from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .utils import fused_add_tanh_sigmoid_multiply
from torch.nn import LayerNorm


# class LayerNorm(nn.Module):
#     def __init__(self, channels, eps=1e-4):
#         super().__init__()
#         self.channels = channels
#         self.eps = eps

#         self.gamma = nn.Parameter(torch.ones(channels))
#         self.beta = nn.Parameter(torch.zeros(channels))

#     def forward(self, x):
#         n_dims = len(x.shape)
#         mean = torch.mean(x, 1, keepdim=True)
#         variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

#         x = (x - mean) * torch.rsqrt(variance + self.eps)

#         shape = [1, -1] + [1] * (n_dims - 2)
#         x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x


class ConvReluNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.norm_layers.append(LayerNorm(hidden_channels, eps=1e-4, elementwise_affine=False))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels, eps=1e-4, elementwise_affine=False))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: Tensor, x_mask: Tensor):
        x_org = x
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x * x_mask)
            x = torch.swapaxes(x, 1, 2)
            x = norm(x)
            x = torch.swapaxes(x, 2, 1)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class WN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert hidden_channels % 2 == 0
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")
        else:
            self.cond_layer = None

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self, x: Tensor, x_mask: Tensor, g: Optional[Tensor] = None
    ):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.tensor([self.hidden_channels], dtype=torch.int)

        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)

        for i, (in_l, skip_l) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_l(x)
            x_in = self.drop(x_in)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)

            res_skip_acts = skip_l(acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    @torch.jit.unused
    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class ActNorm(nn.Module):
    def __init__(self, channels: int, ddi: bool = False):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(
        # TODO: Remove g
        # Although we don't need 'g' we add it to simplify code in the loop in FlowSpecDecoder
        self, x: Tensor, x_mask: Optional[Tensor] = None, reverse: bool = False, g: Optional[Tensor]=None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(
                device=x.device, dtype=x.dtype
            )
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len  # [b]

        return z, logdet

    @torch.jit.unused
    def store_inverse(self):
        pass

    @torch.jit.unused
    def set_ddi(self, ddi: bool):
        self.initialized = not ddi

    @torch.jit.unused
    def initialize(self, x: Tensor, x_mask: Tensor):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m ** 2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (
                (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            )
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):
    def __init__(self, channels: int, n_split: int = 4, no_jacobian: bool = False):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian
        self.weight_inv: Optional[torch.Tensor] = None

        w_init = torch.linalg.qr(
            torch.FloatTensor(self.n_split, self.n_split).normal_()
        )[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(
        # TODO: Remove g
        # Although we don't need 'g' we add it to simplify code in the loop in FlowSpecDecoder
        self, x: Tensor, x_mask: Optional[Tensor] = None, reverse: bool = False, g: Optional[Tensor]=None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        b, c, t = x.size()
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = torch.tensor([1])
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = (
            x.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(b, self.n_split, c // self.n_split, t)
        )

        if reverse:
            logdet = None
            if self.weight_inv is not None:
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = torch.tensor([0])
            else:
                logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len  # [b]

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    @torch.jit.unused
    def store_inverse(self):
        self.weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)

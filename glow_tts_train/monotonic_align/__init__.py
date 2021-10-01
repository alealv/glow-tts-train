import numpy as np
import torch
from .core import maximum_path_c
from torch import Tensor


def maximum_path(value, mask):
    """Cython optimised version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


@torch.jit.script
def maximum_path_each(
    path: Tensor, value: Tensor, t_x: int, t_y: int, max_neg_val: float
):
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[x, y - 1].item()
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[x - 1, y - 1].item()
            value[x, y] += max(v_cur, v_prev)

    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index = index - 1


@torch.jit.script
def ts_maximum_path(value: Tensor, mask: Tensor):
    """Cython optimised version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    path = torch.zeros_like(value)

    t_x_max = mask.sum(1)[:, 0]
    t_y_max = mask.sum(2)[:, 0]

    # Itereate per item in batch
    for i in range(value.shape[0]):
        maximum_path_each(
            path[i],
            value[i],
            int(t_x_max[i].item()),
            int(t_y_max[i].item()),
            max_neg_val=-1e9,
        )

    return path

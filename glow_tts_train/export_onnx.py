#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.export_onnx")

OPSET_VERSION = 12

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-export-onnx")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint (.pth)")
    parser.add_argument("output", type=Path, help="Output model filename (.onnx)")
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="Path to JSON configuration file(s)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug(f"Loading configuration(s) from {args.config}")
        config = TrainingConfig.load_and_merge(config, args.config)

    # Load checkpoint
    _LOGGER.debug(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, config)
    model = checkpoint.model

    _LOGGER.info(
        f"Loaded checkpoint from {args.checkpoint} (global step={checkpoint.global_step})"
    )

    # Inference only
    model.eval()

    # Do not calcuate jacobians for fast decoding
    with torch.no_grad():
        model.decoder.store_inverse()

    old_forward = model.forward

    def infer_forward(text, text_lengths, scales):
        noise_scale = scales[0]
        length_scale = scales[1]
        (mel, mel_lengths, *_), _, _ = old_forward(
            text,
            text_lengths,
            gen=True,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )

        return (mel, mel_lengths)

    model.forward = infer_forward

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = str(args.output) + ".onnx"

    # Create dummy input
    sequences = torch.randint(
        low=0, high=config.model.num_symbols, size=(1, 50), dtype=torch.long
    ).cuda()
    sequence_lengths = torch.IntTensor([sequences.size(1)]).cuda().long()
    scales = torch.FloatTensor([0.667, 1.0])

    dummy_input = (sequences, sequence_lengths, scales)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input", "input_lengths", "scales"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info(f"Exported model to {output}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

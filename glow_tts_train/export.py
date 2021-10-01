#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.export")


def main():
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser(prog="glow-tts-train.export")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint (.pth)")
    parser.add_argument("output", type=Path, help="Output model filename (.pth)")
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
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    # Load checkpoint
    _LOGGER.debug(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, config, use_cuda=False)
    model = checkpoint.model

    _LOGGER.info(
        f"Loaded checkpoint from {args.checkpoint} (global step={checkpoint.global_step})"
    )

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = str(args.output) + ".pth"

    model.eval()

    # Do not calcuate jacobians for fast decoding
    with torch.no_grad():
        model.decoder.store_inverse()

    # Inference only
    # model.forward = model.infer

    jitted_model = torch.jit.script(model)
    torch.jit.save(jitted_model, output)

    _LOGGER.info(f"Saved TorchScript model to {output}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import jsonlines
import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.infer")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-train.infer")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--num-symbols", type=int, help="Number of symbols in the model"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Input format is id|p1 p2 p3..."
    )
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")
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

    # Convert to paths
    if args.config:
        args.config = [Path(p) for p in args.config]

    args.checkpoint = Path(args.checkpoint)

    # Load config
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    if args.num_symbols is not None:
        config.model.num_symbols = args.num_symbols

    _LOGGER.debug(config)

    assert (
        config.model.num_symbols > 0
    ), "Number of symbols not set (did you forget --config or --num-symbols?)"

    # Default mel settings
    output_obj = {
        "id": "",
        "audio": {
            "filter_length": config.audio.filter_length,
            "hop_length": config.audio.hop_length,
            "win_length": config.audio.win_length,
            "mel_channels": config.audio.n_mel_channels,
            "sample_rate": config.audio.sampling_rate,
            "sample_bytes": config.audio.sample_bytes,
            "channels": config.audio.channels,
            "mel_fmin": config.audio.mel_fmin,
            "mel_fmax": config.audio.mel_fmax,
            "normalized": config.audio.normalized,
        },
        "mel": [],
    }

    # Load checkpoint
    start_time = time.perf_counter()
    _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, config, use_cuda=args.cuda)
    end_time = time.perf_counter()

    model, _ = checkpoint.model, checkpoint.optimizer
    _LOGGER.info(
        "Loaded checkpoint from %s in %s second(s) (global step=%s)",
        args.checkpoint,
        end_time - start_time,
        checkpoint.global_step,
    )

    # Do not calcuate jacobians for fast decoding
    model.decoder.store_inverse()
    model.eval()

    if os.isatty(sys.stdin.fileno()):
        print("Reading whitespace-separated phoneme ids from stdin...", file=sys.stderr)

    # Read phoneme ids from standard input.
    # Phoneme ids are separated by whitespace (<p1> <p2> ...)
    writer = jsonlines.Writer(sys.stdout, flush=True)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            utt_id = ""
            if args.csv:
                # Input format is id | p1 p2 p3...
                utt_id, line = line.split("|", maxsplit=1)

            # Phoneme ids as p1 p2 p3...
            phoneme_ids = [int(p) for p in line.split()]
            _LOGGER.debug("%s (id=%s)", phoneme_ids, utt_id)

            # Convert to tensors
            # TODO: Allow batches
            text = torch.autograd.Variable(torch.LongTensor(phoneme_ids).unsqueeze(0))
            text_lengths = torch.LongTensor([text.shape[1]])

            if args.cuda:
                text.contiguous().cuda()
                text_lengths.contiguous().cuda()

            # Infer mel spectrograms
            with torch.no_grad():
                start_time = time.perf_counter()
                (mel, *_), *_, (_attn_gen, *_) = model(
                    text,
                    text_lengths,
                    gen=True,
                    noise_scale=args.noise_scale,
                    length_scale=args.length_scale,
                )
                end_time = time.perf_counter()

                # Write mel spectrogram and settings as a JSON object on one line
                mel_list = mel.squeeze(0).cpu().float().numpy().tolist()
                output_obj["id"] = utt_id
                output_obj["mel"] = mel_list

                writer.write(output_obj)

                _LOGGER.debug(
                    "Generated mel in %s second(s) (%s)", end_time - start_time, utt_id
                )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import logging
import random
import sys
import typing
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import load_checkpoint
from .config import TrainingConfig
from .dataset import PhonemeMelCollate, PhonemeMelLoader, load_mels, load_phonemes
from .ddi import initialize_model
from .models import ModelType
from .optimize import OptimizerType
from .train import train

_LOGGER = logging.getLogger("glow_tts_train")


def parser():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-train")
    parser.add_argument(
        "--output", required=True, type=Path, help="Directory to store model artifacts"
    )
    parser.add_argument(
        "--transcriptions-dir",
        required=True,
        type=Path,
        metavar="transcriptions_dir",
        help="Phones transcriptions directory. The name of each file must be compose as: '<speaker_id>_<dataset_type>.csv'",
    )
    parser.add_argument(
        "--mels-dirs",
        required=True,
        action="append",
        # type=lambda paths: [Path(p) for p in paths.split(" ")],
        type=Path,
        metavar="mels_dirs",
        help="Mels directorie(s)",
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="Path to JSON configuration file(s)",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: use config)"
    )
    parser.add_argument("--checkpoint", type=Path, help="Path to restore checkpoint")
    parser.add_argument("--git-commit", default="", help="Git commit to store in config")
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=1,
        help="Number of epochs between checkpoints",
    )
    parser.add_argument(
        "--skip-missing-mels",
        action="store_true",
        help="Only warn about missing mel files",
    )
    parser.add_argument(
        "--local_rank", type=int, help="Rank passed from torch.distributed.launch"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    return parser.parse_args()


def main(args):

    assert torch.cuda.is_available(), "GPU is required for training"

    is_distributed = args.local_rank is not None

    if is_distributed:
        _LOGGER.info("Setting up distributed run (rank=%s)", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # -------------------------------------------------------------------------

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    config.git_commit = args.git_commit

    _LOGGER.debug(config)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    _LOGGER.debug("Setting random seed to %s", config.seed)
    random.seed(config.seed)

    transcriptions = {}
    for transcription in args.transcriptions_dir.iterdir():
        if transcription.is_file():
            # Extract Speaker ID (must be a number)
            try:
                speaker_id = int(transcription.stem.split("_")[0])
            except ValueError as e:
                _LOGGER.fatal(
                    f"Phones transcription filename must start with Speaker ID and must be a number."
                )
                _LOGGER.debug(e)
                sys.exit(1)

            transcriptions[speaker_id] = transcription

    assert config.model.n_speakers and config.model.n_speakers == len(
        transcriptions
    ), f"Found {len(transcriptions)} speakers in dataset but {config.model.n_speakers} was set."

    _LOGGER.info(f"Gin: {config.model.gin_channels}")
    assert not (
        config.model.n_speakers > 1 and config.model.gin_channels == 0
    ), "Multispeaker model must have gin_channels > 0"

    # Create data loader
    dataset = PhonemeMelLoader(
        transcriptions_by_speaker=transcriptions,
        mels_dirs=args.mels_dirs,
        config=config,
    )
    collate_fn = PhonemeMelCollate(
        n_frames_per_step=config.model.n_frames_per_step,
        multispeaker=len(transcriptions) > 1,
    )

    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    sampler = DistributedSampler(dataset) if is_distributed else None

    train_loader = DataLoader(
        dataset,
        shuffle=(not is_distributed),
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    model: typing.Optional[ModelType] = None
    optimizer: typing.Optional[OptimizerType] = None
    global_step: int = 1

    if args.checkpoint:
        _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
        checkpoint = load_checkpoint(args.checkpoint, config)
        model, optimizer = checkpoint.model, checkpoint.optimizer
        config.learning_rate = checkpoint.learning_rate
        global_step = checkpoint.global_step
        _LOGGER.info(
            "Loaded checkpoint from %s (global step=%s, learning rate=%s)",
            args.checkpoint,
            global_step,
            config.learning_rate,
        )
    else:
        # Data-dependent initialization
        _LOGGER.info("Doing data-dependent initialization...")
        model = initialize_model(train_loader, config)

    if is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Train
    _LOGGER.info("Training started (batch size=%s)", batch_size)

    try:
        train(
            train_loader,
            config,
            args.output,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            checkpoint_epochs=args.checkpoint_epochs,
            rank=(args.local_rank if is_distributed else 0),
        )
        _LOGGER.info("Training finished")
    except KeyboardInterrupt:
        _LOGGER.info("Training stopped")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    main(args)

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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-train")
    parser.add_argument(
        "--output", required=True, help="Directory to store model artifacts"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        nargs=2,
        action="append",
        default=[],
        metavar=("phonemes_csv", "mels"),
        help="Phonemes CSV and JSONL file with mel spectrograms or directory with .npy files (--mels-dir)",
    )
    parser.add_argument(
        "--mels-dir",
        action="store_true",
        help="mels argument is a directory with .npy files",
    )
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: use config)"
    )
    parser.add_argument("--checkpoint", help="Path to restore checkpoint")
    parser.add_argument("--git-commit", help="Git commit to store in config")
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=1,
        help="Number of epochs between checkpoints",
    )
    parser.add_argument(
        "--local_rank", type=int, help="Rank passed from torch.distributed.launch"
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

    assert torch.cuda.is_available(), "GPU is required for training"

    is_distributed = args.local_rank is not None

    if is_distributed:
        _LOGGER.info("Setting up distributed run (rank=%s)", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # -------------------------------------------------------------------------

    # Convert to paths
    args.output = Path(args.output)
    args.dataset = [
        (Path(phonemes_path), Path(mels_path))
        for phonemes_path, mels_path in args.dataset
    ]

    if args.config:
        args.config = [Path(p) for p in args.config]

    if args.checkpoint:
        args.checkpoint = Path(args.checkpoint)

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

    all_id_phonemes = {}
    all_id_mels = {}
    mel_dirs = {}
    num_speakers = len(args.dataset)

    for dataset_idx, (phonemes_path, mels_path) in enumerate(args.dataset):
        # Load phonemes
        _LOGGER.debug("Loading phonemes from %s", phonemes_path)
        with open(phonemes_path, "r") as phonemes_file:
            id_phonemes = load_phonemes(phonemes_file, config)

        _LOGGER.info("Loaded phonemes for %s utterances", len(id_phonemes))

        # Load mels
        id_mels = {}
        if args.mels_dir:
            _LOGGER.debug("Verifying mels in %s", mels_path)
            missing_ids = set()
            for utt_id in id_phonemes:
                mel_path = mels_path / (utt_id + ".npy")
                if not mel_path.is_file():
                    missing_ids.add(utt_id)

            if missing_ids:
                _LOGGER.fatal(
                    "Missing .npy files for utterances: %s", sorted(list(missing_ids))
                )
                sys.exit(1)

            _LOGGER.info("Verified %s mel(s) in %s", len(id_phonemes), mels_path)
            mel_dirs[dataset_idx] = mels_path
        else:
            # TODO: Verify audio configuration
            _LOGGER.debug("Loading JSONL mels from %s", mels_path)
            with open(mels_path, "r") as mels_file:
                id_mels = load_mels(mels_file)

            _LOGGER.info("Loaded mels for %s utterances", len(id_mels))

        # Merge with main set.
        # Disambiguate utterance ids using dataset index.
        for utt_id in id_phonemes:
            all_id_phonemes[(dataset_idx, utt_id)] = id_phonemes[utt_id]

        for utt_id in id_mels:
            all_id_mels[(dataset_idx, utt_id)] = id_mels[utt_id]

    # Set num_symbols
    if config.model.num_symbols < 1:
        config.model.num_symbols = max(max(p_ids) for p_ids in id_phonemes.values()) + 1

    assert config.model.num_symbols > 0, "No symbols"

    config.num_speakers = num_speakers

    # Create data loader
    dataset = PhonemeMelLoader(
        id_phonemes=all_id_phonemes,
        id_mels=all_id_mels,
        mel_dirs=mel_dirs,
        multispeaker=(num_speakers > 1),
    )
    collate_fn = PhonemeMelCollate()

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
    main()

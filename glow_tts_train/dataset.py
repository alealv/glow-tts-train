"""Classes and methods for loading phonemes and mel spectrograms"""
import sys
import csv
import json
import logging
import random
import typing
from pathlib import Path
from functools import reduce

import numpy as np
import torch
import torch.utils.data

from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.dataset")

# -----------------------------------------------------------------------------


class PhonemeMelLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        transcriptions_by_speaker: typing.Dict[int, Path],
        mels_dirs: typing.Iterable[Path],
        config: TrainingConfig,
    ):
        self.multispeaker = len(transcriptions_by_speaker) > 1

        self.dataset = []
        for speaker_id, transcription in transcriptions_by_speaker.items():
            # Load phonemes
            _LOGGER.debug(
                "Loading transcription from %s (speaker=%s)", transcription, speaker_id
            )

            with open(transcription, "r") as phonemes_file:
                transcriptions_dict = load_phonemes(phonemes_file, config)

            _LOGGER.info(
                f"Loaded phonemes for {len(transcriptions_dict)} utterances (speaker={speaker_id})"
            )

            # Extract mels files
            mel_files = {}
            _LOGGER.debug("Verifying mels file for speaker: %s", speaker_id)
            missing_ids = set()
            for utt_id in transcriptions_dict:
                for dir in mels_dirs:
                    mel_path = dir / (utt_id + ".npy")
                    if mel_path.is_file():
                        mel_files[utt_id] = mel_path
                if utt_id not in mel_files.keys():
                    missing_ids.add(utt_id)

            if missing_ids:
                _LOGGER.warning(
                    f"Missing {len(missing_ids)} .npy file(s) out of {len(transcriptions_dict) + len(missing_ids)} for utterances (speaker={speaker_id})"
                )

            intersection = set.intersection(
                set(transcriptions_dict.keys()), set(mel_files.keys())
            )
            _LOGGER.info(
                f"Using {len(intersection)} mel/utterance(s) for speaker: {speaker_id}"
            )

            # Merge with main set.
            # Disambiguate utterance ids using dataset index (speaker id).
            for utt_id in intersection:
                self.dataset.append(
                    {
                        "speaker_id": speaker_id,
                        "utt_id": utt_id,
                        "phonemes": transcriptions_dict[utt_id],
                        "mel_filename": mel_files[utt_id],
                        "mel": None,
                    }
                )

        # Set num_symbols
        num_symbols = len(
            reduce(
                lambda x, y: x.union(set(y["phonemes"].tolist())),
                self.dataset,
                set(),
            )
        )
        if config.model.num_symbols < 1:
            config.model.num_symbols = num_symbols
        elif config.model.num_symbols > num_symbols:
            _LOGGER.warning(
                f"Parsed {num_symbols} amount of symbols but {config.model.num_symbols} was set."
            )
        elif config.model.num_symbols < num_symbols:
            _LOGGER.error(
                f"ABORTING! Parsed {num_symbols} amount of symbols but {config.model.num_symbols} was set."
            )
            sys.exit(1)

    def __getitem__(self, index):
        phonems_seq = self.dataset[index]["phonemes"]

        if self.dataset[index]["mel"] is None:
            self.dataset[index]["mel"] = torch.from_numpy(
                np.load(self.dataset[index]["mel_filename"], allow_pickle=True)
            )

        if self.multispeaker:
            # phonemes, mels, length, speaker
            return (
                phonems_seq,
                self.dataset[index]["mel"],
                len(phonems_seq),
                self.dataset[index]["speaker_id"],
            )

        # phonemes, mels, length
        return (phonems_seq, self.dataset[index]["mel"], len(phonems_seq))

    def __len__(self):
        return len(self.dataset)


class PhonemeMelCollate:
    def __init__(self, n_frames_per_step: int = 1, multispeaker: bool = False):
        self.n_frames_per_step = n_frames_per_step
        self.multispeaker = multispeaker

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        speaker_ids = None
        if self.multispeaker:
            speaker_ids = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

            if speaker_ids is not None:
                speaker_ids[i] = batch[ids_sorted_decreasing[i]][3]

        return text_padded, input_lengths, mel_padded, output_lengths, speaker_ids


# -----------------------------------------------------------------------------


def load_phonemes(
    csv_file: typing.TextIO, config: TrainingConfig
) -> typing.Dict[str, torch.IntTensor]:
    phonemes = {}
    num_too_small = 0
    num_too_large = 0

    reader = csv.reader(csv_file, delimiter="|")
    for row in reader:
        utt_id, phoneme_str = row[0], row[1]
        phoneme_ids = [int(p) for p in phoneme_str.strip().split()]
        num_phonemes = len(phoneme_ids)

        if (config.min_seq_length is not None) and (
            num_phonemes < config.min_seq_length
        ):
            _LOGGER.debug(
                "Dropping %s (%s < %s)", utt_id, num_phonemes, config.min_seq_length
            )
            num_too_small += 1
            continue

        if (config.max_seq_length is not None) and (
            num_phonemes > config.max_seq_length
        ):
            _LOGGER.debug(
                "Dropping %s (%s > %s)", utt_id, num_phonemes, config.max_seq_length
            )
            num_too_large += 1
            continue

        phonemes[utt_id] = torch.IntTensor(phoneme_ids)

    if (num_too_small > 0) or (num_too_large > 0):
        _LOGGER.warning(
            "Dropped some utterance (%s too small, %s too large)",
            num_too_small,
            num_too_large,
        )

    return phonemes


def load_mels(jsonl_file: typing.TextIO) -> typing.Dict[str, torch.FloatTensor]:
    mels = {}
    for line in jsonl_file:
        line = line.strip()
        if not line:
            continue

        mel_obj = json.loads(line)
        utt_id = mel_obj["id"]
        mels[utt_id] = torch.FloatTensor(mel_obj["mel"])

    return mels

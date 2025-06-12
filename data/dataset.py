import torch
from torch.utils.data import Dataset, Sampler
import torchaudio
import numpy as np
from pathlib import Path
import pandas as pd
import os
import logging
from tqdm import tqdm
import sys
from contextlib import contextmanager
import io
from collections import defaultdict

# Disable all possible FunASR output
os.environ["FUNASR_DISABLE_PROGRESS_BAR"] = "1"
os.environ["FUNASR_QUIET"] = "1"
os.environ["FUNASR_LOG_LEVEL"] = "ERROR"
os.environ["FUNASR_DISABLE_TQDM"] = "1"
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)


@contextmanager
def suppress_output():
    # Redirect both stdout and stderr
    new_stdout = io.StringIO()
    new_stderr = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = new_stdout
    sys.stderr = new_stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


from funasr import AutoModel
import librosa
from datasets import load_dataset


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Get max sequence length in the batch
    max_len = max(item["features"].size(0) for item in batch)

    # Initialize padded tensors
    batch_size = len(batch)
    features = torch.zeros(batch_size, max_len, 768)
    labels = torch.zeros(batch_size, dtype=torch.long)
    speaker_ids = []
    datasets = []
    difficulties = []

    # Fill tensors with actual data
    for i, item in enumerate(batch):
        # Features
        seq_len = item["features"].size(0)
        features[i, :seq_len] = item["features"]

        # Labels and metadata
        labels[i] = item["label"]
        speaker_ids.append(item["speaker_id"])
        datasets.append(item["dataset"])
        difficulties.append(
            item.get("difficulty", 0.0)
        )  # Default to 0.0 if not present

    return {
        "features": features,
        "label": labels,
        "speaker_id": speaker_ids,
        "dataset": datasets,
        "difficulty": torch.tensor(difficulties, dtype=torch.float32),
    }


class SpeakerGroupedSampler(Sampler):
    """
    Sampler that groups samples by speaker ID to ensure batches contain samples from the same speaker.
    This helps with speaker disentanglement during training.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Group indices by speaker ID
        self.speaker_groups = defaultdict(list)
        for idx, item in enumerate(dataset.data):
            self.speaker_groups[item["speaker_id"]].append(idx)

        # Sort speakers by ID and create batches
        self.batches = []
        for speaker_id in sorted(self.speaker_groups.keys()):
            indices = self.speaker_groups[speaker_id]
            # Create batches for this speaker, including the last partial batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) > 0:  # Only add non-empty batches
                    self.batches.append(batch)

    def __iter__(self):
        for batch in self.batches:
            yield from batch

    def __len__(self):
        return len(self.dataset)


class EmotionDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        split="train",
        transform=None,
    ):
        """
        Args:
            dataset_name (str): Either 'IEMOCAP' or 'MSP-IMPROV'
            split (str): 'train', 'val', or 'test'
            transform: Optional transform to be applied on a sample
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform

        # Initialize emotion2vec base model for pure emotion representations
        with suppress_output():
            self.model = AutoModel(
                model="emotion2vec/emotion2vec_base",  # Correct HuggingFace model path
                hub="hf",
            )

        # Load dataset from HuggingFace
        if dataset_name.upper() == "IEMOCAP":
            self.dataset = load_dataset("cairocode/IEMO_WAV_Diff", split=split)
        elif dataset_name.upper() == "MSP-IMPROV":
            self.dataset = load_dataset("cairocode/MSPI_WAV", split=split)
        elif dataset_name.upper() == "MSP-PODCAST":
            self.dataset = load_dataset("cairocode/MSPP_WAV_speaker_split")
        else:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Use either 'IEMOCAP' or 'MSP-IMPROV'"
            )

        # Filter for 4-class setup (neutral, happy, sad, anger)
        self.valid_emotions = {"neutral": 0, "happy": 1, "sad": 2, "anger": 3}

        # Filter and store data with proper label mapping
        self.data = []
        for item in self.dataset:
            if item["label"] <= 3:  # Only keep labels 0-3
                # Resample audio to 16kHz if needed
                audio_array = item["audio"]["array"]
                sr = item["audio"]["sampling_rate"]
                if sr != 16000:
                    audio_array = librosa.resample(
                        y=audio_array, orig_sr=sr, target_sr=16000
                    )

                self.data.append(
                    {
                        "audio": audio_array,
                        "label": item["label"],
                        "speaker_id": item["speakerID"],
                        "dataset": dataset_name.upper(),
                        "difficulty": item[
                            "difficulty"
                        ],  # Use existing difficulty column
                    }
                )

    def extract_features(self, audio_array):
        """Extract frame-level features using emotion2vec base model."""
        # Ensure audio is float32
        audio_array = torch.tensor(audio_array, dtype=torch.float32)

        # Generate frame-level features using emotion2vec base model
        with suppress_output():
            rec_result = self.model.generate(
                audio_array,
                output_dir=None,
                granularity="frame",  # Get frame-level features
                extract_embedding=True,
                sr=16000,
            )

        # Get frame-level features [T, 768]
        frame_features = torch.from_numpy(rec_result[0]["feats"]).float()

        return frame_features  # Shape: [T, 768]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract features
        features = self.extract_features(item["audio"])

        if self.transform:
            features = self.transform(features)

        return {
            "features": features,  # Raw emotion2vec features [T, 768]
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "dataset": item["dataset"],
            "difficulty": item["difficulty"],
        }

    def get_speaker_sampler(self, batch_size):
        """
        Create a sampler that groups samples by speaker ID.

        Args:
            batch_size: Size of each batch

        Returns:
            SpeakerGroupedSampler instance
        """
        return SpeakerGroupedSampler(self, batch_size)

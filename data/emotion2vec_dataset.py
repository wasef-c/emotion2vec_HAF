import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import os
import logging
from contextlib import contextmanager
import io
import sys
from funasr import AutoModel
import librosa


# Suppress noisy console output from FunASR
@contextmanager
def suppress_output():
    """Suppress stdout and stderr output"""
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


class Emotion2VecDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        split="train",
        model=None,
    ):
        """
        Args:
            dataset_name (str): Either 'IEMOCAP' or 'MSP-IMPROV'
            split (str): 'train', 'val', or 'test'
            model: Optional pre-loaded emotion2vec model
        """
        self.dataset_name = dataset_name
        self.split = split

        # Initialize emotion2vec model if not provided
        if model is None:
            with suppress_output():
                self.model = AutoModel(model="iic/emotion2vec_base")
                # Freeze the emotion2vec model parameters

        else:
            self.model = model
            # Freeze the emotion2vec model parameters

        # Load dataset from HuggingFace
        if dataset_name.upper() == "IEMOCAP":
            self.dataset = load_dataset("cairocode/IEMO_WAV_002", split=split)
        elif dataset_name.upper() == "MSP-IMPROV":
            self.dataset = load_dataset("cairocode/MSPI_WAV", split=split)
        else:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Use either 'IEMOCAP' or 'MSP-IMPROV'"
            )

        # Filter for 4-class setup (neutral, happy, sad, anger)
        self.valid_emotions = {"neutral": 0, "happy": 1, "sad": 2, "anger": 3}

        # Filter and store data with proper label mapping
        self.data = []
        for item in self.dataset:
            emotion = item["label"]
            if emotion <= 3:
                self.data.append(
                    {
                        "audio": item["audio"],
                        "label": emotion,
                        "speaker_id": item.get(
                            "speaker_id", 0
                        ),  # Default to 0 if not present
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get audio array and resample if needed
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        if sr != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=16000)
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        # Generate emotion2vec features
        with suppress_output():
            features = self.model.generate(
                audio_tensor,
                output_dir=None,
                granularity="utterance",
                extract_embedding=True,
                sr=16000,
            )

        # Convert features to tensor
        features = torch.tensor(features[0]["feats"]).float()

        return {
            "features": features,
            "label": item["label"],
            "speaker_id": item["speaker_id"],
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    """
    features = [item["features"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    speaker_ids = torch.tensor([item["speaker_id"] for item in batch])

    # Pad features to max length in batch
    max_len = max(f.shape[0] for f in features)
    padded_features = []
    for f in features:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, padding], dim=0)
        padded_features.append(f)

    features = torch.stack(padded_features)

    return {"features": features, "label": labels, "speaker_id": speaker_ids}

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Subset


def plot_confusion_matrix(cm, labels, save_path):
    """Plot and optionally save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        return None
    else:
        # Return the figure for wandb logging
        return plt.gcf()


def get_session_splits(dataset):
    """Get indices for each IEMOCAP session.
    Sessions are defined by speaker pairs:
    Session 1 = speakers 1,2
    Session 2 = speakers 3,4
    Session 3 = speakers 5,6
    Session 4 = speakers 7,8
    Session 5 = speakers 9,10
    """
    session_splits = {}
    for i in range(len(dataset)):
        speaker = dataset.data[i]["speaker_id"]
        # Calculate session number: speakers (1,2) -> session 1, (3,4) -> session 2, etc.
        session = (speaker - 1) // 2 + 1
        if session not in session_splits:
            session_splits[session] = []
        session_splits[session].append(i)
    return session_splits


def split_session_data(session_indices, val_ratio=0.2):
    """Split session data into train and validation sets."""
    np.random.shuffle(session_indices)
    val_size = int(len(session_indices) * val_ratio)
    val_indices = session_indices[:val_size]
    train_indices = session_indices[val_size:]
    return train_indices, val_indices


def calculate_class_weights(dataset, class_names):
    """Calculate class weights based on class distribution in the dataset."""
    # Count samples per class
    class_counts = {class_name: 0 for class_name in class_names}

    # Handle both Subset and regular dataset objects
    if isinstance(dataset, Subset):
        # For Subset, we need to access the original dataset and use the indices
        original_dataset = dataset.dataset
        for idx in dataset.indices:
            label = original_dataset.data[idx]["label"]
            class_counts[class_names[label]] += 1
    else:
        # For regular dataset
        for i in range(len(dataset)):
            label = dataset.data[i]["label"]
            class_counts[class_names[label]] += 1

    # Calculate weights (inverse of frequency)
    total_samples = sum(class_counts.values())
    class_weights = {
        class_name: total_samples / (len(class_counts) * count)
        for class_name, count in class_counts.items()
    }

    return class_weights


def get_unique_save_dir(base_name, root="checkpoints"):
    """Get a unique save directory by appending a suffix if needed."""
    root_path = Path(root)
    base_path = root_path / base_name
    if not base_path.exists():
        return base_path

    # Try suffixes like 001, 002, etc.
    for i in range(1, 1000):
        suffixed_path = root_path / f"{base_name}_{i:03d}"
        if not suffixed_path.exists():
            return suffixed_path
    raise RuntimeError("Too many similar experiment names.")


class SpeakerGroupedSampler:
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

        # Get the actual dataset (handle Subset case)
        actual_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

        # Group indices by speaker ID
        self.speaker_groups = defaultdict(list)
        for idx, item in enumerate(actual_dataset.data):
            # If using Subset, we need to map the original indices to subset indices
            if isinstance(dataset, Subset):
                if idx in dataset.indices:
                    subset_idx = dataset.indices.index(idx)
                    self.speaker_groups[item["speaker_id"]].append(subset_idx)
            else:
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
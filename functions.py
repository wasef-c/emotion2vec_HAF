"""
Core functions for emotion recognition with curriculum learning
Consolidates dataset, models, training, evaluation, and difficulty calculation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
import numpy as np
import pandas as pd
from pathlib import Path
import json
import wandb
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean, cosine
import logging
import contextlib
import io
import sys
from collections import defaultdict
from funasr import AutoModel
import librosa
import matplotlib.pyplot as plt


# Suppress outputs for clean execution
@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        yield


# ================================
# CONSTANTS
# ================================
EMOTION2VEC_DIM = 768
EMOTION_CLASSES = ["neutral", "happy", "sad", "anger"]

# ================================
# DATASET AND DATA HANDLING
# ================================


class EmotionDataset(Dataset):
    """Unified emotion dataset for IEMOCAP and MSP-IMPROV"""

    def __init__(self, dataset_name="IEMOCAP", split="train"):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []

        print(f"Loading {dataset_name} dataset...")

        # Load emotion2vec model
        print("Loading emotion2vec model...")

        # Suppress FunASR progress bars and verbose output
        import os

        os.environ["FUNASR_DISABLE_PROGRESS_BAR"] = "1"
        os.environ["FUNASR_QUIET"] = "1"
        os.environ["FUNASR_LOG_LEVEL"] = "ERROR"

        # Suppress tqdm and other progress indicators
        import sys
        from io import StringIO

        # Temporarily redirect stdout/stderr during model loading
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            self.emotion2vec_model = AutoModel(model="iic/emotion2vec_base")
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        print("Emotion2vec model loaded successfully")

        if dataset_name == "IEMOCAP":
            hf_dataset = load_dataset(
                "cairocode/IEMO_WAV_Diff_2_Curriculum", split=split
            )
        elif dataset_name == "MSP-IMPROV":
            hf_dataset = load_dataset("cairocode/MSPI_WAV_Diff_Curriculum", split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Track speaker IDs and sessions for debugging
        speaker_ids_seen = set()
        session_speaker_mapping = {}
        
        # Process each item
        for item in hf_dataset:
            # Create session from speaker_id for LOSO
            speaker_id = item.get("speaker_id", None)
            session = None
            if speaker_id is not None:
                if dataset_name == "IEMOCAP":
                    # IEMOCAP: speakers 1&2→session 1, 3&4→session 2, etc.
                    session = (speaker_id - 1) // 2 + 1
                elif dataset_name == "MSP-IMPROV":
                    # MSP-IMPROV: speakers 947&948→session 1, 949&950→session 2, etc.
                    # Normalize to start from 1: (947-947)//2 + 1 = 1
                    session = (speaker_id - 947) // 2 + 1
                else:
                    # Default fallback for unknown datasets
                    session = (speaker_id - 1) // 2 + 1
                
                # Debug tracking
                speaker_ids_seen.add(speaker_id)
                if session not in session_speaker_mapping:
                    session_speaker_mapping[session] = set()
                session_speaker_mapping[session].add(speaker_id)

            # Resample audio to 16kHz if needed (like original)
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            if sr != 16000:
                import librosa

                audio_array = librosa.resample(
                    y=audio_array, orig_sr=sr, target_sr=16000
                )

            # Convert string values to float, handle None values
            def safe_float(value, default=None):
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=None):
                if value is None:
                    return default
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default

            processed_item = {
                "audio": audio_array,
                "label": item["label"],
                "speaker_id": item.get(
                    "speakerID", item.get("speaker_id", speaker_id)
                ),  # Handle both column names
                "session": session,
                "dataset": dataset_name.upper(),
                "difficulty": safe_float(item.get("difficulty"), 0.5),  # Default difficulty
                "curriculum_order": safe_int(item.get("curriculum_order"), 0),  # Preset curriculum order
                "valence": safe_float(item.get("valence")),
                "arousal": safe_float(item.get("arousal")),
                "domination": safe_float(item.get("consensus_dominance")),  # Use 'domination' like original
            }
            self.data.append(processed_item)

        print(f"Loaded {len(self.data)} samples from {dataset_name}")
        
        # Debug: Print speaker ID and session mapping
        print(f"🔍 DEBUG - Speaker IDs found: {sorted(speaker_ids_seen)}")
        print(f"🔍 DEBUG - Total unique speakers: {len(speaker_ids_seen)}")
        print(f"🔍 DEBUG - Session mapping:")
        for session in sorted(session_speaker_mapping.keys()):
            speakers = sorted(session_speaker_mapping[session])
            print(f"   Session {session}: Speakers {speakers}")
        print(f"🔍 DEBUG - Total sessions created: {len(session_speaker_mapping)}")
        
        # Verify IEMOCAP should have exactly 5 sessions
        if dataset_name == "IEMOCAP" and len(session_speaker_mapping) != 5:
            print(f"⚠️  WARNING: IEMOCAP should have 5 sessions, but found {len(session_speaker_mapping)} sessions!")
            print(f"⚠️  Expected speakers 1-10 → sessions 1-5")
            print(f"⚠️  Actual speakers: {sorted(speaker_ids_seen)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract emotion2vec features
        features = self.extract_features(item["audio"])

        return {
            "features": features,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "session": item["session"],
            "dataset": item["dataset"],
            "difficulty": item["difficulty"],
            "curriculum_order": item["curriculum_order"],
            "valence": item.get("valence", None),
            "arousal": item.get("arousal", None),
            "domination": item.get("domination", None),
        }

    def extract_features(self, audio_array):
        """Extract utterance-level features using emotion2vec base model."""
        # Ensure audio is float32 tensor
        if not isinstance(audio_array, torch.Tensor):
            audio_array = torch.tensor(audio_array, dtype=torch.float32)

        # Extract features using emotion2vec with suppressed output
        import sys
        from io import StringIO

        # Temporarily suppress stdout/stderr for inference
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # Generate utterance-level features using emotion2vec base model
            rec_result = self.emotion2vec_model.generate(
                audio_array,
                output_dir=None,
                granularity="utterance",  # Get utterance-level features to match precomputed
                extract_embedding=True,
                sr=16000,
            )
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Get utterance-level features [EMOTION2VEC_DIM] to match precomputed
        utterance_features = torch.from_numpy(rec_result[0]["feats"]).float()

        return utterance_features  # Shape: [EMOTION2VEC_DIM] to match precomputed


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    max_len = max(item["features"].size(0) for item in batch)
    batch_size = len(batch)

    # Initialize tensors
    features = torch.zeros(batch_size, max_len, EMOTION2VEC_DIM)
    labels = torch.zeros(batch_size, dtype=torch.long)
    difficulties = []
    curriculum_orders = []
    speaker_ids = []
    sessions = []
    datasets = []
    valences = []
    arousals = []
    dominations = []

    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item["features"].size(0)
        features[i, :seq_len] = item["features"]
        labels[i] = item["label"]
        difficulties.append(item.get("difficulty", 0.0))
        curriculum_orders.append(item.get("curriculum_order", 0))
        speaker_ids.append(item["speaker_id"])
        sessions.append(item["session"])
        datasets.append(item["dataset"])
        valences.append(item.get("valence", None))
        arousals.append(item.get("arousal", None))
        dominations.append(item["domination"])

    return {
        "features": features,
        "label": labels,
        "speaker_id": speaker_ids,
        "session": sessions,
        "dataset": datasets,
        "difficulty": torch.tensor(difficulties, dtype=torch.float32),
        "curriculum_order": torch.tensor(curriculum_orders, dtype=torch.long),
        "valence": valences,
        "arousal": arousals,
        "domination": dominations,
    }


class CurriculumSampler(Sampler):
    """Curriculum learning sampler that gradually introduces harder samples"""

    def __init__(
        self,
        dataset,
        batch_size,
        num_epochs,
        curriculum_epochs=15,
        pacing_function="linear",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.curriculum_epochs = curriculum_epochs
        self.pacing_function = pacing_function
        self.current_epoch = 0

        # Get the actual dataset and work with subset indices directly
        self.actual_dataset = (
            dataset.dataset if isinstance(dataset, Subset) else dataset
        )
        self.is_subset = isinstance(dataset, Subset)

        # For subsets, work with relative indices (0 to len(subset)-1)
        if self.is_subset:
            self.subset_size = len(dataset)
            # Work with relative indices within the subset
            self.all_indices = list(range(self.subset_size))

            # Get difficulties/curriculum_order for each item in subset
            self.all_difficulties = []
            self.all_curriculum_orders = []
            for subset_idx in range(self.subset_size):
                actual_idx = dataset.indices[subset_idx]
                difficulty = self.actual_dataset.data[actual_idx].get("difficulty", 0.5)
                curriculum_order = self.actual_dataset.data[actual_idx].get(
                    "curriculum_order", 0.5
                )
                self.all_difficulties.append(difficulty)
                self.all_curriculum_orders.append(curriculum_order)
        else:
            self.all_indices = list(range(len(self.actual_dataset)))
            self.all_difficulties = [
                item.get("difficulty", 0.5) for item in self.actual_dataset.data
            ]
            self.all_curriculum_orders = [
                item.get("curriculum_order", 0.5) for item in self.actual_dataset.data
            ]

        # Always sort by difficulty (which could be original, calculated, or preset)
        sorted_pairs = sorted(
            zip(self.all_indices, self.all_difficulties), key=lambda x: x[1]
        )
        self.sorted_indices = [pair[0] for pair in sorted_pairs]
        self.sorted_difficulties = [pair[1] for pair in sorted_pairs]

        # Set smoother quantile-based progression
        if self.sorted_difficulties:
            # Start with easiest 25% and gradually grow to 100%
            self.start_percentile = 25  # Start with easiest 25% (less aggressive)
            self.end_percentile = 100   # End with all samples
        else:
            self.start_threshold = 0.2
            self.end_threshold = 0.8

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _calculate_progress(self, epoch_progress):
        """Calculate curriculum progress based on pacing function"""
        if self.pacing_function == "linear":
            return epoch_progress
        elif self.pacing_function == "exponential":
            return epoch_progress**1.5  # Less aggressive than epoch_progress**2
        elif self.pacing_function == "logarithmic":
            return np.sqrt(epoch_progress)
        elif self.pacing_function == "sqrt":
            return np.sqrt(epoch_progress)
        else:
            return epoch_progress

    def __iter__(self):
        # Calculate current threshold
        epoch_progress = min(1.0, self.current_epoch / self.curriculum_epochs)
        progress = self._calculate_progress(epoch_progress)

        if self.current_epoch >= self.curriculum_epochs:
            valid_indices = self.all_indices.copy()
        else:
            if hasattr(self, 'start_percentile'):
                # Smooth quantile-based progression
                current_percentile = (
                    self.start_percentile + 
                    (self.end_percentile - self.start_percentile) * progress
                )
                
                # Calculate how many samples to include (smooth growth)
                total_samples = len(self.sorted_indices)
                num_samples_to_include = int((current_percentile / 100.0) * total_samples)
                
                # Take the easiest N samples (they're already sorted by difficulty)
                valid_indices = self.sorted_indices[:num_samples_to_include]
            else:
                # Fallback to threshold-based approach
                current_threshold = (
                    self.start_threshold
                    + (self.end_threshold - self.start_threshold) * progress
                )

                valid_indices = [
                    idx
                    for idx, diff in zip(self.sorted_indices, self.sorted_difficulties)
                    if diff <= current_threshold
                ]

            # Ensure we have enough samples
            if len(valid_indices) < self.batch_size:
                valid_indices = self.sorted_indices[
                    : max(self.batch_size, len(self.sorted_indices) // 10)
                ]

            # Ensure all 4 classes are present
            valid_labels = set()
            for idx in valid_indices:
                if self.is_subset:
                    if idx < len(self.dataset.indices):
                        actual_idx = self.dataset.indices[idx]
                        if actual_idx < len(self.actual_dataset.data):
                            valid_labels.add(
                                self.actual_dataset.data[actual_idx]["label"]
                            )
                else:
                    if idx < len(self.actual_dataset.data):
                        valid_labels.add(self.actual_dataset.data[idx]["label"])

            missing_classes = set(range(4)) - valid_labels
            if missing_classes:
                for missing_class in missing_classes:
                    class_indices = []
                    for idx, diff in zip(self.sorted_indices, self.sorted_difficulties):
                        if self.is_subset:
                            if idx < len(self.dataset.indices):
                                actual_idx = self.dataset.indices[idx]
                                if (
                                    actual_idx < len(self.actual_dataset.data)
                                    and self.actual_dataset.data[actual_idx]["label"]
                                    == missing_class
                                ):
                                    class_indices.append(idx)
                        else:
                            if (
                                idx < len(self.actual_dataset.data)
                                and self.actual_dataset.data[idx]["label"]
                                == missing_class
                            ):
                                class_indices.append(idx)

                    if class_indices:
                        valid_indices.extend(
                            class_indices[:2]
                        )  # Add 2 samples from missing class

        # Final validation - ensure all indices are within bounds
        if self.is_subset:
            final_valid_indices = [
                idx for idx in valid_indices if idx < len(self.dataset)
            ]
        else:
            final_valid_indices = [
                idx for idx in valid_indices if idx < len(self.actual_dataset.data)
            ]

        if not final_valid_indices:
            # Fallback to all available indices if we have none
            fallback_size = min(
                self.batch_size,
                len(self.dataset) if self.is_subset else len(self.actual_dataset.data),
            )
            final_valid_indices = list(range(fallback_size))

        # Use deterministic shuffle with fixed seed
        rng = np.random.RandomState(42)
        rng.shuffle(final_valid_indices)
        return iter(final_valid_indices)

    def __len__(self):
        return len(self.all_indices)


class SpeakerDisentanglementSampler(Sampler):
    """
    Speaker disentanglement sampler: ensures one speaker per batch
    This helps prevent speaker bias during training
    """
    
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get the actual dataset 
        self.actual_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        self.is_subset = isinstance(dataset, Subset)
        
        # Group samples by speaker
        self.speaker_groups = {}
        dataset_size = len(dataset)
        
        for i in range(dataset_size):
            if self.is_subset:
                actual_idx = dataset.indices[i]
                speaker_id = self.actual_dataset.data[actual_idx]["speaker_id"]
            else:
                speaker_id = self.actual_dataset.data[i]["speaker_id"]
            
            if speaker_id not in self.speaker_groups:
                self.speaker_groups[speaker_id] = []
            self.speaker_groups[speaker_id].append(i)
        
        self.speakers = list(self.speaker_groups.keys())
        print(f"📊 Speaker Disentanglement: {len(self.speakers)} speakers, groups: {[(k, len(v)) for k, v in self.speaker_groups.items()]}")
    
    def __iter__(self):
        # Create batches with one speaker per batch
        batches = []
        
        for speaker_id in self.speakers:
            speaker_indices = self.speaker_groups[speaker_id].copy()
            
            if self.shuffle:
                rng = np.random.RandomState(42)  # Deterministic shuffle
                rng.shuffle(speaker_indices)
            
            # Create batches for this speaker
            for i in range(0, len(speaker_indices), self.batch_size):
                batch = speaker_indices[i:i + self.batch_size]
                if len(batch) > 0:  # Only add non-empty batches
                    batches.append(batch)
        
        # Shuffle batch order (but keep speaker homogeneity within batches)
        if self.shuffle:
            rng = np.random.RandomState(42)
            rng.shuffle(batches)
        
        # Flatten batches into a single sequence
        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return len(self.dataset)


class CurriculumSpeakerSampler(CurriculumSampler):
    """
    Combined curriculum learning + speaker disentanglement sampler
    Applies curriculum learning while maintaining speaker homogeneity within batches
    """
    
    def __init__(self, dataset, batch_size, num_epochs, curriculum_epochs=15, 
                 pacing_function="exponential", use_speaker_disentanglement=False):
        super().__init__(dataset, batch_size, num_epochs, curriculum_epochs, pacing_function)
        self.use_speaker_disentanglement = use_speaker_disentanglement
        
        if use_speaker_disentanglement:
            # Group samples by speaker (like SpeakerDisentanglementSampler)
            self.speaker_groups = {}
            dataset_size = len(dataset)
            
            for i in range(dataset_size):
                if self.is_subset:
                    actual_idx = dataset.indices[i]
                    speaker_id = self.actual_dataset.data[actual_idx]["speaker_id"]
                else:
                    speaker_id = self.actual_dataset.data[i]["speaker_id"]
                
                if speaker_id not in self.speaker_groups:
                    self.speaker_groups[speaker_id] = []
                self.speaker_groups[speaker_id].append(i)
            
            self.speakers = list(self.speaker_groups.keys())
            print(f"📊 Curriculum + Speaker: {len(self.speakers)} speakers")
    
    def __iter__(self):
        if not self.use_speaker_disentanglement:
            # Use regular curriculum sampling
            return super().__iter__()
        
        # Combined curriculum + speaker sampling
        epoch_progress = min(1.0, self.current_epoch / self.curriculum_epochs)
        progress = self._calculate_progress(epoch_progress)
        
        if self.current_epoch >= self.curriculum_epochs:
            # After curriculum: use all samples with speaker disentanglement
            valid_indices = self.all_indices.copy()
        else:
            if hasattr(self, 'start_percentile'):
                # Smooth quantile-based progression
                current_percentile = (
                    self.start_percentile + 
                    (self.end_percentile - self.start_percentile) * progress
                )
                
                # Calculate how many samples to include (smooth growth)
                total_samples = len(self.sorted_indices)
                num_samples_to_include = int((current_percentile / 100.0) * total_samples)
                
                # Take the easiest N samples (they're already sorted by difficulty)
                valid_indices = self.sorted_indices[:num_samples_to_include]
            else:
                # Fallback: During curriculum: apply difficulty threshold
                current_threshold = (
                    self.start_threshold + 
                    (self.end_threshold - self.start_threshold) * progress
                )
                
                valid_indices = [
                    idx for idx, diff in zip(self.sorted_indices, self.sorted_difficulties)
                    if diff <= current_threshold
                ]
        
        # Group valid indices by speaker
        speaker_valid_indices = {}
        for idx in valid_indices:
            if self.is_subset:
                actual_idx = self.dataset.indices[idx]
                speaker_id = self.actual_dataset.data[actual_idx]["speaker_id"]
            else:
                speaker_id = self.actual_dataset.data[idx]["speaker_id"]
            
            if speaker_id not in speaker_valid_indices:
                speaker_valid_indices[speaker_id] = []
            speaker_valid_indices[speaker_id].append(idx)
        
        # Create speaker-homogeneous batches
        batches = []
        for speaker_id, indices in speaker_valid_indices.items():
            rng = np.random.RandomState(42)
            rng.shuffle(indices)
            
            # Create batches for this speaker
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) > 0:
                    batches.append(batch)
        
        # Shuffle batch order
        rng = np.random.RandomState(42)
        rng.shuffle(batches)
        
        # Flatten and return
        for batch in batches:
            for idx in batch:
                yield idx


# ================================
# MODELS
# ================================


class AdaptiveEmotionalSaliency(nn.Module):
    """
    Novel approach for speech emotion recognition using learnable emotional saliency
    detection within utterances. Instead of detecting boundaries, this model learns
    which temporal regions within an utterance are most emotionally informative.

    Key innovations:
    1. Learns emotional saliency/importance at each time step
    2. Creates variable-attention segments based on emotional intensity
    3. Processes segments with learned importance weighting
    4. Focuses on emotionally salient regions rather than uniform temporal processing
    """

    def __init__(
        self,
        input_dim=EMOTION2VEC_DIM,
        num_classes=4,
        min_segment_length=4,
        max_segments=15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.min_segment_length = min_segment_length
        self.max_segments = max_segments

        # Emotional saliency network - learns which frames are most emotionally informative
        self.saliency_detector = nn.Sequential(
            nn.Conv1d(
                input_dim, 256, kernel_size=7, padding=3
            ),  # Larger kernel for context
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Saliency scores 0-1
        )

        # Multi-scale temporal processing
        # Short-term: capture rapid emotional variations (prosodic stress, emphasis)
        self.local_processor = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_dim),
        )

        # Long-term: capture emotional contours across utterance
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=input_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.global_processor = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Adaptive pooling based on emotional saliency
        self.saliency_pooling = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Multi-head attention for different emotional aspects
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 512),  # Concatenate local + global features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def detect_emotional_saliency(self, features):
        """
        Detect emotionally salient regions within the utterance.

        Args:
            features: [batch_size, seq_len, input_dim]

        Returns:
            saliency: [batch_size, seq_len] - emotional saliency scores
        """
        # Conv1d expects [batch, channels, length]
        x = features.transpose(1, 2)  # [batch, input_dim, seq_len]
        saliency_scores = self.saliency_detector(x).squeeze(1)  # [batch, seq_len]
        return saliency_scores

    def adaptive_saliency_pooling(self, features, saliency, top_k_ratio=0.3):
        """
        Pool features based on emotional saliency, focusing on most important regions.

        Args:
            features: [batch_size, seq_len, input_dim]
            saliency: [batch_size, seq_len] - saliency scores
            top_k_ratio: proportion of frames to focus on

        Returns:
            pooled_features: [batch_size, input_dim]
        """
        batch_size, seq_len, input_dim = features.shape

        # Select top-k most salient frames
        k = max(1, int(seq_len * top_k_ratio))
        top_k_values, top_k_indices = torch.topk(saliency, k, dim=1)  # [batch, k]

        # Gather top-k features
        batch_indices = (
            torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        )  # [batch, k]
        top_k_features = features[batch_indices, top_k_indices]  # [batch, k, input_dim]

        # Weighted pooling of top-k features
        normalized_weights = F.softmax(top_k_values, dim=1).unsqueeze(
            -1
        )  # [batch, k, 1]
        weighted_features = (top_k_features * normalized_weights).sum(
            dim=1
        )  # [batch, input_dim]

        return weighted_features

    def process_multi_scale(self, features, saliency):
        """
        Process features at multiple temporal scales with saliency guidance.

        Args:
            features: [batch_size, seq_len, input_dim]
            saliency: [batch_size, seq_len]

        Returns:
            local_features: [batch_size, input_dim] - short-term emotional patterns
            global_features: [batch_size, input_dim] - long-term emotional contours
        """
        # Local processing: capture rapid emotional variations
        local_conv = self.local_processor(
            features.transpose(1, 2)
        )  # [batch, input_dim, seq_len]
        local_conv = local_conv.transpose(1, 2)  # [batch, seq_len, input_dim]

        # Apply residual connection with original features
        local_enhanced = features + local_conv

        # Pool local features using saliency (focus on emotionally intense moments)
        local_pooled = self.adaptive_saliency_pooling(
            local_enhanced, saliency, top_k_ratio=0.4
        )

        # Global processing: capture long-term emotional contours
        global_processed = self.global_processor(
            features
        )  # [batch, seq_len, input_dim]

        # Self-attention to capture emotional dependencies
        attended_global, attention_weights = self.emotion_attention(
            global_processed, global_processed, global_processed
        )  # [batch, seq_len, input_dim]

        # Pool global features using saliency (focus on sustained emotional regions)
        global_pooled = self.adaptive_saliency_pooling(
            attended_global, saliency, top_k_ratio=0.6
        )

        return local_pooled, global_pooled

    def forward(self, x):
        """
        Forward pass with adaptive emotional saliency detection.

        Args:
            x: [batch_size, seq_len, input_dim] - emotion2vec features

        Returns:
            outputs: dict with logits and auxiliary outputs
        """
        batch_size, seq_len, input_dim = x.shape

        # Detect emotional saliency within utterance
        saliency_scores = self.detect_emotional_saliency(x)  # [batch_size, seq_len]

        # Process at multiple temporal scales with saliency guidance
        local_features, global_features = self.process_multi_scale(x, saliency_scores)

        # Combine local and global representations
        combined_features = torch.cat(
            [local_features, global_features], dim=1
        )  # [batch, input_dim*2]

        # Final emotion classification
        logits = self.classifier(combined_features)  # [batch_size, num_classes]

        # Return format compatible with existing code
        return {
            "logits": logits,
            "saliency_scores": saliency_scores,
            "local_features": local_features,
            "global_features": global_features,
            "combined_features": combined_features,
        }


class AdaptiveSaliencyLoss(nn.Module):
    """
    Combined loss function for adaptive emotional saliency model.
    Includes emotion classification and saliency regularization.
    Enhanced for imbalanced classification with focal loss.
    """

    def __init__(
        self,
        class_weights=None,
        saliency_weight=0.01,
        diversity_weight=0.01,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        label_smoothing=0.0,
        use_difficulty_scaling=True,
    ):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        if use_focal_loss:
            # Focal loss handles imbalance better than weighted CE
            self.emotion_loss = self.focal_loss
        else:
            self.emotion_loss = self.focal_loss

        self.saliency_weight = saliency_weight  # Reduced from 0.1
        self.diversity_weight = diversity_weight  # Reduced from 0.05
        self.class_weights = class_weights
        self.use_difficulty_scaling = use_difficulty_scaling

    def focal_loss(self, logits, targets, difficulties):
        """Focal Loss for addressing class imbalance with optional label smoothing."""
        # Apply manual label smoothing if specified
        if self.label_smoothing > 0:
            # Create smooth labels
            num_classes = logits.size(-1)
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Calculate cross entropy with smooth labels manually
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
            
            # Apply class weights manually if provided
            if self.class_weights is not None:
                weight_mask = self.class_weights[targets]
                ce_loss = ce_loss * weight_mask
        else:
            # Standard cross entropy for focal loss
            ce_loss = F.cross_entropy(
                logits, targets, weight=self.class_weights, reduction="none"
            )
        
        # Calculate focal weight
        pt = torch.exp(-ce_loss)
        if difficulties is None or not self.use_difficulty_scaling:
            # Standard focal loss without difficulty scaling
            focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        else:
            # Difficulty-scaled focal loss
            adaptive_gamma = self.focal_gamma * (1.0 + 0.5 * difficulties)
            focal_loss = self.focal_alpha * (1 - pt) ** adaptive_gamma * ce_loss
        return focal_loss.mean()

    def saliency_regularization(self, saliency_scores, epoch=None, total_epochs=None):
        """
        Adaptive saliency regularization that becomes less restrictive over time.
        """
        # Adaptive target saliency - starts at 0.3, relaxes to 0.4 over training
        if epoch is not None and total_epochs is not None:
            progress = epoch / total_epochs
            target_saliency = 0.3 + 0.1 * progress  # 0.3 -> 0.4
            target_std = 0.15 + 0.1 * progress  # 0.15 -> 0.25
        else:
            target_saliency = 0.3
            target_std = 0.2

        mean_saliency = saliency_scores.mean()

        # Softer sparsity constraint using smooth L1 loss
        sparsity_loss = F.smooth_l1_loss(
            mean_saliency, torch.tensor(target_saliency, device=mean_saliency.device)
        )

        # Encourage diversity but with softer constraint
        saliency_std = saliency_scores.std(dim=1).mean()
        diversity_loss = F.smooth_l1_loss(
            saliency_std, torch.tensor(target_std, device=saliency_std.device)
        )

        return 0.5 * sparsity_loss + 0.5 * diversity_loss

    def forward(
        self, outputs, targets, speaker_ids=None, epoch=None, total_epochs=None, difficulties = None
    ):
        """
        Compute combined loss.

        Args:
            outputs: dict with model outputs including logits and auxiliary outputs
            targets: [batch_size] - emotion labels
            speaker_ids: optional speaker IDs
            epoch: current epoch for adaptive regularization
            total_epochs: total epochs for adaptive regularization

        Returns:
            total_loss: combined loss value
        """
        # Main emotion classification loss
        logits = outputs["logits"]
        difficulties = difficulties
        emotion_loss = self.emotion_loss(logits, targets, difficulties)

        total_loss = emotion_loss

        # # Add adaptive saliency regularization
        # if self.saliency_weight > 0 and "saliency_scores" in outputs:
        #     saliency_reg = self.saliency_regularization(
        #         outputs["saliency_scores"], epoch, total_epochs
        #     )
        #     total_loss = total_loss + self.saliency_weight * saliency_reg

        # # Add feature diversity regularization (encourage using both local and global)
        # if (
        #     self.diversity_weight > 0
        #     and "local_features" in outputs
        #     and "global_features" in outputs
        # ):
        #     local_norm = outputs["local_features"].norm(dim=1).mean()
        #     global_norm = outputs["global_features"].norm(dim=1).mean()

        #     # Encourage balanced use of both feature types
        #     balance_loss = F.mse_loss(local_norm, global_norm)
        #     total_loss = total_loss + self.diversity_weight * balance_loss

        return total_loss


# ================================
# DIFFICULTY CALCULATION
# ================================


class DifficultyCalculator:
    """Base class for difficulty calculation"""

    def __init__(self, name, expected_vad):
        self.name = name
        self.expected_vad = expected_vad

    def calculate_difficulty(self, samples):
        raise NotImplementedError


class CorrelationDifficulty(DifficultyCalculator):
    """Difficulty based on correlation between expected and actual VAD"""

    def __init__(self, correlation_type="pearson", expected_vad=None):
        super().__init__(f"correlation_{correlation_type}", expected_vad)
        self.correlation_type = correlation_type

    def calculate_difficulty(self, samples):
        difficulties = []

        for sample in samples:
            label = sample["label"]

            # Check for VAD values
            if (
                sample.get("valence") is not None
                and sample.get("arousal") is not None
                and sample.get("domination") is not None
            ):
                actual_vad = [
                    sample["valence"],
                    sample["arousal"],
                    sample["domination"],
                ]
            else:
                raise ValueError(f"Sample missing VAD values: {sample}")

            expected_vad = self.expected_vad[label]

            # Calculate correlation
            try:
                if self.correlation_type == "pearson":
                    corr, _ = pearsonr(expected_vad, actual_vad)
                else:
                    corr, _ = spearmanr(expected_vad, actual_vad)

                if np.isnan(corr):
                    corr = 0.0
            except:
                corr = 0.0

            # Convert to difficulty (higher correlation = easier)
            difficulty = (1 - corr) / 2
            difficulties.append(max(0, min(1, difficulty)))

        return difficulties


class EuclideanDistanceDifficulty(DifficultyCalculator):
    """Difficulty based on directional Euclidean distance"""

    def __init__(self, expected_vad=None, reversed = False):
        super().__init__("euclidean_distance", expected_vad)
        self.reversed = reversed

    def calculate_difficulty(self, samples):
        difficulties = []

        for sample in samples:
            label = sample["label"]

            if (
                sample.get("valence") is not None
                and sample.get("arousal") is not None
                and sample.get("domination") is not None
            ):
                actual_vad = [
                    sample["valence"],
                    sample["arousal"],
                    sample["domination"],
                ]
            else:
                raise ValueError(f"Sample missing VAD values: {sample}")

            expected_vad = self.expected_vad[label]
            difficulty = self._calculate_directional_distance(
                expected_vad, actual_vad, label
            )
            
            difficulties.append(max(0, min(1, difficulty)))
        if self.reversed:
            max_diff = max(difficulties)
            difficulties = [max_diff - d for d in difficulties]
        return difficulties

    def _calculate_directional_distance(self, expected, actual, label):
        """Calculate directional distance - don't punish extreme emotions in expected direction"""
        v_exp, a_exp, d_exp = expected
        v_act, a_act, d_act = actual

        v_diff = v_act - v_exp
        a_diff = a_act - a_exp
        d_diff = d_act - d_exp

        # Simple Euclidean distance - no directional logic, just straight distance
        v_penalty = abs(v_diff)
        a_penalty = abs(a_diff) 
        d_penalty = abs(d_diff)

        distance = np.sqrt(v_penalty**2 + a_penalty**2 + d_penalty**2)
        max_distance = 5 * np.sqrt(3)  # Normalize to [0, 1]
        return distance / max_distance


class WeightedVADDifficulty(DifficultyCalculator):
    """Difficulty based on weighted VAD with directional awareness"""

    def __init__(self, weights=[0.4, 0.4, 0.2], expected_vad=None):
        super().__init__(f"weighted_vad_{weights}", expected_vad)
        self.weights = np.array(weights)

    def calculate_difficulty(self, samples):
        difficulties = []

        for sample in samples:
            label = sample["label"]

            if (
                sample.get("valence") is not None
                and sample.get("arousal") is not None
                and sample.get("domination") is not None
            ):
                actual_vad = [
                    sample["valence"],
                    sample["arousal"],
                    sample["domination"],
                ]
            else:
                raise ValueError(f"Sample missing VAD values: {sample}")

            expected_vad = np.array(self.expected_vad[label])
            actual_vad = np.array(actual_vad)

            penalty = self._calculate_directional_penalty(
                expected_vad, actual_vad, label
            )
            weighted_penalty = np.sum(penalty * self.weights)
            difficulties.append(max(0, min(1, weighted_penalty)))

        return difficulties

    def _calculate_directional_penalty(self, expected, actual, label):
        """Calculate directional penalty"""
        v_diff = actual[0] - expected[0]
        a_diff = actual[1] - expected[1]
        d_diff = actual[2] - expected[2]

        # Same directional logic as Euclidean
        if label == 1:  # happy
            v_penalty = max(0, -v_diff)
            a_penalty = max(0, -a_diff)
            d_penalty = max(0, -d_diff)
        elif label == 2:  # sad
            v_penalty = max(0, v_diff)
            a_penalty = max(0, a_diff)
            d_penalty = max(0, d_diff)
        elif label == 3:  # anger
            v_penalty = max(0, v_diff)
            a_penalty = max(0, -a_diff)
            d_penalty = max(0, -d_diff)
        else:  # neutral
            v_penalty = abs(v_diff)
            a_penalty = abs(a_diff)
            d_penalty = abs(d_diff)

        return np.array([v_penalty, a_penalty, d_penalty])


class BaselineQuadraticDifficulty(DifficultyCalculator):
    """Baseline difficulty using direction-aware quadratic penalty (like original dataset)"""

    def __init__(self, weights=None, expected_vad=None):
        super().__init__("baseline_quadratic", expected_vad)
        # Default weights from baseline: valence=0.5, arousal=0.3, domination=0.2
        self.weights = weights or {"valence": 0.5, "arousal": 0.3, "domination": 0.2}

        # Convert expected_vad format to match baseline (emotion names as keys)
        self.expected_emotions = {0: "neutral", 1: "happy", 2: "sad", 3: "anger"}

        # Expected values from baseline (1-5 scale, converted from our 0-5 scale)
        if expected_vad:
            self.expected = {}
            for label, vad in expected_vad.items():
                emotion = self.expected_emotions[label]
                # Convert from 0-5 scale to 1-5 scale
                self.expected[emotion] = {
                    "valence": vad[0] * 1.25,  # Scale to 1-5 range
                    "arousal": vad[1] * 1.25,
                    "domination": vad[2] * 1.25,
                }
        else:
            # Use baseline values directly
            self.expected = {
                "neutral": {"valence": 3.0, "arousal": 2.5, "domination": 3.0},
                "happy": {"valence": 4.5, "arousal": 4.0, "domination": 3.5},
                "sad": {"valence": 1.5, "arousal": 2.0, "domination": 2.0},
                "anger": {"valence": 1.5, "arousal": 4.5, "domination": 4.0},
            }

    def calculate_difficulty(self, samples):
        difficulties = []

        for sample in samples:
            label = sample["label"]
            emotion = self.expected_emotions[label]

            if (
                sample.get("valence") is not None
                and sample.get("arousal") is not None
                and sample.get("domination") is not None
            ):
                # Convert VAD from 0-5 scale to 1-5 scale to match baseline
                actual_vad = {
                    "valence": sample["valence"] * 1.25,
                    "arousal": sample["arousal"] * 1.25,
                    "domination": sample["domination"] * 1.25,
                }
            else:
                raise ValueError(f"Sample missing VAD values: {sample}")

            difficulty = self._baseline_difficulty_fn(actual_vad, emotion)
            difficulties.append(difficulty)

        return difficulties

    def _baseline_difficulty_fn(self, actual_vad, emotion):
        """Replicate exact baseline difficulty function"""
        features = ["valence", "arousal", "domination"]
        diff = 0.0

        for f in features:
            value = actual_vad[f]
            expected_val = self.expected[emotion][f]

            # Apply directional penalty logic (exactly from baseline)
            penalize = False
            if emotion == "neutral":
                penalize = True  # penalize all deviations
            elif emotion == "happy":
                if f in ["valence", "arousal", "domination"] and value < expected_val:
                    penalize = True
            elif emotion == "sad":
                if f in ["valence", "arousal", "domination"] and value > expected_val:
                    penalize = True
            elif emotion == "anger":
                if f == "valence" and value > expected_val:
                    penalize = True
                if f in ["arousal", "domination"] and value < expected_val:
                    penalize = True

            if penalize:
                deviation = value - expected_val
                weighted_penalty = self.weights[f] * (deviation**2)  # quadratic penalty
                diff += weighted_penalty

        return diff


# ================================
# METRICS AND EVALUATION
# ================================


def calculate_uar(cm):
    """Calculate Unweighted Average Recall"""
    recalls = np.divide(
        cm.diagonal(),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=float),
        where=cm.sum(axis=1) != 0,
    )
    return np.mean(recalls)


def calculate_wa(cm):
    """Calculate Weighted Accuracy"""
    support = cm.sum(axis=1)
    recall_per_class = np.divide(
        np.diag(cm), support, out=np.zeros(cm.shape[0], dtype=float), where=support != 0
    )
    weighted_accuracy = (
        np.sum(recall_per_class * support) / np.sum(cm) if np.sum(cm) > 0 else 0
    )
    return weighted_accuracy


def calculate_metrics(all_labels, all_preds):
    """Calculate comprehensive metrics"""
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    uar = calculate_uar(cm)
    wa = calculate_wa(cm)
    f1_weighted = f1_score(
        all_labels, all_preds, average="weighted", labels=[0, 1, 2, 3], zero_division=0
    )
    f1_macro = f1_score(
        all_labels, all_preds, average="macro", labels=[0, 1, 2, 3], zero_division=0
    )

    return {
        "accuracy": accuracy,
        "wa": wa,
        "uar": uar,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
    }


# ================================
# TRAINING AND EVALUATION
# ================================
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Create a simple progress bar
    total_batches = len(dataloader)
    print(f"Epoch {epoch+1}: ", end="", flush=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):

        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        difficulties = batch["difficulty"].to(device)
        speaker_ids = batch["speaker_id"]

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels, speaker_ids, epoch=epoch, total_epochs=50, difficulties =difficulties )

        loss.backward()
        # Add gradient clipping like in archived version
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Predictions
        preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    print()  # New line after progress bar

    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def evaluate(model, dataloader, criterion, device, return_predictions=False, return_logits=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_speaker_ids = []
    all_difficullties = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            speaker_ids = batch["speaker_id"]
            difficulties = batch["difficulty"].to(device)
            all_difficullties.extend(difficulties)

            outputs = model(features)
            loss = criterion(outputs, labels, speaker_ids, epoch=None, total_epochs=50, difficulties=difficulties)

            total_loss += loss.item()

            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_speaker_ids.extend(speaker_ids.cpu().numpy() if hasattr(speaker_ids, 'cpu') else speaker_ids)
            
            if return_logits:
                all_logits.extend(logits.cpu().numpy())

    # Show prediction distribution
    pred_counts = {}
    label_counts = {}
    for pred in all_preds:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"  Val predictions: {pred_counts}")
    print(f"  Val true labels: {label_counts}")

    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)

    if return_predictions and return_logits:
        return metrics, all_preds, all_labels, all_speaker_ids, all_difficullties, all_logits
    elif return_predictions:
        return metrics, all_preds, all_labels, all_speaker_ids, all_difficullties
        
    return metrics


# ================================
# UTILITY FUNCTIONS
# ================================


def get_session_splits(dataset):
    """Get session-based splits for LOSO (Leave-One-Session-Out)"""
    session_splits = defaultdict(list)

    for i, item in enumerate(dataset.data):
        session = item["session"]

        # Skip if session is None
        if session is not None:
            session_splits[session].append(i)

    return dict(session_splits)


def get_speaker_splits(dataset):
    """Get speaker-based splits for LOSO (Leave-One-Speaker-Out)"""
    speaker_splits = defaultdict(list)

    for i, item in enumerate(dataset.data):
        speaker_id = item["speaker_id"]

        # Skip if speaker_id is None
        if speaker_id is not None:
            speaker_splits[speaker_id].append(i)

    return dict(speaker_splits)


def create_msp_speaker_train_val_splits(dataset, test_speaker_id, val_ratio=0.2):
    """Create train/validation splits for MSP-IMPROV with speaker-based LOSO"""
    # Get all indices
    all_indices = list(range(len(dataset.data)))
    
    # Filter out test speaker
    test_indices = [
        i for i in all_indices if dataset.data[i]["speaker_id"] == test_speaker_id
    ]
    train_val_indices = [
        i for i in all_indices if dataset.data[i]["speaker_id"] != test_speaker_id
    ]
    
    print(f"Test speaker {test_speaker_id}: {len(test_indices)} samples")
    print(f"Train+Val speakers: {len(train_val_indices)} samples")
    
    # Ensure all classes are present in train_val
    class_indices = {0: [], 1: [], 2: [], 3: []}
    for idx in train_val_indices:
        label = dataset.data[idx]["label"]
        class_indices[label].append(idx)

    print(
        f"Class distribution in train+val: {[(k, len(v)) for k, v in class_indices.items()]}"
    )

    # Split train/val ensuring all classes in both sets
    train_indices = []
    val_indices = []

    for emotion_class in range(4):
        class_list = class_indices[emotion_class]
        if len(class_list) == 0:
            print(f"⚠️  Warning: No samples for class {emotion_class} in train+val")
            continue
            
        # Calculate split
        val_size = max(1, int(len(class_list) * val_ratio))  # At least 1 sample for val
        train_size = len(class_list) - val_size
        
        # Random split
        import random
        random.shuffle(class_list)
        
        train_indices.extend(class_list[:train_size])
        val_indices.extend(class_list[train_size:])

    print(f"Final split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_indices, val_indices


def create_subset_dataset(dataset, indices):
    """Create a subset dataset with specified indices"""
    import copy
    subset_dataset = copy.deepcopy(dataset)
    subset_dataset.data = [dataset.data[i] for i in indices]
    return subset_dataset


def create_msp_train_val_splits(dataset, val_ratio=0.2):
    """Create train/validation splits for MSP-IMPROV (no session-based splits)"""
    # Get all indices
    all_indices = list(range(len(dataset.data)))
    
    # Ensure all classes are present in train_val
    class_indices = {0: [], 1: [], 2: [], 3: []}
    for idx in all_indices:
        label = dataset.data[idx]["label"]
        class_indices[label].append(idx)

    print(
        f"Class distribution in MSP: {[(k, len(v)) for k, v in class_indices.items()]}"
    )

    # Split train/val ensuring all classes in both sets
    train_indices = []
    val_indices = []

    for emotion_class in range(4):
        class_list = class_indices[emotion_class]
        if len(class_list) > 0:
            # Use deterministic shuffle with fixed seed
            rng = np.random.RandomState(42)
            rng.shuffle(class_list)
            n_val_class = max(1, int(val_ratio * len(class_list)))
            val_indices.extend(class_list[:n_val_class])
            train_indices.extend(class_list[n_val_class:])

    return train_indices, val_indices


def create_train_val_test_splits(dataset, test_session, val_ratio=0.2):
    """Create train/validation/test splits with session-based test set"""
    # Get all indices
    all_indices = list(range(len(dataset.data)))

    # Debug: Check session distribution
    session_counts = {}
    for i in all_indices:
        session = dataset.data[i].get("session")
        session_counts[session] = session_counts.get(session, 0) + 1
    print(f"Session distribution: {session_counts}")

    # Filter out test set (specific session)
    test_indices = [
        i for i in all_indices if dataset.data[i]["session"] == test_session
    ]
    train_val_indices = [
        i for i in all_indices if dataset.data[i]["session"] != test_session
    ]

    print(f"Test session {test_session}: {len(test_indices)} samples")
    print(f"Train+Val sessions: {len(train_val_indices)} samples")

    # Ensure all classes are present in train_val
    class_indices = {0: [], 1: [], 2: [], 3: []}
    for idx in train_val_indices:
        label = dataset.data[idx]["label"]
        class_indices[label].append(idx)

    print(
        f"Class distribution in train+val: {[(k, len(v)) for k, v in class_indices.items()]}"
    )

    # Split train/val ensuring all classes in both sets
    train_indices = []
    val_indices = []

    for emotion_class in range(4):
        class_list = class_indices[emotion_class]
        if len(class_list) > 0:
            # Use deterministic shuffle with fixed seed
            rng = np.random.RandomState(42)
            rng.shuffle(class_list)
            n_val_class = max(1, int(val_ratio * len(class_list)))
            val_indices.extend(class_list[:n_val_class])
            train_indices.extend(class_list[n_val_class:])

    return train_indices, val_indices, test_indices

# --- Modified function ---
def apply_custom_difficulty(
    dataset, method_name, expected_vad, vad_weights=None
):
    """
    Apply custom difficulty calculation to all data in the dataset.
    This function modifies the 'difficulty' field of each item in dataset.data.

    Args:
        dataset (object): An object with a 'data' attribute (e.g., a list of dictionaries).
        method_name (str): The name of the difficulty calculation method.
        expected_vad (dict): A dictionary with expected VAD values (e.g., {'valence': 0.5, 'arousal': 0.5, 'domination': 0.5}).
        vad_weights (list, optional): A list of weights for VAD dimensions, used by some methods. Defaults to None.
    """

    print(f"🔢 Calculating {method_name} difficulties for all data...")

    # Define the indices for all data in the dataset
    all_indices = range(len(dataset.data))

    # --- Preset method ---
    if method_name == "preset":
        print("📋 Using preset curriculum_order as difficulty values...")
        preset_difficulties = []
        for idx in all_indices:
            item = dataset.data[idx]
            curriculum_order = item.get("curriculum_order", None)
            if curriculum_order is None:
                print(
                    f"⚠️  WARNING: curriculum_order missing for sample {idx}, using 0.5"
                )
                curriculum_order = 0.5
            preset_difficulties.append(curriculum_order)

        # Normalize to [0, 1] range
        min_diff = min(preset_difficulties)
        max_diff = max(preset_difficulties)
        range_diff = max_diff - min_diff

        if range_diff > 0:
            normalized_difficulties = [
                (d - min_diff) / range_diff for d in preset_difficulties
            ]
        else:
            normalized_difficulties = [0.5] * len(preset_difficulties)

        # Update dataset with preset difficulties
        for i, idx in enumerate(all_indices):
            dataset.data[idx]["difficulty"] = normalized_difficulties[i]

        print(
            f"✅ Updated {len(all_indices)} samples with preset curriculum_order"
        )
        print(f"    Original range: {min_diff:.3f} - {max_diff:.3f}")
        print(
            f"    Normalized range: {min(normalized_difficulties):.3f} - {max(normalized_difficulties):.3f}"
        )
        
        # Log per-class statistics (assuming emotion labels exist)
        emotion_names = ["neutral", "happy", "sad", "anger"]
        for emotion_class in range(4):
            class_difficulties = [
                diff
                for i, diff in enumerate(preset_difficulties)
                if dataset.data[all_indices[i]].get("label") == emotion_class
            ]
            if class_difficulties:
                print(
                    f"    {emotion_names[emotion_class]}: mean={np.mean(class_difficulties):.3f}, n={len(class_difficulties)}"
                )

        return  # Early return for preset method

    # --- Initialize calculator for other methods ---
    if method_name == "pearson_correlation":
        calculator = CorrelationDifficulty("pearson", expected_vad)
    elif method_name == "spearman_correlation":
        calculator = CorrelationDifficulty("spearman", expected_vad)
    elif method_name == "euclidean_distance":
        calculator = EuclideanDistanceDifficulty(expected_vad)
    elif method_name == "euclidean_distance_reversed":
        calculator = EuclideanDistanceDifficulty(expected_vad, reversed=True)
    elif method_name.startswith("weighted_vad"):
        weights = vad_weights or [0.4, 0.4, 0.2]
        calculator = WeightedVADDifficulty(weights, expected_vad)
    elif method_name == "baseline_quadratic":
        weights = None
        if vad_weights:
            weights = {
                "valence": vad_weights[0],
                "arousal": vad_weights[1],
                "domination": vad_weights[2],
            }
        calculator = BaselineQuadraticDifficulty(weights, expected_vad)
    else:
        print(f"⚠️  Unknown method: {method_name}")
        return

    # Extract all samples
    all_samples = []
    for idx in all_indices:
        item = dataset.data[idx]
        all_samples.append(
            {
                "label": item.get("label", None),
                "valence": item.get("valence", None),
                "arousal": item.get("arousal", None),
                "domination": item.get("domination", None),
                "original_idx": idx,
            }
        )

    # Calculate difficulties for all samples
    new_difficulties = calculator.calculate_difficulty(all_samples)

    # Apply rescaling with better distribution across [0,1] range
    if new_difficulties:
        difficulties_array = np.array(new_difficulties)
        
        # Choose normalization method based on distribution characteristics
        diff_min = difficulties_array.min()
        diff_max = difficulties_array.max()
        diff_std = difficulties_array.std()
        diff_range = diff_max - diff_min
        
        if diff_max > diff_min:
            # Check if we need better distribution
            # If std is very small relative to range, use percentile-based stretching
            coefficient_of_variation = diff_std / np.mean(difficulties_array) if np.mean(difficulties_array) != 0 else 0
            
            if coefficient_of_variation < 0.2:  # Low variation - use percentile stretching
                print(f"    Using percentile-based distribution (low CV: {coefficient_of_variation:.3f})")
                from scipy.stats import rankdata
                percentile_ranks = rankdata(difficulties_array, method='average') / len(difficulties_array)
                
                # Use a gentler sigmoid that preserves more differences than the original
                stretched = 4 * (percentile_ranks - 0.5)  # Map to [-2, 2] for gentler curve
                sigmoid_stretched = 1 / (1 + np.exp(-stretched))  # Gentler sigmoid
                normalized_difficulties = sigmoid_stretched.tolist()
                
            else:  # Good natural variation - use min-max with optional quantile clipping
                print(f"    Using robust min-max normalization (good CV: {coefficient_of_variation:.3f})")
                
                # Optional: Use quantile-based clipping to remove extreme outliers
                q1, q99 = np.percentile(difficulties_array, [1, 99])
                clipped_difficulties = np.clip(difficulties_array, q1, q99)
                
                # Normalize clipped values
                clipped_min = clipped_difficulties.min()
                clipped_max = clipped_difficulties.max()
                
                if clipped_max > clipped_min:
                    normalized_difficulties = ((clipped_difficulties - clipped_min) / (clipped_max - clipped_min)).tolist()
                else:
                    normalized_difficulties = [0.5] * len(difficulties_array)
        else:
            # All values are the same
            normalized_difficulties = [0.5] * len(difficulties_array)
        
        print(f"    Method: {method_name}")
        print(f"    Original range: {diff_min:.6f} - {diff_max:.6f}")
        print(f"    Original std: {difficulties_array.std():.6f}")
        print(f"    Normalized range: {min(normalized_difficulties):.3f} - {max(normalized_difficulties):.3f}")
        print(f"    Normalized std: {np.array(normalized_difficulties).std():.6f}")
        print(f"    Unique values: {len(np.unique(difficulties_array))}")
        
        # Analyze distribution quality
        norm_array = np.array(normalized_difficulties)
        print(f"    Distribution quartiles: {np.percentile(norm_array, [25, 50, 75])}")
        
        # Debug per-class difficulty if we have labels
        if hasattr(all_samples[0], 'get') and 'label' in all_samples[0]:
            labels_array = np.array([s['label'] for s in all_samples])
            class_names = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Anger'}
            print(f"    Per-class difficulty stats:")
            for class_id in [0, 1, 2, 3]:
                class_mask = labels_array == class_id
                if np.any(class_mask):
                    class_difficulties = norm_array[class_mask]
                    print(f"      {class_names[class_id]}: mean={class_difficulties.mean():.3f}, std={class_difficulties.std():.3f}, n={len(class_difficulties)}")
        elif len(all_samples) > 0 and hasattr(all_samples[0], 'label'):
            # Handle case where samples are objects with .label attribute
            labels_array = np.array([s.label for s in all_samples])
            class_names = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Anger'}
            print(f"    Per-class difficulty stats:")
            for class_id in [0, 1, 2, 3]:
                class_mask = labels_array == class_id
                if np.any(class_mask):
                    class_difficulties = norm_array[class_mask]
                    print(f"      {class_names[class_id]}: mean={class_difficulties.mean():.3f}, std={class_difficulties.std():.3f}, n={len(class_difficulties)}")
        
        # Check how well distributed the values are across [0,1]
        bins = np.linspace(0, 1, 11)  # 10 bins
        hist, _ = np.histogram(norm_array, bins=bins)
        bin_percentages = hist / len(norm_array) * 100
        print(f"    Distribution across 10 bins: {[f'{p:.1f}%' for p in bin_percentages]}")
        
        # Sample some values to see the distribution 
        sample_indices = np.linspace(0, len(difficulties_array)-1, 10, dtype=int)
        print(f"    Sample original: {[f'{difficulties_array[i]:.4f}' for i in sample_indices]}")
        print(f"    Sample normalized: {[f'{normalized_difficulties[i]:.4f}' for i in sample_indices]}")
        
        # Update the dataset with the calculated difficulties
        for i, (sample, difficulty) in enumerate(
            zip(all_samples, normalized_difficulties)
        ):
            original_idx = sample["original_idx"]
            dataset.data[original_idx]["difficulty"] = difficulty

        print(f"✅ Updated {len(all_samples)} samples with {method_name} difficulty")
        print(f"    Original range: {min(new_difficulties):.3f} - {max(new_difficulties):.3f}")
        print(f"    Normalized range: {min(normalized_difficulties):.3f} - {max(normalized_difficulties):.3f}")
    else:
        print("⚠️  No difficulties calculated. Dataset may be empty.")
        return

def difficulty_plot(test_preds,test_labels,all_difficulties,dataset_name="IEMO"):
    # Configuration - increased to 10 bins for better granularity
    num_bins = 10

    # Helper function to convert to numpy safely
    def to_numpy(x):
        if x is None:
            return None
        # Handle torch tensors
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        # Handle lists/arrays that might contain tensors
        if isinstance(x, (list, tuple)):
            return np.array([to_numpy(item) if hasattr(item, 'detach') else item for item in x])
        # Handle regular numpy arrays or other types
        return np.array(x)

    # Convert all inputs to numpy arrays
    test_preds = to_numpy(test_preds)
    test_labels = to_numpy(test_labels)
    all_difficulties = to_numpy(all_difficulties)
    # Calculate per-sample correctness
    correctness = (test_preds == test_labels)
    difficulty_bins = np.linspace(
        start=all_difficulties.min(),
        stop=all_difficulties.max(),
        num=num_bins + 1
    )
    # Use np.digitize to assign each difficulty score to a bin index (1-based)
    bin_indices = np.digitize(all_difficulties, difficulty_bins)
    # Store the results for the plot
    bin_accuracies = []
    bin_centers = []
    bin_counts = []
    bin_labels = []
    
    # Iterate through each bin
    for i in range(1, num_bins + 1):
        # Find all samples that fall into the current bin
        indices_in_bin = np.where(bin_indices == i)[0]
        
        if len(indices_in_bin) > 0:
            # Calculate accuracy for this bin
            accuracy = np.mean(correctness[indices_in_bin])
            
            # Get the center of the bin for plotting
            bin_center = (difficulty_bins[i-1] + difficulty_bins[i]) / 2
            
            # Create a readable label with difficulty range and sample count
            bin_label = f"Bin {i}\n({difficulty_bins[i-1]:.2f}-{difficulty_bins[i]:.2f})\nn={len(indices_in_bin)}"
            
            bin_accuracies.append(accuracy)
            bin_centers.append(bin_center)
            bin_counts.append(len(indices_in_bin))
            bin_labels.append(bin_label)
            
            print(f"  Bin {i}/{num_bins}: Samples={len(indices_in_bin)}, Accuracy={accuracy:.4f}")
        else:
            print(f"  Bin {i}/{num_bins}: No samples found.")

    # Create a WandB Table to log the data with sample counts
    data = [[label, acc, count] for label, acc, count in zip(bin_labels, bin_accuracies, bin_counts)]
    table = wandb.Table(data=data, columns=["difficulty_range", "accuracy", "sample_count"])

    # Create the plot object from the table - using bar chart for better readability
    accuracy_plot = wandb.plot.bar(
        table,
        "difficulty_range",
        "accuracy",
        title=f"Model Accuracy vs. Data Difficulty ({dataset_name})")
        
    return accuracy_plot


def confidence_vs_difficulty_session_analysis(logits, difficulties, labels, predictions, dataset_name="IEMO"):
    """Create clean difficulty vs confidence analysis for per-session data (smaller, more interpretable)"""
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats
    
    # Convert to numpy
    logits = np.array([l.cpu().numpy() if hasattr(l, 'cpu') else l for l in logits])
    difficulties = np.array([d.cpu().numpy() if hasattr(d, 'cpu') else d for d in difficulties])
    labels = np.array([l.cpu().numpy() if hasattr(l, 'cpu') else l for l in labels])
    predictions = np.array([p.cpu().numpy() if hasattr(p, 'cpu') else p for p in predictions])
    
    # Calculate confidence metrics from logits  
    from scipy.special import softmax
    probs = softmax(logits, axis=1)  # Convert logits to probabilities
    max_probs = np.max(probs, axis=1)  # Max probability (confidence)
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)  # Entropy (lower = more confident)
    
    # Normalized entropy (0-1, where 0 = most confident, 1 = least confident)
    max_entropy = np.log(logits.shape[1])  # log(num_classes)
    normalized_entropy = entropy / max_entropy
    
    # DEBUGGING: Print stats to verify differences across experiments
    print(f"=== CONFIDENCE/DIFFICULTY DEBUG for {dataset_name} ===")
    print(f"Difficulty stats: min={difficulties.min():.6f}, max={difficulties.max():.6f}, std={difficulties.std():.6f}")
    print(f"Confidence stats: min={max_probs.min():.6f}, max={max_probs.max():.6f}, std={max_probs.std():.6f}")
    print(f"Entropy stats: min={normalized_entropy.min():.6f}, max={normalized_entropy.max():.6f}, std={normalized_entropy.std():.6f}")
    print(f"Unique difficulty values: {len(np.unique(difficulties))}")
    print(f"Sample difficulties: {np.percentile(difficulties, [0, 25, 50, 75, 100])}")
    print(f"Sample confidences: {np.percentile(max_probs, [0, 25, 50, 75, 100])}")
    
    # VALIDATION: Check if difficulty semantics are correct
    # Easy samples (low difficulty) should have higher accuracy than hard samples (high difficulty)
    correct_predictions = (predictions == labels)
    easy_mask = difficulties < np.percentile(difficulties, 25)  # Bottom 25% = easiest
    hard_mask = difficulties > np.percentile(difficulties, 75)  # Top 25% = hardest
    
    easy_accuracy = np.mean(correct_predictions[easy_mask])
    hard_accuracy = np.mean(correct_predictions[hard_mask])
    
    print(f"DIFFICULTY VALIDATION:")
    print(f"  Easy samples (bottom 25%) accuracy: {easy_accuracy:.4f}")
    print(f"  Hard samples (top 25%) accuracy: {hard_accuracy:.4f}")
    print(f"  Difference (easy - hard): {easy_accuracy - hard_accuracy:.4f}")
    
    if easy_accuracy > hard_accuracy:
        print(f"  ✅ CORRECT: Easy samples have higher accuracy than hard samples")
    else:
        print(f"  ❌ INVERTED: Hard samples have higher accuracy - difficulty may be inverted!")
        print(f"     Consider using a different difficulty method or adding '_reversed' suffix")
    
    # Create figure with scatter plot + binned analyses
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Confidence vs Difficulty Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Calculate statistical relationships instead of binning
    from scipy import stats
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    
    # Compute correlation statistics
    pearson_corr, pearson_p = stats.pearsonr(difficulties, max_probs)
    spearman_corr, spearman_p = stats.spearmanr(difficulties, max_probs)
    
    # Compute R² using linear regression
    reg = LinearRegression()
    X = difficulties.reshape(-1, 1)
    reg.fit(X, max_probs)
    confidence_r2 = r2_score(max_probs, reg.predict(X))
    
    # Same for accuracy relationship
    correct_predictions = (predictions == labels).astype(float)
    acc_pearson_corr, acc_pearson_p = stats.pearsonr(difficulties, correct_predictions) 
    acc_reg = LinearRegression()
    acc_reg.fit(X, correct_predictions)
    accuracy_r2 = r2_score(correct_predictions, acc_reg.predict(X))
    
    # Same for entropy relationship
    entropy_pearson_corr, entropy_pearson_p = stats.pearsonr(difficulties, normalized_entropy)
    entropy_reg = LinearRegression()
    entropy_reg.fit(X, normalized_entropy)
    entropy_r2 = r2_score(normalized_entropy, entropy_reg.predict(X))
    
    print(f"STATISTICAL RELATIONSHIPS:")
    print(f"  Difficulty vs Confidence: r={pearson_corr:.4f} (p={pearson_p:.4f}), R²={confidence_r2:.4f}")
    print(f"  Difficulty vs Accuracy: r={acc_pearson_corr:.4f} (p={acc_pearson_p:.4f}), R²={accuracy_r2:.4f}")
    print(f"  Difficulty vs Entropy: r={entropy_pearson_corr:.4f} (p={entropy_pearson_p:.4f}), R²={entropy_r2:.4f}")
    
    # Create finer bins for trend overlay (not main analysis)
    n_trend_bins = 30
    bin_edges = np.linspace(difficulties.min(), difficulties.max(), n_trend_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_confidence_means = []
    bin_accuracy_means = []
    bin_entropy_means = []
    
    for i in range(n_trend_bins):
        mask = (difficulties >= bin_edges[i]) & (difficulties < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_confidence_means.append(np.mean(max_probs[mask]))
            bin_accuracy_means.append(np.mean(correct_predictions[mask]))
            bin_entropy_means.append(np.mean(normalized_entropy[mask]))
        else:
            bin_confidence_means.append(np.nan)
            bin_accuracy_means.append(np.nan)
            bin_entropy_means.append(np.nan)
    
    # 1. Hexagonal binning plot for density visualization
    axes[0, 0].hexbin(difficulties, max_probs, gridsize=30, cmap='Blues', alpha=0.8)
    
    # Add regression line
    x_line = np.linspace(difficulties.min(), difficulties.max(), 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    axes[0, 0].plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8, label=f'Regression (R²={confidence_r2:.3f})')
    
    axes[0, 0].set_xlabel('Difficulty')
    axes[0, 0].set_ylabel('Confidence (Max Probability)')
    axes[0, 0].set_title(f'Confidence vs Difficulty\nr={pearson_corr:.3f}, p={pearson_p:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Alternative Confidence vs Difficulty view (different style from plot 1)
    # Sample if too many points
    if len(difficulties) > 3000:
        sample_idx = np.random.choice(len(difficulties), 3000, replace=False)
        plot_diff = difficulties[sample_idx]
        plot_conf = max_probs[sample_idx]
        plot_correct = correct_predictions[sample_idx]
    else:
        plot_diff = difficulties
        plot_conf = max_probs
        plot_correct = correct_predictions
    
    # Color by actual correctness for confidence
    axes[0, 1].scatter(plot_diff[plot_correct == 1], plot_conf[plot_correct == 1], 
                      alpha=0.4, c='green', s=1, label='Correct')
    axes[0, 1].scatter(plot_diff[plot_correct == 0], plot_conf[plot_correct == 0], 
                      alpha=0.4, c='red', s=1, label='Incorrect')
    
    # Add regression line for confidence (same as plot 1 but different style)
    y_conf_line = reg.predict(x_line.reshape(-1, 1))
    axes[0, 1].plot(x_line, y_conf_line, 'b-', linewidth=2, alpha=0.8, label=f'Regression (R²={confidence_r2:.3f})')
    
    # Add trend line from binning for smoother overlay
    valid_bins = ~np.isnan(bin_confidence_means)
    if np.any(valid_bins):
        axes[0, 1].plot(np.array(bin_centers)[valid_bins], np.array(bin_confidence_means)[valid_bins], 
                       'o-', color='darkblue', linewidth=2, markersize=6, alpha=0.9, label='Binned Trend')
    
    axes[0, 1].set_xlabel('Difficulty')
    axes[0, 1].set_ylabel('Confidence (Max Probability)')
    axes[0, 1].set_title(f'Confidence vs Difficulty (Scatter)\nr={pearson_corr:.3f}, p={pearson_p:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Confidence vs Accuracy scatter (direct relationship)
    conf_acc_corr, conf_acc_p = stats.pearsonr(max_probs, correct_predictions)
    
    # Use scatter with jitter for binary accuracy  
    jitter_strength = 0.05  # Define jitter strength here too
    jittered_accuracy_conf = correct_predictions + np.random.normal(0, jitter_strength, len(correct_predictions))
    
    # Sample if too many points
    if len(max_probs) > 3000:
        sample_idx_conf = np.random.choice(len(max_probs), 3000, replace=False)
        plot_conf = max_probs[sample_idx_conf]
        plot_acc_conf = jittered_accuracy_conf[sample_idx_conf]
        plot_correct_conf = correct_predictions[sample_idx_conf]
    else:
        plot_conf = max_probs
        plot_acc_conf = jittered_accuracy_conf
        plot_correct_conf = correct_predictions
    
    # Color by actual correctness
    axes[1, 0].scatter(plot_conf[plot_correct_conf == 1], plot_acc_conf[plot_correct_conf == 1], 
                      alpha=0.4, c='green', s=1, label='Correct')
    axes[1, 0].scatter(plot_conf[plot_correct_conf == 0], plot_acc_conf[plot_correct_conf == 0], 
                      alpha=0.4, c='red', s=1, label='Incorrect')
    
    # Add regression line for confidence vs accuracy
    conf_reg = LinearRegression()
    conf_reg.fit(max_probs.reshape(-1, 1), correct_predictions)
    x_conf_line = np.linspace(max_probs.min(), max_probs.max(), 100)
    y_conf_acc_line = conf_reg.predict(x_conf_line.reshape(-1, 1))
    conf_acc_r2 = r2_score(correct_predictions, conf_reg.predict(max_probs.reshape(-1, 1)))
    
    axes[1, 0].plot(x_conf_line, y_conf_acc_line, 'purple', linewidth=2, alpha=0.8, 
                   label=f'Regression (R²={conf_acc_r2:.3f})')
    
    axes[1, 0].set_xlabel('Confidence (Max Probability)')
    axes[1, 0].set_ylabel('Accuracy (with jitter)')
    axes[1, 0].set_title(f'Confidence vs Accuracy\nr={conf_acc_corr:.3f}, p={conf_acc_p:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(-0.1, 1.1)
    
    # 4. Uncertainty (entropy) by difficulty
    axes[1, 1].hexbin(difficulties, normalized_entropy, gridsize=30, cmap='Reds', alpha=0.8)
    
    # Add regression line for entropy
    y_entropy_line = entropy_reg.predict(x_line.reshape(-1, 1))
    axes[1, 1].plot(x_line, y_entropy_line, 'r-', linewidth=2, alpha=0.8, 
                   label=f'Regression (R²={entropy_r2:.3f})')
    
    # Add trend line from binning
    valid_entropy_bins = ~np.isnan(bin_entropy_means)
    if np.any(valid_entropy_bins):
        axes[1, 1].plot(np.array(bin_centers)[valid_entropy_bins], np.array(bin_entropy_means)[valid_entropy_bins], 
                       'o-', color='darkred', linewidth=1, markersize=4, alpha=0.7, label='Trend')
    
    axes[1, 1].set_xlabel('Difficulty')
    axes[1, 1].set_ylabel('Normalized Entropy (Uncertainty)')
    axes[1, 1].set_title(f'Uncertainty vs Difficulty\nr={entropy_pearson_corr:.3f}, p={entropy_pearson_p:.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Convert to wandb Image
    import wandb
    wandb_image = wandb.Image(plt)
    plt.close()
    
    return wandb_image


def confidence_vs_difficulty_analysis(logits, difficulties, labels, predictions, dataset_name="IEMO"):
    """Create difficulty vs confidence analysis plot using logits entropy and max probability"""
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats
    
    # Convert to numpy
    logits = np.array([l.cpu().numpy() if hasattr(l, 'cpu') else l for l in logits])
    difficulties = np.array([d.cpu().numpy() if hasattr(d, 'cpu') else d for d in difficulties])
    labels = np.array([l.cpu().numpy() if hasattr(l, 'cpu') else l for l in labels])
    predictions = np.array([p.cpu().numpy() if hasattr(p, 'cpu') else p for p in predictions])
    
    # Calculate confidence metrics from logits  
    from scipy.special import softmax
    probs = softmax(logits, axis=1)  # Convert logits to probabilities
    max_probs = np.max(probs, axis=1)  # Max probability (confidence)
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)  # Entropy (lower = more confident)
    
    # Normalized entropy (0-1, where 0 = most confident, 1 = least confident)
    max_entropy = np.log(logits.shape[1])  # log(num_classes)
    normalized_entropy = entropy / max_entropy
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Difficulty vs Confidence Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Difficulty vs Max Probability
    axes[0, 0].scatter(difficulties, max_probs, alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Difficulty')
    axes[0, 0].set_ylabel('Max Probability (Confidence)')
    axes[0, 0].set_title('Difficulty vs Max Probability')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_maxprob, p_maxprob = scipy.stats.pearsonr(difficulties, max_probs)
    axes[0, 0].text(0.05, 0.95, f'Corr: {corr_maxprob:.3f} (p={p_maxprob:.3f})', 
                    transform=axes[0, 0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 2. Scatter plot: Difficulty vs Entropy
    axes[0, 1].scatter(difficulties, normalized_entropy, alpha=0.6, s=20, color='red')
    axes[0, 1].set_xlabel('Difficulty')
    axes[0, 1].set_ylabel('Normalized Entropy (Uncertainty)')
    axes[0, 1].set_title('Difficulty vs Entropy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_entropy, p_entropy = scipy.stats.pearsonr(difficulties, normalized_entropy)
    axes[0, 1].text(0.05, 0.95, f'Corr: {corr_entropy:.3f} (p={p_entropy:.3f})', 
                    transform=axes[0, 1].transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
    
    # 3. Binned analysis: Average confidence by difficulty
    # Create 5 difficulty bins consistent with other analyses
    bin_edges = np.linspace(difficulties.min(), difficulties.max(), 6)
    bin_indices = np.digitize(difficulties, bin_edges)
    
    bin_centers = []
    bin_max_probs = []
    bin_entropies = []
    bin_accuracies = []
    
    for i in range(1, len(bin_edges)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)
            bin_max_probs.append(np.mean(max_probs[mask]))
            bin_entropies.append(np.mean(normalized_entropy[mask]))
            bin_accuracies.append(np.mean(predictions[mask] == labels[mask]))
    
    axes[0, 2].plot(bin_centers, bin_max_probs, 'o-', label='Max Probability', color='blue')
    axes[0, 2].set_xlabel('Difficulty (Bin Centers)')
    axes[0, 2].set_ylabel('Average Max Probability')
    axes[0, 2].set_title('Confidence by Difficulty Bins')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # 4. Accuracy vs Confidence
    axes[1, 0].plot(bin_centers, bin_accuracies, 's-', label='Accuracy', color='green')
    axes[1, 0].plot(bin_centers, bin_max_probs, 'o-', label='Confidence', color='blue')
    axes[1, 0].set_xlabel('Difficulty (Bin Centers)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Accuracy vs Confidence by Difficulty')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 5. Entropy by Difficulty Bins
    axes[1, 1].plot(bin_centers, bin_entropies, '^-', label='Normalized Entropy', color='red')
    axes[1, 1].set_xlabel('Difficulty (Bin Centers)')
    axes[1, 1].set_ylabel('Average Normalized Entropy')
    axes[1, 1].set_title('Uncertainty by Difficulty Bins')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 6. Confidence statistics table
    stats_text = f"{'Metric':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n"
    stats_text += "-" * 60 + "\n"
    stats_text += f"{'Max Probability':<20} {np.mean(max_probs):<8.3f} {np.std(max_probs):<8.3f} {np.min(max_probs):<8.3f} {np.max(max_probs):<8.3f}\n"
    stats_text += f"{'Normalized Entropy':<20} {np.mean(normalized_entropy):<8.3f} {np.std(normalized_entropy):<8.3f} {np.min(normalized_entropy):<8.3f} {np.max(normalized_entropy):<8.3f}\n"
    stats_text += f"{'Difficulty':<20} {np.mean(difficulties):<8.3f} {np.std(difficulties):<8.3f} {np.min(difficulties):<8.3f} {np.max(difficulties):<8.3f}\n"
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                   fontfamily='monospace', fontsize=9, verticalalignment='center')
    axes[1, 2].set_title('Confidence Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Convert to wandb Image
    import wandb
    wandb_image = wandb.Image(plt)
    plt.close()
    
    return wandb_image


def difficulty_analysis(difficulties, labels, dataset_name="IEMO"):
    """Create comprehensive difficulty analysis plots using matplotlib"""
    
    # Convert to numpy
    difficulties = np.array([d.cpu().numpy() if hasattr(d, 'cpu') else d for d in difficulties])
    labels = np.array([l.cpu().numpy() if hasattr(l, 'cpu') else l for l in labels])
    
    plots = {}
    emotion_names = ["neutral", "happy", "sad", "anger"]
    colors = ['blue', 'green', 'red', 'orange']
    
    # Create a comprehensive subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Difficulty Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Create consistent bin edges for all plots (5 bins from min to max)
    bin_edges = np.linspace(difficulties.min(), difficulties.max(), 6)  # 6 edges for 5 bins
    
    # 1. Overall difficulty distribution
    axes[0, 0].hist(difficulties, bins=bin_edges, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Overall Difficulty Distribution')
    axes[0, 0].set_xlabel('Difficulty')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Difficulty by emotion class (overlapping histograms)
    for emotion_class in range(4):
        class_mask = labels == emotion_class
        class_difficulties = difficulties[class_mask]
        if len(class_difficulties) > 0:
            axes[0, 1].hist(class_difficulties, bins=bin_edges, alpha=0.6, 
                          label=emotion_names[emotion_class], color=colors[emotion_class])
    
    axes[0, 1].set_title('Difficulty by Emotion Class')
    axes[0, 1].set_xlabel('Difficulty')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot by emotion
    emotion_diffs = []
    emotion_labels_for_box = []
    for emotion_class in range(4):
        class_mask = labels == emotion_class
        class_difficulties = difficulties[class_mask]
        if len(class_difficulties) > 0:
            emotion_diffs.append(class_difficulties)
            emotion_labels_for_box.append(emotion_names[emotion_class])
    
    if emotion_diffs:
        axes[0, 2].boxplot(emotion_diffs, labels=emotion_labels_for_box)
        axes[0, 2].set_title('Difficulty Distribution by Emotion')
        axes[0, 2].set_ylabel('Difficulty')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Statistics table as text
    stats_text = f"{'Emotion':<8} {'Count':<6} {'Mean':<6} {'Std':<6} {'Min':<6} {'Max':<6}\n"
    stats_text += "-" * 50 + "\n"
    
    for emotion_class in range(4):
        class_mask = labels == emotion_class
        class_difficulties = difficulties[class_mask]
        
        if len(class_difficulties) > 0:
            stats_text += f"{emotion_names[emotion_class]:<8} {len(class_difficulties):<6} "
            stats_text += f"{np.mean(class_difficulties):<6.3f} {np.std(class_difficulties):<6.3f} "
            stats_text += f"{np.min(class_difficulties):<6.3f} {np.max(class_difficulties):<6.3f}\n"
    
    axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes, 
                   fontfamily='monospace', fontsize=10, verticalalignment='center')
    axes[1, 0].set_title('Difficulty Statistics')
    axes[1, 0].axis('off')
    
    # 5. Difficulty vs sample index (to see any patterns)
    axes[1, 1].scatter(range(len(difficulties)), difficulties, alpha=0.6, s=10)
    axes[1, 1].set_title('Difficulty vs Sample Index')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Difficulty')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Difficulty percentiles
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = [np.percentile(difficulties, p) for p in percentiles]
    axes[1, 2].bar(range(len(percentiles)), percentile_values, 
                   tick_label=[f'{p}%' for p in percentiles], color='lightcoral')
    axes[1, 2].set_title('Difficulty Percentiles')
    axes[1, 2].set_ylabel('Difficulty Value')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to wandb image
    plots[f"difficulty_analysis/{dataset_name}_comprehensive_analysis"] = wandb.Image(fig)
    plt.close(fig)
    
    return plots


def create_data_loader(dataset, config, is_training=False):
    """Create data loader with optional curriculum learning and speaker disentanglement"""

    if is_training:
        # Debug: Print config values
        print(f"🔍 DEBUG: use_curriculum_learning = {config.use_curriculum_learning} (type: {type(config.use_curriculum_learning)})")
        print(f"🔍 DEBUG: use_speaker_disentanglement = {config.use_speaker_disentanglement} (type: {type(config.use_speaker_disentanglement)})")
        
        # Determine which sampler to use
        if config.use_curriculum_learning and config.use_speaker_disentanglement:
            # Combined curriculum + speaker disentanglement
            try:
                sampler = CurriculumSpeakerSampler(
                    dataset,
                    config.batch_size,
                    config.num_epochs,
                    config.curriculum_epochs,
                    config.curriculum_pacing,
                    use_speaker_disentanglement=True
                )
                print("📊 Using CurriculumSpeakerSampler (curriculum + speaker)")
            except Exception as e:
                print(f"⚠️  CurriculumSpeakerSampler failed: {e}")
                sampler = None
                
        elif config.use_curriculum_learning:
            # Curriculum learning only
            try:
                sampler = CurriculumSampler(
                    dataset,
                    config.batch_size,
                    config.num_epochs,
                    config.curriculum_epochs,
                    config.curriculum_pacing,
                )
                print("📚 Using CurriculumSampler (curriculum only)")
            except Exception as e:
                print(f"⚠️  CurriculumSampler failed: {e}")
                sampler = None
                
        elif config.use_speaker_disentanglement:
            # Speaker disentanglement only
            try:
                sampler = SpeakerDisentanglementSampler(
                    dataset,
                    config.batch_size,
                    shuffle=True
                )
                print("🔊 Using SpeakerDisentanglementSampler (speaker only)")
            except Exception as e:
                print(f"⚠️  SpeakerDisentanglementSampler failed: {e}")
                sampler = None
        else:
            # No special sampling
            sampler = None
            print("🎲 Using standard random sampling")
        
        # Create DataLoader with appropriate sampler
        if sampler is not None:
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn,
                persistent_workers=False,
                generator=torch.Generator().manual_seed(42),
            )
        else:
            # Fallback to standard shuffling
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn,
                persistent_workers=False,
                generator=torch.Generator().manual_seed(42),
            )
    else:
        # For validation/test: no special sampling
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(42),
        )

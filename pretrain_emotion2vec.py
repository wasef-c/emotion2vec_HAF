import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import load_dataset, ClassLabel
from datetime import datetime
import tqdm
import argparse
from pathlib import Path
import sys
import io
from contextlib import contextmanager
import librosa

# Import your model
from models.hierarchical_attention import HierarchicalAttention


# Add suppression context manager and imports for emotion2vec
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


# Import emotion2vec after setting up suppression
from funasr import AutoModel

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration for pretraining - adapted from pretrain.py
PRETRAIN_CONFIG = {
    "learning_rate": 0.0001,
    "batch_size": 28,
    "epochs": 15,
    "patience": 5,  # Early stopping patience
    "use_focal_loss": True,
    "focal_loss_gamma": 2.0,
    "tasks": [
        {
            "name": "Gender",
            "type": "classification",
            "num_classes": 2,
            "class_weights": None,  # Will be set dynamically
        },
        {
            "name": "SpkrID",
            "type": "classification",
            "num_classes": None,  # Will be set based on dataset
            "class_weights": None,  # Will be set dynamically
        },
        {
            "name": "EmoVal",
            "type": "regression",
            "loss": "ccc",  # Concordance Correlation Coefficient
        },
        {
            "name": "EmoAct",
            "type": "regression",
            "loss": "ccc",  # Concordance Correlation Coefficient
        },
        {
            "name": "EmoDom",
            "type": "regression",
            "loss": "ccc",  # Concordance Correlation Coefficient
        },
    ],
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class CCCLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CCCLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        # Convert inputs to float and ensure correct shapes
        preds = preds.float().squeeze(-1)  # Remove last dimension if it's 1
        targets = targets.float()

        # Calculate means
        pred_mean = torch.mean(preds)
        target_mean = torch.mean(targets)

        # Calculate variances
        pred_var = torch.var(preds, unbiased=False)
        target_var = torch.var(targets, unbiased=False)

        # Calculate covariance
        cov = torch.mean((preds - pred_mean) * (targets - target_mean))

        # Calculate CCC
        ccc = (
            2 * cov / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
        )  # Add epsilon to avoid division by zero

        # Convert to loss (1 - CCC)
        loss = 1 - ccc

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MSPPodcastDataset(Dataset):
    """
    Custom dataset for pretraining that works with MSP-PODCAST data
    and extracts emotion2vec features directly from the HuggingFace dataset
    """

    def __init__(self, msp_data, labels, task_name, task_type):
        self.msp_data = msp_data  # This is the actual HuggingFace dataset split
        self.task_type = task_type
        self.task_name = task_name

        if task_type == "classification":
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:  # regression
            self.labels = torch.tensor(labels, dtype=torch.float32)

        # Initialize emotion2vec model for feature extraction
        with suppress_output():
            self.model = AutoModel(
                model="emotion2vec/emotion2vec_base",
                hub="hf",
            )

    def extract_features(self, audio_array, sampling_rate):
        """Extract frame-level features using emotion2vec base model."""
        # Resample to 16kHz if needed
        if sampling_rate != 16000:
            audio_array = librosa.resample(
                y=audio_array, orig_sr=sampling_rate, target_sr=16000
            )

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
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the corresponding item from MSP-PODCAST data
        item = self.msp_data[idx]

        # Extract emotion2vec features from audio
        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
        features = self.extract_features(audio_array, sampling_rate)

        return {"features": features, "label": self.labels[idx]}  # Shape: [T, 768]


def pretrain_collate_fn(batch):
    """
    Custom collate function for pretraining with MSP-PODCAST data.
    Handles variable-length emotion2vec feature sequences and different label types.
    """
    # Get max sequence length in the batch
    max_len = max(item["features"].size(0) for item in batch)
    batch_size = len(batch)

    # Initialize padded tensors
    features = torch.zeros(batch_size, max_len, 768)  # emotion2vec features are 768-dim

    # Check if labels are tensors (classification) or scalars (regression)
    first_label = batch[0]["label"]
    if isinstance(first_label, torch.Tensor):
        if first_label.dim() == 0:  # scalar tensor
            labels = torch.zeros(batch_size, dtype=first_label.dtype)
        else:
            labels = torch.zeros(
                batch_size, *first_label.shape, dtype=first_label.dtype
            )
    else:
        # Handle different label types
        if isinstance(first_label, int):
            labels = torch.zeros(batch_size, dtype=torch.long)
        else:  # float for regression
            labels = torch.zeros(batch_size, dtype=torch.float32)

    # Fill tensors with actual data
    for i, item in enumerate(batch):
        # Features - pad to max length
        seq_len = item["features"].size(0)
        features[i, :seq_len] = item["features"]

        # Labels
        if isinstance(item["label"], torch.Tensor):
            labels[i] = item["label"]
        else:
            labels[i] = torch.tensor(item["label"], dtype=labels.dtype)

    return {
        "features": features,
        "label": labels,
    }


def calculate_ccc(y_true, y_pred):
    """Calculate Concordance Correlation Coefficient."""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate means
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    # Calculate variances
    y_true_var = np.var(y_true, ddof=0)
    y_pred_var = np.var(y_pred, ddof=0)

    # Calculate covariance
    cov = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))

    # Calculate CCC
    ccc = 2 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)

    return ccc


def create_hierarchical_model_for_task(task_type, num_classes, pretrained_model=None):
    """
    Create a hierarchical attention model adapted for the specific task.
    The emotion2vec features remain frozen, only the attention layers are trained.
    """
    if pretrained_model is not None:
        # Load pretrained model and adapt for new task
        model = (
            HierarchicalAttention(input_dim=768, num_classes=4).to(device).float()
        )  # Initialize with default

        # Load state dict excluding task-specific layers
        state_dict = pretrained_model.state_dict()
        # Remove task-specific classifier weights
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("classifier.")
        }
        model.load_state_dict(state_dict, strict=False)

        # Update classifier for current task
        # Your model has hidden_dim = 384
        classifier_input_dim = model.hidden_dim  # 384

        if task_type == "classification":
            model.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_input_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(classifier_input_dim, num_classes),
            ).to(device)
        else:  # regression
            model.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_input_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(classifier_input_dim, 1),
            ).to(device)
    else:
        # Create new model with correct num_classes for the task
        if task_type == "classification":
            model = (
                HierarchicalAttention(input_dim=768, num_classes=num_classes)
                .to(device)
                .float()
            )
        else:  # regression - use 1 output
            model = (
                HierarchicalAttention(input_dim=768, num_classes=1).to(device).float()
            )

    return model


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    patience,
    output_dir,
    task_name,
    task_type,
):
    """Train and evaluate the model with early stopping."""
    best_val_metric = float("-inf") if task_type == "regression" else 0
    train_losses = []
    val_losses = []
    val_metrics = []

    # For early stopping
    patience_counter = 0

    # Create checkpoints directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints", task_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in tqdm.tqdm(
            train_loader, desc=f"{task_name} - Epoch {epoch+1}/{epochs} - Training"
        ):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward pass through hierarchical attention model
            logits, attention_info = model(features)

            # Add entropy regularization loss (from your original model)
            classification_loss = criterion(
                logits.squeeze() if task_type == "regression" else logits, labels
            )
            entropy_loss = attention_info["entropy_loss"]
            total_loss = classification_loss + entropy_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm.tqdm(
                val_loader, desc=f"{task_name} - Epoch {epoch+1}/{epochs} - Validation"
            ):
                features = batch["features"].to(device)
                labels = batch["label"].to(device)

                logits, attention_info = model(features)
                classification_loss = criterion(
                    logits.squeeze() if task_type == "regression" else logits, labels
                )
                entropy_loss = attention_info["entropy_loss"]
                total_loss = classification_loss + entropy_loss

                val_loss += total_loss.item()

                if task_type == "classification":
                    _, preds = torch.max(logits, 1)
                else:  # regression
                    preds = logits.squeeze()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Calculate metrics
        if task_type == "classification":
            val_metric = recall_score(all_labels, all_preds, average="macro")  # UAR
        else:  # regression
            if isinstance(criterion, CCCLoss):
                val_metric = calculate_ccc(all_labels, all_preds)  # CCC
            else:
                val_metric = -mean_squared_error(
                    all_labels, all_preds
                )  # Negative MSE for maximization

        val_metrics.append(val_metric)

        print(f"{task_name} - Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if task_type == "classification":
            print(f"Val UAR: {val_metric:.4f}")
        else:
            if isinstance(criterion, CCCLoss):
                print(f"Val CCC: {val_metric:.4f}")
            else:
                print(f"Val MSE: {-val_metric:.4f}")
            print(f"Val R2: {r2_score(all_labels, all_preds):.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metric": val_metric,
        }
        torch.save(
            checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Check if this is the best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0

            # Save best model
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{task_name}_best_model.pt"),
            )

            # Save visualization
            if task_type == "classification":
                # Save confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix - {task_name} - Best Model")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.savefig(
                    os.path.join(output_dir, f"{task_name}_confusion_matrix.png")
                )
                plt.close()
            else:
                # Save regression plot
                plt.figure(figsize=(10, 8))
                plt.scatter(all_labels, all_preds, alpha=0.5)
                plt.plot(
                    [min(all_labels), max(all_labels)],
                    [min(all_labels), max(all_labels)],
                    "r--",
                )
                plt.title(f"Regression Plot - {task_name} - Best Model")
                plt.xlabel("True Values")
                plt.ylabel("Predicted Values")
                plt.savefig(
                    os.path.join(output_dir, f"{task_name}_regression_plot.png")
                )
                plt.close()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} as metric hasn't improved for {patience} epochs"
                )
                break

    return best_val_metric, val_metrics[-1]


def get_task_index(task_name):
    """Get the index of a task in the PRETRAIN_CONFIG tasks list."""
    for idx, task in enumerate(PRETRAIN_CONFIG["tasks"]):
        if task["name"] == task_name:
            return idx
    raise ValueError(f"Task '{task_name}' not found in configuration")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Pretrain hierarchical attention model starting from a specific task"
    )
    parser.add_argument(
        "--start_task",
        type=str,
        default="Gender",
        help='Name of the task to start from (e.g., "Gender", "SpkrID", "EmoVal")',
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=None,
        help="Directory containing the pre-trained model checkpoint (optional)",
    )
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"pretrain_hierarchical_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, "pretrain_config.json"), "w") as f:
        json.dump(PRETRAIN_CONFIG, f, indent=4)

    # Get starting task index
    try:
        start_task_idx = get_task_index(args.start_task)
    except ValueError as e:
        print(f"Error: {e}")
        print("Available tasks:", [task["name"] for task in PRETRAIN_CONFIG["tasks"]])
        return

    # Load MSP-PODCAST dataset
    print("Loading MSP-PODCAST dataset...")
    dataset = load_dataset("cairocode/MSPP_WAV_speaker_split")

    # Split train set into train and validation (80/20)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Initialize model for the starting task
    task = PRETRAIN_CONFIG["tasks"][start_task_idx]
    task_name = task["name"]
    task_type = task["type"]

    # Load pre-trained model if available
    pretrained_model = None
    if args.pretrained_dir and start_task_idx > 0:
        prev_task = PRETRAIN_CONFIG["tasks"][start_task_idx - 1]
        prev_task_name = prev_task["name"]
        prev_model_path = os.path.join(
            args.pretrained_dir, f"{prev_task_name}_best_model.pt"
        )

        if os.path.exists(prev_model_path):
            print(f"Loading pre-trained model from: {prev_model_path}")
            pretrained_model = HierarchicalAttention(input_dim=768).to(device).float()

            # Load the state dict
            checkpoint = torch.load(prev_model_path)

            # Filter out classifier parameters
            model_state = {
                k: v for k, v in checkpoint.items() if not k.startswith("classifier")
            }

            # Load with strict=False to ignore missing keys
            pretrained_model.load_state_dict(model_state, strict=False)
        else:
            print(f"Pre-trained model not found at {prev_model_path}. Starting fresh.")

    # Process remaining tasks starting from the specified task
    for task_idx, task in enumerate(
        PRETRAIN_CONFIG["tasks"][start_task_idx:], start=start_task_idx
    ):
        task_name = task["name"]
        task_type = task["type"]
        print(f"\n=== Starting pretraining for {task_name} ({task_type}) ===")

        # Prepare labels based on task
        if task_name == "Gender":
            train_labels = np.array(
                [1 if x["Gender"] == "Male" else 0 for x in train_dataset]
            )
            val_labels = np.array(
                [1 if x["Gender"] == "Male" else 0 for x in val_dataset]
            )
            test_labels = np.array(
                [1 if x["Gender"] == "Male" else 0 for x in test_dataset]
            )
            num_classes = 2
        elif task_name == "SpkrID":
            # Count samples per speaker in training set
            speaker_counts = {}
            for x in train_dataset:
                speaker_id = x["SpkrID"]
                speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

            # Select speakers with at least 2 samples in training set
            valid_speakers = sorted(
                [spk for spk, count in speaker_counts.items() if count >= 2]
            )
            if len(valid_speakers) > 200:
                valid_speakers = valid_speakers[:200]

            print(
                f"Selected {len(valid_speakers)} speakers with sufficient samples (>=2)"
            )

            # Create mapping from original speaker IDs to consecutive IDs starting from 0
            speaker_id_map = {
                old_id: new_id for new_id, old_id in enumerate(valid_speakers)
            }

            # Filter training data and split into train/val
            filtered_train_data = []
            train_labels_list = []
            for x in train_dataset:
                if x["SpkrID"] in valid_speakers:
                    filtered_train_data.append(x)
                    train_labels_list.append(speaker_id_map[x["SpkrID"]])

            # Split the filtered training data into train/val (e.g., 80/20 split)

            train_data, val_data, train_labels, val_labels = train_test_split(
                filtered_train_data,
                train_labels_list,
                test_size=0.2,  # 20% for validation
                random_state=42,
                stratify=train_labels_list,  # Ensure balanced split across speakers
            )

            # Use validation set as test set (same speakers, same data)
            test_data = val_data.copy()
            test_labels = val_labels.copy()

            # Print dataset sizes for debugging
            print(f"Original dataset sizes:")
            print(
                f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
            )
            print(f"New dataset sizes:")
            print(
                f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
            )

            # Convert to numpy arrays
            train_labels = np.array(train_labels)
            val_labels = np.array(val_labels)
            test_labels = np.array(test_labels)
            num_classes = len(valid_speakers)

            # Update datasets to use new splits
            train_dataset = train_data
            val_dataset = val_data
            test_dataset = test_data
        elif task_name == "SpkrID2":
            # Count samples per speaker
            speaker_counts = {}
            for x in train_dataset:
                speaker_id = x["SpkrID"]
                speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

            # Select speakers with at least 2 samples in training set
            valid_speakers = sorted(
                [spk for spk, count in speaker_counts.items() if count >= 2]
            )
            if len(valid_speakers) > 200:
                valid_speakers = valid_speakers[:200]
            print(
                f"Selected {len(valid_speakers)} speakers with sufficient samples (>=2)"
            )

            # Create mapping from original speaker IDs to consecutive IDs starting from 0
            speaker_id_map = {
                old_id: new_id for new_id, old_id in enumerate(valid_speakers)
            }

            # Filter and remap speaker IDs for all splits
            def filter_and_remap_dataset(dataset_split):
                filtered_data = []
                labels = []
                for x in dataset_split:
                    if x["SpkrID"] in valid_speakers:
                        filtered_data.append(x)
                        labels.append(speaker_id_map[x["SpkrID"]])
                return filtered_data, labels

            train_data, train_labels = filter_and_remap_dataset(train_dataset)
            val_data, val_labels = filter_and_remap_dataset(val_dataset)
            test_data, test_labels = filter_and_remap_dataset(test_dataset)

            # Print dataset sizes for debugging
            print(f"Original dataset sizes:")
            print(
                f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
            )
            print(f"Filtered dataset sizes:")
            print(
                f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
            )

            # Check if any dataset is empty after filtering
            if len(train_data) == 0:
                raise ValueError("Training set is empty after filtering speakers")
            if len(val_data) == 0:
                raise ValueError("Validation set is empty after filtering speakers")
            if len(test_data) == 0:
                raise ValueError("Test set is empty after filtering speakers")

            train_labels = np.array(train_labels)
            val_labels = np.array(val_labels)
            test_labels = np.array(test_labels)
            num_classes = len(valid_speakers)

            # Update datasets to use filtered data
            train_dataset = train_data
            val_dataset = val_data
            test_dataset = test_data

        else:  # EmoVal, EmoAct, or EmoDom
            train_labels = np.array([x[task_name] for x in train_dataset])
            val_labels = np.array([x[task_name] for x in val_dataset])
            test_labels = np.array([x[task_name] for x in test_dataset])
            num_classes = 1

        # Create model for this task
        model = create_hierarchical_model_for_task(
            task_type, num_classes, pretrained_model
        )

        # Create datasets using MSPPodcastDataset
        train_dataset_obj = MSPPodcastDataset(
            train_dataset, train_labels, task_name, task_type
        )
        val_dataset_obj = MSPPodcastDataset(
            val_dataset, val_labels, task_name, task_type
        )
        test_dataset_obj = MSPPodcastDataset(
            test_dataset, test_labels, task_name, task_type
        )

        # Create data loaders with the pretraining collate function
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=PRETRAIN_CONFIG["batch_size"],
            shuffle=True,
            collate_fn=pretrain_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset_obj,
            batch_size=PRETRAIN_CONFIG["batch_size"],
            shuffle=False,
            collate_fn=pretrain_collate_fn,
        )
        test_loader = DataLoader(
            test_dataset_obj,
            batch_size=PRETRAIN_CONFIG["batch_size"],
            shuffle=False,
            collate_fn=pretrain_collate_fn,
        )

        # Loss function and optimizer
        if task_type == "classification":
            class_counts = np.bincount(train_labels)
            total_samples = len(train_labels)
            class_weights = torch.FloatTensor(
                total_samples / (len(class_counts) * class_counts)
            )
            class_weights = class_weights.to(device)

            if PRETRAIN_CONFIG["use_focal_loss"]:
                criterion = FocalLoss(
                    alpha=class_weights, gamma=PRETRAIN_CONFIG["focal_loss_gamma"]
                )
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:  # regression
            if task["loss"] == "ccc":
                criterion = CCCLoss()
            else:
                criterion = nn.MSELoss()

        optimizer = optim.AdamW(
            model.parameters(), lr=PRETRAIN_CONFIG["learning_rate"], weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3, factor=0.5
        )

        try:
            # Train model
            best_val_metric, final_metric = train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                PRETRAIN_CONFIG["epochs"],
                PRETRAIN_CONFIG["patience"],
                output_dir,
                task_name,
                task_type,
            )

            # Load best model for testing
            model.load_state_dict(
                torch.load(os.path.join(output_dir, f"{task_name}_best_model.pt"))
            )
            model.eval()

            # Perform inference on test set
            print(f"\n=== Testing {task_name} on test set ===")
            all_test_preds = []
            all_test_labels = []
            test_loss = 0

            with torch.no_grad():
                for batch in tqdm.tqdm(test_loader, desc="Testing"):
                    features = batch["features"].to(device)
                    labels = batch["label"].to(device)

                    logits, attention_info = model(features)
                    classification_loss = criterion(
                        logits.squeeze() if task_type == "regression" else logits,
                        labels,
                    )
                    entropy_loss = attention_info["entropy_loss"]
                    total_loss = classification_loss + entropy_loss
                    test_loss += total_loss.item()

                    if task_type == "classification":
                        _, preds = torch.max(logits, 1)
                    else:  # regression
                        preds = logits.squeeze()
                        # Ensure preds is at least 1D for numpy conversion
                        if preds.dim() == 0:
                            preds = preds.unsqueeze(0)

                    all_test_preds.extend(preds.cpu().numpy())
                    all_test_labels.extend(labels.cpu().numpy())

            test_loss /= len(test_loader)

            # Calculate and print test metrics
            if task_type == "classification":
                test_uar = recall_score(
                    all_test_labels, all_test_preds, average="macro"
                )
                print(f"Test Loss: {test_loss:.4f}")
                print(f"Test UAR: {test_uar:.4f}")
                print("\nClassification Report:")
                print(classification_report(all_test_labels, all_test_preds))

                # Save test confusion matrix
                cm = confusion_matrix(all_test_labels, all_test_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Test Confusion Matrix - {task_name}")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.savefig(
                    os.path.join(output_dir, f"{task_name}_test_confusion_matrix.png")
                )
                plt.close()
            else:
                if isinstance(criterion, CCCLoss):
                    test_ccc = calculate_ccc(all_test_labels, all_test_preds)
                    print(f"Test Loss: {test_loss:.4f}")
                    print(f"Test CCC: {test_ccc:.4f}")
                    print(f"Test R2: {r2_score(all_test_labels, all_test_preds):.4f}")

                    # Save test regression plot
                    plt.figure(figsize=(10, 8))
                    plt.scatter(all_test_labels, all_test_preds, alpha=0.5)
                    plt.plot(
                        [min(all_test_labels), max(all_test_labels)],
                        [min(all_test_labels), max(all_test_labels)],
                        "r--",
                    )
                    plt.title(f"Test Regression Plot - {task_name}")
                    plt.xlabel("True Values")
                    plt.ylabel("Predicted Values")
                    plt.savefig(
                        os.path.join(
                            output_dir, f"{task_name}_test_regression_plot.png"
                        )
                    )
                    plt.close()

            # Save test results
            test_results = {
                "task_name": task_name,
                "task_type": task_type,
                "test_loss": float(test_loss),
                "test_metrics": {
                    "uar" if task_type == "classification" else "ccc": float(
                        test_uar if task_type == "classification" else test_ccc
                    )
                },
            }

            with open(
                os.path.join(output_dir, f"{task_name}_test_results.json"), "w"
            ) as f:
                json.dump(test_results, f, indent=4)

            # Set current model as pretrained for next task
            pretrained_model = model

        except Exception as e:
            print(f"Error during training task {task_name}: {e}")
            raise e

    print("\nPretraining completed. Results saved to:", output_dir)


if __name__ == "__main__":
    main()

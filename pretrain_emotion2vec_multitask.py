#!/usr/bin/env python
"""pretrain_emotion2vec_multitask.py

Multi-task pre-training for the HierarchicalAttention encoder.
Unlike the original `pretrain_emotion2vec.py`, this script keeps a *single* encoder
(backbone + fusion) and attaches a separate head (classifier/regressor) for each
configured task via a `ModuleDict`.  The backbone is never re-initialised, so
knowledge accumulates across tasks.  At the end of training we export:

1.  `multitask_backbone.pt` – **encoder only** (no heads) – the file you will
    load in your downstream LOSO script.
2.  `{task_name}_best_model.pt` – full model (encoder + head) for each task.

The code re-uses the dataset and utility classes from the original file but
re-organises the model pipeline.
"""

# NOTE: Only *new* or *changed* code is shown here.  Everything else (imports,
# dataset classes, loss functions, etc.) is identical to the original file and
# can be copied verbatim from `pretrain_emotion2vec.py`.

# --- New / refactored sections -------------------------------------------------

import os
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import recall_score, mean_squared_error, r2_score, f1_score

from models.hierarchical_attention import HierarchicalAttention

# -----------------------------------------------------------------------------
# Additional utilities copied from the original single-task script so that this
# file is self-contained and executable.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt  # kept for parity with original utils
from torch.utils.data import Dataset
from contextlib import contextmanager
import sys, io
import librosa
from funasr import AutoModel

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration (matches original)
PRETRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 28,
    "epochs": 20,
    "patience": 5,
    "use_focal_loss": True,
    "focal_loss_gamma": 2.0,
    "tasks": [
        {"name": "Gender", "type": "classification", "num_classes": 2},
        {"name": "EmoAct", "type": "regression", "loss": "ccc"},
        {"name": "EmoVal", "type": "regression", "loss": "ccc"},
        {"name": "EmoDom", "type": "regression", "loss": "ccc"},
    ],
}


# Suppress noisy console output from FunASR downloads
@contextmanager
def suppress_output():
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CCCLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        preds, targets = preds.float().squeeze(-1), targets.float()
        pred_mean, target_mean = torch.mean(preds), torch.mean(targets)
        pred_var, target_var = torch.var(preds, unbiased=False), torch.var(
            targets, unbiased=False
        )
        cov = torch.mean((preds - pred_mean) * (targets - target_mean))
        ccc = 2 * cov / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
        loss = 1 - ccc
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MSPPodcastDataset(Dataset):
    """MSP-PODCAST split that extracts emotion2vec frame embeddings on the fly."""

    def __init__(self, msp_data, labels, task_name: str, task_type: str):
        self.msp_data = msp_data
        self.task_type = task_type
        self.task_name = task_name
        self.labels = torch.tensor(
            labels, dtype=torch.long if task_type == "classification" else torch.float32
        )
        with suppress_output():
            self.model = AutoModel(model="emotion2vec/emotion2vec_base", hub="hf")

    def extract_features(self, audio_arr, sr):
        if sr != 16000:
            audio_arr = librosa.resample(y=audio_arr, orig_sr=sr, target_sr=16000)
        audio_tensor = torch.tensor(audio_arr, dtype=torch.float32)
        with suppress_output():
            rec = self.model.generate(
                audio_tensor,
                output_dir=None,
                granularity="frame",
                extract_embedding=True,
                sr=16000,
            )
        return torch.from_numpy(rec[0]["feats"]).float()  # [T, 768]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.msp_data[idx]
        feats = self.extract_features(
            item["audio"]["array"], item["audio"]["sampling_rate"]
        )
        return {"features": feats, "label": self.labels[idx]}


def pretrain_collate_fn(batch):
    max_len = max(b["features"].size(0) for b in batch)
    batch_size = len(batch)
    features = torch.zeros(batch_size, max_len, 768)
    first_label = batch[0]["label"]
    labels = (
        torch.zeros(batch_size, *first_label.shape, dtype=first_label.dtype)
        if isinstance(first_label, torch.Tensor)
        else torch.zeros(
            batch_size,
            dtype=torch.long if isinstance(first_label, int) else torch.float32,
        )
    )
    for i, b in enumerate(batch):
        seq_len = b["features"].size(0)
        features[i, :seq_len] = b["features"]
        labels[i] = b["label"]
    return {"features": features, "label": labels}


def calculate_ccc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return 2 * cov / (y_true.var() + y_pred.var() + (mean_true - mean_pred) ** 2)


# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Model wrapper with a persistent encoder and a per-task head dictionary
# ------------------------------------------------------------------------------


class MultiTaskHierarchical(nn.Module):
    """HierarchicalAttention backbone + ModuleDict of task-specific heads."""

    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        # Build encoder and strip its internal classifier so that it outputs the
        # fused representation directly.
        self.encoder = HierarchicalAttention(input_dim=768, num_classes=4)
        self.encoder.classifier = nn.Identity()  # logits == fused features
        self.hidden_dim = self.encoder.hidden_dim
        self.heads = nn.ModuleDict()

    def to(self, device):
        """Override to() to ensure all components are moved to the device."""
        super().to(device)
        self.encoder = self.encoder.to(device)
        for task_name in self.heads:
            self.heads[task_name] = self.heads[task_name].to(device)
        return self

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def add_head(self, task_name: str, out_dim: int):
        """Add a new task-specific MLP head if it does not already exist."""
        if task_name in self.heads:
            return  # already present

        self.heads[task_name] = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, out_dim),
        )
        # Move the new head to the same device as the model
        if hasattr(self, "device"):
            self.heads[task_name] = self.heads[task_name].to(self.device)

    def forward(self, x: torch.Tensor, task_name: str):
        """Run forward pass for a given task (returns logits and attentions)."""
        features, attn_info = self.encoder(x)  # features shape: [B, hidden_dim]
        logits = self.heads[task_name](features)
        return logits, attn_info


# ------------------------------------------------------------------------------
# Training / evaluation helpers (slimmed version of original loops)
# ------------------------------------------------------------------------------


def train_one_epoch(
    model: MultiTaskHierarchical,
    task_name: str,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        feats = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, attn_info = model(feats, task_name)

        cls_loss = criterion(logits.squeeze() if logits.ndim == 1 else logits, labels)
        total_loss = cls_loss + attn_info["entropy_loss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += total_loss.item()
        if logits.ndim == 1:
            preds = logits.detach()
        else:
            preds = torch.argmax(logits.detach(), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    return epoch_loss, all_preds, all_labels


def evaluate(
    model: MultiTaskHierarchical,
    task_name: str,
    loader: DataLoader,
    criterion: nn.Module,
    task_type: str,
):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits, attn_info = model(feats, task_name)
            cls_loss = criterion(
                logits.squeeze() if logits.ndim == 1 else logits, labels
            )
            total_loss = cls_loss + attn_info["entropy_loss"]
            running_loss += total_loss.item()

            if task_type == "classification":
                preds = torch.argmax(logits, dim=1)
            else:
                preds = logits.squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)

    # Calculate metrics based on task type
    metrics = {}
    if task_type == "classification":
        metrics["uar"] = recall_score(all_labels, all_preds, average="macro")
        metrics["accuracy"] = np.mean(np.array(all_preds) == np.array(all_labels))
        metrics["f1"] = f1_score(all_labels, all_preds, average="weighted")
    else:
        metrics["ccc"] = calculate_ccc(all_labels, all_preds)
        metrics["mse"] = mean_squared_error(all_labels, all_preds)
        metrics["r2"] = r2_score(all_labels, all_preds)

    return epoch_loss, metrics


# ------------------------------------------------------------------------------
# Main training routine
# ------------------------------------------------------------------------------


def main():
    import argparse
    from datasets import load_dataset
    import numpy as np

    parser = argparse.ArgumentParser(description="Multi-task pre-training script")
    parser.add_argument(
        "--start_task",
        type=str,
        default="Gender",
        help="Task name to start from (see PRETRAIN_CONFIG)",
    )
    args = parser.parse_args()

    # Timestamped output directory
    output_dir = Path(
        f"pretrain_hierarchical_multitask_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "pretrain_config.json", "w") as f:
        json.dump(PRETRAIN_CONFIG, f, indent=4)

    # Load dataset (same split logic as original)
    print("Loading MSP-PODCAST dataset (speaker split)…")
    dataset = load_dataset("cairocode/MSPP_WAV_speaker_split")
    train_raw, val_raw, test_raw = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )

    # Build model and move to device
    model = MultiTaskHierarchical().to(device)
    model.device = device  # Store device for later use

    # ------------------------------------------------------------------
    # Build heads and per-task criteria (all tasks active simultaneously)
    # ------------------------------------------------------------------
    tasks = PRETRAIN_CONFIG["tasks"]  # Use full list (order irrelevant now)

    # Add heads + criterion dicts
    criterion_dict = {}
    for t_cfg in tasks:
        t_name, t_type = t_cfg["name"], t_cfg["type"]
        # Determine output dim (regression -> 1)
        out_dim = t_cfg.get("num_classes", 1) if t_type == "classification" else 1
        model.add_head(t_name, out_dim)
        # Criterion
        if t_type == "classification":
            # Compute class_weights from train set below – placeholder None for now
            criterion_dict[t_name] = None  # will fill after dataset is built
        else:
            criterion_dict[t_name] = (
                CCCLoss() if t_cfg.get("loss", "ccc") == "ccc" else nn.MSELoss()
            ).to(
                device
            )  # Move criterion to device immediately

    # ------------------------------------------------------------------
    # Multi-task Dataset / DataLoaders
    # ------------------------------------------------------------------
    class MSPPodcastMultiTaskDataset(Dataset):
        """Return features & label dict for *all* configured tasks."""

        def __init__(self, split_data):
            self.data = split_data
            with suppress_output():
                self.e2v = AutoModel(model="emotion2vec/emotion2vec_base", hub="hf")

        def extract(self, arr, sr):
            if sr != 16000:
                arr = librosa.resample(y=arr, orig_sr=sr, target_sr=16000)
            with suppress_output():
                rec = self.e2v.generate(
                    torch.tensor(arr, dtype=torch.float32),
                    output_dir=None,
                    granularity="frame",
                    extract_embedding=True,
                    sr=16000,
                )
            return torch.from_numpy(rec[0]["feats"]).float()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            feats = self.extract(item["audio"]["array"], item["audio"]["sampling_rate"])
            label_dict = {
                "Gender": torch.tensor(
                    1 if item["Gender"] == "Male" else 0, dtype=torch.long
                ),
                "EmoAct": torch.tensor(item["EmoAct"], dtype=torch.float32),
                "EmoVal": torch.tensor(item["EmoVal"], dtype=torch.float32),
                "EmoDom": torch.tensor(item["EmoDom"], dtype=torch.float32),
            }
            return {"features": feats, "labels": label_dict}

    def multitask_collate(batch):
        max_len = max(b["features"].size(0) for b in batch)
        bs = len(batch)
        feat_tensor = torch.zeros(bs, max_len, 768)
        label_batch = {t["name"]: [] for t in tasks}
        for i, b in enumerate(batch):
            l = b["features"].size(0)
            feat_tensor[i, :l] = b["features"]
            for k in label_batch.keys():
                label_batch[k].append(b["labels"][k])
        # Stack labels
        for k in label_batch.keys():
            label_batch[k] = torch.stack(label_batch[k])
        return {"features": feat_tensor, "labels": label_batch}

    train_ds = MSPPodcastMultiTaskDataset(train_raw)
    val_ds = MSPPodcastMultiTaskDataset(val_raw)

    # Initialize criterion for Gender classification
    criterion_dict["Gender"] = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=PRETRAIN_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=multitask_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=PRETRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=multitask_collate,
    )

    # ------------------------------------------------------------------
    # Optimizer – encoder lower LR, all heads higher LR (same)
    # ------------------------------------------------------------------
    head_params = []
    for t_name in model.heads:
        head_params.extend(list(model.heads[t_name].parameters()))
    optimizer = optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": PRETRAIN_CONFIG["learning_rate"] / 10,
            },
            {"params": head_params, "lr": PRETRAIN_CONFIG["learning_rate"]},
        ],
        weight_decay=0.01,
    )

    # Initialize metrics tracking
    best_metrics = {t_cfg["name"]: float("-inf") for t_cfg in tasks}
    task_metrics_history = {t_cfg["name"]: [] for t_cfg in tasks}

    # ------------------------------------------------------------------
    # Training loop (joint multi-task)
    # ------------------------------------------------------------------
    best_val_loss, patience_counter = float("inf"), 0
    for epoch in range(PRETRAIN_CONFIG["epochs"]):
        # ---- Train ----
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            feats = batch["features"].to(device)
            label_dict = {k: v.to(device) for k, v in batch["labels"].items()}
            optimizer.zero_grad()
            # Single forward through encoder
            fused, attn_info = model.encoder(feats)
            loss = attn_info["entropy_loss"]
            for t_cfg in tasks:
                t_name, t_type = t_cfg["name"], t_cfg["type"]
                logits = model.heads[t_name](fused)
                if t_type == "classification":
                    l = criterion_dict[t_name](logits, label_dict[t_name])
                else:
                    l = criterion_dict[t_name](logits.squeeze(), label_dict[t_name])
                loss += l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0
        epoch_metrics = {t_cfg["name"]: {} for t_cfg in tasks}

        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                label_dict = {k: v.to(device) for k, v in batch["labels"].items()}
                fused, attn_info = model.encoder(feats)
                loss = attn_info["entropy_loss"]

                for t_cfg in tasks:
                    t_name, t_type = t_cfg["name"], t_cfg["type"]
                    logits = model.heads[t_name](fused)
                    if t_type == "classification":
                        l = criterion_dict[t_name](logits, label_dict[t_name])
                    else:
                        l = criterion_dict[t_name](logits.squeeze(), label_dict[t_name])
                    loss += l

                    # Calculate task-specific metrics
                    if t_type == "classification":
                        preds = torch.argmax(logits, dim=1)
                    else:
                        preds = logits.squeeze()

                    task_loss, task_metrics = evaluate(
                        model,
                        t_name,
                        DataLoader(
                            [{"features": feats, "label": label_dict[t_name]}],
                            batch_size=1,
                            collate_fn=multitask_collate,
                        ),
                        criterion_dict[t_name],
                        t_type,
                    )
                    epoch_metrics[t_name].update(task_metrics)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{PRETRAIN_CONFIG['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print("\nTask-specific Validation Metrics:")
        for t_cfg in tasks:
            t_name = t_cfg["name"]
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in epoch_metrics[t_name].items()
            )
            print(f"{t_name}: {metrics_str}")

            # Track best metrics
            if t_cfg["type"] == "classification":
                if epoch_metrics[t_name]["uar"] > best_metrics[t_name]:
                    best_metrics[t_name] = epoch_metrics[t_name]["uar"]
            else:
                if epoch_metrics[t_name]["ccc"] > best_metrics[t_name]:
                    best_metrics[t_name] = epoch_metrics[t_name]["ccc"]

            # Store metrics history
            task_metrics_history[t_name].append(epoch_metrics[t_name])

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "multitask_best_model.pt")
            print("\nSaved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= PRETRAIN_CONFIG["patience"]:
                print("\nEarly stopping triggered.")
                break

    # After training loop, evaluate on test set
    print("\nEvaluating on Test Set...")
    test_ds = MSPPodcastMultiTaskDataset(test_raw)
    test_loader = DataLoader(
        test_ds,
        batch_size=PRETRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=multitask_collate,
    )

    # Load best model for final evaluation
    model.load_state_dict(torch.load(output_dir / "multitask_best_model.pt"))
    model.eval()

    test_metrics = {t_cfg["name"]: {} for t_cfg in tasks}
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            label_dict = {k: v.to(device) for k, v in batch["labels"].items()}
            fused, attn_info = model.encoder(feats)

            for t_cfg in tasks:
                t_name, t_type = t_cfg["name"], t_cfg["type"]
                logits = model.heads[t_name](fused)

                # Calculate task-specific metrics
                if t_type == "classification":
                    preds = torch.argmax(logits, dim=1)
                else:
                    preds = logits.squeeze()

                task_loss, task_metrics = evaluate(
                    model,
                    t_name,
                    DataLoader(
                        [{"features": feats, "label": label_dict[t_name]}],
                        batch_size=1,
                        collate_fn=multitask_collate,
                    ),
                    criterion_dict[t_name],
                    t_type,
                )
                # Update test metrics (average across batches)
                for metric_name, value in task_metrics.items():
                    if metric_name not in test_metrics[t_name]:
                        test_metrics[t_name][metric_name] = []
                    test_metrics[t_name][metric_name].append(value)

    # Average test metrics across batches
    for t_name in test_metrics:
        for metric_name in test_metrics[t_name]:
            test_metrics[t_name][metric_name] = np.mean(
                test_metrics[t_name][metric_name]
            )

    # Save final metrics including test results
    final_metrics = {
        "best_metrics": best_metrics,
        "task_metrics_history": task_metrics_history,
        "test_metrics": test_metrics,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # Print final results with test metrics
    print("\nFinal Results:")
    print("=" * 50)
    for t_cfg in tasks:
        t_name = t_cfg["name"]
        print(f"\n{t_name} Task:")
        print(
            f"Best {'UAR' if t_cfg['type'] == 'classification' else 'CCC'} (Validation): {best_metrics[t_name]:.4f}"
        )
        print("\nTest Set Metrics:")
        for metric_name, value in test_metrics[t_name].items():
            print(f"{metric_name}: {value:.4f}")

    # Save backbone
    backbone_state = {k: v for k, v in model.encoder.state_dict().items()}
    torch.save(backbone_state, output_dir / "multitask_backbone.pt")
    print(
        "\nTraining finished. Backbone saved to", output_dir / "multitask_backbone.pt"
    )


if __name__ == "__main__":
    main()

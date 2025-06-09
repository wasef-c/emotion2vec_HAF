import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_recall_fscore_support,
)

from models.hierarchical_attention import HierarchicalAttention
from data.dataset import EmotionDataset


def calculate_metrics(y_true, y_pred):
    """Calculate UA, WA, F1, and UAR metrics."""
    # Weighted Accuracy (WA) - same as standard accuracy
    wa = accuracy_score(y_true, y_pred)

    # Unweighted Accuracy (UA) - average recall per class
    ua = recall_score(y_true, y_pred, average="macro")

    # F1 Score (macro average)
    f1 = f1_score(y_true, y_pred, average="macro")

    # Unweighted Average Recall (UAR) - same as UA for balanced datasets
    uar = recall_score(y_true, y_pred, average="macro")

    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred
    )

    return {
        "wa": wa,
        "ua": ua,
        "f1": f1,
        "uar": uar,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support_per_class": support.tolist(),
    }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, attention_info = model(features)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": loss.item()})

    epoch_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics, attention_info


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits, attention_info = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return val_loss, metrics, conf_matrix, attention_info


def main():
    # Initialize wandb
    wandb.init(project="emotion-recognition-hierarchical")

    # Configuration
    config = {
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    wandb.config.update(config)

    # Create save directory
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    # Initialize datasets
    train_dataset = EmotionDataset("IEMOCAP", split="train")
    val_dataset = EmotionDataset("MSP-IMPROV", split="train")  # Cross-corpus evaluation

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # Initialize model
    model = HierarchicalAttention().to(config["device"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    # Training loop
    best_metrics = {"ua": 0, "wa": 0, "f1": 0, "uar": 0}
    emotion_classes = ["neutral", "happy", "sad", "anger"]

    for epoch in range(config["num_epochs"]):
        # Training
        train_loss, train_metrics, train_attention = train_epoch(
            model, train_loader, criterion, optimizer, config["device"], epoch
        )

        # Validation
        val_loss, val_metrics, conf_matrix, val_attention = evaluate(
            model, val_loader, criterion, config["device"]
        )

        # Update learning rate
        scheduler.step()

        # Log metrics
        wandb.log(
            {
                "train_loss": train_loss,
                "train_wa": train_metrics["wa"],
                "train_ua": train_metrics["ua"],
                "train_f1": train_metrics["f1"],
                "train_uar": train_metrics["uar"],
                "val_loss": val_loss,
                "val_wa": val_metrics["wa"],
                "val_ua": val_metrics["ua"],
                "val_f1": val_metrics["f1"],
                "val_uar": val_metrics["uar"],
                "learning_rate": scheduler.get_last_lr()[0],
                "fusion_weights": val_attention["fusion_weights"].tolist(),
            }
        )

        # Log per-class metrics
        for i, emotion in enumerate(emotion_classes):
            wandb.log(
                {
                    f"val_{emotion}_precision": val_metrics["precision_per_class"][i],
                    f"val_{emotion}_recall": val_metrics["recall_per_class"][i],
                    f"val_{emotion}_f1": val_metrics["f1_per_class"][i],
                }
            )

        # Save best model based on UAR (primary metric for imbalanced emotion recognition)
        if val_metrics["uar"] > best_metrics["uar"]:
            best_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "config": config,
                },
                save_dir / "best_model.pt",
            )

        print(
            f"Epoch {epoch}:\n"
            f"Train: WA={train_metrics['wa']:.4f}, UA={train_metrics['ua']:.4f}, "
            f"F1={train_metrics['f1']:.4f}, UAR={train_metrics['uar']:.4f}\n"
            f"Val: WA={val_metrics['wa']:.4f}, UA={val_metrics['ua']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, UAR={val_metrics['uar']:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()

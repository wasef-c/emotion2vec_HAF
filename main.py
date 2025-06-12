from data.dataset import EmotionDataset, collate_fn
from models.hierarchical_attention import HierarchicalAttention
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, Sampler
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
import json
from datasets import load_dataset
import torch.multiprocessing as mp
import os
import logging
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict


# Disable all possible FunASR output
os.environ["FUNASR_DISABLE_PROGRESS_BAR"] = "1"
os.environ["FUNASR_QUIET"] = "1"
os.environ["FUNASR_LOG_LEVEL"] = "ERROR"
os.environ["FUNASR_DISABLE_TQDM"] = "1"
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method("spawn", force=True)
pretrain_dir = "/home/rml/Documents/pythontest/emotion2vec_HA/pretrain_hierarchical_multitask_20250611_104521/multitask_backbone.pt"
SD = False
CURR = True  # Flag for curriculum learning
experiment_name = "pretrain_N1p1_H2p5_noSD"


def calculate_uar(cm):
    """Calculate Unweighted Average Recall (UAR) from confusion matrix."""
    recalls = cm.diagonal() / cm.sum(axis=1)
    return np.mean(recalls)


def calculate_wa(cm):
    """
    Calculate Weighted Accuracy (WA) from confusion matrix.
    WA = sum over classes of (support_i * recall_i) / total_samples
    """
    # support (number of true samples per class) is the sum over rows
    support = cm.sum(axis=1)
    # recall per class: TP / (TP + FN)
    recall_per_class = np.diag(cm) / support
    # weighted sum of recalls
    weighted_accuracy = np.sum(recall_per_class * support) / np.sum(cm)
    return weighted_accuracy


def calculate_comprehensive_metrics(all_labels, all_preds, class_names=None):
    """
    Calculate all emotion recognition metrics: Accuracy, UAR, WA, F1
    """
    if class_names is None:
        class_names = ["neutral", "happy", "sad", "anger"]

    # Basic metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)

    # Advanced metrics
    uar = calculate_uar(cm)
    # This should equal accuracy, but good for verification
    wa = calculate_wa(cm)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    # Class-wise accuracies (recalls)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    return {
        "accuracy": accuracy,
        "wa": wa,  # Weighted Accuracy
        "uar": uar,  # Unweighted Average Recall
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_accuracies": {
            class_names[i]: float(acc) for i, acc in enumerate(class_accuracies)
        },
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

        # Get the actual dataset (handle Subset case)
        actual_dataset = dataset.dataset if isinstance(
            dataset, Subset) else dataset

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
                batch = indices[i: i + self.batch_size]
                if len(batch) > 0:  # Only add non-empty batches
                    self.batches.append(batch)

    def __iter__(self):
        for batch in self.batches:
            yield from batch

    def __len__(self):
        return len(self.dataset)


def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def enhanced_evaluate(
    model,
    val_loader,
    criterion,
    device,
    epoch=None,
    save_dir=None,
    eval_type="held_out",
    session_info=None,
):
    """Enhanced evaluation with comprehensive metrics including WA"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Track learnable parameters
    all_fusion_weights = []
    attention_entropies = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits, attention_info = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Collect fusion weights for analysis
            fusion_weights = attention_info["fusion_weights"].detach(
            ).cpu().numpy()
            all_fusion_weights.append(fusion_weights)

            # Calculate attention entropy
            frame_attn = attention_info["frame_attn"].detach()
            entropy = -torch.sum(frame_attn *
                                 torch.log(frame_attn + 1e-8), dim=-1)
            attention_entropies.append(entropy.mean().item())

    # Calculate comprehensive metrics
    class_names = ["neutral", "happy", "sad", "anger"]
    metrics = calculate_comprehensive_metrics(
        all_labels, all_preds, class_names)

    # Add loss
    metrics["loss"] = total_loss / len(val_loader)

    # Add session information
    if session_info:
        metrics["session_info"] = session_info

    # Analyze fusion weight patterns
    fusion_weights_array = np.concatenate(all_fusion_weights, axis=0)
    mean_fusion_weights = np.mean(fusion_weights_array, axis=0)
    std_fusion_weights = np.std(fusion_weights_array, axis=0)

    # Add fusion analysis
    metrics["fusion_analysis"] = {
        "frame_mean": float(mean_fusion_weights[0]),
        "segment_mean": float(mean_fusion_weights[1]),
        "utterance_mean": float(mean_fusion_weights[2]),
        "frame_std": float(std_fusion_weights[0]),
        "segment_std": float(std_fusion_weights[1]),
        "utterance_std": float(std_fusion_weights[2]),
    }

    metrics["attention_entropy"] = {
        "mean": float(np.mean(attention_entropies)),
        "std": float(np.std(attention_entropies)),
    }

    # Convert numpy arrays to lists for JSON serialization
    metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    # Print detailed results
    # Print detailed results (in two lines)
    if epoch is not None:
        session_str = f"Session {session_info}" if session_info else "Full Dataset"
        print(
            f"\nEpoch {epoch} Results for {session_str}: "
            f"Acc: {metrics['accuracy']:.4f}, WA: {metrics['wa']:.4f}, UAR: {metrics['uar']:.4f}, "
            f"F1-W: {metrics['f1_weighted']:.4f}, F1-M: {metrics['f1_macro']:.4f} | "
            f"Class Acc: {', '.join(f'{k}: {v:.4f}' for k, v in metrics['class_accuracies'].items())}"
        )
        print(
            f"Fusion Weights -> Frame: {metrics['fusion_analysis']['frame_mean']:.4f}±{metrics['fusion_analysis']['frame_std']:.4f}, "
            f"Segment: {metrics['fusion_analysis']['segment_mean']:.4f}±{metrics['fusion_analysis']['segment_std']:.4f}, "
            f"Utterance: {metrics['fusion_analysis']['utterance_mean']:.4f}±{metrics['fusion_analysis']['utterance_std']:.4f} | "
            f"Attention Entropy: {metrics['attention_entropy']['mean']:.4f}±{metrics['attention_entropy']['std']:.4f}"
        )

    # Save results if epoch is provided
    if epoch is not None and save_dir is not None:
        # Create session-specific filename
        session_str = f"_session_{session_info}" if session_info else "_full_dataset"

        # Save confusion matrix plot with session info
        cm_path = (
            save_dir /
            f"confusion_matrix_{eval_type}{session_str}_epoch_{epoch}.png"
        )
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), class_names, cm_path
        )

        # Save comprehensive metrics with session info
        report_path = save_dir / \
            f"metrics_{eval_type}{session_str}_epoch_{epoch}.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Log to wandb with session info
        wandb.log(
            {
                f"confusion_matrix_{eval_type}{session_str}_epoch_{epoch}": wandb.Image(
                    str(cm_path)
                ),
                f"accuracy_{eval_type}{session_str}": metrics["accuracy"],
                f"wa_{eval_type}{session_str}": metrics["wa"],
                f"uar_{eval_type}{session_str}": metrics["uar"],
                f"f1_weighted_{eval_type}{session_str}": metrics["f1_weighted"],
                f"f1_macro_{eval_type}{session_str}": metrics["f1_macro"],
                f"loss_{eval_type}{session_str}": metrics["loss"],
                f"fusion_frame_mean_{eval_type}{session_str}": metrics[
                    "fusion_analysis"
                ]["frame_mean"],
                f"fusion_segment_mean_{eval_type}{session_str}": metrics[
                    "fusion_analysis"
                ]["segment_mean"],
                f"fusion_utterance_mean_{eval_type}{session_str}": metrics[
                    "fusion_analysis"
                ]["utterance_mean"],
                f"attention_entropy_mean_{eval_type}{session_str}": metrics[
                    "attention_entropy"
                ]["mean"],
            }
        )

    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Track learnable parameters
    all_fusion_weights = []
    attention_entropies = []

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch in progress_bar:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, attention_info = model(features)

        # Calculate classification loss
        classification_loss = criterion(logits, labels)

        # Add entropy regularization loss
        entropy_loss = attention_info["entropy_loss"]
        total_batch_loss = classification_loss + entropy_loss

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_batch_loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Collect fusion weights for analysis
        fusion_weights = attention_info["fusion_weights"].detach(
        ).cpu().numpy()
        all_fusion_weights.append(fusion_weights)

        # Calculate attention entropy
        frame_attn = attention_info["frame_attn"].detach()
        entropy = -torch.sum(frame_attn * torch.log(frame_attn + 1e-8), dim=-1)
        attention_entropies.append(entropy.mean().item())

        progress_bar.set_postfix(
            {
                "loss": f"{total_batch_loss.item():.4f}",
                "cls_loss": f"{classification_loss.item():.4f}",
                "ent_loss": f"{entropy_loss.item():.4f}",
            }
        )

    # Calculate comprehensive metrics
    class_names = ["neutral", "happy", "sad", "anger"]
    metrics = calculate_comprehensive_metrics(
        all_labels, all_preds, class_names)

    # Add loss
    metrics["loss"] = total_loss / len(train_loader)

    # Analyze fusion weight patterns
    fusion_weights_array = np.concatenate(all_fusion_weights, axis=0)
    mean_fusion_weights = np.mean(fusion_weights_array, axis=0)
    std_fusion_weights = np.std(fusion_weights_array, axis=0)

    # Add fusion analysis
    metrics["fusion_analysis"] = {
        "frame_mean": float(mean_fusion_weights[0]),
        "segment_mean": float(mean_fusion_weights[1]),
        "utterance_mean": float(mean_fusion_weights[2]),
        "frame_std": float(std_fusion_weights[0]),
        "segment_std": float(std_fusion_weights[1]),
        "utterance_std": float(std_fusion_weights[2]),
    }

    metrics["attention_entropy"] = {
        "mean": float(np.mean(attention_entropies)),
        "std": float(np.std(attention_entropies)),
    }

    return metrics


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


def main():
    # Configuration
    config = {
        "batch_size": 38,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "class_weights": {
            "neutral": 1.1,
            "happy": 2.5,  # Higher weight for happy class
            "sad": 1.0,
            "anger": 1.0,
        },
        "early_stopping": {
            "patience": 5,
            "min_delta": 0.001,  # Minimum change in monitored value to qualify as an improvement
        },
        "curriculum": {
            "start_threshold": 0.0,  # Start with easiest samples
            "end_threshold": 1.0,    # End with all samples
        } if CURR else None,
    }

    # Create directories
    save_dir = get_unique_save_dir(experiment_name)
    run_name = os.path.basename(experiment_name)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize wandb
    wandb.init(project="emotion-recognition-hierarchical",
               config=config, name=run_name)

    # Load datasets
    print("Loading datasets...")
    iemocap_train = EmotionDataset("IEMOCAP", split="train")
    msp_improv = EmotionDataset("MSP-IMPROV", split="train")

    # Get session splits for IEMOCAP
    session_splits = get_session_splits(iemocap_train)

    # Initialize lists for storing results
    cross_corpus_results = []  # Initialize as a list
    all_results = []
    all_was = []
    all_uars = []
    all_f1s_weighted = []
    all_f1s_macro = []
    all_class_accuracies = []
    all_fusion_analysis = []
    all_attention_entropy = []

    for test_session in session_splits.keys():
        print(f"\nLeave-One-Session-Out: Testing on Session {test_session}")

        # Create session-specific directories
        session_dir = results_dir / f"session_{test_session}"
        session_dir.mkdir(exist_ok=True)

        # Split IEMOCAP data
        test_indices = session_splits[test_session]
        train_indices = []
        for session, indices in session_splits.items():
            if session != test_session:
                train_indices.extend(indices)

        # Create data loaders
        train_dataset = Subset(iemocap_train, train_indices)
        test_dataset = Subset(iemocap_train, test_indices)

        # Calculate class weights for this split
        class_names = ["neutral", "happy", "sad", "anger"]
        split_class_weights = calculate_class_weights(
            train_dataset, class_names)
        print(
            f"\nClass distribution for Session {test_session} training split:")
        for class_name, weight in split_class_weights.items():
            print(f"{class_name}: {weight:.2f}")

        # Combine with base weights
        final_class_weights = {
            class_name: base_weight * split_weight
            for (class_name, base_weight), (_, split_weight) in zip(
                config["class_weights"].items(), split_class_weights.items()
            )
        }

        # Create class weights tensor
        class_weights = torch.tensor(
            [
                final_class_weights["neutral"],
                final_class_weights["happy"],
                final_class_weights["sad"],
                final_class_weights["anger"],
            ]
        ).to(config["device"])

        print("Final class weights (base * split):")
        for class_name, weight in final_class_weights.items():
            print(f"{class_name}: {weight:.2f}")
        if SD and CURR:
            # Create combined sampler for both curriculum learning and speaker grouping
            combined_sampler = CombinedSampler(
                train_dataset,
                config["batch_size"],
                config["num_epochs"],
                config["curriculum"]["start_threshold"],
                config["curriculum"]["end_threshold"]
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                sampler=combined_sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        elif SD:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                sampler=SpeakerGroupedSampler(
                    train_dataset, config["batch_size"]),
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        elif CURR:
            # Create curriculum sampler
            curriculum_sampler = CurriculumSampler(
                train_dataset,
                config["batch_size"],
                config["num_epochs"],
                config["curriculum"]["start_threshold"],
                config["curriculum"]["end_threshold"]
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                sampler=curriculum_sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Initialize model
        model = HierarchicalAttention(
            input_dim=768).to(config["device"]).float()
        if pretrain_dir != None:
            # Load the backbone state dict
            checkpoint = torch.load(pretrain_dir)

            # Load the backbone weights directly since we saved only the backbone
            model.load_state_dict(checkpoint, strict=False)

            print(f"Loaded pretrained backbone from {pretrain_dir}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"]
        )

        # Training loop
        best_val_acc = 0
        best_val_wa = 0
        best_val_uar = 0
        best_val_f1_weighted = 0
        best_val_f1_macro = 0

        # Early stopping variables
        patience_counter = 0
        best_val_wa_so_far = 0
        best_model_state = None

        for epoch in range(config["num_epochs"]):
            # Update curriculum sampler epoch if using curriculum learning
            if CURR:
                train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, config["device"], epoch
            )

            # Validate
            val_metrics = enhanced_evaluate(
                model,
                test_loader,
                criterion,
                config["device"],
                epoch,
                session_dir,
                eval_type="held_out",
                session_info=test_session,
            )

            # Update learning rate
            scheduler.step()

            # Early stopping check
            current_val_wa = val_metrics["wa"]
            if (
                current_val_wa
                > best_val_wa_so_far + config["early_stopping"]["min_delta"]
            ):
                best_val_wa_so_far = current_val_wa
                patience_counter = 0
                # Save best model state
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_metrics["accuracy"],
                    "val_wa": val_metrics["wa"],
                    "val_uar": val_metrics["uar"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "val_f1_macro": val_metrics["f1_macro"],
                }
                # Save best model
                torch.save(
                    best_model_state,
                    save_dir / f"best_model_session_{test_session}.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= config["early_stopping"]["patience"]:
                    print(
                        f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation WA: {best_val_wa_so_far:.4f}")
                    break

            # Log metrics
            wandb.log(
                {
                    f"session_{test_session}/train_loss": train_metrics["loss"],
                    f"session_{test_session}/train_acc": train_metrics["accuracy"],
                    f"session_{test_session}/train_wa": train_metrics["wa"],
                    f"session_{test_session}/train_uar": train_metrics["uar"],
                    f"session_{test_session}/train_f1_weighted": train_metrics[
                        "f1_weighted"
                    ],
                    f"session_{test_session}/train_f1_macro": train_metrics["f1_macro"],
                    f"session_{test_session}/val_loss": val_metrics["loss"],
                    f"session_{test_session}/val_acc": val_metrics["accuracy"],
                    f"session_{test_session}/val_wa": val_metrics["wa"],
                    f"session_{test_session}/val_uar": val_metrics["uar"],
                    f"session_{test_session}/val_f1_weighted": val_metrics[
                        "f1_weighted"
                    ],
                    f"session_{test_session}/val_f1_macro": val_metrics["f1_macro"],
                    f"session_{test_session}/learning_rate": scheduler.get_last_lr()[0],
                }
            )

        # Load best model for final evaluation
        print(f"\nLoading best model for Session {test_session}...")
        if best_model_state is None:
            # If no model was saved during training (shouldn't happen), load the last saved one
            checkpoint = torch.load(
                save_dir / f"best_model_session_{test_session}.pt")
        else:
            checkpoint = best_model_state

        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded model from epoch {checkpoint['epoch']} with validation WA: {checkpoint['val_wa']:.4f}"
        )

        # Create evaluation directories
        held_out_dir = session_dir / "held_out_evaluation"
        cross_corpus_dir = session_dir / "cross_corpus_evaluation"
        held_out_dir.mkdir(exist_ok=True)
        cross_corpus_dir.mkdir(exist_ok=True)

        # Final evaluation on held-out test set
        print(
            f"\nPerforming final evaluation on held-out Session {test_session}...")
        final_metrics = enhanced_evaluate(
            model,
            test_loader,
            criterion,
            config["device"],
            epoch=test_session,
            save_dir=held_out_dir,
            eval_type="held_out",
            session_info=test_session,
        )

        # Save held-out results
        held_out_results = {
            "metrics": {
                "accuracy": float(final_metrics["accuracy"]),
                "wa": float(final_metrics["wa"]),
                "uar": float(final_metrics["uar"]),
                "f1_weighted": float(final_metrics["f1_weighted"]),
                "f1_macro": float(final_metrics["f1_macro"]),
                "class_accuracies": final_metrics["class_accuracies"],
                "fusion_analysis": final_metrics["fusion_analysis"],
                "attention_entropy": final_metrics["attention_entropy"],
            },
            "confusion_matrix": final_metrics["confusion_matrix"],
            "classification_report": final_metrics["classification_report"],
        }

        with open(held_out_dir / "results.json", "w") as f:
            json.dump(held_out_results, f, indent=4)

        print(f"\nHeld-out Session {test_session} Results:")
        print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"WA: {final_metrics['wa']:.4f}")
        print(f"UAR: {final_metrics['uar']:.4f}")
        print(f"F1-Weighted: {final_metrics['f1_weighted']:.4f}")
        print(f"F1-Macro: {final_metrics['f1_macro']:.4f}")
        print("\nClass-wise Accuracies:")
        for class_name, acc in final_metrics["class_accuracies"].items():
            print(f"{class_name}: {acc:.4f}")

        # Cross-corpus evaluation on MSP-IMPROV
        print("\nEvaluating on MSP-IMPROV...")
        msp_loader = DataLoader(
            msp_improv,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Evaluate on MSP-IMPROV using best model
        msp_metrics = enhanced_evaluate(
            model,
            msp_loader,
            criterion,
            config["device"],
            epoch=test_session,
            save_dir=cross_corpus_dir,
            eval_type="cross_corpus",
            session_info=test_session,
        )

        # Save cross-corpus results
        cross_corpus_results.append(msp_metrics)

        # Store results for final summary
        all_results.append(final_metrics["accuracy"])
        all_was.append(final_metrics["wa"])
        all_uars.append(final_metrics["uar"])
        all_f1s_weighted.append(final_metrics["f1_weighted"])
        all_f1s_macro.append(final_metrics["f1_macro"])
        all_class_accuracies.append(final_metrics["class_accuracies"])
        all_fusion_analysis.append(final_metrics["fusion_analysis"])
        all_attention_entropy.append(final_metrics["attention_entropy"])

    # Calculate and print averaged LOSO results
    print("\nAveraged LOSO Results:")
    print("=" * 50)

    # Calculate average confusion matrix
    avg_confusion_matrix = np.zeros((4, 4))  # 4x4 for 4 emotion classes
    for session, results in enumerate(all_results, 1):
        session_dir = results_dir / f"session_{session}"
        held_out_dir = session_dir / "held_out_evaluation"
        with open(held_out_dir / "results.json", "r") as f:
            session_results = json.load(f)
            avg_confusion_matrix += np.array(
                session_results["confusion_matrix"])
    avg_confusion_matrix /= len(all_results)

    # Save averaged confusion matrix
    avg_cm_path = results_dir / "averaged_confusion_matrix.png"
    plot_confusion_matrix(
        avg_confusion_matrix, ["neutral", "happy", "sad", "anger"], avg_cm_path
    )

    # Calculate and print averaged metrics
    print("\nAveraged Metrics:")
    print(f"Accuracy: {np.mean(all_results):.4f} ± {np.std(all_results):.4f}")
    print(f"WA: {np.mean(all_was):.4f} ± {np.std(all_was):.4f}")
    print(f"UAR: {np.mean(all_uars):.4f} ± {np.std(all_uars):.4f}")
    print(
        f"F1-Weighted: {np.mean(all_f1s_weighted):.4f} ± {np.std(all_f1s_weighted):.4f}"
    )
    print(
        f"F1-Macro: {np.mean(all_f1s_macro):.4f} ± {np.std(all_f1s_macro):.4f}")

    # Calculate averaged class accuracies
    avg_class_accuracies = {}
    for class_name in ["neutral", "happy", "sad", "anger"]:
        accuracies = [acc[class_name] for acc in all_class_accuracies]
        avg_class_accuracies[class_name] = {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
        }

    print("\nAveraged Class-wise Accuracies:")
    for class_name, stats in avg_class_accuracies.items():
        print(f"{class_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Save averaged results
    averaged_results = {
        "metrics": {
            "accuracy": {
                "mean": float(np.mean(all_results)),
                "std": float(np.std(all_results)),
            },
            "wa": {"mean": float(np.mean(all_was)), "std": float(np.std(all_was))},
            "uar": {"mean": float(np.mean(all_uars)), "std": float(np.std(all_uars))},
            "f1_weighted": {
                "mean": float(np.mean(all_f1s_weighted)),
                "std": float(np.std(all_f1s_weighted)),
            },
            "f1_macro": {
                "mean": float(np.mean(all_f1s_macro)),
                "std": float(np.std(all_f1s_macro)),
            },
            "class_accuracies": avg_class_accuracies,
        },
        "confusion_matrix": avg_confusion_matrix.tolist(),
    }

    with open(results_dir / "averaged_results.json", "w") as f:
        json.dump(averaged_results, f, indent=4)

    # Log averaged results to wandb
    wandb.log(
        {
            "averaged_confusion_matrix": wandb.Image(str(avg_cm_path)),
            "averaged_accuracy": np.mean(all_results),
            "averaged_wa": np.mean(all_was),
            "averaged_uar": np.mean(all_uars),
            "averaged_f1_weighted": np.mean(all_f1s_weighted),
            "averaged_f1_macro": np.mean(all_f1s_macro),
        }
    )

    for class_name, stats in avg_class_accuracies.items():
        wandb.log({f"averaged_{class_name}_accuracy": stats["mean"]})

    # After all LOSO evaluations, do final cross-corpus evaluation
    print("\nPerforming final cross-corpus evaluation on entire IEMOCAP...")

    # Create directory for final cross-corpus results
    final_cross_corpus_dir = results_dir / "final_cross_corpus"
    final_cross_corpus_dir.mkdir(exist_ok=True)

    # Create data loaders for full IEMOCAP
    full_train_loader = DataLoader(
        iemocap_train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    msp_loader = DataLoader(
        msp_improv,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # For full IEMOCAP training
    print("\nCalculating class weights for full IEMOCAP training...")
    full_class_weights = calculate_class_weights(iemocap_train, class_names)
    print("Class distribution for full IEMOCAP:")
    for class_name, weight in full_class_weights.items():
        print(f"{class_name}: {weight:.2f}")

    # Combine with base weights
    final_full_class_weights = {
        class_name: base_weight * split_weight
        for (class_name, base_weight), (_, split_weight) in zip(
            config["class_weights"].items(), full_class_weights.items()
        )
    }

    print("Final class weights for full IEMOCAP (base * split):")
    for class_name, weight in final_full_class_weights.items():
        print(f"{class_name}: {weight:.2f}")

    # Create class weights tensor for full training
    class_weights = torch.tensor(
        [
            final_full_class_weights["neutral"],
            final_full_class_weights["happy"],
            final_full_class_weights["sad"],
            final_full_class_weights["anger"],
        ]
    ).to(config["device"])

    # Initialize model for full training
    model = HierarchicalAttention(input_dim=768).to(config["device"]).float()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    # Training loop for full IEMOCAP
    best_val_wa = 0
    patience_counter = 0
    best_model_state = None

    print("Training on full IEMOCAP dataset...")
    for epoch in range(config["num_epochs"]):
        # Train
        train_metrics = train_epoch(
            model, full_train_loader, criterion, optimizer, config["device"], epoch
        )

        # Validate on a small portion of IEMOCAP for early stopping
        val_size = int(len(iemocap_train) * 0.1)  # Use 10% for validation
        val_indices = np.random.choice(
            len(iemocap_train), val_size, replace=False)
        val_dataset = Subset(iemocap_train, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        val_metrics = enhanced_evaluate(
            model,
            val_loader,
            criterion,
            config["device"],
            epoch,
            final_cross_corpus_dir,
            eval_type="full_iemocap_val",
            session_info="full_dataset",
        )

        # Update learning rate
        scheduler.step()

        # Early stopping check
        current_val_wa = val_metrics["wa"]
        if current_val_wa > best_val_wa + config["early_stopping"]["min_delta"]:
            best_val_wa = current_val_wa
            patience_counter = 0
            # Save best model state
            best_model_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_wa": val_metrics["wa"],
            }
            # Save best model
            torch.save(
                best_model_state,
                save_dir / "best_model_full_iemocap.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping"]["patience"]:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation WA: {best_val_wa:.4f}")
                break

    # Load best model for final cross-corpus evaluation
    print("\nLoading best model for final cross-corpus evaluation...")
    if best_model_state is None:
        checkpoint = torch.load(save_dir / "best_model_full_iemocap.pt")
    else:
        checkpoint = best_model_state

    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with validation WA: {checkpoint['val_wa']:.4f}"
    )

    # Final cross-corpus evaluation on MSP-IMPROV
    print("\nPerforming final cross-corpus evaluation on MSP-IMPROV...")
    final_cross_corpus_metrics = enhanced_evaluate(
        model,
        msp_loader,
        criterion,
        config["device"],
        epoch="final",
        save_dir=final_cross_corpus_dir,
        eval_type="final_cross_corpus",
        session_info="full_dataset",
    )

    # Save final cross-corpus results
    final_cross_corpus_results = {
        "metrics": {
            "accuracy": float(final_cross_corpus_metrics["accuracy"]),
            "wa": float(final_cross_corpus_metrics["wa"]),
            "uar": float(final_cross_corpus_metrics["uar"]),
            "f1_weighted": float(final_cross_corpus_metrics["f1_weighted"]),
            "f1_macro": float(final_cross_corpus_metrics["f1_macro"]),
            "class_accuracies": final_cross_corpus_metrics["class_accuracies"],
            "fusion_analysis": final_cross_corpus_metrics["fusion_analysis"],
            "attention_entropy": final_cross_corpus_metrics["attention_entropy"],
        },
        "confusion_matrix": final_cross_corpus_metrics["confusion_matrix"],
        "classification_report": final_cross_corpus_metrics["classification_report"],
    }

    with open(final_cross_corpus_dir / "final_cross_corpus_results.json", "w") as f:
        json.dump(final_cross_corpus_results, f, indent=4)

    print("\nFinal Cross-Corpus Results (Full IEMOCAP → MSP-IMPROV):")
    print(f"Accuracy: {final_cross_corpus_metrics['accuracy']:.4f}")
    print(f"WA: {final_cross_corpus_metrics['wa']:.4f}")
    print(f"UAR: {final_cross_corpus_metrics['uar']:.4f}")
    print(f"F1-Weighted: {final_cross_corpus_metrics['f1_weighted']:.4f}")
    print(f"F1-Macro: {final_cross_corpus_metrics['f1_macro']:.4f}")
    print("\nClass-wise Accuracies:")
    for class_name, acc in final_cross_corpus_metrics["class_accuracies"].items():
        print(f"{class_name}: {acc:.4f}")

    # Plot and save confusion matrix before logging to wandb
    cm_path = final_cross_corpus_dir / "confusion_matrix_final_cross_corpus_final.png"
    plot_confusion_matrix(
        np.array(final_cross_corpus_metrics["confusion_matrix"]),
        ["neutral", "happy", "sad", "anger"],
        cm_path,
    )

    # Log final cross-corpus results to wandb
    wandb.log(
        {
            "final_cross_corpus_confusion_matrix": wandb.Image(str(cm_path)),
            "final_cross_corpus_accuracy": final_cross_corpus_metrics["accuracy"],
            "final_cross_corpus_wa": final_cross_corpus_metrics["wa"],
            "final_cross_corpus_uar": final_cross_corpus_metrics["uar"],
            "final_cross_corpus_f1_weighted": final_cross_corpus_metrics["f1_weighted"],
            "final_cross_corpus_f1_macro": final_cross_corpus_metrics["f1_macro"],
        }
    )

    for class_name, acc in final_cross_corpus_metrics["class_accuracies"].items():
        wandb.log({f"final_cross_corpus_{class_name}_accuracy": acc})

    # Save final results
    results = {
        "iemocap_results": {
            "session_results": {
                f"session_{i+1}": {
                    "accuracy": float(acc),
                    "wa": float(wa),
                    "uar": float(uar),
                    "f1_weighted": float(f1_weighted),
                    "f1_macro": float(f1_macro),
                    "class_accuracies": class_accuracies,
                    "fusion_analysis": fusion_analysis,
                    "attention_entropy": attention_entropy,
                }
                for i, (
                    acc,
                    wa,
                    uar,
                    f1_weighted,
                    f1_macro,
                    class_accuracies,
                    fusion_analysis,
                    attention_entropy,
                ) in enumerate(
                    zip(
                        all_results,
                        all_was,
                        all_uars,
                        all_f1s_weighted,
                        all_f1s_macro,
                        all_class_accuracies,
                        all_fusion_analysis,
                        all_attention_entropy,
                    )
                )
            },
            "averages": {
                "accuracy": {
                    "mean": float(np.mean(all_results)),
                    "std": float(np.std(all_results)),
                },
                "wa": {
                    "mean": float(np.mean(all_was)),
                    "std": float(np.std(all_was)),
                },
                "uar": {
                    "mean": float(np.mean(all_uars)),
                    "std": float(np.std(all_uars)),
                },
                "f1_weighted": {
                    "mean": float(np.mean(all_f1s_weighted)),
                    "std": float(np.std(all_f1s_weighted)),
                },
                "f1_macro": {
                    "mean": float(np.mean(all_f1s_macro)),
                    "std": float(np.std(all_f1s_macro)),
                },
            },
        },
        "cross_corpus_results": {
            "session_results": {
                f"session_{i+1}": {
                    "accuracy": float(results["accuracy"]),
                    "wa": float(results["wa"]),
                    "uar": float(results["uar"]),
                    "f1_weighted": float(results["f1_weighted"]),
                    "f1_macro": float(results["f1_macro"]),
                    "class_accuracies": results["class_accuracies"],
                    "fusion_analysis": results["fusion_analysis"],
                    "attention_entropy": results["attention_entropy"],
                }
                for i, results in enumerate(cross_corpus_results)
            },
            "averages": {
                "accuracy": {
                    "mean": float(
                        np.mean([r["accuracy"] for r in cross_corpus_results])
                    ),
                    "std": float(np.std([r["accuracy"] for r in cross_corpus_results])),
                },
                "wa": {
                    "mean": float(np.mean([r["wa"] for r in cross_corpus_results])),
                    "std": float(np.std([r["wa"] for r in cross_corpus_results])),
                },
                "uar": {
                    "mean": float(np.mean([r["uar"] for r in cross_corpus_results])),
                    "std": float(np.std([r["uar"] for r in cross_corpus_results])),
                },
                "f1_weighted": {
                    "mean": float(
                        np.mean([r["f1_weighted"]
                                for r in cross_corpus_results])
                    ),
                    "std": float(
                        np.std([r["f1_weighted"]
                               for r in cross_corpus_results])
                    ),
                },
                "f1_macro": {
                    "mean": float(
                        np.mean([r["f1_macro"] for r in cross_corpus_results])
                    ),
                    "std": float(np.std([r["f1_macro"] for r in cross_corpus_results])),
                },
            },
        },
    }

    with open(save_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=4)

    wandb.finish()


if __name__ == "__main__":
    main()

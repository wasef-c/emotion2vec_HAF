import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
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

# Disable all possible FunASR output
os.environ["FUNASR_DISABLE_PROGRESS_BAR"] = "1"
os.environ["FUNASR_QUIET"] = "1"
os.environ["FUNASR_LOG_LEVEL"] = "ERROR"
os.environ["FUNASR_DISABLE_TQDM"] = "1"
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)

from models.hierarchical_attention import HierarchicalAttention
from data.dataset import EmotionDataset, collate_fn

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method("spawn", force=True)


def calculate_uar(cm):
    """Calculate Unweighted Average Recall (UAR) from confusion matrix."""
    recalls = cm.diagonal() / cm.sum(axis=1)
    return np.mean(recalls)


def calculate_wa(cm):
    """
    Calculate Weighted Accuracy (WA) from confusion matrix.
    WA = sum(class_accuracy * class_frequency) / total_samples
    This is the same as regular accuracy, but explicitly shows the weighting.
    """
    # Class-wise accuracies (recall for each class)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    # Class frequencies (support for each class)
    class_frequencies = cm.sum(axis=1)
    total_samples = cm.sum()

    # Weighted accuracy
    wa = np.sum(class_accuracies * class_frequencies) / total_samples
    return wa


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
    wa = calculate_wa(cm)  # This should equal accuracy, but good for verification
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


def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def enhanced_evaluate(model, val_loader, criterion, device, epoch=None, save_dir=None):
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
            fusion_weights = attention_info["fusion_weights"].detach().cpu().numpy()
            all_fusion_weights.append(fusion_weights)

            # Calculate attention entropy
            frame_attn = attention_info["frame_attn"].detach()
            entropy = -torch.sum(frame_attn * torch.log(frame_attn + 1e-8), dim=-1)
            attention_entropies.append(entropy.mean().item())

    # Calculate comprehensive metrics
    class_names = ["neutral", "happy", "sad", "anger"]
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, class_names)

    # Add loss
    metrics["loss"] = total_loss / len(val_loader)

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

    # Save results if epoch is provided
    if epoch is not None and save_dir is not None:
        # Save confusion matrix plot
        cm_path = save_dir / f"confusion_matrix_epoch_{epoch}.png"
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), class_names, cm_path
        )

        # Save comprehensive metrics
        report_path = save_dir / f"metrics_epoch_{epoch}.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Log to wandb
        wandb.log(
            {
                f"confusion_matrix_epoch_{epoch}": wandb.Image(str(cm_path)),
                f"accuracy_epoch_{epoch}": metrics["accuracy"],
                f"wa_epoch_{epoch}": metrics["wa"],
                f"uar_epoch_{epoch}": metrics["uar"],
                f"f1_weighted_epoch_{epoch}": metrics["f1_weighted"],
                f"f1_macro_epoch_{epoch}": metrics["f1_macro"],
                f"loss_epoch_{epoch}": metrics["loss"],
                f"fusion_frame_mean_epoch_{epoch}": metrics["fusion_analysis"][
                    "frame_mean"
                ],
                f"fusion_segment_mean_epoch_{epoch}": metrics["fusion_analysis"][
                    "segment_mean"
                ],
                f"fusion_utterance_mean_epoch_{epoch}": metrics["fusion_analysis"][
                    "utterance_mean"
                ],
                f"attention_entropy_mean_epoch_{epoch}": metrics["attention_entropy"][
                    "mean"
                ],
            }
        )

    # Print detailed results
    if epoch is not None:
        print(f"\nEpoch {epoch} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"WA (Weighted Accuracy): {metrics['wa']:.4f}")
        print(f"UAR (Unweighted Average Recall): {metrics['uar']:.4f}")
        print(f"F1-Weighted: {metrics['f1_weighted']:.4f}")
        print(f"F1-Macro: {metrics['f1_macro']:.4f}")
        print("\nClass-wise Accuracies:")
        for class_name, acc in metrics["class_accuracies"].items():
            print(f"{class_name}: {acc:.4f}")
        print("\nFusion Weights:")
        print(
            f"Frame: {metrics['fusion_analysis']['frame_mean']:.4f} ± {metrics['fusion_analysis']['frame_std']:.4f}"
        )
        print(
            f"Segment: {metrics['fusion_analysis']['segment_mean']:.4f} ± {metrics['fusion_analysis']['segment_std']:.4f}"
        )
        print(
            f"Utterance: {metrics['fusion_analysis']['utterance_mean']:.4f} ± {metrics['fusion_analysis']['utterance_std']:.4f}"
        )
        print(
            f"\nAttention Entropy: {metrics['attention_entropy']['mean']:.4f} ± {metrics['attention_entropy']['std']:.4f}"
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
        fusion_weights = attention_info["fusion_weights"].detach().cpu().numpy()
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
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, class_names)

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


def main():
    # Configuration
    config = {
        "batch_size": 36,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "class_weights": {
            "neutral": 1.0,
            "happy": 2.0,  # Higher weight for happy class
            "sad": 1.0,
            "anger": 1.0,
        },
    }

    # Create directories
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize wandb
    wandb.init(project="emotion-recognition-hierarchical", config=config)

    # Load datasets
    print("Loading datasets...")
    iemocap_train = EmotionDataset("IEMOCAP", split="train")
    msp_improv = EmotionDataset("MSP-IMPROV", split="train")

    # Get session splits for IEMOCAP
    session_splits = get_session_splits(iemocap_train)

    # Cross-corpus evaluation results
    cross_corpus_results = []

    # LOSO Cross-validation on IEMOCAP
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
        model = HierarchicalAttention(input_dim=768).to(config["device"]).float()

        # Create class weights tensor
        class_weights = torch.tensor(
            [
                config["class_weights"]["neutral"],
                config["class_weights"]["happy"],
                config["class_weights"]["sad"],
                config["class_weights"]["anger"],
            ]
        ).to(config["device"])

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

        for epoch in range(config["num_epochs"]):
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, config["device"], epoch
            )

            # Validate
            val_metrics = enhanced_evaluate(
                model, test_loader, criterion, config["device"], epoch, session_dir
            )

            # Update learning rate
            scheduler.step()

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

            print(
                f"Epoch {epoch}: Train Acc = {train_metrics['accuracy']:.4f}, WA = {train_metrics['wa']:.4f}, UAR = {train_metrics['uar']:.4f}, F1-Weighted = {train_metrics['f1_weighted']:.4f}, F1-Macro = {train_metrics['f1_macro']:.4f} | "
                f"Val Acc = {val_metrics['accuracy']:.4f}, WA = {val_metrics['wa']:.4f}, UAR = {val_metrics['uar']:.4f}, F1-Weighted = {val_metrics['f1_weighted']:.4f}, F1-Macro = {val_metrics['f1_macro']:.4f}"
            )

            # Save best model based on validation accuracy
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_val_wa = val_metrics["wa"]
                best_val_uar = val_metrics["uar"]
                best_val_f1_weighted = val_metrics["f1_weighted"]
                best_val_f1_macro = val_metrics["f1_macro"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_metrics["accuracy"],
                        "val_wa": val_metrics["wa"],
                        "val_uar": val_metrics["uar"],
                        "val_f1_weighted": val_metrics["f1_weighted"],
                        "val_f1_macro": val_metrics["f1_macro"],
                    },
                    save_dir / f"best_model_session_{test_session}.pt",
                )

        all_results.append(best_val_acc)
        all_was.append(best_val_wa)
        all_uars.append(best_val_uar)
        all_f1s_weighted.append(best_val_f1_weighted)
        all_f1s_macro.append(best_val_f1_macro)
        all_class_accuracies.append(val_metrics["class_accuracies"])
        all_fusion_analysis.append(val_metrics["fusion_analysis"])
        all_attention_entropy.append(val_metrics["attention_entropy"])

        print(f"Best results for Session {test_session}:")
        print(f"Accuracy: {best_val_acc:.4f}")
        print(f"WA: {best_val_wa:.4f}")
        print(f"UAR: {best_val_uar:.4f}")
        print(f"F1-Weighted: {best_val_f1_weighted:.4f}")
        print(f"F1-Macro: {best_val_f1_macro:.4f}")

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

        # Load best model
        checkpoint = torch.load(save_dir / f"best_model_session_{test_session}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on MSP-IMPROV
        msp_metrics = enhanced_evaluate(
            model,
            msp_loader,
            criterion,
            config["device"],
            epoch="final",
            save_dir=session_dir,
        )
        cross_corpus_results.append(msp_metrics)

        print(f"MSP-IMPROV results for Session {test_session}:")
        print(f"Accuracy: {msp_metrics['accuracy']:.4f}")
        print(f"WA: {msp_metrics['wa']:.4f}")
        print(f"UAR: {msp_metrics['uar']:.4f}")
        print(f"F1-Weighted: {msp_metrics['f1_weighted']:.4f}")
        print(f"F1-Macro: {msp_metrics['f1_macro']:.4f}")

    # Print final results
    print("\nFinal Results:")
    print("=" * 50)
    print("IEMOCAP Results:")
    for session, (
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
        ),
        1,
    ):
        print(f"Session {session}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  WA: {wa:.4f}")
        print(f"  UAR: {uar:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        print(f"  F1-Macro: {f1_macro:.4f}")
        print("\nClass-wise Accuracies:")
        for class_name, acc in class_accuracies.items():
            print(f"{class_name}: {acc:.4f}")
        print("\nFusion Analysis:")
        for key, value in fusion_analysis.items():
            print(f"{key}: {value:.4f}")
        print("\nAttention Entropy:")
        for key, value in attention_entropy.items():
            print(f"{key}: {value:.4f}")

    print(f"\nIEMOCAP Averages:")
    print(f"Accuracy: {np.mean(all_results):.4f} ± {np.std(all_results):.4f}")
    print(f"WA: {np.mean(all_was):.4f} ± {np.std(all_was):.4f}")
    print(f"UAR: {np.mean(all_uars):.4f} ± {np.std(all_uars):.4f}")
    print(
        f"F1-Weighted: {np.mean(all_f1s_weighted):.4f} ± {np.std(all_f1s_weighted):.4f}"
    )
    print(f"F1-Macro: {np.mean(all_f1s_macro):.4f} ± {np.std(all_f1s_macro):.4f}")

    print("\nCross-Corpus Results (MSP-IMPROV):")
    for session, results in enumerate(cross_corpus_results, 1):
        print(f"Session {session}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  WA: {results['wa']:.4f}")
        print(f"  UAR: {results['uar']:.4f}")
        print(f"  F1-Weighted: {results['f1_weighted']:.4f}")
        print(f"  F1-Macro: {results['f1_macro']:.4f}")

    print(f"\nMSP-IMPROV Averages:")
    msp_accs = [r["accuracy"] for r in cross_corpus_results]
    msp_was = [r["wa"] for r in cross_corpus_results]
    msp_uars = [r["uar"] for r in cross_corpus_results]
    msp_f1s_weighted = [r["f1_weighted"] for r in cross_corpus_results]
    msp_f1s_macro = [r["f1_macro"] for r in cross_corpus_results]
    print(f"Accuracy: {np.mean(msp_accs):.4f} ± {np.std(msp_accs):.4f}")
    print(f"WA: {np.mean(msp_was):.4f} ± {np.std(msp_was):.4f}")
    print(f"UAR: {np.mean(msp_uars):.4f} ± {np.std(msp_uars):.4f}")
    print(
        f"F1-Weighted: {np.mean(msp_f1s_weighted):.4f} ± {np.std(msp_f1s_weighted):.4f}"
    )
    print(f"F1-Macro: {np.mean(msp_f1s_macro):.4f} ± {np.std(msp_f1s_macro):.4f}")

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
                    "mean": float(np.mean(msp_accs)),
                    "std": float(np.std(msp_accs)),
                },
                "wa": {
                    "mean": float(np.mean(msp_was)),
                    "std": float(np.std(msp_was)),
                },
                "uar": {
                    "mean": float(np.mean(msp_uars)),
                    "std": float(np.std(msp_uars)),
                },
                "f1_weighted": {
                    "mean": float(np.mean(msp_f1s_weighted)),
                    "std": float(np.std(msp_f1s_weighted)),
                },
                "f1_macro": {
                    "mean": float(np.mean(msp_f1s_macro)),
                    "std": float(np.std(msp_f1s_macro)),
                },
            },
        },
    }

    with open(save_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=4)

    wandb.finish()


if __name__ == "__main__":
    main()

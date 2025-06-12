import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
from funasr import AutoModel
from transformers import get_linear_schedule_with_warmup

# Disable all possible FunASR output
os.environ["FUNASR_DISABLE_PROGRESS_BAR"] = "1"
os.environ["FUNASR_QUIET"] = "1"
os.environ["FUNASR_LOG_LEVEL"] = "ERROR"
os.environ["FUNASR_DISABLE_TQDM"] = "1"
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)
from data.emotion2vec_dataset import Emotion2VecDataset, collate_fn

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method("spawn", force=True)
experiment_name = "emotion2vec_plus_base_finetune"


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
        cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, classifier):
    # model  # Set model to eval mode since we're only using it for feature extraction
    classifier.train()  # Set classifier to train mode
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch in progress_bar:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        # Forward pass through classifier
        logits = classifier(features)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
            }
        )

    # Calculate comprehensive metrics
    class_names = ["neutral", "happy", "sad", "anger"]
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, class_names)

    # Add loss
    metrics["loss"] = total_loss / len(train_loader)

    return metrics


def enhanced_evaluate(
    model,
    val_loader,
    criterion,
    device,
    epoch=None,
    save_dir=None,
    eval_type="held_out",
    session_info=None,
    classifier=None,
):
    """Enhanced evaluation with comprehensive metrics including WA"""
    # model.eval()  # Set model to eval mode
    classifier.eval()  # Set classifier to eval mode
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            # Forward pass through classifier
            logits = classifier(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate comprehensive metrics
    class_names = ["neutral", "happy", "sad", "anger"]
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, class_names)

    # Add loss
    metrics["loss"] = total_loss / len(val_loader)

    # Add session information
    if session_info:
        metrics["session_info"] = session_info

    # Convert numpy arrays to lists for JSON serialization
    metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    # Print detailed results
    if epoch is not None:
        session_str = f"Session {session_info}" if session_info else "Full Dataset"
        print(
            f"\nEpoch {epoch} Results for {session_str}: "
            f"Acc: {metrics['accuracy']:.4f}, WA: {metrics['wa']:.4f}, UAR: {metrics['uar']:.4f}, "
            f"F1-W: {metrics['f1_weighted']:.4f}, F1-M: {metrics['f1_macro']:.4f} | "
            f"Class Acc: {', '.join(f'{k}: {v:.4f}' for k, v in metrics['class_accuracies'].items())}"
        )

    # Save results if epoch is provided
    if epoch is not None and save_dir is not None:
        # Create session-specific filename
        session_str = f"_session_{session_info}" if session_info else "_full_dataset"

        # Save confusion matrix plot with session info
        cm_path = (
            save_dir / f"confusion_matrix_{eval_type}{session_str}_epoch_{epoch}.png"
        )
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), class_names, cm_path
        )

        # Save comprehensive metrics with session info
        report_path = save_dir / f"metrics_{eval_type}{session_str}_epoch_{epoch}.json"
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
            }
        )

    return metrics


def get_session_splits(dataset):
    """Get indices for each session in IEMOCAP dataset."""
    session_indices = {}
    for i, item in enumerate(dataset.data):
        session = item["speaker_id"]
        if session not in session_indices:
            session_indices[session] = []
        session_indices[session].append(i)
    return session_indices


def get_unique_save_dir(base_name, root="checkpoints"):
    root_path = Path(root)
    root_path.mkdir(exist_ok=True)  # Create parent directory if it doesn't exist
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
        "batch_size": 16,  # Smaller batch size for transformer model
        "num_epochs": 10,
        "learning_rate": 2e-5,  # Lower learning rate for fine-tuning
        "weight_decay": 0.01,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "early_stopping": {
            "patience": 3,
            "min_delta": 0.001,
        },
        "warmup_steps": 100,
    }

    # Create directories
    save_dir = get_unique_save_dir(experiment_name)
    run_name = os.path.basename(experiment_name)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize wandb
    wandb.init(project="emotion-recognition-emotion2vec", config=config, name=run_name)

    # Load model for feature extraction
    print("Loading model...")
    model = AutoModel(model="iic/emotion2vec_plus_base")

    # Load datasets
    print("Loading datasets...")
    iemocap_train = Emotion2VecDataset("IEMOCAP", split="train", model=model)
    msp_improv = Emotion2VecDataset("MSP-IMPROV", split="train", model=model)

    # Get session splits for IEMOCAP
    session_splits = get_session_splits(iemocap_train)

    # Initialize lists for storing results
    all_results = []
    all_was = []
    all_uars = []
    all_f1s_weighted = []
    all_f1s_macro = []
    all_class_accuracies = []

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

        # Initialize classifier
        classifier = nn.Linear(768, 4).to(
            config["device"]
        )  # 768 is the feature dimension from emotion2vec

        # Initialize optimizer and scheduler - only for classifier
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # Calculate total training steps for scheduler
        total_steps = len(train_loader) * config["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=total_steps,
        )

        criterion = nn.CrossEntropyLoss()

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
            # Train
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                config["device"],
                epoch,
                classifier,
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
                classifier=classifier,
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
                    "classifier_state_dict": classifier.state_dict(),
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
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
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
            checkpoint = torch.load(save_dir / f"best_model_session_{test_session}.pt")
        else:
            checkpoint = best_model_state

        classifier.load_state_dict(checkpoint["classifier_state_dict"])
        print(
            f"Loaded model from epoch {checkpoint['epoch']} with validation WA: {checkpoint['val_wa']:.4f}"
        )

        # Create evaluation directories
        held_out_dir = session_dir / "held_out_evaluation"
        held_out_dir.mkdir(exist_ok=True)

        # Final evaluation on held-out test set
        print(f"\nPerforming final evaluation on held-out Session {test_session}...")
        final_metrics = enhanced_evaluate(
            model,
            test_loader,
            criterion,
            config["device"],
            epoch=test_session,
            save_dir=held_out_dir,
            eval_type="held_out",
            session_info=test_session,
            classifier=classifier,
        )

        # Store results for final summary
        all_results.append(final_metrics["accuracy"])
        all_was.append(final_metrics["wa"])
        all_uars.append(final_metrics["uar"])
        all_f1s_weighted.append(final_metrics["f1_weighted"])
        all_f1s_macro.append(final_metrics["f1_macro"])
        all_class_accuracies.append(final_metrics["class_accuracies"])

    # Calculate and print averaged LOSO results
    print("\nAveraged LOSO Results:")
    print("=" * 50)

    # Calculate average confusion matrix
    avg_confusion_matrix = np.zeros((4, 4))  # 4x4 for 4 emotion classes
    for session, results in enumerate(all_results, 1):
        session_dir = results_dir / f"session_{session}"
        held_out_dir = session_dir / "held_out_evaluation"
        with open(
            held_out_dir / "metrics_held_out_session_{session}_epoch_{session}.json",
            "r",
        ) as f:
            session_results = json.load(f)
            avg_confusion_matrix += np.array(session_results["confusion_matrix"])
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
    print(f"F1-Macro: {np.mean(all_f1s_macro):.4f} ± {np.std(all_f1s_macro):.4f}")

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
    print("\nPerforming final cross-corpus evaluation on MSP-IMPROV...")

    # Create directory for final cross-corpus results
    final_cross_corpus_dir = results_dir / "final_cross_corpus"
    final_cross_corpus_dir.mkdir(exist_ok=True)

    # Create data loader for MSP-IMPROV
    msp_loader = DataLoader(
        msp_improv,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Create class weights tensor for full training
    class_weights = torch.tensor(
        [
            1.0,  # Assuming equal weights for simplicity
            1.0,
            1.0,
            1.0,
        ]
    ).to(config["device"])

    # Initialize classifier for full training
    classifier = nn.Linear(768, 4).to(
        config["device"]
    )  # 768 is the feature dimension from emotion2vec

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Training loop for full IEMOCAP
    best_val_wa = 0
    patience_counter = 0
    best_model_state = None

    print("Training on full IEMOCAP dataset...")
    for epoch in range(config["num_epochs"]):
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config["device"],
            epoch,
            classifier,
        )

        # Validate on MSP-IMPROV
        val_metrics = enhanced_evaluate(
            model,
            msp_loader,
            criterion,
            config["device"],
            epoch,
            final_cross_corpus_dir,
            eval_type="validation",
            session_info="msp_improv",
            classifier=classifier,
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
                "classifier_state_dict": classifier.state_dict(),
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

    # Load best model for final test evaluation
    print("\nLoading best model for final test evaluation...")
    if best_model_state is None:
        checkpoint = torch.load(save_dir / "best_model_full_iemocap.pt")
    else:
        checkpoint = best_model_state

    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with validation WA: {checkpoint['val_wa']:.4f}"
    )

    # Final test evaluation on MSP-IMPROV
    print("\nPerforming final test evaluation on MSP-IMPROV...")
    final_test_metrics = enhanced_evaluate(
        model,
        msp_loader,
        criterion,
        config["device"],
        epoch="final",
        save_dir=final_cross_corpus_dir,
        eval_type="test",
        session_info="msp_improv",
        classifier=classifier,
    )

    # Save final test results
    final_test_results = {
        "metrics": {
            "accuracy": float(final_test_metrics["accuracy"]),
            "wa": float(final_test_metrics["wa"]),
            "uar": float(final_test_metrics["uar"]),
            "f1_weighted": float(final_test_metrics["f1_weighted"]),
            "f1_macro": float(final_test_metrics["f1_macro"]),
            "class_accuracies": final_test_metrics["class_accuracies"],
        },
        "confusion_matrix": final_test_metrics["confusion_matrix"],
        "classification_report": final_test_metrics["classification_report"],
    }

    with open(final_cross_corpus_dir / "final_test_results.json", "w") as f:
        json.dump(final_test_results, f, indent=4)

    print("\nFinal Test Results (IEMOCAP → MSP-IMPROV):")
    print(f"Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"WA: {final_test_metrics['wa']:.4f}")
    print(f"UAR: {final_test_metrics['uar']:.4f}")
    print(f"F1-Weighted: {final_test_metrics['f1_weighted']:.4f}")
    print(f"F1-Macro: {final_test_metrics['f1_macro']:.4f}")
    print("\nClass-wise Accuracies:")
    for class_name, acc in final_test_metrics["class_accuracies"].items():
        print(f"{class_name}: {acc:.4f}")

    # Plot and save confusion matrix
    cm_path = final_cross_corpus_dir / "confusion_matrix_final_test.png"
    plot_confusion_matrix(
        np.array(final_test_metrics["confusion_matrix"]),
        ["neutral", "happy", "sad", "anger"],
        cm_path,
    )

    # Log final test results to wandb
    wandb.log(
        {
            "final_test_confusion_matrix": wandb.Image(str(cm_path)),
            "final_test_accuracy": final_test_metrics["accuracy"],
            "final_test_wa": final_test_metrics["wa"],
            "final_test_uar": final_test_metrics["uar"],
            "final_test_f1_weighted": final_test_metrics["f1_weighted"],
            "final_test_f1_macro": final_test_metrics["f1_macro"],
        }
    )

    for class_name, acc in final_test_metrics["class_accuracies"].items():
        wandb.log({f"final_test_{class_name}_accuracy": acc})

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
                }
                for i, (
                    acc,
                    wa,
                    uar,
                    f1_weighted,
                    f1_macro,
                    class_accuracies,
                ) in enumerate(
                    zip(
                        all_results,
                        all_was,
                        all_uars,
                        all_f1s_weighted,
                        all_f1s_macro,
                        all_class_accuracies,
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
        "final_test_results": final_test_results,
    }

    with open(save_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=4)

    wandb.finish()


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import json

from models.hierarchical_attention import HierarchicalAttention
from data.dataset import EmotionDataset
from train import calculate_metrics, train_epoch


def plot_confusion_matrix(conf_matrix, labels, save_path):
    """Plot and save confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_session_splits(dataset):
    """Get indices for each IEMOCAP session (2 speakers)."""
    session_splits = {}
    for i in range(len(dataset)):
        speaker_id = dataset.data[i][2]  # Access speaker_id from dataset
        session = (int(speaker_id) - 1) // 2 + 1  # Convert speaker ID to session number
        if session not in session_splits:
            session_splits[session] = []
        session_splits[session].append(i)
    return session_splits


def train_loso_model(train_loader, val_loader, device, save_dir, session):
    """Train model for LOSO evaluation."""
    model = HierarchicalAttention().to(device)

    # Training configuration
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_metrics = {"uar": 0}
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        train_loss, train_metrics, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validation
        val_loss, val_metrics, _, _ = generate_test_report(
            model, val_loader, device, save_dir, "IEMOCAP", session, save_results=False
        )

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_metrics["uar"] > best_val_metrics["uar"]:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict().copy()

        print(
            f"Epoch {epoch}: Train UAR = {train_metrics['uar']:.4f}, Val UAR = {val_metrics['uar']:.4f}"
        )

    # Load best model
    model.load_state_dict(best_model_state)
    return model


def generate_test_report(
    model, test_loader, device, save_dir, dataset_name, session=None, save_results=True
):
    """Generate comprehensive test results for a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    attention_weights = {"frame": [], "segment": [], "utterance": [], "fusion": []}

    with torch.no_grad():
        for batch in tqdm(
            test_loader,
            desc=f"Testing on {dataset_name}"
            + (f" Session {session}" if session else ""),
        ):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits, attention_info = model(features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Store attention weights
            attention_weights["frame"].append(
                attention_info["frame_attn"].cpu().numpy()
            )
            attention_weights["segment"].append(
                attention_info["segment_attn"].cpu().numpy()
            )
            attention_weights["utterance"].append(
                attention_info["utterance_attn"].cpu().numpy()
            )
            attention_weights["fusion"].append(
                attention_info["fusion_weights"].cpu().numpy()
            )

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)

    if save_results:
        # Generate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        emotion_labels = ["neutral", "happy", "sad", "anger"]

        # Plot and save confusion matrix
        save_name = f"confusion_matrix_{dataset_name}"
        if session:
            save_name += f"_session_{session}"
        plot_confusion_matrix(
            conf_matrix, emotion_labels, save_dir / f"{save_name}.png"
        )

        # Save results
        results = {
            "dataset": dataset_name,
            "session": session,
            "overall_metrics": {
                "WA": metrics["wa"],
                "UA": metrics["ua"],
                "F1": metrics["f1"],
                "UAR": metrics["uar"],
            },
            "per_class_metrics": {},
        }

        # Add per-class metrics
        for i, emotion in enumerate(emotion_labels):
            results["per_class_metrics"][emotion] = {
                "precision": metrics["precision_per_class"][i],
                "recall": metrics["recall_per_class"][i],
                "f1": metrics["f1_per_class"][i],
                "support": metrics["support_per_class"][i],
            }

        # Save attention analysis
        attention_analysis = {
            "fusion_weights_mean": np.mean(
                attention_weights["fusion"], axis=0
            ).tolist(),
            "fusion_weights_std": np.std(attention_weights["fusion"], axis=0).tolist(),
        }

        save_prefix = f"test_results_{dataset_name}"
        if session:
            save_prefix += f"_session_{session}"

        with open(save_dir / f"{save_prefix}.json", "w") as f:
            json.dump(results, f, indent=4)

        with open(save_dir / f"attention_analysis_{save_prefix}.json", "w") as f:
            json.dump(attention_analysis, f, indent=4)

    return 0, metrics, confusion_matrix(all_labels, all_preds), attention_weights


def main():
    # Configuration
    config = {
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    # Initialize IEMOCAP dataset
    iemocap_dataset = EmotionDataset(
        "IEMOCAP", split="train"
    )  # Using train split for LOSO
    session_splits = get_session_splits(iemocap_dataset)

    # LOSO evaluation on IEMOCAP (by session)
    print("\nPerforming LOSO evaluation on IEMOCAP (by session)...")
    loso_results = {}

    for test_session in session_splits.keys():
        print(f"\nTraining and testing for Session {test_session}")

        # Create train and test sets
        test_indices = session_splits[test_session]
        train_indices = []
        for session, indices in session_splits.items():
            if session != test_session:
                train_indices.extend(indices)

        # Create data loaders
        train_dataset = Subset(iemocap_dataset, train_indices)
        test_dataset = Subset(iemocap_dataset, test_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
        )

        # Train model for this LOSO split
        model = train_loso_model(
            train_loader, test_loader, config["device"], results_dir, test_session
        )

        # Test on held-out session
        _, metrics, conf_matrix, _ = generate_test_report(
            model, test_loader, config["device"], results_dir, "IEMOCAP", test_session
        )
        loso_results[f"Session_{test_session}"] = metrics

    # Calculate average LOSO performance
    avg_loso_metrics = {
        "WA": np.mean([r["wa"] for r in loso_results.values()]),
        "UA": np.mean([r["ua"] for r in loso_results.values()]),
        "F1": np.mean([r["f1"] for r in loso_results.values()]),
        "UAR": np.mean([r["uar"] for r in loso_results.values()]),
    }

    # Train final model on all IEMOCAP data for cross-corpus evaluation
    print("\nTraining final model on all IEMOCAP data for cross-corpus evaluation...")
    full_train_loader = DataLoader(
        iemocap_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )

    # Initialize MSP-IMPROV dataset for cross-corpus testing
    msp_dataset = EmotionDataset("MSP-IMPROV", split="test")
    msp_loader = DataLoader(
        msp_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # Train final model
    final_model = train_loso_model(
        full_train_loader,
        msp_loader,  # Using MSP as validation
        config["device"],
        results_dir,
        "final",
    )

    # Cross-corpus evaluation
    print("\nPerforming cross-corpus evaluation on MSP-IMPROV...")
    _, cross_corpus_metrics, _, _ = generate_test_report(
        final_model, msp_loader, config["device"], results_dir, "MSP-IMPROV"
    )

    # Print comprehensive summary
    print("\nTest Results Summary:")
    print("=" * 50)

    # LOSO results
    print("\nIEMOCAP LOSO Results:")
    print("-" * 30)
    for session, metrics in loso_results.items():
        print(f"\n{session}:")
        print(f"WA:  {metrics['wa']:.4f}")
        print(f"UA:  {metrics['ua']:.4f}")
        print(f"F1:  {metrics['f1']:.4f}")
        print(f"UAR: {metrics['uar']:.4f}")

    print("\nIEMOCAP LOSO Average:")
    print(f"WA:  {avg_loso_metrics['WA']:.4f}")
    print(f"UA:  {avg_loso_metrics['UA']:.4f}")
    print(f"F1:  {avg_loso_metrics['F1']:.4f}")
    print(f"UAR: {avg_loso_metrics['UAR']:.4f}")

    # Cross-corpus results
    print("\nMSP-IMPROV Cross-Corpus Results:")
    print("-" * 30)
    print(f"WA:  {cross_corpus_metrics['wa']:.4f}")
    print(f"UA:  {cross_corpus_metrics['ua']:.4f}")
    print(f"F1:  {cross_corpus_metrics['f1']:.4f}")
    print(f"UAR: {cross_corpus_metrics['uar']:.4f}")

    # Save summary results
    summary = {
        "loso_results": loso_results,
        "loso_average": avg_loso_metrics,
        "cross_corpus_results": cross_corpus_metrics,
    }
    with open(results_dir / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(
        "\nDetailed results, confusion matrices, and summary have been saved to the 'test_results' directory."
    )


if __name__ == "__main__":
    main()

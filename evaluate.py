import torch
import numpy as np
import json
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
from metrics import calculate_comprehensive_metrics
from utils import plot_confusion_matrix


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
    all_saliency_scores = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits, aux_outputs = model(features)
            loss_dict = criterion(logits, labels, aux_outputs)
            loss = loss_dict["total_loss"]

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Collect saliency scores - flatten each sample's sequence
            saliency_scores = aux_outputs["saliency_scores"].detach().cpu().numpy()
            # Flatten each sample in the batch and extend the list
            for sample_saliency in saliency_scores:
                all_saliency_scores.extend(sample_saliency.flatten())

    # Calculate comprehensive metrics
    class_names = ["neutral", "happy", "sad", "anger"]
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, class_names)

    # Add loss
    metrics["loss"] = total_loss / len(val_loader)

    # Add session information
    if session_info:
        metrics["session_info"] = session_info

    # Analyze saliency patterns - now we have a flat list
    saliency_array = np.array(all_saliency_scores)
    mean_saliency = np.mean(saliency_array)
    std_saliency = np.std(saliency_array)

    metrics["saliency_analysis"] = {
        "mean_saliency": float(mean_saliency),
        "std_saliency": float(std_saliency),
        "high_saliency_ratio": float(np.mean(saliency_array > 0.5)),
        "saliency_entropy": float(
            -np.mean(saliency_array * np.log(saliency_array + 1e-8))
        ),
    }

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
        print(
            f"Saliency Analysis -> Mean: {metrics['saliency_analysis']['mean_saliency']:.4f}±{metrics['saliency_analysis']['std_saliency']:.4f}, "
            f"High Ratio: {metrics['saliency_analysis']['high_saliency_ratio']:.4f}, "
            f"Entropy: {metrics['saliency_analysis']['saliency_entropy']:.4f}"
        )

    # Log to wandb with session info
    if epoch is not None:
        # Create figure for wandb logging
        fig = plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), class_names, None
        )
        wandb.log(
            {
                f"confusion_matrix_{eval_type}_session_{session_info}_epoch_{epoch}": wandb.Image(
                    fig
                ),
                f"accuracy_{eval_type}_session_{session_info}": metrics["accuracy"],
                f"wa_{eval_type}_session_{session_info}": metrics["wa"],
                f"uar_{eval_type}_session_{session_info}": metrics["uar"],
                f"f1_weighted_{eval_type}_session_{session_info}": metrics[
                    "f1_weighted"
                ],
                f"f1_macro_{eval_type}_session_{session_info}": metrics["f1_macro"],
                f"loss_{eval_type}_session_{session_info}": metrics["loss"],
                f"saliency_mean_{eval_type}_session_{session_info}": metrics[
                    "saliency_analysis"
                ]["mean_saliency"],
                f"high_saliency_ratio_{eval_type}_session_{session_info}": metrics[
                    "saliency_analysis"
                ]["high_saliency_ratio"],
                f"saliency_entropy_{eval_type}_session_{session_info}": metrics[
                    "saliency_analysis"
                ]["saliency_entropy"],
            }
        )
        plt.close(fig)  # Close the figure after logging to wandb

    # Only save results if this is the final evaluation (no epoch number)
    if epoch is None and save_dir is not None:
        # Ensure save directory exists
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save confusion matrix plot
        cm_path = save_dir / f"confusion_matrix_{eval_type}.png"
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), class_names, cm_path
        )

        # Save comprehensive metrics
        report_path = save_dir / "results.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics
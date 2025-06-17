import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from metrics import calculate_comprehensive_metrics


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Training function for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_saliency_scores = []

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch in progress_bar:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, aux_outputs = model(features)
        loss_dict = criterion(logits, labels, aux_outputs)
        total_batch_loss = loss_dict["total_loss"]

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_batch_loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Collect saliency scores - flatten each sample's sequence
        saliency_scores = aux_outputs["saliency_scores"].detach().cpu().numpy()
        # Flatten each sample in the batch and extend the list
        for sample_saliency in saliency_scores:
            all_saliency_scores.extend(sample_saliency.flatten())

        progress_bar.set_postfix(
            {
                "loss": f"{total_batch_loss.item():.4f}",
                "emotion_loss": f"{loss_dict['emotion_loss'].item():.4f}",
                "saliency_reg": f"{loss_dict.get('saliency_reg', 0):.4f}",
            }
        )

    # Calculate comprehensive metrics
    class_names = ["neutral", "happy", "sad", "anger"]
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, class_names)

    # Add loss
    metrics["loss"] = total_loss / len(train_loader)

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

    return metrics

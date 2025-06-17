import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


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
        self, input_dim=768, num_classes=4, min_segment_length=4, max_segments=15
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

    def detect_emotional_saliency(self, features: torch.Tensor) -> torch.Tensor:
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

    def adaptive_saliency_pooling(
        self, features: torch.Tensor, saliency: torch.Tensor, top_k_ratio: float = 0.3
    ) -> torch.Tensor:
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

    def process_multi_scale(
        self, features: torch.Tensor, saliency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with adaptive emotional saliency detection.

        Args:
            x: [batch_size, seq_len, input_dim] - emotion2vec features

        Returns:
            logits: [batch_size, num_classes]
            aux_outputs: dict with saliency maps and other info
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

        # Auxiliary outputs for analysis and visualization
        aux_outputs = {
            "saliency_scores": saliency_scores,
            "local_features": local_features,
            "global_features": global_features,
            "combined_features": combined_features,
        }

        return logits, aux_outputs


class AdaptiveSaliencyLoss(nn.Module):
    """
    Combined loss function for adaptive emotional saliency model.
    Includes emotion classification and saliency regularization.
    """

    def __init__(self, class_weights=None, saliency_weight=0.1, diversity_weight=0.05):
        super().__init__()
        self.emotion_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.saliency_weight = saliency_weight
        self.diversity_weight = diversity_weight

    def saliency_regularization(self, saliency_scores: torch.Tensor) -> torch.Tensor:
        """
        Regularize saliency scores to be neither too sparse nor too uniform.
        """
        # Encourage moderate sparsity (not everything is salient, but not too sparse)
        mean_saliency = saliency_scores.mean()
        target_saliency = 0.3  # Target ~30% of frames to be highly salient
        sparsity_loss = F.mse_loss(
            mean_saliency, torch.tensor(target_saliency, device=mean_saliency.device)
        )

        # Encourage diversity in saliency (avoid all frames having same score)
        saliency_std = saliency_scores.std(dim=1).mean()  # Average std across batch
        target_std = 0.2  # Target reasonable variation in saliency
        diversity_loss = F.mse_loss(
            saliency_std, torch.tensor(target_std, device=saliency_std.device)
        )

        return sparsity_loss + diversity_loss

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, aux_outputs: dict
    ) -> dict:
        """
        Compute combined loss.

        Args:
            logits: [batch_size, num_classes] - emotion predictions
            targets: [batch_size] - emotion labels
            aux_outputs: dict with auxiliary outputs from model

        Returns:
            loss_dict: dict with total loss and component losses
        """
        # Main emotion classification loss
        emotion_loss = self.emotion_loss(logits, targets)

        total_loss = emotion_loss
        loss_dict = {
            "total_loss": total_loss,
            "emotion_loss": emotion_loss,
        }

        # Add saliency regularization
        if self.saliency_weight > 0 and "saliency_scores" in aux_outputs:
            saliency_reg = self.saliency_regularization(aux_outputs["saliency_scores"])

            total_loss = total_loss + self.saliency_weight * saliency_reg
            loss_dict["saliency_reg"] = saliency_reg
            loss_dict["total_loss"] = total_loss

        # Add feature diversity regularization (encourage using both local and global)
        if self.diversity_weight > 0:
            local_norm = aux_outputs["local_features"].norm(dim=1).mean()
            global_norm = aux_outputs["global_features"].norm(dim=1).mean()

            # Encourage balanced use of both feature types
            balance_loss = F.mse_loss(local_norm, global_norm)

            total_loss = total_loss + self.diversity_weight * balance_loss
            loss_dict["balance_loss"] = balance_loss
            loss_dict["total_loss"] = total_loss

        return loss_dict


# Convenience function matching your interface
def create_model_and_criterion(
    input_dim=768, num_classes=4, class_weights=None, device="cuda"
):
    """
    Create model and criterion for adaptive emotional saliency detection.
    """
    model = (
        AdaptiveEmotionalSaliency(
            input_dim=input_dim,
            num_classes=num_classes,
            min_segment_length=4,
            max_segments=15,
        )
        .to(device)
        .float()
    )

    criterion = AdaptiveSaliencyLoss(
        class_weights=class_weights,
        saliency_weight=0.1,  # Weight for saliency regularization
        diversity_weight=0.05,  # Weight for feature balance
    )

    return model, criterion


# Training helper function
def train_step_adaptive(model, criterion, features, labels, optimizer):
    """
    Training step for adaptive saliency model.
    """
    optimizer.zero_grad()

    logits, aux_outputs = model(features)
    loss_dict = criterion(logits, labels, aux_outputs)

    total_loss = loss_dict["total_loss"]
    total_loss.backward()
    optimizer.step()

    return loss_dict, logits


# Visualization helper
def visualize_saliency(model, features, labels=None, save_path=None):
    """
    Visualize learned emotional saliency maps.

    Args:
        model: trained AdaptiveEmotionalSaliency model
        features: [batch_size, seq_len, input_dim]
        labels: optional emotion labels for plotting
        save_path: optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    with torch.no_grad():
        logits, aux_outputs = model(features)
        saliency = aux_outputs["saliency_scores"]  # [batch, seq_len]
        predictions = torch.softmax(logits, dim=1)

    # Emotion labels for visualization
    emotion_names = ["Neutral", "Happy", "Sad", "Angry"]  # Adjust based on your classes

    # Plot first few examples
    n_samples = min(4, len(saliency))
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        ax = axes[i]

        # Plot saliency scores
        time_steps = np.arange(len(saliency[i]))
        saliency_np = saliency[i].cpu().numpy()

        # Create filled plot
        ax.fill_between(
            time_steps, saliency_np, alpha=0.3, color="blue", label="Saliency"
        )
        ax.plot(time_steps, saliency_np, color="blue", linewidth=2)

        # Add threshold line
        threshold = 0.5
        ax.axhline(
            y=threshold, color="red", linestyle="--", alpha=0.7, label="Threshold"
        )

        # Highlight most salient regions
        high_saliency = saliency_np > threshold
        if np.any(high_saliency):
            ax.fill_between(
                time_steps,
                0,
                1,
                where=high_saliency,
                alpha=0.2,
                color="red",
                transform=ax.get_xaxis_transform(),
                label="High Saliency",
            )

        # Set labels and title
        pred_idx = torch.argmax(predictions[i]).item()
        pred_conf = predictions[i][pred_idx].item()

        title = f"Sample {i+1} - Predicted: {emotion_names[pred_idx]} ({pred_conf:.3f})"
        if labels is not None:
            true_idx = labels[i].item()
            title += f" | True: {emotion_names[true_idx]}"

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Steps (frames)")
        ax.set_ylabel("Emotional Saliency")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics
        mean_saliency = np.mean(saliency_np)
        max_saliency = np.max(saliency_np)
        salient_ratio = np.mean(saliency_np > threshold)

        stats_text = f"Mean: {mean_saliency:.3f} | Max: {max_saliency:.3f} | Salient: {salient_ratio:.1%}"
        ax.text(
            0.02,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saliency visualization saved to: {save_path}")

    plt.show()

    # Return saliency statistics for analysis
    saliency_stats = {
        "mean_saliency": saliency.mean().item(),
        "std_saliency": saliency.std().item(),
        "max_saliency": saliency.max().item(),
        "min_saliency": saliency.min().item(),
        "salient_frame_ratio": (saliency > 0.5).float().mean().item(),
    }

    return saliency_stats


# Additional analysis function
def analyze_saliency_patterns(model, dataloader, emotion_names=None, device="cuda"):
    """
    Analyze saliency patterns across different emotions.

    Args:
        model: trained AdaptiveEmotionalSaliency model
        dataloader: DataLoader with emotion data
        emotion_names: list of emotion class names
        device: device to run analysis on

    Returns:
        analysis_results: dict with saliency statistics per emotion
    """
    if emotion_names is None:
        emotion_names = ["Neutral", "Happy", "Sad", "Angry"]

    model.eval()
    emotion_saliency = {name: [] for name in emotion_names}

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            _, aux_outputs = model(batch_features)
            saliency_scores = aux_outputs["saliency_scores"]  # [batch, seq_len]

            # Group by emotion labels
            for i, label in enumerate(batch_labels):
                emotion_idx = label.item()
                if emotion_idx < len(emotion_names):
                    emotion_saliency[emotion_names[emotion_idx]].append(
                        saliency_scores[i].cpu().numpy()
                    )

    # Analyze patterns
    analysis_results = {}
    for emotion, saliency_list in emotion_saliency.items():
        if len(saliency_list) > 0:
            all_saliency = np.concatenate(saliency_list)
            analysis_results[emotion] = {
                "mean_saliency": np.mean(all_saliency),
                "std_saliency": np.std(all_saliency),
                "median_saliency": np.median(all_saliency),
                "salient_frame_ratio": np.mean(all_saliency > 0.5),
                "num_samples": len(saliency_list),
            }

    return analysis_results

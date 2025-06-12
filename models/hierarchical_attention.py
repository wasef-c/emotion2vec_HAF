import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

# Add this at the top of your dataset.py file, before importing FunASR
os.environ["FUNASR_CACHE_DIR"] = "/tmp/funasr_cache"
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Linear projections and reshape
        q = self.query(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # Reshape and project
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        return self.proj(context), attn


class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x, attn_weights


class EntropyRegularizedFusion(nn.Module):
    def __init__(self, hidden_dim, entropy_weight=0.1):
        super().__init__()
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )
        self.entropy_weight = entropy_weight

    def forward(self, frame_pooled, segment_pooled, utterance_pooled):
        fusion_input = torch.cat(
            [frame_pooled, segment_pooled, utterance_pooled], dim=-1
        )
        fusion_weights = self.fusion_network(fusion_input)

        # Calculate entropy regularization loss
        entropy = -torch.sum(fusion_weights * torch.log(fusion_weights + 1e-8), dim=-1)
        max_entropy = torch.log(torch.tensor(3.0))  # log(3) for 3 scales
        entropy_loss = -self.entropy_weight * torch.mean(entropy / max_entropy)

        # Weighted combination
        fused_features = (
            fusion_weights[:, 0:1] * frame_pooled
            + fusion_weights[:, 1:2] * segment_pooled
            + fusion_weights[:, 2:3] * utterance_pooled
        )

        return fused_features, fusion_weights, entropy_loss


class HierarchicalAttention(nn.Module):
    def __init__(self, input_dim=768, num_classes=4):
        super().__init__()
        self.hidden_dim = 384

        # Single projection for all scales
        self.proj = nn.Linear(input_dim, self.hidden_dim)

        # Three transformers for different temporal scales
        self.frame_transformer = TemporalAttentionBlock(self.hidden_dim, 8)
        self.segment_transformer = TemporalAttentionBlock(self.hidden_dim, 8)
        self.utterance_transformer = TemporalAttentionBlock(self.hidden_dim, 8)

        # Replace simple fusion with entropy-regularized fusion
        self.fusion = EntropyRegularizedFusion(self.hidden_dim, entropy_weight=0.1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, num_classes),
        )

    def create_windows(self, x, window_size, stride=None):
        """Create overlapping windows from sequence.
        Args:
            x: [batch_size, seq_len, hidden_dim]
            window_size: Size of each window
            stride: Step size between windows (default: window_size//2 for 50% overlap)
        Returns:
            windows: [batch_size, num_windows, window_size, hidden_dim]
        """
        if stride is None:
            stride = window_size // 2  # 50% overlap by default

        batch_size, seq_len, hidden_dim = x.shape

        # Handle short sequences
        if seq_len < window_size:
            pad_size = window_size - seq_len
            x = F.pad(x, (0, 0, 0, pad_size))
        else:
            # Ensure we can create at least one complete window
            remaining = (seq_len - window_size) % stride
            pad_size = (stride - remaining) % stride if remaining != 0 else 0
            if pad_size > 0:
                x = F.pad(x, (0, 0, 0, pad_size))

        # Create windows and fix dimensions
        windows = x.unfold(1, window_size, stride).transpose(-1, -2)
        return windows  # [batch, num_windows, window_size, hidden]

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] - Raw emotion2vec features
        """
        batch_size = x.size(0)

        # Project input features
        x = self.proj(x)  # [batch, seq_len, hidden]

        # Create different temporal scales
        frame_windows = self.create_windows(
            x, window_size=4
        )  # [batch, n_windows, 4, hidden]
        segment_windows = self.create_windows(
            x, window_size=16
        )  # [batch, n_windows, 16, hidden]

        # Process each scale independently
        # Frame level (4-frame windows)
        frame_features = frame_windows.reshape(
            -1, 4, self.hidden_dim
        )  # [batch*n_windows, 4, hidden]
        frame_features, frame_attn = self.frame_transformer(frame_features)
        frame_features = frame_features.reshape(batch_size, -1, 4, self.hidden_dim)
        frame_pooled = frame_features.mean(dim=(1, 2))  # [batch, hidden]

        # Segment level (16-frame windows)
        segment_features = segment_windows.reshape(
            -1, 16, self.hidden_dim
        )  # [batch*n_windows, 16, hidden]
        segment_features, segment_attn = self.segment_transformer(segment_features)
        segment_features = segment_features.reshape(batch_size, -1, 16, self.hidden_dim)
        segment_pooled = segment_features.mean(dim=(1, 2))  # [batch, hidden]

        # Utterance level (full sequence)
        utterance_features, utterance_attn = self.utterance_transformer(x)
        utterance_pooled = utterance_features.mean(dim=1)  # [batch, hidden]

        # Use entropy-regularized fusion
        fused_features, fusion_weights, entropy_loss = self.fusion(
            frame_pooled, segment_pooled, utterance_pooled
        )

        # Classification
        logits = self.classifier(fused_features)

        return logits, {
            "frame_attn": frame_attn,
            "segment_attn": segment_attn,
            "utterance_attn": utterance_attn,
            "fusion_weights": fusion_weights,
            "entropy_loss": entropy_loss,
        }

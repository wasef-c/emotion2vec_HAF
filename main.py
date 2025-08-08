"""
Main script for emotion recognition with curriculum learning
"""

import torch
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
import wandb
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# CRITICAL: Set deterministic behavior IMMEDIATELY at import time
def force_deterministic_behavior():
    """Force deterministic behavior with maximum settings"""
    seed = 238
    
    print(f"🎲 FORCING DETERMINISTIC BEHAVIOR (seed={seed})")
    
    # Python random
    random.seed(seed)
    
    # NumPy random  
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch random
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Environment variables BEFORE any CUDA operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_DETERMINISTIC_OPS'] = '1'
    
    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Force deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print("✅ DETERMINISTIC BEHAVIOR FORCED")

# Call this IMMEDIATELY at import
force_deterministic_behavior()

from config import Config
from functions import (
    EmotionDataset,
    AdaptiveEmotionalSaliency,
    AdaptiveSaliencyLoss,
    train_epoch,
    evaluate,
    get_session_splits,
    get_speaker_splits,
    create_train_val_test_splits,
    create_msp_train_val_splits,
    create_msp_speaker_train_val_splits,
    create_subset_dataset,
    apply_custom_difficulty,
    create_data_loader,
    calculate_metrics,
    difficulty_plot,
    difficulty_analysis,
    confidence_vs_difficulty_analysis,
    confidence_vs_difficulty_session_analysis,
)


def set_all_seeds(seed=42):
    """Set all possible random seeds for reproducibility"""
    print(f"🎲 Setting all random seeds to {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Additional environment variables for determinism
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_DETERMINISTIC_OPS'] = '1'
    
    # Use deterministic algorithms where possible
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"⚠️  Could not set deterministic algorithms: {e}")
    
    # Additional PyTorch deterministic settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic behavior for specific operations
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    print("✅ All seeds set for reproducibility")


def run_single_experiment(
    config,
    train_dataset_raw,
    test_dataset_raw,
    train_dataset_name,
    test_dataset_name,
    difficulty_method=None,
    vad_weights=None,
    experiment_name=None,
):
    """Run a single emotion recognition experiment with LOSO evaluation"""

    # CRITICAL: Set seeds at start of EVERY experiment run
    set_all_seeds(42)
    
    # Clear GPU cache for clean state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Setup
    device = config.device
    if experiment_name:
        wandb.init(
            project=config.wandb_project, name=experiment_name, config=config.to_dict()
        )

    print(f"🚀 Starting experiment: {experiment_name or 'Single Run'}")
    print(f"Device: {device}")
    print(f"Difficulty method: {difficulty_method or 'original'}")

    # Get splits for LOSO evaluation
    if config.training_direction == "IEMOCAP_to_MSP":
        # IEMOCAP training with sessions, MSP testing
        session_splits = get_session_splits(train_dataset_raw)
        if config.single_test_session:
            test_sessions = [config.single_test_session]
            print(
                f"🎯 Single session mode: Testing on session {config.single_test_session}"
            )
        else:
            test_sessions = list(session_splits.keys())
            print(f"🔄 Full LOSO mode: Testing on sessions {test_sessions}")
    elif config.training_direction == "MSP_to_IEMOCAP":
        # MSP training with sessions for LOSO, IEMOCAP testing
        session_splits = get_session_splits(train_dataset_raw)  # Get MSP sessions for LOSO
        if config.single_test_session:
            test_sessions = [config.single_test_session]
            print(
                f"🎯 Single session mode: Testing MSP session {config.single_test_session}"
            )
        else:
            test_sessions = list(session_splits.keys())
            print(f"🔄 Full LOSO mode: Testing on MSP sessions {test_sessions}")
    else:
        # Default fallback
        test_sessions = [1]  # Dummy session for single iteration
        print(f"🔄 Default mode: Single iteration")

    # Apply custom difficulty calculation if specified (once per experiment)
    if difficulty_method and difficulty_method != "original":
        print(f"🔧 Applying {difficulty_method} to training dataset ({train_dataset_name})")
        apply_custom_difficulty(
            train_dataset_raw,
            difficulty_method,
            config.expected_vad,
            vad_weights,
        )
        print(f"🔧 Applying {difficulty_method} to test dataset ({test_dataset_name})")
        apply_custom_difficulty(
            test_dataset_raw,
            difficulty_method,
            config.expected_vad,
            vad_weights,
        )

    # Store results
    all_results = []
    all_was = []
    all_uars = []
    all_f1s_weighted = []
    all_f1s_macro = []
    cross_corpus_results = []

    # LOSO evaluation
    for test_session in test_sessions:
        print(f"\n{'='*60}")
        print(f"Leave-One-Session-Out: Testing on Session {test_session}")
        print(f"{'='*60}")
        
        # Create train/val/test splits based on training direction
        if config.training_direction == "MSP_to_IEMOCAP":
            # MSP training with session-based LOSO, IEMOCAP testing (all data)
            train_indices, val_indices, msp_test_indices = create_train_val_test_splits(
                train_dataset_raw, test_session, val_ratio=0.2
            )
            # Test on all IEMOCAP data (cross-corpus evaluation)
            test_indices = list(range(len(test_dataset_raw.data)))
            print(f"   IEMOCAP Test (all data): {len(test_indices)} samples")
        else:
            # IEMOCAP training (with sessions), MSP testing
            train_indices, val_indices, iemocap_test_indices = create_train_val_test_splits(
                train_dataset_raw, test_session, val_ratio=0.2
            )
            test_indices = list(range(len(test_dataset_raw.data)))

        print(
            f"{train_dataset_name} Train: {len(train_indices)}, {train_dataset_name} Val: {len(val_indices)}, {test_dataset_name} Test: {len(test_indices)}"
        )

        # Debug: Check if indices are valid
        print(f"Sample train indices: {train_indices[:10]}")
        print(f"{train_dataset_name} dataset size: {len(train_dataset_raw)}")
        print(f"{test_dataset_name} dataset size: {len(test_dataset_raw)}")

        # Create datasets based on configuration
        train_dataset = Subset(train_dataset_raw, train_indices)
        val_dataset = Subset(train_dataset_raw, val_indices)
        test_dataset = Subset(test_dataset_raw, test_indices)

        print(
            f"Actual subset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        

        # Create data loaders
        train_loader = create_data_loader(train_dataset, config, is_training=True)
        val_loader = create_data_loader(val_dataset, config, is_training=False)
        test_loader = create_data_loader(test_dataset, config, is_training=False)

        print(
            f"Data loader batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
        )
        print(f"Batch size: {config.batch_size}")
        print(
            f"Expected train batches: {len(train_dataset) // config.batch_size + (1 if len(train_dataset) % config.batch_size > 0 else 0)}"
        )

        # Create flexible classifier with configurable architecture
        import torch.nn as nn
        import torch.nn.functional as F
        
        class AdvancedClassifier(nn.Module):
            def __init__(self, input_dim=768, hidden_dim=256, num_classes=4, dropout=0.2, 
                         pooling="mean", layer_norm=False, architecture="simple", 
                         feature_normalization=None, input_dropout=0.0, residual_connections=False, **kwargs):
                super().__init__()
                self.architecture = architecture
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.num_classes = num_classes
                self.dropout = dropout
                self.pooling = pooling
                self.layer_norm = layer_norm
                self.feature_normalization = feature_normalization
                self.input_dropout = input_dropout
                self.residual_connections = residual_connections
                
                # Input feature processing
                if input_dropout > 0:
                    self.input_dropout_layer = nn.Dropout(input_dropout)
                else:
                    self.input_dropout_layer = None
                
                # Build architecture based on type
                if architecture == "simple":
                    self._build_simple_architecture()
                elif architecture == "transformer":
                    self._build_transformer_architecture(**kwargs)
                elif architecture == "temporal_cnn":
                    self._build_temporal_cnn_architecture(**kwargs)
                elif architecture == "convformer":
                    self._build_convformer_architecture(**kwargs)
                elif architecture == "residual_mlp":
                    self._build_residual_mlp_architecture(**kwargs)
                else:
                    raise ValueError(f"Unknown architecture: {architecture}")
            
            def _build_simple_architecture(self):
                """Build simple MLP classifier with advanced pooling"""
                # Calculate effective input dimension based on pooling strategy
                if self.pooling == "max_mean":
                    effective_input_dim = self.input_dim * 2
                elif self.pooling == "first_last":
                    effective_input_dim = self.input_dim * 2
                elif self.pooling == "multi_scale":
                    effective_input_dim = self.input_dim * 3  # mean + max + attention
                else:
                    effective_input_dim = self.input_dim
                
                # Build classifier layers
                layers = []
                
                # Optional layer normalization at input
                if self.layer_norm:
                    layers.append(nn.LayerNorm(effective_input_dim))
                
                # First linear layer
                layers.extend([
                    nn.Linear(effective_input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
                
                # Optional layer normalization before output
                if self.layer_norm:
                    layers.append(nn.LayerNorm(self.hidden_dim))
                
                # Output layer
                layers.append(nn.Linear(self.hidden_dim, self.num_classes))
                
                self.classifier = nn.Sequential(*layers)
                
                # For attention pooling
                if self.pooling in ["attention", "multi_scale"]:
                    self.attention = nn.Sequential(
                        nn.Linear(self.input_dim, self.hidden_dim // 4),
                        nn.Tanh(),
                        nn.Linear(self.hidden_dim // 4, 1)
                    )
            
            def _build_transformer_architecture(self, num_heads=8, num_layers=4, **kwargs):
                """Build transformer-based classifier"""
                # Input projection
                self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
                
                # Learnable positional encodings
                self.positional_encoding = nn.Parameter(torch.randn(1000, self.hidden_dim))
                
                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout,
                    activation='relu',
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Output layers
                self.layer_norm_out = nn.LayerNorm(self.hidden_dim)
                self.classifier = nn.Sequential(
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 2, self.num_classes)
                )
            
            def _build_temporal_cnn_architecture(self, kernel_sizes=[3, 5, 7, 9], **kwargs):
                """Build temporal CNN with multi-scale convolutions"""
                # Multi-scale convolution branches
                self.conv_branches = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(self.input_dim, self.hidden_dim // len(kernel_sizes), 
                                 kernel_size=k, padding=k//2),
                        nn.BatchNorm1d(self.hidden_dim // len(kernel_sizes)),
                        nn.ReLU(),
                        nn.Dropout(self.dropout)
                    ) for k in kernel_sizes
                ])
                
                # Additional temporal processing
                self.temporal_conv = nn.Sequential(
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
                
                # Global pooling and classification
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 2, self.num_classes)
                )
            
            def _build_convformer_architecture(self, num_heads=8, num_layers=2, **kwargs):
                """Build hybrid CNN-Transformer classifier"""
                # CNN feature extraction
                self.cnn_features = nn.Sequential(
                    nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=7, padding=3),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU()
                )
                
                # Transformer processing
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Output layers
                self.layer_norm_out = nn.LayerNorm(self.hidden_dim)
                self.classifier = nn.Sequential(
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 2, self.num_classes)
                )
            
            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                batch_size, seq_len, input_dim = x.shape
                
                # Apply feature normalization
                if self.feature_normalization == "l2":
                    x = F.normalize(x, p=2, dim=-1)
                elif self.feature_normalization == "layer":
                    x = F.layer_norm(x, (input_dim,))
                
                # Apply input dropout
                if self.input_dropout_layer is not None:
                    x = self.input_dropout_layer(x)
                
                if self.architecture == "simple":
                    return self._forward_simple(x)
                elif self.architecture == "transformer":
                    return self._forward_transformer(x)
                elif self.architecture == "temporal_cnn":
                    return self._forward_temporal_cnn(x)
                elif self.architecture == "convformer":
                    return self._forward_convformer(x)
                elif self.architecture == "residual_mlp":
                    return self._forward_residual_mlp(x)
                else:
                    raise ValueError(f"Unknown architecture: {self.architecture}")
            
            def _forward_simple(self, x):
                """Forward pass for simple architecture"""
                # Apply pooling strategy
                if self.pooling == "mean":
                    pooled = x.mean(dim=1)
                elif self.pooling == "max_mean":
                    max_pooled = x.max(dim=1)[0]
                    mean_pooled = x.mean(dim=1)
                    pooled = torch.cat([max_pooled, mean_pooled], dim=1)
                elif self.pooling == "first_last":
                    first_frame = x[:, 0, :]
                    last_frame = x[:, -1, :]
                    pooled = torch.cat([first_frame, last_frame], dim=1)
                elif self.pooling == "attention":
                    attention_weights = self.attention(x)
                    attention_weights = F.softmax(attention_weights, dim=1)
                    pooled = (x * attention_weights).sum(dim=1)
                elif self.pooling == "multi_scale":
                    # Combine multiple pooling strategies
                    mean_pooled = x.mean(dim=1)
                    max_pooled = x.max(dim=1)[0]
                    attention_weights = self.attention(x)
                    attention_weights = F.softmax(attention_weights, dim=1)
                    attention_pooled = (x * attention_weights).sum(dim=1)
                    pooled = torch.cat([mean_pooled, max_pooled, attention_pooled], dim=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling}")
                
                logits = self.classifier(pooled)
                return {"logits": logits}
            
            def _forward_transformer(self, x):
                """Forward pass for transformer architecture"""
                # Input projection
                x = self.input_projection(x)
                
                # Add positional encoding
                seq_len = x.size(1)
                pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0)
                x = x + pos_encoding
                
                # Transformer processing
                x = self.transformer_encoder(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Layer norm and classification
                x = self.layer_norm_out(x)
                logits = self.classifier(x)
                
                return {"logits": logits}
            
            def _forward_temporal_cnn(self, x):
                """Forward pass for temporal CNN architecture"""
                x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
                
                # Multi-scale convolution
                branch_outputs = []
                for branch in self.conv_branches:
                    branch_outputs.append(branch(x))
                
                # Concatenate branches
                x = torch.cat(branch_outputs, dim=1)
                
                # Additional temporal processing
                x = self.temporal_conv(x)
                
                # Global pooling
                x = self.global_pool(x)
                x = x.squeeze(2)
                
                # Classification
                logits = self.classifier(x)
                
                return {"logits": logits}
            
            def _build_residual_mlp_architecture(self, num_layers=4, **kwargs):
                """Build residual MLP with skip connections"""
                # Calculate effective input dimension based on pooling strategy
                if self.pooling == "max_mean":
                    effective_input_dim = self.input_dim * 2
                elif self.pooling == "first_last":
                    effective_input_dim = self.input_dim * 2
                elif self.pooling == "multi_scale":
                    effective_input_dim = self.input_dim * 3
                else:
                    effective_input_dim = self.input_dim
                
                # Input projection to hidden dim
                self.input_projection = nn.Linear(effective_input_dim, self.hidden_dim)
                
                # Residual blocks
                self.residual_blocks = nn.ModuleList()
                for i in range(num_layers):
                    block = nn.Sequential(
                        nn.LayerNorm(self.hidden_dim) if self.layer_norm else nn.Identity(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.Dropout(self.dropout)
                    )
                    self.residual_blocks.append(block)
                
                # Output layer
                self.output_layer = nn.Linear(self.hidden_dim, self.num_classes)
                
                # For attention pooling
                if self.pooling in ["attention", "multi_scale"]:
                    self.attention = nn.Sequential(
                        nn.Linear(self.input_dim, self.hidden_dim // 4),
                        nn.Tanh(),
                        nn.Linear(self.hidden_dim // 4, 1)
                    )
            
            def _forward_residual_mlp(self, x):
                """Forward pass for Residual MLP architecture"""
                # Apply pooling
                if self.pooling == "mean":
                    pooled = x.mean(dim=1)
                elif self.pooling == "max":
                    pooled = x.max(dim=1)[0]
                elif self.pooling == "max_mean":
                    max_pooled = x.max(dim=1)[0]
                    mean_pooled = x.mean(dim=1)
                    pooled = torch.cat([max_pooled, mean_pooled], dim=1)
                elif self.pooling == "first_last":
                    first_frame = x[:, 0, :]
                    last_frame = x[:, -1, :]
                    pooled = torch.cat([first_frame, last_frame], dim=1)
                elif self.pooling == "attention":
                    attention_weights = self.attention(x)
                    attention_weights = F.softmax(attention_weights, dim=1)
                    pooled = (x * attention_weights).sum(dim=1)
                elif self.pooling == "multi_scale":
                    mean_pooled = x.mean(dim=1)
                    max_pooled = x.max(dim=1)[0]
                    attention_weights = self.attention(x)
                    attention_weights = F.softmax(attention_weights, dim=1)
                    attention_pooled = (x * attention_weights).sum(dim=1)
                    pooled = torch.cat([mean_pooled, max_pooled, attention_pooled], dim=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling}")
                
                # Input projection
                x = F.relu(self.input_projection(pooled))
                
                # Residual blocks
                for block in self.residual_blocks:
                    if self.residual_connections:
                        x = x + block(x)  # Skip connection
                    else:
                        x = block(x)
                
                # Output
                logits = self.output_layer(x)
                return {"logits": logits}
            
            def _forward_convformer(self, x):
                """Forward pass for ConvFormer architecture"""
                x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
                
                # CNN feature extraction
                x = self.cnn_features(x)
                
                # Prepare for transformer
                x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]
                
                # Transformer processing
                x = self.transformer(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Layer norm and classification
                x = self.layer_norm_out(x)
                logits = self.classifier(x)
                
                return {"logits": logits}
        
        # Get classifier config from experiment config
        classifier_config = getattr(config, 'classifier_config', {})
        
        model = AdvancedClassifier(
            input_dim=768,
            hidden_dim=classifier_config.get('hidden_dim', 256),
            num_classes=4,
            dropout=classifier_config.get('dropout', 0.2),
            pooling=classifier_config.get('pooling', 'mean'),
            layer_norm=classifier_config.get('layer_norm', False),
            architecture=classifier_config.get('architecture', 'simple'),
            num_heads=classifier_config.get('num_heads', 8),
            num_layers=classifier_config.get('num_layers', 4),
            kernel_sizes=classifier_config.get('kernel_sizes', [3, 5, 7, 9]),
            feature_normalization=classifier_config.get('feature_normalization', None),
            input_dropout=classifier_config.get('input_dropout', 0.0),
            residual_connections=classifier_config.get('residual_connections', False)
        ).to(device)
        
        print(f"🏗️  Classifier Config: {classifier_config}")

        # Create class weights
        class_weights = torch.tensor(
            [
                config.class_weights["neutral"],
                config.class_weights["happy"],
                config.class_weights["sad"],
                config.class_weights["anger"],
            ]
        ).to(device)

        # Configure focal loss if specified
        use_focal_loss = getattr(config, 'focal_loss', False)
        focal_alpha = getattr(config, 'focal_alpha', 0.25)
        focal_gamma = getattr(config, 'focal_gamma', 2.0)
        label_smoothing = getattr(config, 'label_smoothing', 0.0)
        
        # Check if we should use difficulty scaling
        use_difficulty_scaling = getattr(config, 'use_difficulty_scaling', True)
        
        criterion = AdaptiveSaliencyLoss(
            class_weights=class_weights, 
            saliency_weight=0.0, 
            diversity_weight=0.0,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            use_difficulty_scaling=use_difficulty_scaling
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Configure learning rate scheduler based on config
        scheduler = None
        if config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=30 #config.num_epochs
            )
        elif config.lr_scheduler == "cosine_warmup":
            # Cosine annealing with warmup
            warmup_epochs = getattr(config, 'warmup_epochs', 10)
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs - warmup_epochs)
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        elif config.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config.num_epochs//3, gamma=0.1
            )
        elif config.lr_scheduler == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.95
            )
        elif config.lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )
        # If None or unrecognized, no scheduler will be used

        # Training loop
        best_val_uar = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(config.num_epochs):
            # Initialize curriculum_epochs to avoid UnboundLocalError
            curriculum_epochs = 0
            
            # Update curriculum sampler epoch
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

                # Log curriculum progress if using curriculum learning
                if config.use_curriculum_learning:
                    sampler = train_loader.sampler
                    epoch_progress = min(1.0, epoch / sampler.curriculum_epochs)
                    progress = sampler._calculate_progress(epoch_progress)
                    curriculum_epochs = sampler.curriculum_epochs
                    if epoch < sampler.curriculum_epochs:
                        if hasattr(sampler, 'start_percentile'):
                            # Smooth quantile-based progression
                            current_percentile = (
                                sampler.start_percentile + 
                                (sampler.end_percentile - sampler.start_percentile) * progress
                            )
                            total_samples = len(sampler.sorted_indices)
                            curriculum_samples = int((current_percentile / 100.0) * total_samples)
                            current_metric = current_percentile
                            metric_name = "percentile"
                        else:
                            # Fallback to threshold-based
                            current_threshold = (
                                sampler.start_threshold
                                + (sampler.end_threshold - sampler.start_threshold)
                                * progress
                            )
                            curriculum_samples = len(
                                [
                                    idx
                                    for idx, diff in zip(
                                        sampler.sorted_indices, sampler.sorted_difficulties
                                    )
                                    if diff <= current_threshold
                                ]
                            )
                            current_metric = current_threshold
                            metric_name = "threshold"
                    else:
                        current_metric = 100.0 if hasattr(sampler, 'start_percentile') else 1.0
                        curriculum_samples = len(sampler.all_indices)
                        metric_name = "percentile" if hasattr(sampler, 'start_percentile') else "threshold"

                    if wandb.run:
                        wandb.log(
                            {
                                f"session_{test_session}/curriculum_{metric_name}": current_metric,
                                f"session_{test_session}/curriculum_samples": curriculum_samples,
                                f"session_{test_session}/curriculum_progress": progress,
                            }
                        )
                else:
                    curriculum_epochs = 0
            # Train
            model.train()  # Ensure training mode
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # Validate
            val_metrics = evaluate(model, val_loader, criterion, device)

            # Update learning rate if scheduler is configured
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                if config.lr_scheduler == "plateau":
                    # ReduceLROnPlateau needs the validation metric
                    scheduler.step(val_metrics["uar"])
                else:
                    # Other schedulers don't need metrics
                    scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}: LR {current_lr:.1e} -> {new_lr:.1e} (scheduler: {config.lr_scheduler})")
            
            # Early stopping (but not before epoch 15)
            if val_metrics["uar"] > best_val_uar + config.early_stopping_min_delta:
                best_val_uar = val_metrics["uar"]
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"Val UAR: {val_metrics['uar']:.4f} ⭐ NEW BEST")
            else:
                if epoch >= curriculum_epochs:
                    patience_counter += 1
                print(
                    f"Val UAR: {val_metrics['uar']:.4f} (Best: {best_val_uar:.4f}, Patience: {patience_counter}/{config.early_stopping_patience})"
                )
                if patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Log to wandb
            if wandb.run:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log(
                    {
                        f"session_{test_session}/train/uar": train_metrics["uar"],
                        f"session_{test_session}/train/loss": train_metrics["loss"],
                        f"session_{test_session}/val/uar": val_metrics["uar"],
                        f"session_{test_session}/val/loss": val_metrics["loss"],
                        f"session_{test_session}/val/wa": val_metrics["wa"],
                        f"session_{test_session}/val/f1_macro": val_metrics["f1_macro"],
                        f"session_{test_session}/val/f1_weighted": val_metrics[
                            "f1_weighted"
                        ],
                        f"session_{test_session}/val/accuracy": val_metrics["accuracy"],
                        f"session_{test_session}/learning_rate": current_lr,
                    }
                )

                # Also log aggregated validation metrics across all sessions so far
                if len(all_uars) > 0:
                    wandb.log(
                        {
                            "aggregated/val/uar": np.mean(all_uars + [val_metrics["uar"]]),
                            "aggregated/val/wa": np.mean(all_was + [val_metrics["wa"]]),
                        }
                    )

        # Load best model and evaluate
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Cross-corpus evaluation on different dataset
        print(f"\nCross-corpus evaluation on {test_dataset_name}...")
        cross_metrics, cross_preds, cross_labels, cross_speaker_ids, cross_difficulties, cross_logits = evaluate(model, test_loader, criterion, device, return_predictions=True, return_logits=True)

        # LOSO in-domain evaluation on held-out session from training dataset
        if config.training_direction == "IEMOCAP_to_MSP":
            # Create test set from held-out IEMOCAP session
            loso_test_indices = [i for i in range(len(train_dataset_raw.data)) if train_dataset_raw.data[i]["session"] == test_session]
            loso_test_dataset = create_subset_dataset(train_dataset_raw, loso_test_indices)
            loso_test_loader = create_data_loader(loso_test_dataset, config, is_training=False)
            print(f"LOSO in-domain evaluation on {train_dataset_name} session {test_session} ({len(loso_test_indices)} samples)...")
        else:
            # MSP_to_IEMOCAP: Create test set from held-out MSP session  
            loso_test_indices = [i for i in range(len(train_dataset_raw.data)) if train_dataset_raw.data[i]["session"] == test_session]
            loso_test_dataset = create_subset_dataset(train_dataset_raw, loso_test_indices) 
            loso_test_loader = create_data_loader(loso_test_dataset, config, is_training=False)
            print(f"LOSO in-domain evaluation on {train_dataset_name} session {test_session} ({len(loso_test_indices)} samples)...")
        
        loso_metrics, loso_preds, loso_labels, loso_speaker_ids, loso_difficulties, loso_logits = evaluate(model, loso_test_loader, criterion, device, return_predictions=True, return_logits=True)
        
        # Enhanced per-session wandb logging
        if wandb.run:
            # Create session-specific confusion matrices
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create confusion matrices based on configuration
            cross_corpus_cm = confusion_matrix(cross_labels, cross_preds, labels=[0, 1, 2, 3])
            difficulty_plot_cross = difficulty_plot(cross_preds, cross_labels, cross_difficulties, f"{test_dataset_name}-CrossCorpus")
            
            # Add difficulty analysis for cross-corpus
            cross_difficulty_plots = difficulty_analysis(cross_difficulties, cross_labels, f"{test_dataset_name}-CrossCorpus")

            loso_cm = confusion_matrix(loso_labels, loso_preds, labels=[0, 1, 2, 3])
            difficulty_plot_loso = difficulty_plot(loso_preds, loso_labels, loso_difficulties, f"{train_dataset_name}-LOSO-S{test_session}")
            
            # Add difficulty analysis for LOSO
            loso_difficulty_plots = difficulty_analysis(loso_difficulties, loso_labels, f"{train_dataset_name}-LOSO-S{test_session}")
            
            # Per-session confidence analysis
            cross_confidence_plot = confidence_vs_difficulty_session_analysis(
                cross_logits, cross_difficulties, cross_labels, cross_preds, f"{test_dataset_name}-CrossCorpus-S{test_session}"
            )
            loso_confidence_plot = confidence_vs_difficulty_session_analysis(
                loso_logits, loso_difficulties, loso_labels, loso_preds, f"{train_dataset_name}-LOSO-S{test_session}"
            )

            
            # Calculate metrics for session and cross-corpus
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            cross_wa = accuracy_score(cross_labels, cross_preds)
            cross_uar = balanced_accuracy_score(cross_labels, cross_preds)
            loso_wa = accuracy_score(loso_labels, loso_preds)
            loso_uar = balanced_accuracy_score(loso_labels, loso_preds)
            
            # Speaker-specific analysis removed - not needed for main analysis
            
            # Plot and log session confusion matrix
            # Plot cross-corpus confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cross_corpus_cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=["Neutral", "Happy", "Sad", "Anger"], yticklabels=["Neutral", "Happy", "Sad", "Anger"])
            plt.title(f'Cross-Corpus ({train_dataset_name}→{test_dataset_name}) Session {test_session}\nWA: {cross_wa:.4f}, UAR: {cross_uar:.4f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cross_corpus_cm_img = wandb.Image(plt)
            plt.close()

            
            # Plot LOSO confusion matrix (held-out session)
            plt.figure(figsize=(8, 6))
            sns.heatmap(loso_cm, annot=True, fmt='d', cmap='Oranges',
                       xticklabels=["Neutral", "Happy", "Sad", "Anger"], yticklabels=["Neutral", "Happy", "Sad", "Anger"])
            plt.title(f'LOSO In-Domain ({train_dataset_name} Session {test_session})\nWA: {loso_wa:.4f}, UAR: {loso_uar:.4f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            loso_cm_img = wandb.Image(plt)
            plt.close()
            
            # Speaker-specific analysis removed for cleaner logging
            
            
            
            
            
            # Log only essential results and confusion matrices  
            log_dict = {
                # Core metrics - LOSO is in-domain, other dataset is cross-corpus
                f"session_{test_session}/loso_wa": loso_metrics["wa"],
                f"session_{test_session}/loso_uar": loso_metrics["uar"],
                f"session_{test_session}/cross_corpus_wa": cross_metrics["wa"],
                f"session_{test_session}/cross_corpus_uar": cross_metrics["uar"],
                
                # Confusion matrices and difficulty plots
                f"session_{test_session}/cm_cross_corpus": cross_corpus_cm_img,
                f"session_{test_session}/cm_loso": loso_cm_img,
                f"session_{test_session}/difficulty_cross_corpus": difficulty_plot_cross,
                f"session_{test_session}/difficulty_loso": difficulty_plot_loso,
                
                # Confidence analysis plots
                f"session_{test_session}/confidence_cross_corpus": cross_confidence_plot,
                f"session_{test_session}/confidence_loso": loso_confidence_plot,

                # Add difficulty analysis plots
                **cross_difficulty_plots,
                **loso_difficulty_plots,

            }
            
            # Speaker confusion matrices removed - cleaner wandb interface
            
            wandb.log(log_dict)
            
            print(f"📊 Logged session {test_session}: Cross-corpus WA={cross_metrics['wa']:.4f}, LOSO WA={loso_metrics['wa']:.4f}, LOSO UAR={loso_metrics['uar']:.4f}")

        # Store results - LOSO is the main in-domain result, cross-corpus is generalization
        all_results.append(loso_metrics["accuracy"])  # LOSO accuracy for main results
        all_was.append(loso_metrics["wa"])
        all_uars.append(loso_metrics["uar"])
        all_f1s_weighted.append(loso_metrics["f1_weighted"])
        all_f1s_macro.append(loso_metrics["f1_macro"])
        cross_corpus_results.append(cross_metrics)  # Cross-corpus for generalization
        
        # Store predictions for proper aggregated confusion matrix
        if 'all_test_preds' not in locals():
            all_test_preds = []
            all_test_labels = []
            all_cross_preds = []
            all_cross_labels = []
            all_test_difficulties = []
            all_cross_difficulties = []
            all_test_logits = []
            all_cross_logits = []
        
        all_test_preds.extend(loso_preds)
        all_test_labels.extend(loso_labels)
        all_cross_preds.extend(cross_preds)  # Cross-corpus = test results (IEMOCAP)
        all_cross_labels.extend(cross_labels)
        all_test_difficulties.extend(loso_difficulties)
        all_cross_difficulties.extend(cross_difficulties)
        all_test_logits.extend(loso_logits)
        all_cross_logits.extend(cross_logits)  # Same as test for confidence analysis

        print(f"Session {test_session} Results:")
        print(f"  {test_dataset_name} UAR: {cross_metrics['uar']:.4f}")
        print(f"  {train_dataset_name} LOSO UAR: {loso_metrics['uar']:.4f}")

    # Calculate final averaged results
    loso_key = f"{train_dataset_name.lower()}_loso_results"
    final_results = {
        loso_key: {
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
        },
        "cross_corpus_results": {
            "accuracy": {
                "mean": float(np.mean([r["accuracy"] for r in cross_corpus_results])),
                "std": float(np.std([r["accuracy"] for r in cross_corpus_results])),
            },
            "uar": {
                "mean": float(np.mean([r["uar"] for r in cross_corpus_results])),
                "std": float(np.std([r["uar"] for r in cross_corpus_results])),
            },
            "f1_macro": {
                "mean": float(np.mean([r["f1_macro"] for r in cross_corpus_results])),
                "std": float(np.std([r["f1_macro"] for r in cross_corpus_results])),
            },
        },
    }

    # Calculate final confusion matrices for logging
    if wandb.run:
        # Use the correctly collected predictions from each session's LOSO evaluation
        # (predictions are already stored from the proper per-session evaluations)

        # Create confusion matrices
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        # IEMOCAP confusion matrix (from proper LOSO predictions)
        iemocap_cm = confusion_matrix(
            all_test_labels, all_test_preds, labels=[0, 1, 2, 3]
        )
        
        # Calculate UAR and WA directly from the aggregated confusion matrix
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        aggregated_accuracy = accuracy_score(all_test_labels, all_test_preds)
        aggregated_uar = balanced_accuracy_score(all_test_labels, all_test_preds)
        
        # Also show the per-session averages for comparison
        per_session_uar = final_results[loso_key]["uar"]["mean"]
        per_session_uar_std = final_results[loso_key]["uar"]["std"]
        per_session_wa = final_results[loso_key]["wa"]["mean"]
        per_session_wa_std = final_results[loso_key]["wa"]["std"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            iemocap_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neutral", "Happy", "Sad", "Anger"],
            yticklabels=["Neutral", "Happy", "Sad", "Anger"],
        )
        plt.title(
            f"{train_dataset_name} LOSO Aggregated Confusion Matrix\n"
            f"Aggregated: UAR={aggregated_uar:.4f}, WA={aggregated_accuracy:.4f}\n"
            f"Per-Session Avg: UAR={per_session_uar:.4f}±{per_session_uar_std:.4f}, WA={per_session_wa:.4f}±{per_session_wa_std:.4f}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        wandb.log(
            {
                f"final/{train_dataset_name.lower()}_loso_confusion_matrix": wandb.Image(
                    plt,
                    caption=f"{train_dataset_name} LOSO Aggregated - UAR: {aggregated_uar:.4f}, WA: {aggregated_accuracy:.4f} | Per-Session Avg - UAR: {per_session_uar:.4f}±{per_session_uar_std:.4f}, WA: {per_session_wa:.4f}±{per_session_wa_std:.4f}",
                )
            }
        )
        plt.close()
        
        # Final aggregated difficulty analysis
        print(f"\n🔍 Final Aggregated Difficulty Analysis")
        final_difficulty_plot_test = difficulty_plot(
            all_test_preds, all_test_labels, all_test_difficulties, 
            f"Final-Aggregated-{test_dataset_name}"
        )
        final_difficulty_analysis_test = difficulty_analysis(
            all_test_difficulties, all_test_labels, 
            f"Final-Aggregated-{test_dataset_name}"
        )
        
        final_difficulty_plot_cross = difficulty_plot(
            all_cross_preds, all_cross_labels, all_cross_difficulties, 
            f"Final-Aggregated-CrossCorpus"
        )
        final_difficulty_analysis_cross = difficulty_analysis(
            all_cross_difficulties, all_cross_labels, 
            f"Final-Aggregated-CrossCorpus"
        )
        
        # Aggregated confidence/uncertainty analysis with all sessions combined
        print(f"\n🔍 Final Aggregated Confidence Analysis")
        final_confidence_plot_loso = confidence_vs_difficulty_session_analysis(
            all_test_logits, all_test_difficulties, all_test_labels, all_test_preds, 
            f"Final-Aggregated-{train_dataset_name}-LOSO-AllSessions"
        )
        final_confidence_plot_cross = confidence_vs_difficulty_session_analysis(
            all_cross_logits, all_cross_difficulties, all_cross_labels, all_cross_preds, 
            f"Final-Aggregated-CrossCorpus-AllSessions"
        )

        # Cross-corpus confusion matrix
        cross_cm = confusion_matrix(
            all_cross_labels, all_cross_preds, labels=[0, 1, 2, 3]
        )
        
        # Calculate metrics directly from aggregated cross-corpus predictions
        cross_aggregated_accuracy = accuracy_score(all_cross_labels, all_cross_preds)
        cross_aggregated_uar = balanced_accuracy_score(all_cross_labels, all_cross_preds)
        
        # Per-session averages for comparison
        cross_per_session_uar = final_results["cross_corpus_results"]["uar"]["mean"]
        cross_per_session_uar_std = final_results["cross_corpus_results"]["uar"]["std"]
        cross_per_session_wa = final_results["cross_corpus_results"]["accuracy"]["mean"]
        cross_per_session_wa_std = final_results["cross_corpus_results"]["accuracy"]["std"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cross_cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=["Neutral", "Happy", "Sad", "Anger"],
            yticklabels=["Neutral", "Happy", "Sad", "Anger"],
        )
        plt.title(
            f"Cross-Corpus ({test_dataset_name}) Aggregated Confusion Matrix\n"
            f"Aggregated: UAR={cross_aggregated_uar:.4f}, WA={cross_aggregated_accuracy:.4f}\n"
            f"Per-Session Avg: UAR={cross_per_session_uar:.4f}±{cross_per_session_uar_std:.4f}, WA={cross_per_session_wa:.4f}±{cross_per_session_wa_std:.4f}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        wandb.log(
            {
                "final/cross_corpus_confusion_matrix": wandb.Image(
                    plt,
                    caption=f"Cross-Corpus {test_dataset_name} Aggregated - UAR: {cross_aggregated_uar:.4f}, WA: {cross_aggregated_accuracy:.4f} | Per-Session Avg - UAR: {cross_per_session_uar:.4f}±{cross_per_session_uar_std:.4f}, WA: {cross_per_session_wa:.4f}±{cross_per_session_wa_std:.4f}",
                )
            }
        )
        plt.close()

        # Log final metrics (both aggregated and per-session averages)
        wandb.log(
            {
                # Dynamic naming based on actual datasets
                f"final/{train_dataset_name.lower()}_loso/uar_avg": final_results[loso_key]["uar"]["mean"],
                f"final/{train_dataset_name.lower()}_loso/uar_std": final_results[loso_key]["uar"]["std"],
                f"final/{train_dataset_name.lower()}_loso/wa_avg": final_results[loso_key]["wa"]["mean"], 
                f"final/{train_dataset_name.lower()}_loso/wa_std": final_results[loso_key]["wa"]["std"],
                f"final/cross_corpus_{test_dataset_name.lower()}/uar_avg": final_results["cross_corpus_results"]["uar"]["mean"],
                f"final/cross_corpus_{test_dataset_name.lower()}/uar_std": final_results["cross_corpus_results"]["uar"]["std"],
                f"final/cross_corpus_{test_dataset_name.lower()}/wa_avg": final_results["cross_corpus_results"]["accuracy"]["mean"],
                f"final/cross_corpus_{test_dataset_name.lower()}/wa_std": final_results["cross_corpus_results"]["accuracy"]["std"],
                
                # # Aggregated metrics (for comparison)
                # "final/msp_training/uar_aggregated": cross_aggregated_uar,
                # "final/cross_corpus_iemocap/uar_aggregated": aggregated_uar,
                # "final/cross_corpus_iemocap/wa_aggregated": aggregated_accuracy,
                # "final/msp_training/wa_aggregated": cross_aggregated_accuracy,
                
                # Final aggregated difficulty analysis plots
                f"final/Difficulty_Aggregated_{test_dataset_name}": final_difficulty_plot_test,
                f"final/Difficulty_Aggregated_CrossCorpus": final_difficulty_plot_cross,
                **final_difficulty_analysis_test,
                **final_difficulty_analysis_cross,
                
            }
        )
        
        # Add summary metrics for easy access in wandb UI
        if wandb.run:
            # LOSO metrics (main in-domain performance)
            wandb.summary[f"{train_dataset_name.lower()}_loso_uar_mean"] = final_results[loso_key]["uar"]["mean"]
            wandb.summary[f"{train_dataset_name.lower()}_loso_uar_std"] = final_results[loso_key]["uar"]["std"]
            wandb.summary[f"{train_dataset_name.lower()}_loso_wa_mean"] = final_results[loso_key]["wa"]["mean"]
            wandb.summary[f"{train_dataset_name.lower()}_loso_wa_std"] = final_results[loso_key]["wa"]["std"]
            
            # Cross-corpus metrics (generalization performance)
            wandb.summary[f"cross_corpus_{test_dataset_name.lower()}_uar_mean"] = final_results["cross_corpus_results"]["uar"]["mean"]
            wandb.summary[f"cross_corpus_{test_dataset_name.lower()}_uar_std"] = final_results["cross_corpus_results"]["uar"]["std"]
            wandb.summary[f"cross_corpus_{test_dataset_name.lower()}_wa_mean"] = final_results["cross_corpus_results"]["accuracy"]["mean"]
            wandb.summary[f"cross_corpus_{test_dataset_name.lower()}_wa_std"] = final_results["cross_corpus_results"]["accuracy"]["std"]

    # Print final summary
    # Final training run with ALL sessions for best cross-corpus performance
    print(f"\n{'='*80}")
    print(f"🎯 FINAL TRAINING WITH ALL SESSIONS")
    print(f"Training on ALL {train_dataset_name} sessions for optimal cross-corpus performance")
    print(f"{'='*80}")
    
    # Create training set with all sessions (no held-out sessions)
    all_session_indices = []
    for session_id in session_splits.keys():
        all_session_indices.extend(session_splits[session_id])
    
    # Create a validation split from all training data (20% for validation)
    import random
    random.shuffle(all_session_indices)
    val_split_size = int(0.2 * len(all_session_indices))
    all_train_indices = all_session_indices[val_split_size:]
    all_val_indices = all_session_indices[:val_split_size]
    
    print(f"All-sessions split: {len(all_train_indices)} train, {len(all_val_indices)} val")
    
    train_subset_all = Subset(train_dataset_raw, all_train_indices)
    val_subset_all = Subset(train_dataset_raw, all_val_indices)
    
    # Create data loaders for all-sessions training
    train_loader_all = create_data_loader(train_subset_all, config, is_training=True)
    val_loader_all = create_data_loader(val_subset_all, config, is_training=False)
    
    # Train model on all sessions (reuse existing training infrastructure)
    print(f"🚀 Training on ALL {len(all_session_indices)} samples from {train_dataset_name}")
    
    # Create a fresh model for all-sessions training (AdvancedClassifier is already defined in this scope)
    classifier_config = getattr(config, 'classifier_config', {})
    final_model = AdvancedClassifier(
        input_dim=768,
        hidden_dim=classifier_config.get('hidden_dim', 256),
        num_classes=4,
        architecture=classifier_config.get('architecture', 'simple'),
        pooling=classifier_config.get('pooling', 'mean'),
        layer_norm=classifier_config.get('layer_norm', False),
        dropout=classifier_config.get('dropout', 0.1),
        input_dropout=classifier_config.get('input_dropout', 0.0)
    ).to(device)
    
    # Setup training components (copy exact pattern from main LOSO training)
    class_weights = torch.tensor(
        [
            config.class_weights["neutral"],
            config.class_weights["happy"],
            config.class_weights["sad"],
            config.class_weights["anger"],
        ]
    ).to(device)

    # Configure focal loss if specified
    use_focal_loss = getattr(config, 'focal_loss', False)
    focal_alpha = getattr(config, 'focal_alpha', 0.25)
    focal_gamma = getattr(config, 'focal_gamma', 2.0)
    label_smoothing = getattr(config, 'label_smoothing', 0.0)
    
    # Check if we should use difficulty scaling
    use_difficulty_scaling = getattr(config, 'use_difficulty_scaling', True)
    
    # Setup training components (ALWAYS use AdaptiveSaliencyLoss like LOSO does)
    criterion = AdaptiveSaliencyLoss(
        class_weights=class_weights, 
        saliency_weight=0.0, 
        diversity_weight=0.0,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        use_difficulty_scaling=use_difficulty_scaling
    )
    
    optimizer = optim.AdamW(final_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Training loop with validation and early stopping (same as LOSO training)
    best_val_uar = 0
    best_model_state = None
    patience_counter = 0
    
    final_model.train()
    for epoch in range(config.num_epochs):
        if hasattr(train_loader_all.sampler, "set_epoch"):
            train_loader_all.sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(
            final_model, train_loader_all, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = evaluate(final_model, val_loader_all, criterion, device)
        
        # Early stopping logic (same as LOSO training)
        if val_metrics["uar"] > best_val_uar:
            best_val_uar = val_metrics["uar"]
            patience_counter = 0
            best_model_state = final_model.state_dict().copy()
            print(f"All-sessions epoch {epoch}: Val UAR: {val_metrics['uar']:.4f} ⭐ NEW BEST")
        else:
            patience_counter += 1
            print(f"All-sessions epoch {epoch}: Val UAR: {val_metrics['uar']:.4f} (Best: {best_val_uar:.4f}, Patience: {patience_counter}/{config.early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (patience: {config.early_stopping_patience})")
            break
    
    # Load best model
    if best_model_state is not None:
        final_model.load_state_dict(best_model_state)
        print(f"Loaded best model with val UAR: {best_val_uar:.4f}")
    
    # Test on cross-corpus (same as before, but with model trained on all sessions)
    print(f"🎯 Testing final model on {test_dataset_name} cross-corpus...")
    final_cross_metrics, final_cross_preds, final_cross_labels, final_cross_speaker_ids, final_cross_difficulties, final_cross_logits = evaluate(
        final_model, test_loader, criterion, device, return_predictions=True, return_logits=True
    )
    
    print(f"🏆 FINAL ALL-SESSIONS CROSS-CORPUS RESULTS:")
    print(f"   {test_dataset_name} WA: {final_cross_metrics['wa']:.4f}")
    print(f"   {test_dataset_name} UAR: {final_cross_metrics['uar']:.4f}")
    print(f"   (Compare to LOSO avg: WA={final_results['cross_corpus_results']['accuracy']['mean']:.4f}, UAR={final_results['cross_corpus_results']['uar']['mean']:.4f})")
    
    # Difficulty vs accuracy analysis for all-sessions cross-corpus test
    print(f"\n🔍 Final All-Sessions Cross-Corpus Difficulty Analysis")
    final_all_sessions_difficulty_plot = difficulty_plot(
        final_cross_preds, final_cross_labels, final_cross_difficulties, 
        f"Final-AllSessions-CrossCorpus-{test_dataset_name}"
    )
    final_all_sessions_difficulty_analysis = difficulty_analysis(
        final_cross_difficulties, final_cross_labels, 
        f"Final-AllSessions-CrossCorpus-{test_dataset_name}"
    )
    
    # Confidence analysis for all-sessions cross-corpus test
    final_all_sessions_confidence_plot = confidence_vs_difficulty_session_analysis(
        final_cross_logits, final_cross_difficulties, final_cross_labels, final_cross_preds, 
        f"Final-AllSessions-CrossCorpus-{test_dataset_name}"
    )
    
    # Confusion matrix for all-sessions cross-corpus test
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    final_all_sessions_cm = confusion_matrix(final_cross_labels, final_cross_preds, labels=[0, 1, 2, 3])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        final_all_sessions_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Neutral", "Happy", "Sad", "Anger"],
        yticklabels=["Neutral", "Happy", "Sad", "Anger"]
    )
    plt.title(f"Final All-Sessions Cross-Corpus Confusion Matrix ({test_dataset_name})\nWA: {final_cross_metrics['wa']:.4f}, UAR: {final_cross_metrics['uar']:.4f}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    final_all_sessions_cm_img = wandb.Image(
        plt,
        caption=f"Final All-Sessions Cross-Corpus {test_dataset_name} - WA: {final_cross_metrics['wa']:.4f}, UAR: {final_cross_metrics['uar']:.4f}"
    )
    plt.close()
    
    # Add to results
    final_results["all_sessions_cross_corpus"] = {
        "wa": final_cross_metrics['wa'],
        "uar": final_cross_metrics['uar'],
        "accuracy": final_cross_metrics['accuracy'],
        "f1_weighted": final_cross_metrics['f1_weighted'],
        "f1_macro": final_cross_metrics['f1_macro'],
        "description": f"Cross-corpus performance when trained on ALL {train_dataset_name} sessions"
    }
    
    # Log to wandb
    if wandb.run:
        wandb.log({
            f"final_all_sessions/{test_dataset_name.lower()}_cross_wa": final_cross_metrics['wa'],
            f"final_all_sessions/{test_dataset_name.lower()}_cross_uar": final_cross_metrics['uar'],
            f"final_all_sessions/confusion_matrix_Final-AllSessions-CrossCorpus-{test_dataset_name}": final_all_sessions_cm_img,
            f"final_all_sessions/difficulty_plot_Final-AllSessions-CrossCorpus-{test_dataset_name}": final_all_sessions_difficulty_plot,
            f"final_all_sessions/confidence_plot_Final-AllSessions-CrossCorpus-{test_dataset_name}": final_all_sessions_confidence_plot,
            **{f"final_all_sessions/{k}": v for k, v in final_all_sessions_difficulty_analysis.items()},
        })
        wandb.summary[f"all_sessions_cross_{test_dataset_name.lower()}_wa"] = final_cross_metrics['wa']
        wandb.summary[f"all_sessions_cross_{test_dataset_name.lower()}_uar"] = final_cross_metrics['uar']

    # # Finish wandb run after all logging is complete
    # if wandb.run:
    #     wandb.finish()

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(
        f"{train_dataset_name} LOSO UAR: {final_results[loso_key]['uar']['mean']:.4f} ± {final_results[loso_key]['uar']['std']:.4f}"
    )
    print(
        f"Cross-corpus UAR: {final_results['cross_corpus_results']['uar']['mean']:.4f} ± {final_results['cross_corpus_results']['uar']['std']:.4f}"
    )
    print(
        f"{train_dataset_name} LOSO WA: {final_results[loso_key]['wa']['mean']:.4f} ± {final_results[loso_key]['wa']['std']:.4f}"
    )

    return final_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Emotion Recognition with Curriculum Learning"
    )
    parser.add_argument(
        "--difficulty-method",
        type=str,
        default="original",
        choices=[
            "original",
            "preset",
            "pearson_correlation",
            "spearman_correlation",
            "euclidean_distance",
            "weighted_vad_valence",
            "weighted_vad_balanced",
            "weighted_vad_arousal",
        ],
        help="Difficulty calculation method",
    )
    parser.add_argument(
        "--curriculum-epochs",
        type=int,
        default=15,
        help="Number of curriculum learning epochs",
    )
    parser.add_argument(
        "--curriculum-pacing",
        type=str,
        default="exponential",
        choices=["linear", "exponential", "logarithmic"],
        help="Curriculum pacing function",
    )
    parser.add_argument(
        "--use-speaker-disentanglement",
        action="store_true",
        help="Use speaker disentanglement",
    )
    parser.add_argument(
        "--single-session",
        type=int,
        default=5,
        help="Test on single session (default: 5)",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Experiment name for wandb"
    )
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate"
    )

    args = parser.parse_args()

    # Create config
    config = Config()
    config.curriculum_epochs = args.curriculum_epochs
    config.curriculum_pacing = args.curriculum_pacing
    config.use_speaker_disentanglement = args.use_speaker_disentanglement
    config.single_test_session = args.single_session
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate

    # Load datasets based on configuration
    print("Loading datasets...")
    print(f"🔄 Training direction: {config.training_direction}")
    
    if config.training_direction == "MSP_to_IEMOCAP":
        print("Training on MSP-IMPROV, Testing on IEMOCAP")
        train_dataset_raw = EmotionDataset("MSP-IMPROV", split="train")
        test_dataset_raw = EmotionDataset("IEMOCAP", split="train")
        train_dataset_name = "MSP-IMPROV"
        test_dataset_name = "IEMOCAP"
    else:  # Default: IEMOCAP_to_MSP
        print("Training on IEMOCAP, Testing on MSP-IMPROV")
        train_dataset_raw = EmotionDataset("IEMOCAP", split="train")
        test_dataset_raw = EmotionDataset("MSP-IMPROV", split="train")
        train_dataset_name = "IEMOCAP"
        test_dataset_name = "MSP-IMPROV"

    # Set VAD weights based on method
    vad_weights = None
    if args.difficulty_method == "weighted_vad_valence":
        vad_weights = [0.5, 0.4, 0.1]
    elif args.difficulty_method == "weighted_vad_balanced":
        vad_weights = [0.4, 0.4, 0.2]
    elif args.difficulty_method == "weighted_vad_arousal":
        vad_weights = [0.4, 0.5, 0.1]

    # Run experiment
    results = run_single_experiment(
        config,
        train_dataset_raw,
        test_dataset_raw,
        train_dataset_name,
        test_dataset_name,
        difficulty_method=args.difficulty_method,
        vad_weights=vad_weights,
        experiment_name=args.experiment_name,
    )

    return results


if __name__ == "__main__":
    main()

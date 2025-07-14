#!/usr/bin/env python3
"""
Simplified emotion recognition training script
Defaults to exact 70.8% WA baseline configuration
Only essential parameters are configurable
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
from pathlib import Path

from functions import (
    EmotionDataset,
    AdaptiveEmotionalSaliency,
    AdaptiveSaliencyLoss,
    train_epoch,
    evaluate,
    get_session_splits,
)

class SimpleConfig:
    """Minimal configuration class with 70.8% baseline as defaults"""
    def __init__(self):
        # FIXED BASELINE PARAMETERS (achieved 70.8% WA)
        self.learning_rate = 9e-5
        self.weight_decay = 5e-6
        self.focal_loss = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.label_smoothing = 0.1
        self.num_epochs = 80
        self.batch_size = 16
        self.lr_scheduler = "cosine"
        
        # Class weights (optimized for IEMOCAP)
        self.class_weights = {
            'neutral': 0.8,
            'happy': 3.5,
            'sad': 1.8,
            'anger': 1.6
        }
        
        # Classifier architecture (fixed to working config)
        self.architecture = "simple"
        self.hidden_dim = 1536
        self.pooling = "attention"
        self.layer_norm = True
        self.dropout = 0.4
        self.input_dropout = 0.2
        
        # CONFIGURABLE PARAMETERS
        self.use_speaker_disentanglement = True  # Can be toggled
        self.use_curriculum_learning = True      # Can be toggled
        self.curriculum_epochs = 30              # Can be adjusted
        self.difficulty_method = "pearson_correlation"
        
        # Paths and logging
        self.wandb_project = "emotion2vec_simple"
        self.experiment_name = "baseline"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def create_model_and_criterion(config):
    """Create model and loss function with exact baseline config"""
    
    # Class weights tensor
    class_weights = torch.tensor([
        config.class_weights['neutral'],
        config.class_weights['happy'], 
        config.class_weights['sad'],
        config.class_weights['anger']
    ], dtype=torch.float32).to(config.device)
    
    # Model with fixed architecture
    model = AdaptiveEmotionalSaliency(
        input_dim=1024,  # emotion2vec dimension
        hidden_dim=config.hidden_dim,
        num_classes=4,
        num_speakers=10,  # IEMOCAP speakers
        pooling=config.pooling,
        use_speaker_disentanglement=config.use_speaker_disentanglement,
        dropout=config.dropout,
        input_dropout=config.input_dropout,
        layer_norm=config.layer_norm
    ).to(config.device)
    
    # Loss function with exact baseline config
    criterion = AdaptiveSaliencyLoss(
        class_weights=class_weights,
        saliency_weight=0.0,  # Disabled for baseline
        diversity_weight=0.0,  # Disabled for baseline
        use_focal_loss=config.focal_loss,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing
    )
    
    return model, criterion


def create_optimizer_and_scheduler(model, config):
    """Create optimizer and scheduler with baseline config"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.num_epochs
    )
    
    return optimizer, scheduler


def create_dataloaders(config):
    """Create training and validation dataloaders"""
    
    # Get session splits
    train_sessions, val_sessions = get_session_splits()
    
    # Create datasets
    train_dataset = EmotionDataset(
        sessions=train_sessions,
        use_curriculum=config.use_curriculum_learning,
        curriculum_epochs=config.curriculum_epochs,
        difficulty_method=config.difficulty_method
    )
    
    val_dataset = EmotionDataset(
        sessions=val_sessions,
        use_curriculum=False  # No curriculum for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_simple_model(config):
    """Main training function with simplified config"""
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config=vars(config)
    )
    
    print(f"Starting training with config:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Speaker disentanglement: {config.use_speaker_disentanglement}")
    print(f"  Curriculum learning: {config.use_curriculum_learning}")
    print(f"  Curriculum epochs: {config.curriculum_epochs}")
    
    # Create model, criterion, optimizer
    model, criterion = create_model_and_criterion(config)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Training loop
    best_wa = 0.0
    best_model_path = f"best_model_{config.experiment_name}.pth"
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, 
            config.device, epoch, config.num_epochs
        )
        
        # Validation
        val_metrics = evaluate(model, val_loader, config.device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_wa': val_metrics['wa'],
            'val_uar': val_metrics['uar'],
            'val_cross_uar': val_metrics.get('cross_uar', 0.0),
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_metrics['wa'] > best_wa:
            best_wa = val_metrics['wa']
            torch.save(model.state_dict(), best_model_path)
            print(f"New best WA: {best_wa:.4f}")
        
        print(f"Train Loss: {train_loss:.4f}, Val WA: {val_metrics['wa']:.4f}, Val UAR: {val_metrics['uar']:.4f}")
    
    print(f"\nTraining completed! Best WA: {best_wa:.4f}")
    wandb.finish()
    
    return best_wa


def main():
    parser = argparse.ArgumentParser(description='Simple emotion recognition training')
    
    # Only essential configurable parameters
    parser.add_argument('--lr', type=float, default=9e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--no-speaker', action='store_true', help='Disable speaker disentanglement')
    parser.add_argument('--no-curriculum', action='store_true', help='Disable curriculum learning')
    parser.add_argument('--curriculum-epochs', type=int, default=30, help='Curriculum epochs')
    parser.add_argument('--name', type=str, default='baseline', help='Experiment name')
    
    args = parser.parse_args()
    
    # Create config
    config = SimpleConfig()
    
    # Override with command line arguments
    config.learning_rate = args.lr
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.use_speaker_disentanglement = not args.no_speaker
    config.use_curriculum_learning = not args.no_curriculum
    config.curriculum_epochs = args.curriculum_epochs
    config.experiment_name = args.name
    
    # Train model
    best_wa = train_simple_model(config)
    
    print(f"Final result: {best_wa:.4f} WA")


if __name__ == "__main__":
    main()
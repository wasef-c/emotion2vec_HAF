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
    seed = 42
    
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
    create_train_val_test_splits,
    apply_custom_difficulty,
    create_data_loader,
    calculate_metrics,
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
    iemocap_dataset,
    msp_dataset,
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

    # Get session splits
    session_splits = get_session_splits(iemocap_dataset)

    # Determine test sessions
    if config.single_test_session:
        test_sessions = [config.single_test_session]
        print(
            f"🎯 Single session mode: Testing on session {config.single_test_session}"
        )
    else:
        test_sessions = list(session_splits.keys())
        print(f"🔄 Full LOSO mode: Testing on sessions {test_sessions}")

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

        # Create splits using simplified function
        train_indices, val_indices, test_indices = create_train_val_test_splits(
            iemocap_dataset, test_session, val_ratio=0.2
        )

        print(
            f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
        )

        # Debug: Check if indices are valid
        print(f"Sample train indices: {train_indices[:10]}")
        print(f"Dataset size: {len(iemocap_dataset)}")

        # Create datasets
        train_dataset = Subset(iemocap_dataset, train_indices)
        val_dataset = Subset(iemocap_dataset, val_indices)
        test_dataset = Subset(iemocap_dataset, test_indices)

        print(
            f"Actual subset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        # Apply custom difficulty calculation if specified
        if difficulty_method and difficulty_method != "original":
            apply_custom_difficulty(
                iemocap_dataset,
                train_indices,
                difficulty_method,
                config.expected_vad,
                vad_weights,
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

        # Create simple classifier (recommended architecture from arXiv)
        import torch.nn as nn
        class SimpleClassifier(nn.Module):
            def __init__(self, input_dim=768, hidden_dim=256, num_classes=4, dropout=0.2):
                super().__init__()
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
            
            def forward(self, x):
                # x: [batch, seq_len, input_dim] -> average pool -> [batch, input_dim]
                x = x.mean(dim=1)  # Global average pooling over time dimension
                logits = self.classifier(x)
                return {"logits": logits}
        
        model = SimpleClassifier().to(device)

        # Create class weights
        class_weights = torch.tensor(
            [
                config.class_weights["neutral"],
                config.class_weights["happy"],
                config.class_weights["sad"],
                config.class_weights["anger"],
            ]
        ).to(device)

        criterion = AdaptiveSaliencyLoss(class_weights=class_weights, saliency_weight=0.0, diversity_weight=0.0)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Configure learning rate scheduler based on config
        scheduler = None
        if config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.num_epochs
            )
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
            # Update curriculum sampler epoch
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

                # Log curriculum progress if using curriculum learning
                if config.use_curriculum_learning:
                    sampler = train_loader.sampler
                    epoch_progress = min(1.0, epoch / sampler.curriculum_epochs)
                    progress = sampler._calculate_progress(epoch_progress)

                    if epoch < sampler.curriculum_epochs:
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
                    else:
                        current_threshold = 1.0
                        curriculum_samples = len(sampler.all_indices)

                    if wandb.run:
                        wandb.log(
                            {
                                f"session_{test_session}/curriculum_threshold": current_threshold,
                                f"session_{test_session}/curriculum_samples": curriculum_samples,
                                f"session_{test_session}/curriculum_progress": progress,
                            }
                        )

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
                patience_counter += 1
                print(
                    f"Val UAR: {val_metrics['uar']:.4f} (Best: {best_val_uar:.4f}, Patience: {patience_counter}/{config.early_stopping_patience})"
                )
                # Only allow early stopping after epoch 15
                if (
                    epoch >= 14 and patience_counter >= config.early_stopping_patience
                ):  # epoch 14 = 15th epoch (0-indexed)
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
                        "epoch": epoch,
                        "global_step": epoch
                        + test_session
                        * config.num_epochs,  # Unique step for proper timeline
                    }
                )

                # Also log aggregated validation metrics across all sessions so far
                if len(all_uars) > 0:
                    wandb.log(
                        {
                            "aggregated/val/uar": np.mean(all_uars + [val_metrics["uar"]]),
                            "aggregated/val/wa": np.mean(all_was + [val_metrics["wa"]]),
                            "epoch": epoch,
                            "global_step": epoch + test_session * config.num_epochs,
                        }
                    )

        # Load best model and evaluate
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Test evaluation
        print(f"\nFinal evaluation on Session {test_session}...")
        test_metrics = evaluate(model, test_loader, criterion, device)

        # Cross-corpus evaluation
        print("Cross-corpus evaluation on MSP-IMPROV...")
        msp_loader = create_data_loader(msp_dataset, config, is_training=False)
        msp_metrics = evaluate(model, msp_loader, criterion, device)

        # Store results
        all_results.append(test_metrics["accuracy"])
        all_was.append(test_metrics["wa"])
        all_uars.append(test_metrics["uar"])
        all_f1s_weighted.append(test_metrics["f1_weighted"])
        all_f1s_macro.append(test_metrics["f1_macro"])
        cross_corpus_results.append(msp_metrics)

        print(f"Session {test_session} Results:")
        print(f"  IEMOCAP UAR: {test_metrics['uar']:.4f}")
        print(f"  Cross-corpus UAR: {msp_metrics['uar']:.4f}")

    # Calculate final averaged results
    final_results = {
        "iemocap_results": {
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
        # Collect all test predictions and labels for final confusion matrix
        all_test_preds = []
        all_test_labels = []
        all_cross_preds = []
        all_cross_labels = []

        # Re-evaluate to get predictions for confusion matrix
        for test_session in test_sessions:
            train_indices, val_indices, test_indices = create_train_val_test_splits(
                iemocap_dataset, test_session, val_ratio=0.2
            )
            test_dataset = Subset(iemocap_dataset, test_indices)
            test_loader = create_data_loader(test_dataset, config, is_training=False)

            # Get test predictions
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    features = batch["features"].to(device)
                    labels = batch["label"]
                    outputs = model(features)
                    preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
                    all_test_preds.extend(preds)
                    all_test_labels.extend(labels.numpy())

        # Get cross-corpus predictions
        msp_loader = create_data_loader(msp_dataset, config, is_training=False)
        model.eval()
        with torch.no_grad():
            for batch in msp_loader:
                features = batch["features"].to(device)
                labels = batch["label"]
                outputs = model(features)
                preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
                all_cross_preds.extend(preds)
                all_cross_labels.extend(labels.numpy())

        # Create confusion matrices
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        # IEMOCAP confusion matrix
        iemocap_cm = confusion_matrix(
            all_test_labels, all_test_preds, labels=[0, 1, 2, 3]
        )
        iemocap_uar = final_results["iemocap_results"]["uar"]["mean"]
        iemocap_wa = final_results["iemocap_results"]["wa"]["mean"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            iemocap_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neutral", "Happy", "Sad", "Anger"],
            yticklabels=["Neutral", "Happy", "Sad", "Anger"],
        )
        plt.title(
            f"IEMOCAP LOSO Confusion Matrix\nUAR: {iemocap_uar:.4f}, WA: {iemocap_wa:.4f}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        wandb.log(
            {
                "iemocap_confusion_matrix": wandb.Image(
                    plt,
                    caption=f"IEMOCAP LOSO - UAR: {iemocap_uar:.4f}, WA: {iemocap_wa:.4f}",
                )
            }
        )
        plt.close()

        # Cross-corpus confusion matrix
        cross_cm = confusion_matrix(
            all_cross_labels, all_cross_preds, labels=[0, 1, 2, 3]
        )
        cross_uar = final_results["cross_corpus_results"]["uar"]["mean"]
        cross_wa = final_results["cross_corpus_results"]["accuracy"][
            "mean"
        ]  # Note: cross-corpus uses 'accuracy' not 'wa'

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cross_cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=["Neutral", "Happy", "Sad", "Anger"],
            yticklabels=["Neutral", "Happy", "Sad", "Anger"],
        )
        plt.title(
            f"Cross-Corpus (MSP-IMPROV) Confusion Matrix\nUAR: {cross_uar:.4f}, WA: {cross_wa:.4f}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        wandb.log(
            {
                "cross_corpus_confusion_matrix": wandb.Image(
                    plt,
                    caption=f"Cross-Corpus MSP-IMPROV - UAR: {cross_uar:.4f}, WA: {cross_wa:.4f}",
                )
            }
        )
        plt.close()

        # Log final metrics
        wandb.log(
            {
                "final/iemocap/uar": final_results["iemocap_results"]["uar"]["mean"],
                "final/cross_corpus/uar": final_results["cross_corpus_results"]["uar"][
                    "mean"
                ],
                "final/iemocap/accuracy": final_results["iemocap_results"]["accuracy"][
                    "mean"
                ],
                "final/cross_corpus/accuracy": final_results["cross_corpus_results"][
                    "accuracy"
                ]["mean"],
            }
        )
        wandb.finish()

    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(
        f"IEMOCAP UAR: {final_results['iemocap_results']['uar']['mean']:.4f} ± {final_results['iemocap_results']['uar']['std']:.4f}"
    )
    print(
        f"Cross-corpus UAR: {final_results['cross_corpus_results']['uar']['mean']:.4f} ± {final_results['cross_corpus_results']['uar']['std']:.4f}"
    )
    print(
        f"IEMOCAP WA: {final_results['iemocap_results']['wa']['mean']:.4f} ± {final_results['iemocap_results']['wa']['std']:.4f}"
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

    # Load datasets once
    print("Loading datasets...")
    iemocap_dataset = EmotionDataset("IEMOCAP", split="train")
    msp_dataset = EmotionDataset("MSP-IMPROV", split="train")

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
        iemocap_dataset,
        msp_dataset,
        difficulty_method=args.difficulty_method,
        vad_weights=vad_weights,
        experiment_name=args.experiment_name,
    )

    return results


if __name__ == "__main__":
    main()

"""
Modified main script for running ablation studies.
This allows programmatic configuration changes for systematic experimentation.
"""

from data.dataset import (
    EmotionDataset,
    collate_fn,
    CombinedSampler,
    CurriculumSampler,
)
from models.AdaptiveEmotionalSalience import (
    AdaptiveEmotionalSaliency,
    AdaptiveSaliencyLoss,
    visualize_saliency,
)
from config import ExperimentConfig
from train import train_epoch
from evaluate import enhanced_evaluate
from utils import (
    get_session_splits,
    split_session_data,
    calculate_class_weights,
    get_unique_save_dir,
    plot_confusion_matrix,
    SpeakerGroupedSampler,
)
from metrics import calculate_comprehensive_metrics

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import wandb
import json
import torch.multiprocessing as mp
import os


def run_experiment_with_config(config):
    """
    Run the emotion recognition experiment with a given configuration.
    Returns aggregated results for ablation analysis.
    """
    
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    
    print(f"Experiment name: {config.experiment_name}")

    # Create directories
    save_dir = get_unique_save_dir(config.experiment_name)
    run_name = os.path.basename(config.experiment_name)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize wandb
    wandb.init(project=config.wandb_project, config=config.to_dict(), name=run_name)

    # Load datasets
    print("Loading datasets...")
    iemocap_train = EmotionDataset("IEMOCAP", split="train")
    msp_improv = EmotionDataset("MSP-IMPROV", split="train")

    # Get session splits for IEMOCAP
    session_splits = get_session_splits(iemocap_train)

    # Initialize lists for storing results
    all_results = []
    all_was = []
    all_uars = []
    all_f1s_weighted = []
    all_f1s_macro = []
    all_class_accuracies = []
    all_saliency_analysis = []
    cross_corpus_results = []

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

        # Calculate class weights for this split
        split_class_weights = calculate_class_weights(train_dataset, config.class_names)
        print(f"\nClass distribution for Session {test_session} training split:")
        for class_name, weight in split_class_weights.items():
            print(f"{class_name}: {weight:.2f}")

        # Combine with base weights
        final_class_weights = {
            class_name: base_weight * split_weight
            for (class_name, base_weight), (_, split_weight) in zip(
                config.class_weights.items(), split_class_weights.items()
            )
        }

        # Create class weights tensor
        class_weights = torch.tensor(
            [
                final_class_weights["neutral"],
                final_class_weights["happy"],
                final_class_weights["sad"],
                final_class_weights["anger"],
            ]
        ).to(config.device)

        print("Final class weights (base * split):")
        for class_name, weight in final_class_weights.items():
            print(f"{class_name}: {weight:.2f}")

        # Create data loaders based on configuration
        train_loader = create_data_loader(config, train_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Initialize model
        model = (
            AdaptiveEmotionalSaliency(input_dim=768, num_classes=4)
            .to(config.device)
            .float()
        )
        criterion = AdaptiveSaliencyLoss(class_weights=class_weights)

        if config.pretrain_dir is not None:
            checkpoint = torch.load(config.pretrain_dir)
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pretrained backbone from {config.pretrain_dir}")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )

        # Training loop
        best_metrics = None
        best_model_state = None
        patience_counter = 0
        best_val_uar_so_far = 0

        for epoch in range(config.num_epochs):
            # Update sampler epoch if using curriculum learning
            if config.use_curriculum_learning and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, config.device, epoch
            )

            # Validate on held-out test set
            val_metrics = enhanced_evaluate(
                model,
                test_loader,
                criterion,
                config.device,
                epoch,
                None,
                eval_type="validation",
                session_info=test_session,
            )

            # Update learning rate
            scheduler.step()

            # Early stopping check
            current_val_uar = val_metrics["uar"]
            if current_val_uar > best_val_uar_so_far + config.early_stopping["min_delta"]:
                print("--------------New best model saved---------------")
                best_val_uar_so_far = current_val_uar
                patience_counter = 0
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_uar": val_metrics["uar"],
                }
                best_metrics = val_metrics
                
                # Save best model
                torch.save(
                    best_model_state,
                    save_dir / f"best_model_session_{test_session}.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping["patience"]:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation UAR: {best_val_uar_so_far:.4f}")
                    break

            # Log metrics to wandb
            wandb.log({
                f"session_{test_session}/train_uar": train_metrics["uar"],
                f"session_{test_session}/val_uar": val_metrics["uar"],
                f"session_{test_session}/train_loss": train_metrics["loss"],
                f"session_{test_session}/val_loss": val_metrics["loss"],
            })

        # Load best model for cross-corpus evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state["model_state_dict"])

        # Cross-corpus evaluation on MSP-IMPROV
        print("\nEvaluating on MSP-IMPROV...")
        msp_loader = DataLoader(
            msp_improv,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        msp_metrics = enhanced_evaluate(
            model,
            msp_loader,
            criterion,
            config.device,
            epoch=None,
            save_dir=session_dir / "cross_corpus_evaluation",
            eval_type="cross_corpus",
            session_info=test_session,
        )

        # Store results
        if best_metrics is not None:
            all_results.append(best_metrics["accuracy"])
            all_was.append(best_metrics["wa"])
            all_uars.append(best_metrics["uar"])
            all_f1s_weighted.append(best_metrics["f1_weighted"])
            all_f1s_macro.append(best_metrics["f1_macro"])
            all_class_accuracies.append(best_metrics["class_accuracies"])
            all_saliency_analysis.append(best_metrics["saliency_analysis"])
            cross_corpus_results.append(msp_metrics)

    # Calculate averaged results
    averaged_results = {
        "iemocap_results": {
            "accuracy": {"mean": float(np.mean(all_results)), "std": float(np.std(all_results))},
            "wa": {"mean": float(np.mean(all_was)), "std": float(np.std(all_was))},
            "uar": {"mean": float(np.mean(all_uars)), "std": float(np.std(all_uars))},
            "f1_weighted": {"mean": float(np.mean(all_f1s_weighted)), "std": float(np.std(all_f1s_weighted))},
            "f1_macro": {"mean": float(np.mean(all_f1s_macro)), "std": float(np.std(all_f1s_macro))},
        },
        "cross_corpus_results": {
            "accuracy": {"mean": float(np.mean([r["accuracy"] for r in cross_corpus_results])), 
                        "std": float(np.std([r["accuracy"] for r in cross_corpus_results]))},
            "uar": {"mean": float(np.mean([r["uar"] for r in cross_corpus_results])), 
                   "std": float(np.std([r["uar"] for r in cross_corpus_results]))},
            "f1_macro": {"mean": float(np.mean([r["f1_macro"] for r in cross_corpus_results])), 
                        "std": float(np.std([r["f1_macro"] for r in cross_corpus_results]))},
        }
    }

    # Log final averaged results to wandb
    wandb.log({
        "final_iemocap_uar": averaged_results["iemocap_results"]["uar"]["mean"],
        "final_cross_corpus_uar": averaged_results["cross_corpus_results"]["uar"]["mean"],
        "final_iemocap_accuracy": averaged_results["iemocap_results"]["accuracy"]["mean"],
        "final_cross_corpus_accuracy": averaged_results["cross_corpus_results"]["accuracy"]["mean"],
    })

    # Save final results
    with open(save_dir / "averaged_results.json", "w") as f:
        json.dump(averaged_results, f, indent=4)

    wandb.finish()

    print(f"\nExperiment completed!")
    print(f"IEMOCAP UAR: {averaged_results['iemocap_results']['uar']['mean']:.4f} ± {averaged_results['iemocap_results']['uar']['std']:.4f}")
    print(f"Cross-corpus UAR: {averaged_results['cross_corpus_results']['uar']['mean']:.4f} ± {averaged_results['cross_corpus_results']['uar']['std']:.4f}")

    return averaged_results


def create_data_loader(config, train_dataset):
    """Create appropriate data loader based on configuration."""
    
    if config.use_curriculum_learning and config.use_speaker_disentanglement:
        # Use CombinedSampler for both curriculum learning and speaker grouping
        train_sampler = CombinedSampler(
            train_dataset,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            start_threshold=config.curriculum["start_threshold"],
            end_threshold=config.curriculum["end_threshold"],
            curriculum_epochs=config.curriculum_epochs,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    elif config.use_curriculum_learning:
        # Use CurriculumSampler for curriculum learning only
        train_sampler = CurriculumSampler(
            train_dataset,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            start_threshold=config.curriculum["start_threshold"],
            end_threshold=config.curriculum["end_threshold"],
            curriculum_epochs=config.curriculum_epochs,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    elif config.use_speaker_disentanglement:
        # Use SpeakerGroupedSampler for speaker grouping only
        train_sampler = SpeakerGroupedSampler(
            train_dataset, batch_size=config.batch_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        # Use regular shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    return train_loader


if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig()
    results = run_experiment_with_config(config)
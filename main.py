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


# os.environ["WANDB_MODE"] = "offline"

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method("spawn", force=True)


def main():
    # Configuration - experiment name will be generated automatically
    config = ExperimentConfig()

    # Model configuration
    config.pretrain_dir = None  # "/path/to/pretrained/model.pt"

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
    cross_corpus_results = []  # Initialize as a list
    all_results = []
    all_was = []
    all_uars = []
    all_f1s_weighted = []
    all_f1s_macro = []
    all_class_accuracies = []
    all_saliency_analysis = []  # Changed from fusion_analysis
    cross_corpus_best_metrics = []  # To store best cross-corpus metrics per session

    for test_session in session_splits.keys():
        print(f"\nLeave-One-Session-Out: Testing on Session {test_session}")

        # Create session-specific directories
        session_dir = results_dir / f"session_{test_session}"
        session_dir.mkdir(exist_ok=True)

        # Split IEMOCAP data
        test_indices = session_splits[test_session]  # This is the held-out session
        train_indices = []
        for session, indices in session_splits.items():
            if session != test_session:
                train_indices.extend(indices)  # All other sessions

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

        # Create data loaders
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
            # Load the backbone state dict
            checkpoint = torch.load(config.pretrain_dir)
            # Load the backbone weights directly since we saved only the backbone
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
        best_val_acc = 0
        best_val_wa = 0
        best_val_uar = 0
        best_val_f1_weighted = 0
        best_val_f1_macro = 0
        best_metrics = None
        best_confusion_matrix = None
        best_epoch = -1

        # Early stopping variables
        patience_counter = 0
        best_val_uar_so_far = 0
        best_model_state = None
        best_cross_corpus_mean = -float("inf")
        best_cross_corpus_metrics = None
        best_cross_corpus_epoch = -1

        for epoch in range(config.num_epochs):
            # Update sampler epoch if using curriculum learning
            if config.use_curriculum_learning:
                train_sampler.set_epoch(epoch)

            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, config.device, epoch
            )

            # Validate on held-out test set
            val_metrics = enhanced_evaluate(
                model,
                test_loader,  # Using test_loader for validation
                criterion,
                config.device,
                epoch,
                None,  # Don't save during training
                eval_type="validation",
                session_info=test_session,
            )

            # Update learning rate
            scheduler.step()

            # Early stopping check on validation set
            current_val_uar = val_metrics["uar"]
            if (
                current_val_uar
                > best_val_uar_so_far + config.early_stopping["min_delta"]
            ):
                print("--------------New best model saved---------------")
                best_val_uar_so_far = current_val_uar
                patience_counter = 0
                # Save best model state and metrics
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_metrics["accuracy"],
                    "val_wa": val_metrics["wa"],
                    "val_uar": val_metrics["uar"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "val_f1_macro": val_metrics["f1_macro"],
                }
                best_metrics = val_metrics
                best_confusion_matrix = val_metrics["confusion_matrix"]
                best_epoch = epoch
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

        # Save best metrics and confusion matrix
        if best_metrics is not None:
            # Create evaluation directories
            held_out_dir = session_dir / "held_out_evaluation"
            held_out_dir.mkdir(exist_ok=True)

            # Save best confusion matrix
            cm_path = held_out_dir / "best_confusion_matrix.png"
            plot_confusion_matrix(
                np.array(best_confusion_matrix), config.class_names, cm_path
            )

            # Save best metrics
            metrics_path = held_out_dir / "best_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"epoch": best_epoch, "metrics": best_metrics}, f, indent=4)

        # Load best model for final evaluation
        print(f"\nLoading best model for Session {test_session}...")
        if best_model_state is None:
            # If no model was saved during training (shouldn't happen), load the last saved one
            checkpoint = torch.load(save_dir / f"best_model_session_{test_session}.pt")
        else:
            checkpoint = best_model_state

        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded model from epoch {checkpoint['epoch']} with validation UAR: {checkpoint['val_uar']:.4f}"
        )

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
            epoch=None,  # Final evaluation
            save_dir=session_dir / "cross_corpus_evaluation",
            eval_type="cross_corpus",
            session_info=test_session,
        )

        # Store results for final summary
        all_results.append(best_metrics["accuracy"])
        all_was.append(best_metrics["wa"])
        all_uars.append(best_metrics["uar"])
        all_f1s_weighted.append(best_metrics["f1_weighted"])
        all_f1s_macro.append(best_metrics["f1_macro"])
        all_class_accuracies.append(best_metrics["class_accuracies"])
        all_saliency_analysis.append(best_metrics["saliency_analysis"])
        cross_corpus_results.append(msp_metrics)

    # Calculate and print averaged LOSO results
    print("\nAveraged LOSO Results:")
    print("=" * 50)

    # Calculate average confusion matrix
    avg_confusion_matrix = np.zeros((4, 4))  # 4x4 for 4 emotion classes
    for session, results in enumerate(all_results, 1):
        session_dir = results_dir / f"session_{session}"
        held_out_dir = session_dir / "held_out_evaluation"
        with open(held_out_dir / "best_metrics.json", "r") as f:
            session_results = json.load(f)
            avg_confusion_matrix += np.array(
                session_results["metrics"]["confusion_matrix"]
            )
    avg_confusion_matrix /= len(all_results)

    # Save averaged confusion matrix
    avg_cm_path = results_dir / "averaged_confusion_matrix.png"
    plot_confusion_matrix(avg_confusion_matrix, config.class_names, avg_cm_path)

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
    for class_name in config.class_names:
        accuracies = [acc[class_name] for acc in all_class_accuracies]
        avg_class_accuracies[class_name] = {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
        }

    print("\nAveraged Class-wise Accuracies:")
    for class_name, stats in avg_class_accuracies.items():
        print(f"{class_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Calculate averaged saliency analysis
    avg_saliency_analysis = {}
    saliency_keys = [
        "mean_saliency",
        "std_saliency",
        "high_saliency_ratio",
        "saliency_entropy",
    ]
    for key in saliency_keys:
        values = [analysis[key] for analysis in all_saliency_analysis]
        avg_saliency_analysis[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    print("\nAveraged Saliency Analysis:")
    for key, stats in avg_saliency_analysis.items():
        print(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

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
            "saliency_analysis": avg_saliency_analysis,
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

    for key, stats in avg_saliency_analysis.items():
        wandb.log({f"averaged_{key}": stats["mean"]})

    # Calculate and print averaged cross-corpus results
    print("\nAveraged Cross-Corpus Results:")
    print("=" * 50)

    cross_corpus_metrics = {
        "accuracy": [r["accuracy"] for r in cross_corpus_results],
        "wa": [r["wa"] for r in cross_corpus_results],
        "uar": [r["uar"] for r in cross_corpus_results],
        "f1_weighted": [r["f1_weighted"] for r in cross_corpus_results],
        "f1_macro": [r["f1_macro"] for r in cross_corpus_results],
    }

    print("\nAveraged Cross-Corpus Metrics:")
    for metric_name, values in cross_corpus_metrics.items():
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

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
                    "saliency_analysis": saliency_analysis,
                }
                for i, (
                    acc,
                    wa,
                    uar,
                    f1_weighted,
                    f1_macro,
                    class_accuracies,
                    saliency_analysis,
                ) in enumerate(
                    zip(
                        all_results,
                        all_was,
                        all_uars,
                        all_f1s_weighted,
                        all_f1s_macro,
                        all_class_accuracies,
                        all_saliency_analysis,
                    )
                )
            },
            "averages": {
                "accuracy": {
                    "mean": float(np.mean(all_results)),
                    "std": float(np.std(all_results)),
                },
                "wa": {"mean": float(np.mean(all_was)), "std": float(np.std(all_was))},
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
        "cross_corpus_results": {
            "session_results": {
                f"session_{i+1}": {
                    "accuracy": float(results["accuracy"]),
                    "wa": float(results["wa"]),
                    "uar": float(results["uar"]),
                    "f1_weighted": float(results["f1_weighted"]),
                    "f1_macro": float(results["f1_macro"]),
                    "class_accuracies": results["class_accuracies"],
                    "saliency_analysis": results["saliency_analysis"],
                }
                for i, results in enumerate(cross_corpus_results)
            },
            "averages": {
                "accuracy": {
                    "mean": float(np.mean(cross_corpus_metrics["accuracy"])),
                    "std": float(np.std(cross_corpus_metrics["accuracy"])),
                },
                "wa": {
                    "mean": float(np.mean(cross_corpus_metrics["wa"])),
                    "std": float(np.std(cross_corpus_metrics["wa"])),
                },
                "uar": {
                    "mean": float(np.mean(cross_corpus_metrics["uar"])),
                    "std": float(np.std(cross_corpus_metrics["uar"])),
                },
                "f1_weighted": {
                    "mean": float(np.mean(cross_corpus_metrics["f1_weighted"])),
                    "std": float(np.std(cross_corpus_metrics["f1_weighted"])),
                },
                "f1_macro": {
                    "mean": float(np.mean(cross_corpus_metrics["f1_macro"])),
                    "std": float(np.std(cross_corpus_metrics["f1_macro"])),
                },
            },
        },
    }

    with open(save_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=4)

    wandb.finish()


if __name__ == "__main__":
    main()

"""
Quick validation script - runs minimal experiments to test setup
without waiting for full LOSO validation.
"""

import torch
from config import ExperimentConfig
from data.dataset import EmotionDataset, collate_fn
from models.AdaptiveEmotionalSalience import AdaptiveEmotionalSaliency, AdaptiveSaliencyLoss
from train import train_epoch
from evaluate import enhanced_evaluate
from torch.utils.data import DataLoader, Subset
import numpy as np
import time


def quick_setup_test():
    """
    Run a minimal test to validate the setup is working.
    Uses only 1 session and a few epochs to quickly verify everything works.
    """
    
    print("🧪 QUICK SETUP VALIDATION")
    print("="*50)
    print("This will test your setup with minimal computation (~10-15 minutes)")
    
    try:
        # Test basic imports and GPU
        print("✅ Testing imports and GPU...")
        config = ExperimentConfig()
        print(f"   Device: {config.device}")
        print(f"   Experiment name: {config.experiment_name}")
        
        # Test dataset loading
        print("✅ Testing dataset loading...")
        dataset = EmotionDataset("IEMOCAP", split="train")
        print(f"   Dataset size: {len(dataset)}")
        
        # Test model creation
        print("✅ Testing model creation...")
        model = AdaptiveEmotionalSaliency(input_dim=768, num_classes=4).to(config.device)
        criterion = AdaptiveSaliencyLoss(class_weights=torch.ones(4).to(config.device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Test data loading with small subset
        print("✅ Testing data loading...")
        small_subset = Subset(dataset, list(range(100)))  # Just 100 samples
        test_loader = DataLoader(
            small_subset, 
            batch_size=8,  # Small batch
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Test one training step
        print("✅ Testing training step...")
        start_time = time.time()
        
        model.train()
        batch = next(iter(test_loader))
        features = batch["features"].to(config.device)
        labels = batch["label"].to(config.device)
        
        optimizer.zero_grad()
        logits, aux_outputs = model(features)
        loss_dict = criterion(logits, labels, aux_outputs)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()
        
        step_time = time.time() - start_time
        print(f"   Training step time: {step_time:.2f} seconds")
        print(f"   Loss: {total_loss.item():.4f}")
        
        # Test evaluation
        print("✅ Testing evaluation...")
        model.eval()
        with torch.no_grad():
            logits, aux_outputs = model(features)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            print(f"   Accuracy on test batch: {acc.item():.4f}")
        
        print("\n🎉 SETUP VALIDATION SUCCESSFUL!")
        print("✅ All components are working correctly")
        print(f"✅ Estimated time per epoch: ~{step_time * len(test_loader):.1f} seconds")
        
        # Estimate timing for full experiments
        samples_per_session = len(dataset) // 5  # Approximate
        batches_per_session = samples_per_session // config.batch_size
        epochs_per_session = 15  # Estimated with early stopping
        
        time_per_session = (step_time * batches_per_session * epochs_per_session) / 60
        time_per_experiment = time_per_session * 5  # 5 LOSO sessions
        
        print(f"\n📊 TIMING ESTIMATES:")
        print(f"   Time per session: ~{time_per_session:.1f} minutes")
        print(f"   Time per experiment: ~{time_per_experiment:.1f} minutes")
        print(f"   Full ablation (37 experiments): ~{time_per_experiment * 37 / 60:.1f} hours")
        
        return True, time_per_experiment
        
    except Exception as e:
        print(f"\n❌ SETUP VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0


def run_single_session_test():
    """
    Run one complete session to get accurate timing.
    This takes ~30-60 minutes but gives real timing data.
    """
    
    print("\n🔬 SINGLE SESSION TEST")
    print("="*40)
    print("Running one complete LOSO session for accurate timing...")
    
    from utils import get_session_splits
    from torch.utils.data import DataLoader
    
    config = ExperimentConfig()
    
    # Load dataset and get one session
    dataset = EmotionDataset("IEMOCAP", split="train")
    session_splits = get_session_splits(dataset)
    
    # Use session 1 as test
    test_session = 1
    test_indices = session_splits[test_session]
    train_indices = []
    for session, indices in session_splits.items():
        if session != test_session:
            train_indices.extend(indices)
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders
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
    model = AdaptiveEmotionalSaliency(input_dim=768, num_classes=4).to(config.device).float()
    criterion = AdaptiveSaliencyLoss(class_weights=torch.ones(4).to(config.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print(f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")
    
    # Train for a few epochs
    start_time = time.time()
    best_uar = 0
    
    for epoch in range(5):  # Just 5 epochs for timing
        print(f"\nEpoch {epoch+1}/5")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.device, epoch)
        
        # Evaluate
        val_metrics = enhanced_evaluate(
            model, test_loader, criterion, config.device, epoch, None, "validation", test_session
        )
        
        print(f"Train UAR: {train_metrics['uar']:.4f}, Val UAR: {val_metrics['uar']:.4f}")
        
        if val_metrics['uar'] > best_uar:
            best_uar = val_metrics['uar']
    
    session_time = time.time() - start_time
    
    print(f"\n📊 SINGLE SESSION RESULTS:")
    print(f"   Time for 5 epochs: {session_time/60:.1f} minutes")
    print(f"   Best validation UAR: {best_uar:.4f}")
    print(f"   Estimated time per full session (~15 epochs): {session_time * 3 / 60:.1f} minutes")
    
    return session_time * 3  # Estimate for full session with early stopping


def main():
    """Main validation function with progressive testing."""
    
    print("🎯 SETUP VALIDATION OPTIONS")
    print("="*50)
    print("Choose validation level:")
    print("1. 🚀 Quick setup test (~5 minutes)")
    print("2. 🔬 Single session test (~30-60 minutes)")
    print("3. 🧪 Two full experiments (~2-3 hours)")
    print("4. ❌ Skip to full ablation")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        success, est_time = quick_setup_test()
        if success:
            print(f"\n✅ Setup validated! Each experiment should take ~{est_time:.1f} minutes")
            proceed = input("\nProceed to full ablation? (y/N): ").strip().lower()
            if proceed in ['y', 'yes']:
                print("🚀 Starting full ablation...")
                from run_all_ablations import main
                main()
        else:
            print("❌ Please fix setup issues before proceeding")
    
    elif choice == "2":
        session_time = run_single_session_test()
        full_experiment_time = session_time * 5 / 60  # 5 sessions per experiment
        print(f"\n✅ Single session validated!")
        print(f"📊 Estimated time per full experiment: {full_experiment_time:.1f} minutes")
        print(f"📊 Estimated time for full ablation: {full_experiment_time * 37 / 60:.1f} hours")
        
        proceed = input("\nProceed to full ablation? (y/N): ").strip().lower()
        if proceed in ['y', 'yes']:
            print("🚀 Starting full ablation...")
            from run_all_ablations import main
            main()
    
    elif choice == "3":
        print("🧪 Running two full experiments for accurate timing...")
        from start_ablation import run_test_experiments
        run_test_experiments()
    
    elif choice == "4":
        print("🚀 Starting full ablation directly...")
        from run_all_ablations import main
        main()
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
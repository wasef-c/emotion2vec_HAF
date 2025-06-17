"""
Estimate time and computational requirements for ablation studies.
"""

def estimate_experiment_time():
    """Estimate time per experiment based on your current setup."""
    
    print("🕐 ABLATION STUDY TIME ESTIMATION")
    print("="*50)
    
    # Based on typical emotion recognition training
    time_per_session = 45  # minutes (5 LOSO sessions, ~9 min each)
    
    studies = [
        ("Quick Ablation", 5),
        ("Architecture Ablation", 12),  # 2×2×3 combinations
        ("Curriculum Epochs", 5),
        ("Class Weights", 6),
        ("Hyperparameters", 9),
    ]
    
    total_experiments = sum(count for _, count in studies)
    total_time_hours = (total_experiments * time_per_session) / 60
    
    print(f"📊 EXPERIMENT BREAKDOWN:")
    print(f"{'Study':<25} {'Experiments':<12} {'Est. Hours':<10}")
    print("-" * 50)
    
    for study_name, exp_count in studies:
        hours = (exp_count * time_per_session) / 60
        print(f"{study_name:<25} {exp_count:<12} {hours:.1f}")
    
    print("-" * 50)
    print(f"{'TOTAL':<25} {total_experiments:<12} {total_time_hours:.1f}")
    
    print(f"\n⚡ COMPUTATIONAL REQUIREMENTS:")
    print(f"• GPU Memory: ~8-12 GB (depending on batch size)")
    print(f"• Storage: ~10-20 GB for all results")
    print(f"• CPU: Multi-core recommended for data loading")
    
    print(f"\n💡 OPTIMIZATION TIPS:")
    print(f"• Run overnight or over weekend")
    print(f"• Use smaller batch sizes if GPU memory limited")
    print(f"• Monitor first few experiments to validate timing")
    print(f"• Consider running studies separately if time-constrained")
    
    return total_time_hours, total_experiments


def estimate_subset_time():
    """Estimate time for running just priority studies."""
    
    print(f"\n🎯 PRIORITY SUBSET ESTIMATION:")
    print("="*50)
    
    time_per_session = 45  # minutes
    
    priority_studies = [
        ("Quick Ablation", 5),
        ("Best Component Deep Dive", 8),  # Focus on promising components
    ]
    
    total_experiments = sum(count for _, count in priority_studies)
    total_time_hours = (total_experiments * time_per_session) / 60
    
    print(f"📊 PRIORITY BREAKDOWN:")
    print(f"{'Study':<25} {'Experiments':<12} {'Est. Hours':<10}")
    print("-" * 50)
    
    for study_name, exp_count in priority_studies:
        hours = (exp_count * time_per_session) / 60
        print(f"{study_name:<25} {exp_count:<12} {hours:.1f}")
    
    print("-" * 50)
    print(f"{'TOTAL':<25} {total_experiments:<12} {total_time_hours:.1f}")
    
    return total_time_hours, total_experiments


def get_user_choice():
    """Get user's preference for ablation scope."""
    
    print(f"\n🤔 CHOOSE YOUR ABLATION SCOPE:")
    print("1. 🚀 Full comprehensive ablation (~16-20 hours)")
    print("2. 🎯 Priority ablation only (~5-7 hours)")
    print("3. 🧪 Quick test run (3-5 experiments, ~2-4 hours)")
    print("4. ❌ Cancel")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("✅ Running full comprehensive ablation")
        return "full"
    elif choice == "2":
        print("✅ Running priority ablation")
        return "priority"
    elif choice == "3":
        print("✅ Running quick test")
        return "quick"
    else:
        print("❌ Cancelled")
        return "cancel"


def show_recommendations():
    """Show recommendations based on computational resources."""
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("="*50)
    
    print("🖥️  HARDWARE RECOMMENDATIONS:")
    print("• GPU: RTX 3080/4080 or better (8-12GB VRAM)")
    print("• RAM: 16GB+ system memory")
    print("• Storage: SSD with 20GB+ free space")
    
    print(f"\n⚙️  SOFTWARE OPTIMIZATION:")
    print("• Use mixed precision training (if not already)")
    print("• Set num_workers=0 to avoid multiprocessing issues")
    print("• Monitor GPU utilization with nvidia-smi")
    
    print(f"\n📋 EXECUTION STRATEGY:")
    print("• Start with quick ablation (5 experiments)")
    print("• Analyze results to guide further experiments")
    print("• Use monitoring script to track progress")
    print("• Save intermediate results frequently")
    
    print(f"\n⚠️  TROUBLESHOOTING:")
    print("• If OOM errors: reduce batch_size in config")
    print("• If slow training: check data loading bottlenecks")
    print("• If experiments fail: check GPU memory and logs")


def main():
    full_hours, full_experiments = estimate_experiment_time()
    priority_hours, priority_experiments = estimate_subset_time()
    
    show_recommendations()
    
    choice = get_user_choice()
    
    if choice == "full":
        print(f"\n🚀 STARTING FULL ABLATION STUDY")
        print(f"📊 {full_experiments} experiments, ~{full_hours:.1f} hours")
        print("Run: python run_all_ablations.py")
        
    elif choice == "priority":
        print(f"\n🎯 STARTING PRIORITY ABLATION")
        print(f"📊 {priority_experiments} experiments, ~{priority_hours:.1f} hours")
        print("Run: python quick_ablation.py")
        
    elif choice == "quick":
        print(f"\n🧪 STARTING QUICK TEST")
        print("📊 3-5 experiments, ~2-4 hours")
        print("Run: python quick_ablation.py (option 1)")
        
    print(f"\n📈 MONITORING:")
    print("Use: python monitor_ablation.py --monitor")


if __name__ == "__main__":
    main()
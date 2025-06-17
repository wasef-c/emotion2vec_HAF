"""
Main entry point for ablation studies with error handling and restart capability.
"""

import sys
import traceback
from pathlib import Path
from run_all_ablations import ComprehensiveAblationRunner
from estimate_ablation_time import estimate_experiment_time, show_recommendations


def check_environment():
    """Check if the environment is ready for ablation studies."""
    
    print("🔍 CHECKING ENVIRONMENT...")
    
    issues = []
    
    # Check if GPU is available
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("❌ CUDA not available - training will be slow on CPU")
        else:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} ({gpu_count} device(s))")
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    # Check wandb
    try:
        import wandb
        print("✅ Wandb available for logging")
    except ImportError:
        issues.append("❌ Wandb not installed - no experiment tracking")
    
    # Check required modules
    required_modules = [
        'data.dataset', 'models.AdaptiveEmotionalSalience', 
        'config', 'train', 'evaluate', 'utils', 'metrics'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} module available")
        except ImportError as e:
            issues.append(f"❌ Module {module} not available: {str(e)}")
    
    # Check disk space (approximate)
    try:
        current_dir = Path(".")
        stat = current_dir.stat()
        print("✅ Current directory accessible")
    except Exception as e:
        issues.append(f"❌ Directory access issue: {str(e)}")
    
    if issues:
        print(f"\n⚠️  ENVIRONMENT ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        
        proceed = input("\nDo you want to proceed anyway? (y/N): ").strip().lower()
        if proceed not in ['y', 'yes']:
            return False
    
    print("✅ Environment check passed!")
    return True


def show_ablation_menu():
    """Show menu for different ablation options."""
    
    print("\n" + "="*80)
    print("🎯 EMOTION RECOGNITION ABLATION STUDY")
    print("="*80)
    
    print("\nChoose your ablation study:")
    print("1. 🚀 Full Comprehensive Ablation (~16-20 hours)")
    print("   • All architectural components")
    print("   • Curriculum epochs variation")
    print("   • Class weighting strategies")
    print("   • Hyperparameter optimization")
    
    print("\n2. 🎯 Quick Core Ablation (~3-5 hours)")
    print("   • 5 key architectural combinations")
    print("   • Essential component analysis")
    
    print("\n3. 📊 Component-Specific Ablation")
    print("   • Focus on specific components")
    print("   • Curriculum, weights, or architecture")
    
    print("\n4. 🧪 Test Run (2-3 experiments)")
    print("   • Validate setup and timing")
    
    print("\n5. ❓ Get time estimates and recommendations")
    print("\n6. ❌ Exit")
    
    return input("\nEnter your choice (1-6): ").strip()


def run_full_ablation():
    """Run the full comprehensive ablation study."""
    
    print("\n🚀 STARTING FULL COMPREHENSIVE ABLATION")
    print("="*60)
    
    try:
        runner = ComprehensiveAblationRunner()
        results = runner.run_all_ablations()
        
        print("\n🎉 FULL ABLATION COMPLETED SUCCESSFULLY!")
        return results
        
    except Exception as e:
        print(f"\n💥 ABLATION FAILED: {str(e)}")
        print("\nFull error traceback:")
        traceback.print_exc()
        
        print(f"\n💡 You can restart from where it left off using the monitor script")
        return None


def run_quick_ablation():
    """Run the quick ablation study."""
    
    print("\n🎯 STARTING QUICK ABLATION")
    print("="*40)
    
    try:
        from quick_ablation import run_quick_ablation
        results = run_quick_ablation()
        
        print("\n🎉 QUICK ABLATION COMPLETED!")
        return results
        
    except Exception as e:
        print(f"\n💥 QUICK ABLATION FAILED: {str(e)}")
        traceback.print_exc()
        return None


def run_component_specific():
    """Run component-specific ablation."""
    
    print("\n📊 COMPONENT-SPECIFIC ABLATION")
    print("="*40)
    
    print("Choose component to study:")
    print("1. Curriculum Learning Epochs")
    print("2. Class Weighting Strategies") 
    print("3. Architecture Components")
    
    choice = input("Enter choice (1-3): ").strip()
    
    try:
        if choice == "1":
            from quick_ablation import run_curriculum_epochs_ablation
            results = run_curriculum_epochs_ablation()
        elif choice == "2":
            from quick_ablation import run_class_weights_ablation
            results = run_class_weights_ablation()
        elif choice == "3":
            print("Running core architecture ablation...")
            from quick_ablation import run_quick_ablation
            results = run_quick_ablation()
        else:
            print("Invalid choice")
            return None
        
        print(f"\n🎉 COMPONENT ABLATION COMPLETED!")
        return results
        
    except Exception as e:
        print(f"\n💥 COMPONENT ABLATION FAILED: {str(e)}")
        traceback.print_exc()
        return None


def run_test_experiments():
    """Run a few test experiments to validate setup."""
    
    print("\n🧪 RUNNING TEST EXPERIMENTS")
    print("="*40)
    
    try:
        from ablation_study import AblationStudy
        
        study = AblationStudy(results_dir="test_ablation")
        
        # Run just 2-3 quick experiments
        test_configs = [
            {
                'use_curriculum_learning': False,
                'use_speaker_disentanglement': False,
                'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                'description': 'Test 1: Baseline'
            },
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': True,
                'curriculum_epochs': 15,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Test 2: Full system'
            },
        ]
        
        for i, config in enumerate(test_configs, 1):
            print(f"\nRunning test experiment {i}/{len(test_configs)}: {config['description']}")
            study.run_single_experiment(config)
        
        study.analyze_results()
        print("\n🎉 TEST EXPERIMENTS COMPLETED!")
        print("✅ Setup is working correctly - you can now run full ablations")
        
        return study
        
    except Exception as e:
        print(f"\n💥 TEST FAILED: {str(e)}")
        print("❌ Please fix the issues before running full ablation")
        traceback.print_exc()
        return None


def main():
    """Main entry point for ablation studies."""
    
    print("Welcome to the Emotion Recognition Ablation Study!")
    
    # Check environment
    if not check_environment():
        print("Please fix environment issues before proceeding.")
        return
    
    while True:
        choice = show_ablation_menu()
        
        if choice == "1":
            results = run_full_ablation()
            break
            
        elif choice == "2":
            results = run_quick_ablation()
            break
            
        elif choice == "3":
            results = run_component_specific()
            break
            
        elif choice == "4":
            results = run_test_experiments()
            # Don't break here - let user choose another option
            
        elif choice == "5":
            estimate_experiment_time()
            show_recommendations()
            # Don't break here - let user choose another option
            
        elif choice == "6":
            print("👋 Goodbye!")
            return
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("📋 NEXT STEPS:")
    print("• Monitor progress: python monitor_ablation.py --monitor")
    print("• View detailed results: python monitor_ablation.py --detailed")
    print("• Results are saved in timestamped directories")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
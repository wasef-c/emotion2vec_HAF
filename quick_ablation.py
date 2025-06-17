"""
Quick ablation study - runs only the most important comparisons.
Good for testing the ablation setup and getting initial results.
"""

from ablation_study import AblationStudy
from config import ExperimentConfig


def run_quick_ablation():
    """Run a minimal but informative ablation study."""
    
    study = AblationStudy(results_dir="quick_ablation_results")
    
    # Define minimal but comprehensive ablation
    quick_configs = [
        # 1. Baseline: No special techniques
        {
            'use_curriculum_learning': False,
            'use_speaker_disentanglement': False,
            'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
            'description': 'Baseline: No special techniques'
        },
        
        # 2. Only class weighting (current default weights)
        {
            'use_curriculum_learning': False,
            'use_speaker_disentanglement': False,
            'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
            'description': 'Class weighting only (happy=3.0)'
        },
        
        # 3. Curriculum learning only
        {
            'use_curriculum_learning': True,
            'use_speaker_disentanglement': False,
            'curriculum_epochs': 15,
            'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
            'description': 'Curriculum learning only (15 epochs)'
        },
        
        # 4. Speaker disentanglement only
        {
            'use_curriculum_learning': False,
            'use_speaker_disentanglement': True,
            'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
            'description': 'Speaker disentanglement only'
        },
        
        # 5. Full system (current approach)
        {
            'use_curriculum_learning': True,
            'use_speaker_disentanglement': True,
            'curriculum_epochs': 15,
            'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
            'description': 'Full system (all techniques)'
        },
    ]
    
    print(f"Starting quick ablation study with {len(quick_configs)} experiments...")
    print("This will help identify which components contribute most to performance.\n")
    
    for i, config in enumerate(quick_configs):
        print(f"\n{'='*70}")
        print(f"Running experiment {i+1}/{len(quick_configs)}")
        print(f"Description: {config['description']}")
        print(f"{'='*70}")
        
        try:
            result = study.run_single_experiment(config)
            if result['status'] == 'completed':
                metrics = result['results']
                print(f"✅ COMPLETED - UAR: {metrics['uar']:.3f}, Acc: {metrics['accuracy']:.3f}")
            else:
                print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ FAILED - {str(e)}")
    
    # Analyze results
    print(f"\n{'='*70}")
    print("QUICK ABLATION RESULTS")
    print(f"{'='*70}")
    study.analyze_results()
    
    return study


def run_curriculum_epochs_ablation():
    """Run ablation specifically for curriculum learning epochs."""
    
    study = AblationStudy(results_dir="curriculum_epochs_ablation")
    
    # Test different curriculum epoch values
    curriculum_configs = []
    for epochs in [5, 10, 15, 20, 30]:
        curriculum_configs.append({
            'use_curriculum_learning': True,
            'use_speaker_disentanglement': True,
            'curriculum_epochs': epochs,
            'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
            'description': f'Full system with {epochs} curriculum epochs'
        })
    
    print(f"Starting curriculum epochs ablation with {len(curriculum_configs)} experiments...")
    
    for i, config in enumerate(curriculum_configs):
        print(f"\nRunning experiment {i+1}/{len(curriculum_configs)}")
        study.run_single_experiment(config)
    
    study.analyze_results()
    return study


def run_class_weights_ablation():
    """Run ablation specifically for class weighting strategies."""
    
    study = AblationStudy(results_dir="class_weights_ablation")
    
    # Test different class weighting strategies
    weight_strategies = [
        # Equal weights (baseline)
        {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
        # Boost happy (current default)
        {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
        # Boost happy and sad (minority classes)
        {"neutral": 1.0, "happy": 2.5, "sad": 2.0, "anger": 1.0},
        # Conservative boost
        {"neutral": 1.0, "happy": 2.0, "sad": 1.5, "anger": 1.0},
        # Boost all non-neutral
        {"neutral": 1.0, "happy": 2.0, "sad": 2.0, "anger": 2.0},
    ]
    
    weight_configs = []
    for i, weights in enumerate(weight_strategies):
        weight_str = "_".join([f"{k[0].upper()}{v}" for k, v in weights.items() if v != 1.0])
        if not weight_str:
            weight_str = "equal"
        
        weight_configs.append({
            'use_curriculum_learning': True,
            'use_speaker_disentanglement': True,
            'curriculum_epochs': 15,
            'class_weights': weights,
            'description': f'Class weights: {weight_str}'
        })
    
    print(f"Starting class weights ablation with {len(weight_configs)} experiments...")
    
    for i, config in enumerate(weight_configs):
        print(f"\nRunning experiment {i+1}/{len(weight_configs)}")
        study.run_single_experiment(config)
    
    study.analyze_results()
    return study


if __name__ == "__main__":
    print("Quick Ablation Study Options:")
    print("1. Quick ablation (5 key experiments)")
    print("2. Curriculum epochs ablation")
    print("3. Class weights ablation")
    print("4. All quick ablations")
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == "1":
        run_quick_ablation()
    elif choice == "2":
        run_curriculum_epochs_ablation()
    elif choice == "3":
        run_class_weights_ablation()
    elif choice == "4":
        print("Running all quick ablations...")
        run_quick_ablation()
        run_curriculum_epochs_ablation() 
        run_class_weights_ablation()
    else:
        print("Invalid choice. Running quick ablation...")
        run_quick_ablation()
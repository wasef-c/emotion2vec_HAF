"""
Comprehensive ablation study for emotion recognition with adaptive saliency.
This script runs a grid search over key hyperparameters to understand their impact.
"""

import itertools
import json
from pathlib import Path
from config import ExperimentConfig
from main import main as run_experiment
import wandb


class AblationStudy:
    """Class to manage ablation study experiments."""
    
    def __init__(self, results_dir="ablation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
        
    def define_ablation_grid(self):
        """Define the parameter grid for ablation study."""
        
        # Core architectural ablations
        architectural_params = {
            # Core system components (ablation study)
            'use_curriculum_learning': [True, False],
            'use_speaker_disentanglement': [True, False],
            
            # Curriculum learning variations
            'curriculum_epochs': [10, 15, 20, 30],  # When curriculum is enabled
            
            # Class weighting strategies
            'class_weight_strategies': [
                # Baseline: equal weights
                {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                # Current default: boost happy
                {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                # Boost minorities (happy + sad)
                {"neutral": 1.0, "happy": 2.5, "sad": 2.0, "anger": 1.0},
                # Conservative boost
                {"neutral": 1.0, "happy": 2.0, "sad": 1.5, "anger": 1.0},
            ]
        }
        
        # Training hyperparameter grid (smaller search)
        training_params = {
            'learning_rate': [5e-5, 1e-4, 2e-4],
            'batch_size': [32, 38, 48],  # Current is 38
            'weight_decay': [1e-5, 1e-4, 1e-3],
        }
        
        return architectural_params, training_params
    
    def get_priority_experiments(self):
        """Define high-priority experiments for initial ablation."""
        
        priority_configs = [
            # 1. Baseline: No curriculum, no speaker disentanglement
            {
                'use_curriculum_learning': False,
                'use_speaker_disentanglement': False,
                'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                'description': 'Baseline: No special techniques'
            },
            
            # 2. Only curriculum learning
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': False,
                'curriculum_epochs': 15,
                'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                'description': 'Curriculum learning only'
            },
            
            # 3. Only speaker disentanglement
            {
                'use_curriculum_learning': False,
                'use_speaker_disentanglement': True,
                'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                'description': 'Speaker disentanglement only'
            },
            
            # 4. Only class weighting
            {
                'use_curriculum_learning': False,
                'use_speaker_disentanglement': False,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Class weighting only'
            },
            
            # 5. Curriculum + Speaker Disentanglement
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': True,
                'curriculum_epochs': 15,
                'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                'description': 'Curriculum + Speaker Disentanglement'
            },
            
            # 6. Curriculum + Class Weighting
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': False,
                'curriculum_epochs': 15,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Curriculum + Class Weighting'
            },
            
            # 7. Speaker Disentanglement + Class Weighting
            {
                'use_curriculum_learning': False,
                'use_speaker_disentanglement': True,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Speaker Disentanglement + Class Weighting'
            },
            
            # 8. Full system (current default)
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': True,
                'curriculum_epochs': 15,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Full system with all techniques'
            },
            
            # 9. Curriculum epoch variations (when curriculum is enabled)
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': True,
                'curriculum_epochs': 10,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Full system, short curriculum (10 epochs)'
            },
            
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': True,
                'curriculum_epochs': 30,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Full system, long curriculum (30 epochs)'
            },
        ]
        
        return priority_configs
    
    def get_hyperparameter_grid(self):
        """Get smaller hyperparameter grid for best architectural configuration."""
        
        # Best config from architectural ablation (you'll need to update this)
        best_architectural_config = {
            'use_curriculum_learning': True,
            'use_speaker_disentanglement': True,
            'curriculum_epochs': 15,
            'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
        }
        
        # Hyperparameter variations
        hyperparam_configs = []
        
        learning_rates = [5e-5, 1e-4, 2e-4]
        batch_sizes = [32, 38, 48]
        weight_decays = [1e-5, 1e-4, 1e-3]
        
        for lr, bs, wd in itertools.product(learning_rates, batch_sizes, weight_decays):
            config = best_architectural_config.copy()
            config.update({
                'learning_rate': lr,
                'batch_size': bs,
                'weight_decay': wd,
                'description': f'Hyperparameter: LR={lr:.0e}, BS={bs}, WD={wd:.0e}'
            })
            hyperparam_configs.append(config)
        
        return hyperparam_configs
    
    def run_single_experiment(self, config_params):
        """Run a single experiment with given configuration."""
        
        # Create configuration
        config = ExperimentConfig()
        
        # Apply parameters
        for key, value in config_params.items():
            if key == 'class_weights':
                config.class_weights = value
            elif key == 'description':
                continue  # Skip description
            else:
                setattr(config, key, value)
        
        # Regenerate experiment name
        config.regenerate_experiment_name()
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {config_params.get('description', 'No description')}")
        print(f"Experiment name: {config.experiment_name}")
        print(f"Configuration: {config_params}")
        print(f"{'='*60}")
        
        try:
            # Run the experiment (you'll need to adapt this to your main function)
            # This assumes your main function can accept a config parameter
            results = self.run_experiment_with_config(config)
            
            # Store results
            experiment_result = {
                'config': config_params,
                'experiment_name': config.experiment_name,
                'results': results,
                'status': 'completed'
            }
            
        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            experiment_result = {
                'config': config_params,
                'experiment_name': config.experiment_name,
                'error': str(e),
                'status': 'failed'
            }
        
        self.results.append(experiment_result)
        self.save_results()
        
        return experiment_result
    
    def run_experiment_with_config(self, config):
        """Adapter to run your main experiment with a given config."""
        from run_ablation import run_experiment_with_config
        
        # Run the experiment and get averaged results
        full_results = run_experiment_with_config(config)
        
        # Extract key metrics for ablation analysis
        iemocap_results = full_results["iemocap_results"]
        cross_corpus_results = full_results["cross_corpus_results"]
        
        return {
            'accuracy': iemocap_results['accuracy']['mean'],
            'uar': iemocap_results['uar']['mean'],
            'wa': iemocap_results['wa']['mean'],
            'f1_weighted': iemocap_results['f1_weighted']['mean'],
            'f1_macro': iemocap_results['f1_macro']['mean'],
            'cross_corpus_uar': cross_corpus_results['uar']['mean'],
            'cross_corpus_accuracy': cross_corpus_results['accuracy']['mean'],
            'full_results': full_results  # Store complete results
        }
    
    def save_results(self):
        """Save current results to file."""
        results_file = self.results_dir / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def analyze_results(self):
        """Analyze and summarize ablation results."""
        if not self.results:
            print("No results to analyze yet.")
            return
        
        print(f"\n{'='*80}")
        print("ABLATION STUDY RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Sort by UAR (primary metric)
        completed_results = [r for r in self.results if r['status'] == 'completed']
        completed_results.sort(key=lambda x: x['results']['uar'], reverse=True)
        
        print(f"\nTop 5 configurations by UAR:")
        print(f"{'Rank':<5} {'UAR':<6} {'Acc':<6} {'F1-M':<6} {'Description':<50}")
        print("-" * 80)
        
        for i, result in enumerate(completed_results[:5]):
            metrics = result['results']
            desc = result['config'].get('description', 'No description')[:47]
            print(f"{i+1:<5} {metrics['uar']:<6.3f} {metrics['accuracy']:<6.3f} "
                  f"{metrics['f1_macro']:<6.3f} {desc:<50}")
        
        # Component analysis
        self.analyze_component_effects(completed_results)
    
    def analyze_component_effects(self, results):
        """Analyze the effect of individual components."""
        print(f"\n{'='*80}")
        print("COMPONENT EFFECT ANALYSIS")
        print(f"{'='*80}")
        
        # Group by components
        curriculum_effect = self.compare_component(results, 'use_curriculum_learning')
        speaker_effect = self.compare_component(results, 'use_speaker_disentanglement')
        
        print(f"\nCurriculum Learning Effect:")
        print(f"  With curriculum:    UAR = {curriculum_effect['with']:.3f}")
        print(f"  Without curriculum: UAR = {curriculum_effect['without']:.3f}")
        print(f"  Improvement: {curriculum_effect['with'] - curriculum_effect['without']:.3f}")
        
        print(f"\nSpeaker Disentanglement Effect:")
        print(f"  With speaker disent.: UAR = {speaker_effect['with']:.3f}")
        print(f"  Without speaker disent.: UAR = {speaker_effect['without']:.3f}")
        print(f"  Improvement: {speaker_effect['with'] - speaker_effect['without']:.3f}")
    
    def compare_component(self, results, component_key):
        """Compare results with and without a specific component."""
        with_component = [r for r in results if r['config'].get(component_key, False)]
        without_component = [r for r in results if not r['config'].get(component_key, False)]
        
        avg_with = sum(r['results']['uar'] for r in with_component) / len(with_component) if with_component else 0
        avg_without = sum(r['results']['uar'] for r in without_component) / len(without_component) if without_component else 0
        
        return {
            'with': avg_with,
            'without': avg_without
        }


def run_priority_ablation():
    """Run the priority ablation study."""
    
    study = AblationStudy()
    priority_configs = study.get_priority_experiments()
    
    print(f"Starting ablation study with {len(priority_configs)} priority experiments...")
    
    for i, config in enumerate(priority_configs):
        print(f"\nProgress: {i+1}/{len(priority_configs)}")
        study.run_single_experiment(config)
    
    # Analyze results
    study.analyze_results()
    
    print(f"\nAblation study completed! Results saved to: {study.results_dir}")


def run_hyperparameter_search():
    """Run hyperparameter search on best architectural configuration."""
    
    study = AblationStudy(results_dir="hyperparameter_search")
    hyperparam_configs = study.get_hyperparameter_grid()
    
    print(f"Starting hyperparameter search with {len(hyperparam_configs)} configurations...")
    
    for i, config in enumerate(hyperparam_configs):
        print(f"\nProgress: {i+1}/{len(hyperparam_configs)}")
        study.run_single_experiment(config)
    
    # Analyze results
    study.analyze_results()
    
    print(f"\nHyperparameter search completed! Results saved to: {study.results_dir}")


if __name__ == "__main__":
    print("Emotion Recognition Ablation Study")
    print("1. Priority ablation (architectural components)")
    print("2. Hyperparameter search")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        run_priority_ablation()
    elif choice == "2":
        run_hyperparameter_search()
    else:
        print("Invalid choice. Running priority ablation...")
        run_priority_ablation()
"""
Comprehensive ablation study runner.
Runs all ablations in optimal order and tracks progress.
"""

import time
import json
from pathlib import Path
from datetime import datetime
from ablation_study import AblationStudy
from quick_ablation import run_quick_ablation, run_curriculum_epochs_ablation, run_class_weights_ablation


class ComprehensiveAblationRunner:
    """Manages running all ablation studies in sequence."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results_summary = {
            'start_time': self.start_time.isoformat(),
            'studies_completed': [],
            'studies_failed': [],
            'best_configurations': {},
            'key_insights': []
        }
        
        # Create main results directory
        self.main_results_dir = Path(f"comprehensive_ablation_{self.start_time.strftime('%Y%m%d_%H%M%S')}")
        self.main_results_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Starting Comprehensive Ablation Study")
        print(f"📁 Results will be saved to: {self.main_results_dir}")
        print(f"⏰ Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def run_all_ablations(self):
        """Run all ablation studies in optimal order."""
        
        studies = [
            ("Quick Ablation", self.run_quick_ablation_study),
            ("Architecture Ablation", self.run_architecture_ablation_study),
            ("Curriculum Epochs Ablation", self.run_curriculum_epochs_study),
            ("Class Weights Ablation", self.run_class_weights_study),
            ("Hyperparameter Search", self.run_hyperparameter_study),
        ]
        
        total_studies = len(studies)
        
        for i, (study_name, study_func) in enumerate(studies, 1):
            print(f"\n{'='*80}")
            print(f"📊 STUDY {i}/{total_studies}: {study_name}")
            print(f"{'='*80}")
            
            try:
                study_start = time.time()
                result = study_func()
                study_duration = time.time() - study_start
                
                self.results_summary['studies_completed'].append({
                    'name': study_name,
                    'duration_minutes': study_duration / 60,
                    'best_config': self.extract_best_config(result),
                    'results_dir': str(result.results_dir)
                })
                
                print(f"✅ {study_name} completed in {study_duration/60:.1f} minutes")
                
            except Exception as e:
                print(f"❌ {study_name} failed: {str(e)}")
                self.results_summary['studies_failed'].append({
                    'name': study_name,
                    'error': str(e)
                })
                
                # Continue with other studies even if one fails
                continue
            
            # Save progress after each study
            self.save_progress()
        
        # Final analysis
        self.perform_final_analysis()
        
        print(f"\n{'='*80}")
        print("🎉 COMPREHENSIVE ABLATION STUDY COMPLETED!")
        print(f"{'='*80}")
        
        total_duration = (datetime.now() - self.start_time).total_seconds() / 60
        print(f"⏱️  Total duration: {total_duration:.1f} minutes")
        print(f"✅ Studies completed: {len(self.results_summary['studies_completed'])}")
        print(f"❌ Studies failed: {len(self.results_summary['studies_failed'])}")
        print(f"📁 Results saved to: {self.main_results_dir}")
        
        return self.results_summary
    
    def run_quick_ablation_study(self):
        """Run the quick ablation study."""
        print("Running quick ablation to identify key components...")
        
        study = AblationStudy(results_dir=self.main_results_dir / "01_quick_ablation")
        
        quick_configs = [
            {
                'use_curriculum_learning': False,
                'use_speaker_disentanglement': False,
                'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                'description': 'Baseline: No special techniques'
            },
            {
                'use_curriculum_learning': False,
                'use_speaker_disentanglement': False,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Class weighting only (happy=3.0)'
            },
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': False,
                'curriculum_epochs': 15,
                'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                'description': 'Curriculum learning only'
            },
            {
                'use_curriculum_learning': False,
                'use_speaker_disentanglement': True,
                'class_weights': {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},
                'description': 'Speaker disentanglement only'
            },
            {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': True,
                'curriculum_epochs': 15,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': 'Full system'
            },
        ]
        
        for i, config in enumerate(quick_configs):
            print(f"\n  Experiment {i+1}/{len(quick_configs)}: {config['description']}")
            study.run_single_experiment(config)
        
        study.analyze_results()
        return study
    
    def run_architecture_ablation_study(self):
        """Run comprehensive architectural ablation."""
        print("Running comprehensive architectural ablation...")
        
        study = AblationStudy(results_dir=self.main_results_dir / "02_architecture_ablation")
        
        # All combinations of architectural components
        curriculum_options = [False, True]
        speaker_options = [False, True]
        class_weight_options = [
            {"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0},  # Equal
            {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},  # Boost happy
            {"neutral": 1.0, "happy": 2.5, "sad": 2.0, "anger": 1.0},  # Boost minorities
        ]
        
        configs = []
        for curr in curriculum_options:
            for speaker in speaker_options:
                for weights in class_weight_options:
                    weight_desc = "equal" if all(w == 1.0 for w in weights.values()) else f"H{weights['happy']}_S{weights['sad']}"
                    
                    config = {
                        'use_curriculum_learning': curr,
                        'use_speaker_disentanglement': speaker,
                        'curriculum_epochs': 15 if curr else 20,  # Default when not using curriculum
                        'class_weights': weights,
                        'description': f"Curr:{curr}, Speaker:{speaker}, Weights:{weight_desc}"
                    }
                    configs.append(config)
        
        for i, config in enumerate(configs):
            print(f"\n  Experiment {i+1}/{len(configs)}: {config['description']}")
            study.run_single_experiment(config)
        
        study.analyze_results()
        return study
    
    def run_curriculum_epochs_study(self):
        """Run curriculum epochs ablation."""
        print("Running curriculum epochs ablation...")
        
        study = AblationStudy(results_dir=self.main_results_dir / "03_curriculum_epochs")
        
        epoch_values = [5, 10, 15, 20, 30]
        
        for epochs in epoch_values:
            config = {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': True,
                'curriculum_epochs': epochs,
                'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
                'description': f'Curriculum epochs: {epochs}'
            }
            print(f"\n  Testing {epochs} curriculum epochs...")
            study.run_single_experiment(config)
        
        study.analyze_results()
        return study
    
    def run_class_weights_study(self):
        """Run class weights ablation."""
        print("Running class weights ablation...")
        
        study = AblationStudy(results_dir=self.main_results_dir / "04_class_weights")
        
        weight_strategies = [
            ({"neutral": 1.0, "happy": 1.0, "sad": 1.0, "anger": 1.0}, "Equal weights"),
            ({"neutral": 1.0, "happy": 2.0, "sad": 1.0, "anger": 1.0}, "Moderate happy boost"),
            ({"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0}, "Strong happy boost"),
            ({"neutral": 1.0, "happy": 2.5, "sad": 2.0, "anger": 1.0}, "Boost minorities"),
            ({"neutral": 1.0, "happy": 2.0, "sad": 1.5, "anger": 1.5}, "Boost all emotions"),
            ({"neutral": 0.8, "happy": 3.0, "sad": 1.2, "anger": 1.2}, "Suppress neutral"),
        ]
        
        for weights, description in weight_strategies:
            config = {
                'use_curriculum_learning': True,
                'use_speaker_disentanglement': True,
                'curriculum_epochs': 15,
                'class_weights': weights,
                'description': description
            }
            print(f"\n  Testing: {description}")
            study.run_single_experiment(config)
        
        study.analyze_results()
        return study
    
    def run_hyperparameter_study(self):
        """Run hyperparameter search on best configuration."""
        print("Running hyperparameter search...")
        
        study = AblationStudy(results_dir=self.main_results_dir / "05_hyperparameters")
        
        # Use best configuration from previous studies (you may want to update this)
        base_config = {
            'use_curriculum_learning': True,
            'use_speaker_disentanglement': True,
            'curriculum_epochs': 15,
            'class_weights': {"neutral": 1.0, "happy": 3.0, "sad": 1.0, "anger": 1.0},
        }
        
        # Hyperparameter grid
        learning_rates = [5e-5, 1e-4, 2e-4]
        batch_sizes = [32, 38, 48]
        weight_decays = [1e-5, 1e-4, 1e-3]
        
        # Test a subset of combinations (full grid would be 27 experiments)
        hyperparam_configs = [
            # Default
            {'learning_rate': 1e-4, 'batch_size': 38, 'weight_decay': 1e-4},
            # Learning rate variations
            {'learning_rate': 5e-5, 'batch_size': 38, 'weight_decay': 1e-4},
            {'learning_rate': 2e-4, 'batch_size': 38, 'weight_decay': 1e-4},
            # Batch size variations
            {'learning_rate': 1e-4, 'batch_size': 32, 'weight_decay': 1e-4},
            {'learning_rate': 1e-4, 'batch_size': 48, 'weight_decay': 1e-4},
            # Weight decay variations
            {'learning_rate': 1e-4, 'batch_size': 38, 'weight_decay': 1e-5},
            {'learning_rate': 1e-4, 'batch_size': 38, 'weight_decay': 1e-3},
            # Best combinations
            {'learning_rate': 5e-5, 'batch_size': 48, 'weight_decay': 1e-5},
            {'learning_rate': 2e-4, 'batch_size': 32, 'weight_decay': 1e-3},
        ]
        
        for i, hyperparams in enumerate(hyperparam_configs):
            config = base_config.copy()
            config.update(hyperparams)
            config['description'] = f"LR:{hyperparams['learning_rate']:.0e}, BS:{hyperparams['batch_size']}, WD:{hyperparams['weight_decay']:.0e}"
            
            print(f"\n  Experiment {i+1}/{len(hyperparam_configs)}: {config['description']}")
            study.run_single_experiment(config)
        
        study.analyze_results()
        return study
    
    def extract_best_config(self, study):
        """Extract the best configuration from a study."""
        if not study.results:
            return None
        
        completed_results = [r for r in study.results if r['status'] == 'completed']
        if not completed_results:
            return None
        
        # Sort by UAR (primary metric)
        best_result = max(completed_results, key=lambda x: x['results']['uar'])
        
        return {
            'config': best_result['config'],
            'uar': best_result['results']['uar'],
            'accuracy': best_result['results']['accuracy'],
            'experiment_name': best_result['experiment_name']
        }
    
    def save_progress(self):
        """Save current progress to file."""
        progress_file = self.main_results_dir / "progress_summary.json"
        with open(progress_file, 'w') as f:
            json.dump(self.results_summary, f, indent=2)
    
    def perform_final_analysis(self):
        """Perform final analysis across all studies."""
        print(f"\n{'='*80}")
        print("📊 FINAL ANALYSIS")
        print(f"{'='*80}")
        
        # Find overall best configurations
        all_best_configs = []
        for study in self.results_summary['studies_completed']:
            if study['best_config']:
                all_best_configs.append(study['best_config'])
        
        if all_best_configs:
            overall_best = max(all_best_configs, key=lambda x: x['uar'])
            
            print(f"\n🏆 OVERALL BEST CONFIGURATION:")
            print(f"   UAR: {overall_best['uar']:.4f}")
            print(f"   Accuracy: {overall_best['accuracy']:.4f}")
            print(f"   Experiment: {overall_best['experiment_name']}")
            print(f"   Config: {overall_best['config']}")
            
            self.results_summary['best_configurations']['overall_best'] = overall_best
        
        # Save final summary
        self.results_summary['end_time'] = datetime.now().isoformat()
        final_summary_file = self.main_results_dir / "final_summary.json"
        with open(final_summary_file, 'w') as f:
            json.dump(self.results_summary, f, indent=2)
        
        print(f"\n📄 Final summary saved to: {final_summary_file}")


def main():
    """Run the comprehensive ablation study."""
    print("🎯 COMPREHENSIVE ABLATION STUDY")
    print("This will run all ablation studies in sequence.")
    print("Expected duration: 4-8 hours depending on your hardware.")
    print()
    
    confirm = input("Do you want to proceed? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Ablation study cancelled.")
        return
    
    runner = ComprehensiveAblationRunner()
    results = runner.run_all_ablations()
    
    return results


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
YAML-based ablation study script for emotion recognition experiments using precomputed features.
Supports flexible experiment configuration via YAML files.
Uses precomputed features from cairocode/IEMO_Features_Curriculum and cairocode/MSPI_Features_Curriculum.
"""

import yaml
import json
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
import wandb
import sys
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from main import run_single_experiment


class PrecomputedEmotionDataset(Dataset):
    """Dataset class for precomputed emotion2vec features"""

    def __init__(self, dataset_name, split="train"):
        self.dataset_name = dataset_name
        self.split = split

        # Load precomputed features dataset
        if dataset_name == "IEMOCAP":
            self.hf_dataset = load_dataset(
                "cairocode/IEMO_Emotion2Vec",
                split=split,
                trust_remote_code=True,
            )

        elif dataset_name == "MSP-IMPROV":
            self.hf_dataset = load_dataset(
                "cairocode/MSPI_Emotion2Vec",
                split=split,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Create session mapping using existing get_session_splits function
        session_splits = self._get_session_splits_from_hf_dataset()
        speaker_to_session = {}
        for session_id, indices in session_splits.items():
            for idx in indices:
                try:
                    speaker_id = self.hf_dataset[idx]["speaker_id"]
                except:
                    speaker_id = self.hf_dataset[idx]["speakerID"]
                speaker_to_session[speaker_id] = session_id

        # Process and store data
        self.data = []
        for item in self.hf_dataset:
            # Extract features (assuming they're already computed and stored)
            features = torch.tensor(
                item["emotion2vec_features"][0]["feats"], dtype=torch.float32
            )

            # Get metadata
            try:
                speaker_id = item["speaker_id"]
            except:
                speaker_id = item["speakerID"]
            session = speaker_to_session[speaker_id]  # Map speaker_id to session
            label = item["label"]

            # Get curriculum-related data
            difficulty = item.get("difficulty", 0.0)
            curriculum_order = item.get("curriculum_order", 0)

            # Get VAD values (handle missing values)
            valence = item.get("valence", None)
            arousal = item.get("arousal", None)
            domination = item.get("domination", None)

            data_point = {
                "features": features,
                "label": label,
                "speaker_id": speaker_id,
                "session": session,
                "dataset": dataset_name,
                "difficulty": difficulty,
                "curriculum_order": curriculum_order,
                "valence": valence,
                "arousal": arousal,
                "domination": domination,
            }

            self.data.append(data_point)

        print(f"Loaded {len(self.data)} samples from {dataset_name}")

    def _get_session_splits_from_hf_dataset(self):
        """Get session-based splits from HuggingFace dataset using proper IEMOCAP session mapping"""
        session_splits = defaultdict(list)
        speaker_ids_seen = set()
        session_speaker_mapping = {}

        for i, item in enumerate(self.hf_dataset):
            try:
                speaker_id = item["speaker_id"]
            except:
                speaker_id = item["speakerID"]

            # CRITICAL FIX: Map speaker_id to proper session (same logic as EmotionDataset)
            # speakers 1&2→session 1, 3&4→session 2, etc.
            session = (speaker_id - 1) // 2 + 1
            
            # Debug tracking
            speaker_ids_seen.add(speaker_id)
            if session not in session_speaker_mapping:
                session_speaker_mapping[session] = set()
            session_speaker_mapping[session].add(speaker_id)
            
            session_splits[session].append(i)

        # Debug output for precomputed dataset
        print(f"🔍 PRECOMPUTED DEBUG - {self.dataset_name} Speaker IDs: {sorted(speaker_ids_seen)}")
        print(f"🔍 PRECOMPUTED DEBUG - Session mapping:")
        for session in sorted(session_speaker_mapping.keys()):
            speakers = sorted(session_speaker_mapping[session])
            print(f"   Session {session}: Speakers {speakers}")
        print(f"🔍 PRECOMPUTED DEBUG - Total sessions: {len(session_speaker_mapping)}")
        
        if self.dataset_name == "IEMOCAP" and len(session_speaker_mapping) != 5:
            print(f"⚠️  PRECOMPUTED WARNING: IEMOCAP should have 5 sessions, found {len(session_speaker_mapping)}!")

        return dict(session_splits)

    def get_session_splits(self):
        """Get session-based splits for LOSO (Leave-One-Session-Out)"""
        session_splits = defaultdict(list)

        for i, item in enumerate(self.data):
            session = item["session"]

            # Skip if session is None
            if session is not None:
                session_splits[session].append(i)

        return dict(session_splits)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get item by index"""
        item = self.data[idx]

        # Return data in the same format as original EmotionDataset
        return {
            "features": item["features"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "session": item["session"],
            "dataset": item["dataset"],
            "difficulty": item["difficulty"],
            "curriculum_order": item["curriculum_order"],
            "valence": item["valence"],
            "arousal": item["arousal"],
            "domination": item["domination"],
        }


class YamlAblationStudyPrecomputed:
    """Manages ablation study using YAML configuration with precomputed features"""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_study()
        self.load_datasets()

    def load_config(self):
        """Load experiment configuration from YAML file"""
        print(f"📖 Loading configuration from {self.config_path}")

        with open(self.config_path, "r") as f:
            self.yaml_config = yaml.safe_load(f)

        # Extract key sections
        self.study_name = self.yaml_config.get(
            "study_name", "yaml_ablation_study_precomputed"
        )
        self.wandb_project = self.yaml_config.get(
            "wandb_project", "emotion2vec_yaml_ablation_precomputed"
        )
        self.base_config = self.yaml_config.get("base_config", {})
        self.experiments = self.yaml_config.get("experiments", [])
        self.expected_results = self.yaml_config.get("expected_results", {})
        self.analysis_config = self.yaml_config.get("analysis", {})

        print(f"✅ Loaded {len(self.experiments)} experiments")
        print(f"📊 Study: {self.study_name}")
        print(f"🔬 Wandb project: {self.wandb_project}")

    def setup_study(self):
        """Setup study directories and progress tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save experiment results in experiments directory
        experiments_base = Path("experiments")
        experiments_base.mkdir(exist_ok=True)
        self.study_dir = experiments_base / f"{self.study_name}_{timestamp}"
        self.study_dir.mkdir(exist_ok=True)

        # Progress tracking files
        self.progress_file = self.study_dir / "progress.json"
        self.experiments_file = self.study_dir / "experiments_queue.json"
        self.completed_file = self.study_dir / "completed_experiments.json"
        self.results_file = self.study_dir / "all_results.json"

        # Initialize progress
        self.progress = {
            "study_name": self.study_name,
            "config_file": str(self.config_path),
            "start_time": datetime.now().isoformat(),
            "total_experiments": len(self.experiments),
            "completed_count": 0,
            "failed_count": 0,
            "status": "initialized",
            "using_precomputed_features": True,
        }

        self.completed_experiments = []
        self.save_progress()

        print(f"📁 Study directory: {self.study_dir}")
        print(f"⚡ Using precomputed features for faster execution")

    def load_datasets(self):
        """Load datasets once for all experiments"""
        print("🔄 Loading precomputed feature datasets...")
        self.iemocap_dataset = PrecomputedEmotionDataset("IEMOCAP", split="train")
        self.msp_dataset = PrecomputedEmotionDataset("MSP-IMPROV", split="train")
        print(f"✅ IEMOCAP: {len(self.iemocap_dataset)} samples")
        print(f"✅ MSP-IMPROV: {len(self.msp_dataset)} samples")

    def save_progress(self):
        """Save current progress and experiment state"""
        self.progress["last_update"] = datetime.now().isoformat()

        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)
        with open(self.experiments_file, "w") as f:
            json.dump(self.experiments, f, indent=2)
        with open(self.completed_file, "w") as f:
            json.dump(self.completed_experiments, f, indent=2)

    def create_experiment_config(self, experiment):
        """Create Config object for experiment"""
        config = Config()

        # Define which config parameters should be converted to specific types
        float_params = ["learning_rate", "weight_decay"]
        int_params = [
            "batch_size",
            "num_epochs",
            "curriculum_epochs",
            "single_test_session",
        ]
        bool_params = ["use_curriculum_learning", "use_speaker_disentanglement"]
        string_params = ["curriculum_pacing", "lr_scheduler"]

        def convert_value(key, value):
            """Convert config values to appropriate types"""
            if key in float_params:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value
            elif key in int_params:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return value
            elif key in bool_params:
                if isinstance(value, str):
                    return value.lower() in ["true", "1", "yes", "on"]
                return bool(value)
            elif key in string_params:
                return str(value)
            return value

        # Apply base configuration with type conversion
        for key, value in self.base_config.items():
            if hasattr(config, key):
                # Special handling for class_weights (dict)
                if key == "class_weights" and isinstance(value, dict):
                    setattr(config, key, value)
                else:
                    converted_value = convert_value(key, value)
                    setattr(config, key, converted_value)

        # Apply experiment-specific overrides with type conversion
        for key, value in experiment.items():
            if key in [
                "id",
                "name",
                "description",
                "category",
                "difficulty_method",
                "vad_weights",
                "classifier_config",
            ]:
                # Handle special parameters
                if key == "classifier_config":
                    setattr(config, key, value)
                elif key in ["difficulty_method", "vad_weights"]:
                    continue  # These are handled separately in run_single_experiment
                else:
                    continue  # Skip other metadata
            elif hasattr(config, key):
                # Special handling for class_weights (dict)
                if key == "class_weights" and isinstance(value, dict):
                    setattr(config, key, value)
                else:
                    converted_value = convert_value(key, value)
                    setattr(config, key, converted_value)

        # Set wandb project
        config.wandb_project = self.wandb_project
        
        # FORCE ALL SESSIONS: Override single_test_session if not explicitly set
        # This forces full LOSO evaluation instead of just session 5
        if not hasattr(config, 'single_test_session') or config.single_test_session == 5:
            # Check if single_test_session was explicitly set in base_config or experiment
            explicitly_set = (
                'single_test_session' in self.base_config or 
                'single_test_session' in experiment
            )
            if not explicitly_set:
                print("🔄 FORCING ALL SESSIONS: Setting single_test_session = None for full LOSO evaluation")
                config.single_test_session = None

        return config

    def run_single_experiment(self, experiment):
        """Run a single experiment"""
        print(f"\n{'='*80}")
        print(f"🔧 Starting: {experiment['name']}")
        print(f"📝 Description: {experiment['description']}")
        print(f"🏷️  Category: {experiment['category']}")
        print(
            f"🔢 Progress: {self.progress['completed_count'] + 1}/{self.progress['total_experiments']}"
        )
        print(f"⚡ Using precomputed features")
        print(f"{'='*80}")

        # Update progress
        self.progress["current_experiment"] = experiment["id"]
        self.progress["status"] = "running"
        self.save_progress()

        # Create config
        config = self.create_experiment_config(experiment)

        try:
            # Clean up any existing wandb runs
            try:
                wandb.finish()
            except:
                pass

            # Run experiment
            start_time = time.time()
            results = run_single_experiment(
                config,
                self.iemocap_dataset,
                self.msp_dataset,
                difficulty_method=experiment.get("difficulty_method", "original"),
                vad_weights=experiment.get("vad_weights", None),
                experiment_name=experiment["name"],
            )
            duration = time.time() - start_time

            # Add experiment metadata to results
            results["experiment_duration_minutes"] = duration / 60
            results["experiment_id"] = experiment["id"]
            results["experiment_name"] = experiment["name"]
            results["category"] = experiment["category"]
            results["difficulty_method"] = experiment.get(
                "difficulty_method", "original"
            )
            results["using_precomputed_features"] = True

            self.mark_experiment_completed(experiment, results)
            return results

        except Exception as e:
            self.mark_experiment_failed(experiment, e)
            raise

    def mark_experiment_completed(self, experiment, results):
        """Mark experiment as completed"""
        experiment["completion_time"] = datetime.now().isoformat()
        experiment["results"] = results
        experiment["status"] = "completed"

        self.completed_experiments.append(experiment)

        # Remove from queue (if this was queue-based)
        self.progress["completed_count"] = len(self.completed_experiments)
        self.progress["current_experiment"] = None
        self.save_progress()

        # Save individual results
        results_file = self.study_dir / f"{experiment['id']}_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment": experiment,
                    "results": results,
                    "completion_time": experiment["completion_time"],
                },
                f,
                indent=2,
            )

        print(f"✅ Completed: {experiment['name']}")

        # Show quick results
        iemocap_wa = results["iemocap_results"]["wa"]["mean"]
        iemocap_uar = results["iemocap_results"]["uar"]["mean"]
        cross_wa = results["cross_corpus_results"]["accuracy"]["mean"]
        cross_uar = results["cross_corpus_results"]["uar"]["mean"]

        print(f"   📊 IEMOCAP WA: {iemocap_wa:.4f} | UAR: {iemocap_uar:.4f}")
        print(f"   🌐 Cross-Corpus WA: {cross_wa:.4f} | UAR: {cross_uar:.4f}")
        print(f"   ⏱️  Duration: {results['experiment_duration_minutes']:.1f} minutes")
        print(f"   ⚡ Speed improvement from precomputed features!")

        # Compare to expected results if available
        exp_id = experiment["id"]
        if exp_id in self.expected_results:
            expected = self.expected_results[exp_id]
            print(f"   📈 Comparison to previous:")
            if "iemocap_wa" in expected:
                diff_wa = iemocap_wa - expected["iemocap_wa"]
                print(
                    f"      WA: {iemocap_wa:.4f} vs {expected['iemocap_wa']:.4f} ({diff_wa:+.4f})"
                )
            if "iemocap_uar" in expected:
                diff_uar = iemocap_uar - expected["iemocap_uar"]
                print(
                    f"      UAR: {iemocap_uar:.4f} vs {expected['iemocap_uar']:.4f} ({diff_uar:+.4f})"
                )
            if "cross_uar" in expected:
                diff_cross = cross_uar - expected["cross_uar"]
                print(
                    f"      Cross-UAR: {cross_uar:.4f} vs {expected['cross_uar']:.4f} ({diff_cross:+.4f})"
                )

    def mark_experiment_failed(self, experiment, error):
        """Mark experiment as failed"""
        full_traceback = traceback.format_exc()

        experiment["last_failure"] = {
            "time": datetime.now().isoformat(),
            "error": str(error),
            "traceback": full_traceback,
        }
        experiment["failure_count"] = experiment.get("failure_count", 0) + 1

        self.progress["failed_count"] += 1
        self.progress["current_experiment"] = None
        self.save_progress()

        print(f"❌ Failed: {experiment['name']}")
        print("=" * 80)
        print(f"Error Type: {type(error).__name__}")
        print(f"Error Message: {str(error)}")
        print("\nFull Traceback:")
        print(full_traceback)
        print("=" * 80)

    def run_ablation_study(self, max_failures_per_experiment=2):
        """Run the complete ablation study"""
        print(f"\n🚀 STARTING YAML-BASED ABLATION STUDY WITH PRECOMPUTED FEATURES")
        print(f"📁 Results directory: {self.study_dir}")
        print(f"📊 Total experiments: {len(self.experiments)}")
        print(
            f"⏱️  Estimated time: {len(self.experiments) * 0.2:.1f} hours (much faster with precomputed features!)"
        )
        print(f"🔬 Wandb project: {self.wandb_project}")
        print(
            f"⚡ Using precomputed features from cairocode/IEMO_Features_Curriculum and cairocode/MSPI_Features_Curriculum"
        )

        start_time = time.time()
        remaining_experiments = self.experiments.copy()

        while remaining_experiments:
            experiment = remaining_experiments[0]

            # Skip if too many failures
            if experiment.get("failure_count", 0) >= max_failures_per_experiment:
                print(f"⏭️  Skipping {experiment['name']} (too many failures)")
                remaining_experiments.pop(0)
                continue

            try:
                self.run_single_experiment(experiment)
                remaining_experiments.pop(0)

            except KeyboardInterrupt:
                print(f"\n🛑 Study interrupted by user")
                print(
                    f"💾 Progress saved. Resume with: python {__file__} {self.config_path}"
                )
                return

            except Exception as e:
                print(f"💥 Experiment failed, continuing...")
                continue

        # Study completed
        total_time = time.time() - start_time
        self.progress["status"] = "completed"
        self.progress["completion_time"] = datetime.now().isoformat()
        self.progress["total_duration_hours"] = total_time / 3600
        self.save_progress()

        self.analyze_results()

    def analyze_results(self):
        """Analyze and summarize results"""
        print(f"\n{'='*80}")
        print(f"🎉 YAML ABLATION STUDY WITH PRECOMPUTED FEATURES COMPLETED!")
        print(f"{'='*80}")

        if not self.completed_experiments:
            print("❌ No experiments completed successfully")
            return

        print(f"📊 Final Statistics:")
        print(f"   Total experiments: {len(self.experiments)}")
        print(f"   Completed: {len(self.completed_experiments)}")
        print(f"   Failed: {self.progress['failed_count']}")
        print(f"   Duration: {self.progress.get('total_duration_hours', 0):.1f} hours")
        print(f"   ⚡ Speed improvement from precomputed features!")

        # Get primary metric for ranking
        primary_metric = self.analysis_config.get("primary_metric", "iemocap_wa")

        # Find best performers
        best_overall = max(
            self.completed_experiments,
            key=lambda x: x["results"]["iemocap_results"]["wa"]["mean"],
        )

        best_cross = max(
            self.completed_experiments,
            key=lambda x: x["results"]["cross_corpus_results"]["uar"]["mean"],
        )

        print(f"\n🥇 BEST PERFORMERS:")
        print(
            f"Best Overall (WA): {best_overall['name']} - {best_overall['results']['iemocap_results']['wa']['mean']:.4f}"
        )
        print(
            f"Best Cross-Corpus: {best_cross['name']} - {best_cross['results']['cross_corpus_results']['uar']['mean']:.4f}"
        )

        # Top 3 ranking
        print(f"\n📈 TOP 3 OVERALL (by WA):")
        sorted_by_wa = sorted(
            self.completed_experiments,
            key=lambda x: x["results"]["iemocap_results"]["wa"]["mean"],
            reverse=True,
        )

        for i, exp in enumerate(sorted_by_wa[:3]):
            results = exp["results"]
            print(f"   {i+1}. {exp['name']}")
            print(f"      WA: {results['iemocap_results']['wa']['mean']:.4f}")
            print(f"      UAR: {results['iemocap_results']['uar']['mean']:.4f}")
            print(
                f"      Cross-UAR: {results['cross_corpus_results']['uar']['mean']:.4f}"
            )
            print(f"      Category: {exp['category']}")

        # Speaker effect analysis
        print(f"\n🔊 SPEAKER DISENTANGLEMENT ANALYSIS:")
        speaker_pairs = {}

        for exp in self.completed_experiments:
            base_name = exp["name"].replace(" + Speaker", "").replace("+ Speaker", "")
            is_speaker = (
                exp.get("use_speaker_disentanglement", False)
                or "Speaker" in exp["name"]
            )

            if base_name not in speaker_pairs:
                speaker_pairs[base_name] = {"without": None, "with": None}

            key = "with" if is_speaker else "without"
            speaker_pairs[base_name][key] = exp

        for base_name, pair in speaker_pairs.items():
            if pair["without"] and pair["with"]:
                without_wa = pair["without"]["results"]["iemocap_results"]["wa"]["mean"]
                with_wa = pair["with"]["results"]["iemocap_results"]["wa"]["mean"]
                improvement = with_wa - without_wa

                print(f"   {base_name}:")
                print(f"      Without Speaker: {without_wa:.4f}")
                print(f"      With Speaker: {with_wa:.4f}")
                print(f"      Improvement: {improvement:+.4f}")

        # Save comprehensive results
        comprehensive_results = {
            "study_config": self.yaml_config,
            "progress": self.progress,
            "completed_experiments": self.completed_experiments,
            "using_precomputed_features": True,
            "feature_datasets": {
                "iemocap": "cairocode/IEMO_Emotion2Vec",
                "msp": "cairocode/MSPI_Emotion2Vec",
            },
            "analysis": {
                "best_overall": best_overall["name"],
                "best_cross_corpus": best_cross["name"],
                "top_3": [exp["name"] for exp in sorted_by_wa[:3]],
                "speaker_analysis": speaker_pairs,
            },
        }

        with open(self.results_file, "w") as f:
            json.dump(comprehensive_results, f, indent=2)

        print(f"\n💾 Comprehensive results saved to: {self.results_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YAML-based ablation study for emotion recognition using precomputed features"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show experiments without running"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        return

    study = YamlAblationStudyPrecomputed(args.config)

    if args.dry_run:
        print("\n🔍 DRY RUN - Experiments to be run:")
        for i, exp in enumerate(study.experiments):
            print(f"   {i+1}. {exp['name']} ({exp['category']})")
            print(f"      Method: {exp['difficulty_method']}")
            if "vad_weights" in exp:
                print(f"      VAD weights: {exp['vad_weights']}")
            if exp.get("use_speaker_disentanglement"):
                print(f"      Speaker: Yes")
            print(f"      Description: {exp['description']}")
            print()
        print("⚡ Note: Using precomputed features for faster execution!")
        return

    study.run_ablation_study()


if __name__ == "__main__":
    main()

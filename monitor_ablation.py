"""
Monitor ablation study progress and display real-time results.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import argparse


def find_latest_ablation_dir():
    """Find the most recent ablation results directory."""
    current_dir = Path(".")
    ablation_dirs = list(current_dir.glob("comprehensive_ablation_*"))
    
    if not ablation_dirs:
        return None
    
    # Sort by creation time and return the latest
    latest_dir = max(ablation_dirs, key=lambda p: p.stat().st_mtime)
    return latest_dir


def load_progress(results_dir):
    """Load progress from the results directory."""
    progress_file = results_dir / "progress_summary.json"
    
    if not progress_file.exists():
        return None
    
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def display_progress(progress):
    """Display the current progress."""
    if not progress:
        print("❌ No progress data found")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 ABLATION STUDY PROGRESS")
    print(f"{'='*80}")
    
    start_time = datetime.fromisoformat(progress['start_time'])
    current_time = datetime.now()
    elapsed = (current_time - start_time).total_seconds() / 60
    
    print(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Elapsed: {elapsed:.1f} minutes")
    
    completed = len(progress['studies_completed'])
    failed = len(progress['studies_failed'])
    total_planned = 5  # Number of planned studies
    
    print(f"✅ Completed: {completed}/{total_planned} studies")
    print(f"❌ Failed: {failed} studies")
    
    if completed > 0:
        print(f"\n📈 COMPLETED STUDIES:")
        for i, study in enumerate(progress['studies_completed'], 1):
            print(f"  {i}. {study['name']} ({study['duration_minutes']:.1f} min)")
            if study['best_config']:
                print(f"     Best UAR: {study['best_config']['uar']:.4f}")
    
    if failed > 0:
        print(f"\n💥 FAILED STUDIES:")
        for i, study in enumerate(progress['studies_failed'], 1):
            print(f"  {i}. {study['name']}: {study['error']}")
    
    # Show best configuration so far
    if 'best_configurations' in progress and 'overall_best' in progress['best_configurations']:
        best = progress['best_configurations']['overall_best']
        print(f"\n🏆 CURRENT BEST CONFIGURATION:")
        print(f"   UAR: {best['uar']:.4f} | Accuracy: {best['accuracy']:.4f}")
        print(f"   Experiment: {best['experiment_name']}")


def display_detailed_results(results_dir):
    """Display detailed results from individual studies."""
    print(f"\n{'='*80}")
    print(f"📋 DETAILED RESULTS")
    print(f"{'='*80}")
    
    study_dirs = [
        ("01_quick_ablation", "Quick Ablation"),
        ("02_architecture_ablation", "Architecture Ablation"),
        ("03_curriculum_epochs", "Curriculum Epochs"),
        ("04_class_weights", "Class Weights"),
        ("05_hyperparameters", "Hyperparameters"),
    ]
    
    for study_dir, study_name in study_dirs:
        study_path = results_dir / study_dir
        results_file = study_path / "ablation_results.json"
        
        if not results_file.exists():
            print(f"\n⏳ {study_name}: Not started yet")
            continue
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            completed_results = [r for r in results if r['status'] == 'completed']
            
            if not completed_results:
                print(f"\n🔄 {study_name}: In progress ({len(results)} experiments)")
                continue
            
            # Sort by UAR
            completed_results.sort(key=lambda x: x['results']['uar'], reverse=True)
            
            print(f"\n✅ {study_name}: {len(completed_results)} experiments completed")
            print(f"   Top 3 results:")
            
            for i, result in enumerate(completed_results[:3], 1):
                metrics = result['results']
                desc = result['config'].get('description', 'No description')[:50]
                print(f"   {i}. UAR: {metrics['uar']:.4f} | Acc: {metrics['accuracy']:.4f} | {desc}")
        
        except Exception as e:
            print(f"\n❌ {study_name}: Error reading results - {str(e)}")


def monitor_continuous(results_dir, interval=30):
    """Monitor progress continuously."""
    print(f"🔄 Monitoring ablation progress every {interval} seconds...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            print("\033[2J\033[H")  # Clear screen
            progress = load_progress(results_dir)
            display_progress(progress)
            
            print(f"\n{'='*80}")
            print(f"🔄 Next update in {interval} seconds... (Ctrl+C to stop)")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Monitor ablation study progress")
    parser.add_argument("--dir", help="Ablation results directory (auto-detect if not provided)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results from each study")
    parser.add_argument("--monitor", action="store_true", help="Monitor progress continuously")
    parser.add_argument("--interval", type=int, default=30, help="Update interval for monitoring (seconds)")
    
    args = parser.parse_args()
    
    # Find results directory
    if args.dir:
        results_dir = Path(args.dir)
    else:
        results_dir = find_latest_ablation_dir()
    
    if not results_dir or not results_dir.exists():
        print("❌ No ablation results directory found")
        print("Make sure you've started the ablation study with run_all_ablations.py")
        return
    
    print(f"📁 Monitoring: {results_dir}")
    
    if args.monitor:
        monitor_continuous(results_dir, args.interval)
    else:
        progress = load_progress(results_dir)
        display_progress(progress)
        
        if args.detailed:
            display_detailed_results(results_dir)


if __name__ == "__main__":
    main()
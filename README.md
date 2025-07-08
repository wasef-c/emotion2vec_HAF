# Emotion Recognition with Curriculum Learning

A clean, simplified implementation for emotion recognition using curriculum learning and adaptive emotional saliency.

## Structure

### Core Files
- **`config.py`** - Configuration management and hyperparameters
- **`functions.py`** - All core functionality (dataset, models, training, evaluation, difficulty calculation)
- **`main.py`** - Single experiment runner with command-line interface
- **`ablation_script.py`** - Systematic ablation study for difficulty calculation methods

### Archive
- **`archive/`** - All previous experimental code, checkpoints, and results

## Quick Start

### Run Single Experiment
```bash
# Basic run with original difficulty
python main.py --experiment-name "test_run"

# Test weighted VAD method  
python main.py --difficulty-method weighted_vad_valence --experiment-name "vad_test"

# With speaker disentanglement
python main.py --difficulty-method euclidean_distance --use-speaker-disentanglement
```

### Run Full Ablation Study
```bash
python ablation_script.py
```

## Available Difficulty Methods
- `original` - Use dataset's original difficulty column
- `pearson_correlation` - Pearson correlation between expected and actual VAD
- `spearman_correlation` - Spearman correlation between expected and actual VAD  
- `euclidean_distance` - Directional Euclidean distance from expected VAD
- `weighted_vad_valence` - Weighted VAD emphasizing valence [0.5, 0.4, 0.1]
- `weighted_vad_balanced` - Balanced weighted VAD [0.4, 0.4, 0.2]
- `weighted_vad_arousal` - Weighted VAD emphasizing arousal [0.4, 0.5, 0.1]

## Key Features
- ✅ Fixed curriculum learning bugs (index mapping, validation contamination)
- ✅ Realistic VAD thresholds for emotion classes
- ✅ Adaptive curriculum thresholds based on actual difficulty distribution
- ✅ LOSO evaluation with proper train/val/test splits
- ✅ Cross-corpus evaluation on MSP-IMPROV
- ✅ Wandb logging and experiment tracking
- ✅ Progress tracking and resumable ablation studies

## Expected VAD Values (0-5 scale)
- **Neutral**: [2.5, 2.5, 2.5] - Middle values
- **Happy**: [3.8, 3.2, 3.2] - High valence, high arousal, moderate-high dominance  
- **Sad**: [1.8, 2.0, 2.0] - Low valence, low arousal, low dominance
- **Anger**: [2.0, 3.8, 3.5] - Low valence, high arousal, high dominance
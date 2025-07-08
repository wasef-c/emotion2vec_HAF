# Curriculum Learning Options Documentation

## Overview

The emotion recognition system supports sophisticated curriculum learning with multiple difficulty calculation methods and pacing strategies. This document covers all available options.

## 1. Curriculum Pacing Functions

Controls **how** the curriculum progresses over training epochs.

### Available Options:

#### `linear` (Default)
- **Description**: Linear progression from easy to hard samples
- **Formula**: `progress = epoch / curriculum_epochs`
- **Characteristics**: Steady, uniform progression
- **Use Case**: Baseline curriculum learning

#### `exponential` 
- **Description**: Slow start, accelerated progression toward end
- **Formula**: `progress = (epoch / curriculum_epochs) ^ 1.5`
- **Characteristics**: More time on easy samples, rapid transition to hard
- **Use Case**: When model needs more time to learn basics

#### `logarithmic`
- **Description**: Fast start, slower progression toward end
- **Formula**: `progress = sqrt(epoch / curriculum_epochs)`
- **Characteristics**: Quick introduction of harder samples, gradual refinement
- **Use Case**: When model learns quickly initially


## 2. Difficulty Calculation Methods

Controls **what** makes a sample difficult.

### 2.1 Baseline Methods

#### `original`
- **Description**: Uses original `difficulty` column from dataset
- **Source**: Pre-computed difficulty values
- **Range**: [0, 1] (easy to hard)

#### `preset` ⭐ **NEW**
- **Description**: Uses predefined `curriculum_order` column as difficulty values
- **Source**: Expert-curated curriculum ordering from dataset
- **Range**: Normalized to [0, 1] (easy to hard)
- **Use Case**: When domain experts have defined optimal learning sequence

#### `baseline_quadratic` ⭐ **NEW**
- **Description**: Replicates original baseline algorithm with quadratic VAD penalties
- **Formula**: `difficulty = Σ(weight_i × (actual_i - expected_i)²)`
- **Default Weights**: `{valence: 0.5, arousal: 0.3, domination: 0.2}`
- **Variants**:
  - Original: `[0.5, 0.3, 0.2]`
  - Balanced: `[0.33, 0.33, 0.34]`
  - Valence-heavy: `[0.6, 0.25, 0.15]`
  - Arousal-heavy: `[0.3, 0.5, 0.2]`

### 2.2 Correlation Methods

#### `pearson_correlation`
- **Description**: Difficulty based on Pearson correlation between expected and actual VAD
- **Formula**: `difficulty = 1 - abs(pearson_corr(expected_VAD, actual_VAD))`
- **Range**: [0, 1] (high correlation = easy, low correlation = hard)

#### `spearman_correlation`
- **Description**: Difficulty based on Spearman rank correlation
- **Formula**: `difficulty = 1 - abs(spearman_corr(expected_VAD, actual_VAD))`
- **Characteristics**: Less sensitive to outliers than Pearson

### 2.3 Distance Methods

#### `euclidean_distance`
- **Description**: Direction-aware Euclidean distance with emotion-specific penalties
- **Formula**: `difficulty = sqrt(Σ(penalty_i²))` where penalties depend on emotion
- **Special Logic**:
  - Happy: Only penalize lower valence/arousal/dominance
  - Sad: Only penalize higher valence/arousal/dominance  
  - Anger: Only penalize higher valence, lower arousal/dominance
  - Neutral: Penalize any deviation

### 2.4 Weighted VAD Methods

#### `weighted_vad_balanced`
- **Description**: Weighted combination of VAD deviations
- **Weights**: `[0.4, 0.4, 0.2]` (V, A, D)
- **Use Case**: Equal emphasis on valence and arousal

#### `weighted_vad_valence`
- **Description**: Valence-focused difficulty calculation
- **Weights**: `[0.5, 0.4, 0.1]` (V, A, D)
- **Use Case**: When valence is most important

#### `weighted_vad_arousal`
- **Description**: Arousal-focused difficulty calculation
- **Weights**: `[0.4, 0.5, 0.1]` (V, A, D)
- **Use Case**: When arousal is most important

## 3. Configuration Parameters

### Core Settings
```python
config.use_curriculum_learning = True/False
config.curriculum_epochs = 15          # Number of epochs for curriculum
config.curriculum_pacing = "exponential"  # Pacing function
```

### Advanced Settings
```python
config.start_threshold = 0.2          # Start with easiest 20%
config.end_threshold = 0.95           # End with 95% of samples
```

## 4. Usage Examples

### Command Line Usage

#### Basic curriculum with original difficulty:
```bash
python main.py --curriculum-pacing exponential --curriculum-epochs 15
```

#### Preset curriculum using expert ordering:
```bash
python main.py --curriculum-pacing preset --curriculum-epochs 20
```

#### Custom difficulty method:
```bash
python main.py --difficulty-method baseline_quadratic --curriculum-pacing linear
```

### Programmatic Usage

```python
from config import Config
from main import run_single_experiment

# Configure curriculum learning
config = Config()
config.use_curriculum_learning = True
config.curriculum_epochs = 15
config.curriculum_pacing = "preset"  # Use expert-defined ordering

# Run experiment
results = run_single_experiment(
    config, 
    iemocap_dataset, 
    msp_dataset,
    difficulty_method="baseline_quadratic",
    vad_weights=[0.5, 0.3, 0.2]
)
```

## 5. Dataset Requirements

### Required Columns
- `difficulty`: Numerical difficulty values [0, 1]
- `valence`: VAD valence values [1, 5] 
- `arousal`: VAD arousal values [1, 5]
- `domination`: VAD dominance values [1, 5]

### Optional Columns
- `curriculum_order`: Expert-defined curriculum ordering [0, 1] (required for `preset` pacing)

## 6. Expected VAD Values by Emotion

```python
EXPECTED_VAD = {
    0: [2.5, 2.5, 2.5],  # neutral - middle values
    1: [3.8, 3.2, 3.2],  # happy - high valence, moderate arousal/dominance
    2: [1.8, 2.0, 2.0],  # sad - low values across dimensions
    3: [2.0, 3.8, 3.5],  # anger - low valence, high arousal/dominance
}
```

## 7. Ablation Study Categories

The system automatically categorizes experiments:

- **baseline**: Original difficulty methods
- **baseline_quadratic**: Quadratic VAD penalty variants
- **preset_curriculum**: Expert-defined curriculum
- **correlation**: Correlation-based difficulty
- **distance**: Distance-based difficulty  
- **weighted**: Weighted VAD combinations
- **speaker_enhanced**: Methods with speaker disentanglement

## 8. Performance Monitoring

During training, the system logs:
- Curriculum threshold per epoch
- Number of samples included
- Class distribution validation
- Progress indicators

### Wandb Logging
```python
wandb.log({
    f"session_{test_session}/curriculum_threshold": current_threshold,
    f"session_{test_session}/curriculum_samples": curriculum_samples,
    f"session_{test_session}/curriculum_progress": progress,
})
```

## 9. Advanced Features

### Class Balance Validation
- Ensures all 4 emotion classes present at each curriculum stage
- Adds emergency samples if classes missing
- Warns about severe class imbalance (>10:1 ratio)

### Adaptive Thresholds
- Automatically sets start/end thresholds based on difficulty distribution
- `start_threshold = 20th percentile`
- `end_threshold = 95th percentile`

### Emergency Fallbacks
- Uses easiest 10% if no samples pass threshold
- Prevents empty batches during training

## 10. Best Practices

### Choosing Pacing Function:
- **Linear**: Good baseline, works for most cases
- **Exponential**: Use when model struggles initially
- **Logarithmic**: Use for fast-learning models
- **Preset**: Use when you have expert domain knowledge

### Choosing Difficulty Method:
- **Original**: Baseline comparison
- **Baseline Quadratic**: Proven effective method with variants
- **Euclidean Distance**: Good for direction-aware difficulty
- **Weighted VAD**: When specific VAD dimensions matter most

### Curriculum Epochs:
- **5-10 epochs**: Fast curriculum for quick models
- **15-20 epochs**: Standard setting for most experiments  
- **25-30 epochs**: Extended curriculum for complex tasks

## 11. Troubleshooting

### Common Issues:

1. **Missing curriculum_order column**:
   - Solution: Ensure dataset has `curriculum_order` column for preset pacing
   - Fallback: Use other pacing functions

2. **All samples same difficulty**:
   - System adds small random noise automatically
   - Check difficulty calculation method

3. **Class imbalance warnings**:
   - Normal for early curriculum stages
   - System automatically adds minority class samples

4. **Empty curriculum stages**:
   - System uses emergency fallback (easiest 10%)
   - Consider adjusting curriculum_epochs

This documentation covers all curriculum learning options available in the system. Each method can be combined for comprehensive ablation studies.
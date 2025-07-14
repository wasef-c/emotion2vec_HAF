#!/usr/bin/env python3
"""
Test the existing EmotionDataset to reproduce 70.8% baseline
Uses existing functions.py without any modifications
"""

import sys
import os
from config import Config
from main import run_single_experiment
from functions import EmotionDataset


def create_baseline_config():
    """Create the exact 70.8% baseline config"""
    config = Config()
    
    # EXACT 70.8% baseline parameters
    config.learning_rate = 9e-5
    config.weight_decay = 5e-6
    config.focal_loss = True
    config.focal_alpha = 0.25
    config.focal_gamma = 2.0
    config.label_smoothing = 0.1
    config.num_epochs = 80
    config.batch_size = 16
    config.lr_scheduler = "cosine"
    
    # Model architecture
    config.use_speaker_disentanglement = True
    config.use_curriculum_learning = True
    config.difficulty_method = "pearson_correlation"
    config.curriculum_epochs = 30
    
    # Class weights
    config.class_weights = {
        'neutral': 0.8,
        'happy': 3.5,
        'sad': 1.8,
        'anger': 1.6
    }
    
    # Classifier config
    config.classifier_config = {
        'architecture': 'simple',
        'hidden_dim': 1536,
        'pooling': 'attention',
        'layer_norm': True,
        'dropout': 0.4,
        'input_dropout': 0.2
    }
    
    config.experiment_name = "test_existing_baseline"
    config.wandb_project = "emotion2vec_test_existing"
    
    return config


def main():
    print("🚀 Testing existing EmotionDataset to reproduce 70.8% baseline")
    print("Using functions.py as-is with utterance-level features")
    
    # Create the exact baseline config
    config = create_baseline_config()
    
    print(f"Config: LR={config.learning_rate}, focal_loss={config.focal_loss}, label_smoothing={config.label_smoothing}")
    
    # Load datasets using existing EmotionDataset
    print("🔄 Loading IEMOCAP with existing EmotionDataset...")
    iemocap_dataset = EmotionDataset("IEMOCAP", split="train")
    
    print("🔄 Loading MSP-IMPROV with existing EmotionDataset...")
    msp_dataset = EmotionDataset("MSP-IMPROV", split="train")
    
    print(f"✅ IEMOCAP: {len(iemocap_dataset)} samples")
    print(f"✅ MSP-IMPROV: {len(msp_dataset)} samples")
    
    # Run the experiment
    try:
        results = run_single_experiment(
            config, 
            iemocap_dataset, 
            msp_dataset,
            difficulty_method=config.difficulty_method,
            experiment_name=config.experiment_name
        )
        
        wa = results.get('wa', 0)
        print(f"🎯 Results: WA={wa:.4f}")
        
        if wa > 0.70:
            print("✅ SUCCESS: Reproduced 70%+ baseline with existing EmotionDataset!")
        else:
            print("❌ Did not reach 70% - need to investigate")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
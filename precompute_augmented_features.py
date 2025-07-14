#!/usr/bin/env python3
"""
Precompute emotion2vec features with different audio augmentations
Creates datasets for fast loading during training experiments
"""

import numpy as np
import soundfile as sf
import tempfile
import torchaudio
import torch
from contextlib import contextmanager
import os
import sys
from datasets import load_dataset, DatasetDict
from funasr import AutoModel
from tqdm import tqdm
import argparse


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def resample_audio_to_16k(audio_data, sampling_rate):
    """Convert audio to mono and resample to 16kHz."""
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)  # stereo to mono
    tensor_audio = torchaudio.functional.resample(
        torch.tensor(audio_data), orig_freq=sampling_rate, new_freq=16000
    )
    return tensor_audio.numpy()


def apply_audio_augmentation(audio_data, augmentation_type, **kwargs):
    """Apply audio augmentation to raw audio"""
    
    if augmentation_type == "baseline":
        return audio_data
    
    elif augmentation_type == "gaussian_noise":
        noise_level = kwargs.get("noise_level", 0.01)
        noise = np.random.normal(0, noise_level, audio_data.shape)
        return audio_data + noise
    
    elif augmentation_type == "volume_scale":
        scale_factor = kwargs.get("scale_factor", 0.8)
        return audio_data * scale_factor
    
    elif augmentation_type == "speed_perturbation":
        # Speed perturbation - modifies both tempo and pitch via resampling
        speed_factor = kwargs.get("speed_factor", 1.1)
        # Resample to change speed (affects both tempo and pitch)
        new_length = int(len(audio_data) / speed_factor)
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        return np.interp(indices, np.arange(len(audio_data)), audio_data)
    
    elif augmentation_type == "time_stretch":
        stretch_factor = kwargs.get("stretch_factor", 1.1)
        # Simple time stretching by resampling
        stretched_length = int(len(audio_data) / stretch_factor)
        indices = np.linspace(0, len(audio_data) - 1, stretched_length)
        return np.interp(indices, np.arange(len(audio_data)), audio_data)
    
    elif augmentation_type == "low_pass_filter":
        # Simple low-pass by removing high frequencies
        cutoff_freq = kwargs.get("cutoff_freq", 0.8)
        # Apply basic smoothing as approximation
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(audio_data, kernel, mode='same')
    
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def extract_emotion2vec_features(audio_data, emotion2vec_model):
    """Extract emotion2vec features using the same process as precomputed"""
    
    # Save to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio_data, 16000)
        temp_wav_path = temp_file.name

    # Extract features
    try:
        with suppress_output():
            result = emotion2vec_model.generate(
                temp_wav_path,
                output_dir=None,
                granularity="utterance",
                extract_embedding=True,
            )
        embedding = result if isinstance(result, list) else result.get("embedding", []) if isinstance(result, dict) else []
    except Exception as e:
        print(f"Error processing sample: {e}")
        embedding = []
    finally:
        os.remove(temp_wav_path)

    return embedding


def create_augmented_dataset(original_dataset, emotion2vec_model, augmentations):
    """Create dataset with multiple augmentation variants"""
    
    augmented_data = []
    
    for i in tqdm(range(len(original_dataset))):
        item = original_dataset[i]
        audio = item["audio"]
        audio_data = audio["array"]
        sampling_rate = audio["sampling_rate"]
        
        # Convert to mono + resample to 16kHz
        resampled_audio = resample_audio_to_16k(audio_data, sampling_rate)
        
        # Create one entry for each augmentation
        for aug_name, aug_config in augmentations.items():
            aug_type = aug_config["type"]
            aug_params = aug_config.get("params", {})
            
            # Apply augmentation
            augmented_audio = apply_audio_augmentation(
                resampled_audio, aug_type, **aug_params
            )
            
            # Extract features
            features = extract_emotion2vec_features(augmented_audio, emotion2vec_model)
            
            # Create new item
            new_item = {**item}  # Copy all original fields
            del new_item["audio"]  # Remove audio to save space
            new_item["emotion2vec_features"] = features
            new_item["augmentation"] = aug_name
            new_item["augmentation_type"] = aug_type
            new_item["augmentation_params"] = aug_params
            
            augmented_data.append(new_item)
    
    return augmented_data


def main():
    parser = argparse.ArgumentParser(description='Precompute augmented emotion2vec features')
    parser.add_argument('--dataset', choices=['IEMOCAP', 'MSP-IMPROV'], 
                       default='IEMOCAP', help='Dataset to process')
    parser.add_argument('--output-name', type=str, 
                       help='Output dataset name (default: auto-generated)')
    
    args = parser.parse_args()
    
    print(f"🚀 Precomputing augmented features for {args.dataset}")
    
    # Define augmentation strategies
    augmentations = {
        "baseline": {
            "type": "baseline",
            "params": {}
        },
        # Speed perturbation - most robust traditional augmentation
        "speed_0.9": {
            "type": "speed_perturbation",
            "params": {"speed_factor": 0.9}  # Slower (lower pitch)
        },
        "speed_1.1": {
            "type": "speed_perturbation", 
            "params": {"speed_factor": 1.1}  # Faster (higher pitch)
        },
        "speed_0.95": {
            "type": "speed_perturbation",
            "params": {"speed_factor": 0.95}  # Slightly slower
        },
        "speed_1.05": {
            "type": "speed_perturbation",
            "params": {"speed_factor": 1.05}  # Slightly faster
        },
        # Gaussian noise augmentations
        "light_noise": {
            "type": "gaussian_noise", 
            "params": {"noise_level": 0.005}
        },
        "medium_noise": {
            "type": "gaussian_noise",
            "params": {"noise_level": 0.01}
        },
        # Volume scaling
        "volume_down": {
            "type": "volume_scale",
            "params": {"scale_factor": 0.8}
        },
        "volume_up": {
            "type": "volume_scale", 
            "params": {"scale_factor": 1.2}
        },
        # Low-pass filtering
        "low_pass": {
            "type": "low_pass_filter",
            "params": {"cutoff_freq": 0.8}
        }
    }
    
    print(f"📊 Will create {len(augmentations)} augmentation variants:")
    for name, config in augmentations.items():
        print(f"  - {name}: {config['type']} {config['params']}")
    
    # Load emotion2vec model
    print("🔄 Loading emotion2vec model...")
    emotion2vec_model = AutoModel(model="iic/emotion2vec_base")
    print("✅ emotion2vec model loaded")
    
    # Load original dataset
    if args.dataset == "IEMOCAP":
        dataset_name = "cairocode/IEMO_WAV_Diff_2_Curriculum"
    else:
        dataset_name = "cairocode/MSPI_WAV_Diff_Curriculum"
    
    print(f"🔄 Loading {args.dataset} dataset...")
    original_dataset = load_dataset(dataset_name, trust_remote_code=True)
    
    # Process each split
    augmented_dataset = DatasetDict()
    
    for split_name, split_data in original_dataset.items():
        print(f"\n📝 Processing {split_name} split ({len(split_data)} samples)...")
        
        augmented_data = create_augmented_dataset(
            split_data, emotion2vec_model, augmentations
        )
        
        # Convert to HuggingFace dataset
        from datasets import Dataset
        augmented_dataset[split_name] = Dataset.from_list(augmented_data)
        
        total_samples = len(augmented_data)
        samples_per_aug = total_samples // len(augmentations)
        print(f"✅ Created {total_samples} augmented samples ({samples_per_aug} per augmentation)")
    
    # Generate output name
    if args.output_name:
        output_name = args.output_name
    else:
        dataset_prefix = "IEMO" if args.dataset == "IEMOCAP" else "MSPI"
        output_name = f"{dataset_prefix}_Emotion2Vec_Augmented"
    
    print(f"\n💾 Pushing to HuggingFace Hub: {output_name}")
    try:
        augmented_dataset.push_to_hub(output_name)
        print(f"✅ Successfully uploaded to: cairocode/{output_name}")
        
        print(f"\n🎯 Usage:")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('cairocode/{output_name}')")
        print(f"# Filter by augmentation:")
        print(f"baseline = dataset.filter(lambda x: x['augmentation'] == 'baseline')")
        print(f"noisy = dataset.filter(lambda x: x['augmentation'] == 'medium_noise')")
        
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        print(f"💾 Saving locally to: ./augmented_dataset")
        augmented_dataset.save_to_disk("./augmented_dataset")


if __name__ == "__main__":
    main()
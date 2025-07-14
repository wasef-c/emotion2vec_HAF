from datasets import DatasetDict
from tqdm import tqdm
import numpy as np
import soundfile as sf
import tempfile
import torchaudio
from contextlib import contextmanager
import os
import sys
from datasets import load_dataset
from funasr import AutoModel
import torch

# Context manager to suppress stdout/stderr
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


def modify_dataset_with_emotion2vec(
    original_dataset: DatasetDict, emotion2vec_model
) -> DatasetDict:
    modified_dataset = DatasetDict()

    for split_name, split_data in original_dataset.items():
        print(f"Processing split: {split_name}")

        features = []
        for i in tqdm(range(len(split_data))):
            # Get raw audio array and sampling rate
            audio = split_data[i]["audio"]
            audio_data = audio["array"]
            sampling_rate = audio["sampling_rate"]

            # Convert to mono + resample to 16kHz
            resampled_audio = resample_audio_to_16k(audio_data, sampling_rate)

            # Save to temp WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, resampled_audio, 16000)
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
                print(f"Error processing sample {i}: {e}")
                embedding = []

            features.append(embedding)

            # Clean up temp file
            os.remove(temp_wav_path)

        # Drop the audio column
        split_data = split_data.remove_columns("audio")

        # Add the emotion2vec_features column
        split_data = split_data.add_column("emotion2vec_features", features)

        modified_dataset[split_name] = split_data

    return modified_dataset

def main():

    dataset = load_dataset("cairocode/MSPI_WAV_Diff_Curriculum")
    emotion2vec_model = AutoModel(model="iic/emotion2vec_base")
    modified_dataset = modify_dataset_with_emotion2vec(dataset, emotion2vec_model)
    modified_dataset.push_to_hub("MSPI_Emotion2Vec")


if __name__ == "__main__":
    main()

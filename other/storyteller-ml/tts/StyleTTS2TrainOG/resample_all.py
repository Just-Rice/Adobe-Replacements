import os
import torchaudio

def load_and_resample_audio(file_path, target_sample_rate=24000):
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample to target_sample_rate if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono by averaging the channels if it's not already mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, target_sample_rate

def process_and_overwrite_audio_files(folder_paths, target_sample_rate=24000):
    for folder_path in folder_paths:
        # Iterate through each file in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Check if the path is an actual file to prevent trying to load directories
            if os.path.isfile(file_path):
                try:
                    # Load, resample, and convert to mono
                    waveform, _ = load_and_resample_audio(file_path, target_sample_rate)
                    
                    # Overwrite the original file with the processed audio
                    torchaudio.save(file_path, waveform, target_sample_rate)
                    print(f"Processed and overwritten: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Example usage:
# Replace these with the paths to your folders containing audio files
folder_paths = ['ood', 'wavs','val']
process_and_overwrite_audio_files(folder_paths)
import pathlib
import numpy as np
input_path = pathlib.Path.cwd() / pathlib.Path("..") / pathlib.Path("Vall-E-mount") / pathlib.Path("input")
models_path = pathlib.Path.cwd() / pathlib.Path("..") / pathlib.Path("Vall-E-mount") / pathlib.Path("models") 
presets_path = pathlib.Path.cwd() / pathlib.Path("..") / pathlib.Path("Vall-E-mount") / pathlib.Path("presets")
prompts_path = pathlib.Path.cwd() / pathlib.Path("..") / pathlib.Path("Vall-E-mount") / pathlib.Path("prompts")
temp_path = pathlib.Path.cwd() / pathlib.Path("..") / pathlib.Path("Vall-E-mount") / pathlib.Path("temp")
import soundfile as sf
import torchaudio
import torchaudio.transforms as T
import torch


def stitch_wav_tensors_with_crossfade(wav_tensor1,wav_tensor2,sample_rate:int):
    # Load two audio signals  as NumPy tensors

    # Compute the length of the cross-fade window in samples
    fade_len = int(0.01 * sample_rate)

    # Create a cross-fade window
    fade_window = np.hanning(2*fade_len)

    # Concatenate the signals with cross-fade
    audio_out = np.concatenate((wav_tensor1[:-fade_len], 
                                wav_tensor1[-fade_len:] * fade_window[:fade_len] + wav_tensor2[:fade_len] * fade_window[fade_len:], 
                                wav_tensor2[fade_len:]))

    sf.write("audio_out.wav", audio_out, sample_rate)
    return audio_out


def trim_audio_range(file_path, start_seconds, end_seconds, output_path="range_trimmed_audio.wav"):
    """
    Trims an audio file to a specified range and saves the trimmed audio.

    Args:
    - file_path (str): Path to the audio file to be trimmed.
    - start_seconds (float): Start time of the desired audio segment in seconds.
    - end_seconds (float): End time of the desired audio segment in seconds.
    - output_path (str): Path to save the trimmed audio. Defaults to "range_trimmed_audio.wav".
    
    Returns:
    - None
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert the range to sample indices
    start_sample = int(start_seconds * sample_rate)
    end_sample = int(end_seconds * sample_rate)

    # Trim the audio to the specified range
    trimmed_waveform = waveform[:, start_sample:end_sample]

    # Save the trimmed audio
    torchaudio.save(output_path, trimmed_waveform, sample_rate)
    
    
def cut_audio_range(file_path, start_seconds, end_seconds, output_path="cut_audio.wav"):
    """
    Cuts out a specific segment from an audio file and saves the result.

    Args:
    - file_path (str): Path to the audio file from which the segment will be cut.
    - start_seconds (float): Start time of the segment to be cut out in seconds.
    - end_seconds (float): End time of the segment to be cut out in seconds.
    - output_path (str): Path to save the resulting audio. Defaults to "cut_audio.wav".
    
    Returns:
    - None
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert the range to sample indices
    start_sample = int(start_seconds * sample_rate)
    end_sample = int(end_seconds * sample_rate)

    # Cut out the specified range
    cut_waveform = torch.cat([waveform[:, :start_sample], waveform[:, end_sample:]], dim=1)

    # Save the resulting audio
    torchaudio.save(output_path, cut_waveform, sample_rate)

def test():
    try:
        path = pathlib.Path("/home/tensor/code/TTSDockerContainer/VALL-E-X/../Vall-E-mount/temp/audio_concat_a521f31b-baf3-49a3-aca6-924b5191aad3.wav")
        wav, sr = torchaudio.load(path)
        t = wav[:,int(5 * sr):int(15 * sr)]
        torchaudio.save(pathlib.Path("./testfull.wav"),t,sr)
    except Exception as e:
        print(e)
    
test()
# TTS and RVC Inference Script

## Overview

This Python script runs both Text-To-Speech (TTS) and Retrieval-based-Voice-Conversion (RVC) inferences. It uses Piper for TTS and a custom RVC voice conversion model. The script takes a variety of command-line arguments to specify the models, configurations, and audio files to use.

## Prerequisites

- Python 3.x
- PyTorch 2.0 or higher
- Piper TTS
- ffmpeg (for audio processing)

### Installation

1. Install Python 3.x and PyTorch 2.0 or higher.
2. Install Piper TTS via pip:

    ```bash
    pip install piper-tts
    ```
3. **Download Piper Voice Models**

    Piper voice models can be downloaded from [Hugging Face's Piper Voices Repository](https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0/en/en_US).
   
4. Make sure `ffmpeg` is installed and available in the system's PATH.

    - For Ubuntu:

        ```bash
        sudo apt-get install ffmpeg
        ```

    - For macOS:

        ```bash
        brew install ffmpeg
        ```

## Usage

To run the script, you need to pass in several command-line arguments.

### TTS Arguments

- `--tts_model_path`: Required. The path to the TTS model.
- `--tts_config_path`: Required. The path to the TTS configuration file.
- `--text`: Required. The text you want to convert to speech.

#### Additional TTS Options

- `--length_scale`: Optional. Phoneme length scale for TTS. Default is 1.0.
- `--noise_scale`: Optional. Noise scale for TTS. Default is 0.5.
- `--noise_w`: Optional. Phoneme width noise for TTS. Default is 0.3.
- `--sentence_silence`: Optional. Seconds of silence after each sentence in TTS. Default is 0.0.

### RVC Arguments

- `--model_path`: Path to the VC model.
- `--model_index_path`: (Optional) Path to the VC model index.
- `--hubert_model_path`: (Optional) Path to the Hubert model. Default is `hubert_base.pt`.
- `--input_audio_filename`: Path to the input audio file for VC.
- `--output_audio_filename`: Path to save the output audio file from VC.

... (Other optional arguments for fine-tuning the VC process)

`--f0up_key`: The f0 key for the input audio file.

`--index_path`: Path to the index file.

`--f0method`: F0 estimation method to be used ('pm' 'harvest' 'crepe' or rmvpe).

`--index_rate`: The rate for the index. (Search feature ratio)

`--device`: The computation device to be used ('cuda:0' for GPU or 'cpu' for CPU).

`--is_half`: Boolean value indicating whether to use half precision. Use 'True' for half-precision and 'False' for full-precision.

`--filter_radius`: The radius of the filter. (If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.)

`--resample_sr`: The sample rate for resampling. (Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling)

`--rms_mix_rate`: The RMS mix rate. (Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used)

`--protect`: The protect value. (Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:)


### Example:

```bash
python fakeyou_rvc_tts_infer.py --tts_model_path "path/to/tts/model" --tts_config_path "path/to/tts/config" --text "Hello, World" --model_path "path/to/vc/model" --input_audio_filename "path/to/input/audio.wav" --output_audio_filename "path/to/output/audio.wav"
```

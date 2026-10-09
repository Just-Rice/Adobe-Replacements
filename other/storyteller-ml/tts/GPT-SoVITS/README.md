
# GPT-SoVITS CLI

This repository contains a command-line interface (CLI) script for the GPT-SoVITS project, which allows you to perform text-to-speech (TTS) synthesis using custom models.

## Prerequisites

- Docker

## Setup

### Build the Docker Image

To build the Docker image, run the following command:

```sh
docker build --no-cache -t gpt-sovits:latest .
```

### Run the Docker Container

To run the Docker container, use the following command:

```sh
docker run -it --privileged --gpus all gpt-sovits:latest
```

## Usage

To run the CLI script, use the following command:

```sh
python fakeyou_infer.py [OPTIONS]
```

### Options

- `--use_pretrained_gpt`: Use pretrained GPT model. If enabled, the script will use the default pretrained models for GPT and SoVITS.
- `--use_pretrained_sovits`: Use pretrained SoVITS model. If enabled, the script will use the default pretrained models for GPT and SoVITS.
- `--gpt_model`: Path to GPT model checkpoint. Required if `--use_pretrained_gpt` is not enabled.
- `--sovits_model`: Path to SoVITS model checkpoint. Required if `--use_pretrained_sovits` is not enabled.
- `--ref_audio`: Path to reference wav file. Required.
- `--ref_text`: Path to reference text file (optional). If left blank \`""\` or not passed, it will disable the reference text and use the reference audio.
- `--target_language`: Language of the target text. Choices: `chinese`, `english`, `japanese`, `chinese+english`, `japanese+english`, `automatic`.
- `--ref_language`: Language of the reference text. Choices: `chinese`, `english`, `japanese`, `chinese+english`, `japanese+english`, `automatic`.
- `--target_text`: Path to the target text file. Required.
- `--how_to_cut`: How to cut the text for synthesis. Default: `No slice`. Options: `No slice`, `Slice once every 4 sentences`, `Cut per 50 characters`, `Slice by Chinese punct`, `Slice by English punct`, `Slice by every punct`.
- `--top_k`: Top K sampling. Default: 20.
- `--top_p`: Top P sampling. Default: 0.6.
- `--temperature`: Sampling temperature. Default: 0.6.
- `--speed`: Speed of the synthesized speech. Default: 1.0.
- `--output_path`: Path to save the output wav file. Default: `output`.

### Example Commands

#### Using Custom Models with Reference Audio

```sh
python fakeyou_infer.py --gpt_model GPT_weights/your_gpt_model.ckpt --sovits_model SoVITS_weights/your_sovits_model.pth --ref_audio data/voice/sponge.wav --ref_text data/voice/ref.txt --target_text data/voice/input.txt --target_language english --ref_language english --how_to_cut "Slice once every 4 sentences" --speed 1.0 --output_path output/
```

#### Using Custom Models Without Reference Text

```sh
python fakeyou_infer.py --gpt_model GPT_weights/your_gpt_model.ckpt --sovits_model SoVITS_weights/your_sovits_model.pth --ref_audio data/voice/sponge.wav --target_text data/voice/input.txt --target_language english --ref_language english --how_to_cut "Slice once every 4 sentences" --speed 1.0 --output_path output/
```

#### Using Pretrained Models

```sh
python fakeyou_infer.py --use_pretrained_gpt --use_pretrained_sovits --ref_audio data/voice/sponge.wav --ref_text data/voice/ref.txt --target_text data/voice/input.txt --target_language english --ref_language english --how_to_cut "Slice once every 4 sentences" --speed 1.0 --output_path output/
```

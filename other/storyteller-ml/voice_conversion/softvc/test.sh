#!/bin/bash
source python/bin/activate && ./cli_inference.py \
  --acoustic_model_filename /home/bt/models/voice-conversion/trump-acoustic.pt \
  --hifigan_model_filename /home/bt/models/voice-conversion/trump-hifigan.pt \
  --source_audio_filename samples/source/medium/gongiveittoya.wav \
  --output_audio_filename output.wav \
  --truncate_seconds 5


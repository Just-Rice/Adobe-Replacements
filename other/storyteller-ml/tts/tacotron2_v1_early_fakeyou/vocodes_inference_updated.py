#!/usr/bin/env python

"""
Vocodes Tacotron Inference Engine
This is a work in progress and is likely going to be slow for now.
"""

# ========== ARGUMENTS (FAIL EARLY IF ABSENT) ==========

import argparse

parser = argparse.ArgumentParser(description='Run TTS inference')

# Model parameters
parser.add_argument('--synthesizer_checkpoint_path', type=str, help='path the TTS synthesizer model', required=True)
parser.add_argument('--text_pipeline_type', type=str, help='', required=True)
parser.add_argument('--vocoder_type', type=str, help='', required=True)
parser.add_argument('--waveglow_vocoder_checkpoint_path', type=str, help='path the TTS vocoder model')
parser.add_argument('--hifigan_vocoder_checkpoint_path', type=str, help='path the TTS vocoder model')
parser.add_argument('--hifigan_superres_vocoder_checkpoint_path', type=str, help='path the TTS vocoder model')

# Optional mel scaling before vocoding
parser.add_argument('--use_default_mel_multiply_factor', help='', action='store_true')
parser.add_argument('--custom_mel_multiply_factor', type=float, help='')

# Premium features
parser.add_argument('--max_decoder_steps', type=int, help='')

# User input 
parser.add_argument('--input_text_filename', type=str, help='path the file containing text to run', required=True)

# Output files
parser.add_argument('--output_audio_filename', type=str, help='where to save result audio', required=True)
parser.add_argument('--output_spectrogram_filename', type=str, help='where to save result spectrogram', required=True)
parser.add_argument('--output_metadata_filename', type=str, help='where to save extra metadata', required=True)

args = parser.parse_args()


# ========== IMPORTS ==========

import sys
sys.path.append('waveglow/')
import shutil
import json

import numpy as np
import torch

#from hparams import create_hparams
#from model import Tacotron2
#from text import text_to_sequence
#from denoiser import Denoiser

from vocodes_common import FakeYouTt2Pipeline
from vocodes_common import print_gpu_info
from vocodes_common import parse_int_or_default

# For metadata
import subprocess
import magic
import os

from logger import LOGGER


# ========== PRINT GPU INFO ==========

import tensorflow as tf
print("TensorFlow version: {}".format(tf.version.VERSION))
print_gpu_info()


# ========== LOAD MODEL + SETUP ==========

pipeline = FakeYouTt2Pipeline()

# NB: Not sure if there was a historical decision to load the models first outside of the pipeline. 
# Probably not, though. Seems like this could be cleaned up.
if args.vocoder_type == 'hifigan-superres':
    pipeline.maybe_load_hifigan(args.hifigan_vocoder_checkpoint_path)
    pipeline.maybe_load_hifigan_super_resolution(args.hifigan_superres_vocoder_checkpoint_path)
elif args.vocoder_type == 'waveglow':
    pipeline.maybe_load_waveglow_model(args.waveglow_vocoder_checkpoint_path)
else:
    raise Exception('Wrong vocoder type: {}'.format(args.vocoder_type))


pipeline.maybe_load_tacotron_model_to_cache(args.synthesizer_checkpoint_path)

# NB: 1000 is the default and is roughly 12 seconds.
max_decoder_steps = parse_int_or_default(args.max_decoder_steps, 1000) 

input_text = open(args.input_text_filename, 'r').read().strip()

inference_args = {
    'raw_text': input_text,
    'use_default_mel_multiply_factor': args.use_default_mel_multiply_factor,
    'maybe_custom_mel_multiply_factor': args.custom_mel_multiply_factor,
    'synthesizer_checkpoint_path': args.synthesizer_checkpoint_path,
    'vocoder_type': args.vocoder_type,
    'text_pipeline_type': args.text_pipeline_type,
    'output_audio_filename': args.output_audio_filename,
    'output_spectrogram_filename': args.output_spectrogram_filename,
    'output_metadata_filename': args.output_metadata_filename,
    'max_decoder_steps': max_decoder_steps,
}

LOGGER.info("running inference with args: {}".format(str(inference_args)))


# ========== INFERENCE ==========

pipeline.infer(inference_args)


#!/usr/bin/env python

"""
Vocodes Tacotron Model Check
"""

import argparse

parser = argparse.ArgumentParser(description='Check that the model is valid')

parser.add_argument('--synthesizer_checkpoint_path', type=str, help='path the TTS synthesizer model', required=True)
parser.add_argument('--output_metadata_filename', type=str, help='where to save extra metadata', required=True)

args = parser.parse_args()

import sys
sys.path.append('waveglow/')
import json

import torch

from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2

# For metadata
import os

print('========================================')
print('Python interpreter', sys.executable)
print('PyTorch version', torch.__version__)
print('CUDA Available?', torch.cuda.is_available())
print('CUDA Device count', torch.cuda.device_count())
print('========================================', flush=True)

# NB(bt, 2021-05-31): Trying to get everything on the same device
#torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = create_hparams()
hparams.sampling_rate = 22050 # Don't change this
hparams.max_decoder_steps = 1000 # How long the audio will be before it cuts off (1000 is about 11 seconds)
hparams.gate_threshold = 0.1 # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)

# Load synthesizer
checkpoint_path = args.synthesizer_checkpoint_path

#model = load_model(hparams)
model = Tacotron2(hparams)

model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=True)

_ = model.cuda().eval().half()

print('Model is valid')

file_size_bytes = os.path.getsize(args.synthesizer_checkpoint_path)

metadata = {
    'file_size_bytes': file_size_bytes,
}

with open(args.output_metadata_filename, 'w') as json_file:
    json.dump(metadata, json_file)

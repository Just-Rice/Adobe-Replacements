#!/usr/bin/env python

"""
Vocodes HiFi-GAN Model Check
"""

import argparse
import json
import os
import sys

# ML
import torch

# Hifi-Gan
from hifigan.env import AttrDict
from hifigan.models import Generator
from hifigan.denoiser import Denoiser as HifiDenoiser

from utils import print_title

print_title('Check Hifigan Model (Tacotron)')

parser = argparse.ArgumentParser(description='Check that the HiFi-GAN model is valid')

parser.add_argument('--checkpoint_path', type=str, help='path the hifigan model', required=True)
parser.add_argument('--output_metadata_filename', type=str, help='where to save extra metadata', required=True)

print('========================================')
print('Python interpreter', sys.executable)
print('PyTorch version', torch.__version__)
print('CUDA Available?', torch.cuda.is_available())
print('CUDA Device count', torch.cuda.device_count())
print('========================================', flush=True)

def load_hifigan_model(checkpoint_path):
    conf = 'hifigan/config_v1.json'
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cuda"))
    state_dict_g = torch.load(checkpoint_path, map_location=torch.device("cuda"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = HifiDenoiser(hifigan, mode="normal")
    return hifigan, h, denoiser

args = parser.parse_args()

print('Checking model validity...', flush=True)

# Check that the model loads
hifigan, h, denoiser = load_hifigan_model(args.checkpoint_path)

_ = hifigan.cuda().eval().half()
_ = denoiser.cuda().eval().half()

print('Model is valid', flush=True)

file_size_bytes = os.path.getsize(args.checkpoint_path)

metadata = {
    'file_size_bytes': file_size_bytes,
}

with open(args.output_metadata_filename, 'w') as json_file:
    json.dump(metadata, json_file)

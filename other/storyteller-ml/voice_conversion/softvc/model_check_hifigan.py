#!/usr/bin/env python

"""
HiFi-GAN (SoftVC) Model Check
"""

import argparse
import json
import os
import sys

import importlib.util

# ML
import torch

# Hifi-Gan
sys.path.append("./hifigan/")
from hifigan.generator import HifiganGenerator
from hifigan.generator import consume_prefix_in_state_dict_if_present

from utils import print_title

print_title('Check Hifigan Model (SoftVC)')

from gpu_utils import print_gpu_info

print_gpu_info()

parser = argparse.ArgumentParser(description='Check that the HiFi-GAN model is valid')

parser.add_argument('--checkpoint_path', type=str, help='path the hifigan (softvc) model', required=True)
parser.add_argument('--output_metadata_filename', type=str, help='where to save extra metadata', required=True)


def load_hifigan_model(hifigan_checkpoint_path):
    # NB(bt, 2022-12-07): Not sure why this "spec loading" is needed.
    spec = importlib.util.spec_from_file_location("hifigan.utils", "./hifigan/hifigan/utils.py")
    u = importlib.util.module_from_spec(spec)
    sys.modules["hifigan.utils"] = u
    spec.loader.exec_module(u)

    spec = importlib.util.spec_from_file_location("hifigan.generator", "./hifigan/hifigan/generator.py")
    g = importlib.util.module_from_spec(spec)
    sys.modules["hifigan.generator"] = g
    spec.loader.exec_module(g)

    checkpoint = torch.load(hifigan_checkpoint_path)["generator"]["model"]
    hifigan = g.HifiganGenerator()
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    hifigan.load_state_dict(checkpoint)
    hifigan.remove_weight_norm()
    hifigan.eval()

    hifigan.cuda()
    return hifigan


args = parser.parse_args()

print('Checking model validity...', flush=True)

# Check that the model loads
_ = load_hifigan_model(args.checkpoint_path)

print('Model is valid', flush=True)

file_size_bytes = os.path.getsize(args.checkpoint_path)

metadata = {
    'file_size_bytes': file_size_bytes,
}

with open(args.output_metadata_filename, 'w') as json_file:
    json.dump(metadata, json_file)

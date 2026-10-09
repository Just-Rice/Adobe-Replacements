#!/usr/bin/env python

import IPython.display as display
import argparse
import importlib.util
import json
import magic
import os
import soundfile
import subprocess
import sys
import sys  
import torch
import torchaudio

from gpu_utils import print_gpu_info

print_gpu_info()

from collections import OrderedDict
from time import time
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from typing import Dict, Any

sys.path.append("./acoustic-model/")
from acoustic import AcousticModel

#sys.path.insert(0, './acoustic-model/acoustic/')
#from model import AcousticModel

sys.path.append("./hifigan/")
from hifigan.generator import HifiganGenerator
from hifigan.generator import consume_prefix_in_state_dict_if_present


# ===========================================

parser = argparse.ArgumentParser()
parser.add_argument('--acoustic_model_filename', type=str, required=True)
parser.add_argument('--hifigan_model_filename', type=str, required=True)
parser.add_argument('--source_audio_filename', type=str, required=True)
parser.add_argument('--output_audio_filename', type=str, default='output.wav')
parser.add_argument('--temporary_audio_filename', type=str, default='temp.wav') # used for truncation, etc.
parser.add_argument('--output_metadata_filename', type=str, default='metadata.json')
parser.add_argument('--truncate_seconds', type=int, default=-1) # -1 is "do not truncate", 0 is invalid
args = parser.parse_args()

print('\n'.join(f'[arg] {k} = {v}' for k, v in vars(args).items()))

# ===========================================

def load_acoustic_model(checkpoint_path):

    def modify_model(state_dict: Dict[str,Any]):
        new_state_dict = OrderedDict()
        for k in state_dict['acoustic-model'].keys():
            value = state_dict['acoustic-model'][k]
            name = k[7:] # remove `module.` because they used ddp training to save
            new_state_dict[name] = value
        return new_state_dict

    # Load the acoustic model (either hubert_soft or hubert_discrete) 
    acoustic_dict = torch.load(checkpoint_path) # this used ddp
    acoustic = AcousticModel().cuda()
    acoustic.eval()

    new_state_dict = modify_model(state_dict=acoustic_dict)
    acoustic.load_state_dict(new_state_dict)
    return acoustic

# ===========================================

def load_hifigan_model(hifigan_checkpoint_path=None):
    # NB(bt, 2022-12-07): Not sure why this "spec loading" is needed.
    spec = importlib.util.spec_from_file_location("hifigan.utils", "./hifigan/hifigan/utils.py")
    u = importlib.util.module_from_spec(spec)
    sys.modules["hifigan.utils"] = u
    spec.loader.exec_module(u)

    spec = importlib.util.spec_from_file_location("hifigan.generator", "./hifigan/hifigan/generator.py")
    g = importlib.util.module_from_spec(spec)
    sys.modules["hifigan.generator"] = g
    spec.loader.exec_module(g)

    if hifigan_checkpoint_path:
        checkpoint = torch.load(hifigan_checkpoint_path)["generator"]["model"]
        hifigan = g.HifiganGenerator()
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        hifigan.load_state_dict(checkpoint)
        hifigan.remove_weight_norm()
        hifigan.eval()
    else:
        hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft")

    hifigan.cuda()
    return hifigan

# ===========================================


def load_wav_data(file_name):
    source, sr = torchaudio.load(f"{file_name}")
    source = torchaudio.functional.resample(source, sr, 16000)
    source = torch.mean(source, dim=0).unsqueeze(0)
    source = source.unsqueeze(0).cuda()
    return source

# ===========================================

def do_voice_conversion(hubert, acoustic, hifigan, audio_source):
    #@title Run the inference and display the result
    # Convert to the target speaker
    with torch.inference_mode():
        # Extract speech units
        start = time()
        units = hubert.units(audio_source)
        end = time()
        # Generate target spectrogram
        print(f"time for feature embedding:{end-start}")

        start = time()
        mel = acoustic.generate(units).transpose(1, 2)
        end = time()
        print(f"time for inference:{end-start}")
        # Generate audio waveform
        start = time()
        target = hifigan(mel)
        end = time()
        print(f"time for wavform {end-start}")
        return target

# ===========================================

def truncate_source_audio(audio_source_filename, output_filename, seconds):
    # Run ffmpeg's ffprobe
    command = [
        "ffmpeg",
        "-y", # Force file overwrite
        "-i",  
        audio_source_filename,
        "-t", 
        str(seconds),
        "-c",
        "copy",
        output_filename,
    ]

    print(f"Truncating source audio to {seconds} seconds", flush=True)
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = pipe.communicate()


def save_output_audio(audio_data, output_audio_filename):
  soundfile.write(args.output_audio_filename, audio_data.squeeze().cpu(), samplerate=16000)

def save_metadata(audio_filename, output_metadata_filename):
    # Run ffmpeg's ffprobe
    command = [
        "ffprobe",
        "-loglevel",  "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        audio_filename
    ]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = pipe.communicate()
    ffmpeg_metadata = json.loads(out)

    # Collect metadata
    print(ffmpeg_metadata)
    mime_type = magic.from_file(audio_filename, mime=True)
    file_size_bytes = os.path.getsize(audio_filename)

    duration_millis = int(float(ffmpeg_metadata['format']['duration']) * 1000)

    metadata = {
        'duration_millis': duration_millis,
        'mimetype': mime_type,
        'file_size_bytes': file_size_bytes,
    }

    with open(output_metadata_filename, 'w') as json_file:
        json.dump(metadata, json_file)

intermediate_audio_filename = args.source_audio_filename

if args.truncate_seconds not in [-1, 0]:
    intermediate_audio_filename = args.temporary_audio_filename
    truncate_source_audio(args.source_audio_filename, intermediate_audio_filename, args.truncate_seconds)

hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()

acoustic = load_acoustic_model(args.acoustic_model_filename)
hifigan = load_hifigan_model(args.hifigan_model_filename)

source_audio = load_wav_data(intermediate_audio_filename)
target_audio = do_voice_conversion(hubert, acoustic, hifigan, source_audio)

save_output_audio(target_audio, args.output_audio_filename)
save_metadata(args.output_audio_filename, args.output_metadata_filename)


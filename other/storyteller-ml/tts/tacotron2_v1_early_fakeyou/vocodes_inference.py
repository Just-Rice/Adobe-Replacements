#!/usr/bin/env python

"""
Vocodes Tacotron Inference Engine
This is a work in progress and is likely going to be slow for now.
"""

import argparse

parser = argparse.ArgumentParser(description='Run TTS inference')

parser.add_argument('--synthesizer_checkpoint_path', type=str, help='path the TTS synthesizer model', required=True)
parser.add_argument('--vocoder_checkpoint_path', type=str, help='path the TTS vocoder model', required=True)
parser.add_argument('--input_text_filename', type=str, help='path the file containing text to run', required=True)
parser.add_argument('--output_audio_filename', type=str, help='where to save result audio', required=True)
parser.add_argument('--output_spectrogram_filename', type=str, help='where to save result spectrogram', required=True)
parser.add_argument('--output_metadata_filename', type=str, help='where to save extra metadata', required=True)

args = parser.parse_args()

import sys
sys.path.append('waveglow/')
import shutil
import json

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from denoiser import Denoiser

# For metadata
import subprocess
import magic
import os

print('========================================')
print('Python interpreter', sys.executable)
print('PyTorch version', torch.__version__)
print('CUDA Available?', torch.cuda.is_available())
print('CUDA Device count', torch.cuda.device_count())
print('========================================', flush=True)

# NB(bt, 2021-05-31): Trying to get everything on the same device
#torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix bug running on CPU?
#torch.set_default_dtype(torch.float16)
# NB(bt, 2021-05-31): Perhaps this is blocking waveglow? It complains of the wrong tensor type.
#torch.set_default_tensor_type('torch.DoubleTensor') # 2021-05-31 commented out
#torch.set_default_tensor_type('torch.HalfTensor') # 2021-05-31 added

hparams = create_hparams()
hparams.sampling_rate = 22050 # Don't change this
hparams.max_decoder_steps = 1000 # How long the audio will be before it cuts off (1000 is about 11 seconds)
hparams.gate_threshold = 0.1 # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)

# Load synthesizer
checkpoint_path = args.synthesizer_checkpoint_path

#model = load_model(hparams)
model = Tacotron2(hparams)

print('Loading synthesizer at path: {}'.format(checkpoint_path))
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()


# NB(bt, 2021-05-31): Trying to get everything on the same device
device = "cuda:0"
model = model.to(device)
#_ = model.cuda().eval()

# Load vocoder
waveglow_path = args.vocoder_checkpoint_path

#for k in waveglow.convinv:
#    k.float()
#denoiser = Denoiser(waveglow)


waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

text = open(args.input_text_filename, 'r').read()

"""
seq = text_to_sequence(text, ['english_cleaners'])

sequence = np.array(seq)[None, :]

print("Sequence np.array: {}".format(type(sequence)))
print("Sequence np.array: {}".format(sequence))

sequence = torch.from_numpy(sequence)

print("Sequence from_numpy: {}".format(type(sequence)))
print("Sequence from_numpy: {}".format(sequence))
print("Sequence from_numpy dtype: {}".format(sequence.dtype))

sequence = torch.autograd.Variable(sequence).long()

print("Sequence torch autograd: {}".format(sequence))

# NB(bt, 2021-05-31): Trying to get everything to the same device
sequence = sequence.to(device)
"""



raw_input = True
sequence = None

for i in text.split("\n"):
    if len(i) < 1:
        continue
    print(i)
    if raw_input:
        if i[-1] != ";":
            i = i+";"
    else:
        #i = ARPA(i)
        pass
    print(i)
    with torch.no_grad():
        # save VRAM by not including gradients
        sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        break # TODO: Handle multiple lines.

with torch.no_grad():
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

#print("Mel outputs:")
#print(mel_outputs)
#print(mel_outputs.shape)
#print(mel_outputs_postnet)
#print(mel_outputs_postnet.shape)

print('Saving mel as JSON...')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# squeeze(0) -> 3D to 2D (by removing the singular 1-length "wrapper" dimension)
# transpose() -> originally 80x{N}, we turn to {N}x80
mel_for_json = mel_outputs_postnet.cpu().numpy().squeeze(0).transpose()
mel_for_scaling = mel_for_json.copy()

max_value = np.amax(mel_for_scaling)
min_value = np.amin(mel_for_scaling)
mel_range = max_value - min_value

mel_for_scaling -= min_value
mel_for_scaling *= (255.0 / mel_range)
mel_for_scaling = mel_for_scaling.astype('int32')

json_data = {
    'mel': mel_for_json,
    'mel_scaled': mel_for_scaling,
}

with open(args.output_spectrogram_filename, 'w') as outfile:
    json.dump(json_data, outfile, cls=NumpyEncoder)

print('Encoding and saving audio...')

with torch.no_grad():
    sigma = 0.8
    audio = waveglow.infer(mel_outputs_postnet, sigma=sigma)

from scipy.io.wavfile import write

output_audio = audio[0].data.cpu().numpy().astype(np.float32)

write(args.output_audio_filename, hparams.sampling_rate, output_audio)

"""
try:
    debug_location = "/home/bt/dev/storyteller/storyteller-web/test"
    print('Writing to debug location: {}'.format(debug_location))
    shutil.copy(args.output_audio_filename, debug_location)

except Exception as e:
    print("Error with saving file for local debugging.")
    print(e)
    pass
"""

# ==== METADATA (1) ====

command = [
    "ffprobe",
    "-loglevel",  "quiet",
    "-print_format", "json",
    "-show_format",
    "-show_streams",
    args.output_audio_filename
]

pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
out, err = pipe.communicate()
ffmpeg_metadata = json.loads(out)

# ==== METADATA (2) ====
mime_type = magic.from_file(args.output_audio_filename, mime=True)
file_size_bytes = os.path.getsize(args.output_audio_filename)

print(ffmpeg_metadata)

duration_millis = int(float(ffmpeg_metadata['format']['duration']) * 1000)

metadata = {
    'duration_millis': duration_millis,
    'mimetype': mime_type,
    'file_size_bytes': file_size_bytes,
}

with open(args.output_metadata_filename, 'w') as json_file:
    json.dump(metadata, json_file)


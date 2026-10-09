import os
import json
import argparse

# Adjust NLTK paths
nltk_cache = os.environ.get('NLTK_DATA', None)
if nltk_cache:
    from pathlib import Path
    Path(nltk_cache).mkdir(parents=True, exist_ok=True)
    import nltk
    nltk.data.path = [nltk_cache]

#import nltk
#print(nltk.data.path)

import torch
import numpy as np
import soundfile as sf
from tsvitsfe import TSVITSFE

# For metadata
import subprocess
import magic
import sys

def print_gpu_info():
    print('========================================')
    print('Python interpreter', sys.executable)
    print('PyTorch version', torch.__version__)
    print('CUDA Available?', torch.cuda.is_available())
    print('CUDA Device count', torch.cuda.device_count())
    print('CUDA architectures library was compiled for', torch.cuda.get_arch_list())
    #try:
    #    from tensorflow.python.client import device_lib
    #    print('local devices', str(device_lib.list_local_devices()).replace("\n", "\n  "))
    #except ImportError:
    #    print('no tensorflow - cannot list devices')
    #    pass
    print('========================================', flush=True)

def generate_metadata_file(output_audio_filename, output_metadata_filename):
    # ==== METADATA (1) ====
    command = [
        "ffprobe",
        "-loglevel",  "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        output_audio_filename
    ]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = pipe.communicate()
    ffmpeg_metadata = json.loads(out)

    print(ffmpeg_metadata)

    # ==== METADATA (2) ====
    mime_type = magic.from_file(output_audio_filename, mime=True)
    file_size_bytes = os.path.getsize(output_audio_filename)

    duration_millis = int(float(ffmpeg_metadata['format']['duration']) * 1000)

    metadata = {
        'duration_millis': duration_millis,
        'mimetype': mime_type,
        'file_size_bytes': file_size_bytes,
    }

    with open(output_metadata_filename, 'w') as json_file:
        json.dump(metadata, json_file)


# NB: This is a copy of `infer_ts.py` meant for handling requests from the job runner.
# Inference text is loaded from the filesystem to protect against attack surface area and carry larger payloads.
# Additionally, we save metadata about the generated output.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to TorchScript-exported model file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="device. cuda or cpu",
    )

    parser.add_argument(
        "--input-text-filename",
        type=str,
        default=None,
        help="filename containing raw text to synethesize",
    )

    parser.add_argument(
        "--output-audio-filename",
        type=str,
        required=True,
        help="Path to store files",
    )
    parser.add_argument(
        "--output-metadata-filename",
        type=str,
        default=None,
        help="filename where we save metadata",
    )

    #parser.add_argument(
    #    "--output-spectrogram-filename",
    #    type=str,
    #    default=None,
    #    help="filename where we save the spectrogram",
    #)

    args = parser.parse_args()

    print_gpu_info()

    input_text = open(args.input_text_filename, 'r').read().strip()
    print(f'Input text: {input_text}')

    vits_fe = TSVITSFE()
    print("Loading model...")
    vits_fe.load(args.checkpoint,args.config,args.device)
    print("Doing inference...")
    out_aud = vits_fe.infer(input_text)

    out_fn = args.output_audio_filename
    sf.write(out_fn, out_aud, vits_fe.hps.data.sampling_rate)
    print(f"Wrote audio file {out_fn}")

    generate_metadata_file(out_fn, args.output_metadata_filename)


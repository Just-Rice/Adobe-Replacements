# FakeYou StyleTTS 2 Inference

import os
import torch
import argparse
import numpy as np
import soundfile as sf
from styletts2importable import compute_style, inference
from tortoise.utils.text import split_and_recombine_text

def clsynthesize(text_file_path, custom_voice, vcsteps):
    # Load the text from the file
    with open(text_file_path, 'r', encoding="utf-8") as file:
        text = file.read()

    texts = split_and_recombine_text(text)
    audios = [inference(t, custom_voice, alpha=0.3, beta=0.7, diffusion_steps=vcsteps, embedding_scale=1) for t in texts]

    return 24000, np.concatenate(audios)

def main():
    parser = argparse.ArgumentParser(description='CLI for StyleTTS Voice Synthesis')
    parser.add_argument('--input', type=str, required=True, help='Path to the text file')
    parser.add_argument('--voice', type=str, help='Path to the custom voice audio file (optional if using --input-style-npz)')
    parser.add_argument('--vcsteps', type=int, required=True, help='Number of diffusion steps')
    parser.add_argument('--output', type=str, help='Output path for the synthesized audio file (optional)')
    parser.add_argument('--input-style-npz', type=str, help='Path to the input style vector NPZ file')
    parser.add_argument('--output-style-npz', type=str, help='Output path for the style vector NPZ file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    custom_voice = None
    if args.input_style_npz and os.path.exists(args.input_style_npz):
        # Load style vector from NPZ file
        style_vector_np = np.load(args.input_style_npz)['style_vector']
        custom_voice = torch.from_numpy(style_vector_np).float().to(device)
    elif args.voice:
        # Compute and save style vector
        npz_output_path = args.output_style_npz or "default_output_path.npz"
        custom_voice = compute_style(args.voice, save_npz=True, npz_path=npz_output_path).to(device)

    # Checking if the custom_voice tensor is initialized
    if custom_voice is None:
        raise ValueError("Unable to load or compute style vector")

    # Perform synthesis and save output if the output path is provided
    if args.output:
        sample_rate, audio = clsynthesize(args.input, custom_voice, args.vcsteps)
        sf.write(args.output, audio, sample_rate)

if __name__ == "__main__":
    main()

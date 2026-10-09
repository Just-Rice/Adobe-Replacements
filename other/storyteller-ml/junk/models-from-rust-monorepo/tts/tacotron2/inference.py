#!/usr/bin/env python

# 1
#import matplotlib
#import matplotlib.pylab as plt

#import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
#from denoiser import Denoiser

print('========================================')
print('Python interpreter', sys.executable)
print('PyTorch version', torch.__version__)
print('CUDA Available?', torch.cuda.is_available())
print('CUDA Device count', torch.cuda.device_count())
print('========================================', flush=True)

# NB(bt, 2021-05-31): Trying to get everything on the same device
torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NB(bt, 2021-05-31): This is just a utility for testing mel output
# The outputs look good so far...
#from vocodes_wav_images import render_histogram

# 2
def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')

# 3
hparams = create_hparams()
hparams.sampling_rate = 22050

# Fix bug running on CPU?
#torch.set_default_dtype(torch.float16)
# NB(bt, 2021-05-31): Perhaps this is blocking waveglow? It complains of the wrong tensor type.
#torch.set_default_tensor_type('torch.DoubleTensor') # 2021-05-31 commented out
torch.set_default_tensor_type('torch.HalfTensor') # 2021-05-31 added

# 4
checkpoint_path = "/home/bt/models/tacotron2-nvidia/tacotron2_statedict.pt"
#checkpoint_path = "/home/bt/models/tigger_tt_model.pt"
#checkpoint_path = "/home/bt/models/uber_test_2.pt"
checkpoint_path = "/home/bt/models/tacotron2/tacotron2_uberduck_JorgenVonStrangle"
checkpoint_path = "/home/bt/models/tacotron2/tacotron2_uberduck_Noire.pt"
model = load_model(hparams)
#model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
#model.load_state_dict(torch.jit.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
#_ = model.cuda().eval().half()
#_ = model.eval().half()


# NB(bt, 2021-05-31): Trying to get everything on the same device
device = "cuda:0"
model = model.to(device)


_ = model.cuda().eval()

# 5
#waveglow_path = '/home/bt/models/waveglow_256channels.pt'
waveglow_path = '/home/bt/models/waveglow/waveglow_256channels_universal_v5.pt'
waveglow_path = '/home/bt/models/waveglow/waveglow_256channels_v4_uberduck.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
#for k in waveglow.convinv:
#    k.float()
#denoiser = Denoiser(waveglow)

# 6
print("\n\n===TEXT===\n")
text = "This is the song that never ends. It goes on and on my friend."
#text = "hi you"
print("Text: {}".format(text))
print("Text len: {}".format(len(text)))
seq = text_to_sequence(text, ['english_cleaners'])
print("Seq: {}".format(type(seq)))
print("Seq: {}".format(seq))
sequence = np.array(seq)[None, :]
print("Sequence np.array: {}".format(type(sequence)))
print("Sequence np.array: {}".format(sequence))
sequence = torch.from_numpy(sequence)
print("Sequence from_numpy: {}".format(type(sequence)))
print("Sequence from_numpy: {}".format(sequence))
print("Sequence from_numpy dtype: {}".format(sequence.dtype))
sequence = torch.autograd.Variable(sequence).long()
    #torch.from_numpy(sequence)).cuda().long()
    #torch.from_numpy(sequence))

print("Sequence torch autograd: {}".format(sequence))

# NB(bt, 2021-05-31): This works. Don't need to do it.
#print('Saving text sequence tensor')
#torch.save(sequence, 'text_sequence.pt')

print("\n\n===END TEXT===\n")


# NB(bt, 2021-05-31): Trying to get everything to the same device
sequence = sequence.to(device)

# 7
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
#plot_data((mel_outputs.float().data.cpu().numpy()[0],
#           mel_outputs_postnet.float().data.cpu().numpy()[0],
#           alignments.float().data.cpu().numpy()[0].T))

# Per melgan preprocess.py, need to output:
# mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
print("Mel outputs:")
print(mel_outputs)
print(mel_outputs.shape)
print(mel_outputs_postnet)
print(mel_outputs_postnet.shape)

print('Saving mels')
torch.save(mel_outputs, 'mel_outputs.mel')
torch.save(mel_outputs_postnet, 'mel_outputs_postnet.mel')

#print('Rendering histograms')
#render_histogram(mel_outputs, 'mel_outputs.png')
#render_histogram(mel_outputs_postnet, 'mel_outputs_postnet.png')

# 8
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    pass


#ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

# https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050
from scipy.io.wavfile import write
write("audio.wav", rate, audio_numpy)


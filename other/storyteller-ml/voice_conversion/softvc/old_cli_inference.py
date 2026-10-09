#!/usr/bin/env python

import IPython.display as display
import torch, torchaudio
import torch.nn as nn
import sys  
import soundfile as sf

from time import time
import torch.jit
sys.path.insert(0, './hubert/hubert')
from model import HubertSoft

sys.path.insert(0, './acoustic-model/')
from acoustic.model import AcousticModel

# ===============================


#@title select input wav
#file_name = "/content/drive/MyDrive/VCModels/Source/brandon1.wav"#@param { type: "string"}
file_name = "samples/source/easy/brandon1.wav"

source, sr = torchaudio.load(f"{file_name}")
source = torchaudio.functional.resample(source, sr, 16000)
source = source.unsqueeze(0).cuda()
print(f"Wav Shape and Type {source.shape} {source.dtype}")
if source.shape[1] == 2: # remove channel
    source = torch.mean(source, dim=0).unsqueeze(0)
    print(f"Removing Channel to mono Wav Shape and Type {source.shape} {source.dtype}")

display.Audio(source.squeeze().cpu(), rate=16000)


# ===============================

import datetime
from utils_acoustic import AcousticModelScripted
from utils_hubert import modify_model

hubert_soft = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()
hubert_soft.eval()

script_acoustic = True #@param {type: "boolean"}
print(script_acoustic)

# Load the acoustic model (either hubert_soft or hubert_discrete) 
#input_acoustic_model_file = "/content/drive/MyDrive/VCModels/Padme/model-24000.pt" #@param {type: "string"}
input_acoustic_model_file = "/home/bt/models/voice-conversion/trump-acoustic.pt"
acoustic_dict = torch.load(input_acoustic_model_file) # this used ddp ## REPLACE THIS! <-- !
if script_acoustic == True:
    acoustic = AcousticModelScripted().cuda()
else:
    acoustic = AcousticModel().cuda()

acoustic.eval()

new_state_dict = modify_model(state_dict=acoustic_dict)
acoustic.load_state_dict(new_state_dict)

import importlib.util
import sys

spec = importlib.util.spec_from_file_location("hifigan.utils", "/content/hifigan/hifigan/utils.py")
u = importlib.util.module_from_spec(spec)
sys.modules["hifigan.utils"] = u
spec.loader.exec_module(u)

spec = importlib.util.spec_from_file_location("hifigan.generator", "/content/hifigan/hifigan/generator.py")
g = importlib.util.module_from_spec(spec)
sys.modules["hifigan.generator"] = g
spec.loader.exec_module(g)

new_state_dict = modify_model(state_dict=acoustic_dict)
acoustic.load_state_dict(new_state_dict)
input_hifigan_checkpoint_path = "/content/drive/MyDrive/VCModels/Hifigan/Padme.pt" #@param {type: "string"}
if input_hifigan_checkpoint_path:
  checkpoint = torch.load(input_hifigan_checkpoint_path)["generator"]["model"]
  hifigan = g.HifiganGenerator()
  consume_prefix_in_state_dict_if_present(checkpoint, "module.")
  hifigan.load_state_dict(checkpoint)
  hifigan.remove_weight_norm()
else:
  hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft")

hifigan.cuda()
hifigan.eval()

#utils.remove_weight_norm(hubert_soft.positional_embedding.conv)
# Convert to the target speaker
units = None
mel = None 

unit_start_time = 0.0
unit_end_time = 0.0

acoustic_start_time = 0.0
acoustic_end_time = 0.0 

hifi_start_time = 0.0
hifi_end_time = 0.0

with torch.inference_mode():
    unit_start_time = datetime.datetime.now()
    units = hubert_soft.units(source)
    print(f"Units and Type {units.shape} {units.dtype}")
    unit_end_time = datetime.datetime.now()

    hubert_soft_script = torch.jit.trace_module(hubert_soft,{"units":source})
    torch.jit.save(hubert_soft_script, 'hubert_soft.jit')

    print(f"Unit Generation Time {(unit_end_time - unit_start_time).total_seconds() * 1000}")

    print("Hubert Soft Exported")

    acoustic_start_time = datetime.datetime.now()
    mel = acoustic.generate(units).transpose(1, 2)
    print(f"Mel and Type {mel.shape} {mel.dtype}")
    acoustic_end_time = datetime.datetime.now()
    print(f"Acoustic Time {(acoustic_end_time - acoustic_start_time).total_seconds() * 1000}")
    if script_acoustic == True:
        output_acoustic_scripted = '/content/drive/MyDrive/VCModels/FinishedModels/Padme.recast' #@param {type: "string"}
        print("Exporting Scripted")
        acoustic_model_script = torch.jit.script(acoustic)
        torch.jit.save(acoustic_model_script, output_acoustic_scripted)
    else:
        print("Exporting Traced")
        acoustic_model_script = torch.jit.trace_module(acoustic,{"generate":units})
        output_acoustic_traced = '' #@param {type: "string"}
        torch.jit.save(acoustic_model_script, output_acoustic_traced)

    print("Acoustic Model Exported")
    hifi_start_time = datetime.datetime.now()

    target = hifigan(mel)
    print(f"Wavform and Type {target.shape} {target.dtype}")
    hifi_end_time = datetime.datetime.now()
    print(f"MEL {target.shape}")
    print(f"Hifi Gan Time {(hifi_end_time - hifi_start_time).total_seconds() * 1000}")

    hifi_gan_script = torch.jit.trace_module(hifigan,{"forward":mel})
    output_hifigan_jit = '/content/drive/MyDrive/VCModels/FinishedModels/Padme-hifigan-jit.pt' #@param {type: "string"}
    torch.jit.save(hifi_gan_script, output_hifigan_jit)
    print("Hifi Gan Model Exported")


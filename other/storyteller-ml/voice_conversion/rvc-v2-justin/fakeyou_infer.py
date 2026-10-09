"""
Our own copy of RVC's (somewhat difficult to use) script, `infer_batch_rvc.py`.
"""
import os, sys, pdb, torch

print("Env vars:")
print(os.environ)


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

print_gpu_info()


now_dir = os.getcwd()
sys.path.append(now_dir)
import sys
import torch
import tqdm as tq
from multiprocessing import cpu_count


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

import argparse

parser = argparse.ArgumentParser(description='Run VC inference')

parser.add_argument('--model_path', type=str, help='path to the .pth file', required=True)
parser.add_argument('--model_index_path', type=str, help='path to the .index file (empty string for none)', required=False, default='')
parser.add_argument('--hubert_model_path', type=str, help='path to the hubert model .pth file (default hubert_base.pt)', 
                    required=False, default='hubert_base.pt')

parser.add_argument('--input_audio_filename', type=str, help='source audio file', required=True)
parser.add_argument('--output_audio_filename', type=str, help='result audio file', required=True)

########
# Default values are taken from the Gradio UI
########
parser.add_argument('--f0_up_key', type=str, 
                    help='the f0 key for the input audio file(default "0")', 
                    required=False, default='0')
parser.add_argument('--f0_method', type=str, 
                    help='f0 estimation method to use: harvest (default), crepe, or pm', 
                    required=False, default='harvest')
parser.add_argument('--index_rate', type=float, 
                    help='The rate for the index (search feature ratio) (default 0.75)', 
                    required=False, default=0.75) # Gradio default 0.75
parser.add_argument('--filter_radius', type=int, 
                    help='The radius of the filter. If >=3, apply median filtering to harvested pitch results. Represents the filter radius and can reduce breathiness. (default 3)', 
                    required=False, default=3) # Gradio default 3
parser.add_argument('--resample_sr', type=int, 
                    help='The sample rate for resampling (post process). Set to zero for no resample. (default 0)', 
                    required=False, default=0)
parser.add_argument('--rms_mix_rate', type=float, 
                    help='RMS mix rate. Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio to 1, the more the output envelope is used (default 0.25)', 
                    required=False, default=0.25) # Gradio default 0.25
parser.add_argument('--protect', type=float, 
                    help='The protect value. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but may reduce indexing accuracy (default 0.33 ???)', 
                    required=False, default=0.33) # Gradio default 0.33

# Device / precision
parser.add_argument('--device', type=str, help='compute device (eg. "cuda:0"), default "cuda:0".', required=False, default='cuda:0')
parser.add_argument('--is_half', type=bool, help='halfwidth tensors (default false)', required=False, default=False)

args = parser.parse_args()

f0up_key = args.f0_up_key
input_path = args.input_audio_filename
index_path = args.model_index_path
f0method = args.f0_method
opt_path = args.output_audio_filename
model_path = args.model_path
index_rate = args.index_rate
device = args.device
is_half = args.is_half
filter_radius = args.filter_radius
resample_sr = args.resample_sr
rms_mix_rate = args.rms_mix_rate
protect = args.protect
hubert_path = args.hubert_model_path

config = Config(device, is_half)
now_dir = os.getcwd()
sys.path.append(now_dir)
from lib.train.vc_infer_pipeline import VC
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.audio import load_audio
from fairseq import checkpoint_utils
from scipy.io import wavfile

hubert_model = None


def load_hubert(hubert_path="hubert_base.pt"):
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(sid, input_audio, f0_up_key, f0_file, f0_method, file_index, index_rate, hubert_path="hubert_base.pt"):
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio, 16000)
    times = [0, 0, 0]
    if hubert_model == None:
        load_hubert(hubert_path)
    if_f0 = cpt.get("f0", 1)
    # audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,times,f0_up_key,f0_method,file_index,file_big_npy,index_rate,if_f0,f0_file=f0_file)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        input_audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        f0_file=f0_file,
    )
    print(times)
    return audio_opt


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, device, is_half, version
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:  #
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净，真奇葩
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}


get_vc(model_path)

file_path = input_path
wav_opt = vc_single(
    0, file_path, f0up_key, None, f0method, index_path, index_rate, hubert_path
)
out_path = opt_path
wavfile.write(out_path, tgt_sr, wav_opt)

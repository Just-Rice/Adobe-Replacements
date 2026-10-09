import os, sys, pdb, torch
from piper import PiperVoice
import wave
import argparse
from scipy.io import wavfile
import tqdm as tq
from multiprocessing import cpu_count


# Environment and GPU Info
print("Env vars:")
print(os.environ)
print('========================================')
print('Python interpreter', sys.executable)
print('PyTorch version', torch.__version__)
print('CUDA Available?', torch.cuda.is_available())
print('CUDA Device count', torch.cuda.device_count())
print('CUDA architectures library was compiled for', torch.cuda.get_arch_list())
print('========================================', flush=True)

now_dir = os.getcwd()
sys.path.append(now_dir)

# Config Class
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
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            self.is_half = True
        if self.n_cpu == 0:
            self.n_cpu = cpu_count()
        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
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

# Argument Parser
parser = argparse.ArgumentParser(description='Run Piper-TTS and RVC inference')

# Arguments for Piper-TTS
parser.add_argument('--tts_model_path', type=str, required=True)
parser.add_argument('--tts_config_path', type=str, required=True)
parser.add_argument('--text', type=str, required=True)

# Additional Arguments for Piper TTS
parser.add_argument('--length_scale', type=float, required=False, default=1.0, help='Phoneme length scale for TTS')
parser.add_argument('--noise_scale', type=float, required=False, default=0.5, help='Noise scale for TTS')
parser.add_argument('--noise_w', type=float, required=False, default=0.3, help='Phoneme width noise for TTS')
parser.add_argument('--sentence_silence', type=float, required=False, default=0.0, help='Seconds of silence after each sentence in TTS')

parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--model_index_path', type=str, required=False, default='')
parser.add_argument('--hubert_model_path', type=str, required=False, default='hubert_base.pt')
parser.add_argument('--input_audio_filename', type=str, required=True)
parser.add_argument('--output_audio_filename', type=str, required=True)
parser.add_argument('--f0_up_key', type=str, required=False, default='0')
parser.add_argument('--f0_method', type=str, required=False, default='harvest')
parser.add_argument('--index_rate', type=float, required=False, default=0.75)
parser.add_argument('--filter_radius', type=int, required=False, default=3)
parser.add_argument('--resample_sr', type=int, required=False, default=0)
parser.add_argument('--rms_mix_rate', type=float, required=False, default=0.25)
parser.add_argument('--protect', type=float, required=False, default=0.33)
parser.add_argument('--device', type=str, required=False, default='cuda:0')
parser.add_argument('--is_half', type=bool, required=False, default=False)

args = parser.parse_args()

# Load Piper Voice with additional settings
voice = PiperVoice.load(model_path=args.tts_model_path, config_path=args.tts_config_path)
synthesize_args = {
    "length_scale": args.length_scale,
    "noise_scale": args.noise_scale,
    "noise_w": args.noise_w,
    "sentence_silence": args.sentence_silence,
}

with wave.open('temp.wav', "wb") as wav_file:
    voice.synthesize(args.text, wav_file, **synthesize_args)

config = Config(args.device, args.is_half)
now_dir = os.getcwd()
sys.path.append(now_dir)
from vc_infer_pipeline import VC
from lib.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from lib.audio import load_audio
from fairseq import checkpoint_utils

hubert_model = None

def load_hubert(hubert_path="hubert_base.pt"):
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([hubert_path])
    hubert_model = models[0]
    hubert_model = hubert_model.to(args.device)
    if args.is_half:
        hubert_model = hubert_model.half()

def vc_single(sid, input_audio, f0_up_key, f0_file, f0_method, file_index, index_rate):
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio, 16000)
    times = [0, 0, 0]
    if hubert_model == None:
        load_hubert()
    if_f0 = cpt.get("f0", 1)
    audio_opt = vc.pipeline(hubert_model, net_g, sid, audio, input_audio, times, f0_up_key, f0_method, file_index, index_rate, if_f0, args.filter_radius, tgt_sr, args.resample_sr, args.rms_mix_rate, version, args.protect, f0_file=f0_file)
    print(times)
    return audio_opt

def get_vc(model_path):
    global tgt_sr, net_g, vc, cpt, version
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=args.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=args.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(args.device)
    if args.is_half:
        net_g = net_g.half()
    vc = VC(tgt_sr, config)

get_vc(args.model_path)
file_path = args.input_audio_filename
wav_opt = vc_single(0, file_path, args.f0_up_key, None, args.f0_method, args.model_index_path, args.index_rate)
out_path = args.output_audio_filename
wavfile.write(out_path, tgt_sr, wav_opt)

import sys
sys.path.append('waveglow/')
import shutil
import json

import numpy as np
import torch

#from tacotron2.hparams import create_hparams
#from tacotron2.model import Tacotron2
#from tacotron2.layers import TacotronSTFT
#from tacotron2.audio_processing import griffin_lim
#from tacotron2.text import text_to_sequence

#from waveglow.denoiser import Denoiser

from scipy.io.wavfile import write as write_wav

# Hifi-Gan
#from hifigan.env import AttrDict
#from hifigan.models import Generator
#from hifigan.denoiser import Denoiser as HifiDenoiser
#from hifigan.meldataset import mel_spectrogram
import resampy
import scipy

# For metadata
import subprocess
import magic
import os

# For talknet
from ControllableTalkNet.controllable_talknet import arpa_parse
from ControllableTalkNet.controllable_talknet import to_arpa
from nemo.collections.tts.models import TalkNetSpectModel
from nemo.collections.tts.models import TalkNetDursModel
from nemo.collections.tts.models import TalkNetPitchModel
from ControllableTalkNet.controllable_talknet import load_hifigan
from hifigan.meldataset import MAX_WAV_VALUE
from hifigan.meldataset import mel_spectrogram

def print_gpu_info():
    print('========================================')
    print('Python interpreter', sys.executable)
    print('PyTorch version', torch.__version__)
    print('CUDA Available?', torch.cuda.is_available())
    print('CUDA Device count', torch.cuda.device_count())
    #try:
    #    from tensorflow.python.client import device_lib
    #    print('local devices', str(device_lib.list_local_devices()).replace("\n", "\n  "))
    #except ImportError:
    #    print('no tensorflow - cannot list devices')
    #    pass
    print('========================================', flush=True)

def load_tacotron_model(checkpoint_path):
    hparams = create_hparams()
    hparams.sampling_rate = 22050 # Don't change this
    hparams.max_decoder_steps = 1000 # How long the audio will be before it cuts off (1000 is about 11 seconds)
    hparams.gate_threshold = 0.1 # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)
    model = Tacotron2(hparams)
    print('Loading synthesizer at path: {}'.format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    # NB(bt, 2021-05-31): Trying to get everything on the same device
    device = "cuda:0"
    model = model.to(device)
    return model

def load_waveglow_model(checkpoint_path):
    waveglow = torch.load(checkpoint_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    return waveglow

def load_hifigan_model(checkpoint_path, super_resolution = False):
    conf_name = 'config_32k' if super_resolution else 'config_v1'
    conf = os.path.join("hifigan", conf_name + ".json")
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

def preprocess_text(text, raw_input=True):
    # TODO: Handle ARPAbet
    pre_sequence = ""
    for line in text.split("\n"):
        if len(line) < 1:
            continue
        if raw_input:
            if line[-1] != ";":
                line = line + ";"
        else:
            #line = ARPA(line)
            pass
        pre_sequence += line

    with torch.no_grad():
        # save VRAM by not including gradients
        sequence = np.array(text_to_sequence(pre_sequence, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    return sequence

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_spectorgram_json_file(mel_outputs_postnet, output_spectrogram_filename):
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

    with open(output_spectrogram_filename, 'w') as outfile:
        json.dump(json_data, outfile, cls=NumpyEncoder)

def save_wav_audio_file(audio, output_audio_filename, sampling_rate=22050):
    output_audio = audio[0].data.cpu().numpy().astype(np.float32)
    write_wav(output_audio_filename, sampling_rate, output_audio)

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


class TacotronWaveglowPipeline:

    def __init__(self):
        # Cache of "path" => loaded model
        #self.tacotron_model_cache = {}
        #self.waveglow = None
        #self.waveglow_filename = None
        #self.hifigan = None
        #self.hifigan_denoiser = None
        #self.hifigan_h = None
        #self.hifigan_filename = None
        #self.hifigan_super_resolution = None
        #self.hifigan_super_resolution_denoiser = None
        #self.hifigan_super_resolution_h = None
        #self.hifigan_super_resolution_filename = None
        pass

    def maybe_load_hifigan(self, checkpoint_path):
        if self.hifigan_filename == checkpoint_path:
            print('Hifi-gan model already loaded into memory: {}'.format(checkpoint_path))
            return
        print('Dumping previous Hifi-gan model from memory: {}'.format(self.hifigan_filename))
        print('Loading Hifi-gan model into memory: {}'.format(checkpoint_path), flush=True)
        hifigan, h, denoiser = load_hifigan_model(checkpoint_path, False)
        self.hifigan = hifigan
        self.hifigan_denoiser = denoiser
        self.hifigan_h = h
        self.hifigan_filename = checkpoint_path

    def maybe_load_hifigan_super_resolution(self, checkpoint_path):
        if self.hifigan_super_resolution_filename == checkpoint_path:
            print('Hifi-gan (SuperRes) model already loaded into memory: {}'.format(checkpoint_path))
            return
        print('Dumping previous Hifi-gan (SuperRes) model from memory: {}'.format(self.hifigan_super_resolution_filename))
        print('Loading Hifi-gan (SuperRes) model into memory: {}'.format(checkpoint_path), flush=True)
        hifigan, h, denoiser = load_hifigan_model(checkpoint_path, True)
        self.hifigan_super_resolution = hifigan
        self.hifigan_super_resolution_denoiser = denoiser
        self.hifigan_super_resolution_h = h
        self.hifigan_super_resolution_filename = checkpoint_path

    def maybe_load_waveglow_model(self, checkpoint_path):
        if self.waveglow_filename == checkpoint_path:
            print('Waveglow model already loaded into memory: {}'.format(checkpoint_path))
            return
        print('Dumping previous waveglow model from memory: {}'.format(self.waveglow_filename))
        print('Loading waveglow model into memory: {}'.format(checkpoint_path), flush=True)
        self.waveglow = load_waveglow_model(checkpoint_path)
        self.waveglow_filename = checkpoint_path

    def maybe_load_tacotron_model_to_cache(self, checkpoint_path):
        if checkpoint_path in self.tacotron_model_cache:
            print('Tacotron model already loaded into memory: {}'.format(checkpoint_path))
            return
        print('Loading tacotron model into memory: {}'.format(checkpoint_path), flush=True)
        model = load_tacotron_model(checkpoint_path)
        self.tacotron_model_cache[checkpoint_path] = model
        print('Now {} synthesizer models in memory'.format(len(self.tacotron_model_cache)))

    def uncache_tacotron_model(self, checkpoint_path):
        print('Currently {} synthesizer models in memory'.format(
            len(self.tacotron_model_cache)))
        print('Clearing synthesizer from memory: {}'.format(checkpoint_path))
        self.tacotron_model_cache.pop(checkpoint_path, None)
        print('Now reduced to {} synthesizer models in memory'.format(
            len(self.tacotron_model_cache)))

    def infer(self, args):
        assert('synthesizer_checkpoint_path' in args)
        assert('vocoder_type' in args)
        assert('raw_text' in args)
        assert('output_audio_filename' in args)
        assert('output_spectrogram_filename' in args)
        assert('output_metadata_filename' in args)

        # ======================= BEGIN TALKNET EXPERIMENT ========================

        print('load models')

        with torch.no_grad():

            # TODO: Singer model?
            #  tnmodel = TalkNetSingerModel.restore_from(singer_path
            talknet_path = "/home/bt/models/talknet/david-attenborough/TalkNetSpect.nemo"
            tnmodel = TalkNetSpectModel.restore_from(talknet_path)

            durs_path = "/home/bt/models/talknet/david-attenborough/TalkNetDurs.nemo"
            pitch_path = "/home/bt/models/talknet/david-attenborough/TalkNetPitch.nemo"

            if os.path.exists(durs_path):
                print('loading duration and pitch models')
                tndurs = TalkNetDursModel.restore_from(durs_path)
                tnmodel.add_module("_durs_model", tndurs)
                tnpitch = TalkNetPitchModel.restore_from(pitch_path)
                tnmodel.add_module("_pitch_model", tnpitch)

            tnmodel.eval()

            print('done loading model?')

            # ==== ARPABET ====

            print('Raw text: {}'.format(args['raw_text']))

            text = args['raw_text']

            token_list = arpa_parse(text, tnmodel)
            print('token_list', token_list)

            DEVICE = "cuda:0"
            tokens = torch.IntTensor(token_list).view(1, -1).to(DEVICE)
            print('tokens', tokens)

            arpa = to_arpa(token_list)

            print('arpa', arpa)
            print('done with arpabet')

            # ==== END ARPABET ====

            # TODO: Either pitch prediction (if model given), or use a source audio file (TODO)

            print("spectrogram estimation")
            spect = tnmodel.generate_spectrogram(tokens=tokens)


            # ==== HIFIGAN ====

            print("load hifigan")

            hifigan_path = "/home/bt/models/talknet/david-attenborough/hifiganmodel"

            hifigan, h, denoiser = load_hifigan(hifigan_path, "config_v1")
            hifipath = hifigan_path

            y_g_hat = hifigan(spect.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
            audio_np = (
                audio_denoised.detach().cpu().numpy().reshape(-1).astype(np.int16)
            )

            # ==== HIFIGAN SR ====

            print("load hifigan SR")

            sr_path = "/home/bt/models/hifigan/Superres_Twilight_33000"
            hifigan_sr, h2, denoiser_sr = load_hifigan(sr_path, "config_32k")

            # TODO
            # TODO
            # TODO -- Auto-tuning
            # TODO
            # TODO

            print("resample 32k")

            # Resample to 32k
            wave = resampy.resample(
                audio_np,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            print("hifi SR")

            # HiFi-GAN super-resolution
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device(DEVICE))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                h2.n_fft,
                h2.num_mels,
                h2.sampling_rate,
                h2.hop_size,
                h2.win_size,
                h2.fmin,
                h2.fmax,
            )
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]

            # High-pass filter, mixing and denormalizing
            audio2_denoised = audio2_denoised.detach().cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(
                101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
            )
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            y *= 4.0  # superres strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded

            #buffer = io.BytesIO()
            #wavfile.write(buffer, 32000, sr_mix.astype(np.int16))
            #b64 = base64.b64encode(buffer.getvalue())
            #sound = "data:audio/x-wav;base64," + b64.decode("ascii")

            print("done!")


        output_audio = sr_mix.astype(np.int16)
        #rate = self.hifigan_super_resolution_h.sampling_rate
        rate = h2.sampling_rate
        write_wav(args['output_audio_filename'], rate, output_audio)

        print('Saving spectrogram as JSON...')
        save_spectorgram_json_file(spect, args['output_spectrogram_filename'])

        print('Generating metadata file...')
        generate_metadata_file(args['output_audio_filename'], args['output_metadata_filename'])


        # ======================= END TALKNET EXPERIMENT ========================

        #tacotron = self.tacotron_model_cache[args['synthesizer_checkpoint_path']]

        #with torch.no_grad():
        #    mel_outputs, mel_outputs_postnet, _, alignments = tacotron.inference(sequence)

        #if args['vocoder_type'] == 'hifigan-superres':
        #    self.vocode_hifigan_superres(args, mel_outputs_postnet)
        #elif args['vocoder_type'] == 'waveglow':
        #    self.vocode_waveglow(args, mel_outputs_postnet)
        #else:
        #    raise Exception('Wrong vocoder type: {}'.format(args['vocoder_type']))


    def vocode_waveglow(self, args, mel_outputs_postnet):
        print('Running waveglow...')
        with torch.no_grad():
            sigma = 0.8
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=sigma)

        print('Encoding and saving audio...')
        save_wav_audio_file(audio, args['output_audio_filename'])

    def vocode_hifigan_superres(self, args, mel_outputs_postnet):
        print('Running hifigan...')
        with torch.no_grad():
            y_g_hat = self.hifigan(mel_outputs_postnet.float())

            MAX_WAV_VALUE = 32768.0
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_denoised = self.hifigan_denoiser(audio.view(1, -1), strength=35)[:, 0]

            # Resample to 32k
            audio_denoised = audio_denoised.cpu().numpy().reshape(-1)

            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
            audio_denoised = audio_denoised * normalize
            wave = resampy.resample(
                audio_denoised,
                self.hifigan_h.sampling_rate,
                self.hifigan_super_resolution_h.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # HiFi-GAN super-resolution
            print('Running hifigan super resolution...')
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device("cuda"))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                self.hifigan_super_resolution_h.n_fft,
                self.hifigan_super_resolution_h.num_mels,
                self.hifigan_super_resolution_h.sampling_rate,
                self.hifigan_super_resolution_h.hop_size,
                self.hifigan_super_resolution_h.win_size,
                self.hifigan_super_resolution_h.fmin,
                self.hifigan_super_resolution_h.fmax,
            )
            y_g_hat2 = self.hifigan_super_resolution(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio2_denoised = self.hifigan_denoiser(audio2.view(1, -1), strength=35)[:, 0]

            # High-pass filter, mixing and denormalizing
            audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(
                101, cutoff=10500, fs=self.hifigan_super_resolution_h.sampling_rate, pass_zero=False
            )
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            superres_strength = 4.0
            y *= superres_strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded
            sr_mix = sr_mix / normalize

            print('Encoding and saving audio...')
            rate = self.hifigan_super_resolution_h.sampling_rate
            output_audio = sr_mix.astype(np.int16)
            write_wav(args['output_audio_filename'], rate, output_audio)

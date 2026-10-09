import sys

sys.path.append('waveglow/')
import shutil
import json

import numpy as np
import torch
import time
from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2
from tacotron2.layers import TacotronSTFT
from tacotron2.audio_processing import griffin_lim
from tacotron2.text import text_to_sequence

from waveglow.denoiser import Denoiser

from scipy.io.wavfile import write as write_wav

# Hifi-Gan
from hifigan.env import AttrDict
from hifigan.models import Generator
from hifigan.denoiser import Denoiser as HifiDenoiser
from hifigan.meldataset import mel_spectrogram
import resampy
import scipy

# For metadata
import subprocess
import magic
import os

from text.pipeline.english_pipeline_v1 import EnglishPipelineV1
from text.pipeline.espeak_pipeline import EspeakPipeline
from text.pipeline.legacy_fakeyou_pipeline import LegacyFakeYouPipeline
from text.pipeline_components.phonetization.cmudict import DEFAULT_CMUDICT_PATH, CMUDict
from text.pipeline_components.phonetization.english_phonetization import EnglishPhonetization

from logger import LOGGER

# Multiplying the mel output postnet by a constant factor scales it before vocoding,
# often resulting in "better" (less tinny) sound.
# This value was determined by Ozzy via experimentation.
DEFAULT_MEL_MULTIPLIER = 1.4

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

def load_tacotron_model(checkpoint_path):
    hparams = create_hparams()
    hparams.sampling_rate = 22050 # Don't change this
    hparams.max_decoder_steps = 1000 # How long the audio will be before it cuts off (1000 is about 11 seconds)
    hparams.gate_threshold = 0.1 # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)
    model = Tacotron2(hparams)
    LOGGER.info('Loading synthesizer at path: {}'.format(checkpoint_path))
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

# Modern pipelines
SPANISH_ESPEAK_PIPELINE = EspeakPipeline(lang='es')
ENGLISH_ESPEAK_PIPELINE = EspeakPipeline(lang='en')

# Legacy pipelines
ENGLISH_PIPELINE_V1 = EnglishPipelineV1(EnglishPhonetization(CMUDict(DEFAULT_CMUDICT_PATH)))
LEGACY_FAKEYOU_PIPELINE = LegacyFakeYouPipeline()

def parse_int_or_default(maybe_int, default_int):
    if maybe_int:
        try:
            return int(maybe_int)
        except:
            pass
    return default_int

def preprocess_text(raw_user_text: str, text_pipeline_type: str):
    integer_sequence = []

    if text_pipeline_type == 'legacy_fakeyou':
        integer_sequence = LEGACY_FAKEYOU_PIPELINE.user_input_to_sequence(raw_user_text)
    elif text_pipeline_type == 'english_v1':
        integer_sequence = ENGLISH_PIPELINE_V1.user_input_to_sequence(raw_user_text)
    elif text_pipeline_type == 'spanish_v2':
        integer_sequence = SPANISH_ESPEAK_PIPELINE.user_input_to_sequence(raw_user_text)
    else: 
        # TODO: Fall back to the newest pipeline once server code has been deployed.
        integer_sequence = LEGACY_FAKEYOU_PIPELINE.user_input_to_sequence(raw_user_text)

    with torch.no_grad():
        # save VRAM by not including gradients
        sequence = np.array(integer_sequence)[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    return sequence

def preprocess_text_old(text, raw_input=True):
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


class FakeYouTt2Pipeline:

    def __init__(self):
        # Cache of "path" => loaded model
        self.tacotron_model_cache = {}
        self.tacotron_model_cache_size_limit = None
        self.tacotron_model_cache_usage_tracker = {}
        self.tacotron_model_cache_eviction_percent = 0.1
        self.waveglow = None
        self.waveglow_filename = None

        self.hifigan = None
        self.hifigan_denoiser = None
        self.hifigan_h = None
        self.hifigan_filename = None
        self.hifigan_super_resolution = None
        self.hifigan_super_resolution_denoiser = None
        self.hifigan_super_resolution_h = None
        self.hifigan_super_resolution_filename = None


    # method to set cache limit
    def set_cache_limit(self, limit):
        self.tacotron_model_cache_size_limit = limit

    def maybe_load_hifigan(self, checkpoint_path):
        if not checkpoint_path or checkpoint_path == 'BOGUS_NOT_USED':
            # Requests no longer have to specify a hifigan to use (if using waveglow)
            return
        if self.hifigan_filename == checkpoint_path:
            LOGGER.info('Hifi-gan model already loaded into memory: {}'.format(checkpoint_path))
            return
        LOGGER.info('Dumping previous Hifi-gan model from memory: {}'.format(self.hifigan_filename))
        LOGGER.info('Loading Hifi-gan model into memory: {}'.format(checkpoint_path))
        hifigan, h, denoiser = load_hifigan_model(checkpoint_path, False)
        self.hifigan = hifigan
        self.hifigan_denoiser = denoiser
        self.hifigan_h = h
        self.hifigan_filename = checkpoint_path

    def maybe_load_hifigan_super_resolution(self, checkpoint_path):
        if not checkpoint_path or checkpoint_path == 'BOGUS_NOT_USED':
            # Requests no longer have to specify a hifigan to use (if using waveglow)
            return
        if self.hifigan_super_resolution_filename == checkpoint_path:
            LOGGER.info('Hifi-gan (SuperRes) model already loaded into memory: {}'.format(checkpoint_path))
            return
        LOGGER.info('Dumping previous Hifi-gan (SuperRes) model from memory: {}'.format(self.hifigan_super_resolution_filename))
        LOGGER.info('Loading Hifi-gan (SuperRes) model into memory: {}'.format(checkpoint_path))
        hifigan, h, denoiser = load_hifigan_model(checkpoint_path, True)
        self.hifigan_super_resolution = hifigan
        self.hifigan_super_resolution_denoiser = denoiser
        self.hifigan_super_resolution_h = h
        self.hifigan_super_resolution_filename = checkpoint_path

    def maybe_load_waveglow_model(self, checkpoint_path):
        if not checkpoint_path or checkpoint_path == 'BOGUS_NOT_USED':
            # Requests no longer have to specify a waveglow to use (if using hifigan)
            return
        if self.waveglow_filename == checkpoint_path:
            LOGGER.info('Waveglow model already loaded into memory: {}'.format(checkpoint_path))
            return
        LOGGER.info('Dumping previous waveglow model from memory: {}'.format(self.waveglow_filename))
        LOGGER.info('Loading waveglow model into memory: {}'.format(checkpoint_path))
        self.waveglow = load_waveglow_model(checkpoint_path)
        self.waveglow_filename = checkpoint_path

    def maybe_load_tacotron_model_to_cache(self, checkpoint_path):
        if checkpoint_path in self.tacotron_model_cache:
            LOGGER.info('Tacotron model already loaded into memory: {}'.format(checkpoint_path))
            # update usage tracker with current timestamp
            self.tacotron_model_cache_usage_tracker[checkpoint_path] = time.time()
            return
    
        if self.tacotron_model_cache_size_limit and len(self.tacotron_model_cache) >= self.tacotron_model_cache_size_limit:
            # uncache at least a few least recently used models
            num_to_uncache = int(self.tacotron_model_cache_size_limit * self.tacotron_model_cache_eviction_percent)
            LOGGER.info('Cache limit reached, uncaching {} least recently used models'.format(num_to_uncache))
            # find the oldest models
            oldest_models = sorted(self.tacotron_model_cache_usage_tracker.items(), key=lambda x: x[1])[:num_to_uncache]
            for model_path, _ in oldest_models:
                self.uncache_tacotron_model(model_path)
                # update usage tracker to remove the uncached model
                self.tacotron_model_cache_usage_tracker.pop(model_path, None)

        LOGGER.info('Loading tacotron model into memory: {}'.format(checkpoint_path))
        model = load_tacotron_model(checkpoint_path)
        self.tacotron_model_cache[checkpoint_path] = model
        # update usage tracker with current timestamp
        self.tacotron_model_cache_usage_tracker[checkpoint_path] = time.time()
        LOGGER.info('Now {} synthesizer models in memory'.format(len(self.tacotron_model_cache)))

    def uncache_tacotron_model(self, checkpoint_path):
        LOGGER.info('Currently {} synthesizer models in memory'.format(
            len(self.tacotron_model_cache)))
        LOGGER.info('Clearing synthesizer from memory: {}'.format(checkpoint_path))
        self.tacotron_model_cache.pop(checkpoint_path, None)
        LOGGER.info('Now reduced to {} synthesizer models in memory'.format(
            len(self.tacotron_model_cache)))

    def infer(self, args):
        # Model parameters
        assert('synthesizer_checkpoint_path' in args)
        assert('vocoder_type' in args)
        assert('text_pipeline_type' in args)
        # Optional mel scaling before vocodinng
        assert('use_default_mel_multiply_factor' in args)
        assert('maybe_custom_mel_multiply_factor' in args)
        # Premium features
        assert('max_decoder_steps' in args)
        # User input
        assert('raw_text' in args)
        # Output files
        assert('output_audio_filename' in args)
        assert('output_spectrogram_filename' in args)
        assert('output_metadata_filename' in args)

        LOGGER.info(f"raw inference text: {args['raw_text']}")

        sequence = preprocess_text(args['raw_text'], args['text_pipeline_type'])

        tacotron = self.tacotron_model_cache[args['synthesizer_checkpoint_path']]

        tacotron.fakeyou_set_max_decoder_steps(args['max_decoder_steps'])

        LOGGER.info("running inference...")

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = tacotron.inference(sequence)
        
        # Multiply the mel outputs for "better" sound.
        use_multiplier = False
        mel_multiplier = DEFAULT_MEL_MULTIPLIER

        if args['maybe_custom_mel_multiply_factor']:
            use_multiplier = True
            mel_multiplier = float(args['maybe_custom_mel_multiply_factor'])
        elif args['use_default_mel_multiply_factor']:
            use_multiplier = True

        if use_multiplier:
            LOGGER.info(f'Scaling audio by {mel_multiplier}')
            mel_outputs_postnet = mel_outputs_postnet * mel_multiplier

        if args['vocoder_type'] == 'hifigan-superres':
            self.vocode_hifigan_superres(args, mel_outputs_postnet)
        elif args['vocoder_type'] == 'waveglow':
            self.vocode_waveglow(args, mel_outputs_postnet)
        else:
            raise Exception('Wrong vocoder type: {}'.format(args['vocoder_type']))

        LOGGER.info('Saving spectrogram as JSON...')
        save_spectorgram_json_file(mel_outputs_postnet, args['output_spectrogram_filename'])

        LOGGER.info('Generating metadata file...')
        generate_metadata_file(args['output_audio_filename'], args['output_metadata_filename'])

    def vocode_waveglow(self, args, mel_outputs_postnet):
        LOGGER.info('Running waveglow...')
        with torch.no_grad():
            sigma = 0.8
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=sigma)

        LOGGER.info('Encoding and saving audio...')
        save_wav_audio_file(audio, args['output_audio_filename'])

    def vocode_hifigan_superres(self, args, mel_outputs_postnet):
        LOGGER.info('Running hifigan...')
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
            LOGGER.info('Running hifigan super resolution...')
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

            LOGGER.info('Encoding and saving audio...')
            rate = self.hifigan_super_resolution_h.sampling_rate
            output_audio = sr_mix.astype(np.int16)
            write_wav(args['output_audio_filename'], rate, output_audio)

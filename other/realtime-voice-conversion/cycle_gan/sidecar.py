#!/usr/bin/env python3

"""
Starts a demo HTTP server to capture and transform audio
as a live demonstration of the trained model.

Brandon Thomas 2019-07-29 <bt@brand.io> <echelon@gmail.com>
"""

# noinspection PyInterpreter
import argparse
import io
import librosa
import numpy as np
import os
import pathlib
import scipy
import soundfile
import struct
import subprocess
import tempfile
import tensorflow as tf
import zmq

from model import CycleGAN
from preprocess import *
from protos.audio_pb2 import VocodeAudioRequest
from protos.audio_pb2 import VocodeAudioResponse

print("TensorFlow version: {}".format(tf.version.VERSION))


class Converter():
    def __init__(self, model_dir, model_name):
        self.num_features = 24
        self.frame_period = 5.0

        self.model = CycleGAN(num_features = self.num_features, mode = 'test')

        self.model.load(filepath = os.path.join(model_dir, model_name))

        self.mcep_normalization_params = np.load(os.path.join(model_dir, 'mcep_normalization.npz'))
        self.mcep_mean_A = self.mcep_normalization_params['mean_A']
        self.mcep_std_A = self.mcep_normalization_params['std_A']
        self.mcep_mean_B = self.mcep_normalization_params['mean_B']
        self.mcep_std_B = self.mcep_normalization_params['std_B']

        self.logf0s_normalization_params = np.load(os.path.join(model_dir,
          'logf0s_normalization.npz'))
        self.logf0s_mean_A = self.logf0s_normalization_params['mean_A']
        self.logf0s_std_A = self.logf0s_normalization_params['std_A']
        self.logf0s_mean_B = self.logf0s_normalization_params['mean_B']
        self.logf0s_std_B = self.logf0s_normalization_params['std_B']

    def convert_partial(self, wav, conversion_direction='A2B', model_sampling_rate=16000):
        wav = wav_padding(wav = wav,
            sr = model_sampling_rate,
            frame_period = self.frame_period,
            multiple = 4)

        #librosa.output.write_wav('wav_padding.wav', wav, model_sampling_rate)

        f0, timeaxis, sp, ap = world_decompose(wav = wav,
            fs = model_sampling_rate,
            frame_period = self.frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp,
            fs = model_sampling_rate,
            dim = self.num_features)
        coded_sp_transposed = coded_sp.T

        f0_converted = pitch_conversion(f0 = f0,
            mean_log_src = self.logf0s_mean_A,
            std_log_src = self.logf0s_std_A,
            mean_log_target = self.logf0s_mean_B,
            std_log_target = self.logf0s_std_B)
        coded_sp_norm = (coded_sp_transposed - self.mcep_mean_A) / self.mcep_std_A

        coded_sp_converted_norm = self.model.test(inputs = np.array([coded_sp_norm]),
            direction = conversion_direction)[0]
        coded_sp_converted = coded_sp_converted_norm * self.mcep_std_B + self.mcep_mean_B

        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted,
            fs = model_sampling_rate)
        wav_transformed = world_speech_synthesis(f0 = f0_converted,
            decoded_sp = decoded_sp_converted,
            ap = ap,
            fs = model_sampling_rate,
            frame_period = self.frame_period)

        # For debugging model output, uncomment the following line:
        #librosa.output.write_wav('model_output.wav', wav_transformed, model_sampling_rate)

        # TODO: Perhaps ditch this. It's probably unnecessary work.
        #upsampled = librosa.resample(wav_transformed, model_sampling_rate, 48000)
        #pcm_data = upsampled.astype(np.float64)
        #stereo_pcm_data = np.tile(pcm_data, (2,1)).T
        #return stereo_pcm_data.astype(np.float32)
        #librosa.output.write_wav('model_output.wav', pcm_data, 48000)

        #return wav
        return wav_transformed

    #def convert(self, wav, conversion_direction='A2B'):
    #    pcm_data = self.convert_partial(wav, conversion_direction=conversion_direction)
    #    buf = io.BytesIO()
    #    # pcm_data: A 1-D or 2-D numpy array of either integer or float data-type.
    #    # To write multiple-channels, use a 2-D array of shape (Nsamples, Nchannels).
    #    scipy.io.wavfile.write(buf, 48000, pcm_data)
    #    return buf

# Set up model
# This should live long in memory, so we do it up front.
model_dir_default = './model/sf1_tm1'
model_name_default = 'sf1_tm1.ckpt'

# TODO: UNCOMMENT
converter = Converter(model_dir_default, model_name_default)

TEMP_DIR = tempfile.TemporaryDirectory(prefix='queue_audio')

def temp_file_name(suffix='.wav'):
    # NB: Not actually using the tempfile. Just the random name.
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
    name = os.path.basename(temp_file.name)
    return os.path.join(TEMP_DIR.name, name)

def convert(audio,
            vocode_params=None,
            request_batch_number=0,
            skip_vocode=False,
            save_files=False,
            discard_vocoded_audio=False):
    #audio = np.array(audio, dtype=np.int16)
    #data, samplerate = soundfile.read(audio)
    #print('samplerate', samplerate)
    """
    samplerate 44100
    data.shape (77824, 2)
    data.dtype float64
    mono [0.0050354  0.00518799 0.0050354  ... 0.11651611 0.11935425 0.1164856 ]
    mono.shape (77824,)
    mono.dtype float64
    downsampled [0.00329925 0.00577342 0.00474898 ... 0.0851728  0.1166483  0.        ]
    downsampled.shape (28236,)
    downsampled.dtype float64

    1) Data should be float64
    2) Output is BYTES!! Not floats.
    3) Result is still somehow mono!?
    --------------------------------------------------

    demo_server.py

       capture sample rate = 44100
       downsample for algorithm = 16000
       upsample after done = 48000

    """

    audio = np.array(audio, dtype=np.float64)
    print('audio.shape', audio.shape)
    print('audio.type', audio.dtype)

    #source_rate = 88000 # Experimentally determined for Rust library 'CPAL'
    #source_rate = 44100

    if vocode_params.original_source_save_file:
        #filename = temp_file_name('.wav')
        filename = 'debug/{}_original_source.wav'.format(request_batch_number)
        print('----- Original wav file out: {}'.format(filename))
        scipy.io.wavfile.write(filename, vocode_params.original_source_rate, audio)

    # NB: Convert the input stereo signal into mono.
    # In the future the frontend should be responsible for sampling details.
    #audio = audio[:, 0]

    if vocode_params.pre_convert_resample:
        print("Resampling audio from {} Hz to {} Hz".format(vocode_params.original_source_rate,
                                                            vocode_params.pre_convert_resample_rate))
        audio = librosa.resample(audio, vocode_params.original_source_rate, vocode_params.pre_convert_resample_rate)
        print('resampled_audio.shape', audio.shape)
        print('resampled_audio.type', audio.dtype)
        if vocode_params.pre_convert_resample_save_file:
            #filename = temp_file_name('.wav')
            filename = 'debug/{}_pre_convert_resample.wav'.format(request_batch_number)
            print('----- Pre-downsampled file out: {}'.format(filename))
            scipy.io.wavfile.write(filename, vocode_params.pre_convert_resample_rate, audio)

    if skip_vocode:
        return audio

    results = converter.convert_partial(audio,
                                        conversion_direction='A2B',
                                        model_sampling_rate=vocode_params.model_hyperparameter_sampling_rate)

    if vocode_params.model_save_file:
        #filename = temp_file_name('.wav')
        filename = 'debug/{}_model_output.wav'.format(request_batch_number)
        print('----- Model file out: {}'.format(filename))
        scipy.io.wavfile.write(filename, vocode_params.model_hyperparameter_sampling_rate, results)

    print('results.type', type(results))
    print('results.len', len(results))

    if vocode_params.post_convert_resample:
        print("Resampling output audio from {} Hz to {} Hz".format(vocode_params.model_hyperparameter_sampling_rate,
                                                                   vocode_params.post_convert_resample_rate))
        #consume_rate = 68000 # Experimentally determined for Rust lib 'CPAL'
        results = librosa.resample(results,
                                   vocode_params.model_hyperparameter_sampling_rate,
                                   vocode_params.post_convert_resample_rate)
        if vocode_params.post_convert_resample_save_file:
            #filename = temp_file_name('.wav')
            filename = 'debug/{}_post_convert_resample.wav'.format(request_batch_number)
            print('----- Upsampled (transformed) file out: {}'.format(filename))
            scipy.io.wavfile.write(filename, vocode_params.post_convert_resample_rate, results)

    if discard_vocoded_audio:
        return audio
    else:
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:{}".format(args.port))
    print('Running server...')

    queue = []
    while True:
        #  Wait for next request from client
        message = socket.recv()

        vocode_request = VocodeAudioRequest.FromString(message)
        vocode_params = vocode_request.vocode_params

        queue.extend(vocode_request.float_audio)

        if len(queue) >= vocode_request.buffer_size_minimum:
            #results = queue[:]
            results = convert(queue,
                              vocode_params=vocode_params,
                              request_batch_number=vocode_request.request_batch_number,
                              skip_vocode=vocode_request.skip_vocode,
                              discard_vocoded_audio=vocode_request.discard_vocoded_audio)
            queue = []

            vocode_response = VocodeAudioResponse()
            vocode_response.float_audio[:] = results

            socket.send(vocode_response.SerializeToString())
        else:
            # Must send reply back to client
            socket.send(b"OK")

if __name__ == '__main__':
    main()


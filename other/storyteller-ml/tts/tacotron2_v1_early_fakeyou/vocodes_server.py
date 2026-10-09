#!/usr/bin/env python3

"""
Starts a demo HTTP server to capture and transform audio
as a live demonstration of the trained model.
Brandon Thomas 2019-07-29 <bt@brand.io> <echelon@gmail.com>
"""

import argparse
import falcon
import io
import librosa
import numpy as np
import os
import scipy
import soundfile
import tensorflow as tf
import subprocess
import tempfile
import json
import time

from falcon_multipart.middleware import MultipartMiddleware
#from model import CycleGAN
#from preprocess import *
from wsgiref import simple_server

from vocodes_common import FakeYouTt2Pipeline
from vocodes_common import load_hifigan_model
from vocodes_common import load_tacotron_model
from vocodes_common import load_waveglow_model
from vocodes_common import parse_int_or_default
from vocodes_common import print_gpu_info

from logger import LOGGER

print("TensorFlow version: {}".format(tf.version.VERSION))

print_gpu_info()

INDEX_HTML = '''
<!doctype html>
<html>
  <head>
    <style>
      input {
        width: 600px;
        margin: 1em;
        padding: 0.5em;
      }
    </style>
  </head>
  <body>
    <meta charset="utf-8" />
    <h1>TTS Inference</h1>
    <script src="./script/recorder.js" type="application/javascript"></script>
    <script type="application/javascript">
      function handleSubmit(ev) {
        ev.preventDefault();
        
        let inference_text = document.getElementById('inference_text').value;
        let vocoder_checkpoint_path = document.getElementById('vocoder_checkpoint_path').value;
        let synthesizer_checkpoint_path = document.getElementById('synthesizer_checkpoint_path').value;
        
        console.log(inference_text);
        
        postInference(inference_text, synthesizer_checkpoint_path, vocoder_checkpoint_path);
        return false;
      }
      
      function postInference(inference_text, synthesizer_checkpoint_path, vocoder_checkpoint_path) {
        const request_payload = {
          'inference_text': inference_text,
          'synthesizer_checkpoint_path': synthesizer_checkpoint_path,
          'vocoder_checkpoint_path': vocoder_checkpoint_path,
          'output_audio_filename': 'web_output_audio.wav',
          'output_spectrogram_filename': 'web_spectrogram.wav',
          'output_metadata_filename': 'web_metadata.wav',
        };
        
        let xhr = new XMLHttpRequest();
        xhr.open("POST", "/infer");
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.send(JSON.stringify(request_payload));
      }
      window.onload = function init() {
        document.getElementById('form').addEventListener('submit', (ev) => handleSubmit(ev));
      };
    </script>
    
    <form id="form">
      <label> Inference Text </label>
      <input id="inference_text" type="text" name="inference_text" />
      <br />
      
      <label> Synthesizer Path </label>
      <input id="synthesizer_checkpoint_path" type="text" name="synthesizer_checkpoint_path" value="/home/bt/models/tacotron2/tacotron2_uberduck_Noire.pt" />
      <br />
      
      <label> Vocoder Path </label>
      <input id="vocoder_checkpoint_path" type="text" name="vocoder_checkpoint_path" value="/home/bt/models/waveglow/waveglow_256channels_universal_v5.pt" />
      <br />
      
      <button>Submit TTS</button>
    </form>
  </body>
</html>
'''


#class Converter():
#    def __init__(self, model_dir, model_name):
#        self.num_features = 24
#        self.sampling_rate = 16000
#        self.frame_period = 5.0
#
#        self.model = CycleGAN(num_features = self.num_features, mode = 'test')
#
#        self.model.load(filepath = os.path.join(model_dir, model_name))
#
#        """
#        # NB: Save the graph
#        definition = self.model.sess.graph_def
#        directory = 'saved_model_2'
#        tf.train.write_graph(definition, directory, 'saved_model_2.pb', as_text=True)
#        # https://github.com/tensorflow/models/issues/3530#issuecomment-395968881
#        output_dir = './saved_model/'
#        builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
#        builder.add_meta_graph_and_variables(
#            self.model.sess,
#            [tf.saved_model.tag_constants.SERVING],
#            main_op=tf.tables_initializer(),
#        )
#        builder.save()
#        """
#
#        """
#        builder.add_meta_graph_and_variables(
#            self.model.sess,
#            [tf.saved_model.tag_constants.SERVING],
#            signature_def_map={
#                'predict_images':
#                    prediction_signature,
#                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                    classification_signature,
#            },
#            main_op=tf.tables_initializer())
#        """
#
#        self.mcep_normalization_params = np.load(os.path.join(model_dir, 'mcep_normalization.npz'))
#        self.mcep_mean_A = self.mcep_normalization_params['mean_A']
#        self.mcep_std_A = self.mcep_normalization_params['std_A']
#        self.mcep_mean_B = self.mcep_normalization_params['mean_B']
#        self.mcep_std_B = self.mcep_normalization_params['std_B']
#
#        self.logf0s_normalization_params = np.load(os.path.join(model_dir, 'logf0s_normalization.npz'))
#        self.logf0s_mean_A = self.logf0s_normalization_params['mean_A']
#        self.logf0s_std_A = self.logf0s_normalization_params['std_A']
#        self.logf0s_mean_B = self.logf0s_normalization_params['mean_B']
#        self.logf0s_std_B = self.logf0s_normalization_params['std_B']
#
#    def convert(self, wav, conversion_direction='A2B'):
#        wav = wav_padding(wav = wav, sr = self.sampling_rate, frame_period = self.frame_period, multiple = 4)
#        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = self.sampling_rate, frame_period = self.frame_period)
#        coded_sp = world_encode_spectral_envelop(sp = sp, fs = self.sampling_rate, dim = self.num_features)
#        coded_sp_transposed = coded_sp.T
#
#        if conversion_direction == 'A2B':
#            f0_converted = pitch_conversion(f0 = f0, mean_log_src = self.logf0s_mean_A, std_log_src = self.logf0s_std_A, mean_log_target = self.logf0s_mean_B, std_log_target = self.logf0s_std_B)
#            coded_sp_norm = (coded_sp_transposed - self.mcep_mean_A) / self.mcep_std_A
#            coded_sp_converted_norm = self.model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
#            coded_sp_converted = coded_sp_converted_norm * self.mcep_std_B + self.mcep_mean_B
#        else:
#            f0_converted = pitch_conversion(f0 = f0, mean_log_src = self.logf0s_mean_B, std_log_src = self.logf0s_std_B, mean_log_target = self.logf0s_mean_A, std_log_target = self.logf0s_std_A)
#            coded_sp_norm = (coded_sp_transposed - self.mcep_mean_B) / self.mcep_std_B
#            coded_sp_converted_norm = self.model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
#            coded_sp_converted = coded_sp_converted_norm * self.mcep_std_A + self.mcep_mean_A
#
#        coded_sp_converted = coded_sp_converted.T
#        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
#        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = self.sampling_rate)
#        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = self.sampling_rate, frame_period = self.frame_period)
#
#        # For debugging model output, uncomment the following line:
#        # librosa.output.write_wav('model_output.wav', wav_transformed, self.sampling_rate)
#
#        # TODO: Perhaps ditch this. It's probably unnecessary work.
#        upsampled = librosa.resample(wav_transformed, self.sampling_rate, 48000)
#        pcm_data = upsampled.astype(np.float64)
#        stereo_pcm_data = np.tile(pcm_data, (2,1)).T
#
#        buf = io.BytesIO()
#        scipy.io.wavfile.write(buf, 48000, stereo_pcm_data.astype(np.float32))
#        return buf

## Set up model
## This should live long in memory, so we do it up front.
#model_dir_default = './model/sf1_tm1'
#model_name_default = 'sf1_tm1.ckpt'
#converter = Converter(model_dir_default, model_name_default)

#tacotron = load_tacotron_model('/home/bt/models/tacotron2/tacotron2_uberduck_Noire.pt')
#waveglow = load_waveglow_model('/home/bt/models/waveglow/waveglow_256channels_universal_v5.pt')
pipeline = FakeYouTt2Pipeline()

class IndexHandler():
    def on_get(self, request, response):
        response.content_type = 'text/html'
        response.body = INDEX_HTML

class StatusHandler():
    def on_get(self, request, response):
        LOGGER.info("/_status endpoint requested.")
        response.content_type = 'application/json'
        # FakeYou / Storyteller universal health checking semantics:
        # success (bool) - calling the endpoint worked (no middle man proxy, etc.)
        # is_healthy (bool) - dependencies are ok
        response.body = '{"success": true, "is_healthy": true}'

class ApiHandler():
    def on_post(self, request, response):
        raw_data = json.load(request.bounded_stream)

        LOGGER.info(f"TTS Request: {raw_data}")

        # ========== RUST SERVICE ARGUMENTS (REQUEST PARAMETERS) ==========

        # Model parameters
        synthesizer_checkpoint_path = raw_data.get('synthesizer_checkpoint_path')
        text_pipeline_type = raw_data.get('text_pipeline_type')
        vocoder_type = raw_data.get('vocoder_type')
        waveglow_vocoder_checkpoint_path = raw_data.get('waveglow_vocoder_checkpoint_path')
        hifigan_vocoder_checkpoint_path = raw_data.get('hifigan_vocoder_checkpoint_path')
        hifigan_superres_vocoder_checkpoint_path = raw_data.get('hifigan_superres_vocoder_checkpoint_path')

        #  Optional mel scaling before vocoding
        use_default_mel_multiply_factor = raw_data.get('use_default_mel_multiply_factor')
        maybe_custom_mel_multiply_factor = raw_data.get('maybe_custom_mel_multiply_factor')

        # Premium features
        maybe_max_decoder_steps = raw_data.get('max_decoder_steps')

        # User input
        inference_text = raw_data.get('inference_text')

        # Output files
        output_audio_filename = raw_data.get('output_audio_filename')
        output_spectrogram_filename = raw_data.get('output_spectrogram_filename')
        output_metadata_filename = raw_data.get('output_metadata_filename')

        # Optional request parameters
        maybe_clear_synthesizer_checkpoint_path = \
            raw_data.get('maybe_clear_synthesizer_checkpoint_path')

        # ========== INFERENCE ==========

        pipeline.maybe_load_waveglow_model(waveglow_vocoder_checkpoint_path)
        pipeline.maybe_load_hifigan(hifigan_vocoder_checkpoint_path)
        pipeline.maybe_load_hifigan_super_resolution(hifigan_superres_vocoder_checkpoint_path)


        # if maybe_clear_synthesizer_checkpoint_path:
        #     pipeline.uncache_tacotron_model(maybe_clear_synthesizer_checkpoint_path)

        pipeline.maybe_load_tacotron_model_to_cache(synthesizer_checkpoint_path)

        # NB: 1000 is the default and is roughly 12 seconds.
        max_decoder_steps = parse_int_or_default(maybe_max_decoder_steps, 1000) 

        inference_args = {
            'raw_text': inference_text,
            'use_default_mel_multiply_factor': use_default_mel_multiply_factor,
            'maybe_custom_mel_multiply_factor': maybe_custom_mel_multiply_factor,
            'synthesizer_checkpoint_path': synthesizer_checkpoint_path,
            'vocoder_type': vocoder_type,
            'text_pipeline_type': text_pipeline_type,
            'output_audio_filename': output_audio_filename,
            'output_spectrogram_filename': output_spectrogram_filename,
            'output_metadata_filename': output_metadata_filename,
            'max_decoder_steps': max_decoder_steps,
        }

        LOGGER.info(f"running inference with args: {inference_args}")

        pipeline.infer(inference_args)

        LOGGER.info(f"Request complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model-cache-limit', type=int, default=100)
    parser.add_argument('--model-cache-prewarm-list-file', type=str, default='/model_cache_prewarm_list.txt')

    args = parser.parse_args()

    pipeline.set_cache_limit(args.model_cache_limit)

    # read prewarm list, _before_ starting the server
    # this happens before the health check, so we don't risk
    # starting the server and then mutating the cache state
    try:
      if os.path.exists(args.model_cache_prewarm_list_file):
          with open(args.model_cache_prewarm_list_file, 'r') as f:
              prewarm_list = f.read().splitlines()
              for model_path in prewarm_list:
                  if os.path.exists(model_path):
                    pipeline.maybe_load_tacotron_model_to_cache(model_path)
    except Exception as e:
        LOGGER.error(f"Failed to prewarm model cache: {e}")

    api = falcon.API(middleware=[MultipartMiddleware()])
    api.add_route('/', IndexHandler())
    api.add_route('/_status', StatusHandler())
    api.add_route('/infer', ApiHandler())

    LOGGER.info(f"Starting server on 0.0.0.0:{args.port}")
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()

if __name__ == '__main__':
    main()

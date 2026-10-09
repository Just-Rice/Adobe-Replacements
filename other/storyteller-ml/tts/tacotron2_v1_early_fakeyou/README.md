FakeYou TTS Pipeline
====================

The following files are entrypoints and an essential part of FakeYou.com functionality: 

* `vocodes_server.py` - Runs as an HTTP server sidecar to the TTS jobs. It loads Tacotron and HifiGan 
   models into memory and keeps an LRU cache. (The Rust job is responsible for managing which models 
   are loaded into GPU memory.) Long term it would be better to quantize the models and not stand up 
   a server to handle requests.

* `vocodes_model_check_tacotron.py` - This checks newly uploaded models to make sure that they have 
   the expected weights and dimensions. This will block invalid files from being uploaded (eg. video 
   files).


Run Dockerized in Development
-----------------------------

### TTS

```bash

docker build .
docker run --rm --gpus all -it \
  -p 8000:8000 \
  --mount type=bind,source=/tmp,target=/tmp \
  --entrypoint ./start_tts_server.sh [image name]
```

To Run Tests
------------

```
pytest text/
```

Vocodes Server Parameters
-------------------------

The `vocodes_server.py` HTTP server accepts HTTP POST requests to the `/infer` endpoint. The requests
are JSON-encoded and have the following parameters:

* `inference_text` - REQUIRED. The text to convert to speech.
* `synthesizer_checkpoint_path` - REQUIRED. Where the Tacotron model lives on the filesystem.
* `vocoder_type` - REQUIRED. Either `hifigan-superres` or `waveglow`.
* `waveglow_vocoder_checkpoint_path` - OPTIONAL. Where the waveglow vocoder lives on the filesystem. 
   Inference will use either Waveglow or HifiGan (preferred).
* `hifigan_vocoder_checkpoint_path` - OPTIONAL. Where the HifiGan vocoder lives on the filesystem. 
   Inference will use either Waveglow or HifiGan (preferred).
* `hifigan_superres_vocoder_checkpoint_path` - OPTIONAL. Where the superres HifiGan vocoder lives on the
   filesystem. This can be used instead of HifiGan.
* `output_audio_filename` - REQUIRED. The name of the output audio file. This controls where it is saved, 
   which is important for the code that uploads it to GCP.
* `output_spectrogram_filename` - REQUIRED. The name of the spectrogram file. This controls where it is saved,
   which is important for the code that uploads it to GCP.
* `output_metadata_filename` - REQUIRED. The name of the metadata file. It contains stats such as audio duration.
   This controls where it is saved, which is important for the code that uploads it to GCP.
* `maybe_clear_synthesizer_checkpoint_path` - OPTIONAL. Rust can instruct Python to remove models from memory.

This interface between Rust and Python can be changed at any time, of course.


TODO
----

- [ ] Fix `NeMo/` and `nemo/` paths. These do not work on Macs as the filesystem is case-insensitive (ugh).

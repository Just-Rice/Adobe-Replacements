```sh
virtualenv env
source env/bin/activate
pip install -r requirements.txt
python3

import fakeyou_infer
fakeyou_infer.infer(
  "/home/paul/projects/ai/rvc/weights/biggie-smalls.pth", # main model path
  "/home/paul/projects/ai/rvc/weights/added_IVF2933_Flat_nprobe_10.index", # index model path
  "/home/paul/projects/ai/so-vits-svc/checkpoint_best_legacy_500.pt", # hubert path
  "/mnt/8terra/tts/voice-conversion-sources/crazyfrog.wav", # input audio
  "/tmp/test.wav" # output audio
)
```
See more optional parameters in the `fakeyou_infer.py` `infer` function

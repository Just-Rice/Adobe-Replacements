Voice Converter Sidecar
=======================
This project spins up a Rust binary to handle audio input and output, then 
ships microphone audio over proto/zeromq to a Python sidecar running
CycleGAN-VC to convert it into target speech.

### CycleGAN
I've included a pared down copy of CycleGAN for the purposes of model
evaluation. It handles all of the audio buffering / sidecar integration
pieces. [The original CycleGAN I used is here](https://github.com/leimao/Voice_Converter_CycleGAN).

#### Note on Training
It's possible to train and evaluate at the same time using dual GPUs
(at least my dual 1080Ti setup). To run the sidecar on a particular
GPU (0-indexded), use:

```
CUDA_VISIBLE_DEVICES=1 ./sidecar.py
```

### Proto Compilation (for Python)
Codegen for Rust is built in. Codegen for Python uses,

```
protoc --python_out=cycle_gan protos/audio.proto
```

### Current Results
Currently there is 4.39 seconds of delay between speaking and generated output
with the sidecar setup on my desktop computer. This is really great and seems
promising.

This gets up to 6.0 seconds later. Drift continues to accrue, but it's a slow
build.

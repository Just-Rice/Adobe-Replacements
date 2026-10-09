so-vits-svc
===========

Original docs are in [README\_ORIGINAL.md](./README_ORIGINAL.md).

This is a snapshot of `f9f16d67e44c127b9cf6d02d6eb5a235d28b70d2` (branch `tag v3.12.1`). It was taken
from voicepaw's fork at https://github.com/voicepaw/so-vits-svc-fork


Dev Installation
----------------

`pip install -e .`

There is also a Dockerfile available so if you prefer:
```sh
docker build -t so-vits-svc .`
docker run -it --privileged --gpus 0 so-vits-svc
```

Usage
-----
```
Usage: svc infer [OPTIONS] INPUT_PATH

  Inference

Options:
  -o, --output-path PATH          path to output dir
  -s, --speaker TEXT              speaker name
  -m, --model-path PATH           path to model  [default: logs/44k]
  -c, --config-path PATH          path to config  [default: configs/44k/config.json]
  -k, --cluster-model-path PATH   path to cluster model
  -t, --transpose INTEGER         transpose  [default: 0]
  -db, --db-thresh INTEGER        threshold (DB) (RELATIVE)  [default: -20]
  -fm, --f0-method [crepe|crepe-tiny|parselmouth|dio|harvest]
                                  f0 prediction method  [default: dio]
  -a, --auto-predict-f0 / -na, --no-auto-predict-f0
                                  auto predict f0  [default: na]
  -r, --cluster-infer-ratio FLOAT
                                  cluster infer ratio  [default: 0]
  -n, --noise-scale FLOAT         noise scale  [default: 0.4]
  -p, --pad-seconds FLOAT         pad seconds  [default: 0.5]
  -d, --device TEXT               device  [default: cuda:0]
  -ch, --chunk-seconds FLOAT      chunk seconds  [default: 0.5]
  -ab, --absolute-thresh / -nab, --no-absolute-thresh
                                  absolute thresh  [default: nab]
  -h, --help                      Show this message and exit.```
```

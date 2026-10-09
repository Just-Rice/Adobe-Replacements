so-vits-svc
===========

Original docs are in [README\_ORIGINAL.md](./README_ORIGINAL.md).

This is a snapshot of `81551ce9c6fb7924d184c3c5a4cf9035168b28d2` (branch `main`). It was taken
from 34j's fork at https://github.com/34j/so-vits-svc-fork


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
$svc infer -h
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
  -a, --auto-predict-f0 BOOLEAN   auto predict f0  [default: True]
  -r, --cluster-infer-ratio FLOAT
                                  cluster infer ratio  [default: 0]
  -n, --noise-scale FLOAT         noise scale  [default: 0.4]
  -p, --pad-seconds FLOAT         pad seconds  [default: 0.5]
  -d, --device TEXT               device  [default: cuda]
  -ch, --chunk-seconds FLOAT      chunk seconds  [default: 0.5]
  -ab, --absolute-thresh BOOLEAN  absolute thresh  [default: False]
  -h, --help                      Show this message and exit.
```

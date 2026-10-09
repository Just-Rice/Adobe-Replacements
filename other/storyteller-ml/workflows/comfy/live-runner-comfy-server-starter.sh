#!/usr/bin/env bash


set -ex

source /app/venv/bin/activate

date > /restart.txt

cd /app/ComfyUI
echo /restart.txt | entr -nr timeout -k 5 0 python main.py --listen 0.0.0.0  --highvram --disable-metadata
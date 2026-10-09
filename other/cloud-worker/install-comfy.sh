#!/bin/bash

set -euxo pipefail

git clone https://github.com/Comfy-Org/ComfyUI.git

cd ComfyUI

pip install -r requirements.txt

#!/bin/bash
# One-shot MiniMax H3 setup for a fresh RunPod pod. Idempotent — safe to re-run
# after a pod reset (which wipes everything outside /workspace).
#
# Usage (from your machine, repo root):
#   scripts/local/push-to-pod.sh          # copies repo files to the pod
#   ssh -p <port> root@<pod-ip> 'bash /workspace/minimax-h3/install-minimax-h3.sh'
set -euo pipefail

# --- ComfyUI itself (container disk; gone after any pod edit/reset) ---
if [ ! -d /ComfyUI ]; then
  git clone https://github.com/Comfy-Org/ComfyUI.git /ComfyUI
fi
cd /ComfyUI
pip install -q --break-system-packages --no-warn-script-location -r requirements.txt
pip install -q --break-system-packages hf_transfer

# --- configs from the repo (staged under /workspace/minimax-h3 by push-to-pod) ---
STAGE=/workspace/minimax-h3
cp "$STAGE"/extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
mkdir -p /ComfyUI/user/default/workflows
cp "$STAGE"/workflows/*.json /ComfyUI/user/default/workflows/

# --- test/reference images (synthetic + real reference sets) ---
python3 "$STAGE"/make-test-images.py
# prefer the volume-staged copies (immune to fetch rate limits); fetch only to backfill
if [ -d "$STAGE"/bench-images ]; then
  cp "$STAGE"/bench-images/* /ComfyUI/input/ 2>/dev/null || true
fi
python3 "$STAGE"/fetch-bench-images.py || echo "WARN: v1 image fetch incomplete"
python3 "$STAGE"/fetch-bench-images-v2.py || echo "WARN: v2 image fetch incomplete"
mkdir -p "$STAGE"/bench-images
cp /ComfyUI/input/bench2_*.jpg /ComfyUI/input/bench2_*.png /ComfyUI/input/bench_*.png "$STAGE"/bench-images/ 2>/dev/null || true

# --- weights (skips files already on the volume) ---
tmux new -s downloads -d "bash $STAGE/download-weights.sh" 2>/dev/null \
  || echo "downloads session already exists"

# --- ComfyUI headless ---
bash "$STAGE"/start-comfyui.sh

echo "Done. Downloads: tail -f /workspace/minimax-h3/download.log"

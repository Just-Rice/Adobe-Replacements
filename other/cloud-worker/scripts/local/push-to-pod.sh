#!/bin/bash
# Stage everything the pod needs onto the persistent volume, so a wiped
# container can be rebuilt with one command. Run from the repo root.
set -euo pipefail

POD_HOST=${POD_HOST:-216.243.220.136}
POD_PORT=${POD_PORT:-17561}
DEST=root@$POD_HOST

ssh -p "$POD_PORT" "$DEST" 'mkdir -p /workspace/minimax-h3/workflows'
scp -O -P "$POD_PORT" \
  scripts/pod/install-minimax-h3.sh \
  scripts/pod/download-weights.sh \
  scripts/pod/start-comfyui.sh \
  scripts/pod/make-test-images.py \
  scripts/pod/fetch-bench-images.py \
  scripts/pod/fetch-bench-images-v2.py \
  comfyui/extra_model_paths.yaml \
  "$DEST":/workspace/minimax-h3/
scp -O -P "$POD_PORT" comfyui/workflows/*.json "$DEST":/workspace/minimax-h3/workflows/
echo "staged. rebuild with: ssh -p $POD_PORT $DEST 'bash /workspace/minimax-h3/install-minimax-h3.sh'"

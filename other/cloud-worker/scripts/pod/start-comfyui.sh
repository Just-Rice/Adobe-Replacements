#!/bin/bash
# Start ComfyUI headless on the pod, listening on 127.0.0.1:8188.
# Uses tmux when available (attachable session), else nohup.
# Log is per-host: pods sharing a network volume must not share log files.
# Extra launch flags via COMFY_ARGS, e.g.:
#   COMFY_ARGS="--reserve-vram 2.5" bash start-comfyui.sh   # 24GB cards
set -euo pipefail

LOG=/workspace/minimax-h3/comfyui-$(hostname).log
# reduce near-limit CUDA OOMs from fragmentation (community-verified for H3)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p /workspace/minimax-h3/output
ARGS=${COMFY_ARGS:-}

# wait out any process that is mid-shutdown before deciding it's "running"
for i in 1 2 3 4 5; do
  pgrep -f "python3? main[.]py" >/dev/null || break
  sleep 2
done
if pgrep -f "python3? main[.]py" >/dev/null; then
  echo "ComfyUI already running"
  exit 0
fi

if command -v tmux >/dev/null; then
  tmux new -s comfyui -d \
    "cd /ComfyUI && python3 main.py --listen 127.0.0.1 --port 8188 --output-directory /workspace/minimax-h3/output $ARGS 2>&1 | tee -a $LOG"
else
  cd /ComfyUI
  nohup python3 main.py --listen 127.0.0.1 --port 8188 --output-directory /workspace/minimax-h3/output $ARGS >>"$LOG" 2>&1 &
fi
echo "started; tail $LOG"

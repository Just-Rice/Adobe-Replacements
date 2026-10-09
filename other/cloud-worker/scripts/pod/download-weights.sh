#!/bin/bash
# Download all MiniMax H3 weight families from Comfy-Org/MiniMax-H3 to the
# RunPod network volume. Run on the pod, ideally inside tmux:
#   tmux new -s downloads -d 'bash /workspace/minimax-h3/download-weights.sh'
#
# Ordered so ComfyUI becomes usable as early as possible:
#   phase 1: VAEs + nvfp4 text encoder + pruned-int8 diffusion models (smallest family)
#   phase 2: int8_convrot family
#   phase 3: bf16 family (largest)
#
# Everything lands under /workspace/minimax-h3/models/{diffusion_models,text_encoders,vae}
# which ComfyUI picks up via extra_model_paths.yaml.

set -uo pipefail

REPO="Comfy-Org/MiniMax-H3"
DEST="/workspace/minimax-h3/models"
LOG="/workspace/minimax-h3/download.log"

mkdir -p "$DEST"
export HF_HUB_ENABLE_HF_TRANSFER=1

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

dl() {
  local file="$1"
  # hf download preserves the repo subpath (diffusion_models/..., vae/...) under --local-dir
  if [ -f "$DEST/$file" ]; then
    log "SKIP $file (already present)"
    return 0
  fi
  log "START $file"
  if hf download "$REPO" "$file" --local-dir "$DEST" >>"$LOG" 2>&1; then
    log "DONE $file ($(du -h "$DEST/$file" | cut -f1))"
  else
    log "FAILED $file"
    return 1
  fi
}

log "=== MiniMax H3 weight download run starting ==="

# --- Phase 1: minimum viable set (VAEs, text encoder, pruned int8 models) ---
dl vae/minimax_h3_video_vae_fp16.safetensors
dl vae/minimax_h3_audio_vae_fp32.safetensors
dl text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
dl diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
dl diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors

# --- Phase 2: int8_convrot (mid-size family) ---
dl diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors
dl diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors

# --- Phase 3: bf16 (full-precision family) ---
dl diffusion_models/minimax_h3_ref2va_bf16.safetensors
dl diffusion_models/minimax_h3_fl2va_bf16.safetensors

rm -rf "$DEST/.cache"
log "=== All downloads complete ==="

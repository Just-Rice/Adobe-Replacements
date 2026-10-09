# MiniMax H3 — weights log (B200 pod `c306014998a3`, 216.243.220.136)

Status as of **2026-08-06 04:35 UTC**.

## Source

All files downloaded from the HuggingFace repo
**[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)** (the
ComfyUI repack of [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)),
e.g. `https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/<path>`.
Downloaded with `hf download` + hf_transfer by `scripts/pod/download-weights.sh`
and `download-weights-tmpfs.sh`; log at `/workspace/minimax-h3/download.log`.

There are two model lines — **fl2va** (text-to-video + first/last-frame i2v)
and **ref2va** (reference-to-video) — each published in several precisions.
ComfyUI finds both storage locations below via `/ComfyUI/extra_model_paths.yaml`
(installed from `scripts/pod/extra_model_paths.yaml`).

## Storage locations on the pod

| Path                            | Medium                       | Notes                                        |
|---------------------------------|------------------------------|----------------------------------------------|
| `/workspace/minimax-h3/models/` | RunPod network volume (1 TB) | **Persistent** — survives pod resets/edits.  |

The volume was resized 100 GB → 1 TB on 2026-08-06 (which reset the pod), so
the earlier `/dev/shm` tmpfs overflow scheme is retired: **everything now
lives on the volume.**

## Downloaded — all files, `/workspace/minimax-h3/models/` (246 GB total)

| File                                                                 | Family                                 | Size   |
|----------------------------------------------------------------------|----------------------------------------|--------|
| `diffusion_models/minimax_h3_fl2va_bf16.safetensors`                 | **bf16** (fl2va)                       | 66 GB  |
| `diffusion_models/minimax_h3_ref2va_bf16.safetensors`                | **bf16** (ref2va)                      | 66 GB  |
| `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors`         | **int8** (fl2va)                       | 34 GB  |
| `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors`        | **int8** (ref2va)                      | 34 GB  |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`  | **int8 pruned** (fl2va)                | 21 GB  |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | **int8 pruned** (ref2va)               | 21 GB  |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`         | text encoder (Qwen3-VL-32B, nvfp4 AWQ) | 16 GB  |
| `vae/minimax_h3_video_vae_fp16.safetensors`                          | video VAE fp16                         | 5.2 GB |
| `vae/minimax_h3_audio_vae_fp32.safetensors`                          | audio VAE fp32                         | 0.6 GB |

**In progress: none.** All nine files fully downloaded and byte-verified.

## Not downloaded (exist upstream in the same repo)

| File                                                                       | Size       | Why skipped                                                                                |
|----------------------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------|
| `diffusion_models/minimax_h3_{fl2va,ref2va}_pruned_bf16.safetensors`       | 40 GB each | a 4th family (pruned, unquantized) — not in the 3 requested families                       |
| `diffusion_models/minimax_h3_{fl2va,ref2va}_pruned_fp8_scaled.safetensors` | 21 GB each | a 5th family — same size class as pruned int8                                              |
| `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`                    | 52 GB      | every official template uses the nvfp4_awq TE (now fits — grab if quality A/B wanted)      |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`            | 27 GB      | same                                                                                       |

## History / gotchas

- 2026-08-06: the pre-existing manual download of ref2va_pruned_int8 had a
  broken filename (`...safetensors?download=true` from a wget of the HF web
  URL); byte size matched HF exactly, so it was renamed and moved to the
  volume rather than re-downloaded.
- 2026-08-06: bf16 + fl2va_int8 downloads to the volume failed with
  `Disk quota exceeded` at ~93 GB used → tmpfs overflow scheme added.
- After any **pod restart**: tmpfs weights are gone; ComfyUI will still boot
  (extra_model_paths tolerates the missing dir) but the bf16/fl2va-int8 entries
  disappear from loader dropdowns until `download-weights-tmpfs.sh` is re-run.
- 2026-08-06 (later): volume resized to 1 TB → pod reset wiped the container
  disk (ComfyUI reinstalled via `install-minimax-h3.sh`); the three overflow
  files were re-downloaded to the volume. Final state: all weights persistent.

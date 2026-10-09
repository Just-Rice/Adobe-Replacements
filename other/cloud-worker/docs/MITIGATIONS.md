# MiniMax H3 — per-machine-class mitigations & OOM map

Compiled 2026-08-06 from measured failures and fixes across five pods.
Companion to `docs/BENCHMARKS.md` (runtimes) and `bench/failures.py` (stats).

## The two resources that fail, and their fixes

**1. Container RAM (cgroup cap — check `/sys/fs/cgroup/memory/memory.limit_in_bytes`,
cgroup v1 on RunPod consumer pods, or `/sys/fs/cgroup/memory.max` for v2).**
ComfyUI stages every loaded model's weights in host RAM ("prepared for dynamic
VRAM loading"). Task-switching between fl2va and ref2va stacks WITHOUT eviction
stages TE (15 GB) + two DiTs (20 GB each) + overhead ≈ 60 GB+. Exceeding the
cap = silent SIGKILL, no traceback, "server died" mid-run.
Fix: `--cache-none` (one model staged at a time) + `--disable-pinned-memory`
(staging stays pageable/reclaimable). Cost: weights reload per run — measured
~2.3× slowdown on the RTX 4000 (551 s vs an est. ~240 s), ~1.4× on the 4090.

**2. VRAM, in two distinct phases:**
- *TE vision encode* (multi-ref): budget bug, fixed by ComfyUI **PR #15316**
  (apply: `git fetch origin pull/15316/head:pr15316 && git checkout pr15316 --
  comfy/text_encoders/minimax.py`) + `--reserve-vram 4`–`5`. Ref pre-downscale
  (`--ref-downscale 0.2` in our harness) also clears this phase alone.
- *DiT sampling with ref tokens / long clips / large canvas*: needs the
  activations to fit beside the (partially loaded) 20 GB DiT. On 24 GB this is
  the hard wall: 15 s clips, 1344×768, and multi-ref all exceeded it pre-patch.
  `--lowvram` made it WORSE (crashes) — do not use for H3.

Hygiene for all capped machines: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(set automatically by `start-comfyui.sh`), torch ≥ 2.5 (`enable_gqa`), and for
Blackwell (B200/5090/RTX PRO 4000) torch ≥ 2.7 with cu128 wheels.

## Machine-class matrix (all measured, not theoretical)

| Pod class            | VRAM   | RAM cap | Required config                                              | What works                                        | What fails                    |
|----------------------|--------|---------|--------------------------------------------------------------|---------------------------------------------------|-------------------------------|
| B200 ($6.94/hr)      | 180 GB | ~3 TB   | none (defaults)                                              | everything incl. 9-ref, 15 s, full-res, bf16      | nothing observed (96/96)      |
| RTX 5090 ($0.99/hr)  | 32 GB  | 124 GB  | #15316 patch + `--reserve-vram 4` (add `--cache-none` when switching bf16 models) | EVERYTHING: 1–9 refs full-size, 9-ref max (34 min), 15 s, full-res, all aspects, bf16 incl. 8-ref (369 s) | bf16 model switching without cache-none (RAM kill) |
| RTX 4090 ($0.74/hr)  | 24 GB  | 62 GB   | #15316 patch + `--reserve-vram 5` + `--cache-none` + `--disable-pinned-memory` | **multi-ref RESCUED**: 8-ref 429 s, 9-ref 444 s, 8-ref@1344×768 1159 s (17–19 GB peaks), all aspects | 9-ref `max` sizing only |
| RTX PRO 4000 ($0.57/hr) | 24 GB | 29 GB | same as 4090 (mandatory even for t2v)                        | t2v/i2v, 1-ref (551–588 s — uneconomical)          | dropped from fleet on economics |

Key procurement insight: **the container RAM cap, not the GPU, decides the
mitigation burden.** A 24 GB card in a ≥96 GB-RAM container would skip
`--cache-none` and its ~1.4–2.3× tax. When renting, prefer pod configs with
RAM ≥ 4× VRAM.

## Reference-image envelope (node schema + measured)

- The node accepts up to **9 reference images** + 3 reference videos (each
  with optional soundtrack) + 3 standalone audio clips — "12 refs" = 9 images
  + 3 videos.
- `match` sizing (refs scaled down to canvas area) is the default and the
  cheap path; `max` (2048 px short edge) multiplies sampling cost ~9× at
  8 refs and is the most VRAM-hungry config in the whole matrix.
- Ref count scaling on B200: +4 s per extra ref (match). Counts 1→9 and the
  ref-resolution ladder (0.1 MP → native) are being measured on all three
  GPU classes right now (`--suite refmatrix`).

## Resolved verdicts (2026-08-06 late)

- **24 GB multi-ref: RESCUED.** Full stack = #15316 patch + `--reserve-vram 5`
  + `--cache-none` + `--disable-pinned-memory` + valid ref images. Entire
  practical envelope works (1–9 refs, all aspects, full-res canvas); only
  9-ref `max` sizing remains out of reach (and is uneconomical everywhere).
- **5090 runs the complete envelope** including bf16 t2v via layerwise
  streaming at 110 s warm — faster than its own pruned int8 (143 s).
- A recurring *non*-OOM failure class was corrupt reference downloads
  (HTTP 429 partial files) masquerading as capability failures — bench images
  are now staged on each region's network volume by the installer.
- Upstream: file the multi-ref OOM issue referencing #15316 (user decision —
  outward-facing).

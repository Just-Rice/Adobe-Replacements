# cloud-worker

Tooling for running the **MiniMax H3** open-weights video model (native stereo
audio, 24 fps, 5–15 s, ~1 MP native) on RunPod GPU machines via ComfyUI.

Current test rig: 1x B200 (180 GB) pod — connection details in `pod-info.txt`.

## Layout

- `comfyui/` — ComfyUI configs stored for easy rebuild
  - `extra_model_paths.yaml` — installed to `/ComfyUI/extra_model_paths.yaml`
  - `workflows/` — the three MiniMax H3 workflows (t2v, i2v, r2v), installed
    to `/ComfyUI/user/default/workflows/`
- `scripts/pod/` — runs on the pod
  - `install-minimax-h3.sh` — one-shot rebuild (ComfyUI clone + deps + configs
    + workflows + images + weight downloads + launch); idempotent, survives
    pod resets since it stages from `/workspace`
  - `download-weights.sh` — all weight families → `/workspace/minimax-h3/models`
  - `start-comfyui.sh` — headless ComfyUI in tmux on 127.0.0.1:8188
  - `make-test-images.py` — synthetic reference/first-frame images → `/ComfyUI/input`
- `scripts/local/push-to-pod.sh` — stage repo files onto the pod's persistent volume
- `scripts/local/port-forward.sh` — persistent tunnel `localhost:8188` → pod
  ComfyUI panel (auto-reconnects)
- `bench/minimax_bench.py` — API-based benchmark harness (t2v / i2v / ref2v ×
  duration × resolution × weight family); appends to `bench/results.csv` +
  `bench/results.jsonl`
- `docs/WEIGHTS.md` — which weights are downloaded, where, and which family
- `docs/RESEARCH.md` — model/ComfyUI findings, running without ComfyUI
  (SGLang / vLLM-Omni / diffusers), concurrency model, RunPod serverless
- `docs/BENCHMARKS.md` — measured results on the B200
- `docs/CONCURRENCY.md` — concurrency model + cost-per-video math across GPU tiers
- `docs/MITIGATIONS.md` — per-machine-class OOM map and required ComfyUI flags/patches
- `docs/SERVING-PLAN.md` — paid/free pool design, pricing tables, capacity & demand scenarios

## Quick start

```bash
# rebuild the pod from scratch (e.g. after a reset wiped the container disk)
scripts/local/push-to-pod.sh
ssh -p 17561 root@216.243.220.136 'bash /workspace/minimax-h3/install-minimax-h3.sh'

# tunnel to the ComfyUI panel (leave running)
scripts/local/port-forward.sh &
open http://localhost:8188

# one-off generation through the API
python3 bench/minimax_bench.py --task t2v --width 864 --height 480 --seconds 5

# smoke suite / full sweep
python3 bench/minimax_bench.py --suite quick
python3 bench/minimax_bench.py --suite sweep --prefix prunedint8_
```

On the pod, ComfyUI runs in tmux session `comfyui`
(`tmux attach -t comfyui`), logs at `/workspace/minimax-h3/comfyui.log`.

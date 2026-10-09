# MiniMax H3 — research findings

Compiled 2026-08-06. Sources linked inline; benchmark numbers we measured
ourselves live in `bench/results.csv` and `docs/BENCHMARKS.md`.

## The model

[MiniMax H3](https://www.minimax.io/blog/minimax-h3) is an open-weights
omni-modal video model: one 33B dense "Omni-Transformer" DiT (50 layers, hidden
5376) jointly denoises video + **native stereo audio** latents in a single
forward pass, conditioned on hidden states from a **Qwen3-VL-32B** text encoder
(layer 50). CFG-distilled — no negative prompt / CFG pass. Output is 24 fps
fixed, 4–15 s, native canvas ~1 MP (2K comes from a separate regenerate/upscale
module in the hosted API). Video VAE: 16x spatial / 4x temporal. Audio VAE:
stereo 32 kHz, 40 latent fps.

Two task checkpoints, each shipped in several precisions:

- **fl2va** — text-to-video and first/last-frame conditioning (covers t2v + i2v)
- **ref2va** — reference-to-video: up to 9 ref images, 3 ref videos (with
  soundtracks), 3 standalone audio clips, woven in via `<Picture i>` /
  `<Video k>` / `<Audio j>` prompt tags

**License caveat**: the MiniMax H3 Community License restricts open-weight use
to the **EU, UK, South Korea, and US**; elsewhere requires a license from
MiniMax ([Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md)).

## Weight families ("bf16 / int8 / int8 pruned")

- **bf16** — full precision, 66 GB per model line. Reference quality.
- **int8_convrot** — 34 GB. [ConvRot](https://arxiv.org/pdf/2512.03673) is
  rotation-based post-training quantization: a group-wise Hadamard rotation
  (expressed as a conv, hence the name) redistributes activation outliers so
  int8 quantizes near-losslessly. Community consensus: "near-lossless".
- **pruned_int8_convrot** — 21 GB. "Pruned" here is *not* layer dropping: the
  DiT's adaLN modulation weights (~40% of params) are replaced by a
  functionally-equivalent lookup table. Comfy/MiniMax claim no quality loss;
  HF discussions call the pruned line "a bit more experimental".
- (Also upstream: pruned_bf16 40 GB, pruned_fp8_scaled 21 GB, and community
  INT4/NVFP4/W4A8 quants — see [awesome-minimax-H3](https://github.com/wildminder/awesome-minimax-H3).)

Full stack VRAM classes (community): bf16 → 80 GB+; int8 → 48 GB; pruned int8 →
fits 32 GB cards; INT4-class community quants reach 12–16 GB.

## ComfyUI integration

- Support merged day-0: [PR #15224](https://github.com/Comfy-Org/ComfyUI/pull/15224);
  int8_convrot VAE decode (~1.5x faster) in [PR #15334](https://github.com/Comfy-Org/ComfyUI/pull/15334).
- Official tutorial: <https://docs.comfy.org/tutorials/video/minimax/minimax-h3>;
  blog: <https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui>;
  community wiki: <https://comfyui-wiki.com/en/tutorial/advanced/video/minimax/minimax-h3>.
- Nodes (`comfy_extras/nodes_minimax_h3.py`):
  - `MiniMaxH3ImageToVideo` — **both t2v and i2v**: optional `first_frame` /
    `last_frame`; no frames connected = pure t2v. Outputs conditioning + packed
    AV latent.
  - `MiniMaxH3ReferenceToVideo` — ref2v; `ref_image_size` `match` (fast) vs
    `max` (2048px refs, several times slower — ref tokens ride every step).
  - `EmptyMiniMaxH3LatentAV`, `MiniMaxH3SigmaShift` (defaults video 12.0 /
    audio 3.0 — already model defaults; node exists for overrides).
- Template sampling config: `res_multistep` sampler + `simple` scheduler,
  **20 steps**, denoise 1.0, `BasicGuider` (no CFG), `CreateVideo` fps=24.
  Wiki: quality degrades <15 steps, marginal gains >25; `beta`/`normal`
  scheduler reportedly beats `simple` for reference-heavy r2v prompts.
- **Frame length rule**: `n % 17 == 5`, i.e. 5, 22, …, 124 (≈5 s), 243 (≈10 s),
  362 (≈15 s). Trained range 124–362; longer untested. Templates compute
  `max(5, round(sec*24))` snapped **up** to the grid.
- **Resolution rule**: multiples of 32 per axis, native area ≤ 768×1344 ≈ 1 MP.
  Practical floor ~384p. fps is fixed at 24 (baked into the model).
- Known pitfalls (Aug 2026): global `--use-sage-attention` produces noise on H3
  ([#15263](https://github.com/Comfy-Org/ComfyUI/issues/15263)) — use the KJNodes
  "Patch Sage Attention" node instead (~2x speedup reported); EasyCache degrades
  the audio stream ([#15326](https://github.com/Comfy-Org/ComfyUI/issues/15326));
  tiled-VAE and ref-video-encode OOM issues (#15312/#15274/#15246).

## Concurrency: what actually maximizes throughput

The finding that shapes everything: **one H3 generation saturates the GPU**
(DiT denoise ≈ 88% of request wall time per vLLM's own profiling). Consequences:

- Running N generations concurrently on one GPU ≈ N× the latency of one — no
  throughput win (SGLang's 2-outputs-per-prompt numbers scale near-linearly).
- Neither SGLang nor vLLM-Omni does cross-request batching for H3
  ("one generation per diffusion batch"), and ComfyUI's H3 latent node doesn't
  expose batch_size.
- ComfyUI executes its queue **serially** — one workflow at a time per instance.

So "as many concurrent generations as possible" = **one resident worker per
GPU, horizontal scale-out**, plus multi-GPU parallelism to cut per-video
latency where it matters. On a multi-GPU pod you can run one ComfyUI instance
per GPU with `--cuda-device N` pinning and separate ports.

## Running H3 without ComfyUI

Official repo [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)
documents four supported paths (original weights:
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)):

1. **SGLang — recommended resident server.**
   [Cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3).
   `uv pip install "sglang[diffusion]" --prerelease=allow`, then e.g.
   `sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va
   --num-gpus 4 --ulysses-degree 4 --performance-mode speed`.
   Model stays in VRAM; async REST job API (`POST /v1/videos` → poll →
   download). Published perf: 4x H200, 1344×768, 124 fr, 50 steps = **75 s**
   (54 s with Cache-DiT); 4x H100 (TP2+Ulysses2) is the fastest verified combo;
   online FP8 on B200/B300. Supports `num_outputs_per_prompt`.
2. **vLLM-Omni** — [recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3);
   sync endpoint `POST /v1/videos/sync`; ROCm docker for MI300X. FP8 not yet.
3. **diffusers — merged Aug 5 2026** ([PR #14355](https://github.com/huggingface/diffusers/pull/14355)).
   Modular Diffusers only (no classic `DiffusionPipeline`):
   `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")` →
   `MiniMaxH3ModularPipeline`; classes `MiniMaxH3Transformer3D`,
   `AutoencoderKLMiniMaxH3`, `MiniMaxH3Scheduler`. Install from git main until
   released. Best for custom logic; you manage offload/parallelism yourself.
4. **ComfyUI API mode** — what this repo uses today: headless `main.py`,
   models resident between jobs, `POST /prompt` + `/history` (see
   `bench/minimax_bench.py` for a zero-dependency client). Serial queue.

Also: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
(supports H3 incl. NF4, low-VRAM Python path). No TensorRT/xDiT/FastVideo
support found.

**Keep-loaded vs serverless**: keeping the model resident is what ComfyUI/SGLang
already do. For scale-to-zero economics:

- **RunPod Serverless has B200** (180 GB, $0.00240/s ≈ $8.64/hr; H200 $0.00155/s,
  H100 $0.00116/s — [docs](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)).
- Weights on a network volume load at ~200–400 MB/s → **cold start ~3–8 min**
  for 40–115 GB of weights. FlashBoot only helps with steady traffic, not true
  scale-from-zero ([RunPod blog](https://www.runpod.io/blog/serverless-gpu-cold-starts-flashboot)).
- Practical setup: [worker-comfyui](https://github.com/runpod-workers/worker-comfyui)
  or a custom `runpod` SDK handler wrapping SGLang/diffusers; **min-workers 1**
  turns it into a resident server with burst capacity — usually cheaper than
  eating multi-minute cold starts given 1–5 min generation times.
- Managed alternative: [fal.ai hosts H3](https://fal.ai/minimax-h3).

## Published performance reference points

| Setup                                             | Config                     | Time                      |
|---------------------------------------------------|----------------------------|---------------------------|
| 4x H100 80GB (SGLang TP2+Ulysses2)                | 1344×768, 124 fr, 50 steps | 13.3 s                    |
| 8x B300 (SGLang, bf16 / online FP8)               | same                       | 19.0 / 18.0 s             |
| 4x H200 (SGLang Ulysses4)                         | same                       | 75.1 s (53.7 s Cache-DiT) |
| 2x RTX 5090 (layerwise offload, ~380 GB host RAM) | same                       | 560 s                     |
| RTX 4090 Laptop 16 GB (SageAttention)             | 960×540, 5 s, 20 steps     | 182 s                     |
| RTX 3060 12 GB (heavy offload)                    | 864×480, 124 fr, 20 steps  | <9 min                    |

Our single-B200 ComfyUI numbers: see `docs/BENCHMARKS.md`.

## Recommendations for storyteller

1. **Today (single B200 pod)**: ComfyUI headless + API is fine for dev and
   already keeps weights resident; use `bench/minimax_bench.py` patterns for
   programmatic generation. Pruned int8 quality is strong and it's the
   fastest/smallest; keep bf16 for quality A/Bs.
2. **Production serving**: SGLang serve is the purpose-built path (async job
   API, multi-GPU, Cache-DiT). Evaluate 4x H100/H200 vs 1x B200 per-video
   economics — multi-GPU Ulysses cuts latency 4–6x for ~4x the GPU cost.
3. **Concurrency**: scale horizontally (1 worker/GPU). Don't co-schedule two
   generations on one GPU.
4. **Serverless**: viable on RunPod with B200 + network volume + min-workers 1;
   pure scale-from-zero pays 3–8 min cold starts. Consider baking pruned-int8
   weights (~42 GB stack) into the image to cut that.

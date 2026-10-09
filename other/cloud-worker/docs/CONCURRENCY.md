# MiniMax H3 — concurrency & cost-per-video report

Written 2026-08-06. Measured numbers come from our B200 pod
(`bench/results.csv`); everything marked *est.* is extrapolated and needs
validation on real hardware. Provider/worker-availability research is a
follow-up task.

## 1. Can one GPU run two generations at once?

**Compute: no meaningful headroom.** During denoising — 85–90% of wall time —
the H3 DiT saturates the GPU's SMs. Two co-scheduled jobs time-slice and each
run ~2× slower; total throughput is ~flat. This matches vLLM's own profiling
(DiT ≈ 88% of request time) and SGLang's `num_outputs_per_prompt` scaling,
which is near-linear in time (2 outputs ≈ 2× one output's duration).

**What is NOT saturated:**

| Resource                       | Observation on B200                                  |
|--------------------------------|------------------------------------------------------|
| VRAM                           | 48–96 GB peak of 180 GB (pruned int8, by resolution) |
| Non-denoise phases             | text encode, VAE decode, mux, queue gaps: ~10–15%    |

**Utilization caps (MPS / MIG):** the B200 supports both — MPS SM partitioning
(`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`) and MIG (e.g. 2 × 90 GB instances, each
big enough for the pruned-int8 stack). But partitioning divides FLOPs, it
doesn't create them: two half-GPU workers each generate ~2× slower.

**MEASURED (2026-08-06, two ComfyUI instances on the B200):** a sequential
pair of warm 5 s t2v jobs took **128 s**; the same pair run concurrently took
**145 s** — co-scheduling was **13% slower** for identical work (each
concurrent job: 142 s vs 63 s alone). The hypothesized 10–20% pipeline-overlap
gain does not exist in practice; contention eats it and more. Details in
`docs/BENCHMARKS.md`.

**Bottom line (now empirical):** concurrency = number of GPUs. One resident
worker per GPU, queue in front, scale horizontally. Never co-schedule; skip
MPS/MIG for throughput purposes.

## 2. Cost per video across GPU tiers

Reference workload: **5 s clip, 864×480, 20 steps, pruned int8** (the
cheapest-to-run family; quality holds up well in our tests).

Measured baseline on B200 (this pod): **66 s** warm generation.
Scaling anchors from the community/vendors: RTX 4090 Laptop 960×540 ≈ 182 s
(SageAttention); 2×5090 via SGLang layerwise offload, 1344×768/50 steps ≈ 560 s;
RTX 3060 12 GB 864×480 < 9 min. Everything else is interpolated from relative
bf16/int8 compute throughput and should be treated as ±40%.

Prices are RunPod list (2026-08, Community | Secure) with Vast.ai market
ranges where relevant.

| GPU              | VRAM   | $/hr (community) | $/hr (secure) | time/video      | videos/GPU-hr | $/video (community) |
|------------------|--------|------------------|---------------|-----------------|---------------|---------------------|
| B200             | 180 GB | $5.89            | $5.89         | **66 s (meas.)**| 54            | $0.108              |
| H200             | 141 GB | $3.59            | $4.39         | ~85 s *est.*    | ~42           | ~$0.085             |
| H100 SXM        | 80 GB  | $2.69            | $2.99         | ~150 s *est.*   | ~24           | ~$0.112             |
| H100 PCIe       | 80 GB  | $1.99            | $2.89         | ~175 s *est.*   | ~21           | ~$0.095             |
| A100 SXM        | 80 GB  | $1.39            | $1.49         | ~300 s *est.*   | ~12           | ~$0.116             |
| L40S             | 48 GB  | $0.79            | $0.99         | ~300 s *est.*   | ~12           | ~$0.066             |
| RTX Pro 6000     | 96 GB  | $1.69            | $1.99         | ~110 s *est.*   | ~33           | ~$0.051             |
| RTX 6000 Ada     | 48 GB  | $0.74            | $0.77         | ~250 s *est.*   | ~14           | ~$0.051             |
| RTX 5090         | 32 GB  | $0.69            | $0.99         | ~120 s *est.*   | ~30           | ~$0.023             |
| RTX 4090         | 24 GB  | $0.34            | $0.69         | ~180 s *est.*   | ~20           | ~$0.017             |
| RTX 3090         | 24 GB  | $0.22            | $0.46         | ~450 s *est.*   | ~8            | ~$0.028             |

Vast.ai market rates run lower still: RTX 5090 ~$0.30–0.60/hr, RTX 4090 from
~$0.20/hr — pushing the 4090 toward **~$0.010/video**, an order of magnitude
below the B200.

## 3. What the table implies

1. **Cheapest concurrency: fleets of 4090/5090-class cards** running pruned
   int8. ~5–10× cheaper per video than big-iron GPUs, if 2–4 min/video latency
   is acceptable. Concurrency N = N cards, at ~$0.25–0.70/hr each.
2. **Big GPUs buy latency and capability, not economy**: B200/H200 for fast
   turnaround, bf16 quality, full-res 15 s clips, or 8-ref workloads (our 8-ref
   full-res runs peaked near 96 GB — that config simply doesn't fit consumer
   cards).
3. **The dollar-optimal fleet is probably mixed**: consumer-card pool for bulk
   draft/short generations + a small big-GPU pool for premium/long/ref-heavy
   jobs.

## 4. Caveats before betting on the cheap tier

- **VRAM offloading**: 24–32 GB cards need ComfyUI's RAM offload for the
  text encoder; host RAM (64 GB+) and PCIe bandwidth matter. Times above
  include typical offload overhead but vary with host specs.
- **RunPod Community Cloud has no network volumes** (Secure Cloud only) — each
  community worker must pull ~42 GB of pruned-int8 weights at boot (or bake
  them into the image). At consumer-pod bandwidth that's tens of minutes of
  cold start; image-baking is near-mandatory.
- **Interruptibility/reliability**: community/marketplace pods can disappear;
  the queue layer must tolerate worker churn.
- **Quality parity of pruned int8 vs bf16** at these settings is exactly what
  the current benchmark chain is quantifying — check `docs/BENCHMARKS.md`.
- All *est.* numbers need a 30-minute validation pass per GPU type with
  `bench/minimax_bench.py --suite key` on a rented instance of each.

## Sources

- Our measurements: `bench/results.csv` (B200 pod, ComfyUI 0.30, 2026-08-06)
- [RunPod pricing](https://www.runpod.io/pricing) (retrieved 2026-08-06)
- [Vast.ai RTX 5090 pricing](https://vast.ai/pricing/gpu/RTX-5090) ·
  [Vast.ai live pricing guide](https://vast.ai/article/how-much-does-it-cost-to-rent-a-gpu-in-the-cloud-live-pricing-guide) ·
  [getdeploying.com 5090 comparison](https://getdeploying.com/gpus/nvidia-rtx-5090)
- SGLang MiniMax-H3 cookbook (multi-GPU + 2×5090 offload timings):
  https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3
- vLLM-Omni MiniMax-H3 recipe (DiT ≈ 88% of request time):
  https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3
- Community datapoints: [kingy.ai benchmarks](https://kingy.ai/news/minimax-h3-benchmarks-specs-hardware-review/),
  [HF discussion #3](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/3)

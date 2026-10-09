# MiniMax H3 — measured benchmarks

Measured 2026-08-06 with `bench/minimax_bench.py` through the ComfyUI API.
Raw data: `bench/results.jsonl` (B200), `bench/4090/`, `bench/4000/`.
Regenerate the tables with `python3 bench/summarize.py`.

**Method.** ComfyUI 0.30 headless; template-reference sampling everywhere:
`res_multistep` sampler, `simple` scheduler, **20 steps**, no CFG, 24 fps,
`nvfp4_awq` Qwen3-VL-32B text encoder, fp16 video VAE + fp32 audio VAE.
Times are ComfyUI execution time (`execution_start`→`execution_success`), which
excludes client/queue overhead (wall time tracked separately in the raw data;
typically +1–2 s). "Warm" = model already resident; first-load runs are
recorded but excluded from the means. Repeated configs use distinct seeds so
ComfyUI's node cache can't short-circuit. Reference inputs are real images
(film stills, anime characters, forests/volcanos, up to 23040×3840 —
`scripts/pod/fetch-bench-images-v2.py`).

Hardware: **B200** 180 GB (US, $6.94/hr) · **RTX 4090** 24 GB (EU, $0.74/hr) ·
**RTX PRO 4000 Blackwell** 24 GB (EU, $0.57/hr). The EU cards run pruned int8
only (bf16/int8 don't fit 24 GB usefully) with `--reserve-vram 2.5`; the RTX
4000 additionally needs `--cache-none --disable-pinned-memory` because its
container has a 29 GB RAM cap that otherwise OOM-kills weight staging.

## B200: weight-family comparison (warm, 20 steps)

The headline: **bf16 is the fastest family on the B200** — quantization costs
speed here (dequant overhead) and only pays when VRAM is scarce.

| Config                          | bf16        | int8        | pruned int8 | bf16 VRAM | int8 VRAM | pruned VRAM |
|---------------------------------|-------------|-------------|-------------|-----------|-----------|-------------|
| t2v 864×480 5 s                 | **51 s**    | 63 s        | 63 s        | 159 GB    | 63 GB     | 69 GB       |
| t2v 864×480 15 s                | **240 s**   | 270 s       | 270 s       | 182 GB    | 101 GB    | 75 GB       |
| t2v 1344×768 5 s                | **169 s**   | 197 s       | 194 s       | 177 GB    | 99 GB     | 73 GB       |
| ref2v 1 ref 864×480 5 s         | **54 s**    | 68 s        | ~68 s¹      | 186 GB    | 95 GB     | 68 GB       |
| ref2v 4 refs match              | **63 s**    | 86 s        | 81 s        | 154 GB    | 95 GB     | 70 GB       |
| ref2v 8 refs match              | **79 s**    | 99 s        | 95 s        | 185 GB    | 96 GB     | 68 GB       |
| ref2v 8 refs match 1344×768     | **282 s**   | 335 s       | 339 s       | 158 GB    | 103 GB    | 77 GB       |
| ref2v 4 refs max                | **91 s**    | 117 s       | 112 s       | 156 GB    | 96 GB     | 70 GB       |
| ref2v 8 refs max                | **785 s**   | 891 s       | 841 s       | 177 GB    | 112 GB    | 92 GB       |

¹ pruned 1-ref mean is 90±41 s over 3 runs because the first run absorbed a
model swap; the steady-state runs are ~68 s, matching int8.

bf16 peaks at 155–187 GB — it *only* runs on ≥180 GB cards. int8 fits under
112 GB (H200-class); pruned int8 under ~95 GB worst-case, ~70 GB typical
(H100-class without offload, 24 GB cards with offload).

## Scaling behavior (B200, pruned int8)

Duration (864×480 t2v): 5 s → 63 s, 10 s → 154 s, 15 s → 270 s. Cost per
output-second rises from 12.6→15.4→18.0 s — mildly superlinear (attention).

Resolution (5 s t2v): 608×352 → 32 s, 864×480 → 63 s, 1344×768 → 194 s.
Roughly ∝ pixels^1.2. The corner case 1344×768×15 s = **1108 s** (18.5 min).

Reference count (864×480 5 s, `match` sizing): 1 ref ≈ 68 s, 4 refs ≈ 81 s,
8 refs ≈ 95 s — ~+4 s per extra reference image. Cheap.

Reference sizing `max` is the expensive lever: with 8 large refs it goes
95 s → **841 s** (~9×), because 2048px reference tokens ride through every
sampling step. Use `match` unless identity fidelity demands otherwise.

## Cross-GPU (pruned int8, warm, 20 steps — final measured)

t2v 864×480 5 s:

| GPU              | $/hr  | time/video       | videos/hr | $/video | notes                              |
|------------------|-------|------------------|-----------|---------|------------------------------------|
| B200 (bf16)      | $6.94 | 51 s (n=2)       | 70        | $0.098  | fastest family on this card        |
| B200 (pruned)    | $6.94 | 63 s (n=3)       | 57        | $0.121  |                                    |
| RTX 4090         | $0.74 | 222 s (n=3, ±0)  | 16        | $0.046  | --reserve-vram 2.5                 |
| RTX PRO 4000     | $0.57 | 553 s (n=3, ±3)  | 6.5       | $0.087  | --cache-none tax (29 GB RAM cap)   |

ref2v single ref, 864×480 5 s:

| GPU              | time/video       | $/video |
|------------------|------------------|---------|
| B200 (bf16)      | 54 s (n=3)       | $0.104  |
| RTX 4090         | 287–392 s        | $0.059–0.081 |
| RTX PRO 4000     | 581 s (n=3, ±6)  | $0.092  |

**RTX 5090 (32 GB, $0.99/hr) — added later, the fleet workhorse:**

| Config (pruned int8, warm)  | time             | $/video | notes                              |
|-----------------------------|------------------|---------|------------------------------------|
| t2v 864×480 5 s             | 143 s (n=3, ±1)  | $0.039  | cheapest t2v measured              |
| ref2v 1 ref                 | 153 s (n=3, ±0)  | $0.042  |                                    |
| ref2v 8 refs full-size      | 220 s (n=3, ±1)  | $0.060  | vs $0.152 B200-bf16: 2.5× cheaper  |
| ref2v 9 refs full-size      | 237 s            | $0.065  | max ref count                      |
| t2v 15 s                    | 582 s            | $0.160  | works (killed 24 GB pre-patch)     |
| t2v 1344×768                | 455 s            | $0.125  | works                              |
| 9:16 480×864 / 1:1 640×640  | 187 s / 141 s    | —       | aspects fine                       |
| ref2v 9 refs `max` sizing   | 2031 s (34 min)  | $0.559  | works; avoid in production         |

**bf16 on the 5090** (66 GB DiT layer-streamed through 32 GB): warm t2v =
**110 s — faster than its own pruned int8 (143 s)**; ref2v 1/4/8 refs =
310/336/369 s (needs `--cache-none` when switching between bf16 models —
two 66 GB stacks break the 124 GB container RAM cap). B200-quality output
at $0.99/hr for everything except 15 s + full-res extremes.

**Verdict:** the 5090 dominates cost AND capability among consumer cards —
cheapest per video on every task shape, full reference envelope, and a
credible bf16 quality tier. The patched 4090 (see MITIGATIONS.md) is a
viable budget multi-ref tier (~8-9 refs at 429–444 s). The RTX PRO 4000's
low price is consumed by its slow, reload-heavy runs — dropped. Fleet
recipe: 5090 pools for volume, B200 pool only for 15 s/full-res premium
jobs and fastest-latency needs.

## Concurrency (2 ComfyUI instances, one B200)

See `docs/CONCURRENCY.md` for the model. Experiment: two resident instances,
sequential pair vs concurrent pair of identical warm t2v jobs.

Measured (pruned int8, t2v 864×480 5 s, both instances warm):

| Mode                          | Total for 2 videos | Per-video effective |
|-------------------------------|--------------------|---------------------|
| Sequential (one instance)     | **128 s**          | 64 s                |
| Concurrent (two instances)    | **145 s**          | 72.5 s              |

Concurrent was **13% slower** for the same work — each co-scheduled job ran
142 s vs 63 s alone (2.27×). There is no pipeline-overlap win; cache and
scheduler contention make co-scheduling strictly worse. **One worker per GPU,
queue in front, scale horizontally.** (This also removes any argument for
MPS/MIG partitioning for throughput.)

## Reliability & failure statistics

Every run's end state and error is recorded in the JSONL files; regenerate the
failure tables with `python3 bench/failures.py`, and validate output files
with `python3 bench/validate_outputs.py` (downloads each MP4 and ffprobes
streams + duration). Snapshot as of 2026-08-06 ~09:00 UTC:

| GPU               | success rate | dominant failures                                     |
|-------------------|--------------|-------------------------------------------------------|
| B200 180 GB       | 96/96 (100%) | none                                                  |
| RTX 4090 24 GB    | 9/53 (17%)   | VRAM OOM: TE vision (multi-ref), DiT sampling (refs/15s/full-res); cascading server death |
| RTX PRO 4000 24 GB| 7/12 (58%)   | container-RAM OOM (pre-flags); corrupt ref image (fixed); server death on 15 s t2v |

**Output validation**: 100% of successful runs produced valid videos (h264 +
stereo AAC at exactly the requested duration) — every success on all three
GPUs checks out. (Three early B200 outputs 404 on re-download because they
were written to the container disk that the pod reset wiped — outputs now go
to the persistent volume via `--output-directory`.)

**The 24 GB failure pattern is reference images, not duration/resolution.**
t2v succeeds reliably on both consumer cards at every size tried; ref2v OOMs
in `torch._int_mm` during Qwen3-VL image encoding: the 15 GB text encoder plus
multi-image vision activations exceeds 24 GB (observed 22.1 GB peak at OOM).
With aggressive flags (`--cache-none --disable-pinned-memory --reserve-vram
2.5`) a **single** reference works (4090: 392 s); 4-ref and 8-ref
configurations fail consistently, and repeated OOMs eventually killed the
server process, cascading failures across the rest of that suite (harness now
survives this and records `server_died`).

A second environment trap on cheap tiers: the RTX 4000 pod's container has a
**29 GB RAM cap** that OOM-killed weight staging until `--cache-none
--disable-pinned-memory` was applied; both EU pods also shipped torch 2.4.1,
too old for H3 (needs ≥2.5; Blackwell needs ≥2.7+cu128).

**Fleet implication:** consumer 24 GB cards are viable for t2v/i2v and
single-ref ref2v only. Multi-reference work — the priority modality — needs
≥32 GB (5090-class, unverified) and realistically ≥48 GB for headroom, or it
stays on big-GPU pools.

## Multi-ref rescue status (in progress)

Community research (see agent report summarized in commit history) identified
the root cause as **ComfyUI PR #15316**: VRAM is budgeted for TE weights but
not image-encode activations. Two OOM phases were isolated on the 4090:

1. **TE vision encode** — fixed by ref pre-downscaling (`--ref-downscale`,
   now in the harness) and/or the #15316 patch + higher `--reserve-vram`.
   Confirmed: with 0.1–0.2 MP refs the TE phase completes.
2. **DiT sampling with ref tokens** — the 20 GB DiT + ref-token-enlarged
   attention doesn't fit 24 GB. `--lowvram` made things worse (crashes).
   Remaining candidates: #15316 patch active + `--reserve-vram 5` (patched
   file is applied on both EU pods, pending verification), TE eviction via
   ComfyUI-H3-Multishot, or a smaller DiT quant (community INT4/W4A8,
   ~10–12 GB, quality TBD).

Verification was interrupted when all three pods went offline simultaneously
(apparent RunPod account event ~09:30 UTC). Resume plan lives in the repo:
restart pods → `scripts/local/push-to-pod.sh` → rescue matrix via
`bench/minimax_bench.py --suite refheavy --ref-downscale 0.2`.

## Takeaways

1. **On the B200, run bf16.** Fastest and highest quality; quantized families
   exist for smaller cards, not for speed.
2. **Many references are cheap; big references are not.** 8 refs at `match`
   costs ~1.4× a single ref. `max` sizing costs up to ~9×. Default `match`.
3. **Duration and resolution both scale superlinearly**; the trained envelope
   corner (15 s @ 1344×768) costs ~18 min even on a B200 — treat full-res
   long clips as premium jobs.
4. **24 GB consumer cards work but pay a heavy offload tax** (~3.5× B200
   latency rather than the ~3× raw-compute ratio), and cheap pod tiers bring
   operational traps: container RAM caps (29 GB on the RTX 4000 pod) and
   stock torch builds too old for H3 (needs ≥2.5 for `enable_gqa`; Blackwell
   needs ≥2.7+cu128).

# MiniMax H3 serving plan — pools, pricing, capacity scenarios

Prepared 2026-08-06. Every generation time in this document is **measured** on
our pods (see `docs/BENCHMARKS.md`); derived numbers state their formula.
GPU prices are what we actually pay: B200 $6.94/hr, RTX 5090 $0.99/hr,
RTX 4090 $0.74/hr. All jobs are 20 steps, 24 fps, template sampler settings.

Two structural facts drive everything:

1. **One generation saturates one GPU** (measured: co-scheduling is 13%
   *slower*). Throughput = number of workers; latency = service time + queue.
2. **bf16 is the fastest family wherever it fits** — including, surprisingly,
   layer-streamed on the RTX 5090 (110 s vs 143 s pruned for the standard
   clip). Quantized families are for VRAM-constrained multi-model workers,
   not for speed or cost.

## The product surface (what we can offer)

**Modalities**: text-to-video, keyframe-to-video (first and/or last frame),
reference-to-video (1–9 images; also up to 3 ref videos + 3 audio clips,
untested by us). All produce video + native stereo audio.

**Resolutions** (canvas must be multiples of 32, area ≤ ~1.03 MP):

| Tier          | 16:9      | 9:16      | 1:1       | Notes                    |
|---------------|-----------|-----------|-----------|--------------------------|
| Draft (352p)  | 608×352   | 352×608   | 448×448   | fastest, preview quality |
| SD (480p)     | 864×480   | 480×864   | 640×640   | the sweet spot           |
| HD (768p)     | 1344×768  | 768×1344  | 992×992   | native max; ~3.3× cost   |

Intermediate 16:9 rungs exist every 32 px (736×416, 960×544, 1056×608,
1152×640, 1216×672, 1280×736) — we recommend selling the three tiers above.
Other aspects (4:3, 21:9, …) work too — anything multiple-of-32 under the
area cap. 2K does not exist locally (hosted-API upscale only).

**Durations**: frame count must satisfy n ≡ 5 (mod 17); trained range 124–362
frames → **~5.2 s to ~15.1 s in ~0.7 s steps** (5.2, 5.9, 6.6, 7.3, 8.0, 8.7,
9.4, 10.1, 10.8, 11.5, 12.2, 12.9, 13.7, 14.4, 15.1). Sell "5 / 10 / 15
seconds"; internally snap up to the grid. fps is fixed at 24.

## Pool 1 — Paid (bf16, everything)

### Worker type

**B200 (180 GB) is the only worker class that runs the entire paid surface
on bf16** — HD, 15 s, and heavy reference jobs all need its VRAM. However,
55–70 % of paid traffic (≤480p, ≤10 s) runs **4× cheaper on 5090-bf16
workers** at identical weights/quality. Recommended deployment:

- **5090 bf16 pool** — handles t2v/i2v/ref2v at ≤480p, ≤10 s (dedicated
  single-model workers; no model switching, so no cache-none tax).
- **B200 bf16 pool (small)** — handles HD, >10 s, and `max`-sizing reference
  jobs; also absorbs overflow for latency.

Route by job shape. This hybrid cuts blended cost ~2.5–3× vs all-B200.

### Cost per video (B200 bf16, measured/derived)

t2v baseline; keyframe (i2v) ≈ +5 %; reference adder below.

| Resolution | 5 s              | 10 s             | 15 s             |
|------------|------------------|------------------|------------------|
| Draft 352p | 26 s → $0.050    | 63 s → $0.122    | 111 s → $0.214   |
| SD 480p    | 51 s → $0.098    | 125 s → $0.240   | 240 s → $0.463   |
| HD 768p    | 169 s → $0.326   | 490 s → $0.945   | 965 s → $1.861   |

(10 s and 352p/HD-longer cells derived from measured duration/resolution
scaling on identical hardware; 480p-5/15 s, HD-5 s are directly measured.)

Reference-to-video adder (match sizing): **+$0.006 (1 ref) to +$0.054
(8–9 refs)** at ≤480p; **+$0.22** at HD with 8 refs. `max` reference sizing
multiplies job cost ~5–10× — if offered at all, price it as its own SKU at
~5× the base price, or restrict to ≤4 refs (+$0.08).

On the 5090-bf16 pool the same ≤480p jobs cost: 5 s $0.030 · 10 s ~$0.074 ·
1-ref +~$0.01. Blend accordingly; the pricing below uses **B200 costs** as
the conservative basis, so hybrid routing only improves your real margin.

### Pricing — per output-second (the unit to reason in)

All prices are **$ per second of delivered video**. Margin means gross margin
on price (price = cost ÷ (1 − margin)). Note the honest wrinkle: our cost per
output-second RISES with clip length (compute is superlinear in frames), so a
truly flat per-second price quietly gives long clips a discount — the tables
show exact per-duration cost so you can decide flat vs tiered.

**Our cost per output-second (B200 bf16 basis), and price at each margin:**

| Tier  | Clip  | Cost/s   | 0%      | 15%     | 25%     | 50%     | 70%     |
|-------|-------|----------|---------|---------|---------|---------|---------|
| 352p  | 5 s   | $0.0097  | $0.0097 | $0.0114 | $0.0129 | $0.0194 | $0.0323 |
| 352p  | 10 s  | $0.0120  | $0.0120 | $0.0141 | $0.0160 | $0.0240 | $0.0400 |
| 352p  | 15 s  | $0.0142  | $0.0142 | $0.0167 | $0.0189 | $0.0284 | $0.0473 |
| 480p  | 5 s   | $0.0190  | $0.0190 | $0.0224 | $0.0254 | $0.0380 | $0.0634 |
| 480p  | 10 s  | $0.0238  | $0.0238 | $0.0280 | $0.0317 | $0.0476 | $0.0793 |
| 480p  | 15 s  | $0.0307  | $0.0307 | $0.0361 | $0.0409 | $0.0614 | $0.1023 |
| 768p  | 5 s   | $0.0630  | $0.0630 | $0.0741 | $0.0840 | $0.1260 | $0.2101 |
| 768p  | 10 s  | $0.0932  | $0.0932 | $0.1097 | $0.1243 | $0.1865 | $0.3108 |
| 768p  | 15 s  | $0.1234  | $0.1234 | $0.1451 | $0.1645 | $0.2467 | $0.4112 |

**Hybrid-routing true cost** (≤480p/≤10 s jobs on 5090-bf16 workers):
480p drops to **$0.0059/s (5 s)** and **$0.0073/s (10 s)** — roughly 3×
cheaper than the B200 basis above. If you price from the B200 table and route
hybrid, realized margin is far above nominal (e.g. a "25 %" 480p price
carries ~77 % true margin on the 5090 share).

**Flat-rate menu option** (blended 60/25/15 duration mix, hybrid routing):

| Public flat price   | True blended margin              |
|---------------------|----------------------------------|
| SD 480p @ $0.02/s   | ~50 %                            |
| SD 480p @ $0.03/s   | ~67 %                            |
| HD 768p @ $0.10/s   | ~20 % (HD stays on B200 — price HD at $0.13–0.17/s for 40–50 %) |
| Draft 352p @ $0.01/s | ~0 % on B200, ~60 % if drafts route to 5090s |

Per-video equivalents of the B200-basis table (for invoice display):

| Duration | Cost   | 0%     | 15%    | 25%    | 50%    | 70%    |
|----------|--------|--------|--------|--------|--------|--------|
| 5 s      | $0.098 | $0.098 | $0.116 | $0.131 | $0.197 | $0.328 |
| 10 s     | $0.240 | $0.240 | $0.283 | $0.320 | $0.481 | $0.801 |
| 15 s     | $0.463 | $0.463 | $0.545 | $0.617 | $0.926 | $1.543 |

| SKU modifiers                | ×cost | example @25% margin, 10 s      |
|------------------------------|-------|--------------------------------|
| Draft 352p                   | ×0.51 | $0.163                         |
| HD 768p                      | ×3.6+ | $1.260 (5 s: $0.434, 15 s: $2.481) |
| + references (≤9, match)     | +$0.01–0.06 | round: +$0.05            |
| + references at HD           | +$0.22 | round: +$0.30                 |
| keyframe (first/last frame)  | +5 %  | fold into base                 |

Simple public menu suggestion (at ~50 % blended margin, knowing hybrid
routing makes true margin higher): **Draft $0.10 · SD $0.20/5 s · HD $0.65/5 s;
+$0.05 per clip with references; duration billed per 5 s block.**

## Pool 2 — Free (480p, 5 s, ≤3 cropped refs)

Constraint set: 864×480 (any of the three aspects), 5 s, up to 3 reference
images pre-cropped/downscaled (our `--ref-downscale 0.2` path — measured to
also *improve* stability and speed).

### Cost per 1,000 free generations (measured)

| Worker + weights              | time/job | jobs/hr/worker | $/1k videos |
|-------------------------------|----------|----------------|-------------|
| **5090 bf16 (t2v/i2v only)**  | 110 s    | 32.7           | **$30**     |
| 5090 pruned int8, t2v         | 143 s    | 25.2           | $39         |
| **5090 pruned int8, 3 refs**  | ~176 s   | 20.5           | **$48**     |
| 4090 patched, 3 refs          | ~400 s   | 9.0            | $82         |
| B200 bf16, 3 refs             | 60 s     | 60.0           | $116        |

int8 (non-pruned) sits between pruned and bf16 in speed on every card and
fits nothing extra — it has no serving niche; skip it. **Recommendation:
free pool = 5090s running pruned int8 with the ref2va model resident**
(one model handles ref jobs; t2v works via 0-ref… no — t2v needs fl2va).
Practical split: ~⅓ of free workers hold fl2va (t2v/i2v jobs, $39/1k), ~⅔
hold ref2va (ref jobs, $48/1k) — no model switching, no cache-none tax,
blended **~$45 per 1,000 free videos**. bf16-on-5090 is competitive for
t2v-only but its ref2v carries the switch tax; revisit if free tier drops
references.

Quality note: pruned int8's output is strong (community: "near-lossless
lineage"), and free users get 480p/5 s — the quality delta vs bf16 at this
tier is minimal. If you want bf16 everywhere anyway, free costs rise ~25 %.

## Capacity & demand scenarios

Model: workers sized so that **peak-hour utilization ≤ 70 %** (queueing knee).
Peak hour assumed 3× the daily average rate. Wait ≈ service_time × ρ/(1−ρ)
per M/M/1 worker; at 70 % that's ~2.3× service time queued ahead of you —
the sizing keeps typical waits under ~1 job-length even at peak.

Wait-time intuition (176 s free ref job):

| Utilization | Avg queue wait |
|-------------|----------------|
| 50 %        | ~3 min         |
| 70 %        | ~7 min         |
| 85 %        | ~17 min        |
| 95 %        | ~56 min        |

### Free pool (5090 pruned, blended 165 s/job, 21.8 jobs/hr/worker)

| Demand (videos/day) | Peak rate/hr | Workers (70 % peak) | $/day (24/7) | $/day (autoscaled*) | Mean-hour wait | Peak wait |
|---------------------|--------------|----------------------|--------------|----------------------|----------------|-----------|
| 1,000               | 125          | 9                    | $214         | ~$95                 | <1 min         | ~6 min    |
| 5,000               | 625          | 41                   | $974         | ~$430                | <1 min         | ~6 min    |
| 20,000              | 2,500        | 164                  | $3,897       | ~$1,700              | <1 min         | ~6 min    |
| 100,000             | 12,500       | 820                  | $19,483      | ~$8,600              | <1 min         | ~6 min    |

*Autoscaled ≈ 44 % of 24/7 (integrate a 3:1 peak:trough sinusoidal day at
70 % target util). Requires image-baked or volume-attached workers that cold
start in ~4 min (measured: container rebuild ~4 min + first model load
~40 s on a warm volume).

Rule of thumb: **free tier costs ~$45 per 1,000 videos served, i.e.
~$0.0087 per output-second** — the worker math above just determines how
fast you serve them. (For comparison: every paid SD second sold at $0.02
funds ~2.3 free seconds.)

### Paid pool (hybrid; blended job assumed 60 % SD-5s / 20 % SD-10s / 10 % HD-5s / 10 % ref-heavy)

Blended: ~$0.11 cost per job on hybrid routing (~$0.16 all-B200), ~150 s
blended service time on the 5090 share, 84 s on the B200 share.

| Demand (jobs/day) | 5090 workers | B200 workers | $/day (24/7) | Revenue/day @25 % margin | @50 % |
|--------------------|--------------|--------------|--------------|---------------------------|--------|
| 500                | 3            | 1            | $238         | $73 rev / $55 cost*       | $110   |
| 2,500              | 12           | 2            | $618         | $367 / $275               | $550   |
| 10,000             | 46           | 5            | $1,926       | $1,467 / $1,100           | $2,200 |
| 50,000             | 229          | 22           | $9,105       | $7,333 / $5,500           | $11,000 |

*Cost column is per-job cost × volume (what you actually burn on GPU-seconds);
the $/day worker column is capacity cost at 24/7 — the gap is idle headroom,
recovered by autoscaling (×0.44) or by letting free-tier jobs soak idle paid
workers (recommended: one queue, two priorities — paid preempts free).

**Break-even insight**: at 25 % margin the paid pool only covers its 24/7
capacity above ~4,000 jobs/day; below that, either autoscale, raise margin,
or (best) run paid and free as one fleet with priority scheduling so paid
headroom serves free demand instead of idling.

## Concrete deployment recommendation

1. **One fleet, two queues** (paid priority, free backfill) on **RTX 5090
   workers in one region** sharing a network volume: bf16-fl2va workers for
   paid t2v/i2v ≤480p/≤10 s, pruned-ref2va workers for all reference jobs
   and free tier.
2. **Small B200 pool** (1–2 + burst) for HD / 15 s / max-sizing paid SKUs.
3. Config per `docs/MITIGATIONS.md`: 5090s need the #15316 patch +
   `--reserve-vram 4`; keep one model per worker (no switching) to avoid
   every RAM-cap failure mode we found.
4. Autoscale on queue depth; workers cold-start in ~5 min from a shared
   volume (bake the container image to cut pip-install; then ~1 min).
5. Watch for ComfyUI merging #15316 and the KJNodes SageAttention patch
   (~2× potential speedup, currently broken for H3) — either would shift
   every number in this doc favorably.

## Assumptions register (audit before betting real money)

- GPU prices: our current RunPod rates; community-cloud 5090s can be ~30 %
  cheaper, secure-cloud availability at scale unverified (provider/worker-
  availability research is a pending follow-up).
- 10 s and 352p/HD-long cells derived from measured scaling laws, not
  directly timed on bf16; ±15 % error bars.
- 3-ref free-tier time interpolated between measured 1-ref and 4-ref runs.
- Peak:average = 3:1 assumed; measure your real diurnal curve and re-run
  the sizing (formulas inline above).
- No egress/storage/queue-infra costs included (small: ~1–3 MB per video).
- Quality parity of pruned int8 for the free tier is a judgment call —
  A/B the actual outputs (they're all on the pods' volumes).

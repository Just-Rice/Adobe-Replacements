#!/usr/bin/env python3
"""Benchmark MiniMax H3 video generation through the ComfyUI API.

Builds API-format graphs programmatically (no workflow JSON needed) for the
three local H3 task types and times each generation:

  t2v    MiniMaxH3ImageToVideo with no frames connected (pure text)
  i2v    MiniMaxH3ImageToVideo with a first_frame image
  ref2v  MiniMaxH3ReferenceToVideo with one reference image

Model facts baked in (see docs/RESEARCH.md):
  - frame length must satisfy n % 17 == 5; trained range 124-362 (~5-15 s @ 24fps)
  - canvas: multiples of 32, native area <= 768*1344 (~1 MP)
  - reference sampler config: res_multistep + simple scheduler, 20 steps, no CFG

Usage examples (through the SSH tunnel, or on the pod with --host):
  python bench/minimax_bench.py --suite quick
  python bench/minimax_bench.py --suite sweep --model fl2va=minimax_h3_fl2va_pruned_int8_convrot.safetensors \
      --model ref2va=minimax_h3_ref2va_pruned_int8_convrot.safetensors
  python bench/minimax_bench.py --task t2v --width 864 --height 480 --seconds 5 --steps 20

Results are appended to bench/results.csv (one row per run) and full metadata
to bench/results.jsonl. Output videos are saved on the ComfyUI side under
output/bench/.
"""
import argparse
import csv
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_HOST = "http://127.0.0.1:8188"

TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
FL2VA_DEFAULT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_DEFAULT = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
# Real images fetched by scripts/pod/fetch-bench-images-v2.py: Jurassic Park
# stills, anime characters (Ashitaka, Spike), forests, volcanos — large files
# (up to 23040x3840) for realistic reference-encoder load.
BENCH_IMAGE = "bench2_jp_01.jpg"
# Node schema allows up to 9 reference images (plus 3 ref videos + 3 audios).
BENCH_REF_SET = ["bench2_jp_01.jpg", "bench2_jp_gate_01.png",
                 "bench2_ashitaka_01.jpg", "bench2_bebop_01.jpg",
                 "bench2_forest_01.jpg", "bench2_forest_02.jpg",
                 "bench2_volcano_01.jpg", "bench2_volcano_02.png",
                 "bench2_jp_02.jpg"]

T2V_PROMPT = (
    "Cinematic aerial shot slowly orbiting a coastal lighthouse at golden hour, "
    "waves crashing on dark rocks below, seagulls circling, warm sunlight flares, "
    "sound of surf and wind, gentle orchestral swell."
)
I2V_PROMPT = (
    "The camera slowly pushes into the city skyline as dusk settles, window lights "
    "flickering on one by one, light wind, distant traffic hum and a soft synth pad."
)
REF2V_PROMPT = (
    "Use <Picture 1> as the setting. A slow cinematic pan across the sunset city "
    "skyline, clouds drifting, lights turning on in the towers, ambient city sounds."
)


def ref2v_prompt(ref_count):
    if ref_count <= 1:
        return REF2V_PROMPT
    tags = ", ".join(f"<Picture {i}>" for i in range(1, ref_count + 1))
    return (
        f"A sweeping cinematic montage inspired by {tags}: dinosaurs stalking through "
        "ancient forests, an anime hero surveying a volcanic ridge, crossfading between "
        "the scenes and moods of each reference in order, camera drifting forward the "
        "whole time, an adventurous orchestral score building throughout."
    )


def snap_length(seconds: float) -> int:
    """Duration in seconds -> valid H3 frame count (24 fps, n % 17 == 5, snapped up)."""
    n = max(5, round(seconds * 24))
    return n + (5 - (n % 17)) % 17


def build_graph(task, model_file, prompt, width, height, length, steps, seed,
                sampler="res_multistep", scheduler="simple", image=BENCH_IMAGE,
                ref_image_size="match", ref_count=1, ref_downscale=0.0,
                filename_prefix="bench/run"):
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": model_file, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "7": {"class_type": "BasicScheduler",
              "inputs": {"model": ["1", 0], "scheduler": scheduler, "steps": steps,
                         "denoise": 1.0}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "BasicGuider",
              "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["8", 0], "guider": ["9", 0], "sampler": ["6", 0],
                          "sigmas": ["7", 0], "latent_image": ["5", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "14": {"class_type": "CreateVideo",
               "inputs": {"images": ["12", 0], "audio": ["13", 0], "fps": 24, "bit_depth": 8}},
        "15": {"class_type": "SaveVideo",
               "inputs": {"video": ["14", 0], "filename_prefix": filename_prefix,
                          "format": "auto", "codec": "auto"}},
    }
    common = {"prompt": prompt, "width": width, "height": height, "length": length}
    if task == "t2v":
        g["5"] = {"class_type": "MiniMaxH3ImageToVideo",
                  "inputs": {"clip": ["2", 0], "vae": ["3", 0], **common}}
    elif task == "i2v":
        g["10"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g["5"] = {"class_type": "MiniMaxH3ImageToVideo",
                  "inputs": {"clip": ["2", 0], "vae": ["3", 0], "first_frame": ["10", 0],
                             **common}}
    elif task == "ref2v":
        ref_inputs = {}

        def ref_source(nid_load, fname):
            g[nid_load] = {"class_type": "LoadImage", "inputs": {"image": fname}}
            if not ref_downscale:
                return [nid_load, 0]
            nid_scale = str(int(nid_load) + 40)
            g[nid_scale] = {"class_type": "ImageScaleToTotalPixels",
                            "inputs": {"image": [nid_load, 0],
                                       "upscale_method": "lanczos",
                                       "megapixels": ref_downscale,
                                       "resolution_steps": 1}}
            return [nid_scale, 0]

        if ref_count <= 1:
            ref_inputs["ref_images.ref_image_0"] = ref_source("10", image)
        else:
            for i in range(ref_count):
                ref_inputs[f"ref_images.ref_image_{i}"] = ref_source(
                    str(20 + i), BENCH_REF_SET[i % len(BENCH_REF_SET)])
        g["5"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                  "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
                             "ref_image_size": ref_image_size,
                             **ref_inputs, **common}}
    else:
        raise ValueError(f"unknown task {task}")
    return g


class Api:
    def __init__(self, host):
        self.host = host.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _req(self, path, data=None, timeout=30):
        url = self.host + path
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"} if body else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or "{}")

    def queue_prompt(self, graph):
        return self._req("/prompt", {"prompt": graph, "client_id": self.client_id})

    def history(self, prompt_id):
        return self._req(f"/history/{prompt_id}")

    def vram_used(self):
        try:
            s = self._req("/system_stats", timeout=5)
            d = s["devices"][0]
            return d["vram_total"] - d["vram_free"]
        except Exception:
            return None


def run_one(api, cfg, poll=2.0, timeout=3600):
    """Submit one benchmark run; returns result dict."""
    graph = build_graph(**{k: v for k, v in cfg.items() if k in (
        "task", "model_file", "prompt", "width", "height", "length", "steps", "seed",
        "sampler", "scheduler", "image", "ref_image_size", "ref_count",
        "ref_downscale", "filename_prefix")})
    peak = {"vram": 0}
    stop = threading.Event()

    def sample_vram():
        while not stop.is_set():
            v = api.vram_used()
            if v:
                peak["vram"] = max(peak["vram"], v)
            stop.wait(3)

    t = threading.Thread(target=sample_vram, daemon=True)
    t.start()
    t0 = time.time()
    try:
        resp = api.queue_prompt(graph)
    except urllib.error.HTTPError as e:
        stop.set()
        detail = e.read().decode()[:2000]
        return {**cfg, "ok": False, "error": f"HTTP {e.code}: {detail}"}
    prompt_id = resp["prompt_id"]
    exec_start = exec_end = None
    status_str = None
    poll_failures = 0
    while time.time() - t0 < timeout:
        time.sleep(poll)
        try:
            h = api.history(prompt_id)
            poll_failures = 0
        except Exception as e:
            poll_failures += 1
            if poll_failures >= 10:
                stop.set()
                return {**cfg, "ok": False, "status": "server_lost",
                        "wall_s": round(time.time() - t0, 1), "exec_s": None,
                        "peak_vram_gb": round(peak["vram"] / 1e9, 1) if peak["vram"] else None,
                        "outputs": [], "error": f"server unreachable while polling: {e}"}
            continue
        if prompt_id in h:
            entry = h[prompt_id]
            status_str = entry.get("status", {}).get("status_str")
            for msg in entry.get("status", {}).get("messages", []):
                if msg[0] == "execution_start":
                    exec_start = msg[1].get("timestamp")
                if msg[0] in ("execution_success", "execution_error"):
                    exec_end = msg[1].get("timestamp")
            if status_str in ("success", "error"):
                break
    stop.set()
    wall = time.time() - t0
    exec_s = (exec_end - exec_start) / 1000 if exec_start and exec_end else None
    outputs = []
    if status_str == "success":
        for node_out in h[prompt_id].get("outputs", {}).values():
            for key in ("images", "video", "videos"):
                for f in node_out.get(key, []):
                    outputs.append(f.get("filename"))
    return {**cfg, "ok": status_str == "success", "status": status_str,
            "wall_s": round(wall, 1),
            "exec_s": round(exec_s, 1) if exec_s else None,
            "peak_vram_gb": round(peak["vram"] / 1e9, 1) if peak["vram"] else None,
            "outputs": outputs,
            "error": None if status_str == "success" else json.dumps(
                h.get(prompt_id, {}).get("status", {}))[:2000]}


def append_results(row, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    jl = outdir / "results.jsonl"
    with jl.open("a") as f:
        f.write(json.dumps({**row, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    csv_path = outdir / "results.csv"
    fields = ["ts", "label", "task", "model_file", "width", "height", "seconds", "length",
              "steps", "sampler", "scheduler", "ref_image_size", "ref_count", "warm",
              "ok", "wall_s", "exec_s", "peak_vram_gb", "outputs", "error"]
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**row, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "outputs": ";".join(row.get("outputs") or [])})


def default_prompt(task):
    return {"t2v": T2V_PROMPT, "i2v": I2V_PROMPT, "ref2v": REF2V_PROMPT}[task]


def make_cfg(task, model_file, width, height, seconds, steps, seed, label, warm,
             sampler="res_multistep", scheduler="simple", ref_image_size="match",
             ref_count=1, ref_downscale=0.0):
    prompt = ref2v_prompt(ref_count) if task == "ref2v" else default_prompt(task)
    return {
        "label": label, "task": task, "model_file": model_file,
        "prompt": prompt, "width": width, "height": height,
        "seconds": seconds, "length": snap_length(seconds), "steps": steps,
        "seed": seed, "sampler": sampler, "scheduler": scheduler,
        "image": BENCH_IMAGE, "ref_image_size": ref_image_size,
        "ref_count": ref_count, "ref_downscale": ref_downscale, "warm": warm,
        "filename_prefix": f"bench/{label}",
    }


def suite_quick(fl2va, ref2va, prefix=""):
    """Small smoke suite: one tiny run per task type."""
    return [
        make_cfg("t2v", fl2va, 864, 480, 5, 20, 1, f"{prefix}quick_t2v", warm=False),
        make_cfg("i2v", fl2va, 864, 480, 5, 20, 1, f"{prefix}quick_i2v", warm=True),
        make_cfg("ref2v", ref2va, 864, 480, 5, 20, 1, f"{prefix}quick_ref2v", warm=False),
    ]


def suite_refheavy(ref2va, prefix=""):
    """Ref2v saturation: 1/4/8 reference images, match vs max sizing.

    Ref tokens ride through every sampling step, so this measures how reference
    context load scales cost. 864x480 x 5 s base config, plus one max-res run.
    """
    cfgs = []
    for n in (1, 4, 8):
        cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 3,
                             f"{prefix}refheavy_{n}ref_match", warm=True, ref_count=n))
    for n in (4, 8):
        cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 3,
                             f"{prefix}refheavy_{n}ref_max", warm=True,
                             ref_image_size="max", ref_count=n))
    cfgs.append(make_cfg("ref2v", ref2va, 1344, 768, 5, 20, 3,
                         f"{prefix}refheavy_8ref_match_1344", warm=True, ref_count=8))
    return cfgs


def suite_family(fl2va, ref2va, prefix=""):
    """Reduced grid for comparing weight families without the full sweep:
    key t2v/i2v/ref2v points at default + max res, short + long duration."""
    return [
        make_cfg("t2v", fl2va, 864, 480, 5, 20, 2, f"{prefix}fam_t2v_864x480_5s", warm=False),
        make_cfg("t2v", fl2va, 864, 480, 15, 20, 2, f"{prefix}fam_t2v_864x480_15s", warm=True),
        make_cfg("t2v", fl2va, 1344, 768, 5, 20, 2, f"{prefix}fam_t2v_1344x768_5s", warm=True),
        make_cfg("i2v", fl2va, 864, 480, 5, 20, 2, f"{prefix}fam_i2v_864x480_5s", warm=True),
        make_cfg("ref2v", ref2va, 864, 480, 5, 20, 2, f"{prefix}fam_ref2v_864x480_5s", warm=False),
        make_cfg("ref2v", ref2va, 864, 480, 15, 20, 2, f"{prefix}fam_ref2v_864x480_15s", warm=True),
        make_cfg("ref2v", ref2va, 1344, 768, 5, 20, 2, f"{prefix}fam_ref2v_1344x768_5s", warm=True),
    ]


def suite_key(fl2va, ref2va, prefix="", repeats=3):
    """Key configurations with N repeats for real averages, plus scaling singles.

    Repeated (N=repeats, distinct seeds): t2v 864x480 5s; ref2v 1 ref; ref2v
    8 refs (few-vs-many reference context, the priority modality).
    Singles: t2v 15 s, t2v max-res, ref2v 8-ref max sizing.
    """
    cfgs = []
    for i in range(repeats):
        cfgs.append(make_cfg("t2v", fl2va, 864, 480, 5, 20, 10 + i,
                             f"{prefix}key_t2v_5s_r{i}", warm=(i > 0)))
    for i in range(repeats):
        cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 10 + i,
                             f"{prefix}key_ref2v_1ref_r{i}", warm=(i > 0), ref_count=1))
    for i in range(repeats):
        cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 10 + i,
                             f"{prefix}key_ref2v_8ref_r{i}", warm=True, ref_count=8))
    cfgs.append(make_cfg("t2v", fl2va, 864, 480, 15, 20, 20, f"{prefix}key_t2v_15s", warm=True))
    cfgs.append(make_cfg("t2v", fl2va, 1344, 768, 5, 20, 20, f"{prefix}key_t2v_1344", warm=True))
    cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 20,
                         f"{prefix}key_ref2v_8ref_max", warm=True, ref_count=8,
                         ref_image_size="max"))
    return cfgs


def suite_refmatrix(ref2va, prefix=""):
    """Reference-variety matrix: ref count x ref resolution.

    Counts 1/2/4/8/9 at native ref size (match sizing); at 8 refs, a
    resolution ladder from 0.1 MP pre-downscaled up to 'max' (2048px) sizing.
    864x480 5 s canvas throughout.
    """
    cfgs = []
    for n in (1, 2, 4, 8, 9):
        cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 70,
                             f"{prefix}rm_{n}ref_full", warm=True, ref_count=n))
    for ds in (0.1, 0.2, 0.4):
        cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 71,
                             f"{prefix}rm_8ref_ds{int(ds*100):03d}", warm=True,
                             ref_count=8, ref_downscale=ds))
    cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 72,
                         f"{prefix}rm_4ref_max", warm=True, ref_count=4,
                         ref_image_size="max"))
    cfgs.append(make_cfg("ref2v", ref2va, 864, 480, 5, 20, 72,
                         f"{prefix}rm_9ref_max", warm=True, ref_count=9,
                         ref_image_size="max"))
    return cfgs


def suite_sweep(fl2va, ref2va, prefix=""):
    """Full sweep: tasks x durations x resolutions at fixed 20 steps.

    Resolution ladder (16:9, multiples of 32, up to the 1MP native cap):
      608x352 (0.2MP draft), 864x480 (0.4MP default), 1344x768 (1MP max)
    Durations: 5 s (124 fr), 10 s (243 fr), 15 s (362 fr) - the trained range.
    """
    resolutions = [(608, 352), (864, 480), (1344, 768)]
    durations = [5, 10, 15]
    cfgs = []
    warm_fl = False
    for w, h in resolutions:
        for sec in durations:
            cfgs.append(make_cfg("t2v", fl2va, w, h, sec, 20, 1,
                                 f"{prefix}sweep_t2v_{w}x{h}_{sec}s", warm=warm_fl))
            warm_fl = True
    # i2v: model already warm; vary duration at default res, plus max res spot check
    for sec in durations:
        cfgs.append(make_cfg("i2v", fl2va, 864, 480, sec, 20, 1,
                             f"{prefix}sweep_i2v_864x480_{sec}s", warm=True))
    cfgs.append(make_cfg("i2v", fl2va, 1344, 768, 5, 20, 1,
                         f"{prefix}sweep_i2v_1344x768_5s", warm=True))
    # ref2v: separate weights -> first run is a model swap (cold)
    warm_ref = False
    for sec in durations:
        cfgs.append(make_cfg("ref2v", ref2va, 864, 480, sec, 20, 1,
                             f"{prefix}sweep_ref2v_864x480_{sec}s", warm=warm_ref))
        warm_ref = True
    cfgs.append(make_cfg("ref2v", ref2va, 1344, 768, 5, 20, 1,
                         f"{prefix}sweep_ref2v_1344x768_5s", warm=True))
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--suite", choices=["quick", "sweep", "refheavy", "family", "key", "refmatrix"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--task", choices=["t2v", "i2v", "ref2v"])
    ap.add_argument("--model", action="append", default=[],
                    help="override model files, e.g. fl2va=file.safetensors (repeatable)")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sampler", default="res_multistep")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--ref-image-size", default="match", choices=["match", "max"])
    ap.add_argument("--ref-count", type=int, default=1,
                    help="number of reference images for ref2v (1-8)")
    ap.add_argument("--ref-downscale", type=float, default=0.0,
                    help="pre-downscale refs to this many megapixels before the node (0=off)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--prefix", default="", help="label prefix for suites, e.g. prunedint8_")
    ap.add_argument("--warm", action="store_true", help="mark run as warm (model preloaded)")
    ap.add_argument("--outdir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    models = {"fl2va": FL2VA_DEFAULT, "ref2va": REF2VA_DEFAULT}
    for m in args.model:
        k, v = m.split("=", 1)
        models[k] = v

    api = Api(args.host)
    if args.suite == "quick":
        cfgs = suite_quick(models["fl2va"], models["ref2va"], args.prefix)
    elif args.suite == "sweep":
        cfgs = suite_sweep(models["fl2va"], models["ref2va"], args.prefix)
    elif args.suite == "refheavy":
        cfgs = suite_refheavy(models["ref2va"], args.prefix)
    elif args.suite == "family":
        cfgs = suite_family(models["fl2va"], models["ref2va"], args.prefix)
    elif args.suite == "key":
        cfgs = suite_key(models["fl2va"], models["ref2va"], args.prefix, args.repeats)
    elif args.suite == "refmatrix":
        cfgs = suite_refmatrix(models["ref2va"], args.prefix)
    elif args.task:
        model = models["ref2va"] if args.task == "ref2v" else models["fl2va"]
        label = args.label or f"{args.task}_{args.width}x{args.height}_{args.seconds}s"
        cfgs = [make_cfg(args.task, model, args.width, args.height, args.seconds,
                         args.steps, args.seed, label, args.warm,
                         args.sampler, args.scheduler, args.ref_image_size,
                         args.ref_count, args.ref_downscale)]
    else:
        ap.error("need --suite or --task")

    outdir = Path(args.outdir)
    for i, cfg in enumerate(cfgs):
        print(f"[{i+1}/{len(cfgs)}] {cfg['label']}: {cfg['task']} {cfg['model_file']} "
              f"{cfg['width']}x{cfg['height']} {cfg['seconds']}s ({cfg['length']}fr) "
              f"{cfg['steps']} steps ...", flush=True)
        row = run_one(api, cfg)
        append_results(row, outdir)
        if row["ok"]:
            print(f"    OK wall={row['wall_s']}s exec={row['exec_s']}s "
                  f"peak_vram={row['peak_vram_gb']}GB -> {row['outputs']}", flush=True)
        else:
            print(f"    FAILED: {str(row.get('error'))[:400]}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate benchmark output videos: download each recorded output via the
ComfyUI /view endpoint and ffprobe it (streams, duration, size).

Usage:
  python3 bench/validate_outputs.py --host http://127.0.0.1:8189 --jsonl bench/4090/results.jsonl [--sample N]

Appends results to <jsonl-dir>/validation.jsonl and prints a summary table.
A video is VALID if it has an h264 video stream and an audio stream, and its
container duration is within 0.5 s of the requested clip length.
"""
import argparse
import json
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height:format=duration,size",
         "-of", "json", path],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--sample", type=int, default=0, help="validate only every Nth eligible row (0 = all)")
    args = ap.parse_args()

    jl = Path(args.jsonl)
    rows = [json.loads(l) for l in jl.open() if l.strip()]
    ok_rows = [r for r in rows if r.get("ok") is True and r.get("outputs")]
    if args.sample:
        ok_rows = ok_rows[::args.sample]

    results = []
    for r in ok_rows:
        for fname in r["outputs"]:
            q = urllib.parse.urlencode(
                {"filename": fname, "subfolder": "bench", "type": "output"})
            url = f"{args.host}/view?{q}"
            verdict = {"label": r["label"], "file": fname, "valid": False}
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                    with urllib.request.urlopen(url, timeout=300) as resp:
                        data = resp.read()
                    tmp.write(data)
                    tmp.flush()
                    verdict["bytes"] = len(data)
                    info = ffprobe(tmp.name) if len(data) > 10000 else None
                if info:
                    streams = {s["codec_type"]: s for s in info.get("streams", [])}
                    dur = float(info.get("format", {}).get("duration", 0))
                    want = float(r["length"]) / 24.0
                    verdict.update({
                        "video_codec": streams.get("video", {}).get("codec_name"),
                        "audio_codec": streams.get("audio", {}).get("codec_name"),
                        "width": streams.get("video", {}).get("width"),
                        "height": streams.get("video", {}).get("height"),
                        "duration": round(dur, 2),
                        "expected_duration": round(want, 2),
                        "valid": ("video" in streams and "audio" in streams
                                  and abs(dur - want) < 0.5),
                    })
            except Exception as e:
                verdict["error"] = str(e)[:200]
            results.append(verdict)
            v = "VALID" if verdict["valid"] else "INVALID"
            print(f'{v:<8} {fname:<48} {verdict.get("bytes",0)//1024:>7} KB '
                  f'{verdict.get("width","?")}x{verdict.get("height","?")} '
                  f'{verdict.get("duration","?")}s/{verdict.get("expected_duration","?")}s '
                  f'{verdict.get("video_codec","")}+{verdict.get("audio_codec","")}')

    with (jl.parent / "validation.jsonl").open("a") as f:
        for v in results:
            f.write(json.dumps(v) + "\n")
    good = sum(1 for v in results if v["valid"])
    print(f"\n{good}/{len(results)} outputs valid")


if __name__ == "__main__":
    main()

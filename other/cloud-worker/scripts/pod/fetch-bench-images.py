#!/usr/bin/env python3
"""Fetch real benchmark images into /ComfyUI/input via the Wikimedia Commons API.

Categories:
  bench_anime_01..03  anime-style character illustrations (detailed subjects)
  bench_movie_01..03  public-domain film stills (cinematic composition)
  bench_big_01..02    very large images (6000px+), stress ref preprocessing

These replace the synthetic PIL gradients for ref2v/i2v benchmarking — real
detail levels tax the reference encoder realistically. Re-runnable; skips
files that already exist.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("/ComfyUI/input")
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "storyteller-bench/1.0 (model benchmarking; contact: ops@storyteller.ai)"}

SETS = [
    # (prefix, count, min_width, search query)
    ("bench_anime", 3, 1000, 'anime style character illustration filetype:bitmap'),
    ("bench_movie", 3, 1200, 'film still 1920s screenshot filetype:bitmap'),
    ("bench_big", 2, 6000, 'panorama cityscape night filetype:bitmap'),
]


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def search_images(query, need, min_width):
    data = api({
        "action": "query", "generator": "search",
        "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": 40,
        "prop": "imageinfo", "iiprop": "url|size|mime",
    })
    pages = (data.get("query") or {}).get("pages", {})
    hits = []
    for p in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") in ("image/jpeg", "image/png") and ii.get("width", 0) >= min_width:
            hits.append((ii["url"], ii["width"], ii["height"]))
        if len(hits) >= need:
            break
    return hits


def fetch(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for prefix, count, min_width, query in SETS:
        hits = search_images(query, count, min_width)
        if len(hits) < count:
            print(f"WARNING: only {len(hits)}/{count} results for {prefix} ({query})")
        for i, (url, w, h) in enumerate(hits, start=1):
            ext = ".png" if url.lower().endswith(".png") else ".jpg"
            name = f"{prefix}_{i:02d}{ext}"
            dest = OUT / name
            if dest.exists():
                print(f"skip {name} (exists)")
            else:
                print(f"fetch {name} <- {w}x{h} {url}")
                fetch(url, dest)
            manifest[name] = {"source": url, "width": w, "height": h}
    (OUT / "bench_images_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(manifest)} images, manifest written")


if __name__ == "__main__":
    main()

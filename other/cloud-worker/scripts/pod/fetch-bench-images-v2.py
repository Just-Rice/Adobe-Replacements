#!/usr/bin/env python3
"""Fetch the v2 real-reference benchmark set into /ComfyUI/input.

Large (min-width-enforced) images: Jurassic Park stills, anime characters
(Ashitaka, Cowboy Bebop), forests, volcanos. Sources are MediaWiki APIs
(Fandom wikis + Wikimedia Commons). Internal test assets only — not
redistributed with the repo.

Produces bench2_* files + bench2_manifest.json. Re-runnable; skips existing.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("/ComfyUI/input")
UA = {"User-Agent": "storyteller-bench/1.0 (internal model benchmarking)"}

# (wiki api base, prefix, count, min_width, query)
SETS = [
    ("https://jurassicpark.fandom.com/api.php", "bench2_jp",       2, 1400, "Tyrannosaurus rex"),
    ("https://jurassicpark.fandom.com/api.php", "bench2_jp_gate",  1, 1200, "Jurassic Park gate"),
    ("https://ghibli.fandom.com/api.php",       "bench2_ashitaka", 1, 900,  "Ashitaka"),
    ("https://cowboybebop.fandom.com/api.php",  "bench2_bebop",    1, 900,  "Spike Spiegel"),
    ("https://commons.wikimedia.org/w/api.php", "bench2_forest",   2, 4000, "old growth forest filetype:bitmap"),
    ("https://commons.wikimedia.org/w/api.php", "bench2_volcano",  2, 4000, "volcano eruption lava filetype:bitmap"),
]


def api(base, params):
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{base}?{q}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def search_images(base, query, need, min_width):
    data = api(base, {
        "action": "query", "generator": "search",
        "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": 50,
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
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for base, prefix, count, min_width, query in SETS:
        try:
            hits = search_images(base, query, count, min_width)
        except Exception as e:
            print(f"WARNING: search failed for {prefix}: {e}")
            hits = []
        if len(hits) < count:
            # fall back: halve the width requirement once
            try:
                hits = search_images(base, query, count, min_width // 2)
            except Exception:
                pass
        if len(hits) < count:
            print(f"WARNING: only {len(hits)}/{count} for {prefix} ({query})")
        for i, (url, w, h) in enumerate(hits, start=1):
            ext = ".png" if ".png" in url.lower() else ".jpg"
            name = f"{prefix}_{i:02d}{ext}"
            dest = OUT / name
            if dest.exists():
                print(f"skip {name}")
            else:
                print(f"fetch {name} <- {w}x{h}")
                fetch(url, dest)
            manifest[name] = {"source": url, "width": w, "height": h}
    (OUT / "bench2_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(manifest)} images")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic test/reference images into /ComfyUI/input.

Creates:
  - red_superboy_on_city_roof.png / mecha_dragon_lightning.png: stylized
    placeholders matching the refs in the installed r2v workflow so it runs
    out of the box.
  - bench_ref_1344x768.png: generic scene used by the benchmark harness for
    i2v (first frame) and ref2v runs.
"""
import math
import random

from PIL import Image, ImageDraw, ImageFilter

OUT = "/ComfyUI/input"


def city_backdrop(w, h, sky_top, sky_bottom, seed=7):
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line(
            [(0, y), (w, y)],
            fill=tuple(int(a + (b - a) * t) for a, b in zip(sky_top, sky_bottom)),
        )
    # skyline
    x = 0
    while x < w:
        bw = rng.randint(w // 20, w // 8)
        bh = rng.randint(h // 4, int(h * 0.55))
        d.rectangle([x, h - bh, x + bw, h], fill=(18, 16, 28))
        for wy in range(h - bh + 10, h - 10, 24):
            for wx in range(x + 8, x + bw - 8, 20):
                if rng.random() < 0.35:
                    d.rectangle([wx, wy, wx + 8, wy + 12], fill=(255, 220, 120))
        x += bw + rng.randint(4, 24)
    return img


def superboy():
    w, h = 1344, 768
    img = city_backdrop(w, h, (30, 20, 60), (120, 40, 80), seed=11)
    d = ImageDraw.Draw(img)
    # rooftop
    d.rectangle([0, int(h * 0.78), w, h], fill=(28, 26, 36))
    cx, cy = w // 2, int(h * 0.55)
    # cape
    d.polygon(
        [(cx - 28, cy - 40), (cx + 28, cy - 40), (cx + 95, cy + 130), (cx - 60, cy + 120)],
        fill=(190, 20, 30),
    )
    # body
    d.rectangle([cx - 26, cy - 35, cx + 26, cy + 70], fill=(40, 60, 160))
    # legs
    d.rectangle([cx - 24, cy + 70, cx - 6, cy + 170], fill=(190, 20, 30))
    d.rectangle([cx + 6, cy + 70, cx + 24, cy + 170], fill=(190, 20, 30))
    # arms on hips
    d.polygon([(cx - 26, cy - 25), (cx - 62, cy + 25), (cx - 26, cy + 32)], fill=(40, 60, 160))
    d.polygon([(cx + 26, cy - 25), (cx + 62, cy + 25), (cx + 26, cy + 32)], fill=(40, 60, 160))
    # head
    d.ellipse([cx - 24, cy - 88, cx + 24, cy - 40], fill=(250, 210, 170))
    d.arc([cx - 16, cy - 70, cx + 16, cy - 46], 20, 160, fill=(120, 60, 30), width=3)
    # freckles + grin
    for fx, fy in [(-10, -62), (10, -62), (-14, -58), (14, -58)]:
        d.point((cx + fx, cy + fy), fill=(180, 120, 80))
    # chest emblem
    d.polygon([(cx, cy - 20), (cx - 14, cy + 5), (cx + 14, cy + 5)], fill=(250, 210, 60))
    return img


def mecha_dragon():
    w, h = 1344, 768
    img = city_backdrop(w, h, (8, 8, 20), (40, 20, 50), seed=23)
    d = ImageDraw.Draw(img)
    cx, cy = int(w * 0.6), int(h * 0.45)
    # body silhouette
    d.polygon(
        [(cx - 220, h), (cx - 120, cy + 60), (cx - 40, cy - 60), (cx + 60, cy - 140),
         (cx + 180, cy - 100), (cx + 240, cy + 40), (cx + 320, h)],
        fill=(10, 10, 14),
    )
    # head / jaws
    d.polygon(
        [(cx + 60, cy - 140), (cx + 10, cy - 220), (cx + 120, cy - 190), (cx + 180, cy - 100)],
        fill=(14, 14, 18),
    )
    d.polygon([(cx + 20, cy - 210), (cx - 40, cy - 250), (cx + 40, cy - 225)], fill=(16, 16, 20))
    # glowing eyes and chest core
    d.ellipse([cx + 60, cy - 195, cx + 80, cy - 180], fill=(255, 40, 40))
    d.ellipse([cx + 95, cy - 190, cx + 112, cy - 176], fill=(255, 40, 40))
    core = Image.new("RGB", (w, h))
    cd = ImageDraw.Draw(core)
    cd.ellipse([cx - 20, cy - 40, cx + 60, cy + 40], fill=(255, 60, 60))
    core = core.filter(ImageFilter.GaussianBlur(18))
    img.paste(Image.blend(img.crop((0, 0, w, h)), core, 0.5), (0, 0), core.convert("L"))
    # lightning bolts
    rng = random.Random(5)
    for _ in range(4):
        x0 = rng.randint(cx - 100, cx + 200)
        y0 = 0
        pts = [(x0, y0)]
        while pts[-1][1] < cy - 160:
            px, py = pts[-1]
            pts.append((px + rng.randint(-40, 40), py + rng.randint(30, 70)))
        d.line(pts, fill=(120, 180, 255), width=4)
    return img


def bench_ref():
    img = city_backdrop(1344, 768, (200, 120, 60), (240, 200, 140), seed=42)
    return img


# Distinct palettes for the multi-reference ref2v benchmark set. Each image is
# a different scene so 4-8 refs exercise genuinely different reference context.
REF_PALETTES = [
    ((200, 120, 60), (240, 200, 140)),   # amber sunset
    ((20, 30, 80), (90, 140, 200)),      # blue dusk
    ((10, 60, 50), (120, 200, 160)),     # teal dawn
    ((80, 20, 90), (200, 120, 220)),     # violet night
    ((160, 40, 30), (250, 150, 90)),     # red dusk
    ((30, 30, 30), (140, 140, 150)),     # overcast grey
    ((10, 20, 40), (40, 90, 160)),       # deep night blue
    ((190, 160, 40), (250, 240, 170)),   # golden noon
]


def bench_ref_set():
    """8 distinct city scenes, bench_ref_01..08.png (1344x768)."""
    for i, (top, bottom) in enumerate(REF_PALETTES, start=1):
        img = city_backdrop(1344, 768, top, bottom, seed=100 + i)
        d = ImageDraw.Draw(img)
        # distinct celestial marker per scene so refs are visually distinct
        x = 120 + i * 130
        d.ellipse([x, 80, x + 90, 170],
                  fill=(255, 240, 200) if i % 2 else (240, 240, 255))
        img.save(f"{OUT}/bench_ref_{i:02d}.png")


if __name__ == "__main__":
    superboy().save(f"{OUT}/red_superboy_on_city_roof.png")
    mecha_dragon().save(f"{OUT}/mecha_dragon_lightning.png")
    bench_ref().save(f"{OUT}/bench_ref_1344x768.png")
    bench_ref_set()
    print("wrote 11 images to", OUT)

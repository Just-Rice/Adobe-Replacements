#!/usr/bin/env python3
"""Summarize benchmark results across GPUs and weight families.

Reads bench/results.csv (B200) plus bench/4090/results.csv and
bench/4000/results.csv when present, groups repeated configs, and prints
mean +/- spread tables (markdown, padded columns) for docs/BENCHMARKS.md.
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent

SOURCES = [
    ("B200", HERE / "results.csv"),
    ("RTX 4090", HERE / "4090" / "results.csv"),
    ("RTX 4090 (patched)", HERE / "4090c" / "results.csv"),
    ("RTX 5090", HERE / "5090" / "results.csv"),
    ("RTX PRO 4000", HERE / "4000" / "results.csv"),
]

FAMILY_PREFIXES = [
    ("pi8", "pruned int8"), ("i8", "int8"), ("b16", "bf16"),
    ("prunedint8", "pruned int8"), ("int8", "int8"), ("bf16", "bf16"),
    ("g4090_pi8", "pruned int8"), ("g4000_pi8", "pruned int8"),
]


def family_of(row):
    m = row["model_file"]
    if "pruned" in m:
        return "pruned int8"
    if "int8" in m:
        return "int8"
    if "bf16" in m:
        return "bf16"
    return m


def load(path):
    jl = path.with_suffix(".jsonl")
    if jl.exists():
        rows = [json.loads(l) for l in jl.open() if l.strip()]
        return [{k: str(v) if v is not None else "" for k, v in r.items()}
                for r in rows if r.get("ok") is True]
    if not path.exists():
        return []
    return [r for r in csv.DictReader(path.open()) if r.get("ok") == "True"]


def key_of(row):
    return (row["task"], f'{row["width"]}x{row["height"]}', row["seconds"],
            row.get("ref_count") or "1", row.get("ref_image_size") or "match",
            row["steps"])


def fmt_stats(vals):
    if len(vals) == 1:
        return f"{vals[0]:.0f} s"
    return f"{statistics.mean(vals):.0f} ± {statistics.stdev(vals):.0f} s (n={len(vals)})"


def pad_table(rows):
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        out.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
        if i == 0:
            out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(out)


def main():
    for gpu, path in SOURCES:
        rows = load(path)
        if not rows:
            continue
        # group warm runs only (exclude first-load runs marked warm=False)
        groups = defaultdict(lambda: defaultdict(list))
        vram = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r.get("warm") == "False":
                continue
            fam = family_of(r)
            try:
                t = float(r["exec_s"])
            except (ValueError, TypeError):
                continue
            groups[key_of(r)][fam].append(t)
            try:
                vram[key_of(r)][fam].append(float(r["peak_vram_gb"]))
            except (ValueError, TypeError):
                pass
        fams = sorted({f for g in groups.values() for f in g},
                      key=lambda f: ["pruned int8", "int8", "bf16"].index(f)
                      if f in ["pruned int8", "int8", "bf16"] else 9)
        print(f"\n## {gpu}\n")
        header = ["task", "canvas", "sec", "refs", "sizing"] + \
            [f"{f} (exec)" for f in fams] + [f"{f} (peak GB)" for f in fams]
        table = [header]
        for k in sorted(groups):
            task, wh, sec, refs, sizing, steps = k
            row = [task, wh, sec, refs, sizing]
            for f in fams:
                row.append(fmt_stats(groups[k][f]) if groups[k][f] else "—")
            for f in fams:
                row.append(f"{max(vram[k][f]):.0f}" if vram[k][f] else "—")
            table.append(row)
        print(pad_table(table))


if __name__ == "__main__":
    main()

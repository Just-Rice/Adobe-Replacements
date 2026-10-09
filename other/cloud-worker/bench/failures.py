#!/usr/bin/env python3
"""Failure statistics across GPUs and configurations.

Classifies every recorded run (bench/*.jsonl) into success / failure buckets
with an error class, and prints per-GPU tables showing which configurations
fail where. Padded markdown output.
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = [
    ("B200 180GB", HERE / "results.jsonl"),
    ("RTX 4090 24GB", HERE / "4090" / "results.jsonl"),
    ("RTX 4090 24GB (patched)", HERE / "4090c" / "results.jsonl"),
    ("RTX 5090 32GB", HERE / "5090" / "results.jsonl"),
    ("RTX PRO 4000 24GB", HERE / "4000" / "results.jsonl"),
]


def classify(row):
    if row.get("ok") is True:
        return "ok"
    err = str(row.get("error") or "") + str(row.get("status") or "")
    if "server_lost" in err or "unreachable" in err or "Connection" in err:
        return "server_died"
    if "OutOfMemory" in err or "Allocation on device" in err or "OOM" in err:
        return "vram_oom"
    if "enable_gqa" in err:
        return "env_torch_too_old"
    if "400" in err and "prompt" in err.lower():
        return "graph_invalid"
    return "error_other"


def cfg_of(row):
    return (f'{row["task"]} {row["width"]}x{row["height"]} {row["seconds"]}s '
            f'refs={row.get("ref_count") or 1} {row.get("ref_image_size") or "match"}')


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
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open() if l.strip()]
        total = len(rows)
        okn = sum(1 for r in rows if r.get("ok") is True)
        print(f"\n## {gpu} — {okn}/{total} runs succeeded "
              f"({100*okn/max(total,1):.0f}%)\n")
        by_cfg = defaultdict(lambda: defaultdict(int))
        for r in rows:
            by_cfg[cfg_of(r)][classify(r)] += 1
        classes = sorted({c for v in by_cfg.values() for c in v if c != "ok"})
        table = [["configuration", "ok"] + classes]
        for cfg in sorted(by_cfg):
            v = by_cfg[cfg]
            if sum(n for c, n in v.items() if c != "ok") == 0:
                continue  # only show configs with at least one failure
            table.append([cfg, str(v.get("ok", 0))] +
                         [str(v.get(c, 0)) for c in classes])
        if len(table) > 1:
            print(pad_table(table))
        else:
            print("(no failures)")
        # fully-clean configs summary
        clean = [c for c, v in sorted(by_cfg.items())
                 if sum(n for cl, n in v.items() if cl != "ok") == 0]
        print(f"\nConfigs with zero failures: {len(clean)}/{len(by_cfg)}")


if __name__ == "__main__":
    main()

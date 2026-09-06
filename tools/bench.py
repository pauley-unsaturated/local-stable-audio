#!/usr/bin/env python3
"""Phase 0 baseline: wall time, per-stage breakdown, and peak memory for the
upstream MLX CLI on this machine.

Each cell runs `scripts/sa3_mlx.py` in the pinned submodule as a fresh
subprocess under `/usr/bin/time -l`, so peak RSS is the OS's number for that
process, not MLX's own Metal-allocator peak (the CLI prints that too; both
are recorded). Stage timings are parsed from the CLI's own instrumentation.

"Warm" here is derived, not measured: wall minus every "load" sub-stage the
CLI reports (T5Gemma load is folded into its encode stage, so it is estimated
from the difference between the first and later runs; see BASELINE.md). The
true resident-model second-generation number comes from Phase 1's in-process
harness, which has to exist anyway.

Usage (from the repo root, no venv activation needed):
    python3 tools/bench.py                      # the full HANDOFF §2 matrix
    python3 tools/bench.py --quick              # sm-music 10 s only, 1 repeat
    python3 tools/bench.py --json out.json      # also dump raw rows

Weights must already be local (install.sh --download sm-music,medium).
Generated WAVs go to the scratch dir and are deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLX = ROOT / "external" / "stable-audio-3" / "optimized" / "mlx"
PY = MLX / ".venv" / "bin" / "python"
CLI = MLX / "scripts" / "sa3_mlx.py"

PROMPT = "dub techno chord stab, 128 bpm"
SEED = 1

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DONE = re.compile(r"done\s+([\d.]+)s wall\s+→\s+([\d.]+)s audio\s+→\s+([\d.]+)× realtime\s+peak RAM\s+([\d.]+)\s+(MB|GB)")
_STAGE = re.compile(r"^\s*\[(\d/5|3a)\]\s+(.*?)\s*(?:·+\s*(\d+) ms)?\s*$")
_SUB_LOAD = re.compile(r"^\s*load\s+([\d.]+)\s*(s|ms)\b")
_SUB_SAMPLE = re.compile(r"^\s*sample\s+(\d+) ms\s+\((\d+) ms/step\)")
_SUB_DECODE = re.compile(r"^\s*decode\s+.*?→\s+(\d+) ms")
_SUB_ENCODE = re.compile(r"^\s*encode\s+(\d+) ms")
_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)


def parse(stdout: str, stderr: str) -> dict:
    out = _ANSI.sub("", stdout)
    r: dict = {"stages_ms": {}, "loads_ms": {}}
    cur = None
    for line in out.splitlines():
        m = _STAGE.match(line)
        if m:
            cur = m.group(1)
            r["stages_ms"][cur] = int(m.group(3)) if m.group(3) else None
            continue
        m = _SUB_LOAD.match(line)
        if m and cur:
            v = float(m.group(1)) * (1000 if m.group(2) == "s" else 1)
            r["loads_ms"][cur] = v
            continue
        m = _SUB_SAMPLE.match(line)
        if m:
            r["sample_ms"] = int(m.group(1)); r["ms_per_step"] = int(m.group(2)); continue
        m = _SUB_DECODE.match(line)
        if m:
            r["decode_ms"] = int(m.group(1)); continue
        m = _SUB_ENCODE.match(line)
        if m:
            r["encode_ms"] = int(m.group(1)); continue
    m = _DONE.search(out)
    if not m:
        raise RuntimeError("no 'done' line in CLI output:\n" + out[-2000:])
    r["wall_s"] = float(m.group(1)); r["audio_s"] = float(m.group(2)); r["realtime"] = float(m.group(3))
    r["mlx_peak_gb"] = float(m.group(4)) / (1024 if m.group(5) == "MB" else 1)
    m = _RSS.search(stderr)
    r["rss_peak_gb"] = int(m.group(1)) / 1024**3 if m else None
    # Stage [1/5] is T5Gemma load + encode in one number; the encode of a short
    # prompt is a few ms, so treat the whole stage as load.
    loads = sum(r["loads_ms"].values()) + (r["stages_ms"].get("1/5") or 0)
    r["loads_total_ms"] = loads
    r["warm_est_s"] = r["wall_s"] - loads / 1000
    return r


def run_cell(dit: str, decoder: str, seconds: float, free_models: bool, tag: str) -> dict:
    out = Path(tempfile.gettempdir()) / f"lsa_bench_{tag}.wav"
    cmd = ["/usr/bin/time", "-l", str(PY), str(CLI),
           "--prompt", PROMPT, "--dit", dit, "--decoder", decoder,
           "--seconds", str(seconds), "--seed", str(SEED), "--out", str(out),
           "--free-models" if free_models else "--no-free-models"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MLX), timeout=30 * 60)
    ext = time.time() - t0
    out.unlink(missing_ok=True)
    if p.returncode != 0:
        raise RuntimeError(f"{tag}: exit {p.returncode}\n{p.stderr[-2000:]}")
    r = parse(p.stdout, p.stderr)
    r.update(dit=dit, decoder=decoder, seconds=seconds, free_models=free_models,
             tag=tag, external_wall_s=round(ext, 2))
    return r


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if not PY.exists():
        sys.exit(f"venv missing: {PY}. Run install.sh in {MLX}")

    configs = [("sm-music", "same-s"), ("medium", "same-l")]
    if a.quick:
        configs = configs[:1]; a.repeats = 1

    rows: list[dict] = []
    def go(dit, dec, secs, free, tag):
        print(f"→ {tag:32s}", end="", flush=True)
        r = run_cell(dit, dec, secs, free, tag)
        rows.append(r)
        print(f" wall {r['wall_s']:6.2f}s  ext {r['external_wall_s']:6.2f}s  warm≈{r['warm_est_s']:5.2f}s  "
              f"rss {fmt(r['rss_peak_gb'])} GB  mlx {r['mlx_peak_gb']:.2f} GB  "
              f"sample {r.get('sample_ms')} ms  decode {r.get('decode_ms')} ms", flush=True)

    # 1. cold 10 s, repeated; also gives the stage breakdown for the warm estimate
    for dit, dec in configs:
        for i in range(a.repeats):
            go(dit, dec, 10, True, f"{dit}-10s-free-r{i+1}")
    # 2. peak RSS with --no-free-models, 10 s
    for dit, dec in configs:
        go(dit, dec, 10, False, f"{dit}-10s-nofree")
    # 3. 30 s and 120 s, free-models (upstream methodology)
    if not a.quick:
        for dit, dec in configs:
            for secs in (30, 120):
                go(dit, dec, secs, True, f"{dit}-{secs}s-free")

    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))

    # Summary tables (markdown) -------------------------------------------
    print("\n## Cold vs derived-warm, 10 s (median of repeats, --free-models)\n")
    print("| dit | cold wall (s) | loads (s) | warm est. (s) | warm × RT | sample (ms) | ms/step | decode (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for dit, dec in configs:
        rs = [r for r in rows if r["dit"] == dit and r["seconds"] == 10 and r["free_models"]]
        if not rs: continue
        med = lambda k: statistics.median(r[k] for r in rs if r.get(k) is not None)
        w = med("wall_s"); l = med("loads_total_ms") / 1000; we = med("warm_est_s")
        print(f"| {dit} | {w:.2f} | {l:.2f} | {we:.2f} | {10/we:.1f}× | {med('sample_ms'):.0f} | {med('ms_per_step'):.0f} | {med('decode_ms'):.0f} |")
    print("\n## Peak memory, 10 s\n")
    print("| dit | RSS free (GB) | RSS no-free (GB) | MLX peak free (GB) | MLX peak no-free (GB) |")
    print("|---|---|---|---|---|")
    for dit, dec in configs:
        f = [r for r in rows if r["dit"] == dit and r["seconds"] == 10 and r["free_models"]]
        n = [r for r in rows if r["dit"] == dit and r["seconds"] == 10 and not r["free_models"]]
        if not f or not n: continue
        print(f"| {dit} | {fmt(max(r['rss_peak_gb'] or 0 for r in f))} | {fmt(n[0]['rss_peak_gb'])} | "
              f"{max(r['mlx_peak_gb'] for r in f):.2f} | {n[0]['mlx_peak_gb']:.2f} |")
    if not a.quick:
        print("\n## Wall vs length (cold, --free-models)\n")
        print("| dit | 10 s | 30 s | 120 s | decode share @120 s |")
        print("|---|---|---|---|---|")
        for dit, dec in configs:
            by = {}
            for r in rows:
                if r["dit"] == dit and r["free_models"]:
                    by.setdefault(r["seconds"], r)
            r10 = [r for r in rows if r["dit"] == dit and r["seconds"] == 10 and r["free_models"]]
            w10 = statistics.median(r["wall_s"] for r in r10) if r10 else None
            r120 = by.get(120)
            share = f"{100*r120['decode_ms']/1000/r120['wall_s']:.0f}%" if r120 and r120.get("decode_ms") else ""
            print(f"| {dit} | {fmt(w10)} | {fmt(by.get(30,{}).get('wall_s'))} | {fmt(by.get(120,{}).get('wall_s'))} | {share} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

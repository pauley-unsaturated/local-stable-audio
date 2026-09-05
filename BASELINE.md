# BASELINE.md

Phase 0 measurements on the target machine. Every number here was measured on
this hardware with the command shown; the "upstream published" column is
copied from `external/stable-audio-3/optimized/mlx/README.md` (their M4 Pro /
48 GB run) as the expected range. Fill in, do not estimate.

## Machine

| Item | Value |
|---|---|
| Model | Mac mini, Apple M4 Pro |
| Unified memory | 48 GB |
| macOS | 26.6.1 (25G76) |
| Xcode / Swift | 26.5 (17F42) / Swift 6.3.2 |
| Python (MLX venv) | 3.12 via uv (see D-002) |
| `mlx` version | _fill in from `.venv/bin/python -c "import mlx.core as mx; print(mx.__version__)"`_ |
| Upstream commit | `779434a` (submodule `external/stable-audio-3`) |
| Date | _fill in_ |

## Gate 0 artefact

| Item | Value |
|---|---|
| Prompt | `dub techno chord stab, 128 bpm` |
| Command | `./sa3 --prompt "dub techno chord stab, 128 bpm" --dit sm-music --decoder same-s --seconds 10 --seed 1 --out <abs path>` |
| Output path | _fill in (outside the repo; `*.wav` is gitignored)_ |
| Listened, sounds like the prompt? | _yes / no, notes_ |

## 1. Cold vs warm wall time, 10 s clip

Cold = fresh process, models loaded from disk (weights already in the HF cache;
download time excluded). Warm = second generation in the same process with
`--no-free-models`, i.e. the per-generation budget the instruments actually
pay. Measured by `tools/bench.py` (Phase 0 task), three runs each, median.

| `--dit` / `--decoder` | Cold wall (s) | Warm wall (s) | Warm × realtime | Upstream published (5 s cold) |
|---|---|---|---|---|
| `sm-music` / `same-s` | | | | 0.80 s, 6.27× |
| `medium` / `same-l` | | | | 2.20 s, 2.27× |

Per-stage breakdown for the warm case (from the CLI's stage lines or
`bench.py`): T5Gemma conditioning / DiT sample (ms per step × steps) / decode.

| `--dit` | condition (ms) | sample (ms) | ms/step | decode (ms) |
|---|---|---|---|---|
| `sm-music` | | | | |
| `medium` | | | | |

## 2. Peak RSS: `--free-models` vs `--no-free-models`

Peak resident set of the generating process, 10 s clip. `--free-models` is
upstream's default and what the README table assumes; `--no-free-models` is
what the sidecar will run with.

| `--dit` | Peak RSS, `--free-models` | Peak RSS, `--no-free-models` | Upstream published (free, 5 s) |
|---|---|---|---|
| `sm-music` | | | 1.62 GB |
| `medium` | | | 3.82 GB |

Resident footprint with both `sm-music` and `medium` (and both codecs) loaded
in one process, if the app is to hot-swap models: _fill in or mark N/A_.

## 3. Wall time vs output length

Cold, `--free-models` (matches upstream's methodology), so the numbers are
comparable to their table. Establishes whether cost is dominated by sampling
steps or by decode.

| `--dit` | 10 s | 30 s | 120 s | Upstream published 30 s / 120 s |
|---|---|---|---|---|
| `sm-music` | | | | 1.53 s / 4.12 s |
| `medium` | | | | 5.06 s / 14.68 s |

Decode share of wall time at 120 s (from stage lines): sm-music ____ %, medium ____ %.

## 4. Loop Mutator budget check (derived from 1)

For a 4-bar loop at 128 BPM (7.5 s, `T_lat = 81`, see D-005):

| `--dit` | Warm generation of 7.5 s (s) | Margin inside one loop (s) | Usable in a one-loop interval? |
|---|---|---|---|
| `sm-music` | | | |
| `medium` | | | |

## 5. Phase 1 additions

Filled during Phase 1, kept here so all machine-specific numbers live in one file.

| Measurement | Value |
|---|---|
| Same-process repeat, sha256 match (fp16 DiT) | |
| Same-process repeat, sha256 match (fp32 DiT) | |
| Cross-process repeat, sha256 match | |
| SAME-S encode→decode round-trip SNR (dB) on a real clip | |
| SAME-L encode→decode round-trip SNR (dB) on a real clip | |

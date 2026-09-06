# BASELINE.md

Phase 0 measurements on the target machine, 2026-09-05. Every number was
measured here with `tools/bench.py` (raw rows: `docs/baseline/bench-2026-09-05.json`,
console log: `docs/baseline/bench-2026-09-05.log`). The "upstream published"
columns are copied from `external/stable-audio-3/optimized/mlx/README.md`
(their M4 Pro / 48 GB run) as the expected range.

## Machine

| Item | Value |
|---|---|
| Model | Mac mini, Apple M4 Pro |
| Unified memory | 48 GB |
| macOS | 26.6.1 (25G76) |
| Xcode / Swift | 26.5 (17F42) / Swift 6.3.2 |
| Python (MLX venv) | 3.12.13 via uv (D-002) |
| `mlx` / `numpy` / `huggingface_hub` | 0.32.2 / 2.5.2 / 1.30.0 |
| Upstream commit | `779434a` (submodule `external/stable-audio-3`) |
| Weights snapshot | `stabilityai/stable-audio-3-optimized` @ `da6edc54` (HF cache, 7.9 GB for sm-music + medium + T5Gemma) |
| Date | 2026-09-05 |

## Gate 0 artefact

| Item | Value |
|---|---|
| Prompt | `dub techno chord stab, 128 bpm` |
| Command | `.venv/bin/python scripts/sa3_mlx.py --prompt "dub techno chord stab, 128 bpm" --dit sm-music --decoder same-s --seconds 10 --seed 1 --out ~/Music/SA3Local/phase0/gate0_sm-music_seed1_10s.wav` |
| Output path | `~/Music/SA3Local/phase0/gate0_sm-music_seed1_10s.wav` |
| `T_lat` | 108 (10.03 s generated, trimmed to 10.0 s) |
| Objective checks (`tools/wavcheck.py`) | 2 ch 44.1 kHz 10.000 s; peak −0.2 dBFS, RMS −18.3 dBFS, crest 18 dB; 11 % of 50 ms blocks silent (stabs with gaps); mean centroid 5.5 kHz; L/R correlation 0.80; onset autocorrelation tempo 63.8 / 129 BPM, i.e. 0.4 BPM from the prompted 128 |
| Listened, sounds like the prompt? | **Pending Mark.** `afplay ~/Music/SA3Local/phase0/gate0_sm-music_seed1_10s.wav` |

## 1. Cold vs warm wall time, 10 s clip

Cold = fresh process, weights already in the HF cache, `--free-models`
(upstream's default). Median of three runs. **Warm is derived, not
measured:** cold wall minus the CLI's own load timings (T5Gemma stage, DiT
load, decoder load). It is the per-generation budget with models resident,
minus nothing else. Phase 1's in-process harness measures the real
second-generation number; expect it to be slightly *lower* than this because
the T5Gemma stage also contains the text encode.

| `--dit` / `--decoder` | Cold wall (s) | Loads (s) | Warm est. (s) | Warm × realtime | Upstream published (5 s, cold) |
|---|---|---|---|---|---|
| `sm-music` / `same-s` | 1.44 | 0.46 | 0.96 | 10.4× | 0.80 s, 6.27× |
| `medium` / `same-l` | 3.57 | 0.98 | 2.60 | 3.8× | 2.20 s, 2.27× |

Per-stage, warm case (median of the three cold runs; conditioning ≈ 60 ms):

| `--dit` | condition (ms) | sample (ms) | ms/step (8 steps) | decode (ms) | unpatch + WAV (ms) |
|---|---|---|---|---|---|
| `sm-music` | 60 | 781 | 98 | 176 | 4 |
| `medium` | 60 | 2058 | 257 | 481 | 4 |

Observation: the very first `medium` run in a fresh process cost 5.23 s
against 3.57 s for the next two, and the first `sm-music` run of the session
(the Gate 0 clip, made while the download was still running) cost 3.55 s
against 1.44 s. That is Metal shader compilation and page-in, paid once per
process. The sidecar must run a throwaway generation at startup so the first
user-visible generation is not the slow one (bead filed under Phase 2).

## 2. Peak memory: `--free-models` vs `--no-free-models`

10 s clip. RSS is the OS's maximum resident set size for the process (via
`/usr/bin/time -l`). "MLX peak" is the CLI's own Metal allocator peak, which
is the number upstream's README table reports.

| `--dit` | RSS, free | RSS, no-free | MLX peak, free | MLX peak, no-free | Upstream published (MLX peak, free, 5 s) |
|---|---|---|---|---|---|
| `sm-music` | 1.27 GB | 2.14 GB | 1.68 GB | 2.21 GB | 1.62 GB |
| `medium` | 2.96 GB | 5.32 GB | 3.89 GB | 6.55 GB | 3.82 GB |

Resident footprint with both DiTs and both codecs loaded in one process: not
measured (the CLI loads one of each). Upper bound from file sizes: T5Gemma
0.57 + sm-music 0.92 + medium 2.9 + SAME-S 0.43 + SAME-L 3.4 ≈ 8.2 GB of
weights, plus the ~3 GB of activations `medium` needs at 10 s. Fits.

## 3. Wall time vs output length

Cold, `--free-models`, matching upstream's methodology so the numbers are
comparable to their table.

| `--dit` | 10 s | 30 s | 120 s | Upstream published 30 s / 120 s |
|---|---|---|---|---|
| `sm-music` | 1.44 | 2.35 | 5.52 | 1.53 / 4.12 |
| `medium` | 3.57 | 6.36 | 16.44 | 5.06 / 14.68 |

Sample vs decode at 120 s: `sm-music` 3078 ms sample / 1951 ms decode, decode
is 35 % of wall; `medium` 8965 / 6451 ms, decode is 39 %. Cost is dominated by
sampling at every length, but decode grows faster than sampling with length
(chunked decode is linear in `T_lat`, sampling is sub-linear here), so at the
loop lengths this app cares about decode is under 20 % of the budget.

This machine runs 10–50 % slower than upstream's published M4 Pro row at
every cell. Candidates: `mlx` 0.32.2 vs whatever they used, background load
(the HF download finished minutes before the sweep), and thermals. Not
investigated; the absolute numbers are comfortably inside budget either way.

## 4. Loop Mutator budget check (derived from 1)

For a 4-bar loop at 128 BPM (7.5 s, `T_lat = 81`, D-005), scaling the 10 s
(`T_lat = 108`) warm stages by 81/108 for sample and decode, conditioning
constant:

| `--dit` | Warm generation of 7.5 s (s, est.) | Margin inside one 7.5 s loop (s) | Usable in a one-loop interval? |
|---|---|---|---|
| `sm-music` | ≈ 0.8 | ≈ 6.7 | Yes, wide |
| `medium` | ≈ 2.0 | ≈ 5.5 | Yes, with the start-at-previous-top schedule; the handoff's 2× realtime estimate was pessimistic for this machine |

## 5. Phase 1 additions

Filled during Phase 1, kept here so all machine-specific numbers live in one file.

| Measurement | Value |
|---|---|
| True warm (resident models, second generation in-process), sm-music 10 s | |
| True warm, medium 10 s | |
| Same-process repeat, sha256 match (fp16 DiT) | |
| Same-process repeat, sha256 match (fp32 DiT) | |
| Cross-process repeat, sha256 match | |
| SAME-S encode→decode round-trip SNR (dB) on a real clip | |
| SAME-L encode→decode round-trip SNR (dB) on a real clip | |

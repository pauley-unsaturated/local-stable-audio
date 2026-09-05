# local-stable-audio

A native macOS instrument built on [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3),
running entirely on Apple Silicon through Stability's pure-MLX runtime. No
PyTorch, no CUDA, no network after the weights are downloaded.

Two instruments sit on top of local generation:

- **Latent Explorer.** A 2D pad whose four corners are saved `(seed, prompt)`
  anchors. Position blends them: slerp on the initial noise, lerp on the text
  conditioning. Release to regenerate; revisits are cached and instant.
- **Loop Mutator.** A bar/BPM-clocked loop that asks the model for a variation
  every N loops and swaps it in at the boundary, never stalling the render
  thread. Every mutation is a child of the last, so you can walk back up the
  lineage and branch.

Target hardware is a Mac mini M4 Pro with 48 GB. It is Apple Silicon only.

## Status

Scaffolding. Nothing generates audio from this repo yet. The plan is gated:
each phase has acceptance criteria and the next phase does not start until the
previous gate is green. Progress is tracked in [beads](https://github.com/steveyegge/beads)
(`bd ready`), one epic per phase.

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Upstream MLX CLI runs here | `.wav` on disk, `BASELINE.md` filled in |
| 1 | Determinism harness | Bit-identical repeats; slerp midpoint is plausible audio |
| 2 | `sa3d` sidecar + CLI client | Generate, cancel mid-sample, encode/decode round-trip over the socket |
| 3 | Swift shell + audio engine | Play a buffer; swap at a boundary with no glitch under Instruments |
| 4 | Loop Mutator | 30-minute run, no dropout, lineage intact |
| 5 | Latent Explorer | 2D pad, instant cached revisits, no stale-result races |
| 6 | First-run download UX | Clean install reaches audio without a terminal |
| 7 | Packaging | Signed, notarized, runs on a second Mac |

The founding brief is [`docs/HANDOFF.md`](docs/HANDOFF.md). Where this repo
departs from it, or had to choose because the upstream source was ambiguous,
the choice is logged in [`DECISIONS.md`](DECISIONS.md). Machine-specific
numbers live in [`BASELINE.md`](BASELINE.md).

## Architecture

```
┌─────────────────────────────┐
│  SA3Local.app  (SwiftUI)    │
│  ├─ AVAudioEngine graph     │   lock-free render block,
│  ├─ Explorer / Looper UI    │   atomic buffer swap at loop boundaries
│  └─ SidecarClient           │
└──────────┬──────────────────┘
           │  UNIX domain socket, newline-delimited JSON
           │  requests carry run_id; every event echoes it
┌──────────▼──────────────────┐
│  sa3d  (Python, MLX venv)   │
│  ├─ model residency mgr     │   models stay loaded between generations
│  ├─ generate / encode /     │   one accelerator lock; cancel interrupts
│  │   interpolate / continue │   the sampler at step granularity
│  └─ HF download w/ progress │
└─────────────────────────────┘
```

The engine is Python because Stability's MLX implementation is Python. Porting
the DiT, the SAME codec and T5Gemma to `mlx-swift` is deliberately out of scope.
Audio crosses the boundary as a raw float32 file in a cache directory; at
roughly one second per generation, file I/O is not the bottleneck.

## Layout

```
local-stable-audio/
├── CLAUDE.md / AGENTS.md    agent instructions (beads workflow, ground rules)
├── DECISIONS.md             assumptions and departures from the handoff, with file:line
├── BASELINE.md              measured numbers for this machine
├── docs/HANDOFF.md          the founding brief
├── docs/IDEAS.md            speculative directions the architecture affords, tied to the Research Vault
├── external/stable-audio-3  upstream, pinned as a submodule (read, never edited)
├── sidecar/                 sa3d.py, engine.py, latents.py, weights.py, protocol.py   (Phase 2)
├── app/SA3Local/            SwiftUI app: SidecarClient, AudioEngine, Explorer/, Looper/, Lineage/  (Phase 3+)
├── tests/                   test_determinism.py, test_protocol.py, AudioEngineTests.swift
└── tools/                   bench.py, sa3ctl.py
```

## Getting started (Phase 0)

```bash
git clone --recurse-submodules git@github.com:pauley-unsaturated/local-stable-audio.git
cd local-stable-audio/external/stable-audio-3/optimized/mlx
./install.sh -y --python 3.12
./sa3 --prompt "dub techno chord stab, 128 bpm" --dit sm-music --decoder same-s --seconds 10 --seed 1
```

Weights (about 1.3 GB for `sm-music` plus 541 MB for the shared T5Gemma
encoder) are pulled from the ungated `stabilityai/stable-audio-3-optimized`
repo on first use and symlinked into `models/mlx/` from the Hugging Face cache.
No account is needed, though anonymous downloads are rate-limited.

Task tracking:

```bash
brew install beads     # bd
bd ready               # what is unblocked right now
bd show <id>
```

## Prior art

- [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3),
  `optimized/mlx`. The engine. MIT-licensed code; the weights are under the
  Stability AI Community License and T5Gemma under Google's Gemma Terms of Use.
- [little-scale/mucking_around_time_with_stable_audio_3](https://github.com/little-scale/mucking_around_time_with_stable_audio_3).
  A FastAPI + browser tool whose backend event contract (stage, sampler step,
  metrics, audio ready, complete, error) this project's protocol borrows from.
  Its MLX path is its least-tested; upstream is the reference for all MLX code.

## License

Code in this repository: TBD. Model weights and generated outputs remain under
their own licenses (see above); the app surfaces an acknowledgement on first run.

# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## What this is

A native macOS instrument (SwiftUI shell + Python MLX sidecar) on Stable Audio 3,
for a Mac mini M4 Pro / 48 GB. Two instruments: a Latent Explorer (2D pad over
noise and prompt anchors) and a Loop Mutator (clocked loop that mutates at
boundaries and keeps a lineage tree). The founding brief is `docs/HANDOFF.md`;
read it before touching anything. The author and reader are fluent in Darwin
internals, AVFoundation, CoreAudio and Metal. Do not explain basics.

## Ground rules (from HANDOFF §1, binding)

1. **Verify before you build.** Every API signature, tensor shape, latent frame
   rate and repo ID in the handoff is recollection. Read
   `external/stable-audio-3/optimized/mlx` and confirm. Source wins over the
   handoff; note each discrepancy in `DECISIONS.md`.
2. **Phase gates.** One beads epic per phase, each with a `Gate N` task. Do not
   start phase N+1 while Gate N is open. If a gate fails, stop and report; do
   not work around it.
3. **No PyTorch at runtime.** Offline one-time weight conversion only, and only
   if unavoidable.
4. **Keep `DECISIONS.md`.** Every assumption forced by ambiguous source goes in
   with `file:line`. Append-only; supersede, don't edit.
5. **No UI before the engine is proven.** Phases 0 and 1 produce CLI output and
   log files only.
6. **Open questions go to Mark, not decided alone.** They exist as `decision`
   beads labelled `needs-mark`. If a task is blocked on one, say so and stop.

## Layout

| Path | What |
|---|---|
| `docs/HANDOFF.md` | The brief. Phase plan, architecture, traps. |
| `docs/IDEAS.md` | Speculative, unscheduled directions. Read for context, do not build from it without a bead. |
| `DECISIONS.md` | Assumptions and departures, `D-NNN`, with `file:line`. |
| `BASELINE.md` | Numbers measured on this machine. Fill, never estimate. |
| `external/stable-audio-3` | Upstream, git submodule pinned at `779434a`. Read-only. See D-001. |
| `sidecar/` | `sa3d.py` (socket server), `engine.py` (wrapper over upstream), `latents.py`, `weights.py`, `protocol.py`. Phase 2. |
| `app/SA3Local/` | SwiftUI app. `SidecarClient.swift`, `AudioEngine.swift`, `Explorer/`, `Looper/`, `Lineage/`. Phase 3+. |
| `tests/` | `test_determinism.py`, `test_protocol.py` (no MLX), `AudioEngineTests.swift`. |
| `tools/` | `bench.py`, `sa3ctl.py`. |

Prior-art clones for reading (not part of the repo):
`~/Programs/stable-audio-3` (same as the submodule) and
`~/Programs/mucking_around_time_with_stable_audio_3` (reference event contract
in `backends/base.py` and `protocol.py`; its MLX code is a hint, not a dependency).

## Toolchain

- Apple M4 Pro, macOS 26.6.1, Xcode 26.5, Swift 6.3.2. Apple Silicon only.
- `uv` (Homebrew) manages Python. The MLX venv is Python 3.12 (D-002), created
  by upstream's `install.sh` inside the submodule:
  `cd external/stable-audio-3/optimized/mlx && ./install.sh -y --python 3.12`.
  Run upstream via `./sa3 ...` there, or `.venv/bin/python` directly. The venv,
  `models/mlx/` symlinks and `output/` inside the submodule are gitignored.
- `bd` (beads 1.2.2, Homebrew) for tasks; prefix `lsa`.
- No `hf` CLI is installed; `huggingface_hub` inside the venv handles downloads.

## Build & test

Phase 0/1 (now):

```bash
cd external/stable-audio-3/optimized/mlx
./sa3 --prompt "..." --dit sm-music --decoder same-s --seconds 10 --seed 1 --out /abs/path.wav
.venv/bin/python /path/to/tests/test_determinism.py     # Phase 1 harness, runs in this venv
```

Phase 2+: `sidecar/` gets its own `uv` project that puts
`external/stable-audio-3/optimized/mlx` on `sys.path`; `pytest tests/` for the
protocol tests (must pass without `mlx` installed). Phase 3+: `xcodebuild` /
`swift test` under `app/`. Update this section when those exist.

## Verified facts (2026-09-05, upstream `779434a`; details and line numbers in DECISIONS.md)

- Model IDs are `--dit sm-music | sm-sfx | medium`, `--decoder same-s | same-l`.
  Not `small-music`. `sm-*` pairs with `same-s`, `medium` with `same-l`.
- Weights: `stabilityai/stable-audio-3-optimized`, files under `MLX/`. Ungated.
  `ensure_local()` in `scripts/weights.py` downloads into the HF cache and
  symlinks into `models/mlx/`.
- Latents: `x_T` is `(1, 256, T_lat)`, fp16 unless `--dit-dtype fp32`.
  `T_lat = ceil(seconds * 44100 / 4096)`; frame rate ≈ 10.767 Hz.
- Two seeds: `x_T` from `mx.random.key(seed)`; the pingpong sampler re-noises
  every step from `key(seed + 1)`. Both matter for determinism and caching.
- `sample_flow_pingpong(model_fn, x, sigmas, seed, paste_back, on_step,
  before_step)` accepts a caller-supplied `x`. That is the injection seam.
  `on_step(i+1, total)` fires after every step; raising from it aborts the loop.
- Precision: SAME decoder always FP32, T5Gemma always fp16, DiT fp16 by default.
  Never "optimise" the decoder to fp16; it fails silently, not loudly.
- `--free-models` defaults on. The sidecar runs the equivalent of
  `--no-free-models`.
- Audio-to-audio (`--init-audio` + `--init-noise-level` σmax) and inpainting
  (`--inpaint-range`) exist in the MLX runtime. `--seconds` defaults to 30,
  `--steps` to 8, `--cfg` to 1.0.

## Conventions

- **Transport:** UNIX domain socket in the app container, NDJSON. Never TCP.
- **Render thread:** no locks, no allocation, no ObjC/Swift refcount traffic,
  no logging. Buffers swap only at loop boundaries via acquire/release atomics.
- **Cancel** must interrupt the sampler, not discard its result.
- **Sidecar orphans** are a bug: parent-death guard plus atexit in `sa3d`.
- **Don't debug the reference repo's MLX code.** Read upstream directly.
- **Upstream is read-only.** Needed changes are recorded in `DECISIONS.md` and
  carried as a patch or monkeypatch in `sidecar/`, never edited in the submodule.
- **Generated audio, weights, venvs stay out of git** (`.gitignore` covers
  `*.wav`, `*.npz`, `.venv/`, `output/`, `models/mlx/`).
- Commits: imperative subject, body says which bead(s) it closes. Keep the
  `Co-Authored-By` trailer the harness adds.

## Beads workflow for this repo

- One epic per phase (`phase-N` label) with a `Gate N` task that depends on
  every task in the phase. The next phase's tasks depend on that gate, so
  `bd ready` only ever shows work that is actually unblocked.
- `decision` beads labelled `needs-mark` are Mark's to answer. Three of them
  block design tasks (AUv3 → audio engine; default model → looper budget;
  anchor semantics → pad).
- Claim before starting (`bd update <id> --claim`), close with a reason that
  names the artefact (file, number in `BASELINE.md`, `D-NNN`).
- Close the epic only when its gate task is closed.
- This repository opts into the **team-maintainer** profile of the managed
  beads block above: agents may commit when a bead closes and `git push` plus
  `bd dolt push` at session end. A current "do not commit/push" instruction
  from Mark still wins.

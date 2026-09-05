# DECISIONS.md

Every assumption made because the source was ambiguous, and every place this
repo deliberately departs from `docs/HANDOFF.md`, goes here with the file and
line that forced the choice. Append-only; supersede rather than edit.

Format: `D-NNN` · date · status (`proposed` / `accepted` / `superseded by D-MMM`)
· context (with `file:line`) · decision · consequences.

Paths are relative to the repo root unless noted. `upstream/` below is shorthand
for `external/stable-audio-3/optimized/mlx/`.

---

## D-001 · 2026-09-05 · accepted · Upstream is pinned as a git submodule, not vendored

**Context.** HANDOFF §0 says "read it, do not vendor it." HANDOFF §9 has
`sidecar/engine.py` as a "thin wrapper over optimized/mlx", which means the
sidecar must import upstream's `models/defs/*` and `scripts/weights.py`.
Upstream's MLX runtime is a scripts-plus-package-directory layout with no
`pyproject.toml` of its own (`upstream/sa3` execs `.venv/bin/python
scripts/sa3_mlx.py`; the root `pyproject.toml` is the PyTorch package), so it is
not pip-installable.

**Decision.** `external/stable-audio-3` is a git submodule pinned at
`779434a` (main as of 2026-09-05, "Merge pull request #99 ... same-ae-limiter").
The sidecar puts `external/stable-audio-3/optimized/mlx` on `sys.path`, exactly
as `upstream/scripts/sa3_mlx.py:20-22` does for itself. Upstream files are never
edited in place; a needed change is recorded here and carried as a runtime
monkeypatch or a patch file under `sidecar/patches/`.

The reference integration (`little-scale/mucking_around_time_with_stable_audio_3`,
commit `b4c36b3`) is not pinned in the repo. It was read from a sibling clone at
`~/Programs/mucking_around_time_with_stable_audio_3` for its event contract only.

**Consequences.** `git clone --recurse-submodules`. Bumping upstream is a
deliberate commit. Phase 7 packaging copies `optimized/mlx` into the bundle from
the submodule.

## D-002 · 2026-09-05 · accepted · Python 3.12 for the MLX venv

**Context.** `upstream/install.sh:24` defaults to Python 3.11; HANDOFF §2 says
"add `--python 3.12` to pin". `uv` on this machine already manages 3.12.

**Decision.** `./install.sh -y --python 3.12`. Record the resulting `mlx`
version in `BASELINE.md`.

## D-003 · 2026-09-05 · accepted · Model identifiers follow the MLX CLI, not the handoff

**Context.** HANDOFF §2, §6 and §12 say `small-music` / `small-sfx` / `medium`.
The MLX CLI (`upstream/scripts/sa3_mlx.py:377-383`, `upstream/README.md`
"Three models, three modes") uses `--dit sm-music | sm-sfx | medium` and
`--decoder same-s | same-l`. The root README's `small-music` names are the
PyTorch package's model IDs.

**Decision.** This repo, its protocol, and its UI use `sm-music`, `sm-sfx`,
`medium`, `same-s`, `same-l`. `sm-*` pairs with `same-s`; `medium` with `same-l`.

## D-004 · 2026-09-05 · accepted · Hugging Face repo IDs (verified)

**Context.** HANDOFF §5 gives the ungated repo as "reportedly
`stable-audio-3-optimized`".

**Verified.** `upstream/scripts/weights.py:16`: `REPO_ID =
"stabilityai/stable-audio-3-optimized"`, files under an `MLX/` prefix
(`MLX/dit_sm-music_f16.npz`, `MLX/same_s_decoder_f32.npz`,
`MLX/t5gemma_f16.npz`, ...). Bundle sizes from `weights.py:57-61`: sm-music
1.3 GB, sm-sfx 1.3 GB, medium 5.9 GB; T5Gemma 541 MB shared. Gated fp32 repos
are `stabilityai/stable-audio-3-small-music`, `-small-sfx`, `-medium`
(root `README.md:19-21`).

**Decision.** Route 2 (ungated optimized repo) only, as HANDOFF §5 prefers.
`ensure_local()` streams into the normal HF cache and symlinks into
`models/mlx/` (`weights.py:141-160`); the app mirrors this into
`Application Support/SA3Local/models/`. Note `weights.py:78-108` prints a
login nudge when no HF token is set: anonymous downloads have a ~50 GB/day
soft cap. That is a first-run UX consideration for Phase 6, not a blocker.

## D-005 · 2026-09-05 · accepted · Latent geometry (verified)

**Context.** HANDOFF §3 says "the latent frame rate is not something I know —
derive it".

**Verified.**
- `upstream/scripts/sa3_mlx.py:36-37`: `SAMPLE_RATE = 44100`,
  `SAMPLES_PER_LATENT = 4096` ("PatchedPretransform downsample × SAME 16×
  expansion"). `upstream/models/defs/audio_encoding.py:13-15` gives the
  factorisation: `PATCH_SIZE = 256`, `ENCODER_STRIDE = 16`.
- `sa3_mlx.py:510`: `T_lat = max(1, ceil(seconds * 44100 / 4096))`. Decoder
  independent by design (comment at 502-509).
- `sa3_mlx.py:700-701`: `x_T = mx.random.normal((1, 256, T_lat), dtype=dtype,
  key=mx.random.key(seed))`, where `dtype` is fp16 unless `--dit-dtype fp32`.
- Decoder output is patches `[B, 512, T_lat*16]` unpatched to
  `[B, 2, T_lat*4096]` (`sa3_pipeline.py:160`).

**Derived.** Latent frame rate = 44100 / 4096 ≈ **10.767 frames/s**
(one frame ≈ 92.88 ms). A 4-bar loop at 128 BPM (7.5 s) is `T_lat = 81`
frames = 7.523 s generated, trimmed to 7.5 s. Loop lengths are therefore not
latent-aligned in general; Phase 4's seam handling must assume a fractional
trailing frame.

## D-006 · 2026-09-05 · proposed · Two seeds: x_T and the sampler's per-step re-noise

**Context.** HANDOFF §6 treats `x_T` as the only stochastic input. The pingpong
sampler (`upstream/models/defs/sa3_pipeline.py:104-153`) draws **fresh noise at
every step** from `mx.random.key(seed)` with `key, sub = mx.random.split(key)`
(lines 127, 137-140), and `sa3_mlx.py:782` passes `seed=args.seed + 1` for it.

**Decision (to validate in Phase 1).** The Explorer slerps `x_T` between anchor
seeds and holds the sampler key fixed (derived from a separate, constant
`sampler_seed`). Whether the per-step noise should instead also be slerped, or
tracked to the nearest anchor, is decided by listening in Phase 1 task "Slerp
midpoint". The cache key must include both seeds.

## D-007 · 2026-09-05 · accepted · Injection seam for a caller-supplied x_T

**Context.** HANDOFF §3 asks where `x_T` can be injected.

**Verified.** `sample_flow_pingpong(model_fn, x, sigmas, seed, paste_back,
on_step, before_step)` (`sa3_pipeline.py:104-106`) takes the initial latent
`x` directly. There is no seam inside `sa3_mlx.py:main()` (lines 344-891 are one
function) short of the noise construction at 700-708.

**Decision.** `sidecar/engine.py` re-implements the orchestration of `main()`
(load, condition, schedule, sample, decode) by calling the `models.defs` pieces,
and never imports or shells out to `sa3_mlx.py`. The audio-to-audio mix
`noise = init_latents * (1 - σmax) + pure_noise * σmax` (line 705) and the
inpaint `local_add_cond` / `paste_back` construction (lines 711-729) are
replicated verbatim so Phase 4's continuation seam has the same inputs upstream
uses.

## D-008 · 2026-09-05 · proposed · Cancellation lands at step granularity via `on_step`

**Context.** HANDOFF §4.2: "`cancel` must actually interrupt the sampling
loop." `on_step(i+1, total)` fires after `mx.eval(x)` for every step
(`sa3_pipeline.py:142-144`).

**Decision.** Phase 2 raises a `Cancelled` exception from `on_step` when the
run's flag is set; the sampler has no `try` so it propagates out cleanly. Cancel
latency is therefore at most one step (~100 ms on `sm-music`, ~300-600 ms on
`medium` per upstream's numbers). If that proves too coarse, the minimal patch is
a `should_stop: Callable[[], bool]` argument checked before `model_fn`; record
it here if adopted.

## D-009 · 2026-09-05 · accepted · Precision policy (verified)

**Verified.** Decoder always FP32 (`sa3_mlx.py:799-800`, README "Notes on the
design": SAME-S differential attention catastrophically cancels in FP16).
T5Gemma always fp16. DiT fp16 by default, `--dit-dtype fp32` "for bit-exact
reproducibility" (README flag table).

**Decision.** Same policy in the sidecar, not configurable from the UI. Phase 1
runs the determinism harness at both DiT dtypes and records which one the
Explorer cache key assumes.

## D-010 · 2026-09-05 · accepted · Baseline durations: 10 / 30 / 120 s, plus upstream's own table

**Context.** HANDOFF §2 asks for 10 s (cold/warm), 30 s and 120 s. Upstream's
`--seconds` default is 30 (`sa3_mlx.py:415`) and its `scripts/benchmark.py`
sweeps 5/30/120/380 s. Upstream's README already publishes an M4 Pro / 48 GB
table (`upstream/README.md` "Sample run on M4 Pro / 48 GB").

**Decision.** `BASELINE.md` records the handoff's matrix measured here, and
reproduces upstream's published M4 Pro table beside it as the expected range.
"Warm" means the second generation in the same process with `--no-free-models`,
which upstream's benchmark does not measure; `tools/bench.py` adds it.

## D-011 · 2026-09-05 · accepted · Phases stay strictly sequential

**Context.** HANDOFF §1 ground rule 2: "Do not start phase N+1 with phase N
red." Phases 4 (Loop Mutator) and 5 (Latent Explorer) are technically
independent once Phase 3 is green.

**Decision.** The beads graph gates Phase 5 on Phase 4's gate task, as the
handoff orders them. Reordering is a one-line `bd dep` change if Mark prefers.
Beads does not allow an epic to block a task, so each phase has an explicit
"Gate N" task that depends on all of that phase's tasks; the next phase's tasks
depend on that gate.

## D-012 · 2026-09-05 · proposed · Audio handoff format

**Context.** HANDOFF §4.1: raw `float32` interleaved file in a cache dir.
Upstream writes 16-bit PCM WAV (`sa3_mlx.py:231-243`, README `--out`).

**Decision.** The sidecar writes the decoded `[2, N]` float32 array as
interleaved stereo float32 with no header to `<cache>/<run_id>.f32`, plus the
sample count and channel count in the `audio_ready` event. The app maps it into
an `AVAudioPCMBuffer` (deinterleaving once, off the render thread). WAV export
for the user is a separate, later concern (see the open decision on output
location).

## D-013 · 2026-09-05 · accepted · AUv3 is on the roadmap; standalone app with recording ships first

**Context.** HANDOFF §12.4. Mark, 2026-09-05: "Yes, but probably just an app
to begin with that can do recording."

**Decision.** Phase 3's render path is built as an `AUAudioUnit` subclass
whose `internalRenderBlock` reads the atomic buffer pointer, hosted in-app
through `AVAudioUnit`. Nothing in the render graph may assume it runs inside
`SA3Local.app`; parameters go through an `AUParameterTree`, tempo and transport
come from the host musical-context blocks (the app is the host until Phase 8).
Recording of the live output is a Phase 4 task (`lsa-8gc.8`). Phase 8 epic
(`lsa-xmd`) packages the same unit as an app extension after packaging is done.

**Consequences.** The sidecar cannot live inside an app extension's sandbox;
Phase 8 has to choose XPC to the container app or a mandatory companion app.
Flagged there, not solved here.

## D-014 · 2026-09-05 · accepted · Explorer pad: X slerps seeds, Y lerps prompts

**Context.** HANDOFF §12.2 offered "seeds, prompts, or both on separate
axes". Mark, 2026-09-05: "Seeds and prompts please."

**Decision.** Read as both, on separate axes. X interpolates `x_T` between
seed A and seed B by slerp; Y interpolates T5Gemma conditioning between prompt
P and prompt Q by lerp. This is the four-corner bilinear model of HANDOFF §6
with corners constrained to `{A,B} × {P,Q}`, which keeps each axis legible.
`sigma_max` and CFG remain separate sliders. Cache key: `(A, B, P, Q, x, y,
sampler_seed, sigma_max, cfg, steps, seconds, dit, decoder, dit_dtype)`.

If Mark meant a single blended pair per corner, revert `lsa-8s5.1` to the
original description; the sidecar's `interpolate` request supports both.

## D-015 · 2026-09-05 · accepted · Render to disk, play back live

**Context.** HANDOFF §12.3. Mark, 2026-09-05: "Render to disk, play back
live."

**Decision.** Every generation from either instrument is persisted to an
app-managed library as a WAV plus a JSON sidecar (prompt, seeds, `sigma_max`,
CFG, steps, model, parent id, timestamps). Playback is from in-memory buffers.
The library root defaults to `~/Music/SA3Local` and is user-changeable; that
default is my assumption, not Mark's instruction. Filename convention:
`<yyyymmdd-hhmmss>_<dit>_<seed>_<slug>.wav`. Live output is also recordable
(`lsa-8gc.8`), with a sidecar listing the lineage ids played and their
sample-accurate swap offsets. The lineage tree must be rebuildable from
sidecars alone.

## D-016 · 2026-09-05 · proposed · Default model: `sm-music` + `same-s`, `medium` resident as opt-in HQ

**Context.** HANDOFF §12.1. Mark asked for the choices and a recommendation
for the M4 Pro / 48 GB.

**Verified inventory** (`stabilityai/stable-audio-3-optimized`, `MLX/`, listed
2026-09-05): three inference DiTs (`dit_sm-music_f16`, `dit_sm-sfx_f16`,
`dit_medium_f16`), their three `-base` counterparts (rectified-flow weights
without ARC post-training, used by the LoRA trainer), SAME-S and SAME-L
encoder/decoder pairs in f32, and T5Gemma f16. There are no quantized MLX
bundles; INT8/FP8 variants exist only for ONNX, TensorRT, TFLite and cpu-amx.
`large` (2.7B) is API-only. Base checkpoints are not inference defaults.

**Recommendation.** Default to `sm-music` + `same-s` for both instruments.
Keep `medium` + `same-l` resident as an opt-in "HQ" mode for the Explorer and
for render-to-disk, and `sm-sfx` as a third selectable DiT (it shares the
SAME-S codec, so it costs 0.9 GB more, nothing else). All of it resident is
about 9 GB of weights, which the machine absorbs. Rationale: the Loop Mutator's
budget is the binding constraint, and upstream's own M4 Pro numbers put
`sm-music` at well under a second for a 7.5 s loop and `medium` at roughly
three seconds, which fits a one-loop interval only with the
start-at-previous-top scheduling and thin thermal headroom. The Explorer
tolerates `medium`'s two-to-three-second release-to-generate; the looper does
not by default. Confirm after Phase 0 listening: if `sm-music` is audibly
too thin for musical loops, flip the default and let the margin readout police
the looper.

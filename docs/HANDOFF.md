# Handoff: SA3 Local — a native macOS instrument on Stable Audio 3 (MLX)

**Audience:** Claude Code, starting from an empty repo.
**Author of this brief:** Mark (Principal Engineer, CoreAudio / on-device ML background). Assume the reader of your output is fluent in Darwin internals, AVFoundation, and Metal. Do not explain basics.

---

## 0. Target hardware and goal

- **Machine:** Mac mini M4 Pro, 48 GB unified memory. Apple Silicon only; this will never run on Intel or CUDA.
- **Goal:** a native macOS app (SwiftUI) that
  1. downloads Stable Audio 3 weights from Hugging Face on first run, with visible progress;
  2. generates audio locally from text prompts via the pure-MLX runtime (no PyTorch at runtime);
  3. exposes two *instruments* on top of that: a **Latent Explorer** and a **Loop Mutator**.

**Prior art — read it, do not vendor it:**

- Upstream: `https://github.com/Stability-AI/stable-audio-3`, specifically `optimized/mlx`. This is the real engine. Apple-Silicon-native, Metal-backed, no PyTorch / transformers / stable-audio-tools at runtime.
- Reference integration: `https://github.com/little-scale/mucking_around_time_with_stable_audio_3`. A FastAPI + browser tool with CUDA and MLX backends. Its `backends/base.py` event contract (stage, tensor preview, sampler-step, metrics, audio-ready, completion, error) is a good protocol to steal. Its `--backend mlx` path is the least-tested part of it — the author developed against an RTX 4080 — so treat its MLX code as a hint, not a dependency. It pins upstream at commit `124e8a7`.

---

## 1. Ground rules

1. **Verify before you build.** Every API signature, tensor shape, latent frame rate, and repo ID in this document is my recollection or secondhand. Read the actual upstream source and confirm. Where this doc and the source disagree, the source wins — and note the discrepancy.
2. **Phase gates.** Each phase below has acceptance criteria. Do not start phase N+1 with phase N red. If a gate fails, stop and report rather than working around it.
3. **No PyTorch at runtime.** PyTorch is acceptable only for one-time offline weight conversion, and only if unavoidable.
4. **Keep a `DECISIONS.md`.** Every assumption you had to make because the source was ambiguous goes in it, with the file and line that forced the choice.
5. **Don't build UI before the engine is proven.** Phase 0 and 1 produce CLI output and log files, nothing else.

---

## 2. Phase 0 — Prove the runtime (do this first, before writing any Swift)

```bash
git clone https://github.com/Stability-AI/stable-audio-3.git
cd stable-audio-3/optimized/mlx
./install.sh            # add --python 3.12 to pin
./sa3 --prompt "dub techno chord stab, 128 bpm" --seconds 10
```

Missing weights are downloaded from HF on first use and symlinked into `models/mlx/` from the HF cache. That symlink convention matters later (§4).

**Measure and record in `BASELINE.md`:**

| Measurement | Why it matters |
|---|---|
| Cold vs. warm wall time, small-music, 10 s | Cold includes model load; warm is your real per-generation budget |
| Same for `medium` | Decides whether medium is usable in the Loop Mutator at all |
| Peak RSS with default `--free-models` and with `--no-free-models` | Explorer and Looper both need models resident; confirm the resident footprint fits |
| Wall time for 30 s and 120 s outputs | Establishes whether cost is dominated by sampling steps or decode |

**Published figures to check yours against** (these are upstream's, on unspecified hardware — expect deviation): small-music / small-sfx around 1 s for a 10 s clip, roughly 10× realtime, 1.6 GB peak; medium around 5 s for a 10 s clip, roughly 2× realtime, 3.8 GB peak. Those peaks assume `--free-models`, where T5Gemma is freed after conditioning and the DiT after sampling.

**Gate 0:** a `.wav` on disk that sounds like the prompt, plus `BASELINE.md` populated.

---

## 3. Phase 1 — Determinism and the reproducibility gate

This is the phase most likely to kill the Latent Explorer, so do it early.

Write a Python harness (in the `optimized/mlx` venv) that:

1. Generates twice with identical `(seed, prompt, steps, cfg, seconds)` in the **same process**. SHA-256 the raw float output. Must match bit-for-bit.
2. Same, across two **separate process invocations**. Must match.
3. Generates with seeds `A` and `B`, then with a **slerp midpoint** of the two initial noise latents. Confirm the midpoint is a plausible audio interpolation and not noise.
4. Encodes a real audio file through the SAME encoder, decodes it back, and measures round-trip error. This proves you can get *into* latent space, which the audio-conditioning path needs.

**Things to determine here and write down:**

- The exact shape and dtype of the initial latent `x_T`. Latents are documented as stereo, 44.1 kHz, 256-dimensional. **The latent frame rate is not something I know — derive it** from `seconds → latent frame count` in the source. Everything in the Explorer depends on it.
- Whether MLX's RNG is seedable and reproducible across process boundaries (`mx.random.seed`), and whether any op in the DiT or decoder introduces nondeterminism.
- Where in the sampling loop you can inject a caller-supplied `x_T` rather than letting it sample one internally. If there's no clean seam, note what a minimal patch would look like.

**Gate 1:** items 1–3 pass. If (2) fails but (1) passes, the Explorer still works within a session — record that constraint and continue. If (1) fails, **stop and report**; the whole Explorer concept needs rethinking.

---

## 4. Architecture

### 4.1 The core decision: Swift shell + Python MLX sidecar

MLX's SA3 implementation is Python. There is an `mlx-swift`, but porting the DiT, the SAME codec, and T5Gemma conditioning to it is a multi-week project with a large correctness surface. **Do not attempt it now.**

```
┌─────────────────────────────┐
│  SA3Local.app  (SwiftUI)    │
│  ├─ AVAudioEngine graph     │
│  ├─ Explorer / Looper UI    │
│  └─ SidecarClient           │
└──────────┬──────────────────┘
           │  UNIX domain socket
           │  NDJSON control + event stream
┌──────────▼──────────────────┐
│  sa3d  (Python, MLX venv)   │
│  ├─ model residency mgr     │
│  ├─ generate / encode /     │
│  │   interpolate / continue │
│  └─ HF download w/ progress │
└─────────────────────────────┘
```

- **Transport:** UNIX domain socket in the app's container, newline-delimited JSON. Not TCP — no port collisions, no LAN exposure, no firewall prompt. (The reference repo uses `0.0.0.0:7861` with *no authentication*; we are not doing that.)
- **Audio handoff:** write generated audio as raw `float32` interleaved to a file in a cache dir, return the path. A 8-second stereo 44.1 kHz f32 buffer is ~2.8 MB — file I/O is nowhere near the bottleneck when generation costs ~1 s. Do **not** build shared memory or protobuf plumbing in v1. Revisit only if profiling says so.
- **Lifecycle:** app spawns `sa3d` as a child, `posix_spawn` with the venv interpreter. Health ping on an interval. On sidecar crash: restart with exponential backoff, surface a non-modal banner, preserve UI state. Kill on app termination — and also install an atexit/parent-death guard in the sidecar so an orphan can't sit on 4 GB of wired memory.
- **Model residency:** launch the sidecar with the equivalent of `--no-free-models`. Both instruments need repeated generations; paying model load per generation is the difference between an instrument and a batch tool. 48 GB is ample.

### 4.2 Protocol sketch

Requests: `generate`, `encode`, `decode`, `interpolate`, `continue`, `cancel`, `status`, `download_weights`.
Events (mirroring the reference repo's contract): `stage`, `sampler_step {i, n}`, `metrics`, `audio_ready {path, run_id}`, `complete`, `error`.

Every request carries a `run_id`. Every event carries it back. The sidecar serializes work behind a single accelerator lock — concurrent requests queue rather than thrash Metal.

`cancel` must actually interrupt the sampling loop, not just discard the result. Find the step callback and check a cancellation flag there. Without this, the Explorer feels broken as soon as the user moves the pad twice quickly.

---

## 5. Weights and first-run download

Two supply routes — **verify both repo IDs before coding**:

1. **Gated fp32 repos** (e.g. `stabilityai/stable-audio-3-small-sfx`). Requires accepting model terms and `hf auth login`. A 401/403 at first generation is the symptom of skipping this.
2. **Ungated optimized repo** (reportedly `stable-audio-3-optimized`), shipping f16 NumPy archives. Roughly half the size, no HF account needed.

**Prefer route 2** for anything a user might install. Falling back to route 1 requires a token entry UI; build that only if route 2 proves insufficient. Note that the T5Gemma component carries Gemma Terms of Use, and model weights and outputs remain under their own licenses — surface an acknowledgement screen on first run.

Storage: let `huggingface_hub` populate its normal cache and symlink into `Application Support/SA3Local/models/`, matching upstream's convention. Do not invent a second cache. Expose the on-disk size and a "reveal in Finder" affordance; multi-GB downloads that the user can't find or delete are hostile.

Download runs in the sidecar with a progress callback streamed as events. Must be resumable and cancellable.

**Precision constraint — do not "optimize" this away:** the SAME codec must stay FP32. Its differential attention catastrophically cancels in FP16. T5Gemma at FP16 is fine, and quantizing it buys nothing on a short prompt encode. If you later evaluate INT4 DiT bundles, quantize the DiT only.

---

## 6. Feature A — Latent Explorer

Be precise about *which* latent space. There are three, and they behave differently:

| Space | What it is | Interpolation | Character of the move |
|---|---|---|---|
| **Initial noise `x_T`** | `[B, 256, T_frames]` | **slerp** (not lerp — lerp shrinks the norm and desaturates the result) | Same prompt, different "take" |
| **Text conditioning** | T5Gemma embeddings of the prompt | lerp, plus CFG scale as a separate axis | Semantic morph between two descriptions |
| **Audio conditioning** | encode a real clip, re-noise to a partial `sigma_max` | scalar | "Mutation" — how far from the source you travel |

The reference repo uses that third one as its Mutation control: after bootstrap, the knob becomes audio-conditioning `sigma_max`. Steal that framing; it's the most musically legible of the three.

**UI:** a 2D pad with four corner anchors. Each anchor is a saved `(seed, prompt embedding)` pair. Position bilinearly blends — slerp on the noise, lerp on the conditioning. Regenerate on pointer-up, not continuously. A separate `sigma_max` slider and a CFG slider sit outside the pad.

**Latency behaviour:** with models resident and small-music, a warm generation should be around 1 s. That is too slow for continuous scrubbing and fine for release-to-generate. Do not fake continuous motion by crossfading between cached results; it will sound like exactly what it is.

**Cache:** LRU keyed on the full parameter tuple, holding decoded buffers. Revisiting a pad location must be instant. Determinism from Gate 1 is what makes the key valid.

---

## 7. Feature B — Loop Mutator

### 7.1 Clock and budget

`bars × beats_per_bar × 60 / bpm = loop_seconds`. Four bars at 128 BPM is 7.5 s. The mutation interval (N complete loops before requesting the next variation) sets the generation deadline.

The budget is: generation must finish comfortably before the boundary at which it's needed. At ~10× realtime, small-music generates a 7.5 s loop in well under a second — a wide margin. At ~2× realtime, medium needs ~3.75 s for the same loop, which fits a single-loop interval but leaves little headroom under thermal pressure or memory contention. **Start generation at the top of the loop preceding the swap, not partway through**, and expose the measured margin in the UI so a user who pushes bars down or picks medium can see they're near the edge.

If a generation misses its deadline: keep playing the current buffer, swap at the *next* boundary. Never stall the render thread, never glitch.

### 7.2 Audio engine

The render path must be allocation-free and lock-free. Use an `AUAudioUnit` render block (or `AVAudioSourceNode`) reading from an immutable buffer referenced by an atomic pointer. The generation thread prepares a fully-formed buffer off the render thread and publishes it with a release store; the render block swaps to it only at a loop boundary with an acquire load. No `os_unfair_lock` on the render thread, no ObjC/Swift allocation, no logging.

Since the buffer swaps only at boundaries, you get a natural epoch-published-snapshot structure — the old buffer is retired one epoch after the swap, when no render callback can still hold it.

### 7.3 Seamlessness

SA3 does not produce loop-seamless audio by default. The head and tail won't match. Three options, in increasing order of quality and effort:

1. **Equal-power crossfade** of a short tail region into the head. Cheap, works, costs you a few ms of the loop's transient attack.
2. **Zero-crossing trim** to the nearest sample-accurate loop length. Only helps for sustained material.
3. **Continuation / inpainting.** The MLX runtime supports mask-based inpainting and audio continuation. Generate the loop, then re-generate its final region conditioned on the head so the tail resolves into it. This is the right answer and the most work. **Verify the inpainting API surface exists in the runtime you installed before promising it.**

Ship (1), design for (3).

### 7.4 Lineage

Each mutation is a child of the previous. Keep a tree: prompt, seed, `sigma_max`, parent ID, audio path. The user must be able to walk back up and branch. This is the difference between a toy and something you'd actually record from.

---

## 8. Packaging, signing, sandbox

Defer this until phases 0–3 are green, but design for it:

- **Embedding Python:** `python-build-standalone` in `Contents/Resources/`, or a `uv`-created venv relocated at build time. Relocation is the hard part — audit for absolute paths baked into the venv (`pyvenv.cfg`, console scripts, `.dist-info`).
- **Hardened runtime:** the MLX and NumPy `.so`s aren't signed by you. Either sign every dylib in the bundle, or enable `com.apple.security.cs.disable-library-validation`. Prefer signing them; library validation is worth keeping.
- **Sandbox:** the app spawns a child process and makes network requests to HF. Dev builds run unsandboxed. If you ever target the Mac App Store, the sidecar model is a problem — flag it early rather than discovering it at submission.
- **Notarization** of a bundle containing an embedded interpreter and hundreds of native extensions is slow. Budget for it.

---

## 9. Repo layout

```
sa3-local/
├── DECISIONS.md
├── BASELINE.md
├── sidecar/
│   ├── sa3d.py              # socket server, dispatch, accelerator lock
│   ├── engine.py            # thin wrapper over optimized/mlx
│   ├── latents.py           # slerp, lerp, encode/decode helpers
│   ├── weights.py           # HF download, progress, integrity
│   └── protocol.py          # request/event schemas, shared with Swift via codegen or by hand
├── app/
│   ├── SA3Local/
│   │   ├── SidecarClient.swift
│   │   ├── AudioEngine.swift        # render block, atomic buffer swap
│   │   ├── Explorer/
│   │   ├── Looper/
│   │   └── Lineage/
├── tests/
│   ├── test_determinism.py
│   ├── test_protocol.py             # mocked, no MLX required
│   └── AudioEngineTests.swift
└── tools/
    └── bench.py
```

---

## 10. Phase plan

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Upstream MLX CLI runs | `.wav` on disk, `BASELINE.md` filled in |
| 1 | Determinism harness | Bit-identical repeats; slerp midpoint is plausible audio |
| 2 | `sa3d` sidecar + CLI client | Generate, cancel mid-sample, encode/decode round-trip, all over the socket |
| 3 | Swift shell + audio engine | Play a generated buffer; swap buffers at a boundary with no glitch under Instruments |
| 4 | Loop Mutator | 30 min continuous run, no dropout, lineage tree intact |
| 5 | Latent Explorer | 2D pad, cached revisits instant, no stale-result races when moved rapidly |
| 6 | First-run download UX | Clean install on a machine with no HF cache reaches audio without terminal use |
| 7 | Packaging | Signed, notarized, runs on a second Mac |

---

## 11. Known traps

- The reference repo's MLX backend is its least-exercised path. Expect it to be behind upstream. Don't debug *its* MLX code — read upstream directly and use the reference only for the event protocol and UI ideas.
- `--free-models` defaults on. It's correct for one-shot CLI use and wrong for both of our instruments.
- FP16 on the SAME codec is silently catastrophic, not loudly broken. If output goes to garbage after a "harmless" precision change, this is why.
- Medium requires `flash-attn` on the CUDA path; that's irrelevant here, but reference-repo troubleshooting notes will point you at it and waste your time.
- Model gating produces 401/403 at *first generation*, not at install. Fail fast: probe access during the download phase.
- Sidecar orphans. A crashed app leaving a resident 4 GB Python process is the failure mode users will report as "my Mac got slow."

---

## 12. Open questions to raise with Mark, not decide alone

1. Small-music vs. small-sfx vs. medium as the default. Depends on whether the target material is musical loops or texture.
2. Whether the Explorer's pad anchors should be seeds, prompts, or both on separate axes.
3. Whether output should land in a user-chosen folder with a filename convention, or in an app-managed library.
4. AUv3 / Audio Unit host integration as a later phase — the Loop Mutator is one wrapper away from being usable in a DAW, and that changes the audio engine design if it's on the roadmap.

# IDEAS.md — what else this architecture affords

Speculative. Nothing here is scheduled; the phase plan in `docs/HANDOFF.md`
stands. These are the things the Stable Audio 3 architecture makes cheap or
possible that a "text prompt in, WAV out" framing hides, cross-referenced to
the Research Vault (`~/Documents/Vaults/Research Vault`) and to Mark's other
instruments (FLATLINE drums, Vox pulsars, Vestige, Seance, Specter).

The affordances that matter, all verified in `external/stable-audio-3/optimized/mlx`:

- **SAME is an encoder as well as a decoder**, resident in MLX at FP32, 256-d
  latents at 44100/4096 ≈ 10.77 frames/s. Encoding is cheap. Any audio can be
  put *into* the space, edited there, and decoded back; the decoder smooths
  frame boundaries because it was trained to.
- **Inpainting takes an arbitrary per-frame mask** (`(1, 1, T_lat)`), not just
  a contiguous range, and `paste_back` makes unmasked frames bit-exact.
- **CFG is a slider between two prompts** (0 = negative prompt, 0.5 = halfway,
  >1 = extrapolate), with APG. Conditioning is a vector you can do arithmetic on.
- **`seconds_total` is a separate global conditioning** from the number of
  frames actually generated. The model can be told it is making the middle of a
  three-minute piece while you take 81 frames.
- **LoRA training runs natively in MLX** on pre-encoded latents, with stackable
  adapters, per-adapter strength, and per-sampler-step gating at inference.
- **Base (non-ARC) checkpoints exist** for all three DiTs: plain rectified flow,
  which supports many-step deterministic Euler sampling and therefore ODE
  inversion.
- **The latent grid is musical at the right tempo.** One frame is 92.88 ms: a
  16th note at 161.5 BPM, an 8th at 80.75, a quarter at 40.4. Generate at a
  frame-aligned tempo and latent frames become beat-grid cells.

Caveat that kills a whole class of ideas: 10.77 Hz is macrosound. Nothing at
grain or pulsar timescale lives in the latent; Vox's microsound stays in the
time domain. SA3 supplies *material* to it, not grains.

---

## 1. Latent concatenative resynthesis ("Vestige in SAME space")

Encode a corpus of Mark's recordings into 256-d latent frames. Encode a target
(a loop, or live input) and replace each frame with its nearest neighbour (or a
k-NN blend) from the corpus, then decode. Temporal structure of the target,
timbre of the corpus, no windowing, no phase problems. An hour of corpus is
~39k frames; brute-force k-NN is trivial.

- Vault: *Latent Granular Resynthesis using Neural Audio Codecs* (2025, the
  paper is exactly this with EnCodec), *CataRT* (2006), *Morphing of Granular
  Sounds* (2015).
- Instrument: a third Mutation mode next to σmax. "How much of the corpus"
  is a mix ratio in latent space. The Explorer's pad becomes a CataRT
  descriptor browser over the corpus.
- Cost: low. Sidecar `encode` + numpy. No DiT involved unless you want to
  re-noise the mosaic at small σmax to heal seams.

## 2. Latent feedback: "I Am Sitting in a Room", but the room is SAME

Decode → optionally process in the time domain (a FLATLINE or Vox effect, a
real room, a tape machine) → re-encode → mix with the previous latents →
decode again. Every pass through the codec is a *semantic* generation loss,
not a spectral one; it converges toward whatever SAME thinks the sound "is".
Insert a partial DiT pass (σmax 0.1–0.3) per iteration and each repeat is
re-dreamed rather than degraded.

- Vault: *Feature-Based Delay Line Using Real-Time Concatenative Synthesis*
  (2023) for the delay-as-corpus framing; Lucier for the rest.
- Instrument: a delay whose buffer is latent frames; repeats are re-selected by
  latent similarity, or re-dreamed. Ships as an effect on the looper's output.
- Cost: low to medium. Latency is one decode chunk.

## 3. Rhythmic inpainting masks

The mask is per frame, so mask a Euclidean pattern, every other bar, only the
off-beats, only the fills. Down-beats stay bit-exact through `paste_back`; the
model regenerates what happens between them. This is a variation engine that
respects the groove by construction, and it is the seam fix for the looper
(mask the last N frames and the first M, conditioned on the rest).

- Vault: *Break-the-Beat!* (2026): separate rhythmic content from timbral
  identity. Here the content is the unmasked frames.
- Instrument: a step-sequencer row over the loop where each cell is "keep" or
  "regenerate". Mutation interval + mask pattern = a whole performance
  vocabulary.
- Cost: low. It is the upstream inpaint path with a different mask.

## 4. Per-step LoRA gating as a coarse-to-fine style mixer

Upstream already supports `--lora a.safetensors steps=-3 --lora b.safetensors
steps=4-`. Early steps decide structure, late steps decide texture. A
"rhythm" adapter gated to steps 1–3 and a "timbre" adapter gated to 4–8 gives
two independent style axes from one sampler. Train the adapters on-device
from the user's own favourited lineage nodes: pre-encode, `lora_train_mlx.py`,
overnight on the M4 Pro. The instrument learns the player's taste.

- Vault: *Sounderfeit* (2018), *Learning Control of Neural SFX Synthesis from
  Physically Inspired Models* (2025): keep the control vocabulary, learn the
  leftovers. Here the vocabulary is the two gates.
- Cost: medium. Training is a background job; inference cost is a re-merge at
  the step boundary (~80 ms on medium per upstream).

## 5. SAME as the loss function for FLATLINE sound matching

SAME latents are trained to be semantic-acoustic. Use the encoder as the
perceptual front end for "match this drum": target sample → latent trajectory;
FLATLINE(params) → audio → latent trajectory; minimise the distance. MLX gives
gradients through the encoder, so if the FLATLINE voice is expressed as a
differentiable graph the loop is end to end; otherwise it is a gradient-free
search with a good objective.

- Vault: *Perceptual-Neural-Physical Sound Matching* (2023, "the bottleneck is
  the loss"), *wav2shape* (2020), *Real-time Timbre Remapping with DDSP*
  (2024, feature-difference loss on latent deltas between hits), *Compiling
  Differentiable Audio Graphs to Real-Time DSP* (2026) for getting the fitted
  voice back out as FAUST.
- Cost: medium. The encoder is the only SA3 piece involved.

## 6. Prompt-to-patch: SA3 as the teacher, FLATLINE as the student

Chain 5 with generation: text prompt → `sm-sfx` renders a thousand "808 kick
with long pitch glide" variants → each is fitted to FLATLINE parameters via
the SAME-space loss → a distribution over *interpretable* patches. The user
types a description and gets knobs, not a WAV. Language becomes a way into a
physical model's parameter space.

- Vault: *Learning Control of Neural SFX Synthesis* (teacher/student framing),
  *DDSP-SFX* (2024), *Towards Orchestrating Physically Modelled 2D Percussion
  Instruments* (2023, "design spaces become searchable").
- Cost: medium-high. Batch job, not an instrument.

## 7. Hybrid drum voice: physical attack, neural tail ("Latent Seance")

FLATLINE renders the first 100 ms transient. It is pasted back bit-exact; SA3
continues the tail via inpainting conditioned on that head, with σmax or the
mask length as the Real → Unreal slider Seance is built around. The transient
stays a physically legible object; the body is hallucinated.

- Vault: *Differentiable Modelling of Percussive Audio with Transient and
  Spectral Synthesis* (2023: transients need their own path), *Real Time Drum
  Augmentation with Physical Modeling* (2013: real exciter, modelled body,
  inverted here), the 2026-04-21 synthesis note's Real → Unreal continuum.
- Cost: low once the looper's continuation seam exists. Same code path.

## 8. Loop crossover and interactive evolution

Gate 1 determinism means a loop is a pure function of `(x_T, cond, σ, seeds)`.
`x_T` is per frame, so splice noise in time: bars 1–2 from parent A's `x_T`,
bars 3–4 from parent B's. The child inherits "takes" per bar. Add mutation
(slerp toward a fresh seed by a small angle) and the lineage tree becomes a
genetic algorithm with the player as fitness function.

- Vault: nothing direct; this is Dawkins' biomorphs applied to the lineage
  tree the looper already keeps.
- Cost: low. It is latent arithmetic plus UI on the tree.

## 9. Structural position as a knob

Generate 81 frames while conditioning `seconds_total` on 180 s. The model
produces the *middle* of a long piece, which has no intro or outro
artefacts, which is what a loop wants. Conditioning on 7.5 s gives a
self-contained miniature with a beginning and an end. Expose it as
"complete ↔ excerpt" and it is a legibly musical control that costs nothing.

- Verified: `sa3_mlx.py:611-613` feeds `args.seconds` to
  `SecondsTotalEmbedder` (`sa3_pipeline.py:37-56`, clamped to 0–384 s) and
  uses the result as both a cross-attention token and the global conditioning,
  independent of `T_lat`. Whether the model honours the mismatch is a Phase 1
  listening test.
- Cost: trivial. One float.

## 10. ODE inversion with the base checkpoints: morph between two real recordings

The shipped ARC models use a stochastic pingpong sampler, which cannot be
inverted. The `-base` rectified-flow checkpoints can be sampled with
deterministic Euler over many steps, and that ODE runs backwards: real audio
→ SAME latent → integrate to `x_T`. Invert two real loops, slerp their `x_T`,
integrate forward. The morph passes through the model's prior instead of
through a crossfade. This is Vestige's "excavate from recordings" done in
noise space.

- Vault: *Neural Granular Sound Synthesis* (2020: invertible latent spaces are
  what make navigation a control surface), *Augmenting Parametric Synthesis
  with Learned Timbral Controllers* (2019: invertibility keeps a learned
  control trustworthy).
- Cost: high, and quality is unknown: base models are un-post-trained and
  slower (many steps). Research, not roadmap.

## 11. Latent beat-slicing with a tempo that fits the grid

At 161.5 BPM a latent frame is a 16th note. Generate there, and re-ordering
frames is beat-shuffling; repeating a frame is a stutter; reversing a bar is
reversing four frames. The decoder heals the boundaries. Rate-convert or
re-generate at the target tempo afterwards, or just play at 161.5 and call it
footwork.

- Vault: *Combining Zeroth and First-Order Analysis ... Live Concatenative
  Granulation* (2021) is the time-domain version of the boundary problem the
  decoder sidesteps.
- Cost: low.

## 12. Accompanist mode (Phase 8, AUv3)

In a host, encode the bus the plugin sits on, use it as `init_audio` at low
σmax with the looper's prompt, and regenerate each loop one loop ahead of the
swap. The plugin plays along with the session. The same encoder tap can drive
idea 1 live: a latent vocoder that resynthesises the input from the corpus.

- Vault: *GrainProc* (2013) and *Feature-Based Delay Line* for live-input
  framing.
- Cost: medium, and it depends on the Phase 8 sidecar-reach decision (D-013).

## 13. Semantic tilt: PCA over the latent space as EQ

Encode a large varied corpus, PCA the 256 dims, correlate the top components
with spectral centroid and attack time. The vault's timbre review says those
two explain ~70% of similarity judgements. Expose the two best-correlated
directions as knobs applied additively to latents before decode. It is an
equaliser whose bands are whatever SAME thinks brightness and punch are.

- Vault: *Timbre Perception, Representation, and its Neuroscientific
  Exploration* (2024).
- Cost: low to try, may not work: the space might not be linear in those
  directions. One afternoon with the encoder decides it.

---

**If I had to pick three for after Phase 5:** 3 (rhythmic masks, because it is
also the seam fix), 1 (latent concatenative, because it plugs Vestige and
FLATLINE's corpus into the same space with no training), and 4 (gated LoRA
mixer, because upstream already built the hard part).

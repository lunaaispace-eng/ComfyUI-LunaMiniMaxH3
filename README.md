# ComfyUI-LunaMiniMaxH3

Two nodes for **MiniMax H3**: one that works out the numbers a generation needs, and
one that builds the joint audio+video latent the sampler runs on. Both exist because
H3 fixes things the generic nodes leave to you — its canvas follows from the aspect
ratio, its frame count sits on a grid, and its latents come in pairs.

| Node | Category | Class |
| --- | --- | --- |
| `Luna MiniMax H3 Canvas` | `Luna/MiniMax` | `LunaMiniMaxH3Canvas` |
| `Luna H3ConcatAVLatent` | `Luna/MiniMax` | `LunaH3ConcatAVLatent` |

Both carry an **ⓘ** on the title bar with their inputs and outputs documented, built
from the node's own description and tooltips.

## Installation

Copy this folder into your ComfyUI custom nodes directory:

```text
ComfyUI/custom_nodes/ComfyUI-LunaMiniMaxH3
```

No dependencies beyond ComfyUI itself. Restart ComfyUI afterwards.

> `Luna MiniMax H3 Canvas` used to ship in **ComfyUI-SaveSimple**. It moved here so the
> H3 work lives in one pack; the class name is unchanged, so saved workflows keep
> working — but update SaveSimple at the same time, or two installed copies of the same
> class will fight over the registration.

---

# Luna H3ConcatAVLatent

Merges a video latent and an audio latent into the single AV latent H3 samples from.

| Input | |
| --- | --- |
| `video_latent` | `[B,24,T,H/16,W/16]` from VAE Encode with H3's video VAE |
| `audio_latent` | `[B,32,2,T]` from VAE Encode Audio with H3's audio VAE |
| `fit_audio` | cut or extend the audio to the length this video length implies |
| `extend_mode` | `invent` (default) pads a short clip with zeros; `fail` raises instead |
| `force` | OFF raises on channel / stereo / batch / device / dtype mismatch |

Output is a `LATENT` whose `samples` is the video+audio pair.

## What it is for

H3 does not sample a video latent, it samples a **pair**: video and audio together in
one nested latent, denoised side by side. Every H3 node that hands you a latent builds
that pair internally and only ever gives you a fresh, empty one — which is fine for
generating, and a dead end for **redrawing footage you already have**. Encode your
frames, encode your soundtrack, concat them here, and sample at `denoise < 1`: the
model rewrites what is there instead of inventing from noise.

Wire an AV latent into `video_latent` and it keeps that latent's video and swaps in
the new audio — the way to replace a soundtrack without touching the picture.

## Why `fit_audio` is on by default

The two streams are sampled together, so their lengths have to agree: H3 pairs
`n` frames with `round(n / 24 × 40)` audio steps. An encoded soundtrack is almost
never exactly that — a clip is 4.98 s, the video is 124 frames — and the mismatch is
not caught when you wire it up, it surfaces inside the sampler with a shape error.

With `fit_audio` on, the node reads the frame count back off the video latent (H3's
`17k+5` frames encode to `5k+2` latent steps, so the count is recoverable) and cuts or
zero-extends the audio to match. The padded tail is the official **empty-latent prior**,
not encode(silence): at `denoise = 1` the model fully rewrites it; at `denoise < 1` it
smears. `extend_mode=fail` refuses to invent that tail. Fit the soundtrack upstream if
a smear at partial denoise matters.

Turn `fit_audio` off to pass both streams through untouched.

## When it raises

- **Wrong channel count** (24 video / 32 audio), stereo axis, batch, device, or dtype —
  the usual sign of the wrong VAE or a mixed pair. Set `force` to pair anyway.
- **Off-grid video latent** with `fit_audio` on — a `T` that is not `5k+2` has no
  recoverable frame count, so the audio length cannot be derived.
- **Nested AV samples** with a flat `noise_mask`, a nested tensor with fewer than
  two streams, or a nested mask with *more* streams than the samples — those used
  to become a silent reshape or an opaque `IndexError`. A nested mask *shorter*
  than the sample pair is accepted when the missing halves are trailing (video
  masked, audio omitted): that is this node's own emission, and the same rule
  the sampler uses.

Shape errors that would otherwise fail deep in the model (a still image where video
frames belong, audio encoded with the wrong VAE) are raised here with the expected
shape spelled out.

---

# Luna MiniMax H3 Canvas

Aspect ratio and a duration in seconds go in; canvas, frame count and both frame rates
come out. It replaces the usual arrangement of a generic resolution node plus two
hand-written maths expressions.

| Output | |
| --- | --- |
| `width` / `height` | the canvas, always a multiple of 32 |
| `length` | frame count, snapped to H3's sampling grid |
| `fps` | 24 — H3's native rate |
| `output_fps` | `24 × interpolation_factor`, for the save node |
| `interpolation_factor` | the same figure `output_fps` was built from, for the interpolation node |
| `info` | what you actually got, including any snapping |

`interpolation_factor` comes back out on purpose. Drive the frame-interpolation node's
own factor from it and the frame count, the interpolation and the playback rate all
descend from one widget — there is no second number left to disagree.

## Why H3 needs its own node

**H3 does not want a megapixel target.** Its canvas follows from the aspect ratio: a
fixed 768 short edge under a 768×1344 area cap, each axis rounded to 32. A generic
resolution node makes you hand-tune a megapixel figure to arrive back at the number
the model already defines. `H3 canvas` mode skips the guessing.

Ratios are listed in landscape form with a **`portrait`** toggle that turns them on
their side — one name per canvas, rather than two entries and the trap of picking
`9:16` *and* ticking portrait.

| Ratio | Landscape | Flipped | MP | Flipped is |
| --- | --- | --- | --- | --- |
| 2.39:1 | 1568×672 | 672×1568 | 1.05 | |
| 21:9 | 1536×672 | 672×1536 | 1.03 | |
| 2:1 | 1440×704 | 704×1440 | 1.01 | |
| 1.91:1 | 1408×736 | 736×1408 | 1.04 | link preview |
| 16:9 | 1344×768 | 768×1344 | 1.03 | Reels, Shorts, TikTok |
| 16:10 | 1216×768 | 768×1216 | 0.93 | |
| 3:2 | 1152×768 | 768×1152 | 0.89 | 2:3 photo |
| 4:3 | 1024×768 | 768×1024 | 0.79 | 3:4 |
| 5:4 | 960×768 | 768×960 | 0.74 | 4:5 Instagram portrait |
| 1:1 | 768×768 | — | 0.59 | |

Note the area falls away as the ratio squares up: the 768 short edge is fixed, so only
wide ratios reach the 1.03 MP cap. 1:1 is 0.59 MP and there is nothing to be done about
it in `H3 canvas` mode — switch to `megapixels` if you want a bigger square.

**Frame count is not free either.** H3 samples on a grid where the count satisfies
`n % 17 == 5` — 5, 22, 39 … 243, 362. Ask for 7.5 s and you get 192 frames, which is
8.0 s. Generic nodes snap you silently; this one says so.

**`fps` and `output_fps` leave from the same node**, so a frame count and its playback
rate cannot drift apart. Wiring `output_fps` to the save node is what prevents the
classic "interpolation switched off but the fps is still doubled" desync, where half
the frames play at twice the rate.

## Going past the default canvas

Switch `size_mode` to `megapixels` for a chosen area at the same aspect — H3 runs to
2K. Cost scales with tokens: the latent is `(width/16) × (height/16)` per latent frame,
so doubling the area doubles the tokens and roughly quadruples the attention cost. The
readout flags anything above the model's default canvas.

## The readout

The node draws its resolved numbers live as you turn the dials, rather than after a run
— the whole point is that they are derived. `megapixels` hides itself in `H3 canvas`
mode, where it does nothing. Warnings (duration snapped, frame count outside H3's
trained 124–362 range, canvas above default) appear underneath.

The frontend mirrors the Python in `h3_canvas.py`, which is the source of truth.

---

## Licence

Apache-2.0. See `LICENSE`.

The model facts these nodes are built on — the 768 short edge, the 768×1344 cap, the
multiple of 32, 24 fps, the `17k+5` frame grid, the `5k+2` latent grid, the 40-per-second
audio latent, the nested video+audio pair — are requirements of MiniMax H3, observable
from its node signatures. The implementations here are this pack's own. **No code is
taken from ComfyUI core**, which is GPL-3.0 and incompatible with this licence.

The `js/luna_*.mjs` helper modules (theme, help, collapse) are copied from
**ComfyUI-SaveSimple**, same author and same licence.

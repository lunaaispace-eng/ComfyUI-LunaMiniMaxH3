"""Luna H3ConcatAVLatent — pair a video latent with an audio latent for MiniMax H3.

H3 samples one latent carrying both streams: a NestedTensor holding video
[B,24,T,H/16,W/16] and audio [B,32,2,T40]. Every H3 node that exists hands you
that pair already built and only ever a fresh one, so the way to redraw
*existing* footage — encode the frames yourself, sample at denoise < 1 — is to
assemble the pair by hand. That is what this node is for.

It also fits the audio stream to the video's length. H3's frame count sits on a
17k+5 grid, the video latent on 5k+2, and the audio latent runs at 40 steps a
second against video's 24 fps, so the length the sampler expects follows from the
video latent alone. An encoded soundtrack is almost never exactly that long, and
the mismatch surfaces inside the sampler rather than here.

Licence note, same rule as `h3_canvas.py`: the shapes, the pairing and the grid
are requirements of the MiniMax H3 model, observable from its node signatures —
interface facts, not authorship. The implementation is this pack's own. **No code
is taken from ComfyUI core**, which is GPL-3.0 and incompatible with this pack's
Apache-2.0 licence, nor from ptmaster's `ComfyUI-PT_H3ConcatAVLatent`, which
prompted the node and ships no licence at all.
"""

from __future__ import annotations

import logging

import comfy.nested_tensor
import comfy.utils
import torch

# MiniMax H3 interface constants, the same set h3_canvas.py works from.
FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_GRID = 17          # frame count satisfies n % FRAME_GRID == FRAME_PHASE
FRAME_PHASE = 5
LATENT_GRID = 5          # ... and the video latent's T is LATENT_GRID*k + LATENT_PHASE
LATENT_PHASE = 2
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_STEREO_AXIS = 2

LOG = "[Luna H3 Concat AV Latent]"
logger = logging.getLogger("Luna.H3ConcatAVLatent")

# Keys the audio latent may contribute. Everything else stays with the video
# latent so batch_index / type / control from the soundtrack cannot clobber it.
_AUDIO_MERGE_KEYS = frozenset({"samples", "noise_mask"})


def frame_count_from_latent_t(video_t: int):
    """Video latent T -> the frame count it holds, or None if it is off-grid.

    H3's two grids run in step: 17k+5 frames encode to 5k+2 latent steps. Only
    the latent is on the wire here, so read k off it and go back the other way.
    """
    k, remainder = divmod(video_t - LATENT_PHASE, LATENT_GRID)
    if k < 0 or remainder:
        return None
    return FRAME_GRID * k + FRAME_PHASE


def audio_latent_length(frame_count: int) -> int:
    """Audio latent steps H3 pairs with that many frames."""
    return round(frame_count / FPS * AUDIO_LATENT_FPS)


def _fit_audio(audio, mask, length: int, extend_mode: str = "invent"):
    """Cut or extend the audio stream along its time axis to `length` steps.

    Short clips under extend_mode=invent are padded with zeros and the added
    steps are left *unmasked*. That tail is the official empty-latent prior
    (EmptyMiniMaxH3LatentAV), not encode(silence): fully rewritten at denoise=1,
    smeared at denoise < 1. extend_mode=fail refuses to invent a tail.
    """
    have = audio.shape[-1]
    if have == length:
        return audio, mask

    # A latent's mask keeps whatever shape it arrived with until sampling resizes
    # it; conform it here so the same cut applies to both.
    if mask is not None:
        mask = comfy.utils.reshape_mask(mask, audio.shape)

    if have > length:
        return audio[..., :length], (None if mask is None else mask[..., :length])

    if extend_mode != "invent":
        raise RuntimeError(
            f"{LOG} audio is {have} latent steps, video wants {length}; "
            f"extend_mode={extend_mode} refuses to invent a tail. Encode a longer "
            "soundtrack or set extend_mode=invent."
        )

    tail = audio.shape[:-1] + (length - have,)
    audio = torch.cat((audio, audio.new_zeros(tail)), dim=-1)
    if mask is not None:
        mask = torch.cat((mask, mask.new_ones(tail)), dim=-1)
    return audio, mask


def _split(latent, stream: int):
    """(samples, mask) for one stream, unwrapping an AV latent if that is what it is.

    Nested samples need at least two streams. Nested samples with a flat mask
    are refused — reshape_mask would treat a leftover video-shaped mask as audio.
    A nested mask may be shorter than the sample pair when the missing halves
    are trailing (CFGGuider.sample appends ones for those). Extra mask streams
    still raise. Nested mask on a flat stream is refused.
    """
    samples = latent["samples"]
    mask = latent.get("noise_mask", None)
    samples_nested = getattr(samples, "is_nested", False)
    mask_nested = mask is not None and getattr(mask, "is_nested", False)

    if samples_nested:
        streams = samples.unbind()
        if len(streams) < 2:
            raise RuntimeError(
                f"{LOG} nested latent has {len(streams)} stream(s); "
                "H3 AV needs 2 (video, audio)."
            )
        samples = streams[stream]
        if mask is None:
            return samples, mask
        if not mask_nested:
            raise RuntimeError(
                f"{LOG} samples are a nested AV pair but noise_mask is a single "
                "tensor; split the mask or drop it before concat."
            )
        mask_streams = mask.unbind()
        if len(mask_streams) > len(streams):
            raise RuntimeError(
                f"{LOG} nested mask has {len(mask_streams)} stream(s), "
                f"samples have {len(streams)}."
            )
        # Omitted trailing halves: this node emits NestedTensor((video_mask,))
        # when only the video side is masked, and the sampler pads those with
        # ones. A missing leading half is not representable and is never emitted.
        if stream >= len(mask_streams):
            mask = None
        else:
            mask = mask_streams[stream]
    elif mask_nested:
        raise RuntimeError(
            f"{LOG} noise_mask is a nested AV pair but samples are a single stream."
        )
    return samples, mask


def _pair_mismatches(video, audio):
    """Human-readable contract failures. Empty means the pair is internally consistent."""
    problems = []
    if video.shape[1] != VIDEO_LATENT_CHANNELS:
        problems.append(
            f"video has {video.shape[1]} channels, H3 expects {VIDEO_LATENT_CHANNELS} (wrong VAE?)"
        )
    if audio.shape[1] != AUDIO_LATENT_CHANNELS:
        problems.append(
            f"audio has {audio.shape[1]} channels, H3 expects {AUDIO_LATENT_CHANNELS} (wrong VAE?)"
        )
    if audio.shape[2] != AUDIO_STEREO_AXIS:
        problems.append(
            f"audio stereo axis is {audio.shape[2]}, H3 expects {AUDIO_STEREO_AXIS}"
        )
    if video.shape[0] != audio.shape[0]:
        problems.append(
            f"batch size {video.shape[0]} (video) vs {audio.shape[0]} (audio)"
        )
    if video.device != audio.device:
        problems.append(f"device {video.device} (video) vs {audio.device} (audio)")
    if video.dtype != audio.dtype:
        problems.append(f"dtype {video.dtype} (video) vs {audio.dtype} (audio)")
    return problems


class LunaH3ConcatAVLatent:
    DESCRIPTION = (
        "Merge a video latent and an audio latent into the joint AV latent MiniMax "
        "H3 samples from. Built for redrawing existing footage: VAE Encode the "
        "frames, encode the soundtrack with the audio VAE, wire both in here, then "
        "sample at denoise < 1. An AV latent in either socket contributes only the "
        "stream that socket asks for, so wiring one into video_latent keeps its "
        "video and swaps the audio."
    )
    CATEGORY = "Luna/MiniMax"
    FUNCTION = "concat"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT", {
                    "tooltip": "Video stream: [B,24,T,H/16,W/16] from VAE Encode with the H3 video VAE. An AV latent works too — its video stream is kept and its audio replaced.",
                }),
                "audio_latent": ("LATENT", {
                    "tooltip": "Audio stream: [B,32,2,T] from VAE Encode Audio with the H3 audio VAE. An AV latent contributes only its audio stream.",
                }),
                "fit_audio": ("BOOLEAN", {
                    "default": True, "label_on": "fit to video", "label_off": "as-is",
                    "tooltip": "ON cuts or extends the audio to the length H3 expects for this video length (frames/24 x 40) — an encoded soundtrack is rarely exactly that, and the mismatch surfaces during sampling rather than here. A short clip is padded with zeros: that tail is the empty-latent prior, not silence. At denoise=1 the model fully rewrites it; at denoise < 1 it smears. OFF passes both streams through untouched.",
                }),
                "extend_mode": (["invent", "fail"], {
                    "default": "invent",
                    "tooltip": "When fit_audio is on and the soundtrack is shorter than the video-implied length: invent (default) appends zeros — empty-latent prior, not encode(silence); fully rewritten at denoise=1, smeared at denoise < 1. fail raises instead of inventing a tail. Long audio is cut either way.",
                }),
                "force": ("BOOLEAN", {
                    "default": False, "label_on": "force pair", "label_off": "validate",
                    "tooltip": "OFF raises on channel, stereo-axis, batch, device, or dtype mismatch (wrong VAE or mixed batches). ON logs a warning and pairs anyway.",
                }),
            },
        }

    def concat(self, video_latent, audio_latent, fit_audio=True, extend_mode="invent", force=False):
        video, video_mask = _split(video_latent, 0)
        audio, audio_mask = _split(audio_latent, 1)

        if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
            raise RuntimeError("Both sockets need a latent with real samples in it.")
        if video.ndim != 5:
            raise RuntimeError(
                f"video_latent should be 5D [B,{VIDEO_LATENT_CHANNELS},T,H/16,W/16], got {tuple(video.shape)}. "
                "Encode video frames with the H3 video VAE, not a still image.")
        if audio.ndim != 4:
            raise RuntimeError(
                f"audio_latent should be 4D [B,{AUDIO_LATENT_CHANNELS},2,T], got {tuple(audio.shape)}. "
                "Use VAE Encode Audio with H3's audio VAE.")

        problems = _pair_mismatches(video, audio)
        if problems:
            msg = f"{LOG} latent pair mismatch: " + "; ".join(problems)
            if force:
                logger.warning("%s (force=True, pairing anyway)", msg)
            else:
                raise RuntimeError(msg + " — set force to pair anyway.")

        if fit_audio:
            frame_count = frame_count_from_latent_t(video.shape[2])
            if frame_count is None:
                raise RuntimeError(
                    f"{LOG} video latent T={video.shape[2]} is not on H3's 5k+2 grid; "
                    "cannot derive the audio length. Use an H3 video latent or turn fit_audio off."
                )
            wanted = audio_latent_length(frame_count)
            if audio.shape[-1] != wanted:
                logger.info(
                    "%s audio %s -> %s latent steps (%s for %s frames)",
                    LOG, audio.shape[-1], wanted,
                    "cut" if audio.shape[-1] > wanted else "extended",
                    frame_count,
                )
            audio, audio_mask = _fit_audio(audio, audio_mask, wanted, extend_mode)

        out = {k: v for k, v in video_latent.items() if k not in _AUDIO_MERGE_KEYS}
        for key in _AUDIO_MERGE_KEYS:
            if key in audio_latent:
                out[key] = audio_latent[key]
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))

        # A mask on either side has to become a pair as well, or the sampler gets
        # one stream masked and one not. CFGGuider.sample pads only *trailing*
        # nested halves, and stream order is (video, audio), so a missing video
        # mask cannot be omitted — it would be read as stream 0. A missing audio
        # mask can: the sampler appends ones for the trailing half.
        if video_mask is not None or audio_mask is not None:
            video_mask = (torch.ones_like(video) if video_mask is None
                          else comfy.utils.reshape_mask(video_mask, video.shape))
            if audio_mask is None:
                out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask,))
            else:
                audio_mask = comfy.utils.reshape_mask(audio_mask, audio.shape)
                out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        else:
            out.pop("noise_mask", None)

        return (out,)


NODE_CLASS_MAPPINGS = {"LunaH3ConcatAVLatent": LunaH3ConcatAVLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"LunaH3ConcatAVLatent": "Luna H3ConcatAVLatent"}

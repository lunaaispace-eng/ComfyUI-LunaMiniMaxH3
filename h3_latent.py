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

LOG = "[Luna H3 Concat AV Latent]"


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


def _fit_audio(audio, mask, length: int):
    """Cut or extend the audio stream along its time axis to `length` steps.

    Short clips are extended with zeros and the added steps are left *unmasked*,
    so the model writes something there rather than being told to preserve
    silence. At denoise < 1 it only partly rewrites them, so a soundtrack much
    shorter than the footage still trails off — fit it upstream if that matters.
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

    tail = audio.shape[:-1] + (length - have,)
    audio = torch.cat((audio, audio.new_zeros(tail)), dim=-1)
    if mask is not None:
        mask = torch.cat((mask, mask.new_ones(tail)), dim=-1)
    return audio, mask


def _split(latent, stream: int):
    """(samples, mask) for one stream, unwrapping an AV latent if that is what it is."""
    samples = latent["samples"]
    mask = latent.get("noise_mask", None)
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[stream]
        if mask is not None and getattr(mask, "is_nested", False):
            mask = mask.unbind()[stream]
    return samples, mask


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
                    "tooltip": "Audio stream: [B,32,2,T] from VAE Encode Audio with the H3 audio VAE.",
                }),
                "fit_audio": ("BOOLEAN", {
                    "default": True, "label_on": "fit to video", "label_off": "as-is",
                    "tooltip": "ON cuts or extends the audio to the length H3 expects for this video length (frames/24 x 40) — an encoded soundtrack is rarely exactly that, and the mismatch surfaces during sampling rather than here. OFF passes both streams through untouched.",
                }),
            },
        }

    def concat(self, video_latent, audio_latent, fit_audio=True):
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

        # Channel counts are the tell for the wrong VAE: the shapes still look
        # plausible and the failure lands deep inside the model. Warn rather than
        # refuse — the pairing itself is not H3-specific.
        for name, tensor, expected in (("video", video, VIDEO_LATENT_CHANNELS),
                                       ("audio", audio, AUDIO_LATENT_CHANNELS)):
            if tensor.shape[1] != expected:
                print(f"{LOG} {name} latent has {tensor.shape[1]} channels, "
                      f"H3 expects {expected} - wrong VAE?")

        if fit_audio:
            frame_count = frame_count_from_latent_t(video.shape[2])
            if frame_count is None:
                print(f"{LOG} video latent T={video.shape[2]} is not on H3's frame grid; "
                      "leaving the audio length alone.")
            else:
                wanted = audio_latent_length(frame_count)
                if audio.shape[-1] != wanted:
                    print(f"{LOG} audio {audio.shape[-1]} -> {wanted} latent steps "
                          f"({'cut' if audio.shape[-1] > wanted else 'extended'} for {frame_count} frames)")
                audio, audio_mask = _fit_audio(audio, audio_mask, wanted)

        out = {}
        out.update(video_latent)
        out.update(audio_latent)  # keys other than samples: the audio latent's win
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))

        # A mask on either side has to become a pair as well, or the sampler gets
        # one stream masked and one not. Each half is conformed to its own stream:
        # a mask can arrive in any shape (Set Latent Noise Mask hands over image
        # dimensions), and once the two are wrapped together nothing downstream
        # resizes them per stream.
        if video_mask is not None or audio_mask is not None:
            video_mask = (torch.ones_like(video) if video_mask is None
                          else comfy.utils.reshape_mask(video_mask, video.shape))
            audio_mask = (torch.ones_like(audio) if audio_mask is None
                          else comfy.utils.reshape_mask(audio_mask, audio.shape))
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        else:
            out.pop("noise_mask", None)

        return (out,)


NODE_CLASS_MAPPINGS = {"LunaH3ConcatAVLatent": LunaH3ConcatAVLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"LunaH3ConcatAVLatent": "Luna H3ConcatAVLatent"}

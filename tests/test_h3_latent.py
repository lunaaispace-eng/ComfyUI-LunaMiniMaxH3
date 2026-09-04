"""Unit tests for LunaH3ConcatAVLatent against the accepted 1-7 audit list.

Stubs `comfy.nested_tensor` and `comfy.utils` so this pack clone can run
without ComfyUI on the path. A live NestedTensor / reshape_mask is used
when importable.
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType


class _StubNestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def _stub_reshape_mask(mask, output_shape):
    import torch
    if tuple(mask.shape) == tuple(output_shape):
        return mask
    return torch.ones(output_shape, dtype=mask.dtype, device=mask.device)


def _install_comfy_stubs():
    try:
        import comfy.nested_tensor  # noqa: F401
        import comfy.utils  # noqa: F401
        return False
    except ImportError:
        pass

    comfy = ModuleType("comfy")
    nested = ModuleType("comfy.nested_tensor")
    utils = ModuleType("comfy.utils")
    nested.NestedTensor = _StubNestedTensor
    utils.reshape_mask = _stub_reshape_mask
    comfy.nested_tensor = nested
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.nested_tensor"] = nested
    sys.modules["comfy.utils"] = utils
    return True


_STUBBED = _install_comfy_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from h3_latent import (  # noqa: E402
    AUDIO_LATENT_CHANNELS,
    VIDEO_LATENT_CHANNELS,
    LunaH3ConcatAVLatent,
    audio_latent_length,
    frame_count_from_latent_t,
)
import comfy.nested_tensor as nested_tensor  # noqa: E402


def _video(t=7, b=1, h=4, w=4, channels=VIDEO_LATENT_CHANNELS, dtype=torch.float32):
    return torch.randn(b, channels, t, h, w, dtype=dtype)


def _audio(steps=10, b=1, channels=AUDIO_LATENT_CHANNELS, stereo=2, dtype=torch.float32):
    return torch.randn(b, channels, stereo, steps, dtype=dtype)


def _latent(samples, mask=None, **extra):
    out = {"samples": samples, **extra}
    if mask is not None:
        out["noise_mask"] = mask
    return out


def _concat(video, audio, **kwargs):
    # V3: execute is a classmethod and returns io.NodeOutput, whose positional
    # results live on .result (the old V1 form was a plain tuple).
    return LunaH3ConcatAVLatent.execute(video, audio, **kwargs).result[0]


class GridTests(unittest.TestCase):
    def test_official_inverse(self):
        # T=2,7,12,37,72,107 → 5,22,39,124,243,362 frames
        cases = ((2, 5), (7, 22), (12, 39), (37, 124), (72, 243), (107, 362))
        for t, frames in cases:
            self.assertEqual(frame_count_from_latent_t(t), frames)
            self.assertEqual(audio_latent_length(frames), round(frames / 24 * 40))

    def test_off_grid_t(self):
        self.assertIsNone(frame_count_from_latent_t(6))
        self.assertIsNone(frame_count_from_latent_t(1))


class SplitTests(unittest.TestCase):
    def test_av_in_audio_keeps_stream_1(self):
        v0, a0 = _video(), _audio(8)
        v1, a1 = _video(), _audio(11)
        out = _concat(
            _latent(v0),
            _latent(nested_tensor.NestedTensor((v1, a1))),
            fit_audio=False,
        )
        video, audio = out["samples"].unbind()
        self.assertTrue(torch.equal(video, v0))
        self.assertTrue(torch.equal(audio, a1))

    def test_av_in_video_keeps_stream_0(self):
        v0, a0 = _video(), _audio(8)
        a1 = _audio(11)
        out = _concat(
            _latent(nested_tensor.NestedTensor((v0, a0))),
            _latent(a1),
            fit_audio=False,
        )
        video, audio = out["samples"].unbind()
        self.assertTrue(torch.equal(video, v0))
        self.assertTrue(torch.equal(audio, a1))

    def test_av_in_both_swaps_soundtrack(self):
        v0, a0 = _video(), _audio(8)
        v1, a1 = _video(), _audio(11)
        out = _concat(
            _latent(nested_tensor.NestedTensor((v0, a0))),
            _latent(nested_tensor.NestedTensor((v1, a1))),
            fit_audio=False,
        )
        video, audio = out["samples"].unbind()
        self.assertTrue(torch.equal(video, v0))
        self.assertTrue(torch.equal(audio, a1))

    def test_nested_samples_flat_mask_raises(self):
        samples = nested_tensor.NestedTensor((_video(), _audio()))
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(samples, mask=torch.ones(1, 24, 7, 4, 4)), _latent(_audio()), fit_audio=False)
        self.assertIn("single tensor", str(ctx.exception))

    def test_arity_1_nested_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(nested_tensor.NestedTensor((_video(),))), _latent(_audio()), fit_audio=False)
        self.assertIn("1 stream", str(ctx.exception))

    def test_omitted_trailing_mask_accepted(self):
        # 1-stream nested mask + 2-stream samples is this node's own
        # video-only emission; the missing half is trailing.
        video = _video()
        video_mask = torch.ones_like(video)
        samples = nested_tensor.NestedTensor((video, _audio(8)))
        mask = nested_tensor.NestedTensor((video_mask,))
        out = _concat(
            _latent(samples, mask=mask),
            _latent(_audio(11)),
            fit_audio=False,
        )
        v, _ = out["samples"].unbind()
        self.assertTrue(torch.equal(v, video))
        masks = out["noise_mask"].unbind()
        self.assertEqual(len(masks), 1)
        self.assertTrue(torch.equal(masks[0], video_mask))

    def test_mask_arity_too_many_raises(self):
        samples = nested_tensor.NestedTensor((_video(), _audio()))
        mask = nested_tensor.NestedTensor((
            torch.ones_like(_video()),
            torch.ones_like(_audio()),
            torch.ones_like(_audio()),
        ))
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(samples, mask=mask), _latent(_audio()), fit_audio=False)
        self.assertIn("nested mask has 3", str(ctx.exception))

    def test_nested_mask_flat_samples_raises(self):
        mask = nested_tensor.NestedTensor((torch.ones_like(_video()), torch.ones_like(_audio())))
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(_video(), mask=mask), _latent(_audio()), fit_audio=False)
        self.assertIn("samples are a single stream", str(ctx.exception))


class ValidationTests(unittest.TestCase):
    def test_channel_mismatch_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(_video(channels=16)), _latent(_audio()), fit_audio=False)
        self.assertIn("16 channels", str(ctx.exception))

    def test_stereo_axis_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(_video()), _latent(_audio(stereo=1)), fit_audio=False)
        self.assertIn("stereo axis", str(ctx.exception))

    def test_batch_mismatch_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(_video(b=2)), _latent(_audio(b=1)), fit_audio=False)
        self.assertIn("batch size", str(ctx.exception))

    def test_dtype_mismatch_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            _concat(
                _latent(_video(dtype=torch.float32)),
                _latent(_audio(dtype=torch.float16)),
                fit_audio=False,
            )
        self.assertIn("dtype", str(ctx.exception))

    def test_force_pairs_anyway(self):
        with self.assertLogs("Luna.H3ConcatAVLatent", level="WARNING") as logs:
            out = _concat(
                _latent(_video(channels=16)),
                _latent(_audio()),
                fit_audio=False,
                force=True,
            )
        self.assertTrue(any("16 channels" in rec.getMessage() for rec in logs.records))
        self.assertEqual(out["samples"].unbind()[0].shape[1], 16)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a second device")
    def test_device_mismatch_raises(self):
        video = _video().to("cpu")
        audio = _audio().to("cuda")
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(video), _latent(audio), fit_audio=False)
        self.assertIn("device", str(ctx.exception))


class FitAudioTests(unittest.TestCase):
    def test_invent_pads_zeros_unmasked(self):
        # T=7 → 22 frames → 37 audio steps
        audio = _audio(10)
        out = _concat(_latent(_video(t=7)), _latent(audio), fit_audio=True, extend_mode="invent")
        _, fitted = out["samples"].unbind()
        self.assertEqual(fitted.shape[-1], 37)
        self.assertTrue(torch.equal(fitted[..., :10], audio))
        self.assertTrue(torch.all(fitted[..., 10:] == 0))
        self.assertNotIn("noise_mask", out)

    def test_invent_with_mask_ones_on_tail(self):
        audio = _audio(10)
        mask = torch.ones_like(audio)
        out = _concat(
            _latent(_video(t=7)),
            _latent(audio, mask=mask),
            fit_audio=True,
            extend_mode="invent",
        )
        _, audio_mask = out["noise_mask"].unbind()
        self.assertEqual(audio_mask.shape[-1], 37)
        self.assertTrue(torch.all(audio_mask[..., 10:] == 1))

    def test_fail_refuses_short_tail(self):
        with self.assertRaises(RuntimeError) as ctx:
            _concat(
                _latent(_video(t=7)),
                _latent(_audio(10)),
                fit_audio=True,
                extend_mode="fail",
            )
        self.assertIn("refuses to invent", str(ctx.exception))

    def test_cut_long_audio(self):
        out = _concat(_latent(_video(t=7)), _latent(_audio(80)), fit_audio=True)
        self.assertEqual(out["samples"].unbind()[1].shape[-1], 37)

    def test_off_grid_fit_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            _concat(_latent(_video(t=6)), _latent(_audio()), fit_audio=True)
        self.assertIn("5k+2", str(ctx.exception))

    def test_off_grid_as_is_passes(self):
        audio = _audio(13)
        out = _concat(_latent(_video(t=6)), _latent(audio), fit_audio=False)
        self.assertEqual(out["samples"].unbind()[1].shape[-1], 13)


class MaskAndMergeTests(unittest.TestCase):
    def test_audio_only_mask_keeps_ones_like_video(self):
        video = _video(t=12, h=8, w=8)
        audio = _audio(20)
        out = _concat(
            _latent(video),
            _latent(audio, mask=torch.ones_like(audio)),
            fit_audio=False,
        )
        masks = out["noise_mask"].unbind()
        self.assertEqual(len(masks), 2)
        self.assertEqual(tuple(masks[0].shape), tuple(video.shape))
        self.assertTrue(torch.all(masks[0] == 1))
        self.assertEqual(tuple(masks[1].shape), tuple(audio.shape))

    def test_video_only_mask_omits_audio_half(self):
        video = _video()
        out = _concat(
            _latent(video, mask=torch.ones_like(video)),
            _latent(_audio()),
            fit_audio=False,
        )
        masks = out["noise_mask"].unbind()
        self.assertEqual(len(masks), 1)
        self.assertEqual(tuple(masks[0].shape), tuple(video.shape))

    def test_video_only_mask_round_trip_soundtrack_swap(self):
        # The advertised AV-in-video swap: feed this node's own 1-mask
        # output back in and replace the soundtrack.
        video = _video()
        a0 = _audio(8)
        a1 = _audio(11)
        first = _concat(
            _latent(video, mask=torch.ones_like(video)),
            _latent(a0),
            fit_audio=False,
        )
        self.assertEqual(len(first["noise_mask"].unbind()), 1)
        swapped = _concat(first, _latent(a1), fit_audio=False)
        v, a = swapped["samples"].unbind()
        self.assertTrue(torch.equal(v, video))
        self.assertTrue(torch.equal(a, a1))
        masks = swapped["noise_mask"].unbind()
        self.assertEqual(len(masks), 1)
        self.assertTrue(torch.equal(masks[0], first["noise_mask"].unbind()[0]))

    def test_video_only_mask_av_in_audio_has_no_mask(self):
        # AV in the audio socket: omitted trailing half means stream 1
        # has no mask, so a bare video + this AV produces no mask.
        video = _video()
        a0 = _audio(8)
        av = _concat(
            _latent(video, mask=torch.ones_like(video)),
            _latent(a0),
            fit_audio=False,
        )
        other = _video()
        out = _concat(_latent(other), av, fit_audio=False)
        v, a = out["samples"].unbind()
        self.assertTrue(torch.equal(v, other))
        self.assertTrue(torch.equal(a, a0))
        self.assertNotIn("noise_mask", out)

    def test_neither_mask_pops(self):
        out = _concat(
            _latent(_video(), batch_index=[0]),
            _latent(_audio()),
            fit_audio=False,
        )
        self.assertNotIn("noise_mask", out)

    def test_audio_keys_do_not_clobber_video(self):
        out = _concat(
            _latent(_video(), batch_index=[0], type="video"),
            _latent(_audio(), batch_index=[9], type="audio", extra="from-audio"),
            fit_audio=False,
        )
        self.assertEqual(out["batch_index"], [0])
        self.assertEqual(out["type"], "video")
        self.assertNotIn("extra", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

"""Luna MiniMax H3 Canvas — one node for the three numbers H3 actually needs.

Ask for an aspect ratio and a duration in seconds; get back a canvas, a frame
count, and the two frame rates. Replaces the usual pile of a generic resolution
node plus hand-written maths expressions for length and fps.

Why a dedicated node rather than a generic one: **H3 does not want a megapixel
target.** Its canvas is fully determined by the aspect ratio — a fixed 768 short
edge under a fixed 768x1344 area cap, each axis rounded to a multiple of 32. A
generic node makes you hand-tune a megapixel figure to land back on the number
the model already defines (1.032 MP). The `H3 canvas` mode here skips that.

Frame count is not free either: H3 samples on a grid where the count satisfies
`n % 17 == 5`, so 5, 22, 39 ... 243, 362. Ask for 7.5 s and you get 192 frames,
which is 8.0 s — a generic node snaps you silently, this one reports it.

`fps` and `output_fps` both leave from here so a frame count and its playback
rate cannot drift apart. Feeding `output_fps` to the save node is what stops the
classic "interpolation off but the fps still doubled" desync.

The constants (768 short edge, 768x1344 cap, multiple of 32, 24 fps, the 17k+5
grid, the 124-362 trained range) are requirements of the MiniMax H3 model —
interface facts, observable from its ComfyUI node signatures. The implementation
below is this pack's own; no code is taken from ComfyUI core, which is GPL-3.0
and would be incompatible with this pack's Apache-2.0 licence.
"""

from __future__ import annotations

import math

# H3 model parameters.
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24.0
FRAME_GRID = 17          # frame count must satisfy n % FRAME_GRID == FRAME_PHASE
FRAME_PHASE = 5
TRAINED_MIN, TRAINED_MAX = 124, 362

# Landscape form only, widest first, with a `portrait` toggle to flip. Listing both
# orientations would mean two names for one canvas and an obvious trap: pick "9:16"
# AND tick portrait and you are back to landscape. One list, one switch.
#
# Flipped, these cover the social sizes: 16:9 -> 9:16 (Reels, Shorts, TikTok),
# 5:4 -> 4:5 (Instagram portrait), 3:2 -> 2:3, 4:3 -> 3:4.
# ASCII only — these labels are saved into workflows and echoed to the console.
ASPECT_RATIOS = {
    "2.39:1 (anamorphic)": (239, 100),
    "21:9 (ultrawide)": (21, 9),
    "2:1": (2, 1),
    "1.91:1 (link preview)": (191, 100),
    "16:9 (widescreen)": (16, 9),
    "16:10": (16, 10),
    "3:2 (photo)": (3, 2),
    "4:3": (4, 3),
    "5:4": (5, 4),
    "1:1 (square)": (1, 1),
}

SIZE_MODES = ["H3 canvas", "megapixels"]


def _round_to(value: float) -> int:
    return max(CANVAS_MULTIPLE, int(round(value / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def align_frames(n: int) -> int:
    """Next frame count on H3's sampling grid, at or above n."""
    n = max(FRAME_PHASE, int(n))
    return n + ((FRAME_PHASE - n) % FRAME_GRID)


def h3_canvas(ratio: float) -> tuple[int, int]:
    """The model's own canvas for an aspect ratio: 768 short edge, area capped."""
    short, long = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE * max(ratio, 1.0 / ratio)
    if short * long > MAX_PIXELS:
        # The cap binds on wide ratios: shrink both edges, keep the ratio exact.
        shrink = math.sqrt(MAX_PIXELS / (short * long))
        short, long = short * shrink, long * shrink
    width, height = (long, short) if ratio >= 1.0 else (short, long)
    return _round_to(width), _round_to(height)


def megapixel_canvas(ratio: float, megapixels: float) -> tuple[int, int]:
    """Same aspect, a chosen area. For pushing past the model's default canvas."""
    area = max(0.01, float(megapixels)) * 1_000_000.0
    return _round_to(math.sqrt(area * ratio)), _round_to(math.sqrt(area / ratio))


class LunaMiniMaxH3Canvas:
    DESCRIPTION = (
        "Canvas, frame count and frame rates for MiniMax H3, from an aspect ratio "
        "and a duration in seconds.\n\n"
        "`H3 canvas` gives the model's own resolution for the ratio — no megapixel "
        "figure to guess. Switch to `megapixels` to push past it (H3 goes to 2K, at "
        "roughly four times the attention cost for double the area).\n\n"
        "Frame count snaps up to H3's sampling grid, and `info` reports the duration "
        "you actually got. Feed `output_fps` to the save node so interpolation and "
        "playback rate can never disagree."
    )

    CATEGORY = "Luna/MiniMax"
    FUNCTION = "resolve"
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "length", "fps", "output_fps",
                    "interpolation_factor", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Named `ratio`, not `aspect_ratio`: the widget row shows name and value
                # on one line, and the longer name pushed the value into an ellipsis.
                "ratio": (list(ASPECT_RATIOS), {
                    "default": "16:9 (widescreen)",
                    "tooltip": "Shape of the frame, landscape. The canvas follows from it.",
                }),
                "portrait": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Turn the ratio on its side. 16:9 becomes 9:16 for Reels and "
                               "Shorts, 5:4 becomes 4:5 for Instagram, 3:2 becomes 2:3.",
                }),
                "size_mode": (SIZE_MODES, {
                    "default": "H3 canvas",
                    "tooltip": "H3 canvas: the model's own resolution (1.03 MP at 16:9). "
                               "megapixels: your own target area, same aspect.",
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.03, "min": 0.1, "max": 8.0, "step": 0.01,
                    "tooltip": "Only used when size_mode is megapixels.",
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0, "min": 0.2, "max": 150.0, "step": 0.1,
                    "tooltip": "Snapped up to H3's frame grid; info reports what you got.",
                }),
                "interpolation_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wire this to the same boolean that gates your frame "
                               "interpolation nodes. When false the factor is forced to 1 "
                               "and output_fps drops back to 24, so bypassing the "
                               "interpolation cannot leave the frame rate doubled.",
                }),
                "interpolation_factor": ("INT", {
                    "default": 2, "min": 1, "max": 8,
                    "tooltip": "Frames per source frame. Drives both the interpolation "
                               "node and output_fps, so they cannot drift. Keeps its value "
                               "while interpolation_enabled is false.",
                }),
            }
        }

    def resolve(self, ratio, portrait, size_mode, megapixels,
                duration_seconds, interpolation_enabled, interpolation_factor):
        w_ratio, h_ratio = ASPECT_RATIOS[ratio]
        if portrait:
            w_ratio, h_ratio = h_ratio, w_ratio
        aspect = w_ratio / h_ratio

        if size_mode == "megapixels":
            width, height = megapixel_canvas(aspect, megapixels)
        else:
            width, height = h3_canvas(aspect)

        length = align_frames(round(duration_seconds * FPS))
        # Clamped once, then both returned: the factor that leaves this node is the
        # same one output_fps was computed from, so driving the interpolation node
        # from it cannot disagree with the frame rate. Gated off, the factor is 1
        # rather than the widget's value -- bypassing the interpolation nodes while
        # the rate stayed doubled is the exact failure this node exists to prevent.
        factor = max(1, int(interpolation_factor)) if interpolation_enabled else 1
        output_fps = FPS * factor
        actual_seconds = length / FPS

        # ASCII only: this string is printed, and a Windows console on cp1252
        # raises UnicodeEncodeError on the likes of a middle dot or an arrow.
        info = (f"{width}x{height} ({w_ratio}:{h_ratio}) | {width * height / 1e6:.2f} MP | "
                f"{length} frames | {actual_seconds:.2f}s @ {FPS:g} fps")
        if factor > 1:
            info += f" -> x{factor} -> {output_fps:g} fps out"
        elif not interpolation_enabled and int(interpolation_factor) > 1:
            info += f" | interpolation off (x{int(interpolation_factor)} held)"

        notes = []
        # The grid steps in 17-frame jumps, so a fraction of a second of snapping is
        # unavoidable and not worth flagging -- the line above already states the real
        # duration. Only say something when it lands more than half a step away.
        if abs(actual_seconds - duration_seconds) >= (FRAME_GRID / FPS) / 2:
            notes.append(f"asked {duration_seconds:g}s, grid gives {actual_seconds:.2f}s")
        if not TRAINED_MIN <= length <= TRAINED_MAX:
            notes.append(f"outside the trained {TRAINED_MIN}-{TRAINED_MAX} frame range")
        if width * height > MAX_PIXELS:
            notes.append(f"{width * height / MAX_PIXELS:.1f}x the model's default canvas")
        if notes:
            info += "  (" + "; ".join(notes) + ")"

        print(f"[Luna MiniMax H3 Canvas] {info}")
        return (width, height, length, FPS, output_fps, factor, info)


NODE_CLASS_MAPPINGS = {"LunaMiniMaxH3Canvas": LunaMiniMaxH3Canvas}
NODE_DISPLAY_NAME_MAPPINGS = {"LunaMiniMaxH3Canvas": "Luna MiniMax H3 Canvas"}

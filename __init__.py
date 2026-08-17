"""ComfyUI-LunaMiniMaxH3 — the nodes MiniMax H3 needs and does not ship.

Luna MiniMax H3 Canvas: ask for an aspect ratio and a duration, get the canvas,
the frame count and both frame rates, already on H3's 17k+5 grid.

Luna H3ConcatAVLatent: pairs a video latent with an audio latent into the joint
AV latent H3 samples from — the way in for redrawing existing footage at
denoise < 1.

Both are model-specific by design: H3's canvas is determined by its aspect ratio
and its latents come in pairs, so the generic nodes make you rediscover constants
the model already fixes.
"""

from .h3_canvas import (
    NODE_CLASS_MAPPINGS as _CANVAS_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _CANVAS_DISPLAY_MAPPINGS,
)
from .h3_latent import (
    NODE_CLASS_MAPPINGS as _LATENT_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _LATENT_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {**_CANVAS_CLASS_MAPPINGS, **_LATENT_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_CANVAS_DISPLAY_MAPPINGS, **_LATENT_DISPLAY_MAPPINGS}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

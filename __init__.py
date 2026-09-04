"""ComfyUI-LunaMiniMaxH3 — the nodes MiniMax H3 needs and does not ship.

Luna MiniMax H3 Canvas: ask for an aspect ratio and a duration, get the canvas,
the frame count and both frame rates, already on H3's 17k+5 grid.

Luna H3ConcatAVLatent: pairs a video latent with an audio latent into the joint
AV latent H3 samples from — the way in for redrawing existing footage at
denoise < 1.

Both are model-specific by design: H3's canvas is determined by its aspect ratio
and its latents come in pairs, so the generic nodes make you rediscover constants
the model already fixes.

Registration is the V3 API (`comfy_entrypoint` + `ComfyExtension`). ComfyUI's
loader takes `NODE_CLASS_MAPPINGS` first and only falls through to
`comfy_entrypoint` when it is absent (`nodes.py`, "V1 node definition" /
"V3 Extension Definition") — so a pack is one or the other, never both, and
defining both would silently keep the pack on V1. Node ids are unchanged, so
existing workflows still resolve.
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .h3_canvas import LunaMiniMaxH3Canvas
from .h3_latent import LunaH3ConcatAVLatent


class LunaMiniMaxH3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LunaMiniMaxH3Canvas, LunaH3ConcatAVLatent]


async def comfy_entrypoint() -> LunaMiniMaxH3Extension:
    return LunaMiniMaxH3Extension()


WEB_DIRECTORY = "./js"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]

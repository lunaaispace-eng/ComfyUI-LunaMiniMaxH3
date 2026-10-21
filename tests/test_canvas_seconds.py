"""Luna MiniMax H3 Canvas: `length` on the 17k+5 grid and `seconds` = length / 24,
appended as the last output. Stubs comfy_api so it runs without ComfyUI.

    E:\ComfyUI-Easy-Install\python_embeded\python.exe -m unittest discover -s tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _stub_io():
    class _Out:
        def __init__(self, display_name=None, **kw):
            self.display_name = display_name

    class _In:
        def __init__(self, *a, **kw):
            pass

    io = SimpleNamespace(
        ComfyNode=object, Schema=lambda **kw: SimpleNamespace(**kw),
        Int=SimpleNamespace(Input=_In, Output=_Out), Float=SimpleNamespace(Input=_In, Output=_Out),
        String=SimpleNamespace(Input=_In, Output=_Out), Boolean=SimpleNamespace(Input=_In),
        Combo=SimpleNamespace(Input=_In), NodeOutput=lambda *a: a,
    )
    latest = ModuleType("comfy_api.latest")
    latest.io = io
    return {"comfy_api": ModuleType("comfy_api"), "comfy_api.latest": latest}


# The stub is in place only while h3_canvas imports, so the other test files in this
# folder still see whatever comfy_api they set up themselves.
_saved = {k: sys.modules.get(k) for k in ("comfy_api", "comfy_api.latest")}
sys.modules.update(_stub_io())
try:
    _spec = importlib.util.spec_from_file_location("h3_canvas", Path(__file__).parent.parent / "h3_canvas.py")
    h3_canvas = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(h3_canvas)
finally:
    for k, v in _saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
Canvas = h3_canvas.LunaMiniMaxH3Canvas


def run(seconds):
    return Canvas.execute("16:9 (widescreen)", False, "H3 canvas", 1.03, seconds, True, 2)


class CanvasSecondsTests(unittest.TestCase):
    def test_seconds_is_last_output(self):
        names = [o.display_name for o in Canvas.define_schema().outputs]
        self.assertEqual(names[-1], "seconds")
        self.assertEqual(names[:7], ["width", "height", "length", "fps", "output_fps",
                                     "interpolation_factor", "info"])

    def test_seconds_follow_the_grid(self):
        for asked, frames in ((5.0, 124), (7.5, 192), (10.0, 243)):
            out = run(asked)
            self.assertEqual(out[2], frames)
            self.assertAlmostEqual(out[7], frames / 24.0)


if __name__ == "__main__":
    unittest.main()

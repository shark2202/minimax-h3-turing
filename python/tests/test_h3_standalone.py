"""GPU-free tests for the embedded-ComfyUI H3 demo (standalone/h3_demo.py).

Nothing here needs torch or a ComfyUI checkout: the graph helpers run on the
bundled workflow JSONs, and execute_graph runs against fake node mappings
(the torch.inference_mode wrapper degrades to a null context without torch).
"""

import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "standalone"))

import h3_demo  # noqa: E402


class FakeNode:
    """Minimal ComfyUI node protocol: class with FUNCTION + callable inputs."""
    def __init__(self, fn):
        self._fn = fn
        self.FUNCTION = "run"

    def run(self, **kwargs):
        return self._fn(**kwargs)

    __call__ = lambda self: self   # instantiate -> itself


class TestPatchGraph(unittest.TestCase):
    def test_t2v_patch(self):
        g = h3_demo.load_workflow("h3_w4a8_t2v_compat_api.json")
        h3_demo.patch_graph(g, "misty forest", 7, image_name=None)
        self.assertEqual(g["6"]["inputs"]["prompt"], "misty forest")
        self.assertEqual(g["10"]["inputs"]["noise_seed"], 7)

    def test_i2v_patch_and_route_guards(self):
        g = h3_demo.load_workflow("h3_w4a8_i2v_compat_api.json")
        h3_demo.patch_graph(g, "she turns", 9, image_name="staged.png")
        self.assertEqual(g["20"]["inputs"]["image"], "staged.png")
        # i2v workflow without --image, and t2v workflow with it: both refuse
        with self.assertRaises(SystemExit):
            h3_demo.patch_graph(h3_demo.load_workflow("h3_w4a8_i2v_compat_api.json"),
                                "x", 1, image_name=None)
        with self.assertRaises(SystemExit):
            h3_demo.patch_graph(h3_demo.load_workflow("h3_w4a8_t2v_compat_api.json"),
                                "x", 1, image_name="y.png")

    def test_prompt_placeholder_already_consumed(self):
        g = h3_demo.load_workflow("h3_w4a8_t2v_compat_api.json")
        h3_demo.patch_graph(g, "a", 1, image_name=None)
        with self.assertRaises(SystemExit):
            h3_demo.patch_graph(g, "b", 1, image_name=None)


class TestExecuteGraph(unittest.TestCase):
    def _mappings(self, order):
        def make(name):
            def fn(**kw):
                order.append(name)
                return ({"n": name, "kw": kw},)   # tuple like real node outputs
            return FakeNode(fn)
        return {"A": make("A"), "B": make("B"), "C": make("C")}

    def test_topological_order_and_edge_resolution(self):
        order = []
        graph = {
            "1": {"class_type": "C", "inputs": {"x": ["2", 0]}},   # C after B
            "2": {"class_type": "B", "inputs": {"x": ["3", 0]}},   # B after A
            "3": {"class_type": "A", "inputs": {"x": 5}},
        }
        results = h3_demo.execute_graph(graph, self._mappings(order), log=lambda *_: None)
        self.assertEqual(order, ["A", "B", "C"])
        # C's input x resolved to B's output, whose input resolved to A's
        self.assertEqual(results["1"][0]["n"], "C")
        self.assertEqual(results["1"][0]["kw"]["x"]["n"], "B")
        self.assertEqual(results["1"][0]["kw"]["x"]["kw"]["x"],
                         {"n": "A", "kw": {"x": 5}})

    def test_dict_result_nodes(self):
        # some nodes return {"result": (...)} - resolve must handle both
        class DictNode(FakeNode):
            def run(self, **kw):
                return {"result": ("payload",)}
        g = {"1": {"class_type": "A", "inputs": {}},
             "2": {"class_type": "B", "inputs": {"v": ["1", 0]}}}
        res = h3_demo.execute_graph(g, {"A": DictNode(None), "B": FakeNode(lambda v: v)},
                                    log=lambda *_: None)
        self.assertEqual(res["2"], "payload")

    def test_cycle_detected(self):
        g = {"1": {"class_type": "A", "inputs": {"x": ["2", 0]}},
             "2": {"class_type": "A", "inputs": {"x": ["1", 0]}}}
        with self.assertRaises(RuntimeError) as cm:
            h3_demo.execute_graph(g, self._mappings([]), log=lambda *_: None)
        self.assertIn("cycle", str(cm.exception))


class TestCollect(unittest.TestCase):
    def test_only_new_video_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.mp4"; old.write_bytes(b"0"); time.sleep(0.01)
            t0 = time.time(); time.sleep(0.01)
            (root / "sub").mkdir()
            (root / "sub" / "H3Lite_00001.mp4").write_bytes(b"new")
            (root / "note.txt").write_text("not a video")
            got = h3_demo.collect_new_files(root, t0)
            self.assertEqual([p.name for p in got], ["H3Lite_00001.mp4"])


class TestCliGuards(unittest.TestCase):
    DEMO = Path(h3_demo.__file__)

    def run_demo(self, *argv):
        return subprocess.run([sys.executable, str(self.DEMO), *argv],
                              capture_output=True, text=True, timeout=60)

    def test_help_needs_nothing_heavy(self):
        r = self.run_demo("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--comfy-dir", r.stdout)

    def test_missing_comfy_dir_refused(self):
        r = self.run_demo("--prompt", "x")
        self.assertEqual(r.returncode, 1)
        self.assertIn("COMFYUI_PATH", r.stderr)

    def test_bad_comfy_dir_refused_before_bootstrapping(self):
        r = self.run_demo("--comfy-dir", "/tmp", "--prompt", "x")
        self.assertEqual(r.returncode, 1)
        self.assertIn("nodes.py", r.stderr)

    def test_launch_discipline_flags_present(self):
        joined = " ".join(h3_demo.VRAM_FLAGS)
        for flag in ("--reserve-vram 2.5", "--vram-headroom 0.5",
                     "--disable-pinned-memory"):
            self.assertIn(flag, joined)


if __name__ == "__main__":
    unittest.main()

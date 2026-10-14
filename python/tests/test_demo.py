"""Demo pipeline tests - the full text/image -> video flow against an
in-process fake ComfyUI (http.server). No GPU required: the fake implements
exactly the five endpoints the demo touches, so the test proves the wiring
(alive -> upload -> submit -> wait -> download), not the render itself.
"""

import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from h3turing import load, set_image, set_prompt, set_seed  # noqa: E402
from h3turing.cli import main  # noqa: E402
from h3turing.client import ComfyUI, collect_outputs  # noqa: E402

PID = "fake-pid-1234abcd"
VIDEO_BYTES = b"FAKEMP4-" + b"x" * 64

_fake_server = None


def _fake_base() -> str:
    """One shared fake ComfyUI for the whole module (started lazily)."""
    global _fake_server
    if _fake_server is None:
        _fake_server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfyUI)
        threading.Thread(target=_fake_server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{_fake_server.server_address[1]}"


class FakeComfyUI(BaseHTTPRequestHandler):
    """Records what the demo sends; answers like a finished render."""
    uploads = []       # (content_type, body) of /upload/image calls
    graphs = []        # graph dicts POSTed to /prompt

    def log_message(self, *a):  # silence test output
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/system_stats":
            self._json({"system": {"comfyui_version": "0.33.1"}})
        elif u.path == "/queue":
            self._json({"queue_running": [], "queue_pending": []})
        elif u.path == f"/history/{PID}":
            self._json({PID: {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {"15": {"gifs": [
                    {"filename": "H3Lite_00001.mp4", "subfolder": "video",
                     "type": "output"}]}},
            }})
        elif u.path == "/view":
            self.send_response(200)
            self.send_header("Content-Length", str(len(VIDEO_BYTES)))
            self.end_headers()
            self.wfile.write(VIDEO_BYTES)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.path == "/upload/image":
            FakeComfyUI.uploads.append((self.headers.get("Content-Type", ""), body))
            self._json({"name": "demo_first_frame.png", "subfolder": "", "type": "input"})
        elif self.path == "/prompt":
            graph = json.loads(body)["prompt"]
            FakeComfyUI.graphs.append(graph)
            self._json({"prompt_id": PID})
        else:
            self._json({"error": "not found"}, 404)


class TestDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _fake_base()

    def _run(self, argv, capture_err=False):
        buf, err = io.StringIO(), io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, buf.getvalue() + (err.getvalue() if capture_err else "")

    def test_i2v_end_to_end(self):
        """image + prompt -> upload -> i2v-final graph -> downloaded mp4."""
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "first_frame.png"
            img.write_bytes(b"\x89PNG fake bytes")
            code, out = self._run([
                "demo", "--prompt", "the swordsman turns to camera",
                "--image", str(img), "--seed", "7",
                "--base", self.base, "--out", str(Path(tmp) / "out"),
                "--poll-every", "0.01"])
            self.assertEqual(code, 0, out)
            mp4 = Path(tmp) / "out" / "H3Lite_00001.mp4"
            self.assertTrue(mp4.exists())
            self.assertEqual(mp4.read_bytes(), VIDEO_BYTES)
            self.assertIn("saved:", out)
            self.assertIn("i2v-final", out)

            # the graph that reached the server: prompt replaced, seed set,
            # first frame = the name /upload/image returned (not our path)
            g = FakeComfyUI.graphs[-1]
            m = g["6"]["inputs"]
            self.assertEqual(m["prompt"], "the swordsman turns to camera")
            self.assertEqual(m["first_frame"], ["20", 0])
            self.assertEqual(g["20"]["inputs"]["image"], "demo_first_frame.png")
            self.assertEqual(g["10"]["inputs"]["noise_seed"], 7)
            # and the upload was real multipart with the file payload inside
            ctype, body = FakeComfyUI.uploads[-1]
            self.assertIn("multipart/form-data; boundary=h3turing", ctype)
            self.assertIn(b'filename="first_frame.png"', body)
            self.assertIn(b"\x89PNG fake bytes", body)

    def test_t2v_end_to_end(self):
        """prompt only -> t2v-final, no upload happens."""
        with tempfile.TemporaryDirectory() as tmp:
            n_uploads = len(FakeComfyUI.uploads)
            code, out = self._run([
                "demo", "--prompt", "misty bamboo forest",
                "--base", self.base, "--out", str(Path(tmp) / "out"),
                "--poll-every", "0.01"])
            self.assertEqual(code, 0, out)
            self.assertIn("t2v-final", out)
            self.assertEqual(len(FakeComfyUI.uploads), n_uploads)
            g = FakeComfyUI.graphs[-1]
            self.assertEqual(g["6"]["inputs"]["prompt"], "misty bamboo forest")
            self.assertNotIn("20", g)  # t2v graph has no LoadImage node

    def test_out_collision_keeps_old_take(self):
        """A same-named file already on disk -> pid-prefixed name, no overwrite."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "H3Lite_00001.mp4").write_bytes(b"previous take")
            code, _ = self._run([
                "demo", "--base", self.base, "--out", str(out),
                "--poll-every", "0.01"])
            self.assertEqual(code, 0)
            self.assertEqual((out / "H3Lite_00001.mp4").read_bytes(), b"previous take")
            self.assertEqual((out / f"{PID[:8]}_H3Lite_00001.mp4").read_bytes(), VIDEO_BYTES)

    def test_unreachable_server_fails_fast(self):
        buf, err = io.StringIO(), io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = main(["demo", "--base", "http://127.0.0.1:9",
                         "--poll-every", "0.01"])
        self.assertEqual(code, 1)
        self.assertIn("not reachable", err.getvalue() + buf.getvalue())

    def test_tier_route_mismatch_rejected(self):
        code, out = self._run(["demo", "--image", "x.png", "--tier", "t2v-final"],
                              capture_err=True)
        self.assertEqual(code, 1)
        self.assertIn("drop --image", out)


class TestHelpers(unittest.TestCase):
    def test_set_image_roundtrip(self):
        g = set_image(load("h3_w4a8_i2v_compat_api"), "frame.png")
        self.assertEqual(g["20"]["inputs"]["image"], "frame.png")
        with self.assertRaises(ValueError):
            set_image(g, "again.png")          # placeholder already consumed
        with self.assertRaises(ValueError):
            set_image(load("h3_w4a8_t2v_compat_api"), "frame.png")  # t2v: no image

    def test_collect_outputs_scans_all_ui_keys(self):
        history = {"outputs": {
            "15": {"gifs": [{"filename": "a.mp4", "subfolder": "video", "type": "output"}]},
            "9": {"videos": [{"filename": "b.webm", "subfolder": "", "type": "output"}]},
            "3": {"images": [{"filename": "c.png", "subfolder": "", "type": "output"},
                              {"subfolder": "no filename - skipped"}]},
            "7": {"audio": [{"filename": "ignored.wav"}]},
        }}
        names = [f["filename"] for f in collect_outputs(history)]
        self.assertEqual(names, ["a.mp4", "b.webm", "c.png"])
        self.assertEqual(collect_outputs({}), [])

    def test_upload_download_roundtrip_via_client(self):
        """Direct client calls hit the same fake endpoints as the demo."""
        with tempfile.TemporaryDirectory() as tmp:
            comfy = ComfyUI(base=_fake_base())
            img = Path(tmp) / "f.png"
            img.write_bytes(b"pngdata")
            self.assertEqual(comfy.upload_image(img), "demo_first_frame.png")
            dest = comfy.download({"filename": "H3Lite_00001.mp4",
                                   "subfolder": "video", "type": "output"},
                                  Path(tmp) / "dl" / "v.mp4")
            self.assertEqual(dest.read_bytes(), VIDEO_BYTES)


if __name__ == "__main__":
    unittest.main()

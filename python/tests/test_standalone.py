"""Guards for the standalone (no-ComfyUI) demo script.

The demo's contract: `--help` and usage errors work on a box with NO torch/
diffusers installed (heavy imports live inside the run functions), and the
Wan temporal-stride rule (frames = 4k+1) is enforced before any loading.
"""

import subprocess
import sys
import unittest
from pathlib import Path

DEMO = Path(__file__).resolve().parents[2] / "standalone" / "demo.py"


class TestStandaloneDemo(unittest.TestCase):
    def run_demo(self, *argv):
        return subprocess.run([sys.executable, str(DEMO), *argv],
                              capture_output=True, text=True, timeout=60)

    def test_help_needs_no_heavy_deps(self):
        r = self.run_demo("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--prompt", r.stdout)
        self.assertIn("--image", r.stdout)

    def test_bad_frame_count_rejected_before_loading(self):
        r = self.run_demo("--frames", "80", "--prompt", "x")
        self.assertEqual(r.returncode, 1)
        self.assertIn("4k+1", r.stderr + r.stdout)
        self.assertNotIn("loading weights", r.stdout)  # failed before any model work


if __name__ == "__main__":
    unittest.main()

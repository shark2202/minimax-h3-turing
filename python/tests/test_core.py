"""GPU-free tests: tier registry integrity, bundled workflows, client helpers.

Run: python -m unittest discover -s python/tests -v
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from h3turing import TIERS, get, load, render, set_prompt, set_seed  # noqa: E402
from h3turing.client import ValidationError, bump_seed  # noqa: E402
from h3turing.tiers import reproducible_only  # noqa: E402


class TestTiers(unittest.TestCase):
    def test_tiers_present(self):
        self.assertEqual(len(TIERS), 9)  # 出厂六件套 + PDD-Acc 两档 + 三个缓存草稿档
        self.assertTrue({"t2v-final", "t2v-draft", "i2v-final", "i2v-draft", "t2v-ultra", "i2v-ultra"} <= set(TIERS))

    def test_reproducibility_discipline(self):
        """The core rule: T8 tiers are drafts (not reproducible), plain tiers are finals."""
        for key, tier in TIERS.items():
            if tier.cached:
                self.assertFalse(tier.reproducible, f"{key}: T8 tiers must not be reproducible")
            else:
                self.assertTrue(tier.reproducible, f"{key}: plain tiers must be reproducible")

    def test_aliases_resolve(self):
        self.assertIs(get("draft"), TIERS["t2v-draft"])
        self.assertIs(get("fastest"), TIERS["t2v-draft"])
        self.assertIs(get("roughest"), TIERS["t2v-draft-rough"])
        self.assertIs(get("i2v-fastest"), TIERS["i2v-ultra"])
        self.assertIs(get("final"), TIERS["t2v-final"])

    def test_unknown_tier_lists_valid(self):
        with self.assertRaises(KeyError) as cm:
            get("nope")
        self.assertIn("t2v-final", str(cm.exception))

    def test_measured_numbers_positive(self):
        for tier in TIERS.values():
            self.assertGreater(tier.measured_s, 0, tier.key)


class TestWorkflows(unittest.TestCase):
    def test_all_bundled_workflows_parse(self):
        for tier in TIERS.values():
            g = load(tier.workflow)   # raises on invalid JSON
            self.assertIsInstance(g, dict)
            self.assertGreater(len(g), 5, tier.workflow)

    def test_no_local_paths_in_bundled_data(self):
        """Red line: no private machine paths may leak into shipped workflows."""
        banned = ["d:/tools", "e:/work", "c:/users", "fangu", "112.74."]
        for tier in TIERS.values():
            raw = json.dumps(load(tier.workflow)).lower()
            for b in banned:
                self.assertNotIn(b, raw, f"{tier.workflow} leaks {b!r}")

    def test_draft_workflows_carry_explicit_cache_inputs(self):
        """v3-API discipline: cached-tier JSONs must pass ALL cache-node inputs explicitly.

        Covers every cache family in use (T8 block cache, TeaCache step cache,
        FirstBlockCache) - a missing input means HTTP 400 over /prompt, not a
        silent default.
        """
        t8_inputs = {"model", "residual_diff_threshold", "start_percent", "end_percent",
                     "max_consecutive_hits", "cache_device", "metric_stride", "verbose"}
        teacache_inputs = {"model", "rel_l1_thresh", "start_step", "end_step", "total_steps"}
        fbc_inputs = {"model", "mode", "threshold", "start_percent", "end_percent",
                      "max_consecutive_hits", "temporal_guard"}
        for tier in TIERS.values():
            if not tier.cached:
                continue
            g = load(tier.workflow)
            seen = 0
            for node in g.values():
                ct = node.get("class_type")
                if ct == "MiniMaxH3BlockCacheT8":
                    seen += 1
                    self.assertEqual(set(node["inputs"]), t8_inputs, tier.workflow)
                    self.assertEqual(node["inputs"]["residual_diff_threshold"], 1.0, tier.workflow)
                elif ct == "MiniMaxH3TeaCache":
                    seen += 1
                    self.assertEqual(set(node["inputs"]), teacache_inputs, tier.workflow)
                elif ct == "ApplyMiniMaxH3FirstBlockCache":
                    seen += 1
                    self.assertEqual(set(node["inputs"]), fbc_inputs, tier.workflow)
                    self.assertIn(node["inputs"]["threshold"], (0.6, 1.0), tier.workflow)
            self.assertGreaterEqual(seen, 1, f"{tier.workflow}: cached tier with no cache node")

    def test_prompt_seed_helpers(self):
        g = set_seed(set_prompt(load("h3_w4a8_t2v_compat_api"), "hello"), 123)
        prompts = [n["inputs"]["prompt"] for n in g.values()
                   if isinstance(n, dict) and "prompt" in n.get("inputs", {})]
        self.assertEqual(prompts, ["hello"])
        seeds = [n["inputs"]["noise_seed"] for n in g.values()
                 if isinstance(n, dict) and "noise_seed" in n.get("inputs", {})]
        self.assertEqual(seeds, [123])
        with self.assertRaises(ValueError):
            set_prompt(g, "again")   # placeholder already consumed

    def test_render_is_deep_copy(self):
        g = set_seed(load("h3_w4a8_t2v_compat_api"), 1)
        r = render(g)
        r["10"]["inputs"]["noise_seed"] = 999
        self.assertEqual(g["10"]["inputs"]["noise_seed"], 1)


class TestClientHelpers(unittest.TestCase):
    def test_bump_seed_breaks_dedup(self):
        g = set_seed(load("h3_w4a8_t2v_compat_api"), 100)
        before = g["10"]["inputs"]["noise_seed"]
        bump_seed(g)
        self.assertEqual(g["10"]["inputs"]["noise_seed"], before + 1)

    def test_validation_error_parses_missing_inputs(self):
        payload = {"error": {"type": "prompt_outputs_failed_validation"},
                   "node_errors": {"200": {"errors": [
                       {"type": "required_input_missing", "details": "verbose"},
                       {"type": "required_input_missing", "details": "cache_device"}]}}}
        err = ValidationError(payload)
        self.assertIn("v3-API", str(err))
        self.assertEqual(err.missing, [("200", "verbose"), ("200", "cache_device")])


class TestReproducibleOnly(unittest.TestCase):
    def test_returns_plain_tiers(self):
        r = reproducible_only()
        self.assertIn("t2v-final", r)
        self.assertNotIn("t2v-draft", r)


if __name__ == "__main__":
    unittest.main()

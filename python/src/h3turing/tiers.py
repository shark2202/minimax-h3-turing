"""Tier presets - the optimization experience, as data.

Every field here was measured on a 2080Ti 22G (Turing sm_75); see the
handbook's docs/10-benchmarks for the full dataset. The two invariants that
matter most:

* ``cached=True`` tiers are NOT same-seed reproducible (cache hits fork the
  sampling trajectory) - drafts only. Covers every cache family measured on
  this box: T8 block cache, TeaCache step cache, FirstBlockCache.
* PDD tiers require a master-environment ComfyUI (PR #15908 era) - except the
  PDD-Acc node route, which runs on v0.33.x but measured slower and weaker
  here (see RUNBOOK §5.6).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    key: str
    route: str                 # "t2v" | "i2v"
    workflow: str              # bundled data file
    lora: str
    steps: int
    cached: bool
    reproducible: bool
    env: str                   # minimum ComfyUI environment
    measured_s: int            # 2080Ti 22G reference, warm run
    notes: str = ""


TIERS: dict[str, Tier] = {
    "t2v-final": Tier(
        key="t2v-final", route="t2v",
        workflow="h3_w4a8_t2v_compat_api.json",
        lora="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors",
        steps=4, cached=False, reproducible=True, env="0.33.1+",
        measured_s=232,   # 本机实测 232.46s（2026-10-21），旧值 280s 是上游手册的数
        notes="Baseline. Use for everything that enters the cut.",
    ),
    "t2v-draft": Tier(
        key="t2v-draft", route="t2v",
        workflow="h3_w4a8_t2v_t8draft_api.json",
        lora="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors",
        steps=4, cached=True, reproducible=False, env="0.33.1+",
        measured_s=157,   # 本机实测 156.62s（2026-10-21）
        notes="T8 threshold 1.0. Defaults (0.12) are a negative optimization on 4-step routes.",
    ),
    # Fastest usable draft on this box: FirstBlockCache at threshold 0.60.
    # Measured 2026-10-21: 108 s (-53%), frame gradient 92% of the plain tier,
    # audio -17.6 dBFS (only 3 dB under the plain tier) - see RUNBOOK 5.11.
    "t2v-draft-fast": Tier(
        key="t2v-draft-fast", route="t2v",
        workflow="h3_w4a8_t2v_firstblockcache6_api.json",
        lora="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors",
        steps=4, cached=True, reproducible=False, env="0.33.1+ (needs FirstBlockCache node)",
        measured_s=108,
        notes="FirstBlockCache threshold 0.60. Fastest draft that keeps picture and audio: 108 s.",
    ),
    # Roughest tier: same node at threshold 1.0. 67 s (-71%) but the frame
    # gradient drops to 65% and the audio to -29.6 dBFS - look-dev only.
    "t2v-draft-rough": Tier(
        key="t2v-draft-rough", route="t2v",
        workflow="h3_w4a8_t2v_firstblockcache1_api.json",
        lora="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors",
        steps=4, cached=True, reproducible=False, env="0.33.1+ (needs FirstBlockCache node)",
        measured_s=67,
        notes="FirstBlockCache threshold 1.0. 67 s - pick composition only, never the cut.",
    ),
    "i2v-final": Tier(
        key="i2v-final", route="i2v",
        workflow="h3_w4a8_i2v_compat_api.json",
        lora="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors",
        steps=4, cached=False, reproducible=True, env="0.33.1+",
        measured_s=212,
        notes="First-frame anchored (face lock). Needs your own first-frame image.",
    ),
    # i2v draft: same FirstBlockCache as the t2v draft-fast tier, measured 2026-10-21
    # (110.93 s, -48%); the first-frame lock survives the cache (r=0.998, x26).
    "i2v-draft": Tier(
        key="i2v-draft", route="i2v",
        workflow="h3_w4a8_i2v_firstblockcache6_api.json",
        lora="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors",
        steps=4, cached=True, reproducible=False, env="0.33.1+ (needs FirstBlockCache node)",
        measured_s=111,
        notes="FirstBlockCache 0.60 on the i2v route: 111 s, first-frame lock intact.",
    ),
# PDD 8-step via the ComfyUI-MiniMax-H3-PDD-Acc node (no master upgrade needed)
    "t2v-pddacc8": Tier(
        key="t2v-pddacc8", route="t2v",
        workflow="h3_w4a8_t2v_pddacc8_api.json",
        lora="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        steps=8, cached=False, reproducible=True, env="0.33.1+ (needs PDD-Acc node)",
        measured_s=399,
        notes="PDD 8-step through the ComfyUI-MiniMax-H3-PDD-Acc node: no master upgrade. "
              "The node owns the sigmas (euler only), and forbids turbo/cache nodes on top.",
    ),
# Same node route at nfe=4 (officially sanctioned: two blocks per step) - the only PDD
    # configuration that can compete with the 4-step turbo tier on speed.
    "t2v-pddacc4": Tier(
        key="t2v-pddacc4", route="t2v",
        workflow="h3_w4a8_t2v_pddacc4_api.json",
        lora="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        steps=4, cached=False, reproducible=True, env="0.33.1+ (needs PDD-Acc node)",
        measured_s=227,
        notes="PDD at nfe=4 via the PDD-Acc node: same 4 forwards as the turbo tier, so it competes on speed, not just quality.",
    ),
    "t2v-ultra": Tier(
        key="t2v-ultra", route="t2v",
        workflow="h3_w4a8_t2v_pdd8_t8_api.json",
        lora="MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors",
        steps=8, cached=True, reproducible=False, env="master (PDD backport, see docs/09)",
        measured_s=210,
        notes="Fastest t2v + distilled quality. 6/8 hits.",
    ),
    "i2v-ultra": Tier(
        key="i2v-ultra", route="i2v",
        workflow="h3_w4a8_i2v_pdd8_t8_api.json",
        lora="MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors",
        steps=8, cached=True, reproducible=False, env="master (PDD backport, see docs/09)",
        measured_s=192,
        notes="Fastest record in the project (-51%). Face-locked ultra drafts.",
    ),
}

# Aliases for the common questions.
ALIASES = {
    "draft": "t2v-draft",
    "final": "t2v-final",
    "fastest": "t2v-draft-fast",   # 2026-10-21: i2v-ultra needs the master env, t2v-draft-fast does not
    "i2v-fastest": "i2v-ultra",
}


def get(key: str) -> Tier:
    """Resolve a tier by key or alias. Raises KeyError with the valid keys."""
    key = ALIASES.get(key, key)
    if key not in TIERS:
        raise KeyError(f"unknown tier {key!r}; valid: {sorted(TIERS)} + aliases {sorted(ALIASES)}")
    return TIERS[key]


def reproducible_only() -> dict[str, Tier]:
    """Tiers safe for final shots (same-seed reproducible)."""
    return {k: t for k, t in TIERS.items() if t.reproducible}

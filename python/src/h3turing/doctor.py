"""Environment checks - run before a production batch.

Every check maps to a handbook FAQ entry; failures here are the known ways a
batch dies hours later.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Core weight set (docs/05 download list). File names as shipped; substitutes
# are fine if the workflow loader nodes are renamed to match.
CORE_WEIGHTS = {
    "diffusion_models": ["minimax_h3_fl2va_pruned_w4a8_mixed"],
    "text_encoders": ["qwen3vl_4b"],          # fp8_scaled / int4_convrot variants
    "clip_projections": ["mmh3-4b-ClipProj"],
    "vae": ["minimax_h3_video_vae_fp16", "minimax_h3_audio_vae_fp32"],
    "loras": ["fl2v_turbo_4step"],             # prefix match on either LoRA line
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def check_comfyui(base: str = "http://127.0.0.1:8188", timeout: float = 5.0) -> Check:
    try:
        urllib.request.urlopen(base + "/system_stats", timeout=timeout).read()
        return Check("comfyui_alive", True, base)
    except Exception as e:
        return Check("comfyui_alive", False, f"{base} unreachable ({e}) - start it via h3_launch")


def check_queue_clean(base: str = "http://127.0.0.1:8188", timeout: float = 10.0) -> Check:
    """A persisted queue revives jobs on restart (FAQ #3); clear it before prod."""
    try:
        q = json.loads(urllib.request.urlopen(base + "/queue", timeout=timeout).read())
        running, pending = len(q.get("queue_running", [])), len(q.get("queue_pending", []))
        ok = running == 0 and pending == 0
        return Check("queue_clean", ok, f"running={running} pending={pending}"
                     + ("" if ok else " - clear /queue before production"))
    except Exception as e:
        return Check("queue_clean", False, str(e))


def check_t8_node(base: str = "http://127.0.0.1:8188", timeout: float = 10.0) -> Check:
    """Draft tiers need MiniMaxH3BlockCacheT8 registered (docs/08)."""
    try:
        info = json.loads(urllib.request.urlopen(base + "/object_info", timeout=timeout).read())
        ok = "MiniMaxH3BlockCacheT8" in info
        return Check("t8_node", ok, "registered" if ok
                     else "missing - install T8mars node pack (draft tiers will 400)")
    except Exception as e:
        return Check("t8_node", False, str(e))


def check_weights(models_dir: str | Path) -> list[Check]:
    """Prefix-match the core weight set under a ComfyUI models directory."""
    root = Path(models_dir)
    checks: list[Check] = []
    for sub, prefixes in CORE_WEIGHTS.items():
        folder = root / sub
        present = {p.name.lower() for p in folder.glob("*.safetensors")} if folder.is_dir() else set()
        for prefix in prefixes:
            hit = any(name.startswith(prefix.lower()) or prefix.lower() in name for name in present)
            checks.append(Check(f"weights:{sub}/{prefix}", hit,
                                "found" if hit else f"not found under {folder}"))
    return checks

def parse_gpu_procs(csv_text: str) -> list[tuple[int, str, int]]:
    """解析 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 的输出。"""
    out: list[tuple[int, str, int]] = []
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        mib = parts[2].split()[0] if parts[2] else "0"
        try:
            out.append((int(parts[0]), parts[1], int(mib)))
        except ValueError:
            continue
    return out


def check_gpu_quiet(server_pid: int | None = None) -> Check:
    """除了 ComfyUI 自己，没有别的进程占着显存。

    这条对应手册 §4.5 那次真事故：**渲染期间有并发负载（分片下载 / 编译 / 另一个
    模型进程）会让采样出 NaN，成片全黑、音频 NaN、SaveVideo 抛 `returned 22`**——
    3/3 复现，而且 4 分钟 GPU 时间全白烧。事前唯一的判据就是"设备上还有谁"。
    """
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        procs = parse_gpu_procs(q.stdout)
    except Exception as e:
        return Check("gpu_quiet", True, f"nvidia-smi 不可用（{e}）")
    if not procs:
        return Check("gpu_quiet", True, "no process holds the GPU")
    others = [p for p in procs if not (server_pid and p[0] == server_pid)]
    if others:
        detail = "; ".join(f"pid{pid} {name} {mib}MiB" for pid, name, mib in others)
        return Check("gpu_quiet", False,
                     f"other processes hold the GPU: {detail} — 渲染期间并发负载会让采样出 NaN（§4.5）")
    total = sum(m for _, _, m in procs)
    return Check("gpu_quiet", True, f"only ComfyUI holds the GPU ({total} MiB)")



def check_lora_env(baseline_errors: int = 0) -> Check:
    """Heuristic for the Sep-20 quant+LoRA contamination: after installing any pip
    package, load your acceleration LoRA once and count 'ERROR lora' lines in the
    server log. A delta above baseline means the quantized-weight patch path is
    broken (see handbook docs/09 and upstream Comfy-Org/ComfyUI#16420) - the LoRA
    loads 'successfully' but its deltas silently never apply."""
    return Check(
        "lora_env",
        True,  # informational - the operator compares against baseline
        "post-pip-install check: load the accel LoRA once and diff 'ERROR lora' "
        f"count against baseline ({baseline_errors}); any increase = broken "
        "quant+LoRA patch path (docs/09, upstream #16420)",
    )

def check_bypass_path() -> Check:
    """The final tiers must apply the LoRA at run time (bypass), not merged.

    Merging a LoRA into the pruned w4a8 weights costs 29% wall clock for no
    quality gain (handbook 5.13) - and the failure is silent: the picture
    looks the same, it is just slower. The bypass node flips to merge mode
    when ``low_vram`` is True, so check the tier workflows themselves.
    """
    from .tiers import TIERS
    from .workflows import load

    problems = []
    for key, tier in TIERS.items():
        if tier.cached:
            continue                      # draft tiers: cache dominates, merge is fine
        try:
            graph = load(tier.workflow)
        except Exception as e:
            problems.append(f"{key}: {e}")
            continue
        merge_nodes = [n for n in graph.values()
                       if isinstance(n, dict) and n.get("class_type") == "LoraLoaderModelOnly"]
        if not merge_nodes:
            continue          # no merge-path LoRA (e.g. the PDD-Acc node applies its own)
        lora_nodes = [n for n in graph.values()
                      if isinstance(n, dict) and n.get("class_type") == "MiniMaxH3TurboLoRA"]
        if not lora_nodes:
            problems.append(f"{key}: merge-path LoRA with no bypass node (+29% time)")
        elif any(n["inputs"].get("low_vram") for n in lora_nodes):
            problems.append(f"{key}: low_vram=True means merge mode, +29% time")
    if problems:
        return Check("bypass_path", False, "; ".join(problems))
    return Check("bypass_path", True, "final tiers apply the LoRA at run time")



def _server_pid() -> int | None:
    """ComfyUI 服务自己的 pid（用来把它从"占显存的进程"里排除）。"""
    try:
        out = subprocess.run(["pgrep", "-f", "main.py --listen"], capture_output=True,
                             text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


def run_all(base: str = "http://127.0.0.1:8188", models_dir: str | Path | None = None) -> list[Check]:
    results = [check_comfyui(base)]
    if results[0].ok:
        results.append(check_queue_clean(base))
        results.append(check_t8_node(base))
        results.append(check_bypass_path())
        results.append(check_gpu_quiet(_server_pid()))
    if models_dir:
        results.extend(check_weights(models_dir))
    results.append(check_lora_env())
    return results


def report(base: str = "http://127.0.0.1:8188", models_dir: str | Path | None = None) -> bool:
    """Run all checks, print a report, return overall pass."""
    results = run_all(base, models_dir)
    ok_all = True
    for c in results:
        mark = "✓" if c.ok else "✗"
        if not c.ok:
            ok_all = False
        print(f"{mark} {c.name}: {c.detail}")
    print("=> PASS" if ok_all else "=> FAIL")
    return ok_all

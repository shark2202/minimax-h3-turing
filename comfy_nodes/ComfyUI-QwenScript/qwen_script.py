"""QwenScript -- write the H3 prompt (or a whole multi-shot 剧本) with a local
Qwen3.8 GGUF, as a node inside the graph.

The LLM is a separate llama-server process (repo: ~/ai-work/qwen3.8-2080ti-256k,
see docs/SMALL_MODELS.md). This node owns that process for the duration of one
generation, because 22 GB cannot hold a 9B target and the H3 pipeline at once:

  1. free ComfyUI's own model cache (the sampler loads right after this node)
  2. reuse a running server, or start one with CONFIG_PATH=<config>
  3. ask for the shot's English prompt (JSON contract in the repo's
     scripts/screenplay.system.md, so prompt edits need no code change)
  4. kill the server *only if this node started it*, then assert that the VRAM
     the sampler is about to need is actually free

One queue run is therefore "剧本 -> 出片" without either side OOM-ing. Output 0
is the prompt for the selected shot, ready to wire into MiniMaxH3ImageToVideo;
output 1 is the full screenplay, shown in the node.

Measured on the 22 GB 2080 Ti: H3 peak 18926 MiB (resident 17422), llama-server
6808 MiB, card total 22528 MiB -- so the two can never be resident together.

Local install: symlink this directory into ComfyUI/custom_nodes/ (see
minimax-h3-turing/scripts/install_qwen_node.local.sh) and restart ComfyUI.
Independent of ComfyUI at import time, so it can be smoke-tested with plain python.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "/home/a/ai-work/qwen3.8-2080ti-256k"


def _load_screenplay_module(repo: str):
    """Import <repo>/scripts/screenplay.py -- one implementation of the JSON
    handling, shared with the CLI tool."""
    path = Path(repo) / "scripts" / "screenplay.py"
    if not path.is_file():
        raise RuntimeError(f"{path} not found - check the engine_repo path")
    spec = importlib.util.spec_from_file_location("h3_screenplay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3_screenplay"] = mod
    spec.loader.exec_module(mod)
    return mod


def _server_alive(base: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _target_model(config_path: Path) -> str:
    """Pull TargetModel out of the launcher config, so we can kill exactly the
    server we started (same trick the launchers use: match, never guess)."""
    m = re.search(r"^TargetModel=(.*)$", config_path.read_text(encoding="utf-8"), re.M)
    if not m:
        raise RuntimeError(f"no TargetModel in {config_path}")
    return os.path.expanduser(os.path.expandvars(m.group(1).strip().strip("'\"")))


def _kill_server(model_path: str) -> int:
    """Kill llama-server processes serving this model file. Returns how many."""
    killed = 0
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "llama-server" in cmdline and model_path in cmdline:
            try:
                os.kill(int(proc.name), 15)
                killed += 1
            except OSError:
                pass
    if killed:
        time.sleep(1.5)  # let it release VRAM before the sampler allocates
    return killed


def _free_vram_mib() -> int | None:
    """Device-wide free VRAM, measured (torch first, nvidia-smi as fallback).

    Device-wide on purpose: what matters is memory *another* process holds (the
    LLM, or a ComfyUI cache from the previous queue run).
    """
    try:
        import torch
        free, _total = torch.cuda.mem_get_info()
        return int(free // (1024 * 1024))
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _kill_server_on_port(base: str) -> int:
    """Stop llama-server processes bound to the port in `base` (a foreign server)."""
    m = re.search(r":(\d+)", base)
    port = m.group(1) if m else "8080"
    killed = 0
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "llama-server" in cmdline and re.search(rf"--port\s+{port}\b", cmdline):
            try:
                os.kill(int(proc.name), 15)
                killed += 1
            except OSError:
                pass
    if killed:
        time.sleep(1.5)
    return killed


def _free_comfy_cache() -> str:
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
        return "freed"
    except Exception as e:  # never fail a render over this
        return f"skipped ({e})"


class QwenScript:
    """Write the H3 prompt with the local Qwen3.8 server."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brief": ("STRING", {"multiline": True, "default":
                    "竹林剑客，黑色电影风格，复仇；冷调晨雾，手持镜头"}),
                "shot": ("INT", {"default": 1, "min": 1, "max": 24,
                                 "tooltip": "which shot's prompt to output (1-based)"}),
                "shots": ("INT", {"default": 1, "min": 1, "max": 24,
                                  "tooltip": "how many shots the model should write"}),
                "style": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "extra instructions: tone, lens, era, pacing"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05}),
                "thinking": (["none", "low", "medium", "high"], {"default": "low"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "server": ("STRING", {"default": "http://127.0.0.1:8080/v1"}),
                "model": ("STRING", {"default": "qwen3.8-9b-heretic"}),
                "engine_repo": ("STRING", {"default": DEFAULT_REPO}),
                "config": ("STRING", {"default": "config.small.sh"}),
                "script_out": ("STRING", {"default": "", "tooltip":
                    "optional path prefix; writes .md + .shots.json"}),
                "reuse_script": ("BOOLEAN", {"default": True, "tooltip":
                    "reuse an existing .shots.json instead of asking the model again"}),
                "autostart": ("BOOLEAN", {"default": True, "tooltip":
                    "start llama-server here if the endpoint is not answering"}),
                "free_comfy_cache": ("BOOLEAN", {"default": True, "tooltip":
                    "unload ComfyUI's models first: 22 GB cannot hold both"}),
                "min_free_vram_mib": ("INT", {"default": 19700, "min": 0, "max": 262144,
                    "tooltip": "abort before the sampler if less than this is free; measured on this card: "
                               "H3 peak 18926 MiB (resident 17422), LLM 6808, total 22528"}),
                "foreign_server": (["fail", "stop", "ignore"], {"default": "fail",
                    "tooltip": "a llama-server this node did not start is holding VRAM: "
                               "fail fast with the numbers / stop it / ignore and hope"}),
                "timeout_s": ("INT", {"default": 900, "min": 30, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "screenplay")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = True
    DESCRIPTION = ("Write the H3 prompt / a multi-shot 剧本 with the local Qwen3.8 "
                   "GGUF server; starts and stops that server around the call.")

    def run(self, brief, shot, shots, style, temperature, thinking, seed, server,
            model, engine_repo, config, script_out, reuse_script, autostart,
            free_comfy_cache, min_free_vram_mib, foreign_server, timeout_s):
        t0 = time.time()
        base = server.rstrip("/")
        repo = os.path.expanduser(engine_repo)
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = Path(repo) / config
        script_mod = _load_screenplay_module(repo)
        shots_file = Path(os.path.expanduser(script_out) + ".shots.json") if script_out else None

        if free_comfy_cache:
            print(f"[QwenScript] comfy cache: {_free_comfy_cache()}", flush=True)

        # A cached .shots.json needs no model at all: check it before touching VRAM.
        script = None
        if reuse_script and shots_file and shots_file.is_file():
            cached = json.loads(shots_file.read_text(encoding="utf-8"))
            if len(cached.get("shots", [])) >= shot:
                script = cached
                print(f"[QwenScript] reusing {shots_file} (no model needed)", flush=True)

        started_by_us = False
        if script is None:
            if _server_alive(base):
                print(f"[QwenScript] using the llama-server already running at {base}", flush=True)
            else:
                if not autostart:
                    raise RuntimeError(
                        f"no llama-server at {base} and autostart is off - start it with "
                        f"CONFIG_PATH={config_path.name} {repo}/scripts/start-llama.sh")
                if not (Path(repo) / "scripts" / "start-llama.sh").is_file():
                    raise RuntimeError(f"{repo}/scripts/start-llama.sh not found")
                env = dict(os.environ, CONFIG_PATH=str(config_path))
                print(f"[QwenScript] starting llama-server: CONFIG_PATH={config_path.name}", flush=True)
                subprocess.Popen(["bash", "scripts/start-llama.sh"], cwd=repo, env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
                started_by_us = True
                deadline = time.time() + 180
                while time.time() < deadline and not _server_alive(base, 2.0):
                    time.sleep(2.0)
                if not _server_alive(base):
                    raise RuntimeError(f"llama-server did not come up; see {repo}/logs/server-*.err.log")
            try:
                system = (Path(repo) / "scripts" / "screenplay.system.md").read_text(encoding="utf-8")
                user = f"{brief}\n\n镜头数：{shots}。"
                if style.strip():
                    user += f"\n额外要求：{style.strip()}"
                payload = {
                    "model": model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "temperature": temperature,
                    "seed": seed,
                    "response_format": {"type": "json_object"},
                }
                if thinking != "medium":
                    # chat_template_kwargs 才是真正生效的开关：OpenAI 的 reasoning_effort 字段本引擎不映射（实测 no-op）
                    payload["chat_template_kwargs"] = {"reasoning_effort": thinking}
                t1 = time.time()
                try:
                    data = script_mod.post_json(base + "/chat/completions", payload, timeout_s)
                except urllib.error.HTTPError as e:
                    raise RuntimeError(f"HTTP {e.code} from {base}: "
                                       f"{e.read().decode('utf-8', 'replace')[:300]}") from None
                wall = time.time() - t1
                usage = data.get("usage") or {}
                ntokens = usage.get("completion_tokens")
                print(f"[QwenScript] {data.get('model')}: {ntokens} tokens in {wall:.1f}s "
                      f"({ntokens / wall if ntokens else 0:.1f} tok/s)", flush=True)
                text = (data["choices"][0]["message"].get("content") or "")
                try:
                    script = script_mod.extract_json(text)
                    assert script.get("shots"), "empty shot list"
                except Exception as e:
                    if script_out:
                        raw = Path(os.path.expanduser(script_out) + ".raw.txt")
                        raw.write_text(text, encoding="utf-8")
                        raise RuntimeError(f"{e}; raw reply kept at {raw}") from None
                    raise RuntimeError(f"{e}; reply was: {text[:300]}") from None
                if shots_file:
                    shots_file.parent.mkdir(parents=True, exist_ok=True)
                    shots_file.write_text(json.dumps(
                        {"model": model, "brief": brief, "shots": script["shots"]},
                        ensure_ascii=False, indent=2), encoding="utf-8")
            finally:
                if started_by_us:
                    print(f"[QwenScript] stopping the server it started "
                          f"({_kill_server(_target_model(config_path))} process(es))", flush=True)

        # --- VRAM precondition, measured, before the sampler is allowed to load H3 ---
        # The card cannot hold the LLM and the H3 pipeline at once (measured: 6.8 GB +
        # 18.5 GB peak on a 22.5 GB card), so "free VRAM right now" is the one number
        # that decides whether the render survives. Fail here, cheaply, instead of
        # letting the sampler die five minutes in.
        free_mib = _free_vram_mib()
        if free_mib is not None and free_mib < min_free_vram_mib:
            detail = (f"only {free_mib} MiB VRAM free but H3 needs ~{min_free_vram_mib} MiB "
                      f"(measured: LLM 6.8 GB + H3 peak 18.5 GB on a 22.5 GB card)")
            if foreign_server == "stop" and not started_by_us:
                print(f"[QwenScript] {detail} -> stopping the llama-server on {base}", flush=True)
                _kill_server_on_port(base)
                free_mib = _free_vram_mib()
                detail = f"still only {free_mib} MiB VRAM free after stopping it"
            if free_mib < min_free_vram_mib:
                if foreign_server == "ignore":
                    print(f"[QwenScript] WARNING: {detail}; continuing anyway", flush=True)
                else:
                    raise RuntimeError(
                        f"{detail}. Stop the other GPU user (or set foreign_server=stop "
                        f"to let this node do it), or raise min_free_vram_mib if the card "
                        f"is bigger than this one.")
        elif free_mib is not None:
            print(f"[QwenScript] free VRAM before the sampler: {free_mib} MiB "
                  f"(threshold {min_free_vram_mib})", flush=True)

        if shot > len(script["shots"]):
            raise RuntimeError(f"shot {shot} requested but the screenplay has "
                               f"{len(script['shots'])} shot(s)")
        sel = script["shots"][shot - 1]

        md_lines = [f"# {script.get('title', '')}", "",
                    f"**logline**: {script.get('logline', '')}", ""]
        for s in script["shots"]:
            md_lines += [f"## {s.get('index', '')}. {s.get('title', '')} "
                         f"({s.get('frames', 124)} 帧)", "",
                         f"- 画面：{s.get('scene_zh', '')}",
                         f"- 首帧：{s.get('first_frame_hint') or '不需要（文生视频）'}", "",
                         "```text", s.get("prompt_en", ""), "```", ""]
        screenplay_md = "\n".join(md_lines)
        if script_out:
            Path(os.path.expanduser(script_out) + ".md").write_text(screenplay_md, encoding="utf-8")
        print(f"[QwenScript] shot {shot}/{len(script['shots'])} in {time.time() - t0:.1f}s: "
              f"{sel.get('prompt_en', '')[:120]}", flush=True)
        return {"ui": {"text": (screenplay_md,)}, "result": (sel.get("prompt_en", ""), screenplay_md)}


NODE_CLASS_MAPPINGS = {"QwenScript": QwenScript}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenScript": "Qwen3.8 Script (local LLM)"}
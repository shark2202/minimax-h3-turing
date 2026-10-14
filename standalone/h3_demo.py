#!/usr/bin/env python3
"""Minimal MiniMax H3 text/image -> video demo, WITHOUT running the ComfyUI
server - the local ComfyUI checkout is embedded as a Python library.

  t2v: python h3_demo.py --comfy-dir /path/to/ComfyUI --prompt "..."
  i2v: python h3_demo.py --comfy-dir /path/to/ComfyUI --image first.png --prompt "..."

Why ComfyUI must be present as code (and why that is still "no ComfyUI" in
the operational sense): the H3 h3lite W4A8 weights load only through
ComfyUI's comfy-kitchen dequant backend - no other loader exists, and there
is no Diffusers pipeline for H3. So this script imports ComfyUI's own node
implementations directly: no HTTP server, no web UI, no queue, no polling.
It executes the repo's field-tested workflow graphs node by node in-process
and copies the rendered mp4 out. Same weights as the server route (the
docs/05 download list, already under ComfyUI's models/ tree) - ZERO new
model downloads; same expected timings (~280s t2v / ~420s i2v warm).

Node-embedding pattern follows ComfyUI-to-Python-Extension's generated
runtime: argv -> enable_args_parsing -> cuda_malloc -> torch -> node
registry init -> direct node-class calls in dependency order.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"
T2V_WORKFLOW = "h3_w4a8_t2v_compat_api.json"
I2V_WORKFLOW = "h3_w4a8_i2v_compat_api.json"

PROMPT_PLACEHOLDER = "__H3_PROMPT__"
IMAGE_PLACEHOLDER = "__H3_FIRST_FRAME__"

DEFAULT_PROMPT = ("cinematic wide shot, a lone swordsman walking through a "
                  "misty bamboo forest at dawn, natural ambient audio")

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

# The anti-TDR launch discipline from scripts/h3_launch.example.sh, applied
# by faking argv before ComfyUI parses it (TDR black-screens / the OOM'd
# read_file_slice path - see handbook docs/04 and docs/06).
VRAM_FLAGS = ["--reserve-vram", "2.5", "--vram-headroom", "0.5",
              "--disable-pinned-memory"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy-dir", default=None,
                   help="ComfyUI checkout path (default: $COMFYUI_PATH)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--image", default=None,
                   help="first-frame image -> switches to i2v")
    p.add_argument("--seed", type=int, default=3013)
    p.add_argument("--out", default="h3_demo_output", help="directory for the mp4")
    p.add_argument("--workflow", default=None,
                   help="workflow JSON name override (default: compat t2v/i2v by route)")
    p.add_argument("--skip-custom-nodes", action="store_true",
                   help="don't load custom_nodes/ (compat workflows don't need it)")
    return p.parse_args(argv)


# -- pure graph helpers (testable without ComfyUI/torch) -------------------

def load_workflow(name: str) -> dict:
    path = Path(name)
    if not path.is_file():
        path = WORKFLOWS_DIR / (name if name.endswith(".json") else name + ".json")
    return json.loads(path.read_text(encoding="utf-8"))


def patch_graph(graph: dict, prompt: str, seed: int, image_name: str | None) -> None:
    """Fill the placeholders in place. i2v graphs need image_name, t2v must not."""
    load_image_nodes = [n for n in graph.values()
                        if isinstance(n, dict) and n.get("class_type") == "LoadImage"]
    wants_image = bool(load_image_nodes)
    if wants_image != (image_name is not None):
        route = "i2v" if wants_image else "t2v"
        raise SystemExit(f"route mismatch: this is a {route} workflow - "
                         + ("pass --image PATH" if wants_image else "drop --image"))
    hits = 0
    for node in graph.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not inputs:
            continue
        if inputs.get("prompt") == PROMPT_PLACEHOLDER:
            inputs["prompt"] = prompt
            hits += 1
        if "image" in inputs and inputs["image"] == IMAGE_PLACEHOLDER:
            inputs["image"] = image_name
        if "noise_seed" in inputs:
            inputs["noise_seed"] = int(seed)
    if hits == 0:
        raise SystemExit(f"no {PROMPT_PLACEHOLDER!r} in workflow - pick a bundled "
                         "placeholder workflow or set the prompt node directly")


def execute_graph(graph: dict, mappings: dict | None = None, log=print) -> dict:
    """Run an API-format graph node by node in dependency order.

    This is the embedding pattern of ComfyUI-to-Python-Extension's generated
    runtime, generalized: instantiate each class from NODE_CLASS_MAPPINGS and
    call its FUNCTION with exactly the graph's inputs (the bundled graphs
    already carry every input explicitly - the v3-API discipline).
    """
    import contextlib

    if mappings is None:
        import nodes as comfy_nodes
        mappings = comfy_nodes.NODE_CLASS_MAPPINGS
    try:
        import torch
        runner_ctx = torch.inference_mode()
    except ImportError:                      # allows GPU-free graph unit tests
        runner_ctx = contextlib.nullcontext()

    def resolve(value):
        if (isinstance(value, list) and len(value) == 2
                and isinstance(value[0], str) and isinstance(value[1], int)):
            out = results[value[0]]
            try:
                return out[value[1]]
            except (KeyError, TypeError):
                return out["result"][value[1]]     # some nodes return dicts
        return value

    results: dict = {}
    pending = dict(graph)
    with runner_ctx:
        while pending:
            ready = [nid for nid, node in pending.items()
                     if all(dep in results
                            for dep in (v[0] for v in node.get("inputs", {}).values()
                                        if isinstance(v, list) and len(v) == 2
                                        and isinstance(v[0], str)))]
            if not ready:
                raise RuntimeError(f"graph stuck (cycle or missing node): {sorted(pending)}")
            for nid in ready:
                node = pending.pop(nid)
                cls = mappings[node["class_type"]]
                kwargs = {k: resolve(v) for k, v in node.get("inputs", {}).items()}
                out = getattr(cls(), cls.FUNCTION)(**kwargs)
                results[nid] = out
                log(f"  [{nid}] {node['class_type']} done")
    return results


def collect_new_files(root: str | Path, since: float,
                      exts: set[str] = VIDEO_EXTS) -> list[Path]:
    """Files written under ``root`` after ``since`` with a video extension."""
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in exts
                  and p.stat().st_mtime > since)


# -- ComfyUI embedding (heavy imports stay inside) --------------------------

def resolve_comfy_dir(cli_value: str | None) -> Path:
    import os
    cand = cli_value or os.environ.get("COMFYUI_PATH")
    if not cand:
        raise SystemExit("pass --comfy-dir or set $COMFYUI_PATH to the ComfyUI checkout")
    path = Path(cand).expanduser().resolve()
    if not (path / "nodes.py").is_file():
        raise SystemExit(f"{path} does not look like a ComfyUI checkout (no nodes.py)")
    return path


def bootstrap_comfyui(comfy_dir: Path, flags: list[str], custom_nodes: bool = True):
    """Import ComfyUI the way its own generated-script runtime does."""
    import asyncio

    sys.path.insert(0, str(comfy_dir))
    sys.argv = [sys.argv[0], *flags]        # must precede enable_args_parsing
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args         # noqa: F401  (parses our flags)
    import cuda_malloc                      # noqa: F401  (allocator / cu130 path)
    import torch                            # noqa: F401

    # optional extra_model_paths.yaml next to the checkout
    try:
        from main import load_extra_path_config
        yaml = comfy_dir / "extra_model_paths.yaml"
        if yaml.is_file():
            load_extra_path_config(str(yaml))
    except ImportError:
        pass

    # node registry: core + comfy_extras + custom nodes (mirrors the
    # import_custom_nodes() pattern; some custom nodes want the singletons)
    import execution
    import server
    from nodes import init_extra_nodes

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    execution.PromptQueue(server.PromptServer(loop))
    res = init_extra_nodes(init_custom_nodes=custom_nodes)
    if inspect.iscoroutine(res):            # newer ComfyUI made it async
        asyncio.run(res)


def stage_image(image_path: str) -> str:
    """Copy the first frame into ComfyUI's input dir; returns its filename."""
    import folder_paths
    src = Path(image_path).expanduser()
    if not src.is_file():
        raise SystemExit(f"image not found: {src}")
    name = f"h3demo_{uuid.uuid4().hex[:8]}{src.suffix.lower()}"
    shutil.copy(src, Path(folder_paths.get_input_directory()) / name)
    return name


def teardown():
    """Best-effort VRAM cleanup so repeated in-process runs stay healthy."""
    import gc
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()


def main(argv=None):
    args = parse_args(argv)
    comfy_dir = resolve_comfy_dir(args.comfy_dir)
    workflow = args.workflow or (I2V_WORKFLOW if args.image else T2V_WORKFLOW)

    graph = load_workflow(workflow)
    patch_graph(graph, args.prompt, args.seed, image_name="pending" if args.image else None)
    print(f"workflow: {workflow} | seed {args.seed} | "
          f"route: {'i2v' if args.image else 't2v'}")

    t0 = time.time()
    print(f"bootstrapping ComfyUI from {comfy_dir} (no server, no UI)...")
    bootstrap_comfyui(comfy_dir, VRAM_FLAGS, custom_nodes=not args.skip_custom_nodes)

    if args.image:
        name = stage_image(args.image)
        for node in graph.values():         # swap in the real staged filename
            if isinstance(node, dict) and "image" in node.get("inputs", {}):
                node["inputs"]["image"] = name
        print(f"first frame staged: {name}")

    print("executing graph node by node (4-step Turbo; ~280s t2v / ~420s i2v warm)...")
    try:
        import folder_paths
        out_root = folder_paths.get_output_directory()
        execute_graph(graph)
        videos = collect_new_files(out_root, t0)
        if not videos:
            raise SystemExit("render finished but no new video under "
                             f"{out_root} - check the SaveVideo node above")
        dest_dir = Path(args.out)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for v in videos:
            shutil.copy(v, dest_dir / v.name)
            print(f"saved: {(dest_dir / v.name).resolve()}")
        print(f"done in {time.time() - t0:.0f}s (seed {args.seed})")
    finally:
        teardown()


if __name__ == "__main__":
    main()

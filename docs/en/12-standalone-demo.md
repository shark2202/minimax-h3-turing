# 12 · The H3 Demo Without a ComfyUI Server: Embedded-Library Mode

> [Demo 11](11-demo.md) uses the ComfyUI server as its runtime (launch, HTTP, polling, download). This chapter drops the server: **one Python script process** goes from text/image to mp4, with MiniMax H3 itself and zero new weight downloads. For embedding into your own pipelines, headless batch rendering, or anyone who doesn't want a resident service.

## The shape: ComfyUI as a library, not a server

The local H3 route is h3lite **W4A8 quantized weights**, and the only loader is ComfyUI's comfy-kitchen dequantization backend (the cu130 bonus from [01](01-hardware-limits.md) comes exactly from it); there is no Diffusers pipeline. So "without ComfyUI" in the H3 context means: **no server, no UI, no HTTP/queue** — import the local ComfyUI checkout as a Python library and execute node by node, in dependency order, with its own node implementations.

| | Server mode (docs/11) | Embedded mode (this chapter) |
|---|---|---|
| Processes | resident ComfyUI + client script | single process, exits when done |
| Communication | HTTP /prompt + polling | in-process calls |
| Workflows | the same JSON kit | **the same JSON kit** |
| Weights | docs/05 list (models/ tree) | **the same set, zero new downloads** |
| Expected time | t2v ~280s / i2v ~420s (warm) | same engine, same ballpark |
| Extra | resume, multi-job queue | embeds directly into Python pipeline code |

## Usage

```bash
cd standalone
python h3_demo.py --comfy-dir /path/to/ComfyUI \
    --prompt "a lone swordsman in a misty bamboo forest at dawn"

# image -> video: the first frame is copied into ComfyUI's input dir automatically
python h3_demo.py --comfy-dir /path/to/ComfyUI \
    --image first_frame.png --prompt "she slowly turns to camera"
```

Prerequisites: a ComfyUI v0.33.1+ checkout (with its Python deps) and the H3 weights from docs/05 already under its `models/` tree — fully shared with the server route; install once. `--comfy-dir` can also be provided via the `COMFYUI_PATH` environment variable.

| Flag | Default | Notes |
|---|---|---|
| `--comfy-dir` / `$COMFYUI_PATH` | required | ComfyUI checkout (validated by nodes.py) |
| `--prompt` / `--image` / `--seed` | 3013 | same semantics as demo 11; `--image` switches to the i2v workflow |
| `--workflow` | compat t2v/i2v | any of the six-tier kit (e.g. t8draft, needs the T8 node pack) |
| `--out` | h3_demo_output | where the mp4 is copied |
| `--skip-custom-nodes` | off | compat tiers don't need custom_nodes; loads faster |

## What it does in-process

| Step | Handbook knowledge it applies |
|---|---|
| injects `--reserve-vram 2.5 --vram-headroom 0.5 --disable-pinned-memory` into `sys.argv` before `enable_args_parsing()` | the anti-TDR launch discipline (scripts/h3_launch) applies verbatim; TDR black-screens / the read_file_slice OOM path |
| `import cuda_malloc` + torch before node init | the cu130 dequant path activates in embedded mode too |
| `PromptServer(loop) + PromptQueue + init_extra_nodes()` | the canonical embedding pattern (mirrors the ComfyUI-to-Python-Extension runtime); custom nodes keep working |
| topological node-by-node execution, edges resolved to upstream outputs | the graph IS the six-tier JSON — **field-tested parameters unchanged**, and the every-input-explicit v3-API discipline sidesteps HTTP 400 entirely |
| snapshots the output dir before, collects new video files after | no server history to query; equivalent of demo 11's output collection |
| `soft_empty_cache()` before exit | repeated in-process renders don't accumulate VRAM |

## Boundaries

- **One render per run**: the model cache is process-level; the demo is a single-shot runner. Batch renders loop `execute_graph` or return to the server-mode queue.
- **Errors raise immediately**: no server log buffer - node exceptions surface with original tracebacks, earlier than HTTP mode (a missing v3-node input is a direct TypeError, not a 400).
- **Same-seed reproducible** (compat tiers): same discipline as server mode; T8/PDD tiers still follow "drafts only, never finals" ([08](08-t8-blockcache-4step.md)).
- **Run doctor after installing pip packages**: the silent quant+LoRA breakage ([09](09-pdd-backport.md)) is orthogonal to the runtime shape; embedded mode is equally exposed.

## Appendix: the absolutely-no-ComfyUI alternative (Wan 2.1)

If the box shouldn't even hold ComfyUI code: `standalone/demo.py` is the pure-Diffusers variant (Wan 2.1, t2v 1.3B / i2v 14B 480P; download and cache list in [standalone/README](../../standalone/README.md)). **Note it does not generate H3** - that is a model-level trade-off: H3 cannot leave the ComfyUI ecosystem; this appendix only answers the "zero ComfyUI" extreme.

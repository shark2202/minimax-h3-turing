# 05 · The compat Workflows, Explained

## Model Download List (read this first)

The file names referenced by the workflows come from h3lite's W4A8 component Set A (a cloud-drive distribution); every measurement in this repo is based on **Route B (the HF-direct equivalent)** — the right column below is the recommended download source:

| Name referenced by workflows | Recommended actual download | Source | Target directory |
|---|---|---|---|
| minimax_h3_fl2va_pruned_w4a8_mixed_ax1y2jp.safetensors (12.5G) | minimax_h3_fl2va_pruned_w4a8_mixed.safetensors | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental) | models/diffusion_models/ |
| qwen3vl_4b_int4_convrot.safetensors (2.8G) | qwen3vl_4b_fp8_scaled.safetensors (5.2G) | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) → text_encoders/ | models/text_encoders/ |
| mmh3-4b-ClipProj-celeb-mlp.safetensors (304M) | same name, direct | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) | models/clip_projections/ |
| minimax_h3_video_vae_fp16 / audio_vae_fp32 | same name, direct | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) → vae/ | models/vae/ |
| Turbo 4-step LoRA (fl2v/ref2v) | same name, direct | same repo → loras/ | models/loras/ |
| PDD Acc-8Step LoRA (FL2VA = t2v ultra tier / Ref2VA = i2v ultra tier) | same name, direct | Kijai repo → loras/ | models/loras/ |

- **Renaming rule**: when using a substitute, edit the loader node's file name in the workflow JSON to match your actual download (or rename the files); the two W4A8 variants differ by ~26 KB (a quantization-config tweak) and produce identical output in our tests
- **If you insist on the ax1y2jp-named originals**: h3lite Set A, Baidu Pan `pan.baidu.com/s/1x5GGuJv0h8chApgVoDgIaQ` (code `1hjx`)
- **China mirror**: if huggingface.co won't connect, swap the domain for hf-mirror.com — paths unchanged; multi-threaded chunked downloads recommended for the big files

### Copy-paste Download (commands)

The table above is for reading; this block is for pasting. Point `MODELS` at your ComfyUI `models/` tree. In ②④⑤⑥ the repo subfolders (`text_encoders/`, `vae/`, `loras/`) share ComfyUI's directory names, so `--local-dir "$MODELS"` drops the files straight into place; ①③ live at the repo root, hence the explicit target subdirectory (new-style CLI equivalent: `hf download …`, same arguments):

```bash
pip install -U huggingface_hub              # skip if already installed
MODELS=/path/to/ComfyUI/models              # ← change to your path
# export HF_ENDPOINT=https://hf-mirror.com  # China mirror: uncomment when direct access fails

# ① Main W4A8 DiT (12.5G) → diffusion_models/
huggingface-cli download Kijai/MiniMax-H3-experimental \
    minimax_h3_fl2va_pruned_w4a8_mixed.safetensors \
    --local-dir "$MODELS/diffusion_models"

# ② Text encoder qwen3vl-4b fp8 (5.2G) → text_encoders/
huggingface-cli download Comfy-Org/Krea-2 \
    text_encoders/qwen3vl_4b_fp8_scaled.safetensors \
    --local-dir "$MODELS"

# ③ ClipProj (304M) → clip_projections/
huggingface-cli download NicoLab28/ClipProj-MiniMax-H3 \
    mmh3-4b-ClipProj-celeb-mlp.safetensors \
    --local-dir "$MODELS/clip_projections"

# ④ Dual VAE (video fp16 + audio fp32) → vae/
huggingface-cli download Comfy-Org/MiniMax-H3 \
    vae/minimax_h3_video_vae_fp16.safetensors \
    vae/minimax_h3_audio_vae_fp32.safetensors \
    --local-dir "$MODELS"

# ⑤ Both Turbo 4-step LoRAs (compat tier uses v0.1, t8draft tier uses v1.0_768p) → loras/
huggingface-cli download Comfy-Org/MiniMax-H3 \
    loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors \
    loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors \
    --local-dir "$MODELS"

# ⑥ PDD 8-step LoRAs (FL2VA = t2v / Ref2VA = i2v; only for the PDD tiers) → loras/
huggingface-cli download Kijai/MiniMax-H3-experimental \
    loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors \
    loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors \
    --local-dir "$MODELS"
```

Verify afterwards: ①–⑤ total ~21G; the compat workflows reference the Set-A names (ax1y2jp / int4_convrot) in their loader nodes — apply the **renaming rule** above and they run.

```bash
ls -lh "$MODELS"/{diffusion_models,text_encoders,clip_projections,vae,loras}
h3turing doctor --models-dir "$MODELS"     # usage in python/README
```

Without the CLI, paste the direct links below into a browser / `wget -c` / any downloader, then move the files into the matching directories (mirror = domain swap to hf-mirror.com, paths unchanged; `wget -c` resumes by itself and survives where aria2 doesn't — see [FAQ #5](06-faq.md)):

```text
https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3/resolve/main/mmh3-4b-ClipProj-celeb-mlp.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors
https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors
https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors
```

## Why "compat"

H3's official full workflows reference the T8 BlockCache node and full-precision model names — a **dead diagram** on a Turing card without T8 installed (missing nodes, mismatched models). The compat versions are community-maintained degradations:

- Acceleration nodes that can't run on sm_75 removed
- Model file names aligned to the W4A8 mixed weights actually in use
- The 4-step Turbo route + native audio preserved

compat + W4A8 measured as the optimum on this card (see [01](01-hardware-limits.md)) — there is no "loss" from not upgrading to the full workflows.

## Files

> Full tier matrix, three-step import, and pitfall notes: [workflows/README](../workflows/README.md).

| File | Purpose |
|---|---|
| [h3_w4a8_t2v_compat_api.json](../workflows/h3_w4a8_t2v_compat_api.json) | t2v · final-shot tier (280 s/clip, reproducible) |
| [h3_w4a8_t2v_t8draft_api.json](../workflows/h3_w4a8_t2v_t8draft_api.json) | t2v · fast-draft tier (160 s/clip, −43%, not same-seed reproducible) |
| [h3_w4a8_i2v_compat_api.json](../workflows/h3_w4a8_i2v_compat_api.json) | i2v (needs a first frame) |

## Import & Replace

1. Drag the JSON into the ComfyUI UI, or drop it in `user/default/workflows/`
2. Check the model loader nodes' file names against your local `models/` (W4A8 weight file names differ between download eras)
3. **i2v**: point the first-frame loader node at your own image (this repo ships no sample image)
4. Defaults: 640×352 · 124 frames · 4 steps · native audio

## Production Parameter Advice

- Short clips (5–6 s): use defaults, with `--reserve-vram 2.5` (see [04](04-community-tips.md))
- For 1080p deliverables: keep generating at 640×352 → extract frames → RTX VSR upscale. **Do not** raise the workflow resolution (VRAM and time both blow up)
- Write prompts in three beats: subject → action/camera → environment & light (see h3lite's official methodology)

## New-ComfyUI UI Exports and Subgraph Workflows

On the master environment (the prerequisite for the PDD tiers), the newer frontend exports **subgraph nodes** inside UI-format workflows. Such JSON runs fine when dragged into the UI, but **submitting it to `/prompt` gets rejected** — subgraphs are a frontend concept; the API layer only understands a flattened node graph.

Symptoms and handling:

- **Symptoms**: `/prompt` complains about missing nodes or broken links while the UI runs the same workflow fine; the JSON shows `type` fields with subgraph namespaces, link ids as UUIDs, or virtual nodes at `inputs=-10 / outputs=-20`
- **Fix**: flatten before submitting. The mechanics: instantiate the subgraph's inner nodes → bridge input slots (including fan-out) → resolve UUID link chains transitively → skip frontend-only nodes like `MarkdownNote` → auto-generate Primitive nodes carrying defaults for pass-through slots; the widget-bound link marker is `"widget" in t_ins[l[4]]` (newer exports list widget slots inside node `inputs`)
- **Ready-made implementation**: [ComfyAgent](https://github.com/IvenKooLab/comfy-agent) v1.5.2, `server.py: convert_ui_to_api / _flatten_subgraphs` (open source, MIT; 5 unit tests; proven on a 102-node nested graph with zero loss)
- The six workflows in this repo are **native API format** and unaffected — this only matters when you save/derive your own workflows from the UI

## Common Import Failures

| Symptom | Cause |
|---|---|
| Red missing-node badges | H3 support node packs not installed (KJNodes etc.), or ComfyUI too old |
| Model loader node errors | Weight file name mismatch — double-click the node and select your local file |
| Color tearing in output | Wrong weights (W4A4) — switch to W4A8 mixed |

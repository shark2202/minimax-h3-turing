#!/usr/bin/env bash
# Install/refresh the QwenScript node and its workflow inside ComfyUI (idempotent).
# The repo is the single source of truth: ComfyUI gets a symlink, so an edit here
# is what runs after the next ComfyUI restart. Safe to re-run.
#
#   bash scripts/install_qwen_node.local.sh
#   COMFY_DIR=/path/to/ComfyUI bash scripts/install_qwen_node.local.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
COMFY="${COMFY_DIR:-/home/a/ai-work/ComfyUI}"
NODE_SRC="$REPO/comfy_nodes/ComfyUI-QwenScript"
NODE_DST="$COMFY/custom_nodes/ComfyUI-QwenScript"
WF_SRC="$REPO/workflows/h3_w4a8_t2v_qwenscript_api.json"
WF_DST="$COMFY/user/default/workflows"

[ -d "$COMFY" ] || { echo "no ComfyUI at $COMFY (set COMFY_DIR=)" >&2; exit 1; }
[ -f "$NODE_SRC/qwen_script.py" ] || { echo "missing $NODE_SRC/qwen_script.py" >&2; exit 1; }

# A real directory here means an older copy that would shadow the repo - refuse it.
if [ -e "$NODE_DST" ] && [ ! -L "$NODE_DST" ]; then
    echo "$NODE_DST is a real directory, not a symlink." >&2
    echo "Move it aside first (it may hold local edits): mv $NODE_DST ${NODE_DST}.bak" >&2
    exit 1
fi

ln -sfn "$NODE_SRC" "$NODE_DST"
mkdir -p "$WF_DST"
cp -f "$WF_SRC" "$WF_DST/$(basename "$WF_SRC")"

echo "node -> $NODE_DST -> $(readlink -f "$NODE_DST")"
echo "flow -> $WF_DST/$(basename "$WF_SRC")"
echo
echo "next: restart ComfyUI, then"
echo "  curl -s 127.0.0.1:8188/object_info/QwenScript | head -c 60"
echo "weights (not in git): /home/a/ai-work/models/llm/Qwen3.8-9B-heretic/*.gguf"
echo "  sha256 + fetch command: qwen3.8-2080ti-256k/docs/SMALL_MODELS.md"

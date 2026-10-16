#!/usr/bin/env bash
# 本机落地的启动脚本（scripts/h3_launch.example.sh 的实例化版）。
# 2026-10-15 本机实测两条，改环境前先读：
#
# 1) TORCH_DISABLE_NATIVE_JIT=1 —— 必须。
#    torch 2.14 把 outer-product 形状的 bmm 路由到 triton JIT 实现，
#    本机没装 python3.10-dev（/usr/include/python3.10/Python.h 不存在），
#    triton 现场编译 cuda_utils 失败 → 文本编码器第一个 op 就崩（2.4s 报错）。
#    装了 python3.10-dev 之后可以去掉（triton 走 JIT；Turing 上本路线无收益）。
#
# 2) 默认 --listen 0.0.0.0 —— 局域网 Mac 要访问（ComfyUI **无鉴权**，
#    0.0.0.0 = 同网段任何人可提交任务、读 output）。想收紧就覆盖：
#      H3_LISTEN=100.69.133.37 只走 tailscale
#      H3_LISTEN=127.0.0.1     只本机
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/home/a/ai-work/ComfyUI}"
PY="${PY:-$COMFY_DIR/.venv/bin/python}"
LISTEN="${H3_LISTEN:-0.0.0.0}"
PORT="${H3_PORT:-8188}"

export TORCH_DISABLE_NATIVE_JIT=1
cd "$COMFY_DIR"
"$PY" main.py \
  --listen "$LISTEN" --port "$PORT" \
  --reserve-vram 2.5 --vram-headroom 0.5 \
  --disable-pinned-memory "$@" 2>&1 | tee -a "${H3_LOG:-$COMFY_DIR/h3_server.log}"
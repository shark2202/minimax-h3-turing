#!/usr/bin/env python
"""从 ComfyUI-resized 的 LoRA 派生一份"去 diffusion_model. 前缀"的副本。

为什么需要：bypass（运行时应用）路径的节点（Larryvrh/ComfyUI-MiniMax-H3-Turbo）
按**无前缀**键名找模块（`blocks.0.attn.out_proj.lora_A.weight`），而
ComfyUI-resized 的权重带 `diffusion_model.` 前缀 → 直接给它会匹配到
**0 个 adapter**（LoRA 根本没生效，看起来像"加速节点没用"）。

数据逐字节相同，只改键名；派生结果登记在 models/MANIFEST.tsv。
用法：
  /home/a/ai-work/ComfyUI/.venv/bin/python scripts/derive_noprefix_lora.py \
      models/loras/<src>.safetensors models/loras/<dst>.safetensors
"""
import hashlib
import os
import sys

from safetensors.torch import load_file, save_file

PREFIX = "diffusion_model."


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print(f"{dst} already exists ({os.path.getsize(dst)} B); nothing to do")
    else:
        sd = load_file(src)
        out = {k[len(PREFIX):] if k.startswith(PREFIX) else k: v for k, v in sd.items()}
        save_file(out, dst, metadata={"format": "pt", "note": f"prefix-stripped copy of {os.path.basename(src)}"})
        print(f"wrote {dst}: {len(out)} tensors, {os.path.getsize(dst)} B")
    print(f"sha256 {sha256(dst)}  {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
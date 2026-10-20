#!/usr/bin/env python
"""同 seed 成片档对拍：比帧数据与音轨，不比文件名/sha256（MP4 里嵌了 graph metadata，改一个 widget 就换字节）。

用法（用 ComfyUI 的 venv 跑，需要 numpy + PyAV）：
  /home/a/ai-work/ComfyUI/.venv/bin/python scripts/compare_video.py a.mp4 b.mp4
  ... --frames 0,62,123 --png-out h3_ab/frames     # 额外导出并排目检图

判读纪律见 RUNBOOK §5.4 第 4 层：
  完全相同帧 124/124 → 数值等价
  0/124 相同但目检无伪影 → "同质量、不同画面"（如 T8 激进攻，docs/08）
  0/124 相同且目检软化/伪影 → 质量退化（如 kitchen INT8 attention，RUNBOOK §5.5）
"""
import argparse
import sys

import numpy as np
import av


def load(path):
    frames, samples = [], []
    with av.open(path) as c:
        for f in c.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24").astype(np.float32))
    with av.open(path) as c:
        for f in c.decode(audio=0):
            samples.append(f.to_ndarray().astype(np.float64))
    return frames, samples


def side_by_side(a, b, index, path):
    from PIL import Image, ImageDraw
    top, bot = Image.fromarray(a[index].astype(np.uint8)), Image.fromarray(b[index].astype(np.uint8))
    w, h = top.size
    canvas = Image.new("RGB", (w, h * 2 + 22), (0, 0, 0))
    canvas.paste(top, (0, 0))
    canvas.paste(bot, (0, h + 22))
    delta = float(np.abs(a[index] - b[index]).mean())
    ImageDraw.Draw(canvas).text(
        (6, h + 4), f"frame {index}  TOP={path[0]}  BOTTOM={path[1]}  |d|={delta:.2f}", fill=(255, 255, 255)
    )
    canvas.save(f"{path[2]}/frame{index}.png")


def main():
    ap = argparse.ArgumentParser(description="Compare two H3 renders by frame data, not by file hash.")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--frames", default="", help="comma-separated frame indices for side-by-side PNGs")
    ap.add_argument("--png-out", default="", help="directory for the side-by-side PNGs (default: next to a)")
    args = ap.parse_args()

    fa, sa = load(args.a)
    fb, sb = load(args.b)
    print(f"A {args.a}: {len(fa)} frames")
    print(f"B {args.b}: {len(fb)} frames")

    n = min(len(fa), len(fb))
    if n == 0:
        print("!! 没有可比帧")
        return 1

    ma = np.mean([f.mean() for f in fa])
    mb = np.mean([f.mean() for f in fb])
    black_a = int(sum(1 for f in fa if f.max() == 0))
    black_b = int(sum(1 for f in fb if f.max() == 0))
    print(f"亮度 mean: A {ma:.3f}  B {mb:.3f} | 全黑帧: A {black_a}  B {black_b}")

    mae = np.array([np.abs(fa[i] - fb[i]).mean() for i in range(n)])
    mse = np.array([((fa[i] - fb[i]) ** 2).mean() for i in range(n)])
    psnr = 10 * np.log10(255.0 ** 2 / max(mse.mean(), 1e-12))
    print(f"逐帧 |Δ|: mean {mae.mean():.4f} (范围 {mae.min():.4f}-{mae.max():.4f})")
    print(f"整体 PSNR: {psnr:.2f} dB | 完全相同帧: {int((mae == 0).sum())}/{n}")

    if sa and sb:
        wa, wb = np.concatenate(sa).ravel(), np.concatenate(sb).ravel()
        m = min(len(wa), len(wb))
        wa, wb = wa[:m], wb[:m]
        print(f"音轨: {m} samples | peak A {np.abs(wa).max():.4f} B {np.abs(wb).max():.4f} "
              f"| NaN A {bool(np.isnan(wa).any())} B {bool(np.isnan(wb).any())}")
        print(f"音轨 |Δ|: mean {np.abs(wa - wb).mean():.6f}, max {np.abs(wa - wb).max():.6f}")

    if args.frames:
        out = args.png_out or "."
        import os
        os.makedirs(out, exist_ok=True)
        for i in [int(x) for x in args.frames.split(",") if int(x) < n]:
            side_by_side(fa, fb, i, (args.a.split("/")[-1], args.b.split("/")[-1], out))
            print(f"并排图: {out}/frame{i}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
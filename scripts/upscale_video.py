#!/usr/bin/env python
"""640×352 → 1080p 交付：抽帧 → 超分 → 合回（音轨原样 copy）。

手册口径：**不要直接改工作流分辨率**（显存和时间都不划算），而是"生成 640×352 →
抽帧 → 超分 → 合回视频"。本脚本用 ffmpeg 一条命令完成（解码→缩放→编码→复制音轨），
避免用 PyAV 手工交织 A/V（那会踩 PyAV 的 rebase/mux 坑）。

ffmpeg 用的是 imageio-ffmpeg 附带的静态构建（装在**独立 venv** `/home/a/ai-work/.venv`，
不碰 ComfyUI 的 venv —— docs/09 记的 pip 污染量化/LoRA 路径的坑）。

用法：
  /home/a/ai-work/.venv/bin/python scripts/upscale_video.py in.mp4 out.mp4 --scale 3
  # 或指定 ffmpeg 路径 / 换缩放算法
  --ffmpeg /usr/bin/ffmpeg --scaler lanczos|hqx|zscale

超分算法说明（本机只有软件实现，没有 CUDA/NPP 滤镜的静态 ffmpeg）：
  - ``lanczos``：通用高质量重采样（默认）
  - ``hqx``     ：hqx 放大算法（2/3/4×），边缘更"锐"，有时更讨喜但会起振铃
  - ``zscale``  ：zimg，锐度介于两者之间
真·ML 超分（RTX VSR）需要 ``nvidia-vfx``；本机目前装不上（源码构建），故未内置。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_ffmpeg(explicit: str = "") -> str:
    if explicit:
        return explicit
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    p = shutil.which("ffmpeg")
    if p:
        return p
    raise SystemExit("找不到 ffmpeg：装 imageio-ffmpeg 或用 --ffmpeg 指定路径")


def probe(ffmpeg: str, path: Path) -> dict:
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True).stderr
    import re
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", out)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    fps = 24.0
    m2 = re.search(r"Video:.*?(\d+(?:\.\d+)?) fps", out)
    if m2:
        fps = float(m2.group(1))
    has_audio = "Audio:" in out
    return {"w": w, "h": h, "fps": fps, "audio": has_audio}


def main() -> int:
    ap = argparse.ArgumentParser(description="Upscale a rendered clip; audio is copied as-is.")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=int, default=3, help="2..4 (3 -> 640x352 becomes 1920x1056)")
    ap.add_argument("--scaler", choices=["lanczos", "hqx", "zscale"], default="lanczos")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--sharpen", type=float, default=0.0,
                    help="unsharp amount (0 = off); 0.4-0.8 counteracts resampling softness")
    ap.add_argument("--ffmpeg", default="")
    args = ap.parse_args()

    ff = find_ffmpeg(args.ffmpeg)
    src, dst = Path(args.src), Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = probe(ff, src)
    if not info["w"]:
        raise SystemExit(f"读不出视频信息：{src}")
    ow, oh = info["w"] * args.scale, info["h"] * args.scale

    if args.scaler == "hqx":
        vf = f"scale={ow}:{oh}:flags=neighbor,hqx"
    elif args.scaler == "zscale":
        vf = f"zscale={ow}:{oh}:filter= lanczos"
    else:
        vf = f"scale={ow}:{oh}:flags=lanczos"
    if args.sharpen > 0:
        vf += f",unsharp=5:5:{args.sharpen}:5:5:0"

    cmd = [ff, "-y", "-hide_banner", "-i", str(src),
           "-vf", vf,
           "-r", f"{info['fps']:.6f}",
           "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd += ["-c:a", "copy"] if info["audio"] else ["-an"]
    cmd += [str(dst)]

    print("cmd:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2500:], file=sys.stderr)
        return r.returncode

    out_info = probe(ff, dst)
    size = dst.stat().st_size
    print(f"done: {src} ({info['w']}x{info['h']} @{info['fps']:.0f}fps) → "
          f"{dst} ({out_info['w']}x{out_info['h']}), {size / 1024:.0f} KiB, "
          f"audio={'copied' if out_info['audio'] else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
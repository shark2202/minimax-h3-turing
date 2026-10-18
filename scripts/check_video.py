#!/usr/bin/env python3
"""Post-render checks for an H3 clip - including the i2v-specific one.

The three questions, in the order failures happen on this machine:

1. is it black?          NaN sampling leaves an all-black, audio-less mp4 that
                         still exists on disk (RUNBOOK 4.5) - existence proves nothing.
2. is it a real clip?     frames / geometry / fps, plus an audio track with signal
                         and no NaNs (H3 always carries native audio).
3. did the image drive it?  the i2v claim itself. Needs --first-frame; pass
                         --control with an *unrelated* image. A genuine i2v first
                         frame matches the given image far better than the control;
                         a t2v render matches neither. Two separate verdicts:
                         `first_frame` = fidelity (absolute r), `attribution` =
                         is that image the source at all (ratio vs controls).

Needs the ComfyUI venv (av + numpy + PIL), no GPU:
    /home/a/ai-work/ComfyUI/.venv/bin/python scripts/check_video.py shot.mp4 \
        --first-frame first.png --control unrelated.png

Exit 0 = all checks passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str) -> None:
    results.append((ok, name, detail))


def luma64(img, geometry: tuple[int, int] | None = None) -> np.ndarray:
    """64x64 float luma - the common ground for comparing frame to image.

    Small size on purpose: the frame is 640x352 and the source image has its
    own ratio. But an image is resized through `geometry` (the clip's w,h)
    first, because the model squashes the first frame into the latent geometry
    (nodes_minimax_h3._resize with crop="disabled"); comparing an image at its
    own aspect would score the aspect mismatch instead of the content.
    """
    if geometry:
        img = img.resize(geometry, Image.BILINEAR)
    return np.asarray(img.convert("L").resize((64, 64), Image.BILINEAR),
                      dtype=np.float32) / 255.0

def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson r and mean absolute error between two luma64 images."""
    da, db = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((da * da).sum() * (db * db).sum())) + 1e-9
    return float((da * db).sum() / denom), float(np.abs(a - b).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video", help="rendered mp4 to check")
    p.add_argument("--first-frame", default=None,
                   help="the image the i2v render was conditioned on")
    p.add_argument("--control", action="append", default=[],
                   help="unrelated image(s) used as the mismatch baseline (repeatable)")
    p.add_argument("--match-r", type=float, default=0.6,
                   help="minimum Pearson r for the first-frame match (default 0.6)")
    a = p.parse_args()

    path = Path(a.video)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1

    with av.open(str(path)) as c:
        v = c.streams.video[0]
        w, h = v.codec_context.width, v.codec_context.height
        fps = float(v.average_rate) if v.average_rate else 0.0
        frames, chunks = [], []
        streams = [v] + list(c.streams.audio[:1])
        # one pass over both streams: decoding video first leaves no packets
        # for a second decode() call on the same container
        for f in c.decode(*streams):
            if isinstance(f, av.VideoFrame):
                frames.append(luma64(f.to_image()))
            else:
                chunks.append(f.to_ndarray().astype(np.float64).ravel())
        audio = np.concatenate(chunks) if chunks else None

    dur = len(frames) / fps if fps else 0.0
    means = np.array([f.mean() for f in frames]) if frames else np.array([0.0])
    stds = np.array([f.std() for f in frames]) if frames else np.array([0.0])
    check(len(frames) >= 5, "frames",
          f"{len(frames)} frames @ {fps:.0f}fps  {w}x{h}  {dur:.2f}s")

    # black frame == mean 0 / max 0 (RUNBOOK 4.5); healthy baseline is mean 0.29-0.42
    check(means.min() > 0.03 and stds.max() > 0.02, "not_black",
          f"frame mean {means.min():.2f}-{means.max():.2f} (min per frame), "
          f"std max {stds.max():.2f}")

    if audio is None:
        check(False, "audio", "no audio track (the NaN failure also drops audio)")
    else:
        peak = float(np.abs(audio).max())
        if peak > 2.0:  # integer PCM: scale to +/-1
            peak /= 32768.0
        check(not np.isnan(audio).any() and peak > 0.02, "audio",
              f"{audio.size} samples, peak {peak:.3f}, "
              f"NaN {'yes' if np.isnan(audio).any() else 'no'}")

    if a.first_frame:
        given = luma64(Image.open(a.first_frame), (w, h))
        r, mae = similarity(frames[0], given)
        check(r >= a.match_r, "first_frame",
              f"r={r:.3f} MAE={mae:.3f} vs {Path(a.first_frame).name} (need r>={a.match_r})")
        if a.control:
            ctrl_r = {c: similarity(frames[0], luma64(Image.open(c), (w, h)))[0]
                      for c in a.control}
            worst = max(ctrl_r.values())
            # attribution, not fidelity: "is the image an input at all". Expressed
            # as a ratio, because a genuine i2v first frame scores low in absolute
            # terms when the source image fights the prompt (measured: own 0.206 vs
            # other 0.030 = x6.9, while an unrelated render managed ~x0.03).
            check(r > 0.1 and r >= 3 * worst, "attribution",
                  f"own r x{r / worst:.1f} vs best control "
                  f"({', '.join(f'{Path(c).name}={v:.3f}' for c, v in ctrl_r.items())})")
    else:
        print("note: no --first-frame given - the i2v match check was skipped")

    failed = [n for ok, n, _ in results if not ok]
    print(f"\n{path}  ({path.stat().st_size / 1024:.0f} KiB)")
    for ok, name, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<12} {detail}")
    print("=> " + ("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
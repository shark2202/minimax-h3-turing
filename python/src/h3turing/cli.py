"""Command-line interface.

    h3turing demo    --prompt TEXT [--image PATH] [--seed N] [--tier KEY]
                     [--base URL] [--out DIR]      # one command: text/image -> mp4
    h3turing doctor  [--base URL] [--models-dir DIR]
    h3turing bench   --a TIER --b TIER [--seed N] [--prompt TEXT]
                     [--log-path FILE] [--base URL] [--out FILE]
    h3turing tiers

Exit codes: 0 = success, 1 = checks failed / runtime error, 2 = bad usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PROMPT = ("cinematic wide shot, a lone swordsman walking through a misty "
                  "bamboo forest at dawn, natural ambient audio")


def _cmd_doctor(args) -> int:
    from . import doctor
    ok = doctor.report(base=args.base, models_dir=args.models_dir)
    return 0 if ok else 1


def _cmd_demo(args) -> int:
    """The minimal end-to-end pipeline: prompt (+ optional image) -> local mp4.

    Defaults to the plain reproducible tiers (t2v-final / i2v-final); pick a
    draft/ultra tier with --tier once the environment supports it (docs/09).
    """
    import time

    from .client import ComfyUI, collect_outputs
    from .tiers import get
    from .workflows import load, set_image, set_prompt, set_seed

    tier = get(args.tier or ("i2v-final" if args.image else "t2v-final"))
    if args.image and tier.route != "i2v":
        raise RuntimeError(f"tier {tier.key!r} is {tier.route} - drop --image or pick an i2v tier")
    if not args.image and tier.route == "i2v":
        raise RuntimeError(f"tier {tier.key!r} needs a first frame - pass --image PATH")

    comfy = ComfyUI(base=args.base)
    if not ComfyUI(base=args.base, retries=1, backoff=0.1).alive():
        raise RuntimeError(f"ComfyUI not reachable at {args.base} - start it via "
                           "scripts/h3_launch.example.sh, then verify with 'h3turing doctor'")

    graph = load(tier.workflow)
    set_prompt(graph, args.prompt)
    set_seed(graph, args.seed)
    if args.image:
        image = Path(args.image).expanduser()
        if not image.is_file():
            raise RuntimeError(f"image not found: {image}")
        print(f"uploading first frame: {image.name}", flush=True)
        set_image(graph, comfy.upload_image(image))

    print(f"tier {tier.key} | seed {args.seed} | {tier.steps} steps "
          f"(reference {tier.measured_s}s on 2080Ti 22G)", flush=True)
    if not tier.reproducible:
        print("note: this is a draft tier - NOT same-seed reproducible", flush=True)

    t0 = time.time()
    pid = comfy.submit(graph)
    print(f"submitted {pid}; rendering (progress every ~30s)...", flush=True)

    def progress(elapsed, _history, queue):
        if int(elapsed / max(args.poll_every, 1e-9)) % 3 == 0:
            print(f"  {elapsed:5.0f}s | queue: {len(queue['running'])} running, "
                  f"{len(queue['pending'])} pending", flush=True)

    history = comfy.wait(pid, max_s=args.max_s, poll_every=args.poll_every, on_poll=progress)

    files = collect_outputs(history)
    if not files:
        raise RuntimeError("render finished but history has no saved output - "
                           "check the SaveVideo node and the server log")

    out_dir = Path(args.out)
    for entry in files:
        dest = out_dir / entry["filename"]
        if dest.exists():  # server counters reset on restart - keep old takes
            dest = out_dir / f"{pid[:8]}_{entry['filename']}"
        comfy.download(entry, dest)
        print(f"saved: {dest.resolve()}", flush=True)
    print(f"done in {time.time() - t0:.0f}s ({tier.key}, seed {args.seed})", flush=True)
    return 0


def _cmd_tiers(_args) -> int:
    from .tiers import TIERS
    print(f"{'key':<15} {'route':<5} {'cache':<5} {'repro':<5} {'measured':>9}  env")
    for key, t in sorted(TIERS.items()):
        print(f"{key:<15} {t.route:<5} {'yes' if t.cached else 'no':<5} "
              f"{'yes' if t.reproducible else 'NO':<5} {t.measured_s:>7}s  {t.env}")
    print("\ncached tiers (T8 / TeaCache / FirstBlockCache) are NOT same-seed reproducible - finals only from plain tiers")
    return 0


def _cmd_bench(args) -> int:
    from .benchmark import ab_pair, save_report, speedup
    from .client import ComfyUI
    from .tiers import get
    from .workflows import load, set_prompt, set_seed

    comfy = ComfyUI(base=args.base)
    graphs = {}
    for label, key in (("control", args.a), ("treatment", args.b)):
        tier = get(key)
        g = load(tier.workflow)
        set_prompt(g, args.prompt)
        set_seed(g, args.seed)   # same seed across arms - the A/B invariant
        graphs[label] = g

    print(f"A/B: {args.a} (control) vs {args.b} (treatment) | seed {args.seed}", flush=True)
    results = ab_pair(comfy, graphs, log_path=args.log_path, max_s=args.max_s)
    delta = speedup(results["control"], results["treatment"])
    print(f"\n=> treatment {delta:+.1f}% vs control "
          f"({results['treatment'].wall_s:.0f}s vs {results['control'].wall_s:.0f}s)")
    if args.out:
        save_report(results, args.out, meta={"a": args.a, "b": args.b, "seed": args.seed})
        print("report saved:", args.out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="h3turing",
        description="Field-tested toolkit for MiniMax H3 on Turing GPUs (see the handbook)")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("demo", help="minimal pipeline: text/image -> video, one command",
                       description="Render one clip end-to-end: upload first frame (i2v), "
                                   "submit the tier's workflow, wait, download the mp4.")
    m.add_argument("--prompt", default=DEFAULT_PROMPT, help="text prompt (default: builtin demo prompt)")
    m.add_argument("--image", default=None, help="first-frame image path -> switches to i2v")
    m.add_argument("--seed", type=int, default=3013)
    m.add_argument("--tier", default=None,
                   help="tier key (default: t2v-final, or i2v-final with --image)")
    m.add_argument("--base", default="http://127.0.0.1:8188")
    m.add_argument("--out", default="h3_demo_output", help="directory for downloaded videos")
    m.add_argument("--max-s", type=float, default=1800.0)
    m.add_argument("--poll-every", type=float, default=10.0,
                   help="history poll interval in seconds (tests use 0.01)")

    d = sub.add_parser("doctor", help="pre-production checks (ComfyUI alive, queue, T8 node, weights)")
    d.add_argument("--base", default="http://127.0.0.1:8188")
    d.add_argument("--models-dir", default=None, help="ComfyUI models dir for weight checks")

    b = sub.add_parser("bench", help="same-seed A/B benchmark between two tiers")
    b.add_argument("--a", required=True, help="control tier key (e.g. t2v-final)")
    b.add_argument("--b", required=True, help="treatment tier key (e.g. t2v-draft)")
    b.add_argument("--seed", type=int, default=3013)
    b.add_argument("--prompt", default=DEFAULT_PROMPT)
    b.add_argument("--log-path", default=None,
                   help="ComfyUI server log for cache-hit parsing (T8 verbose=True required)")
    b.add_argument("--base", default="http://127.0.0.1:8188")
    b.add_argument("--max-s", type=float, default=1800.0)
    b.add_argument("--out", default=None, help="save JSON report to this path")

    sub.add_parser("tiers", help="list the six tiers with measured timings")

    args = p.parse_args(argv)
    handlers = {"demo": _cmd_demo, "doctor": _cmd_doctor,
                "bench": _cmd_bench, "tiers": _cmd_tiers}
    try:
        return handlers[args.cmd](args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

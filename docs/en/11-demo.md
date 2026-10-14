# 11 · Minimal Demo: text/image → video in one command

> The executable summary of the whole pipeline: once the weights are in place, going from a prompt (or a first-frame image) to a local mp4 takes exactly one command. Zero dependencies, stdlib only.

## Prerequisites (one-time, ~30 min)

1. **ComfyUI ≥ v0.33.1**, launched with the memory-safe flags from [04](04-community-tips.md) (anti-TDR):

   ```bash
   # template in scripts/h3_launch.example.sh; the flags that matter:
   python main.py --listen 127.0.0.1 --port 8188 \
     --reserve-vram 2.5 --vram-headroom 0.5 --disable-pinned-memory
   ```

2. **Weights**: put the W4A8 component set into `models/` per the [download list + copy-paste commands in 05](05-workflows.md) (~21G; verify with `h3turing doctor --models-dir` afterwards).
3. **This toolkit**:

   ```bash
   pip install git+https://github.com/IvenKooLab/minimax-h3-turing.git#subdirectory=python
   ```

## Text → video (t2v)

```bash
h3turing demo --prompt "cinematic wide shot, a lone swordsman walking through a misty bamboo forest at dawn"
```

## Image → video (i2v)

```bash
h3turing demo --image first_frame.png --prompt "she slowly turns to camera, wind in her hair"
```

Adding `--image` switches to the i2v tier and uploads the first frame. Landscape images recommended (the workflow generates at 640×352, roughly 16:9).

## What success looks like

```
tier t2v-final | seed 3013 | 4 steps (reference 280s on 2080Ti 22G)
submitted 6f2a…; rendering (progress every ~30s)...
  120s | queue: 1 running, 0 pending
saved: /home/you/h3_demo_output/H3Lite_00001.mp4
done in 287s (t2v-final, seed 3013)
```

- The clip lands in the `--out` directory (default `./h3_demo_output/`); ComfyUI's `output/video/` keeps a copy
- Reference timings (warm): **~280s** for t2v-final, **~420s** for i2v-final; draft/ultra tiers listed by `h3turing tiers`
- Default parameters: 640×352 · 124 frames (~5 s @ 24 fps) · 4-step Turbo · native audio

## What it does for you

| Step | The trap it avoids |
|---|---|
| Liveness probe with fast fail | Server down → error in 0.1 s with launch hints, not a silent hang |
| Multipart first-frame upload (i2v) | LoadImage wants the **server-side filename**, not your local path |
| Placeholder substitution (prompt/first frame/seed) | v3-API nodes need every input explicit; miss one = HTTP 400 |
| Post-submit polling with stall detection | The HTTP API lags and drops while the GPU is at 100% (FAQ) |
| Output parsing + mp4 download | SaveVideo lands in history `outputs` under a key that drifts across versions — all three are scanned |
| No same-name overwrite | Server counters reset on restart; old takes survive |

## Troubleshooting

- Run `h3turing doctor` first: liveness, queue residue, T8 node, weight set — one command
- `ComfyUI not reachable`: server not started or wrong port (`--base`)
- `render finished but history has no saved output`: check the server log for SaveVideo/encode errors
- LoRA silently dead after installing pip packages (the Sep-20 lesson): the `ERROR lora` check in [09](09-pdd-backport.md)
- The other 13 traps: [06 FAQ](06-faq.md)

## Switching tiers

`--tier` accepts anything `h3turing tiers` lists (e.g. the 160s `t2v-draft`). Remember the one rule: **T8/PDD tiers are not same-seed reproducible — drafts only, never finals**.

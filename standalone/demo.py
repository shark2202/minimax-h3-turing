#!/usr/bin/env python3
"""Minimal text/image -> video demo WITHOUT ComfyUI - pure PyTorch + Diffusers.

Target: single 2080Ti 22G (Turing sm_75 - no BF16, no FlashAttention-2).

  t2v:  python demo.py --prompt "a cat walking on a beach at sunset"
  i2v:  python demo.py --image first_frame.png --prompt "the cat looks up"

Routes (why not MiniMax H3: its W4A8 weights load only through ComfyUI's
comfy-kitchen dequant backend - outside ComfyUI there is nothing to load them):

  t2v  Wan-AI/Wan2.1-T2V-1.3B-Diffusers      ~29GB download (fp32/bf16 as
        shipped); cast to fp16 at load the stack is ~15G and the whole
        pipeline fits the 22G card (transformer 2.8G + umt5 11.4G + vae 0.5G)
  i2v  Wan-AI/Wan2.1-I2V-14B-480P-Diffusers  ~90GB download (transformer
        shipped fp32, 65.6G); even in fp16 it is ~33G > 22G VRAM, so it
        streams layer groups through the card (diffusers group offloading,
        the documented low-VRAM path for 14B)

Weights auto-download to the HuggingFace cache on first run; see
standalone/README.md for the exact cache layout and manual pre-download.

Turing adaptations baked in: fp16 everywhere except the VAE (fp32 - the Wan
VAE in fp16 produces black/NaN frames), stock SDPA attention, no bf16.
First run downloads weights to the HuggingFace cache (~/.cache/huggingface).
"""

from __future__ import annotations

import argparse
import sys
import time

T2V_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
I2V_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# Official Wan negative prompt (diffusers docs) - quality depends on it.
DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards")

DEFAULT_PROMPT = ("A sleek gray cat walking on a beach at sunset, gentle "
                  "waves in the background, warm golden light, cinematic "
                  "tracking shot, shallow depth of field.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)
    p.add_argument("--image", default=None,
                   help="first-frame image -> switches to the i2v route (14B, ~40GB download)")
    p.add_argument("--out", default="demo_out.mp4")
    p.add_argument("--steps", type=int, default=None,
                   help="denoising steps (default 50 t2v / 30 i2v; fewer = faster)")
    p.add_argument("--frames", type=int, default=81,
                   help="must be 4k+1 (81 = ~5s at 16fps)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def run_t2v(args):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    vae = AutoencoderKLWan.from_pretrained(T2V_ID, subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(T2V_ID, vae=vae, torch_dtype=torch.float16)
    pipe.to("cuda")                     # 1.3B fp16 stack ~15GB - fits 22G fully
    pipe.vae.enable_tiling()            # bounds the 81-frame fp32 decode peak
    return pipe(
        prompt=args.prompt, negative_prompt=args.negative,
        height=args.height, width=args.width, num_frames=args.frames,
        num_inference_steps=args.steps or 50, guidance_scale=5.0,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]


def run_i2v(args):
    # 14B transformer = ~28GB fp16 > 22GB VRAM: stream layer groups through
    # the card (the documented low-VRAM path), never hold the whole model.
    import torch
    from diffusers import (AutoencoderKLWan, WanImageToVideoPipeline,
                           WanTransformer3DModel)
    from diffusers.hooks.group_offloading import apply_group_offloading
    from transformers import CLIPVisionModel, UMT5EncoderModel

    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    text_encoder = UMT5EncoderModel.from_pretrained(
        I2V_ID, subfolder="text_encoder", torch_dtype=torch.float16)
    apply_group_offloading(text_encoder, onload_device=cuda, offload_device=cpu,
                           offload_type="block_level", num_blocks_per_group=4)
    image_encoder = CLIPVisionModel.from_pretrained(
        I2V_ID, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(I2V_ID, subfolder="vae",
                                           torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(
        I2V_ID, subfolder="transformer", torch_dtype=torch.float16)
    transformer.enable_group_offload(onload_device=cuda, offload_device=cpu,
                                     offload_type="leaf_level", use_stream=True)
    pipe = WanImageToVideoPipeline.from_pretrained(
        I2V_ID, text_encoder=text_encoder, image_encoder=image_encoder,
        vae=vae, transformer=transformer, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    # Keep the input's aspect ratio inside the 480p budget, snapped to the
    # model's patch grid (formula from the diffusers Wan docs).
    import numpy as np
    from diffusers.utils import load_image
    image = load_image(args.image)
    max_area = args.height * args.width
    aspect = image.height / image.width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round((max_area * aspect) ** 0.5) // mod * mod
    width = round((max_area / aspect) ** 0.5) // mod * mod
    image = image.resize((width, height))
    print(f"first frame resized to {width}x{height}")

    return pipe(
        image=image, prompt=args.prompt, negative_prompt=args.negative,
        height=height, width=width, num_frames=args.frames,
        num_inference_steps=args.steps or 30, guidance_scale=5.0,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]


def main(argv=None):
    args = parse_args(argv)
    if (args.frames - 1) % 4:
        sys.exit("--frames must be 4k+1 (81, 77, 61 ...) - the Wan VAE temporal stride")
    t0 = time.time()
    print(f"route: {'i2v (14B, group-offloaded)' if args.image else 't2v (1.3B)'} | "
          f"{args.frames} frames @ {args.width}x{args.height} | seed {args.seed}")
    print("loading weights (first run downloads ~29G t2v / ~90G i2v - see README)")
    frames = run_i2v(args) if args.image else run_t2v(args)
    from diffusers.utils import export_to_video
    export_to_video(frames, args.out, fps=16)
    print(f"saved: {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()

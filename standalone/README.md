# standalone/ · 脱离 ComfyUI 服务器的最小视频生成 Demo

## 主线：MiniMax H3，嵌入库模式（`h3_demo.py`）

不起服务器、不开 UI、不走 HTTP——把本地 ComfyUI 当 Python 库 import，单进程跑完 文本/图片 → mp4。**模型就是 H3 本身，权重零新增下载**（与服务器路线共用 docs/05 那套，放在 ComfyUI 的 `models/` 树下）。

```bash
cd standalone

# 文生视频（compat 成片档，~280s/镜）
python h3_demo.py --comfy-dir /path/to/ComfyUI \
    --prompt "a lone swordsman in a misty bamboo forest at dawn"

# 图生视频（自动把首帧拷进 ComfyUI input 目录，~420s/镜）
python h3_demo.py --comfy-dir /path/to/ComfyUI \
    --image first_frame.png --prompt "she slowly turns to camera"
```

`--comfy-dir` 可用环境变量 `COMFYUI_PATH` 代替；`--workflow` 可换六件套任意档。完整说明（形态对比、原理、边界）见 [docs/12](../docs/12-standalone-demo.md)。

**所需模型 = docs/05 清单**（[下载表 + 复制粘贴命令](../docs/05-workflows.md)，~21G），按表放入 ComfyUI 的 `models/diffusion_models/`、`text_encoders/`、`clip_projections/`、`vae/`、`loras/` ——与 ComfyUI 服务器路线完全同一套，装过一次两边共用。

## 附录：纯 Diffusers 替代路线（`demo.py`，Wan 2.1 · 不是 H3）

机器上连 ComfyUI 代码都不想放时的极端选项。模型自动下载到 HF 缓存，无需手动放置。

```bash
cd standalone
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 版 torch
pip install -r requirements.txt

python demo.py --prompt "A sleek gray cat walking on a beach at sunset"        # t2v（~29G 下载）
python demo.py --image first_frame.png --prompt "the cat looks up, curious"    # i2v（~90G 下载）
```

输出 `demo_out.mp4`（832×480 · 81 帧 ≈ 5 秒 @16fps）。参数：`--steps`（t2v 50 / i2v 30）、`--frames`（4k+1）、`--seed`、`--out`。

### Wan 路线模型清单与存放目录

| 仓库（route） | 组件 | 下载量 | 运行时（加载转 fp16） |
|---|---|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（t2v） | transformer | 5.7G（fp32 原样） | 2.8G |
| | text_encoder（umt5-xxl） | 22.7G | 11.4G |
| | vae + tokenizer 等 | 0.6G | vae 保持 fp32 ~0.5G |
| | **小计** | **~29G** | **~15G，整卡驻留** |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`（i2v） | transformer | 65.6G（fp32 原样） | 32.8G，超 22G → 层组流式 |
| | text_encoder（umt5-xxl，与 t2v 仓库重复） | 22.7G | 11.4G（同样流式） |
| | image_encoder（CLIP ViT-H） | 1.3G | fp32 ~1.3G |
| | vae + tokenizer 等 | 0.6G | vae 保持 fp32 ~0.5G |
| | **小计** | **~90G** | — |

存放目录（默认 `~/.cache/huggingface/hub/models--Wan-AI--*/snapshots/<版本>/…`）；换盘 `export HF_HOME=/data/hf`；国内镜像 `export HF_ENDPOINT=https://hf-mirror.com`；预下载 `huggingface-cli download <仓库名>`。

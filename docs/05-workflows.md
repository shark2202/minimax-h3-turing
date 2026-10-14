# 05 · compat 工作流说明

## 模型下载清单（先看这个）

工作流引用的文件名来自 h3lite W4A8 组件集 A（网盘分发名）；本仓库全部实测基于**路线 B（HF 直链等价版）**——下表右列即推荐下载源：

| 工作流引用名 | 推荐实际下载 | 来源 | 放置目录 |
|---|---|---|---|
| minimax_h3_fl2va_pruned_w4a8_mixed_ax1y2jp.safetensors (12.5G) | minimax_h3_fl2va_pruned_w4a8_mixed.safetensors | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental) | models/diffusion_models/ |
| qwen3vl_4b_int4_convrot.safetensors (2.8G) | qwen3vl_4b_fp8_scaled.safetensors (5.2G) | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) → text_encoders/ | models/text_encoders/ |
| mmh3-4b-ClipProj-celeb-mlp.safetensors (304M) | 同名直下 | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) | models/clip_projections/ |
| minimax_h3_video_vae_fp16 / audio_vae_fp32 | 同名直下 | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) → vae/ | models/vae/ |
| Turbo 4step LoRA（fl2v/ref2v） | 同名直下 | 同上 → loras/ | models/loras/ |
| PDD Acc-8Step LoRA（FL2VA=t2v 极速档 / Ref2VA=i2v 极速档） | 同名直下 | Kijai 仓 → loras/ | models/loras/ |

- **改名规则**：用替代版时，把工作流 JSON 里 loader 节点的文件名改成实际下载名（或反向重命名文件）；两个 W4A8 版本仅差约 26KB（量化配置微调），实测输出一致
- **坚持要 ax1y2jp 原名版**：h3lite Set A 百度盘 `pan.baidu.com/s/1x5GGuJv0h8chApgVoDgIaQ`（提取码 1hjx）
- **国内镜像**：huggingface.co 直连不动时换 hf-mirror.com，路径不变；大文件建议多线程分块下载

### 复制粘贴下载（命令版）

上表用来看，这一节用来抄——整块复制执行即可。`MODELS` 指向 ComfyUI 的 `models/` 树；②④⑤⑥ 的仓库子目录（`text_encoders/`、`vae/`、`loras/`）与 ComfyUI 子目录同名，`--local-dir "$MODELS"` 直接落位，①③ 在仓库根目录、显式指定目标子目录（新版 CLI 等价写法：`hf download …`，参数相同）：

```bash
pip install -U huggingface_hub              # 已装可跳过
MODELS=/path/to/ComfyUI/models              # ← 改成你的路径
# export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像：直连不动时取消注释

# ① 主权重 W4A8 DiT（12.5G）→ diffusion_models/
huggingface-cli download Kijai/MiniMax-H3-experimental \
    minimax_h3_fl2va_pruned_w4a8_mixed.safetensors \
    --local-dir "$MODELS/diffusion_models"

# ② 文本编码器 qwen3vl-4b fp8（5.2G）→ text_encoders/
huggingface-cli download Comfy-Org/Krea-2 \
    text_encoders/qwen3vl_4b_fp8_scaled.safetensors \
    --local-dir "$MODELS"

# ③ ClipProj（304M）→ clip_projections/
huggingface-cli download NicoLab28/ClipProj-MiniMax-H3 \
    mmh3-4b-ClipProj-celeb-mlp.safetensors \
    --local-dir "$MODELS/clip_projections"

# ④ 双 VAE（视频 fp16 + 音频 fp32）→ vae/
huggingface-cli download Comfy-Org/MiniMax-H3 \
    vae/minimax_h3_video_vae_fp16.safetensors \
    vae/minimax_h3_audio_vae_fp32.safetensors \
    --local-dir "$MODELS"

# ⑤ Turbo 4 步 LoRA 两支（compat 档用 v0.1，t8draft 档用 v1.0_768p）→ loras/
huggingface-cli download Comfy-Org/MiniMax-H3 \
    loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors \
    loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors \
    --local-dir "$MODELS"

# ⑥ PDD 8 步 LoRA（FL2VA=t2v / Ref2VA=i2v，只跑 PDD 档才需要）→ loras/
huggingface-cli download Kijai/MiniMax-H3-experimental \
    loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors \
    loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors \
    --local-dir "$MODELS"
```

下完核对：①–⑤ 约 21G；compat 工作流加载节点里是 Set-A 名（ax1y2jp / int4_convrot），按上方**改名规则**处理即可跑通。

```bash
ls -lh "$MODELS"/{diffusion_models,text_encoders,clip_projections,vae,loras}
h3turing doctor --models-dir "$MODELS"     # 用法见 python/README
```

不方便装 CLI 时，把下面的直链逐条粘进浏览器 / `wget -c` / 任何下载器，下完手动放进对应目录（镜像只换域名 hf-mirror.com，路径不变；`wget -c` 自带断点续传，比 aria2 稳，见 [FAQ #5](06-faq.md)）：

```text
https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3/resolve/main/mmh3-4b-ClipProj-celeb-mlp.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors
https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors
https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors
```

## 为什么是 "compat"

H3 官方完整版工作流引用了 T8 BlockCache 节点和完整精度模型名，在 Turing 卡 + 未装 T8 的环境里是**死图**（节点缺失、模型对不上）。compat 版是社区维护的降级适配：

- 去掉 sm_75 跑不了的加速节点
- 模型文件名对齐 W4A8 mixed 现役权重
- 保留 4 步 Turbo 路线 + 原生音频

实测这张卡上 compat + W4A8 就是最优解（见 [01](01-hardware-limits.md)），不存在"升回完整版"的损失。

## 文件

> 完整用途矩阵、导入三步与坑提示见 [workflows/README](../workflows/README.md)。

| 文件 | 用途 |
|---|---|
| [h3_w4a8_t2v_compat_api.json](../workflows/h3_w4a8_t2v_compat_api.json) | 文生视频 · 成片档（280s/镜，可复现） |
| [h3_w4a8_t2v_t8draft_api.json](../workflows/h3_w4a8_t2v_t8draft_api.json) | 文生视频 · 草稿快跑档（160s/镜，-43%，同 seed 不可复现） |
| [h3_w4a8_i2v_compat_api.json](../workflows/h3_w4a8_i2v_compat_api.json) | 图生视频（需要首帧） |

## 导入与替换

1. ComfyUI 界面拖入 JSON，或放到 `user/default/workflows/`
2. 检查模型加载节点里的文件名与你本地 `models/` 下实际文件一致（不同时间下载的 W4A8 权重命名不同）
3. **图生视频**：把首帧图加载节点指向你自己的图片（本仓库不含示例图）
4. 默认参数：640×352 · 124 帧 · 4 步 · 带原生音频

## 产线参数建议

- 短片（5-6 秒）直接用默认值，`--reserve-vram 2.5`（见 [04](04-community-tips.md)）
- 要 1080p 成片：保持 640×352 生成 → 抽帧 → RTX VSR 超分，**不要**直接改工作流分辨率（显存和时间都不划算）
- prompt 按三段式写：主体 → 动作/运镜 → 环境与光线，具体参考 h3lite 官方方法论

## 新版 ComfyUI 的 UI 导出与子图工作流

在 master 环境（本手册 PDD 档的前提）里，新版前端会把**子图（subgroup）节点**导出进 UI 格式工作流。这类 JSON 直接拖进界面能跑，但**提交 `/prompt` 会被拒**——子图是前端概念，API 层只认展平后的节点图。

判定与处理：

- **症状**：`/prompt` 报缺节点或连线错乱，但界面里明明能跑；打开 JSON 能看到 `type` 带子图命名空间、连线 id 是 UUID、或出现 `inputs=-10 / outputs=-20` 的虚拟节点
- **正解**：先展平再提交。展平要点：实例化子图内部节点 → 桥接输入槽（含 fan-out）→ 级联解析 UUID 连线链 → 跳过 `MarkdownNote` 等前端专属节点 → 数据穿透的槽自动生成 Primitive 节点承载默认值；widget 绑定线的判定特征是 `"widget" in t_ins[l[4]]`（新版导出把 widget 槽列进节点 `inputs`）
- **现成实现**：[ComfyAgent](https://github.com/IvenKooLab/comfy-agent) v1.5.2 的 `server.py: convert_ui_to_api / _flatten_subgraphs`（开源 MIT，含 5 项单测，实测 102 节点嵌套图零丢失展开）
- 本仓库的六件套是**原生 API 格式**，不受此问题影响；只有当你从界面另存/改造出自己的工作流时才需要走这一步

## 常见导入失败

| 现象 | 原因 |
|---|---|
| 节点红色缺失 | 没装 H3 支持节点包（KJNodes 等），或 ComfyUI 版本过老 |
| 模型加载节点报错 | 权重文件名不匹配，双击节点改成你本地实际文件名 |
| 出图彩色撕裂 | 用错了 W4A4 权重，换 W4A8 mixed |

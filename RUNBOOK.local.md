# RUNBOOK.local.md · 本机部署与操作手册

> 这份是**本机（a@2080Ti-22G 服务器）专属**的落地记录，不是上游 handbook 的一部分。
> 上游原理与实测数据见 `docs/`，这里只写"在这台机器上怎么跑"。
> 最后更新：2026-10-15（首次部署并实测跑通 t2v 成片档）

---

## 0. 一页速查

```bash
cd /home/a/ai-work/minimax-h3-turing

# ① 启动服务（局域网 + tailscale 可访问，约 20s 就绪）
scripts/h3_launch.local.sh

# ② 健康检查（服务、队列、T8 节点、权重）
scripts/h3turing.local.sh doctor --models-dir /home/a/ai-work/models
# 档位一览（6 档，含实测耗时与所需环境）
scripts/h3turing.local.sh tiers

# ③ 跑一镜（文生视频·成片档，实测 250s）
scripts/h3turing.local.sh demo --tier t2v-final --out /home/a/ai-work/minimax-h3-turing/h3_demo_output

# ④ 停止服务
pkill -f "main.py --listen"
```

Mac 上：浏览器开 `http://192.168.100.2:8188/`（不通就换 `192.168.8.189`，装了 tailscale 用 `100.69.133.37`）。

---

## 1. 环境事实（2026-10-15 实测）

| 项 | 值 |
|---|---|
| GPU | RTX 2080Ti 22G（Turing sm_75，driver 580.178.04，可用显存 21.5 GiB） |
| ComfyUI | **v0.33.1**，`git status` 干净、无本地补丁 |
| 解释器 | `/home/a/ai-work/ComfyUI/.venv/bin/python`（torch **2.14.1+cu130**、PyAV 17.1、comfy_kitchen 可用） |
| 权重树 | `/home/a/ai-work/models`（经 `ComfyUI/extra_model_paths.yaml` 挂载），9 文件 ≈29.6 GB，sha256 全部核对通过 |
| 节点 | 原生 H3（`comfy_extras/nodes_minimax_h3.py`）+ `custom_nodes/ComfyUI-ClipProj` |
| 网卡 | `192.168.100.2`（enp4s0）/ `192.168.8.189`（USB 网卡）/ `100.69.133.37`（tailscale） |
| 磁盘 | `/` 余量 ≈749 GB |
| 缺的东西 | `python3.10-dev`（无 Python.h）、T8 BlockCache 节点包、master 环境 |

权重清单与核对结果（`sha256sum` 实测 vs 上游 LFS oid）：

```
diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors   12.54 GB  01aa7b92… ✓
text_encoders/qwen3vl_4b_fp8_scaled.safetensors                    5.24 GB  54bd5144… ✓
vae/minimax_h3_video_vae_fp16.safetensors                          5.21 GB  7c1f1314… ✓
vae/minimax_h3_audio_vae_fp32.safetensors                          605 MB   8e505d95… ✓
loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors  1.96 GB  c396a9a0… ✓
loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors  315 MB  3a069f26… ✓
loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors                 1.72 GB  71347e27… ✓
loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors                1.72 GB  d7cd9480… ✓
clip_projections/mmh3-4b-ClipProj-celeb-mlp.safetensors            304 MB   275b3899… ✓
```

软链别名（工作流里用的是 Set-A 旧名，指向等价版本，所以 loader 无需改名）：
`…_w4a8_mixed_ax1y2jp.safetensors → …_mixed.safetensors`、`qwen3vl_4b_int4_convrot → qwen3vl_4b_fp8_scaled`。

---

## 2. 启动 / 停止 / 参数

`scripts/h3_launch.local.sh` 等价于：

```bash
TORCH_DISABLE_NATIVE_JIT=1 ComfyUI/.venv/bin/python main.py \
  --listen 0.0.0.0 --port 8188 \
  --reserve-vram 2.5 --vram-headroom 0.5 --disable-pinned-memory
```

- 后三个是手册的防黑屏/OOM 参数，**不要删**（共享桌面 GPU 防 TDR）
- `TORCH_DISABLE_NATIVE_JIT=1` 是**本机必须**，原因见 §4.1
- 覆盖开关：`H3_LISTEN=`（默认 `0.0.0.0`）、`H3_PORT=`、`H3_LOG=`（默认 `ComfyUI/h3_server.log`）、`COMFY_DIR=`、`PY=`
- 收紧访问：
  ```bash
  H3_LISTEN=100.69.133.37 scripts/h3_launch.local.sh   # 只 tailscale
  H3_LISTEN=127.0.0.1     scripts/h3_launch.local.sh   # 只本机
  ```
- 停止：`pkill -f "main.py --listen"`。启动约 20s 后 `curl -s 127.0.0.1:8188/system_stats` 有响应即就绪
- 本机**不设开机自启**（明确定策）：机器重启 / 会话结束后服务不会自己起来，需要手动跑一次本脚本
- 日志：`ComfyUI/h3_server.log`（`h3turing bench --log-path` 要靠它解析 T8 命中数）

---

## 3. Mac 侧操作

### 3.1 工作流在哪

已放进 ComfyUI 用户目录，**界面左侧 Workflows 面板直接可见**，无需下载：

```
ComfyUI/user/default/workflows/h3_w4a8_t2v_compat_api.json   文生视频·成片档（250s）
ComfyUI/user/default/workflows/h3_w4a8_i2v_compat_api.json   图生视频·成片档（~420s）
```

前端识别 API 格式 JSON 并自动铺成节点图（`isApiJson` / `loadApiJson`）。也可自己导入：

```bash
# 在 Mac 上把全套工作流抓下来，再拖进界面
scp a@192.168.100.2:/home/a/ai-work/minimax-h3-turing/workflows/*.json ~/Downloads/
```

### 3.2 打开后只需要配 3 处

| 节点 | 字段 | 说明 |
|---|---|---|
| `[6] MiniMaxH3ImageToVideo` | `prompt` = `__H3_PROMPT__` | **必改**。三段式写法：主体 → 动作/运镜 → 环境与光线 |
| `[10] RandomNoise` | `noise_seed` | 成片档同 seed 可复现；换 seed 换构图 |
| `[20] LoadImage`（仅 i2v） | `image` = `__H3_FIRST_FRAME__` | **必换**，点开上传首帧图 |

保持默认的：`width 640 / height 352 / length 124`（≈5.2s @24fps）、`[9] steps=4`（turbo 4 步）。
要 1080p 成片不要直接改分辨率——640×352 生成 → 抽帧 → RTX VSR 超分（手册结论）。

### 3.3 出片在哪 / 怎么看

服务端落地：`ComfyUI/output/video/<filename>`；`demo` 命令会再拷一份到 `--out` 目录。
Mac 浏览器直接播放（不用下载）：

```
http://192.168.100.2:8188/view?filename=H3Lite_00001_.mp4&subfolder=video&type=output
```

---

## 4. 已知坑（本机踩过，改环境前先读）

### 4.1 triton JIT 缺 Python.h → 文本编码器 2.4s 崩 ✅ 已绕过

- **症状**：提交后 2.4s 报错，node `MiniMaxH3ImageToVideo`，栈底是
  `gcc … triton/backends/nvidia/driver.c … fatal error: Python.h: 没有那个文件或目录`
- **根因**：torch 2.14 把 outer-product 形状的 bmm（`precompute_freqs_cis` 里那一下）路由到
  `torch.ops._native.bmm_outer_product` 的 triton 实现 → triton JIT 现场编译 `cuda_utils` 需要 Python 头文件 →
  本机**没装 python3.10-dev**（全盘无 Python.h）
- **最小复现**：`a=randn(2,128,1,cuda); b=randn(2,1,4096,cuda); a@b` → 默认必崩
- **修法（当前采用）**：`TORCH_DISABLE_NATIVE_JIT=1`（torch 官方 kill switch，不注册 DSL 实现，回退 eager，0.02s 出结果）
- **根治（可选）**：`sudo apt install python3.10-dev`，然后可从启动脚本里去掉这行。Turing 上 triton JIT 对 H3 路线无收益
- **注意**：`TORCH_DISABLE_NATIVE_JIT=1` 关掉的是 torch 全局 native DSL 加速，本路线靠 comfy_kitchen 的 CUDA 后端，不受影响

### 4.2 ComfyUI 无鉴权

`--listen 0.0.0.0` = 同网段任何人都能提交任务、浏览 `output/`、改设置。要收紧按 §2 换 `H3_LISTEN`。

### 4.3 六件套里 4 个档现在打不开

档位 `t2v-draft` / `i2v-draft` / `t2v-ultra` / `i2v-ultra`（文件 `*_t8draft_api.json` ×2、`*_pdd8_t8_api.json` ×2）引用 `MiniMaxH3BlockCacheT8`，本机没装 → 提示节点缺失、图上发红。
补：`git clone https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8 ComfyUI/custom_nodes/` 后重启（v0.33.1 免升 ComfyUI，GitHub 直连正常）。
PDD 档还要额外上 master 环境（`docs/09` 的 backport），本机是干净 v0.33.1。

### 4.4 权重残留

`models/loras/.cache/huggingface/download/` 有 ≈670 MB 的 `*.incomplete`（v1.0 LoRA 前 5 次 sha 不符留下的），可删。

---

## 5. 实测基线（复现参照）

| 项 | 值 |
|---|---|
| 命令 | `scripts/h3turing.local.sh demo --tier t2v-final` |
| tier / seed | t2v-final（compat + turbo 4step v0.1）/ 3013 |
| 耗时 | **250s**（手册参考 280s），队列正常、无重试 |
| 输出 | 5.17s · 640×352 · 24fps · 124 帧 · AAC 32kHz 立体声 |
| 文件 | `h3_demo_output/H3Lite_00001_.mp4`（同时在 `ComfyUI/output/video/`） |
| 日志健康度 | `ERROR lora` = 0（量化+LoRA 链路正常，无 docs/09 的 #16420 污染） |
| doctor | 7/8 ✓（唯一 ✗ = `t8_node`，见 §4.3） |

---

## 6. 尚未固化 / 待办

| 项 | 现状 | 影响 |
|---|---|---|
| **服务持久化** | 服务**手动启动，不设自启**（本机明确定策）：重启机器/结束会话后，手动跑一次 `scripts/h3_launch.local.sh`（约 20s 就绪） | 无 |
| git | `RUNBOOK.local.md`、`scripts/*.local.sh`、`h3_demo_output/` 均未提交 | 想留档就 `git add` 后自行提交 |
| T8 节点包 | 未安装 | 4/6 档位不可用 |
| PDD 档 | 未实测（缺 master 环境 + node pack） | — |
| i2v 档 | 未在本机实测（工作流已就位） | 首帧图需自备 |
| `h3turing` 命令 | 未装进 PATH，用 `scripts/h3turing.local.sh` 包装（纯标准库，无需装包） | 无 |
| python3.10-dev | 未安装（无 sudo 权限） | 目前靠 §4.1 绕过 |

---

## 7. 换机器/重装时的复用清单

1. 权重按 `docs/05` 的下载清单拉齐到 `models/`，逐文件核对 sha256
2. ComfyUI 用 `v0.33.1` + `custom_nodes/ComfyUI-ClipProj`；写 `extra_model_paths.yaml` 指向权重树
3. 抄 `scripts/h3_launch.local.sh`（改 `COMFY_DIR`/`PY`）
4. 装 `python3.10-dev` 就删掉脚本里的 `TORCH_DISABLE_NATIVE_JIT=1`；不装就必须留着
5. 跑 `scripts/h3turing.local.sh doctor --models-dir <models>` 全绿再开跑
6. 先跑 `demo --tier t2v-final` 对时（本机基线 250s）；明显变慢先看杀软实时防护（手册 FAQ #5）
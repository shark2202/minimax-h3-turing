# RUNBOOK.local.md · 本机部署与操作手册

> 这份是**本机（a@2080Ti-22G 服务器）专属**的落地记录，不是上游 handbook 的一部分。
> 上游原理与实测数据见 `docs/`，这里只写"在这台机器上怎么跑"。
> 最后更新：2026-10-18（新增 §1.3「共存/同时/并行」三问的结论；图生视频 i2v 实测三次跑通 + 产物断言脚本 `scripts/check_video.py`；上一条：2026-10-15 首次部署并实测跑通 t2v 成片档）

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
# ③b 图生视频（首帧锚定，实测 220-260s；首帧按 640×352 备，见 §3.5）
scripts/h3turing.local.sh demo --image h3_demo_output/i2v_first_frame.png --prompt "the swordsman slowly turns his head toward the camera, bamboo leaves drift in the wind, dawn light flickers through the mist" --seed 3013
# ③c 产物断言（i2v 专属：首帧真的被用了吗？无 GPU，秒级）
/home/a/ai-work/ComfyUI/.venv/bin/python scripts/check_video.py h3_demo_output/H3Lite_00009_.mp4 --first-frame h3_demo_output/i2v_first_frame.png --control /home/a/ai-work/ComfyUI/input/example.png

# ④ 停止服务
pkill -f "main.py --listen"
```
手册里的三张"账本"：**§1.1 模型清单**（18 件 / 59.53 GiB + 出处 + sha256）、**§1.2 显存账本**（22 GB 卡的占用实测与卸载机制）、
**§5.x 实测基线**（t2v / i2v / 剧本→出片链的耗时与产物）；验收怎么算"成了"见 **§5.4 验收四层**。

Mac 上：浏览器开 `http://192.168.100.2:8188/`（不通就换 `192.168.8.189`，装了 tailscale 用 `100.69.133.37`）。

---

## 1. 环境事实（2026-10-15 实测）

| 项 | 值 |
|---|---|
| GPU | RTX 2080Ti 22G（Turing sm_75，driver 580.178.04，可用显存 21.5 GiB） |
| ComfyUI | **v0.33.1**，`git status` 干净、无本地补丁 |
| 解释器 | `/home/a/ai-work/ComfyUI/.venv/bin/python`（torch **2.14.1+cu130**、PyAV 17.1、comfy_kitchen 可用） |
| 权重树 | `/home/a/ai-work/models`（经 `ComfyUI/extra_model_paths.yaml` 挂载），H3 9 文件 ≈29.6 GB，sha256 全部核对通过；**全机 18 个模型的统一清单见 §1.1** |
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

### 1.1 模型清单与下载（断点续传）

全机 **18 个模型 / 59.53 GiB**（H3 9 + SDXL 1 + 27B 6 + 9B 1 + chat 模板 1）统一登记在
`models/MANIFEST.tsv`（`sha256 | 字节数 | 目标路径 | 主源 | 备用源 | 角色`），校验与下载都用 `scripts/fetch_models.sh`：

| 处 | 文件 | 大小 | sha256（前 12 位；全值见清单） | 角色 |
|---|---|---:|---|---|
| `ai-work/models/` | `diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | 11.68 GiB | `01aa7b92c007` | H3 主模型 W4A8 mixed（t2v/i2v 共用） |
| `ai-work/models/` | `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` | 4.88 GiB | `54bd5144df0b` | H3 文本编码器（工作流里用别名 int4_convrot） |
| `ai-work/models/` | `vae/minimax_h3_video_vae_fp16.safetensors` | 4.85 GiB | `7c1f131492e7` | H3 视频 VAE |
| `ai-work/models/` | `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.56 GiB | `8e505d95dd15` | H3 音频 VAE（§4.5 的 NaN 就出在这一路） |
| `ai-work/models/` | `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p…safetensors` | 1.82 GiB | `c396a9a06f58` | 4 步 Turbo v1.0（成片档） |
| `ai-work/models/` | `loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1…safetensors` | 0.29 GiB | `3a069f26fbc3` | 4 步 Turbo v0.1（compat 工作流实际引用） |
| `ai-work/models/` | `loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors` | 1.61 GiB | `71347e272513` | PDD 8 步（t2v-ultra，需 master 环境） |
| `ai-work/models/` | `loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors` | 1.61 GiB | `d7cd9480c000` | PDD 8 步 · Ref2VA（i2v-ultra） |
| `ai-work/models/` | `clip_projections/mmh3-4b-ClipProj-celeb-mlp.safetensors` | 0.28 GiB | `275b38999127` | ClipProj 投影头 |
| `ai-work/models/` | `checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` | 6.62 GiB | `c9e3e68f89b8` | SDXL 出图（§3.4；主源 ModelScope） |
| `ai-work/models/llm/` | `Q2-LynnStyle/Qwen3.8-27B-EfficientThink-SimPO-Q2-LynnStyle.gguf` | 12.11 GiB | `8a84f7ef93b0` | 27B 目标（config.sh 默认） |
| `ai-work/models/llm/` | `Q2-LynnStyle/dflash2-qwen38-27b-Q4_K_M.gguf` | 1.06 GiB | `e83676f81b66` | 27B 草稿（DFlash2，depth 6） |
| `ai-work/models/llm/` | `Q2-LynnStyle/dflash2-qwen38-27b-Q8_0.gguf` | 1.92 GiB | `1086ea5d44e9` | 27B 草稿备选（多占 ~1 GiB） |
| `ai-work/models/llm/` | `Q2-LynnStyle/mmproj-Qwen3.8-27B-Q4_K_M.gguf` | 0.49 GiB | `0d22c439a59f` | 27B 视觉投影（可选，放主机内存） |
| `ai-work/models/llm/` | `Q2-LynnStyle/mtp-Qwen3.8-27B-Q4_0.gguf` | 1.56 GiB | `051a1764cff8` | MTP 头（Q2 目标无 nextn → 不可用，留档） |
| `ai-work/models/llm/` | `Q2-LynnStyle/mtp-Qwen3.8-27B-Q8_0.gguf` | 2.95 GiB | `cbf60a0c48b4` | 同上，留档 |
| `ai-work/models/llm/` | `Qwen3.8-9B-heretic/Qwen3.8-9B-heretic-uncensored.Q4_K_M.gguf` | 5.24 GiB | `8eb51ad276c3` | 写剧本的小模型（heretic 去审查；config.small.sh 默认） |
| `ai-work/models/llm/` | `chat_template.jinja` | 27 KB | `c47c82b05447` | Qwen 聊天模板 v22.4（两个启动器都引用） |

**合计 18 个文件 / 59.53 GiB**（H3 9 件 27.6 GiB + SDXL 1 件 6.62 + 27B 6 件 20.09 + 9B 1 件 5.24 + 模板）。
上表与 `models/MANIFEST.tsv` 同源；§1 上方那段 9 行手写块是 10-15 下载时的原始核对记录，数值与此表一致，留作历史痕迹。
- **2026-10-18 归集**：全机模型统一到 `/home/a/ai-work/models/`——ComfyUI 那五棵树按类型放根下（原地未动），
  LLM 侧放进新增的 `llm/`：`llm/Q2-LynnStyle/`（27B 六件）、`llm/Qwen3.8-9B-heretic/`（9B）、`llm/chat_template.jinja`。
  旧路径 `/home/a/models/` **已删除**；两个工程的配置与文档（`config.sh`、`config.small.sh`、`scripts/download-models.sh`、
  `docs/MODELS.md`、`docs/SMALL_MODELS.md`、`docs/DEPLOY_UBUNTU.md`、本手册、`models/MANIFEST.tsv`）全部同步改完，
  ComfyUI 侧 `extra_model_paths.yaml` 本来就指向 `ai-work/models`，无需改动。
全值 sha256、字节数、主源/备用源都在清单里；`scripts/fetch_models.sh --check` 会逐行核对字节数 + sha256。

### 1.2 显存账本与卸载机制（22 GB 卡）

实测账本（MiB）：

| 时刻 | 卡上占用 | 说明 |
|---|---:|---|
| 无模型 | 168–292 | 系统/显示 + 各进程 CUDA 上下文 |
| LLM 就位（9B Q4 + 64K q8_0 KV） | **6808** | 权重 5.24 GiB + KV ~1.35 GiB + workspace |
| H3 常驻 / 采样峰值 | 17444 / **18926** | DiT + 动态装载的 VAE/TE（日志 `4965MB Staged`） |
| **两者同住的峰值** | **25734** | 超 22528 → **任何时刻都不能同时在场** |

卸载三层（"腾显存"能成功的全部原因）：

1. **进程内丢引用**：`unload_all_models()` 其实就是 `free_memory(1e30, device)`（`comfy/model_management.py:2063`）
   → 对每个 `ModelPatcher` 调 `detach()`（`comfy/model_patcher.py:1295`：`eject_model()` + 补丁搬到 offload 设备 +
   `unpatch_model(unpatch_weights=True)`）→ 对象变 `real_model() is None`，`cleanup_models()` 弹表 + `gc.collect()`
2. **把块还给驱动**：`soft_empty_cache()`（`comfy/model_management.py:2045`）= `torch.cuda.synchronize()` +
   `torch.cuda.empty_cache()` + `ipc_collect()`。**只有这一步做完，别的进程才看得见那些空闲**——只做第 1 步，
   torch 的缓存分配器仍替自己留着块。实测落差 **17444 → 168 MiB**
3. **ComfyUI 自己的预留**（不信任 OS）：`--reserve-vram 2.5` → `EXTRA_RESERVED_VRAM`（`:853`），
   `minimum_inference_memory()` 永不动用最后 2.5 GB；`--vram-headroom 0.5` 交给 `comfy_aimdo` 做显存压力管理；
   0.33 的**动态显存装载**按模型声明的 `get_dynamic_vram__units()` 从主机内存流式上位
   → 22.5 GB 卡实际可用 ≈19.7 GB，H3 峰值 18926 MiB 时仍余 ~3.6 GB

跨进程只有一条路：**杀进程**。外部无权释放别人的 CUDA 上下文（`cudaFree` 只作用于自己的上下文），
SIGTERM 后驱动销毁上下文 → 实测 **6808 → 260 MiB**（节点等 1.5 s 就是给这个）。

顺序靠**图内依赖**保证（节点是采样器的上游 → H3 必然在节点返回之后才装载），不靠 sleep；返回前用
`torch.cuda.mem_get_info()` 读**设备级**空闲并断言 ≥19700 MiB，不够就报错，而不是 5 分钟后收 CUDA OOM
（`foreign_server=fail|stop|ignore` 决定对"别人起的服务"怎么办）。

不能同住时的替代路径与代价：LLM 跑 CPU（`-ngl 0`，9B 28 线程几 tok/s，5 镜剧本 2–4 分钟）／换 4B（≈2.5 GB，
理论上剩 19.7 GB ≥ 18.9 GB 峰值，但只剩 0.8 GB 余量，撞 VAE 尖峰就翻车）／半卸载（省不下多少还付 PCIe 往返）。

**不保证**：有引用的张量（latent/输出/cuDNN workspace）不释放，所以腾完也不是 0；断言是某一时刻的采样、
不是持续监控；VAE 解码尖峰归 ComfyUI 自己预算；2080Ti 没有 MPS/ECC 隔离，多进程共享靠自觉。

```bash
cd /home/a/ai-work/minimax-h3-turing
scripts/fetch_models.sh --list      # 清单 + 本机状态（不哈希，秒级）
scripts/fetch_models.sh --check     # 逐文件校验字节数 + sha256（全量读盘，1-2 分钟）
scripts/fetch_models.sh             # 缺什么下什么；已就位且校验通过的跳过
scripts/fetch_models.sh --only 9b --jobs 12 --mirror hf-mirror
```

- **断点续传**：分片落在 `<目标>.parts/`，每片每次请求 `Range: bytes=<续传起点>-<片尾>` 并追加写入；
  中断后**重跑同一条命令**即接着下。装前必须字节数 + sha256 都对（不符 exit 2，分片保留可重跑）
- `--mirror hf-mirror|official` 做域名互换（本机 hf 直连不通，清单里主源本就是 hf-mirror）；
  SDXL 那条主源是 ModelScope——本机实测它比 hf-mirror 快
- **纪律**：下载期间别渲染（§4.5）；检测到 ComfyUI/llama-server 在跑会大声警告
- 2026-10-18 复核：18 个 sha256 与下载时记录**逐个一致**（H3 九件另见 `models/h3_all_verified.sha256`，
  27B 六件见 `qwen3.8-2080ti-256k/docs/MODELS.md`，9B 见其 `docs/SMALL_MODELS.md`）；
  续传逻辑有独立自检：120 MiB 掐断后，第二次只补缺的 75.0 MiB 且 sha256 一致
- **`--check` 全量复核过一遍：18/18 OK，exit 0，用时 6m18s**（单线程 sha256，60 GiB → 约 160 MB/s）。
  这才是"清单可复现"的实证：任何一件权重被换过、截断过、或下载时被污染，这一条命令都会指出是哪一行（exit 1 = 缺/长度不符，2 = sha 不符）。
- **归集后复验（2026-10-18）**：`--check --only llm/` 对搬走的 8 件重新哈希 → **8/8 OK，exit 0，2m33s**（26 GiB）。
  同一天还实测：27B（19810 MiB）与 9B（6776 MiB）都能从新路径加载并生成，ComfyUI 剧本→出片整链跑通
  （`H3Lite_00014_.mp4`，节点日志显示 `starting llama-server` 走的就是新路径）。所以"搬家"是零风险操作：
  同一文件系统内 `mv` 只改目录项，内容不动；真被换过/截断过，这一条命令会指出是哪一行。

### 1.3 同一张卡上「共存 / 同时 / 并行」三种问法

| 问法 | 结论 | 依据 |
|---|---|---|
| 两套环境同机共存 | ✅ 已实测（§5.3，`H3Lite_00012/13/14_.mp4`） | 串行错峰 + 数字断言，不靠自律 |
| LLM 与 H3 同时驻留显存 | ❌ 25734 > 22528 MiB，物理装不下 | §1.2 账本 |
| 两个模型真并行推理 | ❌ 同上，单卡无解 | 替代路线及代价见 §1.2 末段 |

- **错峰机制长在图依赖里，不靠 sleep**：`QwenScript` 节点先卸载 ComfyUI 模型缓存 → 拉起 llama-server（6.8 GB）→ 出剧本 → **只杀自己拉起的那个进程**（按 `TargetModel` 路径匹配，不误杀）→ 采样器进场前断言设备级空闲 ≥19700 MiB。
  源码 `comfy_nodes/ComfyUI-QwenScript/qwen_script.py`（文件头注释即协议）；机制另见 qwen 仓库 `docs/SMALL_MODELS.md` 工作流 B
- **`llamacpp` 与 `kvmem-llamacpp` 之间也不能同时**：两者都绑 `127.0.0.1:8080`，且 KVMem 的 256K 逻辑上下文照样要吃 GPU working set（`KVMemBudget` / `GenerationReserve`）——是**二选一**，不是"再加一个"（qwen 仓库 `docs/LAUNCH_PARAMS.md`）
- **内存（33.5 GB）是唯一真正能共享的一层**：权重走 mmap / 动态装载，两边不同时把所有页拉起就够用（H3 动态装载 staging 13.5 GB 落在主机内存 + LLM 权重 5.24 GiB 走 mmap 页缓存 ≈ 19 GB）。
  ⚠️ 但**这一层没有断言**：抢页缓存的退化路径是 swap（变慢），不会像显存那样当场报错——所以"显存断言通过"不等于"内存没问题"

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
- **起服务要用"脱离会话"的方式**：`setsid nohup scripts/h3_launch.local.sh >/dev/null 2>&1 < /dev/null &`。
  2026-10-17 踩过：把服务挂在带超时/带回收的包装里起，到点包装把**服务本身**杀了（渲染完 2 小时后服务凭空消失）。
  一句话：服务是常驻进程，别让它挂在任何"会自己结束"的东西下面。
- 日志：`ComfyUI/h3_server.log`（`h3turing bench --log-path` 要靠它解析 T8 命中数）

---

## 3. Mac 侧操作

### 3.1 工作流在哪

已放进 ComfyUI 用户目录，**界面左侧 Workflows 面板直接可见**，无需下载：

```
ComfyUI/user/default/workflows/h3_w4a8_t2v_compat_api.json   文生视频·成片档（250s）
ComfyUI/user/default/workflows/h3_w4a8_i2v_compat_api.json   图生视频·成片档（~420s）
ComfyUI/user/default/workflows/sdxl_juggernaut_t2i_api.json   文生图·SDXL Juggernaut XL v9（本机新增）
ComfyUI/user/default/workflows/h3_w4a8_t2v_qwenscript_api.json  t2v 成片档 + 本地 Qwen3.8 写剧本节点（§3.6）
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

### 3.4 文生图（本机新增 · SDXL）

与 H3 无关的独立路线，纯出图用：`sdxl_juggernaut_t2i_api.json`

- 权重：`models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors`（7.10 GB，RunDiffusion/Juggernaut-XL-v9，目标 sha256 `c9e3e68f…`）
—— **状态：✅ 已落位并验证通过（2026-10-18 01:09 完成）**：`7105348188` 字节、sha256 `c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca`（独立复算过一遍），
  ComfyUI `ckpt_name` 列表实时可见（无需重启），出图验证：**18.1s** 一张 1024×1024（服务端 17.59s）、产物 `h3_demo_output/Juggernaut_00001_.png`。
  浏览器看：`http://192.168.100.2:8188/view?filename=Juggernaut_00001_.png&subfolder=image&type=output`
  出图命令（可选，CLI 里没有图片档，直接用自带客户端）：`/home/a/ai-work/ComfyUI/.venv/bin/python /tmp/h3_img_verify.py`（该脚本是 2026-10-17 的临时件，需要的话把逻辑抄进 `python/` 再删）
  重下/换源的跑法：`bash models/ranged_get.sh juggernaut "<URL>" 7105348188 c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca /home/a/ai-work/models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors 12`
  - **首选源 = ModelScope**：`https://modelscope.cn/api/v1/models/RunDiffusion/Juggernaut-XL-v9/repo?Revision=master&FilePath=Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors`
    单连接实测 1.31 MB/s，**且 CDN 路径是内容寻址的**（路径里嵌着期望 sha256 `…/lfs-objects/c9/e3/e68f89b8…`）→ 能确认是同一个 LFS 对象。
    备选 hf-mirror 同路径单连接仅 51 KB/s（8 连接聚合 0.5-2 MB/s，且被反复掐连接）。两边都是**按客户端限总带宽**，加连接数不再涨。
  - 装完脚本会自校验 sha256 才落位，不匹配不落位。**下载期间不要同时跑 H3 渲染（§4.5）**；要渲染就先停下载。
- **⚠️ 停止/重启下载必须连 curl 子进程一起杀**（2026-10-17 用 5.3 GB 换来的教训）：
  每个分片由一个 `curl` 子进程写，只 `pkill -f ranged_get` 会留下 curl **继续以 append 模式写**，
  与新进程的 curl 同时写同一文件 → 内容交错污染。而且 `truncate` 会把长度修回正确值，
  **大小检查看不出问题，只有最终 sha256 能发现**（脚本 exit 2 = sha 不符，`exit 1` = 还缺数据）。
  正确做法：先 `pkill -f "ranged_get.sh <tag>"`，**再** `pkill -f "curl.*<文件名片段>"`（或用 `[r]` 技巧避免自匹配），确认干净再重启。
  重启换新连接确实能提速（实测 0.52 → 1.03 MB/s），但必须干净重启。
- 只需改 `[2] CLIPTextEncode` 的 `text` = `__PROMPT__` 和 `[5] KSampler` 的 `seed`
- 默认 1024×1024 · dpmpp_2m + karras · 30 步 · CFG 5（模型卡推荐区间 30-40 步 / CFG 4-7）
- 出图落 `ComfyUI/output/image/`，命名 `Juggernaut_00001_.png`

### 3.5 图生视频（i2v · 本机已实测三次）

> ✅ 2026-10-18 跑通：成片档 i2v 三次实测、同 seed 逐字节可复现、换首帧确实换片子（证据在 §5.2）。

与 t2v 只差一处：`h3_w4a8_i2v_compat_api.json` 多一个 `[20] LoadImage` → `[6].first_frame`；
CLI 加 `--image`（首帧走 `/upload/image`，图里填**服务器端文件名**，不是本地路径）。

```bash
scripts/h3turing.local.sh demo --image h3_demo_output/i2v_first_frame.png \
  --prompt "the swordsman slowly turns his head toward the camera, bamboo leaves drift in the wind, dawn light flickers through the mist" \
  --seed 3013 --out ./h3_demo_output
```

- **首帧几何要按 640×352 备**：`nodes_minimax_h3.py` 里是 `_resize(..., crop="disabled")` = **直接压扁不裁切**，
  方形图进 16:9 会被拉变形。本机已切好 `h3_demo_output/i2v_first_frame.png`
  （Juggernaut 那张 → 中心裁 16:9 → 640×352，竹林剑客）。
- 服务端耗时（种子 3013）：**259.9s** 冷启 / 223.2s 换首帧 / **219.7s** 热复跑；手册参考 420s，本机更快。
- 三层验证，少一层就会漏判：
  1. **接线**（0 GPU，3s）：`/home/a/ai-work/ComfyUI/.venv/bin/python -m unittest discover -s python/tests` → 37/37。
     `test_i2v_end_to_end` 断言 multipart 上传里含首帧字节、`graph[20].image` = 服务器返回名（不是本地路径）、
     `graph[6].first_frame=["20",0]`、t2v 图不含节点 20。**它只证明接线对，不证明模型用了那张图。**
  2. **真渲染**：上面那条命令。
  3. **产物断言**（无 GPU，秒级）：`scripts/check_video.py`（PyAV+numpy+PIL，用 ComfyUI venv 的 python 跑）
     ```bash
     /home/a/ai-work/ComfyUI/.venv/bin/python scripts/check_video.py \
       h3_demo_output/H3Lite_00009_.mp4 \
       --first-frame h3_demo_output/i2v_first_frame.png \
       --control /home/a/ai-work/ComfyUI/input/example.png
     ```
     四条断言：帧几何 / **不全黑**（存在≠成功，§4.5 的全黑片也照样躺在盘上）/ 音轨峰值且无 NaN / **首帧吻合**。
     后者的两条判据要分清：
     - `first_frame` = **保真度**（绝对 r，默认 ≥0.6）：输出首帧是不是那张图
     - `attribution` = **归因**（own r ÷ 最好对照 r ≥ 3）：那张图到底有没有被当输入用
     标定：主跑 r=**0.998**（×24.5）；拿 t2v 产物当 i2v 检则两条都 FAIL（r=0.010、×0.0）——
     即脚本不会"什么都不看就 PASS"。
- ⚠️ **首帧保真度不是无条件的**：换成与提示词冲突的卡通图（`example.png` + 剑客提示词）那次，
  输出首帧是**半透明叠影**（r=0.206，肉眼见 `h3_demo_output/i2v_00010_frame0_ghost.png`）：
  归因仍成立（×6.8），但 4 步 turbo 档在"图与提示词打架"时会把两者混起来。
  所以**别拿单次绝对 r 判 i2v 好坏**——判"用没用图"看归因；要首帧忠实就换个和提示词一致的图重跑。
- Mac 界面同路：导入 workflow → `[20] LoadImage` 里上传首帧，节点上出现缩略图 = 图已到服务端。

### 3.6 剧本（本地 Qwen3.8-9B · ComfyUI 节点）

2026-10-18 落地：**小参数 Qwen3.8 当场写剧本，同一张卡直接出片**——同一次排队跑完，不用手工倒提示词。

- 权重：`/home/a/ai-work/models/llm/Qwen3.8-9B-heretic/Qwen3.8-9B-heretic-uncensored.Q4_K_M.gguf`
  （5.24 GiB，sha256 `8eb51ad2…c96510f`，**heretic 去审查版**——按"优先无审查"选的；
  选型、下载、实测全在 `~/ai-work/qwen3.8-2080ti-256k/docs/SMALL_MODELS.md`）
- 节点：**源码在本仓库** `comfy_nodes/ComfyUI-QwenScript/`（同目录 README 写了显存协议），用
  `bash scripts/install_qwen_node.local.sh` 符号链接进 `ComfyUI/custom_nodes/`；UI 里叫
  **Qwen3.8 Script (local LLM)**（分类 MiniMax H3）。ComfyUI 里是链接不是拷贝 → 改源码重启即生效，不会两份漂移
- 工作流：`ComfyUI/user/default/workflows/h3_w4a8_t2v_qwenscript_api.json`
  （= t2v 成片档 + 该节点，`[6].prompt` 由字面量改成接节点 `prompt` 输出）
- **22G 装不下两套，所以节点自己错峰**：先卸载 ComfyUI 的模型缓存 → 拉起 llama-server（占 **6.8 GB**）
  → 生成剧本 → **只杀自己拉起的那个进程** → 采样器拿到整卡。整链实测见 §5.3
- `reuse_script=true` + `script_out=<前缀>`：同一份 `.shots.json` 后续排队直接复用，
  改 `shot` 序号逐镜出片，不必反复问模型
- 只写剧本、暂不渲染（不占 ComfyUI、更快）：
  ```bash
  cd ~/ai-work/qwen3.8-2080ti-256k
  CONFIG_PATH=config.small.sh scripts/start-llama.sh
  python3 scripts/screenplay.py --brief "竹林剑客，黑色电影，复仇前夜；冷调晨雾，手持镜头" \
      --shots 5 --seed 3013 --out ~/ai-work/minimax-h3-turing/h3_demo_output/screenplay_bamboo
  pkill -f "[Q]wen3.8-9B-heretic-uncensored"   # 出片前腾显存；[Q] 是为了不让 pkill 匹配到自己
  ```
- 速度：**70 tok/s**（9B Q4_K_M + ngram-mod，无草稿），5 镜剧本 ≈ 15s
- ⚠️ 重装/换机：节点与工作流**已进本仓库**，一条命令重装（§7 第 7 步）；**权重不进 git**
  （`~/ai-work/models/llm/Qwen3.8-9B-heretic/`，字节数与 sha256 见 SMALL_MODELS.md，拉取见 §7 第 8 步）

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

### 4.5 并发下载负载 → 采样出 NaN（视频全黑 + 音频 NaN，SaveVideo 崩）⚠️ 已定位

- **症状**：提交后 2-4 分钟，`SaveVideo` 抛
  `av.error.ArgumentError: Invalid argument: 'avcodec_send_frame()' returned 22`
  （栈底是 `comfy_api/latest/_input_impl/video_types.py` 的 `output.mux(audio_stream.encode(frame))`）。
  同一次跑在 `output/video/` 留下的那个十几 KB 小 mp4：**视频轨全黑**（解帧 `mean=0.00 / max=0`）、**无音轨**。
  ≈236s 的 GPU 时间全白烧——报错发生在最后一步（封装），前面采样/解码全跑完了。
- **判据（怎么确认是 NaN 而不是别的）**：把 mux 那一步单独复现，**只有波形含 NaN 才吐这个错**：
  正常 165334 样本 ✓、1 样本 ✓、单声道 ✓、零长度 → MemoryError（另一种错）、NaN → 正是 `returned 22`。
  所以看到 `returned 22` 就可以直接判定"音频数据里有 NaN"。
- **触发条件（2026-10-17 本机实测对照）**：

  | 条件 | 结果 |
  |---|---|
  | 渲染期间并发 8 路分片下载（31GB 内存的机器上还压着 ~20GB page cache） | **3/3 全 NaN**（124 帧 ×2、60 帧 ×1） |
  | 停掉下载 + 全新进程，同 seed 同长度 | **2/2 正常出片**，两次帧统计完全一致 |
  | 对照里再换掉 `--reserve-vram/--vram-headroom/--disable-pinned-memory` | 与带参数结果一模一样 → **启动参数不是原因，别去动它** |

- **为什么只可能是运行时数值**：采样起点是**全零 latent**（`_empty_av_latent` 用 `torch.zeros`，噪声由 `RandomNoise` 给），
  输入不脏；权重 sha 与图 JSON 都与 10-15 成功那次一致，ComfyUI/PyAV 版本未变。
  坏的是模型前向的运行时数值（翻页/搬运权重那条路），不是数据。
- **处置**：出片前确认机器没在跑下载或其它吃内存的活（`free -g` 看 available、`uptime` 看 load）；
  撞上了别在原进程反复重跑，**停负载 + 重启服务**再跑。
- **成片自检**（一条命令）：用 `ComfyUI/.venv/bin/python` + PyAV 解帧看 `mean/std/max`，
  全 0 就是这一条；正常片子首帧 `mean≈73 / max=255`（帧统计与 seed 一起可作回归基线）。

### 4.6 下载器踩过的两个坑（写 `fetch_models.sh` 时实测）✅ 已修

- **`curl -C -` 不能和 `--range` 联用**：实测这组合会把 range 的**结尾丢掉**，发成 `Range: bytes=<offset>-`
  （直到 EOF）。后果不是报错而是**静默超量**：3 片各从断点下到文件尾，一次续传传了 315 MiB（应 75 MiB），
  最后 sha 不符 exit 2。正确做法是**自己算续传起点**：`--range $((base+have))-$end` 然后追加写入；
  另外别加 curl `--retry`（重试会从 range 头重来，而我们已经 append 过一段 → 重复写入）。
- **`read` 配 `IFS=$'\t'` 存不住空字段**：TAB 属于 IFS 空白，连续 TAB 会被折成一个，中间的空列被吞掉
  （`a⇥b⇥c⇥d⇥⇥e f` 读出来是 `alt="e f"`、`role=""`）。所以清单分隔符用**竖线** `|`，
  并让下载器对"段数不足"直接报错退出，而不是拿错字段去下文件。
- 两个坑都是**自检先于上线**抓到的：`/tmp/resume_test.py` 起一个会记账的本地 HTTP 服务，掐断续传后
  断言"第二次只补缺的字节"（现在 5/5 PASS）。

### 4.7 运维三连坑（本机实测）✅ 已固化进脚本/手册

- **`pkill -f "<长字符串>"` 会匹配到它自己**：那次我把自己那条 bash 命令杀了（exit 143，命令只跑到一半就没了）。
  用方括号技巧 `pkill -f "[Q]wen3.8-9B-heretic"`（模式里含 `[Q]`，不再自匹配），或先 `pgrep -f` 看清再杀。
  §7 与本文里的 kill 命令都写成 `[Q]`/`[m]ain.py` 形式。
- **长跑辅助进程必须 `setsid`**：`nohup … &` 起的显存采样器被会话回收掐掉两次（159 s、313 s），日志停在半路，
  看着像"脚本自己退了"。§0 那条"别把常驻进程挂在会自己结束的东西下面"同样适用于采样器/监视器：
  `setsid nohup … >/dev/null 2>&1 < /dev/null &`。
- **`set -u` 下 `local a=1 b=$((a+1))` 会炸**（`a: 未绑定的变量`）：整条命令的所有展开发生在 `local` 赋值**之前**。
  拆成两行。这个坑是下载器自检抓到的——当时 60 GB 的高压路径传了 **0 字节**，而不是悄悄算错。

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

### 5.1 2026-10-17 复跑核验 + 速度换算

同 seed（3013）同图重跑，产物与 10-15 基线**逐字节相同** —— 这是"没坏"的最强证据，
也说明 compat 档的复现性是真的：

```
365ab00c45ae090eace62f899a6fa18ab10bb180c37e6b0a43184eb6db0a8bd7  h3_demo_output/H3Lite_00001_.mp4  (10-15)
365ab00c45ae090eace62f899a6fa18ab10bb180c37e6b0a43184eb6db0a8bd7  h3_demo_output/H3Lite_00007_.mp4  (10-17)
365ab00c45ae090eace62f899a6fa18ab10bb180c37e6b0a43184eb6db0a8bd7  h3_demo_output/H3Lite_00008_.mp4  (10-18，图片模型落位后再次复跑)
```

10-17 那条：服务端 `Prompt executed in 254.88s`，客户端墙钟 260s，`ERROR lora` = 0，
124 帧 / 5.17s / 640×352 / 24fps / AAC 32kHz 立体声，音轨峰值 0.949（有真实声音，非静音）、无 NaN，
解帧均值得 `mean≈73→106 / std≈72→75 / max=255`（全黑故障是 mean=0/max=0，一眼可分）。
**图片模型（SDXL Juggernaut XL v9）同批验证（10-18）**：1024×1024 / 30 步 / dpmpp_2m+karras，
服务端 `Prompt executed in 17.59s`（客户端墙钟 18.1s）→ **每张 1024² 出图 ≈ 18s**，
产物 `h3_demo_output/Juggernaut_00001_.png`（1.55MB，extrema 0-255、stddev 67/60/50 = 真图非全黑）。
即：图片路线的单张成本比视频路线低两个数量级（≈18s vs ≈250s），H3 与 SDXL 可同进程按需换入换出。

**「1 秒成片要跑多久」**（t2v-final，640×352，4 步，服务端执行时间）：

| 成片时长 | 帧数 | 服务端执行 | 每 1s 成片 | 备注 |
|---|---|---|---|---|
| 5.17s | 124 | 254.9s | **49.3s** | 冷进程第 1 镜（含 ~41s 模型装载） |
| 3.04s | 73 | 144.5s | **47.5s** | 冷进程第 1 镜 |
| 3.04s | 73 | 102.8s | **33.8s** | 热：同进程第 2 镜，模型已驻留 |

- 一条 5s 镜头 ≈ **4.2-4.3 分钟**；短视频单价更贵（固定开销摊不薄）。
- 采样 4 步占大头：124 帧时 4 步 ≈ 184s（46.05s/步，第 1 步含初始化 87.06s），其余 ~71s 是
  文本编码 + 视频/音频 VAE 解码 + 封装。
- 成本对帧数**超线性**：帧数 ×2.07（60→124）时步时 ×2.2。

**帧数会被吸附到 17k+5 网格**（`comfy_extras/nodes_minimax_h3.py: align_frame_count`：
`while n % 17 != 5: n += 1`，**只上不下**）：合法长度是 5/22/39/56/73/90/107/124/141…。
实测请求 60 帧 → 实际出 73 帧（3.04s）。124 正好在网格上，所以成片档不会被改。

### 5.2 2026-10-18 i2v 三次实测（首帧锚定 · 证据）

命令同 §3.5，三次同为 seed 3013、同提示词，**只换"首帧"这一个变量**：

| 跑次 | 产物 | 首帧 | 服务端执行 | 客户端墙钟 | first_frame（保真度） | attribution（归因） |
|---|---|---|---|---|---|---|
| R1 | `H3Lite_00009_.mp4` | 剑客图 640×352 | 259.94s | 260s | r=**0.998** / MAE 0.020 ✅ | ×24.5 ✅ |
| R2 | `H3Lite_00010_.mp4` | `example.png` 768×768 | 223.16s | 230s | r=0.206（叠影）❌ | ×6.8 ✅ |
| R3 | `H3Lite_00011_.mp4` | 剑客图（与 R1 同参） | 219.67s | 401s（含 ~180s 排队） | r=0.998 ✅ | ×24.5 ✅ |

- **复现性**：R3 与 R1 **逐字节相同** ——
  `ba3998bb055fe34a07a5121438fdedc0bc387889cc6b9a7f41f08c0fa29835c6`（两条都是它）
  → i2v 继承了 compat 档的同 seed 可复现性（§5.1 的同一手法，可以放心当回归基线）。
- **对时别拿墙钟比**：R3 的客户端墙钟 401s 里有 ~180s 是排队（它和 R2 是一起提交的，客户端日志能看到
  前 150s 是 `1 running, 1 pending`）——它自己的服务端执行仍是 219.67s。单跑一条时间就是服务端耗时。
- 顺带确认队列串行是对的：两次同 seed 重跑（R1/R3）在**同一进程里跑出了逐字节相同的产物**，
  说明"排队不影响结果"，也说明热进程（R3 219.7s vs R1 259.9s）省下来的正是模型装载那 ~40s。
- **换首帧 → 换片子**：R1 与 R2 的首帧之间 r=0.031、全片逐帧平均差异 **0.287**（同 seed 同提示词）。
- 2×2 交叉矩阵（产物首帧 × 首帧图，两图都先按 640×352 压扁再比）：

  | | imgA（剑客） | imgB（example） |
  |---|---|---|
  | video R1 fp0 | **0.998** | 0.041 |
  | video R2 fp0 | 0.030 | **0.206** |

  对角线明显高一档 = 图像确实是输入，不是装饰。
- 肉眼：`h3_demo_output/i2v_00009_strip.png`（第 0/31/62/93/123 帧横排）显示剑客从背对镜头逐步回头面向镜头，
  斗笠/刀/竹林/晨雾晨光全程保持；R2 的首帧叠影见 `i2v_00010_frame0_ghost.png`。
- 其余指标与 t2v 一致：124 帧 / 640×352 / 24fps / 5.17s / AAC 32kHz 立体声；帧 mean 0.46-0.56
  （t2v 是 0.31-0.44）；音轨峰值 R1 0.273、R2 1.025；两次都无 NaN；
  服务端日志 `returned 22` = 0（§4.5 的 NaN 故障没出现）、`ERROR lora` = 0。

### 5.3 2026-10-18 剧本 → 出片 整链（ComfyUI 节点 · 显存实测）

同一次排队完成：`[30] Qwen3.8 Script` 现场写剧本 → `[6].prompt` 接它的输出 → t2v 成片档渲染。

| 跑次 | 节点 30（LLM 阶段） | 服务端 | 产物 | 备注 |
|---|---|---|---|---|
| 首次（无缓存剧本） | 拉起 llama-server → 698 tokens / 10.1s（**68.9 tok/s**）→ 停掉自己拉起的进程；整段 **17.7s** | 250.24s | `H3Lite_00012_.mp4` | 3 镜剧本，JSON 契约 0 违例 |
| 复用（`reuse_script=true`） | `reusing … (no model needed)` → **0.0s**，全程没起模型 | 232.10s | `H3Lite_00013_.mp4` | 节点打印 `free VRAM 21831 MiB ≥ 19700` |
| 原样重提 | —— | **0.01s** | 无新产物 | ComfyUI 执行缓存命中：**改一个输入（`seed`/`shot`）才是真复跑** |

- 两条产物都过 `scripts/check_video.py`：124 帧 / 640×352 / 24fps / 5.17s、非全黑、音轨峰值 0.446 与 0.432、无 NaN
  （`h3_demo_output/H3Lite_0001{2,3}_.mp4`）。用的就是**模型现场写的第一镜 prompt**（"A lone swordsman in black
  robes stands still in a bamboo forest at dawn, …"）。
- **显存曲线**（`nvidia-smi` 1s 采样，313 样本）：节点返回前空闲 **21831 MiB** → 装载+采样**峰值 18926 MiB** 后回落到驻留 17444 MiB；
  峰值时卡上仍余 ~3.6 GB，与 `--reserve-vram 2.5 + --vram-headroom 0.5` 的预留吻合；VAE 解码那一下没有超过采样峰值。
- 这条曲线就是节点里 `min_free_vram_mib=19700` 的来历：`18926（H3 峰值）+ 6808（LLM）= 25734 > 22528`，
  两者**任何时刻都不能同时在场**，所以"腾显存"必须在采样器进场前完成并**当场用数字断言**，否则要在 5 分钟后收 CUDA OOM。
- **复现性检查别用 sha256**：两条产物字节不同（696338 vs 696392），但**逐帧像素完全相同（124/124）**。
  原因是 `SaveVideo` 把整份 workflow graph 写进 MP4 的 `prompt` metadata —— 本轮只差 `seed 3013→3014` 和两个新 widget，
  字节就变了。所以"同 seed 是否复现"要比**帧数据**，不是比文件哈希（§5.1 那三条字节相同，是因为它们的 graph 真的一字不差）。

### 5.4 验收四层（本机通用套路：缺一层就漏判）

1. **接线**（0 GPU、秒级）：`python -m unittest discover -s python/tests`、`fetch_models.sh --check`、
   `check-config.sh`、`curl -s 127.0.0.1:8188/object_info/<节点>` —— 只证明"接对了线"，不证明有效。
2. **真跑一次**：真渲染/真生成，而且**改一个输入值**（`seed`/`shot`/`script_out`）—— 图一字不差会命中
   ComfyUI 执行缓存：`Prompt executed in 0.01 seconds`，什么都没跑（§5.3 的坑）。
3. **产物断言**（"文件存在"不算）：`scripts/check_video.py` 查帧几何 / 非全黑 / 音轨峰值与 NaN /
   首帧吻合 + 归因；日志侧 `returned 22` = 0、`ERROR lora` = 0；数值与 §5.1/§5.2 基线比对。
4. **复现 / 回归**：同 seed 比**帧数据**（不是 sha256：MP4 里嵌了 graph metadata，改一个 widget 就换字节）；
   sha256 只用在校验**权重与下载**（18 件 59.53 GiB 全绿是这条的基准）。
5. **记账**：服务端耗时、峰值显存、关键数字写回 §5.x —— 没有数字的结论不算结论。

---

## 6. 尚未固化 / 待办

| 项 | 现状 | 影响 |
|---|---|---|
| **服务持久化** | 服务**手动启动，不设自启**（本机明确定策）：重启机器/结束会话后，手动跑一次 `scripts/h3_launch.local.sh`（约 20s 就绪） | 无 |
| git（本仓库） | ✅ 已入库（`6d160a5` 等）；2026-10-18 复核时工作区干净，本次仅新增 §1.3 与本节两行。`h3_demo_output/` 与 `.pi/` 已进 `.gitignore` | 无 |
| git（qwen 仓库） | ✅ 已入库（`8576dd9` 等）：小模型支持（`start-llama.sh`/`start-kvmem.sh`/`check-config.sh`/`config.example.sh`）与剧本工具（`scripts/screenplay.py` + `screenplay.system.md`）。`config.small.sh` 按设计本地（已 gitignore），`docs/SMALL_MODELS.md` 是本地笔记（并入上游前要英译+合并） | 无 |
| ComfyUI 剧本节点 | ✅ 已进本仓库 `comfy_nodes/ComfyUI-QwenScript/`；ComfyUI 侧是符号链接（`scripts/install_qwen_node.local.sh` 幂等重装，遇真目录会拒绝覆盖） | 换机 = 一条安装命令 + 拉权重，不再有"只在本机"的孤本 |
| 模型清单 | `models/MANIFEST.tsv`（18 个模型 / 59.53 GiB / 主源+备用源+sha256 的唯一记录）+ `scripts/fetch_models.sh`（断点续传，自检 5/5） | 换机/重装照它拉权重；权重本身不进 git（`.gitignore` 只放行 MANIFEST.tsv） |
| T8 节点包 | 未安装 | 4/6 档位不可用 |
| PDD 档 | 未实测（缺 master 环境 + node pack） | — |
| i2v 档 | ✅ **已实测**（2026-10-18，§3.5 / §5.2）：成片档三次跑通、同 seed 逐字节可复现、换首帧确实换片子 | 首帧按 640×352 自备（§3.5） |
| i2v 首帧叠影 | 只证到"图与提示词打架"这一层（§5.2 R2）：未跑"卡通图 + 卡通提示词"对照，所以还不能断定成因是提示词冲突还是图像类型 | 想钉死成因再跑一次（≈220s，命令同 §3.5 换 --prompt） |
| 并发压测 · RAM 断言 | 只做了**串行错峰**实测（§1.3）；显存侧有 `min_free_vram_mib=19700` 断言，**内存侧没有任何断言**（H3 staging 13.5 GB 与 LLM 页缓存同抢，退化到 swap 不报错） | 想上"多模型同时跑"之前先补这两项 |
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
7. 剧本节点：`bash scripts/install_qwen_node.local.sh`（符号链接进 `ComfyUI/custom_nodes/` + 拷贝工作流），
   重启 ComfyUI 后 `curl -s 127.0.0.1:8188/object_info/QwenScript` 应返回节点定义（18 个输入）
8. 模型权重（**不在 git**）：用清单一次性核对/补齐 —— `scripts/fetch_models.sh --check`（18 文件 59.53 GiB），
   缺的跑 `scripts/fetch_models.sh`（断点续传；清单与用法见 §1.1）。只想补 9B：
   ```bash
   cd ~/ai-work/minimax-h3-turing && scripts/fetch_models.sh --only 9B-heretic --jobs 12
   ```
   （清单里的 URL / 字节数 / sha256 就是权威；旧的一次性脚本 `~/ai-work/models/ranged_get.sh` 仍在，
   重下 SDXL 那种单文件的跑法见 §3.4）
9. 剧本链路自检：`cd ~/ai-work/qwen3.8-2080ti-256k && CONFIG_PATH=config.small.sh scripts/check-config.sh`
   （草稿应显示 `absent (optional)`）→ `CONFIG_PATH=config.small.sh scripts/start-llama.sh` →
   `python3 scripts/screenplay.py --brief "竹林剑客，黑色电影" --shots 3 --out /tmp/t`（≈15s，70 tok/s）
# 13 · 社区加速方案调研（2026-10-19）

> **本文性质：调研 backlog，不是判决书。** 全部社区数字来自**别人的卡**（3060 / 4070 / 4090 / 5080 / 5090 / DGX Spark），步骤数与分辨率大多也不同——按手册 [§5.4 验收四层](../RUNBOOK.local.md) 的纪律，**只有本机同 seed 帧数据对拍才算数**。本机已实测项单独标注。

## 0. 一张表看完

| # | 路线 | 门槛 | 社区数字 | 本机可用性（v0.33.1 锁版） |
|---|---|---|---|---|
| A1 | kitchen **INT8 attention**（`--use-ck-attention`） | 无 | 5090 sampler **≈2×**；4090 1.34–1.79× | ✅ **kernel 已确认可用，从未启用** |
| A2 | **EasyCache**（ComfyUI 内置） | 无 | 3060 20 步 782.3→428.2s（**-45.3%**） | ✅ 节点已注册，未测 |
| A3 | TeaCache / FirstBlockCache / AdaptiveCache | 装节点 | 3060 20 步 **1.94× / 1.64× / 1.58×** | 未测（与 T8 同纪律） |
| B1 | **PDD-Acc 节点** | ComfyUI ≥0.33.0 | 官方 8 步路线，免 core 升级 | ✅ **可解锁 `-ultra`，不需 master** |
| B2 | Zironic **H3-Optimizations** | ComfyUI ≥0.33.0 | Memory Opt **1.27×**（134→106s） | 可装，代价=改动数值 |
| B3 | **comfyui-turing-utils** ⭐ | 需自编译 CUDA kernel | 无公开 H3 端到端数字 | ✅ 唯一**点名 sm_75** 的包 |
| C1 | core Sol-attn / Block Sparse / Comfy Compiler | 需升版本 | kitchen PR #117 int8 Sol **1.80×** | ❌ v0.33.1 无这些节点 |
| C2 | kijai SolAttn（Triton） | — | 仅 4090/5090 验证 | ⚠️ Triton → sm_75 待验 |
| C3 | xmarre ComfyUI-Sol-H3 | CuTe sm120/121 | — | ❌ 架构不符 |
| C4 | Spectrum | 需 30–50 步 | 4 步路线无意义 | ❌ 本机无此节点 |
| D | Turbo/FastH3/TaoMate/HyperFlow LoRA 换代 | 下载权重 | FastH3 **1.87×** vs Turbo-8step；TaoMate **2.2×** | ✅ 只换权重 |
| E1 | TE-Speed | — | 3060 实测 **0.999×**（改产物零省时） | ❌ 维持判死 |
| E2 | H3 Super-Acceleration on Turing | HF gated | 672×384 draft 4 步 → LTX-2.5 精修 3 步 | ⚠️ 见 §5 |

---

## 1. 零安装，本机可即刻验证

### 1.1 kitchen INT8 attention（最便宜的一张牌）

H3 是**单个 packed 序列**，注意力 `O(S²)` 而块内其余 `O(S)`——实测占编译块的 **41%（960×544）/ 57%（1344×768）**，画布越大占比越高；"这是 8-bit GEMM 唯一碰不到的部分"（comfy-kitchen PR #117、multimodalart 的 `h3_int8_attention.py`）。

**本机已验证内核可用**（2026-10-19 实测）：

```
comfy_kitchen 0.2.31 | torch 2.14.1+cu130 | cap (7,5)
int8_attention_is_available() -> True
```
**本机已实测（2026-10-19，t2v 成片档同 seed A/B）**：提速 **1.31×**（服务端 232.46s → 177.64s），但**画面明显软化**、0/124 帧相同、PSNR 18.21 dB → **不作产线默认**，详见 `RUNBOOK.local.md` §5.5。推论：社区 ≈2× 的数字出自 20 步环境，**4 步低 NFE 没有余量吸收 INT8 量化误差**。

代码路径也已确认可达 H3：`comfy/ldm/modules/attention.py` 里 `--use-ck-attention` → `optimized_attention = attention_comfy_kitchen_int8`，而 `comfy/ldm/minimax/model.py` 的 `DiTBlock` 正是调 `optimized_attention`。

**关键事实：本机启动脚本从未带这个 flag** → 手册现有的 280s / 160s / 420s 基线都是**不含** INT8 attention 的数字。

### 1.2 EasyCache（内置，零安装）

- 3060 12GB / 20 步：782.3s → **428.2s（-45.3%）**；再叠 SageAttention 2 → 399.7s（-48.9%）——两者**非线性叠加**（缓存跳过的正是 Sage 加速的那些步）
- ⚠️ **社区说法直接矛盾**：一侧称"EasyCache **不可**与 Turbo LoRA 组合"，另一侧给出"Turbo 4 步 + EasyCache = 420s vs 官方 20 步 1006s"。→ 本机 4 步路线必须实测
- 接线：`UNETLoader → EasyCache → BasicScheduler / BasicGuider`（两个都要接）
- 纪律与 T8 相同：**改数值、同 seed 不可复现**，草稿用

### 1.3 缓存节点对照组（3060 / 20 步 / 同 seed 实测）

| 节点 | 加速 | 判定 |
|---|---|---|
| TeaCache | **1.937×** | 最强 |
| ApplyFirstBlockCache | 1.644× | |
| AdaptiveCache | 1.581× | |
| CombinedAcceleration | 1.336× | |
| **TESpeedMiniMaxH3** | **0.999×** | **改产物、零省时** |
| SpeedCache | 崩（`AttributeError: time_shift_slope`） | |

本机 4 步路线 T8 实测 **1.75×**（280→160s）。20 步比值不可照搬，但 TeaCache 值得对拍。
踩坑提醒（同源实测）：缓存节点接 `MODEL` 后必须**同时**接到 scheduler 与 guider，只接一处会"加载成功、无报错、什么都不做"。

---

## 2. 装节点即可，**不必动版本锁**（均要求 ComfyUI ≥0.33.0）

### 2.1 PDD-Acc：`-ultra` 两档的真正解锁器

`Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc` —— 把 PDD 的 **head bank** 在自定义节点里装到 `final_layer`、按 sigma 逐步 armed，**因此不需要 core 升级**（`docs/07`/`docs/09` 认为"必须 master backport"的判断被这条路绕过）。

- 约束（硬性）：nfe **只能 4 或 8**（训练包络）；sigma shift **必须 12/3**；sampler `euler`；CFG 1.0
- **不与任何蒸馏 LoRA 叠加**（lightx2v turbo 等）；**不与 blockcache / EasyCache 叠加**；角色 LoRA 可叠
- 风险：FL2VA / Ref2VA 配错**静默出错**（带 fingerprint 校验：trunk 相对距离 0.0503、容差 0.015、cast 噪声 ~2e-3）
- 权重放 `ComfyUI/models/pdd_acc/`（本机已有 Kijai 转换版两个 PDD 文件，sha256 已核对，见 `models/MANIFEST.tsv`）
- ⚠️ **格式陷阱（本机实测 2026-10-19）**：`Kijai/MiniMax-H3-experimental` 的 `MiniMax-H3-*-Acc-8Step_comfy.safetensors` **不被这个节点接受** —— 报 `has no 'proj_out.weight' — not a PDD Acc file from alibaba-pai/MiniMax-H3-Acc-LoRAs (or a converted copy)?`。Kijai 那版是按 **core 的 `set_bias` / `reshape_weight` 布局**打包的（给 master 路线用），本节点认的是原版 `proj_out` / `audio_proj_out` 堆叠布局。**必须用 alibaba-pai 原版或 aptech0081 预转换版**（后者仓库已改名为 `aptech0081/MiniMax-H3-PDD-ComfyUI`）
- 适配 pruned / w4a8 再量化：通过 adaln rebasing（拟合残差 ~1.4e-5）

### 2.2 Zironic/H3-Optimizations

- `H3 Memory Optimization`：1.27×（134→106s）；**代价：同 seed 输出不再匹配**（平均像素差 2.27/255），GGUF 直接失败
- `H3 Sparse Attention`：默认 15% budget，**有意**丢弃部分视频间注意力连接 → 质量权衡，无"无损比例"
- `H3 VSA Attention`：需 FastH3/VSA 训练过的 checkpoint
- 要求 ComfyUI 0.33.0+

### 2.3 comfyui-turing-utils ⭐（唯一点名 sm_75 的包）

`wjie98/comfyui-turing-utils` 自述能力：`bundled sm75+ W8A8 attention`、`exact-sm75 Sage`、`native sm75+ Sol 稀疏注意力`、固定 Top-K **SLA**、**Spark-H3** 稀疏注意力、ConvRot W8A8/W4A8 加载、`MiniMax H3 Latent Upscale` 节点。

- 要求：Turing/Ampere/Ada+、Python 3.10+、`comfy-kitchen>=0.2.26`（本机 0.2.31 ✅）、**需自行编译** `comfyui-turing-utils-kernel>=0.29.1`（仓库内已含 CUDA 13.0 交叉编译的 sm75 cubin；原生 W8A8/Sol 需目标 GPU 的 cubin/PTX）
- 自述限制：NVFP4 路径"**2080 Ti runtime validation 仍待完成**"、Sage1/Sage2 源码 experimental、"不替代最终 exact-sm75 占用与端到端测试"
- **意义：这是对 `docs/03`「SageAttention 判死」的直接挑战**——判死的是 pip wheel + Triton 路线，这条走原生 sm75 cubin，不吃 Triton
- 现状：D128 specialization、dense/sparse 双核、`Configure Attention Strategy` 选 Sol/SLA/Veda/Spark/none

---

## 3. 需要升版本或换架构（与"锁 v0.33.1"直接冲突）

- **core 的 Sol-attn / Block Sparse Attention / Comfy Compiler**：v0.33.1 的 `object_info` 里**没有**（本机已核）
- **comfy-kitchen PR #117「Add int8 Sol-attention (CORE-391)」**：T=80k / 56 heads 下 **23.4 → 13.0 s/it（1.80×）**，cos ≈0.97 vs dense
- `kijai/ComfyUI-SolAttn_triton`：作者自述"仅 4090/5090 测过"、**依赖 Triton** → sm_75 大概率重演 SageAttention 的结局（待验）
- `xmarre/ComfyUI-Sol-H3`：执行 Sana 的 `cute_sm120` 后端 → **SM120/121 only**
- **Spectrum**：需 30–50 步才有效，10 步级近乎无用 → **本机 4 步路线无意义**

---

## 4. 权重侧：最便宜的 A/B（不碰版本、不装节点）

- **Turbo LoRA 换代**：`larryvrh v4_step600_ema`（v4 线）、ckpt500/850 EMA、lightx2v v1.0。社区做过 **25 配置 × 10 压力场景**系统对拍（持续音高 / 重叠对白 / 口音与语码切换 / 硬切 / 镜面一致性 / 手指 / 节拍网格），并发现 sampler 与 audio shift 是独立变量——**audio shift 6 会让 whisper 热 ~9dB**
- **FastH3 4 步（DMD2 蒸馏）**：118s vs Turbo 8 步 225s（**1.87×**），30 段 MV 零失败
- **TaoMate-H3 3 步**：**2.2×** vs Turbo 8 步；1.2× vs 4 步 LoRA
- **HyperFlow 8 步**（Video Rebirth，drbaph 打包 4 个）
- `H3-Anti-Burn-LoRA`：链式续接抗烧焦（非提速）

本机现状：`lightx2v 4step v0.1` + `turbo v1.0_768p`（compat 档实际引用 v0.1）。

---

## 5. 复核后仍判死 / 读不到

- **TE-Speed**：3060 实测 **0.999×**（改产物、零省时），与 `docs/06` 第 9 条"短步数语义崩坏"同向；3.x 号称适配 PDD / Block Sparse / Comfy Compiler，但**要求"升到最新 ComfyUI"**（与锁版冲突）且**无任何数字**
- **SageAttention（pip wheel）/ W4A4 / cache_device=gpu**：无新证据翻案
- **H3 Super-Acceleration on Turing（coolthor）**：HF 仓库 **gated（HTTP 401，需登录授权）**，正文读不到。检索摘要与 NVIDIA Sol-H3 Spark 线一致：672×384 draft · 124 帧 · 4 步 → LTX-2.5 精修 3 步 → 1344×768 / 5s@24fps，跑在同款 2080Ti 22G 上。**即本机 `RUNBOOK §8`「已评估、未实施」那条**。硬卡点：本机 31GB 内存 vs §8 对照机 128GB

---

## 6. 建议的最小实验序列

| 序 | 实验 | 成本 | 预期 |
|---|---|---|---|
| 1 | 加 `--use-ck-attention` 重启 + 同 seed 成片档对拍 | 一次重启 + 一镜 | kernel 已确认可用，**当天可做** |
| 2 | EasyCache 一镜对拍（4 步） | 0 安装 | 解决社区矛盾 |
| 3 | PDD-Acc 节点 | clone + 移动已有权重 | **解锁 `-ultra`，不升版本** |
| 4 | Turbo LoRA 换代（v4 / FastH3） | 下载权重 | 纯收益侧 |
| 5 | turing-utils | 编译 CUDA kernel | 最对口 sm_75，最重且未最终验证 |

---

## 7. 对现有手册判决的影响

| 现有条目 | 状态 | 说明 |
|---|---|---|
| `docs/08` T8 需 ComfyUI ≥0.34（误判） | 已纠正 | 本机 2026-10-19 实装 v1.0.4 并实跑 160s / 2-4 命中 |
| `docs/07` PDD 必须 master backport | **部分过期** | PDD-Acc 自定义节点在 ≥0.33.0 即可（见 §2.1），本机锁版不必动 |
| `docs/01` 「加速节点路线在 sm_75 上放弃」 | **需重估** | turing-utils 提供 sm75 原生 W8A8/Sage/Sol/SLA（见 §2.3）；且 kitchen INT8 attention 本机可用（§1.1） |
| `docs/06` TE-Speed 判死 | 维持 | 社区单卡实测 0.999×，方向一致 |
| `docs/03` SageAttention 判死 | 维持（pip wheel 路线） | 原生 sm75 路线另论，见 §2.3 |

## 8. 证据质量与未验证项

- 所有社区数字均为**异构硬件 + 不同步数/分辨率**，不可直接外推到本机；引用时一律带出处，落地一律本机对拍
- **未验证**：本机 INT8 attention 的实际提速与画质；EasyCache 在 4 步 + Turbo LoRA 下能否共存；turing-utils 的 CUDA kernel 能否在本机编译通过；PDD-Acc 与本地 pruned w4a8 + T8 组合的实际行为
- 调研工具限制（2026-10-19）：Exa 搜索经代理多次失败（3 条查询未完成）、Brave 无 API key、HF 两个页面分别被墙与 gated → **C 类缺少 sm_75 的直接实测报告**
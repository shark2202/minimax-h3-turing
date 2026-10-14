# 12 · 脱离 ComfyUI 服务器的 H3 Demo：嵌入库模式

> [11 号 demo](11-demo.md) 以 ComfyUI 服务器为运行时（起服务、走 HTTP、轮询下载）。这一章去掉服务器：**一个 Python 脚本进程**完成 文本/图片 → mp4，模型就是 MiniMax H3 本身，权重零新增下载。适合嵌入自己的训练/产线脚本、无头批量渲染、不想维护常驻服务的场景。

## 形态：ComfyUI 作为库，而不是服务

H3 本机路线是 h3lite **W4A8 量化权重**，加载器只有 ComfyUI 的 comfy-kitchen 反量化后端（[01](01-hardware-limits.md) 的 cu130 红利正来自它），也没有 Diffusers 管线。所以「脱离 ComfyUI」在 H3 语境下的正确形态是：**不起服务器、不开 UI、不走 HTTP/队列**，把本地 ComfyUI 代码当 Python 库 import 进来，用它自己的节点实现按拓扑序逐节点执行。

| | 服务器模式（docs/11） | 嵌入模式（本篇） |
|---|---|---|
| 进程 | ComfyUI 常驻 + 客户端脚本 | 单进程，跑完即退 |
| 通信 | HTTP /prompt + 轮询 | 进程内直接调用 |
| 工作流 | 同一批 JSON（六件套） | **同一批 JSON** |
| 权重 | docs/05 清单（models/ 树） | **同一套，零新增下载** |
| 预期耗时 | t2v ~280s / i2v ~420s（热机） | 相同引擎，同量级 |
| 附加能力 | 断点续跑、多任务排队 | 可直接嵌进 Python 产线代码 |

## 用法

```bash
cd standalone
python h3_demo.py --comfy-dir /path/to/ComfyUI \
    --prompt "a lone swordsman in a misty bamboo forest at dawn"

# 图生视频：自动把首帧拷进 ComfyUI input 目录再执行
python h3_demo.py --comfy-dir /path/to/ComfyUI \
    --image first_frame.png --prompt "she slowly turns to camera"
```

前置：一份 ComfyUI v0.33.1+ checkout（含其 Python 依赖）+ docs/05 的 H3 权重已放在它的 `models/` 树下——和服务器路线完全共用，装过一次即可。`--comfy-dir` 也可用环境变量 `COMFYUI_PATH` 代替。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--comfy-dir` / `$COMFYUI_PATH` | 必填 | ComfyUI checkout 路径（有 nodes.py 才算合法） |
| `--prompt` / `--image` / `--seed` | 3013 | 同 demo 11 的语义；给 `--image` 自动切 i2v 工作流 |
| `--workflow` | compat t2v/i2v | 可换六件套任意一档（如 t8draft，需 T8 节点包） |
| `--out` | h3_demo_output | mp4 拷出目录 |
| `--skip-custom-nodes` | 关 | compat 档不需要 custom_nodes，加载更快 |

## 它在进程内做了什么

| 步骤 | 对应手册知识 |
|---|---|
| `sys.argv` 注入 `--reserve-vram 2.5 --vram-headroom 0.5 --disable-pinned-memory` 再 `enable_args_parsing()` | 防黑屏启动参数原样生效（scripts/h3_launch），防 TDR / read_file_slice OOM |
| `import cuda_malloc` + torch 后再 init 节点 | cu130 反量化路径在嵌入模式下同样激活 |
| `PromptServer(loop) + PromptQueue + init_extra_nodes()` | ComfyUI 生成式脚本的标准嵌入姿势（参考 ComfyUI-to-Python-Extension runtime），custom nodes 兼容 |
| 逐节点拓扑执行，输入边解析为上游输出 | 内置图就是六件套 JSON 本身——**实测参数一个不差**，v3-API 全显式输入的纪律也顺带绕开了 HTTP 400 |
| 渲染前快照 output 目录，渲染后收新视频文件 | 无服务器 history 可查，等价于 demo 11 的输出收集 |
| 退出前 `soft_empty_cache()` | 单进程内反复渲染不积显存 |

## 边界与注意

- **单次一渲染**：进程内模型缓存是进程级的，demo 定位是单镜跑通；批量渲染用循环调 `execute_graph` 或回到服务器模式的队列。
- **执行错误直接抛**：没有服务器日志兜底，节点异常会带原始堆栈抛出——比 HTTP 模式更早暴露问题（v3 节点缺输入会直接 TypeError 而不是 400）。
- **同 seed 可复现**（compat 档）：与服务器模式同纪律；T8/PDD 档仍遵守「草稿用、成片不用」（[08](08-t8-blockcache-4step.md)）。
- **装新 pip 包后先跑一次 doctor**：量化+LoRA 静默失效坑（[09](09-pdd-backport.md)）与运行形态无关，嵌入模式同样会中招。

## 附录：完全不装 ComfyUI 的替代路线（Wan 2.1）

如果目标机器连 ComfyUI 代码都不想放：`standalone/demo.py` 提供纯 Diffusers 版（Wan 2.1，t2v 1.3B / i2v 14B 480P，下载与缓存清单见 [standalone/README](../standalone/README.md)）。**注意它生成的不是 H3**——这是模型层面的取舍：H3 出不了 ComfyUI 生态，此附录只回答「绝对零 ComfyUI」的极端需求。

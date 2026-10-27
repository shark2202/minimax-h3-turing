# Workflows · 目录结构（API 格式 vs 可视化 UI 格式）

> 这份只讲**目录怎么分、两种格式各给谁用、怎么转**。各档位的用法见 [README.zh-CN.md](README.zh-CN.md)（六件套说明）。

## 目录

```
workflows/
├── api/     ← API 格式（{"3": {"class_type": ...}}）—— 机器消费的那份
│               服务端 /prompt 只吃这个格式；h3turing 库打包的也是它
│               （python/src/h3turing/data/ 是它的包内副本）
├── ui/      ← UI 格式（{"nodes": [...], "links": [...]}，带坐标）—— 人在界面里用的那份
│               由 scripts/wf_convert.py 从 api/ 生成，1:1 对应（<同名>_ui.json）
├── preview/ ← 预览截图
└── README*.md
```

## 两种格式各是什么、给谁用

| | API 格式 | UI 格式（可视化） |
|---|---|---|
| 长相 | `{"3": {"class_type": "UNETLoader", "inputs": {...}}}` | `{"nodes": [{id, type, pos, inputs, outputs, widgets_values}], "links": [[...]]}` |
| 谁消费 | 服务端 `/prompt`、`h3turing` 库、脚本 | ComfyUI 界面（拖进画布 / Workflows 面板） |
| 布局 | 无 | 有坐标；本目录的 UI 版是**脚本自动分层排版**的（按依赖深度分列） |
| 能直接拖进界面吗 | 能（前端 `isApiJson` 会自动转），但**排版是自动生成的、不好读** | 直接就是画布上的样子 |

**为什么两种都要留**：API 版是"真相"（可被服务端执行、可被脚本批量处理）；UI 版是"给人看的"（在界面里改参数、拖节点、读结构）。同一次改动要同步两边——`scripts/wf_convert.py` 就是干这个的。

## 转换器

```bash
# API → UI（生成可视化工作流）
python scripts/wf_convert.py api2ui workflows/api/h3_w4a8_t2v_bypass2_api.json out.json
# 批量：整个目录
python scripts/wf_convert.py api2ui_dir workflows/api workflows/ui
# UI → API（把可视化工作流交回服务端）
python scripts/wf_convert.py ui2api some_ui.json out_api.json
```

转换器用**服务端 `/object_info`**（每个节点类的 `input_order` / `input` / `output` 定义）推算
`widgets_values` 的顺序——这是最容易出错的地方（顺序错位 = 提示词串进 seed 框，静默出错）。
另外处理了三种现实情况：前端专属节点（Note / PrimitiveFloat 等不在 `/object_info` 里，转换时跳过并断开相关连线）、
`widgets_values` 是 dict 的节点（LoadVideo）、缺失的可选输入不塞 null。

**保真度已验证**（2026-10-21）：把生成的 UI 文件转回 API 提交给 ComfyUI 实际渲染，
产物与 API 直跑**逐帧完全相同**（124/124 帧，PSNR 168 dB）。

## 装到 ComfyUI

ComfyUI 的 Workflows 面板读 `ComfyUI/user/default/workflows/`（前端用
`userdata?dir=workflows&recurse=true` 列举，**支持子目录分组**）。本机部署：

```
ComfyUI/user/default/workflows/
├── ui/              ← 产线的可视化工作流（9 个）—— 日常从这里拖
├── api/             ← 产线的 API 工作流（9 个）
├── _experiments/    ← 实验 / 已判死 / 被支配的（api + ui 各 20 个）
│   ├── api/
│   └── ui/
└── demo.json        ← ComfyUI 自带示例
```

重装：

```bash
bash scripts/install_qwen_node.local.sh   # 节点（符号链接）+ 剧本工作流
# 其余工作流：
mkdir -p ComfyUI/user/default/workflows/{api,ui}
cp workflows/api/*.json ComfyUI/user/default/workflows/api/
cp workflows/ui/*.json  ComfyUI/user/default/workflows/ui/
```

## 哪些是产线

见 [python/src/h3turing/tiers.py](../python/src/h3turing/tiers.py)（`h3turing tiers` 可查）
或 [RUNBOOK §3.1](../RUNBOOK.local.md)。`_experiments/` 里的**不要用在产线上**——
里面有已判死的（EasyCache 0 命中、larryvrh 音轨近静音）和被支配的旧档位。

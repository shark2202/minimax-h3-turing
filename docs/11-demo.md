# 11 · 最小 Demo：一条命令跑通 文本/图片 → 视频

> 全流程的「可执行摘要」：装好权重之后，从一句 prompt（或一张首帧图）到本地 mp4，只需要一条命令。零依赖，纯标准库。

## 前置（一次性，约 30 分钟）

1. **ComfyUI ≥ v0.33.1**，按 [04](04-community-tips.md) 的启动参数跑起来（防 TDR 黑屏）：

   ```bash
   # 模板见 scripts/h3_launch.example.sh，关键参数：
   python main.py --listen 127.0.0.1 --port 8188 \
     --reserve-vram 2.5 --vram-headroom 0.5 --disable-pinned-memory
   ```

2. **权重**：按 [05 的下载清单与复制粘贴命令](05-workflows.md) 把 W4A8 组件放进 `models/`（约 21G，下完可用 `h3turing doctor --models-dir` 核对）。
3. **本工具包**：

   ```bash
   pip install git+https://github.com/IvenKooLab/minimax-h3-turing.git#subdirectory=python
   ```

## 文生视频（t2v）

```bash
h3turing demo --prompt "cinematic wide shot, a lone swordsman walking through a misty bamboo forest at dawn"
```

## 图生视频（i2v）

```bash
h3turing demo --image first_frame.png --prompt "she slowly turns to camera, wind in her hair"
```

加 `--image` 自动切换到 i2v 档并上传首帧。首帧建议横构图（工作流按 640×352 生成，约 16:9）。

## 成功长什么样

```
tier t2v-final | seed 3013 | 4 steps (reference 280s on 2080Ti 22G)
submitted 6f2a…; rendering (progress every ~30s)...
  120s | queue: 1 running, 0 pending
saved: /home/you/h3_demo_output/H3Lite_00001.mp4
done in 287s (t2v-final, seed 3013)
```

- 视频落在 `--out` 目录（默认 `./h3_demo_output/`），ComfyUI 的 `output/video/` 里同时留档
- 参考耗时（热机）：t2v 成片档 **~280s**、i2v 成片档 **~420s**；草稿/极速档见 `h3turing tiers`
- 默认参数：640×352 · 124 帧（约 5 秒 @24fps）· 4 步 Turbo · 带原生音频

## 它替你做了什么

| 步骤 | 对应坑 |
|---|---|
| 探活 + 快速失败 | 服务器没开时 0.1s 报错并给出启动指引，而不是干等 |
| 首帧 multipart 上传（i2v） | LoadImage 认的是**服务器端文件名**，不是本地路径 |
| 占位符替换（prompt/首帧/seed） | v3-API 节点要求全参数显式，漏一个 = HTTP 400 |
| 提交后带停滞检测的轮询 | GPU 100% 时 HTTP 接口会滞后/丢包（FAQ） |
| 输出解析 + mp4 下载到本地 | SaveVideo 在 history 的 `outputs` 里，键名随版本漂移，三键全扫 |
| 同名不覆盖 | 服务器重启后计数器归零，旧 take 不会被新 take 覆盖 |

## 排错

- 先跑 `h3turing doctor`：探活、队列残留、T8 节点、权重齐全一站查
- `ComfyUI not reachable`：服务器没起或端口不对（`--base` 指定）
- `render finished but history has no saved output`：查服务器日志的 SaveVideo/编码错误
- 装过新 pip 包后 LoRA 静默失效（9-20 教训）：看 [09](09-pdd-backport.md) 的 `ERROR lora` 检查法
- 其余 13 条坑见 [06 FAQ](06-faq.md)

## 换档

`--tier` 接受 `h3turing tiers` 列出的任意档（如 `t2v-draft` 160s 草稿档）。记住唯一纪律：**T8/PDD 档同 seed 不可复现——草稿用，成片不用**。

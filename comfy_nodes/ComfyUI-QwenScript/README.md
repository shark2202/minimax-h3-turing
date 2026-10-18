# QwenScript · ComfyUI 节点

让**本机的小参数 Qwen3.8**（`~/models/Qwen3.8-9B-heretic/`，heretic 去审查版）在图里写
H3 提示词或整份逐镜剧本，并把"写剧本 → 出片"串成**一次排队**。UI 里叫
**Qwen3.8 Script (local LLM)**，分类 `MiniMax H3`。

## 安装 / 更新

```bash
bash scripts/install_qwen_node.local.sh     # 符号链接到 ComfyUI/custom_nodes + 拷贝工作流
pkill -f "main[.]py --listen" && setsid nohup scripts/h3_launch.local.sh >/dev/null 2>&1 < /dev/null &
```

单一真源就是本目录：ComfyUI 里装的是**符号链接**，改这里的 `.py` 重启即生效，不存在两份拷贝漂移。
（`install_qwen_node.local.sh` 遇到"真目录"会拒绝覆盖并让你先改名，避免又出现两份。）

## 显存协议（这是本节点的存在理由）

22 GB 卡上 LLM 与 H3 **不可能同时驻留**（实测：LLM 6808 MiB、H3 峰值 18926 MiB、卡 22528 MiB，
相加 25734 → 必然 OOM）。所以节点按下面的顺序做**串行**，而不是"共享"：

1. `free_comfy_cache`：调 `comfy.model_management.unload_all_models() + soft_empty_cache()`，
   把 ComfyUI 的模型和 torch 缓存块还给驱动（后面的采样器才装得下）
2. 有 `script_out` 缓存剧本 → **根本不起模型**；否则复用端点上的服务，或按 `CONFIG_PATH` 拉起
3. 生成 → `finally` 里**只杀自己拉起的那个进程**（按 `TargetModel` 路径匹配 `/proc/*/cmdline`，
   靠 `--port` 匹配处理别人的），并等 1.5s 让驱动回收
4. **前置断言**：`_free_vram_mib()`（torch `mem_get_info()`，回退 `nvidia-smi`）测**设备级**空闲，
   低于 `min_free_vram_mib`（默认 19700 = 实测峰值 + ~0.8 GiB）就报错并打印实际数字，
   而不是让采样器在 5 分钟后 CUDA OOM。别人占着显存时按 `foreign_server`：
   `fail`（默认，打印数字 + 处理办法）/ `stop`（自己停掉端口上的 llama-server）/ `ignore`

**不保证**的地方：断言是"某时刻采样"，不是持续监控——检查之后有别的东西来抢显存，失败点就变成
采样器里的 OOM；VAE 解码那一下的峰值属 ComfyUI 自己的预算（`--reserve-vram/--vram-headroom`）。

## 输入 / 输出

- 输出：`prompt`（接到 `MiniMaxH3ImageToVideo` 的 `prompt`）+ `screenplay`（整份剧本，节点上直接显示）
- 关键输入：`brief`（题材/梗概）、`shot`（出第几镜）、`shots`（写几镜）、`style`（额外要求）、
  `seed`（同 prompt 同 seed = 同剧本；**改一个输入值才会绕开 ComfyUI 的执行缓存**）、
  `script_out`（`.md` / `.shots.json` 前缀）、`reuse_script`、`min_free_vram_mib`、`foreign_server`
- 提示词契约在 `qwen3.8-2080ti-256k/scripts/screenplay.system.md`（改风格改那个文件，不用碰代码）

## 自检（不依赖 ComfyUI 运行时）

```bash
python3 -c "import sys;sys.path.insert(0,'comfy_nodes/ComfyUI-QwenScript');import qwen_script as q;print(len(q.QwenScript.INPUT_TYPES()['required']), q._free_vram_mib())"
```

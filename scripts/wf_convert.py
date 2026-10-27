#!/usr/bin/env python
"""API 格式 ↔ UI 格式（可视化）工作流互转。

为什么需要：ComfyUI 的 `/prompt` 只吃 **API 格式**（`{"3": {"class_type": ...}}`），
而界面里好看、能拖能看的是 **UI 格式**（`{"nodes": [...], "links": [...]}`，带坐标）。
前端其实能自动把 API JSON 铺成节点图（`isApiJson` / `loadApiJson`），但那是**自动
排版**——节点堆在一起、连线交叉，没有可读性。

本脚本用**服务端 `/object_info`**（每个节点类的 input_order / input / output 定义）
做双向转换，保证 `widgets_values` 的顺序与前端定义严格一致（错位 = 静默串值，
是最容易出的错）。

用法：
  # API → UI（生成可视化工作流）
  python scripts/wf_convert.py api2ui workflows/api/h3_w4a8_t2v_bypass2_api.json out.json
  # UI → API（用于把可视化工作流提交给服务端）
  python scripts/wf_convert.py ui2api some_ui_workflow.json out_api.json
  # 批量：把一个目录里的 API 工作流全转成 UI
  python scripts/wf_convert.py api2ui_dir workflows/api workflows/ui
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PRIMITIVE = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def load_object_info(base: str = "http://127.0.0.1:8188") -> dict:
    with urllib.request.urlopen(base + "/object_info", timeout=20) as r:
        return json.loads(r.read())


def _input_order(defn: dict) -> list[str]:
    """节点声明的输入顺序（前端 widget 顺序就按这个来）。"""
    order = defn.get("input_order") or {}
    req = order.get("required")
    opt = order.get("optional")
    if req is None or opt is None:
        inp = defn.get("input", {})
        req = list(inp.get("required", {}).keys())
        opt = list(inp.get("optional", {}).keys())
    return list(req) + list(opt)


def _input_spec(defn: dict, name: str) -> dict:
    inp = defn.get("input", {})
    for bucket in ("required", "optional"):
        spec = inp.get(bucket, {}).get(name)
        if spec is not None:
            raw = spec[0]
            # COMBO 的声明是 [[选项1, 选项2], {...}] —— spec[0] 是列表而非类型名
            if isinstance(raw, list):
                return {"type": "COMBO", "opts": spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}}
            return {"type": raw, "opts": spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}}
    return {"type": "?", "opts": {}}


def _is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


# ---------------------------------------------------------------- API -> UI

def api_to_ui(api: dict, object_info: dict, title: str = "") -> dict:
    # 1) 分配 UI 节点 id：数字原样保留，"105:201" 这种换成新的唯一整数
    api_ids = list(api.keys())
    ui_id: dict[str, int] = {}
    used: set[int] = set()
    next_id = 1
    for a in api_ids:
        if a.isdigit():
            ui_id[a] = int(a)
            used.add(int(a))
    for a in api_ids:
        if a not in ui_id:
            while next_id in used:
                next_id += 1
            ui_id[a] = next_id
            used.add(next_id)
            next_id += 1

    nodes: list[dict] = []
    links: list[list] = []
    link_id = 1
    # 记录每个 API 节点每个输出槽被谁连了，用于填 outputs[].links
    out_links: dict[tuple[str, int], list[int]] = {}

    for a in api_ids:
        node = api[a]
        ctype = node["class_type"]
        defn = object_info.get(ctype, {})
        inputs: list[dict] = []
        widgets: list = []

        for name in _input_order(defn):
            if name not in node.get("inputs", {}):
                continue
            value = node["inputs"][name]
            spec = _input_spec(defn, name)
            if _is_link(value):
                origin_api, origin_slot = value
                lid = link_id
                link_id += 1
                origin_type = "?"
                if origin_api in api:
                    odef = object_info.get(api[origin_api]["class_type"], {})
                    outs = odef.get("output", [])
                    if origin_slot < len(outs):
                        origin_type = outs[origin_slot]
                links.append([lid, ui_id[origin_api], origin_slot, ui_id[a], len(inputs), origin_type])
                out_links.setdefault((origin_api, origin_slot), []).append(lid)
                inputs.append({"name": name, "type": spec["type"], "link": lid})
            elif spec["type"] in PRIMITIVE:
                widgets.append(value)
                if spec["opts"].get("control_after_generate"):
                    # 前端在 seed 之后插一个"控制"槽（randomize/fixed）；API 里没有它
                    widgets.append("fixed")
            else:
                # 非原始类型却是字面量：界面上无处安放（通常是合法的"直接给值"），保留为 widget
                widgets.append(value)

        outs = defn.get("output", [])
        out_names = defn.get("output_name", outs)
        outputs = []
        for i, otype in enumerate(outs):
            name = out_names[i] if i < len(out_names) else otype
            outputs.append({"name": name, "type": otype, "links": out_links.get((a, i), []) or []})

        nodes.append({
            "id": ui_id[a],
            "type": ctype,
            "pos": [0, 0],                      # 稍后统一排版
            "size": [300, 80],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": ctype},
            "widgets_values": widgets,
        })

    _layout(nodes, links)
    return {
        "id": "uuid-will-be-generated-by-frontend",
        "revision": 0,
        "last_node_id": max((n["id"] for n in nodes), default=0),
        "last_link_id": max((l[0] for l in links), default=0),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.7, "offset": [60, 60]}},
        "version": 0.4,
    }


def _layout(nodes: list[dict], links: list[list]) -> None:
    """按依赖深度分层排版：列 = 深度，行 = 同层内序号。"""
    by_id = {n["id"]: n for n in nodes}
    parents: dict[int, list[int]] = {n["id"]: [] for n in nodes}
    for lid, src, _s, dst, _t, _ty in links:
        parents[dst].append(src)

    depth: dict[int, int] = {}

    def calc(nid, stack=()) -> int:
        if nid in depth:
            return depth[nid]
        if nid in stack:                       # 环：兜底
            return 0
        ps = parents.get(nid, [])
        d = 0 if not ps else 1 + max(calc(p, stack + (nid,)) for p in ps)
        depth[nid] = d
        return d

    for n in nodes:
        calc(n["id"])

    cols: dict[int, list[int]] = {}
    for n in nodes:
        cols.setdefault(depth[n["id"]], []).append(n["id"])
    for d in cols:
        cols[d].sort()

    # 执行顺序（拓扑序，供 order 字段用）
    order: list[int] = []
    seen: set[int] = set()

    def visit(nid):
        if nid in seen:
            return
        seen.add(nid)
        for p in parents.get(nid, []):
            visit(p)
        order.append(nid)

    for n in nodes:
        visit(n["id"])

    order_index = {nid: i for i, nid in enumerate(order)}
    x0, y0, colw, rowh = 60, 60, 340, 210
    for d, ids in sorted(cols.items()):
        for row, nid in enumerate(ids):
            n = by_id[nid]
            n["pos"] = [x0 + d * colw, y0 + row * rowh]
            n["order"] = order_index[nid]
            n["size"] = [300, max(80, 40 + 30 * len(n.get("widgets_values", [])))]


# ---------------------------------------------------------------- UI -> API

def ui_to_api(ui: dict, object_info: dict) -> dict:
    """UI 格式 → API 格式（可提交给 /prompt）。

    三个必须处理的现实情况：
    1. 前端专属节点（Note / MarkdownNote / PrimitiveFloat…）不在 /object_info 里，
       服务端不认识 → 直接跳过（它们只是画布上的说明，不参与计算）。
    2. 有些节点的 widgets_values 是 **dict**（如 LoadVideo），要按输入名取值。
    3. 缺失的**可选**输入不要塞默认值（塞 null 会让 v3 节点以为"传了值"），
       只有**必填**输入缺了才用默认值兜底。
    """
    link_map = {l[0]: (l[1], l[2], l[3], l[4]) for l in ui.get("links", [])}
    kept = {n["id"] for n in ui["nodes"] if n["type"] in object_info}

    api: dict[str, dict] = {}
    for n in ui["nodes"]:
        ctype = n["type"]
        defn = object_info.get(ctype)
        if defn is None:
            continue                                   # 前端专属节点，跳过
        order = _input_order(defn)
        raw_wv = n.get("widgets_values", [])
        is_dict = isinstance(raw_wv, dict)
        widget_vals = [] if is_dict else list(raw_wv)
        wi = 0
        api_inputs: dict[str, object] = {}

        for name in order:
            sock = next((i for i in n.get("inputs", []) if i.get("name") == name), None)
            if sock and sock.get("link") is not None and sock["link"] in link_map:
                src, sslot, _dst, _t = link_map[sock["link"]]
                if src in kept:                         # 源节点被跳过（前端专属）→ 断开这条连线
                    api_inputs[name] = [str(src), sslot]
                continue
            spec = _input_spec(defn, name)
            if is_dict:
                if name in raw_wv and isinstance(raw_wv[name], (str, int, float, bool)):
                    api_inputs[name] = raw_wv[name]
                continue
            if wi < len(widget_vals):
                api_inputs[name] = widget_vals[wi]
                wi += 1
                if spec["opts"].get("control_after_generate"):
                    wi += 1                            # 跳过"控制"槽
            else:
                required = name in defn.get("input", {}).get("required", {})
                if required:
                    api_inputs[name] = spec["opts"].get("default")
        api[str(n["id"])] = {"class_type": ctype, "inputs": api_inputs}
    return api


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode, src = sys.argv[1], sys.argv[2]
    dst = sys.argv[3] if len(sys.argv) > 3 else ""
    oi = load_object_info()

    if mode == "api2ui":
        g = json.load(open(src))
        out = api_to_ui(g, oi)
        Path(dst).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {dst}: {len(out['nodes'])} 节点 / {len(out['links'])} 连线")
        return 0

    if mode == "ui2api":
        g = json.load(open(src))
        out = ui_to_api(g, oi)
        Path(dst).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {dst}: {len(out)} 节点")
        return 0

    if mode == "api2ui_dir":
        outdir = Path(dst)
        outdir.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(src).glob("*.json")):
            try:
                g = json.load(open(f))
            except Exception as e:
                print(f"skip {f.name}: {e}")
                continue
            if not isinstance(g, dict) or "nodes" in g:
                continue
            out = api_to_ui(g, oi)
            (outdir / (f.stem + "_ui.json")).write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {f.name} → {f.stem}_ui.json ({len(out['nodes'])} 节点)")
        return 0

    print("未知模式:", mode)
    return 2


if __name__ == "__main__":
    sys.exit(main())
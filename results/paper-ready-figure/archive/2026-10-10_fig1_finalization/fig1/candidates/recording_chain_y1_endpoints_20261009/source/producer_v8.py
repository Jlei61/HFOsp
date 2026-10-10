#!/usr/bin/env python3
"""Move the distal zoom guide to Y1's deepest A-shaft contact; freeze all else."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig1a_recording_chain as render
from scripts.paper_figures import build_fig1a_y1_recording_chain as y1

np = render.np
BASE = render.CANONICAL/"candidates/recording_chain_y1_view_20261009"
OUT = render.CANONICAL/"candidates/recording_chain_y1_endpoints_20261009"


def main():
    for sub in ("source", "figures"):
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    old = json.loads((BASE/"metadata.json").read_text())
    source = json.loads((BASE/"source/selection.json").read_text())
    source["plot"]["zoom_anchor_mode"] = "physical_ends"
    projection = json.loads((BASE/"source/brain_projection.json").read_text())
    for filename in ("y1_brain.png", "brain_projection.json", "recording_and_geometry.npz"):
        shutil.copy2(BASE/"source"/filename, OUT/"source"/filename)
    with np.load(OUT/"source/recording_and_geometry.npz") as frozen:
        traces = frozen["signals_V"].copy()
        by_name = dict(zip(frozen["names"].tolist(), frozen["coords_mm"]))
    display = render.draw_candidate(OUT, source, traces, by_name, projection)
    for key in ("display_marker_mapping", "readout_locations_mm", "closeup_coordinates_mm",
                "trace_spacing_V", "trace_labels", "linked_channels", "canvas_size_inches", "external_lead"):
        assert display[key] == old["display"][key], key
    assert display["magnification_leaders"][0] == old["display"]["magnification_leaders"][0]
    assert display["magnification_leaders"][1][1] == old["display"]["magnification_leaders"][1][1]
    complete = y1.compose_current(OUT)
    metadata = {**old, "schema":"fig1a_y1_deep_contact_guide_v8", "producer":str(Path(__file__).resolve()),
                "reference_candidate":str(BASE), "display":display, "complete_layout":complete,
                "human_visual_acceptance":"PENDING",
                "change_scope":"only the distal zoom-guide source: move beside deepest A1; keep upper guide, brain, electrode and signals unchanged"}
    y1.write_json(OUT/"metadata.json", metadata)
    y1.write_json(OUT/"source/selection.json", source)
    for src, name in ((__file__, "producer_v8.py"), (render.__file__, "renderer_v8.py"),
                      (y1.__file__, "layout_helper_v8.py")):
        shutil.copy2(src, OUT/"source"/name)
    (OUT/"figures/README.md").write_text(
        "# Fig1A：放大引导线对应深端\n\n候选版，待作者目视检查。\n\n"
        "### fig1-panela.png / .pdf / .svg\n"
        "下方引导线改从脑内A杆最深触点A1旁的椭圆边界引出，对应中央电极的深端圆头；上方引导线仍从浅端附近引向尾部。"
        "脑模型、观察方向、真实触点及颜色、中间电极和右侧波形均保留上一版。\n"
        "**关注点**：两条引导线分别体现深、浅两端，未穿入圈选椭圆，彼此不相交；双极通道定义和数据未改变。\n\n"
        "### fig1-complete-layout.png / .pdf\n"
        "使用当前Figure 1修订的数据及绘图函数，仅替换A为本次引导线候选。"
        "当前修订及旧候选均保留。\n"
        "**关注点**：核对整页中的引导线位置；B–F保持当前版本。\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

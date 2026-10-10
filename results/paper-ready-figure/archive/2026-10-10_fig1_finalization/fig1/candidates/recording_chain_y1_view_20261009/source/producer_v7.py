#!/usr/bin/env python3
"""Adjust only the Y1 brain camera while retaining the accepted electrode/readout."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig1a_recording_chain as render
from scripts.paper_figures import build_fig1a_y1_recording_chain as y1
from scripts.paper_figures.patient_public_labels import artifact_subject_from_public

np = render.np
BASE = render.CANONICAL / "candidates/recording_chain_y1_20261009"
OUT = render.CANONICAL / "candidates/recording_chain_y1_view_20261009"


def main():
    for sub in ("source", "figures"):
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    old = json.loads((BASE/"metadata.json").read_text())
    source = json.loads((BASE/"source/selection.json").read_text())
    with np.load(BASE/"source/recording_and_geometry.npz") as cached:
        traces = cached["signals_V"].copy()
        names = cached["names"].copy()
        coords = cached["coords_mm"].copy()
    by_name, meshes, geometry = render.load_geometry(artifact_subject_from_public("yuquan", "Y1"))
    np.testing.assert_array_equal(names, np.array(list(by_name)))
    np.testing.assert_array_equal(coords, np.array(list(by_name.values())))
    assert geometry == old["geometry"]
    surface_points = np.concatenate([mesh.points for mesh in meshes])
    focus = (surface_points.min(0)+surface_points.max(0))/2
    projection = render.render_brain(
        by_name, meshes, OUT/"source/y1_brain.png", shaft_name="A", channel_colors=y1.COLORS,
        camera_focus=focus, parallel_scale_mm=93,
        camera_offset=(-420, 260, 120), camera_roll_degrees=12,
    )
    y1.write_json(OUT/"source/brain_projection.json", projection)
    display = render.draw_candidate(OUT, source, traces, by_name, projection)
    for key in ("readout_locations_mm", "closeup_coordinates_mm", "trace_spacing_V",
                "trace_labels", "linked_channels", "canvas_size_inches", "external_lead"):
        assert display[key] == old["display"][key], key
    for name in ("recording_and_geometry.npz", "selection.json"):
        shutil.copy2(BASE/"source"/name, OUT/"source"/name)
    complete = y1.compose_current(OUT)
    metadata = {
        **old, "schema":"fig1a_y1_front_left_view_v7", "producer":str(Path(__file__).resolve()),
        "reference_candidate":str(BASE), "geometry":geometry, "display":display,
        "complete_layout":complete, "human_visual_acceptance":"PENDING",
        "change_scope":"only the three-dimensional brain camera and dependent projection markers/zoom guides",
        "camera_revision":{
            "offset_mm":[-420,260,120], "roll_degrees":12,
            "intent":"frontal side toward viewer-left, with a small downward inclination",
            "physical_coordinates_unchanged":True,
            "reflection_or_image_warp":False,
        },
    }
    y1.write_json(OUT/"metadata.json", metadata)
    for src, name in ((__file__, "producer_v7.py"), (render.__file__, "renderer_v7.py"),
                      (y1.__file__, "layout_helper_v7.py")):
        shutil.copy2(src, OUT/"source"/name)
    (OUT/"figures/README.md").write_text(
        "# Fig1A：Y1 左前方观察视角\n\n候选版，待作者目视确认。\n\n"
        "### fig1-panela.png / .pdf / .svg\n"
        "Y1脑表面改从更靠左前方的视角观察，减小向下倾斜，使额侧朝向观察者左侧、略偏下；这是三维相机变化，未镜像、拉伸或改变解剖坐标。"
        "脑内102个实测触点、紫蓝双极通道标记和放大圈均重新投影，放大引导线随之更新。"
        "中央电极造型、连接帽、外接线和右侧A1–A12波形使用上一版冻结来源。\n"
        "**关注点**：检查脑模型朝向是否符合此前布局；信号、相邻双极定义、颜色对应和中央/右侧排版均保持不变。\n\n"
        "### fig1-complete-layout.png / .pdf\n"
        "将新A放入current_revision指向的当前完整Figure 1，B–F调用其现有绘图函数和数据。"
        "保留原修订及其指针，候选中仅调整A的脑部观察方向。\n"
        "**关注点**：查看完整页面中的脑模型姿态和对应关系；其余面板须与当前修订保持一致。\n",
        encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

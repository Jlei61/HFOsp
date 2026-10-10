#!/usr/bin/env python3
"""Retained-original Fig1A trial: highlight A7/A9 and magnify event centers."""
from __future__ import annotations

import hashlib
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
BASE = render.CANONICAL/"candidates/recording_chain_y1_endpoints_20261009"
OUT = render.CANONICAL/"candidates/recording_chain_y1_a7_zoom_trial_20261009"
COLORS = {"A7-A8":render.PURPLE, "A9-A10":render.BLUE}


def main():
    for sub in ("source", "figures"):
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    old = json.loads((BASE/"metadata.json").read_text())
    source = json.loads((BASE/"source/selection.json").read_text())
    old_projection = json.loads((BASE/"source/brain_projection.json").read_text())
    baseline_hashes = {str(p):render.sha256(p) for p in BASE.rglob("*") if p.is_file()}
    with np.load(BASE/"source/recording_and_geometry.npz") as original:
        traces = original["signals_V"].copy()
        names = original["names"].copy()
        coords = original["coords_mm"].copy()
        channels = original["channels"].copy()
    fs = float(source["plot"]["fs_out"])
    original_n = int(round(source["plot"]["window_sec"]*fs))
    zoom_n = int(round(.16*fs))
    assert traces.shape == (12,3*original_n) and original_n == 320
    lo = (original_n-zoom_n)//2
    bounds = [[i*original_n+lo,i*original_n+lo+zoom_n] for i in range(3)]
    zoomed = np.concatenate([traces[:,left:right] for left,right in bounds],axis=1)
    assert zoomed.shape == (12,480)
    # Crop only after the established filtering; retain the old common vertical gain.
    source["plot"].update(window_sec=.16, channel_colors=COLORS,
                          time_tick_step_sec=.1, highlight_marker_size_pt2=49,
                          color_highlighted_labels=True)
    source["selection"]["display_zoom"] = {
        "original_excerpt_duration_sec":.32,"display_excerpt_duration_sec":.16,
        "centering":"same original event-window center for every channel",
        "original_filtered_sample_slices_stop_exclusive":bounds,
        "displayed_edf_windows_sec":[[start+.08,start+.24] for start,end in old["signals"]["edf_crop_bounds_sec"]],
        "filtering":"unchanged original .32 s filtering, then exact central crop",
        "amplitude_gain":"unchanged shared gain; no per-channel rescaling",
    }
    by_name,meshes,geometry = render.load_geometry(artifact_subject_from_public("yuquan","Y1"))
    np.testing.assert_array_equal(names,np.array(list(by_name)))
    np.testing.assert_array_equal(coords,np.array(list(by_name.values())))
    assert geometry == old["geometry"]
    camera = old_projection["camera_position_focal_viewup"]
    projection = render.render_brain(by_name,meshes,OUT/"source/y1_brain.png",shaft_name="A",
                                    channel_colors=COLORS,camera_focus=np.array(camera[1]),
                                    parallel_scale_mm=old_projection["parallel_scale_mm"],
                                    camera_offset=tuple(old_projection["camera_offset_mm"]),
                                    camera_roll_degrees=old_projection["additional_camera_roll_degrees"])
    assert projection["contact_pixels"] == old_projection["contact_pixels"]
    assert projection["channel_midpoint_pixels"] == old_projection["channel_midpoint_pixels"]
    display = render.draw_candidate(OUT,source,zoomed,by_name,projection)
    for key in ("readout_locations_mm","closeup_coordinates_mm","trace_spacing_V",
                "trace_labels","canvas_size_inches","external_lead","magnification_leaders"):
        assert display[key] == old["display"][key],key
    y1.write_json(OUT/"source/selection.json",source)
    y1.write_json(OUT/"source/brain_projection.json",projection)
    shutil.copy2(BASE/"source/recording_and_geometry.npz",OUT/"source/original_recording_and_geometry.npz")
    np.savez_compressed(OUT/"source/recording_and_geometry.npz",signals_V=zoomed,
                        channels=channels,names=names,coords_mm=coords)
    signal_meta = {**old["signals"],"original_preprocessing_record":old["signals"],
                   "shape":list(zoomed.shape),"filtered_samples_sha256":hashlib.sha256(zoomed.tobytes()).hexdigest(),
                   "excerpt_duration_sec":.16,
                   "display_time":"three exact central 0.16 s crops, concatenated over 0.48 s; dashed boundaries at 0.16 and 0.32 s",
                   "display_zoom":source["selection"]["display_zoom"]}
    complete = y1.compose_current(OUT)
    metadata = {**old,"schema":"fig1a_y1_a7_zoom_trial_v9","producer":str(Path(__file__).resolve()),
                "reference_candidate":str(BASE),"status":"TRIAL_CANDIDATE","human_visual_acceptance":"PENDING",
                "display":display,"signals":signal_meta,"selection":source["selection"],
                "complete_layout":complete,
                "change_scope":"A7/A9 highlights, larger colored label markers, and central 2x time magnification; original retained"}
    y1.write_json(OUT/"metadata.json",metadata)
    assert baseline_hashes == {str(p):render.sha256(p) for p in BASE.rglob("*") if p.is_file()}
    for src,name in ((__file__,"producer_v9.py"),(render.__file__,"renderer_v9.py"),(y1.__file__,"layout_helper_v9.py")):
        shutil.copy2(src,OUT/"source"/name)
    (OUT/"figures/README.md").write_text(
        "# Fig1A：A7/A9与事件中心放大尝试\n\n原版完整保留，本目录为待作者目视选择的尝试版。\n\n"
        "### fig1-panela.png / .pdf / .svg\n"
        "紫色从A5改为A7，蓝色仍为A9，脑内真实双极中点、中间示意及右侧通道标记同步对应。"
        "右侧高亮圆点面积从25增至49 pt²（直径约增40%），A7/A9标签同步着色。"
        "每个原0.32秒片段只展示中心0.16秒，同一宽度下时间方向放大2倍；直接裁切原滤波结果，保留原共同纵向增益、事件及通道顺序。\n"
        "**关注点**：检查A7/A9辨识度与振荡细节；三个不连续片段以灰虚线分隔，总显示时间为0.48秒，未重新滤波或逐通道移时。A7、A9分别简称A7–A8和A9–A10。\n\n"
        "### fig1-complete-layout.png / .pdf\n"
        "将本次A尝试版放入当前完整Figure 1，B–F继续使用当前数据和绘图函数。"
        "原A候选及当前修订指针均不替换。\n"
        "**关注点**：比较整页中的色点大小和时间放大效果；本版仅供效果选择。\n",encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render Fig1A from the selected Y1 anatomy, A-shaft and original EDF.

Reuse the accepted connector-cap renderer and adjacent-bipolar preprocessing.
The three excerpts are illustrative examples, not a new cohort analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig1a_recording_chain as shared
from scripts.paper_figures.patient_public_labels import artifact_subject_from_public, public_patient_label

np = shared.np
CANON = shared.CANONICAL
OUTPUT = CANON / "candidates/recording_chain_y1_20261009"
COLORS = {"A5-A6": shared.PURPLE, "A9-A10": shared.BLUE}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def select_source(subject):
    record_path = ROOT / "results/interictal_propagation_masked/per_subject" / f"yuquan_{subject}.json"
    record = json.loads(record_path.read_text())
    assert public_patient_label(record["dataset"], record["subject"]) == "Y1"
    # Use the first chronological block already included in Y1's accepted C/E.
    first = record["event_metadata"]["record_names"][0]
    raw_dir = Path("/mnt/yuquan_data/yuquan_24h_edf") / subject
    packed_path = raw_dir / f"{first}_packedTimes.npy"
    lag_path = raw_dir / f"{first}_lagPat.npz"
    packed = np.load(packed_path)
    with np.load(lag_path) as lag:
        names = lag["chnNames"].tolist()
        rows = [i for i, name in enumerate(names) if name.startswith("A")]
        active = lag["eventsBool"] > 0
    assert active.shape[1] == len(packed)
    counts = active[rows].sum(0)
    eligible = ((counts >= 8) & active[names.index("A5")] & active[names.index("A9")])
    indices = []
    for chunk in np.array_split(np.arange(len(packed)), 3):
        choices = chunk[eligible[chunk]]
        assert len(choices), "No eligible A-shaft example in this chronological third"
        indices.append(int(choices[0]))
    channels = [f"A{i}-A{i+1}" for i in range(1, 13)]
    return {
        "source_paths": {"edf": str(raw_dir/f"{first}.edf"), "packed_times": str(packed_path),
                         "participation_artifact": str(lag_path), "accepted_subject_record": str(record_path)},
        "selection": {"display_label": "Yuquan Y1", "selected_channels": channels,
                      "display_labels": [f"A{i}" for i in range(1, 13)],
                      "reference_mode": "adjacent_bipolar", "event_indices": indices,
                      "event_windows_sec": packed[indices].tolist(),
                      "event_active_A_channel_counts": counts[indices].tolist(),
                      "selection_rule": "first chronological C/E source block; divide its event list into thirds; first event in each third with >=8 participating A-shaft channels and participating A5/A9; use eventsBool, never finite lagPatRank",
                      "shaft_selection": "A is a real 14-contact Y1 shaft with ten participating channels in accepted C/E; display A1-A12 from physical A1-A13",
                      "claim_boundary": "three illustrative examples, not prevalence, clinical localization or a new statistical result"},
        "plot": {"window_sec": 0.32, "fs_out": 1000., "band_hz": [80, 250],
                 "brain_image": "y1_brain.png", "channel_colors": COLORS,
                 "brain_electrode_gap_reduction_mm": 12.,
                 "amplitude_scale": "single common gain for all Y1 channels and excerpts; spacing=8*SD of all displayed samples, same rule as the original waveform producer"},
    }


def compose_current(output):
    """Rebuild the current compact Fig1 with its original B-F functions/data."""
    from scripts.paper_figures import build_fig1_y1_selection as current

    helpers = current.shared
    pointer = json.loads((CANON/"current_revision.json").read_text())
    existing_root = Path(pointer["revision_root"])
    existing_meta = json.loads(Path(pointer["metadata"]).read_text())
    assert pointer["selected_patient"] == "Y1"
    assert existing_meta["figure_size_inches"] == [16, 13.80]
    before = {str(p): shared.sha256(p) for p in (existing_root/"figures").iterdir() if p.is_file()}
    previous = json.loads((helpers.OUT/"metadata.json").read_text())
    helpers.old.propagation_plot._apply_masked_paths()
    records = helpers.old._load_temporal_records()
    helpers.old._assert_masked_mi_records(records)
    record = next(r for r in records if public_patient_label(r["dataset"],r["subject"]) == "Y1")
    arr = helpers.old._load_exemplar_arrays(record, max_events=10**9)
    assert len(arr["valid_events"]) == 18190
    assert [int(sum(arr["labels"] == i)) for i in range(2)] == [13160, 5030]
    spectrum = helpers.load_spectrum()
    helpers.plt.rcdefaults()
    helpers.plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42,
                                "svg.fonttype":"none", "axes.unicode_minus":False})
    fig = helpers.plt.figure(figsize=(16, 13.80))
    helpers.image_in_rect(fig, current.rect(fig,.22,9.14,8.65,4.28), output/"figures/fig1-panela.png")
    bmeta = current.draw_b_compact(fig,9.10,9.02,spectrum)
    cmeta,c_axis = current.draw_ce_aligned(fig,4.28,arr)
    emeta,e_axis = current.draw_ce_aligned(fig,.12,arr,True)
    dmeta,d_axis = current.draw_d_aligned(fig,12.30,4.96,records,previous["mechanism"])
    fmeta,f_axis = current.draw_f_aligned(fig,12.30,.80,records)
    current.check_d_group_alignment(fig,c_axis,d_axis)
    current.check_row_alignment(fig,e_axis,f_axis)
    for letter,x,y in [("A",.192,13.57),("B",8.96,13.57),("C",.192,8.97),
                       ("D",11.58,8.97),("E",.192,4.46),("F",11.58,4.46)]:
        fig.text(x/16,y/13.80,letter,fontsize=23,fontweight="bold",va="top")
    current.check_channel_labels(fig,arr)
    assert cmeta == existing_meta["panels"]["c"]
    assert emeta == existing_meta["panels"]["e"]
    assert dmeta == existing_meta["panel_d"]
    assert fmeta == existing_meta["panel_f"]
    assert bmeta == existing_meta["panel_b"]["display"]
    helpers.save(fig,output/"figures/fig1-complete-layout",dpi=300)
    after = {str(p): shared.sha256(p) for p in (existing_root/"figures").iterdir() if p.is_file()}
    assert before == after
    return {"source_revision":str(existing_root), "unchanged_B_F_metadata":True,
            "previous_revision_files_unchanged":True, "B_patient":"Y3", "C_E_patient":"Y1",
            "scope":"only panel A changed; B remains its existing Y3 example"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-brain",action="store_true")
    parser.add_argument("--panel-only",action="store_true")
    args = parser.parse_args()
    for name in ("source","figures"):
        (OUTPUT/name).mkdir(parents=True,exist_ok=True)
    subject = artifact_subject_from_public("yuquan","Y1")
    source = select_source(subject)
    print("Y1 source selected; loading EDF excerpts",flush=True)
    traces,signal_meta = shared.load_recordings(source)
    source["plot"]["trace_spacing_V"] = float(np.std(traces)*8)
    by_name,meshes,geometry_meta = shared.load_geometry(subject)
    for channel in source["selection"]["selected_channels"]:
        assert all(n in by_name for n in channel.split("-"))
    projection_path = OUTPUT/"source/brain_projection.json"
    if args.reuse_brain:
        projection=json.loads(projection_path.read_text())
    else:
        surface_points=np.concatenate([mesh.points for mesh in meshes])
        focus=(surface_points.min(0)+surface_points.max(0))/2
        projection=shared.render_brain(by_name,meshes,OUTPUT/"source/y1_brain.png",
                                      shaft_name="A",channel_colors=COLORS,
                                      camera_focus=focus,parallel_scale_mm=93)
        write_json(projection_path,projection)
    display=shared.draw_candidate(OUTPUT,source,traces,by_name,projection)
    print("Y1 panel rendered",flush=True)
    np.savez_compressed(OUTPUT/"source/recording_and_geometry.npz",signals_V=traces,
                        channels=np.array(signal_meta["channels"]), names=np.array(list(by_name)),
                        coords_mm=np.array(list(by_name.values())))
    write_json(OUTPUT/"source/selection.json",source)
    metadata={"schema":"fig1a_y1_recording_chain_v6", "display_label":"Yuquan Y1",
              "producer":str(Path(__file__).resolve()), "renderer":str(Path(shared.__file__).resolve()),
              "status":"CANDIDATE", "human_visual_acceptance":"PENDING",
              "formal_current_figure_replaced":False, "geometry":geometry_meta,
              "signals":signal_meta,"selection":source["selection"],"display":display}
    if not args.panel_only:
        metadata["complete_layout"]=compose_current(OUTPUT)
    write_json(OUTPUT/"metadata.json",metadata)
    shutil.copy2(__file__,OUTPUT/"source/producer_v6.py")
    shutil.copy2(shared.__file__,OUTPUT/"source/renderer_v6.py")
    (OUTPUT/"figures/README.md").write_text(
        "# Fig1A：Yuquan Y1 的植入定位、A 杆和真实记录\n\n"
        "候选版，待作者目视检查；当前整图和此前候选保留。\n\n"
        "### fig1-panela.png / .pdf / .svg\n"
        f"左侧为 Y1 本人的双侧 FreeSurfer pial 表面及 {geometry_meta['n_contacts']} 个真实触点，使用完整 affine 对齐 scanner RAS；中间沿用已认可的圆头、尾部连接帽及向右弯曲导线。"
        "选用 Y1 实际存在的 A 杆，12个环状符号表示相邻双极通道中点；A5–A6用紫色、A9–A10用蓝色，三个区域各一个对应标记。"
        "右侧为 Y1 原始 EDF 重建的 A1–A12 双极记录，三个0.32秒片段及80–250 Hz处理复用既有函数，全部通道共用一个幅值尺度。"
        "脑图与电极间的水平间距缩短12 mm，画布同步收紧，图形和文字未缩小。\n"
        "**关注点**：A1是A1–A2的简称，依次至A12–A13；电极外形、连接帽及导线为示意，不是厂商尺寸测量。"
        "三个展示事件按source/selection.json中的确定性规则从Y1既有事件中选取，仅作记录示例，不作统计推断。\n\n"
        + ("### fig1-complete-layout.png / .pdf\n"
           "使用当前refined_bd_y1版的原始绘图函数，仅替换A为本次Y1候选；B继续保留现有Y3谱图，C/E仍为已选定的Y1全量事件，D/F统计和其余布局保持原样。"
           "所有旧源文件保留，未切换当前修订指针。\n"
           "**关注点**：检查A的新脑模型、A杆颜色对应与收紧后的整页比例；本轮请求的患者替换范围为A。\n" if not args.panel_only else ""),
        encoding="utf-8")
    print(OUTPUT,flush=True)


if __name__ == "__main__":
    main()

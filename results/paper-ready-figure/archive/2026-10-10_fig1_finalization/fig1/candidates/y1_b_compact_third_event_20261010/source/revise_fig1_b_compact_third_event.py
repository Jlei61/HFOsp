#!/usr/bin/env python3
"""Tighten Figure 1B and replace only its third illustrative event.

The original full-window S**3 spectrum pipeline is reused without changes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import restore_fig1_legacy_spectrum as restored

BASE = restored.OUT / "latest_layout"
OUT = restored.CANON / "candidates/y1_b_compact_third_event_20261010"
CHOICES = [("FA134AX6", 1559), ("FA134AX6", 1562), ("FA134AXF", 1494)]
LAYOUT = dict(
    positions_inches=[[14.38 + i * 1.52, 11.27, 1.44, 3.85] for i in range(3)],
    colorbar_inches=[18.98, 11.27, .10, 3.85],
    title_baseline_inches=15.52,
    replacement_bounds_inches=[13.91, 10.48, 19.28, 15.87],
)


def main():
    (OUT / "source").mkdir(parents=True, exist_ok=True)
    previous = json.loads((restored.OUT / "spectrum_contract.json").read_text())
    contract = copy.deepcopy(previous)
    events = restored.display_events(contract, write_metadata=False, event_selection=CHOICES)
    old_events = restored.display_events(copy.deepcopy(previous), write_metadata=False)
    for old, new in zip(old_events[:2], events[:2]):
        for key in ("specs", "times", "centers", "freqs", "time_bounds"):
            np.testing.assert_array_equal(old[key], new[key])
    assert contract["display_events"][2]["event_index"] != previous["display_events"][2]["event_index"]
    for record in sorted({r for r, _ in CHOICES}):
        z = np.load(restored.OUT / f"source/{record}_segment.npz")
        computed = restored.legacy_spectrum(z["signals_stitched_V"], z["split_borders_sec"])
        for key, value in zip(("specs", "times", "freqs", "centers", "unnormalized_specs"), computed):
            np.testing.assert_array_equal(z[key], value)
    contract["previous_third_event"] = previous["display_events"][2]
    contract["illustration_change_reason"] = (
        "Replace the double-peaked A3 example by a cleaner existing event on the same Y1 A3-A9 shaft; "
        "the first two events and all spectrum/centroid definitions remain unchanged."
    )
    restored.write_json(OUT / "spectrum_contract.json", contract)
    preservation = restored.compose_patch(BASE, OUT, events, latest=True, spectrum_layout=LAYOUT)
    for letter in "acdef":
        for suffix in ("png", "pdf"):
            name = f"fig1-panel{letter}.{suffix}"
            assert (OUT / "figures" / name).read_bytes() == (BASE / "figures" / name).read_bytes()

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5.5))
    axes = restored.draw_spectrum(fig, events,
        [[.55+i*3.02, .65, 2.85, 4.2] for i in range(3)],
        [9.55, .65, .10, 4.2], 5.12, halfwidth=.25)
    for ax in axes[:3]:
        for t in (-150, 150):
            ax.axvline(t, color=".25", lw=.7, ls="--")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figures/fig1-spectrum-full-window-check.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)

    metadata = json.loads((BASE / "metadata.json").read_text())
    metadata.update(producer=str(Path(__file__).resolve()), source_revision=str(BASE),
        status="PENDING_AUTHOR_VISUAL_REVIEW", human_visual_acceptance="PENDING",
        changed_panels=["B spectrum layout and third event"],
        spectrum_contract=str(OUT / "spectrum_contract.json"),
        preservation=preservation, layout_change=LAYOUT,
        validation=str(OUT / "validation.json"))
    metadata["panel_b"].update(events=contract["display_events"], first_two_events_unchanged=True,
        old_third_event=previous["display_events"][2], third_event_replaced=True,
        horizontal_shift_inches=-.44, spectrum_width_inches=1.44,
        previous_spectrum_width_inches=1.28)
    metadata["summaries"]["B"] = metadata["panel_b"]
    metadata["outputs"] = {str(p.relative_to(OUT)): restored.sha(p)
        for p in (OUT / "figures").glob("*") if p.suffix in ("png", ".png", ".pdf", ".svg")}
    metadata["input_hashes"] = {str(p): restored.sha(p) for p in (
        Path(__file__), Path(restored.__file__), BASE / "metadata.json",
        restored.OUT / "spectrum_contract.json")}
    restored.write_json(OUT / "metadata.json", metadata)
    audit = dict(status="PASS", first_two_events_numerically_identical=True,
        third_event_replaced=contract["display_events"][2],
        original_spectrum_functions_exact=True,
        full_window_S3_centroid_unchanged=True,
        stored_lagpat_max_difference_ms=max(e["relative_lagpat_max_difference_ms"] for e in contract["display_events"]),
        preservation=preservation, other_standalone_panels_byte_identical=True,
        human_visual_acceptance="PENDING")
    restored.write_json(OUT / "validation.json", audit)
    for source in (Path(__file__), Path(restored.__file__)):
        shutil.copy2(source, OUT / "source" / source.name)
    descriptions = {
        "fig1-panelb.png / .pdf": "B右侧整体左移0.44英寸，各事件轴宽由1.28增至1.44英寸，标题仍与左侧HFO标题齐平。第三例更换为Y1同一A3–A9杆的FA134AXF/1494；前两例不变，谱及完整事件S³质心算法不变。\n\n**关注点**：布局和第三例均为本次实际修改，统一±150 ms时间轴及完整频率范围保留，待作者检查。",
        "fig1-complete-layout.png / .pdf": "最新18通道布局中仅替换B右侧区域。A、C、D、E、F及左侧178段HFO展示保持上一版。\n\n**关注点**：先看B两部分间距和新第三例；完整图待作者目视确认。",
        "fig1-spectrum-full-window-check.png / .pdf": "展示当前三个事件的完整500 ms计算窗口。虚线表示主图±150 ms的显示范围，质心始终从完整事件计算。\n\n**关注点**：第三例更换后仍与原lagPatRaw对齐，不通过移动质心改善观感。",
    }
    for letter in "acdef":
        descriptions[f"fig1-panel{letter}.png / .pdf"] = f"{letter.upper()}直接保留上一版独立图文件，数据和排版未改动。\n\n**关注点**：文件已与来源逐字节核对。"
    (OUT / "figures" / "README.md").write_text("# Figure 1B：收紧布局与替换第三例\n\n" +
        "\n\n".join(f"### {name}\n\n{text}" for name, text in descriptions.items()) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the preregistered zero-simulation objective controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STAGE = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
    "data_driven_dual_core_interictal_identifiability"
)
CONTROL_ROWS = (
    ("1_minority_removed", "Remove minority mode"),
    ("2_scl_censored", "Censor SCL contacts"),
    ("3_stretched", "Stretch onset times"),
    ("4_permuted_within_shaft", "Permute within shaft"),
)
METRICS = ("D_support", "D_order", "D_lag", "D_cover")
METRIC_LABELS = ("Support", "Order", "Timing", "Coverage")
TARGET = "#C43C39"
INVARIANT = "#2878B5"
DESCRIPTIVE = "#888888"
EMPTY = "#F3F3F3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cells(payload: dict) -> list[list[dict]]:
    controls = payload.get("controls") or {}
    if payload.get("schema_id") != "topic4_rev22_dci_objective_qualification_v2":
        raise RuntimeError("unexpected objective-qualification schema")
    if controls.get("all_pass") is not True:
        raise RuntimeError("objective controls did not all pass")
    cells: list[list[dict]] = []
    for control_id, _ in CONTROL_ROWS:
        record = controls.get(control_id)
        if not isinstance(record, dict) or record.get("pass") is not True:
            raise RuntimeError(f"missing or failed objective control: {control_id}")
        row = []
        for metric in METRICS:
            invariant_key = metric
            invariant_suffix = ""
            if metric == "D_order" and "D_order.ICL-ICL" in (record.get("invariance") or {}):
                invariant_key = "D_order.ICL-ICL"
                invariant_suffix = "\nICL only"
            if metric in (record.get("target") or {}):
                value = float(record["target"][metric]["fraction_worse"])
                row.append({"kind": "target", "value": value, "label": f"{100 * value:.0f}%"})
            elif invariant_key in (record.get("invariance") or {}):
                value = float(record["invariance"][invariant_key]["max_abs_difference"])
                row.append({"kind": "invariant", "value": value,
                            "label": f"max |d|\n{value:.1g}{invariant_suffix}"})
            elif metric in (record.get("reported") or {}):
                value = float(record["reported"][metric]["fraction_worse"])
                row.append({"kind": "descriptive", "value": value, "label": f"{100 * value:.0f}%"})
            else:
                row.append({"kind": "empty", "value": None, "label": ""})
        cells.append(row)
    return cells


def render(payload: dict, output_dir: Path, *, source_sha256: str | None = None) -> dict:
    cells = build_cells(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
    })
    colors = {"target": TARGET, "invariant": INVARIANT,
              "descriptive": DESCRIPTIVE, "empty": EMPTY}
    rgba = np.asarray([[matplotlib.colors.to_rgba(colors[cell["kind"]])
                        for cell in row] for row in cells])
    fig, ax = plt.subplots(figsize=(4.8, 2.55))
    ax.imshow(rgba, aspect="auto", interpolation="none")
    ax.set_xticks(range(len(METRICS)), METRIC_LABELS)
    ax.set_yticks(range(len(CONTROL_ROWS)), [label for _, label in CONTROL_ROWS])
    ax.tick_params(length=0, labelsize=7)
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CONTROL_ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y, row in enumerate(cells):
        for x, cell in enumerate(row):
            if not cell["label"]:
                continue
            ax.text(x, y, cell["label"], ha="center", va="center", color="white",
                    fontsize=6.5, fontweight="bold")
    ax.set_title("Zero-simulation objective controls", fontsize=9, fontweight="bold", pad=9)
    ax.text(0.0, -0.20,
            "Red: expected view worsened (% paired draws)   Blue: protected view unchanged\n"
            "Gray: descriptive only; not an acceptance criterion",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#333333")
    stem = "rev22_dci_objective_qualification_controls"
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    metadata = {
        "schema_id": "topic4_rev22_dci_objective_qualification_figure_v1",
        "source_schema_id": payload["schema_id"],
        "source_status": payload.get("status"),
        "source_git_commit": payload.get("git_commit"),
        "source_config_sha256": payload.get("config_sha256"),
        "source_sha256": source_sha256,
        "control_ids": [control_id for control_id, _ in CONTROL_ROWS],
        "cells": cells,
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "claim_boundary": (
            "This figure validates metric selectivity on patient pseudo-model controls; "
            "it is not evidence that any SNN candidate matches the patient."
        ),
    }
    metadata_path = output_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    (output_dir / "README.md").write_text(
        "### rev22_dci_objective_qualification_controls\n\n"
        "四个零仿真正控分别破坏少数传播模式、SCL 招募、物理时延和杆内顺序。"
        "红格给出预期受影响视图变差的配对抽样比例，蓝格确认不应变化的视图保持不变；"
        "灰格仅作描述，不参与目标函数验收。该图只验证量尺能否看见预期破坏，不证明模型已经拟合患者。\n\n"
        "**关注点**：四个预注册正控是否都选择性影响目标视图，并保护理论上不应变化的视图。\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=STAGE / "objective_qualification/objective_qualification.json")
    parser.add_argument("--output-dir", type=Path,
                        default=STAGE / "figures/objective_qualification")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = render(payload, args.output_dir, source_sha256=_sha256(args.input))
    print(json.dumps({"status": "OBJECTIVE_QUALIFICATION_FIGURE_COMPLETE",
                      "outputs": len(metadata["output_sha256"])}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Data adapter for the existing Topic 5 field-concordance painters.

No plotting code lives here.  The paired-field atlases are rendered by
``plot_topic5_field_concordance.plot_atlas`` and the cohort board is rendered by the original
OR-margin-board painter in ``plot_topic5_field_concordance_best_board``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_topic5_field_concordance as atlas_plotter
from scripts.plot_contact_plane_static import _limits_with_padding
from scripts.plot_topic5_axis_alignment_fields import _rank01
from scripts.plot_topic5_field_concordance_best_board import plot_or_margin_board

RESULT = ROOT / "results/topic5_ictal_recruitment/template_axis_field_frequency_panel"
AXIS_CSV = ROOT / "results/topic5_ictal_recruitment/template_axis_field/axis_cohort.csv"
FIG = RESULT / "figures"

BANDS = [
    {"name": "HFA 60–100 Hz", "slug": "hfa_60_100", "color": "#d95f45"},
    {"name": "Broadband 1–45 Hz", "slug": "broad_1_45", "color": "#2b6cb0"},
    {"name": "Broadband 1–150 Hz", "slug": "broad_1_150", "color": "#2f855a"},
]


def _epilepsiae_axis_rows() -> list[dict]:
    rows = [
        row for row in csv.DictReader(AXIS_CSV.open())
        if row.get("status") == "ok" and row.get("dataset") == "epilepsiae"
    ]
    rows.sort(key=lambda row: int(row["subject"]))
    if len(rows) != 17:
        raise RuntimeError(f"expected 17 Epilepsiae axis-defined patients, found {len(rows)}")
    return rows


def _load_result(subject_id: str, band: dict) -> dict:
    path = RESULT / band["slug"] / "per_subject" / f"{subject_id}.json"
    record = json.loads(path.read_text())
    field = record.get("field") or {}
    if field.get("status") != "ok":
        raise RuntimeError(f"{subject_id} {band['slug']}: field status is {field.get('status')}")
    metric = field["statistics"]["nulls"]["channel"]["metrics"]["own_maxab"]
    observed = field["statistics"]["observed_by_seizure"]
    median_a = float(np.median([float(value["own_a_abs"]) for value in observed.values()]))
    median_b = float(np.median([float(value["own_b_abs"]) for value in observed.values()]))
    winner = "a" if median_a >= median_b else "b"
    n_a = sum(float(value["own_a_abs"]) >= float(value["own_b_abs"])
              for value in observed.values())
    n_b = len(observed) - n_a
    return {
        "record": record,
        "field": field,
        "metric": metric,
        "winner": winner,
        "n_winner_a": n_a,
        "n_winner_b": n_b,
    }


def _soz_mask(subject_id: str, names: list[str]) -> np.ndarray:
    path = atlas_plotter.REAL_DIR / f"{subject_id}_t_a.json"
    if not path.exists():
        return np.zeros(len(names), bool)
    record = json.loads(path.read_text())
    lookup = {str(channel["name"]): bool(channel.get("is_soz"))
              for channel in record.get("channels", [])}
    return np.asarray([lookup.get(name, False) for name in names], bool)


def _atlas_row(subject_id: str, loaded: dict) -> dict:
    field = loaded["field"]
    metric = loaded["metric"]
    winner = loaded["winner"]
    plane = field["planes"][f"own_{winner}"]
    scale = float(plane["scale_mm"])
    points = np.asarray(plane["points"], float) * scale
    rank = np.asarray(field[f"rank_{winner}"], float)
    inter = (rank - np.nanmin(rank)) / max(float(np.nanmax(rank) - np.nanmin(rank)), 1e-12)
    ict = _rank01(field["seizure_mean"])
    ok = np.isfinite(inter) & np.isfinite(ict)
    sign_neg = bool(int(ok.sum()) >= 3 and np.corrcoef(inter[ok], ict[ok])[0, 1] < 0)
    names = [str(name) for name in field["names"]]
    p95 = float(metric["null_q"]["p95"])
    observed = float(metric["obs_subject"])
    return {
        "ds_sid": subject_id,
        "xs": points[:, 0],
        "ys": points[:, 1],
        "inter": inter,
        "ict": ict,
        "support": np.asarray(field[f"support_{winner}"], float),
        "soz": _soz_mask(subject_id, names),
        "xlim": _limits_with_padding(points[:, 0], include_zero=True, min_span=35.0),
        "ylim": _limits_with_padding(points[:, 1], include_zero=True, min_span=35.0),
        "sigma": float(plane["sigma"]) * scale,
        "sign_neg": sign_neg,
        "r": observed,
        "p95": p95,
        "passed": bool(metric["passed"]),
        "margin": observed - p95,
        "n_ch": len(names),
    }


def _write_table(subject_rows: list[dict], loaded: dict[tuple[str, str], dict]) -> Path:
    records = []
    for subject in subject_rows:
        sid = subject["subject_id"]
        row = {"subject_id": sid, "dataset": "epilepsiae", "subject": subject["subject"],
               "display_name": f"E{subject['subject']}"}
        for band in BANDS:
            item = loaded[(sid, band["slug"])]
            metric = item["metric"]
            n_a, n_b = item["n_winner_a"], item["n_winner_b"]
            prefix = band["slug"]
            row.update({
                f"{prefix}_maxab_r": metric["obs_subject"],
                f"{prefix}_n_seizures": metric["n_seizures"],
                f"{prefix}_modal_winner": "A" if n_a > n_b else "B" if n_b > n_a else "A/B",
                f"{prefix}_winner_fraction": max(n_a, n_b) / (n_a + n_b),
                f"{prefix}_n_winner_a": n_a,
                f"{prefix}_n_winner_b": n_b,
            })
        records.append(row)
    path = RESULT / "maxab_early_ictal_frequency_table.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def _write_readme(medians: dict[str, float]) -> None:
    (FIG / "README.md").write_text(f"""# 图说明

### field_concordance_or_margin_board_maxab_frequency_epilepsiae.png

直接复用 `field_concordance_or_margin_board_prototype.png` 的原始绘图函数。队列仅包含 17 名 Epilepsiae 双轴可估患者；Yuquan 中仅有两人有 3 个缓存发作条目，其余无可用发作，因此全部不进入本图。左侧主点是每名患者三个频段中最大的 `real |r| − channel-null p95`，淡色竖线保留三个候选频段，右侧方块表示各频段是否超过自己的 channel-shuffle null。

**关注点**：这是与原 OR-margin board 完全相同的 margin 展示，不是新的相关性热图。

### field_concordance_atlas_maxab_hfa_60_100.png

直接调用 `plot_topic5_field_concordance.py::plot_atlas`，不再实现 atlas 布局、field painter、边框、排序或色标。左侧是该患者 A/B 中患者内相关更高的 own-axis 模板场（viridis），右侧是 HFA 60–100 Hz 平均早期发作场（Reds：低能量浅色、高能量深红）；tile 上的 `|r|` 是逐发作 maxAB 后的患者中位数。

**关注点**：仅看 17 名 Epilepsiae 的逐患者 paired field。

### field_concordance_atlas_maxab_broad_1_45.png

与 HFA atlas 使用完全相同的现有 `plot_atlas` 函数，激活场换为 broadband 1–45 Hz。17 人的患者 maxAB 中位数为 {medians['broad_1_45']:.3f}。

**关注点**：边框、排序和 null 含义与原 atlas 一致。

### field_concordance_atlas_maxab_broad_1_150.png

与前两张使用同一现有 `plot_atlas` 函数，激活场为 line-noise-masked broadband 1–150 Hz。17 人的患者 maxAB 中位数为 {medians['broad_1_150']:.3f}。

**关注点**：与 1–45 Hz 的 paired field 直接对照。
""")


def main() -> None:
    subjects = _epilepsiae_axis_rows()
    loaded = {(subject["subject_id"], band["slug"]): _load_result(subject["subject_id"], band)
              for subject in subjects for band in BANDS}
    table = _write_table(subjects, loaded)

    board_rows = []
    for subject in subjects:
        sid = subject["subject_id"]
        values = {}
        for band in BANDS:
            metric = loaded[(sid, band["slug"])]["metric"]
            real = float(metric["obs_subject"])
            p95 = float(metric["null_q"]["p95"])
            values[band["name"]] = {
                "real": real, "p95": p95, "margin": real - p95,
                "pass": bool(metric["passed"]), "color": band["color"],
            }
        best_name = max(values, key=lambda name: values[name]["margin"])
        best = values[best_name]
        board_rows.append({"subject_id": sid, "vals": values, "best_label": best_name,
                           **best, "or_pass": any(value["pass"] for value in values.values())})
    board_rows.sort(key=lambda row: (not row["or_pass"], -row["margin"]))
    n_pass = sum(row["or_pass"] for row in board_rows)
    board = plot_or_margin_board(
        board_rows, BANDS,
        FIG / "field_concordance_or_margin_board_maxab_frequency_epilepsiae.png",
        f"Epilepsiae OR-over-3 own-axis maxAB board: {n_pass}/{len(board_rows)} pass",
    )

    atlas_plotter.OUT = FIG
    atlases = []
    medians = {}
    for band in BANDS:
        rows = [_atlas_row(subject["subject_id"], loaded[(subject["subject_id"], band["slug"])])
                for subject in subjects]
        medians[band["slug"]] = float(np.median([row["r"] for row in rows]))
        atlas_plotter.ACTIVATION_LABEL[band["slug"]] = band["name"]
        atlases.append(atlas_plotter.plot_atlas(
            rows, band["slug"],
            subtitle_text=(f"per subject:  winning own-axis A/B template-rank field   vs   "
                           f"seizure-onset {band['name']} activation field          "
                           "r$_s$ = | corr$_{mirror}$( F$_{interictal}$ , F$_{seizure}$ ) |"),
            output_name=f"field_concordance_atlas_maxab_{band['slug']}.png",
        )[0])
    _write_readme(medians)
    print(f"wrote {table}")
    print(f"wrote {board}")
    for atlas in atlases:
        print(f"wrote {atlas}")


if __name__ == "__main__":
    main()

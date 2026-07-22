#!/usr/bin/env python3
"""Render frozen interictal TA/TB rank fields for every estimable subject.

The visual grammar is intentionally inherited from
``plot_topic5_ictal_field_dynamics.py``: two viridis rank fields, template A on
the left and template B on the right.  Geometry and support are read from the
frozen ``topic5_interictal_template_fields_v1`` artifact; no axis or plane is
refit here.  Rendering uses a fixed 6-mm display bandwidth by default, while
the patient-specific frozen kernels used for field scoring remain unchanged.

Collinear TA/TB pairs use their common shared plane for both panels.  Other
pairs use the two template-specific planes.  Contact names, swap/source rings
and clinical overlays are deliberately omitted.

The locked data, projection and visual contract is documented in
``docs/topic5_interictal_field_figure_spec.md``.  New interictal-field figures
must reuse this module's public payload, panel, subject or atlas functions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_contact_plane_static import _limits_with_padding
from scripts.plot_topic5_field_vs_ictal_swap import (
    FS_CBAR_LABEL,
    FS_TICK,
    _rank01,
    draw_topic5_field_panel,
)
from src.topic5_template_axis_field import scorers_from_interictal_record


DEFAULT_INPUT = (
    ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
)
DEFAULT_OUTPUT = (
    ROOT / "results/interictal_propagation_masked/template_gradient_fields/figures"
)
DEFAULT_DISPLAY_SIGMA_MM = 6.0
DEFAULT_YUQUAN_CROSSWALK = ROOT / "docs/paper-draft/.private/yuquan_crosswalk.md"
INTERICTAL_FIELD_FIGURE_CONTRACT = "topic5_interictal_ab_field_figure_v1"
TA_COLOR = "#B2182B"
TB_COLOR = "#2166AC"
FS_XLABEL = 17
FS_PANEL_TITLE = 20
FS_SUBJECT_TITLE = 19
CONTACT_OUTLINE_LW = 2.7
CONTACT_SIZE = 92
ATLAS_CONTACT_OUTLINE_LW = 1.5
ATLAS_CONTACT_SIZE = 46

__all__ = [
    "INTERICTAL_FIELD_FIGURE_CONTRACT",
    "build_interictal_ab_panel_payloads",
    "draw_interictal_rank_field_panel",
    "load_interictal_field_records",
    "plot_interictal_ab_atlas",
    "plot_interictal_ab_subject",
]


def _load_yuquan_crosswalk(path: Path) -> Dict[str, str]:
    """Read the ignored private folder-to-public-ID map without exposing it."""
    if not path.exists():
        raise FileNotFoundError(
            f"Yuquan public-ID crosswalk not found: {path}; pass --yuquan-crosswalk"
        )
    labels: Dict[str, str] = {}
    row = re.compile(r"^\|\s*(Y\d+)\s*\|\s*([^|]+?)\s*\|")
    for line in path.read_text().splitlines():
        match = row.match(line)
        if match:
            public_id, folder = match.groups()
            labels[folder.strip()] = public_id
    if not labels:
        raise ValueError(f"no Yuquan public-ID rows found in {path}")
    return labels


def _display_name(subject_id: str, yuquan_labels: Mapping[str, str]) -> str:
    if subject_id.startswith("epilepsiae_"):
        return "E" + subject_id.removeprefix("epilepsiae_")
    if subject_id.startswith("yuquan_"):
        folder = subject_id.removeprefix("yuquan_")
        if folder not in yuquan_labels:
            raise KeyError(f"no public Yuquan ID for artifact folder: {folder}")
        return str(yuquan_labels[folder])
    return subject_id


def _axis_mode(record: Mapping[str, object], models: Mapping[str, object]) -> str:
    relation = (record.get("axis_pair") or {}).get("relation") or {}
    has_shared = all(key in models for key in ("shared_a", "shared_b"))
    if bool(relation.get("collinear")) and has_shared:
        return "shared"
    return "own"


def _canonical_transverse_sign(w: Sequence[float]) -> int:
    """Fix the otherwise arbitrary SVD sign using the dominant 3D component."""
    vec = np.asarray(w, float)
    if vec.shape != (3,) or not np.isfinite(vec).all() or np.linalg.norm(vec) <= 0:
        raise ValueError("transverse basis must be one finite nonzero 3D vector")
    dominant = int(np.argmax(np.abs(vec)))
    return 1 if vec[dominant] >= 0 else -1


def _transverse_display_signs(
    mode: str,
    planes: Mapping[str, object],
    y_a: Sequence[float],
    y_b: Sequence[float],
) -> Tuple[int, int, float]:
    """Choose display-only transverse signs from geometry, never field values.

    The A sign is deterministic in the subject's 3D coordinate system.  For
    separate own planes, B is then oriented to minimize same-contact transverse
    RMSE against A.  Shared-plane A/B panels receive exactly the same sign.
    """
    ya = np.asarray(y_a, float)
    yb = np.asarray(y_b, float)
    if ya.shape != yb.shape:
        raise ValueError("TA/TB transverse vectors must be contact-aligned")
    if mode == "shared":
        sign = _canonical_transverse_sign(planes["shared"]["w"])
        return sign, sign, 0.0
    if mode != "own":
        raise ValueError(f"unknown axis mode: {mode}")

    sign_a = _canonical_transverse_sign(planes["own_a"]["w"])
    target = sign_a * ya
    use = np.isfinite(target) & np.isfinite(yb)
    if int(use.sum()) < 2:
        raise ValueError("fewer than two matched contacts for transverse sign alignment")
    rmse_by_sign = {
        sign: float(np.sqrt(np.mean((target[use] - sign * yb[use]) ** 2)))
        for sign in (1, -1)
    }
    sign_b = min((1, -1), key=lambda sign: (rmse_by_sign[sign], -sign))
    return sign_a, sign_b, rmse_by_sign[sign_b]


def build_interictal_ab_panel_payloads(
    record: Mapping[str, object], *, display_sigma_mm: float = DEFAULT_DISPLAY_SIGMA_MM,
) -> Tuple[Dict[str, object], Dict[str, object], str]:
    """Build locked TA/TB payloads for the shared Topic 5 field renderer."""
    if not np.isfinite(display_sigma_mm) or display_sigma_mm <= 0:
        raise ValueError("display_sigma_mm must be a positive finite value")
    field = record.get("interictal_field") or {}
    models = scorers_from_interictal_record(record)  # fingerprint validation, fail closed
    mode = _axis_mode(record, models)
    planes = field.get("planes") or {}
    names = [str(x) for x in field["contact_order"]]
    rank_a_raw = np.asarray(field["rank_a"], float)
    rank_b_raw = np.asarray(field["rank_b"], float)
    rank_a = _rank01(rank_a_raw)
    rank_b = _rank01(rank_b_raw)

    specs = (
        ("shared_a", "shared", rank_a, rank_a_raw)
        if mode == "shared" else ("own_a", "own_a", rank_a, rank_a_raw),
        ("shared_b", "shared", rank_b, rank_b_raw)
        if mode == "shared" else ("own_b", "own_b", rank_b, rank_b_raw),
    )
    raw = []
    for model_key, plane_key, ranks, rank_values in specs:
        model = models[model_key]
        plane = planes[plane_key]
        scale_mm = float(plane["scale_mm"])
        points_mm = np.asarray(model["points"], float) * scale_mm
        if points_mm.shape != (len(names), 2):
            raise ValueError(
                f"{record.get('subject_id')} {model_key}: point/contact shape mismatch"
            )
        raw.append(
            {
                "names": names,
                "xs": points_mm[:, 0],
                "ys": points_mm[:, 1],
                "vals": np.asarray(ranks, float),
                # Keep the frozen integer-like ranks alongside the 0..1 display values.
                # Embedded paper panels can therefore label their colorbars in actual ranks
                # without changing the established field colours or refitting the field.
                "rank_values": np.asarray(rank_values, float),
                "sup": np.asarray(model["support"], float),
                # The analysis kernel remains frozen in ``model['sigma']``.
                # A fixed display-only bandwidth restores the established
                # spatial-field coverage without changing any field score.
                "sigma_mm": float(display_sigma_mm),
            }
        )

    sign_a, sign_b, transverse_rmse = _transverse_display_signs(
        mode, planes, raw[0]["ys"], raw[1]["ys"],
    )
    for item, sign in zip(raw, (sign_a, sign_b)):
        item["ys"] = sign * item["ys"]

    # One physical-mm display extent per subject.  Shared pairs therefore have
    # identical coordinates; own-axis pairs retain their own orientations but
    # use a common plotting scale, matching the established A|B presentation.
    all_x = np.concatenate([item["xs"] for item in raw])
    all_y = np.concatenate([item["ys"] for item in raw])
    xlim = _limits_with_padding(all_x, include_zero=True, min_span=35.0)
    ylim = _limits_with_padding(all_y, include_zero=True, min_span=35.0)

    payloads = []
    for item, sign in zip(raw, (sign_a, sign_b)):
        payloads.append(
            {
                "ds_sid": str(record["subject_id"]),
                "names": item["names"],
                "xs": item["xs"],
                "ys": item["ys"],
                "sup": item["sup"],
                "soz": np.zeros(len(names), dtype=bool),
                "src_a": set(),
                "src_b": set(),
                "frame": {
                    "xlim": xlim,
                    "ylim": ylim,
                    "sigma_mm": item["sigma_mm"],
                },
                "vals": item["vals"],
                "rank_values": item["rank_values"],
                "transverse_sign": int(sign),
                "transverse_alignment_rmse_mm": float(transverse_rmse),
            }
        )
    return payloads[0], payloads[1], mode


def draw_interictal_rank_field_panel(
    ax, payload: Mapping[str, object], template: str, *,
    compact: bool = False, panel_title: str | None = None,
    contact_outline_lw: float | None = None, contact_size: float | None = None,
    show_template_tag: bool = True,
):
    """Draw one locked TA/TB rank-field panel using the shared renderer."""
    template = str(template).upper()
    if template not in {"TA", "TB"}:
        raise ValueError("template must be TA or TB")
    color = TA_COLOR if template == "TA" else TB_COLOR
    title = (panel_title or "") if compact else (panel_title or template)
    draw_topic5_field_panel(
        ax, payload, payload["vals"], title, "early 0 → late 1",
        compact=compact, labels=False, cbar=False,
        contact_outline_lw=(
            float(contact_outline_lw) if contact_outline_lw is not None
            else (ATLAS_CONTACT_OUTLINE_LW if compact else CONTACT_OUTLINE_LW)
        ),
        contact_size=(
            float(contact_size) if contact_size is not None
            else (ATLAS_CONTACT_SIZE if compact else CONTACT_SIZE)
        ),
    )
    if compact:
        ax.title.set(color="#222222", fontsize=9)
        if show_template_tag:
            ax.text(
                0.035, 0.965, template, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold", color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
                zorder=8,
            )
    else:
        ax.title.set(color=color, fontsize=FS_PANEL_TITLE, fontweight="bold")
    return ax


def plot_interictal_ab_subject(
    record: Mapping[str, object], output_dir: Path, *,
    yuquan_labels: Mapping[str, str],
    display_sigma_mm: float = DEFAULT_DISPLAY_SIGMA_MM,
    output_format: str = "png",
) -> Path:
    dat_a, dat_b, mode = build_interictal_ab_panel_payloads(
        record, display_sigma_mm=display_sigma_mm,
    )
    subject_id = str(record["subject_id"])
    pretty = _display_name(subject_id, yuquan_labels)
    mode_label = "shared" if mode == "shared" else "separate"
    xlim = dat_a["frame"]["xlim"]
    ylim = dat_a["frame"]["ylim"]
    frame_aspect = float((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))
    figure_width = float(np.clip(3.0 + 10.0 * frame_aspect, 8.6, 13.0))

    fig, axes = plt.subplots(
        1, 2, figsize=(figure_width, 6.6), sharex=True, sharey=True,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.025)
    draw_interictal_rank_field_panel(axes[0], dat_a, "TA")
    draw_interictal_rank_field_panel(axes[1], dat_b, "TB")
    axes[0].set_anchor("E")
    axes[1].set_anchor("W")
    for ax in axes:
        ax.set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    colorbar_ax = axes[1].inset_axes([1.045, 0.0, 0.055, 1.0])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
        cax=colorbar_ax,
    )
    colorbar.set_label("early 0 → late 1", fontsize=FS_CBAR_LABEL)
    colorbar.ax.tick_params(labelsize=FS_TICK)
    # Equal-aspect panels can occupy very different vertical fractions across
    # subjects.  Anchor the subject header to the rendered TA/TB title boxes so
    # it stays close without overlapping tall/narrow geometries such as E922.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_top = max(
        ax.title.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
        for ax in axes
    )
    panel_left_x = min(ax.get_position().x0 for ax in axes)
    panel_right_x = max(ax.get_position().x1 for ax in axes)
    panel_bottom_y = min(ax.get_position().y0 for ax in axes)
    fig.text(
        0.5 * (panel_left_x + panel_right_x), panel_bottom_y - 0.055,
        "Main Propagation Axis (mm)",
        ha="center", va="top", fontsize=FS_XLABEL,
    )
    fig.text(
        panel_left_x, title_top + 0.018, f"{pretty} · {mode_label}",
        ha="left", va="bottom", fontsize=FS_SUBJECT_TITLE, fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{subject_id}_interictal_AB.{output_format}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_interictal_ab_atlas(
    records: Sequence[Mapping[str, object]], output_dir: Path, *, subject_columns: int = 4,
    yuquan_labels: Mapping[str, str],
    display_sigma_mm: float = DEFAULT_DISPLAY_SIGMA_MM,
    output_format: str = "png",
) -> Path:
    n = len(records)
    nrows = int(np.ceil(n / subject_columns))
    fig, axes = plt.subplots(
        nrows,
        2 * subject_columns,
        figsize=(5.0 * subject_columns, 3.25 * nrows),
        squeeze=False,
        layout="constrained",
    )
    for index, record in enumerate(records):
        row = index // subject_columns
        col = 2 * (index % subject_columns)
        dat_a, dat_b, mode = build_interictal_ab_panel_payloads(
            record, display_sigma_mm=display_sigma_mm,
        )
        pretty = _display_name(str(record["subject_id"]), yuquan_labels)
        mode_tag = "shared" if mode == "shared" else "separate"
        draw_interictal_rank_field_panel(
            axes[row, col], dat_a, "TA", compact=True,
            panel_title=f"{pretty} · {mode_tag}",
        )
        draw_interictal_rank_field_panel(
            axes[row, col + 1], dat_b, "TB", compact=True,
        )
    for index in range(n, nrows * subject_columns):
        row = index // subject_columns
        col = 2 * (index % subject_columns)
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.018, pad=0.025,
    )
    cbar.set_label("interictal propagation rank: early 0 → late 1")
    fig.suptitle(
        f"Frozen interictal TA/TB rank fields — all {n} axis-estimable subjects",
        fontsize=17, fontweight="bold", x=0.01, ha="left",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"all_subjects_interictal_AB.{output_format}"
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return path


def load_interictal_field_records(
    input_dir: Path, subjects: Iterable[str] | None,
) -> list[Dict[str, object]]:
    if subjects:
        paths = [input_dir / f"{str(subject).removesuffix('.json')}.json" for subject in subjects]
    else:
        paths = sorted(input_dir.glob("*.json"))
    records = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text())
        field = record.get("interictal_field") or {}
        if record.get("status") != "ok" or field.get("status") != "ok":
            continue
        records.append(record)
    return sorted(records, key=lambda value: str(value["subject_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--display-sigma-mm", type=float, default=DEFAULT_DISPLAY_SIGMA_MM,
        help="display-only Gaussian bandwidth in mm; frozen analysis kernels are unchanged",
    )
    parser.add_argument(
        "--yuquan-crosswalk", type=Path, default=DEFAULT_YUQUAN_CROSSWALK,
        help="ignored private markdown mapping Yuquan artifact folders to public Y IDs",
    )
    parser.add_argument(
        "--format", choices=("png", "pdf"), default="png",
        help="output format; use PDF when exporting a paper-figure candidate",
    )
    parser.add_argument("--no-atlas", action="store_true")
    args = parser.parse_args()

    records = load_interictal_field_records(args.input_dir, args.subjects)
    if not records:
        raise RuntimeError("no axis-estimable interictal field records found")
    yuquan_labels = _load_yuquan_crosswalk(args.yuquan_crosswalk)
    for record in records:
        path = plot_interictal_ab_subject(
            record, args.output_dir, yuquan_labels=yuquan_labels,
            display_sigma_mm=args.display_sigma_mm,
            output_format=args.format,
        )
        relation = record["axis_pair"]["relation"]["relation"]
        mode = "shared" if "shared_a" in record["interictal_field"]["field_models"] else "own"
        dat_a, dat_b, _ = build_interictal_ab_panel_payloads(
            record, display_sigma_mm=args.display_sigma_mm,
        )
        print(
            f"[fig] {path.name}  plane={mode} relation={relation} "
            f"transverse_signs={dat_a['transverse_sign']:+d}/{dat_b['transverse_sign']:+d} "
            f"rmse={dat_a['transverse_alignment_rmse_mm']:.2f}mm",
            flush=True,
        )
    if not args.no_atlas:
        path = plot_interictal_ab_atlas(
            records, args.output_dir, yuquan_labels=yuquan_labels,
            display_sigma_mm=args.display_sigma_mm,
            output_format=args.format,
        )
        print(f"[atlas] {path.name}", flush=True)
    print(
        f"[done] {len(records)} subjects, display_sigma={args.display_sigma_mm:g} mm "
        f"-> {args.output_dir}"
    )


if __name__ == "__main__":
    main()

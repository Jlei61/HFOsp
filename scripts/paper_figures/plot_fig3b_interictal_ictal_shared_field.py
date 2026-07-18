#!/usr/bin/env python3
"""Build Fig. 3B: frozen interictal timing vs early-ictal broadband field.

The left panel is the frozen TA timing field on E1146's shared plane.  The
right panel is the broadband (1--150 Hz) enhancement from the complete seizure
with maximum positive shared-TA concordance over clinical onset [0, 10] s.  It
uses the accepted distal-baseline robust-z contract and is projected without
refitting, ranking, or sign flipping.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_event_envelope_field import (  # noqa: E402
    _event_field,
    load_frozen,
)
from scripts.run_topic5_clinical_onset_shared_field_concordance import (  # noqa: E402
    BAND,
    CLINICAL_WINDOW,
    MIN_BASELINE_FRAMES,
    _extract_bounds,
    clinical_relative_times,
    highest_valid_broadband_upper,
)
from scripts.run_topic5_eeg_onset_shared_field_concordance import (  # noqa: E402
    _eeg_offset_from_inventory,
    _inventory_for_subject,
    select_shared_scorers,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE  # noqa: E402
from scripts.run_topic5_tspectral_field_concordance import (  # noqa: E402
    SPECTRAL_HOP_SEC,
    SPECTRAL_WINDOW_SEC,
    _extract_log_band_power,
)
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.topic5_ictal_recruitment import bipolar_alias_label  # noqa: E402
from src.topic5_template_axis_field import scorers_from_interictal_record  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    DISTAL_BASELINE_EEG_SEC,
    aggregate_complete_windows,
    distal_baseline_robust_z,
    exact_name_align_matrix,
    score_observed_bundle,
)


FROZEN_ROOT = (
    ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
)
CHECKPOINT_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/tspectral_field_concordance/per_subject"
    / "clinical_onset_shared_field"
)
OUT_DIR = (
    ROOT
    / "results/paper-ready-figure/fig3b_interictal_ictal_shared_field"
    / "figures"
)

SCHEMA_ID = "fig3b_interictal_ictal_shared_field_candidate_v2"
INTERICTAL_CMAP = "viridis"
ICTAL_CMAP = "magma_r"
TEMPLATE_COLORS = {"A": "#B2182B", "B": "#2166AC"}
FIGSIZE = (7.25, 3.65)
DISPLAY_DPI = 300


def _normalize_minmax(values: Sequence[float]) -> np.ndarray:
    """Continuous min-max display transform; never rank or sign-flip values."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("cannot normalize an all-missing vector")
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanmax(values[finite]))
    if not hi > lo:
        raise ValueError("cannot normalize a constant vector")
    out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def _load_record(ds_sid: str) -> tuple[dict, Path]:
    path = FROZEN_ROOT / f"{ds_sid}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    # Fingerprint verification is intentionally repeated here even though
    # load_frozen() performs the same fail-closed gate for the display payload.
    scorers_from_interictal_record(record)
    return record, path


def _extract_clinical_activation(
    ds_sid: str,
    seizure_idx: int,
    record: Mapping[str, object],
) -> tuple[np.ndarray, dict]:
    dataset, sid = ds_sid.split("_", 1)
    if dataset != "epilepsiae":
        raise NotImplementedError("Fig3-B currently requires Epilepsiae clinical onset")

    inventory = _inventory_for_subject(dataset, sid)
    if not 0 <= int(seizure_idx) < len(inventory):
        raise IndexError(f"seizure index {seizure_idx} outside inventory n={len(inventory)}")
    inventory_row = inventory[int(seizure_idx)]
    eeg_rel_clinical = _eeg_offset_from_inventory(dataset, inventory_row)
    pre_sec, post_sec = _extract_bounds(eeg_rel_clinical)
    seizure = extract_seizure_window(
        f"{dataset}/{sid}",
        int(seizure_idx),
        pre_sec=pre_sec,
        post_sec=post_sec,
        reference=ICTAL_REFERENCE[dataset],
    )

    target_names = [
        str(name)
        for name in (record.get("interictal_field") or {}).get("contact_order", [])
    ]
    raw_names = [bipolar_alias_label(name) for name in seizure.ch_names]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("raw channel aliases are not unique")
    raw_index = {name: index for index, name in enumerate(raw_names)}
    matched_names = [name for name in target_names if name in raw_index]
    if len(matched_names) < 6:
        raise ValueError(f"fewer than 6 exact-name contacts: {len(matched_names)}")

    signal = seizure.signal[[raw_index[name] for name in matched_names]]
    upper_hz = highest_valid_broadband_upper(seizure.fs)
    powers, frame_times_crop = _extract_log_band_power(
        signal,
        seizure.fs,
        [BAND],
        band_hz_override={BAND: (1.0, upper_hz)},
    )
    frame_times_clinical = clinical_relative_times(frame_times_crop, seizure.pre_sec)
    baseline_clinical = (
        float(eeg_rel_clinical + DISTAL_BASELINE_EEG_SEC[0]),
        float(eeg_rel_clinical + DISTAL_BASELINE_EEG_SEC[1]),
    )
    robust = distal_baseline_robust_z(
        powers[BAND],
        frame_times_clinical,
        baseline_clinical,
        min_frames=MIN_BASELINE_FRAMES,
    )
    aligned = exact_name_align_matrix(record, matched_names, robust["delta"])
    window_grid = np.asarray(
        [[CLINICAL_WINDOW[0], CLINICAL_WINDOW[1], np.mean(CLINICAL_WINDOW)]],
        dtype=float,
    )
    rows, complete = aggregate_complete_windows(
        aligned["values"],
        frame_times_clinical,
        window_grid,
        spectral_window_sec=SPECTRAL_WINDOW_SEC,
    )
    if not bool(complete[0]):
        raise ValueError("clinical onset [0,10] s window is incomplete")
    activation = np.asarray(rows[0], dtype=float)
    if int(np.isfinite(activation).sum()) < 6:
        raise ValueError("fewer than 6 finite clinical-onset activation contacts")

    return activation, {
        "seizure_id": str(seizure.seizure_id),
        "reference": str(ICTAL_REFERENCE[dataset]),
        "sample_rate_hz": float(seizure.fs),
        "eeg_onset_minus_clinical_sec": float(eeg_rel_clinical),
        "extraction_pre_sec": float(pre_sec),
        "extraction_post_sec": float(post_sec),
        "band_hz": [1.0, float(upper_hz)],
        "is_exact_1_150": bool(np.isclose(upper_hz, 150.0)),
        "spectral_window_sec": float(SPECTRAL_WINDOW_SEC),
        "spectral_hop_sec": float(SPECTRAL_HOP_SEC),
        "baseline_reference": "EEG onset",
        "baseline_eeg_sec": list(map(float, DISTAL_BASELINE_EEG_SEC)),
        "baseline_clinical_sec": list(map(float, baseline_clinical)),
        "clinical_window_sec": list(map(float, CLINICAL_WINDOW)),
        "n_baseline_frames": int(robust["n_baseline_frames"]),
        "n_target_contacts": int(aligned["n_target"]),
        "n_matched_contacts": int(aligned["n_matched"]),
        "n_finite_contacts": int(np.isfinite(activation).sum()),
        "missing_contacts": list(aligned["missing_names"]),
    }


def _checkpoint_event(ds_sid: str, seizure_idx: int) -> tuple[dict | None, Path]:
    path = CHECKPOINT_ROOT / ds_sid / f"seizure_{int(seizure_idx):03d}.json"
    if not path.exists():
        return None, path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("event"), path


def _best_template_a_seizure(ds_sid: str, n_target_contacts: int) -> tuple[int, list[dict]]:
    """Select the available exact-band seizure with maximum signed TA concordance."""
    subject_dir = CHECKPOINT_ROOT / ds_sid
    rows = []
    for path in sorted(subject_dir.glob("seizure_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        event = payload.get("event") or {}
        if event.get("status") != "included":
            continue
        row = {
            "seizure_idx": int(event["seizure_idx"]),
            "seizure_id": str(event["seizure_id"]),
            "shared_a_signed": float(event["shared_a_signed"]),
            "shared_a_abs": float(event["shared_a_abs"]),
            "shared_b_signed": float(event["shared_b_signed"]),
            "shared_b_abs": float(event["shared_b_abs"]),
            "shared_best_template": str(event["shared_best_template"]),
            "shared_maxab": float(event["shared_maxab"]),
            "is_exact_1_150": bool(event["is_exact_1_150"]),
            "n_finite_contacts": int(event["n_finite_contacts"]),
        }
        if row["is_exact_1_150"] and row["n_finite_contacts"] == int(n_target_contacts):
            rows.append(row)
    if not rows:
        raise ValueError(f"{ds_sid}: no complete exact 1-150 Hz clinical checkpoints")
    rows.sort(key=lambda row: (-row["shared_a_signed"], row["seizure_idx"]))
    return int(rows[0]["seizure_idx"]), rows


def _score_audit(
    record: Mapping[str, object],
    activation: np.ndarray,
    checkpoint: Mapping[str, object] | None,
) -> dict:
    shared = select_shared_scorers(scorers_from_interictal_record(record))
    observed = score_observed_bundle(shared, activation)
    keys = (
        "shared_a_signed",
        "shared_a_abs",
        "shared_b_signed",
        "shared_b_abs",
        "shared_best_template",
        "shared_maxab",
    )
    current = {key: observed.get(key) for key in keys}
    comparison = None
    if checkpoint is not None:
        comparison = {}
        for key in keys:
            expected = checkpoint.get(key)
            actual = current.get(key)
            if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
                comparison[key] = {
                    "expected": float(expected),
                    "actual": float(actual),
                    "abs_error": abs(float(expected) - float(actual)),
                }
            else:
                comparison[key] = {"expected": expected, "actual": actual}
    return {"observed": current, "checkpoint_comparison": comparison}


def _draw_field(
    ax: plt.Axes,
    fz: Mapping[str, object],
    values: np.ndarray,
    support: np.ndarray,
    *,
    cmap: str,
    colorbar_values: np.ndarray,
    title: str,
    title_color: str,
    show_y: bool,
) -> ScalarMappable:
    x_grid, y_grid, field, _, _ = _event_field(fz, values, support)
    points = np.asarray(fz["points_mm"], dtype=float)
    ax.imshow(
        field,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        aspect="equal",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    finite = np.isfinite(values)
    ax.scatter(
        points[finite, 0],
        points[finite, 1],
        c=values[finite],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=38,
        edgecolors="white",
        linewidths=0.85,
        zorder=3,
    )
    ax.set_xlim(float(x_grid.min()), float(x_grid.max()))
    ax.set_ylim(float(y_grid.min()), float(y_grid.max()))
    ax.set_title(title, fontsize=11.0, pad=7, color=title_color, fontweight="bold")
    ax.set_xlabel("shared TA axis (mm)", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8, length=2.2)
    if show_y:
        ax.set_ylabel("transverse (mm)", fontsize=9.5)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    colorbar_values = np.asarray(colorbar_values, dtype=float)
    finite_colorbar = colorbar_values[np.isfinite(colorbar_values)]
    if finite_colorbar.size == 0:
        raise ValueError("colorbar values are all missing")
    return ScalarMappable(
        Normalize(float(finite_colorbar.min()), float(finite_colorbar.max())),
        cmap=cmap,
    )


def render(
    ds_sid: str,
    fz: Mapping[str, object],
    activation: np.ndarray,
    out_png: Path,
    out_pdf: Path,
) -> dict:
    label = "A"
    rank = np.asarray(fz["rank_a" if label == "A" else "rank_b"], dtype=float)
    support = np.asarray(
        fz["support_a" if label == "A" else "support_b"], dtype=float
    )
    interictal_display = _normalize_minmax(rank)
    ictal_display = _normalize_minmax(activation)

    fig = plt.figure(figsize=FIGSIZE, layout="constrained", facecolor="white")
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.045, 1.0, 0.045], wspace=0.08)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 2], sharey=left)
    left_map = _draw_field(
        left,
        fz,
        interictal_display,
        support,
        cmap=INTERICTAL_CMAP,
        colorbar_values=rank,
        title="TA fields",
        title_color=TEMPLATE_COLORS[label],
        show_y=True,
    )
    right_map = _draw_field(
        right,
        fz,
        ictal_display,
        support,
        cmap=ICTAL_CMAP,
        colorbar_values=activation,
        title="Broadband power",
        title_color="black",
        show_y=False,
    )
    pretty = f"E{ds_sid.split('_', 1)[1]}"
    fig.suptitle(pretty, x=0.015, ha="left", fontsize=13.5, fontweight="bold")

    cbar_left = fig.colorbar(left_map, cax=fig.add_subplot(grid[0, 1]))
    rank_lo = float(np.nanmin(rank))
    rank_hi = float(np.nanmax(rank))
    rank_mid = 0.5 * (rank_lo + rank_hi)
    cbar_left.set_ticks([rank_lo, rank_mid, rank_hi])
    cbar_left.set_ticklabels(
        [f"{rank_lo:g} (early)", f"{rank_mid:g}", f"{rank_hi:g} (late)"]
    )
    cbar_left.ax.set_title("propagation\nrank", fontsize=7.5, pad=5)
    cbar_left.ax.tick_params(labelsize=7.5, length=2)

    cbar_right = fig.colorbar(right_map, cax=fig.add_subplot(grid[0, 3]))
    power_lo = float(np.nanmin(activation))
    power_hi = float(np.nanmax(activation))
    power_ticks = np.linspace(power_lo, power_hi, 4)
    cbar_right.set_ticks(power_ticks)
    cbar_right.set_ticklabels([f"{value:.2f}" for value in power_ticks])
    cbar_right.ax.set_title("power\n(robust z)", fontsize=7.5, pad=5)
    cbar_right.ax.tick_params(labelsize=7.5, length=2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DISPLAY_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "figure_size_in": list(map(float, FIGSIZE)),
        "png_dpi": int(DISPLAY_DPI),
        "display_xlim_mm": list(map(float, fz["display_xlim_mm"])),
        "display_ylim_mm": list(map(float, fz["display_ylim_mm"])),
        "display_sigma_mm": float(fz["display_sigma_mm"]),
        "smoothing_contract": "identical support, extent, and 6 mm display sigma in both panels",
        "interictal_colormap": INTERICTAL_CMAP,
        "ictal_colormap": ICTAL_CMAP,
        "color_semantics": "dark means earliest propagation on the left and highest broadband power on the right",
        "interictal_colorbar_quantity": "raw propagation rank",
        "interictal_colorbar_limits": [rank_lo, rank_hi],
        "ictal_colorbar_quantity": "0-10 s mean baseline-normalized log-band-power robust z",
        "ictal_colorbar_limits": [power_lo, power_hi],
        "interictal_transform": "min-max of frozen template rank for interpolation; colorbar restored to raw rank",
        "ictal_transform": "continuous min-max of 0-10 s mean robust-z for interpolation; colorbar restored to robust-z; no rank; no sign flip",
        "template_support_used_for_both_panels": f"support_{label.lower()}",
        "transverse_sign": int(fz["transverse_sign"]),
        "outputs": {"png": str(out_png), "pdf": str(out_pdf)},
    }


def _write_readme(out_dir: Path, stem: str, seizure_idx: int) -> Path:
    text = f"""# Fig3-B：间期时序场与发作早期能量场

### {stem}.png / .pdf

E1146 的冻结 shared plane 配对图。左侧为红色语义标题的 TA early-to-late timing field；右侧为 E1146 全部可用发作中 `shared_a_signed` 最高的 seizure {seizure_idx}，显示 clinical onset `0–10 s` 内 broadband `1–150 Hz` baseline-normalized power，使用 `magma_r`，且不做 rank 或 sign flip。两幅图严格复用同一 contact order、shared TA axis、transverse sign、TA support 与同一个 6 mm display kernel。

**关注点**：这是一个 representative-subject shared-field readout，用于连接间期传播轴与同次发作早期能量分布；不能单独解释为 replay、因果机制或 cohort 结论。

### {stem}_metadata.json

记录 raw seizure、临床窗、远端 baseline、频谱参数、冻结 fingerprint、A/B 匹配分数、逐触点原始值与显示归一化。重画时必须先通过 checkpoint score parity，不能从 ictal 值重拟合轴、平面、support 或 kernel。

**关注点**：两条 colorbar 分别恢复为真实 propagation rank 与 robust-z 数值；`magma_r` 令高 broadband power 为深色，与左图“早期为深色”的视觉方向一致。右图不是 contact rank，也不改变当前 maxAB 科学统计。
"""
    path = out_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument(
        "--seizure-idx", type=int, default=None,
        help="explicit inventory index; default selects maximum positive shared_a_signed",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    ds_sid = str(args.subject).replace("/", "_")
    record, frozen_path = _load_record(ds_sid)
    fz = load_frozen(ds_sid)
    best_seizure_idx, selection_rows = _best_template_a_seizure(ds_sid, len(fz["names"]))
    seizure_idx = best_seizure_idx if args.seizure_idx is None else int(args.seizure_idx)
    activation, extraction = _extract_clinical_activation(
        ds_sid, seizure_idx, record
    )
    checkpoint, checkpoint_path = _checkpoint_event(ds_sid, seizure_idx)
    audit = _score_audit(record, activation, checkpoint)
    template = "A"
    if args.seizure_idx is None and seizure_idx != best_seizure_idx:
        raise AssertionError("automatic TA exemplar selection drifted")
    if args.seizure_idx is None and audit["observed"]["shared_best_template"] != "A":
        raise ValueError("the maximum-positive-TA exemplar is not maxAB winner A")

    errors = []
    for key, comparison in (audit.get("checkpoint_comparison") or {}).items():
        if "abs_error" in comparison:
            errors.append(float(comparison["abs_error"]))
        elif comparison.get("expected") != comparison.get("actual"):
            raise ValueError(f"checkpoint mismatch for {key}: {comparison}")
    max_score_error = max(errors, default=0.0)
    if checkpoint is not None and max_score_error > 1e-12:
        raise ValueError(f"checkpoint score parity failed: max error {max_score_error:g}")

    stem = f"{ds_sid}_seizure_{seizure_idx:02d}_interictal_ictal_shared_field"
    out_png = args.output_dir / f"{stem}.png"
    out_pdf = args.output_dir / f"{stem}.pdf"
    display = render(
        ds_sid,
        fz,
        activation,
        out_png,
        out_pdf,
    )

    rank = np.asarray(fz["rank_a" if template == "A" else "rank_b"], dtype=float)
    support = np.asarray(
        fz["support_a" if template == "A" else "support_b"], dtype=float
    )
    metadata = {
        "schema_id": SCHEMA_ID,
        "status": "paper-ready Fig3-B candidate",
        "paper_role": "Fig3-B interictal timing versus early-ictal shared field",
        "canonical_producer": "scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py",
        "subject": ds_sid,
        "seizure_idx": seizure_idx,
        "frozen_record": str(frozen_path.relative_to(ROOT)),
        "frozen_contract": record.get("contract"),
        "frozen_fingerprint": record["interictal_field"]["fingerprint_sha256"],
        "axis_definition": record.get("axis_definition"),
        "axis_direction_convention": record.get("axis_direction_convention"),
        "field_plane": "shared",
        "selected_template": template,
        "template_selection": "TA fixed; seizure chosen by maximum positive shared_a_signed across complete exact-band E1146 checkpoints",
        "seizure_selection": {
            "criterion": "maximum shared_a_signed",
            "n_candidates": len(selection_rows),
            "automatic": args.seizure_idx is None,
            "selected_seizure_idx": seizure_idx,
            "best_available_seizure_idx": best_seizure_idx,
            "rows_sorted_by_shared_a_signed": selection_rows,
        },
        "contact_order": list(fz["names"]),
        "raw_interictal_rank": rank.tolist(),
        "interictal_display_values": _normalize_minmax(rank).tolist(),
        "template_support": support.tolist(),
        "raw_ictal_robust_z_mean": activation.tolist(),
        "ictal_display_values": _normalize_minmax(activation).tolist(),
        "ictal_extraction": extraction,
        "score_audit": audit,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_present": checkpoint is not None,
        "checkpoint_max_abs_score_error": float(max_score_error),
        "display": display,
        "claim_scope": [
            "representative-subject bridge from a frozen interictal timing field to selected-seizure early-ictal broadband power",
            "not a template-free reconstruction",
            "not replay evidence, a cohort statistic, or a causal/mechanistic test",
        ],
    }
    meta_path = args.output_dir / f"{stem}_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = _write_readme(args.output_dir, stem, seizure_idx)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    print(f"wrote {meta_path}")
    print(f"wrote {readme}")
    print(
        f"shared winner=T{template}; maxAB={audit['observed']['shared_maxab']:.6f}; "
        f"checkpoint max error={max_score_error:.3g}"
    )


if __name__ == "__main__":
    main()

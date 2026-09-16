#!/usr/bin/env python3
"""Animate the Figure 3C ictal field over the Figure 3E peri-onset axis.

The frozen TA plane, contact order, support and display geometry are inherited
from the canonical Figure 3C record. Only the 10-s broadband ictal activation
window changes between frames. A moving cursor links each field frame to its
amplitude-aware expression of the frozen TA and TB interictal fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.patient_public_labels import public_patient_label  # noqa: E402
from scripts.paper_figures.plot_fig3b_interictal_ictal_shared_field import (  # noqa: E402
    BAND,
    CHECKPOINT_ROOT,
    FROZEN_ROOT,
    INTERICTAL_CMAP,
    ICTAL_CMAP,
    ICTAL_REFERENCE,
    LOCKED_SEIZURE_IDX,
    MIN_BASELINE_FRAMES,
    SPECTRAL_HOP_SEC,
    SPECTRAL_WINDOW_SEC,
    TEMPLATE_COLORS,
    _eeg_offset_from_inventory,
    _event_field,
    _extract_bounds,
    _extract_log_band_power,
    _inventory_for_subject,
    _load_record,
    _normalize_minmax,
    clinical_relative_times,
    highest_valid_broadband_upper,
    load_frozen,
)
from scripts.run_topic5_eeg_onset_shared_field_concordance import (  # noqa: E402
    select_shared_scorers,
)
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.supplementary_figure_style import apply_supplementary_rcparams  # noqa: E402
from src.topic5_ictal_recruitment import bipolar_alias_label  # noqa: E402
from src.topic5_template_axis_field import scorers_from_interictal_record  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    DISTAL_BASELINE_EEG_SEC,
    aggregate_complete_windows,
    distal_baseline_robust_z,
    exact_name_align_matrix,
    score_observed_bundle,
)


OUT_ROOT = (
    ROOT
    / "results/paper-ready-figure/supplementary-video-2-fig3c-peri-onset-field"
)
FIG_DIR = OUT_ROOT / "figures"
CANONICAL_GIF = ROOT / "results/paper-ready-figure/supplementary-video-2.gif"
CANONICAL_METADATA = (
    ROOT / "results/paper-ready-figure/supplementary-video-2_metadata.json"
)
FIG3C_METADATA = ROOT / "results/paper-ready-figure/fig3/fig3_panelc_metadata.json"
SCHEMA_ID = "fig3c_peri_onset_field_evolution_video_v1"
DEFAULT_SUBJECT = "epilepsiae_1146"
DEFAULT_START_SEC = -120.0
DEFAULT_STOP_SEC = 20.0
DEFAULT_WINDOW_SEC = 10.0
DEFAULT_STEP_SEC = 2.0
DEFAULT_FPS = 6.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_grid(start_sec: float, stop_sec: float, window_sec: float, step_sec: float) -> np.ndarray:
    last_start = float(stop_sec) - float(window_sec)
    if last_start < float(start_sec):
        raise ValueError("stop_sec - window_sec precedes start_sec")
    starts = np.arange(float(start_sec), last_start + 1e-9, float(step_sec))
    if starts.size == 0 or not np.isclose(starts[-1], last_start):
        starts = np.append(starts, last_start)
    return np.column_stack([starts, starts + window_sec, starts + window_sec / 2.0])


def extract_dynamic_activation(
    ds_sid: str,
    seizure_idx: int,
    record: Mapping[str, object],
    *,
    start_sec: float,
    stop_sec: float,
    window_sec: float,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return target-contact robust-z activation for complete peri-onset windows."""
    dataset, sid = ds_sid.split("_", 1)
    if dataset != "epilepsiae":
        raise NotImplementedError("the Figure 3C video is locked to Epilepsiae")
    inventory = _inventory_for_subject(dataset, sid)
    if not 0 <= int(seizure_idx) < len(inventory):
        raise IndexError(f"seizure index {seizure_idx} outside inventory n={len(inventory)}")
    inventory_row = inventory[int(seizure_idx)]
    eeg_rel_clinical = _eeg_offset_from_inventory(dataset, inventory_row)
    pre_default, post_default = _extract_bounds(eeg_rel_clinical)
    # Preserve the exact extraction anchor used by canonical Figure 3C.  A
    # rounded pre-window shifts the 0.5-s spectral grid and changes robust-z.
    pre_sec = float(pre_default)
    post_sec = max(float(post_default), float(stop_sec) + 30.0)
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
    if matched_names != target_names:
        missing = [name for name in target_names if name not in raw_index]
        raise ValueError(f"Figure 3C exact contact-order match failed; missing={missing}")

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
    windows = _window_grid(start_sec, stop_sec, window_sec, step_sec)
    values, complete = aggregate_complete_windows(
        aligned["values"],
        frame_times_clinical,
        windows,
        spectral_window_sec=SPECTRAL_WINDOW_SEC,
    )
    if not bool(np.all(complete)):
        bad = np.where(~complete)[0].tolist()
        raise ValueError(f"incomplete peri-onset windows: {bad}")
    values = np.asarray(values, dtype=float)
    if values.shape != (len(windows), len(target_names)):
        raise ValueError(
            f"activation shape {values.shape} != {(len(windows), len(target_names))}"
        )
    return windows, values, {
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
        "n_baseline_frames": int(robust["n_baseline_frames"]),
        "n_target_contacts": int(aligned["n_target"]),
        "n_matched_contacts": int(aligned["n_matched"]),
        "missing_contacts": list(aligned["missing_names"]),
    }


def _score_frames(record: Mapping[str, object], values: np.ndarray) -> dict[str, np.ndarray]:
    shared = select_shared_scorers(scorers_from_interictal_record(record))
    rows = [score_observed_bundle(shared, row) for row in values]
    q_ta_signed = np.asarray(
        [row["shared_a_signed_projection_z"] for row in rows], float
    )
    q_tb_signed = np.asarray(
        [row["shared_b_signed_projection_z"] for row in rows], float
    )
    q_ta_abs = np.abs(q_ta_signed)
    q_tb_abs = np.abs(q_tb_signed)
    return {
        "ta_signed": np.asarray([row["shared_a_signed"] for row in rows], float),
        "tb_signed": np.asarray([row["shared_b_signed"] for row in rows], float),
        "maxab_abs": np.asarray([row["shared_maxab"] for row in rows], float),
        "r_winner": np.asarray([row["shared_best_template"] for row in rows], object),
        "q_ta_signed": q_ta_signed,
        "q_tb_signed": q_tb_signed,
        "q_max": np.maximum(q_ta_abs, q_tb_abs),
        "q_winner": np.where(q_ta_abs >= q_tb_abs, "A", "B"),
    }


def _phase(center: float) -> str:
    if center < -90.0:
        return "Baseline"
    if center < 0.0:
        return "Pre-ictal"
    if center <= 10.0:
        return "Early ictal"
    return "Post-onset"


def _time_text(value: float) -> str:
    return "0" if np.isclose(value, 0.0) else f"{value:+.0f}"


def render(
    ds_sid: str,
    seizure_idx: int,
    fz: Mapping[str, object],
    windows: np.ndarray,
    values: np.ndarray,
    scores: Mapping[str, np.ndarray],
    *,
    fps: float,
    out_gif: Path,
    out_poster: Path,
) -> dict:
    apply_supplementary_rcparams()
    rank = np.asarray(fz["rank_a"], float)
    support = np.asarray(fz["support_a"], float)
    interictal_display = _normalize_minmax(rank)
    x_grid, y_grid, ta_field, _, _ = _event_field(
        fz, interictal_display, support
    )
    points = np.asarray(fz["points_mm"], float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise ValueError("no positive robust-z activation in animation window")
    power_vmax = float(np.nanpercentile(positive, 98.0))
    power_vmax = max(1.0, float(np.ceil(power_vmax * 2.0) / 2.0))
    power_norm = Normalize(0.0, power_vmax, clip=True)
    centers = np.asarray(windows[:, 2], float)
    baseline_lo = float(centers.min() - (windows[0, 1] - windows[0, 0]) / 2.0)
    baseline_hi = -90.0

    fig = plt.figure(figsize=(8.9, 6.6), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[3.3, 1.25],
        width_ratios=[1.0, 0.045, 1.0, 0.045],
        left=0.09,
        right=0.93,
        bottom=0.10,
        top=0.94,
        hspace=0.32,
        wspace=0.18,
    )
    ax_ta = fig.add_subplot(grid[0, 0])
    cax_ta = fig.add_subplot(grid[0, 1])
    ax_ictal = fig.add_subplot(grid[0, 2], sharey=ax_ta)
    cax_ictal = fig.add_subplot(grid[0, 3])
    ax_curve = fig.add_subplot(grid[1, :])

    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
    ax_ta.imshow(
        ta_field,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=INTERICTAL_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    ax_ta.scatter(
        points[:, 0], points[:, 1], c=interictal_display,
        cmap=INTERICTAL_CMAP, vmin=0.0, vmax=1.0, s=32,
        edgecolors="white", linewidths=0.75, zorder=3,
    )
    ax_ta.set_title("TA field", color=TEMPLATE_COLORS["A"], fontweight="bold", pad=6)
    ax_ta.set_xlabel("shared TA axis (mm)")
    ax_ta.set_ylabel("Y (mm)")
    cb_ta = fig.colorbar(
        ScalarMappable(Normalize(0.0, 1.0), cmap=INTERICTAL_CMAP), cax=cax_ta
    )
    cb_ta.set_ticks([0.0, 0.5, 1.0])
    cb_ta.set_ticklabels(["0", "0.5", "1"])
    cb_ta.ax.set_title("ranks\nearly→late", loc="left", pad=4, fontsize=8.5)

    first_field = _event_field(fz, values[0], support)[2]
    ictal_im = ax_ictal.imshow(
        first_field,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=ICTAL_CMAP,
        norm=power_norm,
        interpolation="bilinear",
    )
    ictal_scatter = ax_ictal.scatter(
        points[:, 0], points[:, 1], c=values[0], cmap=ICTAL_CMAP,
        norm=power_norm, s=32, edgecolors="white", linewidths=0.75, zorder=3,
    )
    ax_ictal.set_xlabel("shared TA axis (mm)")
    ax_ictal.tick_params(axis="y", left=False, labelleft=False)
    cb_ictal = fig.colorbar(
        ScalarMappable(power_norm, cmap=ICTAL_CMAP), cax=cax_ictal
    )
    cb_ictal.ax.set_title("power\nz", pad=4)

    max_line, = ax_curve.plot(
        centers, scores["q_max"], color="0.12", lw=2.0,
        label="Q",
    )
    winner_colors = [
        TEMPLATE_COLORS["A"] if winner == "A" else TEMPLATE_COLORS["B"]
        for winner in scores["q_winner"]
    ]
    ax_curve.scatter(
        centers, scores["q_max"], c=winner_colors, s=13,
        edgecolors="white", linewidths=0.35, zorder=3,
    )
    ax_curve.axvspan(baseline_lo, baseline_hi, color="#DDE5F4", alpha=0.70, lw=0)
    ax_curve.axvspan(0.0, 10.0, color="#F2D9D5", alpha=0.65, lw=0)
    ax_curve.axvline(0.0, color="0.15", lw=1.0, ls="--")
    cursor = ax_curve.axvline(centers[0], color="0.1", lw=1.2)
    current_dot, = ax_curve.plot(
        [centers[0]], [scores["q_max"][0]], "o",
        ms=7.0, color=winner_colors[0], mec="white", mew=0.8, zorder=4,
    )
    ax_curve.set_xlim(float(windows[0, 0]), float(windows[-1, 1]))
    q_upper = max(0.1, float(np.nanmax(scores["q_max"])) * 1.08)
    ax_curve.set_ylim(0.0, q_upper)
    ax_curve.set_xlabel("Time from clinical onset (s)")
    ax_curve.set_ylabel("Template expression Q (z)")
    ax_curve.legend(
        handles=[
            max_line,
            Line2D([], [], marker="o", ls="none", color=TEMPLATE_COLORS["A"], label="TA dominant"),
            Line2D([], [], marker="o", ls="none", color=TEMPLATE_COLORS["B"], label="TB dominant"),
        ],
        frameon=False, loc="lower left", ncol=1,
    )
    ax_curve.grid(axis="y", color="0.90", lw=0.7)

    dataset, raw_subject = ds_sid.split("_", 1)
    pretty = public_patient_label(dataset, raw_subject)

    def update(index: int):
        lo, hi, center = map(float, windows[index])
        field = _event_field(fz, values[index], support)[2]
        ictal_im.set_data(field)
        ictal_scatter.set_array(np.asarray(values[index], float))
        ax_ictal.set_title(
            f"{pretty} | SZ{seizure_idx + 1}\n"
            f"{_phase(center)} field  [{_time_text(lo)}, {_time_text(hi)}] s",
            pad=6,
        )
        cursor.set_xdata([center, center])
        current_dot.set_data([center], [scores["q_max"][index]])
        current_dot.set_color(winner_colors[index])
        return ictal_im, ictal_scatter, cursor, current_dot

    onset_index = int(np.where(np.isclose(windows[:, 0], 0.0))[0][0])
    update(onset_index)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_poster, dpi=220, bbox_inches="tight", facecolor="white")
    animation = FuncAnimation(
        fig, update, frames=len(windows), interval=1000.0 / fps, blit=False
    )
    animation.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return {
        "n_frames": int(len(windows)),
        "fps": float(fps),
        "frame_duration_ms_requested": float(1000.0 / fps),
        "power_colorbar_limits_robust_z": [0.0, power_vmax],
        "power_colorbar_contract": (
            "fixed across frames; negative robust-z values clip to the light end; "
            "no per-frame rank or min-max normalization"
        ),
        "poster_frame_index": onset_index,
        "poster_window_sec": windows[onset_index].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--seizure-idx", type=int, default=LOCKED_SEIZURE_IDX)
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC)
    parser.add_argument("--stop-sec", type=float, default=DEFAULT_STOP_SEC)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--frozen-root", type=Path, default=FROZEN_ROOT)
    args = parser.parse_args()

    ds_sid = str(args.subject).replace("/", "_")
    record, frozen_path = _load_record(ds_sid, args.frozen_root)
    fz = load_frozen(ds_sid, frozen_root=args.frozen_root)
    windows, values, extraction = extract_dynamic_activation(
        ds_sid,
        int(args.seizure_idx),
        record,
        start_sec=float(args.start_sec),
        stop_sec=float(args.stop_sec),
        window_sec=float(args.window_sec),
        step_sec=float(args.step_sec),
    )
    if list(fz["names"]) != list(record["interictal_field"]["contact_order"]):
        raise ValueError("display and scoring contact order differ")
    scores = _score_frames(record, values)

    fig3c = json.loads(FIG3C_METADATA.read_text(encoding="utf-8"))
    if ds_sid != fig3c["subject"] or int(args.seizure_idx) != int(fig3c["seizure_idx"]):
        raise ValueError("requested case differs from canonical Figure 3C")
    if record["interictal_field"]["fingerprint_sha256"] != fig3c["frozen_fingerprint"]:
        raise ValueError("frozen field fingerprint differs from canonical Figure 3C")
    onset_index = int(np.where(np.isclose(windows[:, 0], 0.0))[0][0])
    static_values = np.asarray(fig3c["raw_ictal_robust_z_mean"], float)
    static_parity_error = float(np.nanmax(np.abs(values[onset_index] - static_values)))
    if static_parity_error > 1e-8:
        raise ValueError(f"Figure 3C 0-10 s parity failed: {static_parity_error:g}")

    stem = f"{ds_sid}_sz{int(args.seizure_idx) + 1}_fig3c_peri_onset_field_evolution"
    out_gif = args.output_dir / f"{stem}.gif"
    out_poster = args.output_dir / f"{stem}_poster.png"
    display = render(
        ds_sid,
        int(args.seizure_idx),
        fz,
        windows,
        values,
        scores,
        fps=float(args.fps),
        out_gif=out_gif,
        out_poster=out_poster,
    )

    with Image.open(out_gif) as encoded:
        encoded_frames = int(getattr(encoded, "n_frames", 1))
        encoded_duration_ms = int(encoded.info.get("duration", 0))
        encoded_loop = int(encoded.info.get("loop", 0))
    if encoded_frames != len(windows):
        raise ValueError(f"encoded GIF frame count {encoded_frames} != {len(windows)}")
    CANONICAL_GIF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_gif, CANONICAL_GIF)

    metadata = {
        "schema_id": SCHEMA_ID,
        "paper_slot": "Supplementary Video 2",
        "status": "author-locked supplementary video",
        "canonical_path": str(CANONICAL_GIF.relative_to(ROOT)),
        "paper_role": "Figure 3C peri-onset field-evolution companion",
        "subject": ds_sid,
        "seizure_idx": int(args.seizure_idx),
        "display_label": (
            f"{public_patient_label(*ds_sid.split('_', 1))} | "
            f"SZ{int(args.seizure_idx) + 1}"
        ),
        "canonical_static_figure": str(FIG3C_METADATA.relative_to(ROOT)),
        "frozen_record": str(frozen_path.relative_to(ROOT)),
        "frozen_fingerprint": record["interictal_field"]["fingerprint_sha256"],
        "field_plane": "shared",
        "selected_static_template": "TA",
        "own_field_fallback": False,
        "contact_order": list(fz["names"]),
        "template_support": np.asarray(fz["support_a"], float).tolist(),
        "window_contract": {
            "start_sec": float(args.start_sec),
            "stop_sec": float(args.stop_sec),
            "window_sec": float(args.window_sec),
            "step_sec": float(args.step_sec),
            "time_anchor": "clinical onset",
            "windows": windows.tolist(),
        },
        "ictal_extraction": extraction,
        "figure3c_static_parity": {
            "matched_window_sec": windows[onset_index].tolist(),
            "max_abs_robust_z_error": static_parity_error,
            "tolerance": 1e-8,
            "passed": True,
        },
        "frame_scores": {
            "ta_signed_r": scores["ta_signed"].tolist(),
            "tb_signed_r": scores["tb_signed"].tolist(),
            "maxab_abs_r": scores["maxab_abs"].tolist(),
            "r_winner": scores["r_winner"].tolist(),
            "q_ta_signed_projection_z": scores["q_ta_signed"].tolist(),
            "q_tb_signed_projection_z": scores["q_tb_signed"].tolist(),
            "Q_max_abs_projection_z": scores["q_max"].tolist(),
            "Q_dominant_template": scores["q_winner"].tolist(),
        },
        "display": display,
        "encoded_gif": {
            "n_frames": encoded_frames,
            "duration_ms_per_frame": encoded_duration_ms,
            "effective_fps": (
                None if encoded_duration_ms <= 0 else 1000.0 / encoded_duration_ms
            ),
            "loop": encoded_loop,
        },
        "outputs": {
            "gif": str(CANONICAL_GIF.relative_to(ROOT)),
            "source_gif": str(out_gif.relative_to(ROOT)),
            "poster": str(out_poster.relative_to(ROOT)),
            "gif_sha256": _sha256(CANONICAL_GIF),
            "source_gif_sha256": _sha256(out_gif),
            "poster_sha256": _sha256(out_poster),
        },
        "claim_boundary": (
            "Representative selected seizure visualization. It displays how the "
            "same Figure 3C field readout changes over time; it is not an independent "
            "cohort test, template-free replay proof, or mechanistic evidence."
        ),
    }
    metadata_path = OUT_ROOT / f"{stem}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    CANONICAL_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme = f"""### {stem}.gif

E10 | SZ3 的 Figure 3C 动态配套视频。上排固定显示冻结间期 TA 场，并用相同 shared plane、15 个触点、TA support 和 6 mm 显示核更新右侧 1–150 Hz 发作场；下排同步显示幅度感知模板表达量 $Q(t)=\max(|q_A(t)|,|q_B(t)|)$，红蓝点表示当时由 TA 或 TB 表达主导，黑色游标对应当前 10 s 滑窗。

**关注点**：动画覆盖临床起始前 120 s 至起始后 20 s，步长 2 s；右侧 power-z 色标在所有帧中固定，未作逐帧 rank 或 min–max 归一化，因此可以比较随时间变化的增强幅度。0–10 s 帧与正式 Figure 3C 的逐触点 robust-z 数值严格一致。

### {stem}_poster.png

GIF 中临床起始后 0–10 s 帧的静态海报，用于快速核对 Figure 3C 的空间模式和下方 $Q$ 表达量游标。

**关注点**：该病例经过代表性形态选择，只是 Figure 3C 的动态可视化补充，不构成独立队列检验、template-free replay 证据或机制证明。
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(out_gif)
    print(out_poster)
    print(metadata_path)


if __name__ == "__main__":
    main()

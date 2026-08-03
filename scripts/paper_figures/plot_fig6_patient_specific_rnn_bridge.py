#!/usr/bin/env python3
"""Build the paper-ready patient-specific, target-free RNN bridge figure.

The figure deliberately separates two claims.  Panels a--d use only
interictal data and show whether a self-supervised recurrent model recovers a
patient's held-out propagation structure.  Panels e--f read the already
frozen model field against the same patient's early-ictal broadband field.
Ictal targets are never used to fit or select a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_patient_specific_rnn_bridge import chronological_60_20_20  # noqa: E402
from src.topic5_rank_distribution import pairwise_precedence  # noqa: E402


SCHEMA_ID = "fig6_patient_specific_target_free_rnn_bridge_v1"
OUT_DIR = ROOT / "results/paper-ready-figure/fig6_patient_specific_rnn_bridge/figures"
FIG_BASENAME = "fig6_patient_specific_rnn_bridge"
REPRESENTATIVE = "epilepsiae_620"
MODEL = "full_history_gru"
CONTROL = "rank_shuffle_gru"
CMAP_NAME = "viridis"
MODEL_COLOR = "#287D78"
CONTROL_COLOR = "#B6B6B6"
TARGET_COLOR = "#A35E48"
SEEDS = (11, 29, 47)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, *roots: Path) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
    return str(resolved)


def _panel_label(ax: plt.Axes, label: str, *, x: float = -0.18) -> None:
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.8, length=3)


def _normalized_groups(groups: np.ndarray, counts: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups, dtype=float)
    counts = np.asarray(counts, dtype=float)
    denominator = np.maximum(counts - 1.0, 1.0)
    return np.where(groups >= 0, groups / denominator[:, None], np.nan)


def _largest_shaft_indices(names: Sequence[str]) -> np.ndarray:
    shafts: dict[str, list[tuple[int, int]]] = {}
    for index, name in enumerate(map(str, names)):
        shaft, ordinal = parse_shaft(name)
        if shaft is None or ordinal is None:
            continue
        shafts.setdefault(shaft, []).append((ordinal, index))
    if not shafts:
        raise ValueError("representative patient has no parseable contact shaft")
    chosen = max(shafts.values(), key=len)
    return np.asarray([index for _, index in sorted(chosen)], dtype=int)


def _event_orientation(groups: np.ndarray, shaft_indices: np.ndarray) -> np.ndarray:
    """Signed propagation direction along the largest displayed shaft."""
    score = np.full(len(groups), np.nan, dtype=float)
    position = np.arange(len(shaft_indices), dtype=float)
    for event_index, event in enumerate(np.asarray(groups, dtype=float)):
        ranks = event[shaft_indices]
        valid = ranks >= 0
        if int(valid.sum()) < 4:
            continue
        value = spearmanr(position[valid], ranks[valid]).statistic
        if np.isfinite(value):
            score[event_index] = float(value)
    return score


def _display_events(
    groups: np.ndarray,
    counts: np.ndarray,
    shaft_indices: np.ndarray,
    *,
    n_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    orientation = _event_orientation(groups, shaft_indices)
    eligible = np.flatnonzero(np.isfinite(orientation))
    if len(eligible) < n_events:
        raise ValueError(f"only {len(eligible)} direction-scoreable events")
    eligible = eligible[np.argsort(orientation[eligible], kind="stable")]
    take = np.rint(np.linspace(0, len(eligible) - 1, n_events)).astype(int)
    selected = eligible[take]
    return _normalized_groups(groups[selected], counts[selected]).T, orientation[selected]


def _plot_event_map(
    ax: plt.Axes,
    matrix: np.ndarray,
    names: Sequence[str],
    *,
    title: str,
    label: str,
    subtitle: str,
) -> None:
    cmap = plt.get_cmap(CMAP_NAME).copy()
    cmap.set_bad("#F0F0F0")
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(names)), list(map(str, names)))
    ax.set_xticks([0, matrix.shape[1] // 2, matrix.shape[1] - 1])
    ax.set_xticklabels(["reverse", "mixed", "forward"])
    ax.set_xlabel("Events sorted by propagation direction for display")
    ax.set_ylabel("SEEG contacts")
    ax.set_title(title, loc="left", pad=14, fontweight="bold")
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=6.5, color="0.35")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_label(ax, label, x=-0.16)


def _load_rollouts_and_fields(
    output: Path,
    subject: str,
    candidates: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[Path]]:
    groups = []
    counts = []
    fields = {candidate: [] for candidate in candidates}
    paths: list[Path] = []
    names = None
    for seed in SEEDS:
        path = output / "units" / subject / MODEL / f"seed_{seed}" / "free_rollouts.npz"
        paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            current = np.asarray(data["contact_names"]).astype(str)
            if names is None:
                names = current
            elif not np.array_equal(names, current):
                raise RuntimeError("contact order drift across model seeds")
            groups.append(np.asarray(data["event_group_ids"], dtype=np.int16))
            counts.append(np.asarray(data["event_group_count"], dtype=np.int16))
            for candidate in candidates:
                fields[candidate].append(np.asarray(data[f"field__{candidate}"], dtype=float))
    median_fields = {
        candidate: np.median(np.stack(values), axis=0)
        for candidate, values in fields.items()
    }
    return (
        np.asarray(names),
        np.concatenate(groups, axis=0),
        np.concatenate(counts, axis=0),
        median_fields,
        paths,
    )


def _load_target(cache_root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    paths = sorted((cache_root / f"outer_{subject}").glob(f"{subject}__*.npz"))
    if not paths:
        raise FileNotFoundError(f"{subject}: no early-ictal target files")
    names = None
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            current = np.asarray(data["contact_names"]).astype(str)
            if names is None:
                names = current
            elif not np.array_equal(names, current):
                raise RuntimeError("target contact order drift across seizures")
            rows.append(np.asarray(data["target_1_150"], dtype=float))
    return np.asarray(names), np.median(np.stack(rows), axis=0), paths


def _select_display_field(
    model_names: np.ndarray,
    fields: dict[str, np.ndarray],
    target_names: np.ndarray,
    target: np.ndarray,
    candidates: Sequence[str],
) -> tuple[np.ndarray, str, float, int]:
    lookup = {name: index for index, name in enumerate(model_names)}
    if not all(name in lookup for name in target_names):
        raise RuntimeError("model/target contact join is incomplete")
    keep = np.asarray([lookup[name] for name in target_names], dtype=int)
    best_name = ""
    best_rho = np.nan
    best_abs = -np.inf
    best_values = None
    for candidate in candidates:
        values = np.asarray(fields[candidate], dtype=float)[keep]
        rho = float(spearmanr(values, target).statistic)
        if abs(rho) > best_abs:
            best_abs = abs(rho)
            best_rho = rho
            best_name = str(candidate)
            best_values = values
    if best_values is None:
        raise RuntimeError("no display field could be selected")
    sign = -1 if best_rho < 0 else 1
    return np.asarray(best_values) * sign, best_name, abs(best_rho), sign


def _rank01(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(np.asarray(values, dtype=float), method="average")
    return (ranked - 1.0) / max(len(ranked) - 1.0, 1.0)


def _pca_projection(coords: np.ndarray) -> np.ndarray:
    centered = np.asarray(coords, dtype=float) - np.mean(coords, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vh[:2].T
    for column in range(2):
        nonzero = np.flatnonzero(np.abs(plane[:, column]) > 1e-9)
        if nonzero.size and plane[nonzero[0], column] > 0:
            plane[:, column] *= -1
    # Display the dominant implanted direction vertically.  This is only a
    # rigid 90-degree rotation of the PCA plane and does not change distances.
    return np.column_stack([plane[:, 1], plane[:, 0]])


def _plot_contact_field(
    ax: plt.Axes,
    xy: np.ndarray,
    names: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    show_y: bool,
) -> None:
    shafts: dict[str, list[tuple[int, int]]] = {}
    for index, name in enumerate(names):
        shaft, ordinal = parse_shaft(str(name))
        if shaft is not None and ordinal is not None:
            shafts.setdefault(shaft, []).append((ordinal, index))
    for items in shafts.values():
        indices = [index for _, index in sorted(items)]
        if len(indices) > 1:
            ax.plot(xy[indices, 0], xy[indices, 1], color="0.78", lw=1.2, zorder=1)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        cmap=CMAP_NAME,
        vmin=0,
        vmax=1,
        s=72,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    for (x, y), name in zip(xy, names):
        ax.annotate(str(name), (x, y), xytext=(3, 2), textcoords="offset points", fontsize=5.7)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=6)
    ax.set_xlabel("off-axis (mm)", fontsize=6.5)
    if show_y:
        ax.set_ylabel("dominant contact axis (mm)", fontsize=6.5)
    else:
        ax.set_yticklabels([])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=5.8, length=2)
    _clean_axis(ax)


def _paired_nll_panel(ax: plt.Axes, interictal: pd.DataFrame, primary_subjects: set[str]) -> dict:
    wide = (
        interictal.loc[interictal.subject.isin(primary_subjects)]
        .pivot(index="subject", columns="model", values="test_nll")
        .dropna(subset=[CONTROL, MODEL])
    )
    shuffled = wide[CONTROL].to_numpy(float)
    real = wide[MODEL].to_numpy(float)
    for left, right in zip(shuffled, real):
        ax.plot([0, 1], [left, right], color="0.78", lw=0.8, zorder=1)
    ax.scatter(np.zeros(len(shuffled)), shuffled, s=28, color=CONTROL_COLOR, edgecolor="white", lw=0.5, zorder=2)
    ax.scatter(np.ones(len(real)), real, s=30, color=MODEL_COLOR, edgecolor="white", lw=0.5, zorder=3)
    ax.plot([0, 1], [np.median(shuffled), np.median(real)], color="0.15", lw=2.0, marker="o", ms=4, zorder=4)
    ax.set_xticks([0, 1], ["Within-event\nranks shuffled", "Real rank\norder"])
    ax.set_ylabel("Held-out event NLL  ↓")
    ax.set_title("Real event order improves prediction", loc="left", fontweight="bold", pad=8)
    ax.text(
        0.03,
        0.04,
        "14/15 patients\nmedian gain 0.059 nats/event\nexact P = 1.2 × 10⁻⁴",
        transform=ax.transAxes,
        fontsize=6.6,
        va="bottom",
    )
    _clean_axis(ax)
    _panel_label(ax, "d")
    return {
        "n": int(len(wide)),
        "n_real_better": int(np.sum(real < shuffled)),
        "median_rank_shuffle_minus_real_nll": float(np.median(shuffled - real)),
    }


def _paired_field_panel(ax: plt.Axes, primary: pd.DataFrame) -> dict:
    frame = primary.loc[primary.model == MODEL].set_index("subject")
    observed = frame.observed_max_abs_rho.to_numpy(float)
    null = frame.all_contact_null_median.to_numpy(float)
    for true, shuffled in zip(observed, null):
        color = MODEL_COLOR if true > shuffled else "0.65"
        ax.plot([0, 1], [shuffled, true], color=color, alpha=0.55, lw=0.85, zorder=1)
    rng = np.random.default_rng(620)
    jitter = rng.normal(0, 0.025, len(frame))
    ax.scatter(jitter, null, s=27, color=CONTROL_COLOR, edgecolor="white", lw=0.5, zorder=2)
    ax.scatter(1 + jitter, observed, s=30, color=TARGET_COLOR, edgecolor="white", lw=0.5, zorder=3)
    ax.plot([0, 1], [np.median(null), np.median(observed)], color="0.15", lw=2.0, marker="o", ms=4, zorder=4)
    ax.set_xticks([0, 1], ["Channel-shuffle\nnull", "RNN-derived field\nvs early ictal"])
    ax.set_ylabel(r"Same-patient field similarity, max $|\rho|$")
    ax.set_ylim(0, 1.04)
    ax.set_title("Target-free cross-state correspondence", loc="left", fontweight="bold", pad=8)
    ax.text(
        0.03,
        0.96,
        "13/15 patients above null\nexact P = 0.026",
        transform=ax.transAxes,
        fontsize=6.6,
        va="top",
    )
    ax.text(
        0.03,
        0.04,
        "Stricter within-shaft sensitivity: n.s.\nIncrement over static scaffold: n.s.",
        transform=ax.transAxes,
        fontsize=6.0,
        color="0.35",
        va="bottom",
    )
    _clean_axis(ax)
    _panel_label(ax, "f")
    return {
        "n": int(len(frame)),
        "n_observed_above_null_median": int(np.sum(observed > null)),
        "median_observed_max_abs_rho": float(np.median(observed)),
        "median_all_contact_null": float(np.median(null)),
        "median_margin": float(np.median(observed - null)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_patient_specific_target_free_rnn_bridge_v0_1.yaml",
    )
    parser.add_argument("--representative", default=REPRESENTATIVE)
    parser.add_argument("--display-events", type=int, default=160)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    artifact_root = Path(config["artifact_root"]).resolve()
    dataset_root = artifact_root / config["dataset_root"]
    cache_root = ROOT / config["target_cache_root"]
    candidates = list(map(str, config["readout"]["candidate_fields"]))
    subject = str(args.representative)

    summary_path = output / "PATIENT_SPECIFIC_RNN_BRIDGE_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise RuntimeError("frozen patient-specific bridge is not complete")
    interictal = pd.read_csv(output / "interictal_patient_metrics.csv")
    ictal = pd.read_csv(output / "early_ictal_patient_metrics.csv")
    primary = ictal.loc[(ictal.band == "1_150") & ~ictal.development_supportive.astype(bool)].copy()
    primary_subjects = set(primary.subject.astype(str))

    record = load_records(dataset_root)[subject]
    _, _, test_indices = chronological_60_20_20(record)
    observed_groups = np.asarray(record.group_ids[test_indices], dtype=np.int16)
    observed_counts = np.asarray(record.group_count[test_indices], dtype=np.int16)
    record_names = np.asarray(record.contact_names).astype(str)

    model_names, rollout_groups, rollout_counts, model_fields, rollout_paths = _load_rollouts_and_fields(
        output, subject, candidates
    )
    if not np.array_equal(record_names, model_names):
        raise RuntimeError("dataset/model contact order mismatch")
    target_names, target, target_paths = _load_target(cache_root, subject)
    display_field, selected_field, display_abs_rho, display_sign = _select_display_field(
        model_names, model_fields, target_names, target, candidates
    )

    shaft_indices = _largest_shaft_indices(record_names)
    observed_map, observed_orientation = _display_events(
        observed_groups, observed_counts, shaft_indices, n_events=args.display_events
    )
    rollout_map, rollout_orientation = _display_events(
        rollout_groups, rollout_counts, shaft_indices, n_events=args.display_events
    )

    observed_precedence = pairwise_precedence(observed_groups)
    rollout_precedence = pairwise_precedence(rollout_groups)
    off_diagonal = ~np.eye(observed_precedence.shape[0], dtype=bool)
    valid_pairs = off_diagonal & np.isfinite(observed_precedence) & np.isfinite(rollout_precedence)
    display_precedence_r = float(
        np.corrcoef(observed_precedence[valid_pairs], rollout_precedence[valid_pairs])[0, 1]
    )
    formal_precedence_r = float(
        interictal.loc[(interictal.subject == subject) & (interictal.model == MODEL), "precedence_correlation"].iloc[0]
    )

    coord_result = load_subject_coords(
        "epilepsiae", subject.split("_", 1)[1], target_names.tolist()
    )
    if not bool(np.all(coord_result.mapped_mask_in_requested_order)):
        raise RuntimeError("representative contact plane is not fully mapped")
    xy = _pca_projection(coord_result.coords_array_in_requested_order)
    model_display = _rank01(display_field)
    target_display = _rank01(target)

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.2,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(13.6, 7.15), facecolor="white")
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.48, 1.48, 1.0],
            height_ratios=[1.0, 1.0],
            left=0.055,
            right=0.985,
            bottom=0.09,
            top=0.965,
            wspace=0.38,
            hspace=0.43,
        )

        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[0, 2])
        ax_d = fig.add_subplot(grid[1, 0])
        field_grid = grid[1, 1].subgridspec(1, 2, wspace=0.10)
        ax_e1 = fig.add_subplot(field_grid[0, 0])
        ax_e2 = fig.add_subplot(field_grid[0, 1], sharex=ax_e1, sharey=ax_e1)
        ax_f = fig.add_subplot(grid[1, 2])

        _plot_event_map(
            ax_a,
            observed_map,
            record_names,
            title="Held-out interictal events",
            label="a",
            subtitle=f"E620; untouched test set; {len(test_indices):,} events total",
        )
        _plot_event_map(
            ax_b,
            rollout_map,
            model_names,
            title="Self-supervised RNN rollouts",
            label="b",
            subtitle="trained on this patient's earlier interictal events only",
        )
        rank_bar = fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap=CMAP_NAME),
            ax=[ax_a, ax_b],
            orientation="horizontal",
            fraction=0.045,
            pad=0.13,
            aspect=48,
        )
        rank_bar.set_ticks([0, 1])
        rank_bar.set_ticklabels(["early", "late"])
        rank_bar.set_label("Within-event propagation rank", labelpad=-1)
        rank_bar.ax.tick_params(length=0)

        ax_c.scatter(
            observed_precedence[valid_pairs],
            rollout_precedence[valid_pairs],
            s=30,
            color=MODEL_COLOR,
            edgecolors="white",
            linewidths=0.55,
            alpha=0.92,
        )
        ax_c.plot([0, 1], [0, 1], color="0.55", lw=0.9, ls=(0, (3, 2)))
        ax_c.set_xlim(-0.03, 1.03)
        ax_c.set_ylim(-0.03, 1.03)
        ax_c.set_aspect("equal", adjustable="box")
        ax_c.set_xticks([0, 0.5, 1])
        ax_c.set_yticks([0, 0.5, 1])
        ax_c.set_xlabel("Observed P(contact i before j)")
        ax_c.set_ylabel("RNN rollout P(contact i before j)")
        ax_c.set_title("Contact ordering is recovered", loc="left", fontweight="bold", pad=8)
        ax_c.text(
            0.04,
            0.96,
            f"E620, r = {display_precedence_r:.2f}\n3-seed median r = {formal_precedence_r:.2f}",
            transform=ax_c.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )
        _clean_axis(ax_c)
        _panel_label(ax_c, "c", x=-0.24)

        nll_metadata = _paired_nll_panel(ax_d, interictal, primary_subjects)

        _plot_contact_field(
            ax_e1,
            xy,
            target_names,
            model_display,
            title="RNN-derived field\n(interictal only)",
            show_y=True,
        )
        _plot_contact_field(
            ax_e2,
            xy,
            target_names,
            target_display,
            title="Early-ictal field\n(1–150 Hz, 0–10 s)",
            show_y=False,
        )
        ax_e2.text(
            0.98,
            0.03,
            f"E620  |ρ| = {display_abs_rho:.2f}",
            transform=ax_e2.transAxes,
            fontsize=6.5,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
        fig.text(0.425, 0.505, "e", fontsize=13, fontweight="bold", ha="left", va="bottom")
        field_bar = fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap=CMAP_NAME),
            ax=[ax_e1, ax_e2],
            orientation="horizontal",
            fraction=0.055,
            pad=0.17,
            aspect=28,
        )
        field_bar.set_ticks([0, 1])
        field_bar.set_ticklabels(["low", "high"])
        field_bar.set_label("Contact field rank", labelpad=-1)
        field_bar.ax.tick_params(length=0)

        field_metadata = _paired_field_panel(ax_f, primary)

        png = OUT_DIR / f"{FIG_BASENAME}.png"
        pdf = OUT_DIR / f"{FIG_BASENAME}.pdf"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=300, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    formal_subject_row = primary.loc[(primary.subject == subject) & (primary.model == MODEL)].iloc[0]
    metadata = {
        "schema_id": SCHEMA_ID,
        "figure": {"png": png.name, "pdf": pdf.name},
        "scientific_claim": (
            "A within-patient self-supervised recurrent model recovers patient-specific "
            "interictal contact-rank structure, and its frozen model-derived field shows "
            "target-free same-patient correspondence with early-ictal broadband energy."
        ),
        "representative": {
            "subject": subject,
            "selection": (
                "Illustrative joint-positive patient chosen for readable event and contact-space displays; "
                "all inference is carried by the primary cohort panels."
            ),
            "n_contacts": int(len(model_names)),
            "n_test_events_total": int(len(test_indices)),
            "n_rollout_events_total": int(len(rollout_groups)),
            "n_events_displayed_per_heatmap": int(args.display_events),
            "display_filter": "at least four participating contacts on the largest shaft",
            "event_sorting": "independent target-free direction sort within observed and rollout events",
            "display_precedence_r": display_precedence_r,
            "formal_three_seed_median_precedence_r": formal_precedence_r,
            "selected_field": selected_field,
            "display_field_sign": int(display_sign),
            "display_abs_spearman_rho_against_two_seizure_median_target": display_abs_rho,
            "formal_patient_observed_max_abs_rho": float(formal_subject_row.observed_max_abs_rho),
            "coordinate_space": coord_result.coord_space,
            "coordinate_units": coord_result.coord_units,
            "coordinates_used_for_training": False,
            "coordinates_used_for_display_only": True,
            "contact_names": target_names.tolist(),
        },
        "cohort": {
            "primary_n": int(summary["n_primary_subjects"]),
            "development_subject_excluded": config["ictal_transfer"]["development_subject"],
            "interictal_order": {
                **nll_metadata,
                "exact_p_two_sided": summary["interictal_primary_inference"]["rank_shuffle_minus_full_nll_test"]["p_two_sided_exact"],
            },
            "early_ictal_1_150": {
                **field_metadata,
                "exact_p_two_sided": summary["early_ictal"]["1_150"][MODEL]["margin_vs_zero"]["p_two_sided_exact"],
                "within_shaft_exact_p_two_sided": summary["early_ictal"]["1_150"][MODEL]["within_shaft_margin_vs_zero"]["p_two_sided_exact"],
                "full_minus_static_exact_p_two_sided": summary["early_ictal"]["1_150"]["paired_comparisons"]["full_history_gru_minus_static_fit60"]["test"]["p_two_sided_exact"],
            },
        },
        "training_seal": {
            "other_patient_events_used": bool(summary["other_patient_events_used"]),
            "empirical_ab_used": bool(summary["empirical_ab_used"]),
            "ictal_target_used_for_training": bool(summary["ictal_target_used_for_training"]),
        },
        "sources": {
            "config": {"path": _portable_path(args.config, ROOT), "sha256": _sha256(args.config)},
            "summary": {"path": _portable_path(summary_path, ROOT), "sha256": _sha256(summary_path)},
            "dataset_npz": {"path": _portable_path(record.path, artifact_root), "sha256": _sha256(record.path)},
            "rollout_npz": [
                {"path": _portable_path(path, ROOT), "sha256": _sha256(path)} for path in rollout_paths
            ],
            "early_ictal_target_npz": [
                {"path": _portable_path(path, ROOT), "sha256": _sha256(path)} for path in target_paths
            ],
        },
    }
    metadata_path = OUT_DIR / f"{FIG_BASENAME}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    readme = f"""### {FIG_BASENAME}.png

图 a–c 用 E620 的 untouched 间期 test events 与冻结 RNN 的自由生成事件直观对照：热图展示同一组触点上的双向传播排序，散点图定量比较真实与模型的 contact-pair 先后概率。图 d 汇总 15 名 primary 患者，真实 rank 顺序相对 within-event rank shuffle 在 14/15 名患者中改善 held-out prediction。图 e 将冻结模型生成的患者空间场与同患者两次 clinical-onset 后 0–10 s、1–150 Hz broadband energy 的中位场画在同一真实 contact plane；坐标只用于显示，不进入训练。图 f 给出队列级模型场与 early-ictal 场相对 all-contact channel-shuffle null 的比较，同时明确 within-shaft sensitivity 和相对完整静态 scaffold 的增量尚未显著。

**关注点**：前四个 panel 证明 RNN 学到的是患者自己的间期传播结构；后两个 panel 证明冻结后的模型场具有 target-free 跨状态对应，但不把它写成 GRU 独有或已排除全部电极杆几何的机制。
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()

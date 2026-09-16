#!/usr/bin/env python3
"""Subject-first cohort analysis of amplitude-aware Fig3-E template expression.

The observed readout is the frozen shared-plane maxAB absolute template
projection ``|q|`` in per-channel baseline robust-z units.  The primary null
permutes activation values only within electrode shafts, then recomputes field
smoothing, identity/mirror orientation and the A/B maximum for every draw.

EEG onset is the primary time anchor; clinical onset is a sensitivity.  The
hierarchy is window -> seizure median -> subject median -> cohort.  Seizures
and overlapping windows are never treated as independent cohort units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    _load_frozen_shared,
    _shared_geometry_metadata,
)
from scripts.plot_topic5_signed_broadband_movie import (  # noqa: E402
    _band_power_trace_chunked,
    _pre_target,
)
from scripts.plot_topic5_signed_broadband_similarity_timecourse import (  # noqa: E402
    _eligibility_status,
)
from scripts.run_topic5_t0_eligibility import (  # noqa: E402
    GUARD_SEC,
    ICTAL_REFERENCE,
    MIN_BASELINE_SEC,
    _inventory_rows,
)
from src import topic5_ictal_recruitment as recruit  # noqa: E402
from src.ictal_onset_extraction import (  # noqa: E402
    extract_seizure_window,
    resolve_baseline_window,
)
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    aggregate_complete_windows,
    exact_name_align_matrix,
    make_complete_window_grid,
    make_contact_permutations,
    score_observed_bundle,
    score_permutation_matrix,
)


CONTRACT = "fig3e_shared_template_expression_cohort_v1"
DEFAULT_SUBJECTS = (
    "epilepsiae_1084",
    "epilepsiae_1146",
    "epilepsiae_384",
    "epilepsiae_548",
    "epilepsiae_583",
    "epilepsiae_590",
    "epilepsiae_958",
)
OUT = ROOT / "results/paper-ready-figure/fig3/candidates/fig3e_template_expression_cohort"
FIGURES = OUT / "figures"
CHECKPOINTS = OUT / "per_subject/checkpoints"
SOURCE_INDEX = (
    ROOT
    / "results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages"
    / "fig3_peri_onset_field_similarity/fig3_peri_onset_subject_index.json"
)

START_SEC, STOP_SEC = -120.0, 20.0
WINDOW_SEC, STEP_SEC = 10.0, 2.0
SPECTRAL_WIN_SEC, HOP_SEC = 1.0, 0.5
BAND_HZ = (1.0, 150.0)
PHASES = {
    "distal_pre": (-120.0, -90.0),
    "proximal_pre": (-30.0, -10.0),
    "early_ictal": (0.0, 20.0),
}
PHASE_LABELS = {
    "distal_pre": "Distal pre",
    "proximal_pre": "Proximal pre",
    "early_ictal": "Early ictal",
}
ANCHORS = ("eeg", "clinical")
COL_OBS = "#A35E48"
COL_NULL = "#3E6D9C"
COL_GRAY = "#A7A9AC"
BASE_SEED = 20260816


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix,
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix,
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _event_seed(subject: str, seizure_idx: int, seed: int) -> int:
    return int(
        (zlib.crc32(f"{subject}:{seizure_idx}:fig3e-q-within-shaft".encode()) + seed)
        % (2**32 - 1)
    )


def _eeg_rel(dataset: str, row: Mapping[str, object]) -> float:
    if dataset == "yuquan":
        return 0.0
    eeg, clinical = row.get("eeg_onset_epoch"), row.get("clin_onset_epoch")
    if eeg in (None, "") or clinical in (None, ""):
        raise ValueError("missing_eeg_or_clinical_onset")
    return float(eeg) - float(clinical)


def _checkpoint_path(subject: str, seizure_idx: int) -> Path:
    return CHECKPOINTS / subject / f"seizure_{seizure_idx:03d}.npz"


def _checkpoint_matches(path: Path, *, n_perm: int, seed: int,
                        field_sha256: str, cache_sha256: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"].item()))
        return bool(
            meta.get("contract") == CONTRACT
            and int(meta.get("n_perm", -1)) == int(n_perm)
            and int(meta.get("seed", -1)) == int(seed)
            and meta.get("field_sha256") == field_sha256
            and meta.get("eligibility_cache_sha256") == cache_sha256
        )
    except Exception:
        return False


def phase_window_mask(windows: np.ndarray, phase: str) -> np.ndarray:
    """Select complete 10-s windows fully contained in a locked phase."""
    lo, hi = PHASES[phase]
    wins = np.asarray(windows, float)
    return (wins[:, 0] >= lo - 1e-9) & (wins[:, 1] <= hi + 1e-9)


def _score_rows(scorers, values: np.ndarray) -> np.ndarray:
    output = []
    for row in np.asarray(values, float):
        scored = score_observed_bundle(scorers, row)
        candidates = [
            scored.get("shared_a_abs_projection_z"),
            scored.get("shared_b_abs_projection_z"),
        ]
        finite = [float(value) for value in candidates if value is not None and np.isfinite(value)]
        output.append(max(finite) if finite else np.nan)
    return np.asarray(output, float)


def _process_event(subject: str, seizure_idx: int, inventory_row: Mapping[str, object],
                   field_record: Mapping[str, object], shared_scorers, *,
                   n_perm: int, seed: int) -> dict:
    dataset, sid = subject.split("_", 1)
    eeg_rel_inventory = _eeg_rel(dataset, inventory_row)
    anchor_offsets = {"eeg": eeg_rel_inventory, "clinical": 0.0}
    union_start = min(offset + START_SEC for offset in anchor_offsets.values())
    union_stop = max(offset + STOP_SEC for offset in anchor_offsets.values())
    pre_sec = _pre_target(dataset, dict(inventory_row), display_start=union_start)
    post_sec = union_stop + max(30.0, WINDOW_SEC + 0.5)
    sw = extract_seizure_window(
        f"{dataset}/{sid}", int(seizure_idx), pre_sec=pre_sec,
        post_sec=post_sec, reference=ICTAL_REFERENCE[dataset],
    )
    eeg_rel_actual = (
        float(sw.eeg_onset_epoch) - float(sw.clin_onset_epoch)
        if dataset == "epilepsiae" and sw.eeg_onset_epoch is not None
        else 0.0
    )
    if not np.isclose(eeg_rel_actual, eeg_rel_inventory, atol=1e-6):
        raise ValueError(
            f"inventory_loader_eeg_offset_mismatch:{eeg_rel_inventory}:{eeg_rel_actual}"
        )
    if sw.fs / 2.0 <= BAND_HZ[1]:
        raise ValueError(f"nyquist_below_150_hz:{sw.fs / 2.0:g}")

    power, time_from_crop = _band_power_trace_chunked(
        sw.signal, sw.fs, band=BAND_HZ, win_sec=SPECTRAL_WIN_SEC,
        hop_sec=HOP_SEC, chunk_ch=16,
    )
    baseline = resolve_baseline_window(
        power.shape[1], hop_sec=HOP_SEC, pre_sec=sw.pre_sec,
        buffer_sec=GUARD_SEC, eeg_onset_rel_sec=eeg_rel_actual,
        min_baseline_valid_sec=MIN_BASELINE_SEC,
    )
    if not baseline.valid:
        raise ValueError(f"invalid_baseline:{baseline}")
    robust_z = recruit.baseline_robust_z(
        power, (baseline.start_idx, baseline.end_idx), hop_sec=HOP_SEC,
        min_baseline_valid_sec=MIN_BASELINE_SEC,
    )
    rel_clinical = np.asarray(time_from_crop, float) - float(sw.pre_sec)

    target_names = [
        str(value) for value in field_record["interictal_field"]["contact_order"]
    ]
    raw_names = [recruit.bipolar_alias_label(value) for value in sw.ch_names]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("raw_channel_aliases_not_unique")
    raw_index = {name: index for index, name in enumerate(raw_names)}
    matched_names = [name for name in target_names if name in raw_index]
    if len(matched_names) < 6:
        raise ValueError(f"fewer_than_6_exact_name_contacts:{len(matched_names)}")
    selected = robust_z[[raw_index[name] for name in matched_names]]
    aligned = exact_name_align_matrix(field_record, matched_names, selected)
    # A name match is not sufficient for the null universe: a channel whose
    # entire robust-z trace is non-finite must remain missing in place rather
    # than be shuffled into a finite contact slot.
    matched_mask = np.isfinite(aligned["values"]).any(axis=1)
    if int(matched_mask.sum()) < 6:
        raise ValueError(f"fewer_than_6_finite_contact_traces:{int(matched_mask.sum())}")
    permutation_seed = _event_seed(subject, seizure_idx, seed)
    permutations = make_contact_permutations(
        target_names, matched_mask, n_perm, permutation_seed,
        mode="within_shaft",
    )

    relative_grid = make_complete_window_grid(
        START_SEC, STOP_SEC, WINDOW_SEC, STEP_SEC
    )
    observed, null = {}, {}
    for anchor in ANCHORS:
        clinical_grid = relative_grid.copy()
        clinical_grid[:, :3] += anchor_offsets[anchor]
        values, complete = aggregate_complete_windows(
            aligned["values"], rel_clinical, clinical_grid,
            spectral_window_sec=SPECTRAL_WIN_SEC,
        )
        if not bool(np.all(complete)):
            raise ValueError(f"incomplete_{anchor}_window_grid")
        if np.any(np.sum(np.isfinite(values), axis=1) < 6):
            raise ValueError(f"fewer_than_6_finite_contacts_{anchor}")
        observed[anchor] = _score_rows(shared_scorers, values)
        permuted = score_permutation_matrix(
            shared_scorers, values, permutations, chunk_draws=50,
        )
        null[anchor] = np.asarray(
            permuted["shared_maxab_projection_z"], dtype=np.float32
        )
        if not np.isfinite(observed[anchor]).all():
            raise ValueError(f"nonfinite_{anchor}_observed_projection")
        finite_fraction = float(np.isfinite(null[anchor]).mean())
        if finite_fraction < 0.99:
            raise ValueError(
                f"insufficient_finite_{anchor}_null_projection:{finite_fraction:.6f}"
            )

    return {
        "relative_grid": relative_grid,
        "observed": observed,
        "null": null,
        "metadata": {
            "subject": subject,
            "seizure_idx": int(seizure_idx),
            "seizure_id": sw.seizure_id,
            "eeg_onset_minus_clinical_sec": float(eeg_rel_actual),
            "n_target_contacts": len(target_names),
            "n_matched_contacts": int(matched_mask.sum()),
            "baseline_sec_rel_clinical": [
                float(baseline.start_sec), float(baseline.end_sec)
            ],
            "baseline_sec_rel_eeg": [
                float(baseline.start_sec - eeg_rel_actual),
                float(baseline.end_sec - eeg_rel_actual),
            ],
            "sample_rate_hz": float(sw.fs),
            "permutation_seed": int(permutation_seed),
        },
    }


def _write_checkpoint(path: Path, result: Mapping[str, object], *, n_perm: int,
                      seed: int, field_sha256: str, cache_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(result["metadata"])
    meta.update({
        "contract": CONTRACT,
        "n_perm": int(n_perm),
        "seed": int(seed),
        "field_sha256": field_sha256,
        "eligibility_cache_sha256": cache_sha256,
    })
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False,
    ) as handle:
        tmp = Path(handle.name)
    np.savez_compressed(
        tmp,
        relative_grid=np.asarray(result["relative_grid"], float),
        eeg_observed=np.asarray(result["observed"]["eeg"], float),
        eeg_null=np.asarray(result["null"]["eeg"], np.float32),
        clinical_observed=np.asarray(result["observed"]["clinical"], float),
        clinical_null=np.asarray(result["null"]["clinical"], np.float32),
        metadata_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    os.replace(tmp, path)


def _load_checkpoint(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "relative_grid": np.asarray(data["relative_grid"], float),
            "observed": {
                "eeg": np.asarray(data["eeg_observed"], float),
                "clinical": np.asarray(data["clinical_observed"], float),
            },
            "null": {
                "eeg": np.asarray(data["eeg_null"], float),
                "clinical": np.asarray(data["clinical_null"], float),
            },
            "metadata": json.loads(str(data["metadata_json"].item())),
        }


def _subject_fold(subject: str, events: Sequence[Mapping[str, object]],
                  anchor: str) -> tuple[list[dict], list[dict]]:
    grid = np.asarray(events[0]["relative_grid"], float)
    observed_events = np.stack([event["observed"][anchor] for event in events])
    null_events = np.stack([event["null"][anchor] for event in events])
    observed = np.nanmedian(observed_events, axis=0)
    null_draws = np.nanmedian(null_events, axis=0)
    null_median = np.nanmedian(null_draws, axis=0)
    trajectory = []
    for index, window in enumerate(grid):
        trajectory.append({
            "subject": subject,
            "anchor": anchor,
            "window_start_sec": float(window[0]),
            "window_end_sec": float(window[1]),
            "window_center_sec": float(window[2]),
            "n_seizures": len(events),
            "q_observed": float(observed[index]),
            "q_null_median": float(null_median[index]),
            "q_excess": float(observed[index] - null_median[index]),
        })

    phases = []
    for phase in PHASES:
        mask = phase_window_mask(grid, phase)
        event_observed = np.nanmedian(observed_events[:, mask], axis=1)
        event_null = np.nanmedian(null_events[:, :, mask], axis=2)
        subject_observed = float(np.nanmedian(event_observed))
        subject_null_draws = np.nanmedian(event_null, axis=0)
        subject_null_median = float(np.nanmedian(subject_null_draws))
        phases.append({
            "subject": subject,
            "anchor": anchor,
            "phase": phase,
            "phase_start_sec": PHASES[phase][0],
            "phase_end_sec": PHASES[phase][1],
            "n_windows": int(mask.sum()),
            "n_seizures": len(events),
            "q_observed": subject_observed,
            "q_null_median": subject_null_median,
            "q_excess": subject_observed - subject_null_median,
        })
    return trajectory, phases


def _bootstrap_median_ci(values: Sequence[float], *, seed: int,
                         n_boot: int = 20000) -> list[float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
    medians = np.median(draws, axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def _exact_sign_flip_mean_p(values: Sequence[float]) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    observed = float(np.mean(values))
    signs = ((np.arange(2 ** len(values))[:, None] >> np.arange(len(values))) & 1)
    signs = signs * 2 - 1
    null = np.mean(signs * values[None, :], axis=1)
    return float(np.mean(null >= observed - 1e-12))


def cohort_test(values: Sequence[float], *, seed: int) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n_subjects": 0}
    nonzero = values[np.abs(values) > 1e-12]
    wilcoxon_p = (
        float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
        if len(nonzero) else 1.0
    )
    n_positive = int(np.sum(values > 0))
    return {
        "n_subjects": int(len(values)),
        "median": float(np.median(values)),
        "bootstrap_95ci": _bootstrap_median_ci(values, seed=seed),
        "n_positive": n_positive,
        "wilcoxon_greater_p": wilcoxon_p,
        "exact_sign_flip_mean_greater_p": _exact_sign_flip_mean_p(values),
        "exact_sign_test_greater_p": float(
            binomtest(n_positive, len(values), 0.5, alternative="greater").pvalue
        ),
    }


def _cohort_statistics(phase: pd.DataFrame, subject_meta: pd.DataFrame,
                       *, seed: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    summary = {}
    strata = {
        "all_eligible": np.ones(len(subject_meta), bool),
        "strict_geometry": subject_meta["geometry_quality_tier"].eq("strict_2d").to_numpy(),
        "coverage_ge_50pct": subject_meta["coverage_fraction"].ge(0.5).to_numpy(),
        "strict_and_coverage": (
            subject_meta["geometry_quality_tier"].eq("strict_2d")
            & subject_meta["coverage_fraction"].ge(0.5)
        ).to_numpy(),
    }
    for anchor in ANCHORS:
        anchor_phase = phase[phase["anchor"].eq(anchor)]
        wide = anchor_phase.pivot(index="subject", columns="phase", values="q_excess")
        wide_observed = anchor_phase.pivot(
            index="subject", columns="phase", values="q_observed"
        )
        wide_null = anchor_phase.pivot(
            index="subject", columns="phase", values="q_null_median"
        )
        contrasts = {
            "pre_proximal_minus_distal_excess": (
                wide["proximal_pre"] - wide["distal_pre"]
            ),
            "early_minus_proximal_excess": (
                wide["early_ictal"] - wide["proximal_pre"]
            ),
            "distal_excess_gt_zero": wide["distal_pre"],
            "proximal_excess_gt_zero": wide["proximal_pre"],
            "early_excess_gt_zero": wide["early_ictal"],
            "observed_pre_proximal_minus_distal": (
                wide_observed["proximal_pre"] - wide_observed["distal_pre"]
            ),
            "within_shaft_null_pre_proximal_minus_distal": (
                wide_null["proximal_pre"] - wide_null["distal_pre"]
            ),
            "observed_early_minus_proximal": (
                wide_observed["early_ictal"] - wide_observed["proximal_pre"]
            ),
            "within_shaft_null_early_minus_proximal": (
                wide_null["early_ictal"] - wide_null["proximal_pre"]
            ),
        }
        summary[anchor] = {}
        for stratum, mask_all in strata.items():
            allowed = set(subject_meta.loc[mask_all, "subject"])
            summary[anchor][stratum] = {}
            for contrast, series in contrasts.items():
                selected = series[series.index.isin(allowed)]
                test = cohort_test(
                    selected.to_numpy(float),
                    seed=seed + zlib.crc32(f"{anchor}:{stratum}:{contrast}".encode()),
                )
                summary[anchor][stratum][contrast] = test
                rows.append({
                    "anchor": anchor,
                    "stratum": stratum,
                    "contrast": contrast,
                    **test,
                })
    return pd.DataFrame(rows), summary


def _plot_anchor(anchor: str, trajectory: pd.DataFrame, phase: pd.DataFrame,
                 cohort_summary: Mapping[str, object], out_png: Path,
                 out_pdf: Path) -> None:
    data = trajectory[trajectory["anchor"].eq(anchor)]
    grouped = data.groupby("window_center_sec", sort=True)
    x = np.asarray(sorted(data["window_center_sec"].unique()), float)

    def quantile(column: str, q: float) -> np.ndarray:
        return np.asarray([grouped.get_group(value)[column].quantile(q) for value in x])

    obs_med, obs_lo, obs_hi = quantile("q_observed", 0.5), quantile("q_observed", 0.25), quantile("q_observed", 0.75)
    null_med, null_lo, null_hi = quantile("q_null_median", 0.5), quantile("q_null_median", 0.25), quantile("q_null_median", 0.75)
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 2.55), gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax, axp = axes
    ax.fill_between(x, obs_lo, obs_hi, color=COL_OBS, alpha=0.18, linewidth=0)
    ax.plot(x, obs_med, color=COL_OBS, lw=2.0, label="Observed |q|")
    ax.fill_between(x, null_lo, null_hi, color=COL_NULL, alpha=0.12, linewidth=0)
    ax.plot(x, null_med, color=COL_NULL, lw=1.5, ls="--", label="Within-shaft null")
    ax.axvline(0, color="0.25", lw=0.9, ls=(0, (3, 2)))
    ax.set_xlim(START_SEC, STOP_SEC)
    ax.set_xlabel(f"Time from {anchor.upper() if anchor == 'eeg' else 'clinical'} onset (s)")
    ax.set_ylabel("Expression |q| (baseline z)")
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.text(0.01, 0.98, "a", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    phases = list(PHASES)
    positions = np.arange(len(phases), dtype=float)
    pdat = phase[phase["anchor"].eq(anchor)]
    for _subject, group in pdat.groupby("subject"):
        values = [float(group.loc[group["phase"].eq(name), "q_excess"].iloc[0]) for name in phases]
        axp.plot(positions, values, color=COL_GRAY, lw=0.8, alpha=0.8, zorder=1)
        axp.scatter(positions, values, s=15, facecolor="white", edgecolor="0.45", lw=0.7, zorder=2)
    medians = [float(pdat.loc[pdat["phase"].eq(name), "q_excess"].median()) for name in phases]
    axp.plot(positions, medians, color=COL_OBS, lw=2.2, zorder=3)
    axp.scatter(positions, medians, s=26, color=COL_OBS, zorder=4)
    axp.axhline(0, color="0.65", lw=0.8)
    axp.set_xticks(positions, [PHASE_LABELS[name].replace(" ", "\n") for name in phases])
    axp.set_ylabel("Spatial-null excess |q|")
    axp.text(0.01, 0.98, "b", transform=axp.transAxes, ha="left", va="top", fontweight="bold")
    primary = cohort_summary[anchor]["all_eligible"]["pre_proximal_minus_distal_excess"]
    raw_pre = cohort_summary[anchor]["all_eligible"]["observed_pre_proximal_minus_distal"]
    axp.text(
        0.98, 0.03,
        f"raw pre Δ: {raw_pre['n_positive']}/{raw_pre['n_subjects']} ↑\n"
        f"excess pre Δ: {primary['n_positive']}/{primary['n_subjects']} ↑, "
        f"P={primary['exact_sign_flip_mean_greater_p']:.3g}",
        transform=axp.transAxes, ha="right", va="bottom", fontsize=7.0,
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.5, width=0.8, length=3)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.96, wspace=0.38)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


def _plot_subject_grid(trajectory: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    data = trajectory[trajectory["anchor"].eq("eeg")]
    subjects = sorted(data["subject"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.25), sharex=True, sharey=True)
    for axis, subject in zip(axes.flat, subjects):
        group = data[data["subject"].eq(subject)].sort_values("window_center_sec")
        x = group["window_center_sec"].to_numpy(float)
        axis.plot(x, group["q_excess"], color=COL_OBS, lw=1.6)
        axis.axhline(0, color="0.7", lw=0.7)
        axis.axvline(0, color="0.3", lw=0.7, ls=(0, (3, 2)))
        axis.set_title(subject.replace("epilepsiae_", "E"), fontsize=8, loc="left")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(subjects):]:
        axis.axis("off")
    axes[1, 0].set_xlabel("Time from EEG onset (s)", fontsize=8)
    axes[0, 0].set_ylabel("Null-excess |q|", fontsize=8)
    axes[1, 0].set_ylabel("Null-excess |q|", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.95, wspace=0.22, hspace=0.35)
    fig.savefig(out_png, dpi=250, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


def _write_readme() -> None:
    text = """### fig3e_template_expression_cohort_eeg_onset.png

主分析按 EEG onset 对齐。左图比较 amplitude-aware shared-template expression |q| 与保持每根电极杆能量结构的 within-shaft spatial null；右图先在发作内和患者内折叠，再展示远端发作前、近端发作前和早期发作的 null-excess |q|。

**关注点**：患者才是独立统计单位；发作数和重叠时间窗不作为 cohort 样本。

### fig3e_template_expression_cohort_clinical_onset.png

与主分析完全相同，仅把时间零点改为 clinical onset，作为 onset-anchor 敏感性分析。

**关注点**：若 EEG 与 clinical onset 不一致，clinical 对齐会预期性地模糊快速变化。

### fig3e_template_expression_subject_grid_eeg_onset.png

逐患者展示 EEG-onset 对齐的 q_excess 轨迹，用于检查 cohort 中位数是否由单个患者驱动。

**关注点**：重点看近端发作前上升是否跨患者同向，以及 E583 低覆盖患者是否异常。
"""
    (FIGURES / "README.md").write_text(text)


def run(args: argparse.Namespace) -> dict:
    subjects = tuple(args.subjects or DEFAULT_SUBJECTS)
    all_events: dict[str, list[dict]] = {}
    drops, subject_meta_rows, event_rows = [], [], []
    for subject_index, subject in enumerate(subjects, start=1):
        eligibility = _eligibility_status(subject)
        eligible = [int(value) for value in eligibility["eligible_idxs"]]
        if not eligible:
            drops.append({"subject": subject, "reason": eligibility["reason_code"]})
            continue
        field_path = ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject" / f"{subject}.json"
        field_sha256 = _sha256(field_path)
        field_record, shared = _load_frozen_shared(subject)
        scorers = {key: shared[key] for key in ("shared_a", "shared_b")}
        cache_path = ROOT / str(eligibility["cache_path"])
        cache_sha256 = _sha256(cache_path)
        dataset, sid = subject.split("_", 1)
        inventory, onset_field = _inventory_rows(dataset, sid)
        # ``extract_seizure_window`` defines seizure_idx on chronological
        # order.  The shared inventory helper historically documented that
        # order but did not enforce it, so sort locally and fail closed on the
        # exact extractor convention instead of inheriting CSV row order.
        inventory = sorted(inventory, key=lambda row: float(row[onset_field]))
        subject_events = []
        print(f"[{subject_index}/{len(subjects)}] {subject}: {len(eligible)} eligible seizures", flush=True)
        for event_index, seizure_idx in enumerate(eligible, start=1):
            path = _checkpoint_path(subject, seizure_idx)
            event_seed = int(args.seed + subject_index * 100003 + seizure_idx)
            if args.resume and _checkpoint_matches(
                path, n_perm=args.n_perm, seed=event_seed,
                field_sha256=field_sha256, cache_sha256=cache_sha256,
            ):
                result = _load_checkpoint(path)
                print(f"  [{event_index}/{len(eligible)}] seizure {seizure_idx}: checkpoint", flush=True)
            else:
                try:
                    result = _process_event(
                        subject, seizure_idx, inventory[seizure_idx], field_record,
                        scorers, n_perm=args.n_perm, seed=event_seed,
                    )
                    _write_checkpoint(
                        path, result, n_perm=args.n_perm, seed=event_seed,
                        field_sha256=field_sha256, cache_sha256=cache_sha256,
                    )
                    result = _load_checkpoint(path)
                    print(f"  [{event_index}/{len(eligible)}] seizure {seizure_idx}: done", flush=True)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    drops.append({"subject": subject, "seizure_idx": seizure_idx, "reason": reason})
                    print(f"  [{event_index}/{len(eligible)}] seizure {seizure_idx}: drop {reason}", flush=True)
                    continue
            subject_events.append(result)
            event_rows.append({
                "subject": subject,
                "seizure_idx": seizure_idx,
                **result["metadata"],
                "checkpoint": str(path.relative_to(ROOT)),
            })
        if not subject_events:
            continue
        all_events[subject] = subject_events
        geometry = _shared_geometry_metadata(field_record)
        subject_meta_rows.append({
            "subject": subject,
            "n_eligible": len(eligible),
            "n_processed": len(subject_events),
            "n_dropped": len(eligible) - len(subject_events),
            "coverage_fraction": len(subject_events) / len(eligible),
            "field_sha256": field_sha256,
            "eligibility_cache": eligibility["cache_path"],
            "eligibility_cache_sha256": cache_sha256,
            **geometry,
        })

    if not all_events:
        raise RuntimeError("no subject produced a cohort event")
    trajectory_rows, phase_rows = [], []
    for subject, events in all_events.items():
        for anchor in ANCHORS:
            trajectory, phases = _subject_fold(subject, events, anchor)
            trajectory_rows.extend(trajectory)
            phase_rows.extend(phases)
    trajectory = pd.DataFrame(trajectory_rows)
    phase = pd.DataFrame(phase_rows)
    subject_meta = pd.DataFrame(subject_meta_rows)
    cohort_table, cohort_summary = _cohort_statistics(
        phase, subject_meta, seed=int(args.seed)
    )

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(event_rows), OUT / "fig3e_template_expression_events.csv")
    _atomic_csv(subject_meta, OUT / "fig3e_template_expression_subjects.csv")
    _atomic_csv(trajectory, OUT / "fig3e_template_expression_subject_trajectories.csv")
    _atomic_csv(phase, OUT / "fig3e_template_expression_subject_phases.csv")
    _atomic_csv(cohort_table, OUT / "fig3e_template_expression_cohort_statistics.csv")
    for anchor in ANCHORS:
        stem = FIGURES / f"fig3e_template_expression_cohort_{anchor}_onset"
        _plot_anchor(
            anchor, trajectory, phase, cohort_summary,
            stem.with_suffix(".png"), stem.with_suffix(".pdf"),
        )
    grid_stem = FIGURES / "fig3e_template_expression_subject_grid_eeg_onset"
    _plot_subject_grid(
        trajectory, grid_stem.with_suffix(".png"), grid_stem.with_suffix(".pdf")
    )
    _write_readme()

    source_index_payload = json.loads(SOURCE_INDEX.read_text())
    summary = {
        "contract": CONTRACT,
        "tier": "exploratory same-contract shared-plane cohort; subject is independent unit",
        "primary_anchor": "eeg",
        "sensitivity_anchor": "clinical",
        "band_hz": list(BAND_HZ),
        "normalization": "per-channel baseline robust-z using the existing Fig3-E baseline resolver",
        "readout": "shared maxAB absolute amplitude-aware projection |q|; identity/mirror selected by abs correlation; A/B max reselected per draw",
        "primary_null": "within-shaft contact permutation with one fixed mapping per seizure and draw reused across all windows and both anchors",
        "window_contract": {
            "range_sec": [START_SEC, STOP_SEC],
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "phases": {key: list(value) for key, value in PHASES.items()},
        },
        "hierarchy": "window -> seizure median -> subject median -> cohort",
        "n_perm": int(args.n_perm),
        "seed": int(args.seed),
        "requested_subjects": list(subjects),
        "source_same_contract_subject_index": str(SOURCE_INDEX.relative_to(ROOT)),
        "source_same_contract_subject_index_sha256": _sha256(SOURCE_INDEX),
        "source_denominator_flow": source_index_payload.get("denominator_flow"),
        "n_subjects": int(len(subject_meta)),
        "n_processed_seizures": int(subject_meta["n_processed"].sum()),
        "n_dropped_seizures": int(subject_meta["n_dropped"].sum()),
        "drops": drops,
        "cohort_statistics": cohort_summary,
        "claim_boundary": (
            "A positive preictal q_excess contrast supports increasing expression of the "
            "frozen interictal spatial scaffold before EEG onset. It does not prove causal "
            "replay, prediction, recruitment order, or a unique propagation mechanism."
        ),
    }
    summary_path = OUT / "fig3e_template_expression_cohort_summary.json"
    _atomic_json(summary, summary_path)
    artifact_paths = [
        OUT / "fig3e_template_expression_events.csv",
        OUT / "fig3e_template_expression_subjects.csv",
        OUT / "fig3e_template_expression_subject_trajectories.csv",
        OUT / "fig3e_template_expression_subject_phases.csv",
        OUT / "fig3e_template_expression_cohort_statistics.csv",
        summary_path,
        FIGURES / "README.md",
        *sorted(FIGURES.glob("*.png")),
        *sorted(FIGURES.glob("*.pdf")),
    ]
    manifest = {
        "contract": CONTRACT,
        "generated_by": "scripts/paper_figures/run_fig3e_template_expression_cohort.py",
        "n_subjects": int(len(subject_meta)),
        "n_processed_seizures": int(subject_meta["n_processed"].sum()),
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
    }
    _atomic_json(manifest, OUT / "fig3e_template_expression_manifest.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    summary = run(args)
    primary = summary["cohort_statistics"]["eeg"]["all_eligible"]
    print(json.dumps({
        "n_subjects": summary["n_subjects"],
        "n_processed_seizures": summary["n_processed_seizures"],
        "preictal": primary["pre_proximal_minus_distal_excess"],
        "early": primary["early_minus_proximal_excess"],
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()

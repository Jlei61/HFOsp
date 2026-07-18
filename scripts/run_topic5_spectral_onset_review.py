#!/usr/bin/env python3
"""Build the provisional blinded/revealed per-seizure T_spectral review set."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_raw_spectral_context import (  # noqa: E402
    _alias_index,
    _load_lagpat_channels,
)
from scripts.run_topic5_energy_timing_pilot import (  # noqa: E402
    BAND_COLORS,
    BAND_LABELS,
    BAND_SHORT,
)
from scripts.run_topic5_onset_energy_cohort import (  # noqa: E402
    CACHE_ROOT,
    _compute_missing_bands,
    _eligible_map,
    _tier_contract,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE, _inventory_rows  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.topic5_spectral_onset import (  # noqa: E402
    SpectralDiagnostics,
    SpectralOnsetConfig,
    TargetEpisodeAssignment,
    assign_target_episode,
    calibration_samples,
    config_to_dict,
    detect_spectral_episodes,
    episode_to_dict,
    fit_spectral_calibration,
    prepare_spectral_event,
)


DEFAULT_OUT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2/seed_v1p1"
)
PRIMARY_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/source_cache/primary_common_1_80hz"
)
NARROW_SENSITIVITY_AUDIT = (
    ROOT
    / "results/topic5_ictal_recruitment/t0_eligibility_audit_narrow_cache_sensitivity.csv"
)
DEFAULT_SUBJECTS = (
    "epilepsiae_1146",
    "epilepsiae_442",
    "epilepsiae_583",
    "epilepsiae_916",
)
SEED = 20260714
ANALYSIS_VERSION = "topic5_tspectral_v1p1"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _status_summary(rows: list[dict]) -> list[dict]:
    statuses = (
        "confirmed_precise_T",
        "broadband_but_imprecise_T",
        "separate_prior_episode",
        "no_detectable_broadband_transition",
    )
    subjects = sorted({str(row["subject"]) for row in rows})
    out: list[dict] = []
    for subject in [*subjects, "ALL_POOLED_DESCRIPTIVE"]:
        use = rows if subject == "ALL_POOLED_DESCRIPTIVE" else [r for r in rows if r["subject"] == subject]
        confirmed = [
            float(r["auto_t_spectral_rel_eeg_sec"])
            for r in use
            if r["auto_status"] == "confirmed_precise_T"
            and str(r["auto_t_spectral_rel_eeg_sec"]) != ""
        ]
        counts = {status: sum(r["auto_status"] == status for r in use) for status in statuses}
        out.append(
            {
                "subject": subject,
                "n_seizures": len(use),
                **{f"n_{status}": counts[status] for status in statuses},
                **{
                    f"fraction_{status}": counts[status] / len(use) if use else float("nan")
                    for status in statuses
                },
                "confirmed_t_rel_eeg_q25_sec": (
                    float(np.quantile(confirmed, 0.25)) if confirmed else float("nan")
                ),
                "confirmed_t_rel_eeg_median_sec": (
                    float(np.median(confirmed)) if confirmed else float("nan")
                ),
                "confirmed_t_rel_eeg_q75_sec": (
                    float(np.quantile(confirmed, 0.75)) if confirmed else float("nan")
                ),
            }
        )
    return out


def _preserve_manual_review(existing_path: Path, rows: list[dict]) -> None:
    if not existing_path.exists():
        return
    old = {row["review_id"]: row for row in csv.DictReader(existing_path.open(encoding="utf-8"))}
    preserve = (
        "manual_blind_class",
        "manual_blind_t_review_sec",
        "manual_episode_match",
        "manual_artifact",
        "manual_final_class",
        "manual_final_t_rel_eeg_sec",
        "reviewer",
        "review_notes",
    )
    for row in rows:
        previous = old.get(str(row["review_id"]))
        if previous is None:
            continue
        for field in preserve:
            row[field] = previous.get(field, row[field])


def _cache_tier(ds_sid: str) -> str:
    if not NARROW_SENSITIVITY_AUDIT.exists():
        return "primary_existing"
    for row in csv.DictReader(NARROW_SENSITIVITY_AUDIT.open(encoding="utf-8")):
        if row.get("subject_id") != ds_sid:
            continue
        if str(row.get("narrow_cache_eligible", "")).strip().lower() in {
            "true",
            "1",
            "yes",
        }:
            return "narrow_sensitivity_min6_overlap"
    return "primary_existing"


def _rank_channels_from_json(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs") or []
    if pairs:
        pair = pairs[0]
        names = [str(value) for value in pair.get("channel_names", [])]
        valid = pair.get("joint_valid", [True] * len(names))
        return [name for name, keep in zip(names, valid, strict=False) if bool(keep)]
    return [str(value) for value in data.get("channel_names", [])]


def _load_subject_events(
    ds_sid: str,
    config: SpectralOnsetConfig,
    *,
    cache_root: Path = CACHE_ROOT,
    cache_tier_override: str | None = None,
    inventory_csv: Path | None = None,
) -> tuple[list[dict], list[str]]:
    specs, _, _, _ = _tier_contract("common_1_80hz")
    band_order = [name for name, _, _ in specs]
    dataset, sid = ds_sid.split("_", 1)
    if inventory_csv is not None:
        with Path(inventory_csv).open(encoding="utf-8") as handle:
            inv_rows = [
                row for row in csv.DictReader(handle)
                if row.get("subject") == sid and row.get("onset_epoch")
            ]
        inv_rows.sort(key=lambda row: float(row["onset_epoch"]))
    else:
        inv_rows, _ = _inventory_rows(dataset, sid)

    cache_obj = np.load(cache_root / f"{ds_sid}.npz", allow_pickle=False)
    cache_meta = json.loads((cache_root / f"{ds_sid}.json").read_text(encoding="utf-8"))
    cache_channels = [str(x) for x in cache_obj["channels"]]
    cache_lookup = {name: i for i, name in enumerate(cache_channels)}
    cached_idx = {int(x) for x in cache_meta["seizure_idxs"]}
    if dataset == "epilepsiae":
        all_subjects = sorted(path.stem for path in CACHE_ROOT.glob("epilepsiae_*.json"))
        eligible_by_subject, _ = _eligible_map(all_subjects, max_band_hz=80.0)
        eligible = eligible_by_subject[ds_sid]
    elif dataset == "yuquan":
        # Yuquan caches are already seizure-admitted by their frozen primary or
        # explicit narrow-sensitivity audit. Their cache zero is the EEG onset.
        eligible = sorted(cached_idx)
    else:
        cache_obj.close()
        raise ValueError(f"unsupported dataset: {dataset}")
    lagpat_channels, lagpat_source = _load_lagpat_channels(ds_sid)
    timing_channels = [name for name in lagpat_channels if name in cache_lookup]
    if len(timing_channels) < config.min_contacts and dataset == "yuquan":
        # A narrow rank record can be real yet have zero montage overlap (observed
        # zhaojinrui: F5-F8 versus the available bipolar-left montage).  Fall back
        # to the masked broad rank record, but only when it improves the overlap;
        # provenance remains explicit and narrow/broad are never pooled silently.
        broad_path = (
            ROOT
            / "results/interictal_propagation_masked_broad/rank_displacement/per_subject"
            / f"{ds_sid}.json"
        )
        broad_channels = _rank_channels_from_json(broad_path)
        broad_timing = [name for name in broad_channels if name in cache_lookup]
        if len(broad_timing) > len(timing_channels):
            lagpat_channels = broad_channels
            timing_channels = broad_timing
            lagpat_source = str(broad_path.relative_to(ROOT))
    if len(timing_channels) < config.min_contacts:
        cache_obj.close()
        raise RuntimeError(f"{ds_sid}: only {len(timing_channels)} timing contacts")
    timing_idx = np.asarray([cache_lookup[name] for name in timing_channels], dtype=int)
    missing = sorted(set(eligible) - cached_idx)
    fallback: dict[str, np.ndarray] = {}
    if missing:
        if dataset != "epilepsiae":
            cache_obj.close()
            raise RuntimeError(f"{ds_sid}: Yuquan cache misses admitted indices {missing}")
        fallback, fallback_meta = _compute_missing_bands(
            ds_sid,
            missing,
            inv_rows,
            specs,
            timing_channels,
            PRIMARY_ROOT,
            "common_1_80hz",
        )
        if fallback_meta.get("drops"):
            cache_obj.close()
            raise RuntimeError(f"{ds_sid}: fallback drops={fallback_meta['drops']}")

    events: list[dict] = []
    for idx in eligible:
        inv = inv_rows[idx]
        clinical_available = dataset == "epilepsiae"
        eeg_rel = (
            float(inv["eeg_onset_epoch"]) - float(inv["clin_onset_epoch"])
            if clinical_available
            else 0.0
        )
        source_obj = cache_obj if idx in cached_idx else fallback
        source = "committed_v2_band_cache" if idx in cached_idx else "raw_short_window_fallback"
        arrays = []
        rel_ref = None
        for band in band_order:
            z = np.asarray(source_obj[f"{band}__zt__{idx}"], dtype=float)
            if idx in cached_idx:
                z = z[timing_idx]
            rel_clin = np.asarray(source_obj[f"{band}__relt__{idx}"], dtype=float)
            rel_eeg = rel_clin - eeg_rel
            if rel_ref is None:
                rel_ref = rel_eeg
            elif rel_eeg.shape != rel_ref.shape or not np.allclose(rel_eeg, rel_ref, atol=1e-6):
                raise ValueError(f"{ds_sid} seizure {idx}: band time grids differ")
            arrays.append(z)
        assert rel_ref is not None
        prepared = prepare_spectral_event(np.stack(arrays), rel_ref, config=config)
        events.append(
            {
                "subject": ds_sid,
                "dataset": dataset,
                "sid": sid,
                "seizure_idx": int(idx),
                "seizure_id": str(
                    inv.get("seizure_id")
                    or inv.get("clean_seizure_id")
                    or f"{sid}_sz_{idx + 1:03d}"
                ),
                "eeg_rel_clinical_sec": eeg_rel,
                "clinical_rel_eeg_sec": -eeg_rel,
                "clinical_onset_available": clinical_available,
                "annotation_mode": (
                    "eeg_and_clinical" if clinical_available else "eeg_only"
                ),
                "cache_zero_reference": (
                    "clinical_onset" if clinical_available else "eeg_onset"
                ),
                "cache_tier": cache_tier_override or _cache_tier(ds_sid),
                "source": source,
                "lagpat_source": lagpat_source,
                "timing_channels": timing_channels,
                "prepared": prepared,
            }
        )
    cache_obj.close()
    return events, band_order


def _stacked_raw(
    ax: plt.Axes,
    raw_sw,
    channel_idx: list[int],
    *,
    eeg_rel_clinical: float,
    xlim_eeg: tuple[float, float],
    blind_origin: float | None,
) -> None:
    raw_rel_eeg = np.asarray(raw_sw.t_axis, dtype=float) - float(eeg_rel_clinical)
    use = np.flatnonzero((raw_rel_eeg >= xlim_eeg[0]) & (raw_rel_eeg <= xlim_eeg[1]))
    if use.size == 0:
        raise ValueError("raw window does not overlap spectral review window")
    decim = max(1, int(round(float(raw_sw.fs) / 180.0)))
    use = use[::decim]
    x = raw_rel_eeg[use]
    if blind_origin is not None:
        x = x - blind_origin
    signal = raw_sw.signal[np.asarray(channel_idx), :][:, use]
    centered = signal - np.nanmedian(signal, axis=1, keepdims=True)
    scale = float(np.nanpercentile(np.abs(centered), 95.0) * 3.0)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    offsets = np.arange(len(channel_idx), dtype=float)[::-1] * scale
    for row, offset in enumerate(offsets):
        ax.plot(x, centered[row] + offset, color="0.20", lw=0.35, alpha=0.85)
    ax.set_yticks(offsets)
    ax.set_yticklabels([str(raw_sw.ch_names[i]) for i in channel_idx], fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _markers(
    axes: list[plt.Axes],
    diagnostics: SpectralDiagnostics,
    assignment: TargetEpisodeAssignment,
    *,
    clinical_rel_eeg: float,
    clinical_onset_available: bool = True,
) -> None:
    target = (
        diagnostics.episodes[assignment.target_index]
        if assignment.target_index is not None
        else None
    )
    for ax in axes:
        ax.axvline(0.0, color="#7A4F9A", ls=":", lw=1.0)
        if clinical_onset_available:
            ax.axvline(float(clinical_rel_eeg), color="0.15", ls="--", lw=0.9)
        for epi_idx, episode in enumerate(diagnostics.episodes):
            color = "#2CA02C" if epi_idx == assignment.target_index else "#9E9E9E"
            ax.axvspan(episode.start_sec, episode.end_sec, color=color, alpha=0.055, lw=0)
            ax.axvline(
                episode.change_sec,
                color=color,
                lw=1.1 if epi_idx == assignment.target_index else 0.7,
                ls="-" if epi_idx == assignment.target_index else ":",
            )
        if target is not None and np.isfinite(target.bootstrap_q05_sec):
            ax.axvspan(
                target.bootstrap_q05_sec,
                target.bootstrap_q95_sec,
                color="#2CA02C",
                alpha=0.10,
                lw=0,
            )


def _plot_review(
    event: dict,
    diagnostics: SpectralDiagnostics,
    assignment: TargetEpisodeAssignment,
    raw_sw,
    raw_idx: list[int],
    band_order: list[str],
    *,
    review_id: str,
    blind: bool,
    out_path: Path,
    refined_time_sec: float | None = None,
    refined_q05_sec: float | None = None,
    refined_q95_sec: float | None = None,
    refined_status: str | None = None,
    refined_label: str = "patient-specific T_best",
) -> None:
    t = diagnostics.rel_t
    clinical_rel_eeg = float(event["clinical_rel_eeg_sec"])
    x0 = max(float(t[0]), -120.0)
    x1 = min(float(t[-1]), max(0.0, clinical_rel_eeg) + 20.0)
    use = (t >= x0) & (t <= x1)
    blind_origin = x0 if blind else None
    x = t[use] - x0 if blind else t[use]
    xlim = (float(x[0]), float(x[-1]))

    fig = plt.figure(figsize=(13.2, 11.8))
    gs = fig.add_gridspec(
        5,
        3,
        height_ratios=[2.25, 0.85, 1.05, 0.85, 0.95],
        hspace=0.52,
        wspace=0.28,
    )
    ax_raw = fig.add_subplot(gs[0, :])
    _stacked_raw(
        ax_raw,
        raw_sw,
        raw_idx,
        eeg_rel_clinical=float(event["eeg_rel_clinical_sec"]),
        xlim_eeg=(x0, x1),
        blind_origin=blind_origin,
    )
    ax_raw.set_xlim(xlim)
    ax_raw.set_title(
        f"a  Raw intracranial traces on fixed timing contacts (n={len(raw_idx)})",
        loc="left",
        fontsize=10,
    )

    ax_band = fig.add_subplot(gs[1, :])
    heat = np.vstack([diagnostics.band_trace[:, use], diagnostics.consensus_trace[use]])
    vmax = max(2.0, min(8.0, float(np.nanpercentile(np.abs(heat), 98.0))))
    image = ax_band.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        extent=[xlim[0], xlim[1], heat.shape[0] - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_band.set_yticks(np.arange(6))
    ax_band.set_yticklabels([BAND_SHORT[name] for name in band_order] + ["ALL"], fontsize=7)
    ax_band.set_title("b  Five-band spatial Q75 energy and multiband consensus", loc="left", fontsize=10)
    fig.colorbar(image, ax=ax_band, pad=0.01, fraction=0.018, label="baseline-centred z")

    ax_trace = fig.add_subplot(gs[2, :])
    for band_idx, band in enumerate(band_order):
        ax_trace.plot(
            x,
            diagnostics.band_trace[band_idx, use],
            color=BAND_COLORS[band],
            lw=0.9,
            alpha=0.85,
            label=BAND_LABELS[band],
        )
    ax_trace.plot(x, diagnostics.consensus_trace[use], color="0.12", lw=1.5, label="consensus")
    ax_trace.axhline(0.0, color="0.6", lw=0.6)
    ax_trace.set_ylabel("delta z")
    ax_trace.set_title("c  Band trajectories", loc="left", fontsize=10)
    ax_trace.legend(frameon=False, ncol=6, fontsize=7, loc="upper left")

    ax_contact = fig.add_subplot(gs[3, :])
    contact_image = ax_contact.imshow(
        diagnostics.contact_active_band_count[:, use],
        aspect="auto",
        interpolation="nearest",
        extent=[xlim[0], xlim[1], len(event["timing_channels"]) - 0.5, -0.5],
        cmap="viridis",
        vmin=0,
        vmax=5,
    )
    ax_contact.set_yticks(np.arange(len(event["timing_channels"])))
    ax_contact.set_yticklabels(event["timing_channels"], fontsize=6)
    ax_contact.set_title("d  Number of background-extreme bands per contact", loc="left", fontsize=10)
    fig.colorbar(contact_image, ax=ax_contact, pad=0.01, fraction=0.018, ticks=range(6))

    ax_step = fig.add_subplot(gs[4, 0])
    ax_step.plot(x, diagnostics.consensus_step_z[use], color="#4C78A8", lw=1.0)
    ax_step.axhline(3.0, color="0.35", ls="--", lw=0.8)
    ax_step.set_ylabel("step robust-z")
    ax_step.set_title("e  Consensus upward step", loc="left", fontsize=9)

    ax_nband = fig.add_subplot(gs[4, 1])
    ax_nband.plot(x, diagnostics.n_level_bands[use], color="#ECA82C", lw=1.0)
    ax_nband.axhline(3.0, color="0.35", ls="--", lw=0.8)
    ax_nband.set_ylim(-0.2, 5.2)
    ax_nband.set_yticks(range(6))
    ax_nband.set_title("f  Active bands", loc="left", fontsize=9)

    ax_ncontact = fig.add_subplot(gs[4, 2])
    ax_ncontact.plot(x, diagnostics.n_level_contacts[use], color="#59A14F", lw=1.0)
    ax_ncontact.axhline(diagnostics.min_spatial_contacts, color="0.35", ls="--", lw=0.8)
    ax_ncontact.set_ylim(-0.2, len(event["timing_channels"]) + 0.5)
    ax_ncontact.set_title("g  Broadband-active contacts", loc="left", fontsize=9)

    axes = [ax_raw, ax_band, ax_trace, ax_contact, ax_step, ax_nband, ax_ncontact]
    if not blind:
        _markers(
            axes,
            diagnostics,
            assignment,
            clinical_rel_eeg=clinical_rel_eeg,
            clinical_onset_available=bool(event.get("clinical_onset_available", True)),
        )
        if refined_time_sec is not None and np.isfinite(refined_time_sec):
            for ax in axes:
                if (
                    refined_q05_sec is not None
                    and refined_q95_sec is not None
                    and np.isfinite(refined_q05_sec)
                    and np.isfinite(refined_q95_sec)
                ):
                    ax.axvspan(
                        refined_q05_sec,
                        refined_q95_sec,
                        color="#1F77B4",
                        alpha=0.09,
                        lw=0,
                    )
                ax.axvline(refined_time_sec, color="#1F77B4", lw=1.35, zorder=8)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    for ax in (ax_step, ax_nband, ax_ncontact):
        ax.set_xlabel("review time (s)" if blind else "time from EEG onset (s)")

    if blind:
        title = f"{review_id} — BLIND spectral-transition review"
        footer = (
            f"blind time origin = left edge; automatic thresholds shown but no onset annotations, "
            f"episode markers, subject, or seizure ID"
        )
    else:
        target = (
            diagnostics.episodes[assignment.target_index]
            if assignment.target_index is not None
            else None
        )
        t_text = "NA" if target is None else f"{target.change_sec:+.1f}s"
        title = (
            f"{event['subject']} seizure {event['seizure_idx']:02d} ({event['seizure_id']}) — "
            f"{assignment.status}; auto T_spectral={t_text}"
        )
        if refined_status is not None:
            refined_text = (
                "NA"
                if refined_time_sec is None or not np.isfinite(refined_time_sec)
                else f"{refined_time_sec:+.1f}s"
            )
            title += f"; {refined_status}; {refined_label}={refined_text}"
        footer = "purple dotted = EEG onset; "
        if event.get("clinical_onset_available", True):
            footer += "black dashed = clinical onset; "
        else:
            footer += "Yuquan EEG-only annotation (no separate clinical marker); "
        footer += "green = assigned episode/T; gray = other detected episode"
        if refined_status is not None:
            footer += f"; blue = {refined_label} (90% resampling interval)"
    fig.suptitle(title, fontsize=13, y=0.992)
    fig.text(0.5, 0.012, footer, ha="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.962, bottom=0.055)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _review_row(
    event: dict,
    diagnostics: SpectralDiagnostics,
    assignment: TargetEpisodeAssignment,
    *,
    review_id: str,
    blind_path: Path,
    revealed_path: Path,
) -> dict:
    target = (
        diagnostics.episodes[assignment.target_index]
        if assignment.target_index is not None
        else None
    )
    target_dict = episode_to_dict(target)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "review_id": review_id,
        "subject": event["subject"],
        "seizure_idx": event["seizure_idx"],
        "seizure_id": event["seizure_id"],
        "source": event["source"],
        "annotation_mode": event.get("annotation_mode", "eeg_and_clinical"),
        "clinical_onset_available": event.get("clinical_onset_available", True),
        "cache_zero_reference": event.get("cache_zero_reference", "clinical_onset"),
        "cache_tier": event.get("cache_tier", "primary_existing"),
        "n_timing_contacts": len(event["timing_channels"]),
        "eeg_onset_rel_cache_zero_sec": event["eeg_rel_clinical_sec"],
        "eeg_onset_rel_clinical_sec": (
            event["eeg_rel_clinical_sec"]
            if event.get("clinical_onset_available", True)
            else ""
        ),
        "blind_time_origin_rel_eeg_sec": max(float(diagnostics.rel_t[0]), -120.0),
        "auto_status": assignment.status,
        "auto_n_episodes": len(diagnostics.episodes),
        "auto_n_connected_episodes": assignment.n_connected_episodes,
        "auto_n_prior_episodes": assignment.n_prior_episodes,
        "auto_t_spectral_rel_eeg_sec": target_dict.get("change_sec", ""),
        "auto_t_spectral_rel_cache_zero_sec": (
            target_dict["change_sec"] + event["eeg_rel_clinical_sec"]
            if target is not None
            else ""
        ),
        "auto_t_spectral_rel_clinical_sec": (
            target_dict["change_sec"] + event["eeg_rel_clinical_sec"]
            if target is not None and event.get("clinical_onset_available", True)
            else ""
        ),
        "auto_episode_start_rel_eeg_sec": target_dict.get("start_sec", ""),
        "auto_episode_end_rel_eeg_sec": target_dict.get("end_sec", ""),
        "auto_step_z": target_dict.get("change_step_z", ""),
        "auto_n_step_bands": target_dict.get("n_step_bands", ""),
        "auto_n_step_contacts": target_dict.get("n_step_contacts", ""),
        "auto_low_step_supported": target_dict.get("low_step_supported", ""),
        "auto_high_step_supported": target_dict.get("high_step_supported", ""),
        "auto_complete_change_gate": target_dict.get("automatic_change_gate", ""),
        "auto_bootstrap_q05_rel_eeg_sec": target_dict.get("bootstrap_q05_sec", ""),
        "auto_bootstrap_q95_rel_eeg_sec": target_dict.get("bootstrap_q95_sec", ""),
        "auto_bootstrap_ci_width_sec": target_dict.get("bootstrap_ci_width_sec", ""),
        "auto_stable_candidate_time": target_dict.get("stable_candidate_time", ""),
        "blind_figure": _portable_path(blind_path),
        "revealed_figure": _portable_path(revealed_path),
        "manual_blind_class": "",
        "manual_blind_t_review_sec": "",
        "manual_episode_match": "",
        "manual_artifact": "",
        "manual_final_class": "",
        "manual_final_t_rel_eeg_sec": "",
        "reviewer": "",
        "review_notes": "",
    }


def _write_readmes(out_root: Path, counts: dict[str, int]) -> None:
    fig_root = out_root / "figures"
    (fig_root / "README.md").write_text(
        "# T_spectral per-seizure review figures\n\n"
        "### blind/\n\n逐 seizure 匿名审查图，不显示患者、seizure ID、EEG/clinical onset 或自动 episode 标记。\n\n"
        "**关注点**：先独立判断是否存在持续、空间分布的宽带转变及其时刻。\n\n"
        "### revealed/\n\n与 blind 图相同的数据，但显示可用的 EEG/clinical onset、所有自动 episode 及目标 episode 分配；Yuquan 只有 EEG onset。\n\n"
        "**关注点**：第二遍判断 episode 是否属于目标 seizure，并排除原始信号 artifact。\n",
        encoding="utf-8",
    )
    (fig_root / "blind" / "README.md").write_text(
        "# Blind T_spectral review\n\n"
        "### TS*.png\n\n匿名、随机编号的逐 seizure 图。横轴从当前审查窗左边界记为 0，不显示任何临床时间标记。\n\n"
        "**关注点**：只依据原始波形、五频带、空间支持和 gate 曲线判断宽带 episode。\n",
        encoding="utf-8",
    )
    revealed_root = fig_root / "revealed"
    (revealed_root / "README.md").write_text(
        "# Revealed T_spectral review\n\n"
        "### <dataset>_<subject>/\n\n按患者整理的第二遍审查图，显示可用的标记和自动 episode 分配。Yuquan 使用 EEG-only annotation，不伪造 clinical marker。\n\n"
        "**关注点**：先完成 blind 判断，再核对 episode assignment；不要反向修改 blind 结论。\n",
        encoding="utf-8",
    )
    for subject, count in counts.items():
        (revealed_root / subject / "README.md").write_text(
            f"# {subject} revealed T_spectral review\n\n"
            f"### {subject}_seizure_*_tspectral_revealed.png\n\n"
            f"该患者共 {count} 次 frequency-complete eligible seizures 的 revealed 图。\n\n"
            "**关注点**：比较 subject 内 episode 形态、自动时刻稳定性以及 EEG/clinical 标记偏移。\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Path:
    config = SpectralOnsetConfig(n_boot=int(args.n_boot))
    all_events: list[dict] = []
    band_order: list[str] | None = None
    for subject in args.subjects:
        events, bands = _load_subject_events(subject, config)
        if args.max_seizures is not None:
            events = events[: int(args.max_seizures)]
        all_events.extend(events)
        band_order = bands if band_order is None else band_order
    assert band_order is not None

    order = np.random.default_rng(SEED).permutation(len(all_events))
    review_ids = {int(event_index): f"TS{rank + 1:04d}" for rank, event_index in enumerate(order)}
    rows: list[dict] = []
    errors: list[dict] = []
    counts: dict[str, int] = {}
    for subject in args.subjects:
        subject_event_idx = [i for i, event in enumerate(all_events) if event["subject"] == subject]
        samples = [calibration_samples(all_events[i]["prepared"]) for i in subject_event_idx]
        counts[subject] = len(subject_event_idx)
        for local_index, event_index in enumerate(subject_event_idx):
            event = all_events[event_index]
            review_id = review_ids[event_index]
            other = [sample for j, sample in enumerate(samples) if j != local_index]
            calibration_scope = "subject_LOSO"
            if not other:
                other = [samples[local_index]]
                calibration_scope = "within_seizure_fallback"
            try:
                calibration = fit_spectral_calibration(other, config=config)
                clinical = float(event["clinical_rel_eeg_sec"])
                search = (
                    config.baseline[1] + config.max_gap_sec,
                    min(
                        float(event["prepared"].rel_t[-1]),
                        max(0.0, clinical) + config.assignment_post_sec,
                    ),
                )
                diagnostics = detect_spectral_episodes(
                    event["prepared"],
                    calibration,
                    search=search,
                    config=config,
                    seed=SEED + int(event["seizure_idx"]),
                )
                assignment = assign_target_episode(
                    diagnostics.episodes,
                    eeg_onset_sec=0.0,
                    clinical_onset_sec=clinical,
                    config=config,
                )
                blind_path = args.out_root / "figures/blind" / f"{review_id}.png"
                revealed_path = (
                    args.out_root
                    / "figures/revealed"
                    / subject
                    / f"{subject}_seizure_{event['seizure_idx']:02d}_tspectral_revealed.png"
                )
                if args.force or not (blind_path.exists() and revealed_path.exists()):
                    x0_eeg = max(float(diagnostics.rel_t[0]), -120.0)
                    x1_eeg = min(
                        float(diagnostics.rel_t[-1]), max(0.0, clinical) + config.assignment_post_sec
                    )
                    x0_clin = x0_eeg + float(event["eeg_rel_clinical_sec"])
                    x1_clin = x1_eeg + float(event["eeg_rel_clinical_sec"])
                    raw_sw = extract_seizure_window(
                        f"{event['dataset']}/{event['sid']}",
                        int(event["seizure_idx"]),
                        # Load exactly the displayed interval.  Extra padding
                        # can turn an otherwise complete edge-of-block event
                        # into an artificial raw-window drop (E253 seizure 1).
                        pre_sec=max(10.0, -x0_clin),
                        post_sec=max(10.0, x1_clin),
                        results_root=ROOT / "results",
                        reference=ICTAL_REFERENCE[event["dataset"]],
                    )
                    raw_lookup = _alias_index(raw_sw.ch_names)
                    absent = [name for name in event["timing_channels"] if name not in raw_lookup]
                    if absent:
                        raise ValueError(f"raw timing contacts missing: {absent}")
                    raw_idx = [int(raw_lookup[name]) for name in event["timing_channels"]]
                    _plot_review(
                        event,
                        diagnostics,
                        assignment,
                        raw_sw,
                        raw_idx,
                        band_order,
                        review_id=review_id,
                        blind=True,
                        out_path=blind_path,
                    )
                    _plot_review(
                        event,
                        diagnostics,
                        assignment,
                        raw_sw,
                        raw_idx,
                        band_order,
                        review_id=review_id,
                        blind=False,
                        out_path=revealed_path,
                    )
                    del raw_sw
                row = _review_row(
                    event,
                    diagnostics,
                    assignment,
                    review_id=review_id,
                    blind_path=blind_path,
                    revealed_path=revealed_path,
                )
                row["calibration_scope"] = calibration_scope
                rows.append(row)
                print(
                    f"[tspectral] {subject} seizure {event['seizure_idx']} {assignment.status}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - preserve event-level failure provenance
                errors.append(
                    {
                        "subject": subject,
                        "seizure_idx": event["seizure_idx"],
                        "review_id": review_id,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
                print(f"[ERROR] {errors[-1]}", flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "review_manifest.csv"
    _preserve_manual_review(manifest_path, rows)
    _write_csv(manifest_path, rows)
    _write_csv(args.out_root / "auto_status_summary.csv", _status_summary(rows))
    (args.out_root / "contract.json").write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "status": "algorithmic_v1p1_manual_adjudication_pending",
                "config": config_to_dict(config),
                "subjects": list(args.subjects),
                "n_events": len(all_events),
                "n_rendered": len(rows),
                "n_errors": len(errors),
                "review_id_seed": SEED,
                "calibration": "subject-level leave-one-seizure-out",
                "band_order": band_order,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_root / "processing_errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_readmes(args.out_root, counts)
    if errors:
        raise RuntimeError(f"{len(errors)} event(s) failed; see {args.out_root / 'processing_errors.json'}")
    return args.out_root / "review_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument(
        "--all-epilepsiae",
        action="store_true",
        help="use every Epilepsiae subject with a v2 band-cache artifact",
    )
    parser.add_argument(
        "--all-yuquan-cache",
        action="store_true",
        help="use every Yuquan subject with a v2 band-cache artifact",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--max-seizures", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.all_epilepsiae:
        args.subjects = sorted(path.stem for path in CACHE_ROOT.glob("epilepsiae_*.json"))
    if args.all_yuquan_cache:
        args.subjects = sorted(path.stem for path in CACHE_ROOT.glob("yuquan_*.json"))
    print(run(args))


if __name__ == "__main__":
    main()

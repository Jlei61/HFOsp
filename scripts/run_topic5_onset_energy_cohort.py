#!/usr/bin/env python3
"""Cohort audit of baseline-extreme multiband energy near EEG onset.

The primary question is deliberately local and annotation-explicit: within a
pre-specified EEG-onset +/-5 s window, does the fixed interictal-analysis
contact set show a sustained (>=2 s) multiband energy excursion above the
seizure's own distal EEG-relative baseline Q99?  The same 10 s window is then
centered on clinical onset and on pseudo-onset offsets for controls.

This is a cohort extension of the E1146 timing pilot.  It uses the seven
primary v2 band-cache traces, the masked rank-displacement lagPat-valid contact
set, and T0 eligibility.  It does not read A/B template ranks or correlations.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_ictal_field_long_cache import GUARD_SEC, MIN_BASELINE_SEC  # noqa: E402
from scripts.paper_figures.plot_fig3_raw_spectral_context import (  # noqa: E402
    _alias_index,
    _load_lagpat_channels,
)
from scripts.run_topic5_energy_timing_pilot import (  # noqa: E402
    BAND_COLORS,
    BAND_LABELS,
    BAND_SHORT,
    _primary_band_specs,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE, _inventory_rows  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window, resolve_baseline_window  # noqa: E402
from src.topic5_energy_timing import (  # noqa: E402
    band_energy_timing,
    centered_window_hit_profile,
    detect_centered_window_enhancement,
    detect_multiband_recruitment_onset,
    max_upward_transition,
)
from src.topic5_ictal_recruitment import _spectrogram_on_hop, bipolar_alias_label  # noqa: E402
from src.topic5_v2_band_scan import (  # noqa: E402
    band_bin_selection,
    line_noise_bin_mask,
    load_phase1_config,
    robust_z_with_flags,
)


CACHE_ROOT = ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
AUDIT_CSV = ROOT / "results/topic5_ictal_recruitment/t0_eligibility_audit.csv"
DEFAULT_OUT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/source_cache"
)

# All timing coordinates below are EEG-onset-relative.
DISTAL_BASELINE = (-120.0, -90.0)
EEG_ONSET_WINDOW = (-5.0, 5.0)
BROAD_CONTEXT = (-80.0, 5.0)
HALF_WIDTH_SEC = 5.0
SMOOTH_SEC = 2.0
BASELINE_QUANTILE = 0.99
SUSTAIN_SEC = 2.0
TRANSITION_FLANK_SEC = 2.0
# Pseudo-onsets use the same +/-5 s window at pre-onset centers and exclude the
# final 10 s before true EEG onset.  This tests whether the onset neighborhood
# is enriched relative to the preceding minute without requiring unavailable
# post-onset data at block boundaries.
PSEUDO_CENTERS = np.asarray(
    [-55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0],
    dtype=float,
)
PROFILE_CENTERS = np.asarray([-50.0, -40.0, -30.0, -20.0, -10.0, 0.0], dtype=float)
N_NULL = 5000
SEED = 20260713


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quartiles(values) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(v) for v in np.percentile(x, [25.0, 50.0, 75.0]))


def _paired_greater(a, b) -> dict:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    delta = x[keep] - y[keep]
    nonzero = delta[np.abs(delta) > 1e-12]
    if delta.size and nonzero.size:
        w_p = float(wilcoxon(delta, alternative="greater").pvalue)
    else:
        w_p = 1.0
    n_pos = int(np.sum(nonzero > 0.0))
    n_neg = int(np.sum(nonzero < 0.0))
    sign_p = (
        float(binomtest(n_pos, n_pos + n_neg, 0.5, alternative="greater").pvalue)
        if n_pos + n_neg
        else 1.0
    )
    return {
        "n_subjects": int(delta.size),
        "median_delta": float(np.median(delta)) if delta.size else float("nan"),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_tied": int(delta.size - nonzero.size),
        "wilcoxon_greater_p": w_p,
        "sign_greater_p": sign_p,
    }


def _subject_bootstrap_ci(values, *, n_boot: int = 10000, seed: int = SEED) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(x, size=(int(n_boot), x.size), replace=True), axis=1)
    return [float(v) for v in np.percentile(boot, [2.5, 97.5])]


def _compute_missing_bands(
    ds_sid: str,
    missing: list[int],
    inv_rows: list[dict],
    specs: list[tuple[str, float, float]],
    timing_channels: list[str],
    out_root: Path,
    tier: str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Recompute only the 23 short-window cache misses on timing contacts."""
    cache_dir = out_root / "per_subject" / ds_sid / "cache"
    npz_path = cache_dir / f"{ds_sid}_{tier}_eeg_relative_missing_bands.npz"
    json_path = cache_dir / f"{ds_sid}_{tier}_eeg_relative_missing_bands.json"
    contract = {
        "seizure_idxs": sorted(int(x) for x in missing),
        "channels": timing_channels,
        "bands": [name for name, _, _ in specs],
        "distal_baseline_eeg_sec": list(DISTAL_BASELINE),
        "broad_context_eeg_sec": list(BROAD_CONTEXT),
    }
    if npz_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if meta.get("contract") == contract and not meta.get("drops"):
            z = np.load(npz_path, allow_pickle=True)
            return {key: np.asarray(z[key]) for key in z.files}, meta

    cfg = load_phase1_config()
    ln = cfg["line_noise"]
    spec_win = float(cfg["power"]["spectrogram_win_sec"])
    spec_hop = float(cfg["power"]["spectrogram_hop_sec"])
    dataset, sid = ds_sid.split("_", 1)
    arrays: dict[str, np.ndarray] = {"channels": np.asarray(timing_channels)}
    meta = {"subject": ds_sid, "contract": contract, "seizure": {}, "drops": []}
    for idx in missing:
        eeg_rel = float(inv_rows[idx]["eeg_onset_epoch"]) - float(inv_rows[idx]["clin_onset_epoch"])
        # Cover EEG baseline/context and the clinical-onset comparison window.
        pre_sec = max(130.0, 125.0 - eeg_rel)
        post_sec = max(6.0, eeg_rel + 6.0)
        try:
            sw = extract_seizure_window(
                f"{dataset}/{sid}",
                idx,
                pre_sec=pre_sec,
                post_sec=post_sec,
                results_root=ROOT / "results",
                reference=ICTAL_REFERENCE[dataset],
            )
            lookup = _alias_index(sw.ch_names)
            absent = [name for name in timing_channels if name not in lookup]
            if absent:
                raise ValueError(f"timing contacts missing from raw window: {absent}")
            raw_idx = [int(lookup[name]) for name in timing_channels]
            f, t, sxx = _spectrogram_on_hop(sw.signal[np.asarray(raw_idx)], sw.fs, spec_win, spec_hop)
            rel_t = np.asarray(t, dtype=float) - float(sw.pre_sec)
            line_mask = line_noise_bin_mask(f, ln["harmonics_hz"], ln["halfwidth_hz"])
            for band, lo, hi in specs:
                bmask, _, _ = band_bin_selection(f, lo, hi, line_mask, half_open=True)
                if not np.any(bmask):
                    raise ValueError(f"{band}: no usable FFT bins")
                power = np.asarray(sxx[:, bmask, :], dtype=float).sum(axis=1)
                logp = np.log(np.maximum(power, 1e-30))
                bl = resolve_baseline_window(
                    logp.shape[1],
                    hop_sec=spec_hop,
                    pre_sec=sw.pre_sec,
                    buffer_sec=GUARD_SEC,
                    eeg_onset_rel_sec=eeg_rel,
                    min_baseline_valid_sec=MIN_BASELINE_SEC,
                )
                zt, _ = robust_z_with_flags(
                    logp,
                    (bl.start_idx, bl.end_idx),
                    spec_hop,
                    MIN_BASELINE_SEC,
                )
                arrays[f"{band}__zt__{idx}"] = zt.astype(np.float32)
                arrays[f"{band}__relt__{idx}"] = rel_t.astype(np.float32)
            meta["seizure"][str(idx)] = {
                "eeg_onset_rel_clinical_sec": eeg_rel,
                "pre_sec": pre_sec,
                "post_sec": post_sec,
                "fs": float(sw.fs),
            }
            del sxx, sw
        except Exception as exc:  # noqa: BLE001 - fail-closed provenance
            meta["drops"].append({"seizure_idx": idx, "reason": f"{type(exc).__name__}:{exc}"})
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return arrays, meta


def _process_subject(
    ds_sid: str,
    specs: list[tuple[str, float, float]],
    out_root: Path,
    *,
    eligible: list[int],
    majority_required: int,
    tier: str,
) -> tuple[list[dict], list[dict], list[np.ndarray], list[dict], dict]:
    dataset, sid = ds_sid.split("_", 1)
    inv_rows, _ = _inventory_rows(dataset, sid)
    cache_npz_path = CACHE_ROOT / f"{ds_sid}.npz"
    cache_json_path = CACHE_ROOT / f"{ds_sid}.json"
    cache_meta = json.loads(cache_json_path.read_text(encoding="utf-8"))
    cache_obj = np.load(cache_npz_path, allow_pickle=True)
    cache_channels = [str(x) for x in cache_obj["channels"]]
    cache_lookup = {name: i for i, name in enumerate(cache_channels)}
    cached_idx = {int(x) for x in cache_meta["seizure_idxs"]}

    lagpat_channels, lagpat_source = _load_lagpat_channels(ds_sid)
    timing_channels = [name for name in lagpat_channels if name in cache_lookup]
    if len(timing_channels) < 6:
        raise RuntimeError(f"{ds_sid}: only {len(timing_channels)} timing contacts in cache")
    timing_idx = np.asarray([cache_lookup[name] for name in timing_channels], dtype=int)
    missing = sorted(set(eligible) - cached_idx)
    fallback, fallback_meta = _compute_missing_bands(
        ds_sid, missing, inv_rows, specs, timing_channels, out_root, tier
    )
    if fallback_meta.get("drops"):
        raise RuntimeError(f"{ds_sid}: fallback drops={fallback_meta['drops']}")

    band_order = [name for name, _, _ in specs]
    seizure_rows: list[dict] = []
    band_rows: list[dict] = []
    pseudo_vectors: list[np.ndarray] = []
    profile_rows: list[dict] = []
    for idx in eligible:
        inv = inv_rows[idx]
        eeg_rel = float(inv["eeg_onset_epoch"]) - float(inv["clin_onset_epoch"])
        clinical_center_eeg = -eeg_rel
        source_obj = cache_obj if idx in cached_idx else fallback
        source = "committed_v2_band_cache" if idx in cached_idx else "raw_short_window_fallback"
        traces: dict[str, np.ndarray] = {}
        rel_ref: np.ndarray | None = None
        n_band_eeg = 0
        n_band_clin = 0
        n_band_step = 0
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
            trace, broad = band_energy_timing(
                z,
                rel_eeg,
                baseline=DISTAL_BASELINE,
                search=BROAD_CONTEXT,
                spatial_q=0.75,
                smooth_sec=SMOOTH_SEC,
                baseline_quantile=BASELINE_QUANTILE,
                sustain_sec=SUSTAIN_SEC,
            )
            eeg_win = detect_centered_window_enhancement(
                trace,
                rel_eeg,
                center_sec=0.0,
                half_width_sec=HALF_WIDTH_SEC,
                baseline=DISTAL_BASELINE,
                baseline_quantile=BASELINE_QUANTILE,
                sustain_sec=SUSTAIN_SEC,
            )
            clin_win = detect_centered_window_enhancement(
                trace,
                rel_eeg,
                center_sec=clinical_center_eeg,
                half_width_sec=HALF_WIDTH_SEC,
                baseline=DISTAL_BASELINE,
                baseline_quantile=BASELINE_QUANTILE,
                sustain_sec=SUSTAIN_SEC,
            )
            _, step = max_upward_transition(
                trace,
                rel_eeg,
                baseline=DISTAL_BASELINE,
                search=EEG_ONSET_WINDOW,
                flank_sec=TRANSITION_FLANK_SEC,
                baseline_quantile=BASELINE_QUANTILE,
            )
            n_band_eeg += int(eeg_win.detected)
            n_band_clin += int(clin_win.detected)
            n_band_step += int(step.detected)
            traces[band] = trace
            band_rows.append(
                {
                    "subject": ds_sid,
                    "seizure_idx": idx,
                    "seizure_id": inv["seizure_id"],
                    "band": band,
                    "band_label": BAND_LABELS[band],
                    "source": source,
                    "n_timing_contacts": len(timing_channels),
                    "eeg_onset_rel_clinical_sec": eeg_rel,
                    "eeg_window_extreme_hit": bool(eeg_win.detected),
                    "clinical_window_extreme_hit": bool(clin_win.detected),
                    "eeg_window_peak_delta_z": eeg_win.peak_value,
                    "baseline_q99_delta_z": eeg_win.threshold,
                    "eeg_window_peak_minus_q99": eeg_win.peak_value - eeg_win.threshold,
                    "eeg_window_longest_above_sec": eeg_win.longest_above_sec,
                    "eeg_window_step_q99_hit": bool(step.detected),
                    "eeg_window_step_time_rel_eeg_sec": step.transition_sec,
                    "broad_first_sustained_rel_eeg_sec": broad.rise_sec,
                    "broad_sustained_detected": bool(broad.detected),
                }
            )

        assert rel_ref is not None
        consensus = np.nanmedian(np.vstack([traces[name] for name in band_order]), axis=0)
        recruitment = detect_multiband_recruitment_onset(
            np.vstack([traces[name] for name in band_order]),
            rel_ref,
            baseline=DISTAL_BASELINE,
            search=BROAD_CONTEXT,
            majority_required=majority_required,
            post_sec=5.0,
            flank_sec=TRANSITION_FLANK_SEC,
            baseline_quantile=BASELINE_QUANTILE,
            sustain_sec=SUSTAIN_SEC,
        )
        eeg_cons = detect_centered_window_enhancement(
            consensus,
            rel_ref,
            center_sec=0.0,
            half_width_sec=HALF_WIDTH_SEC,
            baseline=DISTAL_BASELINE,
            baseline_quantile=BASELINE_QUANTILE,
            sustain_sec=SUSTAIN_SEC,
        )
        clin_cons = detect_centered_window_enhancement(
            consensus,
            rel_ref,
            center_sec=clinical_center_eeg,
            half_width_sec=HALF_WIDTH_SEC,
            baseline=DISTAL_BASELINE,
            baseline_quantile=BASELINE_QUANTILE,
            sustain_sec=SUSTAIN_SEC,
        )
        _, step_cons = max_upward_transition(
            consensus,
            rel_ref,
            baseline=DISTAL_BASELINE,
            search=EEG_ONSET_WINDOW,
            flank_sec=TRANSITION_FLANK_SEC,
            baseline_quantile=BASELINE_QUANTILE,
        )
        pseudo = centered_window_hit_profile(
            consensus,
            rel_ref,
            PSEUDO_CENTERS,
            half_width_sec=HALF_WIDTH_SEC,
            baseline=DISTAL_BASELINE,
            baseline_quantile=BASELINE_QUANTILE,
            sustain_sec=SUSTAIN_SEC,
        )
        profile = centered_window_hit_profile(
            consensus,
            rel_ref,
            PROFILE_CENTERS,
            half_width_sec=HALF_WIDTH_SEC,
            baseline=DISTAL_BASELINE,
            baseline_quantile=BASELINE_QUANTILE,
            sustain_sec=SUSTAIN_SEC,
        )
        pseudo_vectors.append(pseudo)
        for center, hit in zip(PROFILE_CENTERS, profile):
            profile_rows.append(
                {"subject": ds_sid, "seizure_idx": idx, "center_rel_eeg_sec": center, "hit": bool(hit)}
            )
        seizure_rows.append(
            {
                "subject": ds_sid,
                "seizure_idx": idx,
                "seizure_id": inv["seizure_id"],
                "source": source,
                "n_timing_contacts": len(timing_channels),
                "eeg_onset_rel_clinical_sec": eeg_rel,
                "consensus_eeg_window_extreme_hit": bool(eeg_cons.detected),
                "consensus_clinical_window_extreme_hit": bool(clin_cons.detected),
                "consensus_eeg_window_peak_delta_z": eeg_cons.peak_value,
                "consensus_baseline_q99_delta_z": eeg_cons.threshold,
                "consensus_eeg_window_peak_minus_q99": eeg_cons.peak_value - eeg_cons.threshold,
                "consensus_eeg_window_longest_above_sec": eeg_cons.longest_above_sec,
                "consensus_eeg_window_step_q99_hit": bool(step_cons.detected),
                "consensus_eeg_window_step_time_rel_eeg_sec": step_cons.transition_sec,
                "n_band_eeg_window_extreme_hits": n_band_eeg,
                "n_bands_total": len(band_order),
                "majority_required": majority_required,
                "majority_band_eeg_window_hit": bool(n_band_eeg >= majority_required),
                "n_band_clinical_window_extreme_hits": n_band_clin,
                "majority_band_clinical_window_hit": bool(n_band_clin >= majority_required),
                "n_band_eeg_window_step_hits": n_band_step,
                "pseudo_onset_expected_hit_fraction": float(np.mean(pseudo)),
                "energy_recruitment_onset_detected": bool(recruitment.detected),
                "energy_recruitment_onset_rel_eeg_sec": recruitment.onset_sec,
                "energy_recruitment_onset_rel_clinical_sec": recruitment.onset_sec + eeg_rel,
                "energy_recruitment_step_delta_z": recruitment.step_delta,
                "energy_recruitment_step_q99_delta_z": recruitment.step_threshold,
                "energy_recruitment_consensus_post_sustained": recruitment.consensus_post_sustained,
                "energy_recruitment_n_band_post_sustained": recruitment.n_band_post_sustained,
            }
        )

    cache_obj.close()
    subject_meta = {
        "subject": ds_sid,
        "tier": tier,
        "n_eligible": len(eligible),
        "eligible_seizure_idxs": eligible,
        "n_cached": len(set(eligible) & cached_idx),
        "n_fallback": len(missing),
        "timing_contacts": timing_channels,
        "n_timing_contacts": len(timing_channels),
        "timing_contact_source": lagpat_source,
    }
    return seizure_rows, band_rows, pseudo_vectors, profile_rows, subject_meta


def _aggregate_subject_rows(seizure_rows: list[dict]) -> list[dict]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in seizure_rows:
        by_subject[row["subject"]].append(row)
    out = []
    for subject, rows in sorted(by_subject.items()):
        n = len(rows)
        frac = lambda key: float(np.mean([bool(r[key]) for r in rows]))
        out.append(
            {
                "subject": subject,
                "n_seizures": n,
                "consensus_eeg_window_hit_fraction": frac("consensus_eeg_window_extreme_hit"),
                "consensus_clinical_window_hit_fraction": frac("consensus_clinical_window_extreme_hit"),
                "majority_band_eeg_window_hit_fraction": frac("majority_band_eeg_window_hit"),
                "majority_band_clinical_window_hit_fraction": frac("majority_band_clinical_window_hit"),
                "consensus_eeg_window_step_hit_fraction": frac("consensus_eeg_window_step_q99_hit"),
                "mean_n_band_eeg_window_hits": float(np.mean([r["n_band_eeg_window_extreme_hits"] for r in rows])),
                "median_n_band_eeg_window_hits": float(np.median([r["n_band_eeg_window_extreme_hits"] for r in rows])),
                "pseudo_onset_expected_hit_fraction": float(np.mean([r["pseudo_onset_expected_hit_fraction"] for r in rows])),
                "eeg_minus_clinical_hit_fraction": (
                    frac("consensus_eeg_window_extreme_hit") - frac("consensus_clinical_window_extreme_hit")
                ),
                "eeg_minus_pseudo_hit_fraction": (
                    frac("consensus_eeg_window_extreme_hit")
                    - float(np.mean([r["pseudo_onset_expected_hit_fraction"] for r in rows]))
                ),
                "energy_recruitment_onset_detection_fraction": float(
                    np.mean([bool(r["energy_recruitment_onset_detected"]) for r in rows])
                ),
                "energy_recruitment_onset_rel_eeg_median_sec": float(
                    np.median(
                        [
                            r["energy_recruitment_onset_rel_eeg_sec"]
                            for r in rows
                            if r["energy_recruitment_onset_detected"]
                        ]
                    )
                )
                if any(r["energy_recruitment_onset_detected"] for r in rows)
                else float("nan"),
                "energy_recruitment_onset_rel_clinical_median_sec": float(
                    np.median(
                        [
                            r["energy_recruitment_onset_rel_clinical_sec"]
                            for r in rows
                            if r["energy_recruitment_onset_detected"]
                        ]
                    )
                )
                if any(r["energy_recruitment_onset_detected"] for r in rows)
                else float("nan"),
            }
        )
    return out


def _aggregate_band_rows(band_rows: list[dict], band_order: list[str]) -> tuple[list[dict], list[dict]]:
    by_subject_band: dict[tuple[str, str], list[dict]] = defaultdict(list)
    subjects = sorted({row["subject"] for row in band_rows})
    for row in band_rows:
        by_subject_band[(row["subject"], row["band"])].append(row)
    subject_band = []
    cohort_band = []
    for band in band_order:
        vals = []
        for subject in subjects:
            rr = by_subject_band[(subject, band)]
            value = float(np.mean([bool(r["eeg_window_extreme_hit"]) for r in rr]))
            vals.append(value)
            subject_band.append(
                {
                    "subject": subject,
                    "band": band,
                    "band_label": BAND_LABELS[band],
                    "n_seizures": len(rr),
                    "eeg_window_hit_fraction": value,
                    "clinical_window_hit_fraction": float(
                        np.mean([bool(r["clinical_window_extreme_hit"]) for r in rr])
                    ),
                    "eeg_window_step_hit_fraction": float(
                        np.mean([bool(r["eeg_window_step_q99_hit"]) for r in rr])
                    ),
                }
            )
        q25, med, q75 = _quartiles(vals)
        pooled = [r for r in band_rows if r["band"] == band]
        cohort_band.append(
            {
                "band": band,
                "band_label": BAND_LABELS[band],
                "n_subjects": len(subjects),
                "n_seizures": len(pooled),
                "pooled_eeg_window_hit_fraction": float(
                    np.mean([bool(r["eeg_window_extreme_hit"]) for r in pooled])
                ),
                "subject_hit_fraction_q25": q25,
                "subject_hit_fraction_median": med,
                "subject_hit_fraction_q75": q75,
                "subject_hit_fraction_mean": float(np.mean(vals)),
            }
        )
    return subject_band, cohort_band


def _pseudo_null(
    subject_rows: list[dict],
    pseudo_by_subject: dict[str, list[np.ndarray]],
    *,
    n_null: int = N_NULL,
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    subjects = [row["subject"] for row in subject_rows]
    observed = float(np.mean([row["consensus_eeg_window_hit_fraction"] for row in subject_rows]))
    null = np.empty(int(n_null), dtype=float)
    for perm in range(int(n_null)):
        subject_fractions = []
        for subject in subjects:
            vectors = pseudo_by_subject[subject]
            hits = [bool(v[int(rng.integers(0, len(v)))]) for v in vectors]
            subject_fractions.append(float(np.mean(hits)))
        null[perm] = float(np.mean(subject_fractions))
    return {
        "n_null": int(n_null),
        "seed": int(seed),
        "observed_unweighted_subject_mean": observed,
        "null_mean": float(np.mean(null)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "empirical_greater_p": float((1 + np.sum(null >= observed)) / (1 + null.size)),
    }


def _energy_onset_summary(seizure_rows: list[dict], subject_rows: list[dict]) -> dict:
    confirmed = [row for row in seizure_rows if row["energy_recruitment_onset_detected"]]
    rel_eeg = np.asarray([row["energy_recruitment_onset_rel_eeg_sec"] for row in confirmed], dtype=float)
    rel_clin = np.asarray(
        [row["energy_recruitment_onset_rel_clinical_sec"] for row in confirmed], dtype=float
    )
    subject_fraction = np.asarray(
        [row["energy_recruitment_onset_detection_fraction"] for row in subject_rows], dtype=float
    )
    subject_median_eeg = np.asarray(
        [row["energy_recruitment_onset_rel_eeg_median_sec"] for row in subject_rows], dtype=float
    )
    subject_median_eeg = subject_median_eeg[np.isfinite(subject_median_eeg)]
    percentiles = lambda x, q: [float(v) for v in np.percentile(x, q)] if x.size else [float("nan")] * len(q)
    return {
        "definition": (
            "largest multiband-consensus upward step in the broad post-baseline window; "
            "step > baseline Q99, post-step consensus sustained > level Q99, and band-majority confirmation"
        ),
        "search_sec_rel_eeg": list(BROAD_CONTEXT),
        "selection_uses_plus_minus_5_window": False,
        "n_detected": len(confirmed),
        "n_total": len(seizure_rows),
        "pooled_detection_fraction_descriptive": len(confirmed) / len(seizure_rows),
        "subject_detection_fraction_q25_median_q75": percentiles(subject_fraction, [25, 50, 75]),
        "subject_detection_fraction_mean": float(np.mean(subject_fraction)),
        "onset_rel_eeg_pooled_q10_q25_median_q75_q90_sec": percentiles(
            rel_eeg, [10, 25, 50, 75, 90]
        ),
        "subject_median_onset_rel_eeg_q25_median_q75_sec": percentiles(
            subject_median_eeg, [25, 50, 75]
        ),
        "median_abs_distance_to_eeg_sec": float(np.median(np.abs(rel_eeg))) if rel_eeg.size else float("nan"),
        "median_abs_distance_to_clinical_sec": (
            float(np.median(np.abs(rel_clin))) if rel_clin.size else float("nan")
        ),
        "n_closer_to_eeg": int(np.sum(np.abs(rel_eeg) < np.abs(rel_clin))),
        "n_closer_to_clinical": int(np.sum(np.abs(rel_clin) < np.abs(rel_eeg))),
        "n_equal_distance": int(np.sum(np.abs(rel_clin) == np.abs(rel_eeg))),
        "fraction_within_eeg": {
            str(sec): float(np.mean(np.abs(rel_eeg) <= sec)) for sec in (1, 2, 5, 10, 20)
        },
    }


def _plot_energy_onsets(seizure_rows: list[dict], subject_rows: list[dict], out_path: Path) -> None:
    confirmed = [row for row in seizure_rows if row["energy_recruitment_onset_detected"]]
    subject_order = [row["subject"] for row in subject_rows]
    labels = [subject.replace("epilepsiae_", "E") for subject in subject_order]
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in confirmed:
        by_subject[row["subject"]].append(row)

    fig, axs = plt.subplots(2, 2, figsize=(12.2, 9.0), gridspec_kw={"hspace": 0.40, "wspace": 0.32})
    ax = axs[0, 0]
    for yi, subject in enumerate(subject_order):
        vals = [row["energy_recruitment_onset_rel_eeg_sec"] for row in by_subject[subject]]
        ax.scatter(vals, np.full(len(vals), yi), s=13, alpha=0.60, color="#4C78A8")
        if vals:
            ax.scatter([np.median(vals)], [yi], marker="|", s=90, color="black", linewidth=1.4)
    ax.axvline(0.0, color="#7A4F9A", ls=":", lw=1.1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(BROAD_CONTEXT)
    ax.set_xlabel("energy-recruitment onset relative to EEG onset (s)")
    ax.set_title("a  Data-defined onset by subject", loc="left")

    ax = axs[0, 1]
    fractions = [row["energy_recruitment_onset_detection_fraction"] for row in subject_rows]
    ax.bar(np.arange(len(labels)), fractions, color="#4C78A8", alpha=0.82)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("confirmed fraction")
    ax.set_title("b  Within-subject detector coverage", loc="left")

    ax = axs[1, 0]
    rel_eeg = np.asarray([row["energy_recruitment_onset_rel_eeg_sec"] for row in confirmed], dtype=float)
    rel_clin = np.asarray(
        [row["energy_recruitment_onset_rel_clinical_sec"] for row in confirmed], dtype=float
    )
    bins = np.arange(BROAD_CONTEXT[0], max(20.0, float(np.nanmax(rel_clin)) + 5.0), 5.0)
    ax.hist(rel_eeg, bins=bins, density=True, histtype="step", lw=1.8, color="#7A4F9A", label="relative EEG")
    ax.hist(rel_clin, bins=bins, density=True, histtype="step", lw=1.8, color="#C23B22", label="relative clinical")
    ax.axvline(0.0, color="0.25", ls="--", lw=0.8)
    ax.set_xlabel("detected onset latency (s)")
    ax.set_ylabel("density")
    ax.set_title("c  Which annotation is closer after detection?", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[1, 1]
    band_counts = np.asarray(
        [row["energy_recruitment_n_band_post_sustained"] for row in confirmed], dtype=int
    )
    if band_counts.size:
        values, counts = np.unique(band_counts, return_counts=True)
        ax.bar(values, counts / counts.sum(), color="#59A14F", alpha=0.82)
    ax.set_xlabel("bands sustained after detected change point")
    ax.set_ylabel("fraction of confirmed seizures")
    ax.set_title("d  Multiband confirmation", loc="left")
    ax.set_ylim(0.0, 1.0)

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Epilepsiae: data-defined multiband energy-recruitment onset", fontsize=14, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_cohort(
    subject_rows: list[dict],
    subject_band_rows: list[dict],
    profile_rows: list[dict],
    pseudo_result: dict,
    band_order: list[str],
    tier_label: str,
    out_path: Path,
) -> None:
    subjects = [row["subject"].replace("epilepsiae_", "E") for row in subject_rows]
    eeg = np.asarray([row["consensus_eeg_window_hit_fraction"] for row in subject_rows])
    clin = np.asarray([row["consensus_clinical_window_hit_fraction"] for row in subject_rows])
    pseudo = np.asarray([row["pseudo_onset_expected_hit_fraction"] for row in subject_rows])

    fig, axs = plt.subplots(2, 2, figsize=(12.4, 9.2), gridspec_kw={"hspace": 0.38, "wspace": 0.32})
    x = np.arange(len(subjects))

    ax = axs[0, 0]
    for xi, a, b in zip(x, pseudo, eeg):
        ax.plot([xi, xi], [a, b], color="0.75", lw=0.8)
    ax.scatter(x, pseudo, facecolor="white", edgecolor="0.35", s=30, label="pseudo-onset expectation")
    ax.scatter(x, eeg, color="#7A4F9A", s=30, label="EEG onset +/-5 s")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=90, fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("within-subject hit fraction")
    ax.set_title("a  EEG-onset window versus time-shifted pseudo-onsets", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[0, 1]
    for xi, a, b in zip(x, clin, eeg):
        ax.plot([xi, xi], [a, b], color="0.75", lw=0.8)
    ax.scatter(x, clin, facecolor="white", edgecolor="#C23B22", s=30, label="clinical onset +/-5 s")
    ax.scatter(x, eeg, color="#7A4F9A", s=30, label="EEG onset +/-5 s")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=90, fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("within-subject hit fraction")
    ax.set_title("b  EEG onset versus clinical onset", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[1, 0]
    lookup = {(r["subject"], r["band"]): r["eeg_window_hit_fraction"] for r in subject_band_rows}
    full_subjects = [row["subject"] for row in subject_rows]
    mat = np.asarray([[lookup[(s, b)] for b in band_order] for s in full_subjects], dtype=float)
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=7)
    ax.set_xticks(np.arange(len(band_order)))
    ax.set_xticklabels([BAND_SHORT[b] for b in band_order])
    ax.set_title("c  Per-subject EEG-window hit fraction by band", loc="left")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.035, label="fraction")

    ax = axs[1, 1]
    by_subject_center: dict[tuple[str, float], list[bool]] = defaultdict(list)
    for row in profile_rows:
        by_subject_center[(row["subject"], float(row["center_rel_eeg_sec"]))].append(bool(row["hit"]))
    means = []
    q25 = []
    q75 = []
    for center in PROFILE_CENTERS:
        vals = np.asarray(
            [np.mean(by_subject_center[(row["subject"], float(center))]) for row in subject_rows], dtype=float
        )
        means.append(float(np.mean(vals)))
        q25.append(float(np.quantile(vals, 0.25)))
        q75.append(float(np.quantile(vals, 0.75)))
    ax.plot(PROFILE_CENTERS, means, color="#7A4F9A", marker="o", lw=1.5)
    ax.fill_between(PROFILE_CENTERS, q25, q75, color="#7A4F9A", alpha=0.18, lw=0)
    ax.axvline(0.0, color="0.2", ls="--", lw=0.9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("10-s window center relative to EEG onset (s)")
    ax.set_ylabel("unweighted subject-mean hit fraction")
    ax.set_title("d  Temporal specificity profile", loc="left")
    ax.text(
        0.02,
        0.98,
        f"pseudo-null p={pseudo_result['empirical_greater_p']:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Epilepsiae cohort: {tier_label} energy above distal-baseline Q99 near EEG onset",
        fontsize=14,
        y=0.99,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_readme(fig_dir: Path, *, tier_label: str, majority_required: int, n_bands: int) -> None:
    (fig_dir / "README.md").write_text(
        "# EEG-onset multiband energy cohort audit\n\n"
        "### epilepsiae_cohort_onset_energy_alignment.png\n\n"
        f"18 位当前 energy-analysis-eligible Epilepsiae 患者的 `{tier_label}` cohort 审查。Panel a 比较 EEG onset ±5 s 与同宽伪 onset 窗，"
        "Panel b 比较 EEG 与 clinical onset，Panel c 给出逐患者逐频带命中比例，Panel d 展示相同 10 s 窗沿 EEG onset 前后平移后的时间特异性。"
        f"命中要求固定 lagPat-valid contacts 上的空间 Q75 多频带能量持续至少 2 s 超过该 seizure 自身 EEG-relative `[-120,-90] s` baseline Q99；独立频带多数门为 `≥{majority_required}/{n_bands}`。\n\n"
        "**关注点**：cohort 推断以 subject 为统计单位；pooled seizure 比例只作描述。伪 onset 只取 EEG onset 前 `−55` 到 `−10 s` 的同宽窗口。\n",
        encoding="utf-8",
    )
    with (fig_dir / "README.md").open("a", encoding="utf-8") as fh:
        fh.write(
            "\n### epilepsiae_energy_defined_onset_distribution.png\n\n"
            "在 distal baseline 之后的宽窗内直接寻找最大多频带向上 change point，不用 EEG onset ±5 s 决定候选时刻。"
            "图中依次展示逐患者检测时刻、检测覆盖率、检测后相对 EEG/clinical onset 的分布，以及多数频带确认情况。\n\n"
            "**关注点**：这是 energy-recruitment onset，不等同于最早 electrographic seizure onset；当前 baseline 与宽搜索段仍由已有 seizure window 合同提供。\n"
        )


def _tier_contract(tier: str) -> tuple[list[tuple[str, float, float]], int, str, str]:
    all_specs = _primary_band_specs()
    if tier == "common_1_80hz":
        return (
            [spec for spec in all_specs if spec[2] <= 80.0],
            3,
            "common 1-80 Hz five-band",
            "primary_common_1_80hz",
        )
    if tier == "extended_1_250hz":
        return all_specs, 4, "extended 1-250 Hz seven-band", "sensitivity_extended_1_250hz"
    raise ValueError(f"unknown tier: {tier}")


def _eligible_map(subjects: list[str], *, max_band_hz: float) -> tuple[dict[str, list[int]], int]:
    out: dict[str, list[int]] = {subject: [] for subject in subjects}
    total_t0_eligible = 0
    for row in csv.DictReader(AUDIT_CSV.open(encoding="utf-8")):
        if row["dataset"] != "epilepsiae" or row["analysis_eligible"].lower() != "true":
            continue
        if row["subject_id"] not in out:
            continue
        total_t0_eligible += 1
        if float(row["fs"]) / 2.0 > float(max_band_hz):
            out[row["subject_id"]].append(int(row["seizure_idx"]))
    for subject in out:
        out[subject].sort()
    return out, total_t0_eligible


def run_tier(out_root: Path, *, tier: str, n_null: int = N_NULL) -> Path:
    specs, majority_required, tier_label, _ = _tier_contract(tier)
    band_order = [name for name, _, _ in specs]
    subjects = sorted(path.stem for path in CACHE_ROOT.glob("epilepsiae_*.json"))
    if not subjects:
        raise RuntimeError("no Epilepsiae v2 band caches found")
    eligible_by_subject, total_t0_eligible = _eligible_map(
        subjects, max_band_hz=max(hi for _, _, hi in specs)
    )
    if any(not eligible_by_subject[subject] for subject in subjects):
        empty = [subject for subject in subjects if not eligible_by_subject[subject]]
        raise RuntimeError(f"{tier}: subjects without frequency-complete eligible seizures: {empty}")

    all_seizure: list[dict] = []
    all_band: list[dict] = []
    all_profile: list[dict] = []
    pseudo_by_subject: dict[str, list[np.ndarray]] = {}
    subject_meta = []
    for subject in subjects:
        print(f"[{tier}] {subject}", flush=True)
        seizure, band, pseudo, profile, meta = _process_subject(
            subject,
            specs,
            out_root,
            eligible=eligible_by_subject[subject],
            majority_required=majority_required,
            tier=tier,
        )
        all_seizure.extend(seizure)
        all_band.extend(band)
        all_profile.extend(profile)
        pseudo_by_subject[subject] = pseudo
        subject_meta.append(meta)

    subject_rows = _aggregate_subject_rows(all_seizure)
    subject_band_rows, cohort_band_rows = _aggregate_band_rows(all_band, band_order)
    pseudo_result = _pseudo_null(subject_rows, pseudo_by_subject, n_null=n_null)
    eeg = np.asarray([r["consensus_eeg_window_hit_fraction"] for r in subject_rows])
    clinical = np.asarray([r["consensus_clinical_window_hit_fraction"] for r in subject_rows])
    pseudo = np.asarray([r["pseudo_onset_expected_hit_fraction"] for r in subject_rows])
    majority = np.asarray([r["majority_band_eeg_window_hit_fraction"] for r in subject_rows])
    step = np.asarray([r["consensus_eeg_window_step_hit_fraction"] for r in subject_rows])

    _write_csv(out_root / "seizure_level_onset_energy.csv", all_seizure)
    _write_csv(out_root / "band_seizure_level_onset_energy.csv", all_band)
    _write_csv(out_root / "subject_level_onset_energy.csv", subject_rows)
    _write_csv(out_root / "subject_band_onset_energy.csv", subject_band_rows)
    _write_csv(out_root / "cohort_band_onset_energy.csv", cohort_band_rows)
    _write_csv(out_root / "window_offset_profile.csv", all_profile)

    q25, med, q75 = _quartiles(eeg)
    mq25, mmed, mq75 = _quartiles(majority)
    total_eligible = len(all_seizure)
    total_attempted = sum(
        1
        for row in csv.DictReader(AUDIT_CSV.open(encoding="utf-8"))
        if row["dataset"] == "epilepsiae"
    )
    summary = {
        "tier": tier,
        "tier_label": tier_label,
        "analysis_level": "cohort-level onset-alignment audit; subject is the inferential unit",
        "denominator": {
            "epilepsiae_sql_subjects": 27,
            "epilepsiae_sql_seizures": 542,
            "energy_analysis_subjects": len(subjects),
            "t0_attempted_seizures_in_energy_subjects": total_attempted,
            "t0_eligible_seizures_before_frequency_support": total_t0_eligible,
            "frequency_complete_eligible_seizures": total_eligible,
            "frequency_support_exclusions": total_t0_eligible - total_eligible,
            "cached_seizures": sum(m["n_cached"] for m in subject_meta),
            "raw_fallback_seizures": sum(m["n_fallback"] for m in subject_meta),
        },
        "contract": {
            "time_axis": "seconds relative to database-provided EEG onset",
            "distal_baseline_sec": list(DISTAL_BASELINE),
            "onset_window_sec": list(EEG_ONSET_WINDOW),
            "threshold": "within-seizure distal-baseline Q99",
            "sustain_sec": SUSTAIN_SEC,
            "spatial_summary": "Q75 across fixed masked-rank-displacement lagPat-valid contacts",
            "smooth_sec": SMOOTH_SEC,
            "bands": band_order,
            "consensus": f"pointwise median of {len(band_order)} band traces",
            "multiband_concordance": f">={majority_required} of {len(band_order)} individual bands hit",
            "pseudo_centers_rel_eeg_sec": PSEUDO_CENTERS.tolist(),
            "clinical_control": "same +/-5 s window centered on SQL clinical onset",
            "energy_recruitment_detector": {
                "search_sec_rel_eeg": list(BROAD_CONTEXT),
                "candidate": "largest multiband-consensus upward step",
                "confirmation": (
                    "step > baseline-step Q99 AND post-step consensus sustained above level Q99 "
                    f"AND >= {majority_required}/{len(band_order)} bands sustained"
                ),
                "uses_eeg_plus_minus_5_for_selection": False,
            },
        },
        "primary_consensus_eeg_window": {
            "pooled_hits": int(sum(bool(r["consensus_eeg_window_extreme_hit"]) for r in all_seizure)),
            "pooled_total": total_eligible,
            "pooled_fraction_descriptive": float(
                np.mean([bool(r["consensus_eeg_window_extreme_hit"]) for r in all_seizure])
            ),
            "subject_fraction_q25": q25,
            "subject_fraction_median": med,
            "subject_fraction_q75": q75,
            "unweighted_subject_mean": float(np.mean(eeg)),
            "unweighted_subject_mean_bootstrap95": _subject_bootstrap_ci(eeg),
        },
        "majority_band_concordance": {
            "pooled_hits": int(sum(bool(r["majority_band_eeg_window_hit"]) for r in all_seizure)),
            "pooled_total": total_eligible,
            "pooled_fraction_descriptive": float(
                np.mean([bool(r["majority_band_eeg_window_hit"]) for r in all_seizure])
            ),
            "subject_fraction_q25": mq25,
            "subject_fraction_median": mmed,
            "subject_fraction_q75": mq75,
            "unweighted_subject_mean": float(np.mean(majority)),
            "unweighted_subject_mean_bootstrap95": _subject_bootstrap_ci(majority, seed=SEED + 1),
        },
        "local_step_sensitivity": {
            "subject_fraction_median": float(np.median(step)),
            "unweighted_subject_mean": float(np.mean(step)),
        },
        "data_defined_energy_recruitment_onset": _energy_onset_summary(all_seizure, subject_rows),
        "controls": {
            "pseudo_onset": pseudo_result,
            "eeg_vs_subject_expected_pseudo": _paired_greater(eeg, pseudo),
            "eeg_vs_clinical_onset": _paired_greater(eeg, clinical),
            "clinical_unweighted_subject_mean": float(np.mean(clinical)),
            "pseudo_expected_unweighted_subject_mean": float(np.mean(pseudo)),
        },
        "subject_meta": subject_meta,
        "outputs": {
            "seizure_level": str((out_root / "seizure_level_onset_energy.csv").relative_to(ROOT)),
            "subject_level": str((out_root / "subject_level_onset_energy.csv").relative_to(ROOT)),
            "band_level": str((out_root / "cohort_band_onset_energy.csv").relative_to(ROOT)),
            "figure": str(
                (out_root / "figures/epilepsiae_cohort_onset_energy_alignment.png").relative_to(ROOT)
            ),
            "energy_onset_figure": str(
                (out_root / "figures/epilepsiae_energy_defined_onset_distribution.png").relative_to(ROOT)
            ),
        },
    }
    summary_path = out_root / "cohort_summary.json"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fig_dir = out_root / "figures"
    _plot_cohort(
        subject_rows,
        subject_band_rows,
        all_profile,
        pseudo_result,
        band_order,
        tier_label,
        fig_dir / "epilepsiae_cohort_onset_energy_alignment.png",
    )
    _plot_energy_onsets(
        all_seizure,
        subject_rows,
        fig_dir / "epilepsiae_energy_defined_onset_distribution.png",
    )
    _write_readme(
        fig_dir,
        tier_label=tier_label,
        majority_required=majority_required,
        n_bands=len(band_order),
    )
    return summary_path


def _write_combined_summary(out_root: Path) -> Path:
    combined = {
        "primary_tier": "common_1_80hz",
        "sensitivity_tier": "extended_1_250hz",
        "reason": (
            "1-80 Hz is common to all T0-eligible sampling rates; 1-250 Hz excludes "
            "256-Hz seizures whose Nyquist frequency is 128 Hz"
        ),
        "tiers": {},
    }
    for tier in ("common_1_80hz", "extended_1_250hz"):
        _, _, _, dirname = _tier_contract(tier)
        path = out_root / dirname / "cohort_summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        entry = {
            "summary_path": str(path.relative_to(ROOT)),
            "denominator": summary["denominator"],
            "primary_consensus_eeg_window": summary["primary_consensus_eeg_window"],
            "majority_band_concordance": summary["majority_band_concordance"],
            "controls": summary["controls"],
        }
        if "data_defined_energy_recruitment_onset" in summary:
            entry["data_defined_energy_recruitment_onset"] = summary[
                "data_defined_energy_recruitment_onset"
            ]
        combined["tiers"][tier] = entry
    path = out_root / "cohort_summary.json"
    out_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run(out_root: Path, *, n_null: int = N_NULL) -> Path:
    for tier in ("common_1_80hz", "extended_1_250hz"):
        _, _, _, dirname = _tier_contract(tier)
        run_tier(out_root / dirname, tier=tier, n_null=n_null)
    return _write_combined_summary(out_root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--tier", choices=("both", "common_1_80hz", "extended_1_250hz"), default="both")
    args = ap.parse_args()
    if args.tier == "both":
        out = run(args.out_root, n_null=args.n_null)
    else:
        _, _, _, dirname = _tier_contract(args.tier)
        out = run_tier(args.out_root / dirname, tier=args.tier, n_null=args.n_null)
        _write_combined_summary(args.out_root)
    print(out)


if __name__ == "__main__":
    main()

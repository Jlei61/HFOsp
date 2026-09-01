#!/usr/bin/env python3
"""Run the full T_spectral-aligned phenotype-matched energy-field analysis.

The runner reads accepted T_spectral caches only for event inclusion, phenotype,
time-zero and annotation provenance.  Exact 1-150 Hz broadband, 60-100 Hz HFA,
and the explicitly labelled 30-80 Hz sensitivity are recomputed from canonical
raw SEEG.  Frozen interictal fields are loaded through their fingerprint-checking
API; no ictal value can refit any spatial construction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import iirnotch, spectrogram, sosfiltfilt, tf2sos

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE  # noqa: E402
from src import topic5_ictal_recruitment as recruit  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.topic5_template_axis_field import (  # noqa: E402
    INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
    interictal_field_quality_tier,
    scorers_from_interictal_record,
)
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    ANALYSIS_CONTRACT,
    ANALYSIS_RANGE_SEC,
    DISTAL_BASELINE_EEG_SEC,
    FIXED_WINDOWS,
    PHENOTYPE_LABEL_VERSION,
    WINDOW_SCALES,
    aggregate_complete_windows,
    annotation_provenance,
    bootstrap_median_ci,
    distal_baseline_robust_z,
    eligibility_drop_reason,
    exact_name_align_matrix,
    fold_seizure_null_draws,
    fixed_window_sign_flip_maxt,
    independent_label_permutation_maxt,
    jsonable,
    make_complete_window_grid,
    make_contact_permutations,
    paired_sign_flip_p,
    phenotype_selector_sets,
    score_observed_bundle,
    score_permutation_matrix,
    sign_flip_cluster_maxt,
    tspectral_reference_for_raw_eeg,
    tspectral_zeroed_times,
)


ARTIFACT_ROOT = Path(os.environ.get("HFOSP_ARTIFACT_ROOT", ROOT)).resolve()
EPI_CACHE = ARTIFACT_ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_common_1_80hz"
YUQ_CACHE = ARTIFACT_ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_yuquan_common_1_80hz"
FIELD_ROOT = Path(os.environ.get(
    "HFOSP_INTERICTAL_FIELD_DIR",
    ARTIFACT_ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject",
)).resolve()
PHENOTYPE_TABLE = (ARTIFACT_ROOT / "results/topic5_ictal_recruitment/peri_onset_energy_timing/"
                   "early_spectral_phenotype/per_seizure_spectral_overlap_state.csv")
DEFAULT_OUT = ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance"
PAPER_FIGURES = ROOT / "results/paper-ready-figure/fig3-sup-tspectral-field-concordance/figures"

SPECTRAL_WINDOW_SEC = 1.0
SPECTRAL_HOP_SEC = 0.5
MIN_CONTACTS = 6
MIN_BASELINE_FRAMES = 50
NOTCH_FREQS = (50.0, 100.0, 150.0, 200.0)

BANDS = {
    "broadband_1_150": {
        "hz": (1.0, 150.0),
        "label": "Broadband 1–150 Hz",
        "phenotype": "broadband_1_150",
        "role": "primary",
    },
    "hfa_60_100": {
        "hz": (60.0, 100.0),
        "label": "HFA 60–100 Hz",
        "phenotype": "gamma_nonbroadband",
        "role": "primary",
    },
    "gamma_30_80_sensitivity": {
        "hz": (30.0, 80.0),
        "label": "Gamma 30–80 Hz sensitivity",
        "phenotype": "gamma_nonbroadband",
        "role": "label_matched_sensitivity",
    },
}
MAIN_BAND_FOR_PHENOTYPE = {
    "broadband_1_150": "broadband_1_150",
    "gamma_nonbroadband": "hfa_60_100",
}
PHENOTYPE_MATCHED_READOUTS = {
    "primary": {
        "broadband_1_150": "broadband_1_150",
        "gamma_nonbroadband": "hfa_60_100",
    },
    "gamma_30_80_substitution_sensitivity": {
        "broadband_1_150": "broadband_1_150",
        "gamma_nonbroadband": "gamma_30_80_sensitivity",
    },
}
FIXED_WINDOW_ORDER = ("distal", "pre20", "pre10", "post10", "post20", "late20_30")
QUALITY_COLUMNS = (
    "field_ready", "geometry_2d_supported", "strict_stability_pass",
    "axis_quality_tier", "axis_relation", "shared_field_available",
    "field_fingerprint_algorithm", "field_fingerprint_sha256",
)


def _seed(token: str, base: int) -> int:
    return int((zlib.crc32(token.encode("utf-8")) + int(base)) % (2**32 - 1))


def _hash_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _cache_npz_paths(subjects: set[str] | None = None) -> list[Path]:
    out = []
    for root in (EPI_CACHE, YUQ_CACHE):
        for path in sorted(root.glob("*.npz")):
            if subjects is None or path.stem in subjects:
                out.append(path)
    return out


def _hash_manifest(paths: Sequence[Path]) -> Dict[str, str]:
    return {str(path.relative_to(ROOT)): _hash_file(path) for path in paths}


def _write_hash_table(path: Path, before: Mapping[str, str], after: Mapping[str, str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256_before", "sha256_after", "unchanged"])
        writer.writeheader()
        for key in sorted(before):
            writer.writerow({"path": key, "sha256_before": before[key],
                             "sha256_after": after.get(key),
                             "unchanged": before[key] == after.get(key)})


def _load_subject_caches(subject_filter: set[str] | None = None):
    records = []
    for cache_root in (EPI_CACHE, YUQ_CACHE):
        for path in sorted(cache_root.glob("*.json")):
            if path.name in {"cache_alignment_summary.json"}:
                continue
            meta = json.loads(path.read_text())
            subject = str(meta.get("subject", path.stem))
            if "seizure_idxs" not in meta or (subject_filter and subject not in subject_filter):
                continue
            records.append((cache_root, path, meta, phenotype_selector_sets(meta)))
    return records


def _field_quality(record: Mapping[str, object]) -> Dict[str, object]:
    field = record.get("interictal_field") or {}
    pair = record.get("axis_pair") or {}
    relation = pair.get("relation") or {}
    return {
        "field_status": field.get("status"),
        "field_ready": field.get("status") == "ok",
        "geometry_2d_supported": pair.get("geometry_2d_supported"),
        "strict_stability_pass": pair.get("strict_stability_pass"),
        "axis_quality_tier": interictal_field_quality_tier(record),
        "axis_relation": relation.get("relation"),
        "shared_field_available": "shared_a" in (field.get("field_models") or {}),
        "field_fingerprint_algorithm": field.get("fingerprint_algorithm"),
        "field_fingerprint_sha256": field.get("fingerprint_sha256"),
        "n_field_contacts": field.get("n_contacts"),
    }


def _quality_values(quality: Mapping[str, object]) -> Dict[str, object]:
    return {key: quality.get(key) for key in QUALITY_COLUMNS}


def _event_inventory(cache_records) -> tuple[list[dict], list[dict]]:
    inventory, drops = [], []
    for cache_root, sidecar_path, meta, selectors in cache_records:
        subject = str(meta["subject"])
        dataset = subject.split("_", 1)[0]
        field_path = FIELD_ROOT / f"{subject}.json"
        if field_path.exists():
            field_record = json.loads(field_path.read_text())
            quality = _field_quality(field_record)
        else:
            quality = _field_quality({})
        for idx in sorted(selectors["accepted"]):
            event = (meta.get("seizure") or {}).get(str(idx), {})
            label = event.get("early_spectral_phenotype") or {}
            if idx in selectors["broadband_1_150"]:
                group = "broadband_1_150"
            elif idx in selectors["gamma_nonbroadband"]:
                group = "gamma_nonbroadband"
            elif idx in selectors["not_classified"]:
                group = "not_classified"
            else:
                group = "classified_non_target"
            row = {
                "dataset": dataset, "subject": subject, "seizure_idx": int(idx),
                "accepted_tspectral": True,
                "alignment_status": event.get("alignment_status"),
                "label_status": label.get("label_status", "not_classified"),
                "label_version": label.get("label_version"),
                "phenotype": label.get("phenotype"),
                "simple_phenotype": label.get("simple_phenotype"),
                "analysis_group": group,
                "strict_broadband_selector": idx in selectors["broadband_1_150"],
                "gamma_nonbroadband_selector": idx in selectors["gamma_nonbroadband"],
                "cache_sidecar": str(sidecar_path.relative_to(ROOT)),
                "cache_npz": str((cache_root / f"{subject}.npz").relative_to(ROOT)),
                "analysis_status": "candidate" if group in MAIN_BAND_FOR_PHENOTYPE else group,
                **quality,
            }
            inventory.append(row)
            if group not in MAIN_BAND_FOR_PHENOTYPE:
                reason = "not_classified" if group == "not_classified" else "outside_two_primary_phenotypes"
                drops.append({**{k: row[k] for k in ("dataset", "subject", "seizure_idx", "analysis_group")},
                              "band": "", "drop_type": "contract_exclusion", "drop_reason": reason})
    return inventory, drops


def _notch_yuquan(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply the Fig3-B 50-Hz-harmonic notch to Yuquan's raw loader output."""
    out = np.asarray(signal, float)
    for freq in NOTCH_FREQS:
        if freq < fs / 2.0:
            b, a = iirnotch(freq, 30.0, fs)
            out = sosfiltfilt(tf2sos(b, a), out, axis=-1)
    return out


def _extract_log_band_power(signal: np.ndarray, fs: float,
                            band_keys: Sequence[str], *,
                            band_hz_override: Mapping[str, Sequence[float]] | None = None
                            ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    nperseg = int(round(SPECTRAL_WINDOW_SEC * float(fs)))
    hop = int(round(SPECTRAL_HOP_SEC * float(fs)))
    if nperseg > signal.shape[1]:
        raise ValueError("raw extraction shorter than one spectral window")
    freqs, times, sxx = spectrogram(
        np.asarray(signal, float), fs=float(fs), nperseg=nperseg,
        noverlap=max(0, nperseg - hop), scaling="density", mode="psd", axis=-1,
    )
    if sxx.ndim == 2:
        sxx = sxx[np.newaxis, ...]
    output = {}
    for key in band_keys:
        if band_hz_override is not None and key in band_hz_override:
            lo, hi = map(float, band_hz_override[key])
        else:
            lo, hi = BANDS[key]["hz"]
        if hi >= fs / 2.0:
            raise ValueError(f"Nyquist {fs / 2.0:g} <= requested {key} upper edge {hi:g}")
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            raise ValueError(f"no FFT bins in {key} {lo:g}-{hi:g} Hz")
        output[key] = np.log(np.maximum(sxx[:, mask, :].sum(axis=1), 1e-30))
    return output, np.asarray(times, float)


def _event_extract_bounds(event: Mapping[str, object]) -> tuple[float, float]:
    # The raw extractor is EEG-onset referenced.  Epilepsiae cache zero is
    # clinical onset, so the cache-relative T_spectral value is invalid here.
    t_ref_eeg = tspectral_reference_for_raw_eeg(event)
    raw_min = min(t_ref_eeg + ANALYSIS_RANGE_SEC[0] - 2.0,
                  DISTAL_BASELINE_EEG_SEC[0] - 2.0)
    raw_max = max(t_ref_eeg + ANALYSIS_RANGE_SEC[1] + 2.0,
                  DISTAL_BASELINE_EEG_SEC[1] + 2.0)
    return max(1.0, -raw_min), max(1.0, raw_max)


def _window_blocks(delta: np.ndarray, legacy: np.ndarray, rel_t: np.ndarray,
                   baseline_interval: tuple[float, float]):
    blocks, delta_rows, legacy_rows = [], [], []
    cursor = 0
    for scale in WINDOW_SCALES:
        for region, bounds in (("trajectory", ANALYSIS_RANGE_SEC),
                               ("distal_baseline", baseline_interval)):
            grid = make_complete_window_grid(bounds[0], bounds[1], scale.width_sec, scale.step_sec)
            d, complete = aggregate_complete_windows(
                delta, rel_t, grid, spectral_window_sec=SPECTRAL_WINDOW_SEC
            )
            z, complete_z = aggregate_complete_windows(
                legacy, rel_t, grid, spectral_window_sec=SPECTRAL_WINDOW_SEC
            )
            keep = complete & complete_z
            grid, d, z = grid[keep], d[keep], z[keep]
            sl = slice(cursor, cursor + len(grid))
            cursor += len(grid)
            blocks.append({"scale": scale.name, "region": region, "grid": grid, "slice": sl})
            delta_rows.append(d); legacy_rows.append(z)
    fixed_names = list(FIXED_WINDOWS) + ["distal"]
    fixed_bounds = list(FIXED_WINDOWS.values()) + [baseline_interval]
    fixed_grid = np.array([[lo, hi, (lo + hi) / 2.0] for lo, hi in fixed_bounds], float)
    d, complete = aggregate_complete_windows(delta, rel_t, fixed_grid,
                                               spectral_window_sec=SPECTRAL_WINDOW_SEC)
    z, complete_z = aggregate_complete_windows(legacy, rel_t, fixed_grid,
                                                 spectral_window_sec=SPECTRAL_WINDOW_SEC)
    keep = complete & complete_z
    sl = slice(cursor, cursor + int(np.sum(keep)))
    blocks.append({"scale": "fixed", "region": "fixed", "grid": fixed_grid[keep],
                   "fixed_names": [v for v, ok in zip(fixed_names, keep) if ok], "slice": sl})
    delta_rows.append(d[keep]); legacy_rows.append(z[keep])
    return blocks, np.vstack(delta_rows), np.vstack(legacy_rows)


def _score_rows(subject: str, dataset: str, seizure_idx: int, phenotype: str,
                band_key: str, band_meta: Mapping[str, object], blocks, values: np.ndarray,
                legacy_values: np.ndarray, scorers, nulls, align, quality, event_meta,
                baseline_audit: Mapping[str, object]) -> tuple[list[dict], dict, dict, dict]:
    observed = [score_observed_bundle(scorers, row) for row in values]
    legacy = [score_observed_bundle(scorers, row) for row in legacy_values]
    null_summary = {}
    for mode, scored in nulls.items():
        null_summary[mode] = {}
        for plane in ("own", "shared"):
            key = f"{plane}_maxab"
            if key in scored:
                arr = np.asarray(scored[key], float)
                null_summary[mode][plane] = {
                    "median": np.nanmedian(arr, axis=0),
                    "p2p5": np.nanpercentile(arr, 2.5, axis=0),
                    "p97p5": np.nanpercentile(arr, 97.5, axis=0),
                }
    rows, block_lookup = [], {}
    for block in blocks:
        block_rows = []
        grid = block["grid"]
        fixed_names = block.get("fixed_names", [None] * len(grid))
        for local, ((lo, hi, center), fixed_name) in enumerate(zip(grid, fixed_names)):
            global_idx = block["slice"].start + local
            obs, old = observed[global_idx], legacy[global_idx]
            n_finite = int(np.isfinite(values[global_idx]).sum())
            row = {
                "dataset": dataset, "subject": subject, "seizure_idx": int(seizure_idx),
                "phenotype": phenotype, "band": band_key, "band_label": band_meta["label"],
                "band_role": band_meta["role"], "label_version": PHENOTYPE_LABEL_VERSION,
                "t_spectral_source": event_meta.get("source"),
                "t_spectral_rel_cache_zero_sec": event_meta.get("t_spectral_rel_cache_zero_sec"),
                "t_spectral_rel_eeg_sec": event_meta.get("t_spectral_rel_eeg_sec"),
                "cache_zero_reference": event_meta.get("cache_zero_reference"),
                "clinical_onset_rel_tspectral_sec": event_meta.get(
                    "clinical_onset_rel_tspectral_sec") if dataset != "yuquan" else None,
                "eeg_onset_rel_tspectral_sec": event_meta.get("eeg_onset_rel_tspectral_sec"),
                "distal_baseline_start_rel_tspectral_sec": baseline_audit[
                    "interval_rel_tspectral_sec"][0],
                "distal_baseline_end_rel_tspectral_sec": baseline_audit[
                    "interval_rel_tspectral_sec"][1],
                "window_scale": block["scale"], "window_region": block["region"],
                "fixed_window": fixed_name, "window_start_sec": float(lo),
                "window_end_sec": float(hi), "window_center_sec": float(center),
                "n_target_contacts": int(align["n_target"]), "n_matched": int(align["n_matched"]),
                "n_finite": n_finite, "missing_contacts": ";".join(align["missing_names"]),
                "spatial_median_delta_energy": float(np.nanmedian(values[global_idx])),
                "spatial_mean_delta_energy": float(np.nanmean(values[global_idx])),
                "distal_baseline_z_center_max_abs": float(baseline_audit["max_abs_center"]),
                **_quality_values(quality),
            }
            for prefix in ("own_a", "own_b", "shared_a", "shared_b"):
                row[f"{prefix}_signed"] = obs.get(f"{prefix}_signed")
                row[f"{prefix}_abs"] = obs.get(f"{prefix}_abs")
                row[f"{prefix}_mirror_choice"] = obs.get(f"{prefix}_mirror_choice")
                row[f"{prefix}_r_identity"] = obs.get(f"{prefix}_r_identity")
                row[f"{prefix}_r_mirror"] = obs.get(f"{prefix}_r_mirror")
            for plane in ("own", "shared"):
                row[f"{plane}_maxab"] = obs.get(f"{plane}_maxab")
                row[f"{plane}_best_template"] = obs.get(f"{plane}_best_template")
                row[f"legacy_{plane}_maxab"] = old.get(f"{plane}_maxab")
                for mode in ("all_contact", "within_shaft"):
                    summary = null_summary.get(mode, {}).get(plane)
                    for label in ("median", "p2p5", "p97p5"):
                        row[f"{plane}_{mode}_null_{label}"] = (
                            float(summary[label][global_idx]) if summary is not None else np.nan
                        )
                if row.get(f"{plane}_maxab") is not None:
                    row[f"{plane}_M"] = float(row[f"{plane}_maxab"] -
                                               row[f"{plane}_within_shaft_null_median"])
                else:
                    row[f"{plane}_M"] = np.nan
            rows.append(row); block_rows.append(row)
        block_lookup[(block["scale"], block["region"])] = block_rows

    # D(t) is defined event-first from the matching-scale distal baseline M.
    for scale in [s.name for s in WINDOW_SCALES]:
        baseline_rows = block_lookup.get((scale, "distal_baseline"), [])
        for plane in ("own", "shared"):
            base = [r[f"{plane}_M"] for r in baseline_rows if np.isfinite(r.get(f"{plane}_M", np.nan))]
            base_med = float(np.median(base)) if base else np.nan
            for region in ("trajectory", "distal_baseline"):
                for row in block_lookup.get((scale, region), []):
                    row[f"{plane}_D"] = (float(row[f"{plane}_M"] - base_med)
                                          if np.isfinite(row.get(f"{plane}_M", np.nan))
                                          and np.isfinite(base_med) else np.nan)
                    row[f"{plane}_M_distal_scale_median"] = base_med

    fixed_rows = block_lookup.get(("fixed", "fixed"), [])
    fixed_by_name = {r["fixed_window"]: r for r in fixed_rows}
    for plane in ("own", "shared"):
        base = fixed_by_name.get("distal", {}).get(f"{plane}_M", np.nan)
        for row in fixed_rows:
            row[f"{plane}_D"] = (float(row[f"{plane}_M"] - base)
                                  if np.isfinite(row.get(f"{plane}_M", np.nan))
                                  and np.isfinite(base) else np.nan)
            row[f"{plane}_M_distal_scale_median"] = base
    metrics = _fixed_metrics(fixed_by_name)

    trajectory_null = {}
    for block in blocks:
        if block["region"] != "trajectory":
            continue
        trajectory_null[block["scale"]] = {}
        for mode, scored in nulls.items():
            trajectory_null[block["scale"]][mode] = {}
            for plane in ("own", "shared"):
                key = f"{plane}_maxab"
                if key in scored:
                    trajectory_null[block["scale"]][mode][key] = np.asarray(
                        scored[key][:, block["slice"]], np.float32
                    )
    fixed_block = next(block for block in blocks
                       if block["scale"] == "fixed" and block["region"] == "fixed")
    fixed_null = {"window_names": list(fixed_block["fixed_names"])}
    for mode, scored in nulls.items():
        fixed_null[mode] = {}
        for plane in ("own", "shared"):
            key = f"{plane}_maxab"
            if key in scored:
                fixed_null[mode][key] = np.asarray(
                    scored[key][:, fixed_block["slice"]], np.float32
                )
    return rows, metrics, trajectory_null, fixed_null


def _fixed_metrics(fixed: Mapping[str, Mapping[str, object]]) -> dict:
    out = {}
    aliases = {"post10": "M_post10", "post20": "M_post20", "pre10": "M_pre10",
               "pre20": "M_pre20", "distal": "M_distal", "late20_30": "M_late20_30"}
    for plane in ("own", "shared"):
        for name, alias in aliases.items():
            out[f"{plane}_{alias}"] = fixed.get(name, {}).get(f"{plane}_M", np.nan)
        for suffix, a, b in (
            ("M_post10_minus_pre10", "post10", "pre10"),
            ("M_post20_minus_pre20", "post20", "pre20"),
            ("M_post10_minus_distal", "post10", "distal"),
            ("M_post20_minus_distal", "post20", "distal"),
        ):
            va = fixed.get(a, {}).get(f"{plane}_M", np.nan)
            vb = fixed.get(b, {}).get(f"{plane}_M", np.nan)
            out[f"{plane}_{suffix}"] = float(va - vb) if np.isfinite(va) and np.isfinite(vb) else np.nan
    return out


def _process_event(subject: str, dataset: str, seizure_idx: int, phenotype: str,
                   subject_meta: Mapping[str, object], field_record: Mapping[str, object],
                   scorers, quality, n_perm: int, seed: int, *,
                   band_keys_override: Sequence[str] | None = None):
    event = (subject_meta.get("seizure") or {}).get(str(seizure_idx), {})
    band_keys = (list(band_keys_override) if band_keys_override is not None else
                 [key for key, meta in BANDS.items() if meta["phenotype"] == phenotype])
    unknown = sorted(set(band_keys) - set(BANDS))
    if unknown:
        raise ValueError(f"unknown explicit band keys: {unknown}")
    if not band_keys:
        raise ValueError(f"no readout band selected for phenotype={phenotype}")
    pre_sec, post_sec = _event_extract_bounds(event)
    sw = extract_seizure_window(
        f"{dataset}/{subject.split('_', 1)[1]}", seizure_idx,
        pre_sec=pre_sec, post_sec=post_sec, reference=ICTAL_REFERENCE[dataset],
    )
    target_names = [str(v) for v in field_record["interictal_field"]["contact_order"]]
    raw_names = [recruit.bipolar_alias_label(v) for v in sw.ch_names]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("raw channel aliases are not unique")
    raw_index = {name: i for i, name in enumerate(raw_names)}
    matched_names = [name for name in target_names if name in raw_index]
    if len(matched_names) < MIN_CONTACTS:
        raise ValueError(f"fewer_than_6_exact_name_contacts:{len(matched_names)}")
    signal = sw.signal[[raw_index[name] for name in matched_names]]
    if dataset == "yuquan":
        signal = _notch_yuquan(signal, sw.fs)
    powers, times_from_crop = _extract_log_band_power(signal, sw.fs, band_keys)
    rel_reference = np.asarray(times_from_crop, float) - float(sw.pre_sec)
    t_spectral_ref_eeg = tspectral_reference_for_raw_eeg(event)
    t_spectral_ref_cache = float(event["t_spectral_rel_cache_zero_sec"])
    rel_tspectral = tspectral_zeroed_times(rel_reference, t_spectral_ref_eeg)
    provenance = annotation_provenance(dataset, event)
    eeg_rel_tspectral = provenance["eeg_onset_rel_tspectral_sec"]
    if eeg_rel_tspectral is None:
        raise ValueError("missing_eeg_onset_for_distal_baseline")
    baseline_interval = (eeg_rel_tspectral + DISTAL_BASELINE_EEG_SEC[0],
                         eeg_rel_tspectral + DISTAL_BASELINE_EEG_SEC[1])
    target_match_mask = np.array([name in raw_index for name in target_names], bool)
    event_seed = _seed(f"{subject}:{seizure_idx}", seed)
    permutations = {
        "all_contact": make_contact_permutations(target_names, target_match_mask, n_perm,
                                                  event_seed, mode="all_contact"),
        "within_shaft": make_contact_permutations(target_names, target_match_mask, n_perm,
                                                   event_seed + 1, mode="within_shaft"),
    }
    output = {}
    for band_key in band_keys:
        robust = distal_baseline_robust_z(
            powers[band_key], rel_tspectral, baseline_interval, min_frames=MIN_BASELINE_FRAMES
        )
        aligned_delta = exact_name_align_matrix(field_record, matched_names, robust["delta"])
        aligned_legacy = exact_name_align_matrix(field_record, matched_names, robust["legacy_z"])
        n_finite = int(np.isfinite(aligned_delta["values"]).any(axis=1).sum())
        reason = eligibility_drop_reason(
            band_available=True, field_status=quality["field_status"], fingerprint_ok=True,
            n_finite_contacts=n_finite,
        )
        if reason:
            raise ValueError(reason)
        blocks, values, legacy_values = _window_blocks(
            aligned_delta["values"], aligned_legacy["values"], rel_tspectral, baseline_interval
        )
        min_window_finite = int(np.min(np.isfinite(values).sum(axis=1)))
        if min_window_finite < MIN_CONTACTS:
            raise ValueError(f"fewer_than_6_finite_contacts_in_window:{min_window_finite}")
        nulls = {
            mode: score_permutation_matrix(scorers, values, perm, chunk_draws=100)
            for mode, perm in permutations.items()
        }
        baseline_centers = np.asarray(robust["baseline_z_center"], float)
        baseline_audit = {
            "n_frames": robust["n_baseline_frames"],
            "max_abs_center": float(np.nanmax(np.abs(baseline_centers))),
            "median_abs_center": float(np.nanmedian(np.abs(baseline_centers))),
            "interval_rel_tspectral_sec": list(map(float, baseline_interval)),
            "interval_rel_cache_reference_sec": [
                float(baseline_interval[0] + t_spectral_ref_cache),
                float(baseline_interval[1] + t_spectral_ref_cache),
            ],
        }
        rows, metrics, trajectory_null, fixed_null = _score_rows(
            subject, dataset, seizure_idx, phenotype, band_key, BANDS[band_key], blocks,
            values, legacy_values, scorers, nulls, aligned_delta, quality, event, baseline_audit,
        )
        fixed_block = next(block for block in blocks
                           if block["scale"] == "fixed" and block["region"] == "fixed")
        fixed_activation = {
            name: np.asarray(values[fixed_block["slice"].start + j], np.float32)
            for j, name in enumerate(fixed_block["fixed_names"])
        }
        output[band_key] = {
            "rows": rows, "fixed_metrics": metrics, "trajectory_null": trajectory_null,
            "fixed_null": fixed_null, "fixed_activation": fixed_activation,
            "baseline_audit": baseline_audit,
            "n_target": aligned_delta["n_target"], "n_matched": aligned_delta["n_matched"],
            "n_finite": n_finite, "missing_names": aligned_delta["missing_names"],
            "seizure_id": sw.seizure_id, "fs": float(sw.fs),
            "permutation_seed_all_contact": event_seed,
            "permutation_seed_within_shaft": event_seed + 1,
            "t_spectral_provenance": {**provenance,
                                       "t_spectral_rel_cache_zero_sec": t_spectral_ref_cache,
                                       "t_spectral_rel_eeg_sec": t_spectral_ref_eeg,
                                       "source": event.get("source")},
        }
    return output


def _subject_timecourse(subject: str, dataset: str, quality: Mapping[str, object],
                        event_results: list[dict]) -> list[dict]:
    rows = []
    grouped = defaultdict(list)
    for event in event_results:
        for band_key, result in event["bands"].items():
            grouped[(event["phenotype"], band_key)].append((event["seizure_idx"], result))
    for (phenotype, band_key), events in grouped.items():
        for scale in [s.name for s in WINDOW_SCALES]:
            event_rows = []
            for seizure_idx, result in events:
                picked = [r for r in result["rows"] if r["window_scale"] == scale
                          and r["window_region"] == "trajectory"]
                if picked:
                    event_rows.append((seizure_idx, picked, result))
            if not event_rows:
                continue
            centers = np.array([r["window_center_sec"] for r in event_rows[0][1]], float)
            if any(not np.allclose(centers, [r["window_center_sec"] for r in picked])
                   for _, picked, _ in event_rows[1:]):
                raise ValueError(f"{subject} {band_key} {scale}: trajectory grids drifted")
            folded_null = {}
            for mode in ("all_contact", "within_shaft"):
                for plane in ("own", "shared"):
                    arrays = []
                    for _, _picked, result in event_rows:
                        arr = (result["trajectory_null"].get(scale, {}).get(mode, {})
                               .get(f"{plane}_maxab"))
                        if arr is not None:
                            arrays.append(arr)
                    if len(arrays) == len(event_rows) and arrays:
                        folded_null[(mode, plane)] = fold_seizure_null_draws(arrays)
            for j, center in enumerate(centers):
                values = lambda key: np.asarray([picked[j].get(key, np.nan)
                                                 for _, picked, _ in event_rows], float)
                row = {
                    "dataset": dataset, "subject": subject, "phenotype": phenotype,
                    "band": band_key, "band_label": BANDS[band_key]["label"],
                    "band_role": BANDS[band_key]["role"], "window_scale": scale,
                    "window_start_sec": event_rows[0][1][j]["window_start_sec"],
                    "window_end_sec": event_rows[0][1][j]["window_end_sec"],
                    "window_center_sec": float(center), "n_seizures": len(event_rows),
                    "spatial_median_delta_energy": float(np.nanmedian(values("spatial_median_delta_energy"))),
                    "own_maxab": float(np.nanmedian(values("own_maxab"))),
                    "own_M": float(np.nanmedian(values("own_M"))),
                    "own_D": float(np.nanmedian(values("own_D"))),
                    "shared_maxab": float(np.nanmedian(values("shared_maxab")))
                    if np.isfinite(values("shared_maxab")).any() else np.nan,
                    "shared_M": float(np.nanmedian(values("shared_M")))
                    if np.isfinite(values("shared_M")).any() else np.nan,
                    "shared_D": float(np.nanmedian(values("shared_D")))
                    if np.isfinite(values("shared_D")).any() else np.nan,
                    **_quality_values(quality),
                }
                for (mode, plane), null in folded_null.items():
                    row[f"{plane}_{mode}_null_median_folded"] = float(np.nanmedian(null[:, j]))
                    row[f"{plane}_{mode}_null_p2p5_folded"] = float(np.nanpercentile(null[:, j], 2.5))
                    row[f"{plane}_{mode}_null_p97p5_folded"] = float(np.nanpercentile(null[:, j], 97.5))
                rows.append(row)
    return rows


def _fixed_subject_rows(subject: str, dataset: str, quality: Mapping[str, object],
                        event_results: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for event in event_results:
        for band_key, result in event["bands"].items():
            grouped[(event["phenotype"], band_key)].append(result["fixed_metrics"])
    out = []
    for (phenotype, band_key), metrics in grouped.items():
        keys = sorted({key for row in metrics for key in row})
        rec = {"dataset": dataset, "subject": subject, "phenotype": phenotype,
               "band": band_key, "band_label": BANDS[band_key]["label"],
               "band_role": BANDS[band_key]["role"], "n_seizures": len(metrics),
               **_quality_values(quality)}
        for key in keys:
            values = np.asarray([row.get(key, np.nan) for row in metrics], float)
            rec[key] = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
        out.append(rec)
    return out


def _fixed_field_subject_rows(subject: str, dataset: str, quality: Mapping[str, object],
                              event_results: list[dict]) -> tuple[list[dict], dict]:
    """Fold fixed-window observed/null fields seizure-first inside one subject.

    The draw index is preserved while taking the seizure median, so each reported
    subject null quantile comes from the same subject-first randomization contract
    as the time-course null.  Q95 is the one-sided reference used by the accepted
    field-concordance atlas; p values remain continuous Monte Carlo quantities.
    """
    grouped = defaultdict(list)
    for event in event_results:
        for band_key, result in event["bands"].items():
            grouped[(event["phenotype"], band_key)].append((event["seizure_idx"], result))

    rows, display = [], {}
    for (phenotype, band_key), events in grouped.items():
        window_names = list(events[0][1]["fixed_null"]["window_names"])
        if any(list(result["fixed_null"]["window_names"]) != window_names
               for _, result in events[1:]):
            raise ValueError(f"{subject} {band_key}: fixed-window names drifted")
        event_fixed = []
        for seizure_idx, result in events:
            lookup = {row["fixed_window"]: row for row in result["rows"]
                      if row["window_scale"] == "fixed" and row["window_region"] == "fixed"}
            if set(lookup) != set(window_names):
                raise ValueError(f"{subject} {band_key} seizure {seizure_idx}: fixed rows incomplete")
            event_fixed.append((seizure_idx, lookup, result))

        display[band_key] = {
            "phenotype": phenotype,
            "windows": {
                name: np.nanmedian(np.stack(
                    [np.asarray(result["fixed_activation"][name], float)
                     for _, _, result in event_fixed], axis=0), axis=0)
                for name in window_names
            },
        }

        folded_null = {}
        for mode in ("all_contact", "within_shaft"):
            for plane in ("own", "shared"):
                key = f"{plane}_maxab"
                arrays = [result["fixed_null"].get(mode, {}).get(key)
                          for _, _, result in event_fixed]
                if arrays and all(array is not None for array in arrays):
                    folded_null[(mode, plane)] = fold_seizure_null_draws(arrays)

        for j, window_name in enumerate(window_names):
            event_rows = [lookup[window_name] for _, lookup, _ in event_fixed]

            def _median(key: str) -> float:
                vals = np.asarray([row.get(key, np.nan) for row in event_rows], float)
                return float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan

            rec = {
                "dataset": dataset, "subject": subject, "phenotype": phenotype,
                "band": band_key, "band_label": BANDS[band_key]["label"],
                "band_role": BANDS[band_key]["role"], "field_plane_contract": "own_narrow",
                "fixed_window": window_name,
                "window_start_sec": float(np.median(
                    [row["window_start_sec"] for row in event_rows])),
                "window_end_sec": float(np.median(
                    [row["window_end_sec"] for row in event_rows])),
                "window_center_sec": float(np.median(
                    [row["window_center_sec"] for row in event_rows])),
                "n_seizures": len(event_rows),
                "seizure_idxs": ";".join(str(index) for index, _, _ in event_fixed),
                "n_target_contacts": int(min(row["n_target_contacts"] for row in event_rows)),
                "n_matched": int(min(row["n_matched"] for row in event_rows)),
                "n_finite": int(min(row["n_finite"] for row in event_rows)),
                "spatial_median_delta_energy": _median("spatial_median_delta_energy"),
                **_quality_values(quality),
            }
            for plane in ("own", "shared"):
                rec[f"{plane}_a_abs"] = _median(f"{plane}_a_abs")
                rec[f"{plane}_b_abs"] = _median(f"{plane}_b_abs")
                rec[f"{plane}_a_signed"] = _median(f"{plane}_a_signed")
                rec[f"{plane}_b_signed"] = _median(f"{plane}_b_signed")
                observed = _median(f"{plane}_maxab")
                rec[f"{plane}_maxab"] = observed
                a_abs, b_abs = rec[f"{plane}_a_abs"], rec[f"{plane}_b_abs"]
                if np.isfinite(a_abs) or np.isfinite(b_abs):
                    rec[f"{plane}_best_template_subject"] = (
                        "A" if np.nan_to_num(a_abs, nan=-np.inf) >= np.nan_to_num(b_abs, nan=-np.inf)
                        else "B"
                    )
                else:
                    rec[f"{plane}_best_template_subject"] = None
                for mode in ("all_contact", "within_shaft"):
                    null = folded_null.get((mode, plane))
                    if null is None or not np.isfinite(observed):
                        continue
                    draw = np.asarray(null[:, j], float)
                    draw = draw[np.isfinite(draw)]
                    if not len(draw):
                        continue
                    median = float(np.median(draw))
                    p95 = float(np.percentile(draw, 95))
                    rec[f"{plane}_{mode}_null_median_folded"] = median
                    rec[f"{plane}_{mode}_null_p2p5_folded"] = float(np.percentile(draw, 2.5))
                    rec[f"{plane}_{mode}_null_p95_folded"] = p95
                    rec[f"{plane}_{mode}_null_p97p5_folded"] = float(np.percentile(draw, 97.5))
                    rec[f"{plane}_{mode}_delta_null_median"] = float(observed - median)
                    rec[f"{plane}_{mode}_margin_to_p95"] = float(observed - p95)
                    rec[f"{plane}_{mode}_empirical_p_one_sided"] = float(
                        (1 + np.sum(draw >= observed)) / (len(draw) + 1)
                    )
                    rec[f"{plane}_{mode}_exceeds_p95"] = bool(observed > p95)
            rows.append(rec)
    return rows, display


def _phenotype_matched_fixed_subject_rows(
        subject: str, dataset: str, quality: Mapping[str, object],
        event_results: list[dict], *, readout_family: str) -> list[dict]:
    """Fold one phenotype-selected primary value per seizure inside a subject.

    Broadband seizures contribute 1--150 Hz.  Gamma-nonbroadband seizures
    contribute HFA 60--100 Hz in the primary family; the 30--80 Hz family is a
    complete substitution sensitivity and never creates a second observation
    for the same gamma seizure.
    """
    if readout_family not in PHENOTYPE_MATCHED_READOUTS:
        raise ValueError(f"unknown phenotype-matched readout family: {readout_family}")
    band_map = PHENOTYPE_MATCHED_READOUTS[readout_family]
    selected = []
    for event in event_results:
        phenotype = str(event["phenotype"])
        if phenotype not in band_map:
            continue
        band_key = band_map[phenotype]
        if band_key not in event["bands"]:
            raise ValueError(
                f"{subject} seizure {event['seizure_idx']}: missing selected band {band_key}"
            )
        selected.append((int(event["seizure_idx"]), phenotype, band_key,
                         event["bands"][band_key]))
    if not selected:
        return []
    window_names = list(selected[0][3]["fixed_null"]["window_names"])
    if set(window_names) != set(FIXED_WINDOW_ORDER):
        raise ValueError(f"{subject}: pooled fixed-window contract drifted: {window_names}")
    if any(list(result["fixed_null"]["window_names"]) != window_names
           for _, _, _, result in selected[1:]):
        raise ValueError(f"{subject}: pooled fixed-window names differ across seizures")
    event_fixed = []
    for seizure_idx, phenotype, band_key, result in selected:
        lookup = {row["fixed_window"]: row for row in result["rows"]
                  if row["window_scale"] == "fixed" and row["window_region"] == "fixed"}
        if set(lookup) != set(window_names):
            raise ValueError(f"{subject} seizure {seizure_idx}: pooled fixed rows incomplete")
        event_fixed.append((seizure_idx, phenotype, band_key, lookup, result))

    folded_null = {}
    for mode in ("all_contact", "within_shaft"):
        for plane in ("own", "shared"):
            key = f"{plane}_maxab"
            arrays = [result["fixed_null"].get(mode, {}).get(key)
                      for _, _, _, _, result in event_fixed]
            if arrays and all(array is not None for array in arrays):
                folded_null[(mode, plane)] = fold_seizure_null_draws(arrays)

    rows = []
    for window_name in FIXED_WINDOW_ORDER:
        j = window_names.index(window_name)
        event_rows = [lookup[window_name] for _, _, _, lookup, _ in event_fixed]

        def _median(key: str) -> float:
            values = np.asarray([row.get(key, np.nan) for row in event_rows], float)
            return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan

        n_broadband = sum(phenotype == "broadband_1_150"
                          for _, phenotype, _, _, _ in event_fixed)
        n_gamma = sum(phenotype == "gamma_nonbroadband"
                      for _, phenotype, _, _, _ in event_fixed)
        rec = {
            "dataset": dataset, "subject": subject,
            "readout_family": readout_family,
            "readout_contract": (
                "strict broadband -> 1-150 Hz; gamma non-broadband -> 60-100 Hz"
                if readout_family == "primary" else
                "strict broadband -> 1-150 Hz; gamma non-broadband -> 30-80 Hz sensitivity"
            ),
            "field_plane_contract": "own_narrow",
            "fixed_window": window_name,
            "window_start_sec": float(np.median(
                [row["window_start_sec"] for row in event_rows])),
            "window_end_sec": float(np.median(
                [row["window_end_sec"] for row in event_rows])),
            "window_center_sec": float(np.median(
                [row["window_center_sec"] for row in event_rows])),
            "n_seizures": len(event_fixed),
            "n_broadband_seizures": int(n_broadband),
            "n_gamma_seizures": int(n_gamma),
            "seizure_idxs": ";".join(str(v[0]) for v in event_fixed),
            "selected_bands": ";".join(v[2] for v in event_fixed),
            "n_target_contacts": int(min(row["n_target_contacts"] for row in event_rows)),
            "n_matched": int(min(row["n_matched"] for row in event_rows)),
            "n_finite": int(min(row["n_finite"] for row in event_rows)),
            "spatial_median_delta_energy": _median("spatial_median_delta_energy"),
            **_quality_values(quality),
        }
        for plane in ("own", "shared"):
            rec[f"{plane}_a_abs"] = _median(f"{plane}_a_abs")
            rec[f"{plane}_b_abs"] = _median(f"{plane}_b_abs")
            rec[f"{plane}_a_signed"] = _median(f"{plane}_a_signed")
            rec[f"{plane}_b_signed"] = _median(f"{plane}_b_signed")
            observed = _median(f"{plane}_maxab")
            rec[f"{plane}_maxab"] = observed
            for mode in ("all_contact", "within_shaft"):
                null = folded_null.get((mode, plane))
                if null is None or not np.isfinite(observed):
                    continue
                draw = np.asarray(null[:, j], float)
                draw = draw[np.isfinite(draw)]
                if not len(draw):
                    continue
                median = float(np.median(draw))
                p95 = float(np.percentile(draw, 95))
                rec[f"{plane}_{mode}_null_median_folded"] = median
                rec[f"{plane}_{mode}_null_p2p5_folded"] = float(np.percentile(draw, 2.5))
                rec[f"{plane}_{mode}_null_p95_folded"] = p95
                rec[f"{plane}_{mode}_null_p97p5_folded"] = float(np.percentile(draw, 97.5))
                rec[f"{plane}_{mode}_delta_null_median"] = float(observed - median)
                rec[f"{plane}_{mode}_margin_to_p95"] = float(observed - p95)
                rec[f"{plane}_{mode}_empirical_p_one_sided"] = float(
                    (1 + np.sum(draw >= observed)) / (len(draw) + 1)
                )
                rec[f"{plane}_{mode}_exceeds_p95"] = bool(observed > p95)
        rows.append(rec)
    return rows


def _delta_window_pivot(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    pivot = frame.pivot(index="subject", columns="fixed_window", values=value_column)
    pivot = pivot.reindex(columns=FIXED_WINDOW_ORDER)
    return pivot.dropna(axis=0, how="any")


def _phenotype_matched_cohort_statistics(subject_rows: pd.DataFrame, n_boot: int,
                                         n_perm: int, seed: int) -> pd.DataFrame:
    """Combined/Epilepsiae/Yuquan exploratory subject-vs-null statistics."""
    rows = []
    strata = (
        ("combined", subject_rows, "exploratory_subject_cohort"),
        ("epilepsiae", subject_rows[subject_rows.dataset == "epilepsiae"],
         "exploratory_dataset_stratum"),
        ("yuquan", subject_rows[subject_rows.dataset == "yuquan"],
         "exploratory_dataset_stratum"),
    )
    for stratum, frame, role in strata:
        if frame.empty:
            continue
        for plane in ("own", "shared"):
            for mode in ("within_shaft", "all_contact"):
                source = f"{plane}_{mode}_delta_null_median"
                if source not in frame:
                    continue
                pivot = _delta_window_pivot(frame, source)
                if pivot.empty:
                    continue
                inference = fixed_window_sign_flip_maxt(
                    pivot.to_numpy(float), n_perm=n_perm,
                    seed=_seed(f"pooled:{stratum}:{plane}:{mode}", seed),
                )
                by_subject = frame.drop_duplicates("subject").set_index("subject")
                for j, window in enumerate(FIXED_WINDOW_ORDER):
                    values = pivot[window].to_numpy(float)
                    lo, hi = bootstrap_median_ci(
                        values, n_boot=n_boot,
                        seed=_seed(f"pooled-ci:{stratum}:{plane}:{mode}:{window}", seed),
                    )
                    meta = frame[frame.fixed_window == window].set_index("subject").loc[pivot.index]
                    rows.append({
                        "dataset_stratum": stratum, "inference_role": role,
                        "readout_family": str(frame.readout_family.iloc[0]),
                        "field_plane": plane, "null_type": mode,
                        "fixed_window": window,
                        "window_start_sec": float(np.median(meta.window_start_sec)),
                        "window_end_sec": float(np.median(meta.window_end_sec)),
                        "window_center_sec": float(np.median(meta.window_center_sec)),
                        "n_subjects": int(len(values)),
                        "n_seizures": int(by_subject.loc[pivot.index, "n_seizures"].sum()),
                        "n_broadband_seizures": int(
                            by_subject.loc[pivot.index, "n_broadband_seizures"].sum()),
                        "n_gamma_seizures": int(
                            by_subject.loc[pivot.index, "n_gamma_seizures"].sum()),
                        "tested_statistic": "mean_subject_delta",
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q25": float(np.percentile(values, 25)),
                        "q75": float(np.percentile(values, 75)),
                        "bootstrap_median_ci_low": lo,
                        "bootstrap_median_ci_high": hi,
                        "two_sided_sign_flip_raw_p": float(inference["raw_p"][j]),
                        "two_sided_sign_flip_maxt_p": float(inference["maxt_p"][j]),
                        "n_sign_permutations": int(inference["n_permutations"]),
                    })
    return pd.DataFrame(rows)


def _phenotype_matched_relation_statistics(subject_rows: pd.DataFrame, n_perm: int,
                                           seed: int) -> pd.DataFrame:
    """Pre-existing A/B relation heterogeneity, retaining ``same`` separately."""
    rows = []
    scopes = (
        ("geometry_2d_supported", subject_rows[
            subject_rows.geometry_2d_supported.fillna(False).astype(bool)]),
        ("strict_2d_sensitivity", subject_rows[subject_rows.axis_quality_tier == "strict_2d"]),
    )
    for scope, scoped in scopes:
        for stratum, frame in (
            ("combined", scoped),
            ("epilepsiae", scoped[scoped.dataset == "epilepsiae"]),
            ("yuquan", scoped[scoped.dataset == "yuquan"]),
        ):
            if frame.empty:
                continue
            pivot = _delta_window_pivot(frame, "own_within_shaft_delta_null_median")
            if pivot.empty:
                continue
            metadata = frame.drop_duplicates("subject").set_index("subject").loc[pivot.index]
            labels = metadata.axis_relation.astype(str).to_numpy(object)
            inference = independent_label_permutation_maxt(
                pivot.to_numpy(float), labels, "reversed", "different",
                n_perm=n_perm, seed=_seed(f"relation:{scope}:{stratum}", seed),
            )
            for j, window in enumerate(FIXED_WINDOW_ORDER):
                rec = {
                    "quality_scope": scope, "dataset_stratum": stratum,
                    "fixed_window": window,
                    "comparison": "reversed_vs_different",
                    "tested_statistic": "mean_delta_reversed_minus_different",
                    "n_reversed": int(np.sum(labels == "reversed")),
                    "n_different": int(np.sum(labels == "different")),
                    "n_same": int(np.sum(labels == "same")),
                    "mean_difference_reversed_minus_different": float(
                        inference["observed_mean_difference"][j]),
                    "two_sided_label_permutation_raw_p": float(inference["raw_p"][j]),
                    "two_sided_label_permutation_maxt_p": float(inference["maxt_p"][j]),
                    "n_label_permutations": int(inference["n_permutations"]),
                }
                for relation in ("reversed", "different", "same"):
                    values = pivot.loc[metadata.axis_relation.astype(str) == relation, window].to_numpy(float)
                    rec[f"{relation}_mean"] = float(np.mean(values)) if len(values) else np.nan
                    rec[f"{relation}_median"] = float(np.median(values)) if len(values) else np.nan
                    rec[f"{relation}_q25"] = float(np.percentile(values, 25)) if len(values) else np.nan
                    rec[f"{relation}_q75"] = float(np.percentile(values, 75)) if len(values) else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def _exploratory_band_window_statistics(subject_field: pd.DataFrame, n_perm: int,
                                        seed: int) -> pd.DataFrame:
    """Phenotype-separated own/within-shaft fixed-window exploratory tests."""
    rows = []
    for (phenotype, band, band_label, band_role), band_frame in subject_field.groupby(
            ["phenotype", "band", "band_label", "band_role"], sort=True):
        for stratum, frame in (
            ("combined", band_frame),
            ("epilepsiae", band_frame[band_frame.dataset == "epilepsiae"]),
            ("yuquan", band_frame[band_frame.dataset == "yuquan"]),
        ):
            pivot = _delta_window_pivot(frame, "own_within_shaft_delta_null_median")
            if pivot.empty:
                continue
            inference = fixed_window_sign_flip_maxt(
                pivot.to_numpy(float), n_perm=n_perm,
                seed=_seed(f"band:{phenotype}:{band}:{stratum}", seed),
            )
            by_subject = frame.drop_duplicates("subject").set_index("subject")
            for j, window in enumerate(FIXED_WINDOW_ORDER):
                values = pivot[window].to_numpy(float)
                rows.append({
                    "dataset_stratum": stratum, "inference_role": "exploratory",
                    "phenotype": phenotype, "band": band, "band_label": band_label,
                    "band_role": band_role, "fixed_window": window,
                    "n_subjects": int(len(values)),
                    "n_seizures": int(by_subject.loc[pivot.index, "n_seizures"].sum()),
                    "tested_statistic": "mean_subject_delta",
                    "mean": float(np.mean(values)), "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "two_sided_sign_flip_raw_p": float(inference["raw_p"][j]),
                    "two_sided_sign_flip_maxt_p": float(inference["maxt_p"][j]),
                    "n_sign_permutations": int(inference["n_permutations"]),
                })
    return pd.DataFrame(rows)


def _cohort_fixed_field_statistics(subject_field: pd.DataFrame, n_boot: int,
                                   n_perm: int, seed: int) -> pd.DataFrame:
    """Subject-unit cohort summaries of observed maxAB minus null median."""
    rows = []
    strata = (("epilepsiae", subject_field[subject_field.dataset == "epilepsiae"],
               "subject_cohort"),
              ("yuquan", subject_field[subject_field.dataset == "yuquan"],
               "case_sensitivity"),
              ("combined_descriptive", subject_field, "descriptive_only"))
    # Distal baseline is fixed in the original EEG-onset coordinate and therefore
    # legitimately differs after conversion to each event's T_spectral coordinate.
    # It must remain one pre-specified window family at cohort level, not be split
    # into n=1 groups by its descriptive converted coordinates.
    group_cols = ["phenotype", "band", "band_label", "band_role", "fixed_window"]
    for stratum, frame, role in strata:
        for key, group in frame.groupby(group_cols, dropna=False, sort=True):
            base = {name: value for name, value in zip(group_cols, key)}
            base.update({
                "window_start_sec": float(np.nanmedian(group["window_start_sec"])),
                "window_end_sec": float(np.nanmedian(group["window_end_sec"])),
                "window_center_sec": float(np.nanmedian(group["window_center_sec"])),
            })
            for plane in ("own", "shared"):
                for mode in ("within_shaft", "all_contact"):
                    source = f"{plane}_{mode}_delta_null_median"
                    if source not in group:
                        continue
                    values = pd.to_numeric(group[source], errors="coerce").to_numpy(float)
                    valid = np.isfinite(values)
                    x = values[valid]
                    if not len(x):
                        continue
                    lo, hi = bootstrap_median_ci(
                        x, n_boot=n_boot,
                        seed=_seed(f"field:{stratum}:{plane}:{mode}:{key}", seed),
                    )
                    p = (paired_sign_flip_p(
                        x, n_perm=n_perm,
                        seed=_seed(f"field-sign:{stratum}:{plane}:{mode}:{key}", seed),
                    ) if role == "subject_cohort" else np.nan)
                    rows.append({
                        **base, "dataset_stratum": stratum, "inference_role": role,
                        "field_plane": plane, "null_type": mode,
                        "metric": "maxab_minus_null_median", "n_subjects": int(len(x)),
                        "n_seizures": int(group.loc[valid, "n_seizures"].sum()),
                        "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
                        "q75": float(np.percentile(x, 75)), "bootstrap_ci_low": lo,
                        "bootstrap_ci_high": hi, "two_sided_sign_flip_p": p,
                    })
    return pd.DataFrame(rows)


def _strict_2d_sensitivity_tables(subject_field: pd.DataFrame, n_boot: int,
                                  n_perm: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an explicitly named strict-2D sensitivity without altering primary rows."""
    frame = subject_field.copy()
    if "axis_quality_tier" not in frame:
        field_ready = frame.get("field_ready", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
        geometry = frame.get(
            "geometry_2d_supported", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        strict = frame.get(
            "strict_stability_pass", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        frame["axis_quality_tier"] = np.select(
            [~field_ready, ~geometry, strict],
            ["field_unavailable", "geometry_unsupported", "strict_2d"],
            default="non_strict_2d",
        )
    strict_subject = frame[frame.axis_quality_tier == "strict_2d"].copy()
    strict_subject["quality_scope"] = "strict_2d_sensitivity"
    strict_cohort = _cohort_fixed_field_statistics(strict_subject, n_boot, n_perm, seed)
    if len(strict_cohort):
        strict_cohort["quality_scope"] = "strict_2d_sensitivity"
    return strict_subject, strict_cohort


def _cohort_timecourse(subject_df: pd.DataFrame, n_boot: int, seed: int):
    rows = []
    strata = [("epilepsiae", subject_df[subject_df.dataset == "epilepsiae"]),
              ("yuquan", subject_df[subject_df.dataset == "yuquan"]),
              ("combined_descriptive", subject_df)]
    group_cols = ["phenotype", "band", "band_label", "band_role", "window_scale",
                  "window_start_sec", "window_end_sec", "window_center_sec"]
    for stratum, frame in strata:
        for key, group in frame.groupby(group_cols, dropna=False, sort=True):
            for plane in ("own", "shared"):
                finite = pd.to_numeric(group[f"{plane}_D"], errors="coerce").to_numpy(float)
                valid = np.isfinite(finite)
                values = finite[valid]
                if not len(values):
                    continue
                lo, hi = bootstrap_median_ci(
                    values, n_boot=n_boot, seed=_seed(f"{stratum}:{plane}:{key}", seed)
                )
                rec = {name: value for name, value in zip(group_cols, key)}
                rec.update({
                    "dataset_stratum": stratum, "field_plane": plane,
                    "n_subjects": int(len(values)),
                    "n_seizures": int(group.loc[valid, "n_seizures"].sum()),
                    "median_D": float(np.median(values)),
                    "q25_D": float(np.percentile(values, 25)),
                    "q75_D": float(np.percentile(values, 75)), "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi, "maxt_corrected_p": np.nan,
                })
                rows.append(rec)
    return pd.DataFrame(rows)


def _cluster_statistics(subject_df: pd.DataFrame, cohort_df: pd.DataFrame,
                        n_perm: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters = []
    for phenotype, band in MAIN_BAND_FOR_PHENOTYPE.items():
        for plane in ("own", "shared"):
            frame = subject_df[(subject_df.dataset == "epilepsiae") &
                               (subject_df.phenotype == phenotype) &
                               (subject_df.band == band) &
                               (subject_df.window_scale == WINDOW_SCALES[0].name)]
            if frame.empty or not np.isfinite(pd.to_numeric(frame[f"{plane}_D"], errors="coerce")).any():
                continue
            pivot = frame.pivot(index="subject", columns="window_center_sec",
                                values=f"{plane}_D").sort_index(axis=1).dropna(how="all")
            result = sign_flip_cluster_maxt(
                pivot.to_numpy(float), pivot.columns.to_numpy(float), n_perm=n_perm,
                seed=_seed(f"cluster:{plane}:{phenotype}:{band}", seed)
            )
            for cluster_id, cluster in enumerate(result["clusters"], 1):
                clusters.append({"dataset_stratum": "epilepsiae", "field_plane": plane,
                                 "phenotype": phenotype, "band": band,
                                 "cluster_id": cluster_id,
                                 "n_subjects": result["n_subjects"], **cluster})
            mask = ((cohort_df.dataset_stratum == "epilepsiae") &
                    (cohort_df.field_plane == plane) &
                    (cohort_df.phenotype == phenotype) & (cohort_df.band == band) &
                    (cohort_df.window_scale == WINDOW_SCALES[0].name))
            mapping = dict(zip(pivot.columns.to_numpy(float), result["maxt_p"]))
            cohort_df.loc[mask, "maxt_corrected_p"] = cohort_df.loc[
                mask, "window_center_sec"].map(mapping)
    return pd.DataFrame(clusters), cohort_df


def _cohort_fixed_statistics(subject_fixed: pd.DataFrame, n_boot: int,
                             n_perm: int, seed: int) -> pd.DataFrame:
    metric_names = [
        "M_post10", "M_post20", "M_pre10", "M_pre20", "M_distal",
        "M_post10_minus_pre10", "M_post20_minus_pre20",
        "M_post10_minus_distal", "M_post20_minus_distal",
    ]
    rows = []
    strata = [("epilepsiae", subject_fixed[subject_fixed.dataset == "epilepsiae"], "subject_cohort"),
              ("yuquan", subject_fixed[subject_fixed.dataset == "yuquan"], "case_sensitivity"),
              ("combined_descriptive", subject_fixed, "descriptive_only")]
    for stratum, frame, role in strata:
        for (phenotype, band, band_label, band_role), group in frame.groupby(
                ["phenotype", "band", "band_label", "band_role"], sort=True):
            for plane in ("own", "shared"):
                for metric in metric_names:
                    source = f"{plane}_{metric}"
                    if source not in group:
                        continue
                    values = pd.to_numeric(group[source], errors="coerce").to_numpy(float)
                    valid = np.isfinite(values)
                    x = values[valid]
                    if not len(x):
                        continue
                    lo, hi = bootstrap_median_ci(
                        x, n_boot=n_boot,
                        seed=_seed(f"fixed:{stratum}:{plane}:{phenotype}:{band}:{metric}", seed))
                    p = (paired_sign_flip_p(
                        x, n_perm=n_perm,
                        seed=_seed(f"sign:{stratum}:{plane}:{phenotype}:{band}:{metric}", seed))
                         if role == "subject_cohort" else np.nan)
                    rows.append({
                        "dataset_stratum": stratum, "inference_role": role,
                        "field_plane": plane, "phenotype": phenotype,
                        "band": band, "band_label": band_label, "band_role": band_role,
                        "metric": metric, "n_subjects": int(len(x)),
                        "n_seizures": int(group.loc[valid, "n_seizures"].sum()),
                        "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
                        "q75": float(np.percentile(x, 75)), "bootstrap_ci_low": lo,
                        "bootstrap_ci_high": hi, "two_sided_sign_flip_p": p,
                    })
    return pd.DataFrame(rows)


def _axis_quality_reports(subject_fixed: pd.DataFrame, n_boot: int,
                          seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report every pre-existing quality layer without changing the denominator."""
    frame = subject_fixed[subject_fixed.band_role == "primary"].copy()
    if "field_ready" not in frame:
        # Closed subject tables can only exist after status=ok and fingerprint
        # validation; this supports statistics-only regeneration of older rows.
        frame["field_ready"] = True
    variables = ("field_ready", "geometry_2d_supported", "strict_stability_pass",
                 "axis_quality_tier", "axis_relation", "shared_field_available")
    contrasts = ("M_post10_minus_pre10", "M_post20_minus_pre20",
                 "M_post10_minus_distal", "M_post20_minus_distal")
    summaries, descriptive = [], []
    strata = (("epilepsiae", frame[frame.dataset == "epilepsiae"]),
              ("yuquan", frame[frame.dataset == "yuquan"]),
              ("combined_descriptive", frame))
    for stratum, stratum_frame in strata:
        for (phenotype, band), group in stratum_frame.groupby(["phenotype", "band"], sort=True):
            for variable in variables:
                for level, layer in group.groupby(variable, dropna=False, sort=True):
                    base = {
                        "dataset_stratum": stratum, "phenotype": phenotype, "band": band,
                        "quality_variable": variable,
                        "quality_level": "NA" if pd.isna(level) else str(level),
                        "n_subjects": int(layer.subject.nunique()),
                        "n_seizures": int(layer.n_seizures.sum()),
                    }
                    summaries.append(base)
                    for plane in ("own", "shared"):
                        for metric in contrasts:
                            source = f"{plane}_{metric}"
                            values = pd.to_numeric(layer[source], errors="coerce").to_numpy(float)
                            valid = np.isfinite(values)
                            x = values[valid]
                            if not len(x):
                                continue
                            lo, hi = bootstrap_median_ci(
                                x, n_boot=n_boot,
                                seed=_seed(f"quality:{stratum}:{phenotype}:{band}:"
                                           f"{variable}:{level}:{plane}:{metric}", seed),
                            )
                            descriptive.append({
                                **base, "field_plane": plane, "metric": metric,
                                "n_subjects": int(len(x)),
                                "n_seizures": int(layer.loc[valid, "n_seizures"].sum()),
                                "median": float(np.median(x)),
                                "q25": float(np.percentile(x, 25)),
                                "q75": float(np.percentile(x, 75)),
                                "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                                "inference_role": "pre_existing_stratum_descriptive",
                            })
    return pd.DataFrame(summaries), pd.DataFrame(descriptive)


def _write_csv(path: Path, rows) -> None:
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def _refresh_subject_fixed_coordinates(subject_field: pd.DataFrame,
                                       per_event: pd.DataFrame) -> pd.DataFrame:
    """Refresh descriptive subject coordinates from the closed per-event table."""
    fixed = per_event[(per_event.window_scale == "fixed") &
                      (per_event.window_region == "fixed")]
    keys = ["subject", "phenotype", "band", "fixed_window"]
    coords = (fixed.groupby(keys, dropna=False)[
        ["window_start_sec", "window_end_sec", "window_center_sec"]].median()
              .reset_index())
    payload = subject_field.drop(columns=["window_start_sec", "window_end_sec",
                                          "window_center_sec"], errors="ignore")
    refreshed = payload.merge(coords, on=keys, how="left", validate="one_to_one")
    if refreshed[["window_start_sec", "window_end_sec", "window_center_sec"]].isna().any().any():
        raise ValueError("fixed-window coordinate refresh left missing subject rows")
    return refreshed


def recompute_closed_statistics(out_dir: str | Path, *, bootstrap: int = 5000,
                                time_permutations: int = 10000,
                                seed: int = 20260715) -> dict:
    """Recompute cohort folds/statistics/figures from closed subject artifacts.

    This is intentionally downstream of raw extraction and spatial-null
    scoring.  It is used when presentation or an explicitly reported plane is
    added without touching any seizure cache, raw trace, or per-seizure value.
    """
    out = Path(out_dir).resolve()
    subject_df = pd.read_csv(out / "subject_timecourse.csv")
    subject_fixed_df = pd.read_csv(out / "fixed_window_subject_summary.csv")
    subject_field_df = pd.read_csv(out / "fixed_window_field_concordance_subject.csv")
    pooled_primary_df = pd.read_csv(out / "phenotype_matched_fixed_window_subject.csv")
    pooled_gamma30_df = pd.read_csv(
        out / "phenotype_matched_fixed_window_subject_gamma30_80_sensitivity.csv"
    )
    subject_funnel_df = pooled_primary_df.drop_duplicates("subject")[
        ["dataset", "subject", "n_seizures", "n_broadband_seizures",
         "n_gamma_seizures", "seizure_idxs", "field_ready",
         "geometry_2d_supported", "strict_stability_pass", "axis_quality_tier",
         "axis_relation", "shared_field_available", "field_fingerprint_algorithm",
         "field_fingerprint_sha256"]
    ].sort_values(["dataset", "subject"])
    per_event_df = pd.read_csv(out / "per_seizure_timecourse.csv", low_memory=False)
    subject_field_df = _refresh_subject_fixed_coordinates(subject_field_df, per_event_df)
    cohort_df = _cohort_timecourse(subject_df, bootstrap, seed)
    cluster_df, cohort_df = _cluster_statistics(
        subject_df, cohort_df, time_permutations, seed
    )
    fixed_stats_df = _cohort_fixed_statistics(
        subject_fixed_df, bootstrap, time_permutations, seed
    )
    quality_summary_df, quality_stats_df = _axis_quality_reports(
        subject_fixed_df, bootstrap, seed
    )
    field_stats_df = _cohort_fixed_field_statistics(
        subject_field_df, bootstrap, time_permutations, seed
    )
    subject_field_df["quality_scope"] = "registered_all_field_ready"
    field_stats_df["quality_scope"] = "registered_all_field_ready"
    strict_subject_df, strict_field_stats_df = _strict_2d_sensitivity_tables(
        subject_field_df, bootstrap, time_permutations, seed
    )
    pooled_primary_stats_df = _phenotype_matched_cohort_statistics(
        pooled_primary_df, bootstrap, time_permutations, seed
    )
    pooled_gamma30_stats_df = _phenotype_matched_cohort_statistics(
        pooled_gamma30_df, bootstrap, time_permutations, seed
    )
    pooled_primary_strict_df = pooled_primary_df[
        pooled_primary_df.axis_quality_tier == "strict_2d"
    ].copy()
    pooled_primary_strict_stats_df = _phenotype_matched_cohort_statistics(
        pooled_primary_strict_df, bootstrap, time_permutations, seed
    )
    relation_stats_df = _phenotype_matched_relation_statistics(
        pooled_primary_df, time_permutations, seed
    )
    exploratory_band_stats_df = _exploratory_band_window_statistics(
        subject_field_df, time_permutations, seed
    )
    _write_csv(out / "cohort_timecourse.csv", cohort_df)
    _write_csv(out / "fixed_window_cohort_statistics.csv", fixed_stats_df)
    _write_csv(out / "cluster_statistics.csv", cluster_df)
    _write_csv(out / "axis_quality_summary.csv", quality_summary_df)
    _write_csv(out / "axis_quality_fixed_descriptive.csv", quality_stats_df)
    _write_csv(out / "fixed_window_field_concordance_subject.csv", subject_field_df)
    _write_csv(out / "fixed_window_field_concordance_cohort.csv", field_stats_df)
    _write_csv(out / "fixed_window_field_concordance_subject_strict_2d_sensitivity.csv",
               strict_subject_df)
    _write_csv(out / "fixed_window_field_concordance_cohort_strict_2d_sensitivity.csv",
               strict_field_stats_df)
    _write_csv(out / "phenotype_matched_fixed_window_cohort.csv",
               pooled_primary_stats_df)
    _write_csv(out / "phenotype_matched_fixed_window_cohort_gamma30_80_sensitivity.csv",
               pooled_gamma30_stats_df)
    _write_csv(out / "phenotype_matched_fixed_window_subject_strict_2d_sensitivity.csv",
               pooled_primary_strict_df)
    _write_csv(out / "phenotype_matched_fixed_window_cohort_strict_2d_sensitivity.csv",
               pooled_primary_strict_stats_df)
    _write_csv(out / "phenotype_matched_relation_statistics.csv", relation_stats_df)
    _write_csv(out / "exploratory_band_fixed_window_statistics.csv",
               exploratory_band_stats_df)
    _write_csv(out / "cohort_subject_funnel.csv", subject_funnel_df)
    from scripts.paper_figures.plot_fig3_sup_tspectral_field_concordance import plot_all
    manifest = plot_all(out)
    validation = validate_closed_outputs(out)
    return {
        "cohort_timecourse_rows": int(len(cohort_df)),
        "fixed_statistics_rows": int(len(fixed_stats_df)),
        "cluster_rows": int(len(cluster_df)),
        "axis_quality_summary_rows": int(len(quality_summary_df)),
        "axis_quality_descriptive_rows": int(len(quality_stats_df)),
        "fixed_window_field_statistics_rows": int(len(field_stats_df)),
        "strict_2d_subject_rows": int(len(strict_subject_df)),
        "strict_2d_field_statistics_rows": int(len(strict_field_stats_df)),
        "phenotype_matched_cohort_rows": int(len(pooled_primary_stats_df)),
        "phenotype_matched_gamma30_sensitivity_rows": int(len(pooled_gamma30_stats_df)),
        "phenotype_matched_relation_rows": int(len(relation_stats_df)),
        "exploratory_band_rows": int(len(exploratory_band_stats_df)),
        "validation_ok": bool(validation["validation_ok"]),
        **manifest,
    }


def validate_closed_outputs(out_dir: str | Path) -> dict:
    """Fail closed on the final artifact and scientific-contract invariants."""
    out = Path(out_dir).resolve()
    event = pd.read_csv(out / "event_inventory.csv")
    drop = pd.read_csv(out / "drop_inventory.csv")
    per_event = pd.read_csv(out / "per_seizure_timecourse.csv", low_memory=False)
    subject = pd.read_csv(out / "subject_timecourse.csv")
    fixed_field = pd.read_csv(out / "fixed_window_field_concordance_subject.csv")
    pooled = pd.read_csv(out / "phenotype_matched_fixed_window_subject.csv")
    pooled_stats = pd.read_csv(out / "phenotype_matched_fixed_window_cohort.csv")
    pooled_gamma30 = pd.read_csv(
        out / "phenotype_matched_fixed_window_subject_gamma30_80_sensitivity.csv"
    )
    relation_stats = pd.read_csv(out / "phenotype_matched_relation_statistics.csv")
    exploratory_band = pd.read_csv(out / "exploratory_band_fixed_window_statistics.csv")
    subject_funnel = pd.read_csv(out / "cohort_subject_funnel.csv")
    strict_fixed_field = pd.read_csv(
        out / "fixed_window_field_concordance_subject_strict_2d_sensitivity.csv"
    )
    hashes = pd.read_csv(out / "input_cache_hashes.csv")
    contract = json.loads((out / "contract.json").read_text())
    primary = event.analysis_status == "included_primary"
    data_drop = event.analysis_status == "drop"
    target = event.analysis_group.isin(MAIN_BAND_FOR_PHENOTYPE)
    contract_excluded = event.analysis_group.isin(["classified_non_target", "not_classified"])
    gamma = per_event[per_event.phenotype == "gamma_nonbroadband"]
    gamma_bandsets = gamma.groupby(["subject", "seizure_idx"]).band.apply(set)
    expected_gamma_bands = {"hfa_60_100", "gamma_30_80_sensitivity"}
    yuquan_clinical = per_event.loc[per_event.dataset == "yuquan",
                                    "clinical_onset_rel_tspectral_sec"]
    required_cohort = [
        PAPER_FIGURES / f"{stem}{suffix}"
        for stem in ("field_concordance_atlas_broadband",
                     "field_concordance_atlas_hfa",
                     "field_concordance_or_margin_board",
                     "fig3sup1_A_observed_maxAB",
                     "fig3sup1_B_null_per_band",
                     "phenotype_matched_cohort_by_window",
                     "phenotype_matched_subject_data_vs_null_by_window",
                     "phenotype_matched_relation_by_window",
                     "phenotype_matched_exploratory_band_by_window")
        for suffix in (".png", ".pdf")
    ]
    expected_windows = set(FIXED_WINDOWS) | {"distal"}
    complete_fixed_sets = fixed_field.groupby(["subject", "phenotype", "band"])[
        "fixed_window"].apply(set)
    complete_pooled_sets = pooled.groupby("subject")["fixed_window"].apply(set)
    complete_gamma30_sets = pooled_gamma30.groupby("subject")["fixed_window"].apply(set)
    pooled_one = pooled.drop_duplicates("subject").set_index("subject")
    pooled_gamma30_one = pooled_gamma30.drop_duplicates("subject").set_index("subject")
    current_hashes_match = all(
        (ROOT / str(row.path)).exists()
        and _hash_file(ROOT / str(row.path)) == str(row.sha256_after)
        for row in hashes.itertuples()
    )
    geometry_subjects = set(pooled.loc[
        pooled.geometry_2d_supported.fillna(False).astype(bool), "subject"
    ])
    relation_primary = relation_stats[
        (relation_stats.quality_scope == "geometry_2d_supported") &
        (relation_stats.dataset_stratum == "combined")
    ]
    checks = {
        "accepted_event_conservation": int(len(event)) == int(
            primary.sum() + data_drop.sum() + contract_excluded.sum()),
        "target_event_conservation": int(target.sum()) == int(primary.sum() + data_drop.sum()),
        "broadband_gamma_selector_overlap_zero": not bool(
            (event.strict_broadband_selector.astype(bool) &
             event.gamma_nonbroadband_selector.astype(bool)).any()),
        "drop_rows_match_event_status": int(
            (drop.drop_type == "data_eligibility").sum()) == int(data_drop.sum()),
        "all_included_windows_have_at_least_6_finite_contacts": int(
            per_event.n_finite.min()) >= MIN_CONTACTS,
        "gamma_main_and_sensitivity_both_present": bool(
            len(gamma_bandsets) and all(v == expected_gamma_bands for v in gamma_bandsets)),
        "yuquan_clinical_onset_not_fabricated": bool(yuquan_clinical.isna().all()),
        "distal_baseline_center_is_zero": float(
            per_event.distal_baseline_z_center_max_abs.max()) <= 1e-9,
        "all_cache_npz_hashes_unchanged": bool(hashes.unchanged.astype(bool).all()),
        "all_cache_npz_hashes_still_match_live_files": bool(current_hashes_match),
        "formal_axis_definition": contract.get("axis_definition") == "template_propagation_axis_v2",
        "formal_axis_direction": contract.get("axis_direction") == "positive_early_to_late",
        "formal_analysis_contract_v1p1": contract.get("contract") == ANALYSIS_CONTRACT,
        "all_included_fields_use_current_fingerprint_algorithm": bool(
            (per_event.field_fingerprint_algorithm ==
             INTERICTAL_FIELD_FINGERPRINT_ALGORITHM).all()),
        "no_fingerprint_drift_drops": not bool(
            drop.drop_reason.astype(str).str.startswith("fingerprint_drift").any()),
        "axis_quality_tiers_are_known": set(fixed_field.axis_quality_tier.dropna()).issubset(
            {"strict_2d", "non_strict_2d", "geometry_unsupported"}),
        "strict_2d_sensitivity_contains_only_strict_geometry": bool(
            len(strict_fixed_field)
            and strict_fixed_field.geometry_2d_supported.astype(bool).all()
            and strict_fixed_field.strict_stability_pass.astype(bool).all()
            and (strict_fixed_field.axis_quality_tier == "strict_2d").all()),
        "spatial_null_draws_at_least_1000": int(
            contract["nulls"]["n_permutations"]) >= 1000,
        "fixed_window_subject_nulls_are_complete": bool(
            len(complete_fixed_sets) and all(value == expected_windows
                                             for value in complete_fixed_sets)),
        "phenotype_matched_subject_windows_are_complete": bool(
            len(complete_pooled_sets) and all(value == expected_windows
                                              for value in complete_pooled_sets)),
        "gamma30_substitution_subject_windows_are_complete": bool(
            len(complete_gamma30_sets) and all(value == expected_windows
                                               for value in complete_gamma30_sets)),
        "phenotype_matched_one_observation_per_event": bool(
            int(pooled.loc[pooled.fixed_window == "post10", "n_seizures"].sum())
            == int(primary.sum())),
        "gamma30_is_substitution_not_event_duplication": bool(
            pooled_one.index.equals(pooled_gamma30_one.index)
            and (pooled_one[["n_seizures", "n_broadband_seizures", "n_gamma_seizures"]]
                 == pooled_gamma30_one[["n_seizures", "n_broadband_seizures",
                                        "n_gamma_seizures"]]).all().all()),
        "primary_gamma_events_use_hfa_not_gamma30": bool(
            pooled.loc[pooled.n_gamma_seizures > 0, "selected_bands"].str.contains(
                "hfa_60_100", regex=False).all()
            and not pooled.selected_bands.str.contains(
                "gamma_30_80_sensitivity", regex=False).any()),
        "gamma30_sensitivity_replaces_hfa": bool(
            pooled_gamma30.loc[pooled_gamma30.n_gamma_seizures > 0,
                               "selected_bands"].str.contains(
                "gamma_30_80_sensitivity", regex=False).all()
            and not pooled_gamma30.selected_bands.str.contains(
                "hfa_60_100", regex=False).any()),
        "combined_pooled_statistics_include_all_subjects_and_yuquan": bool(
            len(pooled_stats[(pooled_stats.dataset_stratum == "combined") &
                             (pooled_stats.field_plane == "own") &
                             (pooled_stats.null_type == "within_shaft")]) == len(FIXED_WINDOW_ORDER)
            and int(pooled_stats.loc[
                (pooled_stats.dataset_stratum == "combined") &
                (pooled_stats.field_plane == "own") &
                (pooled_stats.null_type == "within_shaft"), "n_subjects"
            ].min()) == pooled.subject.nunique()
            and bool((pooled.dataset == "yuquan").any())),
        "subject_funnel_is_one_row_per_pooled_subject": bool(
            not subject_funnel.subject.duplicated().any()
            and set(subject_funnel.subject) == set(pooled.subject)),
        "relation_primary_uses_only_geometry_2d_subjects": bool(
            len(relation_primary) == len(FIXED_WINDOW_ORDER)
            and int(relation_primary.iloc[0][["n_reversed", "n_different", "n_same"]].sum())
            == len(geometry_subjects)),
        "same_relation_is_reported_separately": bool(
            len(relation_primary) and (relation_primary.n_same > 0).all()),
        "exploratory_band_statistics_include_combined_all_three_readouts": bool(
            set(exploratory_band.loc[
                exploratory_band.dataset_stratum == "combined", "band"
            ]) == set(BANDS)),
        "within_shaft_q95_and_p_are_finite": bool(
            np.isfinite(pd.to_numeric(
                fixed_field["own_within_shaft_null_p95_folded"], errors="coerce")).all()
            and np.isfinite(pd.to_numeric(
                fixed_field["own_within_shaft_empirical_p_one_sided"], errors="coerce")).all()),
        "all_contact_q95_and_p_are_finite": bool(
            np.isfinite(pd.to_numeric(
                fixed_field["own_all_contact_null_p95_folded"], errors="coerce")).all()
            and np.isfinite(pd.to_numeric(
                fixed_field["own_all_contact_empirical_p_one_sided"], errors="coerce")).all()),
        "all_required_cohort_png_pdf_present": all(p.exists() for p in required_cohort),
        "figure_readmes_present": bool((PAPER_FIGURES / "README.md").exists()),
        "no_pilot_debug_parallel_directory": not any(
            token in p.name.lower() for p in out.parent.iterdir()
            for token in ("pilot", "debug") if "tspectral_field_concordance" in p.name.lower()
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    result = {
        "validation_ok": not failed, "failed_checks": failed, "checks": checks,
        "counts": {
            "accepted_events": int(len(event)), "target_events": int(target.sum()),
            "included_primary_events": int(primary.sum()), "data_eligibility_drops": int(data_drop.sum()),
            "subjects": int(subject.subject.nunique()), "cache_npz_files": int(len(hashes)),
            "fixed_window_subject_rows": int(len(fixed_field)),
            "phenotype_matched_subject_rows": int(len(pooled)),
            "phenotype_matched_subjects": int(pooled.subject.nunique()),
        },
    }
    (out / "validation_summary.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n"
    )
    if failed:
        raise RuntimeError(f"closed-output validation failed: {failed}")
    return result


def run(args) -> dict:
    if args.n_perm < 1000:
        raise ValueError("spatial null contract requires --n-perm >= 1000")
    out = Path(args.out_dir).resolve()
    # This analysis has one formal output directory.  Clear only artifacts
    # owned by this runner so subset sanity runs cannot contaminate full-cohort
    # results and no pilot/debug sibling is needed.
    for owned_dir in (out / "per_subject", out / "figures"):
        if owned_dir.exists():
            shutil.rmtree(owned_dir)
    for owned_file in (
        "contract.json", "README.md", "event_inventory.csv", "drop_inventory.csv",
        "per_seizure_timecourse.csv", "subject_timecourse.csv", "cohort_timecourse.csv",
        "fixed_window_subject_summary.csv", "fixed_window_cohort_statistics.csv",
        "fixed_window_field_concordance_subject.csv",
        "fixed_window_field_concordance_cohort.csv",
        "fixed_window_field_concordance_subject_strict_2d_sensitivity.csv",
        "fixed_window_field_concordance_cohort_strict_2d_sensitivity.csv",
        "phenotype_matched_fixed_window_subject.csv",
        "phenotype_matched_fixed_window_cohort.csv",
        "phenotype_matched_fixed_window_subject_gamma30_80_sensitivity.csv",
        "phenotype_matched_fixed_window_cohort_gamma30_80_sensitivity.csv",
        "phenotype_matched_fixed_window_subject_strict_2d_sensitivity.csv",
        "phenotype_matched_fixed_window_cohort_strict_2d_sensitivity.csv",
        "phenotype_matched_relation_statistics.csv",
        "exploratory_band_fixed_window_statistics.csv",
        "cohort_subject_funnel.csv",
        "cluster_statistics.csv", "axis_quality_summary.csv", "validation_summary.json",
        "axis_quality_fixed_descriptive.csv", "input_cache_hashes.csv", "figure_manifest.json",
    ):
        path = out / owned_file
        if path.exists():
            path.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_subject").mkdir(exist_ok=True)
    subject_filter = set(args.subjects) if args.subjects else None
    cache_records = _load_subject_caches(subject_filter)
    if not cache_records:
        raise RuntimeError("no T_spectral cache subject records selected")
    selected_subjects = {str(rec[2]["subject"]) for rec in cache_records}
    cache_npz = _cache_npz_paths(selected_subjects)
    hash_before = _hash_manifest(cache_npz)
    inventory, drop_rows = _event_inventory(cache_records)
    inv_lookup = {(r["subject"], int(r["seizure_idx"])): r for r in inventory}

    all_seizure_rows, subject_rows, subject_fixed_rows, subject_field_rows = [], [], [], []
    pooled_primary_rows, pooled_gamma30_rows = [], []
    start_time = time.time()
    for subject_number, (_cache_root, _sidecar_path, meta, selectors) in enumerate(cache_records, 1):
        subject = str(meta["subject"]); dataset = subject.split("_", 1)[0]
        candidates = [(idx, "broadband_1_150") for idx in sorted(selectors["broadband_1_150"])]
        candidates += [(idx, "gamma_nonbroadband") for idx in sorted(selectors["gamma_nonbroadband"])]
        if not candidates:
            continue
        print(f"[{subject_number:02d}/{len(cache_records):02d}] {subject}: {len(candidates)} target seizures",
              flush=True)
        field_path = FIELD_ROOT / f"{subject}.json"
        if not field_path.exists():
            for idx, phenotype in candidates:
                reason = "missing_axis_or_field"
                drop_rows.append({"dataset": dataset, "subject": subject, "seizure_idx": idx,
                                  "analysis_group": phenotype, "band": MAIN_BAND_FOR_PHENOTYPE[phenotype],
                                  "drop_type": "data_eligibility", "drop_reason": reason})
                inv_lookup[(subject, idx)]["analysis_status"] = "drop"
                inv_lookup[(subject, idx)]["drop_reason"] = reason
            continue
        field_record = json.loads(field_path.read_text())
        quality = _field_quality(field_record)
        try:
            scorers = scorers_from_interictal_record(field_record)
        except Exception as exc:
            reason = f"fingerprint_drift:{type(exc).__name__}:{exc}"
            for idx, phenotype in candidates:
                drop_rows.append({"dataset": dataset, "subject": subject, "seizure_idx": idx,
                                  "analysis_group": phenotype, "band": MAIN_BAND_FOR_PHENOTYPE[phenotype],
                                  "drop_type": "data_eligibility", "drop_reason": reason})
                inv_lookup[(subject, idx)]["analysis_status"] = "drop"
                inv_lookup[(subject, idx)]["drop_reason"] = reason
            continue
        subject_events = []
        for event_number, (idx, phenotype) in enumerate(candidates, 1):
            print(f"  seizure {idx} [{event_number}/{len(candidates)}] {phenotype}", flush=True)
            try:
                bands = _process_event(subject, dataset, idx, phenotype, meta, field_record,
                                       scorers, quality, args.n_perm, args.seed)
                main_band = MAIN_BAND_FOR_PHENOTYPE[phenotype]
                if main_band not in bands:
                    raise RuntimeError(f"main band {main_band} was not produced")
                inv_lookup[(subject, idx)]["analysis_status"] = "included_primary"
                inv_lookup[(subject, idx)]["processed_bands"] = ";".join(sorted(bands))
                event_result = {"subject": subject, "dataset": dataset, "seizure_idx": idx,
                                "phenotype": phenotype, "bands": bands}
                subject_events.append(event_result)
                for result in bands.values():
                    all_seizure_rows.extend(result["rows"])
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                main_band = MAIN_BAND_FOR_PHENOTYPE[phenotype]
                drop_rows.append({"dataset": dataset, "subject": subject, "seizure_idx": idx,
                                  "analysis_group": phenotype, "band": main_band,
                                  "drop_type": "data_eligibility", "drop_reason": reason})
                inv_lookup[(subject, idx)]["analysis_status"] = "drop"
                inv_lookup[(subject, idx)]["drop_reason"] = reason
                print(f"    DROP {reason}", flush=True)
        if subject_events:
            srows = _subject_timecourse(subject, dataset, quality, subject_events)
            frows = _fixed_subject_rows(subject, dataset, quality, subject_events)
            field_rows, display = _fixed_field_subject_rows(
                subject, dataset, quality, subject_events
            )
            pooled_primary = _phenotype_matched_fixed_subject_rows(
                subject, dataset, quality, subject_events, readout_family="primary"
            )
            pooled_gamma30 = _phenotype_matched_fixed_subject_rows(
                subject, dataset, quality, subject_events,
                readout_family="gamma_30_80_substitution_sensitivity",
            )
            subject_rows.extend(srows); subject_fixed_rows.extend(frows)
            subject_field_rows.extend(field_rows)
            pooled_primary_rows.extend(pooled_primary)
            pooled_gamma30_rows.extend(pooled_gamma30)
            subject_json = {
                "contract": ANALYSIS_CONTRACT, "subject": subject, "dataset": dataset,
                "field_quality": quality,
                "n_primary_seizures": {phenotype: sum(e["phenotype"] == phenotype for e in subject_events)
                                        for phenotype in MAIN_BAND_FOR_PHENOTYPE},
                "events": [{"seizure_idx": e["seizure_idx"], "phenotype": e["phenotype"],
                            "bands": {key: {k: v for k, v in result.items()
                                            if k not in {"rows", "trajectory_null", "fixed_null",
                                                         "fixed_activation"}}
                                      for key, result in e["bands"].items()}
                            } for e in subject_events],
                "fixed_window_subject_summary": frows,
                "fixed_window_field_concordance": field_rows,
                "phenotype_matched_fixed_window_primary": pooled_primary,
                "phenotype_matched_fixed_window_gamma_30_80_sensitivity": pooled_gamma30,
                "fixed_window_activation_subject_median": display,
                "contact_order": list(field_record["interictal_field"]["contact_order"]),
            }
            (out / "per_subject" / f"{subject}.json").write_text(
                json.dumps(jsonable(subject_json), ensure_ascii=False, indent=2) + "\n"
            )

    event_df = pd.DataFrame(inventory)
    drop_df = pd.DataFrame(drop_rows)
    seizure_df = pd.DataFrame(all_seizure_rows)
    subject_df = pd.DataFrame(subject_rows)
    subject_fixed_df = pd.DataFrame(subject_fixed_rows)
    subject_field_df = pd.DataFrame(subject_field_rows)
    pooled_primary_df = pd.DataFrame(pooled_primary_rows)
    pooled_gamma30_df = pd.DataFrame(pooled_gamma30_rows)
    subject_funnel_df = pooled_primary_df.drop_duplicates("subject")[
        ["dataset", "subject", "n_seizures", "n_broadband_seizures",
         "n_gamma_seizures", "seizure_idxs", "field_ready",
         "geometry_2d_supported", "strict_stability_pass", "axis_quality_tier",
         "axis_relation", "shared_field_available", "field_fingerprint_algorithm",
         "field_fingerprint_sha256"]
    ].sort_values(["dataset", "subject"])
    if subject_df.empty:
        raise RuntimeError("no subject-level results were produced")
    cohort_df = _cohort_timecourse(subject_df, args.bootstrap, args.seed)
    cluster_df, cohort_df = _cluster_statistics(subject_df, cohort_df,
                                                 args.time_permutations, args.seed)
    fixed_stats_df = _cohort_fixed_statistics(subject_fixed_df, args.bootstrap,
                                               args.time_permutations, args.seed)
    quality_summary_df, quality_stats_df = _axis_quality_reports(
        subject_fixed_df, args.bootstrap, args.seed
    )
    field_stats_df = _cohort_fixed_field_statistics(
        subject_field_df, args.bootstrap, args.time_permutations, args.seed
    )
    subject_field_df["quality_scope"] = "registered_all_field_ready"
    field_stats_df["quality_scope"] = "registered_all_field_ready"
    strict_subject_df, strict_field_stats_df = _strict_2d_sensitivity_tables(
        subject_field_df, args.bootstrap, args.time_permutations, args.seed
    )
    pooled_primary_stats_df = _phenotype_matched_cohort_statistics(
        pooled_primary_df, args.bootstrap, args.time_permutations, args.seed
    )
    pooled_gamma30_stats_df = _phenotype_matched_cohort_statistics(
        pooled_gamma30_df, args.bootstrap, args.time_permutations, args.seed
    )
    pooled_primary_strict_df = pooled_primary_df[
        pooled_primary_df.axis_quality_tier == "strict_2d"
    ].copy()
    pooled_primary_strict_stats_df = _phenotype_matched_cohort_statistics(
        pooled_primary_strict_df, args.bootstrap, args.time_permutations, args.seed
    )
    relation_stats_df = _phenotype_matched_relation_statistics(
        pooled_primary_df, args.time_permutations, args.seed
    )
    exploratory_band_stats_df = _exploratory_band_window_statistics(
        subject_field_df, args.time_permutations, args.seed
    )

    _write_csv(out / "event_inventory.csv", event_df)
    _write_csv(out / "drop_inventory.csv", drop_df)
    _write_csv(out / "per_seizure_timecourse.csv", seizure_df)
    _write_csv(out / "subject_timecourse.csv", subject_df)
    _write_csv(out / "cohort_timecourse.csv", cohort_df)
    _write_csv(out / "fixed_window_subject_summary.csv", subject_fixed_df)
    _write_csv(out / "fixed_window_cohort_statistics.csv", fixed_stats_df)
    _write_csv(out / "fixed_window_field_concordance_subject.csv", subject_field_df)
    _write_csv(out / "fixed_window_field_concordance_cohort.csv", field_stats_df)
    _write_csv(out / "fixed_window_field_concordance_subject_strict_2d_sensitivity.csv",
               strict_subject_df)
    _write_csv(out / "fixed_window_field_concordance_cohort_strict_2d_sensitivity.csv",
               strict_field_stats_df)
    _write_csv(out / "phenotype_matched_fixed_window_subject.csv", pooled_primary_df)
    _write_csv(out / "phenotype_matched_fixed_window_cohort.csv",
               pooled_primary_stats_df)
    _write_csv(out / "phenotype_matched_fixed_window_subject_gamma30_80_sensitivity.csv",
               pooled_gamma30_df)
    _write_csv(out / "phenotype_matched_fixed_window_cohort_gamma30_80_sensitivity.csv",
               pooled_gamma30_stats_df)
    _write_csv(out / "phenotype_matched_fixed_window_subject_strict_2d_sensitivity.csv",
               pooled_primary_strict_df)
    _write_csv(out / "phenotype_matched_fixed_window_cohort_strict_2d_sensitivity.csv",
               pooled_primary_strict_stats_df)
    _write_csv(out / "phenotype_matched_relation_statistics.csv", relation_stats_df)
    _write_csv(out / "exploratory_band_fixed_window_statistics.csv",
               exploratory_band_stats_df)
    _write_csv(out / "cohort_subject_funnel.csv", subject_funnel_df)
    _write_csv(out / "cluster_statistics.csv", cluster_df)
    _write_csv(out / "axis_quality_summary.csv", quality_summary_df)
    _write_csv(out / "axis_quality_fixed_descriptive.csv", quality_stats_df)

    hash_after = _hash_manifest(cache_npz)
    _write_hash_table(out / "input_cache_hashes.csv", hash_before, hash_after)
    hashes_unchanged = hash_before == hash_after
    counts = {
        "accepted_tspectral_events": int(len(event_df)),
        "classified_events": int((event_df.label_status == "classified").sum()),
        "not_classified_events": int((event_df.analysis_group == "not_classified").sum()),
        "strict_broadband_events": int(event_df.strict_broadband_selector.sum()),
        "gamma_nonbroadband_events": int(event_df.gamma_nonbroadband_selector.sum()),
        "included_primary_events": int((event_df.analysis_status == "included_primary").sum()),
        "data_eligibility_drops": int((drop_df.drop_type == "data_eligibility").sum()) if len(drop_df) else 0,
        "subjects_with_primary_results": int(subject_df.subject.nunique()),
        "phenotype_matched_subjects": int(pooled_primary_df.subject.nunique()),
        "phenotype_matched_subject_seizures": int(
            pooled_primary_df.drop_duplicates("subject").n_seizures.sum()),
        "phenotype_matched_broadband_seizures": int(
            pooled_primary_df.drop_duplicates("subject").n_broadband_seizures.sum()),
        "phenotype_matched_gamma_seizures": int(
            pooled_primary_df.drop_duplicates("subject").n_gamma_seizures.sum()),
        "phenotype_matched_subjects_by_dataset": {
            str(key): int(value) for key, value in pooled_primary_df.drop_duplicates(
                "subject"
            ).dataset.value_counts().items()
        },
        "phenotype_matched_subjects_by_axis_relation": {
            str(key): int(value) for key, value in pooled_primary_df.drop_duplicates(
                "subject"
            ).axis_relation.value_counts(dropna=False).items()
        },
        "included_events_by_axis_quality_tier": {
            str(key): int(value) for key, value in event_df.loc[
                event_df.analysis_status == "included_primary", "axis_quality_tier"
            ].value_counts(dropna=False).items()
        },
        "subjects_by_axis_quality_tier": {
            str(key): int(value) for key, value in subject_df.drop_duplicates("subject")[
                "axis_quality_tier"
            ].value_counts(dropna=False).items()
        },
    }
    drop_counts = (drop_df.groupby(["drop_type", "drop_reason"]).size().reset_index(name="n").to_dict("records")
                   if len(drop_df) else [])
    contract = {
        "contract": ANALYSIS_CONTRACT,
        "scientific_question": "phenotype-matched early-ictal energy topography concordance with frozen interictal TA/TB fields",
        "axis_definition": "template_propagation_axis_v2",
        "axis_direction": "positive_early_to_late",
        "field_source": str(FIELD_ROOT.relative_to(ROOT) / "<dataset>_<subject>.json"),
        "field_loading_api": "scorers_from_interictal_record (fingerprint fail-closed)",
        "field_fingerprint_algorithm": INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
        "activation_join": "exact contact name to frozen interictal_field.contact_order",
        "raw_time_zero_contract": {
            "raw_extractor_reference": "EEG onset",
            "tspectral_coordinate_used": "t_spectral_rel_eeg_sec",
            "cache_relative_coordinate": "provenance only; never used to re-zero raw EEG",
        },
        "activation_readouts": {
            "primary": "delta_E = baseline robust-z level minus its explicit distal-baseline median",
            "legacy_sensitivity": "unsubtracted baseline robust-z level, separately named legacy_*",
        },
        "field_plane_reporting": {
            "own": "primary, own_a/own_b then maxAB",
            "shared": "separate sensitivity when pre-existing shared field is available",
            "cross_plane_selection": "forbidden; own and shared are never maximized together",
        },
        "tspectral_caches": [str(EPI_CACHE.relative_to(ROOT)), str(YUQ_CACHE.relative_to(ROOT))],
        "phenotype_label_source": str(PHENOTYPE_TABLE.relative_to(ROOT)),
        "phenotype_label_version": PHENOTYPE_LABEL_VERSION,
        "bands": BANDS,
        "spectral": {"window_sec": SPECTRAL_WINDOW_SEC, "hop_sec": SPECTRAL_HOP_SEC,
                     "input": "notch-filtered raw SEEG", "notch_hz": list(NOTCH_FREQS),
                     "fft_bin_line_mask": False, "power": "closed-band summed PSD then log"},
        "distal_baseline": {"reference": "original EEG onset", "interval_sec": list(DISTAL_BASELINE_EEG_SEC),
                            "normalization": "per-contact median/MAD robust-z, then explicit delta to baseline median"},
        "analysis_range_rel_tspectral_sec": list(ANALYSIS_RANGE_SEC),
        "window_scales": [vars(v) for v in WINDOW_SCALES], "fixed_windows": FIXED_WINDOWS,
        "nulls": {"n_permutations": args.n_perm, "seed": args.seed,
                  "all_contact": "one identity permutation per seizure/draw, reused across all time/scales/bands",
                  "within_shaft": "same, restricted within shaft; primary spatial null",
                  "selection_cost": "mirror and TA/TB maxAB recomputed for own and shared in every draw",
                  "folding": "seizure -> subject median within every draw"},
        "cohort_statistics": {"unit": "subject", "bootstrap": args.bootstrap,
                              "fixed_window_test": "two-sided subject sign-flip",
                              "time_correction": "two-sided subject sign-flip maxT and cluster",
                              "time_permutations": args.time_permutations,
                              "no_scientific_pass_fail_gate": True},
        "phenotype_matched_primary_estimand": {
            "seizure_readout": {
                "broadband_1_150": "Broadband 1-150 Hz",
                "gamma_nonbroadband": "HFA 60-100 Hz",
            },
            "seizure_folding": "one phenotype-matched observation per seizure",
            "subject_folding": "median across all target seizures within subject",
            "cohort_folding": "subject is the independent unit; no band split",
            "tested_statistic": "mean subject observed-own-maxAB minus folded-null median",
            "fixed_window_multiplicity": "same-sign subject sign flip with maxT across all six pre-specified windows",
            "primary_null": "within_shaft",
            "reference_null": "all_contact",
            "files": ["phenotype_matched_fixed_window_subject.csv",
                      "phenotype_matched_fixed_window_cohort.csv",
                      "cohort_subject_funnel.csv"],
        },
        "gamma_30_80_substitution_sensitivity": {
            "rule": "replace every gamma seizure HFA readout by Gamma 30-80 Hz; never add a second seizure observation",
            "files": [
                "phenotype_matched_fixed_window_subject_gamma30_80_sensitivity.csv",
                "phenotype_matched_fixed_window_cohort_gamma30_80_sensitivity.csv",
            ],
        },
        "axis_relation_heterogeneity": {
            "primary_scope": "geometry_2d_supported",
            "comparison": "reversed versus different",
            "same_relation": "retained as a separate descriptive stratum",
            "strict_2d": "pre-specified sensitivity",
            "multiplicity": "label permutation maxT across six fixed windows",
            "file": "phenotype_matched_relation_statistics.csv",
        },
        "fixed_window_field_concordance": {
            "observed": "subject median of seizure-level own maxAB",
            "primary_null": "within_shaft",
            "reference_null": "all_contact",
            "folding": "null draw-wise seizure median before subject quantiles",
            "subject_threshold": "one-sided null Q95, displayed without a cohort gate",
            "cohort_metric": "subject own maxAB minus subject folded-null median",
            "files": ["fixed_window_field_concordance_subject.csv",
                      "fixed_window_field_concordance_cohort.csv",
                      "fixed_window_field_concordance_subject_strict_2d_sensitivity.csv",
                      "fixed_window_field_concordance_cohort_strict_2d_sensitivity.csv"],
        },
        "axis_quality_reporting": {
            "variables": ["field_ready", "geometry_2d_supported", "strict_stability_pass",
                          "axis_quality_tier", "axis_relation", "shared_field_available"],
            "role": "pre-existing strata, descriptive only; never changes the primary denominator",
            "tier_definition": {
                "strict_2d": "field ready, 2D geometry supported, strict stability passed",
                "non_strict_2d": "field ready and 2D geometry supported, strict stability not passed",
                "geometry_unsupported": "field estimable but 2D geometry unsupported",
                "field_unavailable": "no reusable field; data-eligibility drop",
            },
            "strict_2d_sensitivity": "separate downstream table; not substituted for the registered primary denominator",
            "files": ["axis_quality_summary.csv", "axis_quality_fixed_descriptive.csv"],
        },
        "counts": counts, "drop_counts": drop_counts,
        "cache_npz_hashes_unchanged": hashes_unchanged,
        "runtime_seconds": float(time.time() - start_time),
        "selected_subject_filter": sorted(subject_filter) if subject_filter else None,
    }
    (out / "contract.json").write_text(json.dumps(jsonable(contract), ensure_ascii=False, indent=2) + "\n")

    # Plot only after every numeric artifact is closed.
    from scripts.paper_figures.plot_fig3_sup_tspectral_field_concordance import plot_all  # noqa: E402
    plot_all(out)
    validate_closed_outputs(out)
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)
    print(f"cache hashes unchanged: {hashes_unchanged}", flush=True)
    print(f"wrote {out}", flush=True)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="optional dataset_subject subset; outputs still use the one formal directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--time-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--statistics-only", action="store_true",
                        help="recompute cohort statistics/figures from closed subject tables")
    args = parser.parse_args()
    if args.statistics_only:
        print(json.dumps(recompute_closed_statistics(
            args.out_dir, bootstrap=args.bootstrap,
            time_permutations=args.time_permutations, seed=args.seed,
        ), ensure_ascii=False, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()

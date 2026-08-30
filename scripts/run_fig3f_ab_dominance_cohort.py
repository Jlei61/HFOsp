#!/usr/bin/env python3
"""Figure 3F: peri-onset A/B-template dominance in the frozen 17-subject cohort.

The cohort is read directly from the canonical ``own_2d_geometry`` result.  It is
not restricted to the seven shared-plane subjects used by Figure 3C, and the
A/B-axis relation (reversed/same/different) is descriptive rather than an
eligibility gate.

For every successful seizure, a subject-fixed contrast is frozen from the
interictal templates::

    C_AB(t) = corr(E(t), e_A - e_B)

The primary endpoint compares absolute A/B dominance in far pre-ictal
[-120,-60) s and near-onset [-30,+10) s windows.  Each subject is one unit:
far and near are first summarized across seizures, and their difference is
tested against seizure-wise exhaustive circular time shifts.  A within-shaft
spatial-presence gate is retained only as a sensitivity analysis.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    FROZEN_FIELD_DIR,
    _shared_window_extent,
)
from scripts.plot_topic5_signed_broadband_movie import (  # noqa: E402
    _band_power_trace_chunked,
    _offset_rel,
    _pre_target,
    _window_values,
)
from scripts.plot_topic5_signed_broadband_similarity_timecourse import (  # noqa: E402
    _eligible_idxs,
)
from scripts.run_topic5_fig3b_maxab_spatial_null import (  # noqa: E402
    _permutation_indices,
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
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_scaffold_ab_contrast import (  # noqa: E402
    build_D_AB,
    classify_event,
    contrast_timecourse,
    locking_statistic,
    template_pair_tier,
)
from src.topic5_template_axis_field import (  # noqa: E402
    _row_pearson,
    interictal_field_fingerprint,
    scorers_from_interictal_record,
)


ARTIFACT_ROOT = Path(os.environ.get("HFOSP_ARTIFACT_ROOT", ROOT)).resolve()
COHORT_SOURCE = Path(os.environ.get(
    "HFOSP_FIG3F_COHORT_SOURCE",
    ARTIFACT_ROOT / "results/topic5_ictal_recruitment/template_axis_field/cohort_summary.json",
)).resolve()
OUT_DIR = Path(os.environ.get(
    "HFOSP_FIG3F_OUT_DIR",
    ROOT / "results/paper-ready-figure/fig3f_ab_dominance_cohort",
)).resolve()
SUB_DIR = OUT_DIR / "per_subject"
COHORT_CSV = OUT_DIR / "fig3f_ab_dominance_cohort.csv"
COHORT_JSON = OUT_DIR / "fig3f_ab_dominance_cohort.json"

ALGORITHM_VERSION = "fig3f_full17_fixed_ab_contrast_v2"
PERMUTATION_COUPLING = "per_seizure_fixed_spatial_mapping_across_66_windows_v1"
COHORT_NULL_VERSION = "subject_first_independent_circular_shift_draws_v1"
START_SEC, STOP_SEC = -120.0, 20.0
WINDOW_SEC, STEP_SEC = 10.0, 2.0
BAND = (1.0, 150.0)
FAR_PRE = (-120.0, -60.0)
NEAR_ONSET = (-30.0, 10.0)
NEAR_PRE = (-30.0, 0.0)
EARLY_ICTAL = (0.0, 10.0)
DELTA_SIDE = 0.2
ALPHA_PRESENT = 0.05
N_VALID_SEIZURE_MIN = 3
N_VALID_SHIFT_MIN = 40
N_PERM_SUBJECT = 1000
N_PERM_COHORT = 10000
MIN_FINITE_FRACTION = 0.90


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _json_scalar(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _stable_subject_seed(subject: str, seed: int) -> int:
    digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()
    return (int(seed) + int(digest[:8], 16)) % (2**32 - 1)


def _cohort_subjects() -> list[str]:
    """Read the exact canonical 17-subject own-field 2-D denominator."""
    payload = json.loads(COHORT_SOURCE.read_text())
    node = payload["metrics"]["own_2d_geometry"]["within_shaft"]["own_maxab"]
    subjects = [str(value) for value in node["subjects"]]
    if node.get("status") != "ok" or int(node.get("n", -1)) != len(subjects):
        raise RuntimeError("own_2d_geometry cohort summary is internally inconsistent")
    if len(subjects) != 17 or len(subjects) != len(set(subjects)):
        raise RuntimeError(f"expected canonical 17 unique subjects, found {len(subjects)}")
    return subjects


def _load_frozen_record(subject: str) -> dict:
    """Load and fingerprint-check an own-field 2-D record without shared-field gating."""
    path = FROZEN_FIELD_DIR / f"{subject}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text())
    dataset, sid = subject.split("_", 1)
    if record.get("dataset") != dataset or str(record.get("subject")) != sid:
        raise ValueError(f"{subject}: frozen subject identity mismatch")
    scorers_from_interictal_record(record)  # validates contract, own A/B, and fingerprint
    field = record.get("interictal_field") or {}
    if field.get("fingerprint_sha256") != interictal_field_fingerprint(record):
        raise ValueError(f"{subject}: interictal field fingerprint mismatch")
    pair = record.get("axis_pair") or {}
    if pair.get("geometry_2d_supported") is not True:
        raise ValueError(f"{subject}: own 2-D geometry is not supported")
    return record


def _seizure_args(subject: str, seizure_idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        subject=subject,
        seizure_idx=int(seizure_idx),
        start_sec=START_SEC,
        stop_sec=STOP_SEC,
        band_lo=BAND[0],
        band_hi=BAND[1],
        spectral_win_sec=1.0,
        hop_sec=0.5,
        smooth_sec=WINDOW_SEC,
        frame_step_sec=STEP_SEC,
        onset_win_sec=10.0,
        chunk_ch=16,
    )


def _extract_frozen_values(args: SimpleNamespace, record: dict) -> dict:
    """Extract one seizure on the fixed grid in frozen contact order."""
    subject = args.subject
    dataset, sid = subject.split("_", 1)
    seizure_idx = int(args.seizure_idx)
    inventory, _ = _inventory_rows(dataset, sid)
    if not 0 <= seizure_idx < len(inventory):
        raise IndexError(
            f"{subject}: seizure_idx={seizure_idx} out of range (n={len(inventory)})"
        )
    inv = inventory[seizure_idx]
    offset = _offset_rel(dataset, inv)
    pre_sec = _pre_target(dataset, inv, display_start=args.start_sec)
    stop_at, post_sec = _shared_window_extent(
        offset=offset, stop_sec=args.stop_sec, smooth_sec=args.smooth_sec
    )
    window = extract_seizure_window(
        f"{dataset}/{sid}",
        seizure_idx,
        pre_sec=pre_sec,
        post_sec=post_sec,
        reference=ICTAL_REFERENCE[dataset],
        results_root=ARTIFACT_ROOT / "results",
    )
    if window.fs / 2.0 <= args.band_hi:
        raise RuntimeError(
            f"{subject}: Nyquist {window.fs / 2.0:g} <= band_hi {args.band_hi:g}"
        )
    power, sample_time = _band_power_trace_chunked(
        window.signal,
        window.fs,
        band=(args.band_lo, args.band_hi),
        win_sec=args.spectral_win_sec,
        hop_sec=args.hop_sec,
        chunk_ch=args.chunk_ch,
    )
    eeg_rel = None
    if window.eeg_onset_epoch is not None and window.clin_onset_epoch is not None:
        eeg_rel = window.eeg_onset_epoch - window.clin_onset_epoch
    baseline = resolve_baseline_window(
        power.shape[1],
        hop_sec=args.hop_sec,
        pre_sec=window.pre_sec,
        buffer_sec=GUARD_SEC,
        eeg_onset_rel_sec=eeg_rel,
        min_baseline_valid_sec=MIN_BASELINE_SEC,
    )
    if not baseline.valid:
        raise RuntimeError(f"{subject} sz{seizure_idx}: invalid baseline {baseline}")
    z = recruit.baseline_robust_z(
        power,
        (baseline.start_idx, baseline.end_idx),
        hop_sec=args.hop_sec,
        min_baseline_valid_sec=MIN_BASELINE_SEC,
    )
    rel_time = np.asarray(sample_time, float) - float(window.pre_sec)

    target_names = [str(name) for name in record["interictal_field"]["contact_order"]]
    raw_names = [recruit.bipolar_alias_label(name) for name in window.ch_names]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError(f"{subject}: raw channel aliases are not unique")
    raw_index = {name: index for index, name in enumerate(raw_names)}
    names = [name for name in target_names if name in raw_index]
    z_selected = z[np.asarray([raw_index[name] for name in names], int)]
    finite_contact = np.isfinite(z_selected).any(axis=1)
    names = [name for name, keep in zip(names, finite_contact) if keep]
    z_selected = z_selected[finite_contact]
    if len(names) < 6:
        raise RuntimeError(f"{subject}: insufficient matched contacts ({len(names)})")

    stop_start = stop_at - args.smooth_sec
    starts = np.arange(args.start_sec, stop_start + 1e-9, args.frame_step_sec)
    values = _window_values(z_selected, rel_time, starts, args.smooth_sec)
    centers = starts + args.smooth_sec / 2.0
    expected = np.arange(
        START_SEC + WINDOW_SEC / 2.0,
        STOP_SEC - WINDOW_SEC / 2.0 + 1e-9,
        STEP_SEC,
    )
    if len(centers) != 66 or not np.allclose(centers, expected, atol=1e-9, rtol=0):
        raise RuntimeError(
            f"{subject} seizure {seizure_idx}: noncanonical grid ({len(centers)} windows)"
        )
    return {
        "seizure_idx": seizure_idx,
        "names": names,
        "centers": centers,
        "values": np.asarray(values, float),
        "offset_sec": float(offset),
    }


def _load_subject_windows(subject: str, record: dict) -> tuple[list[dict], list[dict], int]:
    requested = [int(value) for value in _eligible_idxs(subject)]
    seizures: list[dict] = []
    drops: list[dict] = []
    for seizure_idx in requested:
        try:
            seizures.append(
                _extract_frozen_values(_seizure_args(subject, seizure_idx), record)
            )
        except Exception as exc:
            drops.append({
                "seizure_idx": int(seizure_idx),
                "reason": f"{type(exc).__name__}: {exc}",
            })
    if not seizures:
        raise RuntimeError(f"{subject}: no successful seizures; drops={drops}")
    return seizures, drops, len(requested)


def _subject_fixed_joint(
    seizures: list[dict], record: dict
) -> tuple[list[str], list[np.ndarray]]:
    frozen_order = [str(name) for name in record["interictal_field"]["contact_order"]]
    common = set(frozen_order)
    for seizure in seizures:
        common.intersection_update(seizure["names"])
    names = [name for name in frozen_order if name in common]
    matrices: list[np.ndarray] = []
    for seizure in seizures:
        index = {name: i for i, name in enumerate(seizure["names"])}
        matrices.append(seizure["values"][:, [index[name] for name in names]])
    finite_fraction = np.isfinite(np.vstack(matrices)).mean(axis=0)
    keep = finite_fraction >= MIN_FINITE_FRACTION
    names = [name for name, ok in zip(names, keep) if ok]
    matrices = [matrix[:, keep] for matrix in matrices]
    if len(names) < 6:
        raise RuntimeError(f"subject-fixed joint set has only {len(names)} contacts")
    return names, matrices


def _shaft_qc(names: list[str]) -> dict:
    sizes: dict[str, int] = {}
    for name in names:
        shaft = parse_shaft(name)[0]
        sizes[shaft] = sizes.get(shaft, 0) + 1
    n_contacts_shuffled = sum(size for size in sizes.values() if size >= 2)
    fraction = n_contacts_shuffled / len(names) if names else 0.0
    n_multi = sum(size >= 2 for size in sizes.values())
    low_dof = n_multi < 2 or fraction < 0.60
    return {
        "n_contacts": len(names),
        "n_shafts": len(sizes),
        "shaft_sizes": dict(sorted(sizes.items())),
        "n_multi_contact_shafts": int(n_multi),
        "n_contacts_shuffled": int(n_contacts_shuffled),
        "fraction_contacts_shuffled": float(fraction),
        "n_singleton_contacts": int(len(names) - n_contacts_shuffled),
        "testable": not low_dof,
        "low_dof": bool(low_dof),
    }


def axis_present_fixed_mapping(
    values: np.ndarray,
    names: list[str],
    e_a: np.ndarray,
    e_b: np.ndarray,
    rng: np.random.Generator,
    *,
    n_perm: int,
    alpha: float = ALPHA_PRESENT,
) -> dict:
    """Pointwise within-shaft maxAB gate, used only for sensitivity analysis."""
    values = np.asarray(values, float)
    e_a = np.asarray(e_a, float)
    e_b = np.asarray(e_b, float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("values/contact shape mismatch")
    if e_a.shape != (len(names),) or e_b.shape != (len(names),):
        raise ValueError("template/contact shape mismatch")
    qc = _shaft_qc(names)
    mappings = _permutation_indices(names, rng, "within_shaft", int(n_perm))
    observed = np.maximum(
        np.abs(_row_pearson(e_a, values)), np.abs(_row_pearson(e_b, values))
    )
    pvals = np.full(values.shape[0], np.nan, float)
    for window_index, row in enumerate(values):
        if not np.isfinite(observed[window_index]):
            continue
        shuffled = row[mappings]
        null = np.maximum(
            np.abs(_row_pearson(e_a, shuffled)),
            np.abs(_row_pearson(e_b, shuffled)),
        )
        finite = np.isfinite(null)
        if finite.any():
            pvals[window_index] = (
                1 + int(np.sum(null[finite] >= observed[window_index]))
            ) / (int(finite.sum()) + 1)
    return {
        "present": np.isfinite(pvals) & (pvals < alpha),
        "within_shaft_p": pvals,
        "maxAB": observed,
        "testable": bool(qc["testable"]),
        "low_dof": bool(qc["low_dof"]),
        "qc": qc,
    }


def _fixed_contrast(record: dict, names: list[str]) -> dict:
    field = record["interictal_field"]
    order = [str(name) for name in field["contact_order"]]
    if len(order) != len(set(order)) or len(names) != len(set(names)):
        raise ValueError("duplicate contact name in frozen/current order")
    index = {name: i for i, name in enumerate(order)}
    missing = [name for name in names if name not in index]
    if missing:
        raise ValueError(f"contacts outside frozen interictal field: {missing}")
    full = build_D_AB(field["rank_a"], field["rank_b"])
    take = np.asarray([index[name] for name in names], int)
    return {
        "eA": np.asarray(full["eA"], float)[take],
        "eB": np.asarray(full["eB"], float)[take],
        "D_AB": np.asarray(full["D_AB"], float)[take],
        "rho_AB": float(full["rho_AB"]),
        "sd_D_AB": float(full["sd_D_AB"]),
        "template_pair_tier": template_pair_tier(float(full["rho_AB"])),
    }


def circular_shift_polar_null(
    c_ab: np.ndarray,
    present: np.ndarray,
    centers: np.ndarray,
    far_pre=FAR_PRE,
    near_onset=NEAR_ONSET,
    *,
    n_valid_shift_min: int = N_VALID_SHIFT_MIN,
) -> dict:
    """Enumerate all non-zero circular shifts and retain paired far/near values."""
    c_ab = np.asarray(c_ab, float)
    present = np.asarray(present, bool)
    centers = np.asarray(centers, float)
    observed = locking_statistic(c_ab, present, centers, far_pre, near_onset)
    pairs = []
    for shift in range(1, len(centers)):
        shifted = locking_statistic(
            np.roll(c_ab, shift),
            np.roll(present, shift),
            centers,
            far_pre,
            near_onset,
        )
        if np.isfinite(shifted["polar_far"]) and np.isfinite(shifted["polar_near"]):
            pairs.append([shifted["polar_far"], shifted["polar_near"]])
    null_pairs = np.asarray(pairs, float).reshape((-1, 2))
    valid = (
        np.isfinite(observed["polar_far"])
        and np.isfinite(observed["polar_near"])
        and len(null_pairs) >= n_valid_shift_min
    )
    return {
        "status": "ok" if valid else "insufficient",
        "polar_far_obs": float(observed["polar_far"]),
        "polar_near_obs": float(observed["polar_near"]),
        "null_pairs": null_pairs,
        "n_valid_shift": int(len(null_pairs)),
    }


def subject_paired_null(
    per_seizure: list[dict], *, n_perm: int = N_PERM_SUBJECT, seed: int = 0
) -> dict:
    """Subject-first paired endpoint and circular-time null across seizures."""
    usable = [
        item for item in per_seizure
        if item.get("status") == "ok" and len(item.get("null_pairs", [])) > 0
    ]
    if not usable:
        return {
            "polar_far": np.nan,
            "polar_near": np.nan,
            "delta": np.nan,
            "null_p95": np.nan,
            "p": np.nan,
            "subject_locked": False,
            "n_valid_seizures": 0,
            "null_delta": np.full(n_perm, np.nan),
        }
    far = float(np.median([item["polar_far_obs"] for item in usable]))
    near = float(np.median([item["polar_near_obs"] for item in usable]))
    delta = near - far
    rng = np.random.default_rng(seed)
    null_delta = np.full(n_perm, np.nan, float)
    for replicate in range(n_perm):
        draws = [item["null_pairs"][rng.integers(len(item["null_pairs"]))] for item in usable]
        draws = np.asarray(draws, float)
        null_delta[replicate] = np.median(draws[:, 1]) - np.median(draws[:, 0])
    p95 = float(np.percentile(null_delta, 95))
    p = float((1 + np.sum(null_delta >= delta)) / (n_perm + 1))
    return {
        "polar_far": far,
        "polar_near": near,
        "delta": float(delta),
        "null_p95": p95,
        "p": p,
        "subject_locked": bool(delta > p95),
        "n_valid_seizures": len(usable),
        "null_delta": null_delta,
    }


def _analysis_summary(result: dict, *, eligible: bool) -> dict:
    return {
        "eligible": bool(eligible),
        "subject_locked": bool(eligible and result["subject_locked"]),
        "n_valid_seizures": int(result["n_valid_seizures"]),
        "polar_far": _json_scalar(result["polar_far"]),
        "polar_near": _json_scalar(result["polar_near"]),
        "delta": _json_scalar(result["delta"]),
        "null_p95": _json_scalar(result["null_p95"]),
        "p": _json_scalar(result["p"]),
    }


def compute_subject(
    subject: str, gate_nperm: int, seed: int
) -> tuple[dict, list[dict], dict]:
    record = _load_frozen_record(subject)
    seizures, drops, n_requested = _load_subject_windows(subject, record)
    names, matrices = _subject_fixed_joint(seizures, record)
    contrast = _fixed_contrast(record, names)
    shaft_qc = _shaft_qc(names)
    subject_seed = _stable_subject_seed(subject, seed)

    rows: list[dict] = []
    primary_nulls: list[dict] = []
    sensitivity_nulls: list[dict] = []
    c_ab_matrix = np.full((len(seizures), 66), np.nan, float)
    present_matrix = np.zeros((len(seizures), 66), bool)
    within_shaft_p = np.full((len(seizures), 66), np.nan, float)

    for row_index, (seizure, values) in enumerate(zip(seizures, matrices)):
        centers = seizure["centers"]
        timecourse = contrast_timecourse(
            values, contrast["D_AB"], contrast["eA"], contrast["eB"]
        )
        c_ab = np.asarray(timecourse["C_AB"], float)
        primary_present = np.isfinite(c_ab)
        primary_shift = circular_shift_polar_null(c_ab, primary_present, centers)
        if primary_shift["status"] == "ok":
            primary_nulls.append(primary_shift)

        gate = axis_present_fixed_mapping(
            values,
            names,
            contrast["eA"],
            contrast["eB"],
            np.random.default_rng(subject_seed + int(seizure["seizure_idx"])),
            n_perm=gate_nperm,
        )
        sensitivity_shift = circular_shift_polar_null(
            c_ab, gate["present"], centers
        )
        if gate["testable"] and sensitivity_shift["status"] == "ok":
            sensitivity_nulls.append(sensitivity_shift)

        primary_event = classify_event(
            c_ab,
            primary_present,
            centers,
            FAR_PRE,
            NEAR_ONSET,
            NEAR_PRE,
            EARLY_ICTAL,
            DELTA_SIDE,
        )
        rows.append({
            "subject": subject,
            "seizure_idx": int(seizure["seizure_idx"]),
            "primary_polar_far": _json_scalar(primary_shift["polar_far_obs"]),
            "primary_polar_near": _json_scalar(primary_shift["polar_near_obs"]),
            "primary_delta": _json_scalar(
                primary_shift["polar_near_obs"] - primary_shift["polar_far_obs"]
            ),
            "primary_n_valid_shift": int(primary_shift["n_valid_shift"]),
            "primary_valid": primary_shift["status"] == "ok",
            "n_axis_present_windows": int(gate["present"].sum()),
            "sensitivity_polar_far": _json_scalar(sensitivity_shift["polar_far_obs"]),
            "sensitivity_polar_near": _json_scalar(sensitivity_shift["polar_near_obs"]),
            "sensitivity_delta": _json_scalar(
                sensitivity_shift["polar_near_obs"] - sensitivity_shift["polar_far_obs"]
            ),
            "sensitivity_n_valid_shift": int(sensitivity_shift["n_valid_shift"]),
            "sensitivity_valid": bool(
                gate["testable"] and sensitivity_shift["status"] == "ok"
            ),
            "far_side": primary_event["far_side"],
            "near_side": primary_event["near_side"],
            "event_class": primary_event["event_class"],
        })
        c_ab_matrix[row_index] = c_ab
        present_matrix[row_index] = gate["present"]
        within_shaft_p[row_index] = gate["within_shaft_p"]

    primary_result = subject_paired_null(
        primary_nulls, n_perm=N_PERM_SUBJECT, seed=subject_seed
    )
    sensitivity_result = subject_paired_null(
        sensitivity_nulls, n_perm=N_PERM_SUBJECT, seed=subject_seed + 1
    )
    primary_eligible = primary_result["n_valid_seizures"] >= N_VALID_SEIZURE_MIN
    sensitivity_eligible = bool(
        shaft_qc["testable"]
        and sensitivity_result["n_valid_seizures"] >= N_VALID_SEIZURE_MIN
    )

    pair = record["axis_pair"]
    relation = pair["relation"]
    summary = {
        "subject": subject,
        "status": "ok",
        "cohort_source": _display_path(COHORT_SOURCE),
        "n_requested_seizures": int(n_requested),
        "n_successful_seizures": int(len(seizures)),
        "coverage_fraction": float(len(seizures) / n_requested),
        "seizure_drops": drops,
        "n_joint": len(names),
        "joint_contact_names": names,
        "geometry_2d_supported": bool(pair.get("geometry_2d_supported")),
        "strict_2d": bool(pair.get("strict_stability_pass")),
        "axis_relation": relation.get("relation"),
        "axis_cosine": _json_scalar(relation.get("cosine")),
        "rank_contrast_rho_AB": _json_scalar(contrast["rho_AB"]),
        "rank_contrast_tier": contrast["template_pair_tier"],
        "shaft_qc": shaft_qc,
        "primary": _analysis_summary(primary_result, eligible=primary_eligible),
        "within_shaft_sensitivity": _analysis_summary(
            sensitivity_result, eligible=sensitivity_eligible
        ),
        "event_class_counts": {
            label: sum(row["event_class"] == label for row in rows)
            for label in ("selection", "switch", "persistent", "none")
        },
        "gate_nperm": int(gate_nperm),
        "subject_null_nperm": N_PERM_SUBJECT,
        "seed": int(seed),
        "subject_seed": int(subject_seed),
        "ranges_sec": {"far_pre": list(FAR_PRE), "near_onset": list(NEAR_ONSET)},
        "algorithm_version": ALGORITHM_VERSION,
        "permutation_coupling": PERMUTATION_COUPLING,
        "field_fingerprint_sha256": record["interictal_field"]["fingerprint_sha256"],
        "caveats": [
            "Primary C_AB is an ungated relative A/B template contrast and is defined for all own-field 2-D subjects.",
            "Only a reversed A/B axis supports source/sink reversal wording; same/different relations remain relative template contrast.",
            "The within-shaft axis-present result is a sensitivity analysis, not the full-cohort inclusion rule.",
            "Near-onset polarization does not by itself prove a literal A-to-B temporal switch.",
        ],
    }
    arrays = {
        "window_center_sec": seizures[0]["centers"],
        "seizure_idx": np.asarray([item["seizure_idx"] for item in seizures], int),
        "C_AB": c_ab_matrix,
        "axis_present": present_matrix,
        "within_shaft_p": within_shaft_p,
        "joint_contact_names": np.asarray(names, str),
        "primary_subject_null_delta": primary_result["null_delta"],
        "sensitivity_subject_null_delta": sensitivity_result["null_delta"],
    }
    return summary, rows, arrays


def _subject_stem(subject: str) -> str:
    return f"{subject}_fig3f_ab_dominance"


def _write_subject(summary: dict, rows: list[dict], arrays: dict | None) -> None:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    stem = _subject_stem(summary["subject"])
    (SUB_DIR / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    columns = [
        "subject", "seizure_idx", "primary_polar_far", "primary_polar_near",
        "primary_delta", "primary_n_valid_shift", "primary_valid",
        "n_axis_present_windows", "sensitivity_polar_far", "sensitivity_polar_near",
        "sensitivity_delta", "sensitivity_n_valid_shift", "sensitivity_valid",
        "far_side", "near_side", "event_class",
    ]
    with (SUB_DIR / f"{stem}_per_seizure.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if arrays is not None:
        np.savez_compressed(SUB_DIR / f"{stem}_timecourse.npz", **arrays)


def _cohort_row(summary: dict) -> dict:
    primary = summary.get("primary", {})
    sensitivity = summary.get("within_shaft_sensitivity", {})
    return {
        "subject": summary["subject"],
        "status": summary["status"],
        "axis_relation": summary.get("axis_relation"),
        "axis_cosine": summary.get("axis_cosine"),
        "strict_2d": summary.get("strict_2d"),
        "n_requested_seizures": summary.get("n_requested_seizures"),
        "n_successful_seizures": summary.get("n_successful_seizures"),
        "coverage_fraction": summary.get("coverage_fraction"),
        "n_joint": summary.get("n_joint"),
        "primary_eligible": primary.get("eligible"),
        "primary_polar_far": primary.get("polar_far"),
        "primary_polar_near": primary.get("polar_near"),
        "primary_delta": primary.get("delta"),
        "primary_p": primary.get("p"),
        "primary_null_p95": primary.get("null_p95"),
        "primary_subject_locked": primary.get("subject_locked"),
        "sensitivity_eligible": sensitivity.get("eligible"),
        "sensitivity_delta": sensitivity.get("delta"),
        "sensitivity_p": sensitivity.get("p"),
        "sensitivity_subject_locked": sensitivity.get("subject_locked"),
    }


def _binomial_summary(records: list[dict], key: str = "primary") -> dict:
    eligible = [record for record in records if record.get(key, {}).get("eligible")]
    k = sum(record[key].get("subject_locked") is True for record in eligible)
    n = len(eligible)
    if not n:
        return {
            "k_locked": 0,
            "n_eligible": 0,
            "one_sided_exact_binomial_p": None,
            "locked_fraction_exact_95ci": [None, None],
        }
    result = binomtest(k, n, p=0.05, alternative="greater")
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return {
        "k_locked": int(k),
        "n_eligible": int(n),
        "one_sided_exact_binomial_p": float(result.pvalue),
        "locked_fraction_exact_95ci": [float(ci.low), float(ci.high)],
    }


def _wilcoxon_greater(values: list[float]) -> dict:
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "statistic": None, "p_one_sided": None}
    try:
        result = wilcoxon(array, alternative="greater", method="auto")
        return {
            "n": int(len(array)),
            "statistic": float(result.statistic),
            "p_one_sided": float(result.pvalue),
        }
    except ValueError:
        return {"n": int(len(array)), "statistic": None, "p_one_sided": None}


def cohort_hierarchical_null(
    records: list[dict], null_draws: list[np.ndarray], *, n_perm: int, seed: int,
    key: str = "primary",
) -> dict:
    """Median-subject cohort test with independent subject-level time-null draws."""
    eligible = [record for record in records if record.get(key, {}).get("eligible")]
    if len(eligible) != len(null_draws):
        raise ValueError("eligible record/null-draw count mismatch")
    if not eligible:
        return {
            "n": 0, "median_delta": None, "null_p95": None,
            "p_one_sided": None, "null_delta": np.full(n_perm, np.nan),
        }
    observed = np.asarray([record[key]["delta"] for record in eligible], float)
    rng = np.random.default_rng(seed)
    cohort_null = np.full(n_perm, np.nan, float)
    for replicate in range(n_perm):
        draws = [values[rng.integers(len(values))] for values in null_draws]
        cohort_null[replicate] = np.median(draws)
    statistic = float(np.median(observed))
    return {
        "n": int(len(observed)),
        "median_delta": statistic,
        "null_p95": float(np.percentile(cohort_null, 95)),
        "p_one_sided": float(
            (1 + np.sum(cohort_null >= statistic)) / (n_perm + 1)
        ),
        "null_delta": cohort_null,
    }


def _relation_summary(records: list[dict]) -> dict:
    out = {}
    for relation in ("reversed", "same", "different"):
        values = [
            record["primary"]["delta"] for record in records
            if record.get("axis_relation") == relation
            and record.get("primary", {}).get("eligible")
        ]
        out[relation] = {
            "n": len(values),
            "median_delta": float(np.median(values)) if values else None,
            "wilcoxon_greater": _wilcoxon_greater(values),
        }
    return out


def _write_cohort(records: list[dict], gate_nperm: int, seed: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_cohort_row(record) for record in records]
    with COHORT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    eligible = [record for record in records if record.get("primary", {}).get("eligible")]
    primary_null_draws = []
    for record in eligible:
        npz = np.load(SUB_DIR / f"{_subject_stem(record['subject'])}_timecourse.npz")
        values = np.asarray(npz["primary_subject_null_delta"], float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise RuntimeError(f"{record['subject']}: empty primary subject null")
        primary_null_draws.append(values)
    cohort_null = cohort_hierarchical_null(
        records, primary_null_draws, n_perm=N_PERM_COHORT, seed=seed + 99173
    )
    sensitivity_eligible = [
        record for record in records
        if record.get("within_shaft_sensitivity", {}).get("eligible")
    ]
    sensitivity_null_draws = []
    for record in sensitivity_eligible:
        npz = np.load(SUB_DIR / f"{_subject_stem(record['subject'])}_timecourse.npz")
        values = np.asarray(npz["sensitivity_subject_null_delta"], float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise RuntimeError(f"{record['subject']}: empty sensitivity subject null")
        sensitivity_null_draws.append(values)
    sensitivity_cohort_null = cohort_hierarchical_null(
        records,
        sensitivity_null_draws,
        n_perm=N_PERM_COHORT,
        seed=seed + 27011,
        key="within_shaft_sensitivity",
    )
    primary_cohort_draws = cohort_null.pop("null_delta")
    sensitivity_cohort_draws = sensitivity_cohort_null.pop("null_delta")
    np.savez_compressed(
        OUT_DIR / "fig3f_ab_dominance_cohort_null.npz",
        cohort_null_delta=primary_cohort_draws,
        sensitivity_cohort_null_delta=sensitivity_cohort_draws,
        eligible_subjects=np.asarray([record["subject"] for record in eligible], str),
        sensitivity_eligible_subjects=np.asarray(
            [record["subject"] for record in sensitivity_eligible], str
        ),
    )
    deltas = [record["primary"]["delta"] for record in eligible]
    payload = {
        "contract": ALGORITHM_VERSION,
        "generated_by": "scripts/run_fig3f_ab_dominance_cohort.py",
        "cohort_source": _display_path(COHORT_SOURCE),
        "cohort_source_sha256": hashlib.sha256(COHORT_SOURCE.read_bytes()).hexdigest(),
        "scientific_question": (
            "Does relative A/B-template dominance become stronger near clinical onset "
            "than during far pre-ictal time?"
        ),
        "primary_statistic": (
            "per subject: median-seizure near |mean C_AB| minus median-seizure far "
            "|mean C_AB|; cohort: median across subjects"
        ),
        "primary_null": (
            "exhaustive non-zero circular shifts per seizure, independent seizure draws "
            "within subject, then independent subject draws for the cohort median"
        ),
        "cohort_null_version": COHORT_NULL_VERSION,
        "cohort_unit": "subject",
        "n_canonical_subjects": len(records),
        "n_primary_eligible": len(eligible),
        "n_by_axis_relation": {
            relation: sum(record.get("axis_relation") == relation for record in records)
            for relation in ("reversed", "same", "different")
        },
        "primary_cohort_hierarchical_time_null": cohort_null,
        "primary_wilcoxon_greater": _wilcoxon_greater(deltas),
        "primary_subject_locked_count": _binomial_summary(records, "primary"),
        "within_shaft_sensitivity_cohort_hierarchical_time_null": (
            sensitivity_cohort_null
        ),
        "within_shaft_sensitivity_wilcoxon_greater": _wilcoxon_greater([
            record["within_shaft_sensitivity"]["delta"]
            for record in sensitivity_eligible
        ]),
        "within_shaft_sensitivity_locked_count": _binomial_summary(
            records, "within_shaft_sensitivity"
        ),
        "relation_strata_descriptive": _relation_summary(records),
        "gate_nperm": int(gate_nperm),
        "subject_null_nperm": N_PERM_SUBJECT,
        "cohort_null_nperm": N_PERM_COHORT,
        "seed": int(seed),
        "subjects": records,
        "allowed_claim": (
            "The full-cohort test supports or does not support increased relative A/B-template "
            "dominance near onset. Source/sink reversal wording is restricted to the reversed-axis stratum."
        ),
    }
    COHORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def run_subject(subject: str, gate_nperm: int, seed: int) -> dict:
    if subject not in _cohort_subjects():
        raise ValueError(f"{subject}: not in canonical 17-subject cohort")
    summary, rows, arrays = compute_subject(subject, gate_nperm, seed)
    _write_subject(summary, rows, arrays)
    return summary


def _run_subject_safe(subject: str, gate_nperm: int, seed: int) -> dict:
    try:
        summary, rows, arrays = compute_subject(subject, gate_nperm, seed)
        _write_subject(summary, rows, arrays)
    except Exception as exc:
        summary = {
            "subject": subject,
            "status": "drop",
            "drop_reason": f"{type(exc).__name__}: {exc}",
            "primary": {"eligible": False, "subject_locked": False},
            "within_shaft_sensitivity": {
                "eligible": False, "subject_locked": False
            },
        }
        _write_subject(summary, [], None)
    return summary


def run_all(gate_nperm: int, seed: int, workers: int = 1) -> dict:
    subjects = _cohort_subjects()
    workers = max(1, min(int(workers), 2, len(subjects)))
    print(
        f"processing canonical {len(subjects)}-subject cohort with {workers} worker(s)",
        flush=True,
    )
    records_by_subject: dict[str, dict] = {}
    if workers == 1:
        for index, subject in enumerate(subjects, 1):
            started = time.time()
            summary = _run_subject_safe(subject, gate_nperm, seed)
            records_by_subject[subject] = summary
            print(
                f"[{index}/{len(subjects)}] {subject}: {summary['status']} "
                f"({time.time()-started:.1f}s)",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_subject_safe, subject, gate_nperm, seed): subject
                for subject in subjects
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                subject = futures[future]
                summary = future.result()
                records_by_subject[subject] = summary
                print(
                    f"[{index}/{len(subjects)}] {subject}: {summary['status']}",
                    flush=True,
                )
    records = [records_by_subject[subject] for subject in subjects]
    payload = _write_cohort(records, gate_nperm, seed)
    print(
        json.dumps(payload["primary_cohort_hierarchical_time_null"], indent=2),
        flush=True,
    )
    return payload


def rebuild_cohort(gate_nperm: int, seed: int) -> dict:
    records = []
    for subject in _cohort_subjects():
        path = SUB_DIR / f"{_subject_stem(subject)}_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text())
        if int(record.get("gate_nperm", -1)) != int(gate_nperm):
            raise RuntimeError(
                f"{subject}: gate_nperm={record.get('gate_nperm')} != {gate_nperm}"
            )
        if int(record.get("seed", -1)) != int(seed):
            raise RuntimeError(f"{subject}: seed={record.get('seed')} != {seed}")
        records.append(record)
    return _write_cohort(records, gate_nperm, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--rebuild-cohort", action="store_true")
    parser.add_argument("--gate-nperm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.gate_nperm < 19:
        raise ValueError("gate_nperm must be >=19 so p<0.05 is attainable")
    if args.all and args.rebuild_cohort:
        raise ValueError("choose only one of --all and --rebuild-cohort")
    if args.rebuild_cohort:
        payload = rebuild_cohort(args.gate_nperm, args.seed)
        print(json.dumps(payload["primary_cohort_hierarchical_time_null"], indent=2))
    elif args.all:
        run_all(args.gate_nperm, args.seed, workers=args.workers)
    else:
        print(
            json.dumps(
                run_subject(args.subject, args.gate_nperm, args.seed),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

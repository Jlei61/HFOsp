"""Core helpers for the T_spectral-aligned field-concordance analysis.

The module is deliberately filesystem-free.  It owns the parts of the analysis
whose contracts are easy to violate silently: time-zero conversion, mutually
exclusive phenotype selectors, complete-window construction, exact-name
activation alignment, fixed-per-seizure spatial permutations, repeated
mirror/template selection, subject-first folding, and time-wise sign-flip
correction.

The frozen interictal axes and fields are *only* consumed through
``src.topic5_template_axis_field``.  Nothing here fits an axis, plane, kernel,
bandwidth, support map, or template field from ictal data.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from src.propagation_skeleton_geometry import parse_shaft
from src.topic5_template_axis_field import (
    align_activation_to_interictal_field,
    score_field,
    score_scorer_bundle_batch,
)


ANALYSIS_CONTRACT = "topic5_tspectral_field_concordance_v1p1"
PHENOTYPE_LABEL_VERSION = "topic5_early_spectral_overlap_v3"
DISTAL_BASELINE_EEG_SEC = (-120.0, -90.0)
ANALYSIS_RANGE_SEC = (-30.0, 30.0)


@dataclass(frozen=True)
class WindowScale:
    name: str
    width_sec: float
    step_sec: float


WINDOW_SCALES = (
    WindowScale("2s_step0p5s", 2.0, 0.5),
    WindowScale("5s_step1s", 5.0, 1.0),
    WindowScale("10s_step2s", 10.0, 2.0),
    WindowScale("20s_step2s", 20.0, 2.0),
)

FIXED_WINDOWS = {
    "pre10": (-10.0, 0.0),
    "post10": (0.0, 10.0),
    "pre20": (-20.0, 0.0),
    "post20": (0.0, 20.0),
    "late20_30": (20.0, 30.0),
}


def eligibility_drop_reason(*, band_available: bool, field_status: str,
                            fingerprint_ok: bool, n_finite_contacts: int,
                            minimum_contacts: int = 6) -> str | None:
    """Return the first fail-closed data-eligibility reason, or ``None``."""
    if not band_available:
        return "missing_band"
    if str(field_status) != "ok":
        return "missing_axis_or_field"
    if not bool(fingerprint_ok):
        return "fingerprint_drift"
    if int(n_finite_contacts) < int(minimum_contacts):
        return "fewer_than_6_finite_contacts"
    return None


def tspectral_zeroed_times(times_rel_reference: Sequence[float],
                           t_spectral_rel_reference_sec: float) -> np.ndarray:
    """Convert a clinical/EEG-reference time grid to accepted T_spectral=0."""
    return np.asarray(times_rel_reference, float) - float(t_spectral_rel_reference_sec)


def tspectral_reference_for_raw_eeg(event_meta: Mapping[str, object]) -> float:
    """Return T_spectral in the raw extractor's EEG-onset coordinates.

    Epilepsiae cache zero is clinical onset, whereas ``extract_seizure_window``
    returns time relative to EEG onset.  These coordinates must not be mixed.
    """
    value = _finite_or_none(event_meta.get("t_spectral_rel_eeg_sec"))
    if value is None:
        raise ValueError("missing_t_spectral_rel_eeg_sec")
    return float(value)


def annotation_provenance(dataset: str, event_meta: Mapping[str, object]) -> Dict[str, object]:
    """Normalize annotation provenance without inventing Yuquan clinical onset."""
    dataset = str(dataset)
    clinical = event_meta.get("clinical_onset_rel_tspectral_sec")
    eeg = event_meta.get("eeg_onset_rel_tspectral_sec")
    if dataset == "yuquan":
        clinical = None
        return {
            "annotation_mode": "eeg_only",
            "cache_zero_reference": "eeg_onset",
            "clinical_onset_available": False,
            "clinical_onset_rel_tspectral_sec": None,
            "eeg_onset_rel_tspectral_sec": _finite_or_none(eeg),
        }
    return {
        "annotation_mode": "eeg_and_clinical",
        "cache_zero_reference": "clinical_onset",
        "clinical_onset_available": _finite_or_none(clinical) is not None,
        "clinical_onset_rel_tspectral_sec": _finite_or_none(clinical),
        "eeg_onset_rel_tspectral_sec": _finite_or_none(eeg),
    }


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def phenotype_selector_sets(cache_meta: Mapping[str, object]) -> Dict[str, set[int]]:
    """Return accepted selectors and fail if the two primary phenotypes overlap."""
    selectors = cache_meta.get("early_spectral_phenotype_selectors") or {}
    simple = selectors.get("accepted_tspectral_simple_phenotype_idxs") or {}
    accepted = {int(v) for v in cache_meta.get("seizure_idxs", [])}
    labeled = {int(v) for v in selectors.get("accepted_tspectral_labeled_idxs", [])}
    broadband = {
        int(v) for v in selectors.get("accepted_tspectral_strict_broadband_idxs", [])
    }
    gamma = {int(v) for v in simple.get("gamma_nonbroadband", [])}
    overlap = broadband & gamma
    if overlap:
        raise ValueError(f"broadband and gamma_nonbroadband selectors overlap: {sorted(overlap)}")
    for label, values in (("labeled", labeled), ("broadband", broadband), ("gamma", gamma)):
        extra = values - accepted
        if extra:
            raise ValueError(f"{label} selector contains non-accepted events: {sorted(extra)}")
    return {
        "accepted": accepted,
        "labeled": labeled,
        "broadband_1_150": broadband,
        "gamma_nonbroadband": gamma,
        "not_classified": accepted - labeled,
    }


def make_complete_window_grid(start_sec: float, stop_sec: float,
                              width_sec: float, step_sec: float) -> np.ndarray:
    """Return [start,end,center] rows fully contained in [start_sec, stop_sec]."""
    start_sec, stop_sec = float(start_sec), float(stop_sec)
    width_sec, step_sec = float(width_sec), float(step_sec)
    if width_sec <= 0 or step_sec <= 0 or stop_sec <= start_sec:
        raise ValueError("invalid window-grid parameters")
    last = stop_sec - width_sec
    if last < start_sec - 1e-12:
        return np.empty((0, 3), float)
    starts = start_sec + np.arange(int(np.floor((last - start_sec) / step_sec + 1e-9)) + 1) * step_sec
    ends = starts + width_sec
    keep = ends <= stop_sec + 1e-9
    return np.column_stack((starts[keep], ends[keep], (starts[keep] + ends[keep]) / 2.0))


def aggregate_complete_windows(values_contact_time: np.ndarray,
                               frame_centers_sec: Sequence[float],
                               windows: np.ndarray, *,
                               spectral_window_sec: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Average contact traces in windows whose spectral cells fit completely.

    A spectrogram value at time ``t`` represents a cell centered on ``t`` with
    width ``spectral_window_sec``.  Only cells fully inside the requested
    analysis window are used.  The returned boolean vector marks windows that
    also have complete source-data coverage and at least one cell.
    """
    values = np.asarray(values_contact_time, float)
    times = np.asarray(frame_centers_sec, float)
    wins = np.asarray(windows, float)
    if values.ndim != 2 or values.shape[1] != len(times):
        raise ValueError("values must have shape (contact,time) aligned to frame centers")
    if wins.ndim != 2 or wins.shape[1] < 2:
        raise ValueError("windows must have start/end columns")
    half = float(spectral_window_sec) / 2.0
    source_lo = float(np.nanmin(times) - half)
    source_hi = float(np.nanmax(times) + half)
    rows, complete = [], []
    for lo, hi in wins[:, :2]:
        is_complete = bool(lo >= source_lo - 1e-9 and hi <= source_hi + 1e-9)
        use = (times - half >= lo - 1e-9) & (times + half <= hi + 1e-9)
        is_complete = is_complete and bool(np.any(use))
        if is_complete:
            with np.errstate(invalid="ignore"):
                rows.append(np.nanmean(values[:, use], axis=1))
        else:
            rows.append(np.full(values.shape[0], np.nan))
        complete.append(is_complete)
    return np.asarray(rows, float), np.asarray(complete, bool)


def distal_baseline_robust_z(log_power_contact_time: np.ndarray,
                             frame_times_rel_tspectral: Sequence[float],
                             baseline_rel_tspectral: tuple[float, float], *,
                             min_frames: int = 50) -> Dict[str, object]:
    """Robust-z each contact using the seizure's real distal EEG baseline.

    ``delta`` explicitly subtracts the median robust-z level inside the same
    distal interval even though that value should be numerically near zero.
    """
    power = np.asarray(log_power_contact_time, float)
    times = np.asarray(frame_times_rel_tspectral, float)
    if power.ndim != 2 or power.shape[1] != len(times):
        raise ValueError("log power and time grid are not aligned")
    lo, hi = map(float, baseline_rel_tspectral)
    baseline_mask = (times >= lo - 1e-9) & (times <= hi + 1e-9)
    if int(baseline_mask.sum()) < int(min_frames):
        raise ValueError(
            f"distal baseline has {int(baseline_mask.sum())} frames; need {int(min_frames)}"
        )
    base = power[:, baseline_mask]
    med = np.nanmedian(base, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(base - med), axis=1, keepdims=True)
    mad = np.where(np.isfinite(mad) & (mad > 1e-12), mad, np.nan)
    z = (power - med) / mad
    baseline_z_center = np.nanmedian(z[:, baseline_mask], axis=1)
    delta = z - baseline_z_center[:, None]
    return {
        "legacy_z": z,
        "delta": delta,
        "baseline_mask": baseline_mask,
        "baseline_log_power_median": med[:, 0],
        "baseline_log_power_mad": mad[:, 0],
        "baseline_z_center": baseline_z_center,
        "n_baseline_frames": int(baseline_mask.sum()),
    }


def exact_name_align_matrix(record: Mapping[str, object], activation_names: Sequence[str],
                            activation_contact_time: np.ndarray) -> Dict[str, object]:
    """Align a contact-by-time matrix to the frozen field order by exact name."""
    values = np.asarray(activation_contact_time, float)
    if values.ndim != 2 or values.shape[0] != len(activation_names):
        raise ValueError("activation matrix must be contact-by-time")
    first = align_activation_to_interictal_field(record, activation_names, values[:, 0])
    out = np.full((int(first["n_target"]), values.shape[1]), np.nan)
    source = {str(name): i for i, name in enumerate(activation_names)}
    target = [str(v) for v in (record.get("interictal_field") or {}).get("contact_order", [])]
    for i, name in enumerate(target):
        if name in source:
            out[i] = values[source[name]]
    first = dict(first)
    first["values"] = out
    first["n_finite_rows"] = int(np.sum(np.isfinite(out).any(axis=1)))
    return first


def make_contact_permutations(contact_names: Sequence[str], matched_mask: Sequence[bool],
                              n_perm: int, seed: int, *, mode: str) -> np.ndarray:
    """Create fixed contact-identity permutations for one seizure.

    Unmatched contacts remain missing in place.  The returned permutation for a
    draw is reused for every time point, window, scale, distal baseline and band.
    """
    names = [str(v) for v in contact_names]
    matched = np.asarray(matched_mask, bool)
    if len(names) != len(matched):
        raise ValueError("contact names and matched mask differ in length")
    if mode not in {"all_contact", "within_shaft"}:
        raise ValueError(f"unknown permutation mode: {mode}")
    rng = np.random.default_rng(int(seed))
    base = np.arange(len(names), dtype=int)
    out = np.tile(base, (int(n_perm), 1))
    finite_idx = np.where(matched)[0]
    groups: list[np.ndarray]
    if mode == "all_contact":
        groups = [finite_idx]
    else:
        by_shaft: Dict[str, list[int]] = {}
        for idx in finite_idx:
            by_shaft.setdefault(parse_shaft(names[idx])[0], []).append(int(idx))
        groups = [np.asarray(v, int) for _, v in sorted(by_shaft.items())]
    for draw in range(int(n_perm)):
        for idx in groups:
            if len(idx) > 1:
                out[draw, idx] = rng.permutation(idx)
    return out


def apply_fixed_permutations(values_window_contact: np.ndarray,
                             permutations: np.ndarray) -> np.ndarray:
    """Apply each contact permutation to every window without resampling it."""
    values = np.asarray(values_window_contact, float)
    perms = np.asarray(permutations, int)
    if values.ndim != 2 or perms.ndim != 2 or values.shape[1] != perms.shape[1]:
        raise ValueError("window values and permutations are not contact-aligned")
    # (window, draw, contact) -> (draw, window, contact)
    return np.take(values, perms, axis=1).transpose(1, 0, 2)


def score_observed_bundle(scorers: Mapping[str, Mapping[str, object]],
                          activation: Sequence[float]) -> Dict[str, object]:
    """Score own/shared fields and retain mirror/template diagnostics."""
    out: Dict[str, object] = {}
    for name, scorer in scorers.items():
        score = score_field(scorer, activation)
        for key, value in score.items():
            out[f"{name}_{key}"] = value
        out[f"{name}_signed"] = score["signed_r"]
        out[f"{name}_abs"] = score["abs_r"]
    for prefix in ("own", "shared"):
        candidates = []
        for label in ("a", "b"):
            value = out.get(f"{prefix}_{label}_abs")
            if value is not None and np.isfinite(float(value)):
                candidates.append((label, float(value)))
        if candidates:
            best, value = max(candidates, key=lambda pair: pair[1])
            out[f"{prefix}_best_template"] = best.upper()
            out[f"{prefix}_maxab"] = value
    return out


def score_permutation_matrix(scorers: Mapping[str, Mapping[str, object]],
                             values_window_contact: np.ndarray,
                             permutations: np.ndarray, *,
                             chunk_draws: int = 100) -> Dict[str, np.ndarray]:
    """Recompute smoothing, mirror and TA/TB max selection for every draw/window."""
    values = np.asarray(values_window_contact, float)
    perms = np.asarray(permutations, int)
    pieces: Dict[str, list[np.ndarray]] = {}
    for i0 in range(0, len(perms), int(chunk_draws)):
        block_perm = perms[i0:i0 + int(chunk_draws)]
        permuted = apply_fixed_permutations(values, block_perm)
        flat = permuted.reshape(-1, values.shape[1])
        scored = score_scorer_bundle_batch(scorers, flat)
        for key, array in scored.items():
            pieces.setdefault(key, []).append(
                np.asarray(array, float).reshape(len(block_perm), len(values))
            )
    return {key: np.concatenate(blocks, axis=0) for key, blocks in pieces.items()}


def fold_seizure_null_draws(null_by_seizure: Sequence[np.ndarray]) -> np.ndarray:
    """Fold nulls event->subject for every draw before taking null summaries."""
    arrays = [np.asarray(v, float) for v in null_by_seizure]
    if not arrays:
        raise ValueError("no seizure null arrays to fold")
    shape = arrays[0].shape
    if any(v.shape != shape for v in arrays):
        raise ValueError("seizure null arrays must share (draw,window) shape")
    return np.nanmedian(np.stack(arrays, axis=0), axis=0)


def subject_first_fold(rows: Sequence[Mapping[str, object]], value_key: str,
                       group_keys: Sequence[str]) -> list[Dict[str, object]]:
    """Median-fold seizure rows inside subject; never pool seizures at cohort level."""
    grouped: Dict[tuple, list[float]] = {}
    counts: Dict[tuple, set[int]] = {}
    for row in rows:
        if "subject" not in row:
            raise ValueError("subject-first folding requires a subject key")
        key = tuple(row.get(k) for k in ("subject", *group_keys))
        value = _finite_or_none(row.get(value_key))
        if value is not None:
            grouped.setdefault(key, []).append(value)
        if row.get("seizure_idx") is not None:
            counts.setdefault(key, set()).add(int(row["seizure_idx"]))
    out = []
    for key, values in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        rec = {"subject": key[0]}
        rec.update({name: value for name, value in zip(group_keys, key[1:])})
        rec[value_key] = float(np.median(values))
        rec["n_seizures"] = len(counts.get(key, set()))
        out.append(rec)
    return out


def bootstrap_median_ci(values: Sequence[float], *, n_boot: int = 5000,
                        seed: int = 0) -> tuple[float, float]:
    """Subject bootstrap percentile CI for a cohort median."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    med = np.empty(int(n_boot), float)
    for i0 in range(0, int(n_boot), 1000):
        n = min(1000, int(n_boot) - i0)
        samples = x[rng.integers(0, len(x), size=(n, len(x)))]
        med[i0:i0 + n] = np.median(samples, axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def paired_sign_flip_p(values: Sequence[float], *, n_perm: int = 10000,
                       seed: int = 0) -> float:
    """Two-sided subject-level sign-flip p for the mean paired contrast."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    observed = abs(float(np.mean(x)))
    if x.size <= 15:
        code = np.arange(2 ** x.size, dtype=np.uint64)[:, None]
        bits = ((code >> np.arange(x.size, dtype=np.uint64)) & 1).astype(float)
        signs = bits * 2.0 - 1.0
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice((-1.0, 1.0), size=(int(n_perm), x.size))
    null = np.abs(np.mean(signs * x[None, :], axis=1))
    return float((1 + np.sum(null >= observed - 1e-15)) / (len(null) + 1))


def fixed_window_sign_flip_maxt(subject_window: np.ndarray, *, n_perm: int = 10000,
                                seed: int = 0) -> Dict[str, object]:
    """Two-sided sign-flip raw/maxT p values for fixed-window subject deltas.

    One sign is applied to every window of a subject.  This preserves the
    dependence induced by overlapping 10/20-s windows and makes the black
    cohort bar (the mean subject delta) the statistic that is actually tested.
    Complete fixed-window rows are required so the maxT family is identical at
    every permutation.
    """
    values = np.asarray(subject_window, float)
    if values.ndim != 2:
        raise ValueError("subject_window must be subject-by-window")
    complete = np.isfinite(values).all(axis=1)
    values = values[complete]
    n_subjects, n_windows = values.shape
    if n_subjects < 2:
        return {
            "n_subjects": int(n_subjects), "n_permutations": 0,
            "observed_mean": np.full(n_windows, np.nan),
            "raw_p": np.full(n_windows, np.nan),
            "maxt_p": np.full(n_windows, np.nan),
        }
    observed = np.mean(values, axis=0)
    if n_subjects <= 15:
        code = np.arange(2 ** n_subjects, dtype=np.uint64)[:, None]
        bits = ((code >> np.arange(n_subjects, dtype=np.uint64)) & 1).astype(float)
        signs = bits * 2.0 - 1.0
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice((-1.0, 1.0), size=(int(n_perm), n_subjects))
    null = np.abs((signs[:, :, None] * values[None, :, :]).mean(axis=1))
    max_null = np.max(null, axis=1)
    raw = np.asarray([
        (1 + np.sum(null[:, j] >= abs(observed[j]) - 1e-15)) / (len(null) + 1)
        for j in range(n_windows)
    ], float)
    maxt = np.asarray([
        (1 + np.sum(max_null >= abs(observed[j]) - 1e-15)) / (len(null) + 1)
        for j in range(n_windows)
    ], float)
    return {
        "n_subjects": int(n_subjects), "n_permutations": int(len(signs)),
        "observed_mean": observed, "raw_p": raw, "maxt_p": maxt,
    }


def independent_label_permutation_maxt(
        subject_window: np.ndarray, labels: Sequence[object], group_a: object,
        group_b: object, *, n_perm: int = 10000, seed: int = 0,
        exact_limit: int = 100000) -> Dict[str, object]:
    """Two-sided independent-group mean-difference permutation with maxT.

    Subjects outside ``group_a``/``group_b`` (for example the pre-existing
    ``same`` relation) are retained for descriptive reporting by callers but
    are not silently folded into this declared reversed-vs-different contrast.
    """
    values = np.asarray(subject_window, float)
    labels = np.asarray(labels, object)
    if values.ndim != 2 or len(labels) != len(values):
        raise ValueError("values/labels must be aligned subject rows")
    use = ((labels == group_a) | (labels == group_b)) & np.isfinite(values).all(axis=1)
    values, labels = values[use], labels[use]
    is_a = labels == group_a
    n_a, n_b = int(is_a.sum()), int((~is_a).sum())
    n_windows = values.shape[1]
    if n_a < 2 or n_b < 2:
        return {
            "n_group_a": n_a, "n_group_b": n_b, "n_permutations": 0,
            "observed_mean_difference": np.full(n_windows, np.nan),
            "raw_p": np.full(n_windows, np.nan),
            "maxt_p": np.full(n_windows, np.nan),
        }
    observed = values[is_a].mean(axis=0) - values[~is_a].mean(axis=0)
    total = comb(len(values), n_a)
    assignments = []
    if total <= int(exact_limit):
        for indices in combinations(range(len(values)), n_a):
            mask = np.zeros(len(values), bool)
            mask[list(indices)] = True
            assignments.append(mask)
    else:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(n_perm)):
            mask = np.zeros(len(values), bool)
            mask[rng.choice(len(values), size=n_a, replace=False)] = True
            assignments.append(mask)
    null = np.empty((len(assignments), n_windows), float)
    for i, mask in enumerate(assignments):
        null[i] = values[mask].mean(axis=0) - values[~mask].mean(axis=0)
    abs_null = np.abs(null)
    max_null = np.max(abs_null, axis=1)
    raw = np.asarray([
        (1 + np.sum(abs_null[:, j] >= abs(observed[j]) - 1e-15)) / (len(null) + 1)
        for j in range(n_windows)
    ], float)
    maxt = np.asarray([
        (1 + np.sum(max_null >= abs(observed[j]) - 1e-15)) / (len(null) + 1)
        for j in range(n_windows)
    ], float)
    return {
        "n_group_a": n_a, "n_group_b": n_b,
        "n_permutations": int(len(assignments)),
        "observed_mean_difference": observed, "raw_p": raw, "maxt_p": maxt,
    }


def _nan_t_stat(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, float)
    n = np.sum(np.isfinite(values), axis=0)
    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0, ddof=1)
    out = np.full(values.shape[1], np.nan)
    ok = (n >= 2) & np.isfinite(sd) & (sd > 1e-12)
    out[ok] = mean[ok] / (sd[ok] / np.sqrt(n[ok]))
    zero_sd = (n >= 2) & np.isfinite(sd) & (sd <= 1e-12) & (np.abs(mean) > 1e-12)
    out[zero_sd] = np.sign(mean[zero_sd]) * np.inf
    return out


def _clusters(mask: np.ndarray, statistic: np.ndarray) -> list[tuple[int, int, float, int]]:
    out = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        sign = 1 if np.nanmean(statistic[i:j]) >= 0 else -1
        mass = float(np.nansum(np.abs(statistic[i:j])))
        out.append((i, j, mass, sign))
        i = j
    return out


def sign_flip_cluster_maxt(subject_time: np.ndarray, time_centers: Sequence[float], *,
                           n_perm: int = 10000, seed: int = 0,
                           cluster_forming_alpha: float = 0.05) -> Dict[str, object]:
    """Two-sided sign-flip maxT/cluster correction across the full time series.

    One sign is drawn per subject and applied to all time points, preserving the
    observed within-subject temporal autocorrelation.
    """
    values = np.asarray(subject_time, float)
    times = np.asarray(time_centers, float)
    if values.ndim != 2 or values.shape[1] != len(times):
        raise ValueError("subject_time must be subject-by-time")
    complete_subject = np.isfinite(values).sum(axis=1) >= 2
    values = values[complete_subject]
    if len(values) < 2:
        return {"n_subjects": int(len(values)), "t_observed": np.full(len(times), np.nan),
                "maxt_p": np.full(len(times), np.nan), "clusters": []}
    t_obs = _nan_t_stat(values)
    df = max(1, int(np.nanmin(np.sum(np.isfinite(values), axis=0))) - 1)
    threshold = float(student_t.ppf(1.0 - float(cluster_forming_alpha) / 2.0, df))
    rng = np.random.default_rng(int(seed))
    n_perm = int(n_perm)
    max_t = np.empty(n_perm, float)
    max_cluster = np.empty(n_perm, float)
    for i0 in range(0, n_perm, 250):
        n = min(250, n_perm - i0)
        signs = rng.choice((-1.0, 1.0), size=(n, len(values)))
        for j in range(n):
            stat = _nan_t_stat(values * signs[j, :, None])
            max_t[i0 + j] = float(np.nanmax(np.abs(stat)))
            clusters = _clusters(np.abs(stat) >= threshold, stat)
            max_cluster[i0 + j] = max((c[2] for c in clusters), default=0.0)
    maxt_p = np.array([
        (1 + np.sum(max_t >= abs(v))) / (n_perm + 1) if np.isfinite(v) else np.nan
        for v in t_obs
    ])
    observed_clusters = []
    for i0, i1, mass, sign in _clusters(np.abs(t_obs) >= threshold, t_obs):
        p = float((1 + np.sum(max_cluster >= mass)) / (n_perm + 1))
        observed_clusters.append({
            "start_idx": int(i0),
            "end_idx_exclusive": int(i1),
            "start_center_sec": float(times[i0]),
            "end_center_sec": float(times[i1 - 1]),
            "n_windows": int(i1 - i0),
            "sign": "positive" if sign > 0 else "negative",
            "cluster_mass": mass,
            "corrected_p": p,
        })
    return {
        "n_subjects": int(len(values)),
        "n_permutations": n_perm,
        "cluster_forming_alpha_two_sided": float(cluster_forming_alpha),
        "cluster_forming_abs_t": threshold,
        "t_observed": t_obs,
        "maxt_p": maxt_p,
        "clusters": observed_clusters,
    }


def jsonable(value):
    """Convert NumPy-rich analysis records to strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    return value

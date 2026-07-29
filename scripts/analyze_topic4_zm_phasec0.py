#!/usr/bin/env python
"""Fail-closed Phase-C0 branch-identity adjudication from atomic SNN parts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.topic4_zm_phasec_metrics as PCM  # noqa: E402
import src.topic4_zm_phasec_contract as PCC  # noqa: E402
import src.topic4_zm_phasec_resources as PRES  # noqa: E402


OUT = os.path.join(
    ROOT, "results", "topic4_sef_hfo", "zm_phase_c_tonic_identity"
)
SEEDS = (1, 3, 4)
PHASES = ("bounded_mid__rising", "bounded_mid__peak")
NOISES = ("noise_replay", "noise_resample_1", "noise_resample_2")
GAIN_STATES = ("pre_entry__natural",) + PHASES
DELTAS = (0.05, 0.10)
N_BOOT = 5000
MANIFEST_PATH = os.path.join(OUT, "phasec_manifest.json")
PANELS_PATH = os.path.join(OUT, "phasec_panels.json")
HIERARCHICAL_SCHEMA = PCM.HIERARCHICAL_STATS_VERSION
RESOURCE_RECEIPT_INDEX_SCHEMA = (
    "zm_phasec_resource_receipt_index_v1_2026-07-29"
)
TIME_BLOCK_MS = 500.0
HIERARCHICAL_ARRAY_FIELDS = (
    "rho80_active_core_by_block_window",
    "block_isi_cv2_by_panel_neuron",
    "block_refractory_isi_numerator_by_stratum",
    "block_refractory_isi_denominator_by_stratum",
    "pair_corr_by_block_and_pair",
    "pair_null_median_by_block_and_draw",
    "active_area_fraction_by_block_window",
)
PANEL_ARRAY_FIELDS = ("analysis_panel_E_ids", "pairwise_panel_E_ids")
PAIR_NULL_META_FIELDS = (
    "pair_null_stratum_names",
    "refractory_isi_stratum_names",
)
SPATIAL_META_FIELDS = (
    "spatial_grid_n_occupied_E",
    "spatial_grid_all_E_bins_occupied",
    "spatial_area_denominator",
)
HIERARCHICAL_SCALAR_LOCKS = {
    "block_ms": 500.0,
    "ceiling_window_ms": 250.0,
    "ceiling_stride_ms": 50.0,
    "active_area_window_ms": 25.0,
    "pairwise_bin_ms": 5.0,
    "pairwise_null_draws": 100.0,
    "spatial_grid_n": 16.0,
    "spatial_active_floor_hz": 5.0,
}


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _npz_scalar(value):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError("expected scalar NPZ field")
    return arr.reshape(()).item()


def _resolve_artifact(path):
    p = Path(path)
    return p if p.is_absolute() else Path(ROOT) / p


def _object_sha(payload):
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_panels():
    panels = _load(PANELS_PATH)
    claimed = panels.get("manifest_sha256")
    body = {key: value for key, value in panels.items() if key != "manifest_sha256"}
    if not claimed or _object_sha(body) != claimed:
        raise ValueError("Phase-C panel manifest self-hash mismatch")
    if sorted(int(key) for key in panels.get("seeds", {})) != list(SEEDS):
        raise ValueError("Phase-C panel manifest seed coverage mismatch")
    for seed in SEEDS:
        row = panels["seeds"][str(seed)]
        row_claimed = row.get("panel_sha256")
        row_body = {key: value for key, value in row.items() if key != "panel_sha256"}
        if not row_claimed or _object_sha(row_body) != row_claimed:
            raise ValueError(f"Phase-C panel self-hash mismatch for seed {seed}")
        if row.get("activity_independent") is not True:
            raise ValueError(f"Phase-C panel is not activity-independent for seed {seed}")
    return panels


def _load_hierarchical_npz(row, *, expected_panel=None):
    """Load the preregistered block/neuron/pair sufficient statistics.

    Scalar summaries in ``identity.json`` are deliberately not accepted as a
    fallback.  Without these arrays the requested hierarchical uncertainty is
    not identifiable, so the only honest result is a technical block.
    """
    path_value = row.get("observables_path")
    if not path_value:
        return {"status": "blocked", "reason": "missing_observables_path"}
    path = _resolve_artifact(path_value)
    if not path.exists():
        return {"status": "blocked", "reason": "missing_observables_npz"}
    expected_sha = row.get("observables_sha256")
    if not expected_sha or _sha256(path) != expected_sha:
        return {"status": "blocked", "reason": "observables_sha256_mismatch"}
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [
                key for key in
                ("hierarchical_schema", *HIERARCHICAL_SCALAR_LOCKS,
                 *SPATIAL_META_FIELDS,
                 *HIERARCHICAL_ARRAY_FIELDS,
                 *PANEL_ARRAY_FIELDS, *PAIR_NULL_META_FIELDS)
                if key not in z.files
            ]
            if missing:
                return {
                    "status": "blocked",
                    "reason": "missing_npz_fields:" + ",".join(missing),
                }
            if str(_npz_scalar(z["hierarchical_schema"])) != HIERARCHICAL_SCHEMA:
                return {"status": "blocked", "reason": "hierarchical_schema_mismatch"}
            for key, expected in HIERARCHICAL_SCALAR_LOCKS.items():
                if not np.isclose(float(_npz_scalar(z[key])), expected):
                    return {"status": "blocked", "reason": f"{key}_mismatch"}
            arrays = {
                key: np.asarray(z[key], float)
                for key in HIERARCHICAL_ARRAY_FIELDS
            }
            panel_arrays = {
                key: np.asarray(z[key], int)
                for key in PANEL_ARRAY_FIELDS
            }
            spatial_meta = {
                key: _npz_scalar(z[key]) for key in SPATIAL_META_FIELDS
            }
            pair_null_meta = {
                "pair_null_stratum_names": tuple(
                    str(value) for value in np.asarray(
                        z["pair_null_stratum_names"]
                    ).ravel()
                ),
                "refractory_isi_stratum_names": tuple(
                    str(value) for value in np.asarray(
                        z["refractory_isi_stratum_names"]
                    ).ravel()
                ),
            }
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "blocked", "reason": f"invalid_observables_npz:{exc}"}

    block_count = None
    for key, value in arrays.items():
        expected_ndim = (
            3 if key == "pair_null_median_by_block_and_draw" else 2
        )
        if value.ndim != expected_ndim or value.shape[0] < 2:
            return {
                "status": "blocked",
                "reason": f"invalid_shape:{key}:{tuple(value.shape)}",
            }
        if value.shape[1] < 1:
            return {"status": "blocked", "reason": f"empty_sampling_axis:{key}"}
        block_count = value.shape[0] if block_count is None else block_count
        if value.shape[0] != block_count:
            return {"status": "blocked", "reason": "inconsistent_block_count"}
    cv2 = arrays["block_isi_cv2_by_panel_neuron"]
    ref_numerator = arrays[
        "block_refractory_isi_numerator_by_stratum"
    ]
    ref_denominator = arrays[
        "block_refractory_isi_denominator_by_stratum"
    ]
    expected_ref_shape = (
        int(block_count), len(PCM.REFRACTORY_ISI_STRATUM_NAMES)
    )
    if (
        ref_numerator.shape != expected_ref_shape
        or ref_denominator.shape != expected_ref_shape
    ):
        return {"status": "blocked", "reason": "refractory_count_shape_mismatch"}
    if (
        not np.all(np.isfinite(ref_numerator))
        or not np.all(np.isfinite(ref_denominator))
        or np.any(ref_numerator < 0)
        or np.any(ref_denominator < 0)
        or np.any(ref_numerator > ref_denominator)
        or not np.all(ref_numerator == np.floor(ref_numerator))
        or not np.all(ref_denominator == np.floor(ref_denominator))
    ):
        return {"status": "blocked", "reason": "invalid_refractory_isi_counts"}
    if arrays["rho80_active_core_by_block_window"].shape[1] != 6:
        return {"status": "blocked", "reason": "rho250_window_count_mismatch"}
    if arrays["active_area_fraction_by_block_window"].shape[1] != 20:
        return {"status": "blocked", "reason": "active_area_window_count_mismatch"}
    pair_null = arrays["pair_null_median_by_block_and_draw"]
    if pair_null.shape[1:] != (len(PCM.PAIR_NULL_STRATUM_NAMES), 100):
        return {"status": "blocked", "reason": "pairwise_null_draw_count_mismatch"}
    if pair_null_meta["pair_null_stratum_names"] != PCM.PAIR_NULL_STRATUM_NAMES:
        return {"status": "blocked", "reason": "pairwise_null_strata_mismatch"}
    if pair_null_meta["refractory_isi_stratum_names"] != (
        PCM.REFRACTORY_ISI_STRATUM_NAMES
    ):
        return {"status": "blocked", "reason": "refractory_isi_strata_mismatch"}
    n_grid = int(HIERARCHICAL_SCALAR_LOCKS["spatial_grid_n"])
    n_occupied = int(spatial_meta["spatial_grid_n_occupied_E"])
    if not 1 <= n_occupied <= n_grid * n_grid:
        return {"status": "blocked", "reason": "invalid_spatial_grid_occupancy"}
    if bool(spatial_meta["spatial_grid_all_E_bins_occupied"]) != (
        n_occupied == n_grid * n_grid
    ):
        return {"status": "blocked", "reason": "spatial_grid_occupancy_flag_mismatch"}
    if str(spatial_meta["spatial_area_denominator"]) != (
        "anatomy_occupied_E_grid_bins"
    ):
        return {"status": "blocked", "reason": "spatial_area_denominator_mismatch"}
    if expected_panel is not None:
        expected_ids = {
            "analysis_panel_E_ids": np.asarray(
                expected_panel["analysis_panel_E_ids"], int
            ),
            "pairwise_panel_E_ids": np.asarray(
                expected_panel["pairwise_panel_E_ids"], int
            ),
        }
        for key, expected in expected_ids.items():
            if not np.array_equal(panel_arrays[key], expected):
                return {"status": "blocked", "reason": f"{key}_mismatch"}
        n_analysis = panel_arrays["analysis_panel_E_ids"].size
        if cv2.shape[1] != n_analysis:
            return {"status": "blocked", "reason": "analysis_axis_does_not_match_panel"}
        n_analysis_core = int(expected_panel["analysis_panel_n_core"])
        if not 0 < n_analysis_core < n_analysis:
            return {"status": "blocked", "reason": "invalid_analysis_strata"}
        analysis_strata = np.r_[
            np.zeros(n_analysis_core, np.int8),
            np.ones(n_analysis - n_analysis_core, np.int8),
        ]
        n_pair_neuron = panel_arrays["pairwise_panel_E_ids"].size
        n_pair_core = int(expected_panel["pairwise_panel_n_core"])
        if not 0 < n_pair_core < n_pair_neuron:
            return {"status": "blocked", "reason": "invalid_pairwise_strata"}
        left, right = np.triu_indices(n_pair_neuron, k=1)
        pair_strata = (
            (left >= n_pair_core).astype(np.int8)
            + (right >= n_pair_core).astype(np.int8)
        )
        if pair_strata.size != arrays["pair_corr_by_block_and_pair"].shape[1]:
            return {"status": "blocked", "reason": "pair_axis_does_not_match_panel"}
        if set(np.unique(pair_strata).tolist()) != {0, 1, 2}:
            return {"status": "blocked", "reason": "missing_pairwise_stratum"}
    else:
        analysis_strata = np.zeros(
            arrays["block_isi_cv2_by_panel_neuron"].shape[1], np.int8
        )
        pair_strata = np.zeros(
            arrays["pair_corr_by_block_and_pair"].shape[1], np.int8
        )
    if block_count != 16:
        return {
            "status": "blocked",
            "reason": f"identity_block_count_mismatch:{block_count}!=16",
        }
    return {
        "status": "ok",
        "schema": HIERARCHICAL_SCHEMA,
        "n_blocks": int(block_count),
        "analysis_strata": analysis_strata,
        "pair_strata": pair_strata,
        **pair_null_meta,
        **arrays,
    }


def _identity_path(resolution, seed, phase, noise):
    return os.path.join(
        OUT, "parts", "c0_identity", resolution,
        f"seed{seed}", phase, noise, "identity.json",
    )


def _gain_path(resolution, seed, state, noise, delta, sign):
    label = (
        "d0_zero" if sign == 0
        else f"d{delta:g}_{'plus' if sign > 0 else 'minus'}"
    )
    return os.path.join(
        OUT, "parts", "c0_gain", resolution,
        f"seed{seed}", state, noise, label, "gain.json",
    )


def _resource_receipt_ref(path, *, manifest_sha256, task_key):
    """Return one live-validated immutable part/receipt binding."""
    receipt_path = PRES.resource_receipt_path(path)
    valid, reason, receipt = PRES.validate_resource_receipt(
        receipt_path,
        artifact_path=path,
        artifact_root=ROOT,
        manifest_sha256=manifest_sha256,
        task_key=task_key,
    )
    if not valid or not isinstance(receipt, dict):
        raise ValueError(reason)
    part = _load(path)
    ref = {
        "task_key": str(task_key),
        "part_path": os.path.relpath(path, ROOT),
        "part_file_sha256": _sha256(path),
        "resource_receipt_path": os.path.relpath(receipt_path, ROOT),
        "resource_receipt_file_sha256": _sha256(receipt_path),
        "resource_receipt_sha256": receipt["receipt_sha256"],
    }
    aux_ref = part.get("observables_path")
    aux_sha = part.get("observables_sha256")
    if aux_ref is not None or aux_sha is not None:
        aux_path = _resolve_artifact(aux_ref)
        if (
            not aux_path.is_file()
            or not isinstance(aux_sha, str)
            or _sha256(aux_path) != aux_sha
        ):
            raise ValueError("resource_index_aux_observables_drift")
        ref.update({
            "aux_observables_path": os.path.relpath(aux_path, ROOT),
            "aux_observables_file_sha256": aux_sha,
        })
    return ref


def _resource_receipt_failure(path, *, manifest_sha256, task_key):
    """Return the technical resource-audit failure for one production part."""
    try:
        _resource_receipt_ref(
            path,
            manifest_sha256=manifest_sha256,
            task_key=task_key,
        )
    except (OSError, TypeError, ValueError) as exc:
        return str(exc)
    return None


def build_resource_receipt_index(tasks, *, manifest_sha256):
    """Build a canonical full-part resource index without hiding blockers."""
    entries = []
    issues = []
    seen = set()
    logical = []
    normalized = [
        (
            str(row[0]),
            row[1],
            str(row[2]) if len(row) > 2 else "unspecified",
        )
        for row in tasks
    ]
    for task_key, path, role in sorted(
        normalized, key=lambda row: (row[0], row[2])
    ):
        logical.append({"task_key": task_key, "role": role})
        task_key = str(task_key)
        path = str(path)
        if task_key in seen:
            issues.append({
                "task_key": task_key,
                "part_path": os.path.relpath(path, ROOT),
                "reason": "duplicate_resource_task_key",
            })
            continue
        seen.add(task_key)
        if not os.path.isfile(path):
            issues.append({
                "task_key": task_key,
                "part_path": os.path.relpath(path, ROOT),
                "reason": "missing_part",
            })
            continue
        try:
            entries.append(_resource_receipt_ref(
                path,
                manifest_sha256=manifest_sha256,
                task_key=task_key,
            ))
        except (OSError, TypeError, ValueError) as exc:
            issues.append({
                "task_key": task_key,
                "part_path": os.path.relpath(path, ROOT),
                "reason": str(exc),
            })
    body = {
        "schema": RESOURCE_RECEIPT_INDEX_SCHEMA,
        "manifest_sha256": str(manifest_sha256),
        "status": "complete" if not issues else "incomplete",
        "expected_task_count": len(seen),
        "validated_entry_count": len(entries),
        "expected_logical_consumption_count": len(logical),
        "logical_consumptions": logical,
        "entries": entries,
        "issues": issues,
    }
    return {**body, "index_sha256": _object_sha(body)}


def _identity_task_key(seed, phase, noise):
    return f"identity|s{seed}|{phase}|{noise}"


def _gain_task_key(seed, state, noise, delta, sign):
    return (
        f"gain|s{seed}|{state}|{noise}|"
        f"d{float(delta):g}|{int(sign):+d}"
    )


def expected_paths(resolution, seeds):
    identities = [
        _identity_path(resolution, seed, phase, noise)
        for seed in seeds for phase in PHASES for noise in NOISES
    ]
    gains = [
        _gain_path(resolution, seed, state, noise, delta, sign)
        for seed in seeds for state in GAIN_STATES for noise in NOISES
        for delta in DELTAS for sign in (-1, 1)
    ]
    gains += [
        _gain_path(resolution, seed, state, noise, 0.0, 0)
        for seed in seeds for state in GAIN_STATES for noise in NOISES
    ]
    return identities, gains


def expected_resource_tasks(resolution, seeds):
    """Return the exact C0 part universe consumed by one summary."""
    tasks = [
        (
            _identity_task_key(seed, phase, noise),
            _identity_path(resolution, seed, phase, noise),
            "c0_identity",
        )
        for seed in seeds for phase in PHASES for noise in NOISES
    ]
    tasks.extend(
        (
            _gain_task_key(seed, state, noise, delta, sign),
            _gain_path(resolution, seed, state, noise, delta, sign),
            "c0_gain",
        )
        for seed in seeds for state in GAIN_STATES for noise in NOISES
        for delta in DELTAS for sign in (-1, 1)
    )
    tasks.extend(
        (
            _gain_task_key(seed, state, noise, 0.0, 0),
            _gain_path(resolution, seed, state, noise, 0.0, 0),
            "c0_gain",
        )
        for seed in seeds for state in GAIN_STATES for noise in NOISES
    )
    return tasks


def _finite_median(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def _stratified_resample_indices(n_items, strata, rng):
    labels = np.asarray(
        np.zeros(int(n_items), np.int8) if strata is None else strata
    )
    if labels.shape != (int(n_items),):
        raise ValueError("sampling strata do not align with sufficient-stat axis")
    pieces = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        pieces.append(members[rng.integers(0, members.size, size=members.size)])
    return np.concatenate(pieces)


def _matched_pair_summary(pair, null, pair_strata):
    """Compare each fixed-pair stratum with its matched shift-null draws.

    The null statistic is defined per circular-shift draw.  For every draw we
    first aggregate over the (possibly bootstrap-resampled) 500-ms blocks,
    then take Q0.975 only along the draw axis.  Flattening block and draw axes
    would make blocks act like extra null draws and changes the estimand.
    """
    pair = np.asarray(pair, float)
    null = np.asarray(null, float)
    strata = np.asarray(pair_strata, np.int8)
    if pair.ndim != 2 or strata.shape != (pair.shape[1],):
        raise ValueError("pair observations and strata do not align")
    if null.ndim != 3 or null.shape[1] != len(PCM.PAIR_NULL_STRATUM_NAMES):
        raise ValueError("pair null must be block x stratum x draw")
    if null.shape[0] != pair.shape[0]:
        raise ValueError("pair observation/null block axes do not align")
    excess = []
    null_q = []
    observed = []
    for stratum in range(len(PCM.PAIR_NULL_STRATUM_NAMES)):
        obs = pair[:, strata == stratum]
        nul = null[:, stratum, :]
        # Symmetric block estimator: the fixed-pair panel is reduced within
        # each shared block first.  The producer has already performed that
        # same pair reduction for every null draw.
        observed_by_block = np.asarray(
            [_finite_median(row) for row in obs], float
        )
        obs_median = _finite_median(observed_by_block)
        null_by_draw = np.asarray(
            [_finite_median(nul[:, draw]) for draw in range(nul.shape[1])],
            float,
        )
        null_by_draw = null_by_draw[np.isfinite(null_by_draw)]
        q975 = (
            float(np.percentile(null_by_draw, 97.5))
            if null_by_draw.size else np.nan
        )
        observed.append(obs_median)
        null_q.append(q975)
        excess.append(obs_median - q975)
    return {
        "pairwise_observed_median": _finite_median([
            _finite_median(row) for row in pair
        ]),
        "pairwise_null_q97_5": _finite_median(null_q),
        "pairwise_stratum_max_excess": (
            float(np.max(excess))
            if np.all(np.isfinite(excess)) else np.nan
        ),
    }


def _pooled_refractory_isi_probability(
    numerator, denominator, block=None, *, stratum="core"
):
    """Return P(ISI<=tau_ref+2dt) from pooled event counts.

    The core/surround columns are fixed anatomical strata, not equal-weight
    observations.  The decisive C0/C1 statistic uses the active-core analysis
    stratum; surround and all-panel ratios are supportive diagnostics only.
    """
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    if (
        numerator.ndim != 2
        or denominator.shape != numerator.shape
        or numerator.shape[1] != len(PCM.REFRACTORY_ISI_STRATUM_NAMES)
        or not np.all(np.isfinite(numerator))
        or not np.all(np.isfinite(denominator))
        or np.any(numerator < 0)
        or np.any(denominator < 0)
        or np.any(numerator > denominator)
    ):
        raise ValueError("invalid pooled refractory-ISI sufficient statistics")
    if block is not None:
        index = np.asarray(block, int)
        if index.ndim != 1 or np.any(index < 0) or np.any(index >= numerator.shape[0]):
            raise ValueError("invalid refractory-ISI block bootstrap index")
        numerator = numerator[index]
        denominator = denominator[index]
    try:
        if stratum == "all":
            selected = slice(None)
        else:
            selected = PCM.REFRACTORY_ISI_STRATUM_NAMES.index(str(stratum))
    except ValueError as exc:
        raise ValueError(f"unknown refractory-ISI stratum: {stratum}") from exc
    numerator = numerator[:, selected]
    denominator = denominator[:, selected]
    total = float(np.sum(denominator))
    return float(np.sum(numerator) / total) if total > 0 else np.nan


def _continuation_point(run):
    h = run["hierarchical"]
    rho = np.asarray(h["rho80_active_core_by_block_window"], float)
    cv2 = np.asarray(h["block_isi_cv2_by_panel_neuron"], float)
    ref_numerator = np.asarray(
        h["block_refractory_isi_numerator_by_stratum"], float
    )
    ref_denominator = np.asarray(
        h["block_refractory_isi_denominator_by_stratum"], float
    )
    pair = np.asarray(h["pair_corr_by_block_and_pair"], float)
    null = np.asarray(h["pair_null_median_by_block_and_draw"], float)
    area = np.asarray(h["active_area_fraction_by_block_window"], float)
    gain = np.asarray(run["gain_ratio_samples"], float)
    pair_summary = _matched_pair_summary(
        pair, null, h.get("pair_strata")
    )
    ref_surround = _pooled_refractory_isi_probability(
        ref_numerator, ref_denominator, stratum="surround"
    )
    ref_all = _pooled_refractory_isi_probability(
        ref_numerator, ref_denominator, stratum="all"
    )
    return {
        "rho80_active_core": _finite_median(rho),
        "gain_relative_to_preentry": _finite_median(gain),
        "isi_cv2_median": _finite_median(cv2),
        "refractory_isi_fraction": _pooled_refractory_isi_probability(
            ref_numerator, ref_denominator, stratum="core"
        ),
        "refractory_isi_fraction_surround_supportive":
            (float(ref_surround) if np.isfinite(ref_surround) else None),
        "refractory_isi_fraction_all_panel_supportive":
            (float(ref_all) if np.isfinite(ref_all) else None),
        **pair_summary,
        "active_area_fraction": _finite_median(area),
    }


def _resample_continuation(run, rng):
    """One draw: blocks/null draws, plus locked analysis neurons.

    Pairwise correlations use the entire fixed pair panel as a design census.
    Pairs share neurons and are therefore not an IID bootstrap axis.
    """
    h = run["hierarchical"]
    rho = np.asarray(h["rho80_active_core_by_block_window"], float)
    cv2 = np.asarray(h["block_isi_cv2_by_panel_neuron"], float)
    ref_numerator = np.asarray(
        h["block_refractory_isi_numerator_by_stratum"], float
    )
    ref_denominator = np.asarray(
        h["block_refractory_isi_denominator_by_stratum"], float
    )
    pair = np.asarray(h["pair_corr_by_block_and_pair"], float)
    null = np.asarray(h["pair_null_median_by_block_and_draw"], float)
    area = np.asarray(h["active_area_fraction_by_block_window"], float)
    gain = np.asarray(run["gain_ratio_samples"], float)

    n_block = rho.shape[0]
    block = rng.integers(0, n_block, size=n_block)
    analysis_strata = h.get("analysis_strata")
    pair_strata = h.get("pair_strata")
    cv_neuron = _stratified_resample_indices(
        cv2.shape[1], analysis_strata, rng
    )
    cv_draw = cv2[np.ix_(block, cv_neuron)]
    pair_draw = pair[block, :]
    null_draws = rng.integers(0, null.shape[2], size=null.shape[2])
    null_draw = null[np.ix_(
        block,
        np.arange(null.shape[1], dtype=int),
        null_draws,
    )]
    pair_summary = _matched_pair_summary(
        pair_draw, null_draw, np.asarray(pair_strata, np.int8)
    )
    gain_finite = gain[np.isfinite(gain)]
    gain_draw = (
        gain_finite[rng.integers(0, gain_finite.size, size=gain_finite.size)]
        if gain_finite.size else np.asarray([])
    )
    return {
        # rho is an all-active-core census rather than a sampled panel, so only
        # its 500-ms time blocks are resampled.
        "rho80_active_core": _finite_median(rho[block, :]),
        "gain_relative_to_preentry": _finite_median(gain_draw),
        "isi_cv2_median": _finite_median(cv_draw),
        "refractory_isi_fraction": _pooled_refractory_isi_probability(
            ref_numerator, ref_denominator, block=block, stratum="core"
        ),
        **pair_summary,
        "active_area_fraction": _finite_median(area[block, :]),
    }


def _interval(point, draws):
    x = np.asarray(draws, float)
    x = x[np.isfinite(x)]
    return {
        "point": None if not np.isfinite(point) else float(point),
        "lo": float(np.percentile(x, 2.5)) if x.size else None,
        "hi": float(np.percentile(x, 97.5)) if x.size else None,
        "n_boot_finite": int(x.size),
    }


def hierarchical_seed_bootstrap(runs, *, seed, n_boot=N_BOOT):
    """Three-level bootstrap locked by spec §4.5.

    Each selected continuation first resamples 500-ms blocks.  Analysis-panel
    neurons are resampled within locked core/surround strata for CV2;
    active-core f_ref is recomputed from pooled ISI-event counts over those
    blocks.  The pair panel is a complete, dependent design census (pairs
    share neurons), so it is held fixed while circular-null draws are
    resampled.  Six continuations are finally resampled within seed.  Seeds
    are never pooled inside this function.
    """
    by_phase = {
        phase: [index for index, run in enumerate(runs)
                if run.get("phase") == phase]
        for phase in PHASES
    }
    if any(len(by_phase[phase]) < 2 for phase in PHASES):
        raise ValueError(
            "hierarchical seed bootstrap requires at least two numeric "
            "continuations per fast phase"
        )
    required = set(HIERARCHICAL_ARRAY_FIELDS)
    for run in runs:
        h = run.get("hierarchical")
        if not isinstance(h, dict) or not required.issubset(h):
            raise ValueError("continuation lacks hierarchical sufficient statistics")
        if np.asarray(run.get("gain_ratio_samples", [])).size < 1:
            raise ValueError("continuation lacks gain-ratio block samples")

    keys = (
        "rho80_active_core",
        "gain_relative_to_preentry",
        "isi_cv2_median",
        "refractory_isi_fraction",
        "pairwise_observed_median",
        "pairwise_null_q97_5",
        "pairwise_stratum_max_excess",
        "active_area_fraction",
    )
    points_by_run = [_continuation_point(run) for run in runs]
    point = {
        key: _finite_median([row[key] for row in points_by_run])
        for key in keys
    }
    rng = np.random.default_rng(int(seed))
    draws = {key: np.full(int(n_boot), np.nan) for key in keys}
    for draw_index in range(int(n_boot)):
        # Fast phase is a fixed design stratum, not a random imbalance.  Draw
        # three noise continuations within each phase even when one of the
        # three terminal scientific outcomes has no numeric trajectory.
        selected = np.concatenate([
            rng.choice(by_phase[phase], size=len(NOISES), replace=True)
            for phase in PHASES
        ])
        rows = [_resample_continuation(runs[index], rng) for index in selected]
        for key in keys:
            draws[key][draw_index] = _finite_median([row[key] for row in rows])
    out = {
        key: _interval(point[key], draws[key])
        for key in keys
    }
    out.update({
        "structure": (
            "500ms_blocks_then_cv2_neurons_pooled_core_isi_counts_"
            "and_null_draws_"
            "with_fixed_pair_census_then_continuations"
        ),
        "n_numeric_continuations": len(runs),
        "n_drawn_continuations": len(PHASES) * len(NOISES),
        "n_boot": int(n_boot),
        "seed": int(seed),
    })
    return out


def classify_run_joint(run):
    """Apply the complete run-level conjunction before any aggregation."""
    if run.get("technical_block"):
        return {"klass": "blocked", "reason": run["technical_block"]}
    if run.get("scientific_failure"):
        return {"klass": "mixed", "reason": run["scientific_failure"]}
    try:
        values = _continuation_point(run)
    except (KeyError, TypeError, ValueError) as exc:
        return {"klass": "blocked", "reason": f"invalid_sufficient_stats:{exc}"}
    if run.get("gain_unresolved"):
        return {"klass": "mixed", "reason": "gain_unresolved", **values}
    decisive_keys = (
        "rho80_active_core",
        "gain_relative_to_preentry",
        "isi_cv2_median",
        "refractory_isi_fraction",
        "pairwise_observed_median",
        "pairwise_null_q97_5",
        "pairwise_stratum_max_excess",
        "active_area_fraction",
    )
    if not np.all(np.isfinite([values[key] for key in decisive_keys])):
        return {"klass": "blocked", "reason": "nonfinite_joint_metric"}
    if (
        run.get("runaway")
        or run.get("whole_sheet_plateau")
        or run.get("empirical_rest_dwell")
    ):
        return {"klass": "mixed", "reason": "invalid_carrier_dwell", **values}

    saturation = (
        values["rho80_active_core"] >= 0.50
        and (
            values["gain_relative_to_preentry"] <= 0.20
            or values["refractory_isi_fraction"] >= 0.80
        )
    )
    ai = (
        values["rho80_active_core"] <= 0.20
        and values["gain_relative_to_preentry"] >= 0.50
        and values["isi_cv2_median"] >= 0.70
        and abs(values["pairwise_observed_median"]) < 0.10
        and values["pairwise_stratum_max_excess"] < 0.0
        and values["active_area_fraction"] < 0.50
    )
    if saturation and not ai:
        klass = "refractory_saturated_branch"
    elif ai and not saturation:
        klass = "balanced_AI_tonic_candidate"
    else:
        klass = "mixed"
    return {"klass": klass, **values}


def phase_support_from_runs(runs):
    """Require at least two complete joint passes in each natural fast phase."""
    labels = (
        "refractory_saturated_branch",
        "balanced_AI_tonic_candidate",
    )
    classified = []
    for run in runs:
        result = classify_run_joint(run)
        classified.append({
            "phase": run.get("phase"),
            "noise": run.get("noise"),
            **result,
        })
    support = {}
    for label in labels:
        per_phase = {
            phase: sum(
                row["klass"] == label
                for row in classified if row["phase"] == phase
            )
            for phase in PHASES
        }
        support[label] = {
            "passes": all(per_phase[phase] >= 2 for phase in PHASES),
            "per_phase_pass_count": per_phase,
        }
    support["run_rows"] = classified
    return support


def _seed_class_from_hierarchy(runs, *, seed):
    classified = [classify_run_joint(run) for run in runs]
    blocked = [row for row in classified if row["klass"] == "blocked"]
    if blocked:
        return {
            "seed": int(seed),
            "klass": "C0_blocked_observables",
            "reason": blocked[0]["reason"],
        }
    phases = phase_support_from_runs(runs)
    numeric = [
        run for run, row in zip(runs, classified)
        if row["klass"] != "mixed" or not run.get("scientific_failure")
    ]
    if any(
        sum(run.get("phase") == phase for run in numeric) < 2
        for phase in PHASES
    ):
        return {
            "seed": int(seed),
            "klass": "mixed_or_indeterminate_tonic_branch",
            "reason": "fewer_than_two_numeric_continuations_in_a_fast_phase",
            "phase_support": phases,
        }
    ci = hierarchical_seed_bootstrap(numeric, seed=seed, n_boot=N_BOOT)
    sat = (
        ci["rho80_active_core"]["lo"] is not None
        and ci["rho80_active_core"]["lo"] >= 0.50
        and (
            (
                ci["gain_relative_to_preentry"]["hi"] is not None
                and ci["gain_relative_to_preentry"]["hi"] <= 0.20
            )
            or (
                ci["refractory_isi_fraction"]["lo"] is not None
                and ci["refractory_isi_fraction"]["lo"] >= 0.80
            )
        )
        and phases["refractory_saturated_branch"]["passes"]
    )
    ai = (
        ci["rho80_active_core"]["hi"] is not None
        and ci["rho80_active_core"]["hi"] <= 0.20
        and ci["gain_relative_to_preentry"]["lo"] is not None
        and ci["gain_relative_to_preentry"]["lo"] >= 0.50
        and ci["isi_cv2_median"]["lo"] is not None
        and ci["isi_cv2_median"]["lo"] >= 0.70
        and abs(ci["pairwise_observed_median"]["point"]) < 0.10
        and ci["pairwise_stratum_max_excess"]["hi"] is not None
        and ci["pairwise_stratum_max_excess"]["hi"] < 0.0
        and ci["active_area_fraction"]["hi"] is not None
        and ci["active_area_fraction"]["hi"] < 0.50
        and phases["balanced_AI_tonic_candidate"]["passes"]
    )
    if sat and not ai:
        klass = "refractory_saturated_branch"
    elif ai and not sat:
        klass = "balanced_AI_tonic_candidate"
    else:
        klass = "mixed_or_indeterminate_tonic_branch"
    return {
        "seed": int(seed),
        "klass": klass,
        "hierarchical_ci": ci,
        "phase_support": phases,
    }


def combine_resolution_summaries(native, dt2):
    """Glue the native verdict to its independent homologous dt/2 audit."""
    native_agg = (native or {}).get("aggregate") or {}
    verdict = native_agg.get("verdict")
    positive = {
        "refractory_saturated_branch_supported":
            "refractory_saturated_branch",
        "balanced_AI_tonic_candidate_supported":
            "balanced_AI_tonic_candidate",
    }
    if verdict not in positive:
        return {
            "verdict": verdict or "C0_no_evidence",
            "resolution_gate": "not_required_without_native_positive",
        }
    label = positive[verdict]
    supporting = set(native_agg.get("supporting_seeds") or [])
    if not {1, 3}.issubset(supporting):
        return {
            "verdict": "C0_no_evidence",
            "reason": "resolution_confirmation_unavailable",
            "resolution_gate": "insufficient_homologous_native_support",
            "native_verdict": verdict,
            "native_supporting_seeds": sorted(supporting),
            "required_dt2_seeds": [1, 3],
        }
    by_seed = {
        int(row["seed"]): row
        for row in (dt2 or {}).get("seed_rows", [])
        if isinstance(row, dict) and row.get("seed") is not None
    }
    opposite = {
        "refractory_saturated_branch": "balanced_AI_tonic_candidate",
        "balanced_AI_tonic_candidate": "refractory_saturated_branch",
    }[label]
    checked = [
        seed for seed in supporting
        if seed in by_seed and by_seed[seed].get("homologous_anchor_validated") is True
    ]
    if any(by_seed[seed].get("klass") == opposite for seed in checked):
        return {
            "verdict": "resolution_sensitive_identity",
            "native_verdict": verdict,
            "checked_seeds": sorted(checked),
        }
    agreeing = [seed for seed in checked if by_seed[seed].get("klass") == label]
    if len(agreeing) < 2:
        return {
            "verdict": "C0_no_evidence",
            "reason": "resolution_confirmation_unavailable",
            "resolution_gate": "insufficient_homologous_dt2_confirmation",
            "native_verdict": verdict,
            "agreeing_seeds": sorted(agreeing),
            "required_dt2_seeds": [1, 3],
        }
    return {
        "verdict": verdict,
        "resolution_gate": "passed",
        "dt2_supporting_seeds": sorted(agreeing),
    }


TECHNICAL_END_REASONS = {
    None,
    "truncated_or_missing_observable",
    "technical_exception",
    "schema_mismatch",
    "hash_mismatch",
}
SCIENTIFIC_END_REASONS = {
    "runaway",
    "whole_sheet_plateau",
    "empirical_rest_dwell",
}


def _part_failure_kind(row):
    if row.get("status") == "complete":
        return "ok"
    reason = row.get("scientific_end_reason")
    if reason in SCIENTIFIC_END_REASONS:
        return "scientific"
    return "technical"


def _gain_failure_kind(status):
    if status in {
        "nonlinear_or_nonmonotone",
        "scientific_failure_gain_arm",
        "scientific_failure_gain_baseline",
        "invalid_preentry_denominator",
        "gain_plateau_or_runaway",
    }:
        return "scientific"
    if status == "ok":
        return "ok"
    return "technical"


def _manifest_source(manifest, seed, state, noise, *, resolution="dt"):
    native_row = manifest["per_seed"][str(seed)]
    if resolution == "dt":
        seed_row = native_row
    elif resolution == "dt2":
        seed_row = native_row.get(
            "resolution_confirmations", {}
        ).get("dt2")
        if not isinstance(seed_row, dict):
            raise KeyError(f"seed {seed} lacks independent dt2 source lock")
        if (
            seed_row.get("parent_config_sha")
            != native_row["canonical_config_sha"]
        ):
            raise ValueError(f"seed {seed} dt2 parent-config drift")
    else:
        raise ValueError(f"unsupported resolution: {resolution}")
    if state == "pre_entry__natural":
        source = seed_row["c0_pre_entry_gain_control"]
        native_source = native_row["c0_pre_entry_gain_control"]
    else:
        phase = state.rsplit("__", 1)[-1]
        source = seed_row["c0_carrier_states"][phase]
        native_source = native_row["c0_carrier_states"][phase]
    banks = {
        bank["replicate"]: bank for bank in source["noise_banks"]
    }
    return {
        "state_hash": source["state"]["state_hash"],
        "state_file_sha256": source["state"]["file_sha256"],
        "noise_bank_sha": banks[noise]["bank_sha"],
        "config_sha": (
            native_row["canonical_config_sha"]
            if resolution == "dt" else seed_row["config_sha"]
        ),
        "native_state_hash": native_source["state"]["state_hash"],
        "parent_config_sha": native_row["canonical_config_sha"],
    }


def _runtime_provenance_failure(row, manifest, manifest_file_sha):
    """Return a fail-closed provenance reason, or ``None`` when exact.

    A semantic manifest hash alone is insufficient here: a part must also
    prove that it ran with the exact normalized manifest file and the complete
    locked producer map.  Missing fields are technical failures, never
    scientific negatives.
    """
    expected_producers = manifest.get("provenance", {}).get(
        "producer_file_sha256"
    )
    runtime = row.get("runtime_provenance")
    if not isinstance(expected_producers, dict) or not expected_producers:
        return "manifest_missing_producer_locks"
    if not isinstance(runtime, dict):
        return "missing_runtime_provenance"
    if runtime.get("manifest_sha256") != manifest.get("manifest_sha256"):
        return "runtime_manifest_semantic_sha_mismatch"
    if runtime.get("manifest_file_sha256") != manifest_file_sha:
        return "runtime_manifest_file_sha_mismatch"
    if runtime.get("producer_sha256") != expected_producers:
        return "runtime_producer_hash_mismatch"
    return None


def _validate_gain_payload(
    row, *, manifest, manifest_file_sha, seed, state, noise, resolution,
    expected,
):
    manifest_sha = manifest["manifest_sha256"]
    if row.get("manifest_sha256") != manifest_sha:
        return "gain_manifest_mismatch"
    identity_expected = {
        "seed": int(seed),
        "state_tag": state,
        "replicate": noise,
        "resolution": resolution,
    }
    if any(row.get(key) != value for key, value in identity_expected.items()):
        return "gain_identity_mismatch"
    if (
        row.get("state_hash") != expected["state_hash"]
        or row.get("state_file_sha256") != expected["state_file_sha256"]
        or row.get("noise_bank_sha") != expected["noise_bank_sha"]
        or row.get("config_sha") != expected["config_sha"]
    ):
        return "gain_provenance_mismatch"
    if resolution == "dt2" and (
        row.get("homologous_anchor_validated") is not True
        or row.get("homologous_native_state_hash")
        != expected["native_state_hash"]
        or row.get("homologous_parent_config_sha")
        != expected["parent_config_sha"]
    ):
        return "dt2_gain_homologous_anchor_unvalidated"
    provenance_failure = _runtime_provenance_failure(
        row, manifest, manifest_file_sha
    )
    if provenance_failure is not None:
        return provenance_failure
    kind = _part_failure_kind(row)
    if kind == "technical":
        return "technical_failure_gain_arm"
    if kind == "scientific":
        return "scientific_failure_gain_arm"
    if row.get("gain_plateau_gate_pass") is not True:
        return "gain_plateau_or_runaway"
    blocks = np.asarray(row.get("core_rate_500ms_hz", []), float)
    if blocks.ndim != 1 or blocks.size != 2 or not np.all(np.isfinite(blocks)):
        return "missing_gain_block_rates"
    return "ok"


def _gain_for(resolution, seed, state, noise, manifest):
    manifest_sha = manifest["manifest_sha256"]
    manifest_file_sha = _sha256(MANIFEST_PATH)
    source_expected = _manifest_source(
        manifest, seed, state, noise, resolution=resolution
    )
    baseline_path = _gain_path(resolution, seed, state, noise, 0.0, 0)
    if not os.path.exists(baseline_path):
        return {
            "status": "missing_gain_baseline",
            "failure_kind": "technical",
            "linearity_pass": False,
        }
    baseline = _load(baseline_path)
    receipt_failure = _resource_receipt_failure(
        baseline_path,
        manifest_sha256=manifest_sha,
        task_key=_gain_task_key(seed, state, noise, 0.0, 0),
    )
    if receipt_failure is not None:
        return {
            "status": receipt_failure,
            "failure_kind": "technical",
            "linearity_pass": False,
        }
    status = _validate_gain_payload(
        baseline, manifest=manifest, manifest_file_sha=manifest_file_sha,
        seed=seed, state=state, noise=noise, resolution=resolution,
        expected=source_expected,
    )
    if status != "ok":
        if status == "scientific_failure_gain_arm":
            status = "scientific_failure_gain_baseline"
        return {
            "status": status,
            "failure_kind": _gain_failure_kind(status),
            "linearity_pass": False,
        }
    points = []
    block_points = []
    for delta in DELTAS:
        minus_path = _gain_path(resolution, seed, state, noise, delta, -1)
        plus_path = _gain_path(resolution, seed, state, noise, delta, +1)
        if not (os.path.exists(minus_path) and os.path.exists(plus_path)):
            return {
                "status": "missing_gain_arm", "failure_kind": "technical",
                "linearity_pass": False,
            }
        minus, plus = _load(minus_path), _load(plus_path)
        for arm, arm_path, sign in (
            (minus, minus_path, -1), (plus, plus_path, +1)
        ):
            receipt_failure = _resource_receipt_failure(
                arm_path,
                manifest_sha256=manifest_sha,
                task_key=_gain_task_key(
                    seed, state, noise, delta, sign
                ),
            )
            if receipt_failure is not None:
                return {
                    "status": receipt_failure,
                    "failure_kind": "technical",
                    "linearity_pass": False,
                }
            status = _validate_gain_payload(
                arm, manifest=manifest,
                manifest_file_sha=manifest_file_sha, seed=seed, state=state,
                noise=noise, resolution=resolution,
                expected=source_expected,
            )
            if status != "ok":
                return {
                    "status": status,
                    "failure_kind": _gain_failure_kind(status),
                    "linearity_pass": False,
                }
        if not (
            minus.get("noise_bank_sha")
            == plus.get("noise_bank_sha")
            == baseline.get("noise_bank_sha")
        ):
            return {
                "status": "unpaired_gain_noise", "failure_kind": "technical",
                "linearity_pass": False,
            }
        points.append({
            "delta_mV": delta,
            "rate_vth_minus_hz": minus["core_rate_hz"],
            "rate_vth_plus_hz": plus["core_rate_hz"],
            "rate_baseline_hz": baseline["core_rate_hz"],
        })
        block_points.append((
            delta,
            np.asarray(minus["core_rate_500ms_hz"], float),
            np.asarray(plus["core_rate_500ms_hz"], float),
        ))
    out = PCM.paired_local_gain(points)
    if not out.get("linearity_pass"):
        out["failure_kind"] = _gain_failure_kind(out.get("status"))
        return out
    n_blocks = {minus.size for _delta, minus, _plus in block_points}
    n_blocks.update(plus.size for _delta, _minus, plus in block_points)
    n_blocks.add(np.asarray(baseline["core_rate_500ms_hz"]).size)
    if len(n_blocks) != 1:
        return {
            "status": "gain_block_count_mismatch",
            "failure_kind": "technical",
            "linearity_pass": False,
        }
    per_delta = [
        (minus - plus) / (2.0 * delta)
        for delta, minus, plus in block_points
    ]
    out["gain_hz_per_mV_blocks"] = np.median(np.vstack(per_delta), axis=0).tolist()
    out["failure_kind"] = "ok"
    return out


def _validate_identity_payload(
    row, *, manifest, manifest_file_sha, panels, seed, phase, noise,
    resolution,
):
    manifest_sha = manifest["manifest_sha256"]
    panel = panels["seeds"][str(seed)]
    if row.get("manifest_sha256") != manifest_sha:
        return "technical", "manifest_mismatch"
    expected_identity = {
        "seed": int(seed),
        "state_tag": phase,
        "replicate": noise,
        "resolution": resolution,
        "burn_in_ms": 500.0,
        "measure_ms": 8000.0,
        "evidence_value": "production",
        "panel_sha256": panel["panel_sha256"],
    }
    for key, expected in expected_identity.items():
        value = row.get(key)
        if isinstance(expected, float):
            if value is None or not np.isclose(float(value), expected):
                return "technical", f"identity_field_mismatch:{key}"
        elif value != expected:
            return "technical", f"identity_field_mismatch:{key}"
    expected_source = _manifest_source(
        manifest, seed, phase, noise, resolution=resolution
    )
    if (
        row.get("state_hash") != expected_source["state_hash"]
        or row.get("state_file_sha256")
        != expected_source["state_file_sha256"]
        or row.get("noise_bank_sha") != expected_source["noise_bank_sha"]
        or row.get("config_sha") != expected_source["config_sha"]
    ):
        return "technical", "identity_provenance_mismatch"
    if resolution == "dt2" and (
        row.get("homologous_anchor_validated") is not True
        or row.get("homologous_native_state_hash")
        != expected_source["native_state_hash"]
        or row.get("homologous_parent_config_sha")
        != expected_source["parent_config_sha"]
    ):
        return "technical", "dt2_homologous_anchor_unvalidated"
    provenance_failure = _runtime_provenance_failure(
        row, manifest, manifest_file_sha
    )
    if provenance_failure is not None:
        return "technical", provenance_failure
    kind = _part_failure_kind(row)
    if kind != "ok":
        return kind, str(row.get("scientific_end_reason"))
    gates = row.get("carrier_gates")
    if not isinstance(gates, dict) or any(
        key not in gates
        for key in ("runaway", "whole_sheet_plateau", "empirical_rest_dwell")
    ):
        return "technical", "missing_carrier_gates"
    return "ok", None


def _gain_ratio_samples(carrier, preentry):
    if carrier.get("failure_kind") == "technical":
        return "technical", None
    if preentry.get("failure_kind") == "technical":
        return "technical", None
    if not carrier.get("linearity_pass") or not preentry.get("linearity_pass"):
        return "scientific", None
    numerator = np.asarray(carrier.get("gain_hz_per_mV_blocks", []), float)
    denominator = np.asarray(preentry.get("gain_hz_per_mV_blocks", []), float)
    if (
        numerator.ndim != 1
        or denominator.ndim != 1
        or numerator.size < 2
        or numerator.shape != denominator.shape
    ):
        return "technical", None
    if (
        not np.all(np.isfinite(numerator))
        or not np.all(np.isfinite(denominator))
    ):
        return "technical", None
    if np.any(denominator <= 0):
        return "scientific", None
    return "ok", numerator / denominator


def _seed_summary(resolution, seed, manifest, panels):
    rows = []
    gain_cache = {}
    manifest_file_sha = _sha256(MANIFEST_PATH)
    try:
        for state in GAIN_STATES:
            for noise in NOISES:
                gain_cache[(state, noise)] = _gain_for(
                    resolution, seed, state, noise, manifest
                )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "seed": seed,
            "klass": "C0_blocked_observables",
            "reason": f"invalid_manifest_or_gain_provenance:{exc}",
        }
    for phase in PHASES:
        for noise in NOISES:
            path = _identity_path(resolution, seed, phase, noise)
            if not os.path.exists(path):
                return {
                    "seed": seed,
                    "klass": "C0_insufficient_coverage",
                    "reason": f"missing:{os.path.relpath(path, ROOT)}",
                }
            row = _load(path)
            receipt_failure = _resource_receipt_failure(
                path,
                manifest_sha256=manifest["manifest_sha256"],
                task_key=_identity_task_key(seed, phase, noise),
            )
            if receipt_failure is not None:
                return {
                    "seed": seed,
                    "klass": "C0_blocked_observables",
                    "reason": f"{phase}/{noise}:{receipt_failure}",
                }
            kind, reason = _validate_identity_payload(
                row, manifest=manifest,
                manifest_file_sha=manifest_file_sha, panels=panels,
                seed=seed, phase=phase, noise=noise,
                resolution=resolution,
            )
            if kind == "technical":
                return {
                    "seed": seed,
                    "klass": "C0_blocked_observables",
                    "reason": f"{phase}/{noise}:{reason}",
                }
            if kind == "scientific":
                rows.append({
                    "phase": phase,
                    "noise": noise,
                    "identity_path": os.path.relpath(path, ROOT),
                    "scientific_failure": reason,
                    "technical_block": None,
                    "runaway": reason == "runaway",
                    "whole_sheet_plateau": reason == "whole_sheet_plateau",
                    "empirical_rest_dwell": reason == "empirical_rest_dwell",
                })
                continue
            carrier_gain = gain_cache[(phase, noise)]
            pre_gain = gain_cache[("pre_entry__natural", noise)]
            gain_kind, gain_samples = _gain_ratio_samples(
                carrier_gain, pre_gain
            )
            if gain_kind == "technical":
                return {
                    "seed": seed,
                    "klass": "C0_blocked_observables",
                    "reason": (
                        f"{phase}/{noise}:technical_gain:"
                        f"{carrier_gain.get('status')}/"
                        f"{pre_gain.get('status')}"
                    ),
                }
            loaded = _load_hierarchical_npz(
                row,
                expected_panel=panels["seeds"][str(seed)],
            )
            if loaded["status"] != "ok":
                return {
                    "seed": seed,
                    "klass": "C0_blocked_observables",
                    "reason": f"{phase}/{noise}:{loaded['reason']}",
                }
            hierarchical = {
                key: loaded[key] for key in HIERARCHICAL_ARRAY_FIELDS
            }
            hierarchical.update({
                "analysis_strata": loaded["analysis_strata"],
                "pair_strata": loaded["pair_strata"],
                "pair_null_stratum_names": loaded[
                    "pair_null_stratum_names"
                ],
            })
            gates = row.get("carrier_gates") or {}
            rows.append({
                "phase": phase,
                "noise": noise,
                "identity_path": os.path.relpath(path, ROOT),
                "gain_carrier": carrier_gain,
                "gain_preentry": pre_gain,
                "hierarchical": hierarchical,
                "gain_ratio_samples": np.asarray(
                    gain_samples if gain_samples is not None else [np.nan]
                ),
                "gain_unresolved": gain_kind == "scientific",
                "scientific_failure": reason if kind == "scientific" else None,
                "runaway": bool(gates.get("runaway", False)),
                "whole_sheet_plateau": bool(
                    gates.get("whole_sheet_plateau", False)
                ),
                "empirical_rest_dwell": bool(
                    gates.get("empirical_rest_dwell", False)
                ),
            })
    out = _seed_class_from_hierarchy(rows, seed=seed)
    out["resolution"] = resolution
    out["homologous_anchor_validated"] = bool(resolution == "dt2")
    if resolution == "dt2":
        out["homologous_parent_config_sha"] = manifest["per_seed"][
            str(seed)
        ]["canonical_config_sha"]
        out["homologous_native_state_hashes"] = {
            phase: manifest["per_seed"][str(seed)]["c0_carrier_states"][
                phase.rsplit("__", 1)[-1]
            ]["state"]["state_hash"]
            for phase in PHASES
        }
    out["rows"] = [
        {
            key: value for key, value in row.items()
            if key not in {"hierarchical", "gain_ratio_samples"}
        }
        for row in rows
    ]
    return out


def _aggregate(seed_rows):
    blocked = [
        row for row in seed_rows
        if row["klass"] in {"C0_insufficient_coverage", "C0_blocked_observables"}
    ]
    if blocked:
        return {
            "verdict": "C0_no_evidence",
            "reason": "blocked_seed_coverage_or_observables",
            "blocked_seeds": [row["seed"] for row in blocked],
        }
    support_labels = ("refractory_saturated_branch", "balanced_AI_tonic_candidate")
    counts = {label: sum(row["klass"] == label for row in seed_rows)
              for label in support_labels}
    for label, count in counts.items():
        opposite = support_labels[1] if label == support_labels[0] else support_labels[0]
        if count >= 2 and not any(row["klass"] == opposite for row in seed_rows):
            return {
                "verdict": f"{label}_supported",
                "supporting_seeds": [row["seed"] for row in seed_rows
                                     if row["klass"] == label],
                "seed_classes": {str(row["seed"]): row["klass"] for row in seed_rows},
            }
    if counts[support_labels[0]] and counts[support_labels[1]]:
        verdict = "seed_heterogeneous_identity"
    else:
        verdict = "mixed_or_indeterminate_tonic_branch"
    return {
        "verdict": verdict,
        "seed_classes": {str(row["seed"]): row["klass"] for row in seed_rows},
    }


def analyze(resolution="dt", seeds=SEEDS):
    manifest = _load(MANIFEST_PATH)
    PCC.validate_manifest(manifest)
    panels = _load_panels()
    manifest_sha = manifest["manifest_sha256"]
    identity_paths, gain_paths = expected_paths(resolution, seeds)
    resource_tasks = expected_resource_tasks(resolution, seeds)
    missing = [path for path in identity_paths + gain_paths if not os.path.exists(path)]
    if missing:
        seed_rows = []
        aggregate = {
            "verdict": "C0_no_evidence",
            "reason": "missing_expected_parts",
            "n_missing": len(missing),
        }
    else:
        seed_rows = [
            _seed_summary(resolution, seed, manifest, panels) for seed in seeds
        ]
        aggregate = _aggregate(seed_rows)
    payload = {
        "schema": "zm_phasec_c0_summary_v1",
        "manifest_sha256": manifest_sha,
        "manifest_file_sha256": _sha256(MANIFEST_PATH),
        "panel_manifest_sha256": panels["manifest_sha256"],
        "resolution": resolution,
        "expected_identity_parts": len(identity_paths),
        "expected_gain_parts": len(gain_paths),
        "n_missing": len(missing),
        "missing": [os.path.relpath(path, ROOT) for path in missing],
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "bootstrap_draws": N_BOOT,
        "resource_receipt_index": build_resource_receipt_index(
            resource_tasks, manifest_sha256=manifest_sha
        ),
        "claim_boundary": (
            "source-space identity only; no observation match, entry, offset, "
            "recovery, actuator, or lifecycle claim"
        ),
    }
    path = os.path.join(OUT, f"c0_identity_summary_{resolution}.json")
    _write(path, payload)
    print(
        f"[phasec C0] resolution={resolution} verdict={aggregate['verdict']} "
        f"missing={len(missing)} -> {path}",
        flush=True,
    )
    return payload


def finalize_resolution_gate():
    native_path = os.path.join(OUT, "c0_identity_summary_dt.json")
    dt2_path = os.path.join(OUT, "c0_identity_summary_dt2.json")
    if not os.path.exists(native_path):
        payload = {
            "schema": "zm_phasec_c0_resolution_gate_v1",
            "verdict": "C0_no_evidence",
            "reason": "missing_native_summary",
        }
    else:
        native = _load(native_path)
        dt2 = _load(dt2_path) if os.path.exists(dt2_path) else None
        gate = combine_resolution_summaries(native, dt2)
        payload = {
            "schema": "zm_phasec_c0_resolution_gate_v1",
            **gate,
            "native_summary": os.path.relpath(native_path, ROOT),
            "native_summary_sha256": _sha256(native_path),
            "dt2_summary": (
                os.path.relpath(dt2_path, ROOT)
                if os.path.exists(dt2_path) else None
            ),
            "dt2_summary_sha256": (
                _sha256(dt2_path) if os.path.exists(dt2_path) else None
            ),
            "claim_boundary": (
                "source-space identity only; no observation match, entry, "
                "offset, recovery, actuator, or lifecycle claim"
            ),
        }
    path = os.path.join(OUT, "c0_identity_resolution_gate.json")
    _write(path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=("dt", "dt2"), default="dt")
    parser.add_argument(
        "--seeds", default=None,
        help="defaults to 1,3,4 at dt and the independent anchors 1,3 at dt2",
    )
    parser.add_argument("--finalize-resolution", action="store_true")
    args = parser.parse_args()
    if args.finalize_resolution:
        out = finalize_resolution_gate()
        print(f"[phasec C0 resolution] verdict={out['verdict']}", flush=True)
    else:
        seed_text = args.seeds or (
            "1,3,4" if args.resolution == "dt" else "1,3"
        )
        analyze(
            args.resolution, tuple(int(x) for x in seed_text.split(","))
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rev22-DCI Task 3: structure-only admissibility audit of the fixed-topology ellipse domain.

No simulation. For every fit topology seed the frozen reference substrate build is
intercepted exactly at the point where the producer applies the fixed-topology E-to-E
ellipse redistribution (before the learned local mapper), and the producer's own
reweighting function is evaluated on a dense (theta, AR) grid. Per grid point and
topology we record incoming-budget error, zero-denominator failures, the edge-ratio
distribution and the effective incoming source count relative to the reference, then
freeze the largest axis-aligned admissible rectangle containing the reference (45 deg, 2).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import src.topic4_rev20_dual_core_mechanism as mechanism  # noqa: E402
from src.topic4_core_connectivity import _hash_sparse_bins  # noqa: E402

REFERENCE_ANGLE_DEG = 45.0
REFERENCE_ASPECT_RATIO = 2.0
THRESHOLDS = {
    "budget_error_max": 1e-9,
    "edge_ratio_p01_min": 0.25,
    "edge_ratio_p99_max": 4.0,
    "effective_source_median_ratio_min": 0.75,
    "effective_source_p05_ratio_min": 0.50,
}
IDENTIFIABILITY_THRESHOLDS = {
    "theta_rank_correlation_min": 0.80,
    "theta_achieved_span_deg_min": 10.0,
    "theta_signal_to_topology_range_min": 2.0,
    "aspect_rank_correlation_min": 0.80,
    "log_aspect_achieved_span_min": 0.10,
    "aspect_signal_to_topology_range_min": 2.0,
}
FIT_SEEDS = (2511, 2512, 2513, 2514)


# --------------------------------------------------------------------------- #
# pure helpers (unit tested)
# --------------------------------------------------------------------------- #
def ee_edges(net) -> dict:
    """Canonical (bin, row, col, data) arrays of every E-to-E edge, sorted per bin."""
    n_e = int(net["NE"])
    bins, rows, cols, data = [], [], [], []
    for index, matrix in enumerate(net["ampa_by_delay"]):
        coo = matrix.tocoo(copy=False)
        keep = np.asarray(coo.row) < n_e
        row = np.asarray(coo.row[keep], np.int64)
        col = np.asarray(coo.col[keep], np.int64)
        values = np.asarray(coo.data[keep], float)
        order = np.lexsort((row, col))
        bins.append(np.full(len(order), index, np.int64))
        rows.append(row[order])
        cols.append(col[order])
        data.append(values[order])
    return {
        "n_e": n_e,
        "bin": np.concatenate(bins) if bins else np.zeros(0, np.int64),
        "row": np.concatenate(rows) if rows else np.zeros(0, np.int64),
        "col": np.concatenate(cols) if cols else np.zeros(0, np.int64),
        "data": np.concatenate(data) if data else np.zeros(0, float),
    }


def effective_source_count(rows: np.ndarray, data: np.ndarray, n_e: int) -> np.ndarray:
    """(sum w)^2 / sum(w^2) of incoming weights per E target; NaN when no input."""
    s1 = np.bincount(rows, weights=data, minlength=n_e)
    s2 = np.bincount(rows, weights=data * data, minlength=n_e)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s2 > 0.0, s1 * s1 / np.where(s2 > 0.0, s2, 1.0), np.nan)


def weighted_connection_geometry(dx: np.ndarray, dy: np.ndarray, data: np.ndarray) -> dict:
    """Second-moment geometry actually expressed by a weighted directed edge set.

    Orientation is axial (modulo 180 degrees).  The aspect statistic is the square root of
    the covariance eigenvalue ratio, so it is on the same scale as a Gaussian kernel's
    longitudinal/transverse length ratio; it is descriptive, not assumed equal to the
    requested fixed-topology parameter.
    """
    dx, dy, weight = np.asarray(dx, float), np.asarray(dy, float), np.asarray(data, float)
    if dx.shape != dy.shape or dx.shape != weight.shape or dx.ndim != 1:
        raise ValueError("edge displacement and weight arrays must be aligned vectors")
    if not len(weight) or np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ValueError("edge weights must be non-empty, finite and positive")
    total = float(np.sum(weight))
    moment = np.asarray([
        [np.sum(weight * dx * dx), np.sum(weight * dx * dy)],
        [np.sum(weight * dx * dy), np.sum(weight * dy * dy)],
    ], float) / total
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    if eigenvalues[0] <= 0.0 or not np.isfinite(eigenvalues).all():
        raise RuntimeError("weighted edge covariance is singular or non-finite")
    axis = eigenvectors[:, -1]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    return {
        "achieved_angle_deg": angle,
        "achieved_aspect_ratio": float(np.sqrt(eigenvalues[-1] / eigenvalues[0])),
        "weighted_rms_distance_mm": float(np.sqrt(np.trace(moment))),
        "axial_anisotropy": float((eigenvalues[-1] - eigenvalues[0]) / np.sum(eigenvalues)),
    }


def _axial_delta_deg(angle: float, reference: float) -> float:
    return float((float(angle) - float(reference) + 90.0) % 180.0 - 90.0)


def assess_achieved_geometry(per_topology: dict, rectangle: dict | None, theta: np.ndarray,
                             ar: np.ndarray, i_ref: int, j_ref: int,
                             thresholds: dict = IDENTIFIABILITY_THRESHOLDS) -> dict:
    """Determine whether requested geometry produces a topology-stable realized response."""
    if rectangle is None:
        return {"status": "GEOMETRY_ACHIEVED_RESPONSE_NOT_ESTIMABLE", "axes": {},
                "thresholds": dict(thresholds), "pass": False}
    i0, i1, j0, j1 = rectangle["i0"], rectangle["i1"], rectangle["j0"], rectangle["j1"]

    def record(seed, i, j):
        return next(row for row in per_topology[seed]["records"] if row["i"] == i and row["j"] == j)

    seeds = sorted(per_topology)
    reference_angles = {seed: record(seed, i_ref, j_ref)["achieved_angle_deg"] for seed in seeds}
    theta_requested = np.asarray(theta[i0:i1 + 1], float)
    theta_by_topology = np.asarray([
        [_axial_delta_deg(record(seed, i, j_ref)["achieved_angle_deg"], reference_angles[seed])
         for i in range(i0, i1 + 1)]
        for seed in seeds
    ], float)
    aspect_requested = np.log(np.asarray(ar[j0:j1 + 1], float))
    aspect_by_topology = np.log(np.asarray([
        [record(seed, i_ref, j)["achieved_aspect_ratio"] for j in range(j0, j1 + 1)]
        for seed in seeds
    ], float))

    def summarize(requested, achieved, *, span_key, rank_key, signal_key):
        from scipy.stats import spearmanr

        median = np.median(achieved, axis=0)
        span = float(np.ptp(median))
        topology_range = float(np.max(np.ptp(achieved, axis=0)))
        ratio = float(span / max(topology_range, 1e-12))
        rho = (float(spearmanr(requested, median).statistic)
               if len(requested) >= 2 and np.ptp(median) > 0.0 else None)
        checks = {
            "rank": rho is not None and rho >= thresholds[rank_key],
            "span": span >= thresholds[span_key],
            "signal_to_topology_range": ratio >= thresholds[signal_key],
        }
        return {"requested": requested.tolist(), "achieved_by_topology": achieved.tolist(),
                "achieved_median": median.tolist(), "rank_correlation": rho,
                "achieved_span": span, "maximum_topology_range": topology_range,
                "signal_to_topology_range": ratio, "checks": checks,
                "pass": bool(all(checks.values()))}

    theta_summary = summarize(
        theta_requested, theta_by_topology,
        span_key="theta_achieved_span_deg_min", rank_key="theta_rank_correlation_min",
        signal_key="theta_signal_to_topology_range_min",
    )
    aspect_summary = summarize(
        aspect_requested, aspect_by_topology,
        span_key="log_aspect_achieved_span_min", rank_key="aspect_rank_correlation_min",
        signal_key="aspect_signal_to_topology_range_min",
    )
    passed = theta_summary["pass"] and aspect_summary["pass"]
    return {"status": "GEOMETRY_ACHIEVED_RESPONSE_IDENTIFIABLE" if passed else
            "GEOMETRY_ACHIEVED_RESPONSE_NOT_IDENTIFIABLE",
            "axes": {"theta": theta_summary, "aspect": aspect_summary},
            "thresholds": dict(thresholds), "pass": bool(passed)}


def elliptical_radius(dx: np.ndarray, dy: np.ndarray, *, length_scale: float,
                      angle_deg: float, aspect_ratio: float) -> np.ndarray:
    """Same formula as ``mechanism._elliptical_radius``, on flat displacement arrays."""
    scale, aspect = float(length_scale), float(aspect_ratio)
    angle = np.deg2rad(float(angle_deg))
    if scale <= 0.0 or aspect < 1.0 or not np.isfinite([scale, aspect, angle]).all():
        raise ValueError("ellipse parameters are invalid")
    parallel = scale * np.sqrt(aspect)
    perpendicular = scale / np.sqrt(aspect)
    c, s = np.cos(angle), np.sin(angle)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return np.sqrt((u / parallel) ** 2 + (v / perpendicular) ** 2)


def reweight_edges(graph: dict, *, length_scale: float, angle_deg: float, aspect_ratio: float,
                   reference=(REFERENCE_ANGLE_DEG, REFERENCE_ASPECT_RATIO)):
    """Vectorized twin of the producer's fixed-topology reweighting.

    Identical operation order per edge (log-ratio, clip to +-20, exponentiate, rescale by
    the target's incoming total over the reweighted denominator); only the bin loop is
    flattened. ``validate_fast_path`` checks it against the producer itself.
    """
    n_e = graph["n_e"]
    rows, data = graph["row"], graph["data"]
    log_ratio = (
        -elliptical_radius(graph["dx"], graph["dy"], length_scale=length_scale,
                           angle_deg=angle_deg, aspect_ratio=aspect_ratio)
        + elliptical_radius(graph["dx"], graph["dy"], length_scale=length_scale,
                            angle_deg=float(reference[0]), aspect_ratio=float(reference[1]))
    )
    raw_ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
    incoming = np.bincount(rows, weights=data, minlength=n_e)
    denominator = np.bincount(rows, weights=data * raw_ratio, minlength=n_e)
    if np.any((incoming > 0.0) & (denominator <= 0.0)):
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(denominator > 0.0, incoming / np.where(denominator > 0.0, denominator, 1.0), 0.0)
    return data * raw_ratio * scale[rows]


def validate_fast_path(net, positions, graph: dict, base_edges: dict, points, *,
                       length_scale: float, tolerance: float = 1e-10,
                       reference=(REFERENCE_ANGLE_DEG, REFERENCE_ASPECT_RATIO)) -> list[dict]:
    """Assert the vectorized reweighting reproduces the producer at sample grid points."""
    checks = []
    for angle_deg, aspect_ratio in points:
        new_net, audit = mechanism.fixed_topology_ee_ellipse_redistribution(
            net, positions, length_scale=length_scale, angle_deg=float(angle_deg),
            aspect_ratio=float(aspect_ratio), reference_angle_deg=float(reference[0]),
            reference_aspect_ratio=float(reference[1]),
        )
        produced = ee_edges(new_net)
        if not (np.array_equal(produced["bin"], base_edges["bin"])
                and np.array_equal(produced["row"], base_edges["row"])
                and np.array_equal(produced["col"], base_edges["col"])):
            raise RuntimeError("producer changed the E-to-E edge structure")
        fast = reweight_edges(graph, length_scale=length_scale, angle_deg=float(angle_deg),
                              aspect_ratio=float(aspect_ratio), reference=reference)
        if fast is None:
            raise RuntimeError("fast path reported a zero denominator where the producer did not")
        difference = np.max(np.abs(fast - produced["data"]))
        relative = float(difference / max(np.max(np.abs(produced["data"])), 1e-30))
        checks.append({"angle_deg": float(angle_deg), "aspect_ratio": float(aspect_ratio),
                       "max_abs_difference": float(difference), "max_relative_difference": relative,
                       "producer_exact_noop": bool(audit.get("exact_noop", False)),
                       "producer_topology_unchanged": bool(audit.get("topology_unchanged", True)),
                       "producer_gaba_unchanged": bool(audit.get("gaba_unchanged", True))})
        if relative > tolerance:
            raise RuntimeError(f"fast path diverges from the producer at ({angle_deg}, {aspect_ratio}): "
                               f"relative {relative:.3e}")
        del new_net, produced, fast
    return checks


def audit_grid_point(graph: dict, base_edges: dict, base_effective: np.ndarray, *,
                     length_scale: float, angle_deg: float, aspect_ratio: float,
                     reference=(REFERENCE_ANGLE_DEG, REFERENCE_ASPECT_RATIO)) -> dict:
    """Summarize the structural consequences of one (theta, AR) reweighting."""
    is_reference = (float(angle_deg) == float(reference[0])
                    and float(aspect_ratio) == float(reference[1]))
    new_data = (base_edges["data"] if is_reference else reweight_edges(
        graph, length_scale=length_scale, angle_deg=angle_deg, aspect_ratio=aspect_ratio,
        reference=reference))
    if new_data is None:
        return {
            "angle_deg": float(angle_deg), "aspect_ratio": float(aspect_ratio),
            "status": "ZERO_DENOMINATOR", "error": "zero normalization denominator",
            "exact_noop": False, "budget_error_max": None, "zero_denominator": True,
            "edge_ratio_p01": None, "edge_ratio_p50": None, "edge_ratio_p99": None,
            "effective_source_median_ratio": None, "effective_source_p05_ratio": None,
            "targets_effective_ratio_below_0p5": None,
            "topology_unchanged": None, "gaba_unchanged": None,
            "achieved_angle_deg": None, "achieved_aspect_ratio": None,
            "weighted_rms_distance_mm": None, "axial_anisotropy": None,
        }
    n_e = base_edges["n_e"]
    ratio = new_data / base_edges["data"]
    incoming_old = np.bincount(base_edges["row"], weights=base_edges["data"], minlength=n_e)
    incoming_new = np.bincount(base_edges["row"], weights=new_data, minlength=n_e)
    budget_error = float(np.max(np.abs(incoming_new - incoming_old), initial=0.0))
    effective = effective_source_count(base_edges["row"], new_data, n_e)
    valid = np.isfinite(effective) & np.isfinite(base_effective) & (base_effective > 0)
    eff_ratio = effective[valid] / base_effective[valid]
    geometry = weighted_connection_geometry(graph["dx"], graph["dy"], new_data)
    return {
        "angle_deg": float(angle_deg), "aspect_ratio": float(aspect_ratio),
        "status": "OK", "error": None,
        "exact_noop": bool(is_reference),
        "budget_error_max": budget_error,
        "zero_denominator": False,
        "edge_ratio_p01": float(np.quantile(ratio, 0.01)),
        "edge_ratio_p50": float(np.quantile(ratio, 0.50)),
        "edge_ratio_p99": float(np.quantile(ratio, 0.99)),
        "effective_source_median_ratio": float(np.median(eff_ratio)),
        "effective_source_p05_ratio": float(np.quantile(eff_ratio, 0.05)),
        "targets_effective_ratio_below_0p5": int(np.sum(eff_ratio < 0.5)),
        "topology_unchanged": True, "gaba_unchanged": True,
        **geometry,
    }


def grid_point_passes(record: dict, thresholds: dict = THRESHOLDS) -> bool:
    if record["status"] != "OK" or record["zero_denominator"]:
        return False
    if not record["topology_unchanged"] or not record["gaba_unchanged"]:
        return False
    return (
        record["budget_error_max"] <= thresholds["budget_error_max"]
        and record["edge_ratio_p01"] >= thresholds["edge_ratio_p01_min"]
        and record["edge_ratio_p99"] <= thresholds["edge_ratio_p99_max"]
        and record["effective_source_median_ratio"] >= thresholds["effective_source_median_ratio_min"]
        and record["effective_source_p05_ratio"] >= thresholds["effective_source_p05_ratio_min"]
    )


def largest_admissible_rectangle(pass_mask: np.ndarray, i_ref: int, j_ref: int,
                                 theta_values: np.ndarray | None = None,
                                 reference_angle: float = REFERENCE_ANGLE_DEG) -> dict | None:
    """Largest all-pass axis-aligned rectangle (in grid cells) containing (i_ref, j_ref).

    Axis 0 is theta, axis 1 is aspect ratio. Ties in cell count prefer the rectangle
    whose theta range is most symmetric around the reference angle, then the larger
    theta span, then the earliest enumeration.
    """
    mask = np.asarray(pass_mask, bool)
    if not mask[i_ref, j_ref]:
        return None
    n_i, n_j = mask.shape
    prefix = np.zeros((n_i + 1, n_j + 1), np.int64)
    prefix[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)

    def all_pass(i0, i1, j0, j1):
        total = prefix[i1 + 1, j1 + 1] - prefix[i0, j1 + 1] - prefix[i1 + 1, j0] + prefix[i0, j0]
        return total == (i1 - i0 + 1) * (j1 - j0 + 1)

    theta = np.arange(n_i, dtype=float) if theta_values is None else np.asarray(theta_values, float)
    ref_theta = float(theta[i_ref]) if theta_values is None else float(reference_angle)
    best, best_key = None, None
    for i0 in range(i_ref, -1, -1):
        for i1 in range(i_ref, n_i):
            for j0 in range(j_ref, -1, -1):
                for j1 in range(j_ref, n_j):
                    if not all_pass(i0, i1, j0, j1):
                        continue
                    cells = (i1 - i0 + 1) * (j1 - j0 + 1)
                    asymmetry = abs((theta[i0] + theta[i1]) - 2.0 * ref_theta)
                    key = (cells, -asymmetry, i1 - i0)
                    if best_key is None or key > best_key:
                        best, best_key = (i0, i1, j0, j1), key
    i0, i1, j0, j1 = best
    return {"i0": i0, "i1": i1, "j0": j0, "j1": j1, "cells": int(best_key[0]),
            "theta_asymmetry": float(-best_key[1])}


# --------------------------------------------------------------------------- #
# substrate interception
# --------------------------------------------------------------------------- #
class _Captured(Exception):
    pass


def capture_reference_ee_graph(seed: int, *, artifact_root: Path, rev20_config: dict,
                               node_field: dict) -> dict:
    """Rebuild the frozen reference substrate and intercept the producer's ellipse call.

    The wrapper runs the producer's reference (45 deg, AR 2) call -- an exact no-op --
    and records the net/positions handed to it. The learned mapper that follows in
    ``build_substrate`` never touches the pre-mapping E-to-E bins, so this is the graph
    the producer reweights. The build is aborted after capture to save time.
    """
    from src.topic4_zm_ictal_transition import build_substrate, load_round_config

    transition_path = ROOT / rev20_config["inputs"]["transition_config"]["path"]
    if hashlib.sha256(transition_path.read_bytes()).hexdigest() != rev20_config["inputs"]["transition_config"]["sha256"]:
        raise RuntimeError("transition config hash changed")
    transition = load_round_config(transition_path)
    reference = rev20_config["reference"]
    cache_dir = artifact_root / rev20_config["network_cache"]
    captured: dict = {}
    original = mechanism.fixed_topology_ee_ellipse_redistribution

    def _capture(net, positions, **kwargs):
        out = original(net, positions, **kwargs)
        captured.update(net=net, positions=np.asarray(positions, float), kwargs=dict(kwargs),
                        reference_audit=out[1])
        raise _Captured()

    mechanism.fixed_topology_ee_ellipse_redistribution = _capture
    try:
        build_substrate(
            transition, str(reference["base_substrate_candidate_id"]), int(seed),
            cache_dir=str(cache_dir), ee_dose=float(reference["g_EE"]),
            etoi_dose=float(reference["g_EtoI"]), node_candidate_override=dict(node_field),
            node_depth_shrinkage=float(reference["signed_depth_shrinkage"]),
            node_gain=float(reference["node_gain"]),
            ee_ellipse_angle_deg=float(reference["ellipse_angle_deg"]),
            ee_ellipse_aspect_ratio=float(reference["ellipse_aspect_ratio"]),
            artifact_root=artifact_root,
        )
    except _Captured:
        pass
    finally:
        mechanism.fixed_topology_ee_ellipse_redistribution = original
    if "net" not in captured:
        raise RuntimeError("producer never reached the ellipse redistribution point")
    if not captured["reference_audit"].get("exact_noop"):
        raise RuntimeError("reference call was not an exact no-op")
    return captured


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _grid(theta_half_width, theta_step, ar_range, ar_step, reference):
    """Angle grid as exact offsets around the reference so the reference cell is exact."""
    n_half = int(round(float(theta_half_width) / float(theta_step)))
    offsets = np.arange(-n_half, n_half + 1, dtype=float) * float(theta_step)
    theta = float(reference[0]) + offsets
    theta[n_half] = float(reference[0])  # offset 0 -> bit-exact reference angle
    ar = np.round(np.arange(ar_range[0], ar_range[1] + 1e-9, ar_step), 6)
    i_ref = n_half
    j_ref = int(np.flatnonzero(np.isclose(ar, float(reference[1])))[0])
    if float(ar[j_ref]) != float(reference[1]):
        raise ValueError("aspect-ratio grid must contain the reference aspect ratio exactly")
    return theta, ar, i_ref, j_ref


def audit_one_topology(seed: int, *, artifact_root: Path, rev20_config: dict,
                       node_field: dict, theta: np.ndarray, ar: np.ndarray,
                       i_ref: int, j_ref: int, started: float,
                       reference=(REFERENCE_ANGLE_DEG, REFERENCE_ASPECT_RATIO)) -> tuple[int, dict]:
    """Run the full dense-grid structural audit for one independent topology."""
    t0 = time.time()
    captured = capture_reference_ee_graph(
        seed, artifact_root=artifact_root, rev20_config=rev20_config, node_field=node_field,
    )
    net, positions = captured["net"], captured["positions"]
    length_scale = float(captured["kwargs"]["length_scale"])
    base_edges = ee_edges(net)
    base_effective = effective_source_count(base_edges["row"], base_edges["data"], base_edges["n_e"])
    base_hashes = {
        "ampa_topology": _hash_sparse_bins(net["ampa_by_delay"], include_data=False),
        "ampa_data": _hash_sparse_bins(net["ampa_by_delay"]),
        "gaba": _hash_sparse_bins(net["gaba_by_delay"]),
    }
    n_delay_bins = len(net["ampa_by_delay"])
    print(f"[{time.time()-started:6.0f}s] seed {seed}: graph captured in {time.time()-t0:.0f}s; "
          f"E={base_edges['n_e']} EE edges={len(base_edges['data'])} bins={n_delay_bins}", flush=True)
    graph = {
        "n_e": base_edges["n_e"], "row": base_edges["row"], "data": base_edges["data"],
        "dx": positions[base_edges["col"], 0] - positions[base_edges["row"], 0],
        "dy": positions[base_edges["col"], 1] - positions[base_edges["row"], 1],
    }
    validation = validate_fast_path(
        net, positions, graph, base_edges,
        [(float(reference[0]), float(reference[1])), (float(theta[0]), float(ar[0])),
         (float(theta[-1]), float(ar[-1])), (float(theta[i_ref]), float(ar[-1]))],
        length_scale=length_scale, reference=reference,
    )
    print(f"[{time.time()-started:6.0f}s] seed {seed}: fast path validated against the producer "
          f"(max relative {max(c['max_relative_difference'] for c in validation):.2e})", flush=True)
    after_hashes = {
        "ampa_topology": _hash_sparse_bins(net["ampa_by_delay"], include_data=False),
        "ampa_data": _hash_sparse_bins(net["ampa_by_delay"]),
        "gaba": _hash_sparse_bins(net["gaba_by_delay"]),
    }
    if after_hashes != base_hashes:
        raise RuntimeError("validation mutated the frozen base network")
    cache_record = captured.get("net", {}).get("cache_sha256")
    del net, positions, captured
    records = []
    for i, angle in enumerate(theta):
        for j, aspect in enumerate(ar):
            record = audit_grid_point(
                graph, base_edges, base_effective,
                length_scale=length_scale, angle_deg=float(angle), aspect_ratio=float(aspect),
                reference=reference,
            )
            record.update(seed=int(seed), i=i, j=j, passes=grid_point_passes(record))
            records.append(record)
        print(f"[{time.time()-started:6.0f}s] seed {seed}: theta {angle:.1f} done", flush=True)
    ref_record = next(r for r in records if r["i"] == i_ref and r["j"] == j_ref)
    if not ref_record["exact_noop"] or ref_record["budget_error_max"] != 0.0:
        raise RuntimeError("reference grid point was not an exact no-op")
    output = {
        "length_scale_mm": length_scale,
        "n_e": int(base_edges["n_e"]),
        "n_ee_edges": int(len(base_edges["data"])),
        "n_delay_bins": int(n_delay_bins),
        "base_hashes": base_hashes,
        "network_cache": cache_record,
        "fast_path_validation": validation,
        "records": records,
    }
    return int(seed), output


def _plot(out_dir: Path, theta, ar, worst: dict, rectangle: dict | None, seeds,
          reference=(REFERENCE_ANGLE_DEG, REFERENCE_ASPECT_RATIO)) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    dtheta = float(theta[1] - theta[0]) if len(theta) > 1 else 1.0
    dar = float(ar[1] - ar[0]) if len(ar) > 1 else 1.0
    x_edges = np.concatenate([theta - dtheta / 2, [theta[-1] + dtheta / 2]])
    y_edges = np.concatenate([ar - dar / 2, [ar[-1] + dar / 2]])
    panels = [
        ("budget_error_max", "Max incoming-budget error (worst topology)", "viridis",
         LogNorm(vmin=1e-16, vmax=1e-9), r"$\leq 10^{-9}$"),
        ("zero_denominator", "Zero normalization denominator (any topology)", "Greys",
         None, "none allowed"),
        ("edge_ratio_p01", "Edge-weight ratio p01 (min over topologies)", "cividis",
         TwoSlopeNorm(vmin=0.0, vcenter=0.25, vmax=1.0), r"$\geq 0.25$"),
        ("edge_ratio_p99", "Edge-weight ratio p99 (max over topologies)", "cividis_r",
         TwoSlopeNorm(vmin=1.0, vcenter=4.0, vmax=12.0), r"$\leq 4$"),
        ("effective_source_median_ratio", "Effective source count, median ratio (min over topologies)",
         "cividis", TwoSlopeNorm(vmin=0.0, vcenter=0.75, vmax=1.5), r"$\geq 0.75$"),
        ("effective_source_p05_ratio", "Effective source count, p05 ratio (min over topologies)",
         "cividis", TwoSlopeNorm(vmin=0.0, vcenter=0.50, vmax=1.5), r"$\geq 0.50$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    for ax, (key, title, cmap, norm, threshold_label) in zip(axes.ravel(), panels):
        values = np.asarray(worst[key], float).T  # rows = AR, cols = theta
        if key == "zero_denominator":
            mesh = ax.pcolormesh(x_edges, y_edges, values, cmap=cmap, vmin=0, vmax=1, shading="flat")
        elif key == "budget_error_max":
            mesh = ax.pcolormesh(x_edges, y_edges, np.maximum(values, 1e-16), cmap=cmap, norm=norm,
                                 shading="flat")
        else:
            mesh = ax.pcolormesh(x_edges, y_edges, values, cmap=cmap, norm=norm, shading="flat")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label(f"threshold {threshold_label}")
        if rectangle is not None:
            x0 = theta[rectangle["i0"]] - dtheta / 2
            y0 = ar[rectangle["j0"]] - dar / 2
            width = theta[rectangle["i1"]] - theta[rectangle["i0"]] + dtheta
            height = ar[rectangle["j1"]] - ar[rectangle["j0"]] + dar
            ax.add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="black",
                                   linewidth=1.6))
        ax.plot([float(reference[0])], [float(reference[1])], marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor="black", linestyle="none")
        ax.set_title(title)
        ax.set_xlabel("E-to-E long-axis angle (deg)")
        ax.set_ylabel("E-to-E aspect ratio")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "geometry_domain_admissibility.png"
    pdf = out_dir / "geometry_domain_admissibility.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _write_readme(out_dir: Path, rectangle_text: str, seeds) -> None:
    text = (
        "### geometry_domain_admissibility.png\n\n"
        "固定拓扑 E→E 椭圆重加权的纯结构审计，不含任何仿真。横轴是重加权用的长轴角度，纵轴是"
        "长短轴比，六个面板分别对应六条预注册可采纳判据：每个靶细胞总输入是否守恒、有没有归一化"
        "分母为零、边权重比例的 1% 和 99% 分位是否落在 [0.25, 4] 之内、每个靶细胞的有效来源数相对"
        "参考点的中位数比例和 5% 分位比例是否够高。每格取四张拟合拓扑（seed "
        f"{', '.join(str(s) for s in seeds)}）里最差的一张；白圈是参考点 (45°, 2)，黑框是四张拓扑上"
        f"全部判据同时通过的最大轴对齐矩形，{rectangle_text}\n\n"
        "**关注点**：黑框才是 rev22 允许优化的几何域，框外的角度或长短轴比不是"
        "\"数据不喜欢\"，而是这张固定拓扑图上没有足够的边可以承接权重。\n"
    )
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--rev20-config", type=Path,
                        default=ROOT / "config/topic4_rev20_dc_dual_core_mechanism_atlas.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FIT_SEEDS))
    parser.add_argument("--theta-half-width", type=float, default=22.5,
                        help="angle grid half width around the reference axis (deg)")
    parser.add_argument("--reference-angle-deg", type=float, default=None,
                        help="absolute reference angle; default = rev22 analysis config "
                             "reference.ellipse_angle_deg (registered patient axis)")
    parser.add_argument("--reference-aspect-ratio", type=float, default=None)
    parser.add_argument("--analysis-config", type=Path,
                        default=ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json")
    parser.add_argument("--theta-step", type=float, default=2.5)
    parser.add_argument("--ar-range", type=float, nargs=2, default=(1.0, 3.0))
    parser.add_argument("--ar-step", type=float, default=0.125)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()
    artifact_root = args.artifact_root.resolve()
    rev20_config = json.loads(args.rev20_config.read_text())
    manifest_path = artifact_root / rev20_config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    reference_rows = [c for c in manifest["candidates"] if c.get("is_reference")]
    if len(reference_rows) != 1:
        raise RuntimeError("rev20 manifest must contain exactly one reference candidate")
    node_field = reference_rows[0]["node_field"]
    out_root = args.out_root or (
        artifact_root / "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/geometry_domain"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(args.analysis_config.read_text())
    reference = (
        float(analysis["reference"]["ellipse_angle_deg"]) if args.reference_angle_deg is None
        else float(args.reference_angle_deg),
        float(analysis["reference"]["ellipse_aspect_ratio"]) if args.reference_aspect_ratio is None
        else float(args.reference_aspect_ratio),
    )
    theta, ar, i_ref, j_ref = _grid(args.theta_half_width, args.theta_step, args.ar_range,
                                    args.ar_step, reference)

    if args.workers < 1 or args.workers > len(args.seeds):
        raise ValueError("workers must lie between one and the number of topology seeds")
    per_topology = {}
    if args.workers == 1:
        for seed in args.seeds:
            key, value = audit_one_topology(
                seed, artifact_root=artifact_root, rev20_config=rev20_config,
                node_field=node_field, theta=theta, ar=ar, i_ref=i_ref, j_ref=j_ref,
                started=started, reference=reference,
            )
            per_topology[key] = value
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [pool.submit(
                audit_one_topology, seed, artifact_root=artifact_root,
                rev20_config=rev20_config, node_field=node_field, theta=theta, ar=ar,
                i_ref=i_ref, j_ref=j_ref, started=started, reference=reference,
            ) for seed in args.seeds]
            for future in as_completed(futures):
                key, value = future.result()
                per_topology[key] = value

    # ---- combine ----
    n_i, n_j = len(theta), len(ar)
    pass_all = np.ones((n_i, n_j), bool)
    worst = {k: np.full((n_i, n_j), np.nan) for k in (
        "budget_error_max", "edge_ratio_p01", "edge_ratio_p99",
        "effective_source_median_ratio", "effective_source_p05_ratio")}
    worst["zero_denominator"] = np.zeros((n_i, n_j), float)
    for seed, block in per_topology.items():
        for r in block["records"]:
            i, j = r["i"], r["j"]
            pass_all[i, j] &= bool(r["passes"])
            if r["status"] != "OK":
                worst["zero_denominator"][i, j] = 1.0 if r["zero_denominator"] else worst["zero_denominator"][i, j]
                for k in ("budget_error_max", "edge_ratio_p99"):
                    worst[k][i, j] = np.inf
                for k in ("edge_ratio_p01", "effective_source_median_ratio", "effective_source_p05_ratio"):
                    worst[k][i, j] = 0.0
                continue
            worst["budget_error_max"][i, j] = np.nanmax([worst["budget_error_max"][i, j], r["budget_error_max"]])
            worst["edge_ratio_p99"][i, j] = np.nanmax([worst["edge_ratio_p99"][i, j], r["edge_ratio_p99"]])
            for k in ("edge_ratio_p01", "effective_source_median_ratio", "effective_source_p05_ratio"):
                worst[k][i, j] = np.nanmin([worst[k][i, j], r[k]])
    rectangle = largest_admissible_rectangle(pass_all, i_ref, j_ref, theta_values=theta,
                                             reference_angle=float(reference[0]))
    achieved = assess_achieved_geometry(per_topology, rectangle, theta, ar, i_ref, j_ref)
    if rectangle is None or rectangle["cells"] <= 1:
        status = "GEOMETRY_STRUCTURALLY_NON_ESTIMABLE"
    elif rectangle["i0"] == rectangle["i1"] or rectangle["j0"] == rectangle["j1"]:
        status = "GEOMETRY_DOMAIN_FROZEN_ONE_DIMENSION_DEGENERATE"
    elif not achieved["pass"]:
        status = "GEOMETRY_STRUCTURALLY_NON_ESTIMABLE"
    else:
        status = "GEOMETRY_DOMAIN_FROZEN"

    # which criterion binds each rectangle edge: first failing criterion just outside
    def _failing(i, j):
        names = []
        for seed, block in per_topology.items():
            r = next(x for x in block["records"] if x["i"] == i and x["j"] == j)
            if r["status"] != "OK":
                names.append(f"{seed}:{r['status']}")
                continue
            if r["budget_error_max"] > THRESHOLDS["budget_error_max"]:
                names.append(f"{seed}:budget_error")
            if r["edge_ratio_p01"] < THRESHOLDS["edge_ratio_p01_min"]:
                names.append(f"{seed}:edge_ratio_p01")
            if r["edge_ratio_p99"] > THRESHOLDS["edge_ratio_p99_max"]:
                names.append(f"{seed}:edge_ratio_p99")
            if r["effective_source_median_ratio"] < THRESHOLDS["effective_source_median_ratio_min"]:
                names.append(f"{seed}:effective_source_median_ratio")
            if r["effective_source_p05_ratio"] < THRESHOLDS["effective_source_p05_ratio_min"]:
                names.append(f"{seed}:effective_source_p05_ratio")
        return sorted(set(names))

    binding = {}
    if rectangle is not None:
        i0, i1, j0, j1 = rectangle["i0"], rectangle["i1"], rectangle["j0"], rectangle["j1"]
        binding["theta_low"] = ("grid_edge" if i0 == 0 else
                                sorted({n for j in range(j0, j1 + 1) for n in _failing(i0 - 1, j)}))
        binding["theta_high"] = ("grid_edge" if i1 == n_i - 1 else
                                 sorted({n for j in range(j0, j1 + 1) for n in _failing(i1 + 1, j)}))
        binding["ar_low"] = ("grid_edge" if j0 == 0 else
                             sorted({n for i in range(i0, i1 + 1) for n in _failing(i, j0 - 1)}))
        binding["ar_high"] = ("grid_edge" if j1 == n_j - 1 else
                              sorted({n for i in range(i0, i1 + 1) for n in _failing(i, j1 + 1)}))

    # ---- write grid table ----
    csv_path = out_root / "geometry_audit_grid.csv"
    fields = ["seed", "i", "j", "angle_deg", "aspect_ratio", "status", "exact_noop", "passes",
              "budget_error_max", "zero_denominator", "edge_ratio_p01", "edge_ratio_p50",
              "edge_ratio_p99", "effective_source_median_ratio", "effective_source_p05_ratio",
              "targets_effective_ratio_below_0p5", "topology_unchanged", "gaba_unchanged",
              "achieved_angle_deg", "achieved_aspect_ratio", "weighted_rms_distance_mm",
              "axial_anisotropy"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for seed in sorted(per_topology):
            for r in per_topology[seed]["records"]:
                writer.writerow(r)

    rectangle_bounds = None
    if rectangle is not None:
        rectangle_bounds = {
            "theta_deg": [float(theta[rectangle["i0"]]), float(theta[rectangle["i1"]])],
            "aspect_ratio": [float(ar[rectangle["j0"]]), float(ar[rectangle["j1"]])],
            "n_theta": int(rectangle["i1"] - rectangle["i0"] + 1),
            "n_ar": int(rectangle["j1"] - rectangle["j0"] + 1),
            "cells": int(rectangle["cells"]),
        }
        rect_text = (f"覆盖角度 {rectangle_bounds['theta_deg'][0]:g}°–{rectangle_bounds['theta_deg'][1]:g}°、"
                     f"长短轴比 {rectangle_bounds['aspect_ratio'][0]:g}–{rectangle_bounds['aspect_ratio'][1]:g}。")
    else:
        rect_text = "没有任何非参考格通过。"
    figure_paths = _plot(out_root / "figures", theta, ar, worst, rectangle, sorted(per_topology),
                         reference=reference)
    _write_readme(out_root / "figures", rect_text, sorted(per_topology))

    summary = {
        "schema_id": "topic4_rev22_dci_geometry_domain_v1",
        "status": status,
        "git_commit": _git_commit(),
        "rev20_config_sha256": _sha256(args.rev20_config),
        "rev20_manifest_sha256": _sha256(manifest_path),
        "node_field_sha256": node_field.get("field_sha256"),
        "reference": {"angle_deg": float(reference[0]), "aspect_ratio": float(reference[1]),
                      "source": ("rev22 amendment v5.1: absolute reference = registered patient axis "
                                 "(register_to_sheet theta_deg, the graph kernel long axis) and the "
                                 "engine kernel AR; requested angles are offsets around it"),
                      "theta_half_width_deg": float(args.theta_half_width)},
        "grid": {"theta_deg": theta.tolist(),
                 "theta_offset_deg": (theta - float(reference[0])).tolist(),
                 "aspect_ratio": ar.tolist(),
                 "theta_step": float(args.theta_step), "ar_step": float(args.ar_step)},
        "thresholds": THRESHOLDS,
        "identifiability_thresholds": IDENTIFIABILITY_THRESHOLDS,
        "seeds": sorted(per_topology),
        "admissible_rectangle": rectangle_bounds,
        "achieved_geometry": achieved,
        "binding_criteria": binding,
        "pass_all_topologies": pass_all.astype(int).tolist(),
        "worst_case_maps": {k: np.where(np.isfinite(v), v, None).tolist() for k, v in worst.items()},
        "per_topology": {
            str(seed): {k: v for k, v in block.items() if k != "records"}
            for seed, block in per_topology.items()
        },
        "per_topology_pass_counts": {
            str(seed): int(sum(1 for r in block["records"] if r["passes"]))
            for seed, block in per_topology.items()
        },
        "grid_table_sha256": _sha256(csv_path),
        "figure": figure_paths,
        "elapsed_seconds": float(time.time() - started),
        "claim_boundary": ("Structure-only admissibility and realized weighted-moment identifiability of "
                           "the fixed-topology ellipse reweighting; the final ellipse-plus-learned-mapper "
                           "design audit is separate, theta_FT is not an anatomical direction, and this "
                           "artifact contains no simulation."),
    }
    (out_root / "geometry_domain.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "rectangle": rectangle_bounds, "binding": binding,
                      "elapsed_s": round(time.time() - started)}, indent=2))


if __name__ == "__main__":
    main()

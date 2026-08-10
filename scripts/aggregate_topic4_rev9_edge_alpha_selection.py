"""Aggregate exploratory rev9 Node-to-Edge local-response matching."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_rev9_response_matching import (  # noqa: E402
    positive_map_js_distance,
    pseudo_huber_squared,
    robust_scale,
    scalar_pair_loss,
)


ROOT = Path("results/topic4_sef_hfo/data_driven_core_field_rev9")
CALIBRATION = ROOT / "node_edge_calibration"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return default


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_bundle(json_path, npz_path, *, expected_arm=None):
    payload = json.loads(Path(json_path).read_text())
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if expected_arm is not None and payload["arm"] != expected_arm:
        raise RuntimeError(f"unexpected arm in {json_path}")
    if payload["seeds"] != arrays["seeds"].astype(int).tolist():
        raise RuntimeError(f"seed mismatch in {json_path}")
    if [row["id"] for row in payload["sites"]] != arrays["site_ids"].astype(str).tolist():
        raise RuntimeError(f"site mismatch in {json_path}")
    return payload, arrays


def _bundle_records(payload, arrays, *, primary_window, selection_amplitude):
    windows = arrays["windows_after_pulse_end_ms"].astype(float)
    window_matches = np.flatnonzero(np.all(
        np.isclose(windows, np.asarray(primary_window, float)), axis=1))
    amplitude_matches = np.flatnonzero(np.isclose(
        arrays["amplitude_multipliers"], float(selection_amplitude)))
    if len(window_matches) != 1 or len(amplitude_matches) != 1:
        raise RuntimeError("primary window or selection amplitude is ambiguous")
    wi, ai = int(window_matches[0]), int(amplitude_matches[0])
    site_ids = arrays["site_ids"].astype(str)
    site_roles = arrays["site_roles"].astype(str)
    baselines = {int(row["seed"]): row for row in payload["event_diagnostics"]["sham"]}
    records, baseline_records = {}, {}
    for si, seed in enumerate(arrays["seeds"].astype(int)):
        baseline_records[int(seed)] = baselines[int(seed)]
        for ji, (site_id, role) in enumerate(zip(site_ids, site_roles)):
            axis_ratio = arrays["axis_variance_ratio"][si, ji, ai, wi]
            scalars = np.asarray([
                arrays["source_slopes"][si, ji, wi],
                arrays["downstream_slopes"][si, ji, wi],
                arrays["r90_mm"][si, ji, ai, wi],
                np.log(np.clip(axis_ratio, 1e-3, 1e3))
                if np.isfinite(axis_ratio) else np.nan,
            ], float)
            records[(int(seed), str(site_id))] = dict(
                seed=int(seed), site_id=str(site_id), role=str(role),
                eligible=bool(arrays["site_seed_window_linear_eligible"][si, ji, wi]),
                scalars=scalars,
                positive_map=np.asarray(
                    arrays["positive_maps_per_cell"][si, ji, ai, wi], float),
                downstream_positive=float(
                    arrays["downstream_positive_per_cell"][si, ji, ai, wi]),
            )
    return records, baseline_records


def _baseline_loss(node, edge):
    eps = 1e-6
    values = np.asarray([
        np.log((float(edge["active_fraction_floor"]) + eps)
               / (float(node["active_fraction_floor"]) + eps)),
        np.log((float(edge["active_fraction_peak"]) + eps)
               / (float(node["active_fraction_peak"]) + eps)),
        (float(edge["n_events"]) - float(node["n_events"]))
        / max(float(node["n_events"]), 1.0),
    ])
    return float(np.mean(pseudo_huber_squared(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_rev9_exploratory.json")
    parser.add_argument("--node-raw-json", default=str(
        CALIBRATION / "node_kick_canary_t220_linear.json"))
    parser.add_argument("--node-reconciled-npz", default=str(
        CALIBRATION / "node_kick_canary_t220_linear_window_reconciled.npz"))
    parser.add_argument("--node-reconciled-json", default=str(
        CALIBRATION / "node_kick_canary_t220_linear_window_reconciled.json"))
    parser.add_argument("--node-reference-dir", default=str(
        CALIBRATION / "node_selection_references"))
    parser.add_argument("--edge-dir", default=str(
        CALIBRATION / "edge_alpha_selection"))
    parser.add_argument("--out-json", default=str(
        CALIBRATION / "edge_alpha_selection_summary.json"))
    parser.add_argument("--out-npz", default=str(
        CALIBRATION / "edge_alpha_selection_summary.npz"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    instrument = config["small_kick_instrument"]
    primary_window = instrument["primary_window_after_pulse_end_ms"]
    selection_amplitude = instrument["window_selection_amplitude_multiplier"]
    expected_alphas = np.asarray(
        config["edge"]["alpha_grid"]
        + config["edge"].get("alpha_midpoint_grid", []), float)
    expected_seeds = [int(value) for value in config["edge"]["calibration_seeds"]]
    execution_provenance = dict(
        **provenance(), git_status_porcelain=_git("status", "--porcelain"),
        producer_sha256=_sha256(__file__), config_sha256=_sha256(args.config),
        python_executable=sys.executable, python_version=platform.python_version(),
        systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT"), readout_seed=None)

    source_paths = [args.node_raw_json, args.node_reconciled_npz,
                    args.node_reconciled_json]
    node_payload, node_arrays = _load_bundle(
        args.node_raw_json, args.node_reconciled_npz, expected_arm="Node")
    node_records, node_baselines = _bundle_records(
        node_payload, node_arrays, primary_window=primary_window,
        selection_amplitude=selection_amplitude)
    for json_path in sorted(glob.glob(str(Path(args.node_reference_dir) / "*.json"))):
        npz_path = str(Path(json_path).with_suffix(".npz"))
        payload, arrays = _load_bundle(json_path, npz_path, expected_arm="Node")
        records, baselines = _bundle_records(
            payload, arrays, primary_window=primary_window,
            selection_amplitude=selection_amplitude)
        node_records.update(records)
        node_baselines.update(baselines)
        source_paths.extend((json_path, npz_path))
    if sorted(node_baselines) != expected_seeds:
        raise RuntimeError("Node references do not cover the calibration seeds")

    edge_bundles = {}
    for json_path in sorted(glob.glob(str(Path(args.edge_dir) / "*.json"))):
        npz_path = str(Path(json_path).with_suffix(".npz"))
        payload, arrays = _load_bundle(json_path, npz_path, expected_arm="Edge")
        if payload["provenance"]["tracked_modules_dirty"] is not False:
            raise RuntimeError(f"dirty Edge worker: {json_path}")
        alpha = float(payload["alpha"])
        if len(payload["seeds"]) != 1:
            raise RuntimeError("Edge selection workers must contain one seed")
        seed = int(payload["seeds"][0])
        key = (alpha, seed)
        if key in edge_bundles:
            raise RuntimeError(f"duplicate Edge worker {key}")
        records, baselines = _bundle_records(
            payload, arrays, primary_window=primary_window,
            selection_amplitude=selection_amplitude)
        edge_bundles[key] = dict(
            payload=payload, records=records, baseline=baselines[seed])
        source_paths.extend((json_path, npz_path))
    expected_keys = {(float(alpha), seed) for alpha in expected_alphas
                     for seed in expected_seeds}
    if set(edge_bundles) != expected_keys:
        raise RuntimeError("Edge worker grid is incomplete or contains extra cells")

    node_eligible = [row for row in node_records.values() if row["eligible"]]
    scalar_matrix = np.asarray([row["scalars"] for row in node_eligible])
    scale_floors = np.asarray([1e-6, 1e-6, 0.1, 0.1])
    scales = np.asarray([
        robust_scale(scalar_matrix[:, index], floor=scale_floors[index])
        for index in range(scalar_matrix.shape[1])])
    summaries, pair_rows = [], []
    for alpha in expected_alphas:
        losses, control_losses, baseline_losses, map_distances = [], [], [], []
        scalar_feature_counts = []
        edge_eligible_count = 0
        for seed in expected_seeds:
            bundle = edge_bundles[(float(alpha), seed)]
            baseline_losses.append(_baseline_loss(
                node_baselines[seed], bundle["baseline"]))
            for key, node in node_records.items():
                if key[0] != seed or not node["eligible"]:
                    continue
                edge = bundle["records"][key]
                if edge["eligible"]:
                    edge_eligible_count += 1
                paired = bool(edge["eligible"])
                if not paired:
                    pair_rows.append(dict(
                        alpha=float(alpha), seed=seed, site_id=key[1],
                        role=node["role"], paired_eligible=False,
                        scalar_loss=None, map_js=None, pair_loss=None))
                    continue
                scalar_loss, count = scalar_pair_loss(
                    node["scalars"], edge["scalars"], scales)
                map_distance = positive_map_js_distance(
                    node["positive_map"], edge["positive_map"])
                parts, weights = [], []
                if scalar_loss is not None:
                    parts.append(scalar_loss)
                    weights.append(count)
                    scalar_feature_counts.append(count)
                if map_distance is not None:
                    parts.append(map_distance ** 2)
                    weights.append(1)
                    map_distances.append(map_distance)
                pair_loss = (None if not parts else
                             float(np.average(parts, weights=weights)))
                if pair_loss is not None:
                    losses.append(pair_loss)
                    if node["role"] == "matched_off_field":
                        control_losses.append(pair_loss)
                pair_rows.append(dict(
                    alpha=float(alpha), seed=seed, site_id=key[1],
                    role=node["role"], paired_eligible=True,
                    scalar_loss=scalar_loss, scalar_feature_count=count,
                    map_js=map_distance, pair_loss=pair_loss,
                    node_scalars=node["scalars"].tolist(),
                    edge_scalars=edge["scalars"].tolist()))
        n_node_eligible = len(node_eligible)
        coverage = edge_eligible_count / n_node_eligible
        response_loss = float(np.median(losses)) if losses else None
        control_loss = float(np.median(control_losses)) if control_losses else None
        baseline_loss = float(np.median(baseline_losses))
        missing_penalty = float(1.0 - coverage)
        j_cal = (None if response_loss is None or control_loss is None else
                 float(response_loss + missing_penalty
                       + 0.5 * control_loss + 0.25 * baseline_loss))
        structure = edge_bundles[(float(alpha), expected_seeds[0])][
            "payload"]["networks"][0]["edge_diagnostics"]
        summaries.append(dict(
            alpha=float(alpha), J_cal=j_cal,
            response_loss_median=response_loss,
            off_field_loss_median=control_loss,
            baseline_shift_median=baseline_loss,
            missing_pair_penalty=missing_penalty,
            paired_coverage=coverage, n_paired=edge_eligible_count,
            n_node_eligible=n_node_eligible,
            map_js_median=(float(np.median(map_distances))
                           if map_distances else None),
            scalar_feature_count_median=(float(np.median(scalar_feature_counts))
                                         if scalar_feature_counts else None),
            structure=structure))

    finite = [row for row in summaries if row["J_cal"] is not None]
    selected = min(finite, key=lambda row: (row["J_cal"], row["alpha"]))
    out_npz = Path(args.out_npz)
    _atomic_npz(
        out_npz,
        alpha=np.asarray([row["alpha"] for row in summaries]),
        J_cal=np.asarray([row["J_cal"] for row in summaries]),
        response_loss=np.asarray([row["response_loss_median"] for row in summaries]),
        off_field_loss=np.asarray([row["off_field_loss_median"] for row in summaries]),
        baseline_shift=np.asarray([row["baseline_shift_median"] for row in summaries]),
        missing_pair_penalty=np.asarray([row["missing_pair_penalty"] for row in summaries]),
        paired_coverage=np.asarray([row["paired_coverage"] for row in summaries]),
        node_scalar_scales=scales,
        selected_alpha=np.asarray(selected["alpha"]),
    )
    payload = dict(
        status="REV9_EDGE_ALPHA_EXPLORATORY_SELECTION_COMPLETE",
        scientific_role=(
            "Exploratory Node-to-Edge local-response reference selection; not "
            "mechanism equivalence, patient validation, or a hard acceptance gate"),
        objective=dict(
            formula=(
                "median_pair_response_loss + (1-paired_coverage) + "
                "0.5*median_off_field_loss + 0.25*median_sham_baseline_shift"),
            scalar_features=[
                "source_slope", "downstream_slope", "r90_mm",
                "log_axis_variance_ratio"],
            scalar_loss="pseudo_Huber on robust Node-standardized differences",
            map_loss="squared normalized sqrt-JS of positive response maps",
            primary_window_after_pulse_end_ms=primary_window,
            amplitude_multiplier_for_shape=selection_amplitude,
            pairing="same seed and site; both primary windows event-free",
            node_scalar_scales=scales.tolist()),
        summaries=summaries,
        selection=dict(
            alpha_star=selected["alpha"], J_cal=selected["J_cal"],
            tie_break="lower alpha",
            interpretation="response-matched reference only"),
        pair_rows=pair_rows,
        inputs=[dict(path=str(path), sha256=_sha256(path))
                for path in source_paths],
        arrays=dict(path=str(out_npz), sha256=_sha256(out_npz)),
        provenance=execution_provenance,
    )
    atomic_write_json(payload, args.out_json)
    print(json.dumps(dict(
        status=payload["status"], selection=payload["selection"],
        summaries=summaries, arrays_sha256=payload["arrays"]["sha256"]),
        indent=2))


if __name__ == "__main__":
    main()

"""Evaluate the frozen rev9 alpha reference on seeds 911--922."""
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
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from scripts.aggregate_topic4_rev9_edge_alpha_selection import (  # noqa: E402
    _baseline_loss,
    _bundle_records,
    _load_bundle,
)
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_rev9_response_matching import (  # noqa: E402
    positive_map_js_distance,
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


def _summarize(seed_rows, seeds):
    selected = [row for seed in seeds for row in seed_rows[int(seed)]["pairs"]]
    paired = [row for row in selected if row["paired_eligible"]]
    controls = [row for row in paired if row["role"] == "matched_off_field"]
    node_eligible = sum(seed_rows[int(seed)]["n_node_eligible"] for seed in seeds)
    coverage = len(paired) / node_eligible
    response = float(np.median([row["pair_loss"] for row in paired]))
    control = float(np.median([row["pair_loss"] for row in controls]))
    baseline = float(np.median([
        seed_rows[int(seed)]["baseline_loss"] for seed in seeds]))
    missing = float(1.0 - coverage)
    j_value = float(response + missing + 0.5 * control + 0.25 * baseline)
    source_x = np.asarray([row["node_scalars"][0] for row in paired], float)
    source_y = np.asarray([row["edge_scalars"][0] for row in paired], float)
    downstream_x = np.asarray([row["node_scalars"][1] for row in paired], float)
    downstream_y = np.asarray([row["edge_scalars"][1] for row in paired], float)
    source_valid = np.isfinite(source_x) & np.isfinite(source_y)
    downstream_valid = np.isfinite(downstream_x) & np.isfinite(downstream_y)
    return dict(
        J_eval=j_value, response_loss_median=response,
        off_field_loss_median=control, baseline_shift_median=baseline,
        missing_pair_penalty=missing, paired_coverage=float(coverage),
        n_paired=len(paired), n_node_eligible=int(node_eligible),
        map_js_median=float(np.median([row["map_js"] for row in paired
                                      if row["map_js"] is not None])),
        source_spearman=float(spearmanr(
            source_x[source_valid], source_y[source_valid]).statistic),
        downstream_spearman=float(spearmanr(
            downstream_x[downstream_valid], downstream_y[downstream_valid]).statistic),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_rev9_exploratory.json")
    parser.add_argument("--selection-summary", default=str(
        CALIBRATION / "edge_alpha_selection_summary.json"))
    parser.add_argument("--node-dir", default=str(
        CALIBRATION / "node_out_of_selection_references"))
    parser.add_argument("--edge-dir", default=str(
        CALIBRATION / "edge_alpha_star_out_of_selection"))
    parser.add_argument("--out-json", default=str(
        CALIBRATION / "alpha_star_out_of_selection_summary.json"))
    parser.add_argument("--out-npz", default=str(
        CALIBRATION / "alpha_star_out_of_selection_bootstrap.npz"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    instrument = config["small_kick_instrument"]
    seeds = [int(value) for value in
             config["edge"]["out_of_selection_and_factorial_seeds"]]
    bootstrap_seed = int(config["edge"]["out_of_selection_bootstrap_seed"])
    n_bootstrap = int(config["edge"]["out_of_selection_bootstrap_repeats"])
    selection = json.loads(Path(args.selection_summary).read_text())
    alpha_star = float(selection["selection"]["alpha_star"])
    scales = np.asarray(selection["objective"]["node_scalar_scales"], float)
    primary_window = instrument["primary_window_after_pulse_end_ms"]
    selection_amplitude = instrument["window_selection_amplitude_multiplier"]
    execution_provenance = dict(
        **provenance(), git_status_porcelain=_git("status", "--porcelain"),
        producer_sha256=_sha256(__file__), config_sha256=_sha256(args.config),
        python_executable=sys.executable, python_version=platform.python_version(),
        systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT"),
        readout_seed=bootstrap_seed)

    node, edge, inputs = {}, {}, [args.config, args.selection_summary]
    for json_path in sorted(glob.glob(str(Path(args.node_dir) / "*.json"))):
        npz_path = str(Path(json_path).with_suffix(".npz"))
        payload, arrays = _load_bundle(json_path, npz_path, expected_arm="Node")
        records, baselines = _bundle_records(
            payload, arrays, primary_window=primary_window,
            selection_amplitude=selection_amplitude)
        seed = int(payload["seeds"][0])
        node[seed] = dict(payload=payload, records=records, baseline=baselines[seed])
        inputs.extend((json_path, npz_path))
    for json_path in sorted(glob.glob(str(Path(args.edge_dir) / "*.json"))):
        npz_path = str(Path(json_path).with_suffix(".npz"))
        payload, arrays = _load_bundle(json_path, npz_path, expected_arm="Edge")
        if not np.isclose(payload["alpha"], alpha_star):
            raise RuntimeError("out-of-selection Edge worker changed alpha_star")
        records, baselines = _bundle_records(
            payload, arrays, primary_window=primary_window,
            selection_amplitude=selection_amplitude)
        seed = int(payload["seeds"][0])
        edge[seed] = dict(payload=payload, records=records, baseline=baselines[seed])
        inputs.extend((json_path, npz_path))
    if sorted(node) != seeds or sorted(edge) != seeds:
        raise RuntimeError("out-of-selection worker grid is incomplete")
    if any(bundle["payload"]["provenance"]["tracked_modules_dirty"] is not False
           for bundle in [*node.values(), *edge.values()]):
        raise RuntimeError("dirty out-of-selection worker")

    seed_rows = {}
    all_pairs = []
    for seed in seeds:
        pairs = []
        for key, node_row in node[seed]["records"].items():
            if not node_row["eligible"]:
                continue
            edge_row = edge[seed]["records"][key]
            row = dict(
                seed=seed, site_id=key[1], role=node_row["role"],
                paired_eligible=bool(edge_row["eligible"]))
            if edge_row["eligible"]:
                scalar_loss, count = scalar_pair_loss(
                    node_row["scalars"], edge_row["scalars"], scales)
                map_js = positive_map_js_distance(
                    node_row["positive_map"], edge_row["positive_map"])
                pieces, weights = [], []
                if scalar_loss is not None:
                    pieces.append(scalar_loss)
                    weights.append(count)
                if map_js is not None:
                    pieces.append(map_js ** 2)
                    weights.append(1)
                row.update(
                    scalar_loss=scalar_loss, scalar_feature_count=count,
                    map_js=map_js,
                    pair_loss=float(np.average(pieces, weights=weights)),
                    node_scalars=node_row["scalars"].tolist(),
                    edge_scalars=edge_row["scalars"].tolist())
            pairs.append(row)
            all_pairs.append(row)
        seed_rows[seed] = dict(
            pairs=pairs,
            n_node_eligible=sum(row["eligible"]
                                for row in node[seed]["records"].values()),
            baseline_loss=_baseline_loss(
                node[seed]["baseline"], edge[seed]["baseline"]))

    observed = _summarize(seed_rows, seeds)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = {key: np.empty(n_bootstrap, float) for key in (
        "J_eval", "response_loss_median", "off_field_loss_median",
        "baseline_shift_median", "paired_coverage", "source_spearman",
        "downstream_spearman")}
    for index in range(n_bootstrap):
        sampled = rng.choice(seeds, size=len(seeds), replace=True).tolist()
        result = _summarize(seed_rows, sampled)
        for key in bootstrap:
            bootstrap[key][index] = result[key]
    intervals = {key: np.percentile(values, [2.5, 97.5]).tolist()
                 for key, values in bootstrap.items()}

    selected_row = next(row for row in selection["summaries"]
                        if np.isclose(row["alpha"], alpha_star))
    comparison = dict(
        selection=dict(
            J_eval=selected_row["J_cal"],
            response_loss_median=selected_row["response_loss_median"],
            off_field_loss_median=selected_row["off_field_loss_median"],
            baseline_shift_median=selected_row["baseline_shift_median"],
            paired_coverage=selected_row["paired_coverage"]),
        out_of_selection=observed,
        delta_out_minus_selection=dict(
            J_eval=observed["J_eval"] - selected_row["J_cal"],
            response_loss_median=(observed["response_loss_median"]
                                  - selected_row["response_loss_median"]),
            paired_coverage=(observed["paired_coverage"]
                             - selected_row["paired_coverage"])))

    out_npz = Path(args.out_npz)
    _atomic_npz(out_npz, **bootstrap)
    payload = dict(
        status="REV9_ALPHA_STAR_OUT_OF_SELECTION_COMPLETE",
        scientific_role=(
            "Descriptive out-of-selection local-response replication with frozen "
            "alpha, scales, weights, instrument, and seeds; not patient validation"),
        alpha_star=alpha_star, seeds=seeds, observed=observed,
        seed_bootstrap_95_interval=intervals,
        comparison_to_selection=comparison,
        pair_rows=all_pairs,
        inputs=[dict(path=str(path), sha256=_sha256(path)) for path in inputs],
        arrays=dict(path=str(out_npz), sha256=_sha256(out_npz)),
        provenance=execution_provenance)
    atomic_write_json(payload, args.out_json)
    print(json.dumps(dict(
        status=payload["status"], alpha_star=alpha_star, observed=observed,
        intervals=intervals, comparison=comparison,
        arrays_sha256=payload["arrays"]["sha256"]), indent=2))


if __name__ == "__main__":
    main()

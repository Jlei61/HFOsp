#!/usr/bin/env python3
"""Zero-new-physics canary for the native unsupported-activity regularizer."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_native_activity_regularizer import (  # noqa: E402
    summarize_network_windows,
    unsupported_activity_for_window,
)
from src.topic4_node_dualmode import binned_ee_delay_support  # noqa: E402

SEARCH = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2_1"
OUT = SEARCH / "native_activity_regularizer_canary"

VARIANTS = {
    "primary": dict(rounding="nearest", threshold=0.001, quantile=0.95, reset=18, tail=13),
    "support_lo": dict(rounding="nearest", threshold=0.0005, quantile=0.95, reset=18, tail=13),
    "support_hi": dict(rounding="nearest", threshold=0.002, quantile=0.95, reset=18, tail=13),
    "delay_floor": dict(rounding="floor", threshold=0.001, quantile=0.95, reset=18, tail=13),
    "delay_ceil": dict(rounding="ceil", threshold=0.001, quantile=0.95, reset=18, tail=13),
    "background_median": dict(rounding="nearest", threshold=0.001, quantile=0.5, reset=18, tail=13),
    "reset_short": dict(rounding="nearest", threshold=0.001, quantile=0.95, reset=9, tail=13),
    "reset_long": dict(rounding="nearest", threshold=0.001, quantile=0.95, reset=27, tail=13),
    "tail_short": dict(rounding="nearest", threshold=0.001, quantile=0.95, reset=18, tail=7),
    "tail_long": dict(rounding="nearest", threshold=0.001, quantile=0.95, reset=18, tail=20),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _network_support(execution: dict, topology_seed: int) -> dict[str, np.ndarray]:
    record = execution["corrected_networks"][str(topology_seed)]
    path = Path(record["path"])
    if _sha(path) != record["sha256"]:
        raise RuntimeError(f"network cache changed: {path}")
    with path.open("rb") as stream:
        cache = pickle.load(stream)
    net = cache["net"]
    if int(net["NE"]) != int(cache["NE"]):
        raise RuntimeError("cached E population identity changed")
    output = {}
    for rounding in sorted({row["rounding"] for row in VARIANTS.values()}):
        cached = OUT / f"support_topology_{topology_seed}_{rounding}.npz"
        if cached.exists():
            with np.load(cached) as loaded:
                output[rounding] = np.asarray(loaded["support_by_lag"])
        else:
            output[rounding] = binned_ee_delay_support(
                net["ampa_by_delay"], np.asarray(net["pos"][:net["NE"]], float),
                dt_ms=float(cache["config"]["dt"]), frame_ms=2.0,
                bin_mm=1.0, sheet_mm=float(cache["config"]["L"]),
                delay_rounding=rounding,
            )["support_by_lag"]
            np.savez_compressed(cached, support_by_lag=output[rounding])
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = json.loads((SEARCH / "g3_scores.json").read_text())
    execution_path = SEARCH / "execution/confirmation_24s/execution_config.json"
    execution = json.loads(execution_path.read_text())
    support = {
        seed: _network_support(execution, seed) for seed in (6101, 6102)
    }
    event_rows: list[dict] = []
    unit_rows: list[dict] = []
    sources = {str(execution_path): _sha(execution_path)}
    for candidate in scores["candidates"]:
        candidate_id = candidate["candidate_id"]
        for unit, unit_record in sorted(candidate["units"].items()):
            worker_path = Path(unit_record["worker_path"])
            worker = json.loads(worker_path.read_text())
            topology_seed = int(worker["topology_seed"])
            observation_path = (
                worker_path.parent.parent / "repaired_observation" / worker_path.name
            )
            observation = json.loads(observation_path.read_text())
            with np.load(worker_path.with_suffix(".npz")) as arrays:
                movie = np.asarray(arrays["sheet_activity_counts"], float)
                frame_ms = float(arrays["sheet_activity_frame_ms"])
            if frame_ms != 2.0:
                raise RuntimeError("canary requires frozen 2 ms native movies")
            with np.load(observation_path.with_suffix(".npz")) as arrays:
                primary = np.asarray(arrays["primary_event_indices"], int)
            by_variant = {name: [] for name in VARIANTS}
            for event_id in primary:
                event = observation["events"][int(event_id)]
                start, stop = np.rint(np.asarray(event["window_ms"]) / frame_ms).astype(int)
                baseline_start = max(0, int(start) - 60)  # frozen 120 ms pre-window reference
                if baseline_start >= start:
                    raise RuntimeError("primary event lacks a pre-window baseline")
                for name, spec in VARIANTS.items():
                    result = unsupported_activity_for_window(
                        movie, support[topology_seed][spec["rounding"]],
                        start_frame=int(start), stop_frame=int(stop),
                        baseline_start_frame=baseline_start,
                        baseline_stop_frame=int(start),
                        minimum_active_neurons=2,
                        minimum_parent_support=float(spec["threshold"]),
                        reset_frames=int(spec["reset"]),
                        background_quantile=float(spec["quantile"]),
                        response_tail_frames=int(spec["tail"]),
                        response_tail_fraction=0.5,
                    )
                    by_variant[name].append(result)
                    event_rows.append({
                        "candidate_id": candidate_id, "unit": unit,
                        "topology_seed": topology_seed,
                        "dynamics_seed": int(worker["dynamics_seed"]),
                        "event_id": int(event_id), "variant": name,
                        **result.to_dict(),
                    })
            row = {
                "candidate_id": candidate_id, "unit": unit,
                "topology_seed": topology_seed,
                "dynamics_seed": int(worker["dynamics_seed"]),
                "patient_feature_loss": unit_record["score"]["loss_off"],
                "n_primary_events": int(len(primary)),
                "n_detected_windows": int(len(observation["events"])),
                "primary_window_fraction": float(len(primary) / max(1, len(observation["events"]))),
                "physical_status": worker.get("physical_status"),
            }
            for name, rows in by_variant.items():
                summary = summarize_network_windows(rows)
                row[f"{name}_mean_R"] = summary["mean_unsupported_fraction"]
                row[f"{name}_median_R"] = summary["median_unsupported_fraction"]
                row[f"{name}_background_mass"] = summary["mean_background_mass_per_frame"]
                row[f"{name}_background_active_bin_fraction"] = summary["mean_background_active_bin_fraction"]
            unit_rows.append(row)
            sources[str(worker_path)] = _sha(worker_path)
            sources[str(observation_path)] = _sha(observation_path)

    primary_values = np.asarray([row["primary_mean_R"] for row in unit_rows], float)
    sensitivity = []
    for name in VARIANTS:
        values = np.asarray([row[f"{name}_mean_R"] for row in unit_rows], float)
        rho = spearmanr(primary_values, values).statistic
        sensitivity.append({
            "variant": name, "spearman_vs_primary": float(rho),
            "mean_R": float(np.mean(values)), "median_R": float(np.median(values)),
            "minimum_R": float(np.min(values)), "maximum_R": float(np.max(values)),
        })
    auxiliary = {}
    for name in ["patient_feature_loss", "n_primary_events", "primary_window_fraction",
                 "primary_background_mass", "primary_background_active_bin_fraction"]:
        values = np.asarray([row[name] for row in unit_rows], float)
        auxiliary[name] = float(spearmanr(primary_values, values).statistic)
    nonprimary = [row for row in sensitivity if row["variant"] != "primary"]
    stable = all(np.isfinite(row["spearman_vs_primary"]) and row["spearman_vs_primary"] >= 0.7
                 for row in nonprimary)
    payload = {
        "status": "CANARY_PASS_READY_FOR_FROZEN_LAMBDA" if stable else "CANARY_SENSITIVITY_REVIEW_REQUIRED",
        "zero_new_physics": True,
        "n_units": len(unit_rows), "n_primary_events": len(event_rows) // len(VARIANTS),
        "statistical_unit": "one 24 s topology/dynamics network; events aggregated within network",
        "primary_contract": VARIANTS["primary"],
        "support_interpretation": "delayed frozen E-to-E structural support proxy, not measured synaptic current or causal proof",
        "seed_rule": "largest 8-connected above-background patch after a global reset is exempt",
        "self_legitimisation_forbidden": True,
        "background_and_low_count_mass_reported_separately": True,
        "sensitivity_rank_stable_at_rho_ge_0_7": stable,
        "sensitivity": sensitivity,
        "spearman_primary_R_vs_auxiliary": auxiliary,
        "sources": sources,
    }
    _write_csv(OUT / "events.csv", event_rows)
    _write_csv(OUT / "units.csv", unit_rows)
    _write_csv(OUT / "sensitivity.csv", sensitivity)
    _write_json(OUT / "canary.json", payload)
    print(json.dumps({"status": payload["status"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

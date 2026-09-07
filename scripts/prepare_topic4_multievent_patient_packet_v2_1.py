#!/usr/bin/env python3
"""Freeze real patient IDs, then extract the balanced time-resolved packet."""
from __future__ import annotations

import importlib.util
import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_topic4_xy_research as base


OUT = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2_1/patient_time_packet"
EVALUATOR = ROOT / "results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl"
TARGET = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/shaft_aware_patient_training_target.npz")
PRODUCER_PATH = Path("/home/honglab/leijiaxin/HFOsp/scripts/plot_topic5_interictal_event_envelope_field.py")


def _producer():
    # This task lives in an isolated worktree, while the explicitly requested
    # current producer has a few newly added helpers in the main checkout.
    import scripts as scripts_package
    import src as src_package
    main_root = PRODUCER_PATH.parents[1]
    for package, folder in ((scripts_package, main_root / "scripts"),
                            (src_package, main_root / "src")):
        if str(folder) not in package.__path__:
            package.__path__.append(str(folder))
    spec = importlib.util.spec_from_file_location("topic4_v21_patient_envelope_producer", PRODUCER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("patient envelope producer cannot be loaded")
    spec.loader.exec_module(module)
    return module


def _select_ids(labels, blocks, training_rows, raw_indices, raw_boundaries):
    rng = np.random.default_rng(202609070232)
    boundary_by_block = {int(row["block_id"]): row for row in raw_boundaries}
    records = []
    for mode in sorted(np.unique(labels)):
        chosen = []
        per_block = {}
        candidates_by_block = {}
        for local in np.flatnonzero(labels == mode):
            candidates_by_block.setdefault(int(blocks[local]), []).append(int(local))
        block_order = np.asarray(sorted(candidates_by_block), dtype=int)
        rng.shuffle(block_order)
        for block in block_order:
            rng.shuffle(candidates_by_block[int(block)])
        # Round-robin prevents dense blocks from crowding out block coverage.
        while len(chosen) < 64:
            changed = False
            for block in block_order:
                block = int(block)
                if per_block.get(block, 0) >= 8 or not candidates_by_block[block]:
                    continue
                local = candidates_by_block[block].pop()
                chosen.append(local)
                per_block[block] = per_block.get(block, 0) + 1
                changed = True
                if len(chosen) == 64:
                    break
            if not changed:
                break
        if len(chosen) < 64:
            raise RuntimeError(f"mode {mode} has only {len(chosen)} IDs under block cap")
        for order, local in enumerate(chosen):
            raw_index = int(raw_indices[local])
            block = int(blocks[local])
            boundary = boundary_by_block[block]
            if not boundary["start_event_idx"] <= raw_index < boundary["end_event_idx"]:
                raise RuntimeError("raw event ID does not fall inside its recorded block")
            records.append({
                "mode": int(mode), "selection_order": order,
                "packet_role": "primary" if order < 32 else "reserve",
                "evaluator_fit_row": int(training_rows[local]),
                "raw_global_event_index": raw_index,
                "block_id": block,
                "record_name": boundary["record_name"],
                "block_local_event_index": raw_index - int(boundary["start_event_idx"]),
            })
    return records


def freeze_selection():
    path = OUT / "selection.json"
    if path.exists():
        return base.read(path)
    OUT.mkdir(parents=True, exist_ok=True)
    qualification = base.read(EVALUATOR.parent / "qualification.json")
    if base.sha(EVALUATOR) != qualification["evaluator_sha256"]:
        raise RuntimeError("frozen repaired evaluator changed")
    with open(EVALUATOR, "rb") as stream:
        evaluator = pickle.load(stream)
    producer = _producer()
    frozen = producer.load_frozen("epilepsiae_1146")
    raw = producer.load_events(frozen, "1146")
    with np.load(TARGET) as target:
        all_raw_indices = np.asarray(target["patient_train_event_indices"], int)
        all_blocks = np.asarray(target["patient_train_block_ids"], int)
        all_names = target["contact_names"].astype(str).tolist()
    training_rows = np.asarray(evaluator.index["FIT"], int)
    raw_indices = all_raw_indices[training_rows]
    blocks = all_blocks[training_rows]
    if not np.array_equal(blocks, np.asarray(evaluator.blocks)[training_rows]):
        raise RuntimeError("evaluator blocks do not map to the upstream target rows")
    if all_names != list(evaluator.km.feature_names_in_) if hasattr(evaluator.km, "feature_names_in_") else False:
        raise RuntimeError("unexpected named KMeans feature interface")
    records = _select_ids(
        np.asarray(evaluator.fit_labels, int), blocks, training_rows,
        raw_indices, raw["block_boundaries"] if "block_boundaries" in raw else
        producer.load_subject_propagation_events(producer.LAGPAT_ROOT / "1146" / "all_recs")["block_boundaries"],
    )
    payload = {
        "status": "PATIENT_PACKET_IDS_FROZEN_BEFORE_MODEL_NOMINATION",
        "rng_seed": 202609070232,
        "subject": "epilepsiae_1146",
        "selection_rule": "FIT labels only; round-robin blocks; maximum 8 across primary plus reserve per mode and block",
        "events_per_mode_primary": 32, "events_per_mode_reserve": 32,
        "route_or_model_performance_used": False,
        "natural_mode_frequency_estimated_by_balanced_packet": False,
        "natural_mode_counts_FIT": np.bincount(
            evaluator.fit_labels, minlength=evaluator.k,
        ).astype(int).tolist(),
        "upstream_target_path": str(TARGET), "upstream_target_sha256": base.sha(TARGET),
        "evaluator_path": str(EVALUATOR), "evaluator_sha256": base.sha(EVALUATOR),
        "producer_path": str(PRODUCER_PATH), "producer_sha256": base.sha(PRODUCER_PATH),
        "contact_names": all_names,
        "events": records,
    }
    base.write(path, payload)
    return payload


def _quantiles(time_ms, positive):
    total = float(positive.sum())
    if not total > 0:
        return [None, None, None]
    cumulative = np.cumsum(positive) / total
    return [float(time_ms[min(np.searchsorted(cumulative, q), len(time_ms) - 1)])
            for q in (0.1, 0.5, 0.9)]


def _extract_one(producer, frozen, raw, inventory, selection):
    position = int(selection["raw_global_event_index"])
    event = producer.build_event(raw, position, inventory, "1146", frozen)
    time_ms = (np.asarray(event["t"], float) - float(event["t0"])) * 1000.0
    envelope_z = np.asarray(event["env_z"], np.float32)
    positive = np.maximum(envelope_z, 0.0)
    packed = (np.asarray(event["t"]) >= event["w_lo"]) & (
        np.asarray(event["t"]) < event["w_hi"]
    )
    if not packed.any():
        raise RuntimeError("packed window absent from extracted trace")
    packed_time = time_ms[packed]
    packed_positive = positive[:, packed]
    dt_ms = float(np.median(np.diff(time_ms)))
    edge_n = max(1, int(np.ceil(10.0 / dt_ms)))
    aggregate = packed_positive[np.asarray(event["part"], bool)].sum(axis=0)
    mass = float(aggregate.sum())
    left = float(aggregate[:edge_n].sum() / mass) if mass > 0 else None
    right = float(aggregate[-edge_n:].sum() / mass) if mass > 0 else None
    known_clipped = bool(
        event["w_lo"] <= 0.0
        or np.asarray(event["t"])[-1] + dt_ms / 1000.0 < event["w_hi"]
    )
    suspected = bool(not known_clipped and max(left or 0.0, right or 0.0) > 0.10)
    state = ("known_window_or_record_clipped" if known_clipped else
             "edge_mass_suspected" if suspected else "complete")
    quantiles = _quantiles(packed_time, aggregate)
    output = OUT / "events" / (
        f"mode{selection['mode']}_{selection['packet_role']}_{selection['selection_order']:02d}_"
        f"raw{position}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(
            stream,
            contact_names=np.asarray(frozen["names"]),
            time_ms=time_ms.astype(np.float32),
            envelope_background_robust_z=envelope_z,
            positive_envelope_mass=positive.astype(np.float32),
            participation_mask=np.asarray(event["part"], bool),
            valid_quality_mask=np.isfinite(envelope_z).all(axis=1),
            packed_window_mask=packed,
            stored_lag_ms=np.asarray(event["stored"], np.float32),
            fig1a_centroid_ms=np.asarray(event["centroid_ms"], np.float32),
        )
    temporary.replace(output)
    return {
        **selection,
        "status": state,
        "arrays_path": str(output), "arrays_sha256": base.sha(output),
        "producer_record": event["stem"],
        "packed_window_sec": [float(event["w_lo"]), float(event["w_hi"])],
        "packed_window_ms": float(event["packed_window_ms"]),
        "sampling_frequency_hz": float(event["fs"]),
        "n_participating": int(event["n_part"]),
        "positive_mass": mass,
        "valid_sample_fraction": float(np.isfinite(envelope_z).mean()),
        "t10_t50_t90_ms": quantiles,
        "quantile_interpretation": (
            "window-conditional" if known_clipped else "within frozen packed window"
        ),
        "left_10ms_mass_fraction": left,
        "right_10ms_mass_fraction": right,
        "edge_mass_suspected_threshold": 0.10,
        "edge_threshold_is_sensitivity_diagnostic_not_known_clipping_fact": True,
    }


def extract_packet(selection):
    producer = _producer()
    frozen = producer.load_frozen("epilepsiae_1146")
    raw = producer.load_events(frozen, "1146")
    inventory = producer._inventory("1146")
    rows, failures = [], []
    for mode in (0, 1):
        available = [row for row in selection["events"] if row["mode"] == mode]
        readable = 0
        for selected in available:
            if selected["packet_role"] == "reserve" and readable >= 32:
                break
            try:
                row = _extract_one(producer, frozen, raw, inventory, selected)
                rows.append(row)
                readable += 1
            except Exception as exc:
                failures.append({
                    **selected, "status": "unreadable", "reason": repr(exc),
                    "traceback": traceback.format_exc(limit=3),
                })
    report = {
        "status": "PATIENT_TIME_PACKET_EXTRACTED" if all(
            sum(row["mode"] == mode for row in rows) >= 32 for mode in (0, 1)
        ) else "PATIENT_TIME_PACKET_PARTIAL_RAW_DATA_GAP",
        "selection_sha256": base.sha(OUT / "selection.json"),
        "producer_sha256": base.sha(PRODUCER_PATH),
        "readable_events": rows, "unreadable_events": failures,
        "readable_by_mode": {str(mode): sum(row["mode"] == mode for row in rows)
                             for mode in (0, 1)},
        "balanced_packet_estimates_natural_frequency": False,
        "candidate_comparison_opened": False,
    }
    base.write(OUT / "packet_manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    selection = freeze_selection()
    if args.selection_only:
        print(json.dumps({
            "status": selection["status"],
            "events": len(selection["events"]),
        }, indent=2))
        return
    result = extract_packet(selection)
    print(json.dumps({
        "status": result["status"],
        "readable_by_mode": result["readable_by_mode"],
        "unreadable": len(result["unreadable_events"]),
    }, indent=2))


if __name__ == "__main__":
    main()

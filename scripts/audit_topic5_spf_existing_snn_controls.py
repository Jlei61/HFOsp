#!/usr/bin/env python3
"""Freeze an artifact-only SNN inventory for the SPF-RNN positive control.

This script never imports or calls the simulator. It validates existing Topic
4 readouts, converts their event-level contact ranks into model-ready arrays,
and records exactly which perturbations are already available.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import (  # noqa: E402
    CONTRACT_NAME,
    sha256_file,
    validate_rank_event_arrays,
)
from src.sef_hfo_observation import endpoint_centroid_axis  # noqa: E402

SOURCE_ROOT = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"
OUTPUT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/snn_positive_control"
    / "existing_artifact_reuse"
)
FAMILIES = {
    "source_only": "readout_epilepsiae_1146_source_tsrc_highn_s*.json",
    "sink_only": "readout_epilepsiae_1146_sink_tsrc_highn_s*.json",
    "paired_source_sink": "readout_epilepsiae_1146_paired_tsrc_highn_s*.json",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _dense_rank_event(
    ranks: dict[str, Any], contact_names: list[str]
) -> tuple[np.ndarray, int] | None:
    values = [ranks.get(name) for name in contact_names]
    observed = sorted({float(value) for value in values if value is not None})
    if len(observed) < 2:
        return None
    mapping = {value: index for index, value in enumerate(observed)}
    group_ids = np.full(len(contact_names), -1, dtype=np.int16)
    for index, value in enumerate(values):
        if value is not None:
            group_ids[index] = mapping[float(value)]
    return group_ids, len(observed)


def _family(label: str, pattern: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    paths = sorted(SOURCE_ROOT.glob(pattern))
    if not paths:
        raise RuntimeError(f"missing existing SNN family: {label}")
    contact_names: list[str] | None = None
    groups: list[np.ndarray] = []
    counts: list[int] = []
    seeds: list[int] = []
    source_event: list[int] = []
    reported_signs: list[float] = []
    recomputed_signs: list[float] = []
    files = []
    raw_events = 0
    fwd = rev = 0
    for run_index, path in enumerate(paths):
        payload = _json(path)
        partner = Path(
            str(path).replace("/readout_", "/figdata_").replace(".json", ".npz")
        )
        if not partner.exists():
            raise RuntimeError(f"readout lacks figdata partner: {path}")
        with np.load(partner, allow_pickle=True) as figdata:
            fig_names = [str(value) for value in figdata["names"]]
            contact_coords = np.asarray(figdata["contacts"], dtype=float)
            axis_unit = np.asarray(
                figdata["reg"].item()["axis_unit"], dtype=float
            )
        events = payload.get("events", [])
        raw_events += len(events)
        for event_index, event in enumerate(events):
            rank_map = event.get("ranks", {})
            names = list(rank_map)
            if contact_names is None:
                contact_names = names
            if names != contact_names:
                raise RuntimeError(f"contact order drift in {path}")
            if names != fig_names:
                raise RuntimeError(f"rank/figdata contact order drift in {path}")
            converted = _dense_rank_event(rank_map, contact_names)
            if converted is None:
                continue
            group_ids, group_count = converted
            groups.append(group_ids)
            counts.append(group_count)
            seeds.append(int(payload["seed"]))
            source_event.append(event_index)
            sign = event.get("sign")
            reported_signs.append(np.nan if sign is None else float(sign))
            direction_axis = endpoint_centroid_axis(
                group_ids,
                group_ids >= 0,
                contact_coords,
                k_dir=int(payload["k_dir"]),
                eps_deg=2.0,
            )
            recomputed = (
                np.nan
                if direction_axis is None
                else float(np.sign(np.dot(direction_axis, axis_unit)))
            )
            recomputed_signs.append(recomputed)
            fwd += int(np.isfinite(recomputed) and recomputed > 0)
            rev += int(np.isfinite(recomputed) and recomputed < 0)
        files.append(
            {
                "readout": str(path.relative_to(ROOT)),
                "readout_sha256": sha256_file(path),
                "figdata": str(partner.relative_to(ROOT)),
                "figdata_sha256": sha256_file(partner),
                "seed": int(payload["seed"]),
                "lesion": payload["lesion"],
                "placement": payload.get("placement", "template_source"),
                "n_events_reported": int(payload["n_events"]),
            }
        )
    assert contact_names is not None
    group_array = np.asarray(groups, dtype=np.int16)
    count_array = np.asarray(counts, dtype=np.int16)
    validate_rank_event_arrays(group_array, count_array)
    arrays = {
        "contact_names": np.asarray(contact_names, dtype="U"),
        "event_group_ids": group_array,
        "event_group_count": count_array,
        "source_seed": np.asarray(seeds, dtype=np.int16),
        "source_event_index": np.asarray(source_event, dtype=np.int32),
        # Historical single-core payloads say k_dir=2, while their stored sign
        # was produced through an older imported default. Preserve that value
        # for audit and expose one explicit readout across every family.
        "event_direction_sign": np.asarray(
            recomputed_signs, dtype=np.float32
        ),
        "event_direction_sign_reported": np.asarray(
            reported_signs, dtype=np.float32
        ),
    }
    summary = {
        "family": label,
        "n_files": len(paths),
        "seeds": sorted({int(seed) for seed in seeds}),
        "n_raw_events": int(raw_events),
        "n_model_ready_events": int(len(group_array)),
        "n_contacts": len(contact_names),
        "rank_count_median": float(np.median(count_array)),
        "direction_forward": int(fwd),
        "direction_reverse": int(rev),
        "direction_unreadable": int(
            np.sum(~np.isfinite(np.asarray(recomputed_signs, dtype=float)))
        ),
        "reported_direction_forward": int(
            np.sum(np.asarray(reported_signs, dtype=float) > 0)
        ),
        "reported_direction_reverse": int(
            np.sum(np.asarray(reported_signs, dtype=float) < 0)
        ),
        "reported_direction_unreadable": int(
            np.sum(~np.isfinite(np.asarray(reported_signs, dtype=float)))
        ),
        "direction_readout_contract": (
            "recomputed from saved ranks/coordinates with explicit k_dir="
            f"{int(payload['k_dir'])}, eps_deg=2.0"
        ),
        "files": files,
    }
    return summary, arrays


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    family_summaries = []
    for label, pattern in FAMILIES.items():
        summary, arrays = _family(label, pattern)
        target = OUTPUT_ROOT / f"rank_events_{label}.npz"
        np.savez_compressed(target, **arrays)
        summary["rank_event_artifact"] = str(target.relative_to(ROOT))
        summary["rank_event_sha256"] = sha256_file(target)
        family_summaries.append(summary)

    yield_root = (
        ROOT
        / "results/topic5_shared_propagation_field/snn_positive_control/yield_probe"
    )
    yield_conditions = {}
    for path in sorted(yield_root.glob("readout_*.json")):
        payload = _json(path)
        yield_conditions[path.stem.removeprefix("readout_")] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "n_events": int(payload.get("n_events", 0)),
            "dir_forward": int(payload.get("dir_forward", 0)),
            "dir_reverse": int(payload.get("dir_reverse", 0)),
        }

    payload = {
        "contract": CONTRACT_NAME,
        "status": "EXISTING_SNN_ARTIFACTS_AUDITED_NO_SIMULATION_RUN",
        "source_root": str(SOURCE_ROOT.relative_to(ROOT)),
        "source_code_sha256": {
            "runner": sha256_file(ROOT / "scripts/run_sef_hfo_subject_snn.py"),
            "placement": sha256_file(ROOT / "src/sef_hfo_subject_placement.py"),
        },
        "families": family_summaries,
        "yield_probe_conditions": yield_conditions,
        "mechanism_contract": (
            "Propagation direction is set by the location/identity of the "
            "low-threshold pathological kernel or core. E-to-E anisotropy "
            "shapes the propagation channel but removing anisotropy is not "
            "expected, by itself, to erase direction."
        ),
        "reuse_decision": {
            "rerun_snn_simulation": False,
            "rank_event_conversion_available": True,
            "source_vs_sink_perturbation_available": True,
            "paired_repertoire_available": True,
            "direction_labels_recomputed_from_existing_artifacts": True,
            "reported_direction_labels_preserved_for_audit": True,
            "isotropic_yield_probe_role": (
                "diagnostic_only_not_a_direction-erasure_negative_control"
            ),
            "g0_status": (
                "INPUT_AUDIT_ONLY; SEE "
                "results/topic5_shared_propagation_field/snn_positive_control/"
                "existing_artifact_system_identification/ROUND_STATE.json "
                "FOR THE SEPARATE MODEL SCORE"
            ),
        },
    }
    inventory = OUTPUT_ROOT / "existing_snn_artifact_inventory.json"
    inventory.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (OUTPUT_ROOT / "family_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "n_files",
                "n_raw_events",
                "n_model_ready_events",
                "n_contacts",
                "rank_count_median",
                "direction_forward",
                "direction_reverse",
                "direction_unreadable",
                "reported_direction_forward",
                "reported_direction_reverse",
                "reported_direction_unreadable",
                "rank_event_artifact",
                "rank_event_sha256",
            ],
        )
        writer.writeheader()
        for row in family_summaries:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    done = {
        "contract": CONTRACT_NAME,
        "status": "COMPLETE",
        "inventory_sha256": sha256_file(inventory),
        "source_tree_fingerprint": hashlib.sha256(
            "".join(
                item["readout_sha256"]
                for family in family_summaries
                for item in family["files"]
            ).encode()
        ).hexdigest(),
    }
    (OUTPUT_ROOT / "ARTIFACT_AUDIT_STATE.json").write_text(
        json.dumps(done, indent=2) + "\n"
    )
    print(json.dumps(payload["reuse_decision"], indent=2))


if __name__ == "__main__":
    main()

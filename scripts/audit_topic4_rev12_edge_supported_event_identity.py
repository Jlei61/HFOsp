#!/usr/bin/env python3
"""Audit whether local causal roots split delayed E-to-E propagation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.sef_hfo_events import detect_events  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    assign_detector_fragments_to_directed_lineages,
    binned_ee_delay_support,
    causal_root_event_windows,
    edge_supported_root_families,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev12_nd_causal_root_field_fit.json"
DEFAULT_CANDIDATES = (
    "stage_i_a01_d00_s01_p",
    "stage_i_a01_d00_s01_m",
    "stage_i_a00_d01_s01_p",
    "stage_i_anchor_02",
)
ROUNDINGS = ("floor", "nearest", "ceil")
SUPPORT_THRESHOLDS = (0.0003, 0.001, 0.003)
PRIMARY_ROUNDING = "nearest"
PRIMARY_THRESHOLD = 0.001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant_key(rounding: str, threshold: float) -> str:
    return f"{rounding}_support_{threshold:.4f}"


def _one_variant(*, counts: np.ndarray, local_labels: np.ndarray,
                 active_fraction: np.ndarray, active_dt_ms: float,
                 detector_threshold: float, detector_fragments: list[dict],
                 support: np.ndarray, minimum_active_neurons: int,
                 minimum_parent_support: float,
                 minimum_parent_dominance: float,
                 detector_minimum_dominance: float,
                 frame_ms: float, total_ms: float,
                 original_root_ids: np.ndarray,
                 original_fragment_roots: np.ndarray,
                 original_fragment_compound: np.ndarray) -> dict:
    family = edge_supported_root_families(
        counts, local_labels, support,
        minimum_active_neurons=minimum_active_neurons,
        minimum_parent_support=minimum_parent_support,
        minimum_parent_dominance=minimum_parent_dominance,
    )
    assignments = assign_detector_fragments_to_directed_lineages(
        counts, family["labels"], detector_fragments,
        frame_ms=frame_ms, minimum_dominance=detector_minimum_dominance,
    )
    events, compounds = causal_root_event_windows(
        family["components"], assignments, detector_fragments,
        frame_ms=frame_ms, total_ms=total_ms,
    )
    new_compound = np.asarray([row["compound"] for row in assignments], bool)
    new_roots = np.asarray([
        -1 if row["dominant_lineage_id"] is None
        else int(row["dominant_lineage_id"])
        for row in assignments
    ], int)
    clean_root_merged = np.asarray([
        family["family_map"].get(int(root_id), int(root_id)) != int(root_id)
        for root_id in original_root_ids
    ], bool)
    original_clean = ~original_fragment_compound
    comparable = original_clean & ~new_compound
    expected_new_roots = np.asarray([
        family["family_map"].get(int(root_id), int(root_id))
        if root_id > 0 else -1
        for root_id in original_fragment_roots
    ], int)
    root_disagreement = comparable & (new_roots != expected_new_roots)
    return {
        "n_local_roots": family["n_local_roots"],
        "n_edge_supported_families": family["n_edge_supported_families"],
        "n_merged_births": family["n_merged_births"],
        "n_original_clean_events": int(len(original_root_ids)),
        "n_original_clean_event_roots_merged": int(np.sum(clean_root_merged)),
        "clean_event_root_merge_fraction": float(np.mean(clean_root_merged))
        if len(clean_root_merged) else 0.0,
        "n_original_detector_fragments": int(len(detector_fragments)),
        "original_compound_fraction": float(np.mean(original_fragment_compound)),
        "edge_supported_compound_fraction": float(np.mean(new_compound)),
        "n_compound_to_clean": int(np.sum(original_fragment_compound & ~new_compound)),
        "n_clean_to_compound": int(np.sum(original_clean & new_compound)),
        "n_clean_root_disagreements_after_family_mapping": int(
            np.sum(root_disagreement)
        ),
        "n_edge_supported_clean_events": int(len(events)),
        "n_edge_supported_compound_fragments": int(len(compounds)),
        "event_family_map": {
            str(root_id): int(family["family_map"].get(int(root_id), int(root_id)))
            for root_id in original_root_ids
        },
        "merged_births": [
            row for row in family["birth_audit"] if row["merged"]
        ],
        "detector_threshold_recomputed": float(detector_threshold),
        "active_fraction_samples": int(len(active_fraction)),
        "active_fraction_dt_ms": float(active_dt_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    artifact_root = args.artifact_root.resolve()
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    by_id = {row["candidate_id"]: row for row in manifest["candidates"]}
    candidate_ids = tuple(args.candidate_ids or DEFAULT_CANDIDATES)
    seeds = tuple(args.seeds or config["search"]["fit_network_seeds"])
    missing = sorted(set(candidate_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"audit candidates absent from manifest: {missing}")
    event_unit = config["event_unit"]
    transition = load_round_config(
        ROOT / config["inputs"]["transition_config"]["path"]
    )
    rows = []
    support_cache: dict[tuple[int, str], dict] = {}
    for seed in seeds:
        substrate = build_substrate(
            transition, "node_baseline", int(seed),
            cache_dir=str(artifact_root / config["network_cache"]),
            ee_dose=0.0, etoi_dose=0.0,
            node_candidate_override=by_id[candidate_ids[0]]["node_field"],
            artifact_root=artifact_root,
        )
        for rounding in ROUNDINGS:
            support_cache[(int(seed), rounding)] = binned_ee_delay_support(
                substrate.net["ampa_by_delay"], substrate.positions_e,
                dt_ms=float(substrate.engine["dt"]),
                frame_ms=float(event_unit["movie_frame_ms"]),
                bin_mm=float(event_unit["movie_bin_mm"]),
                sheet_mm=float(substrate.engine["L"]),
                delay_rounding=rounding,
            )
        for candidate_id in candidate_ids:
            stem = f"{candidate_id}_seed_{seed}"
            npz_path = artifact_root / config["output_root"] / "workers" / f"{stem}.npz"
            json_path = artifact_root / config["output_root"] / "workers" / f"{stem}.json"
            if not npz_path.exists() or not json_path.exists():
                raise FileNotFoundError(f"missing fit worker: {stem}")
            worker_json = json.loads(json_path.read_text())
            with np.load(npz_path, allow_pickle=False) as loaded:
                counts = np.asarray(loaded["sheet_activity_counts"])
                local_labels = np.asarray(loaded["directed_lineage_labels"], int)
                active_fraction = np.asarray(loaded["active_fraction"], float)
                active_dt_ms = float(loaded["active_fraction_bin_ms"])
                original_root_ids = np.asarray(loaded["event_directed_root_id"], int)
                original_fragment_roots = np.asarray(
                    loaded["detector_fragment_dominant_lineage_id"], int,
                )
                original_fragment_compound = np.asarray(
                    loaded["detector_fragment_compound"], bool,
                )
            detector_threshold = float(worker_json["event_unit"]["event_on_threshold"])
            detector_fragments = detect_events(
                active_fraction, active_dt_ms,
                event_on_frac=detector_threshold,
            )
            if len(detector_fragments) != len(original_fragment_compound):
                raise RuntimeError("recomputed detector fragments differ from worker")
            variants = {}
            for rounding in ROUNDINGS:
                edge = support_cache[(int(seed), rounding)]
                for threshold in SUPPORT_THRESHOLDS:
                    variants[_variant_key(rounding, threshold)] = _one_variant(
                        counts=counts, local_labels=local_labels,
                        active_fraction=active_fraction,
                        active_dt_ms=active_dt_ms,
                        detector_threshold=detector_threshold,
                        detector_fragments=detector_fragments,
                        support=edge["support_by_lag"],
                        minimum_active_neurons=int(event_unit["minimum_active_neurons"]),
                        minimum_parent_support=float(threshold),
                        minimum_parent_dominance=float(event_unit["minimum_dominance"]),
                        detector_minimum_dominance=float(event_unit["minimum_dominance"]),
                        frame_ms=float(event_unit["movie_frame_ms"]),
                        total_ms=len(active_fraction) * active_dt_ms,
                        original_root_ids=original_root_ids,
                        original_fragment_roots=original_fragment_roots,
                        original_fragment_compound=original_fragment_compound,
                    )
            rows.append({
                "candidate_id": candidate_id,
                "seed": int(seed),
                "worker_json": str(json_path.relative_to(artifact_root)),
                "worker_json_sha256": _sha256(json_path),
                "variants": variants,
            })

    primary_key = _variant_key(PRIMARY_ROUNDING, PRIMARY_THRESHOLD)
    primary = [row["variants"][primary_key] for row in rows]
    sensitivity_keys = [
        _variant_key(rounding, threshold)
        for rounding in ROUNDINGS for threshold in SUPPORT_THRESHOLDS
    ]
    verdict = {
        "primary_key": primary_key,
        "primary_clean_event_roots_merged": int(sum(
            row["n_original_clean_event_roots_merged"] for row in primary
        )),
        "primary_compound_to_clean": int(sum(
            row["n_compound_to_clean"] for row in primary
        )),
        "primary_clean_to_compound": int(sum(
            row["n_clean_to_compound"] for row in primary
        )),
        "maximum_clean_event_root_merge_fraction_over_all_variants": float(max(
            row["variants"][key]["clean_event_root_merge_fraction"]
            for row in rows for key in sensitivity_keys
        )),
    }
    verdict["status"] = (
        "EDGE_SUPPORTED_EVENT_IDENTITY_PRIMARY_UNCHANGED"
        if verdict["primary_clean_event_roots_merged"] == 0
        and verdict["primary_compound_to_clean"] == 0
        and verdict["primary_clean_to_compound"] == 0
        else "EDGE_SUPPORTED_EVENT_IDENTITY_CHANGES_PRIMARY"
    )
    output = {
        "schema_id": "topic4_rev12_edge_supported_event_identity_audit_v1",
        "scientific_role": "event_identity_audit_only_no_field_selection",
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "manifest_path": str(manifest_path.relative_to(artifact_root)),
        "manifest_sha256": _sha256(manifest_path),
        "candidate_ids": list(candidate_ids),
        "network_seeds": list(seeds),
        "delay_roundings": list(ROUNDINGS),
        "minimum_parent_support_values": list(SUPPORT_THRESHOLDS),
        "minimum_parent_dominance": float(event_unit["minimum_dominance"]),
        "support_definition": (
            "active-source E-to-E delayed weight divided by source-bin population "
            "and target-bin total incoming-E budget"
        ),
        "verdict": verdict,
        "workers": rows,
    }
    output_path = args.output or (
        artifact_root / config["output_root"] / "aggregate"
        / "edge_supported_event_identity_audit.json"
    )
    atomic_write_json(output, output_path)
    print(json.dumps(verdict, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()

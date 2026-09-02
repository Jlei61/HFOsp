#!/usr/bin/env python3
"""Audit the frozen ECoG event, split, suffix-null, and graph inputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    split_rows = list(csv.DictReader((ROOT / "feasibility/BLOCK_SPLIT.csv").open()))
    inventory_rows = list(csv.DictReader(Path(
        "/home/honglab/leijiaxin/HFOsp/results/epilepsiae_block_inventory.csv"
    ).open()))
    inventory = {row["block_stem"]: row for row in inventory_rows}
    patient = []
    for subject in ("958", "1084"):
        rows = [row for row in split_rows if row["subject"] == subject]
        stems_by_split = {
            split: {row["block_stem"] for row in rows if row["split"] == split}
            for split in ("train", "validation", "test")
        }
        split_disjoint = all(
            stems_by_split[first].isdisjoint(stems_by_split[second])
            for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))
        )
        event_path = ROOT / "cache" / subject / "events.npz"
        with np.load(event_path, allow_pickle=False) as events:
            ranks = np.asarray(events["ranks"], dtype=np.int16)
            participation = np.asarray(events["participation"], dtype=bool)
            split = np.asarray(events["split"], dtype=np.int8)
            names = [str(value) for value in events["channel_names"].tolist()]
            event_epoch = np.asarray(events["event_epoch"], dtype=float)
            event_block_index = np.asarray(events["block_index"], dtype=int)
        dense = True
        for row in ranks:
            observed = np.unique(row[row >= 0])
            if not np.array_equal(observed, np.arange(len(observed))):
                dense = False
                break
        suffix_audit = []
        for null_index in range(3):
            path = ROOT / "cache" / subject / f"events_suffix_null_seed{null_index}.npz"
            with np.load(path, allow_pickle=False) as null:
                null_ranks = np.asarray(null["ranks"], dtype=np.int16)
            suffix_audit.append({
                "null_index": null_index,
                "test_exact": bool(np.array_equal(null_ranks[split == 2], ranks[split == 2])),
                "development_first_three_exact": bool(np.array_equal(
                    np.where((ranks[split != 2] >= 0) & (ranks[split != 2] < 3), ranks[split != 2], -1),
                    np.where((null_ranks[split != 2] >= 0) & (null_ranks[split != 2] < 3), null_ranks[split != 2], -1),
                )),
                "development_changed_event_fraction": float(np.mean(np.any(
                    null_ranks[split != 2] != ranks[split != 2], axis=1
                ))),
                "sha256": sha256_file(path),
            })
        graph_root = ROOT / "graphs" / subject / "four_neighbour"
        graph_paths = sorted(graph_root.glob("*.npz"))
        graph_masks = []
        graph_records = []
        with np.load(graph_root / "TRUE_GRID.npz", allow_pickle=False) as true_artifact:
            true_mask = np.asarray(true_artifact["mask"], dtype=np.uint8)
        for path in graph_paths:
            with np.load(path, allow_pickle=False) as artifact:
                mask = np.asarray(artifact["mask"], dtype=np.uint8)
                family = str(artifact["family"].item())
            graph_masks.append(mask.tobytes())
            graph_records.append({
                "graph": path.stem,
                "family": family,
                "symmetric": bool(np.array_equal(mask, mask.T)),
                "same_directed_edge_count": int(mask.sum()) == int(true_mask.sum()),
                "same_per_node_degree": bool(np.array_equal(mask.sum(0), true_mask.sum(0))),
                "sha256": sha256_file(path),
            })
        train_participation = participation[split == 0].sum(0)
        test_continue_decisions = int(np.sum(np.max(ranks[split == 2], axis=1)))
        epoch_inside = True
        cache_sql_start_exact = True
        for block_index, row in enumerate(rows):
            inventory_row = inventory[row["block_stem"]]
            start = float(inventory_row["block_start_epoch"])
            end = float(inventory_row["block_end_epoch"])
            selected_epoch = event_epoch[event_block_index == block_index]
            epoch_inside &= bool(np.all((selected_epoch >= start) & (selected_epoch <= end)))
            block_cache = ROOT / "cache" / subject / "per_block" / f"{row['block_stem']}.npz"
            with np.load(block_cache, allow_pickle=False) as block:
                cache_sql_start_exact &= float(block["block_start_epoch"].item()) == start
        patient.append({
            "subject": subject,
            "n_contacts": len(names),
            "n_events": len(ranks),
            "split_counts": {str(code): int(np.sum(split == code)) for code in (0, 1, 2)},
            "block_splits_disjoint": split_disjoint,
            "nonparticipant_minus_one_exact": bool(np.array_equal(ranks < 0, ~participation)),
            "dense_rank_sets": dense,
            "all_train_contacts_observed": bool(np.all(train_participation > 0)),
            "minimum_train_contact_events": int(train_participation.min()),
            "test_continue_decisions": test_continue_decisions,
            "event_epoch_inside_sql_block_range": epoch_inside,
            "per_block_cache_uses_sql_start": cache_sql_start_exact,
            "suffix_nulls": suffix_audit,
            "n_graphs": len(graph_records),
            "n_unique_graph_masks": len(set(graph_masks)),
            "graph_records": graph_records,
            "events_sha256": sha256_file(event_path),
        })
    gates = {
        "all_block_splits_disjoint": all(row["block_splits_disjoint"] for row in patient),
        "all_nonparticipants_minus_one": all(row["nonparticipant_minus_one_exact"] for row in patient),
        "all_rank_sets_dense": all(row["dense_rank_sets"] for row in patient),
        "all_event_epochs_inside_sql_blocks": all(row["event_epoch_inside_sql_block_range"] for row in patient),
        "all_per_block_caches_use_sql_start": all(row["per_block_cache_uses_sql_start"] for row in patient),
        "all_train_contacts_observed": all(row["all_train_contacts_observed"] for row in patient),
        "all_test_decisions_above_1000": all(row["test_continue_decisions"] >= 1000 for row in patient),
        "all_suffix_test_exact": all(
            null["test_exact"] for row in patient for null in row["suffix_nulls"]
        ),
        "all_suffix_prefix_exact": all(
            null["development_first_three_exact"] for row in patient for null in row["suffix_nulls"]
        ),
        "all_63_graphs_unique": all(row["n_graphs"] == row["n_unique_graph_masks"] == 63 for row in patient),
        "all_graph_degrees_matched": all(
            graph["same_per_node_degree"] for row in patient for graph in row["graph_records"]
        ),
    }
    payload = {
        "schema": "topic5_ecog_input_audit_v0.1",
        "gates": gates,
        "pass": all(gates.values()),
        "patients": patient,
    }
    output = ROOT / "INPUT_AUDIT.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["pass"]:
        raise RuntimeError(json.dumps(gates, indent=2))
    print(json.dumps({"pass": True, "gates": gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

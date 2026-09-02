#!/usr/bin/env python3
"""Aggregate per-block ECoG ranks and freeze cross-event suffix controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_train_only_modes_suffix_null_v0_5 import (  # noqa: E402
    apply_suffix_mapping,
    suffix_mapping,
)


NULL_SEEDS = (2026081601, 2026081602, 2026081603)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, subject: str) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["subject"] == subject]
    rows.sort(key=lambda row: (int(row["recording_id"]), int(row["block_index"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=("958", "1084"))
    parser.add_argument("--feasibility-root", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/feasibility"))
    parser.add_argument("--cache-root", type=Path, default=Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"))
    args = parser.parse_args()

    rows = load_rows(args.feasibility_root / "BLOCK_SPLIT.csv", args.subject)
    per_block = args.cache_root / args.subject / "per_block"
    expected_paths = [per_block / f"{row['block_stem']}.npz" for row in rows]
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} per-block rank caches; first={missing[0]}")

    names_ref: list[str] | None = None
    ranks_list: list[np.ndarray] = []
    midpoint_list: list[np.ndarray] = []
    participation_list: list[np.ndarray] = []
    lag_list: list[np.ndarray] = []
    epoch_list: list[np.ndarray] = []
    split_list: list[np.ndarray] = []
    block_index_list: list[np.ndarray] = []
    source_event_list: list[np.ndarray] = []
    input_records: list[dict[str, Any]] = []
    split_code = {"train": 0, "validation": 1, "test": 2}
    for block_index, (row, path) in enumerate(zip(rows, expected_paths)):
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["schema_version"].item()) != "topic5_ecog_rank_cache_v0.1":
                raise ValueError(f"wrong rank cache schema: {path}")
            names = [str(value) for value in cache["channel_names"].tolist()]
            if names_ref is None:
                names_ref = names
            elif names != names_ref:
                raise ValueError(f"channel order mismatch: {path}")
            ranks = np.asarray(cache["ranks"], dtype=np.int16)
            participation = np.asarray(cache["participation"], dtype=np.uint8)
            if not np.array_equal(ranks < 0, participation == 0):
                raise ValueError(f"phantom or missing rank mask mismatch: {path}")
            ranks_list.append(ranks)
            midpoint_list.append(np.asarray(cache["midpoint_ranks"], dtype=np.int16))
            participation_list.append(participation)
            lag_list.append(np.asarray(cache["lag_sec"], dtype=np.float32))
            epoch_list.append(np.asarray(cache["event_epoch"], dtype=np.float64))
            split_name = str(cache["split"].item())
            split_list.append(np.full(ranks.shape[0], split_code[split_name], dtype=np.int8))
            block_index_list.append(np.full(ranks.shape[0], block_index, dtype=np.int16))
            source_event_list.append(np.asarray(cache["source_event_index"], dtype=np.int32))
            input_records.append({
                "block_stem": row["block_stem"],
                "split": split_name,
                "n_events": int(ranks.shape[0]),
                "path": str(path),
                "sha256": sha256_file(path),
            })

    assert names_ref is not None
    ranks = np.concatenate(ranks_list, axis=0)
    midpoint_ranks = np.concatenate(midpoint_list, axis=0)
    participation = np.concatenate(participation_list, axis=0)
    lag_sec = np.concatenate(lag_list, axis=0)
    event_epoch = np.concatenate(epoch_list, axis=0)
    split = np.concatenate(split_list, axis=0)
    block_index = np.concatenate(block_index_list, axis=0)
    source_event_index = np.concatenate(source_event_list, axis=0)

    # Events whose participating contacts all fall inside one 5-ms tie set
    # contain no next-rank decision. Keep them in per-block provenance but not
    # in the RNN event table.
    transition_eligible = np.max(ranks, axis=1) >= 1
    dropped_one_rank_by_split = {
        name: int(np.sum((split == code) & ~transition_eligible))
        for name, code in split_code.items()
    }
    ranks = ranks[transition_eligible]
    midpoint_ranks = midpoint_ranks[transition_eligible]
    participation = participation[transition_eligible]
    lag_sec = lag_sec[transition_eligible]
    event_epoch = event_epoch[transition_eligible]
    split = split[transition_eligible]
    block_index = block_index[transition_eligible]
    source_event_index = source_event_index[transition_eligible]

    for event_index, row in enumerate(ranks):
        observed = np.unique(row[row >= 0])
        if observed.size < 2 or not np.array_equal(observed, np.arange(observed.size)):
            raise ValueError(f"non-contiguous rank sets at event {event_index}")
    tied_sets = 0
    total_sets = 0
    for row in ranks:
        values, counts = np.unique(row[row >= 0], return_counts=True)
        total_sets += len(values)
        tied_sets += int(np.sum(counts > 1))

    out_dir = args.cache_root / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.npz"
    temporary = out_dir / "events.tmp.npz"
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("topic5_ecog_events_v0.1"),
        subject=np.asarray(args.subject),
        channel_names=np.asarray(names_ref),
        ranks=ranks,
        midpoint_ranks=midpoint_ranks,
        participation=participation,
        lag_sec=lag_sec,
        split=split,
        event_epoch=event_epoch,
        block_index=block_index,
        source_event_index=source_event_index,
    )
    temporary.replace(events_path)

    null_audits: list[dict[str, Any]] = []
    null_split = split.copy()
    null_split[split == 2] = -1
    mode = np.zeros(len(ranks), dtype=np.int8)
    for null_index, seed in enumerate(NULL_SEEDS):
        mapping, audit = suffix_mapping(ranks, null_split, mode, seed)
        null_ranks = apply_suffix_mapping(ranks, mapping, seed=seed)
        null_ranks[split == 2] = ranks[split == 2]
        if not np.array_equal(null_ranks[split == 2], ranks[split == 2]):
            raise RuntimeError("suffix null changed held-out test")
        null_path = out_dir / f"events_suffix_null_seed{null_index}.npz"
        temp_null = out_dir / f"events_suffix_null_seed{null_index}.tmp.npz"
        np.savez_compressed(
            temp_null,
            schema_version=np.asarray("topic5_ecog_suffix_null_v0.1"),
            subject=np.asarray(args.subject),
            channel_names=np.asarray(names_ref),
            ranks=null_ranks.astype(np.int16),
            split=split,
            suffix_donor_index=mapping,
            seed=np.asarray(seed),
        )
        temp_null.replace(null_path)
        null_audits.append({
            "null_index": null_index,
            "seed": seed,
            "path": str(null_path),
            "sha256": sha256_file(null_path),
            "heldout_test_unchanged": True,
            **audit,
        })

    counts = {name: int(np.sum(split == code)) for name, code in split_code.items()}
    payload = {
        "schema": "topic5_ecog_events_provenance_v0.1",
        "subject": args.subject,
        "channel_names": names_ref,
        "n_contacts": len(names_ref),
        "n_events": int(len(ranks)),
        "split_counts": counts,
        "n_participations": int(participation.sum()),
        "participant_count_median": float(np.median(participation.sum(axis=1))),
        "rank_set_count_median": float(np.median(np.max(ranks, axis=1) + 1)),
        "dropped_one_rank_set_events_by_split": dropped_one_rank_by_split,
        "tied_rank_set_fraction": float(tied_sets / max(total_sets, 1)),
        "nonparticipant_rank_contract_pass": bool(np.array_equal(ranks < 0, participation == 0)),
        "events_path": str(events_path),
        "events_sha256": sha256_file(events_path),
        "input_records": input_records,
        "suffix_nulls": null_audits,
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "subject", "n_contacts", "n_events", "split_counts",
        "participant_count_median", "rank_set_count_median",
        "tied_rank_set_fraction", "events_sha256",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

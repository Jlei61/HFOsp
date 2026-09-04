#!/usr/bin/env python3
"""Verify that a shorter and longer run share an exact seeded training prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json
from src.topic5_group_event_state.v034_spatial_state.contracts import SEED_CONTRACT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-card", type=Path, required=True)
    parser.add_argument("--long-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    short = json.loads(args.short_card.read_text(encoding="utf-8"))
    long = json.loads(args.long_card.read_text(encoding="utf-8"))
    for label, card in (("short", short), ("long", long)):
        if card.get("seed_contract") != SEED_CONTRACT:
            raise ValueError(f"{label} card does not use {SEED_CONTRACT}")
    short_contract = short["contract"]
    long_contract = long["contract"]
    if short_contract["subject"] != long_contract["subject"]:
        raise ValueError("subjects differ")
    for field in ("arch", "optimizer"):
        if short_contract[field] != long_contract[field]:
            raise ValueError(f"{field} differs")
    short_train = dict(short_contract["train"])
    long_train = dict(long_contract["train"])
    short_steps = int(short_train.pop("max_steps"))
    long_steps = int(long_train.pop("max_steps"))
    if short_train != long_train or short_steps >= long_steps:
        raise ValueError("training contracts are not a shorter/longer pair")
    a = {int(row["step"]): row for row in short["history"]}
    b = {int(row["step"]): row for row in long["history"]}
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("cards have no common evaluated step")
    exact = all(a[step] == b[step] for step in common)
    payload = {
        "format": "group_event_state_v0_3_4_seed_prefix_audit_v1",
        "status": "PASS" if exact and short["initial_state_hash"] == long["initial_state_hash"] else "FAIL",
        "seed_contract": SEED_CONTRACT,
        "short_card": str(args.short_card),
        "long_card": str(args.long_card),
        "same_initial_state_hash": short["initial_state_hash"] == long["initial_state_hash"],
        "common_steps": common,
        "all_common_history_rows_exact": exact,
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

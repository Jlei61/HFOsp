#!/usr/bin/env python3
"""Audit why formal v2.2 uses the frozen 200-epoch count."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def main() -> None:
    lock_path = BASE / "development/DEVELOPMENT_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("status") != "pass"
        or lock.get("selected_objective") != "next_plus_rollout_h3"
        or int(lock.get("H_train")) != 3
    ):
        raise SystemExit("development lock is not the selected H3 contract")
    optimizer = lock["optimizer"]
    maximum = int(optimizer["max_epochs"])
    patience = int(optimizer["patience"])
    subjects = list(map(str, lock["development_subjects"]))
    seeds = list(map(int, optimizer["seeds"]))
    rows = []
    for subject in subjects:
        for seed in seeds:
            path = (
                BASE
                / "development/runs"
                / subject
                / "next_plus_rollout_h3"
                / f"seed_{seed}"
                / "metrics.json"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            for variant in ("full", "local_isotropic"):
                model = record["models"][variant]
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "variant": variant,
                        "best_epoch": int(model["best_epoch"]),
                        "epochs_completed": int(model["epochs_completed"]),
                        "early_stopped": bool(model["early_stopped"]),
                        "metrics_sha256": sha256(path),
                    }
                )
    expected = len(subjects) * len(seeds) * 2
    all_hit_maximum = bool(
        len(rows) == expected
        and all(row["best_epoch"] == maximum - 1 for row in rows)
        and all(row["epochs_completed"] == maximum for row in rows)
        and not any(row["early_stopped"] for row in rows)
    )
    formal_resolved = list(
        (BASE / "formal/claim2_runs").glob("*/seed_*/resolved_config.json")
    )
    formal_epoch_values = sorted(
        {
            int(json.loads(path.read_text(encoding="utf-8"))["epochs"])
            for path in formal_resolved
        }
    )
    pass_contract = bool(
        all_hit_maximum and formal_epoch_values == [maximum]
    )
    payload = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "PASS" if pass_contract else "FAIL",
        "selected_objective": lock["selected_objective"],
        "development_models_expected": expected,
        "development_models_audited": len(rows),
        "maximum_epochs": maximum,
        "patience": patience,
        "all_selected_development_models_best_at_final_epoch": all_hit_maximum,
        "formal_policy": (
            "fixed 200 epochs because every selected-objective development "
            "full/control run improved through epoch 199; heldout20 is never "
            "used for formal early stopping"
        ),
        "formal_resolved_configs_audited": len(formal_resolved),
        "formal_epoch_values": formal_epoch_values,
        "development_rows": rows,
        "development_lock_sha256": sha256(lock_path),
        "target_values_read": False,
    }
    atomic_json(BASE / "formal/TRAINER_EPOCH_AUDIT.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not pass_contract:
        raise SystemExit("formal epoch policy does not match development evidence")


if __name__ == "__main__":
    main()

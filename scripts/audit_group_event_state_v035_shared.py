#!/usr/bin/env python3
"""Machine audit for the v0.3.5 shared S_N/S_G development run.

This audit deliberately separates hard implementation failures from scientific
caveats such as a nearly constant learned trajectory or an INNER-budget-edge
checkpoint.  The latter must be reported, but they do not make a file corrupt.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data/hfosp_group_event_state_v0_3_5_shared")
SUBJECTS = ("epilepsiae_253", "epilepsiae_1096", "epilepsiae_1125")
SEEDS = (20260903, 20260904, 20260905)
FAMILIES = ("S_N", "S_G")
HORIZONS = (7200.0, 21600.0, 28800.0)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    hard_failures: list[str] = []
    caveats: list[str] = []
    cells: list[dict] = []

    queue_path = ROOT / "supervisor/queue_status.json"
    post_path = ROOT / "supervisor/post_status.json"
    for label, path in (("primary", queue_path), ("post", post_path)):
        if not path.exists():
            hard_failures.append(f"missing {label} supervisor status")
            continue
        status = load(path)
        if status.get("status") != "COMPLETE":
            hard_failures.append(f"{label} supervisor is {status.get('status')}")
        if status.get("development_targets_read") is not False:
            hard_failures.append(f"{label} development flag is not false")
        if status.get("sealed_partition_opened") is not False:
            hard_failures.append(f"{label} sealed flag is not false")
        failed = [key for key, value in status.get("units", {}).items()
                  if "FAILED" in str(value)]
        if failed:
            hard_failures.append(f"{label} failed units: {failed}")

    seed_hashes: dict[tuple[str, str], list[str]] = defaultdict(list)
    expected_cells = {(subject, family, seed)
                      for subject in SUBJECTS for family in FAMILIES for seed in SEEDS}
    seen_cells: set[tuple[str, str, int]] = set()
    for subject, family, seed in sorted(expected_cells):
        matches = list((ROOT / "shared_producer" / subject / family).glob(
            f"*_state_seed{seed}/card.json"))
        if len(matches) != 1:
            hard_failures.append(
                f"expected one producer card for {subject}/{family}/{seed}, got {len(matches)}")
            continue
        card_path = matches[0]
        card = load(card_path)
        seen_cells.add((subject, family, seed))
        for flag in ("selection_targets_read", "development_targets_read",
                     "sealed_partition_opened", "seizure_outcomes_read"):
            if card.get(flag) is not False:
                hard_failures.append(f"{card_path}: {flag} is not false")
        if card.get("family") != family or card.get("subject") != subject \
                or int(card.get("seed", -1)) != seed:
            hard_failures.append(f"{card_path}: cell identity mismatch")
        if tuple(float(x) for x in card.get("shared_horizons_seconds", [])) != HORIZONS:
            hard_failures.append(f"{card_path}: shared horizons mismatch")
        if "q is excluded" not in card.get("state_update_inputs", ""):
            hard_failures.append(f"{card_path}: recurrent update does not attest q exclusion")
        if "excludes blocks crossing" not in card.get("future_window_seizure_policy", ""):
            hard_failures.append(f"{card_path}: seizure-crossing policy missing")
        if card.get("selected_epoch") == 0:
            caveats.append(f"{subject}/{family}/seed{seed}: selected epoch 0")
        history = card.get("history", [])
        if history and card.get("selected_epoch") == history[-1].get("epoch"):
            caveats.append(f"{subject}/{family}/seed{seed}: selected last trained epoch")

        trajectory = Path(card.get("state_trajectory", ""))
        checkpoint = Path(card.get("checkpoint", ""))
        if not trajectory.exists() or not checkpoint.exists():
            hard_failures.append(f"{card_path}: checkpoint or trajectory missing")
            continue
        try:
            with np.load(trajectory, allow_pickle=False) as z:
                required = {"event_time", "event_segment", "phase", "state_pre", "state_post",
                            "q_context", "fixed_taus_seconds", "state_mean",
                            "producer_family", "shared_horizons_seconds"}
                missing = required.difference(z.files)
                if missing:
                    hard_failures.append(f"{trajectory}: missing arrays {sorted(missing)}")
                    continue
                event_time = np.asarray(z["event_time"])
                pre = np.asarray(z["state_pre"])
                post = np.asarray(z["state_post"])
                if pre.shape != post.shape or pre.shape[0] != event_time.size:
                    hard_failures.append(f"{trajectory}: state/event shape mismatch")
                if not np.all(np.diff(event_time) >= 0):
                    hard_failures.append(f"{trajectory}: event times are not ordered")
                if not np.isfinite(pre).all() or not np.isfinite(post).all():
                    hard_failures.append(f"{trajectory}: non-finite state")
                state_sd = float(np.std(pre, axis=0).mean()) if pre.size else 0.0
                update_rms = float(np.sqrt(np.mean((post - pre) ** 2))) if pre.size else 0.0
                if state_sd < 1e-6:
                    caveats.append(f"{subject}/{family}/seed{seed}: trajectory is effectively constant")
                if update_rms < 1e-7:
                    caveats.append(f"{subject}/{family}/seed{seed}: event updates are effectively zero")
                cells.append({
                    "subject": subject, "family": family, "seed": seed,
                    "selected_epoch": card.get("selected_epoch"),
                    "n_events": int(event_time.size), "state_dim": int(pre.shape[1]),
                    "mean_state_sd": state_sd, "event_update_rms": update_rms,
                    "trajectory_sha256": sha256(trajectory),
                })
                seed_hashes[(subject, family)].append(sha256(trajectory))
        except Exception as exc:
            hard_failures.append(f"{trajectory}: unreadable trajectory: {exc}")

        adapter = Path(card_path).parents[0]
        # The producer's actual adapter is stored in its checkpoint, but the
        # matching shared adapter card is cheaper and safer to inspect here.
        decoder_seed = SEEDS.index(seed)
        adapter_card = ROOT / "shared_stepwise_adapter" / subject / \
            f"decoder_seed{decoder_seed}_state_seed{seed}" / "card.json"
        if not adapter_card.exists():
            hard_failures.append(f"missing shared adapter card {adapter_card}")
        else:
            adapter_payload = load(adapter_card)
            if adapter_payload.get("config", {}).get("use_rate_trajectory_phases") is not True:
                hard_failures.append(f"{adapter_card}: not trained on shared rate phases")
            if adapter_payload.get("development_targets_read") is not False \
                    or adapter_payload.get("sealed_partition_opened") is not False:
                hard_failures.append(f"{adapter_card}: protected partition flag")

    for key, hashes in seed_hashes.items():
        if len(hashes) == len(SEEDS) and len(set(hashes)) == 1:
            hard_failures.append(f"{key}: all seed trajectories are byte-identical")

    evaluator_paths = list((ROOT / "frozen_evaluator").glob("*/*/*/card.json"))
    evaluator_paths += list((ROOT / "cross_evaluator").glob("*/*/*/card.json"))
    for path in evaluator_paths:
        card = load(path)
        if card.get("producer_frozen") is not True:
            hard_failures.append(f"{path}: producer is not declared frozen")
        if "invariant to duplicate anchor rows" not in card.get("ridge_contract", ""):
            hard_failures.append(f"{path}: scale-invariant ridge contract missing")
        if "excludes blocks crossing" not in card.get("future_window_seizure_policy", ""):
            hard_failures.append(f"{path}: seizure-crossing policy missing")
        if card.get("development_targets_read") is not False \
                or card.get("sealed_partition_opened") is not False:
            hard_failures.append(f"{path}: protected partition flag")
        for horizon_key, horizon in card.get("horizons", {}).items():
            for field in (
                "n_independent_fit_windows", "n_independent_inner_windows",
                "n_independent_selection_windows", "independence_contract",
            ):
                if field not in horizon:
                    hard_failures.append(f"{path}: horizon {horizon_key} lacks {field}")
            if horizon.get("status") == "ESTIMATED" and \
                    int(horizon.get("n_independent_selection_windows", 0)) < 3:
                caveats.append(
                    f"{path}: horizon {horizon_key} has fewer than 3 independent selection windows"
                )
            for endpoint_name, endpoint in horizon.get("endpoints", {}).items():
                contrasts = endpoint.get("contrasts", {})
                if endpoint.get("status") == "ESTIMATED" and \
                        "n_independent_block_shift_windows" not in contrasts:
                    hard_failures.append(
                        f"{path}: {horizon_key}/{endpoint_name} lacks independent shift support"
                    )
        if card.get("target_family") == "S_N":
            for horizon in card.get("horizons", {}).values():
                endpoint = horizon.get("endpoints", {}).get("burden", {})
                if endpoint.get("status") == "ESTIMATED" and \
                        endpoint.get("count_likelihood", {}).get("family") != "negative_binomial":
                    hard_failures.append(f"{path}: burden evaluator is not negative-binomial")

    expected_counts = {
        "producer": 18,
        "evaluator": 18,
        "same_prefix": 9,
        "h2b_binding": 18,
        "cross_evaluator": 36,
    }
    observed_counts = {
        "producer": len(list((ROOT / "shared_producer").glob("*/*/*/card.json"))),
        "evaluator": len(list((ROOT / "frozen_evaluator").glob("*/*/*/card.json"))),
        "same_prefix": len(list((ROOT / "same_prefix").glob("*/*/card.json"))),
        "h2b_binding": len(list((ROOT / "shared_h2b").glob("*/*/*/shared_state_binding.json"))),
        "cross_evaluator": len(list((ROOT / "cross_evaluator").glob("*/*/*/card.json"))),
    }
    for name, expected in expected_counts.items():
        if observed_counts[name] != expected:
            hard_failures.append(
                f"{name}: expected {expected}, observed {observed_counts[name]}")

    payload = {
        "format": "group_event_state_v0_3_5_shared_machine_audit_v1",
        "status": "PASS" if not hard_failures else "FAIL",
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "n_expected_cells": len(expected_cells),
        "n_seen_cells": len(seen_cells),
        "cells": cells,
        "hard_failures": hard_failures,
        "scientific_caveats": caveats,
        "scientific_scope": "three-patient development pilot; completion is not a positive-state claim",
        "development_targets_read": False,
        "sealed_partition_opened": False,
    }
    out = ROOT / "final_reports/shared_state_machine_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(json.dumps({"status": payload["status"], "observed_counts": observed_counts,
                      "hard_failures": hard_failures, "n_caveats": len(caveats)}, indent=2))
    raise SystemExit(0 if not hard_failures else 1)


if __name__ == "__main__":
    main()

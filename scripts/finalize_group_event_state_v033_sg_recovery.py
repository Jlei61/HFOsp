#!/usr/bin/env python3
"""Finalize the bounded synthetic-only S_G recovery diagnostic."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v032_eval.contract import atomic_json  # noqa: E402
from src.topic5_group_event_state.v033_evaluator.grammar_recovery import (  # noqa: E402
    RECIPES,
    select_full_input_recipe,
)


TUNING_REPLICATE = 0
VALIDATION_REPLICATES = (1, 2, 3)
FULL_RECIPE = "t0_lr3e3_constant"
ISOLATION_RECIPE = "t1_nested_h16_w2_marks_only"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _card_row(card: dict) -> dict:
    return {
        "recipe": card["recipe"]["name"], "kind": card["kind"],
        "estimator_seed": card["estimator_seed"], "generator_seed": card["generator_seed"],
        "noise_seed": card["noise_seed"], "h_only_inner_nll": card["h_only_inner_nll"],
        "selected_inner_nll": card["selected_inner_nll"], "selected_step": card["selected_step"],
        "gain_level2": card["gain_level2"], "ci_lower": card["ci_lower"],
        "ci_upper": card["ci_upper"], "detected": card["detected"],
        "truth_alignment_score": card["truth_alignment"]["score"],
        "resources": card["resources"], "inputs": card["inputs"],
    }


def _validation_summary(rows: list[dict]) -> dict:
    return {
        "n_seeds": len(rows), "n_detected": sum(bool(r["detected"]) for r in rows),
        "n_positive_gain": sum(float(r["gain_level2"]) > 0 for r in rows),
        "median_gain": float(statistics.median(float(r["gain_level2"]) for r in rows)),
        "per_seed": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_3/training_lab/sg_synthetic_recovery/cards"))
    ap.add_argument("--output", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_3/training_lab/sg_synthetic_recovery/reports/final_report.json"))
    ap.add_argument("--frozen-oracle-report", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_3/agent_a/oracle_level_0_2.json"))
    args = ap.parse_args()

    tuning_cards, card_hashes = [], {}
    for name in RECIPES:
        path = args.cards / f"D3_rep{TUNING_REPLICATE:03d}_{name}.json"
        card = json.loads(path.read_text())
        tuning_cards.append(card)
        card_hashes[str(path)] = _sha256(path)
    selection = select_full_input_recipe(tuning_cards)
    if selection["selected_recipe"] != FULL_RECIPE:
        raise RuntimeError(f"locked full-input validation recipe drifted: {selection}")

    validation: dict[str, dict] = {}
    for recipe_name in (FULL_RECIPE, ISOLATION_RECIPE):
        for kind in ("D3", "D0"):
            rows = []
            for rep in VALIDATION_REPLICATES:
                path = args.cards / f"{kind}_rep{rep:03d}_{recipe_name}.json"
                card = json.loads(path.read_text())
                if int(card["estimator_seed"]) != rep:
                    raise RuntimeError(f"seed mismatch in {path}")
                rows.append(_card_row(card))
                card_hashes[str(path)] = _sha256(path)
            validation[f"{recipe_name}:{kind}"] = _validation_summary(rows)

    # The already-frozen oracle ladder is sufficient to locate the loss after
    # Level 1.  Do not spend a second tuning pass recomputing those controls.
    frozen_oracle = json.loads(args.frozen_oracle_report.read_text())
    d3 = next(r for r in frozen_oracle["replicates"] if r["spec"]["kind"] == "D3")
    grammar = d3["views"]["grammar"]
    oracle_controls = {
        "source": str(args.frozen_oracle_report), "sha256": _sha256(args.frozen_oracle_report),
        "source_commit": frozen_oracle["source_commit"],
        "levels": [{k: level[k] for k in ("level", "gain", "ci_lower", "ci_upper", "detected")}
                   for level in grammar["levels"] if level["level"] in (0, 1)],
    }

    full_d3 = validation[f"{FULL_RECIPE}:D3"]
    marks_d3 = validation[f"{ISOLATION_RECIPE}:D3"]
    d0_fp = {
        FULL_RECIPE: validation[f"{FULL_RECIPE}:D0"]["n_detected"],
        ISOLATION_RECIPE: validation[f"{ISOLATION_RECIPE}:D0"]["n_detected"],
    }
    report = {
        "format": "group_event_state_v0_3_3_sg_level2_recovery_final",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "synthetic D0/D3 grammar targets on a frozen real-time scaffold only",
        "human_targets_used": False, "development_human_targets_read": False,
        "seizure_outcomes_used": False, "sealed_partition_opened": False,
        "h2a_h2b_h3_outputs_modified": False,
        "predeclared_grid": {name: vars(cfg) for name, cfg in RECIPES.items()},
        "tuning": [_card_row(c) for c in tuning_cards], "selection": selection,
        "validation": validation, "frozen_oracle_level0_level1_control": oracle_controls,
        "d0_false_positive_detected_seeds": d0_fp,
        "decision": {
            "full_input_level2_recovered": full_d3["n_detected"] >= 2,
            "marks_only_level2_recovered_robustly": marks_d3["n_detected"] >= 2,
            "stop_further_search": True,
            "failure_location": "encoder_objective_mismatch_under_frozen_scaffold_nuisance",
            "plain_language": (
                "真状态直接给读出头或固定时间库时能找回；把可见 mark 和 scaffold nuisance 一起交给 encoder 后，"
                "调学习率、训练预算和容量仍在 3/3 独立种子失败。只给 mark 的单个 tuning seed 曾通过，但独立复核仅 1/3，"
                "所以不能把它收口成恢复成功。当前失败应定在 encoder/目标在 nuisance 下的错配，而不是继续盲扫超参数。"
            ),
        },
        "card_sha256": card_hashes,
        "max_resources": {
            "peak_cuda_allocated_mib": max(r["resources"]["peak_cuda_allocated_mib"]
                                           for v in validation.values() for r in v["per_seed"]),
            "peak_rss_mib": max(r["resources"]["peak_rss_mib"]
                                for v in validation.values() for r in v["per_seed"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()

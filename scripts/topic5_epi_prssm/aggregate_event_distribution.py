#!/usr/bin/env python3
"""Aggregate Goal 2 into the H2a evidence card.

The question is whether the *state* changes the event distribution, so every
adapter is read against its own frozen-state control: an adapter that improves
prediction with a state that never moves has shown capacity, not state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
)
from src.topic5_epi_prssm.stats import aggregate_seeds, holm, paired_effect, stratify  # noqa: E402

OUT = OUTPUT_ROOT / "event_distribution"
ENDPOINTS = ["event_nll", "order_nll", "selection_nll", "stop_nll", "participation_nll"]
PRIMARY = ["order_nll", "stop_nll", "event_nll"]
ADAPTERS = ["initial_state", "node_film", "edge_gate"]
SOURCES = ["g0", "g2", "g3"]


def load_runs(cohort: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))
            if json.loads(p.read_text()).get("cohort") == cohort]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()
    runs = [r for r in load_runs(args.cohort) if r.get("evaluation")]
    if not runs:
        raise SystemExit("no completed event-distribution runs")
    dataset = {}
    for run in runs:
        dataset.update(run.get("dataset", {}))

    filtered: dict[str, dict[str, dict[str, float]]] = {}
    for endpoint in ENDPOINTS:
        per_arm: dict[str, list[dict[str, float]]] = {}
        for run in runs:
            per_arm.setdefault(run["arm"], []).append(
                {s: v[endpoint] for s, v in run["evaluation"]["filtered"].items()})
        filtered[endpoint] = {arm: aggregate_seeds(v) for arm, v in per_arm.items()}

    ladder_rows, effect_rows, family = [], [], {}
    for endpoint in ENDPOINTS:
        by_arm = filtered[endpoint]
        for arm, values in by_arm.items():
            ladder_rows.append({"endpoint": endpoint, "arm": arm,
                                "n_patients": len(values),
                                "cohort_median": float(np.median(list(values.values())))})
        for adapter in ADAPTERS:
            for source in SOURCES:
                arm, control = f"{adapter}_{source}", f"{adapter}_frozen"
                if arm not in by_arm or control not in by_arm:
                    continue
                effect = paired_effect(by_arm[arm], by_arm[control],
                                       label=f"{endpoint}::{arm}-vs-{control}")
                row = {"endpoint": endpoint, "adapter": adapter, "state_source": source,
                       "contrast": "state vs frozen-state (capacity-matched)",
                       "n_patients": effect.n_patients, "median_delta": effect.median_delta,
                       "ci_low": effect.ci_low, "ci_high": effect.ci_high,
                       "n_favourable": effect.n_favourable,
                       "sign_test_p": effect.sign_test_p, "wilcoxon_p": effect.wilcoxon_p}
                row.update({f"stratum_{k}": json.dumps(v) for k, v in stratify(effect, dataset).items()})
                effect_rows.append(row)
                if endpoint in PRIMARY:
                    family[f"{endpoint}::{arm}"] = effect.sign_test_p
            if f"{adapter}_frozen" in by_arm and "no_state" in by_arm:
                effect = paired_effect(by_arm[f"{adapter}_frozen"], by_arm["no_state"],
                                       label=f"{endpoint}::{adapter}_frozen-vs-no_state")
                effect_rows.append({"endpoint": endpoint, "adapter": adapter,
                                    "state_source": "frozen", "contrast": "adapter capacity alone",
                                    "n_patients": effect.n_patients,
                                    "median_delta": effect.median_delta,
                                    "ci_low": effect.ci_low, "ci_high": effect.ci_high,
                                    "n_favourable": effect.n_favourable,
                                    "sign_test_p": effect.sign_test_p,
                                    "wilcoxon_p": effect.wilcoxon_p})
    atomic_write_csv(OUT / "adapter_ladder.csv", pd.DataFrame(ladder_rows))
    atomic_write_csv(OUT / "full_event_effects.csv", pd.DataFrame(effect_rows))

    # ---- state swap -------------------------------------------------------
    swap_rows, swap_effects = [], {}
    for endpoint in ("order_nll", "event_nll", "selection_nll"):
        for swap in ("swap_matched", "swap_random"):
            per_arm_correct: dict[str, list[dict[str, float]]] = {}
            per_arm_swapped: dict[str, list[dict[str, float]]] = {}
            for run in runs:
                table = run["evaluation"].get("state_swap") or {}
                per_arm_correct.setdefault(run["arm"], []).append(
                    {s: v[f"correct__{endpoint}"] for s, v in table.items()})
                per_arm_swapped.setdefault(run["arm"], []).append(
                    {s: v[f"{swap}__{endpoint}"] for s, v in table.items()})
            for arm in per_arm_correct:
                correct = aggregate_seeds(per_arm_correct[arm])
                swapped = aggregate_seeds(per_arm_swapped[arm])
                if not correct or not swapped:
                    continue
                effect = paired_effect(correct, swapped, label=f"{endpoint}::{arm}::{swap}")
                swap_effects[f"{endpoint}::{arm}::{swap}"] = effect
                for subject in sorted(set(correct) & set(swapped)):
                    swap_rows.append({"endpoint": endpoint, "arm": arm, "swap": swap,
                                      "subject": subject, "dataset": dataset.get(subject),
                                      "correct": correct[subject], "swapped": swapped[subject],
                                      "delta": correct[subject] - swapped[subject]})
    atomic_write_csv(OUT / "state_swap_effects.csv", pd.DataFrame(swap_rows))

    # ---- ambiguous prefix -------------------------------------------------
    prefix_rows = []
    for run in runs:
        for subject, values in (run["evaluation"].get("ambiguous_prefix") or {}).items():
            for depth in (1, 2, 3):
                if f"depth{depth}_n_events" not in values:
                    continue
                prefix_rows.append({
                    "arm": run["arm"], "seed": run["seed"], "subject": subject,
                    "dataset": dataset.get(subject), "prefix_depth": depth,
                    "n_events": values[f"depth{depth}_n_events"],
                    "suffix_nll_correct": values[f"depth{depth}_suffix_nll_correct"],
                    "suffix_nll_swapped": values[f"depth{depth}_suffix_nll_swapped"],
                    # upstream field is swapped_nll - correct_nll, so positive already
                    # means the correct state helped; do not negate it again
                    "state_gain": values[f"depth{depth}_suffix_state_gain"],
                    "adapter": run["arm"].rsplit("_", 1)[0] if "_" in run["arm"] else run["arm"],
                    "state_source": run["arm"].rsplit("_", 1)[-1]})
    prefix_frame = pd.DataFrame(prefix_rows)
    atomic_write_csv(OUT / "ambiguous_prefix_effects.csv", prefix_frame)

    eligible = sorted({s for run in runs for s in run["evaluation"].get("targeted_eligible", [])})
    not_eligible = sorted({s for run in runs
                           for s in run["evaluation"].get("not_eligible_for_targeted_analysis", [])})
    card = {
        "contract": "topic5_epi_prssm_v0_1_h2a_evidence_card",
        "hypothesis": "H2a: does the pre-event slow state change the distribution of a single "
                      "complete event?",
        "status": "EXPLORATORY_DEVELOPMENT",
        "primary_endpoints": PRIMARY,
        "cohort_wide_state_versus_frozen_state": _best_rows(pd.DataFrame(effect_rows)),
        "state_swap": {k: v.as_dict() for k, v in swap_effects.items()},
        "ambiguous_prefix": _prefix_summary(prefix_frame),
        "targeted_eligible_patients": eligible,
        "not_eligible_for_targeted_analysis": not_eligible,
        "holm_corrected_primary_family": holm(family),
        "denominators": {"n_runs": len(runs), "arms": sorted({r["arm"] for r in runs}),
                         "n_patients": len(dataset),
                         "n_epilepsiae": sum(1 for v in dataset.values() if v == "epilepsiae"),
                         "n_yuquan": sum(1 for v in dataset.values() if v == "yuquan")},
        "claim_boundary": [
            "an adapter that improves prediction with a frozen state has shown capacity, not state",
            "insufficient ambiguous-prefix support is recorded as not eligible, never as a negative",
            "development-partition result; no untouched-test claim is made here",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    atomic_write_json(OUT / "H2A_EVIDENCE_CARD.json", card)
    print(json.dumps({"eligible": len(eligible), "not_eligible": len(not_eligible),
                      "top": card["cohort_wide_state_versus_frozen_state"]}, indent=2)[:1500])


def _best_rows(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    subset = frame[(frame.contrast == "state vs frozen-state (capacity-matched)")
                   & frame.endpoint.isin(PRIMARY)]
    subset = subset.sort_values("median_delta")
    return subset.head(12).to_dict("records")


#: arms whose state cannot move, so their suffix gain is zero by construction.
#: Pooling them with the moving-state arms forces every median to zero.
STATIC_STATE_ARMS = ("no_state",)
STATIC_STATE_SUFFIX = ("_frozen",)


def _is_moving_state_arm(arm: str) -> bool:
    if arm in STATIC_STATE_ARMS or arm.endswith(STATIC_STATE_SUFFIX):
        return False
    return True


def _prefix_summary(frame: pd.DataFrame) -> dict:
    """Suffix-branching gain, reported per arm.

    Every arm whose state is frozen contributes an exact zero, so a pooled median
    over all arms is zero by construction and says nothing about the hypothesis.
    The per-arm breakdown is the readable result; the frozen arms are kept and
    reported as the built-in negative control.
    """
    if frame.empty:
        return {"status": "no_eligible_family"}
    out: dict = {"by_arm": {}, "negative_control_arms": {}}
    for (arm, depth), group in frame.groupby(["arm", "prefix_depth"]):
        per_patient = group.groupby("subject").state_gain.median().to_dict()
        effect = paired_effect(per_patient, {s: 0.0 for s in per_patient},
                               label=f"{arm}::prefix_depth_{depth}",
                               lower_is_better=False).as_dict()
        bucket = "by_arm" if _is_moving_state_arm(arm) else "negative_control_arms"
        out[bucket].setdefault(arm, {})[int(depth)] = effect
    moving = [a for a in out["by_arm"]]
    out["reading"] = (
        "positive gain means the suffix is better predicted with the state that actually "
        "held at that event than with a magnitude-matched state from another moment. "
        f"{len(moving)} moving-state arms and {len(out['negative_control_arms'])} frozen "
        "arms are reported separately; the frozen arms are exactly zero by construction "
        "and are the built-in negative control, not evidence."
    )
    return out


if __name__ == "__main__":
    main()

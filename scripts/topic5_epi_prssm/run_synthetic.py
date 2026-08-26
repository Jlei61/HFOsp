#!/usr/bin/env python3
"""Just-in-time synthetic recovery tests, one truth at a time.

Each truth is compared only against the models adjacent to it, never against a
full truth-by-model Cartesian product.  A truth that turns out to be
unidentifiable limits the interpretation of that comparison; it does not block
any other Goal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_json, code_revision,
    is_complete, package_hash, sha256_obj, torch,
)

import numpy as np  # noqa: E402

from src.topic5_epi_prssm.contracts import LeakageGuard  # noqa: E402
from src.topic5_epi_prssm.evaluate import collect_states, evaluate, state_swap_effects  # noqa: E402
from src.topic5_epi_prssm.model import EpiPRSSM  # noqa: E402
from src.topic5_epi_prssm.synthetic_truths import TRUTH_PURPOSE, TRUTHS, generate  # noqa: E402
from src.topic5_epi_prssm.trainer import TrainConfig, make_split_batches, train_model  # noqa: E402

GOAL = "synthetic"
OUT = OUTPUT_ROOT / "synthetic"

_G0 = dict(generator_level="G0", resource_arm="R0", adapter="node_film")
_G1 = dict(generator_level="G1", resource_arm="R0", adapter="node_film")
_G2 = dict(generator_level="G2", resource_arm="R0", adapter="node_film")
_STATIC = dict(generator_level="G0", resource_arm="R0", adapter="no_state")
_R1 = dict(generator_level="G3", resource_arm="R1", adapter="node_film", tau_r_seconds=1200.0,
           freeze_tau_r=True)
_R2 = dict(generator_level="G3", resource_arm="R2", adapter="node_film", tau_r_seconds=1200.0,
           freeze_tau_r=True)
_R3 = dict(generator_level="G3", resource_arm="R3", adapter="node_film", tau_r_seconds=1200.0,
           freeze_tau_r=True, tau_x_seconds=900.0, exposure_kind="clock")
_R3_EVENTS = dict(_R3, tau_x_seconds=20.0, exposure_kind="event_count")
#: the node-resolved frozen state is the control the human ladder is read against:
#: every adapter parameter present, the state present per node, but time-constant.
_G0_FROZEN_NODE = dict(_G0, node_resolved_frozen_state=True)
_G2_FROZEN_NODE = dict(_G2, node_resolved_frozen_state=True)

COMPARISONS: dict[str, dict[str, dict]] = {
    "no_state": {"static": _STATIC, "g0": _G0, "g2": _G2,
                 "g0_frozen_node": _G0_FROZEN_NODE, "g2_frozen_node": _G2_FROZEN_NODE},
    "leaky_state": {"static": _STATIC, "g0": _G0, "g2": _G2,
                    "g0_frozen_node": _G0_FROZEN_NODE, "g2_frozen_node": _G2_FROZEN_NODE},
    "graph_recurrent_state": {"static": _STATIC, "g0": _G0, "g1": _G1, "g2": _G2,
                              "g0_frozen_node": _G0_FROZEN_NODE,
                              "g2_frozen_node": _G2_FROZEN_NODE},

    "observer_overpowering": {"g0": _G0, "g2": _G2},
    "state_conditioned_suffix": {"no_state": dict(_G2, adapter="no_state"), "node_film": _G2},
    "no_state_false_adapter": {"no_state": dict(_G2, adapter="no_state"), "node_film": _G2,
                               "node_film_frozen": dict(_G2, freeze_state=True)},
    "latent_preictal_drift": {"g0": _G0, "g2": _G2},
    "event_rate_only_drift": {"g0": _G0, "g2": _G2,
                              "g0_frozen_node": _G0_FROZEN_NODE,
                              "g2_frozen_node": _G2_FROZEN_NODE},
    "t1_autonomous_resource": {"r0": _G2, "r1": _R1},
    "r2_impulse": {"r1": _R1, "r2": _R2},
    "r3_integrated_exposure": {"r1": _R1, "r2": _R2, "r3_clock": _R3},
    "hidden_common_cause": {"r1": _R1, "r3_clock": _R3},
    "event_count_only": {"r3_clock": _R3, "r3_events": _R3_EVENTS},
    "switching_state": {"g0": _G0, "g2": _G2, "r1": _R1},
    "observer_resource_substitution": {"r1": _R1,
                                       "r1_flexible": dict(_R1, flexible_resource_correction=True)},
    "resource_direct_excitability": {"r1": _R1, "r2": _R2, "r3_clock": _R3},
}


def run_truth(truth: str, seed: int, *, n_patients: int, n_events: int, epochs: int,
              overwrite: bool = False) -> Path:
    key = JobKey(goal=GOAL, family=truth, arm="recovery", seed=seed, split="synthetic",
                 cohort=f"n{n_patients}x{n_events}",
                 config_hash=sha256_obj({"epochs": epochs, "arms": sorted(COMPARISONS[truth])})[:16],
                 input_hash=sha256_obj({"truth": truth, "seed": seed})[:16],
                 code_revision=package_hash()[:16])
    target = OUT / f"{key.job_id}.json"
    if target.exists() and is_complete(key) and not overwrite:
        print(f"SKIPPED_EXISTING {key.job_id}")
        return target

    with JobRunner(key) as record:
        cohort = generate(truth, seed=seed, n_patients=n_patients, n_events=n_events)
        config = TrainConfig(max_epochs=epochs, min_epochs=2, patience=2, tbptt_length=64,
                             max_train_events_per_patient=None, seed=seed)
        arms: dict[str, dict] = {}
        for name, spec in COMPARISONS[truth].items():
            torch.manual_seed(seed)
            model = EpiPRSSM(feature_dim=cohort.patients[0].node_features.shape[-1], **spec)
            report = train_model(model, cohort.patients, config,
                                 guard=LeakageGuard(stage=f"synthetic:{truth}:{name}"))
            entry = {"spec": spec, "train_report": report.as_dict()}
            if report.status == "COMPLETE":
                train_batch, val_batch = make_split_batches(cohort.patients, config)
                expected = torch.stack([p.load[p.split == 0].mean() for p in cohort.patients])
                result = evaluate(model, train_batch, val_batch, expected_load=expected,
                                  seed=seed, with_reset=False, with_shuffle=False)
                entry["filtered"] = result.filtered
                entry["open_loop_event_nll"] = result.open_loop
                entry["mean_validation"] = float(np.mean(
                    [v["event_nll"] + v["participation_nll"] for v in result.filtered.values()]))
                entry["mean_open_loop_h20"] = float(np.mean(
                    [v[20] for v in result.open_loop.values() if np.isfinite(v[20])]))
                if truth in ("state_conditioned_suffix", "no_state_false_adapter"):
                    states = collect_states(model, train_batch, val_batch)
                    entry["state_swap"] = state_swap_effects(model, val_batch, states, seed=seed)
                if model.resource.active:
                    entry["recovered_tau_r_seconds"] = float(model.resource.tau_r().item())
            arms[name] = entry
        payload = {
            "contract": "topic5_epi_prssm_v0_1_synthetic_recovery",
            "job_id": key.job_id, "truth": truth, "seed": seed,
            "purpose": TRUTH_PURPOSE[truth], "n_patients": n_patients, "n_events": n_events,
            "generator_metadata": {k: v for k, v in cohort.metadata.items() if k != "per_patient"},
            "per_patient": cohort.metadata["per_patient"], "arms": arms,
            "verdict": _verdict(truth, arms),
            "code_revision": code_revision(), "package_hash": package_hash(),
        }
        atomic_write_json(target, payload)
        record.outputs = {"run_json": str(target)}
        record.metrics = payload["verdict"]
    print(f"COMPLETE {key.job_id}: {payload['verdict']}")
    return target


def _verdict(truth: str, arms: dict) -> dict:
    scores = {k: v.get("mean_validation") for k, v in arms.items() if v.get("mean_validation") is not None}
    open_loop = {k: v.get("mean_open_loop_h20") for k, v in arms.items()
                 if v.get("mean_open_loop_h20") is not None}
    if not scores:
        return {"status": "NO_COMPLETED_ARM"}
    winner = min(scores, key=scores.get)
    ordering = sorted(scores, key=scores.get)
    spread = max(scores.values()) - min(scores.values())
    #: The human ladder never reads raw validation loss: it reads each moving-state
    #: arm against a capacity-matched frozen arm.  Scoring the synthetic runs any
    #: other way tests a different question than the one being calibrated.
    capacity_matched = {}
    for moving, frozen in (("g0", "g0_frozen_node"), ("g2", "g2_frozen_node"),
                           ("g1", "g0_frozen_node"), ("node_film", "node_film_frozen")):
        if moving in scores and frozen in scores:
            capacity_matched[f"{moving}-vs-{frozen}"] = scores[moving] - scores[frozen]
    return {
        "status": "IDENTIFIABLE" if spread > 5e-3 else "UNIDENTIFIABLE_AT_THIS_SAMPLE",
        "validation_by_arm": scores, "open_loop_h20_by_arm": open_loop,
        "winner": winner, "ranking": ordering, "spread": spread,
        "capacity_matched_gain": capacity_matched,
        "capacity_matched_note": "negative means the moving state beat its capacity-matched "
                                 "frozen control; on a no-state truth this must be ~0, and "
                                 "whatever it is sets the null band the human ladder gain "
                                 "has to clear",
        "expected": TRUTH_PURPOSE[truth]["expect"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, choices=sorted(TRUTHS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-patients", type=int, default=6)
    parser.add_argument("--n-events", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_truth(args.truth, args.seed, n_patients=args.n_patients, n_events=args.n_events,
              epochs=args.epochs, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

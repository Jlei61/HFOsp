#!/usr/bin/env python3
"""Goal 2 / H2a -- does the slow state change the event distribution?

Each adapter is run against matched state sources so that a gain can be told
apart from the adapter's own capacity: ``frozen`` keeps every adapter parameter
but never moves the graph state.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_json, code_revision,
    dataset_of, expected_load_vector, input_hash, is_complete, load_tensors,
    package_hash, resolve_cohort, sha256_obj, torch,
)

from src.topic5_epi_prssm.contracts import LeakageGuard  # noqa: E402
from src.topic5_epi_prssm.evaluate import (  # noqa: E402
    ambiguous_prefix_effects, collect_states, evaluate, state_swap_effects,
)
from src.topic5_epi_prssm.model import EpiPRSSM  # noqa: E402
from src.topic5_epi_prssm.prefix_families import cohort_families  # noqa: E402
from src.topic5_epi_prssm.trainer import TrainConfig, make_split_batches, train_model  # noqa: E402

GOAL = "goal2_event_distribution"
OUT = OUTPUT_ROOT / "event_distribution"

_STATE_SOURCES = {
    "g0": dict(generator_level="G0", resource_arm="R0"),
    "g2": dict(generator_level="G2", resource_arm="R0"),
    "g3": dict(generator_level="G3", resource_arm="R1"),
    # capacity control: node-resolved but time-constant state.  Freezing the
    # state at zero would give every contact the same adapter shift and would
    # therefore under-match a moving state's capacity -- the no-state synthetic
    # showed that most of a "state gain" over the bare fixed repertoire is
    # exactly this per-contact capacity.
    "frozen": dict(generator_level="G2", resource_arm="R0", freeze_state=True,
                   node_resolved_frozen_state=True),
}
ARMS: dict[str, dict] = {"no_state": dict(adapter="no_state", generator_level="G2",
                                          resource_arm="R0")}
# ``edge_gate`` also carries the node FiLM terms, so it cannot show that edge
# coupling contributes on its own; ``edge_gate_only`` is the separable arm.
for _adapter in ("initial_state", "node_film", "edge_gate", "edge_gate_only"):
    for _source, _spec in _STATE_SOURCES.items():
        ARMS[f"{_adapter}_{_source}"] = dict(adapter=_adapter, **_spec)


def run_one(arm: str, seed: int, cohort: str, *, config: TrainConfig,
            overwrite: bool = False) -> Path:
    subjects = resolve_cohort(cohort)
    patients = load_tensors(subjects)
    spec = dict(ARMS[arm])
    family = f"{spec['generator_level']}/{spec['resource_arm']}/{spec['adapter']}"
    config_payload = {"arm": arm, "spec": spec, "train": vars(config)}
    key = JobKey(goal=GOAL, family=family, arm=arm, seed=seed, split="development",
                 cohort=cohort, config_hash=sha256_obj(config_payload)[:16],
                 input_hash=input_hash(patients)[:16], code_revision=package_hash()[:16])
    target = OUT / "runs" / f"{key.job_id}.json"
    if target.exists() and is_complete(key) and not overwrite:
        print(f"SKIPPED_EXISTING {key.job_id}")
        return target

    with JobRunner(key) as record:
        torch.manual_seed(seed)
        model = EpiPRSSM(feature_dim=patients[0].node_features.shape[-1], **spec)
        report = train_model(model, patients, config,
                             guard=LeakageGuard(stage=f"{GOAL}:{arm}"),
                             progress=lambda m: print(f"[{arm} s{seed}] {m}", flush=True))
        train_batch, val_batch = make_split_batches(patients, config)
        payload = {
            "contract": "topic5_epi_prssm_v0_1_event_distribution_run",
            "job_id": key.job_id, "goal": GOAL, "arm": arm, "family": family,
            "adapter": spec["adapter"], "state_source": arm.split("_")[-1] if "_" in arm else "none",
            "seed": seed, "cohort": cohort, "n_patients": len(patients),
            "subjects": list(subjects), "dataset": dataset_of(patients), "spec": spec,
            "train_config": vars(config), "train_report": report.as_dict(),
            "code_revision": code_revision(), "package_hash": package_hash(),
            "config_hash": key.config_hash, "input_hash": key.input_hash,
        }
        if report.status != "COMPLETE":
            record.state = report.status
            record.failure_reason = report.failure_reason
            payload["evaluation"] = None
            atomic_write_json(target, payload)
            return target

        result = evaluate(model, train_batch, val_batch,
                          expected_load=expected_load_vector(patients), seed=seed,
                          with_reset=False, with_shuffle=False, with_open_loop=False)
        states = collect_states(model, train_batch, val_batch)
        swaps = state_swap_effects(model, val_batch, states, seed=seed)
        families = cohort_families(subjects)
        prefixes = ambiguous_prefix_effects(model, val_batch, states, families, seed=seed)
        payload["evaluation"] = {
            "filtered": result.filtered,
            "state_swap": swaps,
            "ambiguous_prefix": prefixes,
            "targeted_eligible": sorted(prefixes),
            "not_eligible_for_targeted_analysis": sorted(set(subjects) - set(prefixes)),
        }
        atomic_write_json(target, payload)
        checkpoint = OUT / "checkpoints" / f"{key.job_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "spec": spec, "job_id": key.job_id,
                    "arm": arm, "seed": seed, "cohort": cohort,
                    "feature_dim": patients[0].node_features.shape[-1]}, checkpoint)
        record.outputs = {"run_json": str(target), "checkpoint": str(checkpoint)}
        record.metrics = {"best_validation": report.best_validation,
                          "n_targeted_eligible": len(prefixes)}
    print(f"COMPLETE {key.job_id}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--tbptt", type=int, default=64)
    parser.add_argument("--max-train-events", type=int, default=30000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = TrainConfig(max_epochs=args.max_epochs, tbptt_length=args.tbptt,
                         max_train_events_per_patient=args.max_train_events, seed=args.seed)
    run_one(args.arm, args.seed, args.cohort, config=config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

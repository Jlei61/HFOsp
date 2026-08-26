#!/usr/bin/env python3
"""Goal 1 / H1 -- does a slow state exist, and which generator layer does the data support?

One run trains one shared-parameter model over the whole cohort and evaluates it
patient by patient.  No arm is dropped for being negative: an arm that fails to
beat G0, collapses its resource or dies numerically is written out with that
reason attached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import (  # noqa: E402  (path bootstrap happens inside _common)
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_json, code_revision,
    dataset_of, expected_load_vector, input_hash, is_complete, load_tensors,
    package_hash, resolve_cohort, sha256_obj, torch,
)

from src.topic5_epi_prssm.contracts import LeakageGuard  # noqa: E402
from src.topic5_epi_prssm.evaluate import evaluate  # noqa: E402
from src.topic5_epi_prssm.model import EpiPRSSM  # noqa: E402
from src.topic5_epi_prssm.trainer import TrainConfig, make_split_batches, train_model  # noqa: E402

GOAL = "goal1_generator"
OUT = OUTPUT_ROOT / "generator_ladder"

#: The ladder.  Every state-bearing arm shares one adapter so the only thing that
#: changes along the ladder is the generator itself.
ARMS: dict[str, dict] = {
    "static": dict(generator_level="G0", resource_arm="R0", adapter="no_state"),
    # capacity-matched reference: every adapter parameter the state arms have,
    # but the graph state never moves.  The just-in-time no-state synthetic showed
    # that comparing a state arm against `static` alone credits adapter capacity
    # to the state, so this arm is the primary reference for the first rung.
    "frozen_state": dict(generator_level="G2", resource_arm="R0", adapter="node_film",
                         freeze_state=True),
    # strictest capacity control: node-resolved but time-constant state
    "frozen_state_node": dict(generator_level="G2", resource_arm="R0", adapter="node_film",
                              freeze_state=True, node_resolved_frozen_state=True),
    "event_index_ewma": dict(generator_level="G0", resource_arm="R0", adapter="node_film",
                             time_mode="event_index"),
    # observable-timing baseline: the readout is conditioned on causal multi-scale
    # rate, interval, coverage and time of day, and on nothing about which contacts
    # took part.  Topic 2 already establishes that these drift slowly, so a latent
    # state has to beat this arm, not merely beat a static repertoire.
    "nuisance_timing_baseline": dict(generator_level="G0", resource_arm="R0",
                                     adapter="node_film", state_from_nuisance=True),
    "ct_ewma_g0": dict(generator_level="G0", resource_arm="R0", adapter="node_film"),
    "unconstrained_gru": dict(generator_level="G0", resource_arm="R0", adapter="node_film",
                              unconstrained_gru=True),
    "g1_graph_clds": dict(generator_level="G1", resource_arm="R0", adapter="node_film"),
    "g2_graph_gru_ode": dict(generator_level="G2", resource_arm="R0", adapter="node_film"),
    "g3_resource": dict(generator_level="G3", resource_arm="R1", adapter="node_film"),
    # spec section 4.2 puts the bounded resource anchor on G2 *or on the best stable
    # recurrent family*.  On this cohort the best stable family is the linear
    # graph-CLDS, so the faithful resource arm is built on it.
    "g3_resource_on_g1": dict(generator_level="G1", resource_arm="R1", adapter="node_film"),
    "g3_flexible_resource_control": dict(generator_level="G3", resource_arm="R1",
                                         adapter="node_film",
                                         flexible_resource_correction=True),
    # declared sensitivity: a compressed slow state, not a spectral eigen-decomposition
    "g2_compressed_state": dict(generator_level="G2", resource_arm="R0", adapter="node_film",
                                state_dim=2),
}


def run_one(arm: str, seed: int, cohort: str, *, config: TrainConfig,
            overwrite: bool = False, arm_label: str | None = None) -> Path:
    patients = load_tensors(resolve_cohort(cohort))
    spec = dict(ARMS[arm])
    # a sensitivity variant records itself under its own arm label so it is never
    # pooled with the arm it varies
    recorded_arm = arm_label or arm
    family = f"{spec['generator_level']}/{spec['resource_arm']}/{spec['adapter']}"
    config_payload = {"arm": arm, "spec": spec, "train": vars(config),
                      "frozen": {k: FROZEN[k] for k in
                                 ("state_dim_H", "observer_dim", "open_loop_horizons",
                                  "state_reset_horizons", "split_fractions")}}
    key = JobKey(goal=GOAL, family=family, arm=recorded_arm, seed=seed, split="development",
                 cohort=cohort, config_hash=sha256_obj(config_payload)[:16],
                 input_hash=input_hash(patients)[:16], code_revision=package_hash()[:16])
    target = OUT / "runs" / f"{key.job_id}.json"
    if target.exists() and is_complete(key) and not overwrite:
        print(f"SKIPPED_EXISTING {key.job_id}")
        return target

    with JobRunner(key) as record:
        torch.manual_seed(seed)
        model = EpiPRSSM(feature_dim=patients[0].node_features.shape[-1], **spec)
        guard = LeakageGuard(stage=f"{GOAL}:{arm}")
        report = train_model(model, patients, config, guard=guard,
                             progress=lambda m: print(f"[{arm} s{seed}] {m}", flush=True))
        train_batch, val_batch = make_split_batches(patients, config)
        payload = {
            "contract": "topic5_epi_prssm_v0_1_generator_run",
            "job_id": key.job_id, "goal": GOAL, "arm": recorded_arm,
            "base_arm": arm, "family": family,
            "seed": seed, "cohort": cohort, "n_patients": len(patients),
            "subjects": [p.subject for p in patients], "dataset": dataset_of(patients),
            "spec": spec, "train_config": vars(config), "train_report": report.as_dict(),
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
                          expected_load=expected_load_vector(patients), seed=seed)
        payload["evaluation"] = {
            "filtered": result.filtered,
            "open_loop_event_nll": result.open_loop,
            "open_loop_order_nll": result.open_loop_order,
            "state_reset": result.state_reset,
            "delta_t_shuffle": result.delta_t_shuffle,
            "n_open_loop_anchors": result.n_open_loop_anchors,
        }
        payload["state_diagnostics"] = _state_diagnostics(model, result)
        atomic_write_json(target, payload)
        checkpoint = OUT / "checkpoints" / f"{key.job_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "spec": spec, "job_id": key.job_id,
                    "arm": recorded_arm, "seed": seed, "cohort": cohort,
                    "feature_dim": patients[0].node_features.shape[-1]}, checkpoint)
        record.outputs = {"run_json": str(target), "checkpoint": str(checkpoint)}
        record.metrics = {
            "best_validation": report.best_validation,
            "epochs_run": report.epochs_run,
            "correction_energy": report.peak_correction_energy,
            "resource_floor_fraction": report.resource_floor_fraction,
        }
    print(f"COMPLETE {key.job_id}")
    return target


def _state_diagnostics(model: EpiPRSSM, result) -> dict:
    import numpy as np
    tau = model.generator.time_constants().detach().numpy()
    diagnostics = {
        "generator_time_constants_seconds": tau.tolist(),
        "generator_tau_min_seconds": float(tau.min()),
        "generator_tau_max_seconds": float(tau.max()),
        "generator_used": not model.unconstrained_gru,
        "generator_tau_median_seconds": float(np.median(tau)),
        "stability_margin": float(model.generator.stability_margin()),
        "resource_tau_seconds": float(model.resource.tau_r().item()) if model.resource.active else None,
        "resource_arm": model.resource_arm,
        "observer_gain": float(torch.nn.functional.softplus(model.observer.log_gain).item()),
    }
    resources = [v["resource"] for v in result.per_event.values() if "resource" in v]
    if resources:
        flat = np.concatenate(resources)
        diagnostics["resource_quantiles"] = {
            q: float(np.quantile(flat, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)}
        diagnostics["resource_boundary_occupancy"] = float((flat <= 1.01e-3).mean())
        diagnostics["resource_collapsed"] = bool(np.quantile(flat, 0.99) < 0.05)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--tbptt", type=int, default=64)
    parser.add_argument("--max-train-events", type=int, default=30000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--arm-label", default=None)
    parser.add_argument("--order-weight", type=float, default=0.0)
    args = parser.parse_args()
    config = TrainConfig(max_epochs=args.max_epochs, tbptt_length=args.tbptt,
                         max_train_events_per_patient=args.max_train_events, seed=args.seed,
                         order_weight=args.order_weight)
    run_one(args.arm, args.seed, args.cohort, config=config, overwrite=args.overwrite,
            arm_label=args.arm_label)


if __name__ == "__main__":
    main()

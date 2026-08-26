#!/usr/bin/env python3
"""Goal 4 / H3 -- does IED exposure update the slow state?

The ladder is nested and matched: every arm shares graph, observer, decoder,
adapter, state dimension, split, seed and optimisation budget, and differs only
in which resource path is switched on.  ``tau_r`` is frozen on T1/R1 before any
exposure arm runs, so ``tau_r`` and ``tau_x`` are never fitted against each other.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_json, code_revision,
    dataset_of, expected_load_vector, input_hash, is_complete, load_tensors,
    package_hash, resolve_cohort, sha256_obj, torch,
)

from src.topic5_epi_prssm.contracts import LeakageGuard  # noqa: E402
from src.topic5_epi_prssm.evaluate import (  # noqa: E402
    collect_states, evaluate, state_swap_effects,
)
from src.topic5_epi_prssm.model import EpiPRSSM  # noqa: E402
from src.topic5_epi_prssm.trainer import TrainConfig, make_split_batches, train_model  # noqa: E402

GOAL = "goal4_exposure"
OUT = OUTPUT_ROOT / "exposure_mechanism"
TAU_FREEZE = OUTPUT_ROOT / "manifests/RESOURCE_TAU_FREEZE.json"

_BASE = dict(generator_level="G3", adapter="node_film")

#: Clock kernels swept for the exposure timescale curve.  The frozen primary trio
#: (5 min / 30 min / 2 h) is unchanged; Topic 2 shows the event rate itself drifts
#: on a multi-hour scale and that Epilepsiae autocorrelation is still positive at
#: 8 h, so 4 h and 8 h are added as declared exploratory sensitivity.  The whole
#: curve is reported; no single tau is selected afterwards.
EXPOSURE_CLOCK_SECONDS = tuple(FROZEN["exposure_tau_sensitivity_seconds"]) + (14400.0, 28800.0)
EXPOSURE_PRIMARY_SECONDS = tuple(FROZEN["exposure_tau_primary_seconds"])
EXPOSURE_EXPLORATORY_SECONDS = (14400.0, 28800.0)

ARMS: dict[str, dict] = {
    # --- stage 4a: no resource, then autonomous resource on a declared tau grid
    "t1_r0": dict(generator_level="G2", resource_arm="R0", adapter="node_film"),
    **{f"t1_r1_tau{int(t)}": dict(resource_arm="R1", tau_r_seconds=float(t),
                                  freeze_tau_r=True, **_BASE)
       for t in FROZEN["resource_tau_grid_seconds"]},
    "t1_r1_free_tau": dict(resource_arm="R1", freeze_tau_r=False, **_BASE),
    # --- stage 4b: impulse and integrated exposure, tau_r frozen from stage 4a
    "t2_r2": dict(resource_arm="R2", freeze_tau_r=True, **_BASE),
    **{f"t2_r3_clock{int(t)}": dict(resource_arm="R3", freeze_tau_r=True,
                                    tau_x_seconds=float(t), exposure_kind="clock", **_BASE)
       for t in EXPOSURE_CLOCK_SECONDS},
    # review follow-up: the first pass only reached 5 min to 8 h and every winning
    # arm had a collapsed resource, so the timescale question was never actually
    # opened.  These reach 30 s at one end and two days at the other.
    **{f"t2_r3_clock{int(t)}": dict(resource_arm="R3", freeze_tau_r=True,
                                    tau_x_seconds=float(t), exposure_kind="clock", **_BASE)
       for t in (30.0, 60.0, 86400.0, 172800.0)},
    **{f"t2_r3_events{int(n)}": dict(resource_arm="R3", freeze_tau_r=True,
                                     tau_x_seconds=float(n), exposure_kind="event_count",
                                     **_BASE)
       for n in (2.0, 160.0, 320.0)},
    **{f"t2_r3_events{int(n)}": dict(resource_arm="R3", freeze_tau_r=True,
                                     tau_x_seconds=float(n), exposure_kind="event_count", **_BASE)
       for n in FROZEN["exposure_event_count_sensitivity"]},
}

STAGE_A = tuple(a for a in ARMS if a.startswith("t1_"))
STAGE_B = tuple(a for a in ARMS if a.startswith("t2_"))


def frozen_tau_r() -> float:
    if TAU_FREEZE.exists():
        return float(json.loads(TAU_FREEZE.read_text())["tau_r_seconds"])
    raise FileNotFoundError(
        f"{TAU_FREEZE} missing: stage 4b may not run before tau_r is frozen on T1/R1")


def run_one(arm: str, seed: int, cohort: str, *, config: TrainConfig,
            overwrite: bool = False) -> Path:
    patients = load_tensors(resolve_cohort(cohort))
    spec = dict(ARMS[arm])
    if arm in STAGE_B or (spec.get("freeze_tau_r") and "tau_r_seconds" not in spec):
        spec["tau_r_seconds"] = frozen_tau_r()
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
            "contract": "topic5_epi_prssm_v0_1_exposure_run",
            "job_id": key.job_id, "goal": GOAL, "arm": arm, "family": family,
            "resource_arm": spec["resource_arm"], "seed": seed, "cohort": cohort,
            "n_patients": len(patients), "subjects": [p.subject for p in patients],
            "dataset": dataset_of(patients), "spec": spec, "train_config": vars(config),
            "train_report": report.as_dict(), "code_revision": code_revision(),
            "package_hash": package_hash(), "config_hash": key.config_hash,
            "input_hash": key.input_hash,
        }
        if report.status != "COMPLETE":
            record.state = report.status
            record.failure_reason = report.failure_reason
            payload["evaluation"] = None
            atomic_write_json(target, payload)
            return target

        result = evaluate(model, train_batch, val_batch,
                          expected_load=expected_load_vector(patients), seed=seed,
                          with_reset=False, with_shuffle=False, with_open_loop=True)
        states = collect_states(model, train_batch, val_batch)
        swaps = state_swap_effects(model, val_batch, states, seed=seed)
        payload["evaluation"] = {
            "filtered": result.filtered,
            "open_loop_event_nll": result.open_loop,
            "open_loop_order_nll": result.open_loop_order,
            "state_swap": swaps,
        }
        payload["resource_diagnostics"] = _resource_diagnostics(model, states)
        atomic_write_json(target, payload)
        checkpoint = OUT / "checkpoints" / f"{key.job_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "spec": spec, "job_id": key.job_id,
                    "arm": arm, "seed": seed, "cohort": cohort,
                    "feature_dim": patients[0].node_features.shape[-1]}, checkpoint)
        record.outputs = {"run_json": str(target), "checkpoint": str(checkpoint)}
        record.metrics = {"best_validation": report.best_validation}
    print(f"COMPLETE {key.job_id}")
    return target


def _resource_diagnostics(model: EpiPRSSM, states) -> dict:
    import numpy as np
    out = {"resource_arm": model.resource_arm,
           "tau_r_seconds": float(model.resource.tau_r().item()) if model.resource.active else None,
           "tau_x_seconds": model.tau_x_seconds, "exposure_kind": model.exposure_kind}
    if model.resource.active:
        out["gamma_q"] = float(torch.nn.functional.softplus(model.resource.consumption).item())
    if hasattr(model.resource, "log_gamma_L"):
        out["gamma_L"] = float(torch.nn.functional.softplus(model.resource.log_gamma_L).item())
    if hasattr(model.resource, "log_gamma_x"):
        out["gamma_x"] = float(torch.nn.functional.softplus(model.resource.log_gamma_x).item())
    values = [d["resource"].cpu().numpy() for d in states.values()]
    if values:
        flat = np.concatenate(values)
        out["resource_quantiles"] = {str(q): float(np.quantile(flat, q))
                                     for q in (0.01, 0.25, 0.5, 0.75, 0.99)}
        # The resource falls from 1, so "pinned" has two ends and they mean opposite
        # things: pinned low is exhausted, pinned high is never consumed.  Measuring
        # only the low end reported 0.0 -- i.e. "not pinned" -- for a run whose
        # resource never moved off its ceiling, which is maximally degenerate.
        out["resource_floor_occupancy"] = float((flat <= 1.01e-3).mean())
        out["resource_ceiling_occupancy"] = float((flat >= 1.0 - 1.01e-3).mean())
        out["resource_boundary_occupancy"] = float(out["resource_floor_occupancy"]
                                                   + out["resource_ceiling_occupancy"])
        out["resource_collapsed"] = bool(np.quantile(flat, 0.99) < 0.05)
        out["resource_never_consumed"] = bool(np.quantile(flat, 0.01) > 0.99)
        out["resource_static"] = bool(np.std(flat) < 1e-4)
    return out


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

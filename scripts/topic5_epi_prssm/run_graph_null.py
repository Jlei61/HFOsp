#!/usr/bin/env python3
"""H1 specificity: does the relational message need *this patient's* topology?

"G1 beats G0" says a relational message helped.  It does not say the message
travelled along the propagation support built from this patient's own events.
Each arm here keeps the generator, the adapter, the parameter count and the edge
budget identical and changes only what the edges connect:

  patient_swapped           another patient's propagation support, same anatomy
  degree_preserving_rewire  same out-degree and the same weight multiset, random targets
  forward_only_shuffled     this patient's support with the contact labels permuted
  geometry_only             the fixed anatomical kernel, no learned support
  identity                  no relational message at all

If the real graph does not beat these, "flows along the patient's own network"
is not supported and the claim has to fall back to "a relational message helped".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_json, code_revision,
    dataset_of, expected_load_vector, input_hash, is_complete, load_tensors,
    package_hash, resolve_cohort, sha256_obj, torch,
)

from src.topic5_epi_prssm.contracts import LeakageGuard  # noqa: E402
from src.topic5_epi_prssm.evaluate import evaluate  # noqa: E402
from src.topic5_epi_prssm.model import EpiPRSSM  # noqa: E402
from src.topic5_epi_prssm.trainer import TrainConfig, make_split_batches, train_model  # noqa: E402

GOAL = "goal1_graph_null"
OUT = OUTPUT_ROOT / "graph_null"

#: the arm under test is the supported H1 layer; only its graph is replaced
BASE_SPEC = dict(generator_level="G1", resource_arm="R0", adapter="node_film")
NULLS = ("real", "patient_swapped", "degree_preserving_rewire", "forward_only_shuffled",
         "geometry_only", "identity")

FORWARD, REVERSE, GEOMETRY = 0, 1, 2


def _row_stochastic(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(out, 0.0)
    total = out.sum(axis=1, keepdims=True)
    return np.divide(out, total, out=np.zeros_like(out), where=total > 0)


def apply_null(patients, kind: str, seed: int):
    """Replace the relation stack in place-free fashion; anatomy stays the patient's own."""
    if kind == "real":
        return patients
    rng = np.random.default_rng(seed)
    donors = list(range(len(patients)))
    out = []
    for index, patient in enumerate(patients):
        adjacency = patient.adjacency.numpy().astype(np.float64).copy()
        n = adjacency.shape[-1]
        forward = adjacency[FORWARD]
        geometry = adjacency[GEOMETRY]

        if kind == "identity":
            eye = np.eye(n)
            new = np.stack([eye, eye, eye])
        elif kind == "geometry_only":
            new = np.stack([geometry, geometry, geometry])
        elif kind == "patient_swapped":
            # a size-matched donor keeps the edge budget comparable; fall back to
            # the next patient and crop, recording that the budget is only approximate
            sizes = [(abs(patients[d].adjacency.shape[-1] - n), d) for d in donors if d != index]
            donor = patients[min(sizes)[1]]
            d_adj = donor.adjacency.numpy().astype(np.float64)
            m = min(n, d_adj.shape[-1])
            fwd = np.zeros((n, n)); fwd[:m, :m] = d_adj[FORWARD][:m, :m]
            rev = np.zeros((n, n)); rev[:m, :m] = d_adj[REVERSE][:m, :m]
            new = np.stack([_row_stochastic(fwd), _row_stochastic(rev), geometry])
        elif kind == "forward_only_shuffled":
            perm = rng.permutation(n)
            fwd = forward[np.ix_(perm, perm)]
            new = np.stack([_row_stochastic(fwd), _row_stochastic(fwd.T), geometry])
        elif kind == "degree_preserving_rewire":
            rewired = np.zeros((n, n))
            for i in range(n):
                weights = forward[i][forward[i] > 0]
                if weights.size == 0:
                    continue
                choices = [j for j in range(n) if j != i]
                targets = rng.choice(choices, size=min(weights.size, len(choices)),
                                     replace=False)
                rewired[i, targets] = rng.permutation(weights)[:len(targets)]
            new = np.stack([_row_stochastic(rewired), _row_stochastic(rewired.T), geometry])
        else:
            raise ValueError(f"unknown null {kind!r}")

        if not np.isfinite(new).all():
            raise ValueError(f"{patient.subject}: null {kind} produced non-finite relations")
        import dataclasses
        out.append(dataclasses.replace(
            patient, adjacency=torch.as_tensor(new, dtype=torch.float32)))
    return out


def edge_budget(patients) -> dict:
    counts = [float((p.adjacency[FORWARD].numpy() > 0).sum()) for p in patients]
    return {"median_forward_edges": float(np.median(counts)),
            "total_forward_edges": float(np.sum(counts))}


#: which of the two graph paths the null is applied to.  The real graph feeds both
#: the slow generator's propagation and the within-event decoder's spatial prior, so
#: shuffling it wholesale cannot say which path needed it.
PATHS = ("both", "generator", "decoder")


def apply_to_path(real, shuffled, path: str):
    """Return patients with the null applied to the chosen graph path only."""
    import dataclasses
    if path == "both":
        return shuffled
    out = []
    for r, sh in zip(real, shuffled):
        if path == "generator":
            # generator propagates along `adjacency`; decoder reads `decoder_adjacency`
            out.append(dataclasses.replace(sh, decoder_adjacency=r.adjacency))
        else:
            out.append(dataclasses.replace(r, decoder_adjacency=sh.adjacency))
    return out


def run_one(kind: str, seed: int, cohort: str, *, config: TrainConfig,
            overwrite: bool = False, path: str = "both") -> Path:
    real = load_tensors(resolve_cohort(cohort))
    patients = apply_to_path(real, apply_null(real, kind, seed), path)
    spec = dict(BASE_SPEC)
    family = f"{spec['generator_level']}/{spec['resource_arm']}/{spec['adapter']}"
    config_payload = {"arm": kind, "spec": spec, "train": vars(config), "null": kind,
                      "path": path}
    arm_label = kind if path == "both" else f"{kind}@{path}"
    key = JobKey(goal=GOAL, family=family, arm=arm_label, seed=seed, split="development",
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
                             guard=LeakageGuard(stage=f"{GOAL}:{kind}"),
                             progress=lambda m: print(f"[{kind} s{seed}] {m}", flush=True))
        train_batch, val_batch = make_split_batches(patients, config)
        payload = {
            "contract": "topic5_epi_prssm_v0_1_graph_null_run",
            "job_id": key.job_id, "goal": GOAL, "arm": arm_label, "graph_null": kind,
            "graph_path": path,
            "family": family, "seed": seed, "cohort": cohort, "n_patients": len(patients),
            "subjects": [p.subject for p in patients], "dataset": dataset_of(patients),
            "spec": spec, "train_config": vars(config), "train_report": report.as_dict(),
            "edge_budget": edge_budget(patients),
            "code_revision": code_revision(), "package_hash": package_hash(),
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
        }
        atomic_write_json(target, payload)
        record.outputs = {"run_json": str(target)}
        record.metrics = {"best_validation": report.best_validation,
                          "epochs_run": report.epochs_run}
    print(f"COMPLETE {key.job_id}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--null", required=True, choices=NULLS)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--max-train-events", type=int, default=30000)
    parser.add_argument("--path", default="both", choices=PATHS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = TrainConfig(max_epochs=args.max_epochs,
                         max_train_events_per_patient=args.max_train_events)
    run_one(args.null, args.seed, args.cohort, config=config, overwrite=args.overwrite,
            path=args.path)


if __name__ == "__main__":
    main()

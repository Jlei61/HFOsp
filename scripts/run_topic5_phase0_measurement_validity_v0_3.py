#!/usr/bin/env python3
"""Phase 0 — hand the pipeline data whose answer is known and see if it comes back.

For every real SEEG patient this generates four synthetic versions of that patient,
keeping the geometry and the event count and replacing only who fires when.  Two
teachers carry no order information at all (one of them while still producing visibly
directional propagation), two carry a lot.  The same student is then fit on each in
two arms that differ *only* by the transition operator:

``FREE_ORDERED``  the ordered low-dimensional residual on a free basis
``FREE_BAG``      the same residual with the prefix state built from the cumulative
                  contact set instead — permutation invariant by construction

Both are residuals on top of the same frozen unordered baseline, fit with the same
rank selection and the same optimiser budget as the v0.2 matrix, because the point is
to test *that* instrument rather than a new one.

Nothing here touches the v0.2 result tree or the real-data conclusions.
"""
from __future__ import annotations

# A worker must not also fan out inside BLAS: these run many at a time on a shared
# machine and the default OpenMP thread count is the core count.
import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, _os.environ.get("TOPIC5_TORCH_THREADS", "1"))

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_phase0_measurement_validity_v0_3 import (  # noqa: E402
    TEACHER_KINDS,
    synthesise_patient,
)
from src.topic5_strict_history_data_v0_2 import (  # noqa: E402
    PRIMARY_PREFIX_LEN,
    build_sample_set,
    load_seeg_patient,
)
from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    MotifConfig,
    OrderedMotif,
    TrainConfig,
    UnorderedBaseline,
    checkpoint_objective,
    combine_logits,
    evaluate,
    fit,
    primary_field_kind,
    tensors_from_samples,
    training_loss,
    unordered_features,
)

FRAME_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"
RESULT_ROOT = ROOT / "results/topic5_phase0_measurement_validity_v0_3"
V02_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"

# ``ORDERLESS_BAG`` on a free basis is the permutation-invariant control the v0.2
# matrix never had, which is why v0.2 could not say whether its order null meant
# "no information in the order" or "the instrument cannot see order".
ARMS = {"FREE_ORDERED": "DIRECT_HORIZON_UPPER_BOUND", "FREE_BAG": "ORDERLESS_BAG"}
BASELINE_LEVEL = "U_FULL_SET"
BASELINE_RANK_GRID = (2, 4, 8)
BASELINE_SEED = 991
RANK = 4
SEEDS = (0, 1, 2)
TEACHER_SEED = 13


def fit_baseline(batches: dict, samples, contact_xy: torch.Tensor) -> dict:
    """The frozen unordered baseline both arms sit on top of, rank picked on split 1."""
    n_features = unordered_features(batches[0], BASELINE_LEVEL).shape[1]
    best = None
    for rank in BASELINE_RANK_GRID:
        if rank > max(1, min(samples.n_contacts, n_features)):
            continue
        module = UnorderedBaseline(
            level=BASELINE_LEVEL, n_contacts=samples.n_contacts, n_features=n_features,
            n_horizons=batches[0].n_horizons, max_cardinality=samples.max_cardinality,
            rank=rank)
        features = {split: unordered_features(batch, BASELINE_LEVEL)
                    for split, batch in batches.items()}

        def forward(piece, rows, _module=module, _features=features[0]):
            return training_loss(combine_logits({**_module(_features[rows])}, None),
                                 piece, "full_suffix")

        def objective(_module, _features=features[1], _batch=batches[1]):
            with torch.no_grad():
                logits = dict(_module(_features))
            return checkpoint_objective(evaluate(None, logits, _batch, contact_xy), None)

        history = fit(module, forward, batches[0], batches[1], objective,
                      TrainConfig(seed=BASELINE_SEED + rank))
        if best is None or history["best_valid_objective"] < best["objective"]:
            best = {"rank": rank, "module": module,
                    "objective": history["best_valid_objective"], "features": features}
    with torch.no_grad():
        logits = {split: dict(best["module"](best["features"][split])) for split in batches}
    return {"rank": best["rank"], "logits": logits,
            "parameters": int(sum(p.numel() for p in best["module"].parameters()))}


def train_arm(arm: str, samples, batches: dict, baseline: dict, contact_xy: torch.Tensor,
              seed: int) -> dict:
    family = ARMS[arm]
    field_kind = primary_field_kind(family)
    config = MotifConfig(
        structure="H1_FREE_LOW_RANK", family=family, rank=RANK,
        n_contacts=samples.n_contacts, n_horizons=batches[0].n_horizons,
        max_cardinality=samples.max_cardinality, f_form="FULL", free_basis=True)
    torch.manual_seed(seed + 1000)
    model = OrderedMotif(config, None)

    def forward(piece, rows, _logits=baseline["logits"][0]):
        merged = combine_logits({key: value[rows] for key, value in _logits.items()},
                                model(piece))
        return training_loss(merged, piece, field_kind)

    def objective(_module):
        return checkpoint_objective(
            evaluate(model, baseline["logits"][1], batches[1], contact_xy), family)

    history = fit(model, forward, batches[0], batches[1], objective, TrainConfig(seed=seed))
    with torch.no_grad():
        result = evaluate(model, baseline["logits"][2], batches[2], contact_xy)
    total = sum(value / 3.0 for value in result.per_horizon["total_nll"][:3]
                if value is not None and np.isfinite(value))
    return {
        "arm": arm, "family": family, "seed": int(seed),
        "objective": float(total + result.scalars[f"{field_kind}_balanced_bce"]),
        "suffix_balanced_bce": float(result.scalars[f"{field_kind}_balanced_bce"]),
        "total_nll_h1": float(result.per_horizon["total_nll"][0]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "gradient_updates": int(history["gradient_updates"]),
        "best_epoch": int(history["best_epoch"]),
    }


def process(job: tuple[str, str]) -> dict:
    patient_name, teacher = job
    torch.set_num_threads(int(_os.environ.get("TOPIC5_TORCH_THREADS", "1")))
    started = time.time()
    try:
        real = load_seeg_patient(FRAME_ROOT, patient_name)
        if teacher == "REAL_DATA":
            synthetic = real
            census = {"patient": patient_name, "teacher": teacher, "seed": -1,
                      "n_events": int(real.n_events),
                      "n_visible_contacts": int(real.n_contacts),
                      "order_information_nats": float("nan")}
        else:
            synthetic, census = synthesise_patient(real, teacher, seed=TEACHER_SEED)
        samples = build_sample_set(synthetic, prefix_len=PRIMARY_PREFIX_LEN)
        rows = {split: np.flatnonzero(samples.split == split) for split in (0, 1, 2)}
        if min(index.size for index in rows.values()) < 20:
            return {"patient": patient_name, "teacher": teacher,
                    "state": "too_few_events", "census": census}
        all_rows = np.arange(samples.n_samples)
        full = tensors_from_samples(samples, all_rows)
        batches = {split: full.index(torch.as_tensor(index)) for split, index in rows.items()}
        contact_xy = torch.as_tensor(
            np.asarray(synthetic.contacts_xy_mm, dtype=np.float32))

        baseline = fit_baseline(batches, samples, contact_xy)
        scores = [train_arm(arm, samples, batches, baseline, contact_xy, seed)
                  for arm in ARMS for seed in SEEDS]
        return {"patient": patient_name, "teacher": teacher, "state": "complete",
                "census": {**census, "baseline_rank": baseline["rank"],
                           "baseline_parameters": baseline["parameters"],
                           "n_development_test": int(rows[2].size)},
                "scores": scores, "wall_seconds": time.time() - started}
    except Exception:
        return {"patient": patient_name, "teacher": teacher, "state": "failed",
                "error": traceback.format_exc(limit=8), "wall_seconds": time.time() - started}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--patients", type=int, default=0, help="0 = every SEEG patient")
    parser.add_argument("--teachers", nargs="*", default=list(TEACHER_KINDS),
                        help="teacher ids, or REAL_DATA to run the untouched patient")
    parser.add_argument("--tag", default="", help="suffix for a smoke run's outputs")
    arguments = parser.parse_args()

    census = pd.read_csv(V02_ROOT / "INPUT_CENSUS.csv")
    patients = sorted(census[census["dataset"] == "SEEG"]["patient"])
    if arguments.patients:
        patients = patients[:arguments.patients]
    jobs = [(name, teacher) for name in patients for teacher in arguments.teachers]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"patients {len(patients)}  teachers {len(arguments.teachers)}  "
          f"cells {len(jobs)}  units {len(jobs) * len(ARMS) * len(SEEDS)}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for index, payload in enumerate(pool.map(process, jobs), start=1):
            results.append(payload)
            if index % 5 == 0 or index == len(jobs):
                done = sum(r["state"] == "complete" for r in results)
                print(f"  [{index}/{len(jobs)}] complete={done}", flush=True)

    rows, truth = [], []
    for payload in results:
        truth.append({**payload.get("census", {}), "patient": payload["patient"],
                      "teacher": payload["teacher"], "state": payload["state"]})
        for score in payload.get("scores", []):
            rows.append({"patient": payload["patient"], "teacher": payload["teacher"], **score})
    suffix = f"_{arguments.tag}" if arguments.tag else ""
    pd.DataFrame(rows).to_csv(RESULT_ROOT / f"PER_UNIT_SCORES{suffix}.csv", index=False)
    pd.DataFrame(truth).to_csv(RESULT_ROOT / f"GROUND_TRUTH_CENSUS{suffix}.csv", index=False)

    states: dict[str, int] = {}
    for payload in results:
        states[payload["state"]] = states.get(payload["state"], 0) + 1
    (RESULT_ROOT / f"RUN_STATUS{suffix}.json").write_text(json.dumps({
        "contract": "topic5_phase0_measurement_validity_v0_3_run",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "cell_states": states, "n_units": len(rows),
        "arms": ARMS, "rank": RANK, "seeds": list(SEEDS),
        "baseline_level": BASELINE_LEVEL, "teacher_seed": TEACHER_SEED,
        "failures": [{key: payload[key] for key in ("patient", "teacher", "error")}
                     for payload in results if payload["state"] == "failed"],
    }, indent=2) + "\n")
    print(f"cells: {states}  units: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Which propagation motif explains how long the event took to get there?

v0.1 compared the same four motifs on "who fires next" and found isotropic local
diffusion was already the best operator.  Its closeout named the reason it could not
settle the question: every model in that round used the within-event rank index as its
only notion of time, while the data carries a time proxy that no model ever saw, and
that proxy holds distance information the rank index does not.

So the motifs are unchanged and only the target moves.  Four things make the
comparison fair, each of which a first attempt got wrong:

* **Matched information.**  The motif reads its field at the contact that actually
  fired next, so it is told the destination.  A baseline knowing only the step index
  would concede a gain that comes from the destination's distance rather than from any
  dynamics — measured at 0.2 to 5.5 percent of the variance on its own, the same order
  as the motifs' whole effect.  The baselines therefore climb: step index, then step
  index plus the distance actually travelled, then plus that contact's habitual
  earliness.
* **A real warm-start chain.**  Each richer motif inherits the *fitted* simpler model
  and sets its own new parameter to zero, so it starts from the simpler model's
  solution rather than from a shared untrained state.  Inheritance is asserted to be
  numerically exact before training resumes.
* **Optimisation starts are not seeds.**  Every weight here initialises
  deterministically, so three "seeds" reproduced each other bit for bit until the axis
  angle was varied.  The three angles are optimisation starts: validation picks one,
  test scores it once, and the spread between starts is reported as basin sensitivity
  rather than as measurement noise.
* **One objective.**  The round's question is time, so the checkpoint is selected on
  validation time error alone.  The contact score is computed and reported, but it
  never decides anything.

The proxy is the within-event spectral-centroid position: not clinical recruitment
time, not axonal conduction delay, and never a conduction velocity.
"""
from __future__ import annotations

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

from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    MotifConfig,
    MotifRNN,
    StaticReadout,
    trainable_parameter_count,
)
from src.topic5_motif_time_targets_v0_3 import (  # noqa: E402
    TIME_BASELINES,
    FreeLowRankDrive,
    TimeHead,
    adjacent_distance_time_relation,
    build_event_tensors_with_time,
    direction_persistence,
    distance_time_relation,
    free_rank_for_budget,
    recruited_field,
    rollout_states,
    time_baseline_scores,
)

MOTIF_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"
RESULT_ROOT = ROOT / "results/topic5_motif_time_targets_v0_3"
FRAME = "GEOMETRY_ONLY_PCA2"

# Simple to rich.  Each entry names the v0.1 model and the parameter it introduces,
# which must be zero at inheritance for the chain to start from its parent's solution.
CHAIN = (
    ("M0_ISOTROPIC_DIFFUSION", "DM0_ISOTROPIC", ()),
    ("M1_AXIAL_CORRIDOR", "DM1_FREE_AXIS", ("theta", "eta_raw")),
    ("M2_DIRECTED_TRANSPORT", "DM2_LOCAL_DIRECTIONAL", ("beta",)),
    ("M3_AXIAL_FEEDFORWARD_TRANSIENT", "DM3_AXIS_FEEDFORWARD_TRANSIENT", ("gamma_raw",)),
)
# not an upper bound: rank-constrained, so it does not contain the full-rank
# structured kernels.  Renamed for any future run; the completed run's CSV
# still carries the old value and the aggregator records the correction.
FREE_ARM = "MFREE_LOW_RANK_ALTERNATIVE"

# Only the axis-dependent layer needs several starts: the isotropic operator has no
# axis for an angle to move, and the later layers inherit the chosen one.
THETA_INITS = (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0)
FREE_STARTS = (0, 1, 2)
MAX_EPOCHS = 120
PATIENCE = 12
LR = 0.05


def masked_contact_nll(logits: torch.Tensor, target: torch.Tensor,
                       available: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Cross entropy over the contacts still available — a diagnostic, never a selector."""
    masked = logits.masked_fill(~available, -1e9)
    picked = (torch.log_softmax(masked, dim=-1) * target).sum(dim=-1)
    keep = valid & (target.sum(dim=-1) > 0)
    if keep.sum() == 0:
        return logits.sum() * 0.0
    return -picked[keep].mean()


def build_operator(model_id: str, unit, theta: float, seed: int,
                   free_rank: int | None = None) -> MotifRNN:
    config = MotifConfig(
        model_id=model_id, n_contacts=len(unit.contact_names),
        n_nodes=unit.nodes_xy_mm.shape[0], observation_operator=unit.H,
        node_xy_mm=unit.nodes_xy_mm, local_mask=unit.local_mask,
        r_forward_mm=unit.r_local_mm, sigma_s_mm=float(unit.sigma_mm), seed=seed,
        theta_init=float(theta))
    if free_rank is None:
        return MotifRNN(config)
    return FreeLowRankDrive(config, free_rank, seed=seed)


def train_one(operator, static, head, tensors, observation, scaled, mask_time,
              index) -> dict:
    """Fit on the training split; select the checkpoint on validation time error only."""
    parameters = (list(operator.parameters()) + list(static.parameters())
                  + list(head.parameters()))
    optimiser = torch.optim.Adam(parameters, lr=LR)

    def field_for(rows):
        states, _ = rollout_states(operator, tensors["x"][rows], tensors["recruited"][rows],
                                   tensors["displacement"][rows])
        return states, recruited_field(states, tensors["target"][rows], observation)

    def time_error(rows):
        _, field = field_for(rows)
        return head.nll(field, scaled[rows], mask_time[rows])

    def snapshot() -> dict:
        return {"operator": {k: v.detach().clone() for k, v in operator.state_dict().items()},
                "static": {k: v.detach().clone() for k, v in static.state_dict().items()},
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()}}

    with torch.no_grad():
        best = float(time_error(index[1]))
    best_state, stall, epoch = snapshot(), 0, 0
    for epoch in range(MAX_EPOCHS):
        optimiser.zero_grad()
        time_error(index[0]).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimiser.step()
        operator.project_constraints()
        static.project_constraints()
        with torch.no_grad():
            score = float(time_error(index[1]))
        if score < best - 1e-9:
            best, stall, best_state = score, 0, snapshot()
        else:
            stall += 1
            if stall >= PATIENCE:
                break
    operator.load_state_dict(best_state["operator"])
    static.load_state_dict(best_state["static"])
    head.load_state_dict(best_state["head"])
    return {"validation_time_mse": best, "epochs": epoch + 1, "state": best_state}


def fitted_motif_parameters(operator) -> dict:
    """The parameters each layer introduces, after fitting.

    Needed to tell two very different situations apart when a richer layer ties its
    parent: the anisotropy strength is non-negative and starts exactly at its boundary,
    so "isotropic really is best here" and "the parameter never left zero" produce the
    same score.  Only the fitted value separates them.
    """
    out: dict[str, float | None] = {}
    for name in ("theta", "eta_raw", "beta", "gamma_raw"):
        value = getattr(operator, name, None)
        if value is None:
            out[f"fitted_{name}"] = None
        else:
            out[f"fitted_{name}"] = float(value.detach())
            out[f"{name}_is_trainable"] = isinstance(value, torch.nn.Parameter)
    return out


def score_on_test(operator, static, head, tensors, observation, scaled, mask_time,
                  index) -> dict:
    with torch.no_grad():
        rows = index[2]
        states, motif_logits = rollout_states(
            operator, tensors["x"][rows], tensors["recruited"][rows],
            tensors["displacement"][rows])
        field = recruited_field(states, tensors["target"][rows], observation)
        static_logits, _, _ = static(tensors["x"][rows], tensors["recruited"][rows],
                                     tensors["displacement"][rows])
        return {
            "time_mse": float(head.nll(field, scaled[rows], mask_time[rows])),
            "contact_nll": float(masked_contact_nll(
                motif_logits + static_logits, tensors["target"][rows],
                tensors["available"][rows], tensors["valid"][rows])),
        }


def assert_inheritance_is_exact(child, parent, tensors, observation, index) -> float:
    """A richer layer with its new parameter at zero must reproduce its parent exactly.

    Without this the chain is only nominally nested: a silent shape or name mismatch in
    the copy would leave the child at a partly untrained state and its comparison
    against the parent would measure the copy, not the mechanism.
    """
    rows = index[1][:64] if index[1].numel() > 64 else index[1]
    with torch.no_grad():
        child_states, _ = rollout_states(child, tensors["x"][rows], tensors["recruited"][rows],
                                         tensors["displacement"][rows])
        parent_states, _ = rollout_states(parent, tensors["x"][rows], tensors["recruited"][rows],
                                          tensors["displacement"][rows])
        gap = float((child_states - parent_states).abs().max())
    if not gap < 1e-5:
        raise AssertionError(f"warm start is not exact: max state difference {gap:.3e}")
    return gap


def process(patient: str) -> dict:
    torch.set_num_threads(int(_os.environ.get("TOPIC5_TORCH_THREADS", "1")))
    started = time.time()
    try:
        unit = load_frame_unit(MOTIF_ROOT, FRAME, patient)
        tensors = build_event_tensors_with_time(
            unit.ranks, unit.contacts_xy_mm, unit.event_lag_raw)
        split = np.asarray(unit.split)
        index = {value: torch.as_tensor(np.flatnonzero(split == value)) for value in (0, 1, 2)}
        if min(int(rows.numel()) for rows in index.values()) < 30:
            return {"patient": patient, "state": "too_few_events"}

        delta = tensors["time_delta"].numpy()
        valid = tensors["time_valid"].numpy()
        if (delta[valid] < 0).any():
            raise ValueError("negative time increment: the proxy is not monotone in rank "
                             "for this patient, which the target construction assumes")
        logged = np.log1p(delta)
        train_rows = split == 0
        centre = float(logged[train_rows][valid[train_rows]].mean())
        scale = max(float(logged[train_rows][valid[train_rows]].std()), 1e-9)
        standardised = (logged - centre) / scale
        scaled = torch.from_numpy(standardised.astype(np.float32))
        mask_time = tensors["time_valid"]

        centroid = tensors["centroid"].numpy()
        distance = np.zeros_like(delta)
        distance[:, :-1] = np.linalg.norm(centroid[:, 1:] - centroid[:, :-1], axis=-1)
        baselines = time_baseline_scores(
            standardised, valid, distance, tensors["target"].numpy(),
            train_rows, split == 2)

        observation = torch.as_tensor(np.asarray(unit.H, dtype=np.float32))
        covariates = np.column_stack([
            unit.contacts_xy_mm,
            (unit.ranks >= 0)[index[0]].mean(axis=0)]).astype(np.float32)
        n_contacts = len(unit.contact_names)

        rows: list[dict] = []
        parent_state, parent_theta, parent_model = None, 0.0, None
        for arm, model_id, added in CHAIN:
            angles = THETA_INITS if added and "theta" in added else (parent_theta,)
            # Two ways to inherit, both offered to validation.  Carrying the parent's
            # fitted read-out reproduces the parent exactly, which is what guarantees the
            # child cannot score worse; but a converged read-out also leaves almost no
            # gradient, and on one patient it held the child at the parent's solution
            # while a re-fitted read-out found a real improvement.  Offering only the
            # first would have made the protocol asymmetric — the parent fits its head
            # from scratch and the child does not.
            head_modes = ("inherit_readout", "refit_readout") if parent_state else ("fresh",)
            attempts = []
            for start_index, (theta, head_mode) in enumerate(
                    [(a, m) for a in angles for m in head_modes]):
                operator = build_operator(model_id, unit, theta, seed=start_index)
                inheritance = None
                if parent_state is not None:
                    operator.load_warm_start(parent_state)
                    for name in added:
                        parameter = getattr(operator, name, None)
                        if isinstance(parameter, torch.nn.Parameter) and name != "theta":
                            with torch.no_grad():
                                parameter.zero_()
                    inheritance = assert_inheritance_is_exact(
                        operator, parent_model, tensors, observation, index)
                static = StaticReadout(n_contacts, covariates, seed=start_index)
                head = TimeHead()
                if head_mode == "inherit_readout":
                    static.load_state_dict(parent_state["static_full"], strict=False)
                    head.load_state_dict(parent_state["head_full"], strict=False)
                fitted = train_one(operator, static, head, tensors, observation,
                                   scaled, mask_time, index)
                attempts.append({"start_index": start_index, "theta_init": float(theta),
                                 "head_mode": head_mode,
                                 "operator": operator, "static": static, "head": head,
                                 "inheritance_max_state_gap": inheritance, **fitted})
            chosen = min(attempts, key=lambda item: item["validation_time_mse"])
            spread = (max(a["validation_time_mse"] for a in attempts)
                      - min(a["validation_time_mse"] for a in attempts))
            test = score_on_test(chosen["operator"], chosen["static"], chosen["head"],
                                 tensors, observation, scaled, mask_time, index)
            rows.append({
                "patient": patient, "arm": arm, "model_id": model_id,
                "n_starts": len(attempts), "chosen_start": chosen["start_index"],
                "chosen_head_mode": chosen["head_mode"],
                "chosen_theta_init": chosen["theta_init"] if "theta" in added or arm != CHAIN[0][0] else None,
                "validation_time_mse": chosen["validation_time_mse"],
                "validation_spread_across_starts": spread,
                "inheritance_max_state_gap": chosen["inheritance_max_state_gap"],
                "epochs": chosen["epochs"],
                "operator_parameters": trainable_parameter_count(chosen["operator"]),
                **fitted_motif_parameters(chosen["operator"]),
                **test, **{f"baseline_{name}": value for name, value in baselines.items()},
            })
            parent_state = {**chosen["state"]["operator"],
                            "static_full": chosen["state"]["static"],
                            "head_full": chosen["state"]["head"]}
            parent_theta, parent_model = chosen["theta_init"], chosen["operator"]

        # free upper bound: same cell, free recurrent drive, its own random starts
        budget = trainable_parameter_count(build_operator("DM0_ISOTROPIC", unit, 0.0, 0))
        rank = free_rank_for_budget(unit.nodes_xy_mm.shape[0], max(4 * budget, 32))
        attempts = []
        for start_index in FREE_STARTS:
            operator = build_operator("DM0_ISOTROPIC", unit, 0.0, seed=start_index,
                                      free_rank=rank)
            static = StaticReadout(n_contacts, covariates, seed=start_index)
            head = TimeHead()
            attempts.append({"start_index": start_index, "operator": operator,
                             "static": static, "head": head,
                             **train_one(operator, static, head, tensors, observation,
                                         scaled, mask_time, index)})
        chosen = min(attempts, key=lambda item: item["validation_time_mse"])
        test = score_on_test(chosen["operator"], chosen["static"], chosen["head"],
                             tensors, observation, scaled, mask_time, index)
        rows.append({
            "patient": patient, "arm": FREE_ARM, "model_id": f"FREE_RANK_{rank}",
            "n_starts": len(attempts), "chosen_start": chosen["start_index"],
            "chosen_head_mode": "fresh", "chosen_theta_init": None,
            "validation_time_mse": chosen["validation_time_mse"],
            "validation_spread_across_starts": (
                max(a["validation_time_mse"] for a in attempts)
                - min(a["validation_time_mse"] for a in attempts)),
            "inheritance_max_state_gap": None, "epochs": chosen["epochs"],
            "operator_parameters": trainable_parameter_count(chosen["operator"]),
            **fitted_motif_parameters(chosen["operator"]),
            **test, **{f"baseline_{name}": value for name, value in baselines.items()},
        })

        clue = distance_time_relation(unit.ranks, unit.contacts_xy_mm, unit.event_lag_raw)
        adjacent = adjacent_distance_time_relation(tensors)
        for row in rows:
            row["all_pairs_partial_spearman"] = clue["partial_spearman"]
            row["adjacent_partial_spearman"] = adjacent["adjacent_partial_spearman"]
            row["observed_direction_persistence"] = direction_persistence(
                unit.ranks, unit.contacts_xy_mm)
        return {"patient": patient, "state": "complete", "rows": rows,
                "wall_seconds": time.time() - started}
    except Exception:
        return {"patient": patient, "state": "failed",
                "error": traceback.format_exc(limit=8), "wall_seconds": time.time() - started}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--patients", type=int, default=0)
    parser.add_argument("--tag", default="")
    arguments = parser.parse_args()

    census = pd.read_csv(ROOT / "results/topic5_capacity_constrained_history_motif_v0_2/INPUT_CENSUS.csv")
    patients = sorted(census[census["dataset"] == "SEEG"]["patient"])
    if arguments.patients:
        patients = patients[:arguments.patients]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    # M0 once, M1 over three angles x two read-out modes, M2 and M3 over the two
    # read-out modes, then the free upper bound's own random starts
    units = len(patients) * (1 + 2 * len(THETA_INITS) + 2 + 2 + len(FREE_STARTS))
    print(f"patients {len(patients)}  chain {[a for a, _, _ in CHAIN]} + {FREE_ARM}  "
          f"training units {units}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for position, payload in enumerate(pool.map(process, patients), start=1):
            results.append(payload)
            done = sum(item["state"] == "complete" for item in results)
            print(f"  [{position}/{len(patients)}] complete={done}", flush=True)

    rows = [row for payload in results for row in payload.get("rows", [])]
    suffix = f"_{arguments.tag}" if arguments.tag else ""
    pd.DataFrame(rows).to_csv(RESULT_ROOT / f"PER_ARM_SCORES{suffix}.csv", index=False)

    states: dict[str, int] = {}
    for payload in results:
        states[payload["state"]] = states.get(payload["state"], 0) + 1
    (RESULT_ROOT / f"RUN_STATUS{suffix}.json").write_text(json.dumps({
        "contract": "topic5_motif_time_targets_v0_3_run",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "patient_states": states, "n_rows": len(rows),
        "chain": [arm for arm, _, _ in CHAIN], "free_arm": FREE_ARM,
        "theta_starts": [float(v) for v in THETA_INITS],
        "time_baselines": list(TIME_BASELINES),
        "selection_rule": "validation time MSE only; contact NLL is a diagnostic",
        "statistical_unit": "patient; starts are optimisation choices, not seeds",
        "time_proxy_note": "within-event spectral-centroid position; not recruitment "
                           "time, not conduction delay, never a velocity",
        "failures": [{key: payload[key] for key in ("patient", "error")}
                     for payload in results if payload["state"] == "failed"],
    }, indent=2) + "\n")
    print(f"patients: {states}  rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

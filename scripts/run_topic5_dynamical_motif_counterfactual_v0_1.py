#!/usr/bin/env python3
"""Observable-prefix counterfactuals and latent gain for one motif unit.

The primary experiment edits things a clinician could in principle observe --
which contact appears in the second rank set, whether two ranks are tied or
ordered, whether one mid-axis contact participates -- re-encodes the whole
prefix and generates again under the frozen decoder.  The Jacobian section is
secondary and only interprets the finite-horizon gain; a hidden direction is
never presented as a stimulation coordinate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_topic5_dynamical_motif_unseen_v0_1 import (  # noqa: E402
    load_unit_model, write_json,
)
from src.topic5_dynamical_motif_analysis_v0_1 import mode_posterior, sequences_to_ranks  # noqa: E402
from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import build_motif_event_tensors  # noqa: E402
from src.topic5_dynamical_motif_rollout_v0_1 import stochastic_rollout, summarise_sequences  # noqa: E402

BRANCH_DRAWS = 64


def hash_pick(label: str, values: np.ndarray, count: int) -> np.ndarray:
    keys = [hashlib.sha256(f"{label}|{int(v)}".encode()).hexdigest() for v in values]
    return np.asarray(values)[np.argsort(keys)][:count]


def select_reference_states(unit, lengths, unseen, per_stage: int = 4) -> list[dict]:
    """Early / middle / late prefixes, stratified before any perturbation runs."""
    usable = unseen[lengths[unseen] >= 3]
    if usable.size == 0:
        return []
    entropy = -(unit.mode_posterior[usable] * np.log(
        np.clip(unit.mode_posterior[usable], 1e-9, 1.0))).sum(axis=1)
    block = np.argsort(np.argsort(unit.event_abs_time[usable])) * 3 // max(1, usable.size)
    uncertain = (entropy > np.median(entropy)).astype(int)
    states = []
    for stage, name in ((1, "early"), (2, "middle"), (3, "late")):
        pool = usable[lengths[usable] > stage]
        if pool.size == 0:
            continue
        chosen: list[int] = []
        for b in range(3):
            for u_flag in (0, 1):
                members = pool[(block[np.isin(usable, pool)][:len(pool)] == b)
                               & (uncertain[np.isin(usable, pool)][:len(pool)] == u_flag)] \
                    if pool.size == usable.size else pool
                if members.size:
                    chosen.extend(hash_pick(f"{unit.unit_id}|{name}", members, 1).tolist())
        if not chosen:
            chosen = hash_pick(f"{unit.unit_id}|{name}", pool, per_stage).tolist()
        for event in sorted(set(chosen))[:per_stage]:
            states.append({"event_index": int(event), "stage": name,
                           "prefix_length": int(min(stage + 1, lengths[event] - 1))})
    return states


def train_support(unit) -> np.ndarray:
    """Fraction of train events in which each contact participates."""
    train = unit.indices(0)
    return (unit.ranks[train] >= 0).mean(axis=0)


def axial_substitutions(unit, prefix: np.ndarray, support: np.ndarray,
                        min_support: float = 0.02) -> dict[str, np.ndarray | None]:
    """Replace one contact of the second rank set along +axis, -axis, orthogonal."""
    xy = unit.contacts_xy_mm
    if prefix.shape[0] < 2 or prefix[1].sum() == 0:
        return {}
    members = np.flatnonzero(prefix[1] > 0)
    used = np.flatnonzero(prefix.sum(0) > 0)
    target = members[0]
    eligible = np.asarray([c for c in range(unit.n_contacts)
                           if c not in set(used.tolist()) and support[c] >= min_support])
    if eligible.size == 0:
        return {}
    offset = xy[eligible] - xy[target]
    out: dict[str, np.ndarray | None] = {}
    for name, score in (("axis_plus", offset[:, 0]), ("axis_minus", -offset[:, 0])):
        forward = eligible[score > 0]
        if forward.size == 0:
            out[name] = None
            continue
        distance = np.linalg.norm(xy[forward] - xy[target], axis=1)
        replacement = int(forward[np.argmin(distance)])
        edited = prefix.copy()
        edited[1, target] = 0
        edited[1, replacement] = 1
        out[name] = edited
    orthogonal = eligible[np.abs(offset[:, 1]) > np.abs(offset[:, 0])]
    if orthogonal.size:
        distance = np.linalg.norm(xy[orthogonal] - xy[target], axis=1)
        replacement = int(orthogonal[np.argmin(distance)])
        edited = prefix.copy()
        edited[1, target] = 0
        edited[1, replacement] = 1
        out["orthogonal"] = edited
    else:
        out["orthogonal"] = None
    return out


def order_edits(prefix: np.ndarray) -> dict[str, np.ndarray | None]:
    """Adjacent swap, tie merge and a supported tie split of the last two ranks."""
    out: dict[str, np.ndarray | None] = {}
    if prefix.shape[0] < 3:
        return out
    swapped = prefix.copy()
    swapped[[-2, -1]] = swapped[[-1, -2]]
    out["adjacent_swap"] = swapped
    merged = prefix[:-1].copy()
    merged[-1] = np.clip(prefix[-2] + prefix[-1], 0, 1)
    out["tie_merge"] = merged
    members = np.flatnonzero(prefix[-1] > 0)
    if members.size >= 2:
        split = np.zeros((prefix.shape[0] + 1, prefix.shape[1]), dtype=prefix.dtype)
        split[:-1] = prefix
        split[-2] = 0
        split[-2, members[0]] = 1
        split[-1, members[1:]] = 1
        out["tie_split"] = split
    else:
        out["tie_split"] = None
    return out


def extent_edits(unit, prefix: np.ndarray, support: np.ndarray,
                 min_support: float = 0.02) -> dict[str, np.ndarray | None]:
    """Add or drop one mid-axis contact without touching the direction."""
    xy = unit.contacts_xy_mm
    used = np.flatnonzero(prefix.sum(0) > 0)
    out: dict[str, np.ndarray | None] = {}
    span = xy[used, 0]
    middle = 0.5 * (span.min() + span.max())
    candidates = np.asarray([c for c in range(unit.n_contacts)
                             if c not in set(used.tolist()) and support[c] >= min_support])
    if candidates.size:
        pick = int(candidates[np.argmin(np.abs(xy[candidates, 0] - middle))])
        added = prefix.copy()
        added[-1, pick] = 1
        out["extent_add"] = added
    else:
        out["extent_add"] = None
    droppable = [c for c in np.flatnonzero(prefix[-1] > 0) if prefix[-1].sum() > 1]
    if droppable:
        pick = int(droppable[int(np.argmin(np.abs(xy[droppable, 0] - middle)))])
        dropped = prefix.copy()
        dropped[-1, pick] = 0
        out["extent_drop"] = dropped
    else:
        out["extent_drop"] = None
    return out


@torch.no_grad()
def finite_horizon_gain(model, unit, tensors, event_indices, device, horizons=range(1, 11)) -> dict:
    """State gain of the frozen Jacobian product along real trajectories."""
    chosen = torch.as_tensor(np.asarray(event_indices, dtype=int))
    batch = {key: tensors[key][chosen].to(device) for key in ("x", "recruited", "displacement")}
    terms = model.recurrent_terms()
    u, _ = model.axis_unit()
    gate = model.direction_gate(batch["displacement"], u)
    kappa = float(terms["kappa"])
    h = torch.zeros(len(chosen), model.n_nodes, device=device)
    product = torch.eye(model.n_nodes, device=device).expand(len(chosen), -1, -1).clone()
    gains, top_directions = [], None
    steps = min(batch["x"].shape[1], max(horizons))
    for t in range(steps):
        pre_h = h
        h = model.step(h, batch["x"][:, t], gate[:, t], terms)
        weight = model.recurrent_matrix(float(gate[:, t].mean()))
        derivative = 1.0 - torch.tanh(
            (batch["x"][:, t] @ model.H) * model.input_gain
            + model.recurrent_drive(pre_h, gate[:, t], terms) + model.node_bias) ** 2
        jacobian = (1.0 - kappa) * torch.eye(model.n_nodes, device=device)[None] \
            + kappa * derivative[:, :, None] * weight[None]
        product = torch.bmm(jacobian, product)
        singular = torch.linalg.svdvals(product)
        gains.append({"horizon": t + 1,
                      "state_gain_mean": float(singular[:, 0].mean()),
                      "state_gain_median": float(singular[:, 0].median()),
                      "log_state_gain_mean": float(torch.log(singular[:, 0]).mean())})
        if t + 1 == min(3, steps):
            _, _, right = torch.linalg.svd(product)
            top_directions = right[:, 0, :]
    peak = max(gains, key=lambda row: row["state_gain_mean"]) if gains else None
    return {
        "per_horizon": gains,
        "peak_horizon": peak["horizon"] if peak else None,
        "peak_state_gain": peak["state_gain_mean"] if peak else None,
        "returns_after_peak": bool(
            peak is not None and gains[-1]["state_gain_mean"] < peak["state_gain_mean"]),
        "finite_nonreturning": bool(
            peak is not None and gains[-1]["state_gain_mean"] >= peak["state_gain_mean"]
            and np.isfinite(gains[-1]["state_gain_mean"])),
        "top_direction_available": top_directions is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draws", type=int, default=BRANCH_DRAWS)
    parser.add_argument("--gate-rule", default="M2-2RANK")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    unit = load_frame_unit(args.out_root, args.frame, args.unit_id)
    unit_dir = (args.out_root / args.tag / args.frame / args.unit_id / args.model
                / f"seed{args.seed_index}")
    model, head, contract, _ = load_unit_model(unit, unit_dir, device)
    tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm, gate_rule=args.gate_rule)
    lengths = tensors["length"].numpy()
    unseen = unit.indices(-1)
    states = select_reference_states(unit, lengths, unseen)
    support = train_support(unit)
    modes_path = (args.out_root / "frame_cache" / args.frame / args.unit_id / "train_only_modes.npz"
                  if args.frame == "GEOMETRY_ONLY_PCA2"
                  else args.out_root.parent / "topic5_multiscale_effective_scaffold_v0_5"
                  / "cache" / args.unit_id / "train_only_modes.npz")
    modes = np.load(modes_path, allow_pickle=False)
    centers, temperature = np.asarray(modes["centers"]), float(modes["temperature"][0])

    rows = []
    for state in states:
        event = state["event_index"]
        prefix_length = int(state["prefix_length"])
        prefix = tensors["x"][event, :prefix_length].numpy().astype(np.float32)
        branches: dict[str, np.ndarray | None] = {"unperturbed": prefix}
        branches.update(axial_substitutions(unit, prefix, support))
        branches.update(order_edits(prefix))
        branches.update(extent_edits(unit, prefix, support))
        reference = None
        for name, edited in branches.items():
            if edited is None:
                rows.append({"frame": args.frame, "unit_id": args.unit_id,
                             "subject": unit.subject, "model_id": args.model,
                             "seed_index": args.seed_index, "event_index": event,
                             "stage": state["stage"], "prefix_length": prefix_length,
                             "branch": name, "status": "UNMATCHED"})
                continue
            block = torch.as_tensor(edited, dtype=torch.float32)[None].repeat(args.draws, 1, 1)
            result = stochastic_rollout(
                model, head, contract, block, unit.contacts_xy_mm, device,
                mode="FULL_STOP", gate_rule=args.gate_rule,
                rng_label=f"{args.frame}|{args.unit_id}|cf|{event}|{name}")
            fixed = stochastic_rollout(
                model, head, contract, block, unit.contacts_xy_mm, device,
                mode="FIXED_H", horizon=3, gate_rule=args.gate_rule,
                rng_label=f"{args.frame}|{args.unit_id}|cf|{event}|{name}")
            summary = summarise_sequences(result["sequence"], result["n_emitted"],
                                          unit.contacts_xy_mm, np.array([1.0, 0.0]))
            fixed_summary = summarise_sequences(fixed["sequence"], fixed["n_emitted"],
                                                unit.contacts_xy_mm, np.array([1.0, 0.0]))
            posterior = mode_posterior(
                sequences_to_ranks(result["sequence"], result["n_emitted"]),
                centers, temperature).mean(axis=0)
            record = {
                "frame": args.frame, "unit_id": args.unit_id, "subject": unit.subject,
                "model_id": args.model, "seed_index": args.seed_index,
                "event_index": event, "stage": state["stage"],
                "prefix_length": prefix_length, "branch": name, "status": "OK",
                "draws": args.draws,
                "r_late_x": float(summary["r_late"][:, 0].mean()),
                "r_late_y": float(summary["r_late"][:, 1].mean()),
                "r_last_x": float(summary["r_last"][:, 0].mean()),
                "l_axis_full": float(summary["l_axis"].mean()),
                "l_axis_fixed_h3": float(fixed_summary["l_axis"].mean()),
                "n_rank_full": float(summary["n_rank"].mean()),
                "n_contact_full": float(summary["n_contact"].mean()),
                "mode_probability_a": float(posterior[0]),
            }
            if name == "unperturbed":
                reference = record
            elif reference is not None:
                record["delta_r_late_mm"] = float(np.hypot(
                    record["r_late_x"] - reference["r_late_x"],
                    record["r_late_y"] - reference["r_late_y"]))
                record["delta_r_late_axial_mm"] = record["r_late_x"] - reference["r_late_x"]
                record["delta_l_axis_full"] = record["l_axis_full"] - reference["l_axis_full"]
                record["delta_l_axis_fixed_h3"] = (record["l_axis_fixed_h3"]
                                                   - reference["l_axis_fixed_h3"])
                record["delta_n_rank_full"] = record["n_rank_full"] - reference["n_rank_full"]
                record["delta_mode_probability_a"] = (record["mode_probability_a"]
                                                      - reference["mode_probability_a"])
            rows.append(record)

    table = pd.DataFrame(rows)
    table.to_csv(unit_dir / "counterfactual_branches.csv", index=False)

    reference_events = np.asarray([state["event_index"] for state in states], dtype=int)
    latent = (finite_horizon_gain(model, unit, tensors, reference_events[:32], device)
              if reference_events.size else {"per_horizon": []})
    write_json(unit_dir / "counterfactual_summary.json", {
        "contract": "topic5_dynamical_motif_counterfactual_v0_1",
        "frame": args.frame, "unit_id": args.unit_id, "subject": unit.subject,
        "model_id": args.model, "seed_index": args.seed_index,
        "n_reference_states": len(states),
        "reference_states": states,
        "branch_denominators": table.groupby("branch").status.value_counts().unstack(
            fill_value=0).to_dict(orient="index") if not table.empty else {},
        "finite_horizon_gain": latent,
        "seconds": time.time() - started,
    })
    print(json.dumps({"unit_id": args.unit_id, "model": args.model,
                      "n_states": len(states), "n_rows": len(table),
                      "seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()

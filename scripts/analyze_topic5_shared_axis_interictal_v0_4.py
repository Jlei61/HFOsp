#!/usr/bin/env python3
"""Interictal evidence for the v0.4 shared-axis model.

Three things are produced, all from interictal data only and all before any
ictal value is unsealed:

1. held-out next-contact prediction, patient-paired against the static
   participation prior, the dense GRU and the within-event rank shuffle;
2. axis stability, across seeds and across a chronological split of fit60,
   after aligning the coordinate's unidentifiable global sign;
3. source-excluded bidirectional recovery: given a held-out event's real
   first rank set, does the model's own ordering of the *remaining* contacts
   match that event better under its matched direction than under the
   swapped one.

The axis-identifiable subgroup is defined here, from these interictal
quantities alone, so it cannot be tuned against the ictal readout later.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_shared_axis_rnn_unit_v0_4 import (  # noqa: E402
    load_one_patient_record,
)
from src.topic5_patient_specific_rnn_bridge import chronological_60_20_20  # noqa: E402
from src.topic5_shared_axis_rnn import SharedAxisPropagationRNN  # noqa: E402
from src.topic5_shared_scaffold_rnn import (  # noqa: E402
    OrdinaryDenseGRUBaseline,
    batched_exact_conditional_k_subset_log_probability,
    estimate_node_hazard_bias,
)


@torch.no_grad()
def static_contact_metrics(groups, counts, indices, bias, batch_size: int = 2048):
    """fit60 participation prior scored on exactly the test decisions."""

    total, n_continue, hits = 0.0, 0, 0
    for start in range(0, len(indices), batch_size):
        selected = np.asarray(indices[start:start + batch_size], dtype=np.int64)
        event = torch.as_tensor(groups[selected], dtype=torch.long)
        event_count = torch.as_tensor(counts[selected], dtype=torch.long)
        logits = torch.as_tensor(bias, dtype=torch.float32).expand(len(selected), -1)
        seen = torch.zeros_like(event, dtype=torch.bool)
        for step in range(int(event_count.max().item()) - 1):
            seen |= event == step
            active = event_count > step + 1
            target = event == step + 1
            eligible = ~seen
            value = batched_exact_conditional_k_subset_log_probability(
                node_logits=logits, eligible=eligible, next_set=target, active=active
            )
            total += float((-value).sum().item())
            rows = torch.where(active)[0]
            if rows.numel():
                predicted = torch.argmax(logits.masked_fill(~eligible, -torch.inf), dim=1)
                hits += int(target[rows, predicted[rows]].sum().item())
                n_continue += int(rows.numel())
    if not n_continue:
        raise RuntimeError("static prior has no scorable continuation decisions")
    return {"contact_nll_per_continue_decision": total / n_continue,
            "top1_next_contact_accuracy": hits / n_continue}


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=float) + "\n")
    tmp.replace(path)


def load_axis_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    hyper = checkpoint["model_hyperparameters"]
    model = SharedAxisPropagationRNN(
        fixed_adjacency=state["fixed_adjacency"],
        participation_bias=np.asarray(checkpoint["participation_bias"], dtype=np.float32),
        length_scale=float(hyper["length_scale"]),
        delta=float(hyper["delta"]),
        smoothness_weight=float(hyper["smoothness_weight"]),
        direction_gain=float(hyper["direction_gain"]),
    )
    model.load_state_dict(state)
    model.eval()
    return checkpoint, model


def learned_axis(checkpoint_path: Path) -> np.ndarray:
    _, model = load_axis_model(checkpoint_path)
    return model.operator_components()["axis_coordinate"].detach().numpy()


def align_sign(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """The coordinate's global sign is not identifiable; align before comparing."""

    return -other if float(np.dot(reference, other)) < 0 else other


def axis_stability(axes: dict[str, np.ndarray], endpoint_fraction: float, min_endpoint: int):
    keys = sorted(axes)
    if len(keys) < 2:
        return None
    reference = axes[keys[0]]
    aligned = {key: align_sign(reference, axes[key]) for key in keys}
    pairwise = [
        abs(float(spearmanr(aligned[a], aligned[b]).statistic))
        for i, a in enumerate(keys) for b in keys[i + 1:]
    ]
    n = len(reference)
    k = max(int(min_endpoint), int(np.ceil(float(endpoint_fraction) * n)))
    ends = {}
    for key in keys:
        order = np.argsort(aligned[key])
        ends[key] = (frozenset(order[:k].tolist()), frozenset(order[-k:].tolist()))
    jaccard = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            for side in (0, 1):
                left, right = ends[a][side], ends[b][side]
                jaccard.append(len(left & right) / max(len(left | right), 1))
    return {
        "keys": keys,
        "min_pairwise_abs_spearman": float(np.min(pairwise)),
        "median_pairwise_abs_spearman": float(np.median(pairwise)),
        "pairwise_abs_spearman": pairwise,
        "median_endpoint_jaccard": float(np.median(jaccard)),
    }


@torch.no_grad()
def bidirectional_recovery(model, groups: np.ndarray, counts: np.ndarray,
                           indices: np.ndarray, axis: np.ndarray,
                           endpoint_fraction: float, min_endpoint: int,
                           min_events_per_side: int):
    """Matched-minus-swapped ordering agreement, with the source excluded.

    For each held-out event the model is given the event's real first rank
    set and then scored on how well its expected arrival order matches the
    observed order of the contacts it was *not* given.  The same event is
    scored again with the direction state forced to the opposite sign; the
    difference isolates the flow from everything both arms share.
    """

    n = int(model.n_contacts)
    k = max(int(min_endpoint), int(np.ceil(float(endpoint_fraction) * n)))
    order = np.argsort(axis)
    side_of = np.zeros(n, dtype=int)
    side_of[order[:k]] = -1
    side_of[order[-k:]] = +1

    components = model.operator_components()
    symmetric, skew = components["W"], components["W_skew"]
    coordinate_t = torch.as_tensor(axis, dtype=symmetric.dtype)
    per_side = defaultdict(list)
    for index in np.asarray(indices, dtype=int):
        row = groups[index]
        if int(counts[index]) < 3:
            continue
        first = row == 0
        if not first.any():
            continue
        mean_side = float(np.mean(side_of[first]))
        side = -1 if mean_side < 0 else (+1 if mean_side > 0 else 0)
        if side == 0:
            continue
        observed = np.where(row >= 0)[0]
        later = observed[row[observed] > 0]
        if len(later) < 3:
            continue
        x = torch.as_tensor(first, dtype=symmetric.dtype)
        x = x / x.sum().clamp_min(1.0)
        direction = -torch.tanh(model.direction_gain * (x * coordinate_t).sum() / x.sum().clamp_min(1e-9))
        symmetric_drive = symmetric @ x
        skew_drive = skew @ x
        scores = {}
        for label, sign in (("matched", 1.0), ("swapped", -1.0)):
            propagation = symmetric_drive + sign * model.flow_weight * direction * skew_drive
            logits = (
                model.participation_bias
                + model.propagation_weight * propagation
                - model.restraint_weight * symmetric_drive
            ).detach().numpy()
            # earlier predicted arrival = higher logit, so negate before ranking
            scores[label] = float(spearmanr(-logits[later], row[later]).statistic)
        if np.isfinite(scores["matched"]) and np.isfinite(scores["swapped"]):
            per_side[side].append((scores["matched"], scores["swapped"]))

    out = {}
    for side in (-1, +1):
        rows = per_side.get(side, [])
        name = "minus" if side < 0 else "plus"
        if len(rows) < int(min_events_per_side):
            out[name] = {"n_events": len(rows), "eligible": False}
            continue
        matched = np.array([r[0] for r in rows]); swapped = np.array([r[1] for r in rows])
        out[name] = {
            "n_events": len(rows), "eligible": True,
            "matched_median_spearman": float(np.median(matched)),
            "swapped_median_spearman": float(np.median(swapped)),
            "matched_minus_swapped_median": float(np.median(matched - swapped)),
        }
    both = all(out[s].get("eligible") for s in ("minus", "plus"))
    out["both_sides_eligible"] = both
    out["matched_minus_swapped_mean_of_sides"] = (
        float(np.mean([out[s]["matched_minus_swapped_median"] for s in ("minus", "plus")]))
        if both else None
    )
    return out


def paired(frame: pd.DataFrame, a: str, b: str, metric: str, lower_better: bool,
           tie_atol: float = 1e-9):
    wide = frame.pivot(index="subject", columns="model", values=metric)
    if a not in wide or b not in wide:
        return {"status": "NOT_AVAILABLE"}
    wide = wide.dropna(subset=[a, b])
    delta = (wide[b] - wide[a]).to_numpy(float) if lower_better else (wide[a] - wide[b]).to_numpy(float)
    delta[np.abs(delta) <= tie_atol] = 0.0
    nonzero = delta[delta != 0]
    try:
        p_two = float(wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) else 1.0
    except ValueError:
        p_two = 1.0
    rng = np.random.default_rng(4242)
    boot = np.median(rng.choice(delta, size=(5000, len(delta)), replace=True), axis=1)
    return {
        "status": "COMPLETE", "positive_means": f"{a}_better", "n": int(len(delta)),
        "subjects": wide.index.astype(str).tolist(),
        "median_delta": float(np.median(delta)),
        "bootstrap_95ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wilcoxon_two_sided_p": p_two,
        "n_positive": int((delta > 0).sum()), "n_negative": int((delta < 0).sum()),
        "n_tied": int((delta == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/topic5_interictal_ictal_shared_axis_rnn_v0_4.yaml")
    parser.add_argument("--gru-root", type=Path, default=None,
                        help="v0.3 root whose ordinary_gru checkpoints are reused")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output_root"]
    dataset_root = Path(config["dataset_artifact_root"]).resolve() / config["dataset_root"]
    seeds = list(map(int, config["training"]["seeds"]))
    fraction = float(config["rollout"]["endpoint_fraction"])
    min_endpoint = int(config["rollout"]["endpoint_min_contacts"])
    identifiable = config["axis_identifiable"]
    development = sorted(map(str, config["development_audit"]["subjects"]))

    rows, axis_rows = [], {}
    per_subject = output / "per_subject"
    subjects = sorted(p.name for p in per_subject.iterdir() if p.is_dir())
    for subject in subjects:
        record = load_one_patient_record(dataset_root, subject)
        groups = np.asarray(record.group_ids, dtype=np.int64)
        counts = np.asarray(record.group_count, dtype=np.int64)
        fit60, _, test20 = chronological_60_20_20(record)

        for unit in sorted((per_subject / subject).iterdir()):
            for seed_dir in sorted(unit.glob("seed_*")):
                done = seed_dir / "DONE.json"
                if not done.is_file():
                    continue
                payload = json.loads(done.read_text())
                if payload.get("status") != "COMPLETE" or payload.get("smoke"):
                    continue
                rows.append({
                    "subject": subject, "model": payload["model"],
                    "fit_half": payload.get("fit_half"), "seed": int(payload["seed"]),
                    "unit": unit.name,
                    "contact_nll": float(payload["test"]["contact_nll_per_continue_decision"]),
                    "top1_accuracy": float(payload["test"]["top1_next_contact_accuracy"]),
                    "n_contacts": int(payload["n_contacts"]),
                })
                if payload["model"] == "shared_axis":
                    key = (payload.get("fit_half") or "full", int(payload["seed"]))
                    axis_rows.setdefault(subject, {})[key] = seed_dir / "checkpoint.pt"

        # static participation prior on the same test decisions
        hazard = estimate_node_hazard_bias(
            groups[fit60], pseudocount=float(config["training"]["hazard_pseudocount"])
        )
        static = static_contact_metrics(
            groups, counts, test20, np.asarray(hazard["bias"], dtype=np.float32)
        )
        rows.append({
            "subject": subject, "model": "static", "fit_half": None, "seed": -1,
            "unit": "static",
            "contact_nll": float(static["contact_nll_per_continue_decision"]),
            "top1_accuracy": float(static["top1_next_contact_accuracy"]),
            "n_contacts": int(len(record.contact_names)),
        })

    seed_frame = pd.DataFrame(rows)
    seed_frame.to_csv(output / "interictal_seed_metrics.csv", index=False)
    full = seed_frame[seed_frame.fit_half.isna()]
    patient = (full.groupby(["subject", "model"], as_index=False)
               .median(numeric_only=True).sort_values(["subject", "model"]))
    patient.to_csv(output / "interictal_patient_metrics.csv", index=False)

    # -------------------------------------------------- axis stability + recovery
    axis_report = {}
    for subject, checkpoints in axis_rows.items():
        seed_axes = {f"seed_{s}": learned_axis(p) for (half, s), p in checkpoints.items()
                     if half == "full"}
        half_axes = {half: learned_axis(p) for (half, s), p in checkpoints.items()
                     if half in ("first", "second")}
        entry = {
            "seed_stability": axis_stability(seed_axes, fraction, min_endpoint),
            "split_half_stability": axis_stability(half_axes, fraction, min_endpoint),
        }
        primary = checkpoints.get(("full", seeds[0]))
        if primary is not None:
            record = load_one_patient_record(dataset_root, subject)
            groups = np.asarray(record.group_ids, dtype=np.int64)
            counts = np.asarray(record.group_count, dtype=np.int64)
            _, _, test20 = chronological_60_20_20(record)
            _, model = load_axis_model(primary)
            entry["bidirectional"] = bidirectional_recovery(
                model, groups, counts, test20,
                model.operator_components()["axis_coordinate"].detach().numpy(),
                fraction, min_endpoint, int(identifiable["min_test_events_per_side"]),
            )
        axis_report[subject] = entry

    def is_identifiable(entry) -> bool:
        seed = entry.get("seed_stability") or {}
        half = entry.get("split_half_stability") or {}
        bi = entry.get("bidirectional") or {}
        margin = bi.get("matched_minus_swapped_mean_of_sides")
        return bool(
            bi.get("both_sides_eligible")
            and seed.get("min_pairwise_abs_spearman", -1)
            >= float(identifiable["min_seed_axis_abs_spearman"])
            and half.get("min_pairwise_abs_spearman", -1)
            >= float(identifiable["min_split_half_abs_spearman"])
            and (margin is not None and margin > 0
                 if identifiable["require_positive_matched_minus_swapped"] else True)
        )

    for subject, entry in axis_report.items():
        entry["axis_identifiable"] = is_identifiable(entry)
    atomic_json(output / "axis_stability.json", axis_report)

    confirmation = patient[~patient.subject.isin(development)]
    comparisons = {
        scope: {
            f"shared_axis_vs_{other}__{metric}": paired(frame, "shared_axis", other, metric, lower)
            for other in ("static", "ordinary_gru", "shared_axis_rank_shuffle")
            for metric, lower in (("contact_nll", True), ("top1_accuracy", False))
        }
        for scope, frame in (("all_34", patient), ("confirmation_31", confirmation))
    }
    summary = {
        "contract": config["contract"],
        "target_values_read": False,
        "primary_endpoint": "test20 contact identity | continue, k",
        "n_subjects": int(patient.subject.nunique()),
        "n_confirmation_subjects": int(confirmation.subject.nunique()),
        "development_subjects": development,
        "patient_medians": {
            model: {"contact_nll": float(g.contact_nll.median()),
                    "top1_accuracy": float(g.top1_accuracy.median())}
            for model, g in patient.groupby("model")
        },
        "comparisons": comparisons,
        "n_axis_identifiable": int(sum(e["axis_identifiable"] for e in axis_report.values())),
        "axis_identifiable_subjects": sorted(
            s for s, e in axis_report.items() if e["axis_identifiable"]
        ),
    }
    atomic_json(output / "interictal_cohort_statistics.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("n_subjects", "n_confirmation_subjects", "n_axis_identifiable")}))


if __name__ == "__main__":
    main()

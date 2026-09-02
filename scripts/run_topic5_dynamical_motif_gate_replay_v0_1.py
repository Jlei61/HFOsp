#!/usr/bin/env python3
"""Direction-commitment replay on frozen motif checkpoints.

Nothing is re-fitted.  A frozen DM2 checkpoint is re-scored under the two other
direction-commitment rules and under a prefix-length sweep, so the result is an
implementation sensitivity, not a claim about the best each rule could reach.
Also reports how often the direction evidence flips sign inside one event.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_topic5_dynamical_motif_unseen_v0_1 import load_unit_model, write_json  # noqa: E402
from scripts.train_topic5_dynamical_motif_unit_v0_1 import evaluate, place_tensors  # noqa: E402
from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import GATE_RULES, build_motif_event_tensors  # noqa: E402

MAX_PREFIX = 6


@torch.no_grad()
def gate_statistics(model, tensors, indices, device, batch_size=1024) -> dict:
    """Sign stability of the direction evidence inside an event."""
    indices = np.asarray(indices, dtype=int)
    flips, steps, magnitude = 0, 0, []
    u, _ = model.axis_unit()
    for begin in range(0, indices.size, batch_size):
        chosen = torch.as_tensor(indices[begin:begin + batch_size], device=tensors["x"].device)
        displacement = tensors["displacement"][chosen].to(device)
        valid = tensors["valid"][chosen].to(device)
        gate = model.direction_gate(displacement, u)
        sign = torch.sign(gate)
        change = (sign[:, 1:] != sign[:, :-1]) & valid[:, 1:] & valid[:, :-1] \
            & (sign[:, 1:] != 0) & (sign[:, :-1] != 0)
        flips += int(change.sum())
        steps += int((valid[:, 1:] & valid[:, :-1]).sum())
        magnitude.append(gate[valid].abs().cpu().numpy())
    values = np.concatenate(magnitude) if magnitude else np.zeros(1)
    return {"sign_flip_rate": float(flips / max(1, steps)),
            "gate_abs_median": float(np.median(values)),
            "gate_abs_p90": float(np.quantile(values, 0.9))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--model", default="DM2_LOCAL_DIRECTIONAL")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--primary-gate", default="M2-2RANK")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows, prefix_rows = [], []
    base = args.out_root / args.tag / args.frame
    for metrics_path in sorted(base.glob(f"*/{args.model}/seed{args.seed_index}/metrics.json")):
        unit_id = metrics_path.parts[-4]
        unit = load_frame_unit(args.out_root, args.frame, unit_id)
        model, head, contract, record = load_unit_model(unit, metrics_path.parent, device)
        unseen, calibration = unit.indices(-1), unit.indices(1)
        for rule in GATE_RULES:
            tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm, gate_rule=rule)
            tensors, _ = place_tensors(tensors, device)
            unseen_scores = evaluate(model, tensors, unseen, device, 1024, 1.0)
            calibration_scores = evaluate(model, tensors, calibration, device, 1024, 1.0)
            statistics = gate_statistics(model, tensors, unseen, device)
            rows.append({
                "frame": args.frame, "unit_id": unit_id, "subject": unit.subject,
                "model_id": args.model, "seed_index": args.seed_index,
                "gate_rule": rule, "is_primary": rule == args.primary_gate,
                "beta": record["numerical_audit"]["beta"],
                "unseen_contact_nll": unseen_scores["contact_nll"],
                "unseen_next_bce": unseen_scores["next_bce"],
                "unseen_stop_bce": unseen_scores["stop_bce"],
                "unseen_top1": unseen_scores["top1"],
                "calibration_contact_nll": calibration_scores["contact_nll"],
                **statistics,
            })
            if rule != args.primary_gate:
                continue
            # Where in the event does the direction evidence start to matter?
            lengths = tensors["length"].numpy()
            for prefix in range(1, MAX_PREFIX + 1):
                subset = unseen[lengths[unseen] > prefix]
                if subset.size < 20:
                    continue
                scores = evaluate(model, tensors, subset, device, 1024, 1.0)
                prefix_rows.append({
                    "frame": args.frame, "unit_id": unit_id, "subject": unit.subject,
                    "gate_rule": rule, "prefix_length": prefix,
                    "n_events": int(subset.size),
                    "unseen_contact_nll": scores["contact_nll"],
                    "unseen_top1": scores["top1"],
                })
        print(f"[gate] {unit_id} done", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.out_root / "DIRECTION_GATE_PER_PATIENT.csv", index=False)
    pd.DataFrame(prefix_rows).to_csv(
        args.out_root / "DIRECTION_GATE_EMERGENCE_PER_PATIENT.csv", index=False)
    summary = {"contract": "topic5_dynamical_motif_gate_replay_v0_1",
               "frame": args.frame, "model": args.model, "seed_index": args.seed_index,
               "primary_gate": args.primary_gate,
               "n_patients": int(table.subject.nunique()) if not table.empty else 0,
               "interpretation": "frozen-checkpoint replay: an implementation sensitivity, "
                                 "not the best each commitment rule could reach if retrained",
               "seconds": time.time() - started}
    if not table.empty:
        pivot = table.pivot_table(index="subject", columns="gate_rule",
                                  values="unseen_contact_nll")
        summary["median_unseen_contact_nll_by_rule"] = {
            rule: float(pivot[rule].median()) for rule in pivot.columns}
        summary["patients_where_rule_is_best"] = (
            pivot.idxmin(axis=1).value_counts().to_dict())
    write_json(args.out_root / "DIRECTION_GATE_REPLAY_SUMMARY.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_patients", "primary_gate")}), flush=True)


if __name__ == "__main__":
    main()

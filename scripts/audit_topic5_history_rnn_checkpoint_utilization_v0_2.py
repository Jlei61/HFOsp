#!/usr/bin/env python3
"""Audit whether selected G1 checkpoints contain and use a nonzero history branch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _half_life(raw: torch.Tensor) -> np.ndarray:
    rate = torch.nn.functional.softplus(raw).detach().cpu().numpy()
    return np.log(2.0) / np.maximum(rate, np.finfo(float).tiny) / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoints", type=int, default=102)
    args = parser.parse_args()
    g1 = args.g1_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    checkpoint_cycles = set()
    for checkpoint_path in sorted(g1.glob("seed_*/*/checkpoint.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["history_state"]
        done = json.loads((checkpoint_path.parent / "DONE.json").read_text())
        metrics = done["metrics"]
        half_life = _half_life(state["history.raw_decay_rate"])
        log = pd.read_csv(checkpoint_path.parent / "training_log.csv")
        history_log = log.loc[log.stage == "chronological_residual"]
        per_cycle = history_log.groupby("cycle").total.mean()
        history_cycles = int(checkpoint.get("config", {}).get("history_cycles", 0))
        checkpoint_cycles.add(history_cycles)
        rows.append({
            "seed": int(checkpoint["seed"]),
            "subject": str(checkpoint["heldout_subject"]),
            "history_cycles": history_cycles,
            "participation_readout_norm": float(
                state["heads.participation.state_to_query.weight"].norm()
            ),
            "rank_readout_norm": float(
                state["heads.relative_rank.state_to_query.weight"].norm()
            ),
            "gru_input_norm": float(state["history.cell.weight_ih"].norm()),
            "gru_recurrent_norm": float(state["history.cell.weight_hh"].norm()),
            "median_half_life_hours": float(np.median(half_life)),
            "half_life_absolute_drift_from_2h": float(
                np.median(np.abs(half_life - 2.0))
            ),
            "base_to_history_bce_gain": float(
                metrics["base_unordered"]["participation_bce"]
                - metrics["chronological_history"]["participation_bce"]
            ),
            "matched_to_history_bce_gain": float(
                metrics["matched_unordered"]["participation_bce"]
                - metrics["chronological_history"]["participation_bce"]
            ),
            "cycle3_minus_cycle2_train_loss": (
                float(per_cycle.loc[3] - per_cycle.loc[2])
                if {2, 3}.issubset(per_cycle.index) else np.nan
            ),
            "last_minus_previous_cycle_train_loss": (
                float(per_cycle.loc[history_cycles] - per_cycle.loc[history_cycles - 1])
                if history_cycles > 1
                and {history_cycles - 1, history_cycles}.issubset(per_cycle.index)
                else np.nan
            ),
        })
    frame = pd.DataFrame(rows)
    if len(frame) != int(args.expected_checkpoints):
        raise RuntimeError(
            f"checkpoint audit incomplete: {len(frame)}/{args.expected_checkpoints}"
        )
    if len(checkpoint_cycles) != 1:
        raise RuntimeError(f"mixed checkpoint training budgets: {checkpoint_cycles}")
    frame.to_csv(output / "checkpoint_utilization_metrics.csv", index=False)

    direct_rows = []
    for done_path in sorted(args.direct_root.resolve().glob("*/DONE.json")):
        done = json.loads(done_path.read_text())
        diagnostics = done.get("history_state_diagnostics")
        if diagnostics:
            direct_rows.append({"subject": done["heldout_subject"], **diagnostics})
    direct = pd.DataFrame(direct_rows)
    direct.to_csv(output / "direct_state_trajectory_metrics.csv", index=False)
    result = {
        "status": "STATE_BRANCH_ACTIVE_BUT_IDENTIFIABILITY_UNRESOLVED",
        "contract": "topic5_history_rnn_checkpoint_utilization_v0_2",
        "n_checkpoints": len(frame),
        "history_checkpoint_cycles": int(next(iter(checkpoint_cycles))),
        "all_readout_norms_nonzero": bool(
            np.all(frame.participation_readout_norm > 0)
            and np.all(frame.rank_readout_norm > 0)
        ),
        "median_participation_readout_norm": float(
            frame.participation_readout_norm.median()
        ),
        "median_state_branch_gain_over_zero_state_base": float(
            frame.base_to_history_bce_gain.median()
        ),
        "median_increment_over_capacity_matched_M1": float(
            frame.matched_to_history_bce_gain.median()
        ),
        "median_half_life_absolute_drift_from_2h": float(
            frame.half_life_absolute_drift_from_2h.median()
        ),
        "median_cycle3_minus_cycle2_train_loss": float(
            frame.cycle3_minus_cycle2_train_loss.median()
        ),
        "median_last_minus_previous_cycle_train_loss": float(
            frame.last_minus_previous_cycle_train_loss.median()
        ),
        "n_direct_state_trajectory_folds_available": int(len(direct)),
        "median_direct_state_variance": (
            float(direct.history_state_variance.median()) if len(direct) else None
        ),
        "interpretation": (
            "The branch is nonzero and changes over events; this does not establish that "
            "the next-event objective identifies a reproducible biological state."
        ),
    }
    (output / "CHECKPOINT_UTILIZATION_SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract held-out state and raw-residual diagnostics from frozen final checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_history_conditioned_field_fold_v0_4 import (
    _batch_history_states,
    _example_tensors,
    _history_half_life,
    _load_examples,
    _load_history_initialization,
)
from src.topic5_static_anchored_history_residual import (
    DualCandidateResidualHead,
    TimeAwareNonrecurrentResidual,
    unit_eps,
)


def _angle_degrees(left: torch.Tensor, right: torch.Tensor) -> float:
    cosine = torch.sum(left * right) / (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    ).clamp_min(1e-12)
    return float(torch.rad2deg(torch.arccos(cosine.clamp(-1, 1))).cpu())


@torch.no_grad()
def extract_unit(root: Path, subject: str, seed: int, device: torch.device) -> pd.DataFrame:
    directory = root / "per_subject" / f"seed_{seed}" / subject
    output_path = directory / "heldout_residual_diagnostics.csv.gz"
    if output_path.exists():
        return pd.read_csv(output_path)
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    examples, _ = _load_examples(root / "cache" / f"outer_{subject}")
    heldout = [example for example in examples if example.subject == subject]
    if not heldout:
        raise RuntimeError(f"no heldout histories for {subject}")
    state_dim = int(config["history_dim"])
    contact_dim = int(heldout[0].contact_embedding.shape[1])
    head_kwargs = {
        "initial_gain": config["initial_gain"],
        "epsilon": config["unit_epsilon"],
        "norm_threshold": config["residual_norm_threshold"],
    }
    frozen_history = _load_history_initialization(Path(checkpoint["history_initialization"]), device)
    m3_history = _load_history_initialization(Path(checkpoint["history_initialization"]), device)
    m3_history.load_state_dict(checkpoint["m3_history_state"], strict=True)
    m1_head = DualCandidateResidualHead(state_dim, contact_dim, **head_kwargs).to(device)
    m1_head.load_state_dict(checkpoint["m1_head_state"], strict=True)
    m3_head = DualCandidateResidualHead(state_dim, contact_dim, **head_kwargs).to(device)
    m3_head.load_state_dict(checkpoint["m3_head_state"], strict=True)
    m2 = TimeAwareNonrecurrentResidual(
        summary_dim=len(heldout[0].time_summary),
        state_dim=state_dim,
        contact_dim=contact_dim,
        **head_kwargs,
    ).to(device)
    m2.load_state_dict(checkpoint["m2_state"], strict=True)
    for model in (frozen_history, m3_history, m1_head, m2, m3_head):
        model.eval()
    frozen_states = _batch_history_states(
        frozen_history, heldout, device=device, chunk_events=config["chunk_events"]
    )
    m3_states = _batch_history_states(
        m3_history, heldout, device=device, chunk_events=config["chunk_events"]
    )
    rows = []
    for index, example in enumerate(heldout):
        tensor = _example_tensors(example, device)
        m2_output = m2(
            tensor["summary"], tensor["contact"], tensor["static_a"], tensor["static_b"]
        )
        outputs = {
            "M1_FROZEN_HISTORY_HEAD": (m1_head(
                frozen_states[index], tensor["contact"], tensor["static_a"], tensor["static_b"]
            ), frozen_states[index], _history_half_life(frozen_history)),
            "M2_TIME_AWARE_NONRECURRENT": (m2_output, m2_output["state"], None),
            "M3_JOINT_RNN": (m3_head(
                m3_states[index], tensor["contact"], tensor["static_a"], tensor["static_b"]
            ), m3_states[index], _history_half_life(m3_history)),
        }
        for model_name, (output, state, half_life) in outputs.items():
            candidate_a = output["candidate_a"]
            candidate_b = output["candidate_b"]
            base_a = unit_eps(tensor["static_a"])
            base_b = unit_eps(tensor["static_b"])
            common = {
                "subject": subject,
                "seizure_id": example.seizure_id,
                "seed": seed,
                "model": model_name,
                "n_events": len(example.event_time),
                "n_contacts": len(example.contact_names),
                "state_norm": float(torch.linalg.vector_norm(state).cpu()),
                "history_half_life_hours": half_life,
                "gain_a": float(output["gains"][0].cpu()),
                "gain_b": float(output["gains"][1].cpu()),
                "residual_norm_a": float(torch.linalg.vector_norm(output["residual_a"]).cpu()),
                "residual_norm_b": float(torch.linalg.vector_norm(output["residual_b"]).cpu()),
                "candidate_l2_from_static_a": float(torch.linalg.vector_norm(candidate_a - base_a).cpu()),
                "candidate_l2_from_static_b": float(torch.linalg.vector_norm(candidate_b - base_b).cpu()),
                "candidate_angle_from_static_a_degrees": _angle_degrees(candidate_a, base_a),
                "candidate_angle_from_static_b_degrees": _angle_degrees(candidate_b, base_b),
            }
            for contact_index, contact in enumerate(example.contact_names):
                rows.append(
                    {
                        **common,
                        "contact": str(contact),
                        "raw_residual_a": float(output["residual_a"][contact_index].cpu()),
                        "raw_residual_b": float(output["residual_b"][contact_index].cpu()),
                        "candidate_a": float(candidate_a[contact_index].cpu()),
                        "candidate_b": float(candidate_b[contact_index].cpu()),
                        "static_a": float(base_a[contact_index].cpu()),
                        "static_b": float(base_b[contact_index].cpu()),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False, compression="gzip")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "INPUT_MANIFEST.json").read_text())
    subjects = list(manifest["cohort"]["primary_subjects"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    frames = []
    for seed in (11, 29, 47):
        for subject in subjects:
            directory = root / "per_subject" / f"seed_{seed}" / subject
            if not (directory / "DONE.json").exists():
                raise RuntimeError(f"formal unit incomplete: seed={seed}, subject={subject}")
            frames.append(extract_unit(root, subject, seed, device))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / "history_conditioned_field_state_diagnostics.csv.gz", index=False, compression="gzip")
    summary = (
        combined.groupby(["subject", "seed", "model"], as_index=False)
        .agg(
            n_seizures=("seizure_id", "nunique"),
            state_norm_median=("state_norm", "median"),
            gain_a=("gain_a", "first"),
            gain_b=("gain_b", "first"),
            candidate_angle_a_median_degrees=("candidate_angle_from_static_a_degrees", "median"),
            candidate_angle_b_median_degrees=("candidate_angle_from_static_b_degrees", "median"),
        )
    )
    summary.to_csv(root / "history_conditioned_field_state_diagnostics_summary.csv", index=False)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "units": 45,
                "rows": len(combined),
                "models": sorted(combined.model.unique()),
                "output": str(root / "history_conditioned_field_state_diagnostics.csv.gz"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exact-model visibility audit for eta, beta and gamma.

The earlier synthetic generator never put gamma into its event-generating
equation.  This replacement uses the actual tissue RNN forward operator and
real held-out prefixes.  For a known parameter value it asks whether a grid of
the same model family recovers that value from future-contact probabilities.

This is an operator/readout sensitivity audit, not evidence that the optimiser
can recover the motif from finite noisy events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    build_motif_event_tensors,
)

SWEEPS = {
    "eta_raw": {
        "model": "DM1_FREE_AXIS",
        "truth": [0.05, 0.10, 0.20, 0.40],
        "grid": [0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80],
        "contexts": {"standard": {"theta": 0.55, "beta": 0.0, "gamma_raw": 0.0}},
    },
    "beta": {
        "model": "DM2_LOCAL_DIRECTIONAL",
        "truth": [0.15, 0.30, 0.60, 1.00],
        "grid": [-1.0, -0.60, -0.30, -0.15, 0.0, 0.15, 0.30, 0.60, 1.0],
        "contexts": {"standard": {"theta": 0.55, "eta_raw": 0.30, "gamma_raw": 0.0}},
    },
    "gamma_raw": {
        "model": "DM3_AXIS_FEEDFORWARD_TRANSIENT",
        "truth": [0.003, 0.01, 0.03, 0.10],
        "grid": [0.0, 0.003, 0.01, 0.03, 0.10, 0.30],
        "contexts": {
            "zero_parent": {"theta": 0.55, "eta_raw": 0.0, "beta": 0.0},
            "directional_parent": {"theta": 0.55, "eta_raw": 0.30, "beta": 0.60},
        },
    },
}


def load_model(path: Path, device: torch.device) -> MotifRNN:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MotifRNN(MotifConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def set_values(model: MotifRNN, values: dict[str, float]) -> None:
    with torch.no_grad():
        for name, value in values.items():
            getattr(model, name).fill_(float(value))


@torch.no_grad()
def future_probabilities(model: MotifRNN, tensors: dict[str, torch.Tensor],
                         indices: np.ndarray, device: torch.device,
                         batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    probabilities, availabilities = [], []
    for begin in range(0, len(indices), batch_size):
        chosen = torch.as_tensor(indices[begin:begin + batch_size])
        batch = {key: tensors[key][chosen].to(device)
                 for key in ("x", "recruited", "displacement", "available",
                             "valid", "is_last")}
        logits, _, _ = model(batch["x"], batch["recruited"], batch["displacement"])
        predict = batch["valid"] & ~batch["is_last"]
        probabilities.append(torch.sigmoid(logits[predict]).cpu().numpy())
        availabilities.append(batch["available"][predict].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(availabilities)


def bernoulli_kl(truth: np.ndarray, candidate: np.ndarray,
                 available: np.ndarray) -> float:
    p = np.clip(np.asarray(truth, float), 1e-7, 1 - 1e-7)
    q = np.clip(np.asarray(candidate, float), 1e-7, 1 - 1e-7)
    term = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    per_decision = (term * available).sum(axis=1) / np.maximum(available.sum(axis=1), 1)
    return float(np.mean(per_decision))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-events", type=int, default=2048)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = args.root / "repair_v0_2/operator_visibility"
    out.mkdir(parents=True, exist_ok=True)
    census = pd.read_csv(args.root / "GEOMETRY_ONLY_FIT_CENSUS.csv")

    rows = []
    unit_ids = args.subjects or sorted(census["subject"].astype(str))
    for unit_id in unit_ids:
        unit = load_frame_unit(args.root, args.frame, unit_id)
        tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm)
        unseen = unit.indices(-1)
        if unseen.size > args.max_events:
            seed = int.from_bytes(hashlib.sha256(unit_id.encode()).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            unseen = np.sort(rng.choice(unseen, args.max_events, replace=False))
        for parameter, contract in SWEEPS.items():
            checkpoint = (args.root / args.tag / args.frame / unit_id
                          / contract["model"] / "seed0/checkpoint.pt")
            if not checkpoint.exists():
                continue
            model = load_model(checkpoint, device)
            original = {name: float(getattr(model, name))
                        for name in ("theta", "eta_raw", "beta", "gamma_raw")}
            for context_label, context in contract["contexts"].items():
                for truth_value in contract["truth"]:
                    truth_values = {**context, parameter: truth_value}
                    set_values(model, truth_values)
                    truth_probability, available = future_probabilities(
                        model, tensors, unseen, device)
                    profile = []
                    for candidate_value in contract["grid"]:
                        candidate_values = {**context, parameter: candidate_value}
                        set_values(model, candidate_values)
                        probability, candidate_available = future_probabilities(
                            model, tensors, unseen, device)
                        if not np.array_equal(available, candidate_available):
                            raise RuntimeError("available-contact mask changed across candidates")
                        divergence = bernoulli_kl(
                            truth_probability, probability, available)
                        profile.append((float(candidate_value), divergence))
                        rows.append({
                            "subject": unit.subject, "unit_id": unit_id,
                            "parameter": parameter, "model_id": contract["model"],
                            "context": context_label,
                            "truth_value": float(truth_value),
                            "candidate_value": float(candidate_value),
                            "mean_bernoulli_kl_per_available_contact": divergence,
                            "n_decisions": int(available.shape[0]),
                        })
                    best = min(profile, key=lambda pair: pair[1])[0]
                    zero_kl = dict(profile).get(0.0, float("nan"))
                    print(f"[operator-visibility] {unit_id} {parameter}/{context_label} "
                          f"truth={truth_value:g} best={best:g} zeroKL={zero_kl:.3e}",
                          flush=True)
            set_values(model, original)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "OPERATOR_VISIBILITY_PROFILE.csv", index=False)
    minima = (frame.sort_values("mean_bernoulli_kl_per_available_contact")
              .groupby(["subject", "unit_id", "parameter", "context", "truth_value"],
                       as_index=False)
              .first())
    zero = (frame[frame["candidate_value"] == 0]
            [["subject", "unit_id", "parameter", "context", "truth_value",
              "mean_bernoulli_kl_per_available_contact"]]
            .rename(columns={"mean_bernoulli_kl_per_available_contact": "truth_vs_zero_kl"}))
    minima = minima.merge(
        zero, on=["subject", "unit_id", "parameter", "context", "truth_value"], how="left")
    minima["recovered_exact_grid_value"] = np.isclose(
        minima["candidate_value"], minima["truth_value"], atol=1e-12)
    minima.to_csv(out / "OPERATOR_VISIBILITY_PER_PATIENT.csv", index=False)
    summary = []
    for (parameter, context, truth), block in minima.groupby(
            ["parameter", "context", "truth_value"]):
        values = block["truth_vs_zero_kl"].to_numpy(float)
        summary.append({
            "parameter": parameter, "context": context, "truth_value": float(truth),
            "n_patients": int(len(block)),
            "exact_recovery": int(block["recovered_exact_grid_value"].sum()),
            "median_truth_vs_zero_kl": float(np.median(values)),
            "q25_truth_vs_zero_kl": float(np.quantile(values, 0.25)),
            "q75_truth_vs_zero_kl": float(np.quantile(values, 0.75)),
        })
    write = {
        "contract": "exact_topic5_motif_operator_visibility_v0_1",
        "interpretation": "forward-operator/readout visibility, not finite-sample fitting recovery",
        "all_three_motifs_in_generator": True,
        "max_events_per_patient": int(args.max_events),
        "n_patients": int(minima["subject"].nunique()),
        "summary": summary,
    }
    (out / "OPERATOR_VISIBILITY_SUMMARY.json").write_text(
        json.dumps(write, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

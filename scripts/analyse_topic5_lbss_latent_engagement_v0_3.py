#!/usr/bin/env python3
"""Target-free audit of whether zero-H tissue nodes participate in LBSS computation.

The full-tissue contract guarantees that many latent nodes are outside every
contact's direct virtual-SEEG footprint.  This script asks the stronger, purely
interictal question: are those nodes dynamically engaged, and does suppressing
their state harm held-out next-contact prediction?  It never reads an ictal
target and it does not refit or select a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(label: str, salt: int) -> int:
    value = hashlib.sha256(f"{label}|{salt}".encode()).digest()
    return int.from_bytes(value[:4], "little")


@torch.no_grad()
def evaluate_engagement(model, tensors, indices, zero_h, device, batch_size=128):
    """Compare intact dynamics with zero-H state clamped after every rank step."""
    from src.topic5_wiring_economy_rnn import cardinality_conditioned_nll

    model.eval()
    indices = np.asarray(indices, dtype=int)
    zero_h_t = torch.as_tensor(zero_h, dtype=torch.bool, device=device)
    keep = (~zero_h_t).float().repeat_interleave(model.state_dim).unsqueeze(0)
    intact_sum = 0.0
    clamped_sum = 0.0
    decisions = 0.0
    node_abs_sum = torch.zeros(model.n_nodes, dtype=torch.float64, device=device)
    state_steps = 0.0

    for begin in range(0, len(indices), int(batch_size)):
        chosen = torch.as_tensor(indices[begin:begin + int(batch_size)])
        batch = {key: value[chosen].to(device) for key, value in tensors.items()}
        b, steps, _ = batch["x"].shape
        h_intact = torch.zeros(b, model.n_nodes * model.state_dim, device=device)
        h_clamped = torch.zeros_like(h_intact)
        intact_logits = []
        clamped_logits = []
        for step in range(steps):
            h_intact = model._step(h_intact, batch["x"][:, step])
            h_clamped = model._step(h_clamped, batch["x"][:, step]) * keep
            intact_logits.append(model._readout(h_intact))
            clamped_logits.append(model._readout(h_clamped))
            if step >= 1:
                active = batch["valid"][:, step].double().view(-1, 1, 1)
                units = h_intact.reshape(b, model.n_nodes, model.state_dim).abs().double()
                node_abs_sum += (units * active).sum(dim=(0, 2)) / model.state_dim
                state_steps += float(active.sum())

        intact_logits = torch.stack(intact_logits, dim=1)
        clamped_logits = torch.stack(clamped_logits, dim=1)
        predict = batch["valid"] & ~batch["is_last"]
        weight = float(predict.float().sum())
        intact = cardinality_conditioned_nll(
            intact_logits, batch["target"], batch["available"], predict
        )
        clamped = cardinality_conditioned_nll(
            clamped_logits, batch["target"], batch["available"], predict
        )
        intact_sum += float(intact) * weight
        clamped_sum += float(clamped) * weight
        decisions += weight

    mean_abs = (node_abs_sum / max(state_steps, 1.0)).cpu().numpy()
    supported = ~np.asarray(zero_h, dtype=bool)
    zero = np.asarray(zero_h, dtype=bool)
    support_median = float(np.median(mean_abs[supported]))
    engagement_threshold = 0.10 * support_median
    return {
        "intact_contact_nll": intact_sum / max(decisions, 1.0),
        "zero_h_clamped_contact_nll": clamped_sum / max(decisions, 1.0),
        "zero_h_clamp_delta_nll": (clamped_sum - intact_sum) / max(decisions, 1.0),
        "n_continue_decisions": int(decisions),
        "zero_h_mean_abs_state": float(mean_abs[zero].mean()),
        "supported_mean_abs_state": float(mean_abs[supported].mean()),
        "zero_to_supported_state_ratio": float(
            mean_abs[zero].mean() / max(mean_abs[supported].mean(), 1e-12)
        ),
        "zero_h_engaged_fraction": float(np.mean(mean_abs[zero] > engagement_threshold)),
        "engagement_threshold": engagement_threshold,
    }


def patient_table(fit_table: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "zero_h_fraction",
        "intact_contact_nll",
        "zero_h_clamped_contact_nll",
        "zero_h_clamp_delta_nll",
        "zero_h_mean_abs_state",
        "supported_mean_abs_state",
        "zero_to_supported_state_ratio",
        "zero_h_engaged_fraction",
    ]
    return (
        fit_table.groupby(["subject", "arm"], as_index=False)[numeric]
        .mean()
        .sort_values(["subject", "arm"])
    )


def plot_summary(patient: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = patient[patient.arm == "L3_LOCAL_PLUS_LEARNED_LR"].copy()
    selected = selected.sort_values("zero_h_clamp_delta_nll")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axes[0]
    ax.axhline(0, color="#8f969b", lw=1)
    ax.scatter(np.arange(len(selected)), selected.zero_h_clamp_delta_nll,
               s=24, color="#b64b48", edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel("Patient")
    ax.set_ylabel(r"$\Delta$ NLL after clamping")
    ax.set_xticks([])

    ax = axes[1]
    ax.axhline(1, color="#8f969b", lw=1)
    ax.scatter(np.arange(len(selected)), selected.zero_to_supported_state_ratio,
               s=24, color="#397f96", edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel("Patient")
    ax.set_ylabel("Zero-H / supported state")
    ax.set_xticks([])
    for label, axis in zip("AB", axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes,
                  fontsize=13, fontweight="bold", va="bottom")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.xaxis.label.set_size(10)
        axis.yaxis.label.set_size(10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument(
        "--arms", nargs="+", default=["L3_LOCAL_PLUS_LEARNED_LR"],
        choices=ARMS,
        help="Primary audit defaults to the selected-shortcut model only.",
    )
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    formal = out_root / "FORMAL_TRAINING_COMPLETE.json"
    if not formal.exists():
        raise RuntimeError("formal training is not complete")

    # Use the exact model implementation that produced the checkpoints.
    snapshot = (
        args.model_snapshot.resolve() if args.model_snapshot is not None
        else out_root / "run_snapshot"
    )
    sys.path.insert(0, str(snapshot))
    from src.topic5_lbss_rnn_v0_2 import (  # noqa: E402
        LBSSConfig,
        LBSSModel,
        build_pool_contract,
        derange_training_validation_only,
    )
    from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402

    device = torch.device(args.device)
    rows = []
    for fit_dir in sorted((out_root / "cache").iterdir()):
        if not fit_dir.is_dir():
            continue
        fit_id = fit_dir.name
        plane = np.load(fit_dir / "plane.npz", allow_pickle=False)
        events = np.load(fit_dir / "events.npz", allow_pickle=False)
        provenance = json.loads((fit_dir / "provenance.json").read_text())
        keep_event = events["split"] >= 0
        observed_ranks = events["ranks"][keep_event].copy()
        split = events["split"][keep_event]
        test_idx = np.flatnonzero(split == 2)
        zero_h = plane["H"].sum(axis=0) <= 1e-12
        for arm in args.arms:
            ranks = observed_ranks.copy()
            if arm == "C_L3_ORDER_SHUFFLED":
                ranks, _ = derange_training_validation_only(
                    ranks, split, stable_seed(fit_id, 7717)
                )
            tensors = build_event_tensors(ranks)
            for seed in (0, 1, 2):
                unit = out_root / "per_fit" / fit_id / arm / f"seed{seed}"
                metrics = json.loads((unit / "metrics.json").read_text())
                config = metrics["config"]
                try:
                    pools = build_pool_contract(
                        plane["D_mm"], float(config["density"]),
                        float(config["added_fraction"]),
                        float(config.get("r_local_multiplier", 2.0)),
                    )
                except TypeError:
                    # The immutable v0.3 producer predates the explicit multiplier
                    # argument and is mathematically fixed at 2.0.
                    if float(config.get("r_local_multiplier", 2.0)) != 2.0:
                        raise
                    pools = build_pool_contract(
                        plane["D_mm"], float(config["density"]),
                        float(config["added_fraction"]),
                    )
                model = LBSSModel(LBSSConfig(
                    arm=arm,
                    n_contacts=int(provenance["n_contacts"]),
                    n_nodes=int(provenance["n_nodes"]),
                    observation_operator=plane["H"],
                    node_distance_mm=plane["D_mm"],
                    local_mask=pools.local_mask,
                    extra_local_pool=pools.extra_local_pool,
                    nonlocal_pool=pools.nonlocal_pool,
                    k_added=pools.k_added,
                    seed=seed,
                    state_dim=int(metrics["config"]["state_dim"]),
                )).to(device)
                state = torch.load(unit / "weights.pt", map_location=device, weights_only=True)
                model.load_state_dict(state)
                result = evaluate_engagement(model, tensors, test_idx, zero_h, device)
                rows.append({
                    "fit_id": fit_id,
                    "subject": provenance["subject"],
                    "scope": provenance["scope"],
                    "arm": arm,
                    "seed": seed,
                    "n_contacts": int(provenance["n_contacts"]),
                    "n_nodes": int(provenance["n_nodes"]),
                    "n_zero_h_nodes": int(zero_h.sum()),
                    "zero_h_fraction": float(zero_h.mean()),
                    **result,
                })

    output = out_root / "latent_engagement"
    output.mkdir(parents=True, exist_ok=True)
    unit_table = pd.DataFrame(rows)
    fit_table = (
        unit_table.groupby(["fit_id", "subject", "scope", "arm"], as_index=False)
        .mean(numeric_only=True)
    )
    patient = patient_table(fit_table)
    unit_table.to_csv(output / "latent_engagement_per_unit.csv", index=False)
    fit_table.to_csv(output / "latent_engagement_per_fit.csv", index=False)
    patient.to_csv(output / "latent_engagement_per_patient.csv", index=False)
    summary = {
        "contract": "topic5_lbss_zero_h_latent_engagement_v0_3",
        "n_units": int(len(unit_table)),
        "n_fits": int(fit_table.fit_id.nunique()),
        "n_patients": int(patient.subject.nunique()),
        "arms": list(args.arms),
        "model_snapshot": str(snapshot),
        "target_values_read": False,
        "interpretation": (
            "Diagnostic engagement audit only; zero-H clamp is not a matched lesion "
            "claim and does not alter model or field selection."
        ),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    for arm in args.arms:
        selected = patient[patient.arm == arm]
        summary[arm] = {
            "median_zero_h_clamp_delta_nll": float(selected.zero_h_clamp_delta_nll.median()),
            "positive_patients": int((selected.zero_h_clamp_delta_nll > 0).sum()),
            "median_zero_to_supported_state_ratio": float(
                selected.zero_to_supported_state_ratio.median()
            ),
            "median_zero_h_engaged_fraction": float(selected.zero_h_engaged_fraction.median()),
        }
    (output / "LATENT_ENGAGEMENT_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    plot_summary(patient, out_root / "figures" / "stage_d_zero_h_latent_engagement.png")
    marker = {
        "status": "PASS",
        "n_units": int(len(unit_table)),
        "n_fits": int(fit_table.fit_id.nunique()),
        "n_patients": int(patient.subject.nunique()),
        "arms": list(args.arms),
        "target_values_read": False,
        "summary_sha256": sha256_file(output / "LATENT_ENGAGEMENT_SUMMARY.json"),
    }
    (out_root / "LATENT_ENGAGEMENT_COMPLETE.json").write_text(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()

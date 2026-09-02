#!/usr/bin/env python3
"""Model-unseen evaluation for one Topic 5.2 dynamical motif unit.

Prediction on ``split == -1``, then closed-loop stochastic generation under the
frozen decoder: fixed 3- and 5-step horizons (direction and extent without the
termination head) and complete FULL_STOP events (length and terminal field).
Every model draws from the same uniform stream, so the paired comparisons are
common-random-number comparisons.
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

from src.topic5_dynamical_motif_analysis_v0_1 import (  # noqa: E402
    SUMMARY_KEYS,
    covariance_alignment,
    coverage,
    energy_score_batch,
    mode_posterior,
    observed_scale,
    select_reference_events,
    sequences_to_ranks,
    standardise,
)
from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    MotifConfig,
    MotifRNN,
    build_motif_event_tensors,
)
from src.topic5_dynamical_motif_rollout_v0_1 import (  # noqa: E402
    DecoderContract,
    SizeHead,
    stochastic_rollout,
    summarise_sequences,
)
from src.topic5_shared_propagation_field import conditional_k_subset_log_prob  # noqa: E402
from src.topic5_wiring_economy_rnn import next_rank_stop_loss  # noqa: E402

ROLLOUT_MODES = (("FIXED_H3", 3), ("FIXED_H5", 5), ("FULL_STOP", None))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable) + "\n")
    temporary.replace(path)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def load_unit_model(unit, unit_dir: Path, device: torch.device) -> tuple[MotifRNN, SizeHead, DecoderContract, dict]:
    payload = torch.load(unit_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = MotifConfig(**payload["config"])
    model = MotifRNN(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    decoder = torch.load(unit_dir / "decoder.pt", map_location="cpu", weights_only=False)
    head = SizeHead(unit.n_contacts).to(device)
    head.load_state_dict(decoder["size_head"])
    head.eval()
    contract = DecoderContract(**decoder["contract"])
    metrics = json.loads((unit_dir / "metrics.json").read_text())
    return model, head, contract, metrics


@torch.no_grad()
def prediction_metrics(model, tensors, indices, device, batch_size=1024) -> dict[str, float]:
    indices = np.asarray(indices, dtype=int)
    totals = {"next_bce": 0.0, "stop_bce": 0.0, "subset_nll": 0.0, "top1": 0.0}
    stop_total, stop_count, decisions = 0.0, 0.0, 0.0
    for begin in range(0, indices.size, batch_size):
        chosen = torch.as_tensor(indices[begin:begin + batch_size])
        batch = {key: tensors[key][chosen].to(device)
                 for key in ("x", "recruited", "displacement", "target", "available",
                             "valid", "is_last")}
        logits, stops, _ = model(batch["x"], batch["recruited"], batch["displacement"])
        _, next_bce, stop_bce = next_rank_stop_loss(
            logits, stops, batch["target"], batch["available"], batch["valid"], batch["is_last"])
        predict = batch["valid"] & ~batch["is_last"]
        flat_logits = logits[predict]
        flat_target = batch["target"][predict] > 0.5
        flat_available = batch["available"][predict]
        subset = -conditional_k_subset_log_prob(flat_logits, flat_target, flat_available)
        masked = logits.masked_fill(~batch["available"], -1e9)
        hit = ((batch["target"].gather(-1, masked.argmax(-1, keepdim=True)).squeeze(-1) > 0)
               & predict).float().sum()
        weight = float(predict.float().sum())
        totals["next_bce"] += float(next_bce) * weight
        totals["stop_bce"] += float(stop_bce) * weight
        totals["subset_nll"] += float(subset.sum())
        totals["top1"] += float(hit)
        decisions += weight
        valid = batch["valid"]
        stop_total += float((torch.nn.functional.binary_cross_entropy_with_logits(
            stops, batch["is_last"].float(), reduction="none") * valid.float()).sum())
        stop_count += float(valid.float().sum())
    result = {key: value / max(1.0, decisions) for key, value in totals.items()}
    result["stop_nll_per_step"] = stop_total / max(1.0, stop_count)
    result["n_continue_decisions"] = int(decisions)
    return result


def run_rollouts(model, head, contract, unit, tensors, indices, draws, device,
                 gate_rule, rng_prefix, chunk_rows=32768) -> dict:
    """Generate ``draws`` events per index for every rollout mode."""
    indices = np.asarray(indices, dtype=int)
    starts = tensors["x"][:, 0][torch.as_tensor(indices)]
    out: dict[str, dict[str, np.ndarray]] = {}
    for mode, horizon in ROLLOUT_MODES:
        pieces: list[dict[str, np.ndarray]] = []
        per_chunk = max(1, chunk_rows // max(1, draws))
        for begin in range(0, indices.size, per_chunk):
            block = starts[begin:begin + per_chunk]
            repeated = block.repeat_interleave(draws, dim=0)
            result = stochastic_rollout(
                model, head, contract, repeated, unit.contacts_xy_mm, device,
                mode="FIXED_H" if horizon else "FULL_STOP", horizon=horizon,
                gate_rule=gate_rule, rng_label=f"{rng_prefix}|{mode}|{begin}",
            )
            summary = summarise_sequences(
                result["sequence"], result["n_emitted"], unit.contacts_xy_mm, np.array([1.0, 0.0]))
            summary["ranks"] = sequences_to_ranks(result["sequence"], result["n_emitted"])
            pieces.append(summary)
        out[mode] = {key: np.concatenate([piece[key] for piece in pieces], axis=0)
                     for key in pieces[0]}
    return out


def observed_summary(unit, tensors, indices) -> dict[str, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    x = tensors["x"][torch.as_tensor(indices)].numpy()
    lengths = tensors["length"][torch.as_tensor(indices)].numpy()
    return summarise_sequences(x.astype(np.uint8), lengths - 1, unit.contacts_xy_mm,
                               np.array([1.0, 0.0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--reference-draws", type=int, default=128)
    parser.add_argument("--reference-target", type=int, default=24)
    parser.add_argument("--gate-rule", default="M2-2RANK")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    unit = load_frame_unit(args.out_root, args.frame, args.unit_id)
    unit_dir = (args.out_root / args.tag / args.frame / args.unit_id / args.model
                / f"seed{args.seed_index}")
    model, head, contract, train_metrics = load_unit_model(unit, unit_dir, device)
    tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm, gate_rule=args.gate_rule)

    unseen = unit.indices(-1)
    lengths = tensors["length"].numpy()
    reference = select_reference_events(
        f"{args.frame}|{args.unit_id}", unseen, lengths, unit.mode_label, args.reference_target)

    prediction = prediction_metrics(model, tensors, unseen, device)
    observed = observed_summary(unit, tensors, unseen)
    scale = observed_scale(observed)
    observed_matrix = standardise(observed, scale)
    modes = np.load(
        args.out_root / "frame_cache" / args.frame / args.unit_id / "train_only_modes.npz"
        if args.frame == "GEOMETRY_ONLY_PCA2"
        else args.out_root.parent / "topic5_multiscale_effective_scaffold_v0_5" / "cache"
        / args.unit_id / "train_only_modes.npz",
        allow_pickle=False)
    centers, temperature = np.asarray(modes["centers"]), float(modes["temperature"][0])
    observed_mode = np.asarray(unit.mode_posterior)[unseen]

    rng_prefix = f"{args.frame}|{args.unit_id}|seed{args.seed_index}"
    payload = {
        "contract": "topic5_dynamical_motif_unseen_v0_1",
        "frame": args.frame, "unit_id": args.unit_id, "subject": unit.subject,
        "model_id": args.model, "seed_index": args.seed_index,
        "n_model_unseen": int(unseen.size), "n_reference": int(reference.size),
        "draws": args.draws, "reference_draws": args.reference_draws,
        "gate_rule": args.gate_rule,
        "prediction": prediction,
        "decoder": contract.to_dict(),
        "observed_scale": scale,
        # Baseline arms are written by the low-cost fitter and carry a different
        # schema than a trained unit, so every training field is optional here.
        "training": {k: train_metrics.get(k) for k in
                     ("best_validation_score", "best_epoch", "n_epochs", "sigma_s_mm",
                      "selected", "selected_axis", "gain_ratio")},
        "numerical_audit": train_metrics.get("numerical_audit", {}),
        "rollout": {},
    }
    reference_rows = []
    for draw_count, index_set, label in ((args.draws, unseen, "all"),
                                         (args.reference_draws, reference, "reference")):
        generated = run_rollouts(model, head, contract, unit, tensors, index_set,
                                 draw_count, device, args.gate_rule,
                                 f"{rng_prefix}|{label}")
        truth = observed_summary(unit, tensors, index_set)
        truth_matrix = standardise(truth, scale)
        truth_mode = np.asarray(unit.mode_posterior)[index_set]
        for mode, _ in ROLLOUT_MODES:
            summary = generated[mode]
            matrix = standardise(summary, scale).reshape(len(index_set), draw_count, -1)
            energy = energy_score_batch(matrix, truth_matrix)
            field = summary["contact_field"].reshape(len(index_set), draw_count, -1)
            field_energy = energy_score_batch(field, truth["contact_field"])
            covered = coverage(matrix, truth_matrix)
            posterior = mode_posterior(summary["ranks"], centers, temperature)
            posterior = posterior.reshape(len(index_set), draw_count, -1).mean(axis=1)
            observed_class = truth_mode.argmax(axis=1)
            brier = float(np.mean((posterior[np.arange(len(index_set)), observed_class] - 1.0) ** 2))
            log_score = float(-np.mean(np.log(np.clip(
                posterior[np.arange(len(index_set)), observed_class], 1e-6, 1.0))))
            first_draw = matrix[:, 0, :]
            block = {
                "energy_score_median": float(np.median(energy)),
                "energy_score_mean": float(np.mean(energy)),
                "contact_field_energy_median": float(np.median(field_energy)),
                "coverage_mean": float(np.mean(covered)),
                "mode_brier": brier,
                "mode_log_score": log_score,
                "generated_length_mean": float(summary["n_rank"].mean()),
                "observed_length_mean": float(truth["n_rank"].mean()),
                "generated_l_axis_mean": float(summary["l_axis"].mean()),
                "observed_l_axis_mean": float(truth["l_axis"].mean()),
                "generated_l_orth_mean": float(summary["l_orth"].mean()),
                "observed_l_orth_mean": float(truth["l_orth"].mean()),
                "generated_n_contact_mean": float(summary["n_contact"].mean()),
                "observed_n_contact_mean": float(truth["n_contact"].mean()),
                "endpoint_error_r_last_mm": float(np.median(np.linalg.norm(
                    summary["r_last"].reshape(len(index_set), draw_count, 2).mean(axis=1)
                    - truth["r_last"], axis=1))),
                "endpoint_error_r_late_mm": float(np.median(np.linalg.norm(
                    summary["r_late"].reshape(len(index_set), draw_count, 2).mean(axis=1)
                    - truth["r_late"], axis=1))),
                "covariance": covariance_alignment(first_draw, truth_matrix),
                "monte_carlo_error": float(np.std(
                    [energy_score_batch(matrix[:, i::4, :], truth_matrix).mean()
                     for i in range(4)], ddof=1) / 2.0) if draw_count >= 8 else None,
            }
            payload["rollout"][f"{label}|{mode}"] = block
            if label == "reference":
                for position, event_index in enumerate(index_set):
                    reference_rows.append({
                        "frame": args.frame, "unit_id": args.unit_id, "subject": unit.subject,
                        "model_id": args.model, "seed_index": args.seed_index,
                        "rollout_mode": mode, "event_index": int(event_index),
                        "draws": draw_count,
                        "energy_score": float(energy[position]),
                        "contact_field_energy": float(field_energy[position]),
                        "coverage": float(covered[position]),
                        "observed_n_rank": int(truth["n_rank"][position]),
                        "observed_l_axis_mm": float(truth["l_axis"][position]),
                        "generated_n_rank_mean": float(
                            summary["n_rank"].reshape(len(index_set), draw_count)[position].mean()),
                        "generated_l_axis_mean_mm": float(
                            summary["l_axis"].reshape(len(index_set), draw_count)[position].mean()),
                        "mode_probability_observed_class": float(
                            posterior[position, observed_class[position]]),
                    })
    payload["reference_event_indices"] = reference.tolist()
    payload["seconds"] = time.time() - started
    write_json(unit_dir / "unseen_evaluation.json", payload)
    pd.DataFrame(reference_rows).to_csv(unit_dir / "reference_events.csv", index=False)
    print(json.dumps({"unit_id": args.unit_id, "model": args.model,
                      "seed": args.seed_index,
                      "subset_nll": prediction["subset_nll"],
                      "full_stop_energy": payload["rollout"]["all|FULL_STOP"]["energy_score_median"],
                      "seconds": payload["seconds"]}, default=_jsonable), flush=True)


if __name__ == "__main__":
    main()

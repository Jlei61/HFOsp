#!/usr/bin/env python3
"""Repair analyses for the Topic 5.2 dynamical-motif RNN.

This script separates three questions that the original closeout mixed:

1. can a no-recurrence model win after its parameter count is capped at DM0;
2. do motifs help on late, long-event or spatially distal transitions even if
   their all-transition average is zero;
3. does selecting checkpoints by contact prediction, instead of the joint
   contact-plus-STOP loss, change that answer.

No seizure target or TA/TB label is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    MAIN_MODELS,
    MotifConfig,
    MotifRNN,
    build_motif_event_tensors,
)
from src.topic5_shared_propagation_field import conditional_k_subset_log_prob  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def bootstrap_median_ci(values: np.ndarray, seed: int = 51) -> list[float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(values, (10000, values.size), replace=True), axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def signed_summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0]
    p = 1.0
    if nonzero.size:
        p = float(wilcoxon(nonzero, alternative="greater", zero_method="wilcox").pvalue)
    return {
        "n": int(values.size),
        "median": float(np.median(values)) if values.size else None,
        "ci95_median": bootstrap_median_ci(values),
        "positive": int(np.sum(values > 0)),
        "negative": int(np.sum(values < 0)),
        "ties": int(np.sum(values == 0)),
        "p_one_sided": p,
    }


def load_model(path: Path, device: torch.device) -> MotifRNN:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MotifRNN(MotifConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def decision_geometry(tensors: dict[str, torch.Tensor], indices: np.ndarray,
                      xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return event, step, distance and event length for continuation decisions."""
    target = tensors["target"][indices].numpy()
    centroid = tensors["centroid"][indices].numpy()
    valid = tensors["valid"][indices].numpy()
    is_last = tensors["is_last"][indices].numpy()
    lengths = tensors["length"][indices].numpy()
    event_rows, steps = np.where(valid & ~is_last)
    target_rows = target[event_rows, steps]
    target_centroid = target_rows @ np.asarray(xy, float)
    target_centroid /= np.maximum(target_rows.sum(axis=1, keepdims=True), 1.0)
    distance = np.linalg.norm(target_centroid - centroid[event_rows, steps], axis=1)
    return event_rows, steps, distance, lengths[event_rows]


@torch.no_grad()
def score_categories(model: MotifRNN, tensors: dict[str, torch.Tensor], indices: np.ndarray,
                     xy: np.ndarray, train_distance_q75: float,
                     train_length_q75: float, device: torch.device) -> dict[str, dict[str, float]]:
    indices = np.asarray(indices, int)
    event_rows, steps, distance, event_length = decision_geometry(tensors, indices, xy)
    records = []
    batch_size = 512
    for begin in range(0, indices.size, batch_size):
        chosen_np = indices[begin:begin + batch_size]
        chosen = torch.as_tensor(chosen_np)
        batch = {key: tensors[key][chosen].to(device)
                 for key in ("x", "recruited", "displacement", "target",
                             "available", "valid", "is_last")}
        logits, _, _ = model(batch["x"], batch["recruited"], batch["displacement"])
        predict = batch["valid"] & ~batch["is_last"]
        losses = -conditional_k_subset_log_prob(
            logits[predict], batch["target"][predict] > 0.5,
            batch["available"][predict])
        records.append(losses.detach().cpu().numpy())
    losses = np.concatenate(records) if records else np.empty(0)
    if losses.size != steps.size:
        raise RuntimeError(f"decision mismatch: loss={losses.size}, geometry={steps.size}")
    masks = {
        "all": np.ones(losses.size, dtype=bool),
        "early_first_two_predictions": steps < 2,
        "late_after_two_predictions": steps >= 2,
        "distal_train_q75": distance >= float(train_distance_q75),
        "long_event_train_q75": event_length >= float(train_length_q75),
    }
    return {
        name: {"contact_nll": float(np.mean(losses[mask])) if mask.any() else float("nan"),
               "n_decisions": int(mask.sum())}
        for name, mask in masks.items()
    }


@torch.no_grad()
def score_m0_ablation(model: MotifRNN, tensors: dict[str, torch.Tensor], indices: np.ndarray,
                      device: torch.device, mode: str) -> dict[str, float]:
    """Score the full 2x2 of history memory by spatial mixing."""
    allowed = (
        "history_with_spatial_mixing",
        "history_without_spatial_mixing",
        "current_rank_with_spatial_mixing",
        "current_rank_without_spatial_mixing",
    )
    if mode not in allowed:
        raise ValueError(mode)
    indices = np.asarray(indices, int)
    subset_total = stop_total = continue_count = valid_count = 0.0
    original_gain = float(model.log_g)
    use_history = mode.startswith("history_")
    use_spatial_mixing = mode.endswith("with_spatial_mixing")
    if not use_spatial_mixing:
        model.log_g.fill_(-30.0)
    for begin in range(0, indices.size, 512):
        chosen = torch.as_tensor(indices[begin:begin + 512])
        batch = {key: tensors[key][chosen].to(device)
                 for key in ("x", "recruited", "displacement", "target",
                             "available", "valid", "is_last")}
        if use_history:
            logits, stops, _ = model(
                batch["x"], batch["recruited"], batch["displacement"])
        else:
            count, steps, _ = batch["x"].shape
            terms = model.recurrent_terms()
            direction, _ = model.axis_unit()
            gate = model.direction_gate(batch["displacement"], direction)
            output, stop_output = [], []
            denom = max(1, model.n_contacts - 1)
            for step in range(steps):
                state = torch.zeros(count, model.n_nodes, device=device)
                state = model.step(
                    state, batch["x"][:, step], gate[:, step], terms)
                output.append(model.readout(state))
                phase = torch.full((count,), step / denom, device=device)
                stop_output.append(model.stop_logit(model.state_features(
                    state, phase, batch["recruited"][:, step].mean(-1))))
            logits, stops = torch.stack(output, 1), torch.stack(stop_output, 1)
        predict = batch["valid"] & ~batch["is_last"]
        subset = -conditional_k_subset_log_prob(
            logits[predict], batch["target"][predict] > 0.5,
            batch["available"][predict])
        subset_total += float(subset.sum())
        continue_count += float(predict.sum())
        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            stops, batch["is_last"].float(), reduction="none")
        stop_total += float((stop_loss * batch["valid"].float()).sum())
        valid_count += float(batch["valid"].sum())
    model.log_g.fill_(original_gain)
    return {
        "contact_nll": subset_total / max(continue_count, 1.0),
        "stop_bce": stop_total / max(valid_count, 1.0),
        "n_continue": int(continue_count), "n_valid_steps": int(valid_count),
    }


def collect_static(root: Path, frame: str, tag: str) -> pd.DataFrame:
    rows = []
    for path in sorted((root / tag / frame).glob("*/capacity_matched_static_seed*.json")):
        payload = json.loads(path.read_text())
        matched = payload.get("STATIC_READOUT_CAPACITY_MATCHED")
        if not matched:
            continue
        unit_id = str(payload["unit_id"])
        rows.append({
            "subject": str(payload["subject"]), "unit_id": unit_id,
            "seed_index": int(payload["seed_index"]),
            "matched_rank": int(matched["factor_rank"]),
            "static_parameter_count": int(matched["parameter_count"]),
            "dm0_parameter_count": int(matched["dm0_parameter_count"]),
            "static_contact_nll": float(matched["model_unseen"]["contact_nll"]),
            "static_stop_bce": float(matched["model_unseen"]["stop_bce"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--tags", nargs="+", default=["formal", "contact_selected"])
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = args.root / "repair_v0_2"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    census = pd.read_csv(args.root / "GEOMETRY_ONLY_FIT_CENSUS.csv")
    unit_ids = args.subjects or sorted(census["subject"].astype(str).tolist())
    for unit_id in unit_ids:
        unit = load_frame_unit(args.root, args.frame, unit_id)
        tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm)
        _, _, train_distance, train_length = decision_geometry(
            tensors, unit.indices(0), unit.contacts_xy_mm)
        q_distance = float(np.quantile(train_distance, 0.75))
        q_length = float(np.quantile(train_length, 0.75))
        for tag in args.tags:
            for model_id in MAIN_MODELS:
                for seed_index in range(3 if tag == "formal" else 1):
                    checkpoint = (args.root / tag / args.frame / unit_id / model_id
                                  / f"seed{seed_index}/checkpoint.pt")
                    if not checkpoint.exists():
                        continue
                    model = load_model(checkpoint, device)
                    scores = score_categories(
                        model, tensors, unit.indices(-1), unit.contacts_xy_mm,
                        q_distance, q_length, device)
                    for subset, record in scores.items():
                        rows.append({
                            "tag": tag, "selection_metric": (
                                "contact_nll" if tag == "contact_selected" else "joint"),
                            "subject": unit.subject, "unit_id": unit_id,
                            "model_id": model_id, "seed_index": seed_index,
                            "subset": subset, "train_distance_q75_mm": q_distance,
                            "train_length_q75": q_length, **record,
                        })
                    if tag == "formal" and model_id == "DM0_ISOTROPIC" and seed_index == 0:
                        for ablation in (
                            "history_with_spatial_mixing",
                            "history_without_spatial_mixing",
                            "current_rank_with_spatial_mixing",
                            "current_rank_without_spatial_mixing",
                        ):
                            ablation_rows.append({
                                "subject": unit.subject, "unit_id": unit_id,
                                "ablation": ablation,
                                **score_m0_ablation(
                                    model, tensors, unit.indices(-1), device, ablation),
                            })
        print(f"[repair-score] {unit_id}", flush=True)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "HARD_TRANSITION_METRICS_PER_SEED.csv", index=False)
    patient = (per_seed.groupby(
        ["tag", "selection_metric", "subject", "model_id", "subset"], as_index=False)
        .agg(contact_nll=("contact_nll", "mean"), n_decisions=("n_decisions", "sum")))
    patient.to_csv(out / "HARD_TRANSITION_METRICS_PER_PATIENT.csv", index=False)

    comparisons = []
    for (tag, metric, subset), block in patient.groupby(
            ["tag", "selection_metric", "subset"]):
        wide = block.pivot(index="subject", columns="model_id", values="contact_nll")
        for child, parent in zip(MAIN_MODELS[1:], MAIN_MODELS[:-1]):
            if child not in wide or parent not in wide:
                continue
            gain = (wide[parent] - wide[child]).dropna()
            comparisons.append({
                "tag": tag, "selection_metric": metric, "subset": subset,
                "child": child, "parent": parent, **signed_summary(gain.to_numpy()),
            })
    write_json(out / "HARD_TRANSITION_SUMMARY.json", comparisons)

    ablations = pd.DataFrame(ablation_rows)
    if not ablations.empty:
        ablations.to_csv(out / "M0_STATE_PATH_ABLATION_PER_PATIENT.csv", index=False)
        wide_contact = ablations.pivot(
            index="subject", columns="ablation", values="contact_nll")
        wide_stop = ablations.pivot(index="subject", columns="ablation", values="stop_bce")
        write_json(out / "M0_STATE_PATH_ABLATION_SUMMARY.json", {
            "spatial_mixing_given_history_contact_gain": signed_summary(
                (wide_contact["history_without_spatial_mixing"]
                 - wide_contact["history_with_spatial_mixing"]).to_numpy()),
            "spatial_mixing_given_history_stop_gain": signed_summary(
                (wide_stop["history_without_spatial_mixing"]
                 - wide_stop["history_with_spatial_mixing"]).to_numpy()),
            "state_memory_without_spatial_mixing_contact_gain": signed_summary(
                (wide_contact["current_rank_without_spatial_mixing"]
                 - wide_contact["history_without_spatial_mixing"]).to_numpy()),
            "state_memory_without_spatial_mixing_stop_gain": signed_summary(
                (wide_stop["current_rank_without_spatial_mixing"]
                 - wide_stop["history_without_spatial_mixing"]).to_numpy()),
            "spatial_mixing_without_history_contact_gain": signed_summary(
                (wide_contact["current_rank_without_spatial_mixing"]
                 - wide_contact["current_rank_with_spatial_mixing"]).to_numpy()),
            "spatial_mixing_without_history_stop_gain": signed_summary(
                (wide_stop["current_rank_without_spatial_mixing"]
                 - wide_stop["current_rank_with_spatial_mixing"]).to_numpy()),
            "state_memory_given_spatial_mixing_contact_gain": signed_summary(
                (wide_contact["current_rank_with_spatial_mixing"]
                 - wide_contact["history_with_spatial_mixing"]).to_numpy()),
            "state_memory_given_spatial_mixing_stop_gain": signed_summary(
                (wide_stop["current_rank_with_spatial_mixing"]
                 - wide_stop["history_with_spatial_mixing"]).to_numpy()),
        })

    static_seed = collect_static(args.root, args.frame, "formal")
    if not static_seed.empty:
        static_seed.to_csv(out / "CAPACITY_MATCHED_STATIC_PER_SEED.csv", index=False)
        static = (static_seed.groupby(["subject", "unit_id"], as_index=False)
                  .agg(matched_rank=("matched_rank", "first"),
                       static_parameter_count=("static_parameter_count", "max"),
                       dm0_parameter_count=("dm0_parameter_count", "min"),
                       static_contact_nll=("static_contact_nll", "mean"),
                       static_stop_bce=("static_stop_bce", "mean"),
                       n_static_seeds=("seed_index", "nunique")))
        dm0 = (patient[(patient["tag"] == "formal")
                       & (patient["model_id"] == "DM0_ISOTROPIC")
                       & (patient["subset"] == "all")]
               [["subject", "contact_nll"]]
               .rename(columns={"contact_nll": "dm0_contact_nll"}))
        static = static.merge(dm0, on="subject", how="left", validate="one_to_one")
        # STOP is not a hard-transition score; use the already frozen formal
        # model-unseen table rather than silently recomputing a different denominator.
        old = pd.read_csv(args.root / "STATIC_VS_RECURRENT_PER_PATIENT.csv")[
            ["subject", "rnn_stop_bce"]].rename(columns={"rnn_stop_bce": "dm0_stop_bce"})
        static = static.merge(old, on="subject", how="left", validate="one_to_one")
        static["contact_gain_rnn_minus_static"] = (
            static["static_contact_nll"] - static["dm0_contact_nll"])
        static["stop_gain_rnn_minus_static"] = (
            static["static_stop_bce"] - static["dm0_stop_bce"])
        static.to_csv(out / "CAPACITY_MATCHED_STATIC_PER_PATIENT.csv", index=False)
        write_json(out / "CAPACITY_MATCHED_STATIC_SUMMARY.json", {
            "contact_prediction_rnn_minus_static": signed_summary(
                static["contact_gain_rnn_minus_static"].to_numpy()),
            "stop_prediction_rnn_minus_static": signed_summary(
                static["stop_gain_rnn_minus_static"].to_numpy()),
            "capacity_contract_all_static_le_dm0": bool(
                (static["static_parameter_count"] <= static["dm0_parameter_count"]).all()),
            "all_patients_have_three_static_seeds": bool(
                (static["n_static_seeds"] == 3).all()),
        })
    else:
        static = pd.DataFrame()

    write_json(out / "REPAIR_ANALYSIS_STATUS.json", {
        "n_units": len(unit_ids), "tags_scored": args.tags,
        "n_model_rows": int(len(per_seed)),
        "static_capacity_matched_available": bool(not static.empty),
        "target_values_read": False,
    })


if __name__ == "__main__":
    main()

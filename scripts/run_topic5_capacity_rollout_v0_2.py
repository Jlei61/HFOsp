#!/usr/bin/env python3
"""Phase G9: frozen-decoder stochastic rollout (secondary evaluation).

The rollout closes the loop: the model samples a rank set, feeds it back into
its own state, and keeps going until its STOP head fires.  It is scored after
the held-out direct/autonomous results are already known, and an unstable
rollout does not change or excuse any of them.

Every arm shares one split-1 temperature and the same common random numbers, so
differences between arms cannot come from a different sampler.
"""
from __future__ import annotations

# One worker must not also fan out inside BLAS: these processes are run many at a
# time on a shared machine, and the default OpenMP thread count is the core count,
# which produced a load average of ~860 on an 80-core host before this was set.
import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, _os.environ.get("TOPIC5_TORCH_THREADS", "1"))

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    MotifConfig,
    OrderedMotif,
    evaluate,
    sample_subset,
)
from scripts.run_topic5_capacity_queue_v0_2 import PatientWorkspace  # noqa: E402
from scripts.run_topic5_capacity_usephase_v0_2 import load_model, median_angle_null  # noqa: E402

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
FRAME_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"
MAX_EVENTS = 400
N_DRAWS = 20
MAX_STEPS = 12
TEMPERATURE_GRID = tuple(float(v) for v in np.linspace(0.6, 1.8, 13))


def calibrate_temperature(workspace: PatientWorkspace) -> float:
    """One temperature per patient, fitted on the frozen unordered baseline.

    Because it is fitted on an arm-independent object, every structure inherits
    exactly the same sampler.
    """
    batch = workspace.tensors(3)
    rows = torch.as_tensor(np.flatnonzero(workspace.split_mask(3, 1)))
    piece = batch.index(rows)
    baseline = {key: value[rows] for key, value in workspace.baseline("U_FULL_SET", 3).items()}
    best, best_temperature = float("inf"), 1.0
    for temperature in TEMPERATURE_GRID:
        scaled = {"contact": baseline["contact"] / temperature,
                  "cardinality": baseline["cardinality"], "suffix": baseline["suffix"]}
        result = evaluate(None, scaled, piece, workspace.contact_xy)
        score = float(np.nanmean(result.per_horizon["total_nll"][:3]))
        if score < best:
            best, best_temperature = score, temperature
    return best_temperature


def fit_rollout_stop_head(model: OrderedMotif | None, workspace: PatientWorkspace,
                          epochs: int = 400, lr: float = 0.05) -> torch.nn.Module:
    """One-step-ahead stop model on the frozen state, fitted after the spatial freeze.

    Every cached sample has at least ``prefix_len + 1`` rank sets, so "does it end at
    the very next step" is identically false and carries no signal.  The head is
    therefore trained on "having already produced k more rank sets, does the event
    end now", pooled over k, with the state rolled forward k steps and the recruited
    fraction taken at that point.  It never enters the spatial checkpoint, and a stop
    result may not be used to rescue a spatial one.
    """
    batch = workspace.tensors(3)
    horizons = batch.n_horizons
    with torch.no_grad():
        base_state = (model.prefix_state(batch) if model is not None
                      else torch.zeros(batch.n_samples, 0))
        transition = (model.transition() if model is not None
                      and model.config.family == "AUTONOMOUS_SHARED_OPERATOR" else None)
        blocks, targets = [], []
        state = base_state
        for step in range(horizons):
            recruited = 1.0 - batch.target_available[:, step].float().mean(dim=1, keepdim=True)
            blocks.append(torch.cat([state, recruited], dim=1))
            targets.append((~batch.target_valid[:, step]).float().unsqueeze(1))
            if transition is not None:
                state = state @ transition.T
    features = torch.cat(blocks, dim=0)
    target = torch.cat(targets, dim=0)
    n_samples = batch.n_samples
    train_rows = np.flatnonzero(workspace.fraction_mask(3, 100))
    valid_rows = np.flatnonzero(workspace.split_mask(3, 1))
    train = torch.as_tensor(np.concatenate([train_rows + step * n_samples
                                            for step in range(horizons)]))
    valid = torch.as_tensor(np.concatenate([valid_rows + step * n_samples
                                            for step in range(horizons)]))
    head = torch.nn.Linear(features.shape[1], 1)
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(features[train]), target[train])
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            score = float(torch.nn.functional.binary_cross_entropy_with_logits(
                head(features[valid]), target[valid]))
        if score < best - 1e-9:
            best, best_state = score, {k: v.clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    head.validation_bce = best
    head.positive_rate = float(target[train].mean())
    return head


def rollout(model: OrderedMotif | None, workspace: PatientWorkspace, rows: torch.Tensor,
            temperature: float, seed: int, true_remaining: torch.Tensor,
            stop_head: torch.nn.Module) -> dict:
    batch = workspace.tensors(3).index(rows)
    baseline = {key: value[rows] for key, value in workspace.baseline("U_FULL_SET", 3).items()}
    kmax = workspace.samples(3).max_cardinality
    n_samples, n_contacts = batch.n_samples, batch.n_contacts
    generator = torch.Generator().manual_seed(seed)
    cumulative3 = torch.zeros(N_DRAWS, n_samples, n_contacts)
    cumulative5 = torch.zeros(N_DRAWS, n_samples, n_contacts)
    full = torch.zeros(N_DRAWS, n_samples, n_contacts)
    lengths = torch.zeros(N_DRAWS, n_samples)
    persistence_sum = torch.zeros(n_samples)
    persistence_count = torch.zeros(n_samples)
    with torch.no_grad():
        for draw in range(N_DRAWS):
            state = model.prefix_state(batch) if model is not None else None
            available = batch.suffix_eval_mask.clone()
            alive = torch.ones(n_samples, dtype=torch.bool)
            previous_centroid = None
            previous_step = None
            for step in range(MAX_STEPS):
                contact = baseline["contact"][:, 0].clone()
                cardinality = baseline["cardinality"][:, 0].clone()
                if model is not None:
                    if model.config.family == "AUTONOMOUS_SHARED_OPERATOR":
                        stepped = state @ model.transition().T
                        contact = contact + model._decode(stepped, None)
                        cardinality = cardinality + (
                            (stepped @ model.card_w).unsqueeze(1) * model.card_u.unsqueeze(0))
                    else:
                        contact = contact + model._decode(state, 0)
                        cardinality = cardinality + (
                            (state @ model.card_w[0]).unsqueeze(1) * model.card_u[0].unsqueeze(0))
                draw_n = torch.multinomial(torch.softmax(cardinality, dim=1), 1,
                                           generator=generator).squeeze(1) + 1
                draw_n = torch.minimum(draw_n, available.sum(dim=1)).clamp(min=0)
                picked = sample_subset(contact / temperature, available, draw_n, kmax, generator)
                picked = picked & alive.unsqueeze(1)
                if step < 3:
                    cumulative3[draw] += picked.float()
                if step < 5:
                    cumulative5[draw] += picked.float()
                full[draw] += picked.float()
                lengths[draw] += alive.float() * (picked.sum(dim=1) > 0).float()
                # direction persistence: do consecutive generated steps keep going
                # the same way?  Averaged over draws and events, it is compared with
                # the same quantity measured on the real continuation.
                weight = picked.float()
                total = weight.sum(dim=1, keepdim=True)
                centroid = (weight / total.clamp(min=1e-6)) @ workspace.contact_xy
                usable = (total.squeeze(1) > 0) & alive
                if previous_centroid is not None and previous_step is not None:
                    first = previous_centroid - previous_step
                    second = centroid - previous_centroid
                    norms = first.norm(dim=1) * second.norm(dim=1)
                    valid = usable & previous_usable & (norms > 1e-6)
                    cosine = (first * second).sum(dim=1) / norms.clamp(min=1e-6)
                    persistence_sum += (cosine * valid).clamp(-1.0, 1.0)
                    persistence_count += valid.float()
                previous_step = previous_centroid if previous_centroid is not None else centroid
                previous_centroid = centroid
                previous_usable = usable
                available = available & ~picked
                alive = alive & (picked.sum(dim=1) > 0) & (available.sum(dim=1) > 0)
                if model is not None:
                    contribution = model._encode(picked.float())
                    state = (state @ model.transition().T + contribution
                             if model.config.family != "ORDERLESS_BAG" else state + contribution)
                # the frozen stop head decides when the generated event ends; without
                # it the loop simply recruits every remaining contact and the length
                # is a property of the montage rather than of the model
                recruited = (batch.cumulative_set + (~available).float()).clamp(max=1.0)
                stop_features = torch.cat([
                    state if model is not None else torch.zeros(n_samples, 0),
                    recruited.sum(dim=1, keepdim=True) / float(n_contacts)], dim=1)
                stopping = torch.sigmoid(stop_head(stop_features)).squeeze(1)
                alive = alive & (torch.rand(n_samples, generator=generator) >= stopping)
                if not bool(alive.any()):
                    break
    mask = batch.suffix_eval_mask.float()
    truth5 = batch.suffix5_field
    truth_full = batch.full_suffix_field
    field3 = cumulative3.mean(dim=0).clamp(0, 1)
    field5 = cumulative5.mean(dim=0).clamp(0, 1)
    field_full = full.mean(dim=0).clamp(0, 1)
    weights = field_full * mask
    predicted = (weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)) @ workspace.contact_xy
    endpoint = torch.linalg.norm(predicted - batch.late_field_centroid, dim=1)
    # the generated length is compared with how many rank sets the event really had
    # left after the prefix, not with the five-horizon window, which would cap the
    # truth at five and manufacture a positive bias
    true_length = true_remaining.float()
    observed = observed_direction_persistence(batch, workspace.contact_xy)
    generated = float((persistence_sum / persistence_count.clamp(min=1)).mean())
    return {
        "direction_persistence_generated": generated,
        "direction_persistence_observed": observed,
        "direction_persistence_residual": observed - generated,
        "brier_three_step": float((((field3 - truth5) ** 2) * mask).sum() / mask.sum()),
        "brier_five_step": float((((field5 - truth5) ** 2) * mask).sum() / mask.sum()),
        "brier_full_suffix": float((((field_full - truth_full) ** 2) * mask).sum() / mask.sum()),
        "endpoint_distance_mm": float(endpoint.mean()),
        "generated_length_mean": float(lengths.mean()),
        "true_remaining_length_mean": float(true_length.mean()),
        "max_generated_steps": MAX_STEPS,
        "length_bias": float(lengths.mean() - true_length.mean()),
        "within_draw_field_variance": float(full.var(dim=0).mean()),
        "fields": field_full.numpy(),
    }


def observed_direction_persistence(batch, contact_xy: torch.Tensor) -> float:
    """The same quantity measured on the real continuation of the same events."""
    centroids, valid = [], []
    for horizon in range(batch.n_horizons):
        weight = batch.target_sets[:, horizon].float()
        total = weight.sum(dim=1, keepdim=True)
        centroids.append((weight / total.clamp(min=1e-6)) @ contact_xy)
        valid.append(batch.target_valid[:, horizon] & (total.squeeze(1) > 0))
    total_sum = torch.zeros(batch.n_samples)
    total_count = torch.zeros(batch.n_samples)
    for horizon in range(2, batch.n_horizons):
        first = centroids[horizon - 1] - centroids[horizon - 2]
        second = centroids[horizon] - centroids[horizon - 1]
        norms = first.norm(dim=1) * second.norm(dim=1)
        keep = valid[horizon] & valid[horizon - 1] & valid[horizon - 2] & (norms > 1e-6)
        cosine = ((first * second).sum(dim=1) / norms.clamp(min=1e-6)).clamp(-1.0, 1.0)
        total_sum += cosine * keep
        total_count += keep.float()
    usable = total_count > 0
    return float((total_sum[usable] / total_count[usable]).mean()) if bool(usable.any()) else float("nan")


def template_statistics(fields: np.ndarray, modes: np.ndarray) -> dict:
    out: dict = {}
    for label in sorted(set(int(v) for v in modes if v >= 0)):
        member = modes == label
        if member.sum() < 5:
            continue
        block = fields[member]
        out[f"template_{label}"] = {
            "n_events": int(member.sum()),
            "mean_field_norm": float(np.linalg.norm(block.mean(axis=0))),
            "within_template_covariance_trace": float(np.trace(np.cov(block, rowvar=False))),
        }
    if len(out) == 2:
        keys = sorted(out)
        first, second = fields[modes == int(keys[0].split("_")[1])].mean(axis=0), \
            fields[modes == int(keys[1].split("_")[1])].mean(axis=0)
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        out["between_template_cosine"] = float(first @ second / denominator) if denominator > 0 else float("nan")
    return out


def process_patient(payload: dict) -> dict:
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    patient = payload["patient"]
    try:
        workspace = PatientWorkspace(patient)
        samples = workspace.samples(3)
        test = np.flatnonzero(workspace.split_mask(3, 2))
        if test.size == 0:
            return {"patient": patient, "skipped": "no development-test events"}
        chosen = test if test.size <= MAX_EVENTS else np.random.default_rng(7).choice(
            test, MAX_EVENTS, replace=False)
        chosen = np.sort(chosen)
        rows = torch.as_tensor(chosen)
        temperature = calibrate_temperature(workspace)
        stop_heads = {label: fit_rollout_stop_head(
            None if unit is None else load_model(workspace, unit), workspace)
            for label, unit in payload["arms"].items()}

        remaining = torch.as_tensor(
            np.asarray(samples.n_rank_sets)[workspace.observed_rows(3)][chosen].astype(np.float32)
            - 3.0).clamp(min=0.0)
        modes = np.asarray(np.load(FRAME_ROOT / patient / "events.npz",
                                   allow_pickle=True)["prefix_mode"])
        event_ids = np.asarray(samples.event_index)[workspace.observed_rows(3)][chosen]
        record = {"patient": patient, "temperature": temperature,
                  "n_events_rolled_out": int(chosen.size), "n_draws": N_DRAWS,
                  "stop_head": {label: {"validation_bce": float(head.validation_bce),
                                        "training_stop_rate": float(head.positive_rate)}
                                for label, head in stop_heads.items()},
                  "arms": {}}
        for label, unit in payload["arms"].items():
            model = None if unit is None else load_model(workspace, unit)
            result = rollout(model, workspace, rows, temperature, seed=1234,
                             true_remaining=remaining, stop_head=stop_heads[label])
            fields = result.pop("fields")
            result["templates"] = template_statistics(fields, modes[event_ids])
            if label == "unordered_baseline":
                truth = np.asarray(workspace.tensors(3).index(rows).full_suffix_field)
                record["observed_templates"] = template_statistics(truth, modes[event_ids])
            record["arms"][label] = result
        return record
    except Exception:
        return {"patient": patient, "error": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    arguments = parser.parse_args()
    manifest = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    manifest = manifest[manifest["eligible"]]
    core = manifest[(manifest["block"] == "CORE1") & (manifest["rank"] == 4)
                    & (manifest["baseline_level"] == "U_FULL_SET")
                    & (manifest["family"] == "AUTONOMOUS_SHARED_OPERATOR")]

    jobs = []
    for patient in sorted(core["patient"].unique()):
        arms: dict[str, dict | None] = {"unordered_baseline": None}
        for label, structure in (("aligned", "H1_PATIENT_ALIGNED"), ("free", "H1_FREE_LOW_RANK")):
            frame = core[(core["patient"] == patient) & (core["structure"] == structure)]
            best, best_score = None, float("inf")
            for unit in frame.to_dict("records"):
                path = RESULT_ROOT / unit["output_dir"] / "metrics.json"
                if not path.exists():
                    continue
                score = json.loads(path.read_text())["training"]["best_valid_objective"]
                if score < best_score:
                    best, best_score = unit, score
            arms[label] = best
        arms["angle_null_median"] = median_angle_null(
            manifest, patient, "AUTONOMOUS_SHARED_OPERATOR", "U_FULL_SET")
        if arms["aligned"] is None:
            continue
        jobs.append({"patient": patient, "arms": {k: v for k, v in arms.items()
                                                  if v is not None or k == "unordered_baseline"}})

    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        records = list(pool.map(process_patient, jobs))

    rows = []
    for record in records:
        if "arms" not in record:
            continue
        for label, values in record["arms"].items():
            rows.append({"patient": record["patient"], "arm": label,
                         "temperature": record["temperature"],
                         "n_events_rolled_out": record["n_events_rolled_out"],
                         **{k: v for k, v in values.items() if k != "templates"},
                         "n_templates": len(
                             [k for k in values["templates"] if k.startswith("template_")]),
                         "between_template_cosine": values["templates"].get(
                             "between_template_cosine", np.nan)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULT_ROOT / "PER_PATIENT_STOCHASTIC_ROLLOUT.csv", index=False)
    summary = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_rollout",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "secondary evaluation; instability here never changes the direct or "
                "autonomous held-out results",
        "n_patients": int(table["patient"].nunique()) if len(table) else 0,
        "n_failed": sum(1 for record in records if "error" in record),
        "shared_sampler": "one split-1 temperature per patient fitted on the frozen unordered "
                          "baseline, common random numbers across arms",
        "per_arm_median": (table.groupby("arm")[
            ["brier_three_step", "brier_five_step", "brier_full_suffix",
             "endpoint_distance_mm", "length_bias", "between_template_cosine"]
        ].median().to_dict("index") if len(table) else {}),
        "per_patient": records,
    }
    (RESULT_ROOT / "STOCHASTIC_ROLLOUT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n")
    print(f"rollout: {summary['n_patients']} patients, {summary['n_failed']} failed")
    for arm, values in summary["per_arm_median"].items():
        print(f"  {arm:20s} brier5={values['brier_five_step']:.4f} "
              f"endpoint={values['endpoint_distance_mm']:.2f}mm "
              f"length_bias={values['length_bias']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

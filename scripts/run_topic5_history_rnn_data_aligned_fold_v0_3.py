#!/usr/bin/env python3
"""Data-aligned static-first field transfer for Topic 5 HistoryRNN.

The earlier direct-transfer analysis asked for one signed field predicted by a
shared cross-patient ridge.  The established paper result instead scores two
frozen interictal candidates (A/B) with a sign-free maxAB statistic.  This
runner keeps that existing output grammar and asks a deliberately simpler
question: can a low-capacity supervised readout of the frozen static fields,
optionally augmented by M1 or HistoryRNN fields, improve held-out maxAB field
concordance?

The recurrent representation remains target-blind.  Only the small shared
readout below sees early-ictal targets, and it is trained on outer-training
patients before scoring the held-out patient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


TIE_TOL = 1e-9
MODEL_COLUMNS = {
    "STATIC_LEARNED": (),
    "STATIC_M1": ("m1_part", "m1_rank"),
    "STATIC_RNN": ("m1_part", "m1_rank", "history_part", "history_rank"),
}


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _design(frame: pd.DataFrame, model: str, *, control: str = "true") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return matched A/B candidate designs with a shared parameter vector."""

    if model not in MODEL_COLUMNS:
        raise ValueError(f"unknown model {model}")
    common = [
        "scaffold_axis_magnitude",
        "scaffold_support_mean",
        "static",
    ]
    dynamic = list(MODEL_COLUMNS[model])
    work = frame.copy()
    if control == "order_shuffle":
        work["history_part"] = work["history_shuffle_part"]
        work["history_rank"] = work["history_shuffle_rank"]
    elif control == "zero_state":
        work["history_part"] = 0.0
        work["history_rank"] = 0.0
    elif control != "true":
        raise ValueError(f"unknown control {control}")

    names = [
        "branch_field",
        "branch_earliness",
        "axis_magnitude",
        "support_mean",
        "branch_support_difference",
        "static",
        *dynamic,
    ]
    shared = [work[column].to_numpy(float) for column in common]
    tail = [work[column].to_numpy(float) for column in dynamic]
    x_a = np.column_stack(
        [
            work.scaffold_field_a.to_numpy(float),
            work.scaffold_earliness_a.to_numpy(float),
            shared[0], shared[1],
            work.scaffold_support_difference.to_numpy(float),
            shared[2], *tail,
        ]
    )
    x_b = np.column_stack(
        [
            work.scaffold_field_b.to_numpy(float),
            work.scaffold_earliness_b.to_numpy(float),
            shared[0], shared[1],
            -work.scaffold_support_difference.to_numpy(float),
            shared[2], *tail,
        ]
    )
    if x_a.shape != x_b.shape or x_a.shape[1] != len(names):
        raise RuntimeError("dual-field design shape drift")
    return x_a, x_b, names


def _group_contract(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    seizure_key = frame.subject.astype(str) + "::" + frame.seizure_id.astype(str)
    seizure_codes, seizure_levels = pd.factorize(seizure_key, sort=True)
    patient_codes, patient_levels = pd.factorize(frame.subject.astype(str), sort=True)
    seizure_patient = np.empty(len(seizure_levels), dtype=np.int64)
    for code in range(len(seizure_levels)):
        members = np.unique(patient_codes[seizure_codes == code])
        if len(members) != 1:
            raise RuntimeError("a seizure maps to multiple patients")
        seizure_patient[code] = members[0]
    return seizure_codes.astype(np.int64), seizure_patient, len(seizure_levels), len(patient_levels)


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, dtype=values.dtype, device=values.device)
    return out.index_add(0, index, values)


def _patient_balanced_maxab(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    target: torch.Tensor,
    seizure_index: torch.Tensor,
    seizure_patient: torch.Tensor,
    n_seizures: int,
    n_patients: int,
) -> torch.Tensor:
    count = _scatter_sum(torch.ones_like(target), seizure_index, n_seizures).clamp_min(1.0)

    def centered(value: torch.Tensor) -> torch.Tensor:
        mean = _scatter_sum(value, seizure_index, n_seizures) / count
        return value - mean[seizure_index]

    y = centered(target)
    a = centered(pred_a)
    b = centered(pred_b)
    y_norm = torch.sqrt(_scatter_sum(y.square(), seizure_index, n_seizures).clamp_min(1e-12))

    def corr(value: torch.Tensor) -> torch.Tensor:
        numerator = _scatter_sum(value * y, seizure_index, n_seizures)
        value_norm = torch.sqrt(
            _scatter_sum(value.square(), seizure_index, n_seizures).clamp_min(1e-12)
        )
        return numerator / (value_norm * y_norm).clamp_min(1e-12)

    best = torch.maximum(corr(a).abs(), corr(b).abs())
    patient_sum = _scatter_sum(best, seizure_patient, n_patients)
    patient_count = _scatter_sum(torch.ones_like(best), seizure_patient, n_patients).clamp_min(1.0)
    return (patient_sum / patient_count).mean()


def _fit_readout(
    frame: pd.DataFrame,
    model: str,
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> dict:
    x_a, x_b, names = _design(frame, model)
    seizure_index, seizure_patient, n_seizures, n_patients = _group_contract(frame)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
    xa, xb = tensor(x_a), tensor(x_b)
    target = tensor(frame.target_z.to_numpy(float))
    sz_index = torch.as_tensor(seizure_index, dtype=torch.long)
    sz_patient = torch.as_tensor(seizure_patient, dtype=torch.long)
    initial = torch.zeros(x_a.shape[1], dtype=torch.float32)
    initial[0] = 1.0
    candidates = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(int(seed))
        weight = torch.nn.Parameter(initial + 0.01 * torch.randn(initial.shape, generator=generator))
        optimizer = torch.optim.Adam([weight], lr=float(learning_rate))
        best = None
        stale = 0
        for step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            score = _patient_balanced_maxab(
                xa @ weight, xb @ weight, target,
                sz_index, sz_patient, n_seizures, n_patients,
            )
            penalty = float(weight_decay) * torch.mean((weight - initial).square())
            loss = 1.0 - score + penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weight], 5.0)
            optimizer.step()
            record = (float(loss.detach()), float(score.detach()), step + 1, weight.detach().clone())
            if best is None or record[0] < best[0] - 1e-7:
                best = record
                stale = 0
            else:
                stale += 1
            if stale >= 120:
                break
        if best is None:
            raise RuntimeError("readout optimization produced no checkpoint")
        candidates.append(
            {
                "seed": int(seed),
                "loss": best[0],
                "train_patient_balanced_pearson_maxab": best[1],
                "step": int(best[2]),
                "weight": best[3].cpu().numpy(),
            }
        )
    selected = min(candidates, key=lambda row: row["loss"])
    return {
        "model": model,
        "feature_names": names,
        "selected_seed": selected["seed"],
        "selected_step": selected["step"],
        "train_patient_balanced_pearson_maxab": selected["train_patient_balanced_pearson_maxab"],
        "weight": selected["weight"],
        "seed_audit": [
            {key: value for key, value in row.items() if key != "weight"}
            for row in candidates
        ],
    }


def _spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) <= 0 or np.nanstd(b) <= 0:
        return float("nan")
    value = spearmanr(a, b).statistic
    return abs(float(value)) if np.isfinite(value) else float("nan")


def _score_candidates(frame: pd.DataFrame, pred_a: np.ndarray, pred_b: np.ndarray, model: str) -> pd.DataFrame:
    work = frame[["subject", "seizure_id", "seizure_idx", "contact", "target_z"]].copy()
    work["pred_a"] = pred_a
    work["pred_b"] = pred_b
    rows = []
    for (subject, seizure_id), group in work.groupby(["subject", "seizure_id"], sort=True):
        rho_a = _spearman_abs(group.pred_a.to_numpy(float), group.target_z.to_numpy(float))
        rho_b = _spearman_abs(group.pred_b.to_numpy(float), group.target_z.to_numpy(float))
        rows.append(
            {
                "subject": subject,
                "seizure_id": seizure_id,
                "model": model,
                "rho_a_abs": rho_a,
                "rho_b_abs": rho_b,
                "maxab_abs_rho": float(np.nanmax([rho_a, rho_b])),
                "winning_branch": "A" if rho_a >= rho_b else "B",
                "n_contacts": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _predict(frame: pd.DataFrame, fit: dict, *, control: str = "true") -> tuple[np.ndarray, np.ndarray]:
    xa, xb, names = _design(frame, fit["model"], control=control)
    if names != fit["feature_names"]:
        raise RuntimeError("readout feature order drift")
    weight = np.asarray(fit["weight"], dtype=float)
    return xa @ weight, xb @ weight


def _channel_null(
    frame: pd.DataFrame,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    seizure_groups = [
        np.asarray(index, dtype=np.int64)
        for _, index in frame.groupby("seizure_id", sort=True).groups.items()
    ]
    observed = {
        model: float(_score_candidates(frame, pair[0], pair[1], model).maxab_abs_rho.median())
        for model, pair in candidates.items()
    }
    draw_rows = []
    for draw in range(int(n_perm)):
        shuffled = frame.target_z.to_numpy(float).copy()
        for index in seizure_groups:
            shuffled[index] = shuffled[index][rng.permutation(len(index))]
        permuted = frame.copy()
        permuted["target_z"] = shuffled
        row = {"draw": draw}
        for model, pair in candidates.items():
            row[model] = float(
                _score_candidates(permuted, pair[0], pair[1], model).maxab_abs_rho.median()
            )
        draw_rows.append(row)
    draws = pd.DataFrame(draw_rows)
    rows = []
    for model, value in observed.items():
        null = draws[model].to_numpy(float)
        rows.append(
            {
                "subject": str(frame.subject.iloc[0]),
                "model": model,
                "observed_patient_median_maxab": value,
                "channel_null_median": float(np.nanmedian(null)),
                "channel_null_p95": float(np.nanpercentile(null, 95)),
                "margin_vs_null_median": float(value - np.nanmedian(null)),
                "permutation_p_one_sided": float((1 + np.sum(null >= value)) / (len(null) + 1)),
                "pass_channel_null_p95": bool(value > np.nanpercentile(null, 95)),
                "n_perm": int(n_perm),
            }
        )
    return pd.DataFrame(rows), draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--n-perm", type=int, default=5000)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.feature_table)
    required = {
        "subject", "seizure_id", "seizure_idx", "contact", "target_z",
        "scaffold_field_a", "scaffold_field_b", "scaffold_earliness_a",
        "scaffold_earliness_b", "scaffold_axis_magnitude",
        "scaffold_support_mean", "scaffold_support_difference", "static",
        "m1_part", "m1_rank", "history_part", "history_rank",
        "history_shuffle_part", "history_shuffle_rank",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise RuntimeError(f"feature table missing columns: {missing}")
    train = (
        frame.loc[frame.subject.astype(str) != args.heldout_subject]
        .copy().reset_index(drop=True)
    )
    test = (
        frame.loc[frame.subject.astype(str) == args.heldout_subject]
        .copy().reset_index(drop=True)
    )
    if train.empty or test.empty:
        raise RuntimeError("outer train/test split is empty")

    fits = {
        model: _fit_readout(
            train, model, seeds=tuple(args.seeds), steps=args.steps,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        )
        for model in MODEL_COLUMNS
    }
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "RAW_STATIC": (
            test.scaffold_field_a.to_numpy(float),
            test.scaffold_field_b.to_numpy(float),
        )
    }
    for model, fit in fits.items():
        candidates[model] = _predict(test, fit)
    candidates["STATIC_RNN_ORDER_SHUFFLE"] = _predict(
        test, fits["STATIC_RNN"], control="order_shuffle"
    )
    candidates["STATIC_RNN_ZERO_STATE"] = _predict(
        test, fits["STATIC_RNN"], control="zero_state"
    )

    seizure_metrics = pd.concat(
        [_score_candidates(test, pair[0], pair[1], model) for model, pair in candidates.items()],
        ignore_index=True,
    )
    null_metrics, null_draws = _channel_null(
        test, candidates, n_perm=args.n_perm,
        seed=_stable_seed(f"data-aligned-v0.3:{args.heldout_subject}"),
    )
    prediction_rows = []
    for model, (pred_a, pred_b) in candidates.items():
        block = test[["subject", "seizure_id", "seizure_idx", "contact", "target_z", "target_energy"]].copy()
        block["model"] = model
        block["prediction_a"] = pred_a
        block["prediction_b"] = pred_b
        prediction_rows.append(block)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    seizure_metrics.to_csv(output / "heldout_seizure_maxab_metrics.csv", index=False)
    null_metrics.to_csv(output / "heldout_channel_null_metrics.csv", index=False)
    null_draws.to_csv(output / "heldout_channel_null_draws.csv.gz", index=False, compression="gzip")
    predictions.to_csv(output / "heldout_dual_field_predictions.csv.gz", index=False, compression="gzip")
    serializable_fits = {
        model: {**fit, "weight": np.asarray(fit["weight"]).tolist()}
        for model, fit in fits.items()
    }
    done = {
        "status": "COMPLETE",
        "contract": "topic5_history_rnn_data_aligned_static_transfer_v0_3",
        "heldout_subject": args.heldout_subject,
        "n_train_patients": int(train.subject.nunique()),
        "n_test_seizures": int(test.seizure_id.nunique()),
        "n_test_contacts": int(test.contact.nunique()),
        "metric": "per-seizure max(|Spearman(candidate_A,target)|, |Spearman(candidate_B,target)|), patient median",
        "target": "clinical_onset_[0,10]s_1-150Hz_contact_energy",
        "selection_cost": "A/B and sign selection repeated inside every channel-label permutation",
        "fits": serializable_fits,
        "n_perm": int(args.n_perm),
    }
    (output / "DONE.json").write_text(json.dumps(done, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in done.items() if key != "fits"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

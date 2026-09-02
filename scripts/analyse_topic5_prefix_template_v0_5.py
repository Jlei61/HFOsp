#!/usr/bin/env python3
"""Strict train-only prefix-template control for v0.5 held-out suffixes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"


def template_log_prob(templates: np.ndarray, posterior: np.ndarray, available: np.ndarray,
                      temperature: float) -> np.ndarray:
    logits = -np.asarray(templates, float) / float(temperature)
    logits[:, ~available] = -np.inf
    mode_log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
    return logsumexp(np.log(np.maximum(posterior, 1e-12))[:, None] + mode_log_prob, axis=0)


def template_decisions(ranks: np.ndarray, split: np.ndarray, posterior: np.ndarray,
                       entropy: np.ndarray, templates: np.ndarray, temperature: float,
                       selected_split: int) -> pd.DataFrame:
    prepared = prepare_template_decisions(ranks, split, posterior, entropy, selected_split)
    if not len(prepared["event_index"]):
        return pd.DataFrame(columns=(
            "event_index", "rank_index", "template_nll", "prefix_entropy", "n_next",
        ))
    nll = evaluate_prepared_template_nll(prepared, templates, temperature)
    return pd.DataFrame({
        "event_index": prepared["event_index"],
        "rank_index": prepared["rank_index"],
        "template_nll": nll,
        "prefix_entropy": prepared["prefix_entropy"],
        "n_next": prepared["n_next"],
    })


def prepare_template_decisions(ranks: np.ndarray, split: np.ndarray,
                               posterior: np.ndarray, entropy: np.ndarray,
                               selected_split: int) -> dict[str, np.ndarray]:
    """Build each decision's candidate mask and target indices exactly once.

    Temperature calibration changes only the template softmax temperature.  The
    event prefix, available contacts and next-rank targets are invariant across
    the 61-point validation grid, so rebuilding them for every temperature is
    pure duplicated work.
    """
    event_indices: list[int] = []
    rank_indices: list[int] = []
    available_rows: list[np.ndarray] = []
    posterior_rows: list[np.ndarray] = []
    entropy_rows: list[float] = []
    target_parent: list[int] = []
    target_contact: list[int] = []
    n_next: list[int] = []
    for event_index in np.flatnonzero(split == selected_split):
        row = ranks[event_index]
        max_rank = int(row[row >= 0].max()) if np.any(row >= 0) else -1
        recruited: set[int] = set()
        for rank_index in range(max_rank):
            recruited.update(np.flatnonzero(row == rank_index).tolist())
            if rank_index < 2:  # three observed rank sets define the prefix
                continue
            target = np.flatnonzero(row == rank_index + 1)
            available = np.ones(ranks.shape[1], dtype=bool)
            available[list(recruited)] = False
            parent = len(event_indices)
            event_indices.append(int(event_index))
            rank_indices.append(int(rank_index))
            available_rows.append(available)
            # Preserve the original posterior dtype: the registered scalar
            # implementation applies np.log before NumPy promotes it in the
            # mixed float32/float64 expression.
            posterior_rows.append(np.asarray(posterior[event_index]))
            entropy_rows.append(float(entropy[event_index]))
            target_parent.extend([parent] * len(target))
            target_contact.extend(target.astype(int).tolist())
            n_next.append(int(len(target)))
    return {
        "event_index": np.asarray(event_indices, dtype=np.int64),
        "rank_index": np.asarray(rank_indices, dtype=np.int32),
        "available": np.asarray(available_rows, dtype=bool),
        "posterior": np.asarray(posterior_rows, dtype=np.float64),
        "prefix_entropy": np.asarray(entropy_rows, dtype=np.float64),
        "target_parent": np.asarray(target_parent, dtype=np.int64),
        "target_contact": np.asarray(target_contact, dtype=np.int32),
        "n_next": np.asarray(n_next, dtype=np.int32),
    }


def evaluate_prepared_template_nll(prepared: dict[str, np.ndarray],
                                   templates: np.ndarray,
                                   temperature: float) -> np.ndarray:
    """Return the original mean target-contact NLL for all prepared decisions."""
    logits = -np.asarray(templates, dtype=np.float64) / float(temperature)
    available = prepared["available"]
    # Shape: decisions x modes x contacts.  Per-fit arrays remain bounded by
    # held-out event count and are released before the next fit is processed.
    masked = np.where(available[:, None, :], logits[None, :, :], -np.inf)
    log_normalizer = logsumexp(masked, axis=2)
    parent = prepared["target_parent"]
    contact = prepared["target_contact"]
    log_posterior = np.log(np.maximum(prepared["posterior"], 1e-12))
    target_terms = (
        log_posterior[parent]
        + logits[:, contact].T
        - log_normalizer[parent]
    )
    target_log_prob = logsumexp(target_terms, axis=1)
    n_decisions = len(prepared["event_index"])
    summed = np.bincount(parent, weights=-target_log_prob, minlength=n_decisions)
    counts = np.bincount(parent, minlength=n_decisions)
    if np.any(counts == 0):
        raise RuntimeError("prepared template decision has no next-rank target")
    return summed / counts


def calibrate_temperature(ranks: np.ndarray, split: np.ndarray, posterior: np.ndarray,
                          entropy: np.ndarray, templates: np.ndarray) -> tuple[float, pd.DataFrame]:
    prepared = prepare_template_decisions(ranks, split, posterior, entropy, selected_split=1)
    if not len(prepared["event_index"]):
        raise RuntimeError("validation split has no suffix decisions for template calibration")
    values = []
    for temperature in np.geomspace(0.03, 30.0, 61):
        nll = evaluate_prepared_template_nll(prepared, templates, float(temperature))
        values.append({"temperature": temperature, "validation_nll": float(np.mean(nll))})
    curve = pd.DataFrame(values)
    best = float(curve.loc[curve.validation_nll.idxmin(), "temperature"])
    return best, curve


def l3_metrics_paths(out: Path, old: Path, fit_id: str, reused: set[str]) -> list[Path]:
    root = old / "per_fit" if fit_id in reused else out / "formal_units"
    return [root / fit_id / "L3_LOCAL_PLUS_LEARNED_LR" / f"seed{seed}" / "metrics.json"
            for seed in range(3)]


def fit_rows(out: Path, old: Path, fit_id: str, reused: set[str]) -> tuple[pd.DataFrame, dict]:
    cache = out / "cache" / fit_id
    events = np.load(cache / "events.npz", allow_pickle=False)
    modes = np.load(cache / "train_only_modes.npz", allow_pickle=False)
    ranks, split = events["ranks"], events["split"]
    posterior, entropy = events["prefix_posterior"], events["prefix_entropy"]
    temperature, curve = calibrate_temperature(
        ranks, split, posterior, entropy, modes["templates"]
    )
    template = template_decisions(ranks, split, posterior, entropy, modes["templates"],
                                  temperature, selected_split=2)
    kept_full = np.flatnonzero(split >= 0)
    seed_rows = []
    for seed, metrics_path in enumerate(l3_metrics_paths(out, old, fit_id, reused)):
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        decisions = pd.read_json(metrics_path.parent / "distance_decisions.json")
        decisions = decisions.loc[decisions.rank_index >= 2, ["event_index", "rank_index", "contact_nll"]]
        decisions["event_index"] = kept_full[decisions.event_index.to_numpy(dtype=int)]
        decisions["seed"] = seed
        seed_rows.append(decisions)
    rnn = pd.concat(seed_rows, ignore_index=True).groupby(
        ["event_index", "rank_index"], as_index=False
    ).contact_nll.median().rename(columns={"contact_nll": "rnn_nll"})
    joined = template.merge(rnn, on=["event_index", "rank_index"], validate="one_to_one")
    joined["rnn_minus_template_gain"] = joined.template_nll - joined.rnn_nll
    joined["fit_id"] = fit_id
    event = joined.groupby(["fit_id", "event_index"], as_index=False).agg(
        prefix_entropy=("prefix_entropy", "first"),
        template_nll=("template_nll", "mean"), rnn_nll=("rnn_nll", "mean"),
        rnn_minus_template_gain=("rnn_minus_template_gain", "mean"),
        n_suffix_decisions=("rank_index", "size"),
    )
    rho = spearmanr(event.prefix_entropy, event.rnn_minus_template_gain).statistic
    return event, {
        "fit_id": fit_id, "temperature": temperature,
        "validation_grid_min": float(curve.temperature.min()),
        "validation_grid_max": float(curve.temperature.max()),
        "n_test_events": len(event), "n_test_decisions": len(joined),
        "median_rnn_advantage": float(event.rnn_minus_template_gain.median()),
        "entropy_advantage_spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "target_values_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    if not (out / "STAGE_E_TRAINING_COMPLETE.json").exists():
        raise RuntimeError("formal training must finish before prefix-template comparison")
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"].astype(str))
    decisions, fit_summary = [], []
    for fit in census.itertuples():
        event, summary = fit_rows(out, old, fit.fit_id, reused)
        event["subject"] = fit.subject
        decisions.append(event); fit_summary.append(summary | {"subject": fit.subject, "scope": fit.scope})
    all_events = pd.concat(decisions, ignore_index=True)
    # Duplicate A/B fits are two geometry views of the same event. Average
    # them before the patient-level entropy relation is computed.
    patient_events = all_events.groupby(["subject", "event_index"], as_index=False).agg(
        prefix_entropy=("prefix_entropy", "first"),
        template_nll=("template_nll", "mean"), rnn_nll=("rnn_nll", "mean"),
        rnn_minus_template_gain=("rnn_minus_template_gain", "mean"),
        n_suffix_decisions=("n_suffix_decisions", "first"),
    )
    patient_rows = []
    for subject, group in patient_events.groupby("subject", sort=False):
        rho = spearmanr(group.prefix_entropy, group.rnn_minus_template_gain).statistic
        cutoff = group.prefix_entropy.quantile(0.75)
        ambiguous = group.loc[group.prefix_entropy >= cutoff]
        patient_rows.append({
            "subject": subject, "n_events": len(group),
            "median_rnn_advantage": group.rnn_minus_template_gain.median(),
            "entropy_advantage_spearman": rho,
            "ambiguous_q4_rnn_advantage": ambiguous.rnn_minus_template_gain.median(),
        })
    patients = pd.DataFrame(patient_rows)
    valid = patients.entropy_advantage_spearman.dropna().to_numpy()
    nonzero = valid[np.abs(valid) > 1e-9]
    p = 1.0 if len(nonzero) == 0 else float(wilcoxon(nonzero, alternative="greater").pvalue)
    all_events.to_csv(out / "PREFIX_TEMPLATE_PER_FIT_EVENT.csv", index=False)
    patient_events.to_csv(out / "PREFIX_TEMPLATE_PER_PATIENT_EVENT.csv", index=False)
    pd.DataFrame(fit_summary).to_csv(out / "PREFIX_TEMPLATE_PER_FIT_SUMMARY.csv", index=False)
    patients.to_csv(out / "PREFIX_TEMPLATE_PER_PATIENT_SUMMARY.csv", index=False)
    (out / "PREFIX_TEMPLATE_SUMMARY.json").write_text(json.dumps({
        "contract": "topic5_prefix_template_v0_5", "patients": len(patients),
        "median_rnn_advantage": float(patients.median_rnn_advantage.median()),
        "entropy_advantage_spearman_median": float(np.nanmedian(valid)),
        "entropy_advantage_wilcoxon_p_greater": p,
        "n_positive_entropy_slopes": int(np.sum(valid > 1e-9)),
        "n_negative_entropy_slopes": int(np.sum(valid < -1e-9)),
        "target_values_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

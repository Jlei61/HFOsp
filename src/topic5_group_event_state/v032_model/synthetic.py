"""Synthetic residual-positive and H-only-null assays on real patient scaffolds.

Both assays keep the real anchor grid, coverage, event times and event tokens
of a patient and only replace the 30-minute counts:

    positive:  log mu = log mu_H + beta * z,  z = standardised hidden marked
               leaky component (tau = 30 min) of a fixed random non-linear
               projection of the real tokens -- information H does not carry
    null:      log mu = log mu_H

with ``y ~ NB(mu, r)``.  The positive assay must recover the H+S increment
*and* correct-time specificity; the null must not produce a stable gain.
Thresholds are constants here and are echoed into every judgement JSON.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .data import SubjectBundle
from .evaluate import evaluate_arms
from .history_baseline import HistoryBaseline
from .paths import atomic_write_json
from .state import anchor_states, leaky_bank_trajectory
from .trainer import bundle_tensors, load_checkpoint_model, train_residual_model

POSITIVE_BETA = 0.35
SYNTH_DISPERSION_R = 5.0
HIDDEN_TAU_SECONDS = 1800.0
HIDDEN_WIDTH = 4
POSITIVE_MIN_RECOVERED_FRACTION = 2.0 / 3.0
NULL_MAX_FALSE_POSITIVE_REPLICATES = 1
NULL_MAX_MEDIAN_GAIN_NATS = 0.01
ASSAY_FORMAT = "group_event_state_v0_3_2_synthetic_assay"


@dataclass
class SyntheticTargets:
    counts: np.ndarray
    z: np.ndarray
    log_mu_true: np.ndarray
    beta: float
    dispersion_r: float
    generator_seed: int
    noise_seed: int
    horizon: float
    hidden_tau: float

    def as_dict(self) -> dict[str, Any]:
        return {"beta": self.beta, "dispersion_r": self.dispersion_r,
                "generator_seed": self.generator_seed, "noise_seed": self.noise_seed,
                "horizon_seconds": self.horizon, "hidden_tau_seconds": self.hidden_tau,
                "hidden_width": HIDDEN_WIDTH}


def make_synthetic_targets(
    bundle: SubjectBundle,
    *,
    horizon: float,
    beta: float,
    dispersion_r: float,
    generator_seed: int,
    noise_seed: int,
    hidden_tau: float = HIDDEN_TAU_SECONDS,
) -> SyntheticTargets:
    rng = np.random.default_rng(int(generator_seed))
    d = bundle.x_std.shape[1]
    weight = rng.normal(size=(d, HIDDEN_WIDTH)) / np.sqrt(d)
    bias = rng.normal(size=HIDDEN_WIDTH) * 0.5
    g = np.tanh(bundle.x_std.astype(np.float64) @ weight + bias)
    times = torch.from_numpy(bundle.event_times)
    seg = torch.from_numpy(bundle.event_segment)
    _pre, post = leaky_bank_trajectory(
        torch.from_numpy(g).float(), times, seg, torch.tensor([float(hidden_tau)]),
        chunk_seconds=3600.0,
    )
    s = anchor_states(
        post, times, torch.from_numpy(bundle.t_anchor), torch.from_numpy(bundle.last_event_pos),
        torch.full((HIDDEN_WIDTH,), float(hidden_tau)),
    ).numpy().astype(np.float64)
    raw = s @ rng.normal(size=HIDDEN_WIDTH)
    train = bundle.anchor_mask("state_train", horizon)
    z = (raw - raw[train].mean()) / max(float(raw[train].std()), 1e-9)
    log_mu_h = bundle.log_mu_h(horizon)
    log_mu_true = log_mu_h + float(beta) * z
    finite = np.isfinite(log_mu_true)
    scored = np.zeros(bundle.n_anchors, dtype=bool)
    for phase in ("calibration", "state_train", "dev_val", "dev_test"):
        scored |= bundle.anchor_mask(phase, horizon)
    if not finite[scored].all():
        raise ValueError("non-finite log mu_H on a scored anchor; cannot synthesise counts")
    mu = np.exp(np.where(finite, log_mu_true, 0.0))
    noise = np.random.default_rng(int(noise_seed))
    r = float(dispersion_r)
    counts = noise.negative_binomial(r, r / (r + mu)).astype(np.int64)
    counts[~finite] = 0
    return SyntheticTargets(counts=counts, z=z, log_mu_true=log_mu_true, beta=float(beta),
                            dispersion_r=r, generator_seed=int(generator_seed),
                            noise_seed=int(noise_seed), horizon=float(horizon),
                            hidden_tau=float(hidden_tau))


def apply_synthetic_targets(bundle: SubjectBundle, targets: SyntheticTargets) -> SubjectBundle:
    """Copy of the bundle whose counts at the assay horizon are the synthetic ones.

    The H-only NB dispersion for that horizon is dropped so the H arm is
    re-fitted on the synthetic TRAIN counts rather than inheriting a value
    estimated on the real ones.
    """

    h_i = bundle.horizon_index(targets.horizon)
    counts = bundle.counts.copy()
    counts[:, h_i] = targets.counts
    dispersion = dict(bundle.history.nb_log_dispersion)
    dispersion[int(targets.horizon)] = None
    history = HistoryBaseline(log_mu=dict(bundle.history.log_mu), nb_log_dispersion=dispersion,
                              source=bundle.history.source, meta=copy.deepcopy(bundle.history.meta))
    fingerprint = dict(bundle.fingerprint)
    fingerprint["synthetic"] = targets.as_dict()
    return replace(bundle, counts=counts, history=history, fingerprint=fingerprint)


def hidden_component_r2_against_h(
    bundle: SubjectBundle, targets: SyntheticTargets, *, phase: str = "state_train", ridge: float = 1e-2
) -> float:
    """In-sample linear R^2 of the hidden component from the explicit history features."""

    idx = np.flatnonzero(bundle.anchor_mask(phase, targets.horizon))
    z = targets.z[idx]
    if bundle.baseline_x is not None:
        keep = np.array(["seizure" not in n for n in bundle.baseline_names], dtype=bool)
        x = bundle.baseline_x[idx][:, keep]
    else:
        x = bundle.log_mu_h(targets.horizon)[idx][:, None]
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
    xs = np.column_stack([np.ones(idx.size), (x - mean) / scale])
    gram = xs.T @ xs + ridge * np.eye(xs.shape[1])
    beta = np.linalg.solve(gram, xs.T @ z)
    resid = z - xs @ beta
    total = float(((z - z.mean()) ** 2).sum())
    return float(1.0 - (resid ** 2).sum() / max(total, 1e-12))


def run_synthetic_assay(
    bundle: SubjectBundle,
    cfg: ModelConfig,
    *,
    kind: str,
    replicate: int,
    seed: int,
    device: torch.device,
    out_dir: Path,
    beta: float | None = None,
    dispersion_r: float = SYNTH_DISPERSION_R,
    generator_seed: int | None = None,
    noise_seed: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if kind not in ("positive", "null"):
        raise ValueError("kind must be 'positive' or 'null'")
    out_dir = Path(out_dir)
    assay_path = out_dir / "assay.json"
    horizon = float(cfg.horizon_seconds)
    beta_value = (POSITIVE_BETA if kind == "positive" else 0.0) if beta is None else float(beta)
    targets = make_synthetic_targets(
        bundle, horizon=horizon, beta=beta_value, dispersion_r=dispersion_r,
        generator_seed=1000 + int(replicate) if generator_seed is None else int(generator_seed),
        noise_seed=2000 + int(replicate) if noise_seed is None else int(noise_seed),
    )
    synthetic_bundle = apply_synthetic_targets(bundle, targets)
    header = {
        "format": ASSAY_FORMAT, "kind": kind, "replicate": int(replicate), "seed": int(seed),
        "subject": bundle.subject, "architecture": cfg.architecture,
        "config_hash": cfg.config_hash(), "h_source": bundle.history.source,
        "synthetic": {**targets.as_dict(),
                      "r2_hidden_vs_h_state_train": hidden_component_r2_against_h(bundle, targets)},
    }
    if assay_path.exists() and not overwrite:
        previous = __import__("json").loads(assay_path.read_text())
        if previous.get("config_hash") == header["config_hash"] and previous.get("status") == "complete" \
                and previous.get("synthetic") == header["synthetic"]:
            previous["skipped_existing"] = True
            return previous
    train = train_residual_model(synthetic_bundle, cfg, seed, device=device, out_dir=out_dir / "train",
                                 overwrite=overwrite)
    if train["status"] != "complete":
        report = {**header, "status": train["status"], "train": train}
        atomic_write_json(assay_path, report)
        return report
    model = load_checkpoint_model(out_dir / "train" / "checkpoint.pt",
                                  in_dim=synthetic_bundle.x_std.shape[1], device=device)
    tensors = bundle_tensors(synthetic_bundle, device)
    evaluation = {
        phase: evaluate_arms(model, synthetic_bundle, cfg, device=device, phase=phase, horizon=horizon,
                             log_r_h=float(train["log_r_h"]), tensors=tensors)
        for phase in ("dev_val", "dev_test")
    }
    report = {
        **header,
        "status": "complete",
        "train": {k: train[k] for k in ("selected_step", "selected_first_validation",
                                         "selected_at_budget_edge", "best_validation", "n_steps_run",
                                         "stopped_reason", "final_alpha", "final_log_r", "log_r_h",
                                         "n_train_anchors", "n_val_anchors", "elapsed_seconds")},
        "dev_val": evaluation["dev_val"],
        "dev_test": evaluation["dev_test"],
    }
    atomic_write_json(assay_path, report)
    return report


def judge_synthetic(assays: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    criteria = {
        "positive": {
            "rule": "dev_test CI95 lower bound of (H - H+S_correct) > 0 AND mean(shifted - correct) > 0 "
                    f"in at least {POSITIVE_MIN_RECOVERED_FRACTION:.3f} of replicates",
            "min_recovered_fraction": POSITIVE_MIN_RECOVERED_FRACTION,
        },
        "null": {
            "rule": f"at most {NULL_MAX_FALSE_POSITIVE_REPLICATES} replicate with dev_test CI95 lower bound of "
                    f"(H - H+S_correct) > 0 AND median gain < {NULL_MAX_MEDIAN_GAIN_NATS} nats/anchor",
            "max_false_positive_replicates": NULL_MAX_FALSE_POSITIVE_REPLICATES,
            "max_median_gain_nats": NULL_MAX_MEDIAN_GAIN_NATS,
        },
    }[kind]
    complete = [a for a in assays if a.get("status", "complete") == "complete"]
    rows = []
    for a in complete:
        c = a["dev_test"]["contrasts"]
        rows.append({
            "replicate": a.get("replicate"),
            "gain_mean": float(c["h_minus_correct"]["mean"]),
            "gain_ci_low": float(c["h_minus_correct"]["ci_low"]),
            "shifted_minus_correct_mean": float(c["shifted_minus_correct"]["mean"]),
        })
    out: dict[str, Any] = {"kind": kind, "n_replicates": len(complete), "criteria": criteria,
                           "per_replicate": rows, "n_incomplete": len(assays) - len(complete)}
    if not rows:
        out["pass"] = None
        out["reason"] = "no complete replicate"
        return out
    gains = np.array([r["gain_mean"] for r in rows])
    out["median_gain_nats"] = float(np.median(gains))
    if kind == "positive":
        recovered = [r["gain_ci_low"] > 0 and r["shifted_minus_correct_mean"] > 0 for r in rows]
        out["n_recovered_replicates"] = int(sum(recovered))
        out["pass"] = bool(sum(recovered) / len(rows) >= POSITIVE_MIN_RECOVERED_FRACTION - 1e-9)
    else:
        false_pos = [r["gain_ci_low"] > 0 for r in rows]
        out["n_false_positive_replicates"] = int(sum(false_pos))
        out["pass"] = bool(sum(false_pos) <= NULL_MAX_FALSE_POSITIVE_REPLICATES
                           and out["median_gain_nats"] < NULL_MAX_MEDIAN_GAIN_NATS)
    return out

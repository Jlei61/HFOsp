"""One-time v0.3.3 development evaluation after all selection is frozen.

This module is deliberately separate from training/search.  It accepts only a
hash-locked release containing corrected training cards and immutable learned
and random-reservoir checkpoints.  Development rows are opened once for
scoring and cannot feed any optimizer or checkpoint selector.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.topic5_group_event_state.v032_model.evaluate import block_bootstrap_mean_ci
from src.topic5_group_event_state.v032_model.shift import block_circular_donor
from src.topic5_group_event_state.v033_evaluator import canonical as C

from .data import DataView
from .human_view import DEVELOPMENT_PHASE, load_materialized_human_r0_view
from .objective import ResidualCountTrainable
from .paths import atomic_write_json, atomic_write_npz, file_hash
from .trainer import load_trained


RELEASE_FORMAT = "group_event_state_v0_3_3_one_time_development_release"
RELEASE_STATUS = "ACTIVE_ONE_TIME_DEVELOPMENT_EVALUATION"
BOOT = dict(block_len=6, n_boot=2000, seed=0)


def validate_development_release(
    release: Mapping[str, Any],
    *,
    request_id: str,
    request_path: Path,
    card_path: Path,
) -> Mapping[str, Any]:
    if release.get("format") != RELEASE_FORMAT or release.get("status") != RELEASE_STATUS:
        raise PermissionError("one-time development release is absent or inactive")
    if release.get("sealed") is not False or release.get("development_only") is not True:
        raise PermissionError("development release must be unsealed and development-only")
    if release.get("selection_feedback_forbidden") is not True \
            or release.get("retraining_after_open_forbidden") is not True:
        raise PermissionError("development release does not forbid selection/retraining feedback")
    entry = dict((release.get("requests") or {}).get(request_id) or {})
    if not entry:
        raise PermissionError(f"request {request_id!r} is not named in the development release")
    identities = (
        (request_path, entry.get("request_sha256"), "request"),
        (card_path, entry.get("corrected_card_sha256"), "corrected card"),
    )
    for path, expected, label in identities:
        if not Path(path).is_file() or file_hash(Path(path)) != str(expected):
            raise ValueError(f"released {label} is missing or changed")
    if Path(str(entry.get("request_path", ""))).resolve() != request_path.resolve() \
            or Path(str(entry.get("corrected_card_path", ""))).resolve() != card_path.resolve():
        raise ValueError("released request/card paths differ from the supplied files")
    checkpoints = list(entry.get("checkpoints") or [])
    if len(checkpoints) < 2:
        raise ValueError("development release needs at least two frozen seeds")
    seeds: set[int] = set()
    for row in checkpoints:
        seed = int(row["seed"])
        if seed in seeds:
            raise ValueError("development release repeats a seed")
        seeds.add(seed)
        for arm in ("learned", "random_reservoir"):
            path = Path(str(row[f"{arm}_checkpoint_path"]))
            if not path.is_file() or file_hash(path) != str(row[f"{arm}_checkpoint_sha256"]):
                raise ValueError(f"released {arm} checkpoint for seed {seed} is missing or changed")
    return entry


def load_development_view(request: Mapping[str, Any], *, scaling: str) -> tuple[DataView, dict[str, Any]]:
    """Open DEVELOPMENT_EVALUATION rows only after the external release check."""

    bins = tuple(tuple(float(value) for value in pair)
                 for pair in request["scientific_target"]["bins_seconds"])
    training_view, meta = load_materialized_human_r0_view(request, bins=bins, scaling=scaling)
    artifact_path = Path(str(request["input_view"]["artifact_path"]))
    with np.load(artifact_path, allow_pickle=False) as stored:
        phase = np.asarray(stored["phase"]).astype(str)
        eligible = np.asarray(stored["eligible_by_horizon"], dtype=bool)
        counts_all = np.asarray(stored["target_counts"], dtype=np.int64)
        log_mu_all = np.asarray(stored["log_mu_h_mark"], dtype=np.float64)
    development = np.flatnonzero((phase == DEVELOPMENT_PHASE) & eligible.all(axis=1))
    if development.size == 0:
        raise ValueError("released artifact has no fully eligible development-evaluation anchors")
    counts = np.full_like(counts_all, -1)
    log_mu = np.full_like(log_mu_all, np.nan)
    counts[development] = counts_all[development]
    log_mu[development] = log_mu_all[development]
    view = replace(
        training_view,
        phase_index={"development": development},
        counts=counts,
        log_mu_h=log_mu,
        h_meta={**training_view.h_meta, "phase_contract": {
            "evaluation": DEVELOPMENT_PHASE,
            "opened_once_for_scoring": True,
            "selection_or_fitting": False,
        }},
    )
    return view, {**meta, "development_evaluation_exposed": True,
                  "n_development_anchors": int(development.size)}


def _ci(values: np.ndarray, segments: np.ndarray) -> dict[str, Any]:
    out = block_bootstrap_mean_ci(np.asarray(values), np.asarray(segments), **BOOT)
    return {key: (float(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in out.items()}


def merge_development_scores(
    *,
    nll_h: np.ndarray,
    nll_learned_by_seed: np.ndarray,
    nll_shifted_by_seed: np.ndarray,
    nll_random_by_seed: np.ndarray,
    shift_valid: np.ndarray,
    segments: np.ndarray,
) -> dict[str, Any]:
    """Seed median at each anchor, then patient-time-block uncertainty."""

    h = np.asarray(nll_h, dtype=np.float64)
    learned = np.asarray(nll_learned_by_seed, dtype=np.float64)
    shifted = np.asarray(nll_shifted_by_seed, dtype=np.float64)
    random = np.asarray(nll_random_by_seed, dtype=np.float64)
    ok = np.asarray(shift_valid, dtype=bool)
    seg = np.asarray(segments)
    if learned.ndim != 2 or learned.shape[0] < 2 \
            or shifted.shape != learned.shape or random.shape != learned.shape:
        raise ValueError("development scores must be aligned seed x anchor arrays with at least two seeds")
    if h.shape != learned.shape[1:] or ok.shape != h.shape or seg.shape != h.shape:
        raise ValueError("development baseline/mask/segments do not align to anchors")
    learned_median = np.median(learned, axis=0)
    random_median = np.median(random, axis=0)
    shifted_median = np.full(h.shape, np.nan)
    if ok.any():
        shifted_median[ok] = np.median(shifted[:, ok], axis=0)
    return {
        "seed_merge_rule": "median per anchor across frozen seeds, then within-target-segment moving-block bootstrap",
        "n_seeds": int(learned.shape[0]),
        "H_minus_learned": _ci(h - learned_median, seg),
        "shifted_minus_correct": _ci(shifted_median - learned_median, seg),
        "random_minus_learned": _ci(random_median - learned_median, seg),
        "arrays": {
            "nll_h": h,
            "nll_learned_by_seed": learned,
            "nll_shifted_by_seed": shifted,
            "nll_random_by_seed": random,
            "nll_learned_seed_median": learned_median,
            "nll_shifted_seed_median": shifted_median,
            "nll_random_seed_median": random_median,
            "shift_valid": ok.astype(np.uint8),
            "bootstrap_segment": seg,
        },
    }


def evaluate_released_request(
    *,
    request_id: str,
    request_path: Path,
    card_path: Path,
    release_path: Path,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Score all frozen seeds once; no fit, selection or checkpoint mutation."""

    request = json.loads(Path(request_path).read_text())
    card = json.loads(Path(card_path).read_text())
    release = json.loads(Path(release_path).read_text())
    entry = validate_development_release(
        release, request_id=request_id, request_path=Path(request_path), card_path=Path(card_path)
    )
    view, view_meta = load_development_view(request, scaling=str(card["recipe"]["scaling"]))
    trainable = ResidualCountTrainable()
    idx = view.phase_index["development"]
    donor = block_circular_donor(
        view.t_anchor, view.anchor_segment, idx, horizon=view.horizon, fraction=0.5
    )
    shift_valid = donor >= 0
    learned_rows: list[np.ndarray] = []
    shifted_rows: list[np.ndarray] = []
    random_rows: list[np.ndarray] = []
    checkpoint_rows: list[dict[str, Any]] = []
    h_reference: np.ndarray | None = None
    for row in entry["checkpoints"]:
        seed = int(row["seed"])
        learned_model = load_trained(Path(row["learned_checkpoint_path"]).parent, trainable, view, device)
        random_model = load_trained(Path(row["random_reservoir_checkpoint_path"]).parent, trainable, view, device)
        learned_terms = trainable.loss_terms(
            learned_model, view, "development", device=device, differentiable_statistics=False,
            sampling="anchor_balanced", lookback_seconds=float(max(learned_model.arch.taus_seconds)),
        )
        random_terms = trainable.loss_terms(
            random_model, view, "development", device=device, differentiable_statistics=False,
            sampling="anchor_balanced", lookback_seconds=float(max(random_model.arch.taus_seconds)),
        )
        shifted_state = learned_terms.state_raw.clone()
        if shift_valid.any():
            shifted_state[torch.from_numpy(np.flatnonzero(shift_valid)).to(device)] = learned_terms.state_raw[
                torch.from_numpy(donor[shift_valid]).to(device)
            ]
        shifted_terms = trainable.loss_terms(
            learned_model, view, "development", device=device, differentiable_statistics=False,
            sampling="anchor_balanced", lookback_seconds=float(max(learned_model.arch.taus_seconds)),
            state_override=shifted_state,
        )

        per_bin_h, per_bin_learned, per_bin_shifted, per_bin_random = [], [], [], []
        for column in range(view.n_bins):
            table = C.build_per_anchor_table(
                subject=view.subject, seed=seed,
                checkpoint_hash=row["learned_checkpoint_sha256"], split=DEVELOPMENT_PHASE,
                anchor_time=view.t_anchor[idx], target=view.counts[idx, column],
                prediction_H=view.log_mu_h[idx, column],
                prediction_H_plus_state=learned_terms.log_mu[:, column].detach().cpu().numpy(),
                dispersion=float(view.log_r_h[column]), mask=np.ones(idx.size, dtype=bool), weight=None,
                eligibility="fully_eligible_30min", evidence_label=card["evidence_label"],
                dispersion_rule="shared",
                extra_arms={
                    "H_plus_shifted_state": shifted_terms.log_mu[:, column].detach().cpu().numpy(),
                    "H_plus_random_reservoir": random_terms.log_mu[:, column].detach().cpu().numpy(),
                },
            )
            per_bin_h.append(np.asarray(table["per_anchor_NLL_H"], dtype=np.float64))
            per_bin_learned.append(np.asarray(table["per_anchor_NLL_H_plus_state"], dtype=np.float64))
            per_bin_shifted.append(np.asarray(table["per_anchor_NLL_H_plus_shifted_state"], dtype=np.float64))
            per_bin_random.append(np.asarray(table["per_anchor_NLL_H_plus_random_reservoir"], dtype=np.float64))
        h = np.sum(per_bin_h, axis=0)
        learned = np.sum(per_bin_learned, axis=0)
        shifted = np.sum(per_bin_shifted, axis=0)
        random = np.sum(per_bin_random, axis=0)
        if np.max(np.abs(learned - learned_terms.nll.detach().cpu().numpy())) > C.TOLERANCE_NATS \
                or np.max(np.abs(random - random_terms.nll.detach().cpu().numpy())) > C.TOLERANCE_NATS:
            raise C.EvaluatorDisagreement("canonical evaluator and model loss disagree on development rows")
        if h_reference is None:
            h_reference = h
        elif not np.allclose(h, h_reference, atol=C.TOLERANCE_NATS, rtol=0):
            raise C.EvaluatorDisagreement("H score changed across optimization seeds")
        learned_rows.append(learned)
        shifted_rows.append(shifted)
        random_rows.append(random)
        checkpoint_rows.append({
            "seed": seed,
            "learned_checkpoint_sha256": row["learned_checkpoint_sha256"],
            "random_reservoir_checkpoint_sha256": row["random_reservoir_checkpoint_sha256"],
        })

    assert h_reference is not None
    merged = merge_development_scores(
        nll_h=h_reference,
        nll_learned_by_seed=np.asarray(learned_rows),
        nll_shifted_by_seed=np.asarray(shifted_rows),
        nll_random_by_seed=np.asarray(random_rows),
        shift_valid=shift_valid,
        segments=view.bootstrap_segment(idx),
    )
    arrays = merged.pop("arrays")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "per_anchor_scores.npz"
    atomic_write_npz(arrays_path, {
        **arrays,
        "anchor_index": idx,
        "anchor_time": view.t_anchor[idx],
        "target_counts": view.counts[idx],
    })
    report = {
        "format": "group_event_state_v0_3_3_one_time_development_result",
        "request_id": request_id,
        "subject": view.subject,
        "evidence_label_at_freeze": card["evidence_label"],
        "training_adequacy_is_not_a_scientific_result": True,
        "card_path": str(card_path), "card_sha256": file_hash(Path(card_path)),
        "release_path": str(release_path), "release_sha256": file_hash(Path(release_path)),
        "canonical_evaluator_sha256": request["input_view"]["evaluator_contract_sha256"],
        "view": view_meta,
        "n_development_anchors": int(idx.size),
        "effective_independent_windows": view.effective_independent_windows("development"),
        "n_shift_valid": int(shift_valid.sum()),
        "checkpoints": checkpoint_rows,
        "score": {key: value for key, value in merged.items() if key != "arrays"},
        "per_anchor_scores_path": str(arrays_path),
        "per_anchor_scores_sha256": file_hash(arrays_path),
        "selection_feedback_used": False,
        "retraining_after_open": False,
        "sealed_partition_opened": False,
        "interpretation_rule": (
            "H_minus_learned > 0 favours state; shifted_minus_correct > 0 favours correct time; "
            "random_minus_learned > 0 favours learned over capacity-matched random reservoir"
        ),
    }
    atomic_write_json(out_dir / "result.json", report)
    return report

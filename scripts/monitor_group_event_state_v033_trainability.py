#!/usr/bin/env python3
"""Incrementally review v0.3.3 human trainability cards.

The monitor is deliberately read-only with respect to training artefacts.  It
looks only at completed ``training_card.json`` files, renders an atomic current
plain-language report, a technical report, and a machine summary, and keeps a
snapshot whenever the set or content of completed cards changes.

This is an optimization / identifiability report.  It must not turn an
inner-selection result into an H1/H2/H3 claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("/data/hfosp_group_event_state_v0_3_3")
REPORT_SCHEMA_VERSION = 9
EXPECTED_REQUESTS = {
    "epilepsiae_253": ("agent_b", "human-sn-r0-253-trainability-o1a-v1"),
    "epilepsiae_916": ("agent_b", "human-sn-r0-916-trainability-o1a-v1"),
    "epilepsiae_1096": ("agent_b_expansion", "human-sn-r0-1096-trainability-broad-v1"),
    "epilepsiae_1125": ("agent_b_expansion", "human-sn-r0-1125-trainability-broad-v1"),
    "epilepsiae_1146": ("agent_b_expansion", "human-sn-r0-1146-trainability-broad-v1"),
    "epilepsiae_384": ("agent_b_expansion", "human-sn-r0-384-trainability-broad-v1"),
    "epilepsiae_548": ("agent_b_expansion", "human-sn-r0-548-trainability-broad-v1"),
    "epilepsiae_583": ("agent_b_expansion", "human-sn-r0-583-trainability-broad-v1"),
    "epilepsiae_922": ("agent_b_expansion", "human-sn-r0-922-trainability-broad-v1"),
}
SELECTED_RECIPE_AUDIT_REQUIRED = {
    subject for subject, (owner, _) in EXPECTED_REQUESTS.items() if owner == "agent_b_expansion"
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _contrast(container: dict[str, Any] | None, nested: str | None = None) -> dict[str, float | None]:
    current: Any = container or {}
    if nested is not None:
        current = current.get(nested) or {}
    return {
        "mean": _number(current.get("mean")),
        "ci_low": _number(current.get("ci_low")),
        "ci_high": _number(current.get("ci_high")),
    }


def _positive_supported(contrast: dict[str, float | None]) -> bool:
    return contrast["ci_low"] is not None and contrast["ci_low"] > 0.0


def _negative_supported(contrast: dict[str, float | None]) -> bool:
    return contrast["ci_high"] is not None and contrast["ci_high"] < 0.0


def _provisional_label(adequate: bool, gain: bool, correct_time: bool, random: bool,
                       beyond_offset: bool | None = None) -> str:
    """``beyond_offset`` is the period-offset control (schema 9): the increment that survives
    replacing the state by one constant vector.  ``None`` means the control is unavailable."""

    if adequate and gain and correct_time and random and beyond_offset:
        return "identified_candidate_on_state_selection"
    if gain and beyond_offset is False:
        return "increment_explained_by_constant_period_offset"
    if adequate and gain:
        return "predictive_increment_without_full_state_identification"
    if adequate:
        return "trained_but_no_supported_increment"
    if gain:
        return "provisional_increment_training_not_adequate"
    return "not_yet_interpretable"


def _fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:+.{digits}f}"


def _fmt_ci(contrast: dict[str, float | None]) -> str:
    return (
        f"{_fmt(contrast['mean'])} "
        f"[{_fmt(contrast['ci_low'])}, {_fmt(contrast['ci_high'])}]"
    )


def _per_seed_directions(card: dict[str, Any]) -> dict[str, Any]:
    """Summarise seed-level directions without treating seeds as patients."""

    diagnostics = ((card.get("diagnostics") or {}).get("multi_seed_diagnostics") or {})
    rows = list(diagnostics.get("per_seed") or [])

    def values(key: str) -> list[dict[str, float | None]]:
        return [_contrast(row.get(key)) for row in rows]

    gain = values("learned_gain_h_minus_model")
    shifted = values("shifted_minus_correct")
    learned_random = values("learned_minus_random")
    seeds = [row.get("seed") for row in rows]
    learned_hashes = [row.get("learned_checkpoint_sha256") for row in rows]
    random_hashes = [row.get("random_checkpoint_sha256") for row in rows]

    return {
        "n_rows": len(rows),
        "n_unique_seeds": len(set(seeds)),
        "n_unique_learned_checkpoints": len(set(learned_hashes)),
        "n_unique_random_checkpoints": len(set(random_hashes)),
        "gain_positive": sum(item["mean"] is not None and item["mean"] > 0.0 for item in gain),
        "gain_ci_supported": sum(_positive_supported(item) for item in gain),
        "shift_positive": sum(item["mean"] is not None and item["mean"] > 0.0 for item in shifted),
        "shift_ci_supported": sum(_positive_supported(item) for item in shifted),
        "learned_random_negative": sum(
            item["mean"] is not None and item["mean"] < 0.0 for item in learned_random
        ),
        "learned_random_ci_supported": sum(_negative_supported(item) for item in learned_random),
    }


def _selected_recipe_tiny_review(data_root: Path, reviewed: ReviewedCard) -> dict[str, Any] | None:
    path = (
        data_root / "supervisor_reports" / "trainability_incremental" /
        "selected_recipe_tiny_overfit" / reviewed.subject / "selected_recipe_tiny_overfit_review.json"
    )
    value = _read_json(path)
    if value is None:
        return None
    if value.get("card_sha256") != reviewed.sha256:
        raise ValueError(f"selected-recipe tiny-overfit review has stale card hash: {path}")
    if value.get("input_hash") != reviewed.card.get("input_hash") \
            or value.get("split_hash") != reviewed.card.get("split_hash"):
        raise ValueError(f"selected-recipe tiny-overfit review input/split mismatch: {path}")
    if value.get("sealed_partition_opened") or value.get("development_evaluation_read"):
        raise ValueError(f"selected-recipe tiny-overfit review opened a forbidden partition: {path}")
    producer = Path(str(value.get("producer_path", "")))
    if not producer.is_file() or _sha256(producer) != value.get("producer_sha256"):
        raise ValueError(f"selected-recipe tiny-overfit producer mismatch: {path}")
    result = value.get("selected_recipe_tiny_overfit") or {}
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "pass": result.get("pass"),
        "gap_closed": _number(result.get("gap_closed")),
        "threshold": _number(result.get("threshold")),
        "n_slice": result.get("n_slice"),
        "presearch_recipe_mismatch": value.get("presearch_recipe_mismatch"),
        "selected_recipe_full_adequacy_conditions_met": value.get(
            "selected_recipe_full_adequacy_conditions_met"
        ),
        "selected_base_recipe_config_hash": value.get("selected_base_recipe_config_hash"),
        "presearch_base_recipe_config_hash": value.get("presearch_base_recipe_config_hash"),
    }


def _period_offset_control(data_root: Path, reviewed: "ReviewedCard") -> dict[str, Any] | None:
    """Card-native control (schema >= 9 cards) or the separate offset/drift audit bound to the card hash."""

    native = reviewed.card.get("period_offset_control")
    if native:
        return {"source": "card", **{k: _contrast(native, k) for k in (
            "gain_period_offset_h_minus_period_mean", "beyond_period_offset_period_mean_minus_learned")}}
    path = data_root / "supervisor_reports" / "trainability_incremental" / "offset_drift_control" / f"{reviewed.subject}.json"
    value = _read_json(path)
    if value is None:
        return None
    if value.get("card_sha256") != reviewed.sha256:
        raise ValueError(f"offset/drift audit has stale card hash: {path}")
    merged = value.get("merged") or {}
    calib = merged.get("h_mark_calibration") or {}
    return {
        "source": "offset_drift_control_audit", "path": str(path), "sha256": _sha256(path),
        "gain_period_offset_h_minus_period_mean": _contrast(merged, "gain_period_offset_H_minus_period_mean_state"),
        "beyond_period_offset_period_mean_minus_learned": _contrast(
            merged, "gain_beyond_period_offset_period_mean_minus_learned"),
        "segment_offset_gain": _contrast(merged, "gain_segment_offset_H_minus_segment_mean_state"),
        "oracle_offset_gain": _contrast(merged, "oracle_offset_H_minus_oracle"),
        "h_mark_count_over_mu_ratio_inner_val": calib.get("inner_val_count_over_mu_ratio_per_bin"),
        "h_mark_count_over_mu_ratio_train": calib.get("train_count_over_mu_ratio_per_bin"),
    }


def _search_rung0_context(data_root: Path, owner: str, request_id: str) -> dict[str, Any] | None:
    """Winner's-curse context: how many freshly sampled configs already beat H on STATE_SELECTION."""

    trace = _read_json(data_root / owner / "search" / request_id / "batch_00" / "search_trace.json")
    if trace is None:
        return None
    out: dict[str, Any] = {}
    for rung in (0, 1, 2):
        gains = [
            _number(row.get("gain_h_minus_model")) for row in trace.get("rows", [])
            if row.get("rung_index") == rung and row.get("status") == "complete"
        ]
        gains = [g for g in gains if g is not None]
        if gains:
            out[f"rung{rung}"] = {"n": len(gains), "fraction_gain_positive": sum(g > 0 for g in gains) / len(gains),
                                  "median_gain": statistics.median(gains), "max_gain": max(gains)}
    return out or None


def _selected_recipe_tiny_900(data_root: Path, reviewed: "ReviewedCard") -> dict[str, Any] | None:
    path = (data_root / "supervisor_reports" / "trainability_incremental" / "selected_recipe_tiny_overfit_900steps"
            / reviewed.subject / "selected_recipe_tiny_overfit_review.json")
    value = _read_json(path)
    if value is None:
        return None
    if value.get("card_sha256") != reviewed.sha256:
        raise ValueError(f"900-step tiny-overfit review has stale card hash: {path}")
    result = value.get("selected_recipe_tiny_overfit") or {}
    return {"path": str(path), "pass": result.get("pass"), "gap_closed": _number(result.get("gap_closed")),
            "steps": result.get("steps"), "threshold": _number(result.get("threshold"))}


def _recalibrated_baseline(data_root: Path, reviewed: "ReviewedCard") -> dict[str, Any] | None:
    path = (data_root / "supervisor_reports" / "trainability_incremental" / "recalibrated_baseline_arms"
            / reviewed.subject / "review.json")
    value = _read_json(path)
    if value is None:
        return None
    if value.get("card_sha256") != reviewed.sha256:
        raise ValueError(f"recalibrated-baseline review has stale card hash: {path}")
    merged = value.get("merged") or {}
    return {
        "path": str(path), "sha256": _sha256(path),
        "recalibration_gain_H_mark_minus_H_recal": _contrast(merged, "recalibration_gain_H_mark_minus_H_recal_inner_val"),
        "gain_H_recal_minus_learned": _contrast(merged, "gain_H_recal_minus_learned"),
        "beyond_period_offset": _contrast(merged, "beyond_period_offset_period_mean_minus_learned"),
        "shifted_minus_correct": _contrast(merged, "shifted_minus_correct"),
        "learned_minus_random": _contrast(merged, "learned_minus_random"),
        "window_seconds": ((merged.get("recalibration") or {}).get("window_seconds")),
    }


def _prior_development_evaluations(data_root: Path, subject: str) -> list[dict[str, Any]]:
    """Already-consumed one-time development scores for the same subject (different, earlier requests).
    Read-only: these files exist because a release was consumed before this round; nothing is re-scored."""

    out = []
    root = data_root / "agent_b" / "development_evaluation_retry1"
    if not root.exists():
        return out
    for result_path in sorted(root.glob("*/result.json")):
        value = _read_json(result_path)
        if value is None or value.get("subject") != subject:
            continue
        score = value.get("score") or {}
        card = _read_json(Path(str(value.get("card_path", "")))) or {}
        out.append({
            "request_id": value.get("request_id"), "result_path": str(result_path),
            "n_development_anchors": value.get("n_development_anchors"),
            "effective_independent_windows": value.get("effective_independent_windows"),
            "development_H_minus_learned": _contrast(score, "H_minus_learned"),
            "development_shifted_minus_correct": _contrast(score, "shifted_minus_correct"),
            "development_random_minus_learned": _contrast(score, "random_minus_learned"),
            "state_selection_gain_of_that_card": _contrast(card.get("blocked_inner_val_gain")),
            "evidence_label_at_freeze": value.get("evidence_label_at_freeze"),
        })
    return out


@dataclass(frozen=True)
class ReviewedCard:
    subject: str
    request_id: str
    path: Path
    sha256: str
    card: dict[str, Any]
    gain: dict[str, float | None]
    shifted: dict[str, float | None]
    learned_random: dict[str, float | None]
    training_adequate: bool
    gain_supported: bool
    correct_time_supported: bool
    learned_better_random: bool
    provisional_label: str


def _review_card(subject: str, request_id: str, path: Path, card: dict[str, Any]) -> ReviewedCard:
    request = card.get("request") or {}
    recorded_request = request.get("request_id")
    recorded_subject = (request.get("input_view") or {}).get("subject") or card.get("subject")
    if recorded_request != request_id:
        raise ValueError(f"request mismatch in {path}: {recorded_request!r} != {request_id!r}")
    if recorded_subject != subject:
        raise ValueError(f"subject mismatch in {path}: {recorded_subject!r} != {subject!r}")
    if card.get("subject") != subject:
        raise ValueError(f"top-level subject mismatch in {path}: {card.get('subject')!r} != {subject!r}")
    if card.get("sealed_partition_opened") or card.get("development_evaluation_read"):
        raise ValueError(f"forbidden partition flag in {path}")

    input_hashes = {
        card.get("input_hash"),
        request.get("input_hash"),
        (request.get("input_view") or {}).get("artifact_sha256"),
    }
    if None in input_hashes or len(input_hashes) != 1:
        raise ValueError(f"input hash mismatch in {path}: {sorted(str(value) for value in input_hashes)}")
    split_hashes = {card.get("split_hash"), request.get("split_hash")}
    if None in split_hashes or len(split_hashes) != 1:
        raise ValueError(f"split hash mismatch in {path}: {sorted(str(value) for value in split_hashes)}")

    seed_directions = _per_seed_directions(card)
    declared_seeds = (card.get("seed_dispersion") or {}).get("n_seeds")
    if declared_seeds != 5 or seed_directions["n_rows"] != 5 or seed_directions["n_unique_seeds"] != 5:
        raise ValueError(
            f"incomplete final seeds in {path}: declared={declared_seeds}, "
            f"rows={seed_directions['n_rows']}, unique={seed_directions['n_unique_seeds']}"
        )
    if seed_directions["n_unique_learned_checkpoints"] != 5:
        raise ValueError(f"non-unique learned checkpoints in {path}")
    if seed_directions["n_unique_random_checkpoints"] != 5:
        raise ValueError(f"non-unique random checkpoints in {path}")

    gain = _contrast(card.get("blocked_inner_val_gain"))
    shifted = _contrast(card.get("shift_null"), "delta_shifted_minus_correct")
    learned_random = _contrast(card.get("random_reservoir_delta"), "learned_minus_random")
    adequate = card.get("evidence_label") == "TRAINING-ADEQUATE"
    gain_supported = _positive_supported(gain)
    time_supported = _positive_supported(shifted)
    random_supported = _negative_supported(learned_random)

    label = _provisional_label(adequate, gain_supported, time_supported, random_supported)

    return ReviewedCard(
        subject=subject,
        request_id=request_id,
        path=path,
        sha256=_sha256(path),
        card=card,
        gain=gain,
        shifted=shifted,
        learned_random=learned_random,
        training_adequate=adequate,
        gain_supported=gain_supported,
        correct_time_supported=time_supported,
        learned_better_random=random_supported,
        provisional_label=label,
    )


def collect_cards(data_root: Path) -> tuple[list[ReviewedCard], list[dict[str, str]]]:
    cards: list[ReviewedCard] = []
    errors: list[dict[str, str]] = []
    for subject, (owner, request_id) in EXPECTED_REQUESTS.items():
        path = data_root / owner / "search" / request_id / "card" / "training_card.json"
        if not path.exists():
            continue
        value = _read_json(path)
        if value is None:
            errors.append({"path": str(path), "error": "unreadable_or_non_atomic_json"})
            continue
        try:
            cards.append(_review_card(subject, request_id, path, value))
        except ValueError as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return cards, errors


def _controller_status(data_root: Path, owner: str) -> dict[str, Any] | None:
    return _read_json(data_root / owner / "agent_b.status.json")


def _replication_progress(data_root: Path) -> dict[str, Any]:
    log_path = data_root / "agent_b" / "o1b_replications_after_o1a.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    by_subject: dict[str, Any] = {}
    for subject in ("253", "916"):
        cells_root = data_root / "agent_b" / f"o1_optimizer_human_{subject}_v4_replication" / "cells"
        cells = []
        if cells_root.exists():
            for cell_root in sorted(cells_root.iterdir()):
                state = _read_json(cell_root / "cell_state.json") or {}
                result = _read_json(cell_root / "run" / "result.json") or {}
                best = result.get("best_validation") or {}
                cells.append({
                    "cell_id": cell_root.name,
                    "status": state.get("status"),
                    "seed": result.get("seed"),
                    "gain_h_minus_model": _number(best.get("gain_h_minus_model")),
                    "selected_step": result.get("selected_step"),
                    "selected_in_warmup": result.get("selected_in_warmup"),
                    "clipping_fraction": _number(result.get("clipping_fraction")),
                    "development_evaluation_read": bool(result.get("development_evaluation_read", False)),
                    "sealed_partition_opened": bool(result.get("sealed_partition_opened", False)),
                })
        gains = [row["gain_h_minus_model"] for row in cells if row["gain_h_minus_model"] is not None]
        by_subject[f"epilepsiae_{subject}"] = {
            "n_expected": 4,
            "n_complete": len(gains),
            "all_complete": len(gains) == 4,
            "gain_median": statistics.median(gains) if gains else None,
            "gain_min": min(gains) if gains else None,
            "gain_max": max(gains) if gains else None,
            "n_positive": sum(value > 0.0 for value in gains),
            "cells": cells,
        }
    return {
        "log_path": str(log_path),
        "started": text.count(" START subject="),
        "completed": text.count(" DONE subject="),
        "all_complete": "REPLICATION_COMPLETE" in text,
        "last_lines": text.splitlines()[-12:],
        "by_subject": by_subject,
    }


def build_summary(data_root: Path, cards: list[ReviewedCard], errors: list[dict[str, str]]) -> dict[str, Any]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for reviewed in sorted(cards, key=lambda row: row.subject):
        card = reviewed.card
        recipe = card.get("recipe") or {}
        arch = recipe.get("arch") or {}
        selected_tiny = _selected_recipe_tiny_review(data_root, reviewed)
        effective_training_adequate = reviewed.training_adequate
        if selected_tiny is not None:
            effective_training_adequate = bool(
                selected_tiny.get("selected_recipe_full_adequacy_conditions_met")
            )
        offset_control = _period_offset_control(data_root, reviewed)
        beyond_offset_supported: bool | None = None
        if offset_control is not None:
            beyond_offset_supported = _positive_supported(
                offset_control["beyond_period_offset_period_mean_minus_learned"]
            )
        effective_label = _provisional_label(
            effective_training_adequate,
            reviewed.gain_supported,
            reviewed.correct_time_supported,
            reviewed.learned_better_random,
            beyond_offset_supported,
        )
        owner, _rid = EXPECTED_REQUESTS[reviewed.subject]
        rows.append({
            "subject": reviewed.subject,
            "request_id": reviewed.request_id,
            "card_path": str(reviewed.path),
            "card_sha256": reviewed.sha256,
            "training_evidence_label": card.get("evidence_label"),
            "training_adequate_from_card": reviewed.training_adequate,
            "provisional_scientific_label": effective_label,
            "blocked_inner_val_gain_h_minus_model": reviewed.gain,
            "shifted_minus_correct": reviewed.shifted,
            "learned_minus_random": reviewed.learned_random,
            "training_adequate": effective_training_adequate,
            "gain_supported": reviewed.gain_supported,
            "correct_time_supported": reviewed.correct_time_supported,
            "learned_better_random": reviewed.learned_better_random,
            "period_offset_control": offset_control,
            "beyond_period_offset_supported": beyond_offset_supported,
            "search_rung_context": _search_rung0_context(data_root, owner, reviewed.request_id),
            "search_n_batches": (card.get("search") or {}).get("n_batches"),
            "search_stop_reason": (card.get("search") or {}).get("stop_reason"),
            "selected_recipe_tiny_overfit_900steps": _selected_recipe_tiny_900(data_root, reviewed),
            "recalibrated_baseline_arms": _recalibrated_baseline(data_root, reviewed),
            "prior_development_evaluations_same_subject": _prior_development_evaluations(data_root, reviewed.subject),
            "n_final_seeds": (card.get("seed_dispersion") or {}).get("n_seeds"),
            "selected_in_warmup": card.get("selected_in_warmup"),
            "selected_at_budget_edge": card.get("selected_at_budget_edge"),
            "adequacy_reasons": card.get("adequacy_reasons"),
            "effective_independent_windows": (card.get("blocked_inner_val_gain") or {}).get(
                "effective_independent_windows"
            ),
            "best_step": card.get("best_step"),
            "state_dim": (card.get("state_variance_rank") or {}).get("state_dim"),
            "state_effective_rank": (card.get("state_variance_rank") or {}).get("participation_ratio"),
            "state_top1_variance_fraction": (card.get("state_variance_rank") or {}).get(
                "fraction_variance_top1"
            ),
            "per_seed_directions": _per_seed_directions(card),
            "tiny_overfit": {
                key: (card.get("tiny_overfit") or {}).get(key)
                for key in ("pass", "gap_closed", "threshold", "n_slice")
            },
            "selected_recipe_tiny_overfit_review": selected_tiny,
            "output_modulation_rms_inner_val": (
                ((card.get("output_modulation") or {}).get("inner_val") or {}).get("modulation_rms")
            ),
            "selection_metric_is_canonical": card.get("selection_metric_is_canonical"),
            "recipe": {
                "optimizer": recipe.get("optimizer"),
                "schedule": recipe.get("schedule"),
                "warmup_fraction": recipe.get("warmup_fraction"),
                "grad_clip": recipe.get("grad_clip"),
                "weight_decay": recipe.get("weight_decay"),
                "width": arch.get("width"),
                "depth": arch.get("depth"),
                "activation": arch.get("activation"),
                "hidden_norm": arch.get("hidden_norm"),
                "write_width": arch.get("write_width"),
                "write_scale": arch.get("write_scale"),
                "taus_seconds": arch.get("taus_seconds"),
            },
        })

    expected = list(EXPECTED_REQUESTS)
    completed = {row["subject"] for row in rows}
    completed_selected_recipe_audits = {
        row["subject"] for row in rows
        if row["subject"] in SELECTED_RECIPE_AUDIT_REQUIRED
        and row["selected_recipe_tiny_overfit_review"] is not None
    }
    return {
        "format": "group_event_state_v0_3_3_incremental_trainability_review",
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "updated_at": now,
        "scope": "STATE_TRAIN plus chronological STATE_SELECTION; optimization and identifiability diagnostic",
        "scientific_boundary": (
            "Interim state-selection contrasts can identify a candidate predictive increment, but cannot establish "
            "H1/H2/H3 or be used for development-driven hyperparameter selection. Schema 9: the recipe, the "
            "checkpoint step and the reported contrast all use the same STATE_SELECTION anchors, so every interval is "
            "optimistically biased by selection; the period-offset control separates a constant baseline-level "
            "correction from time-resolved state information."
        ),
        "expected_subjects": expected,
        "n_expected": len(expected),
        "n_cards": len(rows),
        "pending_subjects": [subject for subject in expected if subject not in completed],
        "selected_recipe_audits": {
            "required_subjects": sorted(SELECTED_RECIPE_AUDIT_REQUIRED),
            "completed_subjects": sorted(completed_selected_recipe_audits),
            "pending_subjects": sorted(
                SELECTED_RECIPE_AUDIT_REQUIRED - completed_selected_recipe_audits
            ),
            "all_complete": completed_selected_recipe_audits == SELECTED_RECIPE_AUDIT_REQUIRED,
        },
        "counts": {
            "training_adequate": sum(row["training_adequate"] for row in rows),
            "gain_supported": sum(row["gain_supported"] for row in rows),
            "correct_time_supported": sum(row["correct_time_supported"] for row in rows),
            "learned_better_random": sum(row["learned_better_random"] for row in rows),
            "beyond_period_offset_supported": sum(bool(row["beyond_period_offset_supported"]) for row in rows),
            "period_offset_control_available": sum(row["period_offset_control"] is not None for row in rows),
            "increment_explained_by_constant_period_offset": sum(
                row["provisional_scientific_label"] == "increment_explained_by_constant_period_offset" for row in rows),
            "fully_identified_on_state_selection": sum(
                row["provisional_scientific_label"] == "identified_candidate_on_state_selection" for row in rows
            ),
        },
        "subjects": rows,
        "read_errors": errors,
        "replication_o1b": _replication_progress(data_root),
        "controllers": {
            "initial": _controller_status(data_root, "agent_b"),
            "expansion": _controller_status(data_root, "agent_b_expansion"),
        },
        "sealed_partition_opened": False,
        "development_evaluation_read": False,
    }


PLAIN_LABELS = {
    "increment_explained_by_constant_period_offset": "增量可由一个常数偏移解释，不是时刻分辨的状态",
    "identified_candidate_on_state_selection": "内部训练段支持候选状态，但还不是 H1 结论",
    "predictive_increment_without_full_state_identification": "有预测增量，尚未证明是时刻特异状态",
    "trained_but_no_supported_increment": "训练到位，但未见可靠增量",
    "provisional_increment_training_not_adequate": "有初步增量，训练识别条件未满足",
    "not_yet_interpretable": "暂不可作科学判读",
}


def render_plain(summary: dict[str, Any]) -> str:
    rows = summary["subjects"]
    counts = summary["counts"]
    lines = [
        "# Group-Event State v0.3.3 训练与可识别性增量报告（白话版）",
        "",
        f"**更新时间：** {summary['updated_at']}",
        "",
        "## 一句话",
        "",
        (
            f"目前 {summary['n_cards']}/{summary['n_expected']} 位患者完成了完整训练卡。"
            f"其中 {counts['gain_supported']} 位在内部靠后的时间段上显示状态比手工历史基线多出可靠信息，"
            f"{counts['correct_time_supported']} 位同时证明这份信息必须放在正确时刻，"
            f"{counts['fully_identified_on_state_selection']} 位同时通过训练充分性、时刻对齐、随机状态和常数偏移四项比较。"
            f"常数偏移对照可用的 {counts['period_offset_control_available']} 位里，"
            f"{counts['increment_explained_by_constant_period_offset']} 位的可靠增量可以由‘把状态换成一个常数’重现，"
            f"只有 {counts['beyond_period_offset_supported']} 位在常数之外还有区间不跨零的增量。"
        ),
        "",
        (
            "这份报告只回答网络有没有训练到、候选状态在 STATE_SELECTION 上有没有增量。"
            "它不打开 development evaluation、发作结局或正式分区，因此不能单独宣布 H1、H2 或 H3 成立。"
        ),
        "",
        "## 已完成患者",
        "",
        "| 患者 | 相对手工历史的改善 | 其中常数偏移就能拿到 | 常数解释不了的部分 | 错时状态代价 | 学到的状态 vs 随机状态 | 当前判读 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        control = row.get("period_offset_control") or {}
        lines.append(
            "| {subject} | {gain} | {offset} | {beyond} | {shift} | {random} | {label} |".format(
                subject=row["subject"].replace("epilepsiae_", "E"),
                gain=_fmt_ci(row["blocked_inner_val_gain_h_minus_model"]),
                offset=_fmt(_contrast(control, "gain_period_offset_h_minus_period_mean")["mean"]) if control else "NA",
                beyond=_fmt_ci(_contrast(control, "beyond_period_offset_period_mean_minus_learned")) if control else "NA",
                shift=_fmt_ci(row["shifted_minus_correct"]),
                random=_fmt_ci(row["learned_minus_random"]),
                label=PLAIN_LABELS[row["provisional_scientific_label"]],
            )
        )
    if not rows:
        lines.append("| — | — | — | — | 尚无完整患者训练卡 |")

    lines.extend([
        "",
        "读数方向：改善、常数偏移、常数解释不了的部分、错时代价均为正更好；‘学到的状态 vs 随机状态’为负更好。",
        "‘常数偏移’一列 = 把状态换成整个选择期的一个常数向量（只用输入、不看答案）后相对手工历史的改善；"
        "同段错时置换不能识别这种常数，所以它单列。所有区间都在选配方与选 checkpoint 的同一段时间上计算，偏乐观。",
        "",
    ])
    for row in rows:
        subject = row["subject"].replace("epilepsiae_", "E")
        reasons = "；".join(row.get("adequacy_reasons") or []) or "无"
        selected_tiny = row.get("selected_recipe_tiny_overfit_review")
        if selected_tiny is None:
            adequacy_text = (
                f"训练卡为 {row.get('training_evidence_label')}；未通过原因：{reasons}。"
            )
        else:
            adequacy_text = (
                f"原训练卡为 {row.get('training_evidence_label')}；原未通过原因：{reasons}。"
                f"纳入最终配方复核后，有效训练充分性为"
                f"{'通过' if row.get('training_adequate') else '未通过'}。"
            )
        lines.extend([
            f"### {subject}",
            "",
            (
                f"共 {row.get('n_final_seeds')} 个最终种子，有效独立窗口约 "
                f"{row.get('effective_independent_windows')} 个。"
                f"{adequacy_text}"
            ),
            "",
        ])
        seed = row["per_seed_directions"]
        lines.extend([
            (
                f"五种子方向：相对手工历史改善 {seed['gain_positive']}/5；"
                f"错时状态更差 {seed['shift_positive']}/5；"
                f"学到的状态优于随机状态 {seed['learned_random_negative']}/5。"
            ),
            "",
        ])
        if row["provisional_scientific_label"] == "identified_candidate_on_state_selection":
            lines.append(
                "在当前内部时间后段上，模型同时满足预测增量、正确时刻和胜过随机状态；"
                "这是可以进入独立 evaluator 的候选状态，但仍需固定物理时间 future-block 和跨发作任务验证。"
            )
        elif row["provisional_scientific_label"] == "increment_explained_by_constant_period_offset":
            lines.append(
                "相对手工历史的改善虽然区间不跨零，但把状态换成整个选择期的一个常数向量就能拿到几乎同样（或更大）的改善；"
                "常数解释不了的那部分区间跨零。这更像手工基线在选择期的水平失准被补上了，"
                "不能读成时刻分辨的状态信息，也不能作为候选状态进入独立 evaluator。"
            )
        elif row["gain_supported"]:
            if row["learned_better_random"] and seed["shift_positive"] == seed["n_rows"]:
                lines.append(
                    "模型在所有种子上都有正确方向的预测增量，并整体胜过等容量随机状态；"
                    "错时状态也在所有种子上更差，但患者内区间仍跨零。"
                    "因此这是较强的候选预测代码信号，尚不足以称时刻特异的慢状态。"
                )
            else:
                lines.append(
                    "模型出现了预测增量，但至少一项训练充分性、正确时刻或随机状态比较未通过；"
                    "因此只能记为值得继续验证的信号，不能称慢状态。"
                )
        else:
            lines.append(
                "当前区间仍跨过零，尚不能区分真实状态信息、种子波动和没有可泛化增量。"
            )
        lines.append("")
        tiny = row.get("tiny_overfit") or {}
        if tiny.get("pass") is False:
            lines.extend([
                (
                    f"训练充分性未通过的具体原因：在 {tiny.get('n_slice')} 个 TRAIN 小样本上，"
                    f"只闭合了 {100.0 * float(tiny.get('gap_closed') or 0.0):.1f}% 的可拟合差距，"
                    f"低于 {100.0 * float(tiny.get('threshold') or 0.0):.0f}% 合同线。"
                    "这不推翻人体内部后段的预测增量，但说明该网络尚未通过完整的容量/优化 sanity check。"
                ),
                "",
            ])
        if selected_tiny is not None:
            lines.extend([
                (
                    f"最终入选配方已另行在同一 TRAIN 小切片复核：闭合 "
                    f"{100.0 * float(selected_tiny.get('gap_closed') or 0.0):.1f}% 的可拟合差距，"
                    f"合同判定为 {'通过' if selected_tiny.get('pass') else '未通过'}。"
                    "因此训练充分性标签不再依赖搜索前的旧配方。"
                ),
                "",
            ])
        control = row.get("period_offset_control")
        if control is not None:
            beyond = _contrast(control, "beyond_period_offset_period_mean_minus_learned")
            offset = _contrast(control, "gain_period_offset_h_minus_period_mean")
            ratio = control.get("h_mark_count_over_mu_ratio_inner_val")
            ratio_text = "" if not ratio else (
                f"手工基线在选择期的实际事件数/预测事件数之比为 {', '.join(f'{float(v):.2f}' for v in ratio)}"
                f"（三个未来窗），训练段为 {', '.join(f'{float(v):.2f}' for v in (control.get('h_mark_count_over_mu_ratio_train') or []))}。"
            )
            lines.extend([
                (
                    f"常数偏移对照：把状态换成一个常数后仍改善 {_fmt(offset['mean'])}；"
                    f"常数解释不了的部分为 {_fmt_ci(beyond)}"
                    f"（{'区间在零之上' if _positive_supported(beyond) else '区间跨零或为负'}）。{ratio_text}"
                ),
                "",
            ])
        recal = row.get("recalibrated_baseline_arms")
        if recal is not None:
            lines.extend([
                (
                    f"因果重标定基线（只用过去 {float(recal.get('window_seconds') or 0) / 3600:.0f} 小时已观测到的事件数校正基线水平）："
                    f"基线自身改善 {_fmt(recal['recalibration_gain_H_mark_minus_H_recal']['mean'])}；"
                    f"在这个基线上重训后，状态增量 {_fmt_ci(recal['gain_H_recal_minus_learned'])}，"
                    f"常数解释不了的部分 {_fmt_ci(recal['beyond_period_offset'])}，"
                    f"错时代价 {_fmt_ci(recal['shifted_minus_correct'])}，学到−随机 {_fmt_ci(recal['learned_minus_random'])}。"
                ),
                "",
            ])
        tiny900 = row.get("selected_recipe_tiny_overfit_900steps")
        if tiny900 is not None:
            lines.extend([
                (
                    f"最终配方按其自身 {tiny900.get('steps')} 步预算复核小样本可拟合性：闭合 "
                    f"{100.0 * float(tiny900.get('gap_closed') or 0.0):.1f}%，"
                    f"{'通过' if tiny900.get('pass') else '未通过'}（300 步版本见上）。"
                ),
                "",
            ])
        ctx = row.get("search_rung_context") or {}
        if ctx.get("rung0"):
            r0 = ctx["rung0"]
            lines.extend([
                (
                    f"搜索背景：第一轮随机抽到的 {r0['n']} 个配置里，{100.0 * r0['fraction_gain_positive']:.0f}% 在同一段选择期上已经胜过手工历史"
                    f"（中位 {_fmt(r0['median_gain'])}）；本轮只跑了 {row.get('search_n_batches')} 个搜索批次（{row.get('search_stop_reason')}）。"
                ),
                "",
            ])
        for prior in row.get("prior_development_evaluations_same_subject") or []:
            lines.extend([
                (
                    f"同一患者此前已消费的一次性 development 评分（旧请求 {prior['request_id']}，不是本卡）："
                    f"选择期增量 {_fmt_ci(prior['state_selection_gain_of_that_card'])} → development 段 "
                    f"{_fmt_ci(prior['development_H_minus_learned'])}，错时代价 {_fmt_ci(prior['development_shifted_minus_correct'])}。"
                    "这是选择期读数能否外推的直接证据。"
                ),
                "",
            ])
        if row.get("state_effective_rank") is not None:
            state_dim = row.get("state_dim")
            state_dim_text = "未知维" if state_dim is None else f"{int(state_dim)} 维"
            lines.extend([
                (
                    f"{state_dim_text}模型状态的有效秩约 {row['state_effective_rank']:.2f}；"
                    f"第一方向解释约 {100.0 * float(row.get('state_top1_variance_fraction') or 0.0):.1f}% 方差。"
                    "这是模型表示诊断，不是生理状态维数。"
                ),
                "",
            ])

    pending = [value.replace("epilepsiae_", "E") for value in summary["pending_subjects"]]
    rep = summary["replication_o1b"]
    lines.extend([
        "## 当前进度",
        "",
        f"- 待完成患者：{', '.join(pending) if pending else '无'}。",
        f"- O1b 独立复现：{rep['completed']}/8 完成，{rep['started']}/8 已启动。",
        "- 正式检验分区、发作结局和 paper-ready 图均未读取或修改。",
        "",
        "## 最终应怎样使用这些结果",
        "",
        (
            "完成全部患者后，先判断‘网络训练问题’能否解释旧阴性，再把冻结候选状态交给独立 evaluator。"
            "只有在固定真实时间的未来事件块上持续有增量、且正确时刻状态胜过错时状态，才进入 H1；"
            "能进一步预测事件路径或发作，才分别进入 H2a/H2b。"
        ),
        "",
    ])
    for subject, result in rep["by_subject"].items():
        short = subject.replace("epilepsiae_", "E")
        if result["n_complete"]:
            lines.insert(
                -6,
                (
                    f"- {short} O1b 固定配方复现：{result['n_complete']}/4 个种子完成；"
                    f"H−model 增量中位数 {_fmt(result['gain_median'])}，"
                    f"范围 {_fmt(result['gain_min'])} 至 {_fmt(result['gain_max'])}，"
                    f"{result['n_positive']}/{result['n_complete']} 同向。"
                ),
            )
    if rep["all_complete"]:
        mixed = [
            subject.replace("epilepsiae_", "E")
            for subject, result in rep["by_subject"].items()
            if 0 < result["n_positive"] < result["n_complete"]
        ]
        lines.insert(
            -6,
            (
                "- O1b 判读：固定配方在不同种子间"
                + (f"于 {', '.join(mixed)} 都发生了方向翻转；" if mixed else "方向一致；")
                + "这证明优化与初始化会影响人体读数，但 O1b 没有独立的错时状态和随机状态对照，"
                + "不能据此认定 H1。"
            ),
        )
    return "\n".join(lines)


def render_technical(summary: dict[str, Any]) -> str:
    lines = [
        "# Group-Event State v0.3.3 incremental trainability review（技术版）",
        "",
        f"Updated: `{summary['updated_at']}`",
        "",
        "## Scope and estimand",
        "",
        (
            "All reported contrasts use STATE_TRAIN plus chronological STATE_SELECTION only. "
            "The principal interim contrast is H_mark NLL minus candidate-model NLL; positive favours the candidate. "
            "No development evaluation, seizure outcome, sealed partition, or formal test partition is read."
        ),
        "",
        "Training adequacy is reported separately from scientific support. Interim patient results are not pooled while "
        "the support-selected expansion cohort is incomplete.",
        "The search metric is intentionally non-canonical until the independent Agent A evaluator hash is registered.",
        "",
        "## Per-subject machine results",
        "",
        "| subject | card | train label | H-model mean [CI] | period-offset arm | beyond-offset [CI] | shifted-correct mean [CI] | learned-random mean [CI] | seeds | eff. blocks | provisional label |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["subjects"]:
        control = row.get("period_offset_control") or {}
        lines.append(
            "| {subject} | `{sha}` | {train} | {gain} | {offset} | {beyond} | {shift} | {random} | {seeds} | {blocks} | `{label}` |".format(
                subject=row["subject"],
                sha=row["card_sha256"][:12],
                train=row["training_evidence_label"],
                gain=_fmt_ci(row["blocked_inner_val_gain_h_minus_model"]),
                offset=_fmt(_contrast(control, "gain_period_offset_h_minus_period_mean")["mean"]) if control else "NA",
                beyond=_fmt_ci(_contrast(control, "beyond_period_offset_period_mean_minus_learned")) if control else "NA",
                shift=_fmt_ci(row["shifted_minus_correct"]),
                random=_fmt_ci(row["learned_minus_random"]),
                seeds=row["n_final_seeds"],
                blocks=row["effective_independent_windows"],
                label=row["provisional_scientific_label"],
            )
        )

    lines.extend(["", "## Selected recipes and failure reasons", ""])
    for row in summary["subjects"]:
        lines.extend([
            f"### {row['subject']}",
            "",
            f"- request: `{row['request_id']}`",
            f"- card: `{row['card_path']}`",
            f"- card sha256: `{row['card_sha256']}`",
            f"- best step: `{row['best_step']}`; warm-up selected: `{row['selected_in_warmup']}`; budget edge: `{row['selected_at_budget_edge']}`",
            f"- state dimension/effective rank: `{row['state_dim']}` / `{row['state_effective_rank']}`",
            f"- top-1 state variance fraction: `{row['state_top1_variance_fraction']}`",
            f"- per-seed directions: `{json.dumps(row['per_seed_directions'], ensure_ascii=False, sort_keys=True)}`",
            f"- tiny overfit: `{json.dumps(row['tiny_overfit'], ensure_ascii=False, sort_keys=True)}`",
            f"- selected-recipe tiny-overfit review: `{json.dumps(row['selected_recipe_tiny_overfit_review'], ensure_ascii=False, sort_keys=True)}`",
            f"- inner-val output modulation RMS: `{row['output_modulation_rms_inner_val']}`",
            f"- canonical evaluator metric registered: `{row['selection_metric_is_canonical']}`",
            f"- adequacy reasons: `{row['adequacy_reasons']}`",
            f"- period-offset control: `{json.dumps(row['period_offset_control'], ensure_ascii=False, sort_keys=True)}`",
            f"- search context (fresh-config gains per rung, n_batches, stop): `{json.dumps(row['search_rung_context'], ensure_ascii=False, sort_keys=True)}`, `{row['search_n_batches']}`, `{row['search_stop_reason']}`",
            f"- selected-recipe tiny-overfit at the recipe's own 900-step budget: `{json.dumps(row['selected_recipe_tiny_overfit_900steps'], ensure_ascii=False, sort_keys=True)}`",
            f"- causally recalibrated baseline arms: `{json.dumps(row['recalibrated_baseline_arms'], ensure_ascii=False, sort_keys=True)}`",
            f"- prior consumed development evaluations (same subject, earlier requests): `{json.dumps(row['prior_development_evaluations_same_subject'], ensure_ascii=False, sort_keys=True)}`",
            f"- recipe: `{json.dumps(row['recipe'], ensure_ascii=False, sort_keys=True)}`",
            "",
        ])

    lines.extend([
        "## Interim aggregation rule",
        "",
        f"- complete cards: `{summary['n_cards']}/{summary['n_expected']}`",
        f"- counts: `{json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- pending: `{summary['pending_subjects']}`",
        f"- read errors: `{summary['read_errors']}`",
        "- no cohort p value is emitted before all support-selected subjects complete.",
        "- these cards do not establish H1/H2/H3; a frozen independent evaluator is required.",
        "- schema 9: recipe selection, checkpoint selection and the reported interval share the STATE_SELECTION anchors "
        "(optimistic bias); the period-offset control (state -> its inner-val mean) is the minimum control a "
        "same-segment block shift cannot provide; `identified_candidate_on_state_selection` now also requires "
        "`beyond_period_offset_period_mean_minus_learned.ci_low > 0`.",
        "",
        "## O1b replication and controller state",
        "",
        f"```json\n{json.dumps(summary['replication_o1b'], ensure_ascii=False, indent=2)}\n```",
        "",
        "Full controller snapshots are retained in `incremental_summary.json` and are not copied into this report.",
        "",
    ])
    return "\n".join(lines)


def _report_fingerprint(cards: Iterable[ReviewedCard], replication: dict[str, Any], data_root: Path) -> str:
    card_payload = "\n".join(
        f"{row.subject}:{row.sha256}" for row in sorted(cards, key=lambda value: value.subject)
    )
    replication_payload = json.dumps(
        {
            "started": replication.get("started"),
            "completed": replication.get("completed"),
            "all_complete": replication.get("all_complete"),
        },
        sort_keys=True,
    )
    audit_payload = []
    for row in sorted(cards, key=lambda value: value.subject):
        base = data_root / "supervisor_reports" / "trainability_incremental"
        for path in (
            base / "selected_recipe_tiny_overfit" / row.subject / "selected_recipe_tiny_overfit_review.json",
            base / "selected_recipe_tiny_overfit_900steps" / row.subject / "selected_recipe_tiny_overfit_review.json",
            base / "offset_drift_control" / f"{row.subject}.json",
            base / "recalibrated_baseline_arms" / row.subject / "review.json",
        ):
            if path.is_file():
                audit_payload.append(f"{row.subject}:{path.name}:{_sha256(path)}")
    payload = (
        f"schema={REPORT_SCHEMA_VERSION}\n{card_payload}\n{replication_payload}\n"
        + "\n".join(audit_payload)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_reports(output_root: Path, summary: dict[str, Any], *, snapshot: bool) -> None:
    plain = render_plain(summary)
    technical = render_technical(summary)
    _atomic_text(output_root / "incremental_plain.md", plain)
    _atomic_text(output_root / "incremental_technical.md", technical)
    _atomic_json(output_root / "incremental_summary.json", summary)
    if snapshot:
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        snapshot_root = output_root / "snapshots" / stamp
        _atomic_text(snapshot_root / "plain.md", plain)
        _atomic_text(snapshot_root / "technical.md", technical)
        _atomic_json(snapshot_root / "summary.json", summary)


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def monitor(data_root: Path, output_root: Path, poll_seconds: float, once: bool) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "monitor_state.json"
    previous = _read_json(state_path) or {}
    previous_fingerprint = previous.get("report_fingerprint") or previous.get("card_fingerprint")

    while True:
        cards, errors = collect_cards(data_root)
        summary = build_summary(data_root, cards, errors)
        fingerprint = _report_fingerprint(cards, summary["replication_o1b"], data_root)
        changed = fingerprint != previous_fingerprint
        if changed or not (output_root / "incremental_plain.md").exists():
            write_reports(output_root, summary, snapshot=changed)
            previous_fingerprint = fingerprint

        controllers = summary["controllers"]
        expansion_pid = ((controllers.get("expansion") or {}).get("controller_pid"))
        state = {
            "format": "group_event_state_v0_3_3_incremental_monitor_state",
            "heartbeat_at": summary["updated_at"],
            "pid": os.getpid(),
            "report_fingerprint": fingerprint,
            "card_fingerprint": hashlib.sha256(
                "\n".join(
                    f"{row.subject}:{row.sha256}" for row in sorted(cards, key=lambda value: value.subject)
                ).encode()
            ).hexdigest(),
            "n_cards": summary["n_cards"],
            "n_expected": summary["n_expected"],
            "pending_subjects": summary["pending_subjects"],
            "read_errors": errors,
            "expansion_controller_pid": expansion_pid,
            "expansion_controller_alive": _process_alive(expansion_pid),
            "all_expected_cards_complete": summary["n_cards"] == summary["n_expected"],
            "all_replications_complete": summary["replication_o1b"]["all_complete"],
            "all_selected_recipe_audits_complete": summary["selected_recipe_audits"]["all_complete"],
        }
        state["all_outputs_complete"] = (
            state["all_expected_cards_complete"]
            and state["all_replications_complete"]
            and state["all_selected_recipe_audits_complete"]
        )
        _atomic_json(state_path, state)

        if once or state["all_outputs_complete"]:
            return 0
        time.sleep(max(5.0, poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "supervisor_reports" / "trainability_incremental",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(monitor(**vars(parse_args())))

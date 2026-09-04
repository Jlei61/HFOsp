#!/usr/bin/env python3
"""Aggregate the complete v0.3.5 run and render its four scientific figures.

The script never fits or selects a model.  It reads only materialised final
cards, collapses optimisation seeds within patient first, and records missing
or non-estimable patient-endpoint units instead of silently dropping them.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.topic5_group_event_state.v035.contracts import INPUT_ROOT, OUTPUT_ROOT, atomic_json
from src.topic5_group_event_state.v035.feedback_models import _nested_arm_admissibility
from src.topic5_epi_prssm.figure_style import apply_style

import os

# Review 2026-09-04: the original run's q(t) carried a non-causal
# ``segment_fraction`` feature (segment END, which coincides with the next
# seizure onset for most patients).  Reports built from that root are marked
# as contaminated; the causal re-run uses a separate root and report tag.
ORIGINAL_ROOT = Path("/data/hfosp_group_event_state_v0_3_5")
REPORT_TAG = os.environ.get("HFOSP_GES_V035_REPORT_TAG", "full_execution")
REPORT_DATE = os.environ.get("HFOSP_GES_V035_REPORT_DATE", "2026-09-04")
Q_TRAJECTORY_CAUSAL = OUTPUT_ROOT.resolve() != ORIGINAL_ROOT.resolve()


SUBJECTS = (
    "epilepsiae_253", "epilepsiae_922", "epilepsiae_1096", "epilepsiae_548",
    "epilepsiae_583", "epilepsiae_1146", "epilepsiae_384", "epilepsiae_1125",
)
SHORT = {s: "E" + s.rsplit("_", 1)[-1] for s in SUBJECTS}
RATE_COLOR = "#E69F00"
MARK_COLOR = "#0072B2"
SHIFT_COLOR = "#9E9E9E"
BURDEN_COLOR = "#009E73"
NEG_COLOR = "#D55E00"
MEDIAN_COLOR = "#A35E48"
PATIENT_COLOR = "#6B7280"
SUPPORT_COLOR = "#DDE9E5"
HORIZON_LABEL = {300: "5 min", 1800: "30 min", 7200: "120 min"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cards(name: str) -> list[tuple[Path, dict[str, Any]]]:
    root = OUTPUT_ROOT / name
    return [(path, _json(path)) for path in sorted(root.glob("**/card.json"))]


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _arm_metric(arm: dict[str, Any], metric: str) -> float | None:
    """Accept both early scalar cards and the final {mean,n} score schema."""
    value = arm.get(metric)
    if isinstance(value, dict):
        value = value.get("mean")
    return _number(value)


def _arm_metric_on_shift_support(arm: dict[str, Any], metric: str) -> float | None:
    """Same arm restricted to block-shift donor-valid anchors; None on older cards."""
    value = arm.get(metric)
    if isinstance(value, dict):
        return _number(value.get("mean_on_shift_support"))
    return None


def _median(values: Iterable[Any]) -> float | None:
    clean = [_number(v) for v in values]
    clean = [v for v in clean if v is not None]
    return float(np.median(clean)) if clean else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def _patient_summary(rows: list[dict[str, Any]], keys: tuple[str, ...], values: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    output = []
    for group, current in grouped.items():
        item = dict(zip(keys, group))
        item["n_seeds"] = len({row.get("seed") for row in current if row.get("seed") is not None})
        for value in values:
            item[value] = _median(row.get(value) for row in current)
        output.append(item)
    return sorted(output, key=lambda r: tuple(str(r.get(k, "")) for k in keys))


def _state_training_map() -> dict[tuple[str, int], int]:
    """Selected state-training epoch for every final subject/seed unit.

    Epoch zero is a valid engineering output, but it is not evidence that an
    event-content state was learned.  Downstream long tables retain those
    units; primary state claims require at least two updated seeds per patient.
    """
    output: dict[tuple[str, int], int] = {}
    for _path, card in _cards("full_mark_final") or _cards("full_mark_state"):
        seed = card.get("seed")
        epoch = card.get("selected_epoch")
        if seed is not None and epoch is not None:
            output[(str(card["subject"]), int(seed))] = int(epoch)
    return output


def _state_training_rows() -> list[dict[str, Any]]:
    by_subject: dict[str, list[tuple[int, int]]] = {}
    for (subject, seed), epoch in _state_training_map().items():
        by_subject.setdefault(subject, []).append((seed, epoch))
    output = []
    for subject in SUBJECTS:
        rows = by_subject.get(subject, [])
        n_updated = sum(epoch > 0 for _seed, epoch in rows)
        output.append({
            "subject": subject,
            "n_state_seeds": len(rows),
            "n_updated_seeds": n_updated,
            "median_selected_epoch": _median(epoch for _seed, epoch in rows),
            "primary_state_eligible": len(rows) >= 3 and n_updated >= 2,
            "status": "ESTIMATED" if rows else "NOT_ESTIMABLE",
        })
    return output


def _primary_state_subjects() -> set[str]:
    return {row["subject"] for row in _state_training_rows()
            if row["primary_state_eligible"]}


def _attach_state_training(row: dict[str, Any], subject: str, seed: int | None,
                           state_map: dict[tuple[str, int], int]) -> dict[str, Any]:
    epoch = state_map.get((subject, int(seed))) if seed is not None else None
    row["selected_state_epoch"] = epoch
    row["state_updated_seed"] = None if epoch is None else int(epoch > 0)
    return row


def _execution_data_scale() -> dict[str, float | int]:
    """Count the registered input universe and the part actually analysed.

    The stored subject file spans the whole recording, but every v0.3.5 stage
    stops at the registered 80 % boundary.  Reporting only the full-record
    totals overstates the analysed data by about a quarter of the events
    (review 2026-09-04), so both are returned and both are printed.
    """
    n_events = 0
    n_events_analysed = 0
    valid_seconds = 0.0
    analysed_seconds = 0.0
    n_available = 0
    for subject in SUBJECTS:
        manifest_path = INPUT_ROOT / subject / "manifest_v3.json"
        if not manifest_path.exists():
            continue
        manifest = _json(manifest_path)
        boundary = float(manifest["report"]["phase_boundaries_epoch"]["80pct"])
        with np.load(manifest["input_path"], allow_pickle=False) as data:
            event_time = np.asarray(data["event_time"], dtype=np.float64)
            n_events += int(event_time.size)
            n_events_analysed += int((event_time < boundary).sum())
            bounds = np.asarray(data["target_segment_bounds"], dtype=np.float64)
            valid_seconds += float(np.sum(bounds[:, 1] - bounds[:, 0]))
            analysed_seconds += float(np.sum(np.clip(np.minimum(bounds[:, 1], boundary) - bounds[:, 0], 0.0, None)))
        n_available += 1
    return {"n_subjects": n_available, "n_events": n_events,
            "n_events_analysed": n_events_analysed,
            "valid_recording_hours": valid_seconds / 3600.0,
            "analysed_recording_hours": analysed_seconds / 3600.0}


def _nb_nll_numpy(count: np.ndarray, mu: np.ndarray, log_dispersion: float) -> np.ndarray:
    """Numpy twin of ``dynamic_rate.negative_binomial_nll`` for stored predictions."""
    lgamma = np.vectorize(math.lgamma)
    r = max(math.log1p(math.exp(float(log_dispersion))), 1e-4)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-6, 1e8)
    count = np.asarray(count, dtype=np.float64)
    return -(lgamma(count + r) - lgamma(r) - lgamma(count + 1.0)
             + r * (math.log(r) - np.log(r + mu)) + count * (np.log(mu) - np.log(r + mu)))


def _rate_shift_support(card: dict[str, Any], j: int) -> dict[str, Any]:
    """Score the correct-time residual/dynamic arms on the block-shift support.

    The card scores ``block_shift`` only on anchors that have a distant donor,
    while the other arms use every eligible SELECTION anchor.  The timing
    contrast must compare identical anchors (review 2026-09-04); the stored
    per-anchor predictions and the checkpoint dispersion make that possible
    without re-training.  The full-support residual NLL is recomputed as a
    self-check against the card.
    """
    import torch
    out = {"residual_nll_on_shift_support": None, "dynamic_nll_on_shift_support": None,
           "n_shift_anchors": 0, "residual_nll_recomputed": None}
    trajectory = Path(card.get("trajectory_path", ""))
    checkpoint = Path(card.get("checkpoint_path", ""))
    if not trajectory.exists() or not checkpoint.exists():
        return out
    log_dispersion = float(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]["log_dispersion"][j])
    with np.load(trajectory, allow_pickle=False) as z:
        selection = np.asarray(z["phase"]).astype(str) == "SELECTION"
        valid = np.asarray(z["target_valid"], dtype=bool)[:, j] & selection
        support = valid & np.asarray(z["block_shift_valid"], dtype=bool)
        count = np.asarray(z["target_count"], dtype=np.float64)[:, j]
        pred_residual = np.asarray(z["pred_residual"], dtype=np.float64)[:, j]
        pred_dynamic = np.asarray(z["pred_dynamic"], dtype=np.float64)[:, j]
    if valid.any():
        out["residual_nll_recomputed"] = float(np.mean(_nb_nll_numpy(count[valid], pred_residual[valid], log_dispersion)))
    if support.any():
        out["residual_nll_on_shift_support"] = float(np.mean(_nb_nll_numpy(count[support], pred_residual[support], log_dispersion)))
        out["dynamic_nll_on_shift_support"] = float(np.mean(_nb_nll_numpy(count[support], pred_dynamic[support], log_dispersion)))
        out["n_shift_anchors"] = int(support.sum())
    return out


def dynamic_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    dynamic_root = "dynamic_rate_final" if (OUTPUT_ROOT / "dynamic_rate_final").exists() else "dynamic_rate"
    for _path, card in _cards(dynamic_root):
        for j, horizon in enumerate(card["config"]["horizons_seconds"]):
            arms = card["selection_arms"]
            static = _number(arms["static"]["nll"][j])
            dynamic = _number(arms["dynamic"]["nll"][j])
            residual = _number(arms["residual"]["nll"][j])
            shifted = _number(arms["block_shift"]["nll"][j])
            support = _rate_shift_support(card, j)
            recomputed = support["residual_nll_recomputed"]
            if residual is not None and recomputed is not None and abs(recomputed - residual) > 1e-3 * max(1.0, abs(residual)):
                raise RuntimeError(f"{_path}: recomputed residual NLL {recomputed} disagrees with card {residual}")
            residual_support = support["residual_nll_on_shift_support"]
            rows.append({
                "subject": card["subject"], "seed": card["seed"], "horizon_seconds": int(horizon),
                "n_anchors": arms["residual"]["n"][j], "static_nll": static,
                "dynamic_nll": dynamic, "residual_nll": residual, "block_shift_nll": shifted,
                "n_shift_anchors": support["n_shift_anchors"],
                "residual_nll_on_shift_support": residual_support,
                "dynamic_nll_on_shift_support": support["dynamic_nll_on_shift_support"],
                "dynamic_gain_over_static": None if static is None or dynamic is None else static - dynamic,
                "residual_gain_over_static": None if static is None or residual is None else static - residual,
                "residual_gain_over_dynamic": None if dynamic is None or residual is None else dynamic - residual,
                "correct_time_gain_over_shift": (
                    None if shifted is None or residual_support is None else shifted - residual_support
                ),
                "correct_time_support": "block_shift versus residual on identical donor-valid anchors",
                "observed_mean": arms["residual"]["observed_mean"][j],
                "predicted_mean": arms["residual"]["predicted_mean"][j],
                "q_trajectory_causal": Q_TRAJECTORY_CAUSAL,
            })
    background = []
    background_root = "background_rate_final" if (OUTPUT_ROOT / "background_rate_final").exists() else "background_rate"
    for _path, card in _cards(background_root):
        audit = card["background_audit"]
        background.append({"subject": card["subject"], "seed": card["seed"],
                           **card["selection"],
                           "background_source": audit.get("source"),
                           "n_available": audit.get("n_available"),
                           "fraction_available": audit.get("fraction_available"),
                           "median_age_seconds": audit.get("median_age_seconds"),
                           "p95_age_seconds": audit.get("p95_age_seconds"),
                           "maximum_age_seconds": audit.get("maximum_age_seconds"),
                           "event_anchor_required": audit.get("event_anchor_required")})
    return rows, background


def _stepwise_shift_support(card: dict[str, Any], metric: str) -> tuple[float | None, float | None, int]:
    """Block-shift null and correct-time score on donor-valid anchors only.

    Older cards averaged ``block_shift`` over every selection pair, where
    anchors without a distant donor silently kept their correct-time context.
    The per-anchor arrays carry NaN for those anchors, so both sides of the
    timing contrast can be rebuilt on identical support (review 2026-09-04).
    """
    path = Path(card.get("per_anchor_path", ""))
    if not path.exists():
        return None, None, 0
    with np.load(path, allow_pickle=False) as z:
        ok = np.asarray(z["shift_valid"], dtype=bool)
        shift = np.asarray(z[f"block_shift_{metric}"], dtype=np.float64)
        rate = np.asarray(z[f"rate_dynamic_{metric}"], dtype=np.float64)
    rows = np.flatnonzero(ok & np.isfinite(shift))
    if rows.size == 0:
        return None, None, 0
    return float(np.mean(shift[rows])), float(np.mean(rate[rows])), int(rows.size)


def stepwise_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for _path, card in _cards("stepwise_decoder"):
        for metric in ("grammar", "next_bce", "stop_bce", "contact_nll"):
            a = card["selection_means"]
            shift_loss, rate_on_support, n_shift = _stepwise_shift_support(card, metric)
            rows.append({"subject": card["subject"], "seed": card["config"]["seed"],
                         "metric": metric, "static_loss": a["static"][metric],
                         "rate_loss": a["rate_dynamic"][metric], "shift_loss": shift_loss,
                         "rate_loss_on_shift_support": rate_on_support, "n_shift_anchors": n_shift,
                         "rate_gain_over_static": a["static"][metric] - a["rate_dynamic"][metric],
                         "correct_time_gain_over_shift": (
                             None if shift_loss is None or rate_on_support is None else shift_loss - rate_on_support
                         )})
    oracle = []
    for _path, card in _cards("stepwise_oracle"):
        selection = card.get("selection_means", card.get("selection", {}))
        for metric in ("grammar", "next_bce", "stop_bce", "contact_nll"):
            if metric in selection.get("static", {}) and metric in selection.get("future_oracle", {}):
                oracle.append({"subject": card["subject"], "seed": card.get("seed", card.get("config", {}).get("seed")),
                               "metric": metric,
                               "oracle_gain_over_static": selection["static"][metric] - selection["future_oracle"][metric]})
    return rows, oracle


def h1_h2a_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grammar, functional, auxiliary = [], [], []
    state_map = _state_training_map()
    cards = _cards("full_mark_final") or _cards("full_mark_state")
    for _path, card in cards:
        arms = card.get("selection", {}).get("arms", {})
        for horizon, current in arms.items():
            for metric in ("grammar", "next_bce", "stop_bce", "contact_nll", "mean"):
                q = _arm_metric(current.get("rate_only", {}), metric)
                static = _arm_metric(current.get("static_only", {}), metric)
                m = _arm_metric(current.get("mark_only", {}), metric)
                qm = _arm_metric(current.get("rate_plus_mark", {}), metric)
                shift = _arm_metric(current.get("block_shift_mark", {}), metric)
                qm_support = _arm_metric_on_shift_support(current.get("rate_plus_mark", {}), metric)
                constant = _arm_metric(current.get("period_mean_mark", {}), metric)
                if q is None or qm is None:
                    continue
                grammar.append(_attach_state_training({"subject": card["subject"], "seed": card["seed"], "horizon": horizon,
                                "metric": metric, "static_loss": static, "q_loss": q,
                                "mark_only_loss": m, "q_plus_mark_loss": qm,
                                "q_plus_mark_loss_on_shift_support": qm_support,
                                "shift_loss": shift, "period_mean_loss": constant,
                                "rate_gain_over_static": None if static is None else static - q,
                                "mark_only_gain_over_static": None if static is None or m is None else static - m,
                                "mark_gain_over_q": q - qm,
                                "correct_time_gain_over_shift": (
                                    None if shift is None or qm_support is None else shift - qm_support
                                ),
                                "correct_time_support_matched": qm_support is not None,
                                "mark_gain_over_period_mean": None if constant is None else constant - qm},
                                card["subject"], card["seed"], state_map))
        for horizon, current in card.get("physical_selection", {}).get("horizons", {}).items():
            for endpoint in ("count_nll", "extent_bce", "participation_bce"):
                contrast = current.get("contrasts", {}).get(endpoint, {})
                functional.append(_attach_state_training({
                    "subject": card["subject"], "seed": card["seed"],
                    "family": "trained_physical_head", "horizon": horizon,
                    "endpoint": endpoint,
                    "rate_gain_over_static": contrast.get("rate_gain_over_static"),
                    "mark_only_gain_over_static": contrast.get("mark_only_gain_over_static"),
                    "state_gain_over_q": contrast.get("mark_gain_over_q"),
                    "correct_time_gain_over_shift": (
                        contrast.get("correct_time_gain_over_shift")
                        if "correct_on_shift_support" in current else None
                    ),
                    "correct_time_support_matched": "correct_on_shift_support" in current,
                    "mark_gain_over_period_mean": contrast.get("mark_gain_over_period_mean"),
                    "n_values": current.get("q_plus_mark", {}).get("n_anchors"),
                }, card["subject"], card["seed"], state_map))
    for _path, card in _cards("functional_readouts_final") or _cards("functional_readouts"):
        for family in ("event_horizons", "physical_horizons"):
            for horizon, endpoints in card.get(family, {}).items():
                for endpoint, result in endpoints.items():
                    contrast = result.get("contrasts", {})
                    seed = _seed_from_path(_path)
                    functional.append(_attach_state_training({"subject": card["subject"], "seed": seed,
                                       "family": family, "horizon": horizon, "endpoint": endpoint,
                                       "rate_gain_over_static": contrast.get("rate_gain_over_static"),
                                       "mark_only_gain_over_static": contrast.get("state_only_gain_over_static"),
                                       "state_gain_over_q": contrast.get("state_gain_over_q"),
                                       "correct_time_gain_over_shift": (
                                           contrast.get("correct_time_gain_over_shift")
                                           if "correct_state_on_shift_support" in result else None
                                       ),
                                       "correct_time_support_matched": "correct_state_on_shift_support" in result,
                                       "mark_gain_over_period_mean": contrast.get("dynamic_gain_over_fit_period_mean"),
                                       "n_values": result.get("q_plus_state", {}).get("n_values")},
                                       card["subject"], seed, state_map))
    for _path, card in _cards("stepwise_auxiliary_final") or _cards("stepwise_auxiliary"):
        selection = card.get("selection", {})
        for horizon, endpoints in selection.get("q_only", {}).items():
            for endpoint, q in endpoints.items():
                qm = selection.get("q_plus_mark_state", {}).get(horizon, {}).get(endpoint)
                static = selection.get("base_head", {}).get(horizon, {}).get(endpoint)
                mark_only = selection.get("mark_state_only", {}).get(horizon, {}).get(endpoint)
                shift = selection.get("block_shift_mark_state", {}).get(horizon, {}).get(endpoint)
                support = selection.get("q_plus_mark_state_on_shift_support", {}).get(horizon, {}).get(endpoint)
                if _number(q) is None or _number(qm) is None: continue
                seed = _seed_from_path(_path)
                auxiliary.append(_attach_state_training({"subject": card["subject"], "seed": seed,
                                  "horizon": horizon, "endpoint": endpoint,
                                  "rate_gain_over_static": None if _number(static) is None else float(static) - float(q),
                                  "mark_only_gain_over_static": None if _number(static) is None or _number(mark_only) is None else float(static) - float(mark_only),
                                  "mark_gain_over_q": float(q) - float(qm),
                                  "correct_time_gain_over_shift": (
                                      None if _number(shift) is None or _number(support) is None
                                      else float(shift) - float(support)
                                  ),
                                  "correct_time_support_matched": _number(support) is not None},
                                  card["subject"], seed, state_map))
    return grammar, functional, auxiliary


def _seed_from_path(path: Path) -> int | None:
    name = path.parent.name
    if "state_seed" in name:
        try: return int(name.split("state_seed", 1)[1])
        except ValueError: return None
    if name.startswith("seed"):
        try: return int(name[4:])
        except ValueError: return None
    return None


def _validate_registered_designs() -> None:
    """Refuse a polished report when a promised scientific arm is absent.

    ``NOT_ESTIMABLE`` remains a valid scientific result.  Missing a registered
    arm is an execution failure and must never be disguised by an empty panel.
    """

    errors: list[str] = []
    wrong_time_schemas: dict[str, list[str]] = {}
    for path, card in _cards("full_mark_final"):
        for horizon, arms in card.get("selection", {}).get("arms", {}).items():
            required = {"static_only", "rate_only", "mark_only", "rate_plus_mark",
                        "period_mean_mark", "block_shift_mark"}
            missing = required - set(arms)
            if missing: errors.append(f"{path}:{horizon}:missing {sorted(missing)}")
        for horizon, arms in card.get("physical_selection", {}).get("horizons", {}).items():
            required = {"static_only", "q_only", "mark_only", "q_plus_mark",
                        "fit_period_mean_mark"}
            missing = required - set(arms)
            if missing: errors.append(f"{path}:{horizon}:missing {sorted(missing)}")
            # The wrong-time null has two admissible constructions: the original
            # within-segment circular shift and the clock-matched donor set that
            # replaced it on 2026-09-04 (the shift has no donors beyond ~6 h).
            # Either is fine; MIXING them inside one results tree is not, because
            # the two nulls answer different questions.  Record which one this
            # card used and fail loudly below if the tree is not uniform.
            schema = ({"block_shift_mark", "correct_on_shift_support"} <= set(arms) and "circular_shift") or \
                     ({"matched_wrong_time", "correct_on_matched_support"} <= set(arms) and "clock_matched") or None
            if schema is None:
                errors.append(f"{path}:{horizon}:no complete wrong-time null "
                              f"(need block_shift_mark+correct_on_shift_support or "
                              f"matched_wrong_time+correct_on_matched_support); got {sorted(arms)}")
            else:
                wrong_time_schemas.setdefault(schema, []).append(f"{path}:{horizon}")
    for path, card in _cards("functional_readouts_final"):
        for family in ("event_horizons", "physical_horizons"):
            for horizon, endpoints in card.get(family, {}).items():
                for endpoint, arms in endpoints.items():
                    required = {"static_only", "q_only", "state_only", "q_plus_state",
                                "fit_period_mean_state", "block_shift_state"}
                    missing = required - set(arms)
                    if missing: errors.append(f"{path}:{family}:{horizon}:{endpoint}:missing {sorted(missing)}")
    for path, card in _cards("stepwise_auxiliary_final"):
        if card.get("status") == "NOT_ESTIMABLE": continue
        required = {"base_head", "q_only", "mark_state_only", "q_plus_mark_state",
                    "block_shift_mark_state"}
        missing = required - set(card.get("selection", {}))
        if missing: errors.append(f"{path}:missing {sorted(missing)}")
    for path, card in _cards("seizure_transfer_final"):
        required = {"clinical_only", "q_clinical", "mark_state_clinical",
                    "q_clinical_plus_state"}
        missing = required - set(card.get("distance_survival", {}).get("arms", {}))
        if missing: errors.append(f"{path}:hazard missing {sorted(missing)}")
    for path, card in _cards("feedback_models_final"):
        for design, result in card.get("designs", {}).items():
            if result.get("status") != "ESTIMATED": continue
            for endpoint, arms in result.get("endpoints", {}).items():
                required = {"M0_common_drive", "M1_burden_feedback", "M2_mark_feedback"}
                missing = required - set(arms)
                if missing: errors.append(f"{path}:{design}:{endpoint}:missing {sorted(missing)}")
                if "admissibility" not in arms or "raw_contrasts" not in arms:
                    errors.append(f"{path}:{design}:{endpoint}:missing numerical admissibility audit")
    if len(wrong_time_schemas) > 1:
        detail = "; ".join(f"{name}: {len(paths)} units e.g. {paths[0]}" for name, paths in sorted(wrong_time_schemas.items()))
        errors.append(
            "wrong-time null schema is not uniform across this results tree -- cards were written by "
            f"different code versions and their nulls are not comparable ({detail}). "
            "Re-run the affected units under one code version before reporting."
        )
    if errors:
        raise RuntimeError("registered v0.3.5 scientific arms are incomplete:\n" + "\n".join(errors))


def h2b_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    risk, field, support = [], [], []
    state_map = _state_training_map()
    cards = _cards("seizure_transfer_final") or _cards("seizure_transfer")
    for path, card in cards:
        seed = _seed_from_path(path)
        hz = card.get("distance_survival", {})
        support.append(_attach_state_training(
            {"subject": card["subject"], "seed": seed,
             **{f"seizures_{k.lower()}": v for k, v in hz.get("seizures_by_phase", {}).items()}},
            card["subject"], seed, state_map))
        arms = hz.get("arms", {})
        for horizon in ("5min", "15min", "30min", "60min", "120min"):
            current = {name: arm.get("selection", {}).get(horizon, {}) for name, arm in arms.items()
                       if isinstance(arm, dict)}
            clinical = current.get("clinical_only", {}).get("brier")
            q = current.get("q_clinical", {}).get("brier")
            m = current.get("mark_state_clinical", {}).get("brier")
            qm = current.get("q_clinical_plus_state", {}).get("brier")
            shift = current.get("block_shift_state", {}).get("brier")
            correct_support = current.get("correct_state_on_shift_support", {}).get("brier")
            period = current.get("fit_period_mean_state", {}).get("brier")
            base = current.get("clinical_only", {})
            if _number(clinical) is None and base.get("status") == "NOT_ESTIMABLE":
                # Withheld horizons stay in the long table as an estimability
                # record; they carry no contrast, so the summaries skip them.
                risk.append(_attach_state_training(
                    {"subject": card["subject"], "seed": seed, "horizon": horizon,
                     "status": base.get("status"), "withheld_reason": base.get("reason"),
                     "n_anchors": base.get("n_anchors"), "n_positive": base.get("n_positive"),
                     "n_full_followup": base.get("n_full_followup"),
                     "n_event_only": base.get("n_event_only"),
                     "outcome_dependent_eligibility": base.get("outcome_dependent_eligibility")},
                    card["subject"], seed, state_map))
            if any(_number(v) is not None for v in (clinical, q, m, qm)):
                risk.append(_attach_state_training({"subject": card["subject"], "seed": seed, "horizon": horizon,
                             "status": "ESTIMATED",
                             "n_full_followup": base.get("n_full_followup"),
                             "n_event_only": base.get("n_event_only"),
                             "outcome_dependent_eligibility": base.get("outcome_dependent_eligibility"),
                             "clinical_brier": clinical, "q_brier": q, "mark_brier": m, "qm_brier": qm,
                             "rate_gain_over_clinical": None if _number(clinical) is None or _number(q) is None else clinical-q,
                             "mark_gain_over_clinical": None if _number(clinical) is None or _number(m) is None else clinical-m,
                             "mark_gain_over_rate": None if _number(q) is None or _number(qm) is None else q-qm,
                             "correct_time_gain_over_shift": None if _number(shift) is None or _number(correct_support) is None else shift-correct_support,
                             "mark_gain_over_period_mean": None if _number(period) is None or _number(qm) is None else period-qm,
                             "n_anchors": current.get("clinical_only", {}).get("n_anchors"),
                             "n_positive": current.get("clinical_only", {}).get("n_positive")},
                             card["subject"], seed, state_map))
        for lead, endpoints in card.get("early_ictal_field_and_path", {}).items():
            if not isinstance(endpoints, dict): continue
            for endpoint, result in endpoints.items():
                if not isinstance(result, dict): continue
                contrast = result.get("registered_contrasts", {})
                field.append(_attach_state_training({"subject": card["subject"], "seed": seed, "lead": lead,
                              "endpoint": endpoint, "status": result.get("status", "ESTIMATED"),
                              "rate_gain_over_clinical": contrast.get("rate_gain_over_clinical"),
                              "mark_gain_over_clinical": contrast.get("mark_gain_over_clinical"),
                              "mark_gain_over_rate": contrast.get("mark_gain_over_rate", result.get("state_gain_over_q")),
                              "correct_time_gain_over_shift": result.get("correct_time_gain_over_shift"),
                              "mark_gain_over_period_mean": result.get("mark_gain_over_period_mean"),
                              **result.get("support", {})}, card["subject"], seed, state_map))
    return risk, field, support


def h3_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons, innovation = [], []
    state_map = _state_training_map()
    cards = _cards("feedback_models_final") or _cards("feedback_models")
    for path, card in cards:
        seed = _seed_from_path(path)
        for design, result in card.get("designs", {}).items():
            status = result.get("status")
            support = result.get("support", result.get("audit", {}).get("n_nonoverlap_by_phase", {}))
            if status != "ESTIMATED":
                comparisons.append(_attach_state_training(
                    {"subject": card["subject"], "seed": seed, "design": design,
                     "status": status, **{f"n_{k.lower()}": v for k, v in support.items()}},
                    card["subject"], seed, state_map))
                continue
            for endpoint, values in result.get("endpoints", {}).items():
                m0 = values.get("M0_common_drive", {}); m1 = values.get("M1_burden_feedback", {}); m2 = values.get("M2_mark_feedback", {})
                # Re-derive admissibility from the stored per-arm MSEs with the
                # symmetric rule, so cards written before the 2026-09-04 review
                # (child-only 4x test) are judged identically to new ones.
                # Cards written before the 2026-09-04 review do not store the
                # FIT-mean null.  Every H3 outcome is standardized on FIT, so
                # its variance is 1 by construction and a sane model's MSE is
                # O(1); fall back to that unit reference rather than leaving a
                # jointly diverged parent/child pair unchecked.
                null = values.get("null_fit_mean")
                reference = null or {"inner_mse": 1.0, "selection_mse": 1.0}
                burden_a = _nested_arm_admissibility(m0, m1, null=reference)
                mark_s = _nested_arm_admissibility(m1, m2, null=reference)
                mark_ok = bool(burden_a["admissible"] and mark_s["admissible"])
                raw = {
                    "burden_gain_over_common": None if _number(m0.get("selection_mse")) is None or _number(m1.get("selection_mse")) is None
                    else float(m0["selection_mse"]) - float(m1["selection_mse"]),
                    "mark_gain_over_burden": None if _number(m1.get("selection_mse")) is None or _number(m2.get("selection_mse")) is None
                    else float(m1["selection_mse"]) - float(m2["selection_mse"]),
                }
                c = {"burden_gain_over_common": raw["burden_gain_over_common"] if burden_a["admissible"] else None,
                     "mark_gain_over_burden": raw["mark_gain_over_burden"] if mark_ok else None}
                comparisons.append(_attach_state_training({"subject": card["subject"], "seed": seed, "design": design,
                                    "endpoint": endpoint, "status": status,
                                    "burden_gain_over_common": c.get("burden_gain_over_common"),
                                    "mark_gain_over_burden": c.get("mark_gain_over_burden"),
                                    "burden_gain_over_common_raw": raw.get("burden_gain_over_common"),
                                    "mark_gain_over_burden_raw": raw.get("mark_gain_over_burden"),
                                    "burden_admissible": bool(burden_a["admissible"]),
                                    "mark_admissible": mark_ok,
                                    "burden_admissibility_reasons": ";".join(burden_a.get("reasons", [])),
                                    "mark_admissibility_reasons": ";".join(mark_s.get("reasons", []) + ([] if burden_a["admissible"] else ["parent_burden_arm_inadmissible"])),
                                    "admissibility_rule": "withheld when INNER or SELECTION MSE differs from nested parent by >4x in either direction, or exceeds 4x the FIT-mean null MSE (absolute clause only when the card stores null_fit_mean)",
                                    "absolute_reference": "stored FIT-mean null" if null is not None else "unit-variance fallback (FIT-standardized outcome)",
                                    "burden_impulse": values.get("M1_burden_feedback", {}).get("signed_impulse_mean_selection"),
                                    "mark_impulse": values.get("M2_mark_feedback", {}).get("signed_impulse_mean_selection"),
                                    **{f"n_{k.lower()}": v for k, v in support.items()},
                                    }, card["subject"], seed, state_map))
        for horizon, result in card.get("functional_innovation", {}).items():
            if horizon in ("status", "semantics") or not isinstance(result, dict): continue
            innovation.append(_attach_state_training(
                {"subject": card["subject"], "seed": seed, "horizon": horizon, **result},
                card["subject"], seed, state_map))
    return comparisons, innovation


def _style() -> None:
    apply_style()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def _finish(fig: plt.Figure, stem: Path, metadata: dict[str, Any]) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    atomic_json(stem.with_suffix(".metadata.json"), metadata)
    plt.close(fig)


def _finish_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def _point_axis(ax: plt.Axes, rows: list[dict[str, Any]], category_key: str,
                categories: tuple[Any, ...], category_labels: tuple[str, ...],
                series: tuple[tuple[str, str, str], ...], *, ylabel: str,
                title: str, legend: bool = False) -> None:
    """Nature-style patient points plus a cohort-median diamond.

    Rows must already be collapsed within patient.  Positive values always
    favour the hypothesis; the pale region is a direction cue, never a
    significance claim.  Missing patients remain missing and are counted in
    each tick label rather than filled with zero.
    """
    all_values: list[float] = []
    support_by_category: list[int] = []
    width = 0.22 if len(series) > 1 else 0.0
    offsets = np.linspace(-width, width, len(series)) if len(series) > 1 else np.asarray([0.0])
    for j, category in enumerate(categories):
        category_support: set[str] = set()
        for offset, (field, label, color) in zip(offsets, series):
            current = [
                row for row in rows
                if row.get(category_key) == category and _number(row.get(field)) is not None
            ]
            values = [float(row[field]) for row in current]
            category_support.update(str(row.get("subject")) for row in current)
            if not values:
                continue
            jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(j + offset + jitter, values, s=16, color=color, alpha=.68,
                       edgecolor="white", linewidth=.35, zorder=3)
            ax.scatter([j + offset], [np.median(values)], marker="D", s=29,
                       facecolor=MEDIAN_COLOR, edgecolor="white", linewidth=.45, zorder=5)
            all_values.extend(values)
        support_by_category.append(len(category_support))
    limit = max([abs(v) for v in all_values] + [0.02]) * 1.18
    ax.set_ylim(-limit, limit)
    ax.axhspan(0, limit, color=SUPPORT_COLOR, alpha=.5, zorder=-4)
    ax.axhline(0, color="#666666", lw=.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(categories)), [f"{label}\nn={n}" for label, n in zip(category_labels, support_by_category)])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(.98, .96, "favourable  ↑", transform=ax.transAxes, ha="right", va="top",
            color="#4D766B", fontsize=7)
    if legend:
        handles = [Line2D([], [], marker="o", ls="", ms=4, color=color, label=label)
                   for _field, label, color in series]
        handles.append(Line2D([], [], marker="D", ls="", ms=4.5,
                              color=MEDIAN_COLOR, label="patient median"))
        ax.legend(handles=handles, frameon=False, loc="lower left")
    _finish_axis(ax)


def plot_dynamic(rate_rows, background_rows, figures: Path) -> None:
    _style(); fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8)); axes=axes.ravel()
    trace = OUTPUT_ROOT/"dynamic_rate_final"/"epilepsiae_1096"/"seed20260903"/"trajectory_and_scores.npz"
    if not trace.exists():
        trace = OUTPUT_ROOT/"dynamic_rate"/"epilepsiae_1096"/"seed20260903"/"trajectory_and_scores.npz"
    if trace.exists():
        with np.load(trace, allow_pickle=False) as z:
            keep = np.asarray(z["phase"]).astype(str) == "SELECTION"
            t = (z["anchor_time"][keep] - z["anchor_time"][keep][0]) / 3600
            axes[0].plot(t, z["target_count"][keep,0], color="#4D4D4D", lw=.75, label="Observed")
            axes[0].plot(t, z["pred_static"][keep,0], color=SHIFT_COLOR, lw=1.0, label="Static")
            axes[0].plot(t, z["pred_residual"][keep,0], color=RATE_COLOR, lw=1.0, label="Causal dynamic")
            axes[0].set(xlabel="Recorded time (h)", ylabel="Events in next 5 min", title="A  Patient-level rate drift")
            axes[0].legend(frameon=False, ncol=1, loc="upper left", handlelength=1.2)
    _finish_axis(axes[0])
    rate = _patient_summary(rate_rows, ("subject", "horizon_seconds"),
                            ("dynamic_gain_over_static", "residual_gain_over_static",
                             "residual_gain_over_dynamic", "correct_time_gain_over_shift"))
    _point_axis(axes[1], rate, "horizon_seconds", (300,1800,7200), ("5 min","30 min","120 min"),
                (("dynamic_gain_over_static","causal q(t)",RATE_COLOR),
                 ("residual_gain_over_dynamic","learned residual over q(t)",MARK_COLOR)),
                ylabel="Incremental count NLL gain", title="B  Dynamic load", legend=True)
    _point_axis(axes[2], rate, "horizon_seconds", (300,1800,7200), ("5 min","30 min","120 min"),
                (("correct_time_gain_over_shift","correct over shifted",MARK_COLOR),),
                ylabel="Correct-time NLL gain", title="C  Time alignment")
    bg = _patient_summary(background_rows, ("subject",),
                          ("background_gain","correct_time_gain_over_shift"))
    bg_long=[]
    for row in bg:
        bg_long.extend(({"subject":row["subject"],"comparison":"increment","value":row["background_gain"]},
                        {"subject":row["subject"],"comparison":"timing","value":row["correct_time_gain_over_shift"]}))
    _point_axis(axes[3], bg_long, "comparison", ("increment","timing"), ("Beyond q(t)","Correct time"),
                (("value","fixed-clock background",MARK_COLOR),),
                ylabel="Count NLL gain", title="D  Background SEEG")
    fig.tight_layout(w_pad=1.2, h_pad=1.3)
    _finish(fig, figures/"fig_v035_dynamic_baseline", {"question":"Does causal q(t) track within-patient load beyond a static calibration?","patient_aggregation":"median across seeds first","positive_direction":"up/right","source_tables":["dynamic_baseline.csv","background_baseline.csv"]})


def plot_h1_h2a(grammar_rows, functional_rows, auxiliary_rows, figures: Path) -> None:
    _style(); fig, axes = plt.subplots(2,2,figsize=(7.2,5.0)); axes=axes.ravel()
    primary = _primary_state_subjects()
    grammar_rows=[r for r in grammar_rows if r.get("subject") in primary]
    functional_rows=[r for r in functional_rows if r.get("subject") in primary]
    auxiliary_rows=[r for r in auxiliary_rows if r.get("subject") in primary]
    g = [r for r in grammar_rows if r["metric"] in ("grammar","mean")]
    map_h={"next_1_events":1,"next_5_events":5,"next_20_events":20}
    gp=_patient_summary(
        g,("subject","horizon"),
        ("rate_gain_over_static","mark_only_gain_over_static","mark_gain_over_q","correct_time_gain_over_shift"),
    )
    _point_axis(axes[0], gp, "horizon", ("next_1_events","next_5_events","next_20_events"),
                ("Next 1","Next 5","Next 20"),
                (("rate_gain_over_static","q(t) over static",RATE_COLOR),
                 ("mark_only_gain_over_static","m(t) over static",PATIENT_COLOR),
                 ("mark_gain_over_q","q(t)+m(t) over q(t)",MARK_COLOR)),
                ylabel="Grammar loss gain", title="A  Event horizon", legend=True)
    f=[r for r in functional_rows if r["family"]=="physical_horizons"]
    fp=_patient_summary(f,("subject","horizon"),
                        ("rate_gain_over_static","mark_only_gain_over_static","state_gain_over_q",
                         "correct_time_gain_over_shift","mark_gain_over_period_mean"))
    _point_axis(axes[1], fp, "horizon", ("future_5min","future_30min","future_120min"),
                ("5 min","30 min","120 min"),
                (("rate_gain_over_static","q(t) over static",RATE_COLOR),
                 ("mark_only_gain_over_static","m(t) over static",PATIENT_COLOR),
                 ("state_gain_over_q","q(t)+m(t) over q(t)",MARK_COLOR)),
                ylabel="Conditional morphology gain", title="B  Time horizon")
    decisive=[]
    for row in fp:
        if row["horizon"] != "future_30min": continue
        decisive.extend((
            {"subject":row["subject"],"comparison":"residual","gain":row["state_gain_over_q"]},
            {"subject":row["subject"],"comparison":"timing","gain":row["correct_time_gain_over_shift"]},
            {"subject":row["subject"],"comparison":"constant","gain":row["mark_gain_over_period_mean"]},
        ))
    _point_axis(axes[2], decisive, "comparison", ("residual","timing","constant"),
                ("Beyond\nq(t)","Correct\ntime","Over\nmean"), (("gain","30-min mark state",MARK_COLOR),),
                ylabel="Conditional morphology gain", title="C  State controls")
    same=[r for r in grammar_rows if r["horizon"]=="next_1_events" and r["metric"] in ("next_bce","stop_bce","contact_nll")]
    sp=_patient_summary(
        same,("subject","metric"),
        ("rate_gain_over_static","mark_only_gain_over_static","mark_gain_over_q","correct_time_gain_over_shift"),
    )
    metrics=("next_bce","stop_bce","contact_nll")
    _point_axis(axes[3], sp, "metric", metrics, ("Later\ncontact","Continue /\nSTOP","Contact\nNLL"),
                (("rate_gain_over_static","q(t) over static",RATE_COLOR),
                 ("mark_gain_over_q","q(t)+m(t) over q(t)",MARK_COLOR)),
                ylabel="Gain after observed prefix", title="D  Same-prefix")
    fig.tight_layout(w_pad=1.2, h_pad=1.2)
    _finish(fig, figures/"fig_v035_h1_h2a", {"question":"Does full event content define a time-specific predictive state beyond q(t)?","patient_aggregation":"median across seeds first","primary_state_rule":"at least two of three seeds selected a non-initial state checkpoint","primary_subjects":sorted(primary),"positive_direction":"up","same_prefix":"first tied group is observed; later recruitment and STOP are scored","source_tables":["h1_h2a_grammar.csv","h1_h2a_functional.csv","h1_h2a_auxiliary.csv","state_training.csv"]})


def plot_h2b(risk_rows, field_rows, figures: Path) -> None:
    _style(); fig,axes=plt.subplots(1,3,figsize=(7.2,2.55))
    primary = _primary_state_subjects()
    # A horizon with no positive seizure window cannot measure seizure-risk
    # discrimination.  Keep it in the long machine table as an estimability
    # record, but do not draw it as biological evidence (positive or negative).
    risk_rows=[r for r in risk_rows
               if r.get("subject") in primary and (_number(r.get("n_positive")) or 0)>0]
    rp=_patient_summary(
        risk_rows,("subject","horizon"),
        ("rate_gain_over_clinical","mark_gain_over_clinical","mark_gain_over_rate"),
    )
    horizons=("5min","15min","30min","60min","120min")
    _point_axis(axes[0],rp,"horizon",horizons,("5","15","30","60","120"),
                (("rate_gain_over_clinical","q(t) over clinical",RATE_COLOR),
                 ("mark_gain_over_rate","q(t)+m(t) over q(t)",MARK_COLOR)),
                ylabel="Brier gain",title="A  Distance to next seizure",legend=True)
    axes[0].set_xlabel("Forecast horizon (min)")
    controls=[]
    for row in rp:
        if row["horizon"] != "30min": continue
        controls.extend((
            {"subject":row["subject"],"comparison":"residual","gain":row["mark_gain_over_rate"]},
            {"subject":row["subject"],"comparison":"timing","gain":row.get("correct_time_gain_over_shift")},
            {"subject":row["subject"],"comparison":"constant","gain":row.get("mark_gain_over_period_mean")},
        ))
    # Rebuild with the two time-specific control fields retained.
    rp_control=_patient_summary(
        risk_rows,("subject","horizon"),
        ("mark_gain_over_rate","correct_time_gain_over_shift","mark_gain_over_period_mean"),
    )
    controls=[]
    for row in rp_control:
        if row["horizon"] != "30min": continue
        controls.extend((
            {"subject":row["subject"],"comparison":"residual","gain":row["mark_gain_over_rate"]},
            {"subject":row["subject"],"comparison":"timing","gain":row["correct_time_gain_over_shift"]},
            {"subject":row["subject"],"comparison":"constant","gain":row["mark_gain_over_period_mean"]},
        ))
    _point_axis(axes[1],controls,"comparison",("residual","timing","constant"),
                ("Beyond\nq(t)","Correct\ntime","Over FIT\nmean"),
                (("gain","30-min mark state",MARK_COLOR),),
                ylabel="Brier gain",title="B  State controls")
    field_rows=[r for r in field_rows if r.get("subject") in primary and r.get("status")=="ESTIMATED"]
    fp=_patient_summary(
        field_rows,("subject","lead","endpoint"),
        ("rate_gain_over_clinical","mark_gain_over_clinical","mark_gain_over_rate"),
    )
    # A patient may contribute several early-field endpoints.  Collapse those
    # endpoints within patient before plotting so every dot remains one
    # patient, as stated by the figure contract.
    fp_patient=_patient_summary(
        fp,("subject","lead"),
        ("rate_gain_over_clinical","mark_gain_over_clinical","mark_gain_over_rate"),
    )
    leads=("lead_360min","lead_120min","lead_30min","lead_5min")
    _point_axis(axes[2],fp_patient,"lead",leads,("6 h","2 h","30 min","5 min"),
                (("rate_gain_over_clinical","q(t) over clinical",RATE_COLOR),
                 ("mark_gain_over_rate","q(t)+m(t) over q(t)",MARK_COLOR)),
                ylabel="Early-field score gain",title="C  Early ictal field/path",legend=True)
    fig.tight_layout(w_pad=1.2)
    _finish(fig, figures/"fig_v035_h2b", {"question":"Does a state learned only from interictal events transfer to seizure timing and early ictal fields?","state_producer":"frozen before seizure outcome readout","patient_aggregation":"median across seeds first, then endpoints for field panel","primary_state_rule":"at least two of three seeds selected a non-initial state checkpoint","primary_subjects":sorted(primary),"risk_eligibility":"patient-horizon requires at least one positive seizure window; zero-positive rows remain in the long table only","positive_direction":"up","source_tables":["h2b_risk.csv","h2b_field.csv","state_training.csv"]})


def plot_h3(comparison_rows, innovation_rows, figures: Path) -> None:
    _style(); fig,axes=plt.subplots(1,3,figsize=(7.2,2.55))
    primary = _primary_state_subjects()
    estimated=[r for r in comparison_rows
               if r.get("status")=="ESTIMATED" and r.get("subject") in primary]
    pp=_patient_summary(estimated,("subject","design"),("burden_gain_over_common","mark_gain_over_burden"))
    # The 6 h arm has only two patients and numerically unstable extrapolation;
    # it remains in the machine table and support panel, but is not drawn as a
    # core effect size.  5k/10k have no estimable patients.
    core_designs=("physical_1800","physical_7200","event_count_1000")
    _point_axis(axes[0],pp,"design",core_designs,("30 min","2 h","1k events"),
                (("burden_gain_over_common","burden over common",BURDEN_COLOR),
                 ("mark_gain_over_burden","mark over burden",MARK_COLOR)),
                ylabel="Future-block score gain",title="A  Feedback-model comparison",legend=True)
    designs=("physical_1800","physical_7200","physical_21600","event_count_1000","event_count_5000","event_count_10000")
    support_b=[len({r["subject"] for r in estimated if r["design"]==design and
                    _number(r.get("burden_gain_over_common")) is not None}) for design in designs]
    support_m=[len({r["subject"] for r in estimated if r["design"]==design and
                    _number(r.get("mark_gain_over_burden")) is not None}) for design in designs]
    x=np.arange(len(designs)); width=.34
    axes[1].bar(x-width/2,support_b,color=BURDEN_COLOR,width=width,label="burden")
    axes[1].bar(x+width/2,support_m,color=MARK_COLOR,width=width,label="mark")
    axes[1].set_xticks(range(len(designs)),("30 m","2 h","6 h","1k","5k","10k"),rotation=25)
    axes[1].set_ylabel("Patients with stable fit")
    axes[1].set_title("B  Independent-block support",loc="left",fontweight="bold")
    axes[1].set_ylim(0,max(support_b+support_m+[1])*1.32)
    for j,(b,m) in enumerate(zip(support_b,support_m)):
        axes[1].text(j-width/2,b+.08,str(b),ha="center",va="bottom",fontsize=6.5)
        axes[1].text(j+width/2,m+.08,str(m),ha="center",va="bottom",fontsize=6.5)
    axes[1].legend(frameon=False,loc="upper right",fontsize=6.5)
    axes[1].text(.98,.70,"Unstable nested fits withheld;\nraw scores remain in table",transform=axes[1].transAxes,
                 ha="right",va="top",fontsize=6.5,color="#555555")
    _finish_axis(axes[1])
    innovation_rows=[r for r in innovation_rows if r.get("subject") in primary]
    ip=_patient_summary(innovation_rows,("subject","horizon"),("signed_functional_innovation_mean","association_with_current_extent","association_with_current_multiband_energy"))
    hs=("next_1_events","next_5_events","next_20_events")
    _point_axis(axes[2],ip,"horizon",hs,("Next 1","Next 5","Next 20"),
                (("association_with_current_extent","current extent",BURDEN_COLOR),
                 ("association_with_current_multiband_energy","current energy",MARK_COLOR)),
                ylabel="Association with state innovation",title="C  Event-linked innovation",legend=True)
    fig.tight_layout(w_pad=1.2)
    _finish(fig, figures/"fig_v035_h3", {"question":"Is a feedback term from IED burden or content needed beyond common drive?","interpretation_ceiling":"event-feedback-like predictive dependence, not causal proof","patient_aggregation":"endpoint median within seed, seed median within patient","primary_state_rule":"at least two of three seeds selected a non-initial state checkpoint","primary_subjects":sorted(primary),"numerical_admissibility":"added arm withheld if INNER or SELECTION MSE is nonfinite or greater than 4x nested parent; raw score retained","six_hour_effect":"reported only in machine table because support is low","positive_direction":"up","source_tables":["h3_models.csv","h3_innovation.csv","state_training.csv"]})


def _sign_summary(rows: list[dict[str, Any]], value: str) -> dict[str, Any]:
    vals=[r[value] for r in rows if _number(r.get(value)) is not None]
    positive=sum(float(v)>0 for v in vals)
    negative=sum(float(v)<0 for v in vals)
    zero=len(vals)-positive-negative
    n_eff=positive+negative
    if n_eff:
        tail=sum(math.comb(n_eff,k) for k in range(min(positive,negative)+1))/(2**n_eff)
        p=min(1.0,2.0*tail)
    else:
        p=None
    return {"n":len(vals),"n_positive":positive,"n_negative":negative,"n_zero":zero,
            "n_nonzero":n_eff,"median":_median(vals),"two_sided_sign_p":p}


def _report_text(dynamic_p, background_p, grammar_p, functional_p, risk_p, field_p,
                 h3_p, state_training) -> tuple[str,str]:
    scale = _execution_data_scale()
    primary = _primary_state_subjects()
    learned = lambda rows: [r for r in rows if r.get("subject") in primary]
    d5=[r for r in dynamic_p if r["horizon_seconds"]==300]
    d30=[r for r in dynamic_p if r["horizon_seconds"]==1800]
    d120=[r for r in dynamic_p if r["horizon_seconds"]==7200]
    g1=learned([r for r in grammar_p if r["horizon"]=="next_1_events" and r["metric"]=="grammar"])
    g20=learned([r for r in grammar_p if r["horizon"]=="next_20_events" and r["metric"]=="grammar"])
    event20=_patient_summary(
        learned([r for r in functional_p if r["family"]=="event_horizons" and r["horizon"]=="next_20_events"]),
        ("subject",),("state_gain_over_q","correct_time_gain_over_shift","mark_gain_over_period_mean"),
    )
    f30=_patient_summary(
        learned([r for r in functional_p if r["family"]=="physical_horizons" and r["horizon"]=="future_30min"]),
        ("subject",),("state_gain_over_q","correct_time_gain_over_shift","mark_gain_over_period_mean"),
    )
    risk30=learned([r for r in risk_p if r["horizon"]=="30min" and (_number(r.get("n_positive")) or 0)>0])
    field30=_patient_summary(
        learned([r for r in field_p if r["lead"]=="lead_30min" and _number(r.get("mark_gain_over_rate")) is not None]),
        ("subject",),("rate_gain_over_clinical","mark_gain_over_rate","correct_time_gain_over_shift"),
    )
    def h3_design(name):
        return _patient_summary(
            learned([r for r in h3_p if r.get("status")=="ESTIMATED" and r["design"]==name]),
            ("subject",),("burden_gain_over_common","mark_gain_over_burden"),
        )
    h3_30,h3_2h,h3_6h,h3_1k=(h3_design(x) for x in
        ("physical_1800","physical_7200","physical_21600","event_count_1000"))
    h3_5k=h3_design("event_count_5000")
    summary={
        "dynamic_5min":_sign_summary(d5,"dynamic_gain_over_static"),
        "dynamic_30min":_sign_summary(d30,"dynamic_gain_over_static"),
        "dynamic_120min":_sign_summary(d120,"dynamic_gain_over_static"),
        "learned_rate_residual_5min":_sign_summary(d5,"residual_gain_over_dynamic"),
        "background":_sign_summary(background_p,"background_gain"),
        "next_event_grammar":_sign_summary(g1,"mark_gain_over_q"),
        "next20_grammar":_sign_summary(g20,"mark_gain_over_q"),
        "next20_morphology":_sign_summary(event20,"state_gain_over_q"),
        "next20_morphology_correct_time":_sign_summary(event20,"correct_time_gain_over_shift"),
        "next20_morphology_over_mean":_sign_summary(event20,"mark_gain_over_period_mean"),
        "physical_30min_morphology":_sign_summary(f30,"state_gain_over_q"),
        "physical_30min_correct_time":_sign_summary(f30,"correct_time_gain_over_shift"),
        "physical_30min_over_mean":_sign_summary(f30,"mark_gain_over_period_mean"),
        "h2b_risk_rate":_sign_summary(risk30,"rate_gain_over_clinical"),
        "h2b_risk_mark":_sign_summary(risk30,"mark_gain_over_rate"),
        "h2b_risk_correct_time":_sign_summary(risk30,"correct_time_gain_over_shift"),
        "h2b_risk_over_mean":_sign_summary(risk30,"mark_gain_over_period_mean"),
        "h2b_field_mark":_sign_summary(field30,"mark_gain_over_rate"),
        "h3_burden_30min":_sign_summary(h3_30,"burden_gain_over_common"),
        "h3_mark_30min":_sign_summary(h3_30,"mark_gain_over_burden"),
        "h3_burden_2h":_sign_summary(h3_2h,"burden_gain_over_common"),
        "h3_mark_2h":_sign_summary(h3_2h,"mark_gain_over_burden"),
        "h3_burden_1k":_sign_summary(h3_1k,"burden_gain_over_common"),
        "h3_mark_1k":_sign_summary(h3_1k,"mark_gain_over_burden"),
        "h3_burden_5k":_sign_summary(h3_5k,"burden_gain_over_common"),
        "h3_mark_5k":_sign_summary(h3_5k,"mark_gain_over_burden"),
    }
    def fmt(name, digits=4):
        x=summary[name]; med="NA" if x["median"] is None else f'{x["median"]:+.{digits}f}'
        p="NA" if x["two_sided_sign_p"] is None else f'{x["two_sided_sign_p"]:.3g}'
        return f'{x["n_positive"]}/{x["n"]} 为正，中位 {med}，sign p={p}'
    def state_table():
        lines=["| 患者 | 更新的 seed | 中位选择轮次 | 主状态分母 |",
               "|---|---:|---:|---|"]
        for row in state_training:
            med="NA" if row["median_selected_epoch"] is None else f'{row["median_selected_epoch"]:.0f}'
            lines.append(f'| {SHORT[row["subject"]]} | {row["n_updated_seeds"]}/{row["n_state_seeds"]} | {med} | {"是" if row["primary_state_eligible"] else "否"} |')
        return "\n".join(lines)
    def patient_table(rows, fields, labels):
        lines=["| 患者 | "+" | ".join(labels)+" |","|---|"+"---:|"*len(fields)]
        for row in sorted(rows,key=lambda x:x["subject"]):
            vals=[]
            for field in fields:
                value=_number(row.get(field)); vals.append("NA" if value is None else f'{value:+.4f}')
            lines.append(f'| {SHORT[row["subject"]]} | '+" | ".join(vals)+" |")
        return "\n".join(lines)
    q_table=patient_table(d5,("dynamic_gain_over_static","residual_gain_over_dynamic"),
                          ("q(t)−静态","学习残差−q(t)"))
    grammar_table=patient_table(g1,("mark_gain_over_q","correct_time_gain_over_shift","mark_gain_over_period_mean"),
                                ("m(t)−q(t)","正确时刻−错时","m(t)−FIT 均值"))
    risk_table=patient_table(risk30,("rate_gain_over_clinical","mark_gain_over_rate","correct_time_gain_over_shift","mark_gain_over_period_mean"),
                             ("q(t)−临床","m(t)−q(t)","正确时刻−错时","m(t)−FIT 均值"))
    h3_table=[]
    for label,rows in (("30 min",h3_30),("2 h",h3_2h),("6 h",h3_6h),("1,000 events",h3_1k),("5,000 events",h3_5k)):
        b=_sign_summary(rows,"burden_gain_over_common"); m=_sign_summary(rows,"mark_gain_over_burden")
        bmed="NA" if b["median"] is None else f'{b["median"]:+.4f}'
        mmed="NA" if m["median"] is None else f'{m["median"]:+.4f}'
        h3_table.append(f'| {label} | {b["n"]} | {b["n_positive"]}/{b["n"]} | {bmed} | {m["n"]} | {m["n_positive"]}/{m["n"]} | {mmed} |')
    h3_table="\n".join(["| 暴露尺度 | burden 可采信患者 | burden 为正 | burden 中位 | mark 可采信患者 | mark 为正 | mark 中位 |","|---|---:|---:|---:|---:|---:|---:|"]+h3_table)
    if Q_TRAJECTORY_CAUSAL:
        review_status=(f"**因果重跑产物**（结果根目录 `{OUTPUT_ROOT}`）：q(t) 已改为只用因果的记录段位置，"
                       "下文全部数字来自修复后的完整链条。")
    else:
        review_status=(f"**原始产物，仅供对照**（结果根目录 `{OUTPUT_ROOT}`）：下文所有依赖 q(t) 的数字都来自含非因果 "
                       "`segment_fraction` 特征的原始运行，不得作为结论引用；修复后的数字见 "
                       "`group_event_state_v0_3_5_causal_rerun_{plain,technical}_2026-09-04.md`。")
    review_block=f"""## 0. 审阅修正（2026-09-04）

{review_status}

审阅代码时发现三处承重问题，已修正代码并加回归测试：

1. **（P0）动态负荷 q(t) 用了未来信息。**"记录段位置"特征原来算的是 (t − 段起点) / (段终点 − 段起点)，用到了覆盖段的**结束时刻**。而覆盖段的结束时刻多数正好是下一次发作的起点（审计 2026-09-04：E548 27/42、E922 21/29、E1125 14/21、E1146 13/27、E1096 9/23、E384 8/16、E253 7/21、E583 2/7 个覆盖段结束在发作起点上），所以它等价于"离下一次发作或断录还有多远"的倒计时。q(t) 是 W2–W6 的共同上游：H1 的动态负荷增益、H2b 的风险层、H3 的共同驱动臂以及 m(t) 的输入都受影响；8 位患者中 6 位在这个特征上学到了非零权重，E1096 上它是 5 分钟负荷模型里绝对值最大的权重。现已改为只用"距本段开始已过去的时间"（H3 的两个段位置项同样改法），并按 spec §11 的全局停止条件把整条链在独立目录重跑。
2. **（P1）"正确时刻 vs 错时"对照的锚点不一致。**错时臂只在有远距离供体的锚点上打分，正确时刻臂却在全部锚点上打分，动态负荷（E1096 5 min：136 vs 46 个锚点）、事件语法、功能读出（E548 120 min：5580 vs 3240 个值）和辅助头四处都直接相减。现在四处都在同一批锚点上比较；旧卡片里无法配平的对照记为缺失而不是沿用错配数字。
3. **（P1）H3 数值可采信规则只单向检查。**原规则只拦"子模型误差超过父模型 4 倍"，没有拦"父模型本身发散"；6 h 尺度里 +20 / +87 的"增益"正是父模型发散造成的。规则改为双向 4 倍，旧卡片按新规则重新判定。

另有两处不改数字但需注意的表述问题：`mark_only`/`static_only` 在训练好的物理头和语法臂里是"联合模型关掉 q"的消融，不是单独训练的臂；stepwise 正对照与 E922 的不可估记录沿用原运行。

"""
    plain=f"""# Group-Event State v0.3.5 完整执行报告（白话版）

{review_block}
## 一句话结论

这轮已经把承诺的五层问题真正拆开并全部运行：**可以确认患者内群体 IED 负荷随真实时间变化，且因果的动态负荷基线在部分患者明显优于一个不变水平；完整事件内容还包含少量能预测随后数场事件连续形态的信息。** 但是，完整事件状态没有在队列层面稳定改善成熟 contact-sequence grammar；跨到发作任务后，只在 E1146 和 E548 出现同时超过动态负荷、FIT 常数和错时状态的风险候选；发作早期空间场与 H3 反馈均未建立。

这里的“未建立”不是“生物学不存在”：H2b early-field 只有 2 位患者可估，H3 的 5,000/10,000-event 长尺度没有任何合格独立时间块。

## 1. 这次不是 MVP：实际完成了什么

- 注册 pilot 共 {scale['n_subjects']} 位患者；这些患者的完整记录含 {scale['n_events']:,} 次群体 IED、约 {scale['valid_recording_hours']:.1f} 小时有效覆盖，但本轮所有训练与评价都止于登记的 80% 时间边界，**实际进入分析的是 {scale['n_events_analysed']:,} 次事件、约 {scale['analysed_recording_hours']:.1f} 小时**；这不是完整 41 人底座，也不冒充正式确认队列；
- 8 位患者的静态、因果动态负荷 `q(t)`、学习残差和固定时钟背景 SEEG；
- 重新按时间切分并冻结的患者内 contact-sequence decoder，状态在事件内部每一步调制 contact 与 STOP；
- 完整事件输入，包括 participation/tied groups、连续毫秒 lag、bipolar/CAR 波形、多频带能量与跨频带 lag；
- next-1/5/20-event 与 5/30/120-min 的 H1/H2a；
- 完全冻结间期状态后的发作距离及发作早期场/路径 H2b；
- common-drive、burden-feedback、mark-feedback 的不重叠时间块 H3；
- 6 个网络/优化配方、4 位搜索患者、3 seed，共 72 个正式搜索单元；全局配方锁定后再跑 7 位患者×3 seed 的 21 个最终状态单元。

## 2. 先回答“网络到底有没有学”

搜索最终选择 `compact`，但它相对 base 的 rolling-inner 改善中位只有 −0.00077；学习率、宽度和残差改动之间没有数量级差异，因此不能把科学结果归因于“只差一个神奇超参数”。与此同时，也不能把所有患者都算作成功训练：

{state_table()}

E1096 的 3 个 seed 全选 epoch 0；E583 只有 1/3 真正更新；E922 没有可评分的成熟 decoder 事件。这些单元保留在机器长表，但不进入下文的“学到的事件内容状态”主分母。最终主状态分母是 E253、E548、E1146、E384、E1125 五位。

## 3. H1 第一层：动态负荷确实有，但患者差异很大

在预测未来事件数时，`q(t)` 相对静态水平：5 min 为 {fmt('dynamic_5min')}；30 min 为 {fmt('dynamic_30min')}；120 min 为 {fmt('dynamic_120min')}。这不是整段一个常数：它只用 anchor 之前的事件与有效观测时间，在 2 min、10 min、30 min、2 h、8 h 五个固定时间尺度上更新。

{q_table}

读法有三点：

1. E1096、E583、E1125、E253 的动态水平有明显价值；E548 反而明显更差，说明不能把 rate drift 写成全队列统一方向。
2. 在 `q(t)` 上再加学习的非线性残差只有 {fmt('learned_rate_residual_5min')}，当前可解释的多尺度因果滤波已经吸收了大部分可见负荷变化。
3. 固定时钟背景 SEEG 在 `q(t)` 之外为 {fmt('background')}。在当前低容量背景接口下没有额外信息；这不等于原始背景波形没有信息。

因此 H1-rate 的安全结论是：**存在患者内动态负荷/记录阶段状态，但目前不是普遍胜出的单一模型。**

## 4. H1/H2a：完整事件状态预测了什么

成熟 decoder 的 future-oracle 正对照在 E253、E548、E583 的 grammar 上分别改善约 +0.044、+0.037、+0.015，说明逐步状态接口有能力检出“未来传播场真的已知”时的效应。

真正学到的 `m(t)` 在 next-event contact grammar 上只有 {fmt('next_event_grammar')}；到 next-20 仍是 {fmt('next20_grammar')}。患者结果如下：

{grammar_table}

这说明成熟 contact sequence 的 contact/STOP 语法没有形成队列级增益；E253 与 E548 是正向个例，E384 明显反向。

另一面，若不只看 contact grammar，而把连续 lag、参与场、波形和多频带形态作为一个功能读出，next-20-event 的结果为 {fmt('next20_morphology')}；正确时刻为 {fmt('next20_morphology_correct_time')}；相对 FIT 期平均状态为 {fmt('next20_morphology_over_mean')}。逐端点看，最一致且量级最大的增量来自波形形态和多频带能量；连续 lag 也有 4/5 位同向但较小，峰时和跨频带 lag 不稳定。它支持“短到中程的 event-content predictive memory”，但还没有证明稳定的传播 grammar 状态。

换成固定物理时间的未来 30 min，`m(t)` 在 `q(t)` 之外为 {fmt('physical_30min_morphology')}，正确时刻为 {fmt('physical_30min_correct_time')}，相对 FIT 均值为 {fmt('physical_30min_over_mean')}。三个条件没有在同一队列层面同时成立，120 min 的独立块更少。因此**小时尺度网络表达状态仍未建立**。

## 5. H2b：从间期状态跨到发作

风险 readout 使用单一离散 survival 合同；发作标签只训练冻结状态之上的小型读出，不回流到间期模型。只有同时有训练期发作和 selection 阳性窗的患者—时间尺度才进入风险分母。

30 min 风险层中，动态负荷相对临床/历史基线为 {fmt('h2b_risk_rate')}；完整事件状态在 `q(t)` 之上为 {fmt('h2b_risk_mark')}；正确时刻为 {fmt('h2b_risk_correct_time')}；相对 FIT 均值为 {fmt('h2b_risk_over_mean')}。

{risk_table}

E1146 与 E548 在 5–120 min 各层都同时满足“超过 q、超过 FIT 均值、正确时刻优于错时”；E253 为反向，E384 虽有增益但错时状态更好。由于主分母只有 4 位，而且效应很小，当前只能称 **两位患者的 development-level 跨任务候选**，不能写成队列级发作易感状态。

发作早期空间场/路径只有 E548、E1146 可估；30 min lead 的 `m(t)` 额外增量为 {fmt('h2b_field_mark')}，没有检出任何增量。这个结果首先是低效力/低灵敏度，不是发作路径不受间期状态影响的生物学阴性。

## 6. H3：IED 是否反过来塑造状态

H3 比较三种等槽位模型：M0 只含 pre-exposure 的冻结 `q/m`、暴露时长、时钟和覆盖段位置；M1 再加 IED burden；M2 再加在 FIT 中扣除 burden 后的事件内容。暴露+未来块严格不重叠，基本分母是独立块而不是滑动事件窗。最终质检发现，低支持下新增槽有时会让 INNER 或 SELECTION 误差相对父模型放大数十倍；因此统一使用“任一层超过父模型 4 倍即数值不可采信”的规则。该规则不看效应方向，既排除偶然好看的发散，也排除偶然难看的发散；原始数值仍保留在机器长表。

{h3_table}

30 min、2 h 和 1,000-event 三个层级在排除不稳定拟合后仍没有一致方向；6 h 支持极少，5,000/10,000-event 没有合格的独立时间块。因此 H3 当前结论是：**没有获得 event-feedback-like predictive dependence 的稳定支持；用户提出的数千至上万次累积尺度在本批住院记录中仍然没有被有效检验。**

事件前后功能状态的 observer innovation 也没有给出正向累积线索：next-20 时，当前 extent 与能量的关联在稳健训练患者中均为 0/5 正。它只能描述模型如何更新 belief，不能当因果效应。

## 7. 五层证据现在在哪里

| 层级 | 当前力度 | 能说什么 |
|---|---|---|
| 1. 患者/记录阶段慢水平 | 中等 development 支持 | 静态差异和部分患者的动态 rate drift 都真实可预测 |
| 2. 事件历史在简单 `q(t)` 之外有信息 | 有限、端点依赖 | next-20 连续形态 4/5 正，但 contact grammar 仅 2/5 |
| 3. 时刻特异的持续状态 | 弱 | 多事件形态有 correct-time 方向，固定 30/120 min 未形成共同闭环 |
| 4. 网络 repertoire/传播状态 | 未建立 | 成熟 contact/STOP grammar 未在多数患者改善 |
| 5. 跨任务发作易感状态 | 两位患者候选 | E1146、E548 风险层通过三项对照；early ictal field 未复现 |

H3 不属于这五层的自动下一阶：目前无稳定支持，超长尺度不可估。

## 8. 本轮执行中修掉的承重问题

- H2b 的发作恰好位于覆盖段右边界，旧半开区间把全部训练标签变成 0；已改为 `(left, right]`。
- 风险评分曾要求整段 horizon 都在覆盖内，从而又把“horizon 内已观察到发作”的 preictal anchor 删除；现允许已观察发作终止随访，未发作的不完整区间仍删去。
- 120 min 的不可估 null arm 现在显式保留，不再因缺臂而被汇总器跳过。
- q 轨迹临时与最终副本 30/30 逐数组完全一致；患者—decoder 映射和所有结果来源均有机器清单。
- 定向回归测试 66 项全过。它们证明实现符合合同，不替代上述人体证据。

## 9. 当前最终判断

v0.3.5 可以作为**完整 development 执行与科学分层**收口，而不能作为 H1–H3 已确立的终局：H1 的动态负荷层最可靠；H1/H2a 的事件内容层有多事件连续形态信号但没有稳定 contact grammar；H2b 只有两位患者风险候选；H3 仍未决。formal/sealed 分区全程未打开。
"""
    technical=f"""# Group-Event State v0.3.5 完整执行报告（技术版）

{review_block}
## 1. 结论边界

本阶段是完整 development execution，不是 formal confirmation。工程完成、可估性、assay sensitivity 与人体科学支持分别报告。所有 seed 先在患者内取中位；seed 不是独立样本。epoch-0 或仅 1/3 seed 更新的状态保留在长表，但不进入 learned-state 主分母。

## 2. 数据、拆分与模型

- 本轮注册输入：{scale['n_subjects']} 位患者。完整记录 {scale['n_events']:,} 次群体 IED / {scale['valid_recording_hours']:.1f} 小时有效覆盖；**80% 边界之前实际用于训练与评价的是 {scale['n_events_analysed']:,} 次事件 / {scale['analysed_recording_hours']:.1f} 小时**。完整 41 人底座未被冒充为本轮已训练队列。
- 注册患者：{', '.join(SHORT[s] for s in SUBJECTS)}；E922 因成熟 decoder 在注册评价窗无事件，W3–W6 状态链不可估。
- FIT 20–60%、rolling INNER 60–70%、一次性 SELECTION 70–80%；development/formal/sealed 均未读取。
- `q(t)`：固定 2 min/10 min/30 min/2 h/8 h 因果 bank，负二项 future-count likelihood，静态截距为嵌套特例。
- `m(t)`：完整群体事件编码；pre-event state 预测当前/未来，事件观测后才更新；future rollout 不用真实未来事件 teacher forcing。
- frozen contact decoder：每个 recurrence step 使用低秩 FiLM、contact-specific shift 与 STOP shift；decoder 主干不更新。
- H2b：5-min grid discrete survival 和 early-ictal field/path；间期 producer 冻结。
- H3：不重叠 exposure+30-min future block，M0/M1/M2 参数槽位相同；M0 已含冻结 pre-state/q、duration、clock 和 segment position。

## 3. 训练与优化审计

完整事件状态搜索为 6 recipes × 4 subjects × 3 seeds = 72 units。`compact` 由 FIT/INNER 患者中位 rank 选出；相对 base 的 inner delta 为 −0.0007739。最终 21 units 均完成且无 OOM retry；预算审计没有 final-two-epoch unit，结论为 `ORIGINAL_BUDGET_ADEQUATE`。这排除了“普遍只因步数太少”的解释，但 E1096 0/3 和 E583 1/3 仍是患者级训练/选择失败。

{state_table()}

## 4. 机器汇总

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

### 4.1 动态负荷患者表（5 min）

{q_table}

### 4.2 next-event contact grammar（稳健训练患者）

{grammar_table}

### 4.3 30-min seizure risk（有阳性窗且稳健训练患者）

{risk_table}

### 4.4 H3 独立块汇总

{h3_table}

所有 H3 原始拟合均保留。新增反馈臂在两种情况下记为不可采信（两条都不看增益符号）：一、INNER 或 SELECTION 的 MSE 非有限，或与嵌套父模型相差 4 倍以上（**任一方向**——父模型自己发散、子模型有界，同样排除）；二、父或子任一臂的 MSE 超过零模型 4 倍。零模型为 FIT 均值预测器；审阅前写出的卡片未存该字段，回退到"FIT 标准化后结局量方差按构造为 1"的单位参照。6 h 低支持结果不进入核心效应图。5k/10k 设计没有满足 ≥4/2/2 FIT/INNER/SELECTION 独立块的患者。

## 5. Assay sensitivity 与科学读法

future-oracle 在 E253/E548/E583 的 contact grammar 改善分别约 +0.0442/+0.0375/+0.0150，说明状态到逐步 decoder 的路径不是结构零。但 oracle 是泄漏答案的正对照，只证明接口灵敏，不证明真实状态存在。真实 `m(t)` 必须同时看：超过 `q(t)`、correct-time 超过 block-shift、超过 FIT-period mean；三项不同步时不能升级为时刻特异状态。

H2b 风险行若 `n_positive=0`，仍保留在 `h2b_risk.csv` 以审计校准，但不进入风险方向统计。early-field 的基本分母是 held-out seizure；只有 E548/E1146 有可估读出。H3 的基本分母是不重叠时间块，endpoint 先在患者内合并后再数患者。

## 6. 运行中发现并修复的 P0/P1

1. H2b person-period label 使用 `[left,right)`，而 seizure onset 正好定义 coverage segment 右边界，导致训练阳性全为零；改为 `(left,right]` 并加边界回归测试。
2. H2b horizon eligibility 要求完整随访，错误删除了 horizon 内已经观察到 seizure 的 anchor；改为“完整随访或在截尾前已观察到发作”。
3. selection 随访硬限制在 80% 边界，禁止把 80% 后无事件时间当作 negative exposure。
4. functional 120-min 缺少 block-shift/period-mean 时，显式输出 `NOT_ESTIMABLE` arm。
5. finalizer 只对有至少一个阳性 seizure window 的 patient×horizon 计算 primary risk summary。
6. functional/seizure maintenance CLI 改为跟随 card 中登记的 trajectory，不再猜本地同名文件。

## 7. 产物与复现

- `dynamic_baseline.csv/json`：静态、确定性 q、学习残差、block-shift。
- `state_training.csv`：每位患者实际更新 seed 和主状态资格。
- `stepwise_decoder.csv/json`、`stepwise_oracle.csv`：逐步 frozen-decoder 接口与正对照。
- `h1_h2a_{{grammar,functional,auxiliary}}.csv`：next-1/5/20 与 5/30/120 min 全端点。
- `h2b_{{risk,field,support}}.csv`：风险、early field/path 和 seizure 分母。
- `h3_{{models,innovation}}.csv`：M0/M1/M2 与 observer innovation。
- 每张核心图同时有 PNG、矢量 PDF 和 metadata；`figures/README.md` 给出逐图阅读合同。
- 所有 `correct_time_gain_over_shift` 列自 2026-09-04 起为同锚点对照（错时臂与正确时刻臂都只在有远距离供体的锚点上打分）；`correct_time_support_matched=False` 的行表示旧卡片无法配平、对照记为缺失。
- `h3_models.csv` 的 `*_admissible` 列由 finalizer 从各臂 MSE 重新判定：双向 4 倍父子比值，加上父/子任一臂超过零模型 MSE 4 倍即不可采信。零模型为 FIT 均值预测器（卡片 `null_fit_mean`）；审阅前写出的卡片没有该字段，回退到"标准化结局量方差按构造为 1"的单位参照，`absolute_reference` 列记录用的是哪一种。

运行 `scripts/supervise_group_event_state_v035_reporting.py` 可重建审计、机器表、图和两份报告。最终 targeted test suite 为 66 passed。所有 `development_targets_read`/`sealed_partition_opened` 标志为 false。
"""
    return plain,technical


def main() -> None:
    out=OUTPUT_ROOT/"final_reports"; figures=out/"figures"; figures.mkdir(parents=True,exist_ok=True)
    _validate_registered_designs()
    rate,bg=dynamic_rows(); step,oracle=stepwise_rows(); grammar,functional,aux=h1_h2a_rows()
    risk,field,h2support=h2b_rows(); h3,innovation=h3_rows()
    state_training=_state_training_rows(); primary_state=_primary_state_subjects()
    tables={"dynamic_baseline":rate,"background_baseline":bg,"stepwise_decoder":step,
            "stepwise_oracle":oracle,"h1_h2a_grammar":grammar,"h1_h2a_functional":functional,
            "h1_h2a_auxiliary":aux,"h2b_risk":risk,"h2b_field":field,"h2b_support":h2support,
            "h3_models":h3,"h3_innovation":innovation,"state_training":state_training}
    for name,rows in tables.items(): _write_csv(out/f"{name}.csv",rows)
    rate_p=_patient_summary(
        rate,("subject","horizon_seconds"),
        ("dynamic_gain_over_static","residual_gain_over_static",
         "residual_gain_over_dynamic","correct_time_gain_over_shift"),
    )
    bg_p=_patient_summary(bg,("subject",),("background_gain","correct_time_gain_over_shift"))
    step_p=_patient_summary(step,("subject","metric"),("rate_gain_over_static","correct_time_gain_over_shift"))
    grammar_p=_patient_summary(
        grammar,("subject","horizon","metric"),
        ("rate_gain_over_static","mark_only_gain_over_static","mark_gain_over_q",
         "correct_time_gain_over_shift","mark_gain_over_period_mean"),
    )
    functional_p=_patient_summary(
        functional,("subject","family","horizon","endpoint"),
        ("rate_gain_over_static","mark_only_gain_over_static","state_gain_over_q",
         "correct_time_gain_over_shift","mark_gain_over_period_mean"),
    )
    aux_p=_patient_summary(
        aux,("subject","horizon","endpoint"),
        ("rate_gain_over_static","mark_only_gain_over_static","mark_gain_over_q","correct_time_gain_over_shift"),
    )
    risk_p=_patient_summary(risk,("subject","horizon"),("n_anchors","n_positive","rate_gain_over_clinical","mark_gain_over_clinical","mark_gain_over_rate","correct_time_gain_over_shift","mark_gain_over_period_mean"))
    risk_primary=[r for r in risk_p if r.get("subject") in primary_state and
                  (_number(r.get("n_positive")) or 0)>0]
    field_p=_patient_summary(field,("subject","lead","endpoint"),("rate_gain_over_clinical","mark_gain_over_clinical","mark_gain_over_rate","correct_time_gain_over_shift","mark_gain_over_period_mean"))
    h3_p=_patient_summary(h3,("subject","design","endpoint","status"),
                          ("burden_gain_over_common","mark_gain_over_burden",
                           "burden_gain_over_common_raw","mark_gain_over_burden_raw",
                           "burden_impulse","mark_impulse"))
    innovation_p=_patient_summary(innovation,("subject","horizon"),("signed_functional_innovation_mean","association_with_current_extent","association_with_current_multiband_energy"))
    machine={
      "dynamic_baseline":{"per_patient":rate_p,"background_per_patient":bg_p,"stepwise_rate_per_patient":step_p},
      "stepwise_decoder":{"per_patient":step_p,"future_oracle":_patient_summary(oracle,("subject","metric"),("oracle_gain_over_static",))},
      "h1_h2a":{"grammar":grammar_p,"functional":functional_p,"auxiliary":aux_p,
                 "state_training":state_training,"primary_state_subjects":sorted(primary_state)},
      "h2b":{"risk":risk_p,"primary_risk":risk_primary,"early_field":field_p,"support":h2support,
              "risk_primary_rule":"at least one positive seizure window per patient-horizon"},
      "h3":{"model_comparisons":h3_p,"functional_innovation":innovation_p},
    }
    for name,payload in machine.items():
        atomic_json(out/f"{name}.json",{"format":f"group_event_state_v0_3_5_{name}_summary_v1","seed_aggregation":"median within patient first","data":payload,"development_targets_read":False,"sealed_partition_opened":False})
    plot_dynamic(rate,bg,figures); plot_h1_h2a(grammar,functional,aux,figures); plot_h2b(risk,field,figures); plot_h3(h3,innovation,figures)
    (figures/"README.md").write_text("""# v0.3.5 核心图

### fig_v035_dynamic_baseline.png
展示静态校准、因果动态负荷 `q(t)`、时刻对齐和连续背景增量。A 是单患者真实时间轨迹；B 是动态负荷相对静态及学习残差相对 `q(t)` 的患者级增量；C 检验动态轨迹的正确时刻；D 检验背景 SEEG 是否在事件历史之外增加信息。

**关注点**：动态负荷是否稳定胜过静态水平；这不等于传播形态状态。

### fig_v035_h1_h2a.png
展示完整事件内容状态 `m(t)` 在 `q(t)` 之外，对 next-1/5/20 event、未来 5/30/120 min 形态和 same-prefix 后续路径的增量。

**关注点**：正增量必须与正确时刻优于 block shift 一起读。

### fig_v035_h2b.png
展示冻结间期状态对发作距离和发作早期空间场/路径的跨任务增量。

**关注点**：分别看 rate-linked 与 mark-specific 信息；基本分母是患者/发作，不是事件行。

### fig_v035_h3.png
展示 common-drive、事件负荷 feedback、事件内容 feedback 的嵌套预测比较，以及事件前后功能状态 innovation。

**关注点**：只解释为 feedback-like predictive dependence，不作人体因果宣称。
""",encoding="utf-8")
    plain,technical=_report_text(rate_p,bg_p,grammar_p,functional_p,risk_primary,field_p,h3_p,state_training)
    plain_path=ROOT/f"docs/archive/topic5/group_event_state_v0_3_5_{REPORT_TAG}_plain_{REPORT_DATE}.md"
    technical_path=ROOT/f"docs/archive/topic5/group_event_state_v0_3_5_{REPORT_TAG}_technical_{REPORT_DATE}.md"
    plain_path.write_text(plain,encoding="utf-8"); technical_path.write_text(technical,encoding="utf-8")
    atomic_json(out/"scope_summary.json",{"format":"group_event_state_v0_3_5_complete_summary_v1","subjects":list(SUBJECTS),"counts":{k:len(v) for k,v in tables.items()},"reports":[str(plain_path),str(technical_path)],"figures":[str(p) for p in sorted(figures.glob('*.png'))],"development_targets_read":False,"sealed_partition_opened":False})
    print(json.dumps({"out":str(out),"counts":{k:len(v) for k,v in tables.items()}},indent=2))


if __name__ == "__main__":
    main()

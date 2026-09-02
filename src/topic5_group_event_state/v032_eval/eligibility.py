"""A-priori endpoint eligibility: data support only, never a model result."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .partition import EVAL_PHASES, REFIT_PHASE
from .timeline import EvalTimeline

ELIGIBILITY_PHASES = EVAL_PHASES + (REFIT_PHASE,)


def phase_block_counts(tl: EvalTimeline, horizon: float) -> dict[str, int]:
    """Non-overlapping ``horizon`` windows that fit inside (segment ∩ phase) pieces."""

    out = {}
    for phase in ELIGIBILITY_PHASES:
        lo, hi = tl.partition.bounds(phase)
        total = 0
        for seg in tl.segments:
            a, b = max(seg.start_epoch, lo), min(seg.stop_epoch, hi)
            if b > a:
                total += int(np.floor((b - a) / float(horizon)))
        out[phase] = total
    return out


def count_statistics(tl: EvalTimeline, phase: str, horizon_index: int) -> dict[str, Any]:
    idx = tl.anchor_indices(phase, horizon_index)
    if idx.size == 0:
        return {"n_anchors": 0, "mean": None, "variance": None, "dispersion": None, "n_events_in_windows": 0}
    c = tl.window_counts(idx, horizon_index).astype(np.float64)
    mean = float(c.mean())
    var = float(c.var(ddof=1)) if c.size > 1 else None
    return {
        "n_anchors": int(idx.size),
        "mean": mean,
        "variance": var,
        "dispersion": (var / mean) if (var is not None and mean > 0) else None,
        "n_events_in_windows": int(c.sum()),
    }


def seizure_clusters(seizures: Sequence[Mapping[str, Any]], gap_seconds: float) -> list[list[int]]:
    onsets = sorted((float(s["onset_epoch"]), i) for i, s in enumerate(seizures))
    clusters: list[list[int]] = []
    last_t = None
    for t, i in onsets:
        if last_t is None or t - last_t > gap_seconds:
            clusters.append([i])
        else:
            clusters[-1].append(i)
        last_t = t
    return clusters


def contact_entropy_bits(participation: np.ndarray) -> float | None:
    mass = participation.sum(axis=0).astype(np.float64)
    if mass.sum() <= 0:
        return None
    p = mass / mass.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def compute_eligibility(tl: EvalTimeline, cfg: Mapping[str, Any]) -> dict[str, Any]:
    rules = cfg["eligibility"]
    horizons = tl.horizons_seconds
    h30 = horizons.index(1800.0)
    h120 = horizons.index(7200.0)
    labels = tl.event_phase_labels()
    events_by_phase = {name: int((labels == i).sum()) for i, name in enumerate(EVAL_PHASES)}
    events_by_phase[REFIT_PHASE] = events_by_phase["base_fit"] + events_by_phase["inner_val"]
    valid_hours = {}
    for phase in ELIGIBILITY_PHASES:
        lo, hi = tl.partition.bounds(phase)
        valid_hours[phase] = float(sum(max(min(s.stop_epoch, hi) - max(s.start_epoch, lo), 0.0) for s in tl.segments) / 3600.0)
    blocks_30 = phase_block_counts(tl, 1800.0)
    blocks_120 = phase_block_counts(tl, 7200.0)
    blocks_5 = phase_block_counts(tl, 300.0)
    counts = {
        "300s": {p: count_statistics(tl, p, horizons.index(300.0)) for p in ELIGIBILITY_PHASES},
        "1800s": {p: count_statistics(tl, p, h30) for p in ELIGIBILITY_PHASES},
        "7200s": {p: count_statistics(tl, p, h120) for p in ELIGIBILITY_PHASES},
    }
    positive_k = tl.positive_k_mask()
    positive_k_by_phase = {name: int(positive_k[labels == i].sum()) for i, name in enumerate(EVAL_PHASES)}
    continuation_steps_by_phase = {
        name: int(np.clip(tl.group_count[labels == i] - 1, 0, None).sum()) for i, name in enumerate(EVAL_PHASES)
    }
    base_mask = labels == 0
    vocab_part = tl.participation[:, tl.vocab_mask]
    entropy = contact_entropy_bits(vocab_part[base_mask])
    max_entropy = float(np.log2(max(tl.n_vocab, 1)))
    part_mass = tl.participation.sum(axis=1).astype(float)
    vocab_mass = vocab_part.sum(axis=1).astype(float)
    support_stability = {}
    for i, name in enumerate(EVAL_PHASES):
        m = labels == i
        support_stability[name] = float(vocab_mass[m].sum() / part_mass[m].sum()) if m.any() and part_mass[m].sum() > 0 else None

    gap = float(rules["seizure_cluster_gap_seconds"])
    clusters = seizure_clusters(tl.seizures, gap)
    seizure_phase = {name: 0 for name in EVAL_PHASES}
    for s in tl.seizures:
        seizure_phase[tl.partition.phase_of(float(s["onset_epoch"]))] += 1

    r30 = rules["count_30min"]
    reasons_30 = []
    if blocks_30["base_fit"] < r30["base_fit_blocks"]:
        reasons_30.append(f"base_fit_blocks_30min={blocks_30['base_fit']}<{r30['base_fit_blocks']}")
    if blocks_30["inner_val"] < r30["inner_val_blocks"]:
        reasons_30.append(f"inner_val_blocks_30min={blocks_30['inner_val']}<{r30['inner_val_blocks']}")
    if blocks_30["dev_val"] < r30["dev_val_blocks"]:
        reasons_30.append(f"dev_val_blocks_30min={blocks_30['dev_val']}<{r30['dev_val_blocks']}")
    if blocks_30["dev_test"] < r30["dev_test_blocks"]:
        reasons_30.append(f"dev_test_blocks_30min={blocks_30['dev_test']}<{r30['dev_test_blocks']}")
    if events_by_phase["dev_test"] < r30["dev_test_events"]:
        reasons_30.append(f"dev_test_events={events_by_phase['dev_test']}<{r30['dev_test_events']}")
    for phase in ("base_fit", "inner_val", "dev_val", "dev_test"):
        if counts["1800s"][phase]["n_anchors"] == 0:
            reasons_30.append(f"no_eligible_30min_anchor_in_{phase}")

    r120 = rules["count_120min"]
    reasons_120 = []
    for phase, key in (("base_fit", "base_fit_blocks"), ("inner_val", "inner_val_blocks"),
                       ("dev_val", "dev_val_blocks"), ("dev_test", "dev_test_blocks")):
        if blocks_120[phase] < r120[key]:
            reasons_120.append(f"{phase}_blocks_120min={blocks_120[phase]}<{r120[key]}")
        if counts["7200s"][phase]["n_anchors"] == 0:
            reasons_120.append(f"no_eligible_120min_anchor_in_{phase}")

    rh = rules["h2a"]
    reasons_h2a = []
    if tl.n_vocab < rh["min_vocab_contacts"]:
        reasons_h2a.append(f"vocab_contacts={tl.n_vocab}<{rh['min_vocab_contacts']}")
    if events_by_phase["base_fit"] < rh["base_fit_events"]:
        reasons_h2a.append(f"base_fit_events={events_by_phase['base_fit']}<{rh['base_fit_events']}")
    if positive_k_by_phase["dev_test"] < rh["dev_test_positive_k_events"]:
        reasons_h2a.append(f"dev_test_positive_k_events={positive_k_by_phase['dev_test']}<{rh['dev_test_positive_k_events']}")
    if support_stability["dev_test"] is None or support_stability["dev_test"] < rh["support_stability"]:
        reasons_h2a.append(f"support_stability_dev_test={support_stability['dev_test']}<{rh['support_stability']}")
    if events_by_phase["inner_val"] < cfg["grammar"]["min_inner_events"]:
        reasons_h2a.append(f"inner_val_events={events_by_phase['inner_val']}<{cfg['grammar']['min_inner_events']}")

    return {
        "subject": tl.subject,
        "dataset": tl.dataset,
        "n_contacts_legacy": int(tl.n_contacts),
        "n_contacts_vocabulary": int(tl.n_vocab),
        "valid_hours_total": float(tl.partition.total_recorded_seconds / 3600.0),
        "valid_hours_by_phase": valid_hours,
        "events_by_phase": events_by_phase,
        "blocks_5min_by_phase": blocks_5,
        "blocks_30min_by_phase": blocks_30,
        "blocks_120min_by_phase": blocks_120,
        "count_statistics": counts,
        "grammar_prefix_events": events_by_phase["base_fit"],
        "positive_k_events_by_phase": positive_k_by_phase,
        "continuation_steps_by_phase": continuation_steps_by_phase,
        "contact_entropy_bits_base_fit": entropy,
        "contact_entropy_normalised": (entropy / max_entropy) if (entropy is not None and max_entropy > 0) else None,
        "support_stability_by_phase": support_stability,
        "seizures": {
            "n_total": len(tl.seizures),
            "n_clusters": len(clusters),
            "cluster_gap_seconds": gap,
            "n_by_onset_phase": seizure_phase,
            "n_in_dev_val_plus_dev_test": seizure_phase["dev_val"] + seizure_phase["dev_test"],
            "h2b_status": "data support described only; no seizure model is run in v0.3.2",
        },
        "eligibility": {
            "count_30min_primary": {"eligible": not reasons_30, "reasons": reasons_30},
            "count_120min_secondary": {"eligible": not reasons_120, "reasons": reasons_120},
            "h2a_positive_k_prefix": {"eligible": not reasons_h2a, "reasons": reasons_h2a},
            "count_5min_short_range_only": {"eligible": counts["300s"]["dev_test"]["n_anchors"] > 0 and counts["300s"]["inner_val"]["n_anchors"] > 0,
                                             "note": "5 min gains can only be called short-range memory"},
        },
        "rules": dict(rules),
        "frozen_before_any_model_result": True,
    }


def eligibility_csv_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    e = entry["eligibility"]
    c30 = entry["count_statistics"]["1800s"]
    c120 = entry["count_statistics"]["7200s"]
    return {
        "subject": entry["subject"],
        "dataset": entry["dataset"],
        "n_contacts_legacy": entry["n_contacts_legacy"],
        "n_contacts_vocab": entry["n_contacts_vocabulary"],
        "valid_hours_total": round(entry["valid_hours_total"], 3),
        "valid_hours_base_fit": round(entry["valid_hours_by_phase"]["base_fit"], 3),
        "valid_hours_inner_val": round(entry["valid_hours_by_phase"]["inner_val"], 3),
        "valid_hours_dev_val": round(entry["valid_hours_by_phase"]["dev_val"], 3),
        "valid_hours_dev_test": round(entry["valid_hours_by_phase"]["dev_test"], 3),
        "events_base_fit": entry["events_by_phase"]["base_fit"],
        "events_inner_val": entry["events_by_phase"]["inner_val"],
        "events_dev_val": entry["events_by_phase"]["dev_val"],
        "events_dev_test": entry["events_by_phase"]["dev_test"],
        "blocks30_base_fit": entry["blocks_30min_by_phase"]["base_fit"],
        "blocks30_inner_val": entry["blocks_30min_by_phase"]["inner_val"],
        "blocks30_dev_val": entry["blocks_30min_by_phase"]["dev_val"],
        "blocks30_dev_test": entry["blocks_30min_by_phase"]["dev_test"],
        "blocks120_base_fit": entry["blocks_120min_by_phase"]["base_fit"],
        "blocks120_inner_val": entry["blocks_120min_by_phase"]["inner_val"],
        "blocks120_dev_val": entry["blocks_120min_by_phase"]["dev_val"],
        "blocks120_dev_test": entry["blocks_120min_by_phase"]["dev_test"],
        "count30_dev_test_mean": c30["dev_test"]["mean"],
        "count30_dev_test_dispersion": c30["dev_test"]["dispersion"],
        "count120_dev_test_mean": c120["dev_test"]["mean"],
        "count120_dev_test_dispersion": c120["dev_test"]["dispersion"],
        "grammar_prefix_events": entry["grammar_prefix_events"],
        "positive_k_dev_test": entry["positive_k_events_by_phase"]["dev_test"],
        "contact_entropy_bits": entry["contact_entropy_bits_base_fit"],
        "support_stability_dev_test": entry["support_stability_by_phase"]["dev_test"],
        "n_seizures": entry["seizures"]["n_total"],
        "n_seizure_clusters": entry["seizures"]["n_clusters"],
        "eligible_count_30min": e["count_30min_primary"]["eligible"],
        "eligible_count_120min": e["count_120min_secondary"]["eligible"],
        "eligible_h2a": e["h2a_positive_k_prefix"]["eligible"],
        "reasons_30min": "|".join(e["count_30min_primary"]["reasons"]),
        "reasons_120min": "|".join(e["count_120min_secondary"]["reasons"]),
        "reasons_h2a": "|".join(e["h2a_positive_k_prefix"]["reasons"]),
    }

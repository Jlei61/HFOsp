"""INTERICTAL_REPERTOIRE_RETAINED, scored from the frozen Fig.4 contracts.

Nothing here is fitted. The direction classifier, the shaft-aware embedding, the
OOD thresholds and the reference quantiles all come from artifacts that were
frozen before the Z/M round existed; this module only applies them to the
returned pre-onset events of a Z/M trajectory and reports the full distribution
rather than a single verdict flag.

Two boundaries are enforced in code rather than left to the caller:

* the runaway interval is never labelled. Only events that ended before the
  transition are scored, so no A/B call can be attributed to the high state.
* the displayed event is the LAST qualifying event in time. "Clean" is a
  diagnostic produced by the frozen filter, never a hand-picked selection.
"""
from __future__ import annotations

import numpy as np

from src.topic4_d6_natural_kmeans import natural_kmeans
from src.topic4_nlc_pathway_mechanism import (
    formal_mode_assignments, network_mode_endpoints)
from src.topic4_shaft_aware_direction import all_event_shaft_participation

NOT_EVALUABLE = "NOT_EVALUABLE_FROM_EXISTING_ARTIFACTS"
MODE_NAMES = {0: "TA_like", 1: "TB_like"}


def shaft_groups(contact_contract):
    groups = {}
    for row in contact_contract["contacts"]:
        groups.setdefault(row["shaft_id"], []).append(int(row["contact_index"]))
    return groups


def _spatial_range_mm(onsets_row, contact_xy):
    recruited = np.flatnonzero(np.isfinite(onsets_row))
    if len(recruited) < 2:
        return 0.0
    points = np.asarray(contact_xy, float)[recruited]
    delta = points[:, None, :] - points[None, :, :]
    return float(np.sqrt((delta ** 2).sum(axis=-1)).max())


def _rank_profile(onsets_row):
    """Dense within-event rank of the recruited contacts; NaN elsewhere."""
    row = np.asarray(onsets_row, float)
    profile = np.full(row.shape, np.nan)
    recruited = np.flatnonzero(np.isfinite(row))
    if len(recruited):
        order = np.argsort(np.argsort(row[recruited]))
        profile[recruited] = order.astype(float)
    return profile


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    if int(both.sum()) < 3:
        return float("nan"), int(both.sum())
    x, y = a[both], b[both]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan"), int(both.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(both.sum())


def score_events(onsets, event_returned, event_before_onset, *, groups,
                 embedding, classifier, contact_xy, contact_names,
                 event_t_on_ms=None, event_t_off_ms=None, t_ictal_ms=None):
    """Score every returned pre-onset event; never label the runaway interval."""
    onsets = np.asarray(onsets, float)
    returned = np.asarray(event_returned, bool)
    before = np.asarray(event_before_onset, bool)
    eligible = returned & before
    if t_ictal_ms is not None and event_t_off_ms is not None:
        strict = np.asarray(event_t_off_ms, float) < float(t_ictal_ms)
    else:
        strict = np.ones_like(eligible)

    assigned = formal_mode_assignments(
        onsets, eligible, groups=groups, embedding=embedding,
        classifier=classifier)
    labels = np.asarray(assigned["labels"], int)
    clean = np.asarray(assigned["clean"], bool)
    ood = np.asarray(assigned["ood"], bool)
    confidence = np.abs(np.asarray(assigned["probability_B"], float) - 0.5) * 2.0

    rows = []
    for index in range(len(onsets)):
        scored = bool(eligible[index])
        rows.append({
            "event_index": int(index),
            "t_on_ms": (float(event_t_on_ms[index])
                        if event_t_on_ms is not None else None),
            "t_off_ms": (float(event_t_off_ms[index])
                         if event_t_off_ms is not None else None),
            "returned": bool(returned[index]),
            "before_onset": bool(before[index]),
            "before_t_ictal": bool(strict[index]),
            "scored": scored,
            # A label on an event that is not scored would be a label on the
            # transition; it is withheld, not merely ignored downstream.
            "mode": (MODE_NAMES[int(labels[index])] if scored else None),
            "classifier_confidence": (float(confidence[index]) if scored else None),
            "ood": (bool(ood[index]) if scored else None),
            "ood_distance": (float(assigned["ood_distance"][index])
                             if scored else None),
            "clean": bool(clean[index]),
            "n_recruited_contacts": int(np.isfinite(onsets[index]).sum()),
            "spatial_range_mm": _spatial_range_mm(onsets[index], contact_xy),
            "recruited_contacts": [contact_names[c] for c in
                                   np.flatnonzero(np.isfinite(onsets[index]))],
        })
    return {"rows": rows, "labels": labels, "clean": clean, "ood": ood,
            "eligible": eligible, "strict": strict, "assigned": assigned}


def _nanmean_allow_empty(values):
    """Column mean over recruited entries; NaN where nothing was recruited."""
    values = np.asarray(values, float)
    counts = np.sum(np.isfinite(values), axis=0)
    totals = np.nansum(values, axis=0)
    out = np.full(values.shape[1], np.nan)
    good = counts > 0
    out[good] = totals[good] / counts[good]
    return out


def rank_profile_similarity(onsets, labels, clean):
    """Per-mode agreement of every clean event with that mode's mean profile."""
    onsets = np.asarray(onsets, float)
    profiles = np.vstack([_rank_profile(row) for row in onsets]) if len(onsets) \
        else np.zeros((0, 0))
    output = {}
    for mode, name in MODE_NAMES.items():
        members = np.flatnonzero(clean & (labels == mode))
        if len(members) < 2:
            output[name] = {"status": NOT_EVALUABLE, "n_events": int(len(members))}
            continue
        with np.errstate(invalid="ignore"):
            # a contact recruited by no member of the mode gives an all-NaN
            # column; that is the expected "not recruited" case, not an error
            centroid = _nanmean_allow_empty(profiles[members])
        values, coverage = [], []
        for index in members:
            # leave-one-out, so an event is not compared with a centroid it
            # helped define
            others = [j for j in members if j != index]
            reference = _nanmean_allow_empty(profiles[others])
            rho, n = _spearman(profiles[index], reference)
            values.append(rho)
            coverage.append(n)
        finite = np.asarray([v for v in values if np.isfinite(v)], float)
        output[name] = {
            "n_events": int(len(members)),
            "median_leave_one_out_spearman": (float(np.median(finite))
                                              if len(finite) else float("nan")),
            "n_evaluable": int(len(finite)),
            "median_common_contacts": float(np.median(coverage)),
            "mean_rank_profile": [None if not np.isfinite(v) else float(v)
                                  for v in centroid],
        }
    return output


def evaluate_repertoire(onsets, event_returned, event_before_onset, *, groups,
                        embedding, classifier, contact_xy, contact_names,
                        gate, duration_ms, event_t_on_ms=None,
                        event_t_off_ms=None, t_ictal_ms=None):
    """The four historical clauses, plus the full returned-event distribution."""
    scored = score_events(
        onsets, event_returned, event_before_onset, groups=groups,
        embedding=embedding, classifier=classifier, contact_xy=contact_xy,
        contact_names=contact_names, event_t_on_ms=event_t_on_ms,
        event_t_off_ms=event_t_off_ms, t_ictal_ms=t_ictal_ms)
    onsets = np.asarray(onsets, float)
    eligible = scored["eligible"]
    labels, clean, ood = scored["labels"], scored["clean"], scored["ood"]

    n_returned = int(eligible.sum())
    mode_counts = [int(np.sum(clean & (labels == mode))) for mode in (0, 1)]
    ood_fraction = (float(np.mean(ood[eligible])) if n_returned
                    else float("nan"))
    alignment = float("nan")
    alignment_status = NOT_EVALUABLE
    if int(clean.sum()) >= 6:
        km = natural_kmeans(onsets[clean], labels[clean])
        if km.get("status") == "OK":
            alignment = float(km["direction_balanced_alignment"])
            alignment_status = "OK"

    clauses = {
        "n_returned_before_onset_at_least_20":
            n_returned >= int(gate["minimum_returned_events_before_onset"]),
        "ood_at_most_reference_q95":
            bool(np.isfinite(ood_fraction)
                 and ood_fraction <= float(gate["ood_q95"])),
        "both_modes_supported":
            min(mode_counts) >= int(gate["minimum_events_per_mode"]),
        "kmeans_alignment_at_least_reference_q05":
            bool(np.isfinite(alignment)
                 and alignment >= float(gate["balanced_alignment_q05"])),
    }
    endpoints = network_mode_endpoints(
        {**scored["assigned"], "clean": clean, "returned": eligible},
        duration_ms)
    participation = all_event_shaft_participation(onsets[eligible], groups) \
        if n_returned else {"n_events": 0}

    displayed = None
    qualifying = [row for row in scored["rows"] if row["scored"] and row["clean"]]
    if qualifying:
        # spec 7.1: fixed algorithmically by time, never by appearance
        displayed = max(qualifying, key=lambda row: (
            row["t_off_ms"] if row["t_off_ms"] is not None else row["event_index"]))

    confidences = [row["classifier_confidence"] for row in scored["rows"]
                   if row["classifier_confidence"] is not None]
    ranges = [row["spatial_range_mm"] for row in scored["rows"] if row["scored"]]
    recruitment = [row["n_recruited_contacts"] for row in scored["rows"]
                   if row["scored"]]
    return {
        "retained": all(clauses.values()),
        "clauses": clauses,
        "failing_clauses": [name for name, ok in clauses.items() if not ok],
        "measures": {
            "n_returned": n_returned,
            "ood_fraction": ood_fraction,
            "mode_counts": mode_counts,
            "balanced_alignment": alignment,
            "balanced_alignment_status": alignment_status,
            "n_clean": int(clean.sum()),
        },
        "distributions": {
            "event_rate_hz": n_returned / max(float(duration_ms) / 1000.0, 1e-9),
            "mode_endpoints": endpoints,
            "classifier_confidence": {
                "median": float(np.median(confidences)) if confidences else None,
                "q05": float(np.quantile(confidences, 0.05)) if confidences else None,
                "q95": float(np.quantile(confidences, 0.95)) if confidences else None},
            "recruitment_size": {
                "median": float(np.median(recruitment)) if recruitment else None,
                "min": int(np.min(recruitment)) if recruitment else None,
                "max": int(np.max(recruitment)) if recruitment else None},
            "spatial_range_mm": {
                "median": float(np.median(ranges)) if ranges else None,
                "max": float(np.max(ranges)) if ranges else None},
            "shaft_participation": participation,
            "rank_profile_similarity": rank_profile_similarity(
                onsets, labels, clean),
        },
        "strict_t_ictal_sensitivity": {
            "rule": "event ends before t_ictal = t_op - 100 ms",
            "n_returned": int((eligible & scored["strict"]).sum()),
            "n_events_between_t_ictal_and_t_op": int(
                (eligible & ~scored["strict"]).sum()),
        },
        "displayed_event": displayed,
        "displayed_event_rule": gate["displayed_event_rule"],
        "events": scored["rows"],
        "no_label_assigned_to_runaway": bool(
            all(row["mode"] is None for row in scored["rows"]
                if not row["scored"])),
    }

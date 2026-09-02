"""Measurement-layer manifests: exposure, refractory, contact support, provenance.

These are produced before any model is scored and never read a model output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.topic5_group_event_state.v02.timeline import RecordedSession, sessions_from_inventory

from .partition import EVAL_PHASES
from .timeline import EvalTimeline, _load_session_rows


# ----------------------------------------------------------------------------
# interval helpers
# ----------------------------------------------------------------------------

def _clip_intervals(intervals: Sequence[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    out = []
    for a, b in intervals:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 > a2:
            out.append((a2, b2))
    return out


def _union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    items = sorted(intervals)
    total = 0.0
    cur_a, cur_b = items[0]
    for a, b in items[1:]:
        if a > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    total += cur_b - cur_a
    return float(total)


def _subtract(pieces: list[tuple[float, float]], blocked: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    for lo, hi in blocked:
        nxt = []
        for a, b in pieces:
            if hi <= a or lo >= b:
                nxt.append((a, b))
                continue
            if a < lo:
                nxt.append((a, min(lo, b)))
            if b > hi:
                nxt.append((max(hi, a), b))
        pieces = nxt
    return pieces


# ----------------------------------------------------------------------------
# valid exposure
# ----------------------------------------------------------------------------

def exposure_manifest(tl: EvalTimeline, cfg: Mapping[str, Any]) -> dict[str, Any]:
    sessions = sessions_from_inventory(_load_session_rows(tl.subject, Path(cfg["session_inventory"])))
    postictal = float(cfg["timeline"]["postictal_exclusion_seconds"])
    min_seg = float(cfg["timeline"]["min_segment_seconds"])
    seizure_iv = [(float(s["onset_epoch"]), float(s["offset_epoch"])) for s in tl.seizures]
    postictal_iv = [(float(s["offset_epoch"]), float(s["offset_epoch"]) + postictal) for s in tl.seizures]
    blocked = sorted([(a, max(b, a) + postictal) for a, b in seizure_iv])

    excluded: list[dict[str, Any]] = []
    seconds_by_reason = {"seizure": 0.0, "postictal": 0.0, "short_segment": 0.0, "unrecorded_gap": 0.0}
    ordered = sorted(sessions, key=lambda s: s.start_epoch)
    for prev, nxt in zip(ordered[:-1], ordered[1:]):
        if nxt.start_epoch > prev.stop_epoch:
            excluded.append({"start": prev.stop_epoch, "stop": nxt.start_epoch, "reason": "unrecorded_gap"})
            seconds_by_reason["unrecorded_gap"] += nxt.start_epoch - prev.stop_epoch
    for session in ordered:
        lo, hi = float(session.start_epoch), float(session.stop_epoch)
        sz = _clip_intervals(seizure_iv, lo, hi)
        pi = _clip_intervals(postictal_iv, lo, hi)
        seconds_by_reason["seizure"] += _union_seconds(sz)
        seconds_by_reason["postictal"] += max(_union_seconds(pi + sz) - _union_seconds(sz), 0.0)
        for a, b in sz:
            excluded.append({"start": a, "stop": b, "reason": "seizure", "session_id": session.session_id})
        for a, b in pi:
            excluded.append({"start": a, "stop": b, "reason": "postictal", "session_id": session.session_id})
        pieces = _subtract([(lo, hi)], blocked)
        for a, b in pieces:
            if b - a < min_seg:
                excluded.append({"start": a, "stop": b, "reason": "short_segment", "session_id": session.session_id})
                seconds_by_reason["short_segment"] += b - a

    segment_rows = []
    valid_by_phase = {name: 0.0 for name in EVAL_PHASES}
    for seg in tl.segments:
        phase_seconds = {}
        for name in EVAL_PHASES:
            lo, hi = tl.partition.bounds(name)
            a, b = max(seg.start_epoch, lo), min(seg.stop_epoch, hi)
            phase_seconds[name] = float(max(b - a, 0.0))
            valid_by_phase[name] += phase_seconds[name]
        segment_rows.append({
            "segment_id": seg.segment_id, "session_id": seg.session_id,
            "start_epoch": seg.start_epoch, "stop_epoch": seg.stop_epoch,
            "duration_seconds": seg.duration_seconds,
            "phase_at_start": tl.partition.phase_of(seg.start_epoch),
            "phase_at_end": tl.partition.phase_of(np.nextafter(seg.stop_epoch, -np.inf)),
            "phase_seconds": phase_seconds,
            "n_events": int((tl.event_segment == seg.segment_id).sum()),
        })
    total_session = float(sum(s.duration_seconds for s in sessions))
    total_valid = float(sum(s.duration_seconds for s in tl.segments))
    return {
        "subject": tl.subject,
        "dataset": tl.dataset,
        "sessions": [{"session_id": s.session_id, "start_epoch": s.start_epoch, "stop_epoch": s.stop_epoch,
                      "duration_hours": s.duration_seconds / 3600.0} for s in ordered],
        "segments": segment_rows,
        "excluded_intervals": excluded,
        "excluded_seconds_by_reason": seconds_by_reason,
        "recorded_session_seconds": total_session,
        "valid_exposure_seconds": total_valid,
        "valid_exposure_seconds_by_phase": valid_by_phase,
        "valid_exposure_hours_by_phase": {k: v / 3600.0 for k, v in valid_by_phase.items()},
        "n_events_kept": int(tl.n_events),
        "n_events_dropped_outside_segments": int(tl.excluded["n_events_outside_segments"]),
        "n_ictal_events_upstream": int(tl.excluded["n_ictal_events_upstream"]),
        "partition": tl.partition.as_dict(),
        "rule": (
            "count/timing targets use only exposure inside coverage segments; a target window "
            "never crosses a segment edge or a phase edge; gaps, seizures, the 3600 s postictal "
            "exclusion and sub-300 s pieces are not exposure and never count as event-free time"
        ),
    }


# ----------------------------------------------------------------------------
# detector refractory / feature-window overlap
# ----------------------------------------------------------------------------

def refractory_manifest(tl: EvalTimeline, cfg: Mapping[str, Any]) -> dict[str, Any]:
    meas = cfg["measurement"]
    core = float(tl.index["core_seconds_nominal"])
    future_support = float(meas["feature_window_post_seconds"]) + float(meas["filter_pad_seconds"])
    min_gap = float(meas["detector_min_gap_seconds"])
    t = tl.event_times
    same = tl.event_segment[1:] == tl.event_segment[:-1]
    iei = np.diff(t)[same]
    n = int(iei.size)

    def frac(mask: np.ndarray) -> float | None:
        return float(np.mean(mask)) if n else None

    # anchors that fall inside the feature window of the preceding event
    last = np.searchsorted(t, tl.grid.t_anchor, side="left") - 1
    ok = last >= 0
    same_seg = np.zeros_like(ok)
    same_seg[ok] = tl.event_segment[last[ok]] == tl.grid.segment_index[ok]
    dt_anchor = np.full(tl.grid.n_anchors, np.inf)
    dt_anchor[ok & same_seg] = tl.grid.t_anchor[ok & same_seg] - t[last[ok & same_seg]]
    anchors_inside_window = float(np.mean(dt_anchor < core + future_support)) if tl.grid.n_anchors else None
    hist_edges = [core, 2 * core, 3 * core, 4 * core, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0]
    hist = np.histogram(iei, bins=[0.0] + hist_edges + [np.inf])[0].tolist() if n else []
    return {
        "subject": tl.subject,
        "core_seconds_nominal": core,
        "native_rate_hz": float(tl.index["native_rate_hz"]),
        "n_within_segment_intervals": n,
        "iei_seconds": {
            "min": float(iei.min()) if n else None,
            "p1": float(np.percentile(iei, 1)) if n else None,
            "p5": float(np.percentile(iei, 5)) if n else None,
            "median": float(np.median(iei)) if n else None,
        },
        "iei_histogram": {"edges": [0.0] + hist_edges + ["inf"], "counts": hist},
        "structural_refractory": {
            "packer_rule": (
                "legacy packer centres a fixed cut_t window on each co-activation run, then "
                "pick_noOverlap_timeRanges(less_than=2) deletes BOTH members of any overlapping "
                "pair and any window >= 2 s; group events closer than cut_t are therefore removed, "
                "not merged"
            ),
            "minimum_possible_iei_seconds": core,
            "fraction_iei_below_core": frac(iei < core - 1e-9),
            "fraction_iei_below_core_plus_min_gap": frac(iei < core + min_gap),
            "fraction_iei_below_2x_core": frac(iei < 2 * core),
            "false_silence_mechanism": (
                "dense bursts (successive group events < cut_t apart) are deleted pairwise upstream, "
                "so high-rate windows are under-counted; the bias is shared by every arm but "
                "compresses the count dynamic range"
            ),
            "recoverable_here": False,
        },
        "feature_window": {
            "pre_seconds": float(meas["feature_window_pre_seconds"]),
            "post_seconds": float(meas["feature_window_post_seconds"]),
            "filter_pad_seconds": float(meas["filter_pad_seconds"]),
            "zero_phase_filter": "filtfilt (FIR 201 taps) uses samples up to core_end + post + pad",
            "future_support_after_core_end_seconds": future_support,
            "fraction_next_event_inside_feature_window": frac(iei < core + future_support),
            "fraction_next_core_overlaps_context_only": frac(iei < core + float(meas["feature_window_post_seconds"])),
            "earliest_state_update_after_onset_seconds": core + future_support,
            "fraction_anchors_inside_previous_feature_window": anchors_inside_window,
            "rule": (
                "an event's content may enter the state no earlier than core_end + 0.75 s; an anchor "
                "state may only use events whose feature window has closed before the anchor"
            ),
        },
        "background_windows": "v0.1 background anchors end strictly before each event core (no future support)",
    }


# ----------------------------------------------------------------------------
# contact support
# ----------------------------------------------------------------------------

def contact_support_manifest(tl: EvalTimeline, cfg: Mapping[str, Any]) -> dict[str, Any]:
    labels = tl.event_phase_labels()
    root = Path(cfg["dataset_root"]) / tl.subject
    contact_ok = np.asarray(np.load(root / "contact_ok.npy", mmap_mode="r")[tl.raw_positions])
    contacts = []
    for c, name in enumerate(tl.contact_names):
        row = {
            "index": c, "label": name, "shaft": tl.contact_shafts[c],
            "in_vocabulary": bool(tl.vocab_mask[c]),
            "valid_on_base_fit": bool(tl.contact_valid_base[c]),
            "participation_rate_by_phase": {},
            "contact_ok_fraction_by_phase": {},
            "participation_rate_by_segment": {},
        }
        for i, phase in enumerate(EVAL_PHASES):
            m = labels == i
            row["participation_rate_by_phase"][phase] = float(tl.participation[m, c].mean()) if m.any() else None
            row["contact_ok_fraction_by_phase"][phase] = float(contact_ok[m, c].mean()) if m.any() else None
        for seg in tl.segments:
            m = tl.event_segment == seg.segment_id
            row["participation_rate_by_segment"][str(seg.segment_id)] = float(tl.participation[m, c].mean()) if m.any() else None
        contacts.append(row)
    unseen = ~tl.vocab_mask
    events_with_unseen = tl.participation[:, unseen].any(axis=1) if unseen.any() else np.zeros(tl.n_events, bool)
    return {
        "subject": tl.subject,
        "vocabulary_rule": f"contact participates in >= {cfg['measurement']['vocab_min_events']} base_fit events",
        "n_contacts_legacy": int(tl.n_contacts),
        "n_contacts_vocabulary": int(tl.n_vocab),
        "contacts": contacts,
        "events_with_out_of_vocabulary_contact_by_phase": {
            phase: int(events_with_unseen[labels == i].sum()) for i, phase in enumerate(EVAL_PHASES)
        },
        "bad_channel_rule": "contact validity (contact_ok) is derived from base_fit events only; later phases are descriptive",
    }


def nontransductive_support_manifest(tl: EvalTimeline, cfg: Mapping[str, Any],
                                     *, hardware_detector_channels: int | None,
                                     montage_provenance: str | None) -> dict[str, Any]:
    labels = tl.event_phase_labels()
    unseen = ~tl.vocab_mask
    events_with_unseen = tl.participation[:, unseen].any(axis=1) if unseen.any() else np.zeros(tl.n_events, bool)
    part_mass = tl.participation.sum(axis=1).astype(float)
    vocab_mass = tl.participation[:, tl.vocab_mask].sum(axis=1).astype(float)
    stability = {}
    for i, phase in enumerate(EVAL_PHASES):
        m = labels == i
        stability[phase] = float(vocab_mass[m].sum() / part_mass[m].sum()) if m.any() and part_mass[m].sum() > 0 else None
    return {
        "subject": tl.subject,
        "dataset": tl.dataset,
        "hardware_detector_channels": hardware_detector_channels,
        "legacy_vocabulary": {"n": int(tl.n_contacts), "labels": list(tl.contact_names),
                              "source": "legacy full-record refine/packing (count > mean + 1 std over the whole record)"},
        "prefix_vocabulary": {
            "n": int(tl.n_vocab),
            "labels": [n for n, keep in zip(tl.contact_names, tl.vocab_mask) if keep],
            "rule": f">= {cfg['measurement']['vocab_min_events']} participations inside base_fit (0-60% recorded time)",
        },
        "excluded_contacts": [n for n, keep in zip(tl.contact_names, tl.vocab_mask) if not keep],
        "events_containing_excluded_contact_by_phase": {
            phase: int(events_with_unseen[labels == i].sum()) for i, phase in enumerate(EVAL_PHASES)
        },
        "support_stability_by_phase": stability,
        "montage_provenance": montage_provenance,
        "detector_reference": tl.index.get("detector_reference"),
        "fixed_hardware_montage_feasible": False,
        "why_not_fixed_hardware_montage": (
            "the legacy event stream (which group events exist and which contacts they contain) was "
            "defined on the refined channel subset; using every implanted channel requires re-running "
            "detection, refine and packing, i.e. a full cache rebuild outside this package"
        ),
        "measurement_layer_nested_contract": "prefix_vocabulary_on_legacy_event_stream",
        "upstream_transductive_steps": [
            "detector abs_thresh relative to the whole-record envelope median",
            "refine: channel selection by full-record event counts",
            "packing: co-activation of the refined channels; overlapping windows deleted",
        ],
        "v032_nested_steps": [
            "contact vocabulary and bad-channel support from base_fit only",
            "event-feature normalisation, mark PCA, group-size prior, calibration bias from base_fit only",
            "hyper-parameters selected on inner_val only; state checkpoints on dev_val only",
        ],
    }


def detector_provenance_audit(cfg: Mapping[str, Any], patients: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    legacy = cfg["measurement"]["legacy_detector"]
    items = [
        {"item": "detector_threshold", "legacy_source": f"rel_thresh={legacy['rel_thresh']} x per-channel envelope median of the processed batch; abs_thresh={legacy['abs_thresh']} x whole-record envelope median (Epilepsiae producer default rel_thresh=3 in one script, 2 in the run block)",
         "uses_full_record_or_future": True, "v032_handling": "cannot be re-fit here; recorded as upstream transductive", "status": "upstream_transductive_unfixable_here"},
        {"item": "template", "legacy_source": "no template matching in the envelope detector or the packer", "uses_full_record_or_future": False, "v032_handling": "not applicable", "status": "not_applicable"},
        {"item": "refine_packing", "legacy_source": legacy["refine_rule"] + "; " + legacy["pack_rule"] + "; overlapping packed windows deleted pairwise (pick_noOverlap_timeRanges less_than=2)",
         "uses_full_record_or_future": True, "v032_handling": "event stream accepted as given; refractory/false-silence quantified per patient", "status": "upstream_transductive_unfixable_here"},
        {"item": "contact_selection", "legacy_source": "refined channel subset selected on full-record counts", "uses_full_record_or_future": True,
         "v032_handling": "prefix vocabulary: contacts with >= vocab_min_events participations in base_fit", "status": "prefix_only_nested"},
        {"item": "normalization", "legacy_source": "v0.3.1 event encoder stats from grammar_fit prefix (0-16%)", "uses_full_record_or_future": False,
         "v032_handling": "all standardisation constants from base_fit anchors/events (0-60%)", "status": "prefix_only_nested"},
        {"item": "group_size_prior", "legacy_source": "v0.3.1 calibrated on 0-16% prefix", "uses_full_record_or_future": False,
         "v032_handling": "static K prior from base_fit; grammar K head fit on base_fit, epoch on inner_val, refit 0-70%", "status": "prefix_only_nested"},
        {"item": "calibration_bias", "legacy_source": "v0.3.1 contact bias / temperature / stop bias fit on 0-16%", "uses_full_record_or_future": False,
         "v032_handling": "fit on base_fit, selected on inner_val, refit on base_refit", "status": "prefix_only_nested"},
        {"item": "event_feature_normalization", "legacy_source": "v0.1 cache stores raw features; v0.2 marks standardised on 70% TRAIN", "uses_full_record_or_future": False,
         "v032_handling": "mark PCA and standardisation re-fit on base_fit only", "status": "prefix_only_nested"},
        {"item": "tied_group_statistics", "legacy_source": "TIE_TOLERANCE_SECONDS = 0.010 fixed from the producer spectrogram hop", "uses_full_record_or_future": False,
         "v032_handling": "constant, not data-adaptive", "status": "fixed_constant"},
        {"item": "bad_channel_support", "legacy_source": "contact_ok any() over ALL events (v0.1 SubjectSequence.contact_valid)", "uses_full_record_or_future": True,
         "v032_handling": "derived from base_fit events only", "status": "prefix_only_nested"},
        {"item": "checkpoint_selection", "legacy_source": "v0.3.1 state checkpoint on 70-80%, dev-test scored once", "uses_full_record_or_future": False,
         "v032_handling": "grammar/H/adapters selected on inner_val (60-70%); state checkpoints (Agent 1) on dev_val (70-80%); dev_test (80-100%) is a consumed development score", "status": "nested_by_selection_phase"},
    ]
    return {
        "format": "group_event_state_v0_3_2_detector_provenance_audit",
        "legacy_detector_parameters": dict(legacy),
        "items": items,
        "per_patient": dict(patients),
        "measurement_layer_nested_contract": "prefix_vocabulary_on_legacy_event_stream",
        "sealed_evaluation_blocker": "fixed hardware montage cache rebuild (detection + refine + packing) not performed",
    }

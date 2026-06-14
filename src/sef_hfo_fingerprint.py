# src/sef_hfo_fingerprint.py
"""Axis-A propagation-fingerprint measurement contract (Stage A0 schema freeze).

Plan: docs/superpowers/plans/2026-06-14-sef-hfo-axisA-lesion-propagation-fingerprint.md §1 + Stage A0.

WHAT THIS IS (plain): the fingerprint is the MEASURING INSTRUMENT for axis A. Before
any lesion sweep runs, we freeze ONE event-level table contract (fields + units +
primary/secondary/deferred tier) so features cannot be cherry-picked after the fact.
This module:
  1. defines that frozen schema (FINGERPRINT_SCHEMA below),
  2. extracts the per-event table over EXISTING baseline read-out artifacts
     (a readout_*.json + the matching rep_*.npz — NO simulation, NO sweep),
  3. computes the PRIMARY features (axis_dir / pathway_width / onset_jitter) +
     the SECONDARY recruit_extent, and HONESTLY marks the fields that the current
     read-out runner does NOT persist (per-contact onset time, per-contact envelope
     peak) as status='requires_extended_readout_save' — that gap IS part of the freeze.

Reuse-first (no re-invention): geometry/axis estimators come from
src.sef_hfo_observation (endpoint_centroid_axis / axis_angle_error_deg /
direction_readability) and entry dispersion from src.sef_hfo_stage3.entry_jitter_stats.

TIER DISCIPLINE (LOCKED, plan §0.1 / §5):
  primary  = {axis_dir, pathway_width, onset_jitter}
  secondary= {latency_jitter, recruit_extent}
  deferred = {speed, event_size}            -- NOT primary, ever
  amplitude_proxy = provenance/diagnostic   -- persisted, NEVER in a primary comparison
A deferred/diagnostic feature must never enter a primary comparison.

SCHEMA FROZEN (user sign-off 2026-06-14, plan §0.1): field set + tiers, pathway_width
(perp to the LOCKED geometry axis, p95-p5, <3 contacts -> NaN) and onset_jitter
(clean-event min-rank first contact) definitions, n_min_events=6 (A1/A3/A4 floor),
the runner clean gate, the double-focus sidecar-hidden-label hook, and the honest
requires_extended_readout_save gap are all APPROVED. Bump SCHEMA_VERSION on any
post-freeze field add / rename / tier change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.sef_hfo_observation import (
    endpoint_centroid_axis,
    axis_angle_error_deg,
    direction_readability,
)
from src.sef_hfo_stage3 import entry_jitter_stats

# ----------------------------------------------------------------------------
# Read-out clean-event gate (mirrors the runner's _clean discipline so the
# fingerprint counts the SAME clean events the baseline JSON reports).
#   scripts/run_sef_hfo_snn_cm_spontaneous_readout.py::_clean
# A clean readable event = returned (self-terminated) AND axis readable
# (axis_err finite AND < AXIS_ERR_MAX) AND n_part >= PART_MIN.
# ----------------------------------------------------------------------------
PART_MIN: int = 7          # min participating contacts for a readable axis (runner value)
AXIS_ERR_MAX: float = 25.0  # deg; axis_err >= this => not a clean directional read (runner value)
K_DIR: int = 3             # endpoint-centroid k (runner value; for re-derivation paths)

# n_min_events gate (plan §1 P1-lite). PROPOSED at schema-freeze time; pending user
# sign-off. Rationale lives in propose_n_min_events() and the summary JSON.
N_MIN_EVENTS_DEFAULT: int = 6

# Schema/contract version. Bump on any field add/rename/tier change after sign-off.
SCHEMA_VERSION: str = "A0-frozen-2026-06-14"  # user sign-off 2026-06-14 (plan §0.1)


# ----------------------------------------------------------------------------
# Frozen event-level schema (plan §1 contract). Each entry: tier + unit + whether
# it is computable NOW from the existing artifacts, and (if not) what the read-out
# runner would additionally have to persist.
# ----------------------------------------------------------------------------
FINGERPRINT_SCHEMA: Dict[str, dict] = {
    # --- provenance / identity (one row per clean event) ---
    "source_label": dict(
        tier="provenance", unit="categorical", computable_now=True,
        source="single-focus baseline: the run's lesion sign (config.lesion -> "
               "oneend_neg=neg / oneend_pos=pos). DOUBLE-FOCUS LATER: the sidecar "
               "hidden_source_label (NOT direction sign) — left as a clearly marked "
               "hook (sidecar_source_label=None until a two-focus sidecar is wired)."),
    "direction_sign": dict(
        tier="provenance", unit="{-1,+1}", computable_now=True,
        source="read-out sign (events[i].sign); template/direction id"),
    "event_window": dict(
        tier="provenance", unit="ms (t_on,t_off)", computable_now=True,
        source="events[i].t_on / t_off"),
    "peak_time": dict(
        tier="provenance", unit="ms", computable_now=True,
        source="events[i].event_peak_t"),
    "participation_mask": dict(
        tier="provenance", unit="bool per contact", computable_now=True,
        source="events[i].ranks[name] is not None"),
    "per_contact_rank": dict(
        tier="provenance", unit="dense rank (unit-free)", computable_now=True,
        source="events[i].ranks (dict name->rank|None)"),
    "per_contact_coords": dict(
        tier="provenance", unit="mm (model frame)", computable_now=True,
        source="rep_*.npz contacts (GLOBAL/constant across events in a run)"),
    "engine_signature": dict(
        tier="provenance", unit="sha map", computable_now=True,
        source="readout JSON provenance.engine_sha (12-hex per engine file)"),
    "seed": dict(
        tier="provenance", unit="int", computable_now=True,
        source="config.seed"),
    "is_collision": dict(
        tier="provenance", unit="bool", computable_now=True,
        source="single-focus baseline: N/A -> False (only one focus; no two-core "
               "co-ignition possible). DOUBLE-FOCUS LATER: sidecar hidden_source_label=="
               "'collision'."),
    "is_ambiguous": dict(
        tier="provenance", unit="bool", computable_now=True,
        source="axis unreadable (axis_err None / n_part < PART_MIN) -> True"),
    "amplitude_proxy": dict(
        tier="diagnostic", unit="firing/LFP envelope a.u.", computable_now=False,
        status="requires_extended_readout_save",
        what_to_persist="per-event envelope peak (e.g. max active fraction in window, "
                        "or per-contact envelope peak). NOTE (plan P2): provenance/"
                        "diagnostic ONLY — must NEVER enter a primary comparison, so "
                        "even once saved it does not become event_size's back door."),

    # --- PRIMARY features ---
    "axis_dir": dict(
        tier="primary", unit="(sign, axis_err_deg)", computable_now=True,
        source="events[i].sign + events[i].axis_err (endpoint_centroid_axis / "
               "axis_angle_error_deg); ALSO the readability gate"),
    "pathway_width": dict(
        tier="primary", unit="mm (perpendicular to axis)", computable_now=True,
        source="robust span (p95-p5) of participating-contact coords projected onto the "
               "axis-PERPENDICULAR unit of the LOCKED model/geometry axis (theta from the "
               "run geometry / rep_*.npz), NOT a per-event endpoint-fitted axis — the width "
               "is measured relative to the fixed axis so it cannot rotate with the event. "
               "Reuses rep_*.npz coords."),
    "onset_jitter": dict(
        tier="primary", unit="entry-dispersion stats", computable_now=True,
        source="entry_jitter_stats over per-event first-active contact (participating "
               "contact with MIN rank). top1/top3 fraction + n_unique across events."),

    # --- SECONDARY features ---
    "recruit_extent": dict(
        tier="secondary", unit="(n_part, along-axis span mm)", computable_now=True,
        source="n_part + robust along-axis span (p95-p5 of participating-contact "
               "projection onto the axis direction)"),
    "latency_jitter": dict(
        tier="secondary", unit="ms (per-contact peak-time variance across events)",
        computable_now=False, status="requires_extended_readout_save",
        what_to_persist="per-contact onset TIME array per event (the runner computes "
                        "per-contact first-crossing lag inside extract_lagpat/read_event "
                        "but DISCARDS the times, persisting only ranks). Persist the "
                        "per-contact lag_raw vector (ms) per clean event."),

    # --- DEFERRED features (NOT primary, plan §1/§5) ---
    "speed": dict(
        tier="deferred", unit="mm/ms", computable_now=False,
        status="requires_extended_readout_save",
        what_to_persist="per-contact onset TIME array per event (same as latency_jitter). "
                        "DEFERRED regardless: model dt-bin onset vs real HFO ms resolution "
                        "not directly comparable (plan §1) — do not promote to primary."),
    "event_size": dict(
        tier="deferred", unit="amplitude x n_part", computable_now=False,
        status="requires_extended_readout_save",
        what_to_persist="per-contact envelope peak per event. DEFERRED regardless: model "
                        "LFP/firing-proxy amplitude vs real HFO amplitude not directly "
                        "comparable (plan §1)."),
}

# Convenience tier views (single source of truth = FINGERPRINT_SCHEMA above).
PRIMARY_FEATURES = tuple(k for k, v in FINGERPRINT_SCHEMA.items() if v["tier"] == "primary")
SECONDARY_FEATURES = tuple(k for k, v in FINGERPRINT_SCHEMA.items() if v["tier"] == "secondary")
DEFERRED_FEATURES = tuple(k for k, v in FINGERPRINT_SCHEMA.items() if v["tier"] == "deferred")
REQUIRES_EXTENDED_SAVE = tuple(
    k for k, v in FINGERPRINT_SCHEMA.items()
    if v.get("status") == "requires_extended_readout_save")


@dataclass
class EventFingerprint:
    """One row of the frozen event-level table (per clean event)."""
    event_index: int
    source_label: str
    sidecar_source_label: Optional[str]    # double-focus hook; None for single-focus baseline
    direction_sign: Optional[float]
    event_window: tuple                    # (t_on, t_off) ms
    peak_time: Optional[float]
    n_part: int
    participation_mask: list               # bool per contact (rep-npz contact order)
    per_contact_rank: list                 # rank or None per contact (rep-npz order)
    is_collision: bool
    is_ambiguous: bool
    amplitude_proxy: dict                  # {value, status} — diagnostic, never primary
    # primary
    axis_dir: dict                         # {sign, axis_err_deg, readable}
    pathway_width: dict                    # {value_mm, status}
    first_contact: Optional[str]           # min-rank participating contact (feeds onset_jitter)
    # secondary
    recruit_extent: dict                   # {n_part, along_span_mm}
    latency_jitter: dict                   # {value, status}
    # deferred
    speed: dict                            # {value, status}
    event_size: dict                       # {value, status}


@dataclass
class RunFingerprint:
    """One run's fingerprint: provenance + per-event table + aggregated primary features."""
    tag: str
    source_label: str
    engine_signature: dict
    seed: Optional[int]
    config: dict
    contact_names: list
    contact_coords: list                   # (n_contact, 2)
    theta_deg: float
    n_events_total: int
    n_clean_events: int
    insufficient: bool                     # n_clean_events < n_min_events
    n_min_events: int
    collision_count: int
    ambiguous_count: int
    excluded_counts: dict
    clean_event_rate: float
    events: List[EventFingerprint] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    deferred_status: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _axis_units(theta_deg: float):
    """Axis unit (along theta) and perpendicular unit (theta+90deg)."""
    th = np.deg2rad(float(theta_deg))
    along = np.array([np.cos(th), np.sin(th)])
    perp = np.array([-np.sin(th), np.cos(th)])
    return along, perp


def _robust_span(proj: np.ndarray) -> float:
    """Robust span p95-p5 of a 1-D projection; NaN if < 2 points."""
    proj = np.asarray(proj, float)
    proj = proj[np.isfinite(proj)]
    if proj.size < 2:
        return float("nan")
    return float(np.percentile(proj, 95) - np.percentile(proj, 5))


def perp_pathway_width(coords: np.ndarray, part_mask: np.ndarray, theta_deg: float,
                       min_part: int = 3) -> float:
    """PRIMARY pathway_width: robust perpendicular span of participating contacts (mm).

    theta_deg is the LOCKED model/geometry axis (run geometry), NOT a per-event
    endpoint-fitted axis — the width is measured perpendicular to the fixed axis so it
    cannot rotate with the event. Guard against inventing signal from too-few
    participants: < min_part participating contacts -> NaN (do not manufacture a width
    from 1-2 points)."""
    coords = np.asarray(coords, float)
    part = np.asarray(part_mask, bool)
    if int(part.sum()) < min_part:
        return float("nan")
    _, perp = _axis_units(theta_deg)
    proj = coords[part] @ perp
    return _robust_span(proj)


def along_axis_span(coords: np.ndarray, part_mask: np.ndarray, theta_deg: float,
                    min_part: int = 3) -> float:
    """SECONDARY recruit_extent helper: robust along-axis span of participating contacts (mm)."""
    coords = np.asarray(coords, float)
    part = np.asarray(part_mask, bool)
    if int(part.sum()) < min_part:
        return float("nan")
    along, _ = _axis_units(theta_deg)
    proj = coords[part] @ along
    return _robust_span(proj)


# ----------------------------------------------------------------------------
# loaders + name-alignment check (CLAUDE.md: fail LOUDLY on mismatch)
# ----------------------------------------------------------------------------
def load_run_artifacts(readout_json_path, rep_npz_path) -> dict:
    """Load a readout_*.json + its matching rep_*.npz and VERIFY two boundary
    contracts (CLAUDE.md §6 boundary discipline).

    CONTRACT (raise LOUDLY on either failure):
      (a) NAME ALIGNMENT — every event's ranks-dict keys must map to rep-npz 'names'
          (the global montage). A mismatch means ranks indices map to the wrong
          contacts.
      (b) n_part / rank CONSISTENCY — every event's JSON n_part must EQUAL the number
          of non-null ranks. The clean gate uses n_part while participation / width /
          jitter use the rank dict; if they disagree the fingerprint is computed on
          inconsistent participation.
    """
    rj = json.loads(Path(readout_json_path).read_text())
    npz = np.load(str(rep_npz_path), allow_pickle=True)
    names = [str(n) for n in npz["names"].tolist()]
    coords = np.asarray(npz["contacts"], float)
    valid = np.asarray(npz["valid"], int).astype(bool)
    theta = float(npz["theta"])
    foci = np.asarray(npz["foci"], float)
    if len(names) != coords.shape[0]:
        raise ValueError(
            f"name/coord length mismatch in rep npz: {len(names)} names vs "
            f"{coords.shape[0]} coords")
    name_set = set(names)
    for raw_i, e in enumerate(rj.get("events", [])):
        ranks = e.get("ranks")
        # (a) name-alignment FIRST: ranks keys must be a subset of npz names (so an
        #     unknown contact is reported as a name error, not as a count mismatch).
        if ranks:
            keys = set(ranks.keys())
            if not keys.issubset(name_set):
                extra = sorted(keys - name_set)
                raise ValueError(
                    "fingerprint name-alignment FAILED: event ranks keys "
                    f"{extra} are not in rep npz names {names}. ranks indices would map "
                    "to the wrong contacts — refusing to extract.")
        # (b) n_part must equal the number of non-null ranks.
        npart = int(e.get("n_part") or 0)
        nz = 0 if not ranks else sum(1 for v in ranks.values() if v is not None)
        if nz != npart:
            raise ValueError(
                f"fingerprint n_part/rank mismatch at event {raw_i}: JSON n_part={npart} "
                f"but {nz} non-null ranks. n_part drives the clean gate while the rank "
                "dict drives participation/width/jitter — they must agree.")
    return dict(readout=rj, names=names, coords=coords, valid=valid,
                theta=theta, foci=foci)


def _source_label_from_lesion(lesion: str) -> str:
    """Single-focus baseline source label = the run's lesion sign."""
    if lesion == "oneend_neg":
        return "neg"
    if lesion == "oneend_pos":
        return "pos"
    return lesion  # twoend_* etc -> the double-focus hook must supply sidecar labels


def _is_clean(e: dict) -> bool:
    """Mirror the runner's clean-directional gate."""
    return bool(
        e.get("returned")
        and e.get("axis_err") is not None
        and e.get("axis_err") < AXIS_ERR_MAX
        and (e.get("n_part") or 0) >= PART_MIN)


# ----------------------------------------------------------------------------
# main extractor
# ----------------------------------------------------------------------------
def extract_fingerprint(readout_json_path, rep_npz_path,
                        n_min_events: int = N_MIN_EVENTS_DEFAULT,
                        sidecar_path: Optional[str] = None) -> RunFingerprint:
    """Extract the frozen event-level fingerprint over EXISTING artifacts (NO sim).

    sidecar_path: double-focus hook. If given, hidden_source_label per event overrides
    the single-focus lesion-sign source_label and supplies is_collision. For the
    single-focus baseline it is None and source_label = the run's lesion sign.
    """
    art = load_run_artifacts(readout_json_path, rep_npz_path)
    rj, names, coords = art["readout"], art["names"], art["coords"]
    theta = art["theta"]
    cfg = rj.get("config", {})
    lesion = cfg.get("lesion", "")
    run_source_label = _source_label_from_lesion(lesion)
    seed = cfg.get("seed")
    engine_sig = (rj.get("provenance", {}) or {}).get("engine_sha", {})

    # optional double-focus sidecar (hook). Map raw_event_index -> sidecar record.
    sidecar_by_raw = {}
    if sidecar_path is not None and Path(sidecar_path).exists():
        sc = json.loads(Path(sidecar_path).read_text())
        for s in sc.get("events", []):
            sidecar_by_raw[s.get("raw_event_index")] = s

    name_index = {nm: i for i, nm in enumerate(names)}
    n_c = len(names)

    rows: List[EventFingerprint] = []
    first_contacts: List[Optional[str]] = []
    collision_count = 0
    ambiguous_count = 0
    excluded = {"not_returned": 0, "axis_unreadable": 0, "too_few_contacts": 0}

    for raw_i, e in enumerate(rj.get("events", [])):
        ranks_dict = e.get("ranks") or {}
        # participation mask + per-contact rank in rep-npz contact order
        part_mask = np.zeros(n_c, bool)
        rank_vec = [None] * n_c
        for nm, rk in ranks_dict.items():
            j = name_index.get(nm)
            if j is None:
                continue
            if rk is not None:
                part_mask[j] = True
                rank_vec[j] = float(rk)

        readable = (e.get("axis_err") is not None and (e.get("n_part") or 0) >= PART_MIN)
        is_amb = not readable

        # double-focus hook: sidecar overrides source label + collision
        sc = sidecar_by_raw.get(raw_i)
        if sc is not None:
            sidecar_label = sc.get("hidden_source_label")
            src_label = sidecar_label
            is_coll = (sidecar_label == "collision")
            is_amb = is_amb or (sidecar_label == "ambiguous")
        else:
            sidecar_label = None
            src_label = run_source_label
            is_coll = False  # single-focus baseline: no two-core co-ignition -> 0

        if is_coll:
            collision_count += 1
        if is_amb:
            ambiguous_count += 1

        # exclusion bookkeeping (only clean events enter aggregation)
        if not e.get("returned"):
            excluded["not_returned"] += 1
        elif e.get("axis_err") is None:
            excluded["axis_unreadable"] += 1
        elif (e.get("n_part") or 0) < PART_MIN:
            excluded["too_few_contacts"] += 1

        # first-active contact = participating contact with MIN rank (feeds onset_jitter)
        fc = None
        part_idx = np.flatnonzero(part_mask)
        if part_idx.size:
            rks = np.array([rank_vec[j] for j in part_idx], float)
            fc = names[int(part_idx[int(np.argmin(rks))])]

        pw = perp_pathway_width(coords, part_mask, theta)
        asp = along_axis_span(coords, part_mask, theta)

        clean = _is_clean(e)
        row = EventFingerprint(
            event_index=raw_i,
            source_label=src_label,
            sidecar_source_label=sidecar_label,
            direction_sign=e.get("sign"),
            event_window=(e.get("t_on"), e.get("t_off")),
            peak_time=e.get("event_peak_t"),
            n_part=int(e.get("n_part") or 0),
            participation_mask=part_mask.tolist(),
            per_contact_rank=rank_vec,
            is_collision=is_coll,
            is_ambiguous=is_amb,
            amplitude_proxy={"value": None, "status": "requires_extended_readout_save",
                             "note": "diagnostic only; never a primary comparison (plan P2)"},
            axis_dir={"sign": e.get("sign"), "axis_err_deg": e.get("axis_err"),
                      "readable": bool(readable), "readability": e.get("readability")},
            pathway_width={"value_mm": (None if not np.isfinite(pw) else round(pw, 4)),
                           "status": "ok" if np.isfinite(pw) else "insufficient_contacts"},
            first_contact=fc,
            recruit_extent={"n_part": int(e.get("n_part") or 0),
                            "along_span_mm": (None if not np.isfinite(asp) else round(asp, 4))},
            latency_jitter={"value": None, "status": "requires_extended_readout_save"},
            speed={"value": None, "status": "requires_extended_readout_save"},
            event_size={"value": None, "status": "requires_extended_readout_save"},
        )
        rows.append(row)
        # onset_jitter aggregates first-active contact over CLEAN events only
        if clean:
            first_contacts.append(fc)

    n_clean = len(first_contacts)
    n_total = len(rows)

    # aggregate PRIMARY features over clean events
    clean_event_rows = [r for r in rows
                        if _is_clean(rj["events"][r.event_index])]
    axis_errs = [r.axis_dir["axis_err_deg"] for r in clean_event_rows
                 if r.axis_dir["axis_err_deg"] is not None]
    signs = [r.direction_sign for r in clean_event_rows if r.direction_sign is not None]
    widths = [r.pathway_width["value_mm"] for r in clean_event_rows
              if r.pathway_width["value_mm"] is not None]

    aggregate = {
        "axis_dir": {
            "n": len(axis_errs),
            "axis_err_median_deg": (round(float(np.median(axis_errs)), 4) if axis_errs else None),
            "axis_err_iqr_deg": (round(float(np.percentile(axis_errs, 75)
                                             - np.percentile(axis_errs, 25)), 4)
                                 if axis_errs else None),
            "sign_majority": (float(np.sign(np.sum(signs))) if signs else None),
            "sign_consistency": (round(float(np.mean(np.array(signs) == np.sign(np.sum(signs)))), 4)
                                 if signs else None),
        },
        "pathway_width": {
            "n": len(widths),
            "median_mm": (round(float(np.median(widths)), 4) if widths else None),
            "iqr_mm": (round(float(np.percentile(widths, 75) - np.percentile(widths, 25)), 4)
                       if len(widths) >= 2 else None),
        },
        "onset_jitter": entry_jitter_stats(first_contacts),
        "recruit_extent_secondary": {
            "n_part_median": (round(float(np.median([r.n_part for r in clean_event_rows])), 2)
                              if clean_event_rows else None),
            "along_span_median_mm": (
                round(float(np.median([r.recruit_extent["along_span_mm"]
                                       for r in clean_event_rows
                                       if r.recruit_extent["along_span_mm"] is not None])), 4)
                if any(r.recruit_extent["along_span_mm"] is not None for r in clean_event_rows)
                else None),
        },
    }

    insufficient = n_clean < n_min_events
    if insufficient:
        aggregate["INSUFFICIENT"] = (
            f"n_clean_events={n_clean} < n_min_events={n_min_events}: "
            "primary fingerprint NOT entered into any group comparison (plan §1 gate).")

    deferred_status = {
        k: {"tier": FINGERPRINT_SCHEMA[k]["tier"],
            "status": "requires_extended_readout_save",
            "what_to_persist": FINGERPRINT_SCHEMA[k]["what_to_persist"]}
        for k in REQUIRES_EXTENDED_SAVE
    }

    return RunFingerprint(
        tag=rj.get("tag", ""),
        source_label=run_source_label,
        engine_signature=engine_sig,
        seed=seed,
        config=cfg,
        contact_names=names,
        contact_coords=coords.tolist(),
        theta_deg=theta,
        n_events_total=n_total,
        n_clean_events=n_clean,
        insufficient=insufficient,
        n_min_events=n_min_events,
        collision_count=collision_count,
        ambiguous_count=ambiguous_count,
        excluded_counts=excluded,
        clean_event_rate=(round(n_clean / n_total, 4) if n_total else 0.0),
        events=rows,
        aggregate=aggregate,
        deferred_status=deferred_status,
    )


def run_fingerprint_to_dict(rf: RunFingerprint) -> dict:
    """Serialize a RunFingerprint (dataclass -> JSON-safe dict)."""
    d = asdict(rf)
    return d


def propose_n_min_events(clean_counts: Dict[str, int]) -> dict:
    """Propose the n_min_events INSUFFICIENT gate (plan §1) tied to the baseline counts.

    clean_counts: {run_tag: n_clean_events}. Returns the proposed value + rationale.
    The value is a PROPOSAL pending user sign-off (not irreversibly final)."""
    counts = sorted(clean_counts.values())
    min_obs = counts[0] if counts else 0
    value = N_MIN_EVENTS_DEFAULT
    rationale = (
        f"Baseline single-focus runs yield clean readable events = {clean_counts}. "
        f"The smallest observed is {min_obs}. n_min_events={value} is set BELOW the "
        "baseline counts so the validated A0 runs pass the gate, yet high enough that a "
        "future lesion variant whose primary fingerprint rests on only a handful of "
        "events is flagged INSUFFICIENT rather than entering a group comparison "
        "(plan §1 P1-lite). Pending user sign-off; revisable at schema-freeze review.")
    return {"value": value, "min_observed_clean": min_obs, "per_run": clean_counts,
            "rationale": rationale}

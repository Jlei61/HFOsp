"""SEF-ITP Phase 2 runner — H3 mark-independence + H4 normalized rate/geometry instability.

Plan: docs/superpowers/plans/2026-05-23-topic4-phase2-h3-h4-plan.md
Framework: docs/topic4_sef_itp_framework.md v1.0.5
Module: src/sef_itp_phase2.py

Usage:
    # one subject, both hypotheses, default 2h epochs:
    python scripts/run_sef_itp_phase2.py --subject yuquan_chengshuai
    # full cohort:
    python scripts/run_sef_itp_phase2.py --all
    # 1h epoch sensitivity:
    python scripts/run_sef_itp_phase2.py --all --epoch-hours 1.0

Reads (relative to repo root):
  - Phase 1 cohort gate: results/topic4_sef_itp/phase1_spatial_geometry/per_subject/<dataset>_<sid>.json
  - PR-7 pairing JSON:  results/interictal_propagation_masked/template_pairing/per_subject/<...>.json
  - PR-7 burst JSON:    results/interictal_propagation_masked/template_pairing/per_subject_burst/<...>.json
  - PR-6 anchoring:     results/interictal_propagation_masked/template_anchoring/per_subject/<...>.json
  - PR-2 masked:        results/interictal_propagation_masked/per_subject/<...>.json
  - lagPat NPZ:         /mnt/yuquan_data/yuquan_24h_edf/<sid>/ or /mnt/epilepsia_data/.../<sid>/all_recs/

Writes per-subject JSON to:
  results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/<dataset>_<sid>.json

Output schema (v1.0.0):
  {
    "dataset": str, "subject_id": str,
    "schema_version": "sef_itp_phase2_v1_2026_05_23",
    "epoch_hours": float, "n_perm": int, "seed": int,
    "h3": {
      "lag1_same_excess_n2": float,
      "window_excess_n2": {"10.0": float, "30.0": float, "60.0": float, "1800.0": float},
      "run_length_lift_n2": float,
      "endpoint_jaccard_first_half": float,
      "endpoint_jaccard_odd_even": float,
      "source": {"pr7_pairing": str, "pr7_burst": str, "pr6_anchoring": str}
    },
    "h4": {
      "epoch_hours": float,
      "n_epochs": int,
      "per_epoch_rate": [float, ...],
      "per_epoch_jaccard": [float, ...],
      "I_rate_epoch_order_shuffle": {...},
      "I_rate_circular_shift": {...},
      "I_geom": {...}
    },
    "exit_reason": "ok" | "missing_pr7_burst" | "missing_pr6_anchoring" | "lagpat_load_failed" | ...
  }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sef_itp_phase1 import resolve_lagpat_subject_dir
from src.interictal_propagation import load_subject_propagation_events, _valid_event_indices
from src.seeg_coord_loader import load_subject_coords
from src.sef_itp_phase2 import (
    extract_window_excess_from_pairing,
    extract_lag1_and_runlength_from_burst,
    extract_endpoint_jaccard_from_anchoring,
    slice_events_into_epochs,
    compute_local_endpoint,                         # v1.0 participation-field (supplementary now)
    compute_local_rank_endpoint,                    # v1.1 rank-based (main)
    compute_endpoint_spatial_radius,                # v1.1
    compute_source_sink_centroid_distance,          # v1.1
    compute_decision_k_drift,                       # v1.1
    endpoint_jaccard,
    compute_I_rate_normalized,
    compute_I_rate_normalized_circular_shift,
    compute_I_geom_normalized,
)


SCHEMA_VERSION = "sef_itp_phase2_v2_2026_05_24"   # bumped from v1 for v1.1 rank-endpoint


# Path conventions (consistent with Phase 1 + Phase 0).
PHASE1_COHORT_DIR = Path(
    "results/topic4_sef_itp/phase1_spatial_geometry/per_subject"
)
PR7_PAIRING_DIR = Path(
    "results/interictal_propagation_masked/template_pairing/per_subject"
)
PR7_BURST_DIR = Path(
    "results/interictal_propagation_masked/template_pairing/per_subject_burst"
)
PR6_ANCHORING_DIR = Path(
    "results/interictal_propagation_masked/template_anchoring/per_subject"
)
PHASE0A_DIR = Path("results/interictal_propagation_masked/per_subject")
RANK_DISPLACEMENT_DIR = Path(
    "results/interictal_propagation_masked/rank_displacement/per_subject"
)

OUT_DIR = Path("results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject")


# H3 metric windows + targets (framework v1.0.5 §3.3 lock).
H3_WINDOWS_S = (10.0, 30.0, 60.0, 1800.0)
DELTA_EXCESS = 0.05


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _global_endpoint_from_pr6(
    pr6_json: dict, channel_names: List[str]
) -> Dict[int, Dict[str, List[int]]]:
    """Extract per-cluster global endpoint from PR-6 anchoring JSON (top-k source / bottom-k sink).

    PR-6 stores `per_template[].source` and `.sink` as channel-name lists; map back to
    indices in `channel_names` (Phase 0a ordering).
    """
    out: Dict[int, Dict[str, List[int]]] = {}
    name_to_idx = {nm: i for i, nm in enumerate(channel_names)}
    for tpl in pr6_json.get("per_template", []):
        cid = int(tpl["cluster_id"])
        src_names = tpl.get("source") or []
        sink_names = tpl.get("sink") or []
        src_idx = [name_to_idx[nm] for nm in src_names if nm in name_to_idx]
        sink_idx = [name_to_idx[nm] for nm in sink_names if nm in name_to_idx]
        out[cid] = {"source": src_idx, "sink": sink_idx}
    return out


def _valid_mask_from_pr6(
    pr6_json: dict, n_ch: int
) -> np.ndarray:
    """Union of `valid_mask` across PR-6 per_template entries (any cluster's valid pool).

    Used as H4 endpoint-shuffle eligible pool. Channels valid in ANY cluster are eligible.
    """
    union = np.zeros(n_ch, dtype=bool)
    for tpl in pr6_json.get("per_template", []):
        vm = tpl.get("valid_mask")
        if vm is None or len(vm) != n_ch:
            continue
        union |= np.asarray(vm, dtype=bool)
    if not union.any():
        # fall back to all-channels eligible if PR-6 didn't supply a mask
        union[:] = True
    return union


def _spatial_radius_for_endpoint(
    endpoint_indices: List[int],
    coords: np.ndarray,
    mapped_mask: np.ndarray,
) -> Dict[str, float]:
    """Wrap compute_endpoint_spatial_radius with mapped-mask gating (NaN coord rows excluded).

    Returns NaN dict when < 2 mapped endpoint channels (insufficient for radius metrics).
    """
    mapped = [i for i in endpoint_indices if mapped_mask[i]]
    if len(mapped) < 2:
        return {
            "centroid_rms": float("nan"),
            "mean_pairwise": float("nan"),
            "min_enclosing_radius": float("nan"),
            "n_points": len(mapped),
            "n_unmapped_dropped": len(endpoint_indices) - len(mapped),
        }
    out = compute_endpoint_spatial_radius(mapped, coords)
    out["n_unmapped_dropped"] = len(endpoint_indices) - len(mapped)
    return out


def run_subject(
    dataset: str,
    subject_id: str,
    *,
    epoch_hours: float = 2.0,
    min_events: int = 10,
    n_perm: int = 1000,
    seed: int = 0,
    endpoint_k: int = 3,
    out_dir: Path = OUT_DIR,
) -> dict:
    """Run Phase 2 (H3 + H4) on one subject. Returns the output record dict.

    Writes the same record to `out_dir/<dataset>_<sid>.json` for cohort summarization.
    Records `exit_reason` field on failure paths instead of raising — cohort runs should
    not abort on one missing input.
    """
    stem = f"{dataset}_{subject_id}"
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "subject_id": subject_id,
        "epoch_hours": epoch_hours,
        "n_perm": n_perm,
        "seed": seed,
        "endpoint_k": endpoint_k,
        "h3": None,
        "h4": None,
        "exit_reason": "ok",
    }
    started = time.time()

    pairing_path = PR7_PAIRING_DIR / f"{stem}.json"
    burst_path = PR7_BURST_DIR / f"{stem}.json"
    anchoring_path = PR6_ANCHORING_DIR / f"{stem}.json"
    phase0a_path = PHASE0A_DIR / f"{stem}.json"

    # --- gate: required input files present ---
    for path, reason in [
        (pairing_path, "missing_pr7_pairing"),
        (burst_path, "missing_pr7_burst"),
        (anchoring_path, "missing_pr6_anchoring"),
        (phase0a_path, "missing_phase0a"),
    ]:
        if not path.exists():
            record["exit_reason"] = reason
            record["missing_path"] = str(path)
            _write_record(record, out_dir, stem)
            return record

    # --- H3 ingest ---
    pairing = _read_json(pairing_path)
    burst = _read_json(burst_path)
    anchoring = _read_json(anchoring_path)

    try:
        window_excess = extract_window_excess_from_pairing(pairing, H3_WINDOWS_S)
        lag1, run_length = extract_lag1_and_runlength_from_burst(burst)
        ep_fh, ep_oe = extract_endpoint_jaccard_from_anchoring(anchoring)
    except (KeyError, TypeError) as e:
        record["exit_reason"] = "h3_extract_failed"
        record["error"] = str(e)
        _write_record(record, out_dir, stem)
        return record

    record["h3"] = {
        "lag1_same_excess_n2": lag1,
        "window_excess_n2": {f"{w}": v for w, v in window_excess.items()},
        "run_length_lift_n2": run_length,
        "endpoint_jaccard_first_half": ep_fh,
        "endpoint_jaccard_odd_even": ep_oe,
        "delta_excess": DELTA_EXCESS,
        "source": {
            "pr7_pairing": str(pairing_path),
            "pr7_burst": str(burst_path),
            "pr6_anchoring": str(anchoring_path),
        },
    }

    # --- H4 prep: load PR-2 masked labels + lagPat events ---
    phase0a = _read_json(phase0a_path)
    labels = np.asarray(phase0a["adaptive_cluster"]["labels"], dtype=int)
    channel_names: List[str] = list(phase0a["channel_names"])

    try:
        lagpat_dir = resolve_lagpat_subject_dir(dataset, subject_id)
        loaded = load_subject_propagation_events(lagpat_dir)
    except Exception as e:
        record["exit_reason"] = "lagpat_load_failed"
        record["error"] = str(e)
        _write_record(record, out_dir, stem)
        return record

    raw_times = np.asarray(loaded["event_abs_times"], dtype=float)
    block_time_ranges = [tuple(tr) for tr in loaded["block_time_ranges"]]
    bools_full = np.asarray(loaded["bools"], dtype=int)  # (n_ch, n_events_total)

    # Apply PR-2 valid-event filter (matches PR-7 burst alignment exactly):
    #   valid_idx = events with n_participating >= 3 (default min_participating)
    #   finite_mask = events with finite abs_time AND non-negative label
    valid_idx = _valid_event_indices(bools_full, min_participating=3)
    if valid_idx.size != labels.size:
        record["exit_reason"] = "valid_event_count_mismatch"
        record["error"] = (
            f"valid_idx.size={valid_idx.size} != labels.size={labels.size} for {stem}"
        )
        _write_record(record, out_dir, stem)
        return record
    times = raw_times[valid_idx]
    finite_mask = np.isfinite(times) & (labels >= 0)
    event_abs_times = times[finite_mask]
    labels = labels[finite_mask]
    bools_valid = bools_full[:, valid_idx][:, finite_mask]  # (n_ch, n_valid)
    events_bool = bools_valid.T.astype(bool)  # → (n_valid, n_ch)
    n_ch = events_bool.shape[1]

    if events_bool.shape[1] != len(channel_names):
        record["exit_reason"] = "channel_count_mismatch"
        record["error"] = (
            f"events_bool n_ch={events_bool.shape[1]} != channel_names={len(channel_names)}"
        )
        _write_record(record, out_dir, stem)
        return record
    if events_bool.shape[0] == 0:
        record["exit_reason"] = "no_valid_events"
        _write_record(record, out_dir, stem)
        return record

    # Global endpoint from PR-6 anchoring (top-k by template_rank over full data).
    global_endpoint = _global_endpoint_from_pr6(anchoring, channel_names)
    valid_mask = _valid_mask_from_pr6(anchoring, n_ch)
    # Note: per-epoch rank endpoint extraction uses derived_valid (channel participation in
    # this epoch's cluster events), NOT a full-data PR-6 mask — see compute_local_rank_endpoint
    # docstring + user-review catch 2026-05-23. PR-6 per-cluster masks are used only by
    # scripts/b1_calibration_check.py for full-data parity audit.

    # --- v1.1: load 3D coords (mm) for spatial radius metrics ---
    coords: Optional[np.ndarray] = None
    mapped_mask: Optional[np.ndarray] = None
    coord_provenance: Optional[Dict[str, Any]] = None
    try:
        coord_result = load_subject_coords(
            dataset=dataset,
            subject_id=subject_id,
            channel_names_requested=channel_names,
        )
        coords = coord_result.coords_array_in_requested_order
        mapped_mask = coord_result.mapped_mask_in_requested_order
        coord_provenance = {
            "n_mapped": int(mapped_mask.sum()),
            "n_missing": int(len(channel_names) - mapped_mask.sum()),
            "coord_space": coord_result.coord_space,
            "coord_units": coord_result.coord_units,
            "normalization_certainty": coord_result.normalization_certainty,
        }
    except Exception as e:
        # Phase 1 cohort gate guarantees coords exist, but be defensive.
        record["coord_load_error"] = str(e)
        coords = None
        mapped_mask = None

    # --- v1.1: load rank_displacement primary_pair to decide decision-k drift subset ---
    rd_path = RANK_DISPLACEMENT_DIR / f"{stem}.json"
    swap_class: Optional[str] = None
    swap_decision_k: Optional[int] = None
    if rd_path.exists():
        rd = _read_json(rd_path)
        pairs = rd.get("pairs") or []
        if pairs:
            ss = (pairs[0] or {}).get("swap_sweep") or {}
            swap_class = ss.get("swap_class")
            swap_decision_k = ss.get("decision_k")

    # --- H4: epoch slicing ---
    # epoch_tolerance=0.1 accommodates Epilepsiae's natural ~59min41sec blocks vs 1h target
    # (and Yuquan's 24h continuous block which is well above any sensible epoch_hours).
    epochs = slice_events_into_epochs(
        event_abs_times=event_abs_times,
        cluster_labels=labels,
        block_time_ranges=block_time_ranges,
        epoch_hours=epoch_hours,
        min_events=min_events,
        epoch_tolerance=0.1,
    )
    if len(epochs) < 2:
        record["exit_reason"] = "insufficient_epochs"
        record["n_epochs"] = len(epochs)
        _write_record(record, out_dir, stem)
        return record

    # --- v1.0 supplementary (participation field): per-epoch participation top-k endpoint ---
    # Kept as sensitivity reference; framework v1.0.6 demotes this to supplementary.
    per_epoch_rate: List[float] = []
    per_epoch_local_v1: List[Dict[int, Dict[str, List[int]]]] = []
    per_epoch_jaccard_v1: List[float] = []
    # --- v1.1 main: per-epoch rank-based endpoint + spatial radius ---
    per_epoch_local_rank: List[Dict[int, Dict[str, List[int]]]] = []
    per_epoch_jaccard_rank: List[float] = []
    per_epoch_spatial: List[Dict[int, Dict[str, Dict[str, float]]]] = []
    cluster_ids = list(global_endpoint.keys())

    for ep in epochs:
        idx = ep["event_indices"]
        n_ev_epoch = len(idx)
        per_epoch_rate.append(n_ev_epoch / epoch_hours)

        # v1.0 supplementary
        if n_ev_epoch == 0:
            per_epoch_local_v1.append({c: global_endpoint[c] for c in cluster_ids})
            per_epoch_jaccard_v1.append(1.0)
        else:
            local_v1 = compute_local_endpoint(
                events_bool=events_bool[idx],
                labels=labels[idx],
                k=endpoint_k,
                valid_mask=valid_mask,
            )
            for c in cluster_ids:
                if c not in local_v1:
                    local_v1[c] = global_endpoint[c]
            per_epoch_local_v1.append(local_v1)
            js_v1 = [endpoint_jaccard(local_v1, global_endpoint, c) for c in cluster_ids]
            per_epoch_jaccard_v1.append(float(np.mean(js_v1)))

        # v1.1 main: rank-based endpoint via per-epoch template_rank
        # ranks/bools come from loaded; align indices to (n_ch, n_events_valid).
        ranks_v_full = np.asarray(loaded["ranks"])[:, valid_idx][:, finite_mask]
        bools_v_full = np.asarray(loaded["bools"])[:, valid_idx][:, finite_mask].astype(bool)
        # Pass NO external valid_mask — derived_valid (per-epoch participation per cluster)
        # is the correct gate (user-review catch 2026-05-23). PR-6 full-data mask is loaded
        # for B1 calibration parity testing only, not for per-epoch endpoint extraction.
        local_rank = compute_local_rank_endpoint(
            ranks=ranks_v_full,
            bools=bools_v_full,
            labels=labels,
            event_indices=idx,
            k=endpoint_k,
        )
        # Fallback: if a cluster has no events in this epoch, use the global endpoint
        # (perfect-match Jaccard contributes 1.0, no penalty — same convention as v1.0).
        for c in cluster_ids:
            if c not in local_rank:
                local_rank[c] = global_endpoint[c]
        per_epoch_local_rank.append(local_rank)
        js_rank = [endpoint_jaccard(local_rank, global_endpoint, c) for c in cluster_ids]
        per_epoch_jaccard_rank.append(float(np.mean(js_rank)))

        # v1.1: spatial radius per cluster, source + sink + axis (only if coords loaded)
        epoch_spatial: Dict[int, Dict[str, Dict[str, float]]] = {}
        if coords is not None and mapped_mask is not None:
            for c in cluster_ids:
                ep_local = local_rank[c]
                src_radius = _spatial_radius_for_endpoint(ep_local["source"], coords, mapped_mask)
                snk_radius = _spatial_radius_for_endpoint(ep_local["sink"], coords, mapped_mask)
                src_mapped = [i for i in ep_local["source"] if mapped_mask[i]]
                snk_mapped = [i for i in ep_local["sink"] if mapped_mask[i]]
                axis_dist = (
                    compute_source_sink_centroid_distance(src_mapped, snk_mapped, coords)
                    if (src_mapped and snk_mapped)
                    else float("nan")
                )
                epoch_spatial[c] = {
                    "source": src_radius,
                    "sink": snk_radius,
                    "source_sink_centroid_distance": axis_dist,
                }
        per_epoch_spatial.append(epoch_spatial)

    rates_arr = np.asarray(per_epoch_rate, dtype=float)

    # --- H4: I_rate (BOTH null methods; advisor catch B) ---
    I_rate_shuffle = compute_I_rate_normalized(rates_arr, n_perm=n_perm, seed=seed)
    I_rate_circshift = compute_I_rate_normalized_circular_shift(
        event_abs_times=event_abs_times,
        block_time_ranges=block_time_ranges,
        epoch_hours=epoch_hours,
        min_events=min_events,
        n_perm=n_perm,
        seed=seed,
    )

    # --- H4: I_geom on v1.1 rank-based endpoints (main line) ---
    endpoint_size = 2 * endpoint_k  # source + sink
    I_geom_rank = compute_I_geom_normalized(
        per_epoch_local=per_epoch_local_rank,
        global_endpoint=global_endpoint,
        valid_mask=valid_mask,
        endpoint_size=endpoint_size,
        n_perm=n_perm,
        seed=seed,
    )
    # --- H4: I_geom on v1.0 participation-field endpoints (supplementary) ---
    I_geom_participation = compute_I_geom_normalized(
        per_epoch_local=per_epoch_local_v1,
        global_endpoint=global_endpoint,
        valid_mask=valid_mask,
        endpoint_size=endpoint_size,
        n_perm=n_perm,
        seed=seed,
    )

    # --- v1.1: decision-k drift (user-return v2 catch 2026-05-23: compute for ALL subjects,
    # report swap_class as context — don't filter at runner level; let summarizer stratify).
    # For swap-positive (strict/candidate) subjects the drift is informative; for swap=none
    # subjects the per-epoch decision_k is a noise / control baseline.
    ranks_v_full = np.asarray(loaded["ranks"])[:, valid_idx][:, finite_mask]
    bools_v_full = np.asarray(loaded["bools"])[:, valid_idx][:, finite_mask].astype(bool)
    # primary_pair is between the two stable_k=2 clusters → cluster_a=0, cluster_b=1
    decision_k_drift_out = compute_decision_k_drift(
        ranks=ranks_v_full,
        bools=bools_v_full,
        labels=labels,
        epochs=epochs,
        cluster_a=0, cluster_b=1,
        min_events_per_cluster=20,
        n_perm=500,
        seed=seed,
    )

    record["h4"] = {
        "epoch_hours": epoch_hours,
        "n_epochs": len(epochs),
        "endpoint_k": endpoint_k,
        "endpoint_size_for_geom_null": endpoint_size,
        "per_epoch_rate": [float(r) for r in per_epoch_rate],
        "swap_class": swap_class,
        "swap_decision_k": swap_decision_k,
        "coord_provenance": coord_provenance,

        # v1.1 main line: rank-based endpoint geometry drift
        "rank_endpoint": {
            "per_epoch_jaccard": [float(j) for j in per_epoch_jaccard_rank],
            "per_epoch_local": [
                {str(c): v for c, v in le.items()} for le in per_epoch_local_rank
            ],
            "I_geom_rank": I_geom_rank,
        },

        # v1.1: spatial radius drift per cluster per epoch (source / sink / axis)
        "spatial_radius": {
            "per_epoch_per_cluster": [
                {str(c): v for c, v in es.items()} for es in per_epoch_spatial
            ],
        },

        # v1.1: decision-k drift (computed for ALL subjects; swap_class is context
        # for stratification in summarizer, NOT a runner-level gate)
        "decision_k_drift": {
            "computed": True,
            "swap_class_context": swap_class,
            "global_decision_k_context": swap_decision_k,
            "result": decision_k_drift_out,
        },

        # I_rate keep both null variants (no change from v1.0; user pending decision)
        "I_rate_epoch_order_shuffle": I_rate_shuffle,
        "I_rate_circular_shift": I_rate_circshift,

        # v1.0 supplementary: participation-field endpoint drift
        "supplementary_participation_field": {
            "per_epoch_jaccard": [float(j) for j in per_epoch_jaccard_v1],
            "per_epoch_local": [
                {str(c): v for c, v in le.items()} for le in per_epoch_local_v1
            ],
            "I_geom_participation": I_geom_participation,
        },
    }
    record["elapsed_seconds"] = time.time() - started
    _write_record(record, out_dir, stem)
    return record


def _write_record(record: dict, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(record, indent=2, default=float))


def enumerate_cohort(cohort_dir: Path = PHASE1_COHORT_DIR) -> List[Tuple[str, str]]:
    """Return (dataset, subject_id) tuples for all Phase 1 cohort subjects."""
    cohort: List[Tuple[str, str]] = []
    for p in sorted(cohort_dir.glob("*.json")):
        stem = p.stem
        if "_" not in stem:
            continue
        ds, sid = stem.split("_", 1)
        cohort.append((ds, sid))
    return cohort


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=str, default=None,
                    help="Single subject (format: '<dataset>_<sid>', e.g. yuquan_chengshuai). "
                         "Mutually exclusive with --all.")
    ap.add_argument("--all", action="store_true",
                    help="Run on the full Phase 1 cohort (n=23).")
    ap.add_argument("--epoch-hours", type=float, default=0.5,
                    help="Epoch size for H4 slicing (default 0.5h — gives circular-shift null "
                         "non-trivial degrees of freedom on both Yuquan 24h continuous and "
                         "Epilepsiae ~1h blocks). 1.0h works for Yuquan; 1.0h is degenerate "
                         "for Epilepsiae since epoch == block. 2.0h is sensitivity Yuquan-only.")
    ap.add_argument("--min-events", type=int, default=10,
                    help="Drop epochs with fewer events than this (default 10).")
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="Permutations for H4 null distributions (default 1000).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--endpoint-k", type=int, default=3,
                    help="Source/sink top-k for endpoint extraction (framework lock k=3).")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not args.subject and not args.all:
        ap.error("Specify --subject <dataset>_<sid> or --all")
    if args.subject and args.all:
        ap.error("--subject and --all are mutually exclusive")

    if args.subject:
        ds, sid = args.subject.split("_", 1)
        targets = [(ds, sid)]
    else:
        targets = enumerate_cohort()

    print(
        f"[phase2] running on {len(targets)} subject(s), "
        f"epoch_hours={args.epoch_hours}, n_perm={args.n_perm}"
    )
    n_ok = 0
    for (ds, sid) in targets:
        try:
            rec = run_subject(
                ds, sid,
                epoch_hours=args.epoch_hours,
                min_events=args.min_events,
                n_perm=args.n_perm,
                seed=args.seed,
                endpoint_k=args.endpoint_k,
                out_dir=args.output_dir,
            )
            if rec["exit_reason"] == "ok":
                n_ok += 1
                h4 = rec.get("h4") or {}
                rk = (h4.get("rank_endpoint") or {}).get("I_geom_rank") or {}
                ir = (h4.get("I_rate_circular_shift") or {}).get("I_rate", float("nan"))
                ig = rk.get("I_geom", float("nan"))
                dk_block = h4.get("decision_k_drift") or {}
                sc = dk_block.get("swap_class_context")
                dk_res = dk_block.get("result") or {}
                dk_mean = dk_res.get("decision_k_mean", float("nan"))
                print(
                    f"  {ds}_{sid}: ok (n_epochs={h4.get('n_epochs')}, "
                    f"I_rate_circshift={ir:.3f}, I_geom_rank={ig:.3f}, "
                    f"swap_class={sc}, dk_mean={dk_mean:.2f})"
                )
            else:
                print(f"  {ds}_{sid}: SKIP ({rec['exit_reason']})")
        except Exception as e:
            print(f"  {ds}_{sid}: FAILED ({type(e).__name__}: {e})")
    print(f"[phase2] {n_ok}/{len(targets)} subjects ok")


if __name__ == "__main__":
    main()

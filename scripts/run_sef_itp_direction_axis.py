"""SEF-ITP H2b runner — direction-axis disambiguation phase.

Plan: docs/archive/topic4/sef_itp_direction_axis/phase_h2b_direction_axis_plan_2026-05-25.md
Module: src/sef_itp_direction_axis.py

Per-subject pipeline:
  1. Load masked rank_displacement primary_pair JSON (channels, ranks, decision_k, swap_class, soz_mask).
  2. Load 3D mm coords via src.seeg_coord_loader.
  3. universe_mask = joint_valid AND mapped_mask.
  4. Derive source/sink halves for each cluster (lowest-k / highest-k dense rank).
  5. Compute template axis vectors v_A, v_B.
  6. Degeneracy detector (PCA on universe coords). near-1D = overriding verdict.
  7. Axis pair alignment (cos(v_A, ±v_B)) + permutation null on per-cluster role draws.
  8. Axis projection slope (rank_B on axis_A is the discriminative test).
  9. (Optional, --with-events) Per-event direction stats for cluster A and B.
  10. SOZ relation (descriptive only).
  11. Assess verdict (axis_reversal / dual_source / same_direction / inconclusive /
      degenerate_geometry / exit_no_universe).

Cohort step (--all then aggregate): writes cohort_summary.json with subject-level
verdict counts, stratified by swap_class. NO Wilcoxon, NO cohort p-value
(per archive plan §3.8 lock).

Usage:
    python scripts/run_sef_itp_direction_axis.py --subject epilepsiae_1146
    python scripts/run_sef_itp_direction_axis.py --subject epilepsiae_1146 epilepsiae_635 epilepsiae_1084
    python scripts/run_sef_itp_direction_axis.py --all
    python scripts/run_sef_itp_direction_axis.py --all --with-events
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

from src.interictal_propagation import load_subject_propagation_events, _valid_event_indices
from src.seeg_coord_loader import load_subject_coords
from src.sef_itp_phase1 import resolve_lagpat_subject_dir
from src import sef_itp_direction_axis as da


SCHEMA_VERSION = "sef_itp_direction_axis_v1_2026_05_25"

RANK_DISPLACEMENT_DIR = Path(
    "results/interictal_propagation_masked/rank_displacement/per_subject"
)
PHASE0A_DIR = Path("results/interictal_propagation_masked/per_subject")
OUT_DIR = Path("results/topic4_sef_itp/direction_axis/per_subject")
COHORT_OUT = Path("results/topic4_sef_itp/direction_axis/cohort_summary.json")


# ============================================================================
# I/O helpers
# ============================================================================


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _split_stem(stem: str) -> Tuple[str, str]:
    """`epilepsiae_1146` → ('epilepsiae', '1146'); `yuquan_chengshuai` → ('yuquan', 'chengshuai')."""
    if stem.startswith("yuquan_"):
        return "yuquan", stem[len("yuquan_"):]
    if stem.startswith("epilepsiae_"):
        return "epilepsiae", stem[len("epilepsiae_"):]
    raise ValueError(f"unknown stem prefix: {stem!r}")


# ============================================================================
# Per-subject runner
# ============================================================================


def run_one_subject(
    stem: str,
    with_events: bool = False,
    k_event: int = da.DEFAULT_K_EVENT,
    n_perm: int = da.DEFAULT_N_PERM,
    seed: int = da.DEFAULT_SEED,
) -> Dict[str, Any]:
    dataset, subject_id = _split_stem(stem)
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "framework_version": "v1.0.7",
        "dataset": dataset,
        "subject_id": subject_id,
        "stem": stem,
        "k_template": None,
        "k_event_default": int(k_event),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "with_events": bool(with_events),
        "exit_reason": None,
    }

    rd_path = RANK_DISPLACEMENT_DIR / f"{stem}.json"
    if not rd_path.exists():
        record["exit_reason"] = "missing_rank_displacement_json"
        record["verdict"] = {"label": "exit_no_universe", "reason": "rank_displacement JSON missing"}
        return record
    rd = _read_json(rd_path)
    pairs = rd.get("pairs") or []
    if not pairs:
        record["exit_reason"] = "rank_displacement_no_pairs"
        record["verdict"] = {"label": "exit_no_universe", "reason": "rank_displacement.pairs empty"}
        return record
    pp = pairs[0]
    channel_names: List[str] = list(pp["channel_names"])
    joint_valid = np.asarray(pp["joint_valid"], dtype=bool)
    rank_a = np.asarray(pp["rank_a_dense_full"], dtype=float)
    rank_b = np.asarray(pp["rank_b_dense_full"], dtype=float)
    ss = pp.get("swap_sweep") or {}
    swap_class = ss.get("swap_class", "unknown")
    decision_k = int(ss.get("decision_k") or 0)
    cluster_id_a = pp.get("cluster_id_a")
    cluster_id_b = pp.get("cluster_id_b")
    soz_mask = np.asarray(pp["soz_mask"], dtype=bool) if "soz_mask" in pp else None
    record["channel_names"] = channel_names
    record["swap_class"] = swap_class
    record["decision_k_from_rank_displacement"] = decision_k
    record["cluster_id_a"] = cluster_id_a
    record["cluster_id_b"] = cluster_id_b

    if decision_k <= 0:
        record["exit_reason"] = "rank_displacement_no_decision_k"
        record["verdict"] = {
            "label": "exit_no_universe",
            "reason": f"swap_sweep.decision_k={decision_k}",
        }
        return record

    # Load 3D coords (mm)
    try:
        coord_result = load_subject_coords(
            dataset=dataset, subject_id=subject_id, channel_names_requested=channel_names,
        )
        coords = coord_result.coords_array_in_requested_order
        mapped_mask = coord_result.mapped_mask_in_requested_order
        record["coord_space"] = coord_result.coord_space
        record["coord_units"] = coord_result.coord_units
        record["n_channels"] = len(channel_names)
        record["n_mapped"] = int(mapped_mask.sum())
    except Exception as e:
        record["exit_reason"] = "coord_load_failed"
        record["error"] = str(e)
        record["verdict"] = {"label": "exit_no_universe", "reason": f"coord load failed: {e}"}
        return record

    universe_mask = da.compute_universe_mask(joint_valid, mapped_mask)
    n_universe = int(universe_mask.sum())
    record["n_universe"] = n_universe

    if n_universe < da.MIN_UNIVERSE_SIZE or n_universe < 2 * decision_k:
        # Still report degeneracy for diagnostics
        deg = da.detect_degeneracy(coords, universe_mask) if n_universe >= 3 else None
        record["degeneracy"] = da.serialize_degeneracy(deg) if deg is not None else None
        verdict = da.SubjectVerdict(
            label="exit_no_universe",
            reason=f"n_universe={n_universe} < max({da.MIN_UNIVERSE_SIZE}, 2*decision_k={2*decision_k})",
        )
        record["verdict"] = {"label": verdict.label, "reason": verdict.reason}
        record["exit_reason"] = "insufficient_universe"
        return record

    # Layers 0/1: derive source/sink, template axes
    source_a_idx, sink_a_idx = da.derive_source_sink_within_universe(rank_a, universe_mask, decision_k)
    source_b_idx, sink_b_idx = da.derive_source_sink_within_universe(rank_b, universe_mask, decision_k)
    template_A = da.compute_template_axis(coords, source_a_idx, sink_a_idx, cluster_id=cluster_id_a or 0)
    template_B = da.compute_template_axis(coords, source_b_idx, sink_b_idx, cluster_id=cluster_id_b or 1)
    record["k_template"] = decision_k
    record["template_A"] = da.serialize_template_axis(template_A)
    record["template_B"] = da.serialize_template_axis(template_B)
    record["template_A_source_channels"] = [channel_names[i] for i in template_A.source_indices]
    record["template_A_sink_channels"] = [channel_names[i] for i in template_A.sink_indices]
    record["template_B_source_channels"] = [channel_names[i] for i in template_B.source_indices]
    record["template_B_sink_channels"] = [channel_names[i] for i in template_B.sink_indices]

    # Layer 4: degeneracy detector
    deg = da.detect_degeneracy(coords, universe_mask)
    record["degeneracy"] = da.serialize_degeneracy(deg)

    # Layer 1: axis pair alignment
    alignment = da.compute_axis_pair_alignment(
        coords, universe_mask, template_A, template_B, n_perm=n_perm, seed=seed,
    )
    record["axis_pair_alignment"] = da.serialize_alignment(alignment)

    # Layer 3: axis projection slopes (A on A is sanity; B on A is discriminative)
    slope_A_on_axisA = da.compute_axis_projection_slope(
        rank_a, universe_mask, coords, template_A.v_axis,
        cluster_id=cluster_id_a or 0, n_perm=n_perm, seed=seed,
    )
    slope_B_on_axisA = da.compute_axis_projection_slope(
        rank_b, universe_mask, coords, template_A.v_axis,
        cluster_id=cluster_id_b or 1, n_perm=n_perm, seed=seed,
    )
    record["slope_A_on_axisA"] = da.serialize_slope(slope_A_on_axisA)
    record["slope_B_on_axisA"] = da.serialize_slope(slope_B_on_axisA)

    # Layer 5: SOZ relation (v1.0.2 audit fix #5 — mapped_full + joint_universe both reported)
    soz = da.compute_soz_relation(coords, universe_mask, soz_mask, template_A, template_B,
                                  mapped_mask=mapped_mask)
    record["soz_relation"] = da.serialize_soz_relation(soz)

    # Layer 2 (optional): event-level
    if with_events:
        try:
            event_block = _compute_event_level_layer(
                dataset, subject_id, channel_names, universe_mask, coords,
                template_A, template_B, k_event=k_event,
            )
            record["event_direction"] = event_block
        except Exception as e:
            record["event_direction_error"] = str(e)

    # Verdict (strict gates) + descriptive geometry (no perm gate)
    verdict = da.assess_subject_verdict(n_universe, deg, alignment, slope_B_on_axisA)
    descriptive = da.assess_descriptive_geometry(n_universe, deg, alignment, slope_B_on_axisA)
    record["verdict"] = {"label": verdict.label, "reason": verdict.reason}
    record["descriptive_geometry"] = {"label": descriptive.label, "reason": descriptive.reason}
    record["exit_reason"] = None if not verdict.label.startswith("exit") else verdict.label
    return record


def _compute_event_level_layer(
    dataset: str,
    subject_id: str,
    channel_names: List[str],
    universe_mask: np.ndarray,
    coords: np.ndarray,
    template_A: da.TemplateAxis,
    template_B: da.TemplateAxis,
    k_event: int,
) -> Dict[str, Any]:
    """Load PR-2 phase0a labels + lagPat events; compute event-level direction for both clusters."""
    stem = f"{dataset}_{subject_id}"
    phase0a_path = PHASE0A_DIR / f"{stem}.json"
    if not phase0a_path.exists():
        return {"exit_reason": "missing_phase0a", "stats_A": None, "stats_B": None}
    phase0a = _read_json(phase0a_path)
    labels = np.asarray(phase0a["adaptive_cluster"]["labels"], dtype=int)
    p0a_channels = list(phase0a["channel_names"])
    if p0a_channels != channel_names:
        return {
            "exit_reason": "channel_name_mismatch_phase0a_vs_rank_displacement",
            "stats_A": None, "stats_B": None,
        }
    try:
        lagpat_dir = resolve_lagpat_subject_dir(dataset, subject_id)
        loaded = load_subject_propagation_events(lagpat_dir)
    except Exception as e:
        return {"exit_reason": f"lagpat_load_failed: {e}", "stats_A": None, "stats_B": None}

    bools_full = np.asarray(loaded["bools"], dtype=bool)  # (n_ch, n_events_total)
    ranks_full = np.asarray(loaded["ranks"], dtype=float)
    lagpat_channels = list(loaded["channel_names"])
    if ranks_full.shape[0] != len(channel_names):
        # Lagpat channel order may not match rank_displacement / phase0a order; bail with
        # a recorded exit rather than silently mismatching channels.
        return {
            "exit_reason": f"lagpat_channel_count_mismatch ({ranks_full.shape[0]} vs {len(channel_names)})",
            "stats_A": None, "stats_B": None,
        }
    # v1.0.2 audit fix #6: explicitly assert lagpat channel ORDER matches rank_displacement
    # channel ORDER, not just count. AGENTS.md cross-PR contract warns that lagpat NPZ may
    # order channels differently per block. Silent order mismatch → event Layer 2 cos values
    # would be computed with the wrong coords-to-rank mapping → silent science pollution.
    if lagpat_channels != channel_names:
        return {
            "exit_reason": (
                f"lagpat_channel_order_mismatch (vs rank_displacement; "
                f"first 5 lagpat={lagpat_channels[:5]} vs rank_displacement={channel_names[:5]})"
            ),
            "stats_A": None, "stats_B": None,
        }
    valid_idx = _valid_event_indices(bools_full, min_participating=3)
    if valid_idx.size != labels.size:
        return {
            "exit_reason": f"valid_event_count_mismatch ({valid_idx.size} vs {labels.size})",
            "stats_A": None, "stats_B": None,
        }
    finite_mask = labels >= 0
    ranks_valid = ranks_full[:, valid_idx][:, finite_mask]
    bools_valid = bools_full[:, valid_idx][:, finite_mask]
    labels_valid = labels[finite_mask]

    stats_A = da.compute_event_direction_stats(
        ranks_valid, bools_valid, labels_valid, universe_mask, coords,
        template_A.v_axis, cluster_id=template_A.cluster_id, k_event=k_event,
    )
    stats_B = da.compute_event_direction_stats(
        ranks_valid, bools_valid, labels_valid, universe_mask, coords,
        template_A.v_axis, cluster_id=template_B.cluster_id, k_event=k_event,
    )
    return {
        "exit_reason": None,
        "k_event": int(k_event),
        "stats_A_on_axisA": da.serialize_event_stats(stats_A),
        "stats_B_on_axisA": da.serialize_event_stats(stats_B),
    }


# ============================================================================
# Cohort aggregation
# ============================================================================


def aggregate_cohort(per_subject_dir: Path = OUT_DIR) -> Dict[str, Any]:
    """Aggregate per-subject JSONs into cohort_summary.json (counts only — no Wilcoxon)."""
    files = sorted(per_subject_dir.glob("*.json"))
    verdicts = []
    for f in files:
        rec = _read_json(f)
        verdicts.append({
            "stem": rec.get("stem", f.stem),
            "swap_class": rec.get("swap_class", "unknown"),
            "verdict": (rec.get("verdict") or {}).get("label", "missing"),
            "descriptive": (rec.get("descriptive_geometry") or {}).get("label", "missing"),
            "n_universe": rec.get("n_universe", 0),
            "cos_A_neg_B": (rec.get("axis_pair_alignment") or {}).get("cos_A_neg_B"),
            "cos_A_pos_B": (rec.get("axis_pair_alignment") or {}).get("cos_A_pos_B"),
            "p_axis_reversal": (rec.get("axis_pair_alignment") or {}).get("p_one_sided_axis_reversal"),
            "slope_B_on_axisA": (rec.get("slope_B_on_axisA") or {}).get("slope"),
            "r2_B_on_axisA": (rec.get("slope_B_on_axisA") or {}).get("r2"),
            "p_slope_neg": (rec.get("slope_B_on_axisA") or {}).get("p_one_sided_neg_slope"),
            "lambda_ratio_12": (rec.get("degeneracy") or {}).get("lambda_ratio_12"),
            "lambda_ratio_23": (rec.get("degeneracy") or {}).get("lambda_ratio_23"),
        })
    distribution: Dict[str, int] = {}
    desc_distribution: Dict[str, int] = {}
    by_class: Dict[str, Dict[str, int]] = {}
    desc_by_class: Dict[str, Dict[str, int]] = {}
    for v in verdicts:
        label = v["verdict"]
        distribution[label] = distribution.get(label, 0) + 1
        desc = v["descriptive"]
        desc_distribution[desc] = desc_distribution.get(desc, 0) + 1
        c = v["swap_class"] or "unknown"
        by_class.setdefault(c, {})
        by_class[c][label] = by_class[c].get(label, 0) + 1
        desc_by_class.setdefault(c, {})
        desc_by_class[c][desc] = desc_by_class[c].get(desc, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION + "_cohort",
        "framework_version": "v1.0.7",
        "tier_lock": "descriptive_supplementary_per_subject_verdict_only",
        "n_subjects": len(verdicts),
        "verdict_distribution": distribution,
        "verdict_by_swap_class": by_class,
        "descriptive_geometry_distribution": desc_distribution,
        "descriptive_geometry_by_swap_class": desc_by_class,
        "constants": {
            "k_template": "decision_k_from_rank_displacement (per-subject)",
            "k_event_default": da.DEFAULT_K_EVENT,
            "n_perm": da.DEFAULT_N_PERM,
            "seed": da.DEFAULT_SEED,
            "degeneracy_lambda_ratio_12_threshold": da.DEGENERACY_LAMBDA_RATIO_12_THRESHOLD,
            "cos_axis_reversal_threshold": da.COS_AXIS_REVERSAL_THRESHOLD,
            "cos_same_direction_threshold": da.COS_SAME_DIRECTION_THRESHOLD,
            "dual_source_cos_abs_max": da.DUAL_SOURCE_COS_ABS_MAX,
            "r2_dual_source_max": da.R2_DUAL_SOURCE_MAX,
            "slope_pvalue_threshold": da.SLOPE_PVALUE_THRESHOLD,
            "axis_perm_pvalue_threshold": da.AXIS_PERM_PVALUE_THRESHOLD,
        },
        "subjects": verdicts,
    }


# ============================================================================
# CLI
# ============================================================================


def _enumerate_all_stems() -> List[str]:
    return sorted([p.stem for p in RANK_DISPLACEMENT_DIR.glob("*.json")])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subject", nargs="+", help="One or more stems (e.g., epilepsiae_1146).")
    ap.add_argument("--all", action="store_true", help="Run on all rank_displacement subjects.")
    ap.add_argument("--with-events", action="store_true",
                    help="Also compute Layer 2 event-level direction stats (needs lagPat + phase0a).")
    ap.add_argument("--k-event", type=int, default=da.DEFAULT_K_EVENT)
    ap.add_argument("--n-perm", type=int, default=da.DEFAULT_N_PERM)
    ap.add_argument("--seed", type=int, default=da.DEFAULT_SEED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cohort-out", type=Path, default=COHORT_OUT)
    ap.add_argument("--skip-aggregate", action="store_true",
                    help="Skip cohort_summary.json aggregation.")
    args = ap.parse_args()

    if args.all and not args.subject:
        stems = _enumerate_all_stems()
    elif args.subject:
        stems = list(args.subject)
    else:
        ap.error("Pass --subject <stem...> or --all")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        t_s = time.time()
        try:
            rec = run_one_subject(
                stem,
                with_events=args.with_events,
                k_event=args.k_event,
                n_perm=args.n_perm,
                seed=args.seed,
            )
        except Exception as e:
            rec = {
                "schema_version": SCHEMA_VERSION,
                "stem": stem,
                "exit_reason": "runner_crash",
                "error": str(e),
                "verdict": {"label": "exit_no_universe", "reason": f"runner crash: {e}"},
            }
        out_path = args.out_dir / f"{stem}.json"
        _write_json(rec, out_path)
        v = (rec.get("verdict") or {}).get("label", "?")
        print(f"[{i}/{len(stems)}] {stem}: verdict={v} t={time.time()-t_s:.1f}s")
    if not args.skip_aggregate and args.all:
        cohort = aggregate_cohort(args.out_dir)
        _write_json(cohort, args.cohort_out)
        print(f"Cohort summary → {args.cohort_out}")
        print(f"Verdict distribution: {cohort['verdict_distribution']}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

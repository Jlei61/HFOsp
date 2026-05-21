"""SEF-ITP Phase 1 CLI runner — H6 / H1 / H2 spatial geometry hypotheses.

Plan: docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md
Framework: docs/topic4_sef_itp_framework.md v1.0.2
Module: src/sef_itp_phase1.py

CURRENT STATUS (2026-05-21):
  - --dry-run mode: WORKING (synthetic input, exercise the pipeline end-to-end)
  - real-data mode: BLOCKED on:
      (a) Phase 0a phantom-rank fix Step 5f (PR-6 endpoint anchoring on masked)
      (b) Phase 0b coord-loader PR (NOT YET STARTED, see Topic 0 §3.2)
      (c) implementation of load_subject_for_phase1() — currently raises
          NotImplementedError to prevent accidental autonomous execution

When prerequisites land, fill in load_subject_for_phase1() and drop the
NotImplementedError guard.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Allow direct invocation: `python scripts/run_sef_itp_phase1.py ...`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sef_itp_phase1 import (
    compute_h1_full,
    compute_h2_set_reversal,
    compute_h2_spatial_reversal,
    compute_h6_segregation,
    compute_participation_rate,
)


SCHEMA_VERSION = "sef_itp_phase1_v1_2026_05_21"


@dataclass
class SubjectPhase1Data:
    """Per-subject input for Phase 1 hypotheses.

    Real-data wiring (TODO): load from per-subject masked JSON + PR-6 endpoint JSON
    + coord file (Phase 0b output) once those exist.
    """

    subject_id: str
    channel_names: List[str]
    events_bool: np.ndarray  # (n_ch, n_events) boolean
    coords: Optional[np.ndarray]  # (n_ch, 3) — None if Phase 0b not complete
    hfo_rate: Optional[np.ndarray]  # (n_ch,) optional — per-channel HFO rate
    # PR-6 endpoint output per cluster:
    cluster_endpoints: Dict[int, Dict[str, List[int]]]  # cluster_id → {"S": [...], "K": [...]}
    valid_indices_per_cluster: Dict[int, List[int]]  # cluster_id → list of valid channel indices
    forward_reverse_pairs: List[Dict]  # [{"cluster_A_id": int, "cluster_B_id": int, ...}, ...]


def load_subject_for_phase1(subject_json_path: Path) -> SubjectPhase1Data:
    """Load per-subject Phase 1 input from masked Phase 0a output + Phase 0b coords.

    CURRENTLY UNIMPLEMENTED: depends on
      - Phase 0a §5f PR-6 endpoint anchoring on masked features (not done)
      - Phase 0b coord loader integration (loader EXISTS as of 2026-05-21:
        src/seeg_coord_loader.py with v2 strict schema; this function needs to
        wire it up + add coord_units == "mm" assertion before passing into
        compute_h1_compactness / compute_h2_spatial_reversal)

    Will read:
      - results/interictal_propagation_masked/per_subject/<sid>.json (Phase 0a 5a output)
      - results/interictal_propagation_masked/pr6_*/per_subject/<sid>.json (Phase 0a 5f output)

    Integration contract (when wiring up):
      from src.seeg_coord_loader import load_subject_coords, assert_coord_result_is_mm_for_main_analysis
      coord_result = load_subject_coords(dataset, subject_id, channel_names)
      assert_coord_result_is_mm_for_main_analysis(coord_result)   # CRITICAL — voxel rejected
      coords = coord_result.coords_array_in_requested_order        # aligned to channel_names
      mask = coord_result.mapped_mask_in_requested_order            # for filtering
    """
    raise NotImplementedError(
        "load_subject_for_phase1 is blocked on Phase 0a §5f. "
        "Coord loader (Phase 0b) is COMPLETE: src/seeg_coord_loader.py + 28 tests GREEN. "
        "See docs/archive/topic4/sef_itp_phase1/coord_loader_plan_2026-05-21.md."
    )


def make_synthetic_subject(seed: int = 0) -> SubjectPhase1Data:
    """Generate synthetic subject input for --dry-run mode."""
    rng = np.random.default_rng(seed)
    n_ch = 30
    n_events = 500

    # Layout: 3 shafts × 10 channels
    names = []
    coords = []
    for shaft_idx, prefix in enumerate(["A", "B", "C"]):
        x = shaft_idx * 50.0
        for i in range(10):
            names.append(f"{prefix}{i+1}-{prefix}{i+2}")
            coords.append([x, 0, i * 3.0])
    coords = np.asarray(coords, dtype=float)

    # Participation: high on shaft A first few channels, decreasing
    base_p = np.concatenate([
        np.linspace(0.9, 0.3, 10),  # shaft A: high at start, low at end
        np.linspace(0.3, 0.9, 10),  # shaft B: low at start, high at end (cross pattern)
        rng.uniform(0.2, 0.6, size=10),  # shaft C: random mid
    ])
    events_bool = (rng.uniform(0, 1, size=(n_ch, n_events)) < base_p[:, None])

    # Synthetic PR-6 endpoint output: forward template (A's deep channels as source,
    # B's superficial as sink), reverse template (B's superficial as source, A's deep as sink)
    cluster_endpoints = {
        0: {"S": [0, 1, 2], "K": [17, 18, 19]},  # forward
        1: {"S": [17, 18, 19], "K": [0, 1, 2]},  # reverse (perfect swap)
    }
    valid_indices_per_cluster = {0: list(range(30)), 1: list(range(30))}
    fwd_rev_pairs = [{"cluster_A_id": 0, "cluster_B_id": 1, "reproducibility_source": "synthetic"}]

    return SubjectPhase1Data(
        subject_id="synthetic_subject_001",
        channel_names=names,
        events_bool=events_bool,
        coords=coords,
        hfo_rate=rng.uniform(1, 10, size=n_ch),
        cluster_endpoints=cluster_endpoints,
        valid_indices_per_cluster=valid_indices_per_cluster,
        forward_reverse_pairs=fwd_rev_pairs,
    )


def run_h6(
    subject: SubjectPhase1Data,
    distance_metric: str = "shaft_ordinal",
    n_permutations: int = 1000,
    rng_seed: int = 0,
) -> Dict:
    """Run H6 — participation field spatial segregation."""
    participation = compute_participation_rate(subject.events_bool)
    coords = subject.coords if distance_metric == "euclidean" else None
    if distance_metric == "euclidean" and coords is None:
        return {
            "verdict": "GATED_ON_PHASE_0B_COORD_LOADER",
            "reason": "euclidean distance requires 3D coords; switch to shaft_ordinal or wait for Phase 0b",
        }
    out = compute_h6_segregation(
        participation=participation,
        channel_names=subject.channel_names,
        coords=coords,
        distance_metric=distance_metric,
        n_permutations=n_permutations,
        rng=np.random.default_rng(rng_seed),
    )
    return out


def run_h1(
    subject: SubjectPhase1Data,
    distance_metric: str = "euclidean",
    n_null: int = 1000,
    rng_seed: int = 0,
) -> Dict:
    """Run H1 (H1a + H1b + H1c) per cluster."""
    if distance_metric == "euclidean" and subject.coords is None:
        return {
            "verdict": "GATED_ON_PHASE_0B_COORD_LOADER",
            "reason": "H1 main analysis requires 3D coords (euclidean); shaft_ordinal limited",
        }

    participation = compute_participation_rate(subject.events_bool)
    rng = np.random.default_rng(rng_seed)

    per_cluster: Dict[int, Dict] = {}
    for cluster_id, ep in subject.cluster_endpoints.items():
        valid = subject.valid_indices_per_cluster[cluster_id]
        out = compute_h1_full(
            source_indices=ep["S"],
            sink_indices=ep["K"],
            valid_indices=valid,
            coords=subject.coords,
            channel_names=subject.channel_names,
            participation=participation,
            hfo_rate=subject.hfo_rate,
            distance_metric=distance_metric,
            n_null=n_null,
            rng=rng,
        )
        per_cluster[int(cluster_id)] = out

    return {"per_cluster": per_cluster}


def run_h2(
    subject: SubjectPhase1Data,
    n_null: int = 1000,
    rng_seed: int = 0,
) -> Dict:
    """Run H2 (set + spatial reversal index) per forward/reverse pair."""
    rng = np.random.default_rng(rng_seed)
    per_pair: List[Dict] = []
    for pair in subject.forward_reverse_pairs:
        a_id = pair["cluster_A_id"]
        b_id = pair["cluster_B_id"]
        S_A = subject.cluster_endpoints[a_id]["S"]
        K_A = subject.cluster_endpoints[a_id]["K"]
        S_B = subject.cluster_endpoints[b_id]["S"]
        K_B = subject.cluster_endpoints[b_id]["K"]

        h2_set = compute_h2_set_reversal(S_A, K_A, S_B, K_B, n_null=n_null, rng=rng)
        if subject.coords is not None:
            h2_spatial = compute_h2_spatial_reversal(
                S_A, K_A, S_B, K_B, subject.coords, n_null=n_null, rng=rng
            )
        else:
            h2_spatial = {
                "verdict": "GATED_ON_PHASE_0B_COORD_LOADER",
                "reason": "spatial reversal requires 3D coords",
            }

        # Integrated H2 verdict
        v_set = h2_set.get("verdict")
        v_spatial = h2_spatial.get("verdict")
        if v_set == "PASS" and v_spatial == "PASS":
            integrated = "PASS"
        elif v_set == "PASS" or v_spatial == "PASS":
            integrated = "partial_PASS"
        elif v_set == "FAIL" or v_spatial == "FAIL":
            integrated = "FAIL"
        else:
            integrated = "NULL"

        per_pair.append({
            "cluster_A_id": a_id,
            "cluster_B_id": b_id,
            "reproducibility_source": pair.get("reproducibility_source"),
            "h2_set_reversal": h2_set,
            "h2_spatial_reversal": h2_spatial,
            "h2_integrated_verdict": integrated,
        })
    return {"per_pair": per_pair}


def run_phase1_subject(
    subject: SubjectPhase1Data,
    hypothesis: str,
    distance_metric: str,
    n_permutations: int,
    n_null: int,
    rng_seed: int,
) -> Dict:
    """Top-level per-subject runner."""
    out: Dict = {
        "schema_version": SCHEMA_VERSION,
        "subject_id": subject.subject_id,
        "n_channels": len(subject.channel_names),
        "n_events": subject.events_bool.shape[1],
        "distance_metric": distance_metric,
        "framework_ref": "topic4_sef_itp_framework.md v1.0.2",
        "plan_ref": "docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md",
    }
    if hypothesis in ("h6", "all"):
        out["h6"] = run_h6(subject, distance_metric, n_permutations, rng_seed)
    if hypothesis in ("h1", "all"):
        out["h1"] = run_h1(subject, distance_metric, n_null, rng_seed)
    if hypothesis in ("h2", "all"):
        out["h2"] = run_h2(subject, n_null, rng_seed)
    return out


def _to_jsonable(obj):
    """Recursively convert numpy types to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SEF-ITP Phase 1 — H6 / H1 / H2 spatial geometry hypotheses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with synthetic data (no real-data dependencies)")
    parser.add_argument("--dataset", choices=["epilepsiae", "yuquan", "both"], default="both")
    parser.add_argument("--subject", type=str, default=None,
                        help="Single subject ID; omit with --all for cohort")
    parser.add_argument("--all", action="store_true",
                        help="Run all Tier 1 subjects (cohort mode)")
    parser.add_argument("--hypothesis", choices=["h6", "h1", "h2", "all"], default="all")
    parser.add_argument("--distance-metric", choices=["euclidean", "shaft_ordinal"],
                        default="shaft_ordinal",
                        help="euclidean requires Phase 0b coords; shaft_ordinal is fallback")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/topic4_sef_itp/_dry_run_artifacts"),
                        help="default = _dry_run_artifacts/ to avoid polluting real Phase 1 cohort path")
    args = parser.parse_args(argv)

    if not args.dry_run:
        print(
            "ERROR: real-data mode is BLOCKED on Phase 0a §5f + Phase 0b coord-loader.\n"
            "Use --dry-run to exercise the pipeline on synthetic data.\n"
            "See docs/topic0_methodology_audits.md §3.2 and "
            "docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md §1.",
            file=sys.stderr,
        )
        return 2

    print("[dry-run] generating synthetic subject…", file=sys.stderr)
    subject = make_synthetic_subject(seed=args.rng_seed)
    result = run_phase1_subject(
        subject=subject,
        hypothesis=args.hypothesis,
        distance_metric=args.distance_metric,
        n_permutations=args.n_permutations,
        n_null=args.n_null,
        rng_seed=args.rng_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"dry_run_{subject.subject_id}.json"
    with open(out_path, "w") as f:
        json.dump(_to_jsonable(result), f, indent=2)
    print(f"[dry-run] wrote {out_path}", file=sys.stderr)
    print(f"[dry-run] verdict summary:", file=sys.stderr)
    for h in ("h6", "h1", "h2"):
        block = result.get(h, {})
        if h == "h1":
            for cid, body in block.get("per_cluster", {}).items():
                print(f"  H1 cluster {cid}: {body.get('h1_overall_verdict')}", file=sys.stderr)
        elif h == "h2":
            for entry in block.get("per_pair", []):
                print(f"  H2 pair {entry['cluster_A_id']}<->{entry['cluster_B_id']}: "
                      f"{entry.get('h2_integrated_verdict')}", file=sys.stderr)
        elif h == "h6":
            print(f"  H6: {block.get('verdict')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

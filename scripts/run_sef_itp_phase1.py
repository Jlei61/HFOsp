"""SEF-ITP Phase 1 CLI runner — H6 / H1 / H2 spatial geometry hypotheses.

Plan: docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md
Framework: docs/topic4_sef_itp_framework.md v1.0.2
Module: src/sef_itp_phase1.py

CURRENT STATUS (2026-05-21):
  - --dry-run mode: WORKING (synthetic input, exercise the pipeline end-to-end)
  - real-data mode: WIRED via load_subject_for_phase1() — consumes
      (a) masked Phase 0a per-subject JSON
          (results/interictal_propagation_masked/per_subject/<dataset>_<sid>.json)
      (b) masked PR-6 template_anchoring JSON
          (results/interictal_propagation_masked/template_anchoring/per_subject/<dataset>_<sid>.json)
      (c) lagPat NPZ — events_bool aligned to Phase 0a channel_names
          via src.interictal_propagation.load_subject_propagation_events
      (d) SEEG coords via src.seeg_coord_loader.load_subject_coords —
          Yuquan fs_native_ras_mm, Epilepsiae MNI152 1mm via auto-discovered
          MRI affine; assert_coord_result_is_mm_for_main_analysis enforces mm
          before euclidean compute (v1.0.5 voxel-rejection contract).
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
from src.seeg_coord_loader import (
    assert_coord_result_is_mm_for_main_analysis,
    load_subject_coords,
)
from src.interictal_propagation import load_subject_propagation_events


# Path conventions (mirror scripts/run_interictal_propagation.py).
DEFAULT_YUQUAN_LAGPAT_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
DEFAULT_EPILEPSIAE_LAGPAT_ROOT = Path(
    "/mnt/epilepsia_data/interilca_inter_results/all_data_lns"
)
DEFAULT_PHASE0A_ROOT = Path("results/interictal_propagation_masked/per_subject")
DEFAULT_PR6_ROOT = Path(
    "results/interictal_propagation_masked/template_anchoring/per_subject"
)


SCHEMA_VERSION = "sef_itp_phase1_v1_2026_05_21"


@dataclass
class SubjectPhase1Data:
    """Per-subject input for Phase 1 hypotheses.

    Real-data wiring (2026-05-21): populated by load_subject_for_phase1() from
    masked Phase 0a JSON + masked PR-6 JSON + lagPat NPZ + seeg coord loader.
    """

    subject_id: str
    dataset: str  # "yuquan" | "epilepsiae"  — needed for downstream provenance
    channel_names: List[str]
    events_bool: np.ndarray  # (n_ch, n_events) boolean
    coords: Optional[np.ndarray]  # (n_ch, 3) — NaN row for unmapped channels
    coord_units: Optional[str]  # "mm" | "voxel" | None — gates euclidean compute
    coord_space: Optional[str]  # provenance for output JSON
    mapped_mask: Optional[np.ndarray]  # (n_ch,) bool — True where coord is mapped
    hfo_rate: Optional[np.ndarray]  # (n_ch,) optional — per-channel HFO rate
    # PR-6 endpoint output per cluster:
    cluster_endpoints: Dict[int, Dict[str, List[int]]]  # cluster_id → {"S": [...], "K": [...]}
    valid_indices_per_cluster: Dict[int, List[int]]  # cluster_id → list of valid channel indices
    forward_reverse_pairs: List[Dict]  # [{"cluster_A_id": int, "cluster_B_id": int, ...}, ...]
    # Audit / provenance:
    coord_provenance: Optional[Dict] = None
    n_dropped_endpoints_no_coords_per_cluster: Optional[Dict[int, int]] = None


def _parse_dataset_subject_from_filename(stem: str) -> Tuple[str, str]:
    """Parse 'epilepsiae_1073' → ('epilepsiae', '1073'); 'yuquan_chengshuai' → ('yuquan', 'chengshuai')."""
    for prefix in ("epilepsiae_", "yuquan_"):
        if stem.startswith(prefix):
            return prefix[:-1], stem[len(prefix):]
    raise ValueError(
        f"Cannot infer dataset from filename stem {stem!r}; expected "
        f"'epilepsiae_<sid>' or 'yuquan_<sid>'"
    )


def _resolve_lagpat_subject_dir(
    dataset: str,
    subject_id: str,
    yuquan_root: Path,
    epilepsiae_root: Path,
) -> Path:
    """Mirror scripts/run_interictal_propagation.py:_subject_dir + _epilepsiae_subject_dir."""
    if dataset == "yuquan":
        return yuquan_root / subject_id
    legacy = epilepsiae_root / subject_id / "all_recs"
    return legacy if legacy.exists() else epilepsiae_root / subject_id


def load_subject_for_phase1(
    subject_json_path: Path,
    *,
    pr6_anchoring_root: Path = DEFAULT_PR6_ROOT,
    yuquan_lagpat_root: Path = DEFAULT_YUQUAN_LAGPAT_ROOT,
    epilepsiae_lagpat_root: Path = DEFAULT_EPILEPSIAE_LAGPAT_ROOT,
    require_coords: bool = True,
    allow_voxel_fallback: bool = False,
) -> SubjectPhase1Data:
    """Load per-subject Phase 1 input from 4 aligned sources.

    Inputs (all derived from `subject_json_path` + path roots):
      1. Masked Phase 0a per-subject JSON (channel_names, adaptive_cluster
         with cluster_id list + candidate_forward_reverse_pairs).
      2. Masked PR-6 template_anchoring JSON (per_template[] with
         source/sink/valid_mask).
      3. lagPat NPZ → events_bool (via load_subject_propagation_events).
      4. SEEG coords via load_subject_coords; assert mm units before return.

    Contract clauses (CLAUDE.md §6):
      - **Channel alignment**: Phase 0a channel_names MUST equal lagPat-loader
        channel_names; raise on mismatch.
      - **Endpoint name→index**: PR-6 source/sink names converted via Phase 0a
        ordering; raise on any name not found.
      - **coord_units mm**: assert_coord_result_is_mm_for_main_analysis called
        when require_coords=True (forbids silent voxel→euclidean).
      - **Valid pool intersection**: valid_indices = PR-6 valid_mask ∩
        coord mapped_mask (when require_coords=True); endpoint S/K filtered
        to mapped channels; n_dropped audited per cluster.
      - **Subject ID canonicalization**: handled inside load_subject_coords
        for Epilepsiae (e.g., '1073' → '107302').

    Returns:
        SubjectPhase1Data populated end-to-end.

    Raises:
        FileNotFoundError: any of the 4 source files / dirs missing.
        ValueError: subject/dataset mismatch, channel-name mismatch,
                    endpoint name not in channel_names, voxel coords when
                    require_coords=True, etc.
    """
    subject_json_path = Path(subject_json_path)
    if not subject_json_path.exists():
        raise FileNotFoundError(f"Phase 0a JSON not found: {subject_json_path}")

    dataset, subject_id = _parse_dataset_subject_from_filename(subject_json_path.stem)

    # === Step 1: Phase 0a JSON ===
    with open(subject_json_path) as f:
        phase0a = json.load(f)

    if phase0a.get("subject") != subject_id:
        raise ValueError(
            f"Phase 0a JSON subject {phase0a.get('subject')!r} != "
            f"filename subject {subject_id!r}"
        )
    if phase0a.get("dataset") != dataset:
        raise ValueError(
            f"Phase 0a JSON dataset {phase0a.get('dataset')!r} != "
            f"filename dataset {dataset!r}"
        )

    phase0a_channels: List[str] = list(phase0a["channel_names"])
    n_ch = len(phase0a_channels)
    if n_ch != phase0a["n_channels"]:
        raise ValueError(
            f"Phase 0a channel_names length {n_ch} != n_channels {phase0a['n_channels']}"
        )

    # === Step 2: PR-6 anchoring JSON ===
    pr6_path = pr6_anchoring_root / subject_json_path.name
    if not pr6_path.exists():
        raise FileNotFoundError(f"PR-6 anchoring JSON not found: {pr6_path}")
    with open(pr6_path) as f:
        pr6 = json.load(f)
    if pr6.get("subject_id") != subject_id or pr6.get("dataset") != dataset:
        raise ValueError(
            f"PR-6 JSON identity ({pr6.get('dataset')!r}, {pr6.get('subject_id')!r}) "
            f"!= ({dataset!r}, {subject_id!r})"
        )

    # === Step 3: events_bool via lagPat loader ===
    lagpat_dir = _resolve_lagpat_subject_dir(
        dataset, subject_id, yuquan_lagpat_root, epilepsiae_lagpat_root
    )
    if not lagpat_dir.exists():
        raise FileNotFoundError(f"lagPat subject dir not found: {lagpat_dir}")

    loaded = load_subject_propagation_events(lagpat_dir)
    events_bool = np.asarray(loaded["bools"], dtype=bool)
    lagpat_channels = list(loaded["channel_names"])

    if lagpat_channels != phase0a_channels:
        raise ValueError(
            f"Channel-name mismatch between Phase 0a and lagPat loader for "
            f"{dataset}/{subject_id}.\n"
            f"  Phase 0a: {phase0a_channels}\n"
            f"  lagPat:   {lagpat_channels}\n"
            f"Re-run Phase 0a §5a if lagPat NPZ was regenerated."
        )
    if events_bool.shape[0] != n_ch:
        raise ValueError(
            f"events_bool n_ch {events_bool.shape[0]} != Phase 0a n_channels {n_ch}"
        )

    # === Step 4: SEEG coords ===
    coords: Optional[np.ndarray] = None
    coord_units: Optional[str] = None
    coord_space: Optional[str] = None
    mapped_mask: Optional[np.ndarray] = None
    coord_provenance: Optional[Dict] = None

    if require_coords:
        coord_result = load_subject_coords(
            dataset=dataset,
            subject_id=subject_id,
            channel_names_requested=phase0a_channels,
            allow_voxel_fallback=allow_voxel_fallback,
        )
        # v1.0.5 contract: voxel rejected for main analysis
        assert_coord_result_is_mm_for_main_analysis(coord_result)
        coords = coord_result.coords_array_in_requested_order
        mapped_mask = coord_result.mapped_mask_in_requested_order
        coord_units = coord_result.coord_units
        coord_space = coord_result.coord_space
        coord_provenance = dict(coord_result.provenance)
        coord_provenance["n_mapped"] = int(mapped_mask.sum())
        coord_provenance["n_missing"] = int(n_ch - mapped_mask.sum())
        coord_provenance["normalization_certainty"] = coord_result.normalization_certainty

    # === Step 5: cluster endpoints + valid_indices from PR-6 ===
    name_to_idx = {nm: i for i, nm in enumerate(phase0a_channels)}
    cluster_endpoints: Dict[int, Dict[str, List[int]]] = {}
    valid_indices_per_cluster: Dict[int, List[int]] = {}
    n_dropped_no_coords: Dict[int, int] = {}

    for t in pr6["per_template"]:
        cid = int(t["cluster_id"])
        s_names: List[str] = list(t["source"])
        k_names: List[str] = list(t["sink"])
        valid_mask_t = np.asarray(t["valid_mask"], dtype=bool)
        if valid_mask_t.shape != (n_ch,):
            raise ValueError(
                f"PR-6 cluster {cid} valid_mask shape {valid_mask_t.shape} != ({n_ch},)"
            )

        missing_names = [nm for nm in s_names + k_names if nm not in name_to_idx]
        if missing_names:
            raise ValueError(
                f"PR-6 cluster {cid} endpoint name(s) {missing_names} "
                f"not in Phase 0a channel_names. Index map keys: {list(name_to_idx)[:5]}..."
            )
        s_idx_raw = [name_to_idx[nm] for nm in s_names]
        k_idx_raw = [name_to_idx[nm] for nm in k_names]

        valid_set = set(np.where(valid_mask_t)[0].tolist())
        if require_coords:
            mapped_set = set(np.where(mapped_mask)[0].tolist())
            valid_set &= mapped_set
            s_idx = [i for i in s_idx_raw if mapped_mask[i]]
            k_idx = [i for i in k_idx_raw if mapped_mask[i]]
            n_dropped_no_coords[cid] = (
                (len(s_idx_raw) - len(s_idx)) + (len(k_idx_raw) - len(k_idx))
            )
        else:
            s_idx = s_idx_raw
            k_idx = k_idx_raw
            n_dropped_no_coords[cid] = 0

        cluster_endpoints[cid] = {"S": s_idx, "K": k_idx}
        valid_indices_per_cluster[cid] = sorted(valid_set)

    # === Step 6: forward/reverse pairs translation ===
    pairs_raw = phase0a.get("adaptive_cluster", {}).get(
        "candidate_forward_reverse_pairs", []
    )
    forward_reverse_pairs: List[Dict] = []
    for p in pairs_raw:
        a_id = int(p["cluster_a"])
        b_id = int(p["cluster_b"])
        if a_id not in cluster_endpoints or b_id not in cluster_endpoints:
            # PR-6 didn't produce endpoints for this cluster — skip pair.
            continue
        forward_reverse_pairs.append(
            {
                "cluster_A_id": a_id,
                "cluster_B_id": b_id,
                "reproducibility_source": p.get("label", "candidate_forward_reverse"),
                "spearman_r": float(p.get("spearman_r", float("nan"))),
            }
        )

    return SubjectPhase1Data(
        subject_id=subject_id,
        dataset=dataset,
        channel_names=phase0a_channels,
        events_bool=events_bool,
        coords=coords,
        coord_units=coord_units,
        coord_space=coord_space,
        mapped_mask=mapped_mask,
        hfo_rate=None,  # Future: wire from results/spatial_modulation/per_channel_metrics
        cluster_endpoints=cluster_endpoints,
        valid_indices_per_cluster=valid_indices_per_cluster,
        forward_reverse_pairs=forward_reverse_pairs,
        coord_provenance=coord_provenance,
        n_dropped_endpoints_no_coords_per_cluster=n_dropped_no_coords,
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
        dataset="synthetic",
        channel_names=names,
        events_bool=events_bool,
        coords=coords,
        coord_units="mm",
        coord_space="synthetic_mm",
        mapped_mask=np.ones(n_ch, dtype=bool),
        hfo_rate=rng.uniform(1, 10, size=n_ch),
        cluster_endpoints=cluster_endpoints,
        valid_indices_per_cluster=valid_indices_per_cluster,
        forward_reverse_pairs=fwd_rev_pairs,
        coord_provenance={"source_path": "<synthetic>", "loader_version": "make_synthetic_subject"},
        n_dropped_endpoints_no_coords_per_cluster={0: 0, 1: 0},
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
            coord_units=subject.coord_units,
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
                S_A, K_A, S_B, K_B, subject.coords, n_null=n_null, rng=rng,
                coord_units=subject.coord_units,
            )
        else:
            h2_spatial = {
                "verdict": "GATED_NO_COORDS",
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
        "dataset": subject.dataset,
        "n_channels": len(subject.channel_names),
        "n_events": int(subject.events_bool.shape[1]),
        "distance_metric": distance_metric,
        "coord_units": subject.coord_units,
        "coord_space": subject.coord_space,
        "coord_provenance": subject.coord_provenance,
        "n_coord_mapped": int(subject.mapped_mask.sum()) if subject.mapped_mask is not None else None,
        "n_dropped_endpoints_no_coords_per_cluster": subject.n_dropped_endpoints_no_coords_per_cluster,
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
                        default="euclidean",
                        help="euclidean (default, main analysis) requires 3D mm coords; "
                             "shaft_ordinal is the no-coord sensitivity fallback. "
                             "Per framework §6.2, H1 main analysis demands euclidean.")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: _dry_run_artifacts/ for --dry-run, "
                             "results/topic4_sef_itp/per_subject/ for real data")
    parser.add_argument("--phase0a-root", type=Path, default=DEFAULT_PHASE0A_ROOT,
                        help="masked Phase 0a per-subject JSON root")
    parser.add_argument("--pr6-root", type=Path, default=DEFAULT_PR6_ROOT,
                        help="masked PR-6 template_anchoring per-subject JSON root")
    parser.add_argument("--allow-voxel-fallback", action="store_true",
                        help="Epilepsiae sensitivity mode — voxel coords (REJECTED by mm assertion; "
                             "use only with --distance-metric=shaft_ordinal)")
    args = parser.parse_args(argv)

    if args.dry_run:
        return _run_dry_run(args)
    return _run_real_data(args)


def _print_verdict_summary(result: Dict) -> None:
    print("verdict summary:", file=sys.stderr)
    for h in ("h6", "h1", "h2"):
        block = result.get(h, {})
        if h == "h1":
            for cid, body in block.get("per_cluster", {}).items():
                print(f"  H1 cluster {cid}: {body.get('h1_overall_verdict')}", file=sys.stderr)
        elif h == "h2":
            for entry in block.get("per_pair", []):
                print(
                    f"  H2 pair {entry['cluster_A_id']}<->{entry['cluster_B_id']}: "
                    f"{entry.get('h2_integrated_verdict')}",
                    file=sys.stderr,
                )
        elif h == "h6":
            print(f"  H6: {block.get('verdict')}", file=sys.stderr)


def _run_dry_run(args) -> int:
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

    out_dir = args.output_dir or Path("results/topic4_sef_itp/_dry_run_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dry_run_{subject.subject_id}.json"
    with open(out_path, "w") as f:
        json.dump(_to_jsonable(result), f, indent=2)
    print(f"[dry-run] wrote {out_path}", file=sys.stderr)
    _print_verdict_summary(result)
    return 0


def _run_real_data(args) -> int:
    if args.subject is None and not args.all:
        print(
            "ERROR: real-data mode requires --subject <sid> or --all", file=sys.stderr
        )
        return 2
    if args.dataset == "both" and args.subject is not None:
        print(
            "ERROR: --subject requires explicit --dataset (epilepsiae or yuquan)",
            file=sys.stderr,
        )
        return 2

    # Enumerate targets
    targets: List[Path] = []
    if args.subject is not None:
        targets.append(args.phase0a_root / f"{args.dataset}_{args.subject}.json")
    else:
        datasets = ["epilepsiae", "yuquan"] if args.dataset == "both" else [args.dataset]
        for ds in datasets:
            targets.extend(sorted(args.phase0a_root.glob(f"{ds}_*.json")))

    if not targets:
        print(f"ERROR: no Phase 0a JSONs found under {args.phase0a_root}", file=sys.stderr)
        return 2

    out_dir = args.output_dir or Path("results/topic4_sef_itp/per_subject")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    for target in targets:
        print(f"[phase1] {target.name} …", file=sys.stderr)
        try:
            subject = load_subject_for_phase1(
                target,
                pr6_anchoring_root=args.pr6_root,
                require_coords=(args.distance_metric == "euclidean"),
                allow_voxel_fallback=args.allow_voxel_fallback,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {target.stem}: {e}", file=sys.stderr)
            n_fail += 1
            continue

        result = run_phase1_subject(
            subject=subject,
            hypothesis=args.hypothesis,
            distance_metric=args.distance_metric,
            n_permutations=args.n_permutations,
            n_null=args.n_null,
            rng_seed=args.rng_seed,
        )
        out_path = out_dir / f"{subject.dataset}_{subject.subject_id}.json"
        with open(out_path, "w") as f:
            json.dump(_to_jsonable(result), f, indent=2)
        print(f"  wrote {out_path}", file=sys.stderr)
        _print_verdict_summary(result)
        n_ok += 1

    print(f"[phase1] done: {n_ok} ok, {n_fail} skip", file=sys.stderr)
    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    sys.exit(main())

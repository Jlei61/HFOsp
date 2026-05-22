"""SEF-ITP Phase 1 CLI runner — H6 / H1 / H2 spatial geometry hypotheses.

Plan: docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md
Framework: docs/topic4_sef_itp_framework.md v1.0.4
Module: src/sef_itp_phase1.py

CURRENT STATUS (2026-05-21):
  - --dry-run mode: WORKING (synthetic input, exercise the pipeline end-to-end)
  - real-data mode: WIRED via load_subject_for_phase1() — consumes
      (a) masked Phase 0a per-subject JSON
          (results/interictal_propagation_masked/per_subject/<dataset>_<sid>.json)
      (b) masked PR-6 template_anchoring JSON
          (results/interictal_propagation_masked/template_anchoring/per_subject/<dataset>_<sid>.json)
      (c) masked rank_displacement JSON for H2 swap-k labels
          (results/interictal_propagation_masked/rank_displacement/per_subject/<dataset>_<sid>.json)
      (d) lagPat NPZ — events_bool aligned to Phase 0a channel_names
          via src.interictal_propagation.load_subject_propagation_events
      (e) SEEG coords via src.seeg_coord_loader.load_subject_coords —
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow direct invocation: `python scripts/run_sef_itp_phase1.py ...`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sef_itp_phase1 import (
    compute_h1_full,
    compute_h6_segregation,
    compute_participation_rate,
    pairwise_3d_euclidean,
    pairwise_shaft_ordinal,
)
from src.seeg_coord_loader import (
    assert_coord_result_is_mm_for_main_analysis,
    enumerate_subject_all_channels,
    load_subject_coords,
)
from src.interictal_propagation import load_subject_propagation_events
from src.rank_displacement import derive_swap_endpoint


# Path conventions (mirror scripts/run_interictal_propagation.py).
DEFAULT_YUQUAN_LAGPAT_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
DEFAULT_EPILEPSIAE_LAGPAT_ROOT = Path(
    "/mnt/epilepsia_data/interilca_inter_results/all_data_lns"
)
DEFAULT_PHASE0A_ROOT = Path("results/interictal_propagation_masked/per_subject")
DEFAULT_PR6_ROOT = Path(
    "results/interictal_propagation_masked/template_anchoring/per_subject"
)
DEFAULT_RANK_DISPLACEMENT_ROOT = Path(
    "results/interictal_propagation_masked/rank_displacement/per_subject"
)


SCHEMA_VERSION = "sef_itp_phase1_v1_2026_05_21"


@dataclass
class SubjectPhase1Data:
    """Per-subject input for Phase 1 hypotheses.

    v1.0.7 (2026-05-22): channel_names is UNIFIED namespace — first
    `n_lagpat_channels` entries are lagPat-selected channels (where
    events_bool has real data), remainder are all-SEEG channels added
    for H1/H2 null-pool expansion. H6 must slice events_bool / coords /
    channel_names to the first n_lagpat_channels rows (H6 stays on the
    lagPat namespace; H1/H2 use the unified pool).
    """

    subject_id: str
    dataset: str  # "yuquan" | "epilepsiae"  — needed for downstream provenance
    channel_names: List[str]
    n_lagpat_channels: int  # first n rows of events_bool are lagPat-selected
    events_bool: np.ndarray  # (n_unified, n_events) — non-lagPat rows all False
    coords: Optional[np.ndarray]  # (n_unified, 3) — NaN row for unmapped channels
    coord_units: Optional[str]  # "mm" | "voxel" | None — gates euclidean compute
    coord_space: Optional[str]  # provenance for output JSON
    mapped_mask: Optional[np.ndarray]  # (n_unified,) bool — True where coord is mapped
    hfo_rate: Optional[np.ndarray]  # (n_unified,) optional — per-channel HFO rate
    # PR-6 endpoint output per cluster (indices in unified namespace):
    cluster_endpoints: Dict[int, Dict[str, List[int]]]  # cluster_id → {"S": [...], "K": [...]}
    valid_indices_per_cluster: Dict[int, List[int]]  # cluster_id → all mapped SEEG − endpoint
    # H2 input: rank-displacement swap_sweep-derived channel labels.
    h2_swap_check: Optional[Dict] = None
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


def _load_h2_from_rank_displacement(
    rank_displacement_path: Path,
    *,
    dataset: str,
    subject_id: str,
    phase0a_channels: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Load Topic4 H2 channel labels from rank-displacement swap_sweep.

    Contract v1.0.4: PR-2 is provenance, PR-6 fixed top-3 endpoint is an
    audit summary; Topic4 H2 channel labels come from the variable-k
    rank-displacement swap_sweep.
    """
    if not rank_displacement_path.exists():
        return None

    with open(rank_displacement_path) as f:
        rd = json.load(f)

    if rd.get("dataset") != dataset or str(rd.get("subject")) != subject_id:
        raise ValueError(
            f"rank_displacement JSON identity ({rd.get('dataset')!r}, {rd.get('subject')!r}) "
            f"!= ({dataset!r}, {subject_id!r})"
        )
    if rd.get("stable_k") != 2:
        return None

    channel_names = list(rd.get("channel_names") or [])
    if channel_names != list(phase0a_channels):
        raise ValueError(
            f"rank_displacement channel_names mismatch for {dataset}/{subject_id}.\n"
            f"  Phase 0a:          {list(phase0a_channels)}\n"
            f"  rank_displacement: {channel_names}"
        )

    ok_pairs = [p for p in rd.get("pairs", []) if p.get("exit_reason") == "ok"]
    if len(ok_pairs) != 1:
        return None
    pair = ok_pairs[0]
    sw = pair.get("swap_sweep") or {}
    if sw.get("exit_reason") != "ok" or sw.get("decision_k") is None:
        return None

    pair_channels = list(pair.get("channel_names") or channel_names)
    if pair_channels != channel_names:
        raise ValueError(
            f"rank_displacement pair channel_names mismatch for {dataset}/{subject_id}"
        )

    joint_valid = np.asarray(pair.get("joint_valid"), dtype=bool)
    rank_a_dense_full = np.asarray(pair.get("rank_a_dense_full"), dtype=float)
    if joint_valid.shape != rank_a_dense_full.shape or joint_valid.shape != (len(channel_names),):
        raise ValueError(
            f"rank_displacement pair shape mismatch for {dataset}/{subject_id}: "
            f"joint_valid={joint_valid.shape}, rank_a_dense_full={rank_a_dense_full.shape}, "
            f"n_channels={len(channel_names)}"
        )

    decision_k = int(sw["decision_k"])
    joint_channels = [ch for ch, keep in zip(channel_names, joint_valid) if keep]
    joint_rank_a = rank_a_dense_full[joint_valid]
    order = np.argsort(joint_rank_a, kind="stable")
    swap_source_channels = [joint_channels[i] for i in order[:decision_k]]
    swap_sink_channels = [joint_channels[i] for i in order[-decision_k:]]
    swap_endpoint_channels = derive_swap_endpoint(
        channel_names=joint_channels,
        rank_a_dense=joint_rank_a,
        decision_k=decision_k,
    )
    swap_score = sw.get("decision_swap_score", sw.get("T_obs"))

    return {
        "source_contract": "rank_displacement_swap_sweep_v1",
        "source_path": str(rank_displacement_path),
        "cluster_id_a": pair.get("cluster_id_a"),
        "cluster_id_b": pair.get("cluster_id_b"),
        "swap_class": sw.get("swap_class", "none"),
        "decision_k": decision_k,
        "swap_source_channels": swap_source_channels,
        "swap_sink_channels": swap_sink_channels,
        "swap_endpoint_channels": swap_endpoint_channels,
        "swap_score": swap_score,
        "decision_swap_score": swap_score,
        "T_obs": sw.get("T_obs", swap_score),
        "p_fw": sw.get("p_fw"),
        "null_p": sw.get("p_fw"),
        "n_valid": sw.get("n_valid", int(joint_valid.sum())),
        "n_perm": sw.get("n_perm"),
    }


def load_subject_for_phase1(
    subject_json_path: Path,
    *,
    pr6_anchoring_root: Path = DEFAULT_PR6_ROOT,
    rank_displacement_root: Path = DEFAULT_RANK_DISPLACEMENT_ROOT,
    yuquan_lagpat_root: Path = DEFAULT_YUQUAN_LAGPAT_ROOT,
    epilepsiae_lagpat_root: Path = DEFAULT_EPILEPSIAE_LAGPAT_ROOT,
    require_coords: bool = True,
    allow_voxel_fallback: bool = False,
) -> SubjectPhase1Data:
    """Load per-subject Phase 1 input from 5 aligned sources.

    Inputs (all derived from `subject_json_path` + path roots):
      1. Masked Phase 0a per-subject JSON (channel_names, adaptive_cluster).
      2. Masked PR-6 template_anchoring JSON (per_template[] with
         source/sink/valid_mask).
      3. Masked rank_displacement JSON (H2 swap_sweep decision_k labels).
      4. lagPat NPZ → events_bool (via load_subject_propagation_events).
      5. SEEG coords via load_subject_coords; assert mm units before return.

    Contract clauses (CLAUDE.md §6):
      - **Channel alignment**: Phase 0a channel_names MUST equal lagPat-loader
        channel_names; raise on mismatch.
      - **Endpoint name→index**: PR-6 source/sink names converted via Phase 0a
        ordering; raise on any name not found.
      - **coord_units mm**: assert_coord_result_is_mm_for_main_analysis called
        when require_coords=True (forbids silent voxel→euclidean).
      - **Valid pool**: H1 valid_indices = all mapped SEEG channels minus
        endpoints; PR-6 valid_mask is not used to constrain the null pool.
      - **H2 channel labels**: rank_displacement swap_sweep decision_k defines
        swap_endpoint_channels; PR-2 candidate and PR-6 fixed top-3 are not
        H2 label sources.
      - **Subject ID canonicalization**: handled inside load_subject_coords
        for Epilepsiae (e.g., '1073' → '107302').

    Returns:
        SubjectPhase1Data populated end-to-end.

    Raises:
        FileNotFoundError: required Phase 0a / PR-6 / lagPat sources missing.
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
    n_lagpat = len(phase0a_channels)
    if n_lagpat != phase0a["n_channels"]:
        raise ValueError(
            f"Phase 0a channel_names length {n_lagpat} != n_channels {phase0a['n_channels']}"
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
    lagpat_events_bool = np.asarray(loaded["bools"], dtype=bool)
    lagpat_channels = list(loaded["channel_names"])

    if lagpat_channels != phase0a_channels:
        raise ValueError(
            f"Channel-name mismatch between Phase 0a and lagPat loader for "
            f"{dataset}/{subject_id}.\n"
            f"  Phase 0a: {phase0a_channels}\n"
            f"  lagPat:   {lagpat_channels}\n"
            f"Re-run Phase 0a §5a if lagPat NPZ was regenerated."
        )
    if lagpat_events_bool.shape[0] != n_lagpat:
        raise ValueError(
            f"events_bool n_ch {lagpat_events_bool.shape[0]} != Phase 0a n_channels {n_lagpat}"
        )

    # === Step 4: full SEEG enumeration (v1.0.7 — replaces lagPat-restricted pool) ===
    all_seeg_names = enumerate_subject_all_channels(dataset, subject_id)
    # Unified namespace: lagPat channels first (preserves their position),
    # then any non-lagPat SEEG channels not yet seen.
    unified_names: List[str] = list(phase0a_channels)
    seen = set(phase0a_channels)
    for nm in all_seeg_names:
        if nm not in seen:
            unified_names.append(nm)
            seen.add(nm)
    n_unified = len(unified_names)
    n_nonlagpat = n_unified - n_lagpat

    # Expand events_bool to (n_unified, n_events): non-lagPat rows all False
    # (they never participated in any lagPat group event by definition).
    n_events = lagpat_events_bool.shape[1]
    events_bool = np.zeros((n_unified, n_events), dtype=bool)
    events_bool[:n_lagpat, :] = lagpat_events_bool

    # === Step 5: SEEG coords on unified namespace ===
    coords: Optional[np.ndarray] = None
    coord_units: Optional[str] = None
    coord_space: Optional[str] = None
    mapped_mask: Optional[np.ndarray] = None
    coord_provenance: Optional[Dict] = None

    if require_coords:
        coord_result = load_subject_coords(
            dataset=dataset,
            subject_id=subject_id,
            channel_names_requested=unified_names,
            allow_voxel_fallback=allow_voxel_fallback,
        )
        assert_coord_result_is_mm_for_main_analysis(coord_result)
        coords = coord_result.coords_array_in_requested_order
        mapped_mask = coord_result.mapped_mask_in_requested_order
        coord_units = coord_result.coord_units
        coord_space = coord_result.coord_space
        coord_provenance = dict(coord_result.provenance)
        coord_provenance["n_mapped"] = int(mapped_mask.sum())
        coord_provenance["n_missing"] = int(n_unified - mapped_mask.sum())
        coord_provenance["n_lagpat_channels"] = n_lagpat
        coord_provenance["n_unified_channels"] = n_unified
        coord_provenance["n_nonlagpat_seeg_added"] = n_nonlagpat
        coord_provenance["normalization_certainty"] = coord_result.normalization_certainty

    # === Step 6: cluster endpoints in unified namespace ===
    name_to_idx = {nm: i for i, nm in enumerate(unified_names)}
    cluster_endpoints: Dict[int, Dict[str, List[int]]] = {}
    valid_indices_per_cluster: Dict[int, List[int]] = {}
    n_dropped_no_coords: Dict[int, int] = {}

    for t in pr6["per_template"]:
        cid = int(t["cluster_id"])
        s_names: List[str] = list(t["source"])
        k_names: List[str] = list(t["sink"])

        missing_names = [nm for nm in s_names + k_names if nm not in name_to_idx]
        if missing_names:
            raise ValueError(
                f"PR-6 cluster {cid} endpoint name(s) {missing_names} "
                f"not in unified channel_names. Sample keys: {list(name_to_idx)[:5]}..."
            )
        s_idx_raw = [name_to_idx[nm] for nm in s_names]
        k_idx_raw = [name_to_idx[nm] for nm in k_names]

        if require_coords:
            s_idx = [i for i in s_idx_raw if mapped_mask[i]]
            k_idx = [i for i in k_idx_raw if mapped_mask[i]]
            n_dropped_no_coords[cid] = (
                (len(s_idx_raw) - len(s_idx)) + (len(k_idx_raw) - len(k_idx))
            )
            # v1.0.7: valid_indices = ALL mapped SEEG channels minus endpoints.
            # Earlier v1.0.6 used PR-6 valid_mask (lagPat-participating only),
            # which created a circular null where endpoints and null samples
            # came from the same pre-filtered high-HFO subset.
            endpoint_set = set(s_idx + k_idx)
            valid_idx = [
                i for i in range(n_unified)
                if mapped_mask[i] and i not in endpoint_set
            ]
        else:
            s_idx = s_idx_raw
            k_idx = k_idx_raw
            n_dropped_no_coords[cid] = 0
            endpoint_set = set(s_idx + k_idx)
            valid_idx = [i for i in range(n_unified) if i not in endpoint_set]

        cluster_endpoints[cid] = {"S": s_idx, "K": k_idx}
        valid_indices_per_cluster[cid] = valid_idx

    # === Step 7: H2 — rank-displacement swap-k channel labels (v1.0.10) ===
    #
    # Topic4 framework v1.0.4 separates three layers:
    #   PR-2/2.5       = template discovery / provenance
    #   PR-6 endpoint  = fixed top-3 source/sink summary
    #   rank-disp k    = per-channel, variable-k H2 label source
    #
    # Therefore H2 channel labels are loaded from masked rank_displacement
    # swap_sweep. PR-6 h2_swap_check remains audit trail only.
    rd_path = rank_displacement_root / subject_json_path.name
    h2_swap_check = _load_h2_from_rank_displacement(
        rd_path,
        dataset=dataset,
        subject_id=subject_id,
        phase0a_channels=phase0a_channels,
    )

    return SubjectPhase1Data(
        subject_id=subject_id,
        dataset=dataset,
        channel_names=unified_names,
        n_lagpat_channels=n_lagpat,
        events_bool=events_bool,
        coords=coords,
        coord_units=coord_units,
        coord_space=coord_space,
        mapped_mask=mapped_mask,
        hfo_rate=None,
        cluster_endpoints=cluster_endpoints,
        valid_indices_per_cluster=valid_indices_per_cluster,
        h2_swap_check=h2_swap_check,
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
    # Synthetic rank-displacement swap_sweep-derived H2 payload.
    synthetic_h2 = {
        "swap_score": 1.0,
        "decision_swap_score": 1.0,
        "T_obs": 1.0,
        "p_fw": 0.001,
        "null_p": 0.001,
        "n_perm": 1000,
        "swap_class": "strict",
        "decision_k": 3,
        "n_valid": n_ch,
        "swap_source_channels": names[:3],
        "swap_sink_channels": names[-3:],
        "swap_endpoint_channels": names[:3] + names[-3:],
        "source_contract": "rank_displacement_swap_sweep_v1",
        "source_path": "<synthetic>",
    }

    return SubjectPhase1Data(
        subject_id="synthetic_subject_001",
        dataset="synthetic",
        channel_names=names,
        n_lagpat_channels=n_ch,
        events_bool=events_bool,
        coords=coords,
        coord_units="mm",
        coord_space="synthetic_mm",
        mapped_mask=np.ones(n_ch, dtype=bool),
        hfo_rate=rng.uniform(1, 10, size=n_ch),
        cluster_endpoints=cluster_endpoints,
        valid_indices_per_cluster=valid_indices_per_cluster,
        h2_swap_check=synthetic_h2,
        coord_provenance={"source_path": "<synthetic>", "loader_version": "make_synthetic_subject"},
        n_dropped_endpoints_no_coords_per_cluster={0: 0, 1: 0},
    )


def run_h6(
    subject: SubjectPhase1Data,
    distance_metric: str = "shaft_ordinal",
    n_permutations: int = 1000,
    rng_seed: int = 0,
) -> Dict:
    """Run H6 — participation field spatial segregation.

    v1.0.7: H6 stays on lagPat namespace (sliced from unified channel_names).
    The non-lagPat SEEG channels added for H1/H2 null-pool expansion are
    by-construction zero participation; including them would dilute the
    high/low split and change the H6 scientific question. Keep H6 as
    "within the channels that participated in group events, is the
    participation field spatially structured?"
    """
    nl = subject.n_lagpat_channels
    participation = compute_participation_rate(subject.events_bool[:nl, :])
    lagpat_names = subject.channel_names[:nl]
    if distance_metric == "euclidean":
        if subject.coords is None:
            return {
                "verdict": "GATED_NO_COORDS",
                "reason": "euclidean distance requires 3D coords",
            }
        coords = subject.coords[:nl]
    else:
        coords = None
    out = compute_h6_segregation(
        participation=participation,
        channel_names=lagpat_names,
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


def _run_h2_spatial_compactness(
    *,
    subject: SubjectPhase1Data,
    source_idx: Sequence[int],
    sink_idx: Sequence[int],
    endpoint_idx: Sequence[int],
    base_pool: Sequence[int],
    distance_metric: str,
    n_null: int,
    rng_seed: int,
) -> Dict:
    """Test swap-k endpoint spatiality against other valid SEEG nodes.

    Rank-displacement defines the labels; this helper asks the spatial
    question. Source-side and sink-side are primary because the two sides of a
    source/sink swap can live at opposite ends of the trajectory.
    """
    endpoint_set = set(endpoint_idx)
    other_pool = [i for i in base_pool if i not in endpoint_set]
    rng = np.random.default_rng(rng_seed)
    if distance_metric == "euclidean":
        D = pairwise_3d_euclidean(subject.coords)
    elif distance_metric == "shaft_ordinal":
        D = pairwise_shaft_ordinal(subject.channel_names)
    else:
        raise ValueError(f"unknown distance_metric: {distance_metric}")

    def _compact(members: Sequence[int]) -> Dict:
        return _compute_h2_unmatched_compactness(
            members=members,
            candidate_pool=other_pool,
            D=D,
            n_null=n_null,
            rng=rng,
        )

    return {
        "available": True,
        "source_helper": "h2_unmatched_compactness_null",
        "candidate_pool_contract": (
            "all_mapped_seeg_minus_swap_endpoint"
            if distance_metric == "euclidean"
            else "all_seeg_minus_swap_endpoint"
        ),
        "null_contract": "unmatched_random_other_valid_nodes",
        "distance_metric": distance_metric,
        "n_swap_endpoint": len(endpoint_set),
        "n_other_valid": len(other_pool),
        "source_side": _compact(source_idx),
        "sink_side": _compact(sink_idx),
        "combined_endpoint": _compact(endpoint_idx),
    }


def _mean_pairwise_from_distance(D: np.ndarray, indices: Sequence[int]) -> float:
    idx = list(indices)
    if len(idx) < 2:
        return 0.0
    sub = D[np.ix_(idx, idx)]
    vals = sub[np.triu_indices_from(sub, k=1)]
    finite = vals[np.isfinite(vals)]
    return float(finite.mean()) if len(finite) else float("inf")


def _compute_h2_unmatched_compactness(
    *,
    members: Sequence[int],
    candidate_pool: Sequence[int],
    D: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
) -> Dict:
    members = list(members)
    pool = [i for i in candidate_pool if i not in set(members)]
    n_members = len(members)
    if n_members < 2:
        return {"verdict": "INSUFFICIENT_CHANNELS", "n_members": n_members}
    if len(pool) < n_members:
        return {
            "verdict": "INSUFFICIENT_POOL",
            "n_members": n_members,
            "n_candidate_pool": len(pool),
        }

    C_actual = _mean_pairwise_from_distance(D, members)
    pool_arr = np.asarray(pool)
    null_dists = [
        _mean_pairwise_from_distance(
            D, rng.choice(pool_arr, size=n_members, replace=False).tolist()
        )
        for _ in range(n_null)
    ]
    null_arr = np.asarray(null_dists, dtype=float)
    null_lower_p = float((np.sum(null_arr <= C_actual) + 1) / (len(null_arr) + 1))
    null_upper_p = float((np.sum(null_arr >= C_actual) + 1) / (len(null_arr) + 1))

    if null_lower_p < 0.05:
        verdict = "PASS"
    elif null_upper_p < 0.05:
        verdict = "FAIL_DIFFUSE"
    else:
        verdict = "NULL"

    return {
        "verdict": verdict,
        "C_actual": float(C_actual),
        "C_null_median": float(np.median(null_arr)),
        "null_lower_p": null_lower_p,
        "null_upper_p": null_upper_p,
        "n_members": n_members,
        "n_candidate_pool": len(pool),
        "n_null_samples": len(null_arr),
        "relaxation_tier": "unmatched",
    }


def run_h2(
    subject: SubjectPhase1Data,
    distance_metric: str = "euclidean",
    n_null: int = 1000,
    rng_seed: int = 0,
) -> Dict:
    """Run H2 — ingest rank-displacement swap_sweep channel labels."""
    swap = subject.h2_swap_check
    if swap is None:
        return {
            "available": False,
            "reason": (
                "rank-displacement swap_sweep unavailable for this subject "
                "(requires stable_k=2 with exactly one ok pair)"
            ),
        }
    name_to_idx = {name: i for i, name in enumerate(subject.channel_names)}
    source_channels = list(swap.get("swap_source_channels") or [])
    sink_channels = list(swap.get("swap_sink_channels") or [])
    endpoint_channels = list(swap.get("swap_endpoint_channels") or [])
    if not source_channels or not sink_channels:
        source_channels = endpoint_channels[: int(swap.get("decision_k") or 0)]
        sink_channels = endpoint_channels[int(swap.get("decision_k") or 0):]
    missing = [
        ch for ch in source_channels + sink_channels + endpoint_channels
        if ch not in name_to_idx
    ]
    if missing:
        raise ValueError(f"H2 swap endpoint channel(s) not in subject namespace: {missing}")

    source_idx = [name_to_idx[ch] for ch in source_channels]
    sink_idx = [name_to_idx[ch] for ch in sink_channels]
    endpoint_idx = sorted({name_to_idx[ch] for ch in endpoint_channels})

    if distance_metric == "euclidean":
        if subject.coords is None or subject.mapped_mask is None:
            spatial_compactness = {
                "available": False,
                "reason": "euclidean H2 spatiality requires mapped 3D coords",
            }
        else:
            base_pool = [
                i for i in range(len(subject.channel_names))
                if bool(subject.mapped_mask[i])
            ]
            spatial_compactness = _run_h2_spatial_compactness(
                subject=subject,
                source_idx=source_idx,
                sink_idx=sink_idx,
                endpoint_idx=endpoint_idx,
                base_pool=base_pool,
                distance_metric=distance_metric,
                n_null=n_null,
                rng_seed=rng_seed,
            )
    elif distance_metric == "shaft_ordinal":
        base_pool = list(range(len(subject.channel_names)))
        spatial_compactness = _run_h2_spatial_compactness(
            subject=subject,
            source_idx=source_idx,
            sink_idx=sink_idx,
            endpoint_idx=endpoint_idx,
            base_pool=base_pool,
            distance_metric=distance_metric,
            n_null=n_null,
            rng_seed=rng_seed,
        )
    else:
        raise ValueError(f"unknown distance_metric: {distance_metric}")

    return {
        "available": True,
        "tier_lock": "directional_mechanism_sanity_not_cohort_claim",
        "source_contract": swap.get("source_contract"),
        "source_path": swap.get("source_path"),
        "swap_class": swap.get("swap_class"),
        "decision_k": swap.get("decision_k"),
        "swap_source_channels": source_channels,
        "swap_sink_channels": sink_channels,
        "swap_endpoint_channels": swap.get("swap_endpoint_channels"),
        "decision_swap_score": swap.get("decision_swap_score"),
        "T_obs": swap.get("T_obs"),
        "p_fw": swap.get("p_fw"),
        "n_valid": swap.get("n_valid"),
        "swap_score": swap.get("swap_score"),
        "null_p": swap.get("null_p"),
        "n_perm": swap.get("n_perm"),
        "spatial_compactness": spatial_compactness,
        "note_no_verdict": (
            "Topic4 H2 is mechanism sanity only, no PASS/NULL/FAIL verdict. "
            "Channel labels come from rank-displacement swap_sweep decision_k; "
            "spatiality tests swap-k source/sink sides against other valid nodes."
        ),
    }


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
        "framework_ref": "topic4_sef_itp_framework.md v1.0.4",
        "plan_ref": "docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md",
    }
    if hypothesis in ("h6", "all"):
        out["h6"] = run_h6(subject, distance_metric, n_permutations, rng_seed)
    if hypothesis in ("h1", "all"):
        out["h1"] = run_h1(subject, distance_metric, n_null, rng_seed)
    if hypothesis in ("h2", "all"):
        out["h2"] = run_h2(subject, distance_metric, n_null, rng_seed)
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
    parser.add_argument("--rank-displacement-root", type=Path,
                        default=DEFAULT_RANK_DISPLACEMENT_ROOT,
                        help="masked rank_displacement per-subject JSON root for H2 swap-k labels")
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
            if block.get("available"):
                print(
                    f"  H2 (rank-displacement swap-k, mechanism sanity): "
                    f"swap_class={block.get('swap_class')}, "
                    f"decision_k={block.get('decision_k')}, "
                    f"T_obs={block.get('T_obs')}, "
                    f"p_fw={block.get('p_fw')}",
                    file=sys.stderr,
                )
            else:
                print(f"  H2: unavailable ({block.get('reason')})", file=sys.stderr)
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
                rank_displacement_root=args.rank_displacement_root,
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

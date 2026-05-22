"""SEF-ITP framework Phase 2 — H3 mark-independence + H4 normalized rate/geometry instability.

Phase 2 sits on top of Phase 1's n=23 cohort (Phase 1 = H1 sanity / H2 primary cohort claim /
H6 secondary). Phase 2 adds:

- **H3** = mark-independent template sampling at multiple time scales + endpoint geometric
  stability. Mostly ingest of PR-7 burst diagnostic (lag1_same + run_length_lift) + PR-7 pairing
  (window excess at {10, 30, 60, 1800}s) + PR-6 anchoring (split_half_robustness endpoint
  Jaccard). Cohort verdict via bootstrap TOST equivalence vs ±δ_excess=0.05 band.
- **H4** = normalized rate vs geometry instability. New pipeline: slice cohort 24h / multi-day
  data into 2h epochs (preserving time order), compute per-epoch rate(t) and endpoint(t),
  compare normalized instability `I_rate` vs `I_geom` via cohort Wilcoxon + Cohen's d.

**Verdict naming locked at framework time:**

- H3: SUPPORTED / NOT_SUPPORTED_MEMORY / NOT_SUPPORTED_GEOMETRY_UNSTABLE / NOT_SUPPORTED_BOTH /
       CONTRADICTED. NOT PASS/NULL/FAIL — guards against "PASS = proves independence".
- H4: PASS / NULL / FAIL / UNDERPOWERED (standard verdict family).

**Locked contracts:**

- δ_excess = 0.05 lock at framework time. Forbid post-hoc adjustment.
- H3 endpoint stability combinator = **OR** (project convention; mirror of
  AGENTS.md cross-PR `forward_reverse_reproduced` = split-half OR odd-even).
- H3 wording lock: "compatible with mark-independent sampling within tested precision."
- H4 Cohen's d ≥ 0.30 floor for PASS verdict.
- H4 I_rate matched null is a **Phase 2 v1.0.0 spec amendment proposal** (see
  `docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md`). Both the literal
  framework v1.0.5 §3.4 `epoch_order_shuffle` (degenerate by construction) and a non-degenerate
  `circular_shift_within_block` variant are implemented; user decides which enters
  framework on return.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

__version__ = "v1.0.0"


@dataclass
class SubjectPhase2Data:
    """One subject's Phase 2 inputs after ingest.

    H3 fields are scalar / dict, populated by `load_subject_for_phase2_h3` from PR-7 and PR-6
    per-subject JSONs (no recomputation here — pure ingest).

    H4 fields are raw arrays (events, labels, block ranges, template ranks, channel names)
    needed for per-epoch endpoint recomputation downstream.
    """

    dataset: str
    subject_id: str

    # H3 ingest (already-computed per-subject metrics from PR-7 / PR-6 JSONs)
    lag1_same_excess_n2: float
    window_excess_n2: Dict[float, float]  # {10.0, 30.0, 60.0, 1800.0} → excess
    run_length_lift_n2: float
    endpoint_jaccard_first_half: float
    endpoint_jaccard_odd_even: float

    # H4 raw (downstream computes per-epoch features from these)
    event_abs_times: np.ndarray
    cluster_labels: np.ndarray
    block_time_ranges: List[Tuple[float, float]]
    template_ranks: Dict[int, np.ndarray]  # {cluster_id: rank vector aligned to channel_names}
    channel_names: List[str]


# --------------------------------------------------------------------------- #
# H3 ingest extractors
#
# H3's three statistical layers all source from existing per-subject JSONs:
#   1. mark-transition lag1 + window excess: PR-7 burst + PR-7 pairing
#   2. burst run length: PR-7 burst
#   3. endpoint geometric stability: PR-6 anchoring (split_half_robustness)
#
# CLAUDE.md §6.1 re-use check (question-match):
#   - PR-7 burst's lag1_same_excess vs N2: same question as H3 layer 1 (lag-1 same-template
#     frequency vs marginal-preserving null). DIRECT INGEST.
#   - PR-7 pairing's window excess @ {10, 30, 60, 1800}s vs N2: same question as H3 multi-scale
#     mark-transition. DIRECT INGEST.
#   - PR-6's split_half_robustness.subject_mean_jaccard_endpoint: same question as H3
#     endpoint geometric stability (Jaccard recall of endpoint set across temporal split).
#     DIRECT INGEST.
# --------------------------------------------------------------------------- #


def extract_window_excess_from_pairing(
    pairing_json: dict,
    windows: Tuple[float, ...] = (10.0, 30.0, 60.0, 1800.0),
    null_key: str = "N2",
) -> Dict[float, float]:
    """Pull `excess` at the requested Δt windows from a PR-7 pairing per-subject JSON.

    PR-7 schema: `pairing_with_nulls.lift.<null>.<window_str>.excess`. Windows are stored as
    string keys (e.g. `"10.0"`).

    Raises KeyError if `pairing_with_nulls`, `lift`, `null_key`, or any requested window is
    missing. No silent default — H3 must not paper over a missing PR-7 window.
    """
    lift = pairing_json["pairing_with_nulls"]["lift"][null_key]
    return {w: float(lift[f"{w}"]["excess"]) for w in windows}


def extract_lag1_and_runlength_from_burst(
    burst_json: dict,
    null_key: str = "N2",
) -> Tuple[float, float]:
    """Pull (lag1_same_excess, run_length_lift) from a PR-7 burst per-subject JSON.

    PR-7 schema:
      `burst_diagnostic.lag1_same_excess.<null_key>` → float (target=0 in H3 TOST)
      `burst_diagnostic.lift.<null_key>.run_length_lift` → float (target=1 in H3 TOST)
    """
    bd = burst_json["burst_diagnostic"]
    lag1 = float(bd["lag1_same_excess"][null_key])
    run_length = float(bd["lift"][null_key]["run_length_lift"])
    return lag1, run_length


def extract_endpoint_jaccard_from_anchoring(
    anchoring_json: dict,
) -> Tuple[float, float]:
    """Pull (first_half_second_half, odd_even_block) endpoint Jaccard from a PR-6 anchoring
    per-subject JSON.

    PR-6 schema:
      `split_half_robustness.per_split.first_half_second_half.subject_mean_jaccard_endpoint`
      `split_half_robustness.per_split.odd_even_block.subject_mean_jaccard_endpoint`
    """
    per_split = anchoring_json["split_half_robustness"]["per_split"]
    fh = float(per_split["first_half_second_half"]["subject_mean_jaccard_endpoint"])
    oe = float(per_split["odd_even_block"]["subject_mean_jaccard_endpoint"])
    return fh, oe

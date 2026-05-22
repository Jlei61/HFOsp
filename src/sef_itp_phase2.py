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

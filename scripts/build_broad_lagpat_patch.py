#!/usr/bin/env python3
"""INTERIM broad-lagPat patch for single-shaft 4-channel Yuquan subjects.

Why this exists
---------------
huangwanling and zhaojinrui have, in the narrow masked lagPat, only 4 channels
that all sit on ONE electrode shaft (H2-H5 / F5-F8). A single shaft cannot span
a 2D propagation axis, so the geometry layer marks them degenerate_axis and the
A5 fingerprint coverage audit reports axis_available=false for both.

The broad-lagPat re-pack (top_n=20, already produced under results/lagpat_broad_dyn/)
brings in channels across multiple shafts (huangwanling -> E/H/B/F,
zhaojinrui -> F/E/C/D/H), all landing in documented clinical networks. This
driver re-runs the propagation -> rank-displacement -> geometry chain on that
broad lagPat for exactly these two subjects, writing to PARALLEL `_broad` dirs
(narrow tree untouched, AGENTS.md parallel-dir convention).

This is an INTERIM 2-subject patch. The full-cohort broad re-derivation
(per-subject narrow+15 dynamic pool for all Yuquan) is deferred. The same
machinery runs the full cohort by widening --subjects; nothing here is
2-subject-specific except the default list.

zhourongxuan is intentionally NOT included: its broad re-pack has only 28
events (broad-INELIGIBLE per results/lagpat_broad_dyn/COHORT_SUMMARY.md), too
few for a stable template.

Chain (each step's broad output feeds the next; integrate is skipped because the
geometry path_axis COMPONENT rec already carries every field the A5 audits read):
  1. run_interictal_propagation (base)  -> broad PR-2 adaptive_cluster JSON
  2. run_interictal_propagation (--pr25) -> merge time_split_reproducibility
  3. run_rank_displacement                -> broad rank_a/b_dense_full + swap_class
  4. run_propagation_skeleton_geometry    -> broad path_axis (axis_length_mm, ...)

All input/output routing is done by monkeypatching module-level path globals
in-process (same pattern as scripts/pilot_broad_lagpat_repack.py). No shared
config is edited.

Outputs (parallel `_broad` trees):
  results/interictal_propagation_masked_broad/per_subject/yuquan_<sub>.json
  results/interictal_propagation_masked_broad/rank_displacement/per_subject/yuquan_<sub>.json
  results/spatial_modulation/propagation_geometry_broad/components/path_axis/per_subject/yuquan_<sub>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BROAD_LAGPAT = REPO / "results" / "lagpat_broad_dyn"
PROP_OUT = REPO / "results" / "interictal_propagation_masked_broad"
GEOM_OUT = REPO / "results" / "spatial_modulation" / "propagation_geometry_broad"

DEFAULT_SUBJECTS = ["huangwanling", "zhaojinrui"]


def step1_2_propagation(subjects):
    """PR-2 base run then PR-2.5 merge, reading broad lagPat, writing PROP_OUT."""
    import scripts.run_interictal_propagation as rip

    broad_dirs = {BROAD_LAGPAT / s for s in subjects}

    orig_subdir = rip._subject_dir

    def subdir_broad(dataset, root, subject):
        if dataset == "yuquan" and subject in subjects:
            return BROAD_LAGPAT / subject
        return orig_subdir(dataset, root, subject)

    orig_has = rip._has_propagation_inputs

    def has_broad(dataset, subject_dir):
        # broad dir holds *_lagPat.npz only (no *_lagPat_withFreqCent.npz); the
        # canonical gate would reject any subject not in the legacy-variant set
        # (e.g. huangwanling), so accept broad lagPat explicitly for our dirs.
        if subject_dir in broad_dirs:
            return bool(list(subject_dir.glob("*_lagPat.npz")))
        return orig_has(dataset, subject_dir)

    rip._subject_dir = subdir_broad
    rip._has_propagation_inputs = has_broad

    common = ["--dataset", "yuquan", "--subjects", *subjects,
              "--masked-features", "--output-root", str(PROP_OUT)]

    saved = sys.argv
    try:
        print("\n=== STEP 1: broad PR-2 propagation (base) ===", flush=True)
        sys.argv = ["run_interictal_propagation", *common]
        rip.main()

        print("\n=== STEP 2: broad PR-2.5 time_split_reproducibility (merge) ===",
              flush=True)
        sys.argv = ["run_interictal_propagation", *common, "--pr25"]
        rip.main()
    finally:
        sys.argv = saved
        rip._subject_dir = orig_subdir
        rip._has_propagation_inputs = orig_has


def step3_rank_displacement(subjects):
    """Broad rank-displacement (reads broad PR-2, no PR-6 -> pr2_sentinel mask)."""
    import scripts.run_rank_displacement as rd

    rd.PR2_DIR = PROP_OUT / "per_subject"
    rd.PR6_DIR = PROP_OUT / "template_anchoring" / "per_subject"  # absent -> fallback
    rd.OUT_DIR = PROP_OUT / "rank_displacement"
    rd.OUT_PER_SUBJECT = rd.OUT_DIR / "per_subject"

    stems = [f"yuquan_{s}" for s in subjects]
    saved = sys.argv
    try:
        print("\n=== STEP 3: broad rank-displacement ===", flush=True)
        # NOTE: do NOT pass --masked-features; that would call _apply_masked_paths
        # and overwrite the broad globals set above.
        sys.argv = ["run_rank_displacement", "--subjects", *stems]
        rd.main()
    finally:
        sys.argv = saved


def step4_geometry(subjects):
    """Broad path_axis (broad lagPat + broad rank-displacement + coords)."""
    import scripts.run_propagation_skeleton_geometry as geo

    geo.RANKDISP = PROP_OUT / "rank_displacement" / "per_subject"

    orig_subdir = geo._subject_dir

    def subdir_broad(ds, subj):
        if ds == "yuquan" and subj in subjects:
            return BROAD_LAGPAT / subj
        return orig_subdir(ds, subj)

    geo._subject_dir = subdir_broad

    tokens = [f"yuquan:{s}" for s in subjects]
    out = GEOM_OUT / "components" / "path_axis"
    saved = sys.argv
    try:
        print("\n=== STEP 4: broad path_axis geometry ===", flush=True)
        sys.argv = ["run_propagation_skeleton_geometry",
                    "--subjects", *tokens, "--out", str(out)]
        geo.main()
    finally:
        sys.argv = saved
        geo._subject_dir = orig_subdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS,
                    help="Yuquan subject names (default: huangwanling zhaojinrui)")
    ap.add_argument("--only-step", type=int, default=None, choices=[1, 3, 4],
                    help="run a single step (1=propagation+pr25, 3=rankdisp, 4=geometry)")
    args = ap.parse_args()
    subjects = args.subjects
    print(f"broad-lagPat patch for: {subjects}", flush=True)
    print(f"  broad lagPat in : {BROAD_LAGPAT}", flush=True)
    print(f"  propagation out : {PROP_OUT}", flush=True)
    print(f"  geometry out    : {GEOM_OUT}", flush=True)

    if args.only_step in (None, 1):
        step1_2_propagation(subjects)
    if args.only_step in (None, 3):
        step3_rank_displacement(subjects)
    if args.only_step in (None, 4):
        step4_geometry(subjects)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()

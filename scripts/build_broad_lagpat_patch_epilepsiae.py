#!/usr/bin/env python3
"""Broad-lagPat propagation patch for Epilepsiae subjects (mirror of build_broad_lagpat_patch.py).

Adds the broad propagation → rank-displacement chain for Epilepsiae subjects that have a broad
lagPat re-pack (results/lagpat_broad_epilepsiae/<id>/) but LACK the downstream broad propagation
outputs (interictal_propagation_masked_broad/{per_subject,rank_displacement}). Default targets =
E590/E1084/E1146 (narrow field_concordance significant, missing from the broad-geometry cohort).

Same口径 as the existing 13 broad Epilepsiae subjects: top_n=20 broad lagPat (already re-packed),
--masked-features, parallel `_broad` output dirs, monkeypatch routing (no shared config edit).

Full chain (all four steps here — this ONE script produces every broad artifact the energy-field
extrapolation cohort reads; do not assume steps 1-3 are enough):
  1. run_interictal_propagation (base)   -> broad PR-2 adaptive_cluster JSON
  2. run_interictal_propagation (--pr25)  -> merge time_split_reproducibility
  3. run_rank_displacement                -> broad rank_a/b_dense_full + swap_class
  4. run_contact_plane_readout            -> broad observation-plane geometry (t_a/t_b), the
                                             `DEF_AXIS_DIR` that src/topic5_field_extrapolation.py
                                             loads via load_broad_axis_record().

Step-4 routing note: for Epilepsiae, run_contact_plane_readout._subject_dir uses EPILEPSIAE_ROOT,
NOT the --lagpat-root override (that flag only reroutes YUQUAN_ROOT). So step 4 monkeypatches
_subject_dir -> BROAD_LAGPAT/<subj> (same as steps 1-2) and passes --rankdisp-dir (broad) + --out
(the `_broad` geometry dir). --subjects are colon-form `epilepsiae:<id>`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BROAD_LAGPAT = REPO / "results" / "lagpat_broad_epilepsiae"
PROP_OUT = REPO / "results" / "interictal_propagation_masked_broad"
GEOM_OUT = (REPO / "results" / "spatial_modulation" / "propagation_geometry_broad"
            / "observation_readout" / "real_subjects")
DEFAULT_SUBJECTS = ["590", "1084", "1146"]


def step1_2_propagation(subjects):
    import scripts.run_interictal_propagation as rip
    broad_dirs = {BROAD_LAGPAT / s for s in subjects}
    orig_subdir = rip._subject_dir
    orig_has = rip._has_propagation_inputs

    def subdir_broad(dataset, root, subject):
        if dataset == "epilepsiae" and subject in subjects:
            return BROAD_LAGPAT / subject
        return orig_subdir(dataset, root, subject)

    def has_broad(dataset, subject_dir):
        if subject_dir in broad_dirs:
            return bool(list(subject_dir.glob("*_lagPat.npz")))
        return orig_has(dataset, subject_dir)

    rip._subject_dir = subdir_broad
    rip._has_propagation_inputs = has_broad
    common = ["--dataset", "epilepsiae", "--subjects", *subjects,
              "--masked-features", "--output-root", str(PROP_OUT)]
    saved = sys.argv
    try:
        print("\n=== STEP 1: broad PR-2 propagation (base) ===", flush=True)
        sys.argv = ["run_interictal_propagation", *common]
        rip.main()
        print("\n=== STEP 2: broad PR-2.5 time_split_reproducibility (merge) ===", flush=True)
        sys.argv = ["run_interictal_propagation", *common, "--pr25"]
        rip.main()
    finally:
        sys.argv = saved
        rip._subject_dir = orig_subdir
        rip._has_propagation_inputs = orig_has


def step3_rank_displacement(subjects):
    import scripts.run_rank_displacement as rd
    rd.PR2_DIR = PROP_OUT / "per_subject"
    rd.PR6_DIR = PROP_OUT / "template_anchoring" / "per_subject"   # absent -> fallback
    rd.OUT_DIR = PROP_OUT / "rank_displacement"
    rd.OUT_PER_SUBJECT = rd.OUT_DIR / "per_subject"
    stems = [f"epilepsiae_{s}" for s in subjects]
    saved = sys.argv
    try:
        print("\n=== STEP 3: broad rank-displacement ===", flush=True)
        sys.argv = ["run_rank_displacement", "--subjects", *stems]
        rd.main()
    finally:
        sys.argv = saved


def step4_geometry(subjects):
    import scripts.run_contact_plane_readout as cpr

    orig_subdir = cpr._subject_dir
    subject_set = set(subjects)

    def subdir_broad(ds, subj):
        if ds == "epilepsiae" and subj in subject_set:
            return BROAD_LAGPAT / subj
        return orig_subdir(ds, subj)

    cpr._subject_dir = subdir_broad
    tokens = [f"epilepsiae:{s}" for s in subjects]
    saved = sys.argv
    try:
        print("\n=== STEP 4: broad contact-plane readout ===", flush=True)
        sys.argv = ["run_contact_plane_readout",
                    "--subjects", *tokens,
                    "--rankdisp-dir", str(PROP_OUT / "rank_displacement" / "per_subject"),
                    "--out", str(GEOM_OUT)]
        cpr.main()
    finally:
        sys.argv = saved
        cpr._subject_dir = orig_subdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--only-step", type=int, default=None, choices=[1, 3, 4])
    args = ap.parse_args()
    print(f"broad-lagPat patch (Epilepsiae) for: {args.subjects}", flush=True)
    print(f"  broad lagPat in : {BROAD_LAGPAT}\n  propagation out : {PROP_OUT}", flush=True)
    print(f"  geometry out    : {GEOM_OUT}", flush=True)
    if args.only_step in (None, 1):
        step1_2_propagation(args.subjects)
    if args.only_step in (None, 3):
        step3_rank_displacement(args.subjects)
    if args.only_step in (None, 4):
        step4_geometry(args.subjects)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()

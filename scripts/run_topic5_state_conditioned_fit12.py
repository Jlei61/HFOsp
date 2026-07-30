#!/usr/bin/env python
"""Run Figure 6 Fit 1 parity and Fit 2 prefix-only scaffold retention.

Fit 1 reruns the accepted clinical-onset, phenotype-matched BB150/gamma,
all-contact channel-shuffle scorer without changing any scientific input.

Fit 2 swaps only the interictal field root and accepted-event inventory to the
prefix-only artifacts. Activation caches, clinical-onset alignment, scorer,
mirror/maxAB selection, null, and seizure->subject->cohort fold remain identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_topic5_clinical_onset_gradient_field_cohort_stat as accepted


CANONICAL_SUMMARY = (
    ROOT
    / "results/topic5_ictal_recruitment/tspectral_field_concordance/"
    "clinical_onset_gradient_field_cohort_stat_summary.json"
)
CANONICAL_FIELD_ROOT = (
    ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
)
PREFIX_ROOT = ROOT / "results/topic5_state_conditioned_predictor/fit2_prefix_scaffold"
OUTPUT_ROOT = ROOT / "results/topic5_state_conditioned_predictor/fit12_clinical_bb150"
PAPER_ROOT = (
    ROOT / "results/paper-ready-figure/fig6_state_conditioned_predictor/fit12"
)


def _configure_run(mode: str, out: Path, paper: Path) -> None:
    """Configure the accepted scorer without modifying its canonical outputs."""
    accepted.OUT = out
    accepted.PAPER = paper
    accepted.PAPER_FIGURES = paper / "figures"
    accepted.STEM = f"fig6_{mode}_clinical_onset_scaffold"
    accepted.CONTRACT = (
        "topic5_fig6_fit1_exact_clinical_bb150_parity_v2"
        if mode == "fit1"
        else "topic5_fig6_fit2_prefix_only_clinical_bb150_v2"
    )
    if mode == "fit1":
        accepted.FIELD_ROOT = CANONICAL_FIELD_ROOT
        accepted.BB150_CACHE = (
            ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
        )
    else:
        accepted.FIELD_ROOT = PREFIX_ROOT / "per_subject"
        accepted.BB150_CACHE = PREFIX_ROOT / "bb150_cache_view"


def _group(summary: dict, group_id: str) -> dict:
    return next(
        row for row in summary["cohort_statistics"] if row["group_id"] == group_id
    )


def verify_fit1(summary: dict, atol: float = 1e-12) -> dict:
    reference = json.loads(CANONICAL_SUMMARY.read_text())
    keys = (
        "n_subjects",
        "n_seizures",
        "data_median",
        "null_median",
        "margin_median",
        "n_data_gt_null",
        "wilcoxon_one_sided_data_gt_null_p",
    )
    comparisons = []
    for group_id in ("all_phenotype_matched", "strict_broadband", "gamma_nonbroadband"):
        got = _group(summary, group_id)
        ref = _group(reference, group_id)
        for key in keys:
            a, b = got[key], ref[key]
            if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                ok = int(a) == int(b)
                delta = int(a) - int(b)
            else:
                ok = bool(np.isclose(float(a), float(b), atol=atol, rtol=0))
                delta = float(a) - float(b)
            comparisons.append(
                {
                    "group_id": group_id,
                    "field": key,
                    "observed": a,
                    "reference": b,
                    "delta": delta,
                    "pass": ok,
                }
            )
    return {
        "fit1_pass": bool(all(row["pass"] for row in comparisons)),
        "numeric_tolerance": atol,
        "comparisons": comparisons,
    }


def verify_fit2(summary: dict) -> dict:
    """Pre-result frozen coarse-scaffold gate; within-shaft is not a hard gate."""
    strict = _group(summary, "strict_broadband")
    pooled = _group(summary, "all_phenotype_matched")
    checks = {
        "strict_bb150_n_subjects_at_least_8": int(strict["n_subjects"]) >= 8,
        "strict_bb150_margin_positive": float(strict["margin_median"]) > 0,
        "strict_bb150_majority_positive": int(strict["n_data_gt_null"])
        > int(strict["n_subjects"]) / 2,
        "strict_bb150_paired_p_below_0p05": float(
            strict["wilcoxon_one_sided_data_gt_null_p"]
        )
        < 0.05,
    }
    return {
        "fit2_pass": bool(all(checks.values())),
        "primary_gate": "strict clinical-onset BB 1-150 vs all-contact channel-shuffle",
        "checks": checks,
        "strict_broadband": strict,
        "pooled_supportive": pooled,
        "within_shaft_role": "secondary anatomical sensitivity; never a hard stop",
    }


def run_mode(mode: str, n_perm: int, seed: int) -> tuple[dict, dict]:
    out = OUTPUT_ROOT / mode
    paper = PAPER_ROOT / mode
    _configure_run(mode, out, paper)
    summary = accepted.run(SimpleNamespace(n_perm=n_perm, seed=seed))
    verdict = verify_fit1(summary) if mode == "fit1" else verify_fit2(summary)
    (out / f"{mode}_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n"
    )
    return summary, verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fit1", "fit2", "both"], default="both")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    modes = ("fit1", "fit2") if args.mode == "both" else (args.mode,)
    for mode in modes:
        if mode == "fit2":
            required = (
                PREFIX_ROOT / "prefix_field_manifest.json",
                PREFIX_ROOT / "fit2_parent_event_allowlist.csv",
            )
            if not all(path.exists() for path in required):
                raise RuntimeError(
                    "Fit 2 prefix fields are missing; run "
                    "build_topic5_state_conditioned_prefix_fields.py first"
                )
        print(f"[{mode}] clinical-onset accepted scorer", flush=True)
        summary, verdict = run_mode(mode, args.n_perm, args.seed)
        results[mode] = {"summary": summary, "verdict": verdict}
        print(json.dumps(verdict, indent=2), flush=True)
        if mode == "fit1" and not verdict["fit1_pass"]:
            raise SystemExit("Fit 1 parity failed; Fit 2 is not interpretable")
    existing_path = OUTPUT_ROOT / "fit12_verdict.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    fit1_verdict = results.get("fit1", {}).get("verdict") or existing.get("fit1")
    fit2_verdict = results.get("fit2", {}).get("verdict") or existing.get("fit2")
    combined = {
        "contract": "topic5_fig6_fit12_clinical_bb150_v2",
        "old_gate_status": (
            "invalid_for_main_claim_adjudication: EEG-onset 1-8 Hz cross-patient "
            "Ridge is a separate directional task"
        ),
        "fit1": fit1_verdict,
        "fit2": fit2_verdict,
        "continue_formal_rnn": bool(
            (fit1_verdict or {}).get("fit1_pass", False)
            and (fit2_verdict or {}).get("fit2_pass", False)
        ),
    }
    (OUTPUT_ROOT / "fit12_verdict.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()

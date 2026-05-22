"""Smoke tests: --masked-features flag re-routes path globals to
`results/interictal_propagation_masked/template_pairing/...` in the three
PR-7 standalone scripts (runner, P3 equivalence, plotting).

These are minimal plumbing tests. Numerical contract for the masked feature
space lives in tests/test_interictal_propagation.py (Step 5a/5b function-level
tests). What we verify here is the runner-level wiring added for Topic 0
phantom-rank rerun roadmap Step 5g.

PR-7 compute helpers (`compute_pairing_with_nulls`,
`compute_burst_diagnostic_with_nulls`, `compute_transition_odds`) accept
`cluster_labels` directly and never run KMeans themselves — there is no
`use_masked_features` kwarg to plumb through compute. Therefore this file
contains only path-swap tests (3 of them), matching the
`scripts/run_rank_displacement.py` shape rather than the
`scripts/run_pr6_template_anchoring.py` shape.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _fresh_import(modname: str):
    """Reload module so prior in-process global mutations don't bleed in."""
    if modname in sys.modules:
        return importlib.reload(sys.modules[modname])
    return importlib.import_module(modname)


def _is_masked(p) -> bool:
    return "interictal_propagation_masked" in str(p)


def _is_legacy(p) -> bool:
    return "interictal_propagation/" in str(p) and not _is_masked(p)


def test_run_pr7_apply_masked_paths_swaps_all_eight_globals():
    m = _fresh_import("run_pr7_template_pairing")

    # Baseline: legacy results/interictal_propagation/...
    assert _is_legacy(m.PER_SUBJECT_DIR)
    assert _is_legacy(m.PR6_AUDIT_CSV)
    assert _is_legacy(m.OUT_DIR)
    assert _is_legacy(m.PER_SUBJECT_OUT)
    assert _is_legacy(m.PER_SUBJECT_BURST_OUT)
    assert _is_legacy(m.PER_SUBJECT_SWEEP_OUT)
    assert _is_legacy(m.AUDIT_CSV)
    assert _is_legacy(m.COHORT_SUMMARY_JSON)

    m._apply_masked_paths()

    # Post-swap: all 8 path globals carry the `_masked` parallel root
    assert _is_masked(m.PER_SUBJECT_DIR), m.PER_SUBJECT_DIR
    assert _is_masked(m.PR6_AUDIT_CSV), m.PR6_AUDIT_CSV
    assert _is_masked(m.OUT_DIR), m.OUT_DIR
    assert _is_masked(m.PER_SUBJECT_OUT), m.PER_SUBJECT_OUT
    assert _is_masked(m.PER_SUBJECT_BURST_OUT), m.PER_SUBJECT_BURST_OUT
    assert _is_masked(m.PER_SUBJECT_SWEEP_OUT), m.PER_SUBJECT_SWEEP_OUT
    assert _is_masked(m.AUDIT_CSV), m.AUDIT_CSV
    assert _is_masked(m.COHORT_SUMMARY_JSON), m.COHORT_SUMMARY_JSON

    # PR-2 input dir comes from masked per_subject (not from masked template_pairing per_subject)
    assert m.PER_SUBJECT_DIR.parts[-2:] == (
        "interictal_propagation_masked",
        "per_subject",
    )
    # PR-7 outputs go under template_pairing/
    assert m.OUT_DIR.parts[-2:] == (
        "interictal_propagation_masked",
        "template_pairing",
    )
    # PR-6 audit input comes from masked template_anchoring/cohort_audit.csv
    assert m.PR6_AUDIT_CSV.parts[-3:] == (
        "interictal_propagation_masked",
        "template_anchoring",
        "cohort_audit.csv",
    )

    # Raw signal roots are NOT masked (they live on /mnt)
    assert "interictal_propagation_masked" not in str(m.YUQUAN_RAW_ROOT)
    assert "interictal_propagation_masked" not in str(m.EPILEPSIAE_RAW_ROOT)


def test_pr7_addendum_p3_apply_masked_paths_swaps_three_globals():
    m = _fresh_import("pr7_addendum_p3_equivalence")

    assert _is_legacy(m.PER_SUBJECT_DIR)
    assert _is_legacy(m.BURST_DIR)
    assert _is_legacy(m.OUT_DIR)

    m._apply_masked_paths()

    assert _is_masked(m.PER_SUBJECT_DIR), m.PER_SUBJECT_DIR
    assert _is_masked(m.BURST_DIR), m.BURST_DIR
    assert _is_masked(m.OUT_DIR), m.OUT_DIR

    # P3 verdict gets written to the masked tree, not orig
    assert m.OUT_DIR.parts[-2:] == (
        "interictal_propagation_masked",
        "template_pairing",
    )
    # Burst input dir is per_subject_burst (not per_subject!)
    assert m.BURST_DIR.parts[-3:] == (
        "interictal_propagation_masked",
        "template_pairing",
        "per_subject_burst",
    )


def test_plot_pr7_apply_masked_paths_swaps_all_five_globals():
    m = _fresh_import("plot_pr7_template_pairing")

    assert _is_legacy(m.COHORT_SUMMARY)
    assert _is_legacy(m.PER_SUBJECT_DIR)
    assert _is_legacy(m.FIG_DIR)
    assert _is_legacy(m.SWEEP_DIR)
    assert _is_legacy(m.AUDIT_CSV_PATH)

    m._apply_masked_paths()

    assert _is_masked(m.COHORT_SUMMARY), m.COHORT_SUMMARY
    assert _is_masked(m.PER_SUBJECT_DIR), m.PER_SUBJECT_DIR
    assert _is_masked(m.FIG_DIR), m.FIG_DIR
    assert _is_masked(m.SWEEP_DIR), m.SWEEP_DIR
    assert _is_masked(m.AUDIT_CSV_PATH), m.AUDIT_CSV_PATH

    # Figures get written to the masked tree, not orig
    assert m.FIG_DIR.parts[-3:] == (
        "interictal_propagation_masked",
        "template_pairing",
        "figures",
    )

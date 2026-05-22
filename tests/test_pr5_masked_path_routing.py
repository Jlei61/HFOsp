"""Smoke tests: --masked-features flag re-routes path globals to
`results/interictal_propagation_masked/...` in the three PR-5 auxiliary
scripts (run_pr5b_share_extended, run_pr5_transition_windows,
plot_template_share_switching).

These are minimal plumbing tests. PR-5-A and PR-5-B inside
`scripts/run_interictal_propagation.py` are already covered by the main
runner's `RESULTS_DIR` auto-route (since Step 5a) — no separate test
needed here for them. PR-5-A/B do not re-cluster; they consume cluster
labels already produced under masked PR-2 by Step 5a, so path routing
alone is the entire masked fix.

Test pattern mirrors tests/test_pr6_masked_path_routing.py from Step 5f
(Topic 0 phantom-rank rerun roadmap §5e).
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
    """Path is under results/interictal_propagation but NOT under the masked
    parallel tree. Use parts-based check so RESULTS_DIR (ends without slash)
    is correctly classified."""
    parts = Path(p).parts
    return ("interictal_propagation" in parts) and not _is_masked(p)


def test_run_pr5b_share_extended_apply_masked_paths_swaps_globals():
    m = _fresh_import("run_pr5b_share_extended")

    # Baseline: legacy paths
    assert _is_legacy(m.RESULTS_DIR), m.RESULTS_DIR
    assert _is_legacy(m.PR1_SUBJECT_SUMMARY), m.PR1_SUBJECT_SUMMARY
    assert _is_legacy(m.OUT_JSON), m.OUT_JSON
    assert _is_legacy(m.OUT_CSV), m.OUT_CSV
    # Baseline pr1 summary uses the historical _n40 cohort tag
    assert m.PR1_SUBJECT_SUMMARY.name == "pr1_subject_summary_n40.json"

    m._apply_masked_paths()

    # Post-swap: all paths under masked tree
    assert _is_masked(m.RESULTS_DIR), m.RESULTS_DIR
    assert _is_masked(m.PR1_SUBJECT_SUMMARY), m.PR1_SUBJECT_SUMMARY
    assert _is_masked(m.OUT_JSON), m.OUT_JSON
    assert _is_masked(m.OUT_CSV), m.OUT_CSV

    # Critical: masked tree has NO _n40 suffix — Step 5a only writes
    # pr1_subject_summary.json. Mechanical dir-only swap would FileNotFound.
    assert m.PR1_SUBJECT_SUMMARY.name == "pr1_subject_summary.json", (
        m.PR1_SUBJECT_SUMMARY
    )

    # Output is under masked tree at the same relative location
    assert m.OUT_JSON.parts[-2:] == (
        "interictal_propagation_masked",
        "pr5b_recruitment_shift_extended.json",
    )


def test_run_pr5_transition_windows_apply_masked_paths_swaps_globals():
    m = _fresh_import("run_pr5_transition_windows")

    assert _is_legacy(m.RESULTS_DIR), m.RESULTS_DIR
    assert _is_legacy(m.PR1_SUBJECT_SUMMARY), m.PR1_SUBJECT_SUMMARY
    assert _is_legacy(m.OUT_JSON), m.OUT_JSON
    assert _is_legacy(m.OUT_CSV), m.OUT_CSV
    assert m.PR1_SUBJECT_SUMMARY.name == "pr1_subject_summary_n40.json"

    m._apply_masked_paths()

    assert _is_masked(m.RESULTS_DIR), m.RESULTS_DIR
    assert _is_masked(m.PR1_SUBJECT_SUMMARY), m.PR1_SUBJECT_SUMMARY
    assert _is_masked(m.OUT_JSON), m.OUT_JSON
    assert _is_masked(m.OUT_CSV), m.OUT_CSV
    assert m.PR1_SUBJECT_SUMMARY.name == "pr1_subject_summary.json"

    assert m.OUT_JSON.parts[-2:] == (
        "interictal_propagation_masked",
        "pr5_transition_windows.json",
    )


def test_plot_template_share_switching_apply_masked_paths_swaps_globals():
    m = _fresh_import("plot_template_share_switching")

    assert _is_legacy(m.RESULTS_DIR), m.RESULTS_DIR
    assert _is_legacy(m.EXT_JSON), m.EXT_JSON
    assert _is_legacy(m.ORIG_JSON), m.ORIG_JSON
    assert _is_legacy(m.TRANSITION_JSON), m.TRANSITION_JSON
    assert _is_legacy(m.FIGURES_DIR), m.FIGURES_DIR

    m._apply_masked_paths()

    assert _is_masked(m.RESULTS_DIR), m.RESULTS_DIR
    assert _is_masked(m.EXT_JSON), m.EXT_JSON
    assert _is_masked(m.ORIG_JSON), m.ORIG_JSON
    assert _is_masked(m.TRANSITION_JSON), m.TRANSITION_JSON
    assert _is_masked(m.FIGURES_DIR), m.FIGURES_DIR

    # Figures land under the masked template_share_switching/figures path
    assert m.FIGURES_DIR.parts[-3:] == (
        "interictal_propagation_masked",
        "template_share_switching",
        "figures",
    )

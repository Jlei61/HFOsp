"""Smoke tests: --masked-features flag re-routes path globals to
`results/interictal_propagation_masked/...` in the three PR-6 runners
(template_anchoring, step6 held-out, rank_displacement).

These are minimal plumbing tests. Numerical contract for the masked feature
space lives in tests/test_interictal_propagation.py (Step 5a/5b function-level
tests). What we verify here is the runner-level wiring added for Topic 0
phantom-rank rerun roadmap Step 5f.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

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


def test_template_anchoring_apply_masked_paths_swaps_all_five_globals():
    m = _fresh_import("run_pr6_template_anchoring")

    # Baseline: legacy results/interictal_propagation/template_anchoring/
    assert _is_legacy(m.PER_SUBJECT_DIR)
    assert _is_legacy(m.OUT_DIR)
    assert _is_legacy(m.PER_SUBJECT_OUT)
    assert _is_legacy(m.AUDIT_CSV)
    assert _is_legacy(m.COHORT_SUMMARY)

    m._apply_masked_paths()

    # Post-swap: all 5 path globals carry the `_masked` parallel root
    assert _is_masked(m.PER_SUBJECT_DIR), m.PER_SUBJECT_DIR
    assert _is_masked(m.OUT_DIR), m.OUT_DIR
    assert _is_masked(m.PER_SUBJECT_OUT), m.PER_SUBJECT_OUT
    assert _is_masked(m.AUDIT_CSV), m.AUDIT_CSV
    assert _is_masked(m.COHORT_SUMMARY), m.COHORT_SUMMARY

    # Specifically: input PR-2 JSONs come from the masked per_subject dir, not
    # the masked template_anchoring per_subject dir (don't confuse the two).
    assert m.PER_SUBJECT_DIR.parts[-2:] == ("interictal_propagation_masked", "per_subject")
    assert m.PER_SUBJECT_OUT.parts[-3:] == (
        "interictal_propagation_masked",
        "template_anchoring",
        "per_subject",
    )

    # SOZ JSON locations are NOT masked (they live at repo top-level results/)
    assert "interictal_propagation_masked" not in str(m.YUQUAN_SOZ_PATH)
    assert "interictal_propagation_masked" not in str(m.EPILEPSIAE_SOZ_PATH)


def test_step6_apply_masked_paths_swaps_inputs_outputs_and_cohort_summaries():
    m = _fresh_import("run_pr6_step6")

    assert _is_legacy(m.PR2_DIR)
    assert _is_legacy(m.PR6_DIR)
    assert _is_legacy(m.PR6_COHORT_SUMMARY)
    assert _is_legacy(m.RD_COHORT_SUMMARY)
    assert _is_legacy(m.DEFAULT_OUTPUT_DIR)

    m._apply_masked_paths()

    assert _is_masked(m.PR2_DIR), m.PR2_DIR
    assert _is_masked(m.PR6_DIR), m.PR6_DIR
    assert _is_masked(m.PR6_COHORT_SUMMARY), m.PR6_COHORT_SUMMARY
    assert _is_masked(m.RD_COHORT_SUMMARY), m.RD_COHORT_SUMMARY
    assert _is_masked(m.DEFAULT_OUTPUT_DIR), m.DEFAULT_OUTPUT_DIR

    # Sanity: PR-6 cohort summary used to seed step6 cohort lookup must come
    # from masked tree (otherwise we'd seed from the unmasked PR-6 list).
    assert m.PR6_COHORT_SUMMARY.parts[-3:] == (
        "interictal_propagation_masked",
        "template_anchoring",
        "cohort_summary.json",
    )


def test_rank_displacement_apply_masked_paths_swaps_inputs_and_output():
    m = _fresh_import("run_rank_displacement")

    assert _is_legacy(m.PR2_DIR)
    assert _is_legacy(m.PR6_DIR)
    assert _is_legacy(m.OUT_DIR)
    assert _is_legacy(m.OUT_PER_SUBJECT)

    m._apply_masked_paths()

    assert _is_masked(m.PR2_DIR), m.PR2_DIR
    assert _is_masked(m.PR6_DIR), m.PR6_DIR
    assert _is_masked(m.OUT_DIR), m.OUT_DIR
    assert _is_masked(m.OUT_PER_SUBJECT), m.OUT_PER_SUBJECT

    # Output writes to rank_displacement_masked, not template_anchoring_masked
    assert m.OUT_DIR.parts[-2:] == (
        "interictal_propagation_masked",
        "rank_displacement",
    )


def test_template_anchoring_split_half_robustness_forwards_use_masked_features(monkeypatch):
    """Verify Step 5b's `compute_time_split_reproducibility` call inside
    `compute_split_half_robustness` actually receives `use_masked_features=True`
    when the runner is invoked under --masked-features.
    """
    import numpy as np

    m = _fresh_import("run_pr6_template_anchoring")

    captured = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)
        return {"splits": {}}  # minimal valid shape

    monkeypatch.setattr(m, "compute_time_split_reproducibility", fake_compute)

    n_ch, n_ev = 8, 12
    cluster_data = {
        "full_bools": np.ones((n_ch, n_ev), dtype=bool),
        "full_ranks": np.tile(np.arange(n_ch).reshape(-1, 1).astype(float), (1, n_ev)),
        "full_block_ids": np.zeros(n_ev, dtype=int),
        "valid_event_indices": np.arange(n_ev, dtype=int),
        "labels": np.array([0, 1] * (n_ev // 2), dtype=int),
        "n_clusters": 2,
    }
    cand = {"channel_names": [f"C{i}" for i in range(n_ch)]}

    out = m.compute_split_half_robustness(
        cand, cluster_data, n_endpoint=3, use_masked_features=True
    )
    assert captured.get("use_masked_features") is True, captured
    assert out == {"per_split": {}}  # fake returned empty splits dict


def test_step6_process_one_passes_use_masked_features(monkeypatch, tmp_path):
    """Verify `_process_one` forwards `use_masked_features=True` to
    `compute_held_out_endpoint_validation`. We construct minimal fixtures so
    the function reaches the compute call.
    """
    import json

    import numpy as np

    m = _fresh_import("run_pr6_step6")

    # Stage a minimal PR-2 JSON for stem 'yuquan_x'
    pr2_root = tmp_path / "results" / "interictal_propagation" / "per_subject"
    pr2_root.mkdir(parents=True)
    pr6_root = (
        tmp_path
        / "results"
        / "interictal_propagation"
        / "template_anchoring"
        / "per_subject"
    )
    pr6_root.mkdir(parents=True)
    monkeypatch.setattr(m, "PR2_DIR", pr2_root)
    monkeypatch.setattr(m, "PR6_DIR", pr6_root)

    # PR-2 minimal payload: stable_k=2, two clusters with template_rank lists.
    n_ch = 10
    pr2_payload = {
        "channel_names": [f"E{i}" for i in range(n_ch)],
        "adaptive_cluster": {
            "stable_k": 2,
            "clusters": [
                {"template_rank": list(range(n_ch))},
                {"template_rank": list(range(n_ch - 1, -1, -1))},
            ],
        },
    }
    (pr2_root / "yuquan_x.json").write_text(json.dumps(pr2_payload))

    # Force the subject_dir resolution to a tmp dir (so it 'exists') and stub
    # `load_subject_propagation_events` to return a tiny block.
    fake_subj = tmp_path / "subj"
    fake_subj.mkdir()
    monkeypatch.setattr(m, "_subject_dir", lambda dataset, subject: fake_subj)

    n_ev = 8
    fake_loaded = {
        "ranks": np.tile(np.arange(n_ch).reshape(-1, 1).astype(float), (1, n_ev)),
        "bools": np.ones((n_ch, n_ev), dtype=bool),
        "event_abs_times": np.arange(n_ev, dtype=float),
        "block_ids": np.zeros(n_ev, dtype=int),
        "block_time_ranges": [(0.0, float(n_ev))],
        "channel_names": [f"E{i}" for i in range(n_ch)],
    }
    monkeypatch.setattr(m, "load_subject_propagation_events", lambda subj_dir: fake_loaded)

    captured = {}

    def fake_validate(**kwargs):
        captured.update(kwargs)
        return {
            "first_half": {"n_events": 4, "swap_class": "swap"},
            "second_half": {"n_events": 4, "swap_class_projected": "swap"},
            "validation": {"tier": "strong"},
        }

    monkeypatch.setattr(m, "compute_held_out_endpoint_validation", fake_validate)

    soz_lookup = {"yuquan": {"x": {"E0", "E1"}}}

    result = m._process_one("yuquan_x", soz_lookup, use_masked_features=True)
    assert captured.get("use_masked_features") is True, captured
    assert result is not None and result.get("exit_reason") is None

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p0_audit", ROOT / "scripts/audit_topic4_fcxr_lc5v2_p0_separation.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_q99_selection_requires_both_sides_of_separation():
    good = [{"name": "q099", "baseline_active_sample_fraction": 0.009,
             "early_median_active_cells_per_sample": 0.80}]
    assert AUDIT.select_policy(good) == "q099"
    bad_baseline = [{**good[0], "baseline_active_sample_fraction": 0.011}]
    bad_early = [{**good[0], "early_median_active_cells_per_sample": 0.70}]
    assert AUDIT.select_policy(bad_baseline) is None
    assert AUDIT.select_policy(bad_early) is None


def test_multitau_audits_have_distinct_transaction_roots():
    assert AUDIT._output_dir(3000.0) != AUDIT._output_dir(8000.0)
    assert AUDIT._output_dir(8000.0) != AUDIT._output_dir(15000.0)
    with pytest.raises(ValueError):
        AUDIT._output_dir(6000.0)

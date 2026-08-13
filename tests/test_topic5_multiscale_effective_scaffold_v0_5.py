import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_topic5_multiscale_effective_scaffold_v0_5",
    ROOT / "scripts/audit_topic5_multiscale_effective_scaffold_v0_5.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_densify_groups_removes_dropped_rank_gaps():
    groups = np.asarray([[0, 2, -1, 4], [1, -1, 3, 3]], dtype=np.int16)
    observed = MODULE.densify_groups(groups)
    expected = np.asarray([[0, 1, -1, 2], [0, -1, 1, 1]], dtype=np.int16)
    assert np.array_equal(observed, expected)


def test_relative_latency_span_requires_two_rank_sets():
    groups = np.asarray([[0, 1, -1], [0, 0, -1]], dtype=np.int16)
    lag = np.asarray([[0.010, 0.035, np.nan], [0.100, 0.120, np.nan]])
    span = MODULE.relative_latency_span_ms(lag, groups)
    assert np.isclose(span[0], 25.0)
    assert np.isnan(span[1])


def test_recovery_subjects_are_exactly_the_five_missing_spatial_patients():
    assert set(MODULE.RECOVERY_SUBJECTS) == {
        "epilepsiae_1077", "epilepsiae_1096", "epilepsiae_1125",
        "epilepsiae_139", "epilepsiae_635",
    }

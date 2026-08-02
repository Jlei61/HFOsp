import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "topic4_zm_lifecycle_control_selection",
    ROOT / "scripts/select_topic4_zm_lifecycle_control_candidates.py",
)
S = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(S)


def _row(rank, g_m, ratio, *, offset=None, gain=15.0, occupancy=0.5):
    return {
        "status": "complete", "selection_rank": rank, "config_id": f"{rank}-{g_m}",
        "arm": "i2e", "tau_D_ms": 500.0, "d_star": 0.7,
        "strength_scale": 1.0, "g_M": g_m, "tau_M_ms": 500.0, "g_Z": 1.0,
        "onset_ms": 500.0, "offset_ms": offset,
        "duration_right_censored": offset is None,
        "median_energy_gain_db": gain, "energy_occupancy_6db": occupancy,
        "paired_M_response": {"ratio_core_mean_hz": ratio},
    }


def test_selection_keeps_one_persistent_candidate_per_fast_phenotype():
    surface = {"rows": [
        _row(0, 1.0, 0.98), _row(0, 3.0, 0.65),
        _row(1, 1.0, 0.95), _row(1, 10.0, 0.55),
    ]}
    selected = S.select_candidates(surface)
    assert [row["selection_rank"] for row in selected] == [0, 1]
    assert [row["g_M"] for row in selected] == [3.0, 10.0]


def test_selection_rejects_native_offsets_and_low_energy_states():
    surface = {"rows": [
        _row(0, 3.0, 0.6, offset=3000.0),
        _row(0, 10.0, 0.6, gain=3.0),
        _row(0, 1.0, 0.98),
    ]}
    selected = S.select_candidates(surface)
    assert len(selected) == 1
    assert selected[0]["g_M"] == 1.0


def test_selection_deprioritises_near_silencing_suppression():
    surface = {"rows": [_row(0, 3.0, 0.65), _row(0, 10.0, 0.1)]}
    selected = S.select_candidates(surface)
    assert selected[0]["g_M"] == 3.0

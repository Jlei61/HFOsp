import numpy as np
import pytest

from src.topic4_fcxr_lc6_functional import (
    COMPONENTS,
    FunctionalResponseRecorder,
    local_patch_pattern,
    paired_response,
)


class _Cfg:
    use_m = False
    m_frozen_E = None
    use_pump = False
    E_E = 58.0
    e_gaba = 0.0


class _Slow:
    cfg = _Cfg()

    def __init__(self, g_i):
        self._gI_last_E = np.asarray(g_i, float)


def _recorder():
    positions = np.array([
        [0.0, 0.0], [0.2, 0.0], [0.6, 0.0], [-0.6, 0.0],
    ]) + 2.0
    return FunctionalResponseRecorder(
        positions, patch_center=[2.0, 2.0], axis_unit=[1.0, 0.0],
        dt_ms=1.0, window_edges_ms=(0, 2, 4, 6),
        axis_edges_mm=np.arange(-1.0, 1.01, .25), sheet_size_mm=4.0,
        n_map_bins_axis=4,
    )


def test_local_patch_is_finite_and_nonempty():
    got = local_patch_pattern(np.array([[0, 0], [1, 0]]), [0, 0], radius_mm=.2)
    np.testing.assert_array_equal(got, [1.0, 0.0])
    with pytest.raises(RuntimeError, match="no E cells"):
        local_patch_pattern(np.array([[0, 0]]), [2, 2], radius_mm=.2)


def test_recorder_uses_actual_reversal_aware_membrane_contributions():
    recorder = _recorder()
    slow = _Slow([.5, .5, .5, .5])
    for step in range(6):
        recorder.sample_membrane(
            step,
            np.full(4, 10.0),
            np.full(4, 2.0),
            np.full(4, 1.5),
            np.zeros(4),
            slow,
        )
        recorder.sample_spikes(step, np.array([0]))
    got = recorder.finalize()
    f_e = got["components"][:, COMPONENTS.index("F_E")]
    f_i = got["components"][:, COMPONENTS.index("F_I")]
    signed = got["components"][:, COMPONENTS.index("I_syn_signed")]
    np.testing.assert_allclose(f_e, 50.0)
    np.testing.assert_allclose(f_i, 5.0)
    np.testing.assert_allclose(signed, 45.0)
    np.testing.assert_allclose(got["active_fraction_1ms"], .25)


def test_paired_response_subtracts_sham_and_reports_zero_crossing():
    sham_rec = _recorder()
    probe_rec = _recorder()
    slow = _Slow([.5, .5, .5, .5])
    for step in range(6):
        sham_rec.sample_membrane(step, np.full(4, 10.0), np.full(4, 2.0), np.full(4, 1.5), np.zeros(4), slow)
        probe_rec.sample_membrane(step, np.full(4, 10.0), np.array([3.0, 3.0, 1.0, 1.0]), np.full(4, 1.5), np.zeros(4), slow)
    got = paired_response(sham_rec.finalize(), probe_rec.finalize())
    assert got["delta_components"].shape == (3, len(COMPONENTS), 4)
    assert len(got["window_zero_crossings"]) == 3

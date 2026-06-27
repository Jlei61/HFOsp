"""Contract tests for the mini-W_event B1c-rescue sidecar (--emit-ea-aux).

The runner gains an OPTIONAL, default-OFF sidecar `ea_aux_bins.npz` carrying the two
per-bin predictors B1c could not test from `ea_net_bins.npz` alone (DATA_MISSING items 1+3):

  - per-bin core_only (sham) EA-window count   -> a 'local rate' predictor;
  - per-bin first-spike onset time (ms)        -> TRUE recruitment order, replacing the
                                                  early-response-RANK proxy.

Invariants under test (no SNN is run except where noted; the helper is a pure read of
already-cached spikes):

  1. `_bin_first_onset_in_window` returns per-bin EARLIEST spike ms (rel window start),
     NaN for bins with no spike, NaN for an empty window.
  2. `--emit-ea-aux` is default OFF; `_measure_response`'s `collect_aux` defaults False, so
     every existing caller and the whole default path are untouched (byte-identical).
  3. `--emit-ea-aux` without `--emit-ea-bins` fails FAST (before any SNN/network build).

The dense per-bin K_min susceptibility map (DATA_MISSING item 2) is deliberately NOT added
here — it needs a separate dense source sweep and is a documented next step.
"""
import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_m3_kick_calibration import (  # noqa: E402
    _bin_first_onset_in_window,
    _measure_response,
    _build_argparser,
    main,
)


def test_first_onset_per_bin_min_and_nan():
    # 5 steps x 4 cells, dt=1ms, window [0,5). bin map: cells 0,1 -> bin0; cell2 -> bin1;
    # cell3 -> bin2. Spikes: cell0 first at step3, cell1 at step1, cell2 at step4, cell3 never.
    spk = np.zeros((5, 4), dtype=bool)
    spk[3, 0] = True
    spk[1, 1] = True
    spk[4, 2] = True
    res = {"E_spk_bool": spk}
    bin_of_cell = np.array([0, 0, 1, 2])
    onset = _bin_first_onset_in_window(res, bin_of_cell, n_bins=3, t_lo=0.0, t_hi=5.0, dt=1.0)
    assert onset[0] == pytest.approx(1.0)        # min(step3, step1) = 1ms
    assert onset[1] == pytest.approx(4.0)        # cell2 at step4
    assert np.isnan(onset[2])                    # cell3 never fired -> NaN


def test_first_onset_window_offset_is_relative_to_t_lo():
    # window [2,5) so a spike at absolute step 3 is 1ms into the window.
    spk = np.zeros((6, 1), dtype=bool)
    spk[3, 0] = True
    res = {"E_spk_bool": spk}
    onset = _bin_first_onset_in_window(res, np.array([0]), n_bins=1, t_lo=2.0, t_hi=5.0, dt=1.0)
    assert onset[0] == pytest.approx(1.0)        # step3 - lo_step2 = 1 step * 1ms


def test_first_onset_empty_window_all_nan():
    res = {"E_spk_bool": np.zeros((5, 3), dtype=bool)}
    onset = _bin_first_onset_in_window(res, np.array([0, 1, 2]), n_bins=3,
                                       t_lo=2.0, t_hi=2.0, dt=1.0)   # lo==hi -> empty
    assert np.all(np.isnan(onset))


def test_first_onset_no_spikes_all_nan():
    res = {"E_spk_bool": np.zeros((5, 3), dtype=bool)}
    onset = _bin_first_onset_in_window(res, np.array([0, 1, 2]), n_bins=3,
                                       t_lo=0.0, t_hi=5.0, dt=1.0)
    assert np.all(np.isnan(onset))


def test_collect_aux_defaults_false_in_measure_response():
    # Existing single caller passed no collect_aux before; the default MUST be False so the
    # default path never builds/serializes the aux rows -> byte-identical artifacts.
    assert _measure_response.__defaults__ is not None
    sig = inspect.signature(_measure_response)
    assert sig.parameters["collect_aux"].default is False


def test_emit_ea_aux_flag_default_off():
    args = _build_argparser().parse_args([])
    assert args.emit_ea_aux is False
    assert args.emit_ea_bins is False


def test_emit_ea_aux_requires_emit_ea_bins(monkeypatch):
    # --emit-ea-aux without --emit-ea-bins must fail FAST (before any SNN/network build).
    monkeypatch.setattr(sys, "argv",
                        ["run_m3_kick_calibration.py", "--run", "--emit-ea-aux"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert "requires --emit-ea-bins" in str(exc.value)

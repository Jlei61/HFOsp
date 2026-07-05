"""Contract tests for the multi-source kick resolver (--kick-xy), Step D.

mini-W_event needs to place the kick at an arbitrary (x,y) source while the
heterogeneous CORE field stays at core_center. `_resolve_kick_center_and_src`
returns (kick_center, src_bin_idx) and must hold three invariants:

  1. bit-parity: kick_xy=None reproduces the EXISTING behavior exactly
     (core mode -> core_center + the rep bin_idx; bare mode -> bin_centers[bin_idx]).
  2. when kick_xy is set, BOTH the kick center AND the source bin (the bin to
     exclude in the spatial measurement) move to (x,y) together.
  3. the core field is computed elsewhere at core_center and is NOT this function's
     concern — moving the kick must not move the core.

No SNN is run.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_m3_kick_calibration import _resolve_kick_center_and_src, main  # noqa: E402


_BINS = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [10.0, 10.0], [4.0, 4.0]])


def test_kick_xy_none_core_mode_is_bit_parity():
    # core mode, no kick_xy -> kick lands at core_center, source bin = the rep bin_idx
    core_center = np.array([4.0, 4.0])
    kc, src = _resolve_kick_center_and_src(
        kick_xy=None, core_center=core_center, core_mode=True,
        bin_centers=_BINS, bin_idx=4)
    assert np.allclose(kc, [4.0, 4.0])
    assert src == 4                      # unchanged rep bin index (bit-parity)


def test_kick_xy_none_bare_mode_is_bit_parity():
    # bare sheet, no kick_xy -> kick lands at this rep bin's center, src = bin_idx
    kc, src = _resolve_kick_center_and_src(
        kick_xy=None, core_center=None, core_mode=False,
        bin_centers=_BINS, bin_idx=1)
    assert np.allclose(kc, _BINS[1])     # [4,0]
    assert src == 1


def test_kick_xy_set_moves_kick_AND_source_bin_together():
    # kick placed exactly on bin index 3 ([10,10]) while core stays at center
    kc, src = _resolve_kick_center_and_src(
        kick_xy=[10.0, 10.0], core_center=np.array([4.0, 4.0]), core_mode=True,
        bin_centers=_BINS, bin_idx=4)
    assert np.allclose(kc, [10.0, 10.0])     # kick moved
    assert src == 3                          # source bin follows to the nearest bin


def test_kick_xy_offset_snaps_source_to_nearest_bin():
    # an off-grid kick at (3.6, 0.2) snaps the source bin to the nearest center ([4,0]=idx1)
    kc, src = _resolve_kick_center_and_src(
        kick_xy=[3.6, 0.2], core_center=np.array([4.0, 4.0]), core_mode=True,
        bin_centers=_BINS, bin_idx=4)
    assert np.allclose(kc, [3.6, 0.2])       # kick center is the exact (x,y), not snapped
    assert src == 1                          # but the excluded source BIN is the nearest one


def test_emit_ea_bins_requires_kick_xy(monkeypatch):
    # --emit-ea-bins without --kick-xy must fail FAST (before any SNN/network build),
    # so the npz is never a silently-truncated multi-bin bare-sheet sweep.
    monkeypatch.setattr(sys, "argv",
                        ["run_m3_kick_calibration.py", "--run", "--emit-ea-bins"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert "requires --kick-xy" in str(exc.value)

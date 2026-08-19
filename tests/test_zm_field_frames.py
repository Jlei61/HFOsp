"""Frame recording must not perturb the trajectory it is recording."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from kick_probe import simulate_kick  # noqa: E402
from model import build_network  # noqa: E402
from params import Params  # noqa: E402
from src.snn_engine.mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_zm_slow_vars import ZMTracedSlowVars  # noqa: E402


def _slow(n, ne, frames=0):
    module = ZMTracedSlowVars(n, 18.0, MZSlowVarsConfig(
        use_z=True, use_m=True, I_th_EI=0.5, tau_z=500.0, tau_adp=50.0,
        eta_m=0.02, trace_stride_steps=10), NE=ne,
        core_mask_E=np.zeros(ne, bool))
    if frames:
        module.enable_field_frames(frames)
    return module


def test_recording_frames_is_bit_identical_to_not_recording():
    """The figure must show the trajectory that was actually analysed."""
    p = Params(g=3.6, L=1.0, density=4000.0, T=200.0, dt=0.1,
               nu_ext_ratio=0.9, seed=11)
    net = build_network(p, verbose=False)
    out = []
    for frames in (0, 50):
        net["rng"] = np.random.default_rng(p.seed)
        res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9,
                            slow=_slow(net["NE"] + net["NI"], net["NE"], frames))
        out.append(res)
    assert np.array_equal(out[0]["rate_E"], out[1]["rate_E"])
    assert np.array_equal(out[0]["E_spk_bool"], out[1]["E_spk_bool"])


def test_frames_are_recorded_on_the_requested_stride():
    slow = _slow(10, 8, frames=3)
    for _ in range(10):
        slow.apply_currents(np.zeros(10), np.ones(10))
    frames = slow.field_frames()
    assert frames["net_slow_current"].shape == (4, 8)      # calls 0,3,6,9
    assert list(frames["call_index"]) == [0, 3, 6, 9]


def test_no_frames_recorded_by_default():
    assert _slow(10, 8).field_frames() is None


def test_frame_value_is_the_net_slow_current():
    slow = _slow(10, 8, frames=1)
    slow.z[:8] = 0.25
    slow.m[:8] = 4.0
    slow.apply_currents(np.zeros(10), np.full(10, 2.0))
    frame = slow.field_frames()["net_slow_current"][0]
    assert np.allclose(frame, (1.0 - 0.25) * 2.0 - 0.02 * 4.0)

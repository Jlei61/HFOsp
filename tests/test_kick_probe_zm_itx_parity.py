"""Off-by-default engine additions must not move the default path."""
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


def _tiny(T=400.0, seed=11):
    p = Params(g=3.6, L=1.0, density=4000.0, T=T, dt=0.1, nu_ext_ratio=0.9, seed=seed)
    net = build_network(p, verbose=False)
    return p, net


def _run(p, net, **kwargs):
    net["rng"] = np.random.default_rng(p.seed)
    return simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, **kwargs)


def test_post_record_zero_is_byte_identical():
    p, net = _tiny()
    a = _run(p, net, early_stop_runaway=True)
    b = _run(p, net, early_stop_runaway=True, post_runaway_record_ms=0.0)
    assert np.array_equal(a["rate_E"], b["rate_E"])
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert a["runaway_early_stop_ms"] == b["runaway_early_stop_ms"]
    assert b["post_runaway_recorded_ms"] == 0.0


def test_post_record_extends_only_the_tail():
    p, net = _tiny()
    a = _run(p, net, early_stop_runaway=True, es_thresh_hz=1.0, es_dur_ms=10.0)
    b = _run(p, net, early_stop_runaway=True, es_thresh_hz=1.0, es_dur_ms=10.0,
             post_runaway_record_ms=20.0)
    assert a["runaway_early_stop_ms"] == b["runaway_early_stop_ms"]
    n = len(a["rate_E"])
    assert len(b["rate_E"]) == min(n + 200, int(round(p.T / p.dt)))
    assert np.array_equal(a["rate_E"], b["rate_E"][:n])
    assert np.isclose(b["post_runaway_recorded_ms"], (len(b["rate_E"]) - n) * p.dt)


def test_post_record_never_exceeds_the_duration_cap():
    p, net = _tiny(T=60.0)
    b = _run(p, net, early_stop_runaway=True, es_thresh_hz=1.0, es_dur_ms=10.0,
             post_runaway_record_ms=100000.0)
    assert len(b["rate_E"]) <= int(round(p.T / p.dt))


def _tail_params(seed, T=200.0):
    return Params(g=3.6, L=1.0, density=4000.0, T=T, dt=0.1,
                  nu_ext_ratio=0.9, seed=seed)


def _full_with_checkpoint(p, net, steps, **kwargs):
    """Run the WHOLE trajectory while capturing. The checkpoint step must be
    inside range(nsteps): a head run shorter than the checkpoint step never
    reaches it and silently captures nothing."""
    captured = {}
    net["rng"] = np.random.default_rng(p.seed)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9,
                        checkpoint_steps=steps,
                        checkpoint_sink=lambda step, state: captured.setdefault(step, state),
                        **kwargs)
    return res, captured


def test_checkpoint_off_is_byte_identical():
    p, net = _tiny()
    a = _run(p, net)
    b = _run(p, net, checkpoint_steps=None, resume_state=None, time_offset_ms=0.0)
    assert np.array_equal(a["rate_E"], b["rate_E"])
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])


def test_capturing_does_not_perturb_the_trajectory():
    p, net = _tiny(T=400.0)
    plain = _run(p, net)
    withck, captured = _full_with_checkpoint(p, net, [2000])
    assert 2000 in captured
    assert np.array_equal(plain["rate_E"], withck["rate_E"])
    assert np.array_equal(plain["E_spk_bool"], withck["E_spk_bool"])


def test_checkpoint_and_resume_are_byte_identical():
    p, net = _tiny(T=400.0)
    full, captured = _full_with_checkpoint(p, net, [2000])
    assert captured[2000]["absolute_time_ms"] == 200.0

    tail = simulate_kick(_tail_params(p.seed), net, KICK_BOOST=0.0, t_kick=1e9,
                         resume_state=captured[2000], time_offset_ms=200.0)
    assert np.array_equal(tail["rate_E"], full["rate_E"][2000:])
    assert np.array_equal(tail["E_spk_bool"], full["E_spk_bool"][2000:])
    assert np.isclose(tail["times"][0], 200.0)


def test_resume_rejects_a_mismatched_clock():
    p, net = _tiny(T=400.0)
    _, captured = _full_with_checkpoint(p, net, [2000])
    try:
        simulate_kick(_tail_params(p.seed), net, KICK_BOOST=0.0, t_kick=1e9,
                      resume_state=captured[2000], time_offset_ms=0.0)
    except ValueError as exc:
        assert "clock" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError on a mismatched resume clock")


def _packet(net):
    NE, NI = net["NE"], net["NI"]
    mask = np.zeros(NE + NI, bool)
    mask[np.arange(0, NE, max(1, NE // 40))] = True
    return mask


def test_perturbed_resume_equals_full_rerun_with_the_same_packet():
    p, net = _tiny(T=400.0)
    packet = _packet(net)

    net["rng"] = np.random.default_rng(p.seed)
    full = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9,
                         forced_spike_mask=packet, forced_spike_ms=250.0)

    # Capture during an UNPERTURBED full run. The packet fires at step 2500,
    # after the step-2000 checkpoint, so the two trajectories are identical up
    # to the capture point -- assert that rather than assume it.
    sham_full, captured = _full_with_checkpoint(p, net, [2000])
    assert np.array_equal(sham_full["E_spk_bool"][:2500], full["E_spk_bool"][:2500])

    probe = simulate_kick(_tail_params(p.seed), net, KICK_BOOST=0.0, t_kick=1e9,
                          resume_state=captured[2000], time_offset_ms=200.0,
                          forced_spike_mask=packet, forced_spike_ms=250.0)
    assert np.array_equal(probe["E_spk_bool"], full["E_spk_bool"][2000:])
    assert np.array_equal(probe["rate_E"], full["rate_E"][2000:])
    assert probe["forced_spike_collision_count"] == full["forced_spike_collision_count"]


def test_sham_and_probe_from_one_checkpoint_diverge_only_after_injection():
    p, net = _tiny(T=400.0)
    packet = _packet(net)
    _, captured = _full_with_checkpoint(p, net, [2000])

    common = dict(KICK_BOOST=0.0, t_kick=1e9, time_offset_ms=200.0)
    sham = simulate_kick(_tail_params(p.seed), net,
                         resume_state=captured[2000], **common)
    probe = simulate_kick(_tail_params(p.seed), net,
                          resume_state=captured[2000],
                          forced_spike_mask=packet, forced_spike_ms=250.0, **common)
    inject = int(round((250.0 - 200.0) / 0.1))
    assert np.array_equal(sham["E_spk_bool"][:inject], probe["E_spk_bool"][:inject])
    assert not np.array_equal(sham["E_spk_bool"], probe["E_spk_bool"])

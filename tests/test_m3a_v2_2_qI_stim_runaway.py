"""TDD contracts for the M3A-v2.2 q_I stim-vs-no-stim runaway comparison GIF.

The figure is a paper-ready visual diagnostic: two arms on ONE shared substrate /
seed / kick schedule / q_I carrier, differing ONLY by a finite V_th clamp on the
four central ICL contacts (the "stimulation"). The whole comparison is valid only
if the arms are byte-identical BEFORE the stim window opens -- that is the load-
bearing contract here, so it gets a real (slow) simulation test, not a mock.

Run fast set:  pytest tests/test_m3a_v2_2_qI_stim_runaway.py -m "not slow"
Run all:       pytest tests/test_m3a_v2_2_qI_stim_runaway.py
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = os.path.join(ROOT, "src", "snn_engine")
PFIG = os.path.join(ROOT, "scripts", "paper_figures")
for p in (ENG, os.path.join(ROOT, "scripts"), ROOT, PFIG):
    if p not in sys.path:
        sys.path.insert(0, p)

# new comparison script (pure helpers)
from plot_fig_m3a_v2_2_qI_stim_runaway_gif import _select_middle_contacts, _stim_site_center  # noqa: E402
# the edited continuous-sim loop (stim params) lives in the companion runaway script
from plot_fig_m3a_v2_2_hG_runaway_transition_gif import (  # noqa: E402
    ProtocolConfig, _build, _simulate_continuous)


# ===========================================================================
# Pure helper: pick the four central ICL contacts (fast)
# ===========================================================================
def test_select_middle_contacts_picks_central_icl():
    names = ["SCL6", "SCL7", "ICL1", "ICL2", "ICL3", "ICL4",
             "ICL5", "ICL6", "ICL7", "ICL8", "ICL9"]
    # ICL_k at x=k on y=0; SCL off to the side. center at x=5.5 -> closest 4 ICL = 4,5,6,7
    contacts = np.array(
        [[0.0, 5.0], [1.0, 5.0]] + [[float(k), 0.0] for k in range(1, 10)], float)
    center = np.array([5.5, 0.0])
    idx = _select_middle_contacts(names, contacts, center, n=4)
    assert [names[i] for i in idx] == ["ICL4", "ICL5", "ICL6", "ICL7"]
    # returned indices must be sorted ascending (stable downstream plotting)
    assert list(idx) == sorted(idx)


def test_select_middle_contacts_ignores_non_icl_shafts():
    names = ["SCL6", "SCL7", "SCL8", "ICL1", "ICL2", "ICL3", "ICL4"]
    contacts = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0],
         [3.0, 0.0], [3.5, 0.0], [4.0, 0.0], [4.5, 0.0]], float)
    center = np.array([0.15, 0.0])  # nearest points are SCL, but we must pick ICL only
    idx = _select_middle_contacts(names, contacts, center, n=2)
    assert all(names[i].startswith("ICL") for i in idx)


# ===========================================================================
# Stim-site center: "earliest-endpoint" == first-kicked focus (tempA); "middle" == sheet center
# ===========================================================================
def test_stim_site_center_earliest_endpoint_is_first_kicked_focus():
    S = {"center": [5.0, 5.0], "axis_unit": [1.0, 0.0], "L": 10.0,
         "layout": {"foci": [[1.0, 2.0], [8.0, 2.0]]}}
    c = _stim_site_center(S, "earliest-endpoint")
    # tempA = foci[0] is kicked first (pulse k=0), so it is the earliest-onset endpoint
    assert list(np.asarray(c)) == [1.0, 2.0]


def test_stim_site_center_middle_is_sheet_center():
    S = {"center": [5.0, 5.0], "layout": {"foci": [[1.0, 2.0], [8.0, 2.0]]}}
    assert list(np.asarray(_stim_site_center(S, "middle"))) == [5.0, 5.0]


def test_stim_site_center_rejects_unknown():
    with pytest.raises(ValueError):
        _stim_site_center({"center": [0.0, 0.0], "layout": {"foci": [[1.0, 2.0], [8.0, 2.0]]}}, "nope")


# ===========================================================================
# Parity + silencing on a real (small-T) shared substrate (slow)
# ===========================================================================
@pytest.fixture(scope="module")
def arms():
    """Build ONE subject1146 substrate, run three arms that re-seed identically:
    none / inactive-stim (window beyond T) / clamp-all-from-mid."""
    cfg = ProtocolConfig(layout="subject1146", top="qI", use_gK=True, eta_K=0.0,
                         use_hG=False, T=200.0, gif_dt_ms=20.0)
    S = _build(cfg)
    N = S["N"]
    dt = S["p"].dt
    full_mask = np.ones(N, bool)
    none = _simulate_continuous(S, cfg, record_gif=False)
    inactive = _simulate_continuous(S, cfg, record_gif=False,
                                    stim_target=full_mask, stim_on=1e9, stim_off=2e9)
    stim_on = 80.0
    clamp = _simulate_continuous(S, cfg, record_gif=False,
                                 stim_target=full_mask, stim_on=stim_on, stim_off=1e9)
    return {"none": none, "inactive": inactive, "clamp": clamp,
            "stim_on_step": int(round(stim_on / dt)), "dt": dt}


@pytest.mark.slow
def test_inactive_stim_is_byte_identical_to_no_stim(arms):
    # a stim schedule whose window never opens must add no RNG draw, no threshold change
    assert np.array_equal(arms["none"]["E_spk_bool"], arms["inactive"]["E_spk_bool"])


@pytest.mark.slow
def test_arms_share_baseline_before_stim_on(arms):
    k = arms["stim_on_step"]
    assert np.array_equal(arms["none"]["E_spk_bool"][:k],
                          arms["clamp"]["E_spk_bool"][:k])


@pytest.mark.slow
def test_clamp_silences_targets_during_window(arms):
    k = arms["stim_on_step"]
    # baseline keeps firing after the (would-be) window opens...
    assert arms["none"]["E_spk_bool"][k:].sum() > 0
    # ...the clamp drives its targets (all E here) to exactly zero spikes
    assert arms["clamp"]["E_spk_bool"][k:].sum() == 0

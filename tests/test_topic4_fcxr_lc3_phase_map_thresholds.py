"""The per-neuron thresholds must reach every run the state map starts.

`run_fcxr_loop` defaults `v_th_per_neuron` to None and then uses one uniform threshold. The
pathology in this substrate *is* two patches of lowered threshold, so omitting the argument runs a
homogeneous sheet: it completes, it produces numbers, and nothing ever ignites. The first version
of the state map omitted it at all three call sites and built a reference trajectory whose slow
variables never moved -- disinhibition 0.0000 at 12 s where the same working point reaches 0.436.

Nothing about that failure is loud, so it gets a test.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture()
def phase_map(monkeypatch):
    import run_topic4_fcxr_lc3_phase_map as M
    seen = {}

    def _fake(p, net, **kw):
        seen.update(kw)
        seen["T"] = p.T
        return {"checkpoint": object()}

    monkeypatch.setattr(M, "run_fcxr_loop", _fake)
    return M, seen


@dataclasses.dataclass
class _P:
    T: float = 1.0
    dt: float = 0.05


def _substrate():
    return dict(p=_P(), net={}, NE=4, N=5, vth=np.array([17.5, 18.0, 17.5, 18.0]))


def test_the_helper_always_passes_the_per_neuron_thresholds(phase_map):
    M, seen = phase_map
    S = _substrate()
    M._loop(S, T_ms=100.0, slow=None, start=None, n_steps=1)
    assert "v_th_per_neuron" in seen, "the thresholds were dropped on the way to the engine"
    assert np.array_equal(seen["v_th_per_neuron"], S["vth"])


def test_the_helper_is_the_only_way_this_script_reaches_the_engine():
    """A second call site would be free to forget the argument again."""
    src = open(os.path.join(ROOT, "scripts",
                            "run_topic4_fcxr_lc3_phase_map.py")).read()
    calls = src.count("run_fcxr_loop(")
    assert calls == 1, (f"run_fcxr_loop is called {calls} times; every run must go through "
                        f"_loop so the thresholds cannot be dropped at one of them")


def test_a_uniform_threshold_would_remove_the_pathology_this_substrate_is_built_on():
    """Why the argument matters at all: the cores are the only thing making the sheet ignite."""
    vth = _substrate()["vth"]
    assert vth.min() < vth.max(), "the substrate's pathology is a spread in threshold"
    uniform = np.full_like(vth, 18.0)
    assert not np.array_equal(vth, uniform)

# tests/test_sef_hfo_axisA_ei_engine.py
"""TDD for the Axis-A local E/I lesion (guarded edit to connectivity_rot.build_connectivity_rot).

Step-0 contract (docs/superpowers/specs/2026-06-13-sef-hfo-snn-axisA-ei-local-lesion-design.md §1):
  - w_EI down = TARGET-indexed scalar on the E cell's GABA input (never w_II).
  - w_EE up   = EDGE-indexed, BOTH source AND target E in core (source-only would heat
                out-core targets -> breaks bare-sheet-quiet).
  - defaults  = BIT-IDENTICAL to the pre-edit engine (rng draw order unchanged).

These import connectivity_rot DIRECTLY (not through the guarded runner) so they run without
a re-bless.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ENG = Path(__file__).resolve().parents[1] / "results/topic4_sef_hfo/lif_snn/engine"
sys.path.insert(0, str(ENG))

from params import Params                       # noqa: E402
from connectivity import place_neurons          # noqa: E402
import connectivity_rot as cr                    # noqa: E402

THETA = np.deg2rad(45.0)
AR = 2.0


def _net(seed=7, L=8.0, **kw):
    p = Params(g=3.6, L=L, density=100.0, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = cr.build_connectivity_rot(p, pos, labels, NE, NI, rng, THETA, AR, **kw)
    return net, pos, labels, NE, NI


def _core_mask_E(pos, NE, L=8.0, r=1.5):
    posE = pos[:NE]
    center = np.array([L / 2, L / 2])
    focus = center - 0.6 * (L / 2) * np.array([np.cos(THETA), np.sin(THETA)])
    return np.linalg.norm(posE - focus, axis=1) <= r


def _chsum(byd):
    tot, nnz = 0.0, 0
    for M in byd:
        if M is None:
            continue
        d = np.asarray(M.data)
        tot += float(d.sum()); nnz += int(d.size)
    return round(tot, 6), nnz


def _ampa_full(net, NE):
    """Sum the by-delay AMPA sparse matrices -> dense N x NE adjacency (small net only)."""
    Ms = [M for M in net["ampa_by_delay"] if M is not None]
    A = Ms[0].toarray().astype(float)
    for M in Ms[1:]:
        A = A + M.toarray()
    return A


# ---------------------------------------------------------------------------
# 1. defaults bit-identical + deterministic
# ---------------------------------------------------------------------------
def test_default_deterministic_and_explicit_noop_identical():
    a, *_ = _net()
    b, pos, labels, NE, NI = _net()
    assert _chsum(a["ampa_by_delay"]) == _chsum(b["ampa_by_delay"])
    assert _chsum(a["gaba_by_delay"]) == _chsum(b["gaba_by_delay"])
    # explicit no-op args (ones / gain 1.0 / empty mask) == default
    cmE = _core_mask_E(pos, NE)
    c, *_ = _net(local_scale_EI=np.ones(NE + NI), w_EE_gain_core=1.0,
                 core_mask_E=np.zeros(NE, bool))
    assert _chsum(c["ampa_by_delay"]) == _chsum(a["ampa_by_delay"])
    assert _chsum(c["gaba_by_delay"]) == _chsum(a["gaba_by_delay"])


# ---------------------------------------------------------------------------
# 2/3. w_EI down: TARGET-indexed, in-core E GABA scaled, structure + a_tot intact
# ---------------------------------------------------------------------------
def test_w_EI_down_scales_in_core_E_gaba_only():
    base, pos, labels, NE, NI = _net()
    cmE = _core_mask_E(pos, NE)
    ls = np.ones(NE + NI); ls[:NE][cmE] = 0.5
    les, *_ = _net(local_scale_EI=ls)
    b_a, b_g = _chsum(base["ampa_by_delay"]), _chsum(base["gaba_by_delay"])
    l_a, l_g = _chsum(les["ampa_by_delay"]), _chsum(les["gaba_by_delay"])
    assert l_g[0] < b_g[0]            # total GABA mass drops
    assert l_g[1] == b_g[1]           # but no synapse removed (weights only)
    assert l_a == b_a                 # AMPA channel untouched


def test_w_II_never_scaled_by_EI_field():
    base, pos, labels, NE, NI = _net()
    # scale ONLY the I targets (indices >= NE); E targets left at 1.0 -> g must be UNCHANGED
    ls = np.ones(NE + NI); ls[NE:] = 0.1
    les, *_ = _net(local_scale_EI=ls)
    assert _chsum(les["gaba_by_delay"]) == _chsum(base["gaba_by_delay"])


# ---------------------------------------------------------------------------
# 4/5. w_EE up: BOTH-in-core edges only; out-core target NOT scaled even with in-core sources
# ---------------------------------------------------------------------------
def test_w_EE_up_raises_ampa_both_in_core_only():
    base, pos, labels, NE, NI = _net()
    cmE = _core_mask_E(pos, NE)
    les, *_ = _net(w_EE_gain_core=1.5, core_mask_E=cmE)
    b_a, b_g = _chsum(base["ampa_by_delay"]), _chsum(base["gaba_by_delay"])
    l_a, l_g = _chsum(les["ampa_by_delay"]), _chsum(les["gaba_by_delay"])
    assert l_a[0] > b_a[0]            # total AMPA mass rises
    assert l_a[1] == b_a[1]           # no synapse added
    assert l_g == b_g                 # GABA untouched


def test_w_EE_up_out_core_target_with_in_core_sources_not_scaled():
    # The discriminator: a target NOT in core, but with in-core SOURCES, must NOT be scaled
    # (both-in-core, not source-only). Inspect its AMPA row -> all E-source weights equal.
    base, pos, labels, NE, NI = _net()
    cmE = _core_mask_E(pos, NE)
    gain = 2.0
    les, *_ = _net(w_EE_gain_core=gain, core_mask_E=cmE)
    A = _ampa_full(les, NE)           # N x NE
    # find an out-core E target that connects to >=1 in-core source
    out_core_E = np.where(~cmE)[0]
    chosen = None
    for i in out_core_E:
        row = A[i]
        src = np.where(row != 0)[0]
        if src.size and cmE[src].any() and (~cmE[src]).any():
            chosen = (i, row, src); break
    assert chosen is not None, "need an out-core target with mixed in/out-core sources"
    i, row, src = chosen
    vals = row[src]
    # out-core target: every E-source weight identical (no gain applied anywhere)
    assert np.allclose(vals, vals[0]), "out-core target's edges must be unscaled (both-in-core rule)"
    # and an IN-core target WITH in-core sources DOES show the gain (two distinct weight levels)
    in_core_E = np.where(cmE)[0]
    found = False
    for j in in_core_E:
        rj = A[j]; s = np.where(rj != 0)[0]
        if s.size and cmE[s].any() and (~cmE[s]).any():
            uniq = np.unique(np.round(rj[s], 6))
            assert uniq.size >= 2, "in-core target should have scaled + unscaled edges"
            ratio = uniq.max() / uniq.min()
            assert abs(ratio - gain) < 1e-6
            found = True; break
    assert found, "need an in-core target with mixed sources"

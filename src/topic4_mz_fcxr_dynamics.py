"""FCXR Stage D — frozen fast-branch map (D1) + reduced-operator mode analysis (D2).

Question (D1, load-bearing): under the accepted FCXR-RC1 substrate (external additive FF +
recurrent conductance + recurrent-only smooth saturation, g_sat=21.6, dt=0.05) with ALL slow
variables frozen, does the fast E-I system possess a finite, stable, repeatably-enterable high
branch (fixed point or bounded orbit) along the frozen failure coordinate D — or only a
low<->runaway/ceiling cliff? Coexistence across D = the fold signature.

The frozen failure field is z_i(D) = clip(1 - D * p_i, 0, 1), where p_i is the LOCKED
onset-depletion spatial pattern (mean-1 normalized) taken from the upstream state-conditioned
susceptibility snapshots (1 - z_E[onset]). See docs/superpowers/plans/2026-07-20-topic4-mz-fcxr-stage-d.md.

This module holds pure logic + thin orchestration; the blessed SNN engine (kick_probe.py) is
never edited and the frozen-Z field rides the non-blessed mz_slow_vars plugin.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------------------
# D0.1 — locked onset-depletion field p_i + substrate-alignment gate
# ------------------------------------------------------------------------------------

def load_onset_depletion_pi(snapshot_npz):
    """Load the LOCKED per-E-neuron onset-depletion pattern p_i from a susceptibility snapshot.

    p_i = dep / mean(dep) with dep = clip(1 - z_E[onset], 0, inf): mean-depletion normalization
    (matches build_DA_q_field's `shape = dep / nanmean(dep)`), so a scalar failure coordinate D
    obeys mean(D * p_i) = D and z_i(D) = clip(1 - D * p_i, 0, 1) has mean depletion ~= D.

    Returns a dict carrying the field plus the substrate identity fields (pos_E / vth_E) that the
    alignment gate needs, so the field can be verified to map neuron-for-neuron onto the RC1
    substrate rather than by index luck.
    """
    z = np.load(snapshot_npz, allow_pickle=True)
    labels = list(z["snapshot_labels"])
    if "onset" not in labels:
        raise ValueError(f"snapshot {snapshot_npz} has no 'onset' state; labels={labels}")
    onset = z["z_E"][labels.index("onset")].astype(np.float64)      # per-E-neuron z at onset
    dep = np.clip(1.0 - onset, 0.0, None)
    m = float(np.mean(dep))
    if not (m > 0):
        raise ValueError("onset depletion has non-positive mean; snapshot carries no failure signal")
    return dict(
        p_i=dep / m,
        pos_E=z["pos_E"].astype(np.float64),
        vth_E=z["vth_E"].astype(np.float64),
        src_xy=np.asarray(z["src_xy"], float),
        snk_xy=np.asarray(z["snk_xy"], float),
        axis_unit=np.asarray(z["axis_unit"], float),
        L=float(z["L"]),
    )


def assert_field_substrate_aligned(pi_pack, S, *, atol_pos=1e-4, atol_vth=1e-4):
    """Raise ValueError unless the onset-depletion field maps neuron-for-neuron onto substrate S.

    The frozen field is applied by index (self.z[:NE] = z_frozen_E), so if the snapshot's E-neuron
    ordering does not match S's build_substrate ordering, the field is mis-registered and every
    downstream D1 result is silently contaminated (CLAUDE.md §6 paired-key discipline). We verify
    the two invariants that pin the ordering: E-neuron positions and per-neuron V_th.
    """
    NE = int(S["NE"])
    posE = np.asarray(S["posE"], float)[:NE]
    vthE = np.asarray(S["vth"], float)[:NE]
    field_pos = np.asarray(pi_pack["pos_E"], float)
    field_vth = np.asarray(pi_pack["vth_E"], float)
    if field_pos.shape[0] != NE:
        raise ValueError(f"NE mismatch: field has {field_pos.shape[0]} E cells, substrate has {NE}")
    if not np.allclose(field_pos, posE, atol=atol_pos):
        raise ValueError("onset-depletion pos_E does not match RC1 substrate posE (mis-registered field)")
    if not np.allclose(field_vth, vthE, atol=atol_vth):
        raise ValueError("onset-depletion vth_E does not match RC1 substrate vth (mis-registered field)")


# ------------------------------------------------------------------------------------
# D1.3 — frozen failure field z_i(D)
# ------------------------------------------------------------------------------------

def frozen_z_field(p_i, D):
    """Frozen inhibitory-efficacy field z_i(D) = clip(1 - D * p_i, 0, 1) along the failure coordinate D.

    p_i is the mean-1 onset-depletion pattern, so mean(D * p_i) = D and the mean depletion of the
    frozen field is ~= D — the same scalar coordinate as the unsaturated slow-fast-transition line
    (their sharp transition sits at D ~= 0.087)."""
    return np.clip(1.0 - float(D) * np.asarray(p_i, float), 0.0, 1.0)

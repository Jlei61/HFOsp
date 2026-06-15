"""
Rotated-anisotropic E->E connectivity for the axis-tracking discriminator test.

This is a thin variant of `connectivity.build_connectivity` that changes ONLY
the E->E AMPA channel (excitatory inputs onto E targets) to a ROTATED elliptical
exponential kernel with an explicit long-axis angle theta and aspect ratio AR.
E->I, I->E, I->I are kept BIT-IDENTICAL to the original isotropic (rho=0)
versions -- they go through the unchanged `_sample_partners` / `_kernel_logweights`
imported from connectivity.py.

Rotated kernel for displacement dz=(z1,z2), long-axis angle theta (rad), widths
l_par > l_perp:

    u =  cos(theta)*z1 + sin(theta)*z2      # along long axis
    v = -sin(theta)*z1 + cos(theta)*z2      # across
    logw = -sqrt((u/l_par)**2 + (v/l_perp)**2)

with l_par = l_EE*sqrt(AR), l_perp = l_EE/sqrt(AR) (geometric mean stays l_EE).
AR=2.0 -> anisotropic (l_par/l_perp = AR = 2); AR=1.0 -> circular (theta moot).

For theta=45deg, AR=2 the kernel reproduces the original rho_EE=0.6 elongation
direction (along the (1,1) diagonal) and a comparable elongation ratio (~2 vs the
original rho=0.6 ratio of ~1.9); confirmed by `_sanity_check` below.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse

# Reuse the original isotropic machinery verbatim for E->I, I->E, I->I and the
# delay-grouping. ONLY the E->E branch gets the rotated kernel.
from connectivity import _kernel_logweights, _sample_partners, _group_by_delay


def _kernel_logweights_rot(dz, l_par, l_perp, theta):
    """log of unnormalized rotated elliptical-exponential weight.

        u =  cos*z1 + sin*z2  ;  v = -sin*z1 + cos*z2
        logw = -sqrt((u/l_par)^2 + (v/l_perp)^2)
    """
    z1 = dz[:, 0]
    z2 = dz[:, 1]
    c, s = np.cos(theta), np.sin(theta)
    u = c * z1 + s * z2
    v = -s * z1 + c * z2
    q = np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2)
    return -q


def _sample_partners_rot(pos_t, src_pos, C, l_par, l_perp, theta, rng,
                         self_local=None):
    """Weighted reservoir sample (same Efraimidis-Spirakis scheme as
    connectivity._sample_partners) but with the ROTATED kernel weights."""
    dz = src_pos - pos_t                       # (Ns, 2)
    lw = _kernel_logweights_rot(dz, l_par, l_perp, theta)
    w = np.exp(lw - lw.max())
    if self_local is not None:
        w[self_local] = 0.0
    nz = int(np.count_nonzero(w))
    if nz == 0:
        return np.empty(0, dtype=np.int64)
    Cc = min(C, nz)
    Ns = len(src_pos)
    if Cc >= Ns:
        return np.arange(Ns)
    keys = rng.standard_exponential(Ns) / np.where(w > 0.0, w, np.inf)
    return np.argpartition(keys, Cc - 1)[:Cc]


def build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE, AR,
                           verbose=False, local_scale_EI=None,
                           w_EE_gain_core=1.0, core_mask_E=None):
    """Identical to connectivity.build_connectivity except the E->E AMPA channel
    uses a rotated elliptical-exponential kernel (theta_EE, AR). Every other
    channel is bit-identical to the original isotropic build.

    Axis-A local E/I lesion (2026-06-15, guarded edit — Step-0 contract in
    docs/superpowers/specs/2026-06-13-sef-hfo-snn-axisA-ei-local-lesion-design.md §1):
    an OPTIONAL per-neuron synaptic weight-scaling field, applied AFTER partner sampling
    (so the rng draw order — hence every default-path weight — is unchanged):
      - local_scale_EI: length-N (or NE) array; for an E TARGET i in the core it multiplies
        w_EI (the E cell's GABA input) by local_scale_EI[i] -> perisomatic inhibition
        collapse. TARGET-indexed scalar; never touches w_II. None -> no-op.
      - w_EE_gain_core + core_mask_E (length-NE E-local bool): for an E->E edge whose SOURCE
        and TARGET E neurons are BOTH in the core, multiply w_EE by w_EE_gain_core ->
        recurrent excitatory cluster. EDGE-indexed (both-in-core); source-only would heat
        out-core targets and break bare-sheet-quiet. gain==1.0 / mask None -> no-op.
    With all three at their defaults the build is BIT-IDENTICAL to the pre-edit engine.
    """
    w_EE, w_IE, w_EI, w_II = p.weights()
    posE = pos[:NE]
    posI = pos[NE:]
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)

    jump_ampa = tau_m / p.tau_r_AMPA
    jump_gaba = tau_m / p.tau_r_GABA

    inv_vdt = 1.0 / (p.v_axon)
    a_rows, a_cols, a_w, a_dly = [], [], [], []
    g_rows, g_cols, g_w, g_dly = [], [], [], []

    # rotated E->E widths (geometric mean stays l_EE)
    l_par = p.l_EE * np.sqrt(AR)
    l_perp = p.l_EE / np.sqrt(AR)

    for i in range(NE + NI):
        a_is_E = labels[i] == 0
        pt = pos[i]

        # ---- excitatory (AMPA) inputs: sources in E ----
        if a_is_E:
            # E target: ROTATED anisotropic E->E kernel (the only change)
            C = p.C_EE
            wval = w_EE * jump_ampa[i]
            self_local = i
            cols = _sample_partners_rot(pt, posE, C, l_par, l_perp, theta_EE,
                                        rng, self_local=self_local)
        else:
            # I target: original isotropic E->I (bit-identical to connectivity.py)
            C = p.C_IE
            l = p.l_IE
            rho = p.rho_IE
            wval = w_IE * jump_ampa[i]
            cols = _sample_partners(pt, posE, C, l, rho, rng, self_local=None)
        if cols.size:
            d = np.linalg.norm(posE[cols] - pt, axis=1)
            dly = p.tau0 + d * inv_vdt
            w_edge = np.full(cols.size, wval)
            # w_EE up (recurrent cluster): scale BOTH-in-core E->E edges (a_is_E => target i
            # is an E cell; core_mask_E[i] target in core, core_mask_E[cols] E-local source in core).
            if (a_is_E and w_EE_gain_core != 1.0 and core_mask_E is not None
                    and core_mask_E[i]):
                w_edge = np.where(core_mask_E[cols], wval * w_EE_gain_core, wval)
            a_rows.append(np.full(cols.size, i))
            a_cols.append(cols)
            a_w.append(w_edge)
            a_dly.append(dly)

        # ---- inhibitory (GABA) inputs: sources in I (bit-identical) ----
        C = p.C_EI if a_is_E else p.C_II
        l = p.l_EI if a_is_E else p.l_II
        rho = p.rho_EI if a_is_E else p.rho_II
        wval = (w_EI if a_is_E else w_II) * jump_gaba[i]
        # w_EI down (perisomatic inhibition collapse): scale the E TARGET's GABA input in core.
        if a_is_E and local_scale_EI is not None:
            wval = wval * float(local_scale_EI[i])
        self_local = None if a_is_E else (i - NE)
        cols = _sample_partners(pt, posI, C, l, rho, rng, self_local=self_local)
        if cols.size:
            d = np.linalg.norm(posI[cols] - pt, axis=1)
            dly = p.tau0 + d * inv_vdt
            g_rows.append(np.full(cols.size, i))
            g_cols.append(cols)
            g_w.append(np.full(cols.size, wval))
            g_dly.append(dly)

        if verbose and (i % 1000 == 0):
            print(f"  connectivity(rot): target {i}/{NE+NI}", flush=True)

    a_rows = np.concatenate(a_rows); a_cols = np.concatenate(a_cols)
    a_w = np.concatenate(a_w);       a_dly = np.concatenate(a_dly)
    g_rows = np.concatenate(g_rows); g_cols = np.concatenate(g_cols)
    g_w = np.concatenate(g_w);       g_dly = np.concatenate(g_dly)

    step = max(1, int(round(p.delay_dt / p.dt)))
    a_step = np.maximum(1, np.round(a_dly / p.delay_dt).astype(int)) * step
    g_step = np.maximum(1, np.round(g_dly / p.delay_dt).astype(int)) * step
    max_delay_steps = int(max(a_step.max(), g_step.max()))

    N = NE + NI
    ampa_by_delay = _group_by_delay(a_rows, a_cols, a_w, a_step, max_delay_steps, N, NE)
    gaba_by_delay = _group_by_delay(g_rows, g_cols, g_w, g_step, max_delay_steps, N, NI)

    if verbose:
        print(f"  synapses(rot): AMPA={a_w.size:,}  GABA={g_w.size:,}  "
              f"theta_EE={np.degrees(theta_EE):.0f}deg AR={AR}  "
              f"l_par={l_par:.3f} l_perp={l_perp:.3f}  "
              f"max_delay={max_delay_steps*p.dt:.2f} ms", flush=True)
    return dict(pos=pos, labels=labels, NE=NE, NI=NI,
                ampa_by_delay=ampa_by_delay, gaba_by_delay=gaba_by_delay,
                max_delay_steps=max_delay_steps)


def _partner_cov_axis(kernel_logw_fn, l_args, n=200000, rng=None):
    """Weighted partner-offset covariance major-axis angle (deg mod 180) and
    elongation ratio, for a kernel sampled densely around a central target.
    `kernel_logw_fn(dz, *l_args)` returns log-weights."""
    if rng is None:
        rng = np.random.default_rng(0)
    dz = rng.uniform(-2.0, 2.0, size=(n, 2))
    lw = kernel_logw_fn(dz, *l_args)
    w = np.exp(lw - lw.max())
    wsum = w.sum()
    mean = (w[:, None] * dz).sum(0) / wsum
    d = dz - mean
    C = (w[:, None, None] * np.einsum('ij,ik->ijk', d, d)).sum(0) / wsum
    evals, evecs = np.linalg.eigh(C)
    vmaj = evecs[:, 1]
    angle = np.degrees(np.arctan2(vmaj[1], vmaj[0])) % 180.0
    ratio = float(np.sqrt(evals[1] / evals[0]))
    return float(angle), ratio


def _sanity_check():
    """Confirm theta=45deg/AR=2 rotated kernel matches the original rho_EE=0.6
    elongation (direction ~45deg, ratio ~2)."""
    from params import Params
    p = Params()
    # original rho_EE=0.6
    a_orig, r_orig = _partner_cov_axis(_kernel_logweights, (p.l_EE, p.rho_EE))
    # rotated theta=45, AR=2
    theta = np.radians(45.0)
    AR = 2.0
    l_par = p.l_EE * np.sqrt(AR)
    l_perp = p.l_EE / np.sqrt(AR)
    a_rot, r_rot = _partner_cov_axis(_kernel_logweights_rot, (l_par, l_perp, theta))
    # rotated theta=0 and 90 for direction check
    a0, r0 = _partner_cov_axis(_kernel_logweights_rot,
                               (l_par, l_perp, np.radians(0.0)))
    a90, r90 = _partner_cov_axis(_kernel_logweights_rot,
                                 (l_par, l_perp, np.radians(90.0)))
    # isotropic AR=1
    aiso, riso = _partner_cov_axis(_kernel_logweights_rot,
                                   (p.l_EE, p.l_EE, np.radians(45.0)))
    print("=== connectivity_rot sanity check (partner-offset covariance) ===")
    print(f"  original rho_EE=0.6 : angle={a_orig:5.1f} deg  ratio={r_orig:.3f}")
    print(f"  rotated th=45 AR=2  : angle={a_rot:5.1f} deg  ratio={r_rot:.3f}  "
          f"(should match original)")
    print(f"  rotated th=0  AR=2  : angle={a0:5.1f} deg  ratio={r0:.3f}")
    print(f"  rotated th=90 AR=2  : angle={a90:5.1f} deg  ratio={r90:.3f}")
    print(f"  isotropic AR=1      : angle={aiso:5.1f} deg  ratio={riso:.3f}  "
          f"(should be ~1)")
    return dict(orig=(a_orig, r_orig), rot45=(a_rot, r_rot),
                rot0=(a0, r0), rot90=(a90, r90), iso=(aiso, riso))


if __name__ == "__main__":
    _sanity_check()

"""
Spatial layout + distance-dependent anisotropic connectivity + delays.

Implements, paper-exact:
  * neuron placement on an L x L sheet (uniform, density from Table 1 ratios)
  * elliptical-exponential connection kernel p_ab(z), Eq 8
  * distance-dependent conduction delays  tau_L = tau0 + d / v   (Eq 7)
  * per-target fixed in-degree C_ab (Table 1), sampled WITHOUT replacement
    with probability proportional to the kernel

Output weight matrices already bake the per-spike jump applied to the synaptic
gating variable s (Eq 4-5):  for a spike from population b onto a target i in
population a, the jump is   Delta s = (tau_m_i / tau_r_syn) * w_ab .

Returned objects:
  pos      (N,2)  positions in mm
  labels   (N,)   0 = excitatory, 1 = inhibitory  (E indices come first)
  NE, NI
  ampa_by_delay : list over delay-steps of CSR (N x NE) weight matrices
                  (sources = excitatory neurons, drive s_E)
  gaba_by_delay : list over delay-steps of CSR (N x NI) weight matrices
                  (sources = inhibitory neurons, drive s_I, subtracted in V)
  max_delay_steps

NOTE on scale: the per-target sampler is O(N * N_b) and is fine up to ~2e4
neurons (the smoke regime). Paper scale (1.1e5) needs a spatial-bin / KDTree
candidate restriction -- left as a TODO; the kernel + in-degree contract is
identical, only the candidate set is pruned.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse


def place_neurons(p, rng):
    N = int(round(p.density * p.L * p.L))
    NE = int(round(p.f_E * N))
    NI = N - NE
    pos = rng.uniform(0.0, p.L, size=(N, 2)).astype(np.float64)
    labels = np.zeros(N, dtype=np.int8)
    labels[NE:] = 1
    return pos, labels, NE, NI


def _kernel_logweights(dz, l, rho):
    """log of unnormalized p_ab(z), Eq 8 (drop constant prefactor: we sample a
    fixed number of partners, so only relative probabilities matter).

        q = sqrt(z1^2 - 2 rho z1 z2 + z2^2)
        logw = - q / (l * sqrt(1 - rho^2))
    """
    z1 = dz[:, 0]
    z2 = dz[:, 1]
    q = np.sqrt(np.maximum(z1 * z1 - 2.0 * rho * z1 * z2 + z2 * z2, 0.0))
    return -q / (l * np.sqrt(1.0 - rho * rho))


def _sample_partners(pos_t, src_pos, C, l, rho, rng, self_local=None):
    """Sample up to C source indices (without replacement) ~ kernel weight.

    Efraimidis-Spirakis weighted reservoir sampling: draw key_j = Exp(1)/w_j and
    keep the C smallest keys. O(Ns), and weights w=0 get key=inf (never chosen).
    """
    dz = src_pos - pos_t                       # (Ns, 2)
    lw = _kernel_logweights(dz, l, rho)
    w = np.exp(lw - lw.max())
    if self_local is not None:
        w[self_local] = 0.0                    # no autapse
    nz = int(np.count_nonzero(w))
    if nz == 0:
        return np.empty(0, dtype=np.int64)
    Cc = min(C, nz)
    Ns = len(src_pos)
    if Cc >= Ns:
        return np.arange(Ns)
    keys = rng.standard_exponential(Ns) / np.where(w > 0.0, w, np.inf)
    return np.argpartition(keys, Cc - 1)[:Cc]


def build_connectivity(p, pos, labels, NE, NI, rng, verbose=True):
    w_EE, w_IE, w_EI, w_II = p.weights()
    posE = pos[:NE]
    posI = pos[NE:]
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)

    # per-spike jump factor onto s (Eq 4-5): tau_m_target / tau_r_syn
    jump_ampa = tau_m / p.tau_r_AMPA           # (N,)  multiplies w_*E
    jump_gaba = tau_m / p.tau_r_GABA           # (N,)  multiplies w_*I

    inv_vdt = 1.0 / (p.v_axon)                 # mm/ms -> ms per mm
    # accumulate triplets for AMPA (E sources) and GABA (I sources)
    a_rows, a_cols, a_w, a_dly = [], [], [], []
    g_rows, g_cols, g_w, g_dly = [], [], [], []

    for i in range(NE + NI):
        a_is_E = labels[i] == 0
        pt = pos[i]

        # ---- excitatory (AMPA) inputs: sources in E ----
        C = p.C_EE if a_is_E else p.C_IE
        l = p.l_EE if a_is_E else p.l_IE
        rho = p.rho_EE if a_is_E else p.rho_IE
        wval = (w_EE if a_is_E else w_IE) * jump_ampa[i]
        self_local = i if a_is_E else None     # i is an E neuron -> exclude autapse
        cols = _sample_partners(pt, posE, C, l, rho, rng, self_local=self_local)
        if cols.size:
            d = np.linalg.norm(posE[cols] - pt, axis=1)
            dly = p.tau0 + d * inv_vdt
            a_rows.append(np.full(cols.size, i))
            a_cols.append(cols)
            a_w.append(np.full(cols.size, wval))
            a_dly.append(dly)

        # ---- inhibitory (GABA) inputs: sources in I ----
        C = p.C_EI if a_is_E else p.C_II
        l = p.l_EI if a_is_E else p.l_II
        rho = p.rho_EI if a_is_E else p.rho_II
        wval = (w_EI if a_is_E else w_II) * jump_gaba[i]
        self_local = None if a_is_E else (i - NE)   # i is an I neuron -> exclude autapse
        cols = _sample_partners(pt, posI, C, l, rho, rng, self_local=self_local)
        if cols.size:
            d = np.linalg.norm(posI[cols] - pt, axis=1)
            dly = p.tau0 + d * inv_vdt
            g_rows.append(np.full(cols.size, i))
            g_cols.append(cols)
            g_w.append(np.full(cols.size, wval))
            g_dly.append(dly)

        if verbose and (i % 1000 == 0):
            print(f"  connectivity: target {i}/{NE+NI}", flush=True)

    a_rows = np.concatenate(a_rows); a_cols = np.concatenate(a_cols)
    a_w = np.concatenate(a_w);       a_dly = np.concatenate(a_dly)
    g_rows = np.concatenate(g_rows); g_cols = np.concatenate(g_cols)
    g_w = np.concatenate(g_w);       g_dly = np.concatenate(g_dly)

    # quantize delays into integer steps of delay_dt
    step = max(1, int(round(p.delay_dt / p.dt)))
    a_step = np.maximum(1, np.round(a_dly / p.delay_dt).astype(int)) * step
    g_step = np.maximum(1, np.round(g_dly / p.delay_dt).astype(int)) * step
    max_delay_steps = int(max(a_step.max(), g_step.max()))

    N = NE + NI
    ampa_by_delay = _group_by_delay(a_rows, a_cols, a_w, a_step, max_delay_steps, N, NE)
    gaba_by_delay = _group_by_delay(g_rows, g_cols, g_w, g_step, max_delay_steps, N, NI)

    if verbose:
        print(f"  synapses: AMPA={a_w.size:,}  GABA={g_w.size:,}  "
              f"max_delay={max_delay_steps*p.dt:.2f} ms ({max_delay_steps} steps)",
              flush=True)
    return dict(pos=pos, labels=labels, NE=NE, NI=NI,
                ampa_by_delay=ampa_by_delay, gaba_by_delay=gaba_by_delay,
                max_delay_steps=max_delay_steps)


def _group_by_delay(rows, cols, w, steps, max_steps, N, Nsrc):
    """Return list[d] -> CSC (N x Nsrc) holding connections with delay-step d.

    CSC so that spike scatter can column-gather only the neurons that fired:
        contrib = Wd[:, spikers].sum(axis=1)
    """
    out = [None] * (max_steps + 1)
    order = np.argsort(steps, kind="stable")
    steps_s = steps[order]
    rows_s, cols_s, w_s = rows[order], cols[order], w[order]
    uniq, start = np.unique(steps_s, return_index=True)
    start = list(start) + [len(steps_s)]
    for k, d in enumerate(uniq):
        sl = slice(start[k], start[k + 1])
        out[d] = sparse.csc_matrix(
            (w_s[sl], (rows_s[sl], cols_s[sl])), shape=(N, Nsrc))
    empty = sparse.csc_matrix((N, Nsrc))
    out = [m if m is not None else empty for m in out]
    return out

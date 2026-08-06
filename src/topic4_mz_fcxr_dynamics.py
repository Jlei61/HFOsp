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


# ------------------------------------------------------------------------------------
# D1.5 — two-layer 8-label frozen fast-branch classifier (pure; TDD on synthetic rows)
# ------------------------------------------------------------------------------------
# Locked thresholds (clause 7). All are relative to the dt=0.05 slow-off baseline anchor
# (baseline_rate / baseline_sigma / baseline_af_q95), so they inherit dt-robustness.
THRESHOLDS = dict(
    HIGH_MS=1000.0,    # min elevation (on the SMOOTHED envelope) to call a state persistent high (>> ~12ms event)
    HIGH_OCC=0.5,      # trailing-window occupancy (fraction of tail bins above q95) for "still elevated at the end"
    MIN_HIGH_MS=300.0, # a "substantial" excursion (metastable candidate) must last at least this long
    CEIL_FRAC=0.90,    # mean tail active fraction >= this (+ low modulation) -> pinned refractory ceiling
    MOD_CEIL=0.10,     # envelope modulation below this at a ceiling -> pinned (no breathing)
    MOD_ORBIT=0.30,    # envelope modulation >= this on a persistent high -> oscillatory (orbit), else fixed plateau
    ENVELOPE_MS=30.0,  # population-envelope smoothing window (P1: bridges sub-window dips of a bursty sustained high)
    PLATEAU_TOL=0.20,  # two high ICs must land within this relative spread to count as the same plateau
)

_FINITE = ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")


def classify_run_provisional(row, T=THRESHOLDS):
    """Per-RUN provisional label from ONE (D, ic) trajectory at ONE window.

    The dt=0.05 slow-off anchor is near-silent between events (baseline_rate ~0.01Hz), so an
    instantaneous-rate bar is meaningless -- a single brief interictal event would clear it. "High" is
    therefore a SUSTAINED elevation: a long CONTIGUOUS run above the quiet af-q95 (high_duration_ms >>
    the ~12ms interictal event) that is STILL elevated at the window end (tail occupancy). Single-window
    only distinguishes still-elevated (FINITE_HIGH_*) from had-a-long-excursion-but-decayed
    (EXCURSION_DECAYED); attractor vs long-transient needs the two-window resolver (clause 4).
    """
    if row["numerical_unsafe"]:                                   # clause 1: unsafe checked FIRST
        return "NUMERICAL_UNSAFE"
    persistent_high = bool(row["high_duration_ms"] >= T["HIGH_MS"] and row["tail_high_frac"] >= T["HIGH_OCC"])
    if persistent_high:                                           # clause 3: duration AND tail occupancy
        if row["af_tail"] >= T["CEIL_FRAC"] and row["modulation"] < T["MOD_CEIL"]:
            return "REFRACTORY_CEILING"                           # clause 2: pinned ceiling before finite-high
        return "FINITE_HIGH_ORBIT" if row["oscillatory_candidate"] else "FINITE_HIGH_FIXED"
    if row["high_duration_ms"] >= T["MIN_HIGH_MS"]:               # substantial excursion that did not persist
        return "EXCURSION_DECAYED"
    return "DECAYS_TO_LOW"


def resolve_high_ic(prov_T1, prov_T2):
    """Two-window resolver (clause 4, F1): FINITE_HIGH requires present-at-end at BOTH windows; persisted
    at the short window but decayed by the longer window => long transient => METASTABLE_TRANSIENT."""
    if "NUMERICAL_UNSAFE" in (prov_T1, prov_T2):
        return "NUMERICAL_UNSAFE"
    if "REFRACTORY_CEILING" in (prov_T1, prov_T2):
        return "REFRACTORY_CEILING"
    fin1, fin2 = prov_T1 in _FINITE, prov_T2 in _FINITE
    if fin1 and fin2:
        return "FINITE_HIGH_ORBIT" if "FINITE_HIGH_ORBIT" in (prov_T1, prov_T2) else "FINITE_HIGH_FIXED"
    if fin1 and not fin2:                                          # high at T1, gone by T2 -> long transient
        return "METASTABLE_TRANSIENT"
    if "EXCURSION_DECAYED" in (prov_T1, prov_T2):
        return "METASTABLE_TRANSIENT"
    return "DECAYS_TO_LOW"


def _plateau_rel_spread(plateaus):
    p = np.asarray([x for x in plateaus if x is not None and np.isfinite(x)], float)
    if p.size < 2 or p.mean() <= 0:
        return float("nan")
    return float((p.max() - p.min()) / p.mean())


def classify_branch_D(low_label, high_labels, high_plateaus, T=THRESHOLDS):
    """Per-D label (clause 6: distinct layer from per-run) from the native-low run + the resolved high-IC runs.

    high_labels/high_plateaus are the RESOLVED per-high-IC labels (>=2 ICs) and their end-of-run plateau rates.
    """
    all_labels = [low_label] + list(high_labels)
    if "NUMERICAL_UNSAFE" in all_labels:
        return dict(D_label="NUMERICAL_UNSAFE", low_label=low_label, high_labels=list(high_labels),
                    plateau_rel_spread=float("nan"))
    fin_idx = [i for i, l in enumerate(high_labels) if l in _FINITE]
    spread = _plateau_rel_spread([high_plateaus[i] for i in fin_idx]) if fin_idx else float("nan")
    if fin_idx:
        if len(fin_idx) < 2:                                       # only one IC reached high -> not confirmed
            D_label = "UNRESOLVED"
        elif np.isfinite(spread) and spread > T["PLATEAU_TOL"]:    # clause 5: plateaus disagree
            D_label = "UNRESOLVED"
        elif low_label == "DECAYS_TO_LOW":
            D_label = "BISTABLE"                                   # low stays low, high stays high (coexistence)
        elif low_label in _FINITE:
            D_label = "FINITE_HIGH"                                # even the native-low IC settles high
        else:
            D_label = "UNRESOLVED"
    elif any(l == "REFRACTORY_CEILING" for l in high_labels):
        D_label = "REFRACTORY_CEILING"
    elif any(l == "METASTABLE_TRANSIENT" for l in high_labels):
        D_label = "METASTABLE_TRANSIENT"
    elif all(l == "DECAYS_TO_LOW" for l in all_labels):
        D_label = "LOW_ONLY"
    else:
        D_label = "UNRESOLVED"
    return dict(D_label=D_label, low_label=low_label, high_labels=list(high_labels),
                plateau_rel_spread=spread)


# ------------------------------------------------------------------------------------
# D1.5b — envelope-based persistence (reviewer P1: raw 1ms contiguity misses gapped/oscillatory high)
# ------------------------------------------------------------------------------------

def _moving_avg(x, w):
    x = np.asarray(x, float)
    if w <= 1 or x.size == 0:
        return x
    return np.convolve(x, np.ones(int(w)) / float(w), mode="same")


def _longest_true_ms(mask, bin_ms):
    m = np.asarray(mask, bool)
    if not m.any():
        return 0.0
    edges = np.flatnonzero(np.diff(np.r_[False, m, False]))
    return float((edges[1::2] - edges[::2]).max() * bin_ms)


def envelope_metrics(af, af_bin_ms, analysis_start_ms, baseline_af_q95, T=THRESHOLDS):
    """Smoothed population-envelope persistence. A 20-50ms envelope tolerates the brief sub-window dips that
    break raw 1ms-bin contiguity, so a gapped/bursty sustained high (seizure-like orbit) is NOT split into
    short segments and missed. Returns env_high_ms (longest contiguous smoothed elevation above the quiet
    q95), the trailing-window occupancy, and the envelope modulation (flat plateau ~0, oscillatory orbit high)."""
    af = np.asarray(af, float)
    a0 = max(0, int(round(analysis_start_ms / af_bin_ms)))
    seg = af[a0:]
    if seg.size == 0:
        return dict(env_high_ms=0.0, env_end_occ=0.0, env_occ=0.0, env_modulation=0.0)
    w = max(1, int(round(T["ENVELOPE_MS"] / af_bin_ms)))
    env = _moving_avg(seg, w)
    above = env > baseline_af_q95
    env_high_ms = _longest_true_ms(above, af_bin_ms)
    endn = max(1, int(round(500.0 / af_bin_ms)))
    env_end_occ = float(np.mean(above[-endn:]))
    hi = env[above]
    if hi.size >= 4:
        p90, p10 = float(np.percentile(hi, 90)), float(np.percentile(hi, 10))
        env_modulation = float((p90 - p10) / max(p90, 1e-12))
    else:
        env_modulation = 0.0
    return dict(env_high_ms=env_high_ms, env_end_occ=env_end_occ, env_occ=float(np.mean(above)),
                env_modulation=env_modulation, env_window_ms=float(seg.size * af_bin_ms))


def classify_run_envelope(row, T=THRESHOLDS):
    """Per-run label from smoothed-envelope persistence (supersedes the raw-contiguity classifier for the
    scientific verdict; needs the recorded trace -> envelope_metrics fields on the row). Fixes the
    self-contradictory FINITE_HIGH_ORBIT: an orbit is sustained ON THE ENVELOPE (gaps bridged) with high
    envelope modulation; a fixed high is a flat plateau; a ceiling is pinned (near-total participation, ~no
    modulation)."""
    if row["numerical_unsafe"]:
        return "NUMERICAL_UNSAFE"
    # persistent = elevated MOST of a long-enough window (occupancy, so gapped orbits are not missed) AND still
    # elevated at the end. env_high_ms (longest contiguous) only flags "had a substantial excursion".
    persistent = bool(row["env_window_ms"] >= T["HIGH_MS"] and row["env_occ"] >= T["HIGH_OCC"]
                      and row["env_end_occ"] >= T["HIGH_OCC"])
    if persistent:
        if row["af_tail"] >= T["CEIL_FRAC"] and row["env_modulation"] < T["MOD_CEIL"]:
            return "REFRACTORY_CEILING"
        return "FINITE_HIGH_ORBIT" if row["env_modulation"] >= T["MOD_ORBIT"] else "FINITE_HIGH_FIXED"
    if row["env_high_ms"] >= T["MIN_HIGH_MS"] or row["env_occ"] >= T["HIGH_OCC"]:
        return "EXCURSION_DECAYED"                             # substantial activity but not still-high-at-end
    return "DECAYS_TO_LOW"


# ------------------------------------------------------------------------------------
# D1.5c — workpoint-relative classifier (reviewer P0: "high" = sustained deviation ABOVE the accepted
#          interictal band, NOT above the near-zero quiet floor; the interictal workpoint is itself an
#          oscillatory event train and must be classified INTERICTAL_WORKPOINT, not finite-high)
# ------------------------------------------------------------------------------------
WP_THRESHOLDS = dict(
    ROLL_MS=300.0,       # rolling-mean rate window ("sustained" activity, not a single brief event)
    HIGH_MS=1000.0,      # a new high branch must stay above the interictal band for at least ~1 s
    HIGH_OCC=0.5,        # fraction of the window whose rolling rate is above the interictal upper bound
    ELEVATED_OCC=0.10,   # above this occupancy = more active than interictal (event train); below = workpoint
    MOD_ORBIT=0.30,      # rolling-rate modulation on a persistent high -> oscillatory (orbit), else fixed
    BASELINE_Q=99.0,     # interictal band upper edge = this percentile of the baseline rolling-mean rate
    PLATEAU_TOL=0.20,    # two high ICs must land within this relative plateau spread to count as one branch
)


def rolling_rate_upper(rate, dt_ms, win_ms=WP_THRESHOLDS["ROLL_MS"], q=WP_THRESHOLDS["BASELINE_Q"]):
    """Interictal band upper edge: the q-th percentile of the win_ms rolling-mean rate over an accepted
    interictal (slow-off / no-kick) reference run. A test run only counts as 'above the band' when its
    rolling rate exceeds THIS empirical bound -- not the near-zero quiet floor."""
    rate = np.asarray(rate, float)
    w = max(1, int(round(win_ms / dt_ms)))
    roll = _moving_avg(rate, w)
    return float(np.percentile(roll, q)) if roll.size else 0.0


def workpoint_metrics(rate, dt_ms, baseline_roll_hi, analysis_start_ms=0.0, T=WP_THRESHOLDS):
    """Metrics of a test run RELATIVE to the empirical interictal band: occupancy / longest stretch / tail
    occupancy of its ROLL_MS rolling-mean rate ABOVE baseline_roll_hi, plus the modulation of the above-band
    segment (flat high vs oscillatory)."""
    rate = np.asarray(rate, float)
    a0 = max(0, int(round(analysis_start_ms / dt_ms)))
    seg = rate[a0:]
    if seg.size == 0:
        return dict(roll_occ=0.0, roll_end_occ=0.0, roll_high_ms=0.0, roll_modulation=0.0, window_ms=0.0)
    w = max(1, int(round(T["ROLL_MS"] / dt_ms)))
    roll = _moving_avg(seg, w)
    above = roll > baseline_roll_hi
    endn = max(1, int(round(500.0 / dt_ms)))
    hi = roll[above]
    if hi.size >= 4:
        p90, p10 = float(np.percentile(hi, 90)), float(np.percentile(hi, 10))
        mod = float((p90 - p10) / max(p90, 1e-9))
    else:
        mod = 0.0
    return dict(roll_occ=float(np.mean(above)), roll_end_occ=float(np.mean(above[-endn:])),
                roll_high_ms=_longest_true_ms(above, dt_ms), roll_modulation=mod,
                window_ms=float(seg.size * dt_ms))


def classify_run_workpoint(row, T=WP_THRESHOLDS):
    """Per-run label relative to the accepted interictal workpoint (reviewer P0 negative-control contract):
    INTERICTAL_WORKPOINT = stays within the interictal band; ELEVATED_EVENT_TRAIN = above the band but not a
    >=1 s sustained excursion; METASTABLE_TRANSIENT = a >=1 s above-band excursion that decays; FINITE_HIGH_* =
    sustained above the band AND still above at the end (a new high branch), fixed vs orbit by modulation."""
    if row["numerical_unsafe"]:
        return "NUMERICAL_UNSAFE"
    occ, end_occ, hi_ms = row["roll_occ"], row["roll_end_occ"], row["roll_high_ms"]
    persistent = bool(row["window_ms"] >= T["HIGH_MS"] and occ >= T["HIGH_OCC"] and end_occ >= T["HIGH_OCC"])
    if persistent:
        return "FINITE_HIGH_ORBIT" if row["roll_modulation"] >= T["MOD_ORBIT"] else "FINITE_HIGH_FIXED"
    if hi_ms >= T["HIGH_MS"]:                                 # a >=1 s contiguous above-band excursion that decayed
        return "METASTABLE_TRANSIENT"
    if occ >= T["ELEVATED_OCC"]:                             # more active than interictal, but not a high branch
        return "ELEVATED_EVENT_TRAIN"
    return "INTERICTAL_WORKPOINT"                            # within the accepted interictal band


def resolve_high_ic_wp(t1, t2):
    """Two-window (T1,T2) resolution for the workpoint label set: a new high branch must be finite at BOTH
    windows; finite at T1 but gone by the longer T2 = a long transient (METASTABLE_TRANSIENT)."""
    if "NUMERICAL_UNSAFE" in (t1, t2):
        return "NUMERICAL_UNSAFE"
    f1, f2 = t1 in _FINITE, t2 in _FINITE
    if f1 and f2:
        return "FINITE_HIGH_ORBIT" if "FINITE_HIGH_ORBIT" in (t1, t2) else "FINITE_HIGH_FIXED"
    if f1 and not f2:
        return "METASTABLE_TRANSIENT"
    if "METASTABLE_TRANSIENT" in (t1, t2):
        return "METASTABLE_TRANSIENT"
    return t2   # non-finite: the longer window (T2) is the definitive read (elevation that decays -> its T2 label)


def classify_branch_D_wp(low_label, high_labels, high_plateaus, T=WP_THRESHOLDS):
    """Per-D label from the workpoint per-run labels (low + resolved high ICs). BISTABLE = interictal/elevated
    low coexisting with a finite-high branch under kick; FINITE_HIGH = even the native-low settles high."""
    all_labels = [low_label] + list(high_labels)
    if "NUMERICAL_UNSAFE" in all_labels:
        return dict(D_label="NUMERICAL_UNSAFE", low_label=low_label, high_labels=list(high_labels),
                    plateau_rel_spread=float("nan"))
    fin_idx = [i for i, l in enumerate(high_labels) if l in _FINITE]
    spread = _plateau_rel_spread([high_plateaus[i] for i in fin_idx]) if fin_idx else float("nan")
    if fin_idx:
        if len(fin_idx) < 2 or (np.isfinite(spread) and spread > T["PLATEAU_TOL"]):
            D_label = "UNRESOLVED"
        elif low_label in ("INTERICTAL_WORKPOINT", "ELEVATED_EVENT_TRAIN"):
            D_label = "BISTABLE"                              # low stays interictal/elevated, high goes finite
        elif low_label in _FINITE:
            D_label = "FINITE_HIGH"                           # even the native-low IC settles high
        else:
            D_label = "UNRESOLVED"
    elif any(l == "METASTABLE_TRANSIENT" for l in all_labels):
        D_label = "METASTABLE_TRANSIENT"
    elif any(l == "ELEVATED_EVENT_TRAIN" for l in all_labels):
        D_label = "ELEVATED_EVENT_TRAIN"
    else:
        D_label = "INTERICTAL_WORKPOINT"
    return dict(D_label=D_label, low_label=low_label, high_labels=list(high_labels), plateau_rel_spread=spread)


# ------------------------------------------------------------------------------------
# D2.10 — SNN-connectivity sech^2 effective-operator lens at a landmark (reuse P1-2)
# ------------------------------------------------------------------------------------

def snn_landmark_sech2(W_EE, g_raw, g_sat=21.6, k=6):
    """Effective-operator lens on the REAL 40k E->E connectivity at a landmark state (the P1-2 path):
    leading-mode IPR of raw W_EE vs the saturation-weighted operator diag(sech^2(g_raw/g_sat)) @ W_EE.
    Answers whether the smooth saturation localizes or preserves the dominant spatial mode (a localized
    leading mode would be a runaway-mode artifact; ~1/N stays global). g_raw = per-E-cell recurrent
    conductance snapshot (e.g. max_raw_gErec recorded during the branch cell). This is a REDUCED-MODEL
    connectivity lens on the SNN branch map, not a claim on its own (§6.3)."""
    from src.topic4_mz_fcxr_modes import leading_modes, effective_jacobian_modes
    lm = leading_modes(W_EE, k=k)
    eff = effective_jacobian_modes(W_EE, np.asarray(g_raw, float), float(g_sat), k=k)
    raw_ipr = lm.get("right_ipr", [float("nan")])
    return dict(
        raw_lead_ipr=float(raw_ipr[0]) if raw_ipr else float("nan"),
        eff_lead_ipr=float(eff.get("eff_leading_ipr", float("nan"))),
        sech2_min=float(eff.get("sech2_min", float("nan"))),
        sech2_mean=float(eff.get("sech2_mean", float("nan"))),
        sech2_p05=float(eff.get("sech2_p05", float("nan"))),
        N=int(W_EE.shape[0]),
    )

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
    K_HIGH=4.0,        # persist-at-end rate must exceed baseline_rate + K_HIGH*sigma
    CEIL_FRAC=0.90,    # af_tail >= this AND low modulation -> pinned refractory ceiling, not a finite attractor
    MOD_CEIL=0.10,     # modulation below this at a ceiling -> pinned (no breathing)
    MIN_HIGH_MS=300.0, # a "substantial" high excursion (metastable candidate) must last at least this long
    PLATEAU_TOL=0.20,  # two high ICs must land within this relative spread to count as the same plateau
)

_PER_RUN = ("NUMERICAL_UNSAFE", "REFRACTORY_CEILING", "FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT",
            "EXCURSION_DECAYED", "DECAYS_TO_LOW")
_FINITE = ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")


def _end_high(row, T):
    """persist-at-end (clause 3): trailing-window rate above baseline band AND participation above q95."""
    return bool(row["end_rate_hz"] > row["baseline_rate"] + T["K_HIGH"] * row["baseline_sigma"]
                and row["af_tail"] > row["baseline_af_q95"])


def classify_run_provisional(row, T=THRESHOLDS):
    """Per-RUN provisional label from ONE (D, ic) trajectory at ONE observation window.

    Single-window only distinguishes present-at-end (FINITE_HIGH_*) from had-excursion-but-decayed
    (EXCURSION_DECAYED) — the attractor-vs-long-transient call needs the two-window resolver (clause 4).
    """
    if row["numerical_unsafe"]:                                   # clause 1: unsafe checked FIRST
        return "NUMERICAL_UNSAFE"
    if _end_high(row, T):
        if row["af_tail"] >= T["CEIL_FRAC"] and row["modulation"] < T["MOD_CEIL"]:
            return "REFRACTORY_CEILING"                           # clause 2: pinned ceiling before finite-high
        return "FINITE_HIGH_ORBIT" if row["oscillatory_candidate"] else "FINITE_HIGH_FIXED"
    if row["high_duration_ms"] >= T["MIN_HIGH_MS"]:
        return "EXCURSION_DECAYED"                                # substantial excursion, did not persist
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

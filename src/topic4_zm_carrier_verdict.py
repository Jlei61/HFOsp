"""Pre-registered Z/M ictal-carrier gate verdicts (spec docs/superpowers/specs/2026-07-24-topic4-zm-
ictal-carrier-gate-design.md). This module is the SINGLE authority for the locked thresholds; the
metric-extraction code (src/topic4_zm_ictal_carrier.py) imports the constants + analyze_macroepisode
from here so the numbers live in exactly one pre-registered place.

Two verdicts, kept strictly separate from the M4-2 termination `fragment` label (which describes an
activity-trace shape, not whether an ictal carrier exists):

  ictal_carrier_verdict(m)  -> (label, detail)
      fail_runaway | fail_plateau | fail_hfo_like_train | candidate_source_only | candidate_observed_carrier
  lifecycle_verdict(carrier_label, exit_metrics) -> str   (only meaningful once a carrier is established)
      carrier_not_established | no_onset | prevention | persistent |
      terminate_to_silence | terminate_then_reignite | terminate_and_recover

Thresholds frozen 2026-07-24 BEFORE any carrier simulation ran. Do not tune to results.
"""
from __future__ import annotations

import numpy as np

# ---- macroepisode machinery (spec §2) ----
ON_FRAC = 0.30            # event-on level = baseline + ON_FRAC*amp
FLOOR_FRAC = 0.20         # sustained floor; troughs above => "not returned to baseline"
MAX_GAP_MS = 250.0        # max sub-FLOOR gap merged inside a carrier macroepisode
MIN_MACRO_MS = 2000.0     # min carrier macroepisode duration
OCCUPANCY_MIN = 0.80      # min fraction of the macroepisode above FLOOR
PRE_ONSET_MS = 300.0      # window for the interictal baseline (before first ictal onset)
BIN_FINE_MS = 5.0         # source-rate bin
SMOOTH_MS = 10.0          # source-rate moving-average pre-macroepisode
# ---- gate thresholds (spec §3-§4) ----
SEP_FACTOR = 2.0          # a dim is "separated" if it exceeds the interictal-median by this factor
A7_DIMS_REQUIRED = 2      # Gate A7: >= 2 of 3 {peak rate, duration, active-area}
N_CONTACTS_MIN = 2        # Gate B1: >= 2 contacts with sustained low-gamma enhancement
ENH_DB = 6.0              # "enhanced" = >= 6 dB (~4x power) over the pre-onset median
DIMS_REQUIRED = 3         # Gate B6: >= 3 of 4 {duration, duty-cycle, energy, spatial-extent}
FLASH_FRAC = 0.80         # Gate A8: whole-field flash if >= this frac of active area ignites...
FLASH_WINDOW_MS = 50.0    # ...within this window of first onset

CARRIER_CANDIDATES = ("candidate_source_only", "candidate_observed_carrier")


def _longest_subthreshold_run_ms(below, dt_ms):
    """Longest contiguous True run in `below` (bool), in ms."""
    best = cur = 0
    for v in below:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best * dt_ms


def _merge_episodes(above, gap_bins):
    """Contiguous True runs of `above`, merging sub-threshold gaps <= gap_bins. Returns [(i0, i1), ...]."""
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return []
    eps = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap_bins:
            eps.append((start, prev + 1))
            start = i
        prev = i
    eps.append((start, prev + 1))
    return eps


def analyze_macroepisode(e, dt_ms, pre_onset_ms=PRE_ONSET_MS):
    """Longest FLOOR-crossing macroepisode of a 1-D energy trace `e` (spec §2). Baseline from the
    leading `pre_onset_ms`; a macroepisode = the longest run above FLOOR after merging sub-FLOOR gaps
    <= MAX_GAP_MS, requiring at least one supra-ON point (a real event, not floor noise). Returns a dict
    with onset_ms (None if no event), duration_ms, occupancy, max_gap_ms, peak, baseline, amp."""
    e = np.asarray(e, float)
    n = e.size
    base = dict(onset_ms=None, duration_ms=0.0, occupancy=0.0, max_gap_ms=0.0,
                peak=float(e.max()) if n else 0.0, baseline=0.0, amp=0.0)
    if n == 0:
        return base
    n_pre = max(1, int(round(pre_onset_ms / dt_ms)))
    b = float(np.median(e[:n_pre]))
    peak = float(e.max())
    amp = peak - b
    base.update(baseline=b, amp=amp)
    if amp <= 1e-12:                                       # never rose above baseline -> no event
        return base
    on = b + ON_FRAC * amp
    floor = b + FLOOR_FRAC * amp
    gap_bins = int(round(MAX_GAP_MS / dt_ms))
    eps = _merge_episodes(e >= floor, gap_bins)
    real = [(i0, i1) for (i0, i1) in eps if (e[i0:i1] >= on).any()]
    if not real:
        return base
    i0, i1 = max(real, key=lambda ep: ep[1] - ep[0])
    span = e[i0:i1]
    base.update(onset_ms=float(i0 * dt_ms), duration_ms=float((i1 - i0) * dt_ms),
                occupancy=float((span >= floor).mean()),
                max_gap_ms=float(_longest_subthreshold_run_ms(span < floor, dt_ms)))
    return base


def is_sustained(macro):
    """A macroepisode is a sustained carrier iff it has an onset, lasts >= MIN_MACRO_MS, and stays above
    FLOOR for >= OCCUPANCY_MIN of its span (max_gap is <= MAX_GAP_MS by construction of the merge)."""
    return bool(macro.get("onset_ms") is not None
                and macro.get("duration_ms", 0.0) >= MIN_MACRO_MS
                and macro.get("occupancy", 0.0) >= OCCUPANCY_MIN)


def ictal_carrier_verdict(m):
    """Priority-ordered carrier adjudication (spec §5). `m` supplies already-extracted metrics; this
    function applies every locked threshold and returns (label, detail). It is the ONLY place allowed
    to emit a 'carrier' label."""
    detail = {}
    # --- runaway wins (A2): unbounded escalation is a dynamical verdict, not a carrier ---
    if m.get("runaway_early_stop_ms") is not None or m.get("tail_escalating"):
        detail["reason"] = "runaway_early_stop" if m.get("runaway_early_stop_ms") is not None else "tail_escalating"
        return "fail_runaway", detail
    # --- structural failure (A6/A8): whole-field flash / saturation / no recruitment ---
    if m.get("whole_field_flash") or m.get("saturated_plateau") or not m.get("has_recruitment", False):
        detail["reason"] = ("whole_field_flash" if m.get("whole_field_flash")
                            else "saturated_plateau" if m.get("saturated_plateau") else "no_recruitment")
        return "fail_plateau", detail
    # --- Gate A: sustained source macroepisode (A1/A3/A4/A5 via is_sustained) + differs from interictal (A7) ---
    src = m.get("src_macro") or {}
    gate_A = is_sustained(src) and int(m.get("src_sep_count", 0)) >= A7_DIMS_REQUIRED
    detail["gate_A"] = bool(gate_A)
    detail["src_duration_ms"] = src.get("duration_ms")
    detail["src_occupancy"] = src.get("occupancy")
    if not gate_A:
        detail["reason"] = ("source_not_sustained" if not is_sustained(src)
                            else "not_separated_from_interictal")
        return "fail_hfo_like_train", detail
    # --- Gate B: >=2 sustained contacts + high-freq + sustained best contact + 3/4 dims separated ---
    obs = m.get("obs_best_macro") or {}
    gate_B = (int(m.get("obs_n_sustained_contacts", 0)) >= N_CONTACTS_MIN
              and bool(m.get("obs_highfreq_enhanced"))
              and is_sustained(obs)
              and int(m.get("obs_sep_count", 0)) >= DIMS_REQUIRED)
    detail["gate_B"] = bool(gate_B)
    detail["obs_n_sustained_contacts"] = m.get("obs_n_sustained_contacts")
    return ("candidate_observed_carrier" if gate_B else "candidate_source_only"), detail


def lifecycle_verdict(carrier_label, exit_metrics):
    """Lifecycle outcome (spec §6), emitted ONLY once a carrier is established. If the carrier gate did
    not pass, returns the sentinel `carrier_not_established` -- never one of the 6 lifecycle labels, so a
    non-carrier can never masquerade as a lifecycle candidate (task §5.4)."""
    if carrier_label not in CARRIER_CANDIDATES:
        return "carrier_not_established"
    e = exit_metrics or {}
    if e.get("prevented"):
        return "prevention"
    if not e.get("onset_detected", False):
        return "no_onset"
    if not e.get("terminated", False):
        return "persistent"
    if e.get("reignited"):
        return "terminate_then_reignite"
    if e.get("interictal_recovered"):
        return "terminate_and_recover"
    return "terminate_to_silence"

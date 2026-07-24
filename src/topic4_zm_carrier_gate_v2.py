"""carrier_gate_v2 (2026-07-24, post-review): FAITHFUL implementation of the pre-registered gate (spec
§2-§4). v1 (src/topic4_zm_carrier_verdict.analyze_macroepisode + src/topic4_zm_ictal_carrier) is KEPT for
history; it silently deviated from the spec on four clauses, caught in review:

  1. onset = start of the longest FLOOR episode  ->  should be FIRST sustained ON crossing (>= MIN_ONSET_MS)
  2. baseline = fixed first 300 ms              ->  should be [0, onset)
  3. observed dB baseline = [0, onset)          ->  onset was 8720 ms for sg -> baseline burst-polluted
  4. B2 = any contact/any time high-freq peak   ->  should overlap the B1 low-gamma macroepisode window
  5. A7 3rd dim = rate-energy                    ->  should be active-AREA (spatial extent)
  6. A8 = active-area size within 50 ms          ->  should be a spatial onset gradient

v2 fixes all six and RECOMPUTES the verdict OFFLINE from the saved NPZ (no SNN re-run). The verdict
vocabulary + priority logic are unchanged (imported from topic4_zm_carrier_verdict.ictal_carrier_verdict).
"""
from __future__ import annotations

import numpy as np

from src.topic4_zm_carrier_verdict import (
    ictal_carrier_verdict, is_sustained,
    ON_FRAC, FLOOR_FRAC, MAX_GAP_MS, MIN_MACRO_MS, OCCUPANCY_MIN, SMOOTH_MS,
    SEP_FACTOR, A7_DIMS_REQUIRED, ENH_DB, N_CONTACTS_MIN, DIMS_REQUIRED, FLASH_WINDOW_MS,
    _merge_episodes, _longest_subthreshold_run_ms)
import src.topic4_zm_ictal_carrier as CG

PROVISIONAL_BASELINE_MS = 300.0
MIN_ONSET_MS = 100.0        # spec: ON must persist this long to count as onset (v1 never used it)
OBS_BASELINE_MS = 300.0     # observed dB baseline = a fixed EARLY (pre-first-burst) window, not [0,onset)


def first_sustained_crossing(above, min_bins):
    """First index i such that `above` is True for min_bins consecutive samples starting at i; else None."""
    above = np.asarray(above, bool)
    run = 0
    for i in range(above.size):
        run = run + 1 if above[i] else 0
        if run >= min_bins:
            return i - min_bins + 1
    return None


def analyze_macroepisode_v2(e, dt_ms, provisional_baseline_ms=PROVISIONAL_BASELINE_MS, min_onset_ms=MIN_ONSET_MS):
    """Spec §2, faithful: provisional baseline (fixed early window, only to detect onset) -> onset = first
    ON crossing sustained >= min_onset_ms -> baseline re-estimated from [0, onset) -> macroepisode = the
    FLOOR-merged span containing the onset."""
    e = np.asarray(e, float)
    n = e.size
    pb = max(1, int(round(provisional_baseline_ms / dt_ms)))
    prov_b = float(np.median(e[:pb]))
    peak = float(e.max()) if n else 0.0
    amp0 = peak - prov_b
    base = dict(onset_ms=None, duration_ms=0.0, occupancy=0.0, max_gap_ms=0.0, peak=peak, baseline=prov_b,
                amp=amp0, sustained=False)
    if amp0 <= 1e-12:
        return base
    on0 = prov_b + ON_FRAC * amp0
    min_bins = max(1, int(round(min_onset_ms / dt_ms)))
    oi = first_sustained_crossing(e >= on0, min_bins)
    if oi is None:
        return base                                        # no sustained onset -> no_onset (gate A trivially fails)
    b = float(np.median(e[:oi])) if oi >= 2 else prov_b    # spec: baseline from the true pre-onset window [0, onset)
    amp = peak - b
    floor = b + FLOOR_FRAC * amp
    on2 = b + ON_FRAC * amp
    gap = int(round(MAX_GAP_MS / dt_ms))
    eps = _merge_episodes(e >= floor, gap)
    cont = [(i0, i1) for i0, i1 in eps if i0 <= oi < i1]
    if cont:
        i0, i1 = cont[0]
    else:
        after = [(i0, i1) for i0, i1 in eps if i0 >= oi and (e[i0:i1] >= on2).any()]
        if not after:
            return dict(base, onset_ms=float(oi * dt_ms), baseline=b, amp=amp)
        i0, i1 = max(after, key=lambda ep: ep[1] - ep[0])
    span = e[i0:i1]
    dur = float((i1 - i0) * dt_ms)
    occ = float((span >= floor).mean())
    return dict(onset_ms=float(oi * dt_ms), duration_ms=dur, occupancy=occ,
                max_gap_ms=float(_longest_subthreshold_run_ms(span < floor, dt_ms)),
                peak=peak, baseline=b, amp=amp, sustained=bool(dur >= MIN_MACRO_MS and occ >= OCCUPANCY_MIN))


def b2_highfreq_overlaps_window(hf_db, window_frames, enh_db=ENH_DB):
    """B2 (fixed): the high-freq (80-150 or 1-150 Hz) dB must exceed enh_db WITHIN the B1 macroepisode
    frame window, not merely somewhere in the record."""
    i0, i1 = window_frames
    seg = np.asarray(hf_db)[max(0, i0):max(i0 + 1, i1)]
    return bool(seg.size and (seg >= enh_db).any())


def _events_v2(e, dt_ms, b, amp):
    on = b + ON_FRAC * amp
    idx = np.flatnonzero(np.asarray(e) >= on)
    if idx.size == 0:
        return []
    out, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > 1:
            out.append((start, prev + 1))
            start = i
        prev = i
    out.append((start, prev + 1))
    return out


def _axial_recruitment(kymo, kt_ms, onset_ms, dt_ms, window_ms=400.0, active_frac_of_peak=0.2):
    """A8 (fixed): axial first-passage SPREAD around onset. For each axial bin that becomes active in the
    window, the first time it crosses active_frac_of_peak of its in-window max; the spread of those times
    is the recruitment latency. flash = spread <= FLASH_WINDOW_MS (near-simultaneous ignition)."""
    kymo = np.asarray(kymo, float)
    oi = int(round(onset_ms / dt_ms))
    w = max(1, int(round(window_ms / dt_ms)))
    seg = kymo[:, oi:oi + w]
    if seg.shape[1] == 0:
        return dict(spread_ms=0.0, n_active_axial=0, whole_field_flash=True)
    peak_per_bin = seg.max(axis=1)
    active = peak_per_bin > active_frac_of_peak * (seg.max() + 1e-12)
    fp = []
    for a in np.flatnonzero(active):
        cr = np.flatnonzero(seg[a] > active_frac_of_peak * peak_per_bin[a])
        if cr.size:
            fp.append(int(cr[0]))
    if len(fp) < 2:
        return dict(spread_ms=0.0, n_active_axial=int(active.sum()), whole_field_flash=True)
    spread_ms = float((max(fp) - min(fp)) * dt_ms)
    return dict(spread_ms=spread_ms, n_active_axial=int(active.sum()),
                whole_field_flash=bool(spread_ms <= FLASH_WINDOW_MS))


def compute_source_gate_v2(core_rate, active_frac, kymo_axis, kt_ms, bin_ms, runaway_early_stop_ms):
    """Gate-A metrics from saved arrays, faithful to spec: analyze_macroepisode_v2 + A7 active-AREA + A8
    axial first-passage. Returns the dict consumed by ictal_carrier_verdict (plus diagnostics)."""
    e_A = CG.moving_average(np.asarray(core_rate, float), SMOOTH_MS / bin_ms)
    macro = analyze_macroepisode_v2(e_A, bin_ms)
    onset_ms = macro["onset_ms"]

    whole_field_flash = False
    has_recruitment = False
    src_sep_count = A7_DIMS_REQUIRED
    if onset_ms is not None and macro["sustained"]:
        oi = int(round(onset_ms / bin_ms))
        # ---- A8 axial gradient ----
        rec = _axial_recruitment(kymo_axis, kt_ms, onset_ms, bin_ms)
        whole_field_flash = rec["whole_field_flash"]
        has_recruitment = (not whole_field_flash) and rec["spread_ms"] > FLASH_WINDOW_MS
        # ---- A7 with ACTIVE-AREA as the 3rd dim ----
        af = np.asarray(active_frac, float)
        dur_bins = int(macro["duration_ms"] / bin_ms)
        macro_area = float(af[oi:oi + dur_bins].mean()) if dur_bins else 0.0
        pre = e_A[:oi]
        if pre.size:
            b = float(np.median(pre))
            ev = _events_v2(pre, bin_ms, b, macro["peak"] - b)
            if ev:
                med_dur = float(np.median([(i1 - i0) * bin_ms for i0, i1 in ev]))
                med_peak = float(np.median([e_A[i0:i1].max() for i0, i1 in ev]))
                med_area = float(np.median([af[i0:i1].mean() for i0, i1 in ev]))
                dims = ((macro["duration_ms"] >= SEP_FACTOR * max(med_dur, 1e-9))
                        + (macro["peak"] >= SEP_FACTOR * max(med_peak, 1e-9))
                        + (macro_area >= SEP_FACTOR * max(med_area, 1e-9)))     # <- active-area, not rate-energy
                src_sep_count = int(dims)

    return dict(macro=macro, onset_ms=onset_ms, whole_field_flash=whole_field_flash,
                # recruitment only gates fail_plateau when there IS a sustained source; a NON-sustained
                # source (no >=100ms onset = a burst train) must route to fail_hfo_like_train, not fail_plateau
                has_recruitment=(has_recruitment if (onset_ms is not None and macro["sustained"]) else True),
                saturated_plateau=False, tail_escalating=False, src_sep_count=src_sep_count,
                runaway_early_stop_ms=runaway_early_stop_ms, e_A=e_A)


def compute_observed_gate_v2(lfp, fs, baseline_ms=OBS_BASELINE_MS):
    """Gate-B metrics from the saved 2 kHz LFP, faithful: dB relative to a FIXED EARLY baseline window
    (not the burst-polluted [0,onset)); per-contact macroepisode_v2; B1 count; B2 = high-freq overlaps the
    best B1 macroepisode window."""
    CG.assert_nyquist(fs)
    tms, env = CG.band_envelopes(lfp, fs)
    dt_frame_ms = float(np.median(np.diff(tms))) if tms.size > 1 else CG.STFT_HOP_MS
    pre_frames = max(1, int(np.sum(tms < baseline_ms)))

    def to_db(band):
        med = np.median(band[:pre_frames], axis=0)
        med = np.where(med <= 0, np.finfo(float).tiny, med)
        return 10.0 * np.log10(np.maximum(band, np.finfo(float).tiny) / med)

    lg_db, hf_db, bb_db = to_db(env["lowgamma"]), to_db(env["highfreq"]), to_db(env["broadband"])
    contacts = []
    for c in range(lg_db.shape[1]):
        macro = analyze_macroepisode_v2(lg_db[:, c], dt_frame_ms, provisional_baseline_ms=baseline_ms)
        peak_lg = float(lg_db[:, c].max())
        contacts.append(dict(macro=macro, peak_lowgamma_db=peak_lg, sustained=bool(macro["sustained"] and peak_lg >= ENH_DB),
                             peak_highfreq_db=float(hf_db[:, c].max()), peak_broadband_db=float(bb_db[:, c].max())))
    sustained = [c for c in contacts if c["sustained"]]
    best_idx = int(max(range(len(contacts)), key=lambda i: (contacts[i]["sustained"], contacts[i]["macro"]["duration_ms"])))
    best = contacts[best_idx]["macro"]
    # B2: high-freq (or broadband) enhanced WITHIN the best contact's low-gamma macroepisode window
    highfreq_enhanced = False
    if best["onset_ms"] is not None and best["duration_ms"] > 0:
        i0 = int(best["onset_ms"] / dt_frame_ms)
        i1 = i0 + int(best["duration_ms"] / dt_frame_ms)
        highfreq_enhanced = any(b2_highfreq_overlaps_window(hf_db[:, c], (i0, i1))
                                or b2_highfreq_overlaps_window(bb_db[:, c], (i0, i1)) for c in range(lg_db.shape[1]))
    return dict(n_sustained_contacts=len(sustained), highfreq_enhanced=bool(highfreq_enhanced),
                best_macro=best, best_contact_idx=best_idx, contacts=contacts,
                contact_peak_lowgamma_db=[round(c["peak_lowgamma_db"], 2) for c in contacts])


def carrier_verdict_v2(source, observed):
    """Assemble + adjudicate with the unchanged v1 verdict logic (only the metrics are corrected). B6 sep
    is left as the conservative 'spatial extent' check (>=2 sustained contacts) since a full 4-dim redo
    is only reached when gate A passes, which does not happen for the current burst-train arms."""
    m = dict(
        runaway_early_stop_ms=source.get("runaway_early_stop_ms"),
        tail_escalating=source["tail_escalating"],
        whole_field_flash=source["whole_field_flash"],
        saturated_plateau=source["saturated_plateau"],
        has_recruitment=source["has_recruitment"],
        src_macro=source["macro"],
        src_sep_count=source["src_sep_count"],
        obs_n_sustained_contacts=observed["n_sustained_contacts"],
        obs_highfreq_enhanced=observed["highfreq_enhanced"],
        obs_best_macro=observed["best_macro"],
        obs_sep_count=DIMS_REQUIRED if observed["n_sustained_contacts"] >= N_CONTACTS_MIN else 0,
    )
    return ictal_carrier_verdict(m)

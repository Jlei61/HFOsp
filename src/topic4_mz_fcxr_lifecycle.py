"""FCXR-LC1 — dynamic slow-feedback lifecycle classifier (pure logic; TDD on synthetic window sequences).

Question (load-bearing): does ONE continuous simulation, driven only by dynamic slow variables (Z + the
asymmetric X relay), traverse
    statistical interictal  ->  bounded ictal-like bout  ->  autonomous termination  ->  statistical interictal
without any kick, hand reset, spliced window, or sustained external drive?

The classifier consumes a run reduced to an ORDERED list of fixed-size analysis windows (win_ms each), each
carrying observables RELATIVE to the E1 baseline statistical band, and returns one lifecycle label. It never
touches the simulation — the reducer that builds `windows` from a raw run lives in the runner.

Anti-cheat (design §3.3 L5, the single most important clause): recovery REQUIRES returning events whose stats
fall back into the baseline band. A post-ictal window with event_rate==0 is SILENT, never INTERICTAL, so an
entirely-silent tail classifies as PERMANENT_SILENCE and can NEVER be reported as RECOVERED. Occupancy dropping
below the band is necessary but NOT sufficient.

Design: docs/superpowers/plans/2026-07-22-topic4-mz-fcxr-lc1.md §3.
"""
from __future__ import annotations

import numpy as np

# Overall lifecycle labels (>=10, design §3.1).
LIFECYCLE_LABELS = (
    "INTERICTAL_BASELINE", "DENSE_EVENT_TRAIN", "ICTAL_LIKE_BOUNDED", "TERMINATED_REFRACTORY",
    "RECOVERED_INTERICTAL", "PERMANENT_SILENCE", "RAPID_RELAPSE", "RUNAWAY", "NUMERICAL_UNSAFE", "UNRESOLVED",
)
# Per-window regimes.
REGIMES = ("UNSAFE", "ICTAL", "DENSE", "INTERICTAL", "SILENT", "OTHER")

LC_THRESHOLDS = dict(
    HIGH_OCC=0.5,          # rolling-rate occupancy above the interictal band upper edge -> ictal-like
    ELEVATED_OCC=0.10,     # above this occupancy = more active than interictal (dense event train)
    PRE_MS=8000.0,         # required interictal BEFORE the bout (isolated bursts tolerated)
    ICTAL_MS=1000.0,       # a bout must be >= this to count as ictal-like (not a single event)
    POST_MS=8000.0,        # required returning interictal AFTER termination (+ any leading silence)
    RELAPSE_GUARD_MS=2000.0,  # ictal re-entry within this of termination -> rapid relapse
    DENSE_MIN_MS=2000.0,   # a DENSE_EVENT_TRAIN needs a SUSTAINED dense run (isolated bursts don't count)
    MIN_INTERICTAL_WINDOWS=3,  # minimum interictal windows to call a no-bout run an interictal baseline
)


# ------------------------------------------------------------------ per-window regime
def window_regime(w, band, T=LC_THRESHOLDS):
    """Regime of one analysis window RELATIVE to the baseline band.

    ICTAL   = sustained above-band occupancy AND recruitment beyond the baseline P90 (spatial spread).
    DENSE   = above the interictal occupancy floor but not ictal-like (elevated event train).
    SILENT  = no events (event_rate<=0) and low occupancy -> a silent gap, NOT interictal.
    INTERICTAL = low occupancy AND returning events whose rate is inside the accepted band.
    OTHER   = low occupancy but event rate outside the band (e.g., a sub-band trickle) -> not a return.
    """
    if bool(w.get("numerical_unsafe", False)):
        return "UNSAFE"
    occ = float(w["occ"]); er = float(w["event_rate_hz"]); rec = float(w.get("recruit_frac", 0.0))
    if occ >= T["HIGH_OCC"] and rec > float(band["recruit_p90"]):
        return "ICTAL"
    if er <= 0.0 and occ < T["ELEVATED_OCC"]:
        return "SILENT"
    if occ >= T["ELEVATED_OCC"]:
        return "DENSE"
    if float(band["event_rate_lo"]) <= er <= float(band["event_rate_hi"]):
        return "INTERICTAL"
    return "OTHER"


# ------------------------------------------------------------------ sequence helpers
def _first_ictal_bout(regimes, win_ms, ictal_ms):
    """First maximal contiguous ICTAL run whose duration >= ictal_ms -> (start, end) inclusive, else None."""
    n_need = max(1, int(round(ictal_ms / win_ms)))
    i = 0
    while i < len(regimes):
        if regimes[i] == "ICTAL":
            j = i
            while j < len(regimes) and regimes[j] == "ICTAL":
                j += 1
            if (j - i) >= n_need:
                return (i, j - 1)
            i = j
        else:
            i += 1
    return None


def _trailing_run_ms(regimes, target, win_ms):
    c = 0
    for r in reversed(regimes):
        if r == target:
            c += 1
        else:
            break
    return c * win_ms


def _leading_run_ms(regimes, target, win_ms):
    c = 0
    for r in regimes:
        if r == target:
            c += 1
        else:
            break
    return c * win_ms


def _relapse_within(regimes, start_idx, win_ms, guard_ms):
    n_guard = max(1, int(round(guard_ms / win_ms)))
    for k in range(start_idx, min(len(regimes), start_idx + n_guard)):
        if regimes[k] == "ICTAL":
            return True
    return False


def _max_run(regimes, target):
    """Longest contiguous run of `target`."""
    best = cur = 0
    for r in regimes:
        cur = cur + 1 if r == target else 0
        best = max(best, cur)
    return best


def _smooth_isolated(regimes):
    """Relabel an ISOLATED single non-ictal, non-interictal window (a lone DENSE/SILENT/OTHER burst or gap
    surrounded by interictal) to INTERICTAL, so a normal interictal run with occasional bursts is not
    shattered. ICTAL windows are NEVER smoothed -> real bouts and ictal-strength flashes survive; a run of
    >=2 consecutive non-interictal windows (a genuine dense train / real silence) is preserved."""
    out = list(regimes)
    for i in range(1, len(out) - 1):
        if regimes[i] not in ("ICTAL", "INTERICTAL") and regimes[i - 1] == "INTERICTAL" and regimes[i + 1] == "INTERICTAL":
            out[i] = "INTERICTAL"
    return out


def _res(label, regimes, *, bout=None, reasons=None, **extra):
    d = dict(label=label, regimes=list(regimes), bout=bout, reasons=list(reasons or []))
    d.update(extra)
    return d


# ------------------------------------------------------------------ lifecycle classifier
def classify_lifecycle(windows, band, *, runaway=False, T=LC_THRESHOLDS):
    """Classify one continuous run reduced to ordered `windows` (each win_ms = band['win_ms']).

    Priority: RUNAWAY / NUMERICAL_UNSAFE first (safety). Then locate the first ictal-like bout; the overall
    label follows from pre-ictal interictal duration, autonomous termination, relapse, and whether the
    post-ictal segment carries RETURNING interictal events (never satisfiable by a silent tail — L5 anti-cheat).
    """
    win_ms = float(band["win_ms"])
    regimes = [window_regime(w, band, T) for w in windows]
    if runaway:
        return _res("RUNAWAY", regimes, reasons=["global runaway flag set"])
    if "UNSAFE" in regimes:
        return _res("NUMERICAL_UNSAFE", regimes, reasons=["a window failed the numerical-safety gate"])
    sm = _smooth_isolated(regimes)   # isolated single non-ictal bursts -> interictal (ICTAL never smoothed)

    bout = _first_ictal_bout(sm, win_ms, T["ICTAL_MS"])
    if bout is None:
        if "ICTAL" in sm:                                     # ictal-strength flash(es) below the bout duration
            return _res("DENSE_EVENT_TRAIN", regimes, reasons=["ictal-strength flash below the >=ICTAL_MS bout"])
        if _max_run(sm, "DENSE") * win_ms >= T["DENSE_MIN_MS"]:
            return _res("DENSE_EVENT_TRAIN", regimes, reasons=[f">={T['DENSE_MIN_MS']:.0f}ms sustained dense-event train"])
        n_int = sm.count("INTERICTAL")
        if n_int >= T["MIN_INTERICTAL_WINDOWS"] and n_int >= 0.5 * len(sm):
            return _res("INTERICTAL_BASELINE", regimes, reasons=["interictal-dominant (isolated bursts tolerated)"])
        if n_int > 0:
            return _res("INTERICTAL_BASELINE", regimes, reasons=["sparse but interictal (no bout, no sustained dense)"])
        return _res("UNRESOLVED", regimes, reasons=["no interictal windows and no ictal bout"])

    b0, b1 = bout
    pre_ms = _trailing_run_ms(sm[:b0], "INTERICTAL", win_ms)          # interictal run just before bout (smoothed)
    bout_ms = (b1 - b0 + 1) * win_ms

    terminated = (b1 + 1 < len(sm)) and (sm[b1 + 1] != "ICTAL")
    if not terminated:
        return _res("ICTAL_LIKE_BOUNDED", regimes, bout=bout, pre_ms=pre_ms, bout_ms=bout_ms,
                    reasons=["ictal bout runs to the end of the record; no autonomous termination observed"])

    if _relapse_within(sm, b1 + 1, win_ms, T["RELAPSE_GUARD_MS"]):
        return _res("RAPID_RELAPSE", regimes, bout=bout, pre_ms=pre_ms, bout_ms=bout_ms,
                    reasons=["ictal re-entry within the relapse guard window"])

    post = sm[b1 + 1:]
    if post and all(r == "SILENT" for r in post):                    # L5 anti-cheat: entirely silent tail
        return _res("PERMANENT_SILENCE", regimes, bout=bout, pre_ms=pre_ms, bout_ms=bout_ms, post_return_ms=0.0,
                    reasons=["post-ictal segment is entirely silent (no returning events) -> not a recovery"])
    k = 0                                                            # allow a leading post-ictal silence
    while k < len(post) and post[k] == "SILENT":
        k += 1
    post_return_ms = _leading_run_ms(post[k:], "INTERICTAL", win_ms)

    if post_return_ms >= T["POST_MS"] and pre_ms >= T["PRE_MS"]:
        return _res("RECOVERED_INTERICTAL", regimes, bout=bout, pre_ms=pre_ms, bout_ms=bout_ms,
                    post_return_ms=post_return_ms,
                    reasons=[f"{pre_ms:.0f}ms pre-ictal interictal -> {bout_ms:.0f}ms bounded bout -> "
                             f"{post_return_ms:.0f}ms returning interictal events"])
    if pre_ms < T["PRE_MS"]:
        return _res("UNRESOLVED", regimes, bout=bout, pre_ms=pre_ms, bout_ms=bout_ms, post_return_ms=post_return_ms,
                    reasons=[f"insufficient pre-ictal interictal ({pre_ms:.0f}ms < {T['PRE_MS']:.0f}ms)"])
    return _res("TERMINATED_REFRACTORY", regimes, bout=bout, pre_ms=pre_ms, bout_ms=bout_ms,
                post_return_ms=post_return_ms,
                reasons=[f"terminated but returning interictal only {post_return_ms:.0f}ms "
                         f"(< {T['POST_MS']:.0f}ms) -> no statistical return"])


# ------------------------------------------------------------------ run -> ordered analysis windows (reducer)
def build_windows(rate, dt_ms, af, af_bin_ms, roll_hi, events, win_ms, *,
                  roll_ms=300.0, start_ms=0.0, event_lookback_ms=None, finite=True):
    """Reduce ONE continuous run to the ordered window list the classifier consumes. PURE over arrays so it
    is testable without a simulation; the runner detects `events` with the SAME frozen slow-off bar used to
    build the E1 baseline band (apples-to-apples).

    Per window (win_ms wide):
      occ           = fraction of the roll_ms rolling-mean rate above the interictal band edge roll_hi
                      (sustained-duration signal, fine time resolution).
      recruit_frac  = PEAK active-fraction in the window (spatial recruitment / spread; `af` is the codebase
                      participation measure). Compared against the baseline P90 to flag ictal-like spread.
      event_rate_hz = returning-event onsets in a TRAILING window of event_lookback_ms (default win_ms),
                      per second. The trailing estimate keeps sparse-but-normal interictal from reading as
                      SILENT just because a single fine window happened to contain no discrete event.
      numerical_unsafe = window rate non-finite (or the global finite flag).
    events: list of dicts with 't_on' (ms). af: per-af_bin active fraction; af_bin_ms its bin width."""
    rate = np.asarray(rate, float); af = np.asarray(af, float)
    if rate.size == 0:
        return []
    lookback = float(win_ms if event_lookback_ms is None else event_lookback_ms)
    w = max(1, int(round(roll_ms / dt_ms)))
    roll = np.convolve(rate, np.ones(w) / float(w), mode="same")
    above = (roll > float(roll_hi)).astype(float)
    n_per = max(1, int(round(win_ms / dt_ms)))
    a0 = max(0, int(round(start_ms / dt_ms)))
    ev_on = np.array([float(e["t_on"]) for e in events], float) if events else np.zeros(0)
    windows = []
    t = a0
    while t + n_per <= rate.size:
        seg = rate[t:t + n_per]
        w0, w1 = t * dt_ms, (t + n_per) * dt_ms
        af0, af1 = int(round(w0 / af_bin_ms)), int(round(w1 / af_bin_ms))
        seg_af = af[af0:af1]
        lb0 = w1 - lookback
        n_ev = int(((ev_on >= lb0) & (ev_on < w1)).sum()) if ev_on.size else 0
        windows.append(dict(
            occ=float(above[t:t + n_per].mean()),
            recruit_frac=float(seg_af.max()) if seg_af.size else 0.0,
            event_rate_hz=n_ev / (lookback / 1000.0),
            numerical_unsafe=(not (finite and bool(np.all(np.isfinite(seg))))),
            t0_ms=float(w0), t1_ms=float(w1), n_events_lookback=n_ev,
        ))
        t += n_per
    return windows


# ------------------------------------------------------------------ slow-variable phase-portrait coordinates
def depletion_coordinate(v, p_weights):
    """Weighted mean depletion D = sum_i p_i (1 - v_i) / sum_i p_i, for a per-neuron slow variable v (z or x)
    and onset-depletion weights p_i. D=0 when fully available (v==1); rises toward 1 as v depletes."""
    v = np.asarray(v, float); p = np.asarray(p_weights, float)
    if v.shape != p.shape:
        raise ValueError(f"v and p_weights must match: {v.shape} vs {p.shape}")
    s = float(p.sum())
    if not (s > 0):
        raise ValueError("p_weights must have positive sum")
    return float(np.sum(p * (1.0 - v)) / s)

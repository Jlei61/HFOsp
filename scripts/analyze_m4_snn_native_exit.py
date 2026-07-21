#!/usr/bin/env python
"""Phase 1/2 analysis for the SNN-native M4 containment-to-exit line (task brief §4-§5).

Phase 1 (form-then-terminate): the recovery current must be tested on an ALREADY-FORMED bounded state,
NOT engaged at an assumed 2500ms. `formed_state_time` reads the no-p M4 anchor's traces and returns the
earliest time the bounded state is stably formed (ALL of: rate sustained elevated, S_G containment engaged,
q_I depleted toward its floor, spatial extent established -- held continuously for >= window_ms). That time
becomes persist_onset_ms for the form-then-terminate arms.

`classify_phase1_verdict` then labels each intervention arm: invalid (state never formed) / termination-no-go
(fragment/rebound/runaway after formation) / termination-only (clean offset, no returning IEDs) /
lifecycle-candidate (clean offset AND matched returning IEDs).

Pure functions here are unit-tested; the CLI (below) drives them on real anchor/intervention artifacts.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _smooth(x, dt, win_ms=200.0):
    x = np.asarray(x, float)
    n = max(1, int(round(win_ms / dt)))
    if x.size < 3:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def _at(arr, t_ms, dt_arr):
    if arr.size == 0:
        return np.nan
    return float(arr[min(arr.size - 1, max(0, int(t_ms / dt_arr)))])


def formed_state_time(rate_hz, trace_SG, trace_qI_mean, area_frames, dt, movie_bin_ms,
                      window_ms=1500.0, probe_ms=100.0,
                      rate_frac=0.5, sg_frac=0.7, q_frac=0.3, area_frac=0.6,
                      bounded_rate_hz=15.0, bounded_sg=0.1, bounded_qfloor=0.6):
    """Earliest t (ms) after which the M4 bounded state is stably FORMED for >= window_ms.

    Plateau references = median over the LAST THIRD of the run (assumed settled). The end-of-run must
    itself be a bounded state (rate_plateau > bounded_rate_hz AND S_G_plateau > bounded_sg AND
    q_floor < bounded_qfloor) -- otherwise there is nothing formed and t_form is None (this is what makes
    the detector reject a run that never left baseline; the fractional criteria alone would be vacuous).

    At each probe the state is "in the formed regime" iff ALL of:
      rate_s >= rate_frac*rate_plateau ; S_G >= sg_frac*S_G_plateau ;
      q_I_mean <= q_floor + q_frac*(1-q_floor) ; active_area >= area_frac*area_plateau .
    t_form = first probe whose next window_ms of probes are ALL in the formed regime."""
    rate_s = _smooth(rate_hz, dt, 200.0)
    SG = np.asarray(trace_SG, float)
    qI = np.asarray(trace_qI_mean, float)
    area = np.asarray(area_frames, float)
    n = rate_s.size
    if n == 0:
        return dict(t_form=None, reason="empty rate trace")
    T = n * dt
    w0 = int(0.66 * n)
    rate_plat = float(np.median(rate_s[w0:]))
    sg_plat = float(np.median(SG[w0:])) if SG.size else 0.0
    q_floor = float(np.median(qI[w0:])) if qI.size else 1.0
    area_plat = float(np.median(area[int(0.66 * area.size):])) if area.size else 0.0

    bounded = (rate_plat > bounded_rate_hz and sg_plat > bounded_sg and q_floor < bounded_qfloor)
    diag = dict(rate_plateau=round(rate_plat, 2), sg_plateau=round(sg_plat, 3),
                q_floor=round(q_floor, 3), area_plateau=round(area_plat, 3),
                window_ms=window_ms, end_state_is_bounded=bool(bounded))
    if not bounded:
        return dict(t_form=None, reason="end-of-run is not a bounded state", **diag)

    probes = np.arange(0.0, T, probe_ms)
    in_regime = np.array([
        (_at(rate_s, t, dt) >= rate_frac * rate_plat)
        and (True if not (SG.size and sg_plat > 0) else _at(SG, t, dt) >= sg_frac * sg_plat)
        and (True if qI.size == 0 else _at(qI, t, dt) <= q_floor + q_frac * (1.0 - q_floor))
        and (True if not (area.size and area_plat > 0) else _at(area, t, movie_bin_ms) >= area_frac * area_plat)
        for t in probes], bool)
    k = int(round(window_ms / probe_ms))
    t_form = None
    for i in range(len(probes) - k):
        if in_regime[i:i + k + 1].all():
            t_form = float(probes[i])
            break
    return dict(t_form=t_form, reason=("stable window found" if t_form is not None
                                       else "no continuous formed window >= window_ms"), **diag)


def classify_phase1_verdict(termination_class, n_pre_events, n_post_events, recovered_events,
                            state_formed=True):
    """Map an intervention arm to a Phase-1 verdict (task brief §4). recovered_events = post-offset events
    whose duration/cadence/extent match the pre-onset IED distribution (assessed upstream; here a count)."""
    if not state_formed:
        return "invalid"
    if termination_class in ("runaway", "fragment", "persist") or termination_class.startswith("reignite"):
        return "termination-no-go"
    if recovered_events and recovered_events > 0:
        return "lifecycle-candidate"
    return "termination-only"


# ---------------------------------------------------------------------------
def _load_arm(out_dir, tag, seed, label):
    """Return (row, npz-dict-or-None) for one arm from its per-arm files (Phase-0 layout)."""
    ad = os.path.join(out_dir, "per_arm", f"{tag}_seed{seed}")
    jp = os.path.join(ad, f"{label}.json")
    npzp = os.path.join(ad, f"{label}.npz")
    row = json.load(open(jp)) if os.path.exists(jp) else None
    z = dict(np.load(npzp)) if os.path.exists(npzp) else None
    return row, z


def analyze_anchor(out_dir, tag, seed, label="B_m4_anchor", movie_bin_ms=25.0, **kw):
    """Run formed_state_time on an anchor arm's traces. Returns the formed-state dict + provenance."""
    row, z = _load_arm(out_dir, tag, seed, label)
    if z is None:
        raise SystemExit(f"anchor npz not found for {label} in {out_dir}/per_arm/{tag}_seed{seed}")
    movie = z.get("movie")
    area = ((movie > 0.1).mean(axis=(1, 2)) if movie is not None and movie.size else np.zeros(0))
    res = formed_state_time(z["rate"], z.get("trace_SG", np.zeros(0)), z["trace_qI_mean"], area,
                            dt=0.1, movie_bin_ms=movie_bin_ms, **kw)
    res["label"] = label
    res["max_rate_hz"] = row.get("max_rate_hz") if row else None
    return res


def main():
    ap = argparse.ArgumentParser(description="SNN-native M4 exit Phase-1/2 analysis")
    ap.add_argument("--out-dir", required=True, help="run --out dir (contains per_arm/<tag>_seed<seed>/)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--anchor-label", default="B_m4_anchor")
    ap.add_argument("--window-ms", type=float, default=1500.0)
    a = ap.parse_args()
    res = analyze_anchor(a.out_dir, a.tag, a.seed, label=a.anchor_label, window_ms=a.window_ms)
    print(json.dumps(res, indent=2))
    if res["t_form"] is not None:
        print(f"\n[formed-state] t_form = {res['t_form']:.0f} ms  -> use --persist-onset-ms {int(res['t_form'])} "
              f"for the form-then-terminate arms", flush=True)
    else:
        print(f"\n[formed-state] NO stable formed window ({res['reason']}) -> "
              f"anchor did not form a stable bounded state; do not schedule an onset.", flush=True)


if __name__ == "__main__":
    main()

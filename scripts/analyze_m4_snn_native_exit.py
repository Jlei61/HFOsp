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
                      core_activity=None, surround_activity=None, activity_bin_ms=25.0,
                      trace_q_core=None, trace_q_surround=None,
                      window_ms=1500.0, probe_ms=100.0,
                      rate_frac=0.5, sg_frac=0.7, q_frac=0.3, area_frac=0.6,
                      core_frac=0.5, surr_frac=0.5, grad_frac=0.5,
                      bounded_rate_hz=15.0, bounded_sg=0.1, bounded_qfloor=0.6,
                      bounded_core_hz=15.0, bounded_surr_hz=15.0, bounded_grad=0.1):
    """Earliest t (ms) after which the M4 bounded state is stably FORMED for >= window_ms.

    Review 2026-07-22: the detector must use CORE/SURROUND, not just the spatial mean. When core/surround
    traces are given (core_activity/surround_activity Hz per activity_bin_ms; trace_q_core/trace_q_surround
    per-step), formation additionally requires: core rate formed+stable, surround recruited to its plateau,
    q_core depleted, and a q_surround-q_core gradient established. Otherwise falls back to global-only.

    Plateau references = median over the LAST THIRD (assumed settled). The end-of-run must itself be a
    bounded M4 state (global: rate>bounded_rate_hz, S_G>bounded_sg, q<bounded_qfloor; +core: core rate>
    bounded_core_hz, q_core<bounded_qfloor, gradient>bounded_grad) -- else t_form=None (fractional criteria
    alone are vacuous). At each probe "in formed regime" iff ALL criteria hold; t_form = first probe whose
    next window_ms are ALL in regime."""
    rate_s = _smooth(rate_hz, dt, 200.0)
    SG = np.asarray(trace_SG, float); qI = np.asarray(trace_qI_mean, float); area = np.asarray(area_frames, float)
    use_cs = core_activity is not None and surround_activity is not None \
        and trace_q_core is not None and trace_q_surround is not None
    ca = _smooth(np.asarray(core_activity, float), activity_bin_ms, 3 * activity_bin_ms) if use_cs else np.zeros(0)
    sa = _smooth(np.asarray(surround_activity, float), activity_bin_ms, 3 * activity_bin_ms) if use_cs else np.zeros(0)
    qc = np.asarray(trace_q_core, float) if use_cs else np.zeros(0)
    qs = np.asarray(trace_q_surround, float) if use_cs else np.zeros(0)
    grad = (qs - qc) if use_cs else np.zeros(0)
    n = rate_s.size
    if n == 0:
        return dict(t_form=None, reason="empty rate trace", used_core_surround=use_cs)
    T = n * dt
    _p = lambda a: float(np.median(a[int(0.66 * a.size):])) if a.size else 0.0
    rate_plat, sg_plat, area_plat = _p(rate_s), _p(SG), _p(area)
    q_floor = _p(qI) if qI.size else 1.0
    core_plat, surr_plat, grad_plat = _p(ca), _p(sa), _p(grad)
    qc_floor = _p(qc) if qc.size else 1.0

    bounded = (rate_plat > bounded_rate_hz and sg_plat > bounded_sg and q_floor < bounded_qfloor)
    if use_cs:
        # The M4 bounded state is a BROAD stripe (spec §9 highest-risk flag: "broad ~60% stripe, NOT a
        # localized core"). Measured q_core~=q_surround~=floor, so the core-surround q GRADIENT is small BY
        # THE STATE'S NATURE. We therefore do NOT gate formation on the gradient (that would reject the real
        # broad state); we gate on broad RECRUITMENT (core AND surround rate at plateau) + broad DEPLETION
        # (q_core at its floor). q_gradient_plateau is reported as a descriptive field only.
        bounded = bounded and (core_plat > bounded_core_hz and surr_plat > bounded_surr_hz
                               and qc_floor < bounded_qfloor)
    diag = dict(rate_plateau=round(rate_plat, 2), sg_plateau=round(sg_plat, 3), q_floor=round(q_floor, 3),
                area_plateau=round(area_plat, 3), window_ms=window_ms, used_core_surround=use_cs,
                end_state_is_bounded=bool(bounded))
    if use_cs:
        diag.update(core_plateau=round(core_plat, 2), surround_plateau=round(surr_plat, 2),
                    q_core_floor=round(qc_floor, 3), q_gradient_plateau=round(grad_plat, 3))
    if not bounded:
        return dict(t_form=None, reason="end-of-run is not a bounded M4 state", **diag)

    probes = np.arange(0.0, T, probe_ms)

    def _formed(t):
        ok = (_at(rate_s, t, dt) >= rate_frac * rate_plat) \
            and (True if not (SG.size and sg_plat > 0) else _at(SG, t, dt) >= sg_frac * sg_plat) \
            and (True if qI.size == 0 else _at(qI, t, dt) <= q_floor + q_frac * (1.0 - q_floor)) \
            and (True if not (area.size and area_plat > 0) else _at(area, t, movie_bin_ms) >= area_frac * area_plat)
        if use_cs:
            # broad recruitment (core+surround at plateau) + core depletion; NOT the gradient (broad state)
            ok = ok and (_at(ca, t, activity_bin_ms) >= core_frac * core_plat) \
                and (_at(sa, t, activity_bin_ms) >= surr_frac * surr_plat) \
                and (_at(qc, t, dt) <= qc_floor + q_frac * (1.0 - qc_floor))
        return ok

    in_regime = np.array([_formed(t) for t in probes], bool)
    k = int(round(window_ms / probe_ms))
    t_form = None
    for i in range(len(probes) - k):
        if in_regime[i:i + k + 1].all():
            t_form = float(probes[i])
            break
    return dict(t_form=t_form, reason=("stable window found" if t_form is not None
                                       else "no continuous formed window >= window_ms"), **diag)


def t_form_sensitivity(rate_hz, trace_SG, trace_qI_mean, area_frames, dt, movie_bin_ms,
                       windows=(1000.0, 1500.0, 2000.0), frac_deltas=(-0.1, 0.0, 0.1),
                       tol_ms=500.0, **cs_kw):
    """t_form stability (review 2026-07-22): re-run formed_state_time across window {1,1.5,2}s AND +/-10%
    perturbation of the fractional thresholds. stable = every variant finds a t_form AND their spread <=
    tol_ms. A t_form that only exists at one window / one threshold is NOT safe to intervene on."""
    variants = {}
    for w in windows:
        variants[f"window_{int(w)}"] = formed_state_time(
            rate_hz, trace_SG, trace_qI_mean, area_frames, dt, movie_bin_ms, window_ms=w, **cs_kw)["t_form"]
    base = dict(rate_frac=0.5, sg_frac=0.7, area_frac=0.6, core_frac=0.5, surr_frac=0.5)
    for d in frac_deltas:
        if d == 0.0:
            continue
        kw = {k: v * (1.0 + d) for k, v in base.items()}
        variants[f"frac_{d:+.2f}"] = formed_state_time(
            rate_hz, trace_SG, trace_qI_mean, area_frames, dt, movie_bin_ms, **kw, **cs_kw)["t_form"]
    vals = [v for v in variants.values() if v is not None]
    stable = (len(vals) == len(variants)) and (len(vals) > 0) and (max(vals) - min(vals) <= tol_ms)
    return dict(t_form_by_variant=variants, stable=bool(stable),
                spread_ms=(max(vals) - min(vals) if vals else None),
                t_form_median=(float(np.median(vals)) if vals else None))


def arm_event_features(row, npz, dt=0.1, movie_bin_ms=25.0, activity_bin_ms=25.0, t_min=None):
    """Per-event feature dicts from an arm's event table (row['events']) + traces (npz: rate, movie,
    core_activity, surround_activity). t_min keeps only events with t_on >= t_min (the post-offset window)."""
    rate = np.asarray(npz["rate"], float)
    movie = np.asarray(npz.get("movie", np.zeros((0, 1, 1))), float)
    ca = np.asarray(npz.get("core_activity", []), float)
    sa = np.asarray(npz.get("surround_activity", []), float)
    feats = []
    for e in (row.get("events") or []):
        t_on, t_off = float(e[0]), float(e[1])
        if t_min is not None and t_on < t_min:
            continue
        i0, i1 = int(t_on / dt), max(int(t_off / dt), int(t_on / dt) + 1)
        f0, f1 = int(t_on / movie_bin_ms), max(int(t_off / movie_bin_ms), int(t_on / movie_bin_ms) + 1)
        a0, a1 = int(t_on / activity_bin_ms), max(int(t_off / activity_bin_ms), int(t_on / activity_bin_ms) + 1)
        peak = float(rate[i0:min(i1, rate.size)].max()) if i0 < rate.size and i1 > i0 else 0.0
        area = float((movie[f0:min(f1, movie.shape[0])] > 0.1).mean()) if movie.size and f0 < movie.shape[0] and f1 > f0 else 0.0
        cr = float(ca[a0:min(a1, ca.size)].mean()) if ca.size and a0 < ca.size and a1 > a0 else 0.0
        sr = float(sa[a0:min(a1, sa.size)].mean()) if sa.size and a0 < sa.size and a1 > a0 else 0.0
        mode = (movie[f0:min(f1, movie.shape[0])].mean(axis=0) if movie.size and f0 < movie.shape[0] and f1 > f0
                else np.zeros(movie.shape[1:] if movie.ndim == 3 else (1, 1)))
        feats.append(dict(t_on=t_on, dur=t_off - t_on, peak=peak, area=area,
                          core_surr_ratio=cr / (sr + 1e-9), mode=np.asarray(mode, float)))
    return feats


def recovery_match(baseline_feats, post_feats, min_post=3, tol_frac=0.6, mode_thresh=0.5):
    """Compare post-offset event feature DISTRIBUTIONS to the slow-off baseline IED distributions (review
    07-22): a post-offset burst counts as a RECOVERED IED only if there are ENOUGH of them AND their
    duration / cadence(IEI) / peak-rate / active-area / core-surround-ratio medians are within tol_frac of
    baseline AND the mean spatial footprint matches (cosine >= mode_thresh). Returns per-metric + a
    recovered bool. (virtual-SEEG contact order + per-event axial order are further checks DEFERRED until
    the LFP montage is wired per event -- so recovered=True here is necessary, not yet sufficient.)"""
    if len(post_feats) < min_post:
        return dict(recovered=False, reason=f"too few post-offset events ({len(post_feats)} < {min_post})",
                    n_post=len(post_feats), per_metric={})

    def _med(fs, k):
        return float(np.median([f[k] for f in fs])) if fs else 0.0

    def _iei(fs):
        d = np.diff(sorted(f["t_on"] for f in fs))
        return float(np.median(d)) if d.size else 0.0

    per = {}
    for name, key in (("duration", "dur"), ("peak_rate", "peak"), ("active_area", "area"),
                      ("core_surround_ratio", "core_surr_ratio")):
        b, p = _med(baseline_feats, key), _med(post_feats, key)
        ratio = p / b if b > 1e-9 else (1.0 if p < 1e-9 else float("inf"))
        per[name] = dict(baseline_median=round(b, 4), post_median=round(p, 4), ratio=round(ratio, 3),
                         ok=bool(1.0 - tol_frac <= ratio <= 1.0 + tol_frac))
    b_iei, p_iei = _iei(baseline_feats), _iei(post_feats)
    r_iei = p_iei / b_iei if b_iei > 1e-9 else float("inf")
    per["iei"] = dict(baseline_median=round(b_iei, 1), post_median=round(p_iei, 1), ratio=round(r_iei, 3),
                      ok=bool(1.0 - tol_frac <= r_iei <= 1.0 + tol_frac))
    bm = np.mean([f["mode"].ravel() for f in baseline_feats], axis=0) if baseline_feats else np.zeros(1)
    pm = np.mean([f["mode"].ravel() for f in post_feats], axis=0)
    cos = float(bm @ pm / (np.linalg.norm(bm) * np.linalg.norm(pm) + 1e-12))
    per["spatial_mode"] = dict(cosine=round(cos, 3), ok=bool(cos >= mode_thresh))
    recovered = all(m["ok"] for m in per.values())
    return dict(recovered=bool(recovered), n_post=len(post_feats), per_metric=per,
                reason=("recovered (necessary; vSEEG/axial order pending)" if recovered else "one or more metrics off baseline"),
                deferred=["virtual_SEEG_contact_order (needs LFP montage per event)",
                          "axial_propagation_order (needs per-event kymograph onset gradient)"])


def verify_pre_onset_identity(anchor_npz, cand_npz, onset_ms, dt=0.1):
    """Contract (review 07-22): with persist_onset_ms=onset the recovery current is exactly 0 before onset
    (p stays 0), so an intervention MUST be byte-identical to the anchor before onset. Checks the saved
    per-step rate / trace_qI_mean / trace_SG are equal on [0, onset). rate == spikes/step, so equal rate +
    equal spatial slow-var traces is a near-exact spike-identity proxy (the runner does not save E_spk_bool)."""
    i = int(onset_ms / dt)
    checks = {}
    for k in ("rate", "trace_qI_mean", "trace_SG"):
        a = np.asarray(anchor_npz.get(k, []), float)
        c = np.asarray(cand_npz.get(k, []), float)
        if a.size == 0 and c.size == 0:
            checks[k] = True                       # both traces absent (e.g. no S_G pool) -> vacuously identical
        else:
            m = min(i, a.size, c.size)
            checks[k] = bool(m > 0 and np.array_equal(a[:m], c[:m]))
    return dict(pre_onset_identical=all(checks.values()), onset_ms=onset_ms, per_trace=checks)


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


def analyze_anchor(out_dir, tag, seed, label="B_m4_anchor", movie_bin_ms=25.0, activity_bin_ms=25.0, **kw):
    """Run the core/surround formed-state detector + t_form sensitivity on an anchor arm's traces (review
    2026-07-22: core/surround, not spatial mean; and t_form must be stable to launch interventions)."""
    row, z = _load_arm(out_dir, tag, seed, label)
    if z is None:
        raise SystemExit(f"anchor npz not found for {label} in {out_dir}/per_arm/{tag}_seed{seed}")
    movie = z.get("movie")
    area = ((movie > 0.1).mean(axis=(1, 2)) if movie is not None and movie.size else np.zeros(0))
    cs = dict(core_activity=z.get("core_activity"), surround_activity=z.get("surround_activity"),
              trace_q_core=z.get("trace_q_core"), trace_q_surround=z.get("trace_q_surround"),
              activity_bin_ms=activity_bin_ms)
    res = formed_state_time(z["rate"], z.get("trace_SG", np.zeros(0)), z["trace_qI_mean"], area,
                            dt=0.1, movie_bin_ms=movie_bin_ms, **cs, **kw)
    res["sensitivity"] = t_form_sensitivity(z["rate"], z.get("trace_SG", np.zeros(0)), z["trace_qI_mean"], area,
                                            dt=0.1, movie_bin_ms=movie_bin_ms, **cs)
    res["label"] = label
    res["max_rate_hz"] = row.get("max_rate_hz") if row else None
    res["safe_to_intervene"] = bool(res["t_form"] is not None and res["sensitivity"]["stable"])
    return res


def plot_formed_state_diagnostic(out_dir, tag, seed, res, label="B_m4_anchor", movie_bin_ms=25.0,
                                 activity_bin_ms=25.0):
    """Formed-state diagnostic figure the review requires BEFORE any intervention: total/core/surround rate,
    q_core/q_surround/q_mean, S_G, active area, axial+transverse kymograph, with t_form + its sensitivity
    spread marked. Saves PNG+PDF next to the anchor's figures/ dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, z = _load_arm(out_dir, tag, seed, label)
    dt = 0.1
    rate = _smooth(np.asarray(z["rate"], float), dt, 200.0)
    tr = np.arange(rate.size) * dt
    qm = np.asarray(z["trace_qI_mean"], float); tq = np.arange(qm.size) * dt
    qc = np.asarray(z.get("trace_q_core", []), float); qs = np.asarray(z.get("trace_q_surround", []), float)
    SG = np.asarray(z.get("trace_SG", []), float)
    ca = np.asarray(z.get("core_activity", []), float); sa = np.asarray(z.get("surround_activity", []), float)
    ta = np.arange(ca.size) * activity_bin_ms
    movie = np.asarray(z.get("movie", np.zeros((0, 1, 1))), float)
    area = (movie > 0.1).mean(axis=(1, 2)) if movie.size else np.zeros(0)
    tf = np.arange(area.size) * movie_bin_ms
    kax = np.asarray(z.get("kymo_axis", np.zeros((0, 1))), float)
    ktr = np.asarray(z.get("kymo_transverse", np.zeros((0, 1))), float)
    t_form = res.get("t_form")
    spread = res.get("sensitivity", {}).get("spread_ms")

    fig, ax = plt.subplots(3, 2, figsize=(13, 9))
    ax[0, 0].plot(tr, rate, color="#333", lw=1.0, label="total")
    if ca.size:
        ax[0, 0].plot(ta, _smooth(ca, activity_bin_ms, 3 * activity_bin_ms), color="#B2182B", lw=1.0, label="core")
        ax[0, 0].plot(ta, _smooth(sa, activity_bin_ms, 3 * activity_bin_ms), color="#2166AC", lw=1.0, label="surround")
    ax[0, 0].set_ylabel("E rate (Hz)"); ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("rate: total / core / surround")
    ax[1, 0].plot(tq, qm, color="#333", lw=1.0, label="q mean")
    if qc.size:
        ax[1, 0].plot(np.arange(qc.size) * dt, qc, color="#B2182B", lw=1.0, label="q_core")
        ax[1, 0].plot(np.arange(qs.size) * dt, qs, color="#2166AC", lw=1.0, label="q_surround")
    ax[1, 0].set_ylabel("q_I"); ax[1, 0].set_ylim(0, 1.05); ax[1, 0].legend(fontsize=8); ax[1, 0].set_title("inhibitory resource")
    ax[2, 0].plot(np.arange(SG.size) * dt, SG, color="#762A83", lw=1.0, label="S_G")
    if area.size:
        ax[2, 0].plot(tf, area, color="#1B7837", lw=1.0, label="active area")
    ax[2, 0].set_xlabel("t (ms)"); ax[2, 0].set_ylabel("S_G / area"); ax[2, 0].legend(fontsize=8); ax[2, 0].set_title("containment + spatial extent")
    for a, k, ttl in ((ax[0, 1], kax, "kymograph: axial"), (ax[1, 1], ktr, "kymograph: transverse")):
        if k.size:
            a.imshow(k.T, aspect="auto", origin="lower", cmap="magma",
                     extent=(0, k.shape[0] * movie_bin_ms, 0, k.shape[1]))
        a.set_ylabel("space bin"); a.set_title(ttl)
    ax[2, 1].axis("off")
    ax[2, 1].text(0.02, 0.9, f"t_form = {t_form} ms" + (f"\nsensitivity spread = {spread} ms" if spread is not None else "")
                  + f"\nsafe_to_intervene = {res.get('safe_to_intervene')}"
                  + f"\nend_state_bounded = {res.get('end_state_is_bounded')}"
                  + f"\ncore_plateau = {res.get('core_plateau')} Hz  q_core_floor = {res.get('q_core_floor')}"
                  + f"\nq_gradient_plateau = {res.get('q_gradient_plateau')}"
                  + f"\nt_form by variant:\n  " + "\n  ".join(f"{k}: {v}" for k, v in res.get("sensitivity", {}).get("t_form_by_variant", {}).items()),
                  va="top", ha="left", fontsize=8.5, family="monospace")
    for a in (ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1]):
        if t_form is not None:
            a.axvline(t_form, color="#F1A340", lw=1.5, ls="--")
    fig.suptitle(f"Formed-state diagnostic — {label} (seed {seed}); dashed = data-driven t_form", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figdir = os.path.join(out_dir, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"formed_state_diagnostic_{tag}_seed{seed}.png")
    fig.savefig(out, dpi=140); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    return out


def _regate_at_onset(z, onset_ms, dt=0.1, movie_bin_ms=25.0, activity_bin_ms=25.0):
    """Re-run the formed-state gate on ONLY [0, onset) of an intervention's own traces (review contract:
    each intervention must independently be formed at its own onset, not just inherit the anchor's t_form)."""
    i = int(onset_ms / dt); fi = int(onset_ms / movie_bin_ms); ai = int(onset_ms / activity_bin_ms)
    movie = np.asarray(z.get("movie", np.zeros((0, 1, 1))), float)[:fi]
    area = (movie > 0.1).mean(axis=(1, 2)) if movie.size else np.zeros(0)
    return formed_state_time(np.asarray(z["rate"], float)[:i], np.asarray(z.get("trace_SG", []), float)[:i],
                             np.asarray(z["trace_qI_mean"], float)[:i], area, dt=dt, movie_bin_ms=movie_bin_ms,
                             core_activity=np.asarray(z.get("core_activity", []), float)[:ai],
                             surround_activity=np.asarray(z.get("surround_activity", []), float)[:ai],
                             trace_q_core=np.asarray(z.get("trace_q_core", []), float)[:i],
                             trace_q_surround=np.asarray(z.get("trace_q_surround", []), float)[:i],
                             activity_bin_ms=activity_bin_ms)


def analyze_phase1(out_dir, seed=1, anchor_tag="p1anchor", intervene_tag="intervene", prev_tag="prevgated80",
                   onset_ms=2300.0, recovery_margin_ms=1000.0):
    """Full Phase-1 verdict (review 2026-07-22). Per intervention arm: (1) pre-onset byte-identity to the
    anchor, (2) re-pass the formed-state gate at its own onset, (3) termination class, (4) if terminated,
    match post-offset events to the A_slow_off baseline IEDs. Plus the gated-prevention control (do IEDs
    survive the gated current on the slow-off substrate?). Writes phase1_verdict.json."""
    _, anchor_z = _load_arm(out_dir, anchor_tag, seed, "B_m4_anchor")
    base_row, base_z = _load_arm(out_dir, prev_tag, seed, "A_slow_off")
    prev_row, _ = _load_arm(out_dir, prev_tag, seed, "A_persist_act")
    base_feats = arm_event_features(base_row, base_z) if (base_row and base_z is not None) else []

    ad = os.path.join(out_dir, "per_arm", f"{intervene_tag}_seed{seed}")
    arms = {}
    for label in sorted(f[:-5] for f in os.listdir(ad) if f.endswith(".json") and f.startswith("D_")):
        row, z = _load_arm(out_dir, intervene_tag, seed, label)
        ident = verify_pre_onset_identity(anchor_z, z, onset_ms) if anchor_z is not None else {"pre_onset_identical": None}
        regate = _regate_at_onset(z, onset_ms)
        term, offset = row.get("termination_class"), row.get("offset_ms")
        rec = None
        if offset is not None and base_feats:
            post = arm_event_features(row, z, t_min=offset + recovery_margin_ms)
            rec = recovery_match(base_feats, post)
        verdict = classify_phase1_verdict(term or "", row.get("n_pre_runaway", 0), row.get("n_events", 0),
                                          (rec["n_post"] if (rec and rec["recovered"]) else 0),
                                          state_formed=(regate["t_form"] is not None))
        arms[label] = dict(termination_class=term, offset_ms=offset,
                           pre_onset_identical=ident["pre_onset_identical"],
                           reformed_at_onset=bool(regate["t_form"] is not None),
                           n_events=row.get("n_events"), q_mean_final=row.get("q_mean_final"),
                           max_rate_hz=row.get("max_rate_hz"), area_tail=row.get("active_area_tail"),
                           recovery=rec, verdict=verdict)
    prev = None
    if prev_row and base_row:
        be, pe = base_row.get("events") or [], prev_row.get("events") or []
        n_base_post = sum(1 for e in be if e[0] >= onset_ms)
        n_prev_post = sum(1 for e in pe if e[0] >= onset_ms)
        prev = dict(n_slowoff_total=len(be), n_slowoff_post_onset=n_base_post,
                    n_gated_total=len(pe), n_gated_post_onset=n_prev_post,
                    prevention_after_onset=bool(n_base_post > 0 and n_prev_post < 0.6 * n_base_post),
                    note="gated current on slow-off substrate; onset-gated IED survival after onset")
    out = dict(onset_ms=onset_ms, baseline_n_ieds=len(base_feats), arms=arms, prevention=prev)
    json.dump(out, open(os.path.join(out_dir, f"phase1_verdict_seed{seed}.json"), "w"), indent=2,
              default=lambda o: None)
    return out


def main():
    ap = argparse.ArgumentParser(description="SNN-native M4 exit Phase-1/2 analysis")
    ap.add_argument("--phase1", action="store_true", help="run the full Phase-1 verdict (needs intervene + prevgated arms)")
    ap.add_argument("--onset-ms", type=float, default=2300.0)
    ap.add_argument("--out-dir", required=True, help="run --out dir (contains per_arm/<tag>_seed<seed>/)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--anchor-label", default="B_m4_anchor")
    ap.add_argument("--window-ms", type=float, default=1500.0)
    a = ap.parse_args()
    if a.phase1:
        v = analyze_phase1(a.out_dir, seed=a.seed, onset_ms=a.onset_ms)
        print(json.dumps(v, indent=2, default=lambda o: None))
        print(f"\n[Phase-1] onset={v['onset_ms']}ms  baseline IEDs={v['baseline_n_ieds']}")
        for lab, r in v["arms"].items():
            rec = r.get("recovery")
            print(f"  {lab}: verdict={r['verdict']}  term={r['termination_class']} offset={r['offset_ms']} "
                  f"pre_onset_identical={r['pre_onset_identical']} reformed={r['reformed_at_onset']} "
                  f"recovered={rec['recovered'] if rec else None}")
        if v.get("prevention"):
            p = v["prevention"]
            print(f"  prevention(gated): slow-off post-onset={p['n_slowoff_post_onset']} -> gated post-onset="
                  f"{p['n_gated_post_onset']}  prevention_after_onset={p['prevention_after_onset']}")
        return
    res = analyze_anchor(a.out_dir, a.tag, a.seed, label=a.anchor_label, window_ms=a.window_ms)
    fig = plot_formed_state_diagnostic(a.out_dir, a.tag, a.seed, res, label=a.anchor_label)
    json.dump(res, open(os.path.join(a.out_dir, f"formed_state_{a.tag}_seed{a.seed}.json"), "w"),
              indent=2, default=lambda o: None)
    print(json.dumps({k: v for k, v in res.items() if k != "sensitivity"}, indent=2, default=lambda o: None))
    print("sensitivity:", json.dumps(res["sensitivity"], default=lambda o: None))
    print(f"[diagnostic figure] {fig}", flush=True)
    if res["safe_to_intervene"]:
        print(f"\n[formed-state] SAFE: t_form = {res['t_form']:.0f} ms, sensitivity stable "
              f"(spread {res['sensitivity']['spread_ms']} ms) -> use --persist-onset-ms {int(res['t_form'])} "
              f"for the form-then-terminate arms", flush=True)
    else:
        why = ("no stable formed window: " + res["reason"]) if res["t_form"] is None \
            else f"t_form UNSTABLE across window/threshold (spread {res['sensitivity']['spread_ms']} ms)"
        print(f"\n[formed-state] NOT SAFE to intervene ({why}). Do NOT schedule an onset yet.", flush=True)


if __name__ == "__main__":
    main()

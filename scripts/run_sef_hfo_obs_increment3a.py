"""Increment-3a rate-model parity: full four-control runner (2026-06-07 user spec).

Four controls, each isolating one confound:
  C1 connectivity-axis tracking: fixed electrode set, θ_EE ∈ {0,45,90}°,
     kick at the -θ_EE end. Verdict: ALL three n_part≥7 AND axis_err<25°.
  C2 kick-position does NOT drag direction: θ_EE fixed 45°, kick offset
     perpendicularly by -0.35, 0, +0.35 × L/2. Verdict: all positions → ~45°.
  C3 electrode placement does NOT drag direction: θ_EE fixed 45°, kick
     fixed, rotate entire 3-shaft montage 0°,30°,60°,90°. CANNOT be replaced
     by C-track. Verdict: all rotations → ~45°.
  C4 no direction in isotropic connectivity: ell_par=ell_perp (AR=1), kick at
     CENTER. Verdict: n_part≥7 REQUIRED first (cannot accept fizzle as "honest
     negative"); then readability<TAU_FAIL=0.3 OR axis is None.

Locked verdict bars (same as Increment-2, NEVER tuned):
  AXIS_ERR_MAX=25°, KDIR=3, PART_MIN=7, TAU_FAIL=0.3, timing_frac=0.5.

Participation margin = 0.1*(env_max-env_min). The rate field has a spatial
amplitude gradient along the propagating wave (near-kick contacts are ~3-10×
stronger than wavefront contacts), unlike the SNN spike envelope. Using 50%
(SNN convention) cuts off late/front contacts; 10% correctly includes them.
This is a substrate-specific design choice, NOT a change to the verdict bars.

Kick at the NEGATIVE θ_EE end: prevents periodic-BC wrap that occurs when the
kick is at +end and the front reaches > L/2 before decaying.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.sef_hfo_lif import (mean_field, integrate_lif_field, _grid, DETECT,   # noqa: E402
                             ELL_PAR, ELL_PERP, L_INH)
from src.sef_hfo_rate_adapter import pulse_stim_fn, rate_event_envelope        # noqa: E402
from src.sef_hfo_observation import (build_shaft, merge_montages, grid_coords, # noqa: E402
                                     extract_lagpat, attach_geometry,
                                     endpoint_centroid_axis, axis_angle_error_deg,
                                     direction_readability)
from src.sef_hfo_snn_adapter import event_window_for_run                        # noqa: E402
from scripts.plot_sef_hfo_obs_readout import plot_readout_diagnostic            # noqa: E402

# ---- locked verdict bars ----
AXIS_ERR_MAX = 25.0
KDIR = 3
PART_MIN = 2 * KDIR + 1          # 7
TAU_FAIL = 0.3

# ---- locked operating point + pulse ----
RATIO = 0.6
DT, T_MAX = 0.25, 250.0
PULSE = dict(radius=2.0, amp=8.0, t_on=0.0, t_off=30.0)

# ---- participation margin (substrate-specific, NOT a verdict bar) ----
PART_MARGIN_FRAC = 0.10           # 10% of event amplitude (see module docstring)


def _montage(center, pitch, n_contacts, shafts_deg, rotation_deg=0.0):
    """Non-parallel shafts centered on `center`, with optional global rotation."""
    rot = np.deg2rad(rotation_deg)
    return merge_montages([
        build_shaft(np.deg2rad(a) + rot, pitch, n_contacts, tuple(center), chr(65 + i))
        for i, a in enumerate(shafts_deg)
    ])


def _integrate(op, theta_rad, AR, kick_xy, n, L):
    """One on+off pair; returns (on_frames, off_frames, on_ext, off_ext).

    on_ext/off_ext are the RAW FIELD active fractions (fraction of pixels with
    rE > op["nuE"]+DETECT). Used for event window detection instead of the
    contact aggregate: the aggregate is diluted by off-stripe contacts (varies with
    montage rotation) while on_ext reflects the physical field event directly.
    """
    ep = ELL_PAR if AR != 1 else float(np.sqrt(ELL_PAR * ELL_PERP))
    eperp = ELL_PERP if AR != 1 else float(np.sqrt(ELL_PAR * ELL_PERP))
    kw = dict(dt=DT, t_max=T_MAX, theta_EE=theta_rad, n=n, L=L,
              ell_par=ep, ell_perp=eperp, l_inh=L_INH, return_frames=True)
    sf  = pulse_stim_fn(tuple(kick_xy), n=n, L=L, **PULSE)
    sfr = pulse_stim_fn(tuple(kick_xy), n=n, L=L, **{**PULSE, "amp": 0.0})
    on  = integrate_lif_field(op, sf,  **kw)
    off = integrate_lif_field(op, sfr, **kw)
    thr = float(op["nuE"]) + DETECT
    on_ext  = (on[-1]  > thr).mean(axis=(1, 2)).astype(float)   # (nsteps,)
    off_ext = (off[-1] > thr).mean(axis=(1, 2)).astype(float)
    return on[-1], off[-1], on_ext, off_ext


def _read(op, theta_rad, AR, kick_xy, m, theta_ref_rad, n, L, kw_pitch,
          margin_frac=PART_MARGIN_FRAC, window_source="field", save_diag=False):
    """Run one condition; return verdict dict (+ _diag for figure generation).

    window_source / margin_frac are the event-DETECTION FRONT-END knobs. The parity
    claim is the direction estimator + thresholds (endpoint_centroid_axis, 25°, k_dir=3),
    which are byte-identical across substrates and — per the sensitivity harness —
    INVARIANT to these front-end knobs (the knobs move COVERAGE: whether a window
    exists / whether n_part≥7, not the recovered axis).
      window_source="field"  : event window from the raw field active fraction (LOCKED
          default). The kick is a model artifact with no real-data analog, so the
          physical-field event interval is the cleanest, geometry-independent temporal
          gate. Robust to montage rotation.
      window_source="contact": from the contact-aggregate (the SNN-runner convention).
          Geometry-FRAGILE (fails when a shaft lies parallel to the wave: contacts stay
          lit, the aggregate never "returns"). Kept only for the window-source
          invariance comparison.
      margin_frac : participation margin = margin_frac*(env_max-env_min) (LOCKED 0.10).
          The rate field has a 3-10x amplitude gradient along the wave; 0.10 includes
          wavefront contacts that 0.50 (SNN convention) cuts off. Sensitivity-swept."""
    on_f, off_f, on_ext, off_ext = _integrate(op, theta_rad, AR, kick_xy, n, L)
    kw = 0.5 * kw_pitch                          # kernel_width = 0.5*pitch
    env = rate_event_envelope(on_f, n, L, m, kw)
    if window_source == "contact":
        env_ref = rate_event_envelope(off_f, n, L, m, kw)
        win = event_window_for_run(env.mean(axis=0), env_ref.mean(axis=0), DT)
    else:
        win = event_window_for_run(on_ext, off_ext, DT)   # raw field active fraction
    empty = dict(n_part=0, axis_err=None, readability=None, win=None,
                 verdict="insufficient", reason="no_event_detected")
    if win is None:
        return empty
    floor  = float(env.min())
    margin = margin_frac * (float(env.max()) - floor)
    art = extract_lagpat(env, DT, [win], floor, margin, 0.5, DT)
    art = attach_geometry(art, m)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    n_part = int(b0.sum())
    eps = 0.5 * kw_pitch
    ax  = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=eps)
    rd  = direction_readability(r0, b0, art.contact_coords)
    err = (None if ax is None else
           round(float(axis_angle_error_deg(ax, theta_ref_rad)), 1))
    rd_val = (None if rd is None or rd != rd else round(float(rd), 3))
    res = dict(n_part=n_part, axis_err=err, readability=rd_val,
               win=[round(win[0], 1), round(win[1], 1)])
    if save_diag:
        # footprint: peak-activity frame
        active_t = int((on_f > float(op["nuE"]) + DETECT).any(axis=(1, 2)).nonzero()[0].max()
                       if (on_f > float(op["nuE"]) + DETECT).any() else 0)
        res["_diag"] = dict(montage=m, envelopes=env, artifact=art, on_frames=on_f,
                            win=win, kick_xy=kick_xy, theta_EE_deg=np.degrees(theta_rad),
                            sheet_L=L, peak_frame=on_f[active_t].ravel(), kw_pitch=kw_pitch,
                            recovered_axis=ax)
    return res


def _verdict(r, is_iso=False):
    """Assign verdict + reason to a read result."""
    if r["n_part"] < PART_MIN:
        return "insufficient", "too_few_contacts"
    if is_iso:
        rd = r["readability"]
        if rd is None or r["axis_err"] is None or rd < TAU_FAIL:
            return "pass", "no_spurious_direction"
        return "fail", "spurious_axis_in_iso"
    if r["axis_err"] is None:
        return "fail", "no_axis_despite_sufficient_contacts"
    if r["axis_err"] < AXIS_ERR_MAX:
        return "pass", None
    return "fail", f"axis_error_{r['axis_err']}deg"


def run_four_controls(L=24.0, n=96, pitch=4.0, n_contacts=6,
                      shafts=(15.0, 75.0, 135.0),
                      kick_end_frac=0.6, kicktrack_off=0.35,
                      shaft_rotations=(0.0, 30.0, 60.0, 90.0),
                      margin_frac=PART_MARGIN_FRAC, window_source="field",
                      save_diag=False):
    center = np.zeros(2)            # rate grid is origin-centered
    op = mean_field(RATIO)
    m_base = _montage(center, pitch, n_contacts, shafts, rotation_deg=0.0)
    half = L / 2.0
    perp45 = np.array([-np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))])
    rd = dict(margin_frac=margin_frac, window_source=window_source, save_diag=save_diag)

    cfg = dict(substrate="lif_rate_field", L=L, n=n, pitch=pitch,
               n_contacts=n_contacts, shafts=list(shafts),
               kick_end_frac=kick_end_frac, kicktrack_off=kicktrack_off,
               shaft_rotations=list(shaft_rotations), ratio=RATIO,
               part_margin_frac=margin_frac, window_source=window_source)
    res = {"config": cfg, "C1_connectivity": {}, "C2_kicktrack": {},
           "C3_shaft_invariance": {}, "C4_iso": None}

    # --- C1: connectivity-axis tracking (θ_EE ∈ {0,45,90}) ---
    for th in (0.0, 45.0, 90.0):
        # kick at NEGATIVE θ_EE end (avoids periodic-BC wrap for θ=0/90)
        end = center - kick_end_frac * half * np.array([np.cos(np.deg2rad(th)),
                                                        np.sin(np.deg2rad(th))])
        r = _read(op, np.deg2rad(th), 2.0, end, m_base, np.deg2rad(th), n, L, pitch, **rd)
        v, reason = _verdict(r)
        r.update(verdict=v, reason=reason, kick_xy=end.tolist())
        res["C1_connectivity"][f"{th:g}deg"] = r

    # --- C2: kick position does NOT drag direction (θ_EE=45°, kick offset perp) ---
    base_kick45 = center - kick_end_frac * half * np.array([np.cos(np.deg2rad(45)),
                                                             np.sin(np.deg2rad(45))])
    for j in (-kicktrack_off, 0.0, kicktrack_off):
        kxy = base_kick45 + j * half * perp45
        r = _read(op, np.deg2rad(45), 2.0, kxy, m_base, np.deg2rad(45), n, L, pitch, **rd)
        v, reason = _verdict(r)
        r.update(verdict=v, reason=reason, kick_xy=kxy.tolist())
        res["C2_kicktrack"][f"perp{j:+.2f}"] = r

    # --- C3: shaft rotation does NOT drag direction (θ_EE=45°, kick fixed) ---
    for rot in shaft_rotations:
        m_rot = _montage(center, pitch, n_contacts, shafts, rotation_deg=rot)
        r = _read(op, np.deg2rad(45), 2.0, base_kick45, m_rot, np.deg2rad(45), n, L,
                  pitch, **rd)
        v, reason = _verdict(r)
        r.update(verdict=v, reason=reason)
        res["C3_shaft_invariance"][f"rot{rot:g}deg"] = r

    # --- C4: isotropic — must first have n_part≥7, THEN check no spurious direction ---
    iso_kick = center.copy()        # kick at CENTER (removes seed-position confound)
    r = _read(op, np.deg2rad(45), 1.0, iso_kick, m_base, np.deg2rad(45), n, L, pitch, **rd)
    v, reason = _verdict(r, is_iso=True)
    r.update(verdict=v, reason=reason, kick_xy=iso_kick.tolist())
    res["C4_iso"] = r

    # --- overall verdict ---
    c1 = [res["C1_connectivity"][k]["verdict"] for k in res["C1_connectivity"]]
    c2 = [res["C2_kicktrack"][k]["verdict"]     for k in res["C2_kicktrack"]]
    c3 = [res["C3_shaft_invariance"][k]["verdict"] for k in res["C3_shaft_invariance"]]
    c4 = res["C4_iso"]["verdict"]
    res["PASS"] = (all(v == "pass" for v in c1) and
                   all(v == "pass" for v in c2) and
                   all(v == "pass" for v in c3) and
                   c4 == "pass")
    res["summary"] = dict(C1=c1, C2=c2, C3=c3, C4=c4)
    return res


def _make_overview_figure(res, out_path):
    """4-panel summary: one panel per control type."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    def _bar(ax, labels, errs, n_parts, title):
        x = np.arange(len(labels))
        cols = []
        for i, (e, n) in enumerate(zip(errs, n_parts)):
            if n < PART_MIN:
                cols.append("gray")
            elif e is None or e >= AXIS_ERR_MAX:
                cols.append("tab:red")
            else:
                cols.append("tab:green")
        bars = ax.bar(x, [(e if e is not None else 40) for e in errs],
                      color=cols, edgecolor="k", width=0.6)
        ax.axhline(AXIS_ERR_MAX, ls="--", c="k", lw=1.5, label=f"gate {AXIS_ERR_MAX}°")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.set_ylabel("axis error (°)"); ax.set_title(title)
        for xi, (e, n) in enumerate(zip(errs, n_parts)):
            ax.text(xi, (e or 40) + 0.5, f"n={n}", ha="center", fontsize=7)
        ax.set_ylim(0, 50); ax.legend(fontsize=7)

    c1 = res["C1_connectivity"]
    _bar(axes[0], list(c1.keys()),
         [c1[k]["axis_err"] for k in c1],
         [c1[k]["n_part"] for k in c1], "C1: connectivity tracking")

    c2 = res["C2_kicktrack"]
    _bar(axes[1], list(c2.keys()),
         [c2[k]["axis_err"] for k in c2],
         [c2[k]["n_part"] for k in c2], "C2: kick position")

    c3 = res["C3_shaft_invariance"]
    _bar(axes[2], list(c3.keys()),
         [c3[k]["axis_err"] for k in c3],
         [c3[k]["n_part"] for k in c3], "C3: shaft rotation")

    c4 = res["C4_iso"]
    ax4 = axes[3]
    col = "tab:green" if c4["verdict"] == "pass" else ("gray" if c4["n_part"] < PART_MIN else "tab:red")
    ax4.bar([0], [c4["readability"] or 0], color=col, edgecolor="k", width=0.5)
    ax4.axhline(TAU_FAIL, ls="--", c="k", lw=1.5, label=f"readability gate {TAU_FAIL}")
    ax4.set_xticks([0]); ax4.set_xticklabels(["iso (AR=1)"])
    ax4.set_ylabel("direction_readability"); ax4.set_title("C4: isotropic (no direction)")
    ax4.text(0, (c4["readability"] or 0) + 0.02,
             f"n_part={c4['n_part']}\nverdict={c4['verdict']}", ha="center", fontsize=8)
    ax4.set_ylim(0, 1.1); ax4.legend(fontsize=7)

    overall = "PASS" if res["PASS"] else "FAIL/INSUFFICIENT"
    fig.suptitle(f"Increment-3a rate parity — four controls  [{overall}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)


def _make_diag_figures(res, out_dir, L, pitch):
    """Per-condition diagnostic figure for any non-passing condition."""
    os.makedirs(out_dir, exist_ok=True)
    gxy = grid_coords(96, L)

    def _diag(r, fname, label):
        if "_diag" not in r:
            return
        d = r["_diag"]
        try:
            plot_readout_diagnostic(
                os.path.join(out_dir, fname),
                d["montage"], d["envelopes"], DT, d["artifact"],
                d["kick_xy"], d["theta_EE_deg"], d["recovered_axis"],
                d["sheet_L"], d["kw_pitch"],
                source_frame=d["peak_frame"], grid_xy=gxy,
                event_window=d["win"],
                title=f"{label} | n_part={r['n_part']} axis_err={r['axis_err']}° [{r['verdict']}]")
        except Exception as e:
            print(f"  diag figure {fname}: {e}", flush=True)

    for k, r in res["C1_connectivity"].items():
        if r.get("verdict") != "pass":
            _diag(r, f"diag_C1_{k}.png", f"C1 {k}")
    for k, r in res["C2_kicktrack"].items():
        if r.get("verdict") != "pass":
            _diag(r, f"diag_C2_{k}.png", f"C2 {k}")
    for k, r in res["C3_shaft_invariance"].items():
        if r.get("verdict") != "pass":
            _diag(r, f"diag_C3_{k}.png", f"C3 {k}")
    if res["C4_iso"].get("verdict") != "pass":
        _diag(res["C4_iso"], "diag_C4_iso.png", "C4 iso")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L",            type=float, default=24.0)
    ap.add_argument("--n",            type=int,   default=96)
    ap.add_argument("--pitch",        type=float, default=4.0)
    ap.add_argument("--n-contacts",   type=int,   default=6)
    ap.add_argument("--kick-end-frac",type=float, default=0.6)
    ap.add_argument("--kicktrack-off",type=float, default=0.35)
    ap.add_argument("--no-diag",      action="store_true")
    a = ap.parse_args()
    save_diag = not a.no_diag

    print(f"Increment-3a four controls: L={a.L}mm n={a.n} pitch={a.pitch}mm "
          f"n_contacts={a.n_contacts} (save_diag={save_diag})", flush=True)

    res = run_four_controls(
        L=a.L, n=a.n, pitch=a.pitch, n_contacts=a.n_contacts,
        kick_end_frac=a.kick_end_frac, kicktrack_off=a.kicktrack_off,
        save_diag=save_diag)

    out = "results/topic4_sef_hfo/observation_layer/increment3a_rate_parity"
    os.makedirs(out, exist_ok=True)

    # strip _diag (not JSON-serialisable) before saving
    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k != "_diag"}
        return d

    with open(os.path.join(out, "rate_parity_four_controls.json"), "w") as f:
        json.dump(strip(res), f, indent=2, default=lambda o: None)

    _make_overview_figure(res, os.path.join(out, "figures", "rate_parity_overview.png"))
    if save_diag:
        _make_diag_figures(res, os.path.join(out, "figures"), L=a.L, pitch=a.pitch)
        # HEADLINE §13 figure: the primary C1 θ_EE=45 read-out diagnostic (electrode
        # overlay + per-contact traces + lag/axis chain) — always emitted (the locked
        # "every SEEG-read-out claim shows the chain" discipline), even though it PASSES.
        head = res["C1_connectivity"].get("45deg", {})
        if "_diag" in head:
            d = head["_diag"]
            plot_readout_diagnostic(
                os.path.join(out, "figures", "rate_readout_headline_theta45.png"),
                d["montage"], d["envelopes"], DT, d["artifact"],
                d["kick_xy"], d["theta_EE_deg"], d["recovered_axis"],
                d["sheet_L"], d["kw_pitch"],
                source_frame=d["peak_frame"], grid_xy=grid_coords(a.n, a.L),
                event_window=d["win"],
                title=(f"Rate-field virtual-SEEG read-out (C1 θ_EE=45°, PASS) | "
                       f"firing-rate-density envelope (NOT LFP) | "
                       f"n_part={head['n_part']} axis_err={head['axis_err']}°"))

    print(f"\n=== OVERALL {'PASS' if res['PASS'] else 'NOT PASS'} ===", flush=True)
    print("C1:", {k: (v["n_part"], v["axis_err"], v["verdict"])
                  for k, v in res["C1_connectivity"].items()})
    print("C2:", {k: (v["n_part"], v["axis_err"], v["verdict"])
                  for k, v in res["C2_kicktrack"].items()})
    print("C3:", {k: (v["n_part"], v["axis_err"], v["verdict"])
                  for k, v in res["C3_shaft_invariance"].items()})
    print("C4:", res["C4_iso"]["n_part"], res["C4_iso"]["readability"],
          res["C4_iso"]["verdict"])
    print("wrote", os.path.join(out, "rate_parity_four_controls.json"), flush=True)


if __name__ == "__main__":
    main()

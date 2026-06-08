"""Two-electrode read-out figures for the SEF-HFO model section (user 2026-06-07).

One paper-grade figure PER MODEL (drawn separately, not a combined overview):
a virtual electrode laid ALONG the E->E connectivity long axis vs one laid
ACROSS it, on the same propagating event, plus THE SIGNAL EACH CONTACT RECORDS
over time (panel B/C, stacked traces in the style of inc2_cm_currentLFP.png).

An electrode ∥ the connectivity axis is crossed by the travelling event in
sequence -> the per-contact peak sweeps (slanted peak locus = reads direction);
an electrode ⊥ to it is crossed ~together -> peaks aligned (vertical locus).

Reuse, not reinvention:
  * figure    -> src.sef_hfo_plot.two_electrode_readout (house style, shared, ONE function)
  * electrodes-> src.sef_hfo_observation.build_shaft
  * rate read -> src.sef_hfo_rate_adapter.rate_event_envelope (per-contact firing-rate-density)
  * rate field-> scripts.run_sef_hfo_obs_increment3a._integrate (the C1 θ_EE=45° recipe)
  * SNN signal-> results/.../two_electrode_snn_raw.npz (formal current-LFP from an igniting
                 3 mm end-kicked sim; produced by run_sef_hfo_obs_two_electrode_snn_lfp.py)

Substrate-specific signal (honest): rate field is mean-field -> firing-rate-density
envelope (NOT a current); SNN -> formal current-based LFP (|I_E|+|I_I|).
Scale honesty (user LFP lock): rate uses REAL 4 mm contacts; SNN ignites only on a 3 mm
sheet -> SCALED sub-mm contacts (mechanism illustration), current-LFP validated near 2 mm.

Run: PYTHONPATH="$PWD" python scripts/plot_sef_hfo_two_electrode_readout.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.sef_hfo_lif import mean_field, DETECT                              # noqa: E402
from src.sef_hfo_rate_adapter import rate_event_envelope                    # noqa: E402
from src.sef_hfo_observation import (build_shaft, grid_coords,              # noqa: E402
                                     extract_lagpat, attach_geometry)
from src.sef_hfo_snn_adapter import event_window_for_run                    # noqa: E402
from src.sef_hfo_plot import two_electrode_readout                          # noqa: E402
from scripts.run_sef_hfo_obs_increment3a import _integrate, RATIO, DT       # noqa: E402

OUT = "results/topic4_sef_hfo/observation_layer/figures"
SNN_RAW = "results/topic4_sef_hfo/observation_layer/two_electrode_snn_raw.npz"
AXIS_DEG = 45.0
COS, SIN = np.cos(np.deg2rad(AXIS_DEG)), np.sin(np.deg2rad(AXIS_DEG))
U = np.array([COS, SIN])
UPERP = np.array([-SIN, COS])
PT_PAR = "electrode ∥ axis — peaks sweep (reads direction)"
PT_PERP = "electrode ⊥ axis — peaks aligned (no direction)"


def _shaft(angle_deg, pitch, n, origin, prefix):
    return build_shaft(np.deg2rad(angle_deg), pitch, n, tuple(origin), prefix)


def _arclength(montage, u_shaft):
    c = np.asarray(montage.contacts, float)
    return (c - c.mean(0)) @ u_shaft


# ===========================================================================
# RATE field — real 4 mm contacts; firing-rate-density signal (NOT LFP)
# ===========================================================================
def _rate_onset_map(frames, thr):
    crossed = frames > thr
    first = np.argmax(crossed, axis=0).astype(float)
    return np.where(crossed.any(axis=0), first * DT, np.nan).ravel()


def _rate_electrode(frames, n, L, montage, win, u_shaft, panel_title):
    env = rate_event_envelope(frames, n, L, montage, 0.5 * 4.0)        # (n_contact, nsteps)
    floor = float(env.min()); margin = 0.10 * (float(env.max()) - floor)
    art = attach_geometry(extract_lagpat(env, DT, [win], floor, margin, 0.5, DT), montage)
    return dict(contacts=montage.contacts, part=art.bools[:, 0], names=list(montage.names),
                s=_arclength(montage, u_shaft), signal=env, panel_title=panel_title)


def build_rate(n=96, L=24.0, pitch=4.0, par_n=6, perp_n=5,
               par_center=(1.5, 1.5), perp_center=(1.5, 1.5)):
    op = mean_field(RATIO)
    K = -0.6 * (L / 2.0) * U
    frames, _off, on_ext, off_ext = _integrate(op, np.deg2rad(AXIS_DEG), 2.0, K, n, L)
    win = event_window_for_run(on_ext, off_ext, DT)
    if win is None:
        raise RuntimeError("rate field produced no event window")
    onset = _rate_onset_map(frames, float(op["nuE"]) + DETECT)
    onset = onset - np.nanmin(onset)          # relative time: 0 = first activation after the kick
    par = _rate_electrode(frames, n, L, _shaft(AXIS_DEG, pitch, par_n, par_center, "P"),
                          win, U, PT_PAR)
    perp = _rate_electrode(frames, n, L, _shaft(AXIS_DEG + 90, pitch, perp_n, perp_center, "Q"),
                           win, UPERP, PT_PERP)
    t = np.arange(frames.shape[0]) * DT
    half = L / 2.0
    two_electrode_readout(
        os.path.join(OUT, "two_electrode_readout_rate.png"),
        field_xy=grid_coords(n, L), field_c=onset, field_clabel="time after event onset (ms)",
        kick_xy=K, axis_deg=AXIS_DEG, extent=(-half, half, -half, half),
        par=par, perp=perp, t=t, event_window=win,
        signal_ylabel="firing-rate-density (NOT LFP)",
        substrate_label="LIF rate field  ·  24 mm  ·  end-seed",
        contact_note="contact spacing 4.0 mm (real SEEG scale) · mean-field signal = firing-rate-density, NOT a current",
        title="Virtual electrode records the propagating event — LIF rate field")
    print(f"  rate: ∥ part={int(par['part'].sum())}/{len(par['part'])} "
          f"⊥ part={int(perp['part'].sum())}/{len(perp['part'])} win={win}")


# ===========================================================================
# SNN — 3 mm igniting sheet; formal current-LFP traces (scaled contacts)
# ===========================================================================
def _snn_electrode(lfp, contacts, names, t, win, u_shaft, panel_title):
    """lfp: (n_contact, nt) for this shaft. part = peak clearly above pre-event baseline."""
    pre = t < win[0]; inwin = (t >= win[0]) & (t <= win[1])
    base = lfp[:, pre].mean(axis=1); bstd = lfp[:, pre].std(axis=1) + 1e-9
    pz = (lfp[:, inwin].max(axis=1) - base) / bstd
    part = pz > 30.0
    c = np.asarray(contacts, float)
    return dict(contacts=c, part=part, names=list(names), s=(c - c.mean(0)) @ u_shaft,
                signal=lfp, panel_title=panel_title)


def build_snn():
    z = np.load(SNN_RAW, allow_pickle=True)
    lfp = np.asarray(z["on_lfp"], float).T          # (n_contact, nt)
    t = np.asarray(z["times"], float)
    contacts = np.asarray(z["contacts"], float)
    ncp, ncq = int(z["nc_par"]), int(z["nc_perp"])
    names = [str(s) for s in np.asarray(z["names"])]
    L = float(z["L"]); K = np.asarray(z["end"], float); TK = float(z["T_KICK"])
    win = (TK + 5.0, TK + 55.0)                     # event lives ~T_KICK+30..+50 ms
    par = _snn_electrode(lfp[:ncp], contacts[:ncp], names[:ncp], t, win, U, PT_PAR)
    perp = _snn_electrode(lfp[ncp:ncp + ncq], contacts[ncp:ncp + ncq], names[ncp:ncp + ncq],
                          t, win, UPERP, PT_PERP)
    field_c = np.asarray(z["onset_E"], float) if "onset_E" in z.files else None
    if field_c is not None:
        field_c = field_c - np.nanmin(field_c)   # relative time: 0 = first spike after the kick
    posE = np.asarray(z["posE"], float)
    two_electrode_readout(
        os.path.join(OUT, "two_electrode_readout_snn.png"),
        field_xy=posE, field_c=field_c,
        field_clabel=("time after event onset (ms)" if field_c is not None else None),
        E_xy=(None if field_c is not None else posE), name_fs=10, label_endpoints_only=True,
        kick_xy=K, axis_deg=AXIS_DEG, extent=(0.0, L, 0.0, L),
        par=par, perp=perp, t=t, event_window=win,
        signal_ylabel="current-LFP (|I_E|+|I_I|)",
        substrate_label="Spiking network  ·  3 mm  ·  end-seed",
        contact_note=("contacts SCALED to model sheet (~0.26 mm) — NOT real SEEG spacing; "
                      "formal current-LFP (validated near 2 mm); event nucleates centrally, "
                      "electrodes sample the down-axis front"),
        title="Virtual electrode records the propagating event — spiking network")
    print(f"  snn:  ∥ part={int(par['part'].sum())}/{ncp} ⊥ part={int(perp['part'].sum())}/{ncq}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[two-electrode readout] rate field ...", flush=True)
    build_rate()
    if os.path.exists(SNN_RAW):
        print("[two-electrode readout] spiking network ...", flush=True)
        build_snn()
    else:
        print(f"[two-electrode readout] SNN raw not found ({SNN_RAW}); run "
              "run_sef_hfo_obs_two_electrode_snn_lfp.py first", flush=True)


if __name__ == "__main__":
    main()

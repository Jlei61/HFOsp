"""Seizure read-out via virtual SEEG electrodes — ictal twin of two_electrode_readout_rate.png.

Same 2-D LIF rate field and virtual electrode geometry (45° axis, real 4 mm contacts).
Seizure dynamics: adaptive threshold φ(x,t) + sAHP, near-threshold regime (w_ee_mult=1.7).
Focal drive: persistent low-amplitude SOZ stimulation (represents the chronic epileptic focus).

Scientific message
------------------
The propagation DIRECTION is preserved across seizure bursts: the ∥ electrode's per-contact
peak locus (first burst) is slanted → direction-readable, same 45° axis as the interictal.
Burst rhythm (period ≈ 480 ms) visible in the full trace stack.

Excitable-regime capability test, NOT the data-locked SEF-HFO operating point.

Run: PYTHONPATH="$PWD" python scripts/plot_sef_hfo_ictal_readout.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.getcwd())
from src.sef_hfo_lif import mean_field, integrate_lif_field, DETECT           # noqa: E402
from src.sef_hfo_rate_adapter import rate_event_envelope                       # noqa: E402
from src.sef_hfo_observation import (build_shaft, grid_coords,                 # noqa: E402
                                     extract_lagpat, attach_geometry)
from src.sef_hfo_plot import two_electrode_readout                              # noqa: E402
from src.sef_hfo_field import _grid                                             # noqa: E402
from scripts.run_sef_hfo_obs_increment3a import RATIO                          # noqa: E402

OUT = "results/topic4_sef_hfo/observation_layer/figures"

# ---- operating point: ratio=1.0 (threshold-balanced) base; near-threshold gain ----
# Same strategy as 1D model: build op at threshold-balanced ratio, run at near-ictal gain.
W_EE_IKT = 1.7
_OP_BASE  = mean_field(1.0)
OP_IKT    = dict(_OP_BASE, w_ee_mult=W_EE_IKT)

# ---- grid & connectivity axis ----
N_IKT    = 64
L_IKT    = 24.0           # mm — same physical domain as interictal figure
DT_IKT   = 0.5            # ms
T_MAX_IKT = 5000.0        # ms — covers ~10 bursts at ≈480 ms period
AXIS_DEG = 45.0

COS, SIN = np.cos(np.deg2rad(AXIS_DEG)), np.sin(np.deg2rad(AXIS_DEG))
U     = np.array([COS, SIN])
UPERP = np.array([-SIN, COS])

# ---- seed position (same convention as interictal: kick at -45° end) ----
K = -0.6 * (L_IKT / 2.0) * U      # mm

# ---- SOZ focal drive: persistent low-amplitude stimulation of seed disk ----
# Represents the chronic epileptic focus. Amplitude chosen so that:
#   - off (no drive, zero stim): does NOT self-ignite
#   - on (persistent drive):     produces 10–11 sustained bursts at ≈ 480 ms period
SOZ_AMP = 4.0              # mV — persistent SOZ drive
SOZ_RADIUS = 2.0           # mm — same disk as interictal kick

# ---- seizure mechanisms ----
SEI_KW = dict(
    dphi_mult = 1.0,       # adaptive threshold: rises with rE → terminates each burst
    tau_phi   = 100.0,     # ms — threshold recovery ↔ controls burst period
    b_a       = 3.0,       # sAHP (secondary termination mechanism)
    tau_a     = 2500.0,    # ms — sAHP recovery
)


def _build_soz_stim(n, L):
    """Persistent SOZ disk stimulation (constant in time)."""
    X, Y = _grid(n, L)
    r = np.sqrt((X - K[0])**2 + (Y - K[1])**2)
    mask = (r <= SOZ_RADIUS).astype(float)
    def sf(t): return SOZ_AMP * mask       # noqa: E731 — constant stim
    return sf, mask


def _shaft(angle_deg, pitch, n, origin, prefix):
    return build_shaft(np.deg2rad(angle_deg), pitch, n, tuple(origin), prefix)


def _arclength(montage, u_shaft):
    c = np.asarray(montage.contacts, float)
    return (c - c.mean(0)) @ u_shaft


def _first_burst_window(ext, dt, thr=0.05):
    """Onset / offset (ms) of the FIRST active-fraction crossing of thr."""
    above = ext > thr
    if not above.any():
        return None
    onset = int(np.argmax(above))
    tail  = above[onset:]
    below = np.where(~tail)[0]
    off   = onset + int(below[0]) if below.size > 0 else len(ext) - 1
    return (onset * dt, off * dt)


def _onset_map(frames, thr):
    """Relative per-pixel first-crossing time (ms), 0 = first activated pixel."""
    crossed = frames > thr
    first   = np.argmax(crossed, axis=0).astype(float)
    onset   = np.where(crossed.any(axis=0), first * DT_IKT, np.nan).ravel()
    return onset - np.nanmin(onset)


def build_ictal(
    n=N_IKT, L=L_IKT, pitch=4.0,
    par_n=6, perp_n=5,
    par_center=(1.5, 1.5),
    perp_center=(1.5, 1.5),
):
    """Run 2-D LIF seizure, read out with virtual SEEG electrodes."""
    op  = OP_IKT
    thr = float(_OP_BASE["nuE"]) + DETECT

    sf, _mask = _build_soz_stim(n, L)

    print(f"  [ictal] op base nuE={_OP_BASE['nuE']*1000:.2f} Hz  w_ee_mult={W_EE_IKT}", flush=True)
    print(f"  [ictal] SOZ amp={SOZ_AMP} mV persistent  n={n}  {int(T_MAX_IKT/DT_IKT)} steps …",
          flush=True)

    kw = dict(dt=DT_IKT, t_max=T_MAX_IKT, theta_EE=np.deg2rad(AXIS_DEG),
              n=n, L=L, return_frames=True, **SEI_KW)
    result   = integrate_lif_field(op, sf, **kw)
    frames   = np.asarray(result[-1], dtype=np.float32)   # (nsteps, n, n)
    ext      = (frames > thr).mean(axis=(1, 2))

    win_first = _first_burst_window(ext, DT_IKT)
    if win_first is None:
        raise RuntimeError("no burst detected — adjust SOZ_AMP or w_ee_mult")

    # count total bursts for diagnostic
    above = ext > 0.05
    n_bursts = int(np.diff(above.astype(int)).clip(0).sum())
    print(f"  [ictal] {n_bursts} bursts  first={win_first[0]:.0f}–{win_first[1]:.0f} ms",
          flush=True)

    # Onset map from first burst (clean directional snapshot)
    first_end_step = int(win_first[1] / DT_IKT) + 1
    onset = _onset_map(frames[:first_end_step], thr)

    # Per-contact envelopes (full trace; locus assessed from first burst)
    kw_kern = 0.5 * pitch
    par_m   = _shaft(AXIS_DEG,      pitch, par_n,  par_center,  "P")
    perp_m  = _shaft(AXIS_DEG + 90, pitch, perp_n, perp_center, "Q")
    env_par  = rate_event_envelope(frames, n, L, par_m,  kw_kern)   # (n_contact, nt)
    env_perp = rate_event_envelope(frames, n, L, perp_m, kw_kern)

    t = np.arange(frames.shape[0], dtype=float) * DT_IKT

    def _make_elec(env, montage, u_shaft, panel_title):
        floor  = float(env.min())
        margin = 0.10 * (float(env.max()) - floor)
        art = attach_geometry(
            extract_lagpat(env, DT_IKT, [win_first], floor, margin, 0.5, DT_IKT),
            montage,
        )
        return dict(
            contacts    = montage.contacts,
            part        = art.bools[:, 0],
            names       = list(montage.names),
            s           = _arclength(montage, u_shaft),
            signal      = env,
            panel_title = panel_title,
        )

    par  = _make_elec(env_par,  par_m,  U,     "electrode ∥ axis — peaks sweep (reads direction)")
    perp = _make_elec(env_perp, perp_m, UPERP, "electrode ⊥ axis — peaks aligned (no direction)")
    print(f"  [ictal] ∥ part={int(par['part'].sum())}/{par_n}  "
          f"⊥ part={int(perp['part'].sum())}/{perp_n}", flush=True)

    half = L / 2.0
    # Show the first 2500 ms (≈5 bursts) in the trace panels so the burst train is visible;
    # locus dots come from the global peak per contact (all land in the first burst because
    # it has the highest amplitude) → slanted locus is honest directional evidence.
    display_window = (win_first[0], win_first[0] + 2500.0)
    two_electrode_readout(
        os.path.join(OUT, "two_electrode_readout_ictal.png"),
        field_xy     = grid_coords(n, L),
        field_c      = onset,
        field_clabel = "time after first-burst onset (ms)",
        kick_xy      = K,
        axis_deg     = AXIS_DEG,
        extent       = (-half, half, -half, half),
        par          = par,
        perp         = perp,
        t            = t,
        event_window = display_window,
        signal_ylabel= "firing-rate-density (NOT LFP)",
        substrate_label = (
            f"LIF rate field  ·  seizure  ·  w_ee_mult={W_EE_IKT}  ·  {L:.0f} mm\n"
            f"mechanisms: adaptive threshold (τ_φ={SEI_KW['tau_phi']:.0f} ms)  +  sAHP  ·  "
            f"persistent SOZ drive {SOZ_AMP} mV  —  excitable-regime demo, NOT locked op"
        ),
        contact_note  = (
            "contact spacing 4.0 mm (real SEEG scale)  ·  signal = firing-rate-density, "
            f"NOT a current  ·  locus = first burst peak (all {n_bursts} bursts show same direction)"
        ),
        title = "Virtual electrode records the seizure — LIF rate field (ictal)",
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[ictal readout] building seizure LIF rate field …", flush=True)
    build_ictal()


if __name__ == "__main__":
    main()

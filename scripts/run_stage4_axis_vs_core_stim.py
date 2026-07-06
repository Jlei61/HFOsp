"""Axis-vs-core stimulation comparison on the SMALL central core (Topic 4).

Fixed-footprint fairness: core-stim (partial cover of the source, leaves residual) vs axis-stim
(N downstream contacts split symmetrically, block both axial fronts) vs no-stim. Reports the
runaway delay each achieves. SNN-heavy 3-arm run is a CLI (cost-gated); target construction is
pure (unit-tested). See the spec / plan."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "paper_figures"),
           str(ROOT / "src" / "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import topic4_axis_vs_core as AV  # noqa: E402

# small-core substrate + stim defaults (spec §3, §5); N even and < n_source(=5 for r=3/pitch1.2)
CORE_R = 3.0
N_CONTACTS, PITCH, R_STIM, N_FOOT = 11, 1.2, 2.0, 4
STIM_ON, STIM_OFF, T_SIM = 0.0, 300.0, 600.0
CORE_MEAN, CORE_STD, DRIVE = 16.5, 1.5, 0.6
K_Q, TAU_Q, SIGMA_Q, ETA_K, K_K, TAU_K, SIGMA_K = 0.25, 5000.0, 1.5, 0.8, 1.5, 150.0, 0.5
OUT = ROOT / "results" / "topic4_sef_hfo" / "axis_vs_core"


def build_small_core_targets(S, *, core_radius, n_contacts=N_CONTACTS, pitch=PITCH, r_stim=R_STIM, N=N_FOOT):
    import plot_fig_m3a_v2_2_qI_stim_runaway_gif as Q
    center = np.asarray(S["center"], float); u = np.asarray(S["axis_unit"], float)
    contacts, names = AV.linear_montage(center, u, n_contacts, pitch)
    src, ax = AV.split_source_axis(contacts, center, core_radius)
    core_ci, axis_ci = AV.select_footprint(contacts, center, u, src, ax, N)
    is_E = np.asarray(S["labels"]) == 0
    pos = S["net"]["pos"]
    core_mask = Q._electrode_e_mask(pos, is_E, contacts[core_ci], r_stim)
    axis_mask = Q._electrode_e_mask(pos, is_E, contacts[axis_ci], r_stim)
    # Fairness is per-CONTACT (N core == N axis, guaranteed by select_footprint on disjoint contact
    # sets). r_stim(2.0) > half the pitch(1.2), so the core-boundary and nearest-downstream clamp
    # disks intersect. Give the core arm priority for those boundary cells: the axis arm then clamps
    # only the cells DOWNSTREAM of the core (sharpens "axis = block the front past the core") and the
    # two clamp sets are disjoint. Conservative for axis>=core (axis clamps fewer cells).
    axis_mask = axis_mask & ~core_mask
    return dict(contacts=contacts, names=names, source_idx=src, axis_idx=ax,
                core_contact_idx=core_ci, axis_contact_idx=axis_ci,
                core_mask=core_mask, axis_mask=axis_mask)


def run_one_arm(S, cfg, target_mask, stim_on, stim_off, DT):
    import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H
    kw = {} if target_mask is None else dict(stim_target=target_mask, stim_on=stim_on, stim_off=stim_off)
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=S["patch_vth"], **kw)
    rate_hz = np.asarray(res["rate_E"], float)
    runaway = H._first_sustained(H._smooth_rate(rate_hz, DT, 20.0), DT, 120.0, 100.0)
    return dict(runaway_ms=runaway, q_min_final=round(float(np.asarray(res["trace_qI_min"]).min()), 4),
                max_rate_hz=round(float(H._smooth_rate(rate_hz, DT, 20.0).max()), 1),
                n_stim_E=int(0 if target_mask is None else int(np.asarray(target_mask).sum())))


def main():
    import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H
    import run_sef_hfo_snn_cm_spontaneous_readout as C
    os.chdir(ROOT); C._engine_guard()
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, use_hG=False,
                           eta_K=ETA_K, k_K=K_K, tau_K=TAU_K, sigma_K=SIGMA_K,
                           k_q=K_Q, tau_q=TAU_Q, sigma_q=SIGMA_Q, q_min=0.05,
                           core_mean=CORE_MEAN, core_std=CORE_STD, core_radius=CORE_R,
                           drive=DRIVE, L=20.0, T=T_SIM, n_pulses=0, seed=1)
    S = H._build(cfg)
    DT = float(S["p"].dt); assert abs(DT - C.DT) < 1e-12
    tg = build_small_core_targets(S, core_radius=CORE_R)
    assert len(tg["core_contact_idx"]) == len(tg["axis_contact_idx"]) == N_FOOT   # fairness gate
    assert N_FOOT < len(tg["source_idx"])                                          # core not fully coverable
    arms = {"no_stim": None, "core_stim": tg["core_mask"], "axis_stim": tg["axis_mask"]}
    rows = {}
    for name, mask in arms.items():
        t0 = time.time()
        r = run_one_arm(S, cfg, mask, STIM_ON, STIM_OFF, DT)
        r["wall_s"] = round(time.time() - t0, 1)
        rows[name] = r
        print(f"ARM {name} " + json.dumps(r), flush=True)
    base = rows["no_stim"]["runaway_ms"]
    for name in ("core_stim", "axis_stim"):
        rows[name]["runaway_delay_ms"] = AV.runaway_delay_ms(rows[name]["runaway_ms"], base, T_SIM)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "small_core_stim.json").write_text(json.dumps({
        "config": dict(core_r=CORE_R, N=N_FOOT, n_contacts=N_CONTACTS, pitch=PITCH, r_stim=R_STIM,
                       stim_on=STIM_ON, stim_off=STIM_OFF, T=T_SIM, core_mean=CORE_MEAN,
                       eta_K=ETA_K, tau_K=TAU_K, drive=DRIVE),
        "n_source_contacts": int(len(tg["source_idx"])),
        "core_contact_idx": tg["core_contact_idx"].tolist(), "axis_contact_idx": tg["axis_contact_idx"].tolist(),
        "contacts": tg["contacts"].tolist(), "arms": rows}, indent=2))
    print("AXIS_VS_CORE_DELAY " + json.dumps({k: rows[k].get("runaway_delay_ms") for k in ("core_stim", "axis_stim")}), flush=True)
    print("DONE_AXIS_VS_CORE_STIM", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

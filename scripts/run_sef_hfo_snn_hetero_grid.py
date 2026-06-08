"""SNN heterogeneity pathology-core grid (spec 2026-06-08).

Experiment A: same sheet / same kick, compare baseline (uniform wide V_th spread)
vs a pathology core (narrowed spread; matched = mean held / unmatched = mean down).
PAIRED seed+noise+kick+surround — only the in-core E thresholds are swapped, so a
grid difference is the core effect, not threshold-sampling noise (spec §2).

Source-space (oracle) metrics per cell -> grid_metrics.json; per-cell NPZ for the
plotter. Modes:
  --time-one   time ONE L=3 baseline run (size the grid)
  --quick      tiny L=1 smoke over 2 cells (cheap green run)
  --grid       the real coarse grid (default)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ENGINE = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENGINE)
from params import Params, compute_nu_theta             # noqa: E402
from connectivity import place_neurons                  # noqa: E402
from connectivity_rot import build_connectivity_rot     # noqa: E402
from kick_probe import simulate_kick, compute_metrics   # noqa: E402
from lfp import LFPRecorder                              # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_heterogeneity import sample_threshold_fields            # noqa: E402
from src.sef_hfo_snn_metrics import onset_times, onset_axis, peak_active_fraction  # noqa: E402
from src.sef_hfo_snn_engine_guard import assert_versions                 # noqa: E402
from src.sef_hfo_observation import build_shaft, merge_montages          # noqa: E402

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
THETA, AR = 45.0, 2.0
PATCH_R = 0.5
T_KICK = 150.0
PITCH, NC = 0.26, 9          # per shaft (scaled sub-mm contacts; matches existing figure)


def _engine_guard():
    rec = json.loads((OUT / "engine_versions.json").read_text())
    assert_versions(rec)                       # spec §7: drift -> loud fail


def _build(L, density, seed):
    p = Params(g=3.6, L=L, density=density, T=450.0, dt=0.1, nu_ext_ratio=0.6, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(THETA), AR=AR)
    nu_theta = compute_nu_theta(p)[0]
    return p, net, nu_theta, NE


def _montage(L):
    c = (0.85 * L / 3.0, 0.85 * L / 3.0)
    return merge_montages([build_shaft(np.deg2rad(THETA), PITCH, NC, c, "P"),
                           build_shaft(np.deg2rad(THETA + 90.0), PITCH, NC, c, "Q")])


def _run_one(p, net, nu_theta, NE, kick_xy, vth, montage, core_mask):
    """One paired-seed run; source-space metrics + arrays for the plotter."""
    net["rng"] = np.random.default_rng(p.seed)          # paired noise/poisson
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=montage.contacts)
    res = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=list(kick_xy),
                        lfp_recorder=rec, V_th_per_neuron=vth)
    dt = p.dt
    onset = onset_times(res["E_spk_bool"], dt, T_KICK)
    axis = onset_axis(net["pos"][:NE], onset, min_n=20)
    m = compute_metrics(res, dt)
    coreE = np.asarray(core_mask, bool)[:NE]
    core_paf = (peak_active_fraction(res["E_spk_bool"][:, coreE], dt, 150.0, 300.0)
                if coreE.any() else float("nan"))
    return dict(
        peak=m["peak"], returned=bool(m["returned"]), outside=m["outside"],
        peak_active_frac=m["peak_active_frac"], core_paf=core_paf,
        axis_deg=(float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
                  if axis is not None else None),
        onset=onset, lfp=res["lfp_trace"], times=res["times"],
        contacts=montage.contacts, names=np.array(montage.names),
    )


def _undirected_diff(a, b):
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def _grid_cells(L):
    """Pre-registered coarse grid (spec §3). sweep-1: fix patch (mid), vary kick;
    sweep-2: fix kick (axis end), vary patch x {matched, unmatched}."""
    u = np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])
    perp = np.array([-u[1], u[0]])
    ctr = np.array([L / 2, L / 2])
    end = ctr + 0.6 * (L / 2) * u                       # far axis end (kick0)
    mid_patch = tuple(ctr)
    cells = []
    kicks = {"end": end, "nearcore": ctr + 0.25 * (L / 2) * u,
             "opp": ctr - 0.6 * (L / 2) * u, "offaxis": ctr + 0.5 * (L / 2) * perp}
    for kname, k in kicks.items():
        cells.append(dict(sweep=1, kick=tuple(k), kname=kname, pname="mid",
                          patch=mid_patch, cond="matched"))
    patches = {"nearseed": tuple(end - 0.3 * (L / 2) * u), "mid": mid_patch,
               "far": tuple(ctr - 0.4 * (L / 2) * u),
               "offaxis": tuple(ctr + 0.4 * (L / 2) * perp)}
    for pname, pc in patches.items():
        for cond in ("matched", "unmatched"):
            cells.append(dict(sweep=2, kick=tuple(end), kname="end", pname=pname,
                              patch=pc, cond=cond))
    return cells


def _process(cells, L, density, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_cell").mkdir(exist_ok=True)
    p, net, nu_theta, NE = _build(L, density, seed)
    is_E = net["labels"] == 0
    montage = _montage(L)
    rows = []
    base_cache = {}                                      # (kick) -> baseline run dict
    t_start = time.time()
    for i, c in enumerate(cells):
        fields = sample_threshold_fields(net["pos"], is_E, c["patch"], PATCH_R,
                                         np.random.default_rng(seed))
        core_mask = fields["core_mask"]
        kkey = tuple(np.round(c["kick"], 4))
        if kkey not in base_cache:
            base_cache[kkey] = _run_one(p, net, nu_theta, NE, c["kick"],
                                        fields["baseline"], montage, core_mask)
        b = base_cache[kkey]
        core = _run_one(p, net, nu_theta, NE, c["kick"], fields[c["cond"]],
                        montage, core_mask)
        d_paf = core["peak_active_frac"] - b["peak_active_frac"]
        d_core = (core["core_paf"] - b["core_paf"]
                  if np.isfinite(core["core_paf"]) and np.isfinite(b["core_paf"]) else None)
        d_axis = (None if (core["axis_deg"] is None or b["axis_deg"] is None)
                  else _undirected_diff(core["axis_deg"], b["axis_deg"]))
        on_patch = bool(np.linalg.norm(np.array(c["kick"]) - np.array(c["patch"])) <= PATCH_R)
        row = dict(idx=i, sweep=c["sweep"], kname=c["kname"], pname=c["pname"],
                   cond=c["cond"], patch=list(c["patch"]), kick=list(c["kick"]),
                   n_core=int(core_mask.sum()),
                   d_peak_active_frac=d_paf, d_core_paf=d_core, d_axis_deg=d_axis,
                   core_returned=core["returned"], base_returned=b["returned"],
                   kick_on_patch=on_patch)
        rows.append(row)
        np.savez_compressed(
            OUT / "per_cell" / f"cell{i:02d}.npz",
            posE=net["pos"][:NE], onset_core=core["onset"],
            vth=fields[c["cond"]], is_E=is_E, lfp=core["lfp"], times=core["times"],
            contacts=core["contacts"], names=core["names"], nc=NC,
            kick=np.array(c["kick"]), patch=np.array(c["patch"]), patch_r=PATCH_R,
            L=L, theta=THETA, base_lfp=b["lfp"], base_times=b["times"],
            meta=json.dumps(row))
        print(f"[{i+1}/{len(cells)}] sweep{c['sweep']} k={c['kname']} p={c['pname']} "
              f"{c['cond']} dpaf={d_paf:+.3f} dcore={d_core} daxis={d_axis} "
              f"on_patch={on_patch} ({time.time()-t_start:.0f}s)", flush=True)
    (OUT / "grid_metrics.json").write_text(json.dumps(
        dict(L=L, density=density, seed=seed, patch_r=PATCH_R, theta=THETA,
             cells=rows), indent=2))
    print(f"wrote grid_metrics.json ({len(rows)} cells, {time.time()-t_start:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-one", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    _engine_guard()
    if a.time_one:
        p, net, nu_theta, NE = _build(3.0, 1800.0, 1)
        fields = sample_threshold_fields(net["pos"], net["labels"] == 0,
                                         (1.5, 1.5), PATCH_R, np.random.default_rng(1))
        t0 = time.time()
        _run_one(p, net, nu_theta, NE, [2.4, 1.5], fields["baseline"], _montage(3.0),
                 fields["core_mask"])
        print(f"ONE L=3 run wall = {time.time()-t0:.1f}s")
        return
    if a.quick:
        _process(_grid_cells(1.0)[:2], 1.0, 4000.0, 1)
        return
    _process(_grid_cells(3.0), 3.0, 1800.0, 1)


if __name__ == "__main__":
    main()

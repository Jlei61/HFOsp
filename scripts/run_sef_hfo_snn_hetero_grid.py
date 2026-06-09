"""SNN heterogeneity pathology-core grid (spec 2026-06-08; 2x2 + improved metrics
2026-06-08c per review).

Experiment A: same sheet / same kick, compare a quiet baseline (wide-healthy core)
vs a pathology core. The core threshold field is a 2x2 of mean{18,16} x std{wide,
narrow} so variance and mean axes separate:
  matched   = (18, narrow)  variance axis (mean held)
  mean_only = (16, wide)    mean axis (spread held)
  unmatched = (16, narrow)  combined
Surround scalar-quiet; PAIRED seed+noise+kick+surround — only the in-core E
thresholds swap. Source-space (oracle) metrics + pre-kick-ignition split + clean
self-limit (rest-window / decay-ratio / burst-duration) per cell. Modes:
  --time-one / --quick / --grid
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ENGINE = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENGINE)
from params import Params, compute_nu_theta             # noqa: E402
from connectivity import place_neurons                  # noqa: E402
from connectivity_rot import build_connectivity_rot     # noqa: E402
from kick_probe import simulate_kick                     # noqa: E402
from lfp import LFPRecorder                              # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_heterogeneity import sample_threshold_fields            # noqa: E402
from src.sef_hfo_snn_metrics import (                                    # noqa: E402
    onset_times, onset_axis, peak_active_fraction,
    pre_kick_ignition, self_limit, event_peak_time)
from src.sef_hfo_snn_engine_guard import assert_versions                 # noqa: E402
from src.sef_hfo_observation import build_shaft, merge_montages          # noqa: E402

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
THETA, AR = 45.0, 2.0
PATCH_R = 0.5
T_KICK = 150.0
PAF_LO, PAF_HI = 150.0, 350.0   # event-capturing window (localized events ~176ms; max-based)
PITCH, NC = 0.26, 9
METRIC_VERSION = "2026-06-08c"
CONDS = ("matched", "mean_only", "unmatched")


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _provenance(p, seed):
    return dict(
        git_commit=_git_hash(), metric_version=METRIC_VERSION,
        engine_versions=json.loads((OUT / "engine_versions.json").read_text()),
        config=dict(L=p.L, density=p.density, drive=p.nu_ext_ratio, g=p.g, dt=p.dt,
                    T=p.T, theta_EE=THETA, AR=AR, patch_r=PATCH_R, V_th=p.V_th,
                    V_reset=p.V_reset, std_wide=1.5, std_narrow=0.5, core_mean_shift=2.0),
        seeds=dict(network_seed=seed), paf_window=[PAF_LO, PAF_HI],
        self_limit="rest[20,50] / decay-ratio[peak+120,+200] / burst-duration; "
                   "pre-kick ignition split [20,150]@10Hz")


def _engine_guard():
    assert_versions(json.loads((OUT / "engine_versions.json").read_text()))


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


def _run_one(p, net, nu_theta, NE, kick_xy, vth, montage, core_mask, seed):
    """One paired-seed run; source-space metrics + ignition split + self-limit."""
    net["rng"] = np.random.default_rng(seed)            # paired noise/poisson
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=montage.contacts)
    res = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=list(kick_xy),
                        lfp_recorder=rec, V_th_per_neuron=vth)
    dt = p.dt; rate = res["rate_E"]
    onset = onset_times(res["E_spk_bool"], dt, T_KICK)
    axis = onset_axis(net["pos"][:NE], onset, min_n=20)
    coreE = np.asarray(core_mask, bool)[:NE]
    whole_paf = peak_active_fraction(res["E_spk_bool"], dt, PAF_LO, PAF_HI)
    core_paf = (peak_active_fraction(res["E_spk_bool"][:, coreE], dt, PAF_LO, PAF_HI)
                if coreE.any() else float("nan"))
    ig, ig_lat = pre_kick_ignition(rate, dt, T_KICK)
    sl = self_limit(rate, dt, T_KICK)
    return dict(
        core_paf=core_paf, peak_active_frac=whole_paf,
        event_peak_t=event_peak_time(rate, dt, PAF_LO, PAF_HI),
        prekick_ignited=bool(ig), ignition_latency=ig_lat,
        rest_rate=sl["rest_rate"], returned=sl["returned"], tail_complete=sl["tail_complete"],
        decay_ratio=sl["decay_ratio"],
        burst_duration_ms=sl["burst_duration_ms"], peak=sl["peak"],
        axis_deg=(float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
                  if axis is not None else None),
        onset=onset, rate=rate, lfp=res["lfp_trace"], times=res["times"],
        contacts=montage.contacts, names=np.array(montage.names),
    )


def _undirected_diff(a, b):
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def _kicks_patches(L):
    """Canonical kick and core (patch) positions shared by the grid + factorial."""
    u = np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])
    perp = np.array([-u[1], u[0]])
    ctr = np.array([L / 2, L / 2]); end = ctr + 0.6 * (L / 2) * u
    mid = tuple(ctr)
    kicks = {"end": end, "nearcore": ctr + 0.25 * (L / 2) * u,
             "opp": ctr - 0.6 * (L / 2) * u, "offaxis": ctr + 0.5 * (L / 2) * perp}
    patches = {"nearseed": tuple(end - 0.3 * (L / 2) * u), "mid": mid,
               "far": tuple(ctr - 0.4 * (L / 2) * u),
               "offaxis": tuple(ctr + 0.4 * (L / 2) * perp)}
    return kicks, patches, end


def _grid_cells(L):
    """Pre-registered coarse grid. sweep-1: variance axis (matched) across kicks,
    fixed mid patch; sweep-2: all 3 conds across patches, fixed kick=end."""
    kicks, patches, end = _kicks_patches(L)
    cells = []
    for kn, k in kicks.items():
        cells.append(dict(sweep=1, kick=tuple(k), kname=kn, pname="mid",
                          patch=patches["mid"], cond="matched"))
    for pn, pc in patches.items():
        for cond in CONDS:
            cells.append(dict(sweep=2, kick=tuple(end), kname="end", pname=pn,
                              patch=pc, cond=cond))
    return cells


def _grid_cells_factorial_matched(L):
    """Full matched kick×core factorial (4×4=16; matched = clean evoked, so
    propagation direction is comparable across the whole grid) PLUS the end-kick
    mean_only/unmatched cells (8) so the self-igniting-condition figures persist.
    24 unique (kick,core,cond) cells — one mechanism figure each."""
    kicks, patches, end = _kicks_patches(L)
    cells = []
    for kn, k in kicks.items():
        for pn, pc in patches.items():
            cells.append(dict(sweep=3, kick=tuple(k), kname=kn, pname=pn,
                              patch=pc, cond="matched"))
    for pn, pc in patches.items():
        for cond in ("mean_only", "unmatched"):
            cells.append(dict(sweep=2, kick=tuple(end), kname="end", pname=pn,
                              patch=pc, cond=cond))
    return cells


def _process(cells, L, density, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_cell").mkdir(exist_ok=True)
    p, net, nu_theta, NE = _build(L, density, seed)
    is_E = net["labels"] == 0
    montage = _montage(L)
    rows = []; base_cache = {}; t0 = time.time()
    for i, c in enumerate(cells):
        fields = sample_threshold_fields(net["pos"], is_E, c["patch"], PATCH_R,
                                         np.random.default_rng(seed))
        core_mask = fields["core_mask"]
        bkey = (tuple(np.round(c["kick"], 4)), tuple(np.round(c["patch"], 4)))
        if bkey not in base_cache:
            base_cache[bkey] = _run_one(p, net, nu_theta, NE, c["kick"],
                                        fields["baseline"], montage, core_mask, seed)
        b = base_cache[bkey]
        core = _run_one(p, net, nu_theta, NE, c["kick"], fields[c["cond"]],
                        montage, core_mask, seed)
        d_core = (core["core_paf"] - b["core_paf"]
                  if np.isfinite(core["core_paf"]) and np.isfinite(b["core_paf"]) else None)
        d_axis = (None if (core["axis_deg"] is None or b["axis_deg"] is None)
                  else _undirected_diff(core["axis_deg"], b["axis_deg"]))
        on_patch = bool(np.linalg.norm(np.array(c["kick"]) - np.array(c["patch"])) <= PATCH_R)
        # d_core_paf is a within-evoked-event synchrony measure ONLY when neither run
        # ignited pre-kick; else it indexes an ignition regime change (review 1B).
        evoked_clean = (not core["prekick_ignited"]) and (not b["prekick_ignited"])
        row = dict(idx=i, sweep=c["sweep"], kname=c["kname"], pname=c["pname"],
                   cond=c["cond"], patch=list(c["patch"]), kick=list(c["kick"]),
                   n_core=int(core_mask.sum()),
                   d_core_paf=d_core, d_axis_deg=d_axis,
                   d_event_t=core["event_peak_t"] - b["event_peak_t"],
                   core_prekick_ignited=core["prekick_ignited"],
                   core_ignition_latency=core["ignition_latency"],
                   base_prekick_ignited=b["prekick_ignited"],
                   evoked_clean=evoked_clean,
                   core_returned=core["returned"], base_returned=b["returned"],
                   core_decay_ratio=core["decay_ratio"], core_rest_rate=core["rest_rate"],
                   core_burst_ms=core["burst_duration_ms"],
                   base_rest_rate=b["rest_rate"], kick_on_patch=on_patch)
        rows.append(row)
        np.savez_compressed(
            OUT / "per_cell" / f"cell{i:02d}.npz",
            posE=net["pos"][:NE], onset_core=core["onset"], vth=fields[c["cond"]],
            is_E=is_E, lfp=core["lfp"], times=core["times"], rate=core["rate"],
            base_rate=b["rate"], contacts=core["contacts"], names=core["names"], nc=NC,
            kick=np.array(c["kick"]), patch=np.array(c["patch"]), patch_r=PATCH_R,
            L=L, theta=THETA, base_lfp=b["lfp"], base_times=b["times"],
            event_peak_t=core["event_peak_t"], base_event_peak_t=b["event_peak_t"],
            meta=json.dumps(row))
        print(f"[{i+1}/{len(cells)}] s{c['sweep']} k={c['kname']} p={c['pname']} "
              f"{c['cond']} dcore={d_core} ig={core['prekick_ignited']} "
              f"ret={core['returned']} on_patch={on_patch} ({time.time()-t0:.0f}s)",
              flush=True)
    (OUT / "grid_metrics.json").write_text(json.dumps(
        dict(provenance=_provenance(p, seed), cells=rows), indent=2))
    print(f"wrote grid_metrics.json ({len(rows)} cells, {time.time()-t0:.0f}s)")


def _mid_pair_seeds(L, density, seeds=(1, 2, 3)):
    """3-seed robustness for the mid-core 3 conditions vs baseline (advisor)."""
    p, net, nu_theta, NE = _build(L, density, seeds[0])
    is_E = net["labels"] == 0; montage = _montage(L)
    ctr = np.array([L / 2, L / 2])
    u = np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])
    kick = ctr + 0.6 * (L / 2) * u; patch = tuple(ctr)
    recs = []
    for s in seeds:
        fields = sample_threshold_fields(net["pos"], is_E, patch, PATCH_R,
                                         np.random.default_rng(s))
        cm = fields["core_mask"]
        b = _run_one(p, net, nu_theta, NE, kick, fields["baseline"], montage, cm, s)
        out = dict(seed=s, base_prekick_ignited=b["prekick_ignited"],
                   base_rest_rate=b["rest_rate"])
        for cond in CONDS:
            c = _run_one(p, net, nu_theta, NE, kick, fields[cond], montage, cm, s)
            out[f"d_core_{cond}"] = c["core_paf"] - b["core_paf"]
            out[f"ignited_{cond}"] = c["prekick_ignited"]
            out[f"ig_lat_{cond}"] = c["ignition_latency"]
            out[f"returned_{cond}"] = c["returned"]
            out[f"d_event_t_{cond}"] = c["event_peak_t"] - b["event_peak_t"]
        recs.append(out)
        print(f"[mid-seed {s}] " + " ".join(
            f"{k}={out[f'd_core_{k}']:+.3f}(ig={out[f'ignited_{k}']})" for k in CONDS),
            flush=True)

    def agg(key):
        v = np.array([r[key] for r in recs], float)
        return dict(mean=float(v.mean()), std=float(v.std()), vals=v.tolist())

    summary = dict(seeds=list(seeds), patch="mid", kick="end",
                   provenance=_provenance(p, seeds[0]), per_seed=recs)
    for cond in CONDS:
        summary[f"d_core_{cond}"] = agg(f"d_core_{cond}")
        summary[f"d_event_t_{cond}"] = agg(f"d_event_t_{cond}")
        summary[f"ignition_rate_{cond}"] = float(
            np.mean([r[f"ignited_{cond}"] for r in recs]))
    (OUT / "mid_pair_seeds.json").write_text(json.dumps(summary, indent=2))
    print("wrote mid_pair_seeds.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-one", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--factorial-matched", action="store_true",
                    help="full matched kick×core factorial + end-kick igniting cells (24)")
    a = ap.parse_args()
    _engine_guard()
    if a.time_one:
        p, net, nu_theta, NE = _build(3.0, 1800.0, 1)
        fields = sample_threshold_fields(net["pos"], net["labels"] == 0,
                                         (1.5, 1.5), PATCH_R, np.random.default_rng(1))
        t0 = time.time()
        _run_one(p, net, nu_theta, NE, [2.4, 1.5], fields["baseline"], _montage(3.0),
                 fields["core_mask"], 1)
        print(f"ONE L=3 run wall = {time.time()-t0:.1f}s")
        return
    if a.quick:
        cells = (_grid_cells_factorial_matched(1.0)[:3] if a.factorial_matched
                 else _grid_cells(1.0)[:3])
        _process(cells, 1.0, 4000.0, 1)
        if not a.factorial_matched:
            _mid_pair_seeds(1.0, 4000.0, seeds=(1, 2))
        return
    if a.factorial_matched:
        _process(_grid_cells_factorial_matched(3.0), 3.0, 1800.0, 1)   # 24 cells, ~25min
        return
    _process(_grid_cells(3.0), 3.0, 1800.0, 1)
    _mid_pair_seeds(3.0, 1800.0, seeds=(1, 2, 3))


if __name__ == "__main__":
    main()

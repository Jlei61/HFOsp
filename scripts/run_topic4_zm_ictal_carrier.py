#!/usr/bin/env python
"""Z/M ictal-carrier gate runner (spec 2026-07-24). Phase 1: does the Z/M(+S_G) substrate produce a
SUSTAINED ictal carrier on the virtual SEEG, or an HFO-like burst train? H is FROZEN OFF here.

Reuses PP.build_substrate (E1146 twoend_equal, L=20, N=40000) + the canonical LFPRecorder driven by the
engine's built-in `lfp_recorder=` hook (observation-only, no engine edit) + the exit runner's
calibration/cfg/core-mask helpers. Per arm: run -> decimate LFP 10k->2k (Nyquist 1kHz > 150) -> source
+ observed metrics -> ictal_carrier_verdict + lifecycle_verdict. Atomic per-arm provenance manifest,
crash-safe --resume, OMP=1, --confirm-run gated. Outputs -> results/topic4_sef_hfo/zm_ictal_carrier_gate/.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import resource
import sys
import time

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP                # noqa: E402  build_substrate + constants
import run_zm_snn_native_exit as ZM          # noqa: E402  _calibrate_I_th_EI/_zm_cfg/_core_mask_E/ARMS
from kick_probe import simulate_kick          # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from lfp import LFPRecorder                    # noqa: E402

from src.sef_hfo_snn_engine_guard import record_versions            # noqa: E402
import src.topic4_zm_ictal_carrier as CG                            # noqa: E402
from src.topic4_zm_carrier_verdict import ictal_carrier_verdict, lifecycle_verdict  # noqa: E402

DT = 0.1
OUT = os.path.join(PP.ROOT, "results", "topic4_sef_hfo", "zm_ictal_carrier_gate")
ENGINE_FILES = [os.path.join(_ROOT, "src", "snn_engine", f)
                for f in ("slow_field.py", "kick_probe.py", "lfp.py", "mz_slow_vars.py", "connectivity_rot.py")]

# H is FROZEN OFF in every arm (Phase 1). interictal_ctrl = frozen slow (z=1, m=0) = pure substrate baseline.
ARMS = {
    "interictal_ctrl": dict(frozen=True, T=3000.0, es_thresh=250.0),
    "bare":            dict(kw=dict(), T=15000.0, es_thresh=120.0),                    # Z/M -> onset -> runaway
    "sg":              dict(kw=dict(use_SG=True, alpha_G=16.0), T=15000.0, es_thresh=250.0),  # + S_G containment
}


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)   # ru_maxrss is KB on linux


def _mem_snapshot():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")[0], line.split()[1]
            if k in ("MemAvailable", "SwapFree", "SwapTotal"):
                info[k] = int(v) // 1024   # MB
    info["self_rss_gb"] = round(_rss_gb(), 2)
    return info


def _snapshot_times(core_rate, bin_ms, onset_ms, T_ms):
    """4 source-space snapshots snapped to ACTUAL burst peaks (spec §6.4): pre-onset (quiet), first
    recruitment (first post-onset burst), putative carrier (mid burst), late (last burst). Snapping to
    burst peaks (not fixed offsets) keeps a burst-train's snapshots from landing in the black inter-burst
    gaps -- so the panel actually shows the spatial pattern DURING activity."""
    from src.topic4_zm_slowfast import detect_bursts
    pk, _ = detect_bursts(np.asarray(core_rate, float), bin_ms)
    pk_ms = pk * bin_ms
    post = pk_ms[pk_ms >= onset_ms] if onset_ms is not None else pk_ms
    pre = max(0.0, (onset_ms - 300.0) if onset_ms is not None else 0.1 * T_ms)
    if post.size >= 3:
        return [pre, float(post[0]), float(post[post.size // 2]), float(post[-1])]
    if post.size >= 1:
        return [pre, float(post[0]), float(post[min(1, post.size - 1)]), float(post[-1])]
    return [0.1 * T_ms, 0.3 * T_ms, 0.6 * T_ms, 0.9 * T_ms]     # no bursts -> evenly spaced


def _run_arm(S, name, spec, rec, contact_names):
    t0 = time.time()
    T_ms = float(spec["T"])
    p = dataclasses.replace(S["p"], T=T_ms)
    core = ZM._core_mask_E(S)
    if spec.get("frozen"):
        cfg = SpatialSlowFieldConfig(use_qI=False, use_gK=False, use_z=False, use_m=False)
    else:
        cfg = ZM._zm_cfg(S["I_th_EI"], **spec["kw"])
    slow = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], core_mask_E=core, cfg=cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                        t_kick=1e9, V_th_per_neuron=S["vth"], verbose=False,
                        early_stop_runaway=True, es_thresh_hz=float(spec["es_thresh"]), es_dur_ms=100.0,
                        lfp_recorder=rec)
    spk = res["E_spk_bool"]
    runaway_ms = res.get("runaway_early_stop_ms")

    # ---- virtual SEEG: decimate 10k -> 2k, Nyquist gate ----
    lfp_ds, fs = CG.decimate_lfp(np.asarray(res["lfp_trace"], float), fs_in=1e3 / DT)
    CG.assert_nyquist(fs)

    # ---- source-space + observed metrics + carrier + lifecycle verdicts ----
    src = CG.compute_source_metrics(spk, core, S["posE"], S["src_xy"], S["axis_unit"], S["L"], DT, runaway_ms)
    onset_ms = src["onset_ms"] if src["onset_ms"] is not None else 500.0   # baseline window fallback
    obs = CG.compute_observed_metrics(lfp_ds, fs, onset_ms)
    m = CG.carrier_metrics_from(src, obs)
    carrier_label, carrier_detail = ictal_carrier_verdict(m)
    life = lifecycle_verdict(carrier_label, dict(onset_detected=src["onset_ms"] is not None, terminated=False))

    # ---- kymographs + snapshots (mm) for the figure ----
    ax, tr = CG.axis_transverse_coords(S["posE"], S["src_xy"], S["axis_unit"])
    kymo_ax, ax_edges, kt = CG.kymograph(spk, ax, DT)
    kymo_tr, tr_edges, _ = CG.kymograph(spk, tr, DT)
    snap_ms = _snapshot_times(src["rates"]["core"], src["rates"]["bin_ms"], src["onset_ms"], spk.shape[0] * DT)
    snaps = np.stack([CG.field_snapshot(spk, S["posE"], S["L"], t - 60.0, t + 60.0, DT) for t in snap_ms])

    fr = src["rates"]
    npz = os.path.join(OUT, f"{name}_seed{S['seed']}.npz")
    np.savez_compressed(
        npz,
        core_rate=fr["core"].astype(np.float32), surr_rate=fr["surround"].astype(np.float32),
        all_rate=fr["allE"].astype(np.float32), active_frac=fr["active_frac"].astype(np.float32),
        rate_bin_ms=fr["bin_ms"], e_A=src["e_A"].astype(np.float32),
        lfp=lfp_ds, lfp_fs=fs, contact_names=np.asarray(contact_names),
        lowgamma_db=obs["lowgamma_db"].astype(np.float32), highfreq_db=obs["highfreq_db"].astype(np.float32),
        broadband_db=obs["broadband_db"].astype(np.float32), frame_times_ms=obs["times_ms"].astype(np.float32),
        kymo_axis=kymo_ax.astype(np.float32), kymo_transverse=kymo_tr.astype(np.float32),
        kymo_axis_edges=ax_edges.astype(np.float32), kymo_transverse_edges=tr_edges.astype(np.float32),
        kymo_t_ms=kt.astype(np.float32), snapshots=snaps.astype(np.float32), snapshot_ms=np.asarray(snap_ms),
        # slow traces (phi_drive/active_frac/H empty here -- H frozen off in Phase 1)
        z_mean=np.asarray(slow.trace_z_mean, np.float32), z_core=np.asarray(slow.trace_z_core_mean, np.float32),
        z_surround=np.asarray(slow.trace_z_surround_mean, np.float32), z_min=np.asarray(slow.trace_z_min, np.float32),
        m_mean=np.asarray(slow.trace_m_mean, np.float32), m_core=np.asarray(slow.trace_m_core_mean, np.float32),
        m_surround=np.asarray(slow.trace_m_surround_mean, np.float32),
        SG=np.asarray(slow.trace_SG, np.float32), H=np.asarray(slow.trace_H, np.float32),
        phi_drive=np.asarray(slow.trace_phi_drive, np.float32), active_frac_H=np.asarray(slow.trace_active_frac, np.float32),
        p_mean=np.asarray(slow.trace_p_mean, np.float32), p_max=np.asarray(slow.trace_p_max, np.float32))

    def _clean(d):   # macro dicts -> json-safe
        return {k: (None if v is None else float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in d.items()}

    t1 = time.time()
    record = dict(
        arm=name, seed=int(S["seed"]), status="complete", T_ms=T_ms, n_steps=int(spk.shape[0]),
        command=" ".join(sys.argv), config=dataclasses.asdict(cfg),
        git_sha=CG.git_sha(_ROOT), engine_versions=record_versions(ENGINE_FILES),
        t_start=round(t0, 1), t_end=round(t1, 1), runtime_s=round(t1 - t0, 1),
        classifier_version=CG.CLASSIFIER_VERSION, readout_fs_hz=float(fs), nyquist_hz=float(fs / 2.0),
        runaway_early_stop_ms=runaway_ms,
        output_files=[os.path.basename(npz)], output_sha256={os.path.basename(npz): CG.sha256_file(npz)},
        ictal_carrier_verdict=carrier_label, carrier_detail=carrier_detail, lifecycle_verdict=life,
        source_metrics=dict(onset_ms=src["onset_ms"], macro=_clean(src["macro"]), af_macro=_clean(src["af_macro"]),
                            whole_field_flash=src["whole_field_flash"], has_recruitment=src["has_recruitment"],
                            saturated_plateau=src["saturated_plateau"], tail_escalating=src["tail_escalating"],
                            src_sep_count=src["src_sep_count"], onset_area=src["onset_area"], peak_area=src["peak_area"],
                            core_peak_hz=float(fr["core"].max()), all_mean_hz=float(fr["allE"].mean()),
                            all_tail_hz=float(fr["allE"][-max(1, len(fr["allE"]) // 20):].mean())),
        observed_metrics=dict(n_sustained_contacts=obs["n_sustained_contacts"], best_macro=_clean(obs["best_macro"]),
                              highfreq_enhanced=obs["highfreq_enhanced"], best_contact=contact_names[obs["best_contact_idx"]],
                              obs_sep_count=m["obs_sep_count"],
                              contact_peak_lowgamma_db={contact_names[i]: round(c["peak_lowgamma_db"], 2)
                                                        for i, c in enumerate(obs["contacts"])}),
        mem=_mem_snapshot())
    CG.write_json_atomic(os.path.join(OUT, f"{name}_seed{S['seed']}.json"), record)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true", help="required: guards the multi-minute sim")
    ap.add_argument("--arms", default="interictal_ctrl,bare,sg")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="skip arms already complete in the manifest")
    ap.add_argument("--smoke", action="store_true", help="tiny T for a plumbing check (NOT science)")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("refusing to run without --confirm-run (each arm is multi-minute at N=40000)")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    assert all(x in ARMS for x in arms), f"unknown arm; choices={list(ARMS)}"
    os.makedirs(OUT, exist_ok=True)
    manifest_path = os.path.join(OUT, f"carrier_gate_seed{a.seed}.json")

    print(f"[mem] {_mem_snapshot()}", flush=True)
    S = PP.build_substrate(seed=a.seed)
    S["seed"] = a.seed
    S["I_th_EI"] = ZM._calibrate_I_th_EI(S)
    print(f"[calib] I_th_EI=q75={S['I_th_EI']:.4f}", flush=True)
    mont = S["reg"]["montage_sheet"]
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=np.asarray(mont.contacts, float))
    contact_names = list(mont.names)
    print(f"[montage] {len(contact_names)} contacts: {contact_names}", flush=True)

    for name in arms:
        spec = dict(ARMS[name])
        if a.smoke:
            spec["T"] = 500.0
        man = CG.read_manifest(manifest_path)
        if a.resume and CG.arm_completed(man, name) and \
           all(os.path.exists(os.path.join(OUT, f)) for f in man["arms"][name].get("output_files", [])):
            print(f"[skip] {name} already complete (resume)", flush=True)
            continue
        print(f"[arm {name}] T={spec['T']:.0f}ms es_thresh={spec['es_thresh']:.0f} ...", flush=True)
        rec_out = _run_arm(S, name, spec, rec, contact_names)
        CG.write_arm_to_manifest(manifest_path, rec_out)
        sm = rec_out["source_metrics"]
        print(f"[arm {name}] carrier={rec_out['ictal_carrier_verdict']} life={rec_out['lifecycle_verdict']} "
              f"| src macro dur={sm['macro']['duration_ms']:.0f}ms occ={sm['macro']['occupancy']:.2f} "
              f"onset={sm['onset_ms']} core_peak={sm['core_peak_hz']:.0f}Hz all_mean={sm['all_mean_hz']:.2f}Hz "
              f"| obs sustained_contacts={rec_out['observed_metrics']['n_sustained_contacts']} "
              f"| wall={rec_out['runtime_s']}s rss={rec_out['mem']['self_rss_gb']}GB", flush=True)
        print(f"[mem] {_mem_snapshot()}", flush=True)

    print(f"[done] manifest {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

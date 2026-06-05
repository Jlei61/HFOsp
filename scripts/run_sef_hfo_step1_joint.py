"""Step 1 (drive × σ) JOINT analysis on the canonical LIF rate field (contract §10).

Supersedes the 1D-slice runner ``run_sef_hfo_step1_noise.py``. Two corrections the
1D runner lacked (see step1_noise_contract §9.11):

  1. drive is SWEPT jointly with σ (a 2-D map), not held at one guessed value;
  2. the detector amplitude bar is RECALIBRATED per operating point (§10.2) between
     the σ=0/σ_ref coherence floor and the deterministic-kick event peak — so a
     near-zero-rest drive's between-event noise hovering is not mis-called
     "sustained" (the §9.7 confound).

Per drive (window A: w_ee_mult=1.0, recovery off):
  - run the deterministic finite kick  -> event-peak coherence  (§10.2 upper bound)
  - run σ=0 and σ_ref (sub-threshold)   -> worst-case noise floor (§10.2 lower bound)
  - calibrate_detector(...) -> per-op bar, or UndetectableOperatingPoint -> the drive
    is non-excitable / non-interictal (§10.3) and its σ sweep is skipped
  - sweep σ × seed with that bar; report the fraction of discrete-event seeds, rate,
    IEI, envelope per cell.

Acceptance (§10.4) is a robust 2-D REGION (>=60% seeds discrete, rate in [0.01,1]/s),
not a single σ point; σ=0 must be extinction at every drive. Honest null is legal.

Drive band = the step0b finite-pulse excitable window (kick_sweep: 0.5-1.3 all
self_limited_propagation); sampled at the interictal end and up toward the
oscillation-onset shoulder (located by the model's OWN phase structure, §10.3,
NOT by copying the SNN nominal ratio / NOT by E-rate).

Run:
  python scripts/run_sef_hfo_step1_joint.py --smoke        # tiny fast end-to-end check
  python scripts/run_sef_hfo_step1_joint.py                # full grid (hours; background)
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_lif import (  # noqa: E402
    ELL_PAR, ELL_PERP, L_INH, integrate_lif_field, mean_field,
    _grid, _STIM_X0, _STIM_R, _STIM_T,
)
from src.sef_hfo_events import (  # noqa: E402
    UndetectableOperatingPoint, calibrate_detector, classify_run, make_ou_noise,
)

OUT = Path("results/topic4_sef_hfo/step1_noise")
N, L, DT, T_RUN = 64, 16.0, 0.25, 5000.0
TAU_NOISE = 100.0          # slow afferent OU (contract §2; fast tau double-counts Phi_LIF sigma)
KICK_A = 8.0               # deterministic finite-pulse amplitude (matches kick_sweep)
KICK_TMAX = 300.0          # kick run length (ms) — event has self-terminated well within this

# Drive band within the step0b excitable window (kick_sweep self_limited 0.5-1.3);
# interictal end up to the oscillation-onset shoulder, located by phase-structure
# shape per §10.3 (NOT the SNN nominal ratio).
DRIVES = [0.6, 0.7, 0.8, 0.9, 1.0]
SIGMAS = [0.0, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
# σ_ref must be UNIFORMLY sub-threshold across the whole drive band so the floor is
# not contaminated by genuine events (circular). σ=1.0/1.5 already ignite at drive
# 0.6 (1D runs), so they are NOT valid floor references there; σ=0.5 is extinction
# across 0.6-1.0. Pre-registered dedicated reference (contract §10.2, NOT the lowest
# sweep-grid point as v1.3 first wrote — that ignites at the interictal end).
SIGMA_REF = 0.5
SEEDS = [0, 1, 2, 3, 4]
W_EE_MULT = 1.0           # window A (PRIMARY); recovery off
B_A = 0.0


def _coh(op, stim, t_max):
    """Run the field and return the coherence active-fraction series (detector input)."""
    out = integrate_lif_field(
        op, stim, dt=DT, t_max=t_max, b_a=B_A, n=N, L=L,
        ell_par=ELL_PAR, ell_perp=ELL_PERP, l_inh=L_INH, coh_len=ELL_PAR,
    )
    return out[-1]  # ext_coh is appended last when coh_len is set


def kick_coh(op):
    """Coherence series from the deterministic off-center finite pulse (event peak)."""
    X, Y = _grid(N, L)
    mask = ((X - _STIM_X0) ** 2 + Y ** 2 <= _STIM_R ** 2).astype(float)
    stim = lambda t: (KICK_A * mask) if t < _STIM_T else (0.0 * mask)
    return _coh(op, stim, KICK_TMAX)


def noise_coh(op, sigma, seed, t_run):
    """Coherence series from a pure-OU-noise run (σ=0 -> deterministic rest)."""
    stim = make_ou_noise(N, L, DT, sigma_noise=sigma, tau_noise=TAU_NOISE, seed=seed)
    return _coh(op, stim, t_run)


def _iei_and_env(events):
    """Median inter-event interval (s) and median envelope (ms) from a cell's events."""
    if not events:
        return None, None
    t_on = sorted(ev["t_on"] for ev in events)
    iei = np.diff(t_on) / 1000.0 if len(t_on) >= 2 else np.array([])
    env = [ev["dur_ms"] for ev in events]
    return (float(np.median(iei)) if iei.size else None), float(np.median(env))


def run_drive(drive, sigmas, seeds, sigma_ref, t_run):
    """Calibrate the detector at this drive, then sweep σ × seed with the per-op bar."""
    op = mean_field(drive, w_ee_mult=W_EE_MULT)
    # --- §10.2 calibration: kick peak + σ=0 / σ_ref floors (anti-circular: no grid) ---
    kick = kick_coh(op)
    s_floor0 = noise_coh(op, 0.0, 0, t_run)        # σ=0 deterministic rest (= the σ=0 sweep cell)
    s_floorR = noise_coh(op, sigma_ref, 0, t_run)  # σ_ref sub-threshold (= the σ_ref,seed0 cell)
    cache = {(0.0, 0): s_floor0, (sigma_ref, 0): s_floorR}
    try:
        cal = calibrate_detector([s_floor0, s_floorR], kick)
    except UndetectableOperatingPoint as e:
        return dict(drive=drive, undetectable=True, reason=str(e),
                    kick_peak=float(np.max(kick)),
                    floor=max(float(np.max(s_floor0)), float(np.max(s_floorR))),
                    per_sigma=[], cells=[])
    bar = cal["event_on_frac"]

    cells, per_sigma = [], []
    for sigma in sigmas:
        labels, rates, ieis, envs = [], [], [], []
        for seed in seeds:
            coh = cache.get((sigma, seed))
            if coh is None:
                coh = noise_coh(op, sigma, seed, t_run)
            res = classify_run(coh, DT, op, window="A", event_on_frac=bar)
            rate = res["n_events"] / (t_run / 1000.0)
            iei, env = _iei_and_env(res["events"])
            labels.append(res["label"])
            if res["label"] == "discrete_events":
                rates.append(rate)
                if iei is not None:
                    ieis.append(iei)
                if env is not None:
                    envs.append(env)
            cells.append(dict(drive=drive, sigma=sigma, seed=seed, label=res["label"],
                              n_events=res["n_events"], max_ext=res["max_ext"],
                              frac_time_on=res["frac_time_on"], event_rate_per_s=rate))
        frac_discrete = labels.count("discrete_events") / len(labels)
        per_sigma.append(dict(
            sigma=sigma, frac_discrete=frac_discrete, regime_counts=dict(Counter(labels)),
            mean_rate_discrete=(float(np.mean(rates)) if rates else None),
            median_iei_s=(float(np.median(ieis)) if ieis else None),
            median_env_ms=(float(np.median(envs)) if envs else None),
        ))
    sigma0 = next(p for p in per_sigma if p["sigma"] == 0.0)
    return dict(drive=drive, undetectable=False, floor=cal["floor"], kick_peak=cal["peak"],
                event_on_frac=bar, sigma0_extinction=(sigma0["regime_counts"] == {"extinction_only": len(seeds)}),
                per_sigma=per_sigma, cells=cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast end-to-end check (1 drive x 3 σ x 1 seed, short run)")
    args = ap.parse_args()

    if args.smoke:
        drives, sigmas, seeds, sigma_ref, t_run = [1.0], [0.0, 1.5, 1.8], [0], 1.5, 800.0
        tag = "joint_A_smoke"
    else:
        drives, sigmas, seeds, sigma_ref, t_run = DRIVES, SIGMAS, SEEDS, SIGMA_REF, T_RUN
        tag = f"joint_A_tau{int(TAU_NOISE)}"

    t0 = time.time()
    results, all_cells = [], []
    for drive in drives:
        td = time.time()
        r = run_drive(drive, sigmas, seeds, sigma_ref, t_run)
        results.append(r)
        all_cells.extend(r.get("cells", []))
        if r["undetectable"]:
            print(f"drive={drive}: UNDETECTABLE ({r['reason']}) -> skipped  [{time.time()-td:.0f}s]")
        else:
            fr = {p["sigma"]: round(p["frac_discrete"], 2) for p in r["per_sigma"]}
            print(f"drive={drive}: bar={r['event_on_frac']:.3f} floor={r['floor']:.3f} "
                  f"peak={r['kick_peak']:.3f} σ0_ext={r['sigma0_extinction']} "
                  f"frac_discrete={fr}  [{time.time()-td:.0f}s]")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        window="A", tau_noise=TAU_NOISE, kick_A=KICK_A, sigma_ref=sigma_ref,
        params=dict(N=N, L=L, dt=DT, t_run_ms=t_run, drives=drives, sigmas=sigmas,
                    seeds=seeds, w_ee_mult=W_EE_MULT, b_a=B_A,
                    ell_par=ELL_PAR, ell_perp=ELL_PERP, l_inh=L_INH),
        per_drive=[{k: v for k, v in r.items() if k != "cells"} for r in results],
    )
    (OUT / f"{tag}.json").write_text(json.dumps(payload, indent=2, default=float))
    with (OUT / f"{tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drive", "sigma", "seed", "label", "n_events", "max_ext",
                    "frac_time_on", "event_rate_per_s"])
        for c in all_cells:
            w.writerow([c["drive"], c["sigma"], c["seed"], c["label"], c["n_events"],
                        f"{c['max_ext']:.4f}", f"{c['frac_time_on']:.4f}",
                        f"{c['event_rate_per_s']:.4f}"])
    print(f"\nWrote {OUT}/{tag}.json + .csv  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

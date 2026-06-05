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
    ELL_PAR, ELL_PERP, L_INH, classify_response, integrate_lif_field, mean_field,
    _grid, _STIM_X0, _STIM_R, _STIM_T,
)
from src.sef_hfo_events import (  # noqa: E402
    UndetectableOperatingPoint, accepted_cell, calibrate_detector, classify_run, make_ou_noise,
)

OUT = Path("results/topic4_sef_hfo/step1_noise")
N, L, DT, T_RUN = 64, 16.0, 0.25, 5000.0
TAU_NOISE = 100.0          # slow afferent OU (contract §2; fast tau double-counts Phi_LIF sigma)
KICK_A = 8.0               # deterministic finite-pulse amplitude (matches kick_sweep)
KICK_TMAX = 300.0          # kick run length (ms) — event has self-terminated well within this

# CANDIDATE drive grid spanning the step0b excitable range. Each drive is VALIDATED
# PER-RUN here (run_drive re-fires the kick at THIS runner's grid N=64/L=16 and requires
# self_limited_propagation, §10.2 C) — that per-run gate, not the kick_sweep comment, is
# authoritative. NB: kick_sweep.json (step0 default grid) labelled 0.5-1.3 all
# self_limited, but at N=64/L=16 the joint runner finds drive 1.3 = local_bump and
# excludes it; so 1.3 is a CANDIDATE excluded by this run's kick gate, not part of a
# clean "0.5-1.3 excitable window" claim. Located by the model's own phase structure
# (§10.3), NOT the SNN nominal ratio.
DRIVES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3]
# σ_ref=0.5 IS in the grid (so it is classified + its σ=0-like extinction verified),
# AND must be UNIFORMLY sub-threshold across the band so the floor is not contaminated
# by genuine events (circular). σ=1.0/1.5 already ignite at drive 0.6 (1D runs) so are
# NOT valid floor references there; σ=0.5 is extinction across the band. (contract §10.2)
SIGMAS = [0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
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


def kick_run(op):
    """Deterministic off-center finite pulse; return (ext, front, ext_coh).

    ``ext``/``front`` feed ``classify_response`` to VALIDATE the kick is a genuine
    self-limited PROPAGATING event (§10.2 C — a local bump must not pass calibration);
    ``ext_coh`` is the event-peak reference for the detector bar.
    """
    X, Y = _grid(N, L)
    mask = ((X - _STIM_X0) ** 2 + Y ** 2 <= _STIM_R ** 2).astype(float)
    stim = lambda t: (KICK_A * mask) if t < _STIM_T else (0.0 * mask)
    ext, front, ext_coh = integrate_lif_field(
        op, stim, dt=DT, t_max=KICK_TMAX, b_a=B_A, n=N, L=L,
        ell_par=ELL_PAR, ell_perp=ELL_PERP, l_inh=L_INH, coh_len=ELL_PAR,
    )
    return ext, front, ext_coh


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
    """Calibrate the detector at this drive (validated kick + worst-case multi-seed
    floor), then sweep σ × seed with the per-op bar."""
    op = mean_field(drive, w_ee_mult=W_EE_MULT)

    # --- §10.2 C / §10.3: the kick must be a genuine self-limited PROPAGATING event,
    #     not a local bump — else this drive is non-excitable / non-interictal and the
    #     "event peak" is meaningless. classify_response on the RAW ext/front. ---
    k_ext, k_front, k_coh = kick_run(op)
    k_label, _ = classify_response(k_ext, k_front, stim_x0=_STIM_X0, stim_r=_STIM_R, dt=DT)
    if k_label != "self_limited_propagation":
        return dict(drive=drive, undetectable=True, kick_label=k_label,
                    reason=f"kick is {k_label}, not self_limited_propagation -> non-excitable",
                    kick_peak=float(np.max(k_coh)), per_sigma=[], cells=[])

    # --- §10.2 A: floor = worst case over the σ=0 deterministic run AND ALL σ_ref
    #     seeds. One σ_ref seed underestimates the stochastic upper envelope (a too-low
    #     floor -> too-low bar -> other quiet seeds mis-called events/sustained). σ=0 is
    #     deterministic (make_ou_noise(σ=0) ignores the seed) so one run covers all. ---
    s_floor0 = noise_coh(op, 0.0, 0, t_run)
    cache = {(0.0, seed): s_floor0 for seed in seeds}      # σ=0 identical across seeds
    floor_refs = [s_floor0]
    for seed in seeds:
        s = noise_coh(op, sigma_ref, seed, t_run)
        cache[(sigma_ref, seed)] = s                       # reuse as the σ_ref grid cells
        floor_refs.append(s)
    try:
        cal = calibrate_detector(floor_refs, k_coh)        # max over ALL refs (anti-circular)
    except UndetectableOperatingPoint as e:
        return dict(drive=drive, undetectable=True, kick_label=k_label, reason=str(e),
                    kick_peak=float(np.max(k_coh)),
                    floor=max(float(np.max(s)) for s in floor_refs), per_sigma=[], cells=[])
    bar = cal["event_on_frac"]

    cells, per_sigma = [], []
    for sigma in sigmas:
        labels, reasons, rates, ieis, envs = [], [], [], [], []
        for seed in seeds:
            coh = cache.get((sigma, seed))
            if coh is None:
                coh = noise_coh(op, sigma, seed, t_run)
            res = classify_run(coh, DT, op, window="A", event_on_frac=bar)
            rate = res["n_events"] / (t_run / 1000.0)
            iei, env = _iei_and_env(res["events"])
            labels.append(res["label"])
            reasons.append(res["reason"])
            if res["label"] == "discrete_events":
                rates.append(rate)
                if iei is not None:
                    ieis.append(iei)
                if env is not None:
                    envs.append(env)
            # record the FAILURE-MODE fields the first run dropped (so "sustained" can be
            # split into long_plateau / too_frequent / non_returning — the §11 claim
            # depends on this; n_events>0 alone cannot exclude a non-returning segment).
            cells.append(dict(drive=drive, sigma=sigma, seed=seed, label=res["label"],
                              reason=res["reason"], n_events=res["n_events"],
                              n_events_total=res["n_events_total"], all_returned=res["all_returned"],
                              longest_on_ms=res["longest_on_ms"], max_ext=res["max_ext"],
                              frac_time_on=res["frac_time_on"], event_rate_per_s=rate))
        frac_discrete = labels.count("discrete_events") / len(labels)
        mrd = float(np.mean(rates)) if rates else None
        per_sigma.append(dict(
            sigma=sigma, frac_discrete=frac_discrete, regime_counts=dict(Counter(labels)),
            reason_counts=dict(Counter(reasons)),          # failure-mode breakdown
            mean_rate_discrete=mrd,
            median_iei_s=(float(np.median(ieis)) if ieis else None),
            median_env_ms=(float(np.median(envs)) if envs else None),
            accepted=accepted_cell(frac_discrete, mrd),    # §10.4 machined (not eyeballed)
        ))
    sigma0 = next(p for p in per_sigma if p["sigma"] == 0.0)
    return dict(drive=drive, undetectable=False, kick_label=k_label,
                floor=cal["floor"], kick_peak=cal["peak"], event_on_frac=bar,
                sigma0_extinction=(sigma0["regime_counts"] == {"extinction_only": len(seeds)}),
                accepted_sigmas=[p["sigma"] for p in per_sigma if p["accepted"]],
                per_sigma=per_sigma, cells=cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast end-to-end check (1 drive x σ{0,σ_ref,1.8} x 2 seeds, short run)")
    ap.add_argument("--drives", type=str, default=None,
                    help="comma-separated drive override (e.g. 0.6,1.0); default = full DRIVES grid")
    ap.add_argument("--sigmas", type=str, default=None,
                    help="comma-separated σ override; σ_ref (0.5) is auto-included if missing")
    ap.add_argument("--tag", type=str, default=None, help="output file tag override")
    args = ap.parse_args()

    if args.smoke:
        # σ_ref=0.5 (matches the real run) AND in the σ grid; 2 seeds exercise the
        # multi-seed floor path.
        drives, sigmas, seeds, sigma_ref, t_run = [1.0], [0.0, 0.5, 1.8], [0, 1], 0.5, 800.0
        tag = "joint_A_smoke"
    else:
        seeds, sigma_ref, t_run = SEEDS, SIGMA_REF, T_RUN
        drives = [float(x) for x in args.drives.split(",")] if args.drives else DRIVES
        sigmas = [float(x) for x in args.sigmas.split(",")] if args.sigmas else SIGMAS
        if sigma_ref not in sigmas:        # σ_ref must be in the grid (classified + reused as floor)
            sigmas = sorted(set(sigmas) | {sigma_ref})
        if 0.0 not in sigmas:              # σ=0 extinction regression must be in the grid
            sigmas = sorted(set(sigmas) | {0.0})
        tag = args.tag or f"joint_A_tau{int(TAU_NOISE)}"

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
                  f"accepted_σ={r['accepted_sigmas']} frac_discrete={fr}  [{time.time()-td:.0f}s]")

    # --- §10.4 machined region verdict (NOT eyeballed off the heatmap): accepted
    #     cells + whether they form a robust 2-D block (two grid-adjacent drives
    #     sharing two grid-adjacent σ) vs a crack/single point, + the σ=0->extinction
    #     regression at every detectable drive. The geometric read is also eyeballed on
    #     the heatmap, but the verdict is machined so the conclusion can't drift. ---
    acc = {r["drive"]: set(r["accepted_sigmas"]) for r in results if not r["undetectable"]}
    accepted_cells = [(d, s) for d in sorted(acc) for s in sorted(acc[d])]
    robust_block = False
    for i in range(len(drives) - 1):
        d0, d1 = drives[i], drives[i + 1]
        shared = acc.get(d0, set()) & acc.get(d1, set())
        for j in range(len(sigmas) - 1):
            if sigmas[j] in shared and sigmas[j + 1] in shared:
                robust_block = True
    detectable = [r for r in results if not r["undetectable"]]
    region_summary = dict(
        n_accepted_cells=len(accepted_cells), accepted_cells=accepted_cells,
        robust_2d_block=robust_block,
        sigma0_extinction_all_drives=all(r["sigma0_extinction"] for r in detectable),
        undetectable_drives=[r["drive"] for r in results if r["undetectable"]],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        window="A", tau_noise=TAU_NOISE, kick_A=KICK_A, sigma_ref=sigma_ref,
        params=dict(N=N, L=L, dt=DT, t_run_ms=t_run, drives=drives, sigmas=sigmas,
                    seeds=seeds, w_ee_mult=W_EE_MULT, b_a=B_A,
                    ell_par=ELL_PAR, ell_perp=ELL_PERP, l_inh=L_INH),
        region_summary=region_summary,
        per_drive=[{k: v for k, v in r.items() if k != "cells"} for r in results],
    )
    (OUT / f"{tag}.json").write_text(json.dumps(payload, indent=2, default=float))
    with (OUT / f"{tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drive", "sigma", "seed", "label", "reason", "n_events",
                    "n_events_total", "all_returned", "longest_on_ms", "max_ext",
                    "frac_time_on", "event_rate_per_s"])
        for c in all_cells:
            w.writerow([c["drive"], c["sigma"], c["seed"], c["label"], c["reason"],
                        c["n_events"], c["n_events_total"], c["all_returned"],
                        f"{c['longest_on_ms']:.1f}", f"{c['max_ext']:.4f}",
                        f"{c['frac_time_on']:.4f}", f"{c['event_rate_per_s']:.4f}"])
    print(f"\nWrote {OUT}/{tag}.json + .csv  [total {time.time()-t0:.0f}s]")
    print(f"REGION (§10.4): {region_summary['n_accepted_cells']} accepted cells; "
          f"robust_2d_block={region_summary['robust_2d_block']}; "
          f"σ0_extinction_all={region_summary['sigma0_extinction_all_drives']}; "
          f"undetectable_drives={region_summary['undetectable_drives']}")
    print(f"accepted cells (drive,σ): {region_summary['accepted_cells']}")


if __name__ == "__main__":
    main()

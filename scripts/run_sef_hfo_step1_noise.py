"""Step 1 seed x amp noise grid (contract §3) on the canonical LIF rate field.

For each candidate window x sigma_noise x seed: drive the field with PURE OU noise
(no designed pulse), classify the run, and report the FRACTION of seeds producing
discrete self-terminating events at each sigma (not lucky-seed existence). Also
emits the per-cell event rate for the trigger-rate curve (contract §4).

Windows (contract §3):
  A         w_ee_mult=1.0, recovery off  (Brunel-like, PRIMARY)
  B         w_ee_mult=1.4, recovery on   (pathological gain, SENSITIVITY)
  B_ablate  w_ee_mult=1.4, recovery off  (recovery-essential failure control)

Run: python scripts/run_sef_hfo_step1_noise.py --window A
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sef_hfo_lif import ELL_PAR, ELL_PERP, L_INH, integrate_lif_field, mean_field
from src.sef_hfo_events import classify_run, make_ou_noise

OUT = Path("results/topic4_sef_hfo/step1_noise")
N, L, DT, T_RUN = 64, 16.0, 0.25, 5000.0
# Slow noise (default 100 ms): the OU on mu represents the SLOW afferent input
# component, NOT the fast synaptic fluctuations already inside Phi_LIF's sigma
# (tau_noise=5ms ~= TAU_AMPA was confounded -> dense ignition -> sustained, §9.1).
TAU_NOISE = 100.0
# Fine sigma grid around the discrete band found at sigma~2.0 (probe §9.3), to
# measure band WIDTH (risk #6: a single sigma point fails; a pass needs a band).
SIGMAS = [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
SEEDS = [0, 1, 2, 3, 4]

WINDOWS = {
    "A": dict(w_ee_mult=1.0, b_a=0.0, tau_a=80.0, cls="A"),
    "B": dict(w_ee_mult=1.4, b_a=2000.0, tau_a=25.0, cls="B"),
    "B_ablate": dict(w_ee_mult=1.4, b_a=0.0, tau_a=25.0, cls="B"),
}


def run_cell(op, win, sigma, seed, tau_noise):
    stim = make_ou_noise(N, L, DT, sigma_noise=sigma, tau_noise=tau_noise, seed=seed)
    # coh_len=ELL_PAR -> detection on the COHERENCE active-fraction (speckle-robust,
    # contract §1 v1.1). return_field for the window-B capture check.
    ext, _front, rE, coh = integrate_lif_field(
        op, stim, dt=DT, t_max=T_RUN, b_a=win["b_a"], tau_a=win["tau_a"],
        n=N, L=L, ell_par=ELL_PAR, ell_perp=ELL_PERP, l_inh=L_INH,
        return_field=True, coh_len=ELL_PAR,
    )
    res = classify_run(coh, DT, op, rE_final=rE, window=win["cls"])
    res["event_rate_per_s"] = res["n_events"] / (T_RUN / 1000.0)
    res["raw_ext_max"] = float(ext.max())  # keep raw for speckle/coherence comparison
    res.pop("events", None)  # drop per-event detail from the grid summary
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=list(WINDOWS), default="A")
    ap.add_argument("--tau-noise", type=float, default=TAU_NOISE,
                    help="OU noise correlation time (ms); default 100 (slow afferent component)")
    args = ap.parse_args()
    win = WINDOWS[args.window]
    tau_noise = args.tau_noise
    tag = f"{args.window}_tau{int(tau_noise)}"
    op = mean_field(1.0, w_ee_mult=win["w_ee_mult"])

    cells = []
    per_sigma = []
    for sigma in SIGMAS:
        labels = []
        rates = []
        for seed in SEEDS:
            r = run_cell(op, win, sigma, seed, tau_noise)
            cells.append(dict(window=args.window, sigma=sigma, seed=seed, **r))
            labels.append(r["label"])
            if r["label"] == "discrete_events":
                rates.append(r["event_rate_per_s"])
            print(f"  [{args.window}] sigma={sigma:<4} seed={seed} -> {r['label']:<16} "
                  f"n_ev={r['n_events']:<3} max_ext={r['max_ext']:.3f} rate={r['event_rate_per_s']:.3f}/s")
        from collections import Counter
        counts = dict(Counter(labels))
        frac_discrete = labels.count("discrete_events") / len(labels)
        per_sigma.append(dict(sigma=sigma, frac_discrete=frac_discrete, regime_counts=counts,
                              mean_rate_discrete=(float(np.mean(rates)) if rates else None)))
        print(f"  == sigma={sigma}: frac_discrete={frac_discrete:.2f} counts={counts}")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        window=args.window, tau_noise=tau_noise,
        params=dict(N=N, L=L, dt=DT, t_run_ms=T_RUN, sigmas=SIGMAS, seeds=SEEDS,
                    tau_noise=tau_noise, w_ee_mult=win["w_ee_mult"], b_a=win["b_a"], tau_a=win["tau_a"],
                    ell_par=ELL_PAR, ell_perp=ELL_PERP, l_inh=L_INH),
        op=dict(nuE=op["nuE"], n_clean_roots=op["n_clean_roots"],
                roots=[r["nuE"] for r in op["roots"]]),
        per_sigma=per_sigma, cells=cells,
    )
    (OUT / f"grid_{tag}.json").write_text(json.dumps(payload, indent=2, default=float))
    # flat CSV
    import csv
    with (OUT / f"grid_{tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "sigma", "seed", "label", "n_events", "max_ext",
                    "final_ext", "longest_on_ms", "captured_high", "event_rate_per_s"])
        for c in cells:
            w.writerow([c["window"], c["sigma"], c["seed"], c["label"], c["n_events"],
                        f"{c['max_ext']:.4f}", f"{c['final_ext']:.4f}", f"{c['longest_on_ms']:.1f}",
                        c["captured_high"], f"{c['event_rate_per_s']:.4f}"])
    print(f"\nWrote {OUT}/grid_{tag}.json + .csv")
    print("per-sigma fraction discrete:",
          {s["sigma"]: round(s["frac_discrete"], 2) for s in per_sigma})


if __name__ == "__main__":
    main()

"""Step 1 S1.5/S1.6: direction readout + negative control on NOISE-DRIVEN events.

Load-bearing discriminator (contract §5/§6, carried from static 0d into the noisy
regime): noise-triggered events' systematic elongation axis must track the E->E
connectivity anisotropy axis theta_EE, NOT geometry; an ISOTROPIC E->E kernel must
produce NO preferred axis.

Method (contract §5 AGGREGATION — a single noise event's shape is dominated by its
triggering fluctuation, so single-event axes are meaningless; v1 used one peak
field and failed = method artifact, §9.5). Here we accumulate the centered
second-moment tensor of the smoothed active region over ALL event-on frames
(integrate_lif_field axis_accum), summed across seeds, per theta_EE. The principal
eigenvector of the aggregate tensor = the systematic event elongation axis; random
per-event asymmetry averages out.

Operating point = window A (recovery off), discrete band (sigma=2.0, tau=100ms).

Run: python scripts/run_sef_hfo_step1_direction.py
"""
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sef_hfo_lif import ELL_PAR, ELL_PERP, integrate_lif_field, mean_field
from src.sef_hfo_events import make_ou_noise

OUT = Path("results/topic4_sef_hfo/step1_noise")
N, L, DT, T = 64, 16.0, 0.25, 5000.0   # run -> many event-on frames (aggregate is frame-dominated)
SIGMA, TAU = 2.0, 100.0
SEEDS = [0, 1, 2]


def accum_tensor(theta_EE_deg, ell_par, ell_perp, seed):
    op = mean_field(1.0)
    stim = make_ou_noise(N, L, DT, sigma_noise=SIGMA, tau_noise=TAU, seed=seed)
    *_, axis = integrate_lif_field(
        op, stim, dt=DT, t_max=T, theta_EE=np.radians(theta_EE_deg),
        ell_par=ell_par, ell_perp=ell_perp, n=N, L=L, coh_len=ELL_PAR, axis_accum=True,
    )
    return axis


def axis_of(tensor):
    C = np.array([[tensor["Sxx"], tensor["Sxy"]], [tensor["Sxy"], tensor["Syy"]]])
    if not np.all(np.isfinite(C)) or tensor["n_onframes"] == 0:
        return float("nan"), 1.0
    ev, evec = np.linalg.eigh(C)
    v = evec[:, -1]
    return float(np.degrees(np.arctan2(v[1], v[0])) % 180.0), float(np.sqrt(ev[-1] / max(ev[0], 1e-12)))


def axis_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def sum_tensors(ts):
    return dict(Sxx=sum(t["Sxx"] for t in ts), Sxy=sum(t["Sxy"] for t in ts),
                Syy=sum(t["Syy"] for t in ts), n_onframes=sum(t["n_onframes"] for t in ts))


def main():
    aniso, iso = [], []
    for theta in [0.0, 60.0, 90.0]:
        ts = [accum_tensor(theta, ELL_PAR, ELL_PERP, s) for s in SEEDS]
        agg = sum_tensors(ts)
        ang, ratio = axis_of(agg)
        row = dict(theta_EE=theta, theta_prop=ang, axis_err_deg=axis_diff(ang, theta),
                   ratio=ratio, n_onframes=agg["n_onframes"])
        aniso.append(row)
        print(f"  [aniso] theta_EE={theta:5.0f} theta_prop={ang:6.1f} "
              f"axis_err={row['axis_err_deg']:5.1f} ratio={ratio:.2f} n_onframes={agg['n_onframes']}", flush=True)
    ts = [accum_tensor(0.0, 0.4, 0.4, s) for s in SEEDS]  # isotropic E->E
    agg = sum_tensors(ts)
    ang, ratio = axis_of(agg)
    iso.append(dict(theta_EE="isotropic", theta_prop=ang, ratio=ratio, n_onframes=agg["n_onframes"]))
    print(f"  [iso]   theta_prop={ang:6.1f} ratio={ratio:.2f} n_onframes={agg['n_onframes']}", flush=True)

    aniso_ok = all(np.isfinite(r["theta_prop"]) and r["axis_err_deg"] < 20.0 and r["ratio"] > 1.3
                   for r in aniso)
    iso_fail = all(r["ratio"] < 1.3 for r in iso)
    verdict = dict(aniso_tracks_theta_EE=bool(aniso_ok), isotropic_no_axis=bool(iso_fail),
                   discriminator_passed=bool(aniso_ok and iso_fail))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "direction_A.json").write_text(json.dumps(
        dict(params=dict(N=N, L=L, sigma=SIGMA, tau_noise=TAU, t_run_ms=T, seeds=SEEDS,
                         ell_par=ELL_PAR, ell_perp=ELL_PERP, method="aggregated_second_moment"),
             anisotropic=aniso, isotropic=iso, verdict=verdict), indent=2, default=float))
    print("=== anisotropic E->E: aggregate event axis tracks theta_EE? (summed over seeds) ===")
    for r in aniso:
        print(f"  theta_EE={r['theta_EE']:5.0f} theta_prop={r['theta_prop']:6.1f} "
              f"axis_err={r['axis_err_deg']:5.1f} ratio={r['ratio']:.2f} n_onframes={r['n_onframes']}")
    print("=== isotropic control (must produce NO axis) ===")
    for r in iso:
        print(f"  theta_prop={r['theta_prop']:6.1f} ratio={r['ratio']:.2f} n_onframes={r['n_onframes']}")
    print(f"DISCRIMINATOR PASSED = {verdict['discriminator_passed']} "
          f"(aniso tracks={aniso_ok}, iso fails={iso_fail})")


if __name__ == "__main__":
    main()

"""Operating-point sweep: find the synchronous-irregular (oscillatory) band."""
import argparse, numpy as np
from scipy import signal
from params import Params, compute_nu_theta
from model import build_network, simulate


def osc_metrics(rate_hz, dt_ms):
    fs = 1000.0 / dt_ms
    x = rate_hz - rate_hz.mean()
    nper = min(len(x), 1024)
    f, P = signal.welch(x, fs=fs, nperseg=nper)
    band = (f >= 10) & (f <= 45)
    if not band.any():
        return float("nan"), float("nan")
    pk = np.argmax(P[band])
    fpk = f[band][pk]
    prom = P[band][pk] / np.median(P[(f >= 5) & (f <= 80)])
    return fpk, prom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=3.6)
    ap.add_argument("--ratios", type=str, default="0.6,0.8,0.9,1.0,1.1,1.3")
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=4000.0)
    ap.add_argument("--T", type=float, default=600.0)
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    print(f"g={args.g}  L={args.L}  density={args.density}  T={args.T}")
    print(f"{'ratio':>6} {'rateE':>7} {'rateI':>7} {'fpeak':>7} {'promin':>8}")
    p0 = Params(g=args.g, L=args.L, density=args.density, T=args.T)
    net = build_network(p0, verbose=False)
    for r in ratios:
        p = Params(g=args.g, L=args.L, density=args.density, T=args.T, nu_ext_ratio=r)
        res = simulate(p, net, verbose=False)
        dt = p.dt
        k = int(round(1.0 / dt))
        n = (len(res["rate_E"]) // k) * k
        rE = res["rate_E"][:n].reshape(-1, k).mean(1)
        fpk, prom = osc_metrics(rE, 1.0)
        print(f"{r:>6.2f} {res['rate_E'].mean():>7.2f} {res['rate_I'].mean():>7.2f} "
              f"{fpk:>7.1f} {prom:>8.1f}")


if __name__ == "__main__":
    main()

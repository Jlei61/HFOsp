"""Full active-fraction + front-radius time series for one kick-from-quiet-rest event.

Reuses kick_probe.simulate_kick (the verbatim localized-kick mechanism). Records,
per 2-ms bin over the whole run, the fraction of E that spiked and the 90th-pct
radius of spiking E from the kick centre — i.e. the rise (travelling front) and the
return to baseline (self-termination). L=3 / density 1800 matches the onset-front
anisotropy run, so the event stays sub-global (cleaner self-limit, not finite-size).

Writes ../data/kick_timeseries.json (panel c of the spiking-validation supplementary).
Run: python src/snn_engine/kick_timeseries.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import Params, compute_nu_theta          # noqa: E402
from model import build_network                       # noqa: E402
from kick_probe import simulate_kick, T_KICK, DUR_KICK  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kick_timeseries.json")
BIN = 2.0


def main():
    p = Params(g=3.6, L=3.0, density=1800.0, T=340.0, nu_ext_ratio=0.6, seed=1)
    nu_theta = compute_nu_theta(p)[0]
    net = build_network(p, verbose=False)
    net["rng"] = np.random.default_rng(p.seed)
    res = simulate_kick(p, net, KICK_BOOST=2.0 * nu_theta)

    E = res["E_spk_bool"]                       # (nsteps, NE)
    NE = res["NE"]
    posE = net["pos"][:NE]
    center = np.array([p.L / 2.0, p.L / 2.0])
    dt = p.dt

    t_ms, frac, rad = [], [], []
    for b0 in np.arange(100.0, p.T, BIN):
        i0, i1 = int(round(b0 / dt)), int(round((b0 + BIN) / dt))
        m = E[i0:i1].any(axis=0)
        t_ms.append(float(b0 + BIN / 2.0))
        frac.append(float(m.mean()))
        if m.sum() >= 5:
            d = np.linalg.norm(posE[m] - center, axis=1)
            rad.append(float(np.percentile(d, 90.0)))
        else:
            rad.append(float("nan"))

    out = dict(t_ms=t_ms, active_frac=frac, radius_mm=rad,
               kick_on=[float(T_KICK), float(T_KICK + DUR_KICK)],
               L=p.L, drive=p.nu_ext_ratio, g=p.g, density=p.density, seed=p.seed,
               peak_active_frac=float(max(frac)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.normpath(OUT)}  peak_frac={max(frac):.3f}  "
          f"max_radius={np.nanmax(rad):.2f}mm")


if __name__ == "__main__":
    main()

"""B-side finding (spec 2026-06-08 §6/§8): a WIDE threshold field (std 1.5
everywhere — the rate model's "clean" value) SPONTANEOUSLY BURSTS at drive 0.6
(its low-threshold tail self-ignites), while the scalar (homogeneous) net stays
quiet. The SNN sees this finite-size destabilization that the rate mean-field
structurally averages away. kick-OFF (no stimulus) comparison.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENGINE = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENGINE)
from params import Params, compute_nu_theta             # noqa: E402
from connectivity import place_neurons                  # noqa: E402
from connectivity_rot import build_connectivity_rot     # noqa: E402
from kick_probe import simulate_kick                     # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_heterogeneity import sample_threshold_fields   # noqa: E402
from src.sef_hfo_snn_engine_guard import assert_versions        # noqa: E402

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")


def main():
    assert_versions(json.loads((OUT / "engine_versions.json").read_text()))
    p = Params(g=3.6, L=3.0, density=1800.0, T=450.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(45.0), AR=2.0)
    dt = p.dt; t = np.arange(int(p.T / dt)) * dt; N = NE + NI
    # wide EVERYWHERE = surround_std 1.5 + wide core (= the original broken design)
    wide = sample_threshold_fields(net["pos"], labels == 0, (1.5, 1.5), 0.5,
                                   np.random.default_rng(1), surround_std=1.5)["baseline"]
    scalar = np.full(N, 18.0)

    def rate_off(vth):
        net["rng"] = np.random.default_rng(p.seed)
        return simulate_kick(p, net, KICK_BOOST=0.0, kick_center=[2.4, 1.5],
                             V_th_per_neuron=vth)["rate_E"]

    rS, rW = rate_off(scalar), rate_off(wide)

    def pk(r):
        i = int(r.argmax()); return float(r[i]), float(t[i])

    res = dict(scalar_peak_Hz=pk(rS)[0], scalar_peak_t_ms=pk(rS)[1],
               wide_peak_Hz=pk(rW)[0], wide_peak_t_ms=pk(rW)[1],
               note="kick-OFF (no stimulus); wide=std 1.5 everywhere, scalar=18; drive 0.6")
    (OUT / "spontaneous_probe.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, rS, color="0.5", lw=1.0, label="scalar V_th (homogeneous) — quiet")
    ax.plot(t, rW, color="C3", lw=1.0,
            label="wide V_th std=1.5 everywhere — spontaneous bursts")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("E rate (Hz)")
    ax.set_title("No-kick (spontaneous): wide threshold heterogeneity destabilizes the "
                 "quiet point\n(SNN sees finite-size synchronization the rate mean-field "
                 "averages away)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "spontaneous_probe.png", dpi=130); plt.close(fig)
    print("scalar:", pk(rS), "wide:", pk(rW))


if __name__ == "__main__":
    main()

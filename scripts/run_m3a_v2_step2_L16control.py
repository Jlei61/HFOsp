"""L-invariance control for Step 2 (review P1-3): the kicked axial event spans the FULL axis
(axis_reach_frac ~1) at L=10 AND L=16, so 'expanded axial' has no axial room regardless of sheet
size -- the geometric half of the Step-2 negative is auditable, not just asserted. slow vars OFF
(baseline only). Writes L16_control.json. DESCRIPTIVE.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools                                          # noqa: E402
import json                                               # noqa: E402
import multiprocessing as mp                              # noqa: E402
import sys                                                # noqa: E402

import numpy as np                                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m3a_v2_step2_qI as S2                          # noqa: E402
from kick_probe import simulate_kick                      # noqa: E402

OUT = os.path.join(ROOT, "results", "topic4_m3a_v2_step2_qI")


def baseline_at(task):
    sub_id, seed, L = task
    S = S2.build(S2.SUBSTRATES[sub_id], seed, L=L)
    S["net"]["rng"] = np.random.default_rng(seed)
    res = simulate_kick(S["p"], S["net"], KICK_BOOST=S2.KICK, slow=None, kick_center=S["core_xy"],
                        r_kick=0.3, t_kick=S2.T_KICK, V_th_per_neuron=S["vth"])
    r = S2._readout(res, S)
    return dict(substrate=sub_id, seed=seed, L=L, NE=S["NE"], R_area=r["R_area"], S_axis=r["S_axis"],
                F_off=r["F_off"], axis_reach_frac=r["axis_reach_frac"],
                corridor_fill_frac=r["corridor_fill_frac"], returned=r["returned"])


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = list(itertools.product(["primary", "sensitivity", "backup"], [1, 2], [10.0, 16.0]))
    with mp.Pool(min(12, len(tasks))) as pool:
        rows = pool.map(baseline_at, tasks)
    json.dump(dict(meta=dict(purpose="L-invariance of axial reach (Step 2 geometry control)",
                             note="axis_reach_frac~1 at both L => event spans full axis => no axial "
                                  "room to expand; corridor_fill_frac~1 => corridor saturated"),
                   rows=rows), open(os.path.join(OUT, "L16_control.json"), "w"), indent=2)
    print("L-invariance control (axis_reach_frac should be ~1 at BOTH L):")
    for r in sorted(rows, key=lambda r: (r["substrate"], r["seed"], r["L"])):
        print(f"  {r['substrate']:11} s{r['seed']} L={r['L']:>4}: reach={r['axis_reach_frac']} "
              f"corridor_fill={r['corridor_fill_frac']} R={r['R_area']} S={r['S_axis']} NE={r['NE']}")
    print(f"\nwrote {os.path.join(OUT, 'L16_control.json')}")


if __name__ == "__main__":
    main()

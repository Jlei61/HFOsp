"""Bounded follow-up of the unresolved branch gap; preserve the previous audit."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from topic4_fig5_D_physical_model import Equilibrium, OUT as SOURCE
import topic4_fig5_z_bifurcation_preview as continuation

OUT = SOURCE / 'gap_and_fold_focus_20260916'
OUT.mkdir(exist_ok=True)

def trace(branch):
    seed = np.load(SOURCE / f'q1.25_{branch}.npz')
    continuation.OUT = OUT
    continuation.trace_branch(
        Equilibrium(1.25), seed['r_hz'][-1], float(seed['s'][-1]), 1,
        f'q1.25_{branch}_followup', max_points=180, lower_bound=0,
        upper_bound=1, step_max=.035, initial_tangent=seed['tangent'][-1],
        trust_curvature=True)

def inspect():
    eq = Equilibrium(1.25); m = eq.m
    folds = json.loads((SOURCE / 'new_stationary_folds.json').read_text())
    states=[]; rows=[]
    for index, item in enumerate(folds):
        state = np.load(SOURCE / (item['name'] + '.npz'))
        r=state['r_hz']; v=state['right_mode'][:400]
        energy=m.count_e*v*v; energy/=energy.sum()
        order=np.argsort(energy)[::-1]
        k95=int(np.searchsorted(np.cumsum(energy[order]), .95)+1)
        peak=int(order[0]); rates=r[:400]
        row=dict(item, number=index+1, mode_energy_per_cell=energy.tolist(),
                 mode_95_percent_cells=k95,
                 mode_peak_xy_mm=[peak%20+.5,peak//20+.5],
                 cell_rate_hz=rates.tolist(),
                 core_A_rate_hz=float(m.region_rate(rates,'175_0')),
                 core_B_rate_hz=float(m.region_rate(rates,'175_1')),
                 fraction_E_cells_above_10_Hz=float(np.sum(m.count_e[rates>10])/m.count_e.sum()),
                 nearest_Z_path_knot_distance=float(np.min(abs(eq.ss-item['D']))))
        rows.append(row);states.append(r)
    pairs=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            pairs.append(dict(a=i+1,b=j+1,D_difference=abs(rows[i]['D']-rows[j]['D']),
                 E_field_rms_difference_hz=float(np.sqrt(np.average((states[i][:400]-states[j][:400])**2,weights=m.count_e))),
                 full_state_max_difference_hz=float(np.max(abs(states[i]-states[j])))))
    (OUT/'fold_spatial_audit.json').write_text(json.dumps(dict(folds=rows,pairwise_state_distances=pairs),indent=2)+'\n')
    for r in rows:
        print(r['number'],r['name'],r['D'],r['mean_e_hz'],r['mode_95_percent_cells'],r['mode_peak_xy_mm'],r['nearest_Z_path_knot_distance'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['trace','inspect']);p.add_argument('--branch',choices=['middle_extension','high']);a=p.parse_args()
    if a.mode=='trace':trace(a.branch)
    else:inspect()

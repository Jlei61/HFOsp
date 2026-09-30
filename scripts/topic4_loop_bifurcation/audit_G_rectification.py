#!/usr/bin/env python3
"""Read-only activation-corner diagnostic at complete recorded windows.

One-millisecond samples are not the original 0.1 ms update budget. Incomplete
native trajectories are explicitly prefixes; their 20--30 s result is omitted.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import time
import numpy as np
from campaign import ROOT, read, write, sha
from analyze_native import original

OUT = ROOT / 'G_activation_rectification'


def summarize(t, r, g, lo, hi):
    m = (t >= lo-1e-8) & (t < hi-1e-8)
    assert m.sum() == round((hi-lo)*1000)
    rr, gg = r[m], g[m]
    q = np.clip((rr-200)/300, 0., 1.)
    meanrate_G = 30*float(np.clip((rr.mean()-200)/300, 0., 1.))
    sampled_target = 30*float(q.mean())
    # Endpoint term uses first/last available sample, so covers T-1ms.
    boundary = .5*(gg[-1]-gg[0])/(hi-lo-.001)
    return dict(interval_s=[lo, hi],samples=int(m.sum()),mean_R_Hz=float(rr.mean()),
        sd_R_Hz=float(rr.std()),min_R_Hz=float(rr.min()),max_R_Hz=float(rr.max()),
        fraction_R_below200=float((rr<200).mean()),fraction_R_above500=float((rr>500).mean()),
        mean_Graw=float(gg.mean()),G_of_mean_R=meanrate_G,
        sampled_mean_G_target=sampled_target,activation_rectification=sampled_target-meanrate_G,
        approximate_G_boundary_term=boundary,
        approximate_filter_balance_residual=float(gg.mean())-sampled_target+boundary)


def main():
    OUT.mkdir(exist_ok=True)
    jobs=[('exit_return_probes', 'exit_z0.21_k9_fields16p7_high')]
    jobs += [('native_exit_K_bracket', f'exit_z0.21_k{k}_fields16p7_high') for k in ['9.35', '9.5']]
    rows=[]
    for rootname,name in jobs:
        root=ROOT/rootname;folder=root/'runs'/name;job=read(root/'jobs'/f'{name}.json')
        d=original.load(folder/'mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
        t=d['time_ms']/1000-job['branch_start_s']
        assert np.allclose(np.diff(t),.001,rtol=0,atol=1e-10)
        complete=(folder/'result.json').exists() and read(folder/'result.json')['status']=='COMPLETE'
        end=round(float(t[-1]+.001),6)
        windows=[summarize(t,d['global_E_rate_Hz'],d['global_raw_conductance_ratio'],lo,hi)
                 for lo,hi in [(0,1),(1,5),(5,10),(10,20),(20,30)] if end>=hi]
        rows.append(dict(source='native',name=name,complete_30s=complete,recorded_prefix_s=end,windows=windows))
    path=ROOT/'carried_exit_lower_holds/analysis/held_K9p35.npz'
    with np.load(path) as z:
        # Density observations are right endpoints. Align bin labels to left
        # endpoints for selecting exactly 3000 samples of its terminal3s.
        t=z['time_s']-.001;end=round(float(t[-1]+.001),6)
        rows.append(dict(source='density_carried_hold',name='held_K9p35',complete_trajectory=True,
            windows=[summarize(t,z['causal_R_Hz'],z['Graw'],end-3,end)]))
    result=dict(status='COMPLETE_RECORDED_WINDOW_DIAGNOSTIC',rows=rows,updated_epoch=time.time(),
        sampling='1ms R/G samples, not exact0.1ms native updates. Filter boundary/balance are approximate. Means clipped outside200-500 require using mean q, not q(meanR).',
        scope='A native prefix is not a final30s state. Rectification alone cannot establish causal closure error, stability, or a bifurcation.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False)
    write(OUT/'result.json',result)
    print([(r['name'],r['windows'][-1]) for r in rows],flush=True)


if __name__=='__main__':main()

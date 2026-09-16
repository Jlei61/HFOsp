#!/usr/bin/env python3
"""Verify when global input alone forces every E resource to decay.

For each5ms observed Rg, its exponential decay is a lower bound until the next
observation because intervening spikes only add nonnegative increments. This
permits a conservative interval test without interpolating unobserved spikes.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import analyze_topic4_autonomous_recovery as common

def main(root):
    rows=[]
    for folder in sorted((root/'runs').glob('pool*')):
        if not (folder/'chunks').exists():continue
        job=common.read(root/'jobs'/(folder.name+'.json'))
        a=common.load(folder,['slow_time_ms','Z'])
        if a is None:continue
        q={k:[] for k in ['time_ms','rate_Hz']}
        for path in sorted((folder/'pool_chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as v:
                for k in q:q[k].append(v[k])
        q={k:np.concatenate(v) for k,v in q.items()};t=q['time_ms'];r=q['rate_Hz']
        if len(t)<2:continue
        assert np.allclose(np.diff(t),5.)
        bound=job['pool_gain']*np.maximum(r[:-1]*np.exp(-5/(job['pool_tau_s']*1000))-job['pool_threshold_Hz'],0)
        forced=bound>=job['threshold']
        edges=np.diff(np.r_[False,forced,False].astype(int))
        intervals=[dict(start_s=float(t[lo]/1000),end_s=float(t[hi]/1000),duration_s=float((t[hi]-t[lo])/1000))
            for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if t[hi]-t[lo]>=100]
        ts=a['slow_time_ms'];z=a['Z'][:,[0,5,6]]
        checked=[];errors=[]
        for i in range(len(ts)-1):
            lo=int(round(ts[i]/5));hi=int(round(ts[i+1]/5))
            if hi>=len(t) or not forced[lo:hi].all():continue
            assert np.isclose(t[lo],ts[i]) and np.isclose(t[hi],ts[i+1])
            nsteps=round((ts[i+1]-ts[i])/.1)
            expected=z[i]*(1-.1/(job['tau_Z_s']*1000))**nsteps
            errors.append(np.max(np.abs(z[i+1]-expected)));checked.append(float(ts[i]/1000))
        maximum=float(max(errors)) if errors else None
        if errors and maximum>1e-10:raise AssertionError((folder.name,'Expected exact native Euler depletion',maximum))
        row=dict(name=folder.name,observed_s=float(ts[-1]/1000),
            sufficient_Rg_threshold_Hz=job['pool_threshold_Hz']+job['threshold']/job['pool_gain'],
            guaranteed_global_only_depletion_intervals=intervals,
            checked_20ms_intervals=len(checked),maximum_mean_core_A_B_Euler_decay_error=maximum,
            check_status='PASS' if errors else 'NO_GUARANTEED_INTERVAL',
            interpretation='Within these intervals the added global current alone exceeds Ith, so every E Zi must decay despite any reduction of firing. This identifies the inhibitory-resource coupling; it does not establish that a future quiet interval or recovery is impossible.')
        common.write(folder/'global_depletion_coupling.json',row);rows.append(row)
    common.write(root/'global_depletion_coupling.json',dict(updated_at=time.time(),rows=rows,
        method='Conservative5ms causal bound on Rg using nonnegative spike increments. Verify mean Z and each core against the native dt=.1ms Euler recurrence over every enclosed20ms interval. No physics or state threshold modified.'))
    print([(r['name'],r['checked_20ms_intervals'],r['maximum_mean_core_A_B_Euler_decay_error']) for r in rows])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=common.OUT/'activity_global_pool_round3');a=p.parse_args();main(a.root)

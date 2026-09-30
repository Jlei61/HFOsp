#!/usr/bin/env python3
"""Exact recorded counterfactual budgets and between-sample gate bounds.

The20--29s interval has R samples at both endpoints, so the positive-spike
filter bounds cover the entire interval, not an unrecorded terminal millisecond.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_actual_G_history import load


def main():
    root=ROOT/'native_exit_K_bracket';out=root/'unclamped_flow'
    assert read(root/'extended_analysis_summary.json')['status']=='COMPLETE'
    with np.load(root/'geometry.npz') as z:
        distance=np.linalg.norm(z['positions_e'][:,None]-z['centers_mm'][None],axis=2)
        group=np.full(32000,2);group[distance[:,0]<1.75]=0
        group[(distance[:,1]<1.75)&(distance[:,1]<distance[:,0])]=1
        assert np.array_equal(np.bincount(group,minlength=3),z['region_counts'][:3])
    masks=[np.ones(32000,bool)]+[group==i for i in range(3)];rows=[]
    for K in [9.35,9.5]:
        name=f'exit_z0.21_k{K:g}_fields16p7_high';d=load(root,name)
        with np.load(root/'runs'/name/'held_fields.npz') as z:
            meansZ=np.array([z['Z'][m].mean() for m in masks]);meansK=np.array([z['K'][m].mean() for m in masks])
        endpoints=(d['time1']>=20-1e-9)&(d['time1']<=29+1e-9)
        assert endpoints.sum()==9001
        rates=d['R'][endpoints]
        upper=float(rates.max()*np.exp(.001/.015));lower=float(rates.min()*np.exp(-.001/.015))
        assert upper<200,'Only a q=0 interval supports the analytic no-jump K decay check'
        assert lower>5 or upper<5,'No claim across an unresolved K timescale crossing'
        tau=.5 if lower>5 else 5.
        expected=meansK*np.expm1(-.0001/tau)/.0001
        md=(d['drift_time']>20+1e-9)&(d['drift_time']<=29+1e-9);assert md.sum()==450
        drift=d['drift'][md].mean(0);err=float(abs(drift[:,1]-expected).max())
        assert err<1e-9
        eligible=meansZ+5*drift[:,0];assert eligible.min()>-1e-12 and eligible.max()<1+1e-12
        rows.append(dict(name=name,K=K,interval_s=[20,29],
            causal_R_between_samples_bound_Hz=[lower,upper],q_zero_throughout_interval=True,
            K_decay_tau_s=tau,mean_Z_allE_A_B_surround=meansZ.tolist(),
            exact_budget_mean_dZ_dK_per_s=drift.tolist(),
            implied_mean_Z_recovery_eligible_fraction=eligible.tolist(),
            analytic_no_jump_K_flow_per_s=expected.tolist(),maximum_K_budget_error=err))
    write(out/'result.json',dict(status='COMPLETE',rows=rows,
        bound='Causal R decays exponentially15ms with only nonnegative spike jumps. Between neighboring1ms samples use previousR*exp(-1/15) and nextR*exp(1/15). Recorded20ms budgets themselves integrate original0.1ms updates.',
        interpretation='Z/K are held. These are evaluations of their original unclamped vector field, not an actual release trajectory. Positive allE recovery does not imply positive core recovery. Neither conditional state is a fixed point of the full autonomous system.',
        producer_sha256=sha(__file__),counts_as_autonomous_loop=False,formal_bifurcation_allowed=False))
    print(rows,flush=True)


if __name__=='__main__':main()

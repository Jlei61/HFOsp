#!/usr/bin/env python3
"""Descriptive threshold-current budgets along the completed K ramps.

Observed synaptic/M/K group means are used. G is sampled after its update;
its one-step alignment error is explicitly bounded rather than hidden.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from continue_target_high_state import OUT
from coupled_density_exit import ADAPTED

dest=OUT/'current_budget';dest.mkdir(exist_ok=True)
assert not (dest/'analysis.json').exists()
geo=dict(np.load(ADAPTED/'geometry.npz'));E=geo['population']==0;size=geo['group_size'];reg=geo['group_region']
masks=[E]+[E&(reg==j) for j in range(3)];theta=geo['threshold_mv'];rows=[]
names=['excitatory_input','local_inhibition','M','global_G','K','leak']
write(dest/'contract.json',dict(question='What changes the core input balance as the prescribedK trajectory loses recruitment?',
    method='Descriptive posthoc decomposition of mean current evaluated at each group threshold. Synaptic input is measured from the free density trajectory, not replaced with instantaneous W*r. This is not a causal ablation, native measurement or a bifurcation test.',
    timing='Stored G is after the last0.1ms global update, whereas voltage uses preupdateG. Bound its error using0<=q<=1; M sampledafterupdate has a separate per-step bound. Z/K are clamped as recorded.',
    producer_sha256=sha(__file__)))
for name in ['ramp2s_K10p5','ramp10s_K10p5']:
    with np.load(OUT/name/'trajectory.npz') as z:
        v=z['group_output'];G=30*z['global_s'];t=z['elapsed_time_ms']/1000
    components=np.empty((len(t),len(names),4));bounds=np.empty((len(t),4))
    for j,m in enumerate(masks):
        def avg(x):return np.average(x,weights=size[m],axis=1)
        components[:,0,j]=avg(v[:,4,m])
        components[:,1,j]=-avg(v[:,5,m])
        components[:,2,j]=-.0005*avg(v[:,2,m])
        leverage=avg(v[:,1,m]*(theta[m][None,:]+17.662847938268442))
        components[:,3,j]=-G*leverage
        components[:,4,j]=-avg(v[:,3,m]*(theta[m][None,:]+30.))
        components[:,5,j]=-np.average(theta[m],weights=size[m])
        # Exact bounds for differences from the last-step preupdate values.
        bounds[:,j]=30*np.expm1(.1/500)*leverage + .0005*(.0001*avg(v[:,2,m])+1)/.9999
    with np.load(OUT/'analysis'/f'{name}.npz') as z:
        rate=z['rate_Hz'];K=z['K'];drift=z['drift_per_s']
    event=read(OUT/'analysis'/f'{name}.json')['both_cores_sustained_low']['first_low_interval_s'][0]
    windows=[]
    for label,lo,hi in [('start',0,.02),('approaching_core_exit',event-.2,event-.1),('first_joint_core_low',event,event+.02),('late_hold',t[-1]-1,t[-1])]:
        m=(t>lo)&(t<=hi);assert m.any()
        windows.append(dict(label=label,window_s=[lo,hi],mean_K=float(K[m].mean()),
            mean_rates_allE_A_B_surround_Hz=rate[m].mean(0).tolist(),
            threshold_current_components_allE_A_B_surround_mV_equiv={n:components[m,i].mean(0).tolist() for i,n in enumerate(names)},
            net_threshold_current_mV_equiv=components[m].sum(1).mean(0).tolist(),
            bound_on_postupdate_alignment_error_mV_equiv=bounds[m].mean(0).tolist(),
            counterfactual_Z_drift_per_s=drift[m].mean(0).tolist()))
    np.savez_compressed(dest/f'{name}.npz',time_s=t,K=K,components=components,names=names,alignment_error_bound=bounds)
    rows.append(dict(name=name,windows=windows))
    print(name,[(w['label'],w['threshold_current_components_allE_A_B_surround_mV_equiv']['excitatory_input'][1:3],w['net_threshold_current_mV_equiv'][1:3]) for w in windows],flush=True)
    del v,components,bounds
write(dest/'analysis.json',dict(status='COMPLETE_OBSERVED_DENSITY_CURRENT_BUDGET',rows=rows,
    scope='Current at a threshold reference, not average dV/dt. It shows the balance accompanying collapse, but separating mediated necessity from correlation still requires a targeted intervention. Not a native trajectory or autonomousZrecovery.',
    producer_sha256=sha(__file__)))

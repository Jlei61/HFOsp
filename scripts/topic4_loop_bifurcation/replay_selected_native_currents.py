#!/usr/bin/env python3
"""Exact local readout replay, then only replace M(t) with its observed mean."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from observe_source_time_structure import OUT as SOURCE
from observe_source_aggregation import INITIAL,OUT as MOMENTS
from conditional_density_inputs import OPS
from coupled_density_exit import ADAPTED
import run_topic4_loop_zk_conditional as native

OUT=SOURCE/'selected_current_replay_exact_thresholds'


def main(wait):
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_CURRENT_REPLAY',created_epoch=time.time(),
        question='Does fixing native M to its mean account for remaining local response errors, or must temporal/non-Gaussian current structure be retained?',
        design='Replay39targets with all recorded native0.1ms IE/II/M and global s, initialV/ref, heldZ/K. Original membrane arithmetic/order. First verify every selected native spike and pre-stepV, then only replace M(t) with its observed2smean. No source response to changed targetspikes, no noise alteration.',
        limits='Conditional local readout at unchanged recorded inputs; counterfactual M is supplied rather than internally updated. Not a coupled-network causal lesion or autonomous closure.',
        global_conversion='Recorded slow.global_state is s; physical raw conductance is30*s.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    while not (SOURCE/'observer_audit.json').exists():
        write(OUT/'progress.json',dict(status='WAITING_TEMPORAL_OBSERVATION',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    assert read(SOURCE/'observer_audit.json')['status']=='PASS'
    z=dict(np.load(SOURCE/'target_traces.npz'));cells=z['cells'];trace=z['IE_II_V_M'];s=z['causal_R_G'][:,1]
    # The observer's historical field label contains G but holds pre-step s.
    # This conversion is explicit and checked against the original observer.
    old=dict(np.load(MOMENTS/'cell_statistics.npz'));assert np.array_equal(s,old['global_R_and_s'][:,1])
    G=30*s
    state=native.read_pickle(INITIAL)['engine'];p=read(OPS/'prepared.json')['params']
    geo=dict(np.load(ADAPTED/'geometry.npz'));E=cells<32000
    Z=state['slow']['z'][cells];K=np.r_[state['termination_mechanism']['sahp_g'],np.zeros(8000)][cells]
    from native_target_thresholds import load
    actual_theta,theta_source=load();theta=actual_theta[cells]
    tm=np.where(E,p['tau_m_E'],p['tau_m_I']);decay=np.exp(-.1/tm)
    refractory=np.rint(np.where(E,p['tau_ref_E'],p['tau_ref_I'])/.1).astype(int)
    observed=sparse.load_npz(SOURCE/'source_spikes_0p1ms.npz').tocsr()[cells].toarray().T.astype(bool)
    Mmean=trace[:,3].mean(0);data=[];errors=[]
    for mode in ['recorded_M','mean_M']:
        V=state['V'][cells].copy();ref=state['ref'][cells].copy();spikes=np.empty_like(observed)
        error=0.
        for t in range(len(trace)):
            if mode=='recorded_M':error=max(error,float(abs(V-trace[t,2]).max()))
            M=trace[t,3] if mode=='recorded_M' else Mmean
            value=trace[t,0]-Z*trace[t,1]-.0005*M
            value[E]+=K[E]*(-30+17.662847938268442)
            g=K+E*Z*G[t]
            inf=(value+g*(-17.662847938268442))/(1+g)
            ref=np.maximum(ref-1,0);free=ref==0
            V=np.where(free,inf+(V-inf)*decay**(1+g),p['V_reset'])
            sp=free&(V>=theta);V[sp]=p['V_reset'];ref[sp]=refractory[sp]
            spikes[t]=sp
        if mode=='recorded_M':
            assert np.array_equal(spikes,observed),np.count_nonzero(spikes!=observed)
            assert error<1e-9,error
        data.append(spikes);errors.append(error)
    rates=np.stack([x.sum(0)/2 for x in data])
    assay=read(MOMENTS/'local_response_factorial_exact_thresholds/result.json');lookup={r['cell']:r for r in assay['effects']}
    rows=[dict(cell=int(c),selected_by=lookup[int(c)]['selected_by'],native_rate_Hz=float(rates[0,j]),
        actual_current_meanM_rate_Hz=float(rates[1,j]),
        measured_variance_gaussian_meanM_rate_Hz=lookup[int(c)]['rates_Hz'][3]) for j,c in enumerate(cells)]
    np.savez_compressed(OUT/'replay.npz',cells=cells,rates_Hz=rates,spikes_recordedM=data[0],spikes_meanM=data[1])
    result=dict(status='COMPLETE_EXACT_LOCAL_REPLAY_AND_FIXED_M_DIAGNOSIS',rows=rows,
        exact_selected_spikes=True,max_prestep_voltage_error_mV=errors[0],
        raw_G_equals30s=True,observed_raw_G_mean=float(G.mean()),
        native_equations_unchanged=True,autonomous_closure_repaired=False,formal_bifurcation_allowed=False,
        threshold_source=theta_source,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)

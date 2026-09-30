#!/usr/bin/env python3
"""Check a mechanistically consequential resource drift at completed cut points.

Native mean drift is averaged over matched elapsed5-10s. Density drift is the
last-step empirical distribution, evaluated with the recovered pre-step G.
Those observables are labeled separately; no mean-current threshold shortcut.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import time
import numpy as np
from campaign import ROOT,read,write,sha
from exit_branch_density import OUT,NAMES
from coupled_density_exit import ADAPTED
from analyze_native import original


def analyze(name,geo):
    folder=OUT/name;E=geo['population']==0;size=geo['group_size'];regions=geo['group_region']
    masks=[E]+[E&(regions==q) for q in range(3)]
    with np.load(folder/'final_state.npz') as z:
        state=z['state'];ref=z['ref'];glob=z['global_state'];assert int(z['clock'][0])==100000
    # An E ref of20 means it spiked on the last step: older ref20 was reduced
    # before integration. Recover the exact pre-step causalR from that count.
    assert np.max(ref[E])<=20
    firing_fraction=(ref[E]==20).mean(1)
    mean_firing_fraction=np.average(firing_fraction,weights=size[E])
    Rpre=(glob[0]-mean_firing_fraction*1000/15.)/np.exp(-.1/15.)
    q=np.clip((Rpre-200)/300,0,1);a=np.exp(-.1/500.)
    Gpre=30*(glob[1]-(1-a)*q)/a
    assert Gpre>=-1e-10
    eligible=(state[:,:,4]+(18+17.662847938268442)*Gpre<95.19851312666987).mean(1)
    Z=state[:,:,6].mean(1);drift=(eligible-Z)/5.
    def weighted(value):return [float(np.average(value[mask],weights=size[mask])) for mask in masks]
    density=dict(endpoint_elapsed_s=10.,last_step_pre_R_Hz=float(Rpre),last_step_pre_Graw=float(Gpre),
        held_Z=weighted(Z),eligible_fraction=weighted(eligible),counterfactual_dZ_per_s=weighted(drift),
        negative_raw_I_I_fraction=weighted((state[:,:,4]<0).mean(1)),
        observable='Instantaneous last-step particle-time predicate,not5secondaverage. No thresholdofmeanI isused.')
    root=ROOT/'exit_return_probes';job=read(root/'jobs'/f'{name}.json');start=(job['branch_start_s']+5)*1000;end=start+5000
    native=original.load(root/'runs'/name/'conditional_drift_chunks',['time_ms','values'])
    keep=(native['time_ms']>start+1e-8)&(native['time_ms']<=end+1e-8);assert keep.sum()==250
    drift_native=native['values'][keep,:,0].mean(0)
    with np.load(job['held_fields_file']) as z:held=z['Z']
    member_regions=regions[geo['cell_group'][:32000]]
    zz=np.r_[held.mean(),*[held[member_regions==q].mean() for q in range(3)]]
    glob_native=original.load(root/'runs'/name/'mechanism_chunks',['time_ms','global_raw_conductance_ratio'])
    m=(glob_native['time_ms']>=start)&(glob_native['time_ms']<end);assert m.sum()==5000
    # G cannot decay faster than tauG=.5s, even between1ms records.
    minG=float(glob_native['global_raw_conductance_ratio'][m].min())
    lower=minG*np.exp(-.001/.5)
    native_row=dict(interval_elapsed_s=[5,10],held_Z=zz.tolist(),mean_counterfactual_dZ_per_s=drift_native.tolist(),
        eligible_fraction=(zz+5*drift_native).tolist(),sampled_Graw_min=minG,
        conservative_interstep_Graw_lower_bound=lower,
        allE_Z_recovery_blocked_throughout=bool(lower>95.19851312666987/(18+17.662847938268442)),
        observable='Nativepreceding20msstep-budgetaverageoverelapsed5-10s;rawI_I nonnegative,globalG lowerboundvalidbetweenrecords.')
    return dict(name=name,native=native_row,density=density,
        critical_question='Does a visuallysimilar sustainedbranch imply the same Z-restorationdirection?',formal_bifurcation_allowed=False)


def main(wait):
    geo=dict(np.load(ADAPTED/'geometry.npz'));done=[];rows=[]
    while len(done)<4:
        for name in NAMES:
            result=OUT/name/'result.json'
            if name in done or not result.exists():continue
            assert read(result)['status']=='COMPLETE'
            row=analyze(name,geo);rows.append(row);done.append(name)
            write(OUT/name/'resource_drift_audit.json',row)
            print('EXIT RESOURCE DRIFT',name,row['native']['mean_counterfactual_dZ_per_s'],row['density']['counterfactual_dZ_per_s'],flush=True)
        write(OUT/'resource_drift_audit.json',dict(status='COMPLETE' if len(done)==4 else 'PARTIAL',completed=done,rows=rows,
            different_time_statistics_labeled=True,formal_bifurcation_allowed=False,producer_sha256=sha(__file__)))
        if len(done)<4:
            if not wait:return
            write(OUT/'resource_audit_progress.json',dict(status='WAITING_FIXED_FOUR',pid=os.getpid(),completed=done,updated_epoch=time.time()))
            time.sleep(30)
    write(OUT/'resource_audit_progress.json',dict(status='COMPLETE',completed=done,updated_epoch=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');main(parser.parse_args().wait)

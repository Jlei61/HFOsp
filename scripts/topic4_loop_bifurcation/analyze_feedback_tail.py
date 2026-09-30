#!/usr/bin/env python3
"""Existing native paired kinetics: G carryover before K retention and Z recovery."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from pathlib import Path

SOURCE=Path('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923')
OUT=ROOT/'feedback_tail_mechanism'


def load(folder,sub,keys):
    data={k:[] for k in keys}
    for p in sorted((folder/sub).glob('*.npz')):
        with np.load(p) as z:
            for k in keys:data[k].append(z[k])
    return {k:np.concatenate(v) for k,v in data.items()}


def first_run(t,mask,n):
    edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    good=starts[ends-starts>=n];return int(good[0]) if len(good) else None


def main():
    OUT.mkdir(exist_ok=True);rows=[]
    for seed,tau in [(9108402,0.),(9108402,.5),(9108403,0.),(9108403,.5),(9108405,.5)]:
        name=f'G30_response{tau:g}_s{seed}';folder=SOURCE/'runs'/name
        result=read(folder/'result.json');assert result['status']=='COMPLETE'
        analysis=read(SOURCE/'analysis'/f'{name}.json');onset=analysis['primary']['entries'][0]['onset_s']
        rate=load(folder,'mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
        response=load(folder,'global_response_chunks',['time_ms','q','G_raw'])
        adapt=load(folder,'intrinsic_adaptation_chunks',['time_ms','sahp_mean_conductance_ratio'])
        budget=load(folder,'z_budget_chunks',['time_ms','values'])
        assert np.array_equal(rate['time_ms'],response['time_ms']) and np.array_equal(rate['time_ms'],adapt['time_ms'])
        t=rate['time_ms']/1000;R=rate['global_E_rate_Hz'];G=rate['global_raw_conductance_ratio'];q=response['q'];K=adapt['sahp_mean_conductance_ratio']
        assert np.allclose(q,np.clip((R-200)/300,0,1),atol=1e-14,rtol=0)
        assert np.array_equal(G,response['G_raw'])
        if tau==0:assert np.array_equal(G,30*q)
        after=t>=onset
        quiet_i=first_run(t,after&(R<=5),100)
        limit=result['job']['threshold']/(18-result['job']['global_reversal_mV'])
        qb=budget['values'];tb=budget['time_ms']/1000
        assert np.max(abs(qb[:,:,5]))<1e-11
        # Budget endpoints and instantaneous samples are explicitly separate.
        positive_i=first_run(tb,(tb>=onset)&(qb[:,1:3,4]>0).all(1),50)
        row=dict(name=name,seed=seed,tau_G_s=tau,onset_s=onset,
            minimum_sampled_R_after_entry_Hz=float(R[after].min()),
            maximum_G_raw_with_q_zero_after_entry=float(G[after&(q==0)].max()) if (after&(q==0)).any() else None,
            first_R_below5_for100ms_s=float(t[quiet_i]) if quiet_i is not None else None,
            first_core_net_positive_1s_budget_endpoint_s=float(tb[positive_i]) if positive_i is not None else None,
            G_block_threshold=limit)
        if quiet_i is not None:
            # Take first recorded G crossing after this low-rate interval begins.
            candidates=np.flatnonzero((np.arange(len(t))>=quiet_i)&(G<limit))
            cross=int(candidates[0]);take=np.arange(quiet_i,cross+1)
            predicted=G[quiet_i]*np.exp(-(t[take]-t[quiet_i])/tau)
            assert (q[take]==0).all()
            error=float(abs(predicted-G[take]).max());assert error<1e-8
            # Nonnegative spikes and15ms rate decay bound intersample R above;
            # this proves q remains off between the1ms samples in this interval.
            upper=float(R[take].max()*np.exp(.001/.015));assert upper<200
            row.update(G_raw_at_R5=float(G[quiet_i]),K_at_R5=float(K[quiet_i]),
                first_G_below_recovery_block_after_R5_s=float(t[cross]),
                G_exponential_prediction_error=error,between_sample_R_upper_bound_Hz=upper,
                q_zero_interval_s=[float(t[quiet_i]),float(t[cross])],
                K_at_G_unblocking=float(K[cross]))
        rows.append(row)
    write(OUT/'analysis.json',dict(status='COMPLETE_EXISTING_NATIVE_RECORDS',rows=rows,producer_sha256=sha(__file__),
        question='Does delayedGpersistaftertheinstantaneoushigh-rategatecloses, bridging into the presetlow-rateK-retention regime before Z recovery becomes possible?',
        statistical_unit='Two previously paired120s seeds plus the selected240s source; reused data, not additional simulations. One first-entry episode per trajectory for this diagnostic.',
        interpretation='InstantGequals30qandvanishesbelow200Hz. DelayedGcanremainpositiveafterqcloses. PairednativecontrolsestablishtheconsequenceofchangingtauG; this trajectory-levelmediatingsequencealone does not prove Gcarryover is the unique mechanism or identify a bifurcation.',
        timing='R/G/K at1ms samples; R<=5 first100ms is a diagnostic marker, not a new physicalrule. CoreZpositive uses20msintegratedbudgetendpoints sustained1s. q-off continuous bound uses positive spikeincrements and15msdecay.',
        formal_bifurcation_allowed=False))
    print(rows,flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare actual-field initial-G interventions with the unchanged native baseline."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_native import original

OUT=ROOT/'exit_actual_G_history_probes'
BASE=ROOT/'exit_return_probes'
BASE_NAME='exit_z0.21_k9_fields16p7_high'


def load(root,name):
    job=read(root/'jobs'/f'{name}.json');folder=root/'runs'/name
    assert read(folder/'result.json')['status']=='COMPLETE'
    with np.load(root/'extended_analysis'/f'{name}_readouts.npz') as z:
        rate=z['rate_5ms_Hz'];field=z['field_rate_5ms_Hz'];times=z['relative_time_5ms_s']
    raw=original.load(folder/'mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
    tm=raw['time_ms']/1000-job['branch_start_s'];keep=(tm>=0)&(tm<30)
    assert keep.sum()==30000
    drift=original.load(folder/'conditional_drift_chunks',['time_ms','values'])
    td=drift['time_ms']/1000-job['branch_start_s'];keepd=(td>0)&(td<=30);assert keepd.sum()==1500
    inputs=original.load(folder/'chunks',['inputs'])['inputs']
    keepi=(inputs[:,0]>=job['branch_start_s']*1000)&(inputs[:,0]<(job['branch_start_s']+30)*1000)
    assert keepi.sum()==300
    return dict(job=job,rate=rate,field=field,time5=times,time1=tm[keep],R=raw['global_E_rate_Hz'][keep],
        G=raw['global_raw_conductance_ratio'][keep],drift_time=td[keepd],drift=drift['values'][keepd],inputs=inputs[keepi,1:],
        generic=read(root/'extended_analysis'/f'{name}.json'))


def main(wait):
    dest=OUT/'G_history_comparison';dest.mkdir(exist_ok=True)
    assert not (dest/'analysis.json').exists()
    while not (OUT/'extended_analysis_summary.json').exists():
        status=OUT/'status.json'
        if status.exists() and read(status)['stage']=='FAILED':
            write(dest/'progress.json',dict(status='STOPPED_ON_NATIVE_FAILURE'));return
        write(dest/'progress.json',dict(status='WAITING_COMPLETE_NATIVE_ANALYSIS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/'extended_analysis_summary.json')['status']=='COMPLETE'
    reference=load(BASE,BASE_NAME);all_data=[reference]+[load(OUT,n) for n in read(OUT/'queue.json')['names']]
    geo=dict(np.load(OUT/'geometry.npz'));counts=geo['cell_e_counts'];rows=[];threshold=95.19851312666987/(18+17.662847938268442)
    baselineG=read(OUT/'scientific_contract.json')['design']['baseline_initial_G_raw']
    for i,d in enumerate(all_data):
        assert np.array_equal(d['inputs'],reference['inputs']),d['job']['name']
        windows=[]
        for lo,hi in [(0,1),(1,5),(5,10),(20,30)]:
            m=(d['time5']>=lo)&(d['time5']<hi);m1=(d['time1']>=lo)&(d['time1']<hi)
            md=(d['drift_time']>lo)&(d['drift_time']<=hi)
            field=d['field'][m].mean(0);base=reference['field'][m].mean(0)
            windows.append(dict(interval_s=[lo,hi],rates_Hz_allE_A_B_surround=d['rate'][m].mean(0).tolist(),
                mean_Graw=float(d['G'][m1].mean()),minimum_causal_R_Hz=float(d['R'][m1].min()),
                sampled_causal_R_at_or_below5_fraction=float((d['R'][m1]<=5).mean()),
                sampled_G_below_recovery_block_fraction=float((d['G'][m1]<threshold).mean()),
                mean_counterfactual_dZ_per_s_allE_A_B_surround=d['drift'][md,:,0].mean(0).tolist(),
                spatial_field_RMS_from_baseline_Hz=float(np.sqrt(np.average((field-base)**2,weights=counts)))))
        row=dict(name=d['job']['name'],initial_Graw=d['job'].get('G_raw_override',baselineG),
            complete_30s=True,recorded_future_input_samples_bitwise=300,
            finite_window_state=d['generic']['finite_window_state'],tail_brief_events=d['generic']['tail_brief_events'],
            censoring=d['generic']['censoring'],windows=windows)
        rows.append(row)
        np.savez_compressed(dest/f"{d['job']['name']}.npz",time5_s=d['time5'],rate5_Hz=d['rate'],field5_Hz=d['field'],
            time1_s=d['time1'],causal_R_Hz=d['R'],Graw=d['G'],drift_time_s=d['drift_time'],counterfactual_drift=d['drift'])
    result=dict(status='COMPLETE_ACTUAL_FIELD_G_HISTORY_COMPARISON',rows=rows,
        initial_state_qa=read(OUT/'initial_state_qa.json'),
        statistical_unit='One shared endogenous history and future noise, two initial-G interventions plus reused unchanged baseline; no independent seed replication.',
        scope='Z andK remain held. CounterfactualZdrift is not observed recovery. Finite conditional persistence or suppression does not establish attractors, a basin boundary, an autonomous exit, or a bifurcation.',
        counts_as_autonomous_loop=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(dest/'analysis.json',result);write(dest/'progress.json',dict(status=result['status'],completed=len(rows),updated_epoch=time.time()))
    print([(q['initial_Graw'],q['finite_window_state'],q['windows'][-1]['rates_Hz_allE_A_B_surround']) for q in rows],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)

#!/usr/bin/env python3
"""Read-only scientific assay for the single-current adaptation candidate."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np

MAIN=Path('/home/honglab/leijiaxin/HFOsp')
sys.path.insert(0,str(MAIN/'scripts'))
import analyze_topic4_interictal_recurrence as audit
from run_topic4_reset_state_diagnosis import write
ROOT=Path('/data/hfosp/topic4_sef_hfo/unified_current_adaptation_20261030')


def load(folder, stream, keys):
    rows={k:[] for k in keys}
    for f in sorted((folder/stream).glob('*.npz')):
        if '.tmp.' in f.name:continue
        with np.load(f) as a:
            for k in keys:rows[k].append(a[k])
    return {k:np.concatenate(v,axis=0) for k,v in rows.items()}


def analyze(name):
    folder=ROOT/'runs'/name
    d=load(folder,'chunks',['time_ms','spikes_1ms','regions_1ms','field_time_ms',
        'field_5ms','slow_time_ms','Z','M','inputs'])
    g=dict(np.load(ROOT/'geometry.npz'))
    assert d['spikes_1ms'][:,0].sum()==d['regions_1ms'][:,:3].sum()==d['field_5ms'].sum()
    assert d['spikes_1ms'][:,1].sum()==d['regions_1ms'][:,3:].sum()
    end=len(d['spikes_1ms'])/1000.
    assert np.allclose(d['time_ms'],np.arange(len(d['time_ms']))+.5)
    counts=np.r_[32000,g['region_counts'][:3]]
    rate=np.c_[d['spikes_1ms'][:,0],d['regions_1ms'][:,:3]].reshape(-1,10,4).sum(1)/counts/.01
    temporal=audit.temporal_audit(rate)
    t=d['slow_time_ms']/1000.;z=d['Z'][:,[0,5,6]]
    episodes=[]
    first=temporal['entries'][0]['onset_s'] if temporal['entries'] else None
    baseline_valid=first is not None and first>8.
    zref=z[np.searchsorted(t,8.,side='right')-1] if baseline_valid else None
    events=temporal['events']
    def describe(event):
        a,b=round(event['start_s']*1000),round(event['end_s']*1000)
        rr=d['regions_1ms'][a:b,:2]/g['region_counts'][:2]/.001
        if len(rr)>=5:
            n=len(rr)//5
            smooth=rr[:n*5].reshape(n,5,2).mean(1)
            peaks=smooth.max(0)
            peak_t=(a+np.argmax(smooth,axis=0)*5+2.5)/1000
        else:peaks=np.zeros(2);peak_t=np.full(2,event['start_s'])
        return dict(**event,core_peak_5ms_Hz=peaks.tolist(),
            core_peak_times_s=peak_t.tolist(),core_B_minus_A_peak_ms=float(np.diff(peak_t)[0]*1000),
            both_cores_active=bool(np.all(peaks>=20.)))
    for ex in temporal['low_activity_exits']:
        start=ex['start_s']
        nxt=next((x['onset_s'] for x in temporal['entries'] if x['onset_s']>ex['confirmation_s']),end)
        recovered=None
        if zref is not None:
            ids=np.flatnonzero((t>=start)&(t<nxt)&np.all(z[:,1:]>=zref[None,1:],axis=1))
            if len(ids):recovered=float(t[ids[0]])
        ev=[] if recovered is None else [describe(e) for e in events
            if e['start_s']>=recovered and e['end_s']<=nxt and not e.get('right_censored',False)
            and .02-1e-9<=e['duration_s']<=.2+1e-9]
        span=ev[-1]['end_s']-ev[0]['start_s'] if ev else 0.
        episodes.append(dict(exit_start_s=start,exit_confirmed_s=ex['confirmation_s'],
            both_core_Z_reference_s=recovered,observation_end_s=nxt,
            returned_brief_events=ev,brief_count=len(ev),brief_span_s=span,
            both_core_event_count=sum(e['both_cores_active'] for e in ev),
            sustained_return=bool(len(ev)>=10 and span>=5.)))
    baseline=[] if first is None else [describe(e) for e in events
        if .5<=e['start_s'] and e['end_s']<=first and .02-1e-9<=e['duration_s']<=.2+1e-9]
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
    row=dict(model='ZGM_UNIFIED_CURRENT_ADAPTATION_V1',name=name,observed_s=end,
        run_status=result.get('status','RUNNING'),temporal=temporal,episodes=episodes,
        initial_brief_events=baseline,reference_time_s=8. if baseline_valid else None,
        reference_allE_coreA_coreB_Z=None if zref is None else zref.tolist(),
        has_high_entry=bool(temporal['entries']),has_autonomous_exit=bool(episodes),
        has_sustained_return=any(x['sustained_return'] for x in episodes),
        count_conservation=True,statistical_unit='one network and one noise realization',
        no_scheduled_intervention=True,no_K_state=True,human_review='PENDING',
        criteria=dict(high='All E >=200 Hz continuously for200ms,10ms bins',
            exit='Original observer: two1s windows all/core means<50Hz and quiet fraction>=0.2',
            recovery='Both cores recover their own initial-interictal reference at8s; not a bifurcation threshold',
            return_rule='At least10 complete20-200ms events spanning>=5s after both-core recovery'),
        scientific_limit='Operational model high activity, not established clinical seizure or certified attractor.')
    dest=ROOT/'analysis'/name;dest.mkdir(parents=True,exist_ok=True)
    write(dest/'result.json',row)
    np.savez_compressed(dest/'readouts.npz',time10_s=(np.arange(len(rate))+.5)*.01,
        rates10_Hz=rate,slow_time_s=t,Z=z,adaptation_mV=.0005*d['M'])
    print(json.dumps({k:row[k] for k in ['name','observed_s','run_status','has_high_entry',
        'has_autonomous_exit','has_sustained_return']}),flush=True)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='unified_ref48_s9108405')
    analyze(p.parse_args().name)

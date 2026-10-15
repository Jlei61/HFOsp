#!/usr/bin/env python3
"""Separate detected contact events from morphology-window selection, offline."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.topic4_observation_repaired import observe
from scripts.paper_figures.compare_topic4_patient_core_rhythm import OUT,P,interval_metrics,csvwrite,write

def summarize(events,duration_s):
    all_times=np.sort([e['qualifying_interval_ms'][0]/1000 for e in events])
    accepted=np.sort([e['qualifying_interval_ms'][0]/1000 for e in events if e['primary_eligible']])
    d={}
    for label,t in [('all_contact',all_times),('accepted_contact',accepted)]:
        d.update({label+'_'+k:v for k,v in interval_metrics(t).items()})
        d[label+'_rate_hz']=len(t)/duration_s
    d['excluded_fraction']=1-len(accepted)/len(all_times) if len(all_times) else None
    d['overlap_excluded_count']=sum('overlapping_window' in e['primary_exclusion_reasons'] for e in events)
    return d

def main():
    current=[]
    for path in sorted((P/'execution/confirmation/repaired_observation').glob('*.json')):
        d=json.loads(path.read_text());w=json.loads(Path(d['worker_path']).read_text())
        current.append(dict(candidate_id=w['candidate_id'],unit=path.stem,**summarize(d['events'],23.5)))
    csvwrite(OUT/'current_contact_interval_selection.csv',current)
    base=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/execution/paired_round1'
    cs={c['candidate_id']:c for c in json.loads((base/'candidate_manifest.json').read_text())['candidates']}
    contract=json.loads((ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/observation_contract.json').read_text())
    arms=['baseline','tau_d_GABA_ms_low','tau_d_GABA_ms_high','E_to_E_weight_scale_low','E_to_E_weight_scale_high','E_to_I_weight_scale_low','E_to_I_weight_scale_high','I_to_E_weight_scale_low','I_to_E_weight_scale_high','vth_low','vth_high']
    rows=[]
    for path in sorted((base/'workers').glob('*.json')):
        w=json.loads(path.read_text());c=cs[w['candidate_id']]
        if c['anchor'] not in ['historical','old_joint','support_rank'] or c['arm'] not in arms:continue
        assert w['simulation']['runaway_early_stop_ms'] is None
        with np.load(w['arrays']['path']) as z:
            assert z['contact_names'].astype(str).tolist()==contract['contact_names']
            result=observe(z['contact_envelope'].astype(float),float(z['contact_envelope_dt_ms']),contract)
        rows.append(dict(candidate_id=c['candidate_id'],anchor=c['anchor'],seed=w['seed'],arm=c['arm'],worker_path=str(path),**summarize(result['events'],11.5)))
    csvwrite(OUT/'paired_parameter_contact_intervals.csv',rows)
    summary=[]
    for arm in arms:
        rr=[r for r in rows if r['arm']==arm]
        summary.append(dict(arm=arm,n_units=len(rr),
            n_units_with_all_contact_intervals=sum(r['all_contact_interval_median_s'] is not None for r in rr),
            n_units_with_accepted_contact_intervals=sum(r['accepted_contact_interval_median_s'] is not None for r in rr),
            all_contact_event_count_range=[min(r['all_contact_n_events'] for r in rr),max(r['all_contact_n_events'] for r in rr)],
            **{k:float(np.median([r[k] for r in rr if r[k] is not None])) for k in ['all_contact_interval_median_s','accepted_contact_interval_median_s','all_contact_interval_cv','accepted_contact_interval_cv','excluded_fraction']}))
    write(OUT/'contact_interval_selection_audit.json',dict(new_physical_runs=0,current_units=current,paired_parameter_summary=summary,interpretation='Offline application of the unchanged fixed contact observer. All-contact events still include threshold, ≥8 contacts, 30 ms extension/merge and edge-window checks; only morphology-window isolation/prolongation/centroid exclusions differ. Not a measurement of raw neuron waiting times.'))
    for r in current:print(r)
    for r in summary:print(r)

if __name__=='__main__':main()

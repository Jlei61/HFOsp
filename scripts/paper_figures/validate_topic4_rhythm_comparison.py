#!/usr/bin/env python3
"""Check source identities and whether peak-threshold choice changes key effects."""
import json,csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.paper_figures.compare_topic4_patient_core_rhythm import OUT,metrics,regions,csvwrite,write

def main():
    base=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/execution/paired_round1'
    cs={c['candidate_id']:c for c in json.loads((base/'candidate_manifest.json').read_text())['candidates']}
    arms=['baseline','tau_d_GABA_ms_low','tau_d_GABA_ms_high','E_to_E_weight_scale_low','E_to_I_weight_scale_high','I_to_E_weight_scale_high']
    rows=[];checks=[]
    with (OUT/'paired_parameter_core_metrics.csv').open() as f:paths=sorted(set(r['worker_path'] for r in csv.DictReader(f) if r['arm'] in arms))
    for path in paths:
        w=json.loads(Path(path).read_text());c=cs[w['candidate_id']];a=w['multidimensional_parameter_audit']
        assert a['requested']==a['effective'] and a['topology_and_delay_assignments_preserved']
        with np.load(w['arrays']['path']) as z:
            _,_,den,pure,_=regions(z,np.asarray(c['node_field']['centers_mm']),w['xy_geometry_audit']['distance_cutoff_mm'])
            raw=z['sheet_activity_counts'].astype(float)[250:];assert np.all(raw<=den[None])
        for prom in [.1,.2,.3]:
            vals=[metrics(raw[:,pure[k]].sum(1)/den[pure[k]].sum(),prom)[0] for k in range(2)]
            rows.append(dict(anchor=c['anchor'],seed=w['seed'],arm=c['arm'],prominence=prom,
                 **{k:float(np.mean([v[k] for v in vals])) for k in ['interval_median_ms','interval_cv','peak_active_fraction_median']}))
    csvwrite(OUT/'parameter_peak_threshold_sensitivity.csv',rows)
    for prom in [.1,.2,.3]:
        rr=[r for r in rows if r['prominence']==prom];baselines={(r['anchor'],r['seed']):r for r in rr if r['arm']=='baseline'}
        for arm in arms[1:]:
            for key in ['interval_median_ms','interval_cv','peak_active_fraction_median']:
                vv=[r[key]-baselines[(r['anchor'],r['seed'])][key] for r in rr if r['arm']==arm]
                checks.append(dict(prominence=prom,arm=arm,metric=key,n=len(vv),positive=int(np.sum(np.array(vv)>0)),negative=int(np.sum(np.array(vv)<0)),median_delta=float(np.median(vv))))
    manifest=json.loads((OUT/'manifest.json').read_text())
    with (OUT/'model_confirmation_metrics.csv').open() as f:mr=list(csv.DictReader(f))
    assert sum(int(r['n_events']) for r in mr if r['layer']=='contact_events')==431
    assert manifest['patient_total_events']==46683
    write(OUT/'validation.json',dict(source_event_counts_match=True,worker_actual_parameter_checks=len(paths),peak_threshold_sensitivity=checks,new_physical_runs=0))
    print('Validated',len(paths),'existing units and 431 contact events / 46683 archived patient events')
    for c in checks:
        if c['metric']=='interval_median_ms':print(c)

if __name__=='__main__':main()

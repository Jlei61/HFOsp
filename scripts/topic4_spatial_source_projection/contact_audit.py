"""Separate local amplitude failure from fixed-observer event selection."""
from shared_source import *
from src.topic4_observation_repaired import runs
import csv

def main():
    contract=read(V10/'native/a/observation_contract.json');names=contract['contact_names'];rows=[];summaries=[]
    inputs=[(f'native_s{s}',PRIOR/f'native/{s}/trajectory.npz',BASE/'observations'/f'native_s{s}.json') for s in (848101,848102,848103)]
    inputs.extend((p.name,p/'trajectory.npz',OUT/'observations'/f'{p.name}_neuron.json') for p in sorted((OUT/'runs').glob('*')) if (p/'result.json').exists())
    for name,path,observer in inputs:
        z=np.load(path);ob=read(observer);mu=np.array(ob['centroid_ms'],float)
        ids=[i for i,e in enumerate(ob['events']) if e['window_ms'][0]>=2000 and e['window_ms'][1]<=12000]
        for contact in ['SCL9','ICL10']:
            j=names.index(contact);bar=ob['threshold'][j];local=[]
            for i in ids:
                e=ob['events'][i];a,b=(np.array(e['window_ms'])/2).astype(int);v=z['contact_envelope'][a:b,j];detected=runs(v>bar)
                maxlen=max([y-x for x,y in detected],default=0)*2
                row=dict(model=name,event=i,contact=contact,start_ms=e['window_ms'][0],end_ms=e['window_ms'][1],
                    primary_eligible=i in ob['primary_event_indices'],participates=bool(np.isfinite(mu[i,j])),peak_over_threshold=float(v.max()/bar),
                    maximum_above_threshold_ms=maxlen,local_failure='below_threshold' if v.max()<=bar else ('too_brief' if maxlen<4 else 'none'))
                assert row['participates']==(maxlen>=4),row
                rows.append(row);local.append(row)
            for condition in ['all_detected','primary_only']:
                keep=[r for r in local if condition=='all_detected' or r['primary_eligible']]
                summaries.append(dict(model=name,contact=contact,condition=condition,N=len(keep),participation=float(np.mean([r['participates'] for r in keep])),
                    peak_ratio_quantiles=np.quantile([r['peak_over_threshold'] for r in keep],[0,.25,.5,.75,1]),
                    below_threshold=sum(r['local_failure']=='below_threshold' for r in keep),too_brief=sum(r['local_failure']=='too_brief' for r in keep)))
    with (OUT/'contact_amplitude_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(OUT/'contact_amplitude_audit.json',dict(rows=summaries,thresholds_unchanged=True,
        interpretation='Contact participation requires >=4 ms above the fixed native-reference threshold. Weak envelope is not equivalent to no local neural activity.'))
    print(json.dumps(safe(summaries),indent=2),flush=True)

if __name__=='__main__':main()

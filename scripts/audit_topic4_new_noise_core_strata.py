"""Repeat the pre-existing TB core-timing diagnostic on completed noise bridges."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd

BASE=Path('/data/hfosp/topic4_sef_hfo')
SOURCE=BASE/'core_multiseed_response_curves_20260913'
OUT=BASE/'overnight_exploration_20260913/position_new_noise_replay'

def main():
    rows=[]; sources=[]
    for file in sorted((SOURCE/'analysis/units').glob('*/result.json')):
        r=json.loads(file.read_text()); c=r['counts']
        if c['candidate'] not in ['bridge_circle_out125','bridge_circle_out125_xminus075']:continue
        trajectory=Path(r['source']); npz=trajectory.with_suffix('.npz')
        assert hashlib.sha256(npz.read_bytes()).hexdigest()==r['source_sha256']
        with np.load(npz) as a: mu=a['centroid_ms'].copy(); names=list(a['contact_names'])
        i,j=names.index('ICL11'),names.index('ICL9')
        tb=[e for e in r['events'] if e['primary'] and e['mode']=='TB']
        pp={(names.index(q['contact_i']),names.index(q['contact_j'])):q['patient_i_precedes_j']
            for q in r['pairs'] if q['layer']=='primary' and q['mode']=='TB'}
        for timing in ['B_minus_A_t10_ms','B_minus_A_t50_ms']:
            for group in ['all','left earlier','right earlier','tie','not estimable']:
                z=[e for e in tb if group=='all' or (group=='not estimable' and e[timing] is None) or e[timing] is not None and
                   ((group=='left earlier' and e[timing]>0) or (group=='right earlier' and e[timing]<0) or (group=='tie' and e[timing]==0))]
                x=mu[[e['event'] for e in z]]
                lag=x[:,j]-x[:,i] if len(x) else np.array([]); lag=lag[np.isfinite(lag)]
                errors=[]
                for (ii,jj),patient in pp.items():
                    if patient is None or not len(x):continue
                    v=x[:,jj]-x[:,ii];v=v[np.isfinite(v)]
                    if len(v):errors.append(abs(np.mean((v>0)+.5*(v==0))-patient))
                rod=[e['centroid_SCL_minus_ICL_ms'] for e in z if e.get('centroid_SCL_minus_ICL_ms') is not None]
                rows.append(dict(candidate=c['candidate'],topology=c['topology'],noise=c['noise'],timing=timing,group=group,n=len(z),
                    TB_total_n=len(tb),ICL_joint_n=len(lag),ICL11_first_probability=np.mean((lag>0)+.5*(lag==0)) if len(lag) else None,
                    rod_median_ms=np.median(rod) if rod else None,all_pair_order_probability_mae=np.mean(errors) if errors else None,
                    pair_support=len(errors),source=str(file)))
        sources.append(dict(analysis=str(file),analysis_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),arrays_sha256=r['source_sha256']))
    d=pd.DataFrame(rows);OUT.mkdir(exist_ok=True)
    d.to_csv(OUT/'new_noise_core_timing_conditional_observations.csv',index=False)
    (OUT/'new_noise_core_timing_manifest.json').write_text(json.dumps(dict(created_unix=time.time(),sources=sources,
        definition='Existing primary TB events; left/right determined by within-window own-core cumulative10/50 percent times, not causal onset.',
        patient_reference='Full patient TB distribution; no corresponding patient core-state strata.',
        interpretation='Descriptive association; do not filter poor subgroups, change nominations, or infer causal drive.',
        physical_runs_added=0,producer=str(Path(__file__)),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
    print(d[d.timing=='B_minus_A_t10_ms'].to_string(index=False),flush=True)

if __name__=='__main__':main()

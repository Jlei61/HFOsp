"""Canonical finite-time readout of the continuous 12+60 s held-Z experiment."""
from canonical_readouts import assess, OUT, R
import numpy as np
import json


def main():
    contract=json.loads((OUT/'native_Z_irregular_extension.json').read_text())
    assert contract['status'] in ['EXECUTION_COMPLETE_READOUT_PENDING','COMPLETE']
    source=np.load(contract['source'])
    path=OUT/'runs/native_Z_D2196_long_from12s_dt005/trajectory.npz'
    extension=np.load(path)
    assert np.array_equal(source['cell_counts'],extension['cell_counts'])
    assert np.max(abs(source['Z_source']-extension['Z_source']))<1e-12
    assert len(source['field_E_hz'])==12000 and len(extension['field_E_hz'])==60000
    fields=np.concatenate([source['field_E_hz'],extension['field_E_hz']])
    counts=source['cell_counts']
    whole=assess(fields,counts,'native_Z_D2196_continuous72s')
    blocks=[]
    for start in range(0,len(fields),12000):
        q=assess(fields[start:start+12000],counts,f'block_{start//1000}_{(start+12000)//1000}s',t0=start)
        blocks.append(q)
    cell=fields.reshape(-1,10,fields.shape[1]).mean(1)
    rate=cell@(counts/counts.sum())
    occupation=(cell>=50)@(counts/counts.sum())
    runs=R.runs_of((rate>=200)&(occupation>=.75))
    broad=next((int(a*10) for a,b in runs if b-a>=20),None)
    complete=whole['events']
    long=[q for q in complete if q['duration_ms']>=1000]
    result=dict(status='COMPLETE',model='frozen spatial rate field; native checkpoint Z parameter path',
        Z='held',M='dynamic',D=contract['D'],global_Z=contract['global_Z'],
        source_files=[contract['source'],str(path)],duration_ms=len(fields),
        canonical_whole=whole,blocks12s=blocks,
        complete_event_count=len(complete),complete_events_ge1s=len(long),
        maximum_complete_event_ms=max((q['duration_ms'] for q in complete),default=None),
        last_complete_event_end_ms=max((q['end_ms'] for q in complete),default=None),
        first_broad75percent_and200Hz200ms_start_ms=broad,
        high_rate_outcome='ENTRY_OBSERVED' if whole['high_rate'] is not None else 'NO_ENTRY_OBSERVED_FINITE72S',
        inference_limit='No asymptotic attractor, basin boundary, chaos or bifurcation type inferred from finite-time survival or escape.')
    (OUT/'native_Z_long_readout.json').write_text(json.dumps(result,indent=2))
    contract.update(status='COMPLETE',readout='native_Z_long_readout.json',
                    high_rate_outcome=result['high_rate_outcome'])
    (OUT/'native_Z_irregular_extension.json').write_text(json.dumps(contract,indent=2))
    for q in blocks:
        print(q['label'],q['category'],q['tail']['mean_rate_hz'],q['tail']['quiet_fraction'],q['tail']['n_events'])
    print({k:result[k] for k in ['complete_event_count','complete_events_ge1s','maximum_complete_event_ms',
        'last_complete_event_end_ms','high_rate_outcome','first_broad75percent_and200Hz200ms_start_ms']})


if __name__=='__main__':main()

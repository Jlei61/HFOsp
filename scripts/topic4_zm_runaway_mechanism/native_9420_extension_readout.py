"""Canonical readout of the matched native9420-field rate continuation."""
from canonical_readouts import assess,OUT,R
from pathlib import Path
import json
import numpy as np


def load(path):return json.loads(Path(path).read_text())


def main():
    contract_path=OUT/'native_9420_matched_extension_contract.json'
    contract=load(contract_path)
    source_path=Path(contract['source']);extension_path=OUT/contract['extension']
    assert load(extension_path.parent/'result.json')['status']=='COMPLETE'
    source=np.load(source_path);extension=np.load(extension_path)
    runtime=load(extension_path.parent/'contract.json')
    initial=np.load(runtime['initial'])
    expected_history=np.roll(source['final_history'],-int(source['final_tick'])%len(source['final_history']),axis=0)
    checks=dict(initial_state_bitwise=np.array_equal(source['final_state'],initial['state']),
        initial_history_bitwise=np.array_equal(expected_history,initial['history']),
        initial_tick_zero=int(initial['tick'])==0,
        dt=float(initial['dt_ms'])==float(extension['dt_ms'])==contract['dt_ms'],
        source_path=Path(str(initial['source'])).resolve()==source_path.resolve(),
        counts=np.array_equal(source['cell_counts'],extension['cell_counts']),
        identical_Z=np.array_equal(source['Z_source'],extension['Z_source']),
        Z_unchanged=np.array_equal(extension['Z_source'],extension['final_state'][11]),
        M_dynamic=runtime['M']=='dynamic',Z_held=runtime['Z']=='held',
        endpoint_history=runtime['rate_history_scheme']=='instantaneous rate at the labelled endpoint',
        duration=len(source['field_E_hz'])==12000 and len(extension['field_E_hz'])==60000)
    assert all(checks.values()),checks
    fields=np.concatenate([source['field_E_hz'],extension['field_E_hz']])
    counts=source['cell_counts'];whole=assess(fields,counts,'native9420_matched_continuous72s')
    blocks=[assess(fields[t:t+12000],counts,f'block_{t//1000}_{t//1000+12}s',t0=t)
            for t in range(0,72000,12000)]
    cell=fields.reshape(-1,10,fields.shape[1]).mean(1);weights=counts/counts.sum()
    rate=cell@weights;occupation=(cell>=50)@weights
    broad=next((int(a*10) for a,b in R.runs_of((rate>=200)&(occupation>=.75)) if b-a>=20),None)
    events=whole['events']
    q=dict(status='COMPLETE',contract=str(contract_path),restart_checks=checks,
        sources=[str(source_path),str(extension_path)],D=contract['D'],global_Z=contract['global_Z'],
        duration_ms=72000,Z='held',M='dynamic',canonical_whole=whole,blocks12s=blocks,
        complete_event_count=len(events),complete_events_after12s=[e for e in events if e['end_ms']>12000],
        maximum_complete_event_ms=max((e['duration_ms'] for e in events),default=None),
        last_complete_event_end_ms=max((e['end_ms'] for e in events),default=None),
        first_broad75percent_and200Hz200ms_start_ms=broad,
        limitation='One deterministic history, continuous72s at fixed native9420 spatial Z. Times are relative to the held-Z intervention, not original SNN clocks. Finite persistence is not asymptotic stability or a bifurcation certificate.')
    (OUT/'native_9420_matched_extension_readout.json').write_text(json.dumps(q,indent=2)+'\n')
    for b in blocks:print(b['label'],b['category'],b['tail']['mean_rate_hz'],b['tail']['quiet_fraction'],len(b['events']),flush=True)
    print('TOTAL',len(events),'lastend',q['last_complete_event_end_ms'],'broad',broad,flush=True)


if __name__=='__main__':main()

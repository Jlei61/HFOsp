#!/usr/bin/env python3
"""Recheck every completed Fig5 cell from its original population counts."""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import analyze_topic4_fig5_single_seed_scan as scan


def main():
    grid=scan.collect()
    protocol=scan.read(scan.OUT/'protocol.json')
    assert grid['all_complete'] and len(grid['records'])==35
    rows=[]
    for record in sorted(grid['records'],key=lambda r:(r['job']['eta_m'],r['job']['tau_M_s'])):
        source=Path(record['source']);result=scan.read(source/'result.json')
        assert result['identity']==protocol['identity']
        for key in ('seed','eta_m','tau_M_s','tau_z_ms','threshold'):
            assert result['job'][key]==record['job'][key],(source,key)
        assert record['job']['seed']==9108401
        end=0;counts=[];resolution=None;chunks=0
        for path in sorted((source/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as data:
                assert int(data['start_step'])==end,path
                next_end=int(data['end_step'])
                bin_ms=10 if 'spikes_10ms' in data else 1
                if resolution is None:resolution=bin_ms
                assert resolution==bin_ms
                pop=data[f'spikes_{bin_ms}ms'];regions=data[f'regions_{bin_ms}ms']
                assert len(pop)*bin_ms*10==next_end-end
                assert np.array_equal(pop[:,0],regions[:,:3].sum(1))
                assert np.array_equal(pop[:,1],regions[:,3:].sum(1))
                counts.append(pop[:,0]);end=next_end;chunks+=1
        pop=np.concatenate(counts)
        if resolution==1:
            assert len(pop)%10==0
            pop=pop.reshape(-1,10).sum(1)
        rate=pop/320.
        changes=np.diff(np.r_[False,rate>=200,False].astype(int))
        episodes=[(a,b) for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)) if b-a>=20]
        observed=bool(episodes)
        assert observed==record['event_observed']
        first=None if not observed else dict(onset_s=float(episodes[0][0]*.01),confirmation_s=float((episodes[0][0]+20)*.01))
        if first is not None:
            for key in first:assert np.isclose(first[key],record['first_entry'][key],rtol=0,atol=1e-8)
            tracker=result.get('tracker',{})
            # Historical full-observation runs may intervene later; the reused
            # first entry must precede any such intervention.
            if tracker.get('restore_s') is not None:
                assert first['confirmation_s']<tracker['restore_s']
        else:
            assert np.isclose(end*.0001,1000.) and result['censored_at_s']==1000.
        rows.append(dict(name=record['job']['name'],seed=9108401,eta_M=record['job']['eta_m'],
            tau_M_s=record['job']['tau_M_s'],event_observed=observed,
            onset_s=None if first is None else first['onset_s'],
            confirmation_s=None if first is None else first['confirmation_s'],
            censoring_s=None if observed else 1000.,audited_count_coverage_s=end*.0001,
            native_count_bin_ms=resolution,audited_chunks=chunks,
            source=str(source),result_sha256=hashlib.sha256((source/'result.json').read_bytes()).hexdigest()))
    times=[r['confirmation_s'] for r in rows if r['event_observed']]
    new=[r for r in rows if r['tau_M_s']==10000 or r['eta_M']==10]
    summary=dict(status='PASS_ALL35_NATIVE_COUNT_ENDPOINTS',seed=9108401,total=35,
        entered=len(times),censored_at_1000_s=35-len(times),confirmation_range_s=[min(times),max(times)],
        late_entries_after_300_s=sum(t>300 for t in times),new_cells=len(new),
        newly_entering_parameter_cells=[r for r in new if r['event_observed']],
        continued_previously_censored_entries=sum(r['event_observed'] for r in rows if r['name'] in {x['job']['name'] for x in protocol['adopted']}),
        continuous_native_count_coverage=True,E_I_regional_conservation=True,
        all_same_topology_and_noise=True,protocol_sha256=hashlib.sha256((scan.OUT/'protocol.json').read_bytes()).hexdigest(),
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audited_at=time.time(),records=rows,human_review='PENDING')
    assert (summary['entered'],summary['censored_at_1000_s'],summary['late_entries_after_300_s'])==(16,19,0)
    (scan.OUT/'endpoint_audit_complete_20260917.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (scan.OUT/'first_entry_complete_20260917.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','newly_entering_parameter_cells')},indent=2))


if __name__=='__main__':main()

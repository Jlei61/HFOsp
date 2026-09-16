#!/usr/bin/env python3
"""Open registered seizure denominators only after all upstream views freeze."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v035.contracts import atomic_json
from src.topic5_group_event_state.v039.human_data import exposure,phase


def past_query_support(times,observed,seizures,history_seconds=28800):
    """Eligibility uses only pre-query coverage and already-started seizures."""
    seizures=np.asarray(seizures,float).reshape(-1,2)
    return np.array([exposure(observed,t-history_seconds,t)>=.9*history_seconds
        and not np.any((seizures[:,0]<=t)&(seizures[:,1]>t-history_seconds)) for t in times])


def represented_first_onsets(times,phases,onsets,bounds,horizon_seconds=21600):
    """Repeated forecasts target the next onset only, never every later cluster member."""
    onsets=np.unique(np.asarray(onsets,float));nxt=np.searchsorted(onsets,times,side='right')
    next_time=np.full(len(times),np.inf);has=nxt<len(onsets);next_time[has]=onsets[nxt[has]]
    out={}
    for split,hi in [('FIT','60pct'),('INNER','70pct'),('SELECTION','80pct')]:
        good=(phases==split)&(next_time-times<=horizon_seconds)&(next_time<bounds[hi])
        out[split]=dict(onsets=np.unique(next_time[good]).tolist(),positive_queries=int(good.sum()))
    return out


def audit(root,output):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    freezes=[]
    for directory,n in [('frozen_states',54),('frozen_view_states',18)]:
        cards=sorted((root/directory).glob('*/state.json'))
        if len(cards)!=n:raise ValueError('Wait for every registered upstream view to freeze')
        for p in cards:
            c=json.loads(p.read_text())
            if c['status']!='COMPLETE' or sha(c['source_card'])!=c['source_card_sha256']:raise ValueError('Invalid upstream freeze')
            freezes.append(dict(path=str(p),sha256=sha(p),subject=c['subject'],source=c['source_card'],checkpoint_sha256=c['checkpoint_sha256']))
    inventory=Path('/home/honglab/leijiaxin/HFOsp/results/epilepsiae_seizure_inventory.csv')
    with inventory.open() as stream:clinical=list(csv.DictReader(stream))
    rows=[]
    for s in ['epilepsiae_1096','epilepsiae_1125','epilepsiae_253']:
        boundary=root/'input_boundary'/s/'card.json';bounds=json.loads(boundary.read_text())['phase_boundaries']
        index_path=Path('/data/hfosp_group_event_state_v0_1/dataset')/s/'index.json';index=json.loads(index_path.read_text())
        seizures=[v for v in index['seizures'] if v['onset_epoch']<bounds['80pct']]
        indexed={v['seizure_id']:v for v in seizures}
        annotated=[v for v in clinical if v['subject']==s.split('_')[-1] and v['eeg_onset_epoch'] and float(v['eeg_onset_epoch'])<bounds['80pct']]
        if set(indexed)!={v['seizure_id'] for v in annotated}:raise ValueError('Clinical onsets omitted from exclusion/denominator index')
        for a in annotated:
            if abs(float(a['eeg_onset_epoch'])-indexed[a['seizure_id']]['onset_epoch'])>1e-6:raise ValueError('Onset mismatch')
        onsets=np.unique([v['onset_epoch'] for v in seizures]);counts={};clusters={};times={}
        for p,lo,hi in [('FIT',bounds['20pct'],bounds['60pct']),('INNER',bounds['60pct'],bounds['70pct']),('SELECTION',bounds['70pct'],bounds['80pct'])]:
            ts=onsets[(onsets>=lo)&(onsets<hi)];counts[p]=len(ts);times[p]=ts.tolist()
            clusters[p]={f'{hours}h_gap':int(bool(len(ts)))+int(np.sum(np.diff(ts)>hours*3600)) for hours in [8,24]}
        risk_min={'FIT':3,'INNER':1,'SELECTION':1};field_min={'FIT':5,'INNER':1,'SELECTION':1}
        risk_ok=all(counts[p]>=n for p,n in risk_min.items());field_ok=all(counts[p]>=n for p,n in field_min.items())
        data_path=root/'human_data_v2'/f'{s}.pt';data=torch.load(data_path,map_location='cpu',weights_only=False)
        grid=np.arange(np.ceil(bounds['20pct']/300)*300,bounds['80pct'],300.)
        eligible=past_query_support(grid,data['observed_support'],[[v['onset_epoch'],v['offset_epoch']] for v in index['seizures']])
        query_times=grid[eligible];query_phase=phase(query_times,bounds)
        represented=represented_first_onsets(query_times,query_phase,onsets,bounds)
        represented_2h=represented_first_onsets(query_times,query_phase,onsets,bounds,7200)
        observed_counts={p:len(v['onsets']) for p,v in represented.items()}
        observed_risk_ok=all(observed_counts[p]>=n for p,n in risk_min.items())
        output.parent.mkdir(parents=True,exist_ok=True);query_path=output.parent/(s+'_past_only_query_support.npz')
        if query_path.exists():raise FileExistsError(query_path)
        np.savez_compressed(query_path,all_grid_times=grid,past_only_eligible=eligible,query_times=query_times,phase=query_phase,
            historical_measurement_support=data['observed_support'])
        # This audit may stop a non-estimable fit. It must never silently turn
        # an unexpectedly estimable patient into a negative or skip its fitting.
        if (risk_ok and observed_risk_ok) or field_ok:raise RuntimeError('A design patient passes represented onset gates: export frozen features on the past-only risk queries and fit qualified downstream outcomes')
        rows.append(dict(subject=s,status='NOT_ESTIMABLE',onsets_by_phase=counts,onset_times=times,descriptive_clusters_by_phase=clusters,
            risk=dict(status='NOT_ESTIMABLE',minimum_onsets=risk_min,raw_gate_pass=risk_ok,observed_seizures_by_phase=observed_counts,
                reason='Insufficient distinct first onsets represented by past-only supported queries' if risk_ok else 'Insufficient distinct raw onsets'),
            past_only_queries=dict(path=str(query_path),sha256=sha(query_path),human_data_sha256=sha(data_path),
                n_by_phase={p:int(np.sum(query_phase==p)) for p in counts},represented_first_onsets_6h=represented,represented_first_onsets_2h=represented_2h,
                rule='Fresh 5-minute pre80 grid, >=90 percent of previous 8h measured support, no already-started seizure overlapping that past history; no future interictal target filtering',
                interpretation='These are upper bounds before frozen feature numerical validation and follow-up censoring; raw counts or repeated forecasts do not add independent onsets'),
            early_spatial_field_and_path=dict(status='NOT_ESTIMABLE',minimum_raw_onsets=field_min,reason='Insufficient raw FIT/INNER/SELECTION onsets before additional per-contact target support gates'),
            state_count=sum(f['subject']==s for f in freezes),index_sha256=sha(index_path),boundary_sha256=sha(boundary)))
    atomic_json(output,dict(status='COMPLETE',rows=rows,upstream_freezes=freezes,inventory_path=str(inventory),inventory_sha256=sha(inventory),
        support_contract_source='src/topic5_group_event_state/v037/h2b.py:588 raw risk gate; :871 early field first FIT gate; valid INNER remains necessary for probe selection',
        support_contract_sha256=sha(ROOT/'src/topic5_group_event_state/v037/h2b.py'),
        risk_queries_fitted=0,spatial_probes_fitted=0,h1_future_interictal_filtered_anchors_reused_as_risk_queries=False,
        interpretation='Raw and past-only represented first-onset counts are both reported. Evaluable positive queries can only target their next onset, not all subsequent cluster members. Missing denominator is not a biological negative. Clusters at 8h/24h are descriptive sensitivity counts, not assumed independent trials.',
        source_sha256=sha(__file__),development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=True,upstream_updated_with_seizure_outcomes=False))
    print(json.dumps(dict(status='COMPLETE',not_estimable_subjects=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.output)

#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.topic5_group_event_state.v039.human_data import make_human_data
from src.topic5_group_event_state.v035.contracts import atomic_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/data/hfosp_group_event_state_v0_3_9_transition_transfer'))
    args=p.parse_args();data=make_human_data(args.subject,args.root,args.output)
    rows=[]
    for hi,lead in enumerate((0.,2.,6.)):
        for phase in ('FIT','INNER','SELECTION'):
            samples=[s for s in data['samples'] if s['phase']==phase and s['targets'][hi][2]]
            independent=[]
            for s in samples:
                if not independent or s['anchor']>=independent[-1]+1800:independent.append(s['anchor'])
            rows.append(dict(lead_hours=lead,phase=phase,n_anchors=len(samples),n_nonoverlapping_target_windows=len(independent),
                             independent_anchors=independent,n_spatial_anchors=sum(s['targets'][hi][3] for s in samples)))
    card=dict(status='COMPLETE',subject=args.subject,data_path=str(args.output),data_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
              input_dim=data['input_dim'],context_dim=data['context_dim'],n_recruitment=data['n_recruitment'],selected_contacts=data['selected_contacts'],
              histories_hours=[.5,8.],leads_hours=[0,2,6],target_width_hours=.5,support=rows,
              interictal_sensor_selection=data['interictal_sensor_selection'],
              clock_only_restored_events=sum(r['restored'] for r in data['clock_restorations']),
              background_repairs=data['background_repairs'],
              background_contaminated_windows_removed=sum(r.get('n_omitted_core_contaminated_windows',0) for r in data['background_repairs']),
              empty_event_blocks_with_restored_background=sum(r.get('empty_block_background_restored',False) for r in data['background_repairs']),
              missing_fine_mark_contract=data['missing_fine_mark_contract'],
              source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/topic5_group_event_state/v039/human_data.py']},
              development_targets_read=False,sealed_partition_opened=False,seizure_outcomes_used_for_model_selection=False)
    atomic_json(args.output.with_suffix('.json'),card)
    print(json.dumps({k:card[k] for k in ('status','subject','input_dim','context_dim','support')}),flush=True)

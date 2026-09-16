"""Synthetic calibration jobs use an untouched future test after INNER selection."""
from dataclasses import replace
from pathlib import Path
import torch
from .synthetic import make_world
from . import data as D
from .train import atomic_torch,atomic_json,evaluate
from .frozen import load_selected


def generate(root,realization=901,scenario='morphology',n_blocks=24):
    if scenario not in ('morphology','identity','zero'):raise ValueError(scenario)
    p=make_world(realization,n_blocks=n_blocks,morph_gain=1.2,
                 identity_gain=1.2 if scenario=='identity' else 0.,zero_effect=scenario=='zero')
    p['instrument_target']='shared morphology-to-identity transfer' if scenario=='identity' else scenario
    p['subject']=f'synthetic_{scenario}_{realization}'
    # Retain the raw-to-measurement witness in its own file. Training never reads truth.
    witness={k:p.pop(k) for k in ('raw_measurement','raw_manifest','slow_state')}
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    atomic_torch(witness,root/f'{p["subject"]}.witness.pt');atomic_torch(p,root/f'{p["subject"]}.pt')
    return dict(status='COMPLETE',subject=p['subject'],packet_file=str(root/f'{p["subject"]}.pt'),scenario=scenario,
                n_events=len(p['event_time']),contract=p['synthetic_contract'],realization_seed=realization)


def score_untouched_future(selected,out,device='cpu'):
    model,prep,cfg,record=load_selected(selected,device)
    if not prep.payload.get('synthetic_contract'):raise PermissionError('instrument scoring is not an alternate human OUTER selection path')
    split=dict(prep.split);pk=prep.payload['packets'];cut=split['support_start']+.5*(split['support_end']-split['support_start'])
    split['outer_packet']=(pk['start']>=cut)&(pk['end']<=split['support_end'])&split['valid_packet']
    split['split_id']=D.digest((prep.split['split_id'],'untouched_synthetic_future',split['outer_packet']));prep.split=split
    result=evaluate(model,prep,cfg,'outer',collect=True)
    atomic_torch(result,out)
    return dict(status=result['status'],selected_updates=record['selected_updates'],units=result['units'],scores=result['scores'],
                interpretation='Synthetic-only future test, disjoint from checkpoint-selecting INNER; independent realizations are required for power.')

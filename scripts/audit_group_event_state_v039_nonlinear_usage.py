#!/usr/bin/env python3
"""Quantify drift curvature at frozen observed anchor states, without fitting."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition
from src.topic5_group_event_state.v035.contracts import atomic_json


def audit(root,output,main_only=False):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[]
    paths=sorted((root/'frozen_states').glob('*/state.json'))
    if not main_only:paths+=sorted((root/'frozen_view_states').glob('*/state.json'))
    if len(paths)!=(54 if main_only else 72):raise ValueError('Wait for all states in the declared audit scope')
    for path in paths:
        meta=json.loads(path.read_text());source=Path(meta['source_card']);card=json.loads(source.read_text());cfg=card['config']
        if sha(source)!=meta['source_card_sha256'] or sha(card['checkpoint'])!=meta['checkpoint_sha256']:raise ValueError('Upstream changed')
        if cfg['family']!='N':continue
        if sha(meta['export'])!=meta['export_sha256']:raise ValueError('Frozen states changed')
        with np.load(meta['export']) as z:
            states=torch.tensor(z['state'],dtype=torch.float64);phase=z['phase']
        checkpoint=torch.load(card['checkpoint'],map_location='cpu',weights_only=False)
        model=EventTransition(card['input_dim'],'N',width=cfg['width'],seed=cfg['seed']).double();model.load_state_dict(checkpoint['observer']);model.requires_grad_(False)
        with torch.no_grad():
            reference=states[phase=='FIT'].mean(0);held=states[phase=='SELECTION'];matrix=model.generator_matrix()
            z=held@model.v.T+model.bias;z0=reference@model.v.T+model.bias
            linear=held@matrix.T;term=torch.tanh(z)@model.u.T;full=linear+term
            jac0=matrix+(model.u*(1-torch.tanh(z0).square()))@model.v
            jac=matrix[None]+torch.einsum('ir,br,rj->bij',model.u,1-torch.tanh(z).square(),model.v)
            tangent=model._drift(reference,matrix)+(held-reference)@jac0.T
            rms=lambda x:float(x.square().mean().sqrt())
            row=dict(subject=card['subject'],family='N',view=cfg['view'],history_hours=cfg['history_hours'],seed=cfg['seed'],selected_step=card['stages']['event']['selected_step'],
                n_selection_anchors=len(held),drift_rms=rms(full),nonlinear_branch_rms=rms(term),curvature_residual_rms=rms(full-tangent),
                nonlinear_branch_to_full_drift=rms(term)/max(rms(full),1e-12),curvature_to_full_drift=rms(full-tangent)/max(rms(full),1e-12),
                jacobian_variation_relative_to_fit_tangent=rms(jac-jac0)/max(rms(jac0),1e-12),
                selected_transition_deltas=card['selected_transition_deltas'],source_card=str(source),source_card_sha256=sha(source),frozen_feature_card=str(path),frozen_feature_card_sha256=sha(path))
        rows.append(row)
    atomic_json(output,dict(status='COMPLETE',rows=rows,source_sha256=sha(__file__),scope='main_only' if main_only else 'main_and_single_views',
        definition='Nonlinear term and deviation from the affine drift tangent at the FIT-mean state, evaluated at frozen SELECTION anchor states. No state or readout is refitted.',
        interpretation='A nonzero nonlinear branch can still be approximately linear. Curvature documents model usage at sampled states; predictive necessity requires the separately retrained F/L/N comparisons. Neither establishes physiological nonlinear dynamics.',
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False))
    print(json.dumps(dict(status='COMPLETE',nonlinear_states=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--main-only',action='store_true');a=p.parse_args();audit(a.root,a.output,a.main_only)

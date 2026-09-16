#!/usr/bin/env python3
"""Audit remaining exhausted H1 controls without updating scientific models."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_group_event_state_v038_dual_credit import file_hash
from scripts.audit_group_event_state_v038_dual_random_background import _parent_parity_audit
from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.ctssm import DualStreamEventCTSSM, GridBackgroundCTSSM
from src.topic5_group_event_state.v037.h1_train import (
    H1TrainConfig, NestedH1Readout, EventStateComputer, GridEventStateComputer,
    _selection_score, _train_stage as train_event_stage)
from src.topic5_group_event_state.v037.h1_dual_train import (
    H1DualTrainConfig, NestedDualReadout, _train_stage as train_dual_stage)


class FrozenComputer:
    def __init__(self, model, values): self.model, self.values = model, values
    def __call__(self): return self.values


def main():
    p = argparse.ArgumentParser(); p.add_argument('--replay-card', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--stage', choices=('random', 'background_current'), required=True)
    p.add_argument('--preflight-only', action='store_true'); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.time(); replay = json.loads(args.replay_card.read_text())
    source = Path(replay['source_card']); original = json.loads(source.read_text())
    if not original['stages'][args.stage]['training_budget_exhausted']:
        raise ValueError('repair selection must follow original optimization exhaustion')
    inputs = {**replay['input_hashes'], str(args.replay_card): file_hash(args.replay_card),
              replay['replay_bundle']: replay['replay_bundle_sha256']}
    for path, digest in inputs.items():
        if file_hash(Path(path)) != digest: raise ValueError(f'source changed: {path}')
    bundle = torch.load(replay['replay_bundle'], map_location='cpu', weights_only=False)
    data, saved, tr = bundle['data'], bundle['source_checkpoint'], bundle['trajectory']
    family = replay['family']; dual = family == 'dual'
    if args.stage == 'background_current' and not dual: raise ValueError('background control is dual only')
    cfg = (H1DualTrainConfig if dual else H1TrainConfig)(**saved['config'])
    device = torch.device(args.device); tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
    q = bundle['q'].to(device); bm = tensor(tr['fixed_mark_state']); exposure = tensor(tr['target_exposure_seconds'])
    target = {k: v.to(device) for k,v in bundle['target'].items()}; valid = {k: v.to(device) for k,v in bundle['valid'].items()}
    fit, inner, sel = [torch.as_tensor(np.flatnonzero(data.rate.phase == phase), device=device) for phase in ('FIT','INNER','SELECTION')]
    horizons = tuple(original['horizons_seconds']); sel_np = sel.cpu().numpy()
    kwargs = dict(taus_seconds=cfg.taus_seconds, burden_channels_per_tau=cfg.burden_channels_per_tau,
                  grammar_channels_per_tau=cfg.grammar_channels_per_tau)
    torch.manual_seed(original['seed'])
    event = DualStreamEventCTSSM(data.burden_mark.shape[1],data.grammar_mark.shape[1],**kwargs).to(device)
    random = DualStreamEventCTSSM(data.burden_mark.shape[1],data.grammar_mark.shape[1],**kwargs).to(device)
    event.load_state_dict(saved['event_observer' if dual else 'observer'])
    computer = (GridEventStateComputer(data,random,device,grid_seconds=saved['grid_seconds']) if family == 'grid'
                else EventStateComputer(data,random,device))
    with torch.no_grad():
        random_state = torch.clamp((computer()-tensor(saved['random_centre']))/tensor(saved['random_scale']),-12,12)
    es = tensor(tr['event_state' if dual else 'learned_state'])
    bd = len(cfg.taus_seconds)*data.burden_mark.shape[1]; ed = len(cfg.taus_seconds)*cfg.burden_channels_per_tau
    if dual:
        bc, bs = tensor(tr['background_current']), tensor(tr['background_state'])
        bg = GridBackgroundCTSSM(bc.shape[1],taus_seconds=cfg.taus_seconds,channels_per_tau=cfg.background_channels_per_tau).to(device)
        bg.load_state_dict(saved['background_observer'])
        model = NestedDualReadout(q.shape[1],bd,bm.shape[1]-bd,bc.shape[1],bs.shape[1],ed,es.shape[1]-ed,saved['widths'],len(horizons)).to(device)
        arm = 'random_event' if args.stage == 'random' else 'B_mark_current_background'
        pred_kw = dict(bmark=bm,background_current=bc)
        if args.stage == 'random': pred_kw.update(background_state=bs,random_event_state=random_state)
    else:
        model = NestedH1Readout(q.shape[1],bd,bm.shape[1]-bd,ed,es.shape[1]-ed,saved['widths'],len(horizons)).to(device)
        arm = 'random_frozen'; pred_kw = dict(bmark=bm,random_state=random_state)
    model.load_state_dict(saved['readout'])
    def score():
        with torch.no_grad():
            return _selection_score(model.predict(q,**pred_kw),target,valid,exposure,model.log_dispersion,sel,sel_np,horizons)
    try: parity = _parent_parity_audit(score(),original['selection_scores'][arm],horizons)
    except ValueError as error: parity = dict(all_endpoint_total_auditable=False,reason=str(error))
    common = dict(repair_source_card=str(source),repair_source_sha256=inputs[str(source)],repair_input_hashes=inputs,
                  repair_stage=args.stage,repair_original_stage=original['stages'][args.stage],repair_score_parity=parity,
                  repair_upstream_checkpoint_updated=False,development_targets_read=False,sealed_partition_opened=False,
                  seizure_targets_read=False,repair_preflight_only=args.preflight_only)
    if not parity['all_endpoint_total_auditable']:
        atomic_json(args.output,{**original,**common,'status':'NOT_ESTIMABLE',
                                'repair_reason':'original control score could not be reproduced; source-limited, not a scientific negative'})
        print('NOT_ESTIMABLE: source parity',flush=True); return
    if args.preflight_only:
        atomic_json(args.output,{'status':'COMPLETE',**common}); print('COMPLETE: parity preflight',flush=True); return
    for layer in getattr(model,args.stage).values():
        with torch.no_grad(): layer.weight.zero_()
    extended = replace(cfg,**{f'max_steps_{args.stage}':2700})
    if dual:
        stage = train_dual_stage(args.stage,model,q,bm,bc,FrozenComputer(bg,bs),FrozenComputer(event,es),random_state,
                                 target,valid,exposure,fit,inner,extended)
    else:
        stage = train_event_stage(stage='random',readout=model,q=q,bmark=bm,state_computer=None,
                                   fixed_random=random_state,target=target,valid=valid,exposure=exposure,
                                   fit_rows=fit,inner_rows=inner,config=extended)
    for name,value in model.state_dict().items():
        if not name.startswith(args.stage+'.') and not torch.equal(value.cpu(),saved['readout'][name]):
            raise ValueError(f'non-control parameter changed: {name}')
    args.output.parent.mkdir(parents=True,exist_ok=True); checkpoint = args.output.with_name('control_checkpoint.pt')
    torch.save({'control':getattr(model,args.stage).state_dict(),'source_hashes':inputs,'config':asdict(extended),
                'random_observer':random.state_dict(),'input_boundary':replay['replay_bundle']},checkpoint)
    result = {**original,**common,'status':'COMPLETE','stages':{**original['stages'],args.stage:stage},
              'selection_scores':{**original['selection_scores'],arm:score()},
              'repair_control_checkpoint':str(checkpoint),'repair_control_checkpoint_sha256':file_hash(checkpoint),
              'repair_config':asdict(extended),'elapsed_seconds':time.time()-started,
              'repair_interpretation':'independent control head from original zero start; original scientific model and other arms remain frozen; no downstream reoptimization'}
    for path,digest in inputs.items():
        if file_hash(Path(path)) != digest: raise ValueError(f'input changed during training: {path}')
    atomic_json(args.output,result); print(json.dumps({'status':'COMPLETE','stage':args.stage,**{k:stage[k] for k in ('steps_run','selected_step','training_budget_exhausted')}}),flush=True)


if __name__ == '__main__': main()

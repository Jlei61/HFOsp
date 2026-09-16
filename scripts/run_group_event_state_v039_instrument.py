#!/usr/bin/env python3
"""Finite, reproducible transition instrument fit; no human outcomes are opened."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition, FutureReadout, endpoint_loss
from src.topic5_group_event_state.v039.synthetic import CASES, generate
from src.topic5_group_event_state.v035.contracts import atomic_json


def run(args):
    start=time.time(); torch.set_num_threads(1); torch.manual_seed(args.seed)
    output=args.output; output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    data=generate(args.case,n=args.episodes,strength=args.strength)
    device=torch.device(args.device)
    x=torch.from_numpy(data['inputs']).to(device); dt=torch.from_numpy(data['dt_hours']).to(device)
    context=torch.from_numpy(data['context']).to(device)
    y={k:{name:torch.from_numpy(value).to(device) for name,value in values.items()} for k,values in data['targets'].items()}
    fit_end=data['fit_end']; inner_end=data['inner_end']; lead=args.lead
    observer=EventTransition(x.shape[-1],args.family,width=args.width,seed=args.seed).to(device)
    readout=FutureReadout(observer.width,2,2).to(device)
    initial=copy.deepcopy(observer.state_dict())
    optimizer=torch.optim.AdamW(list(observer.parameters())+list(readout.parameters()),lr=args.lr,weight_decay=1e-4)
    rng=torch.Generator(device=device).manual_seed(args.seed+1)
    def predict(rows,requires_grad=False):
        state=observer.scan(x[rows],dt[rows],checkpoint_chunk=32 if requires_grad else 0)
        if args.forecast=='rollout': state=observer.advance(state,lead)
        return readout(state,context[rows],lead)
    def score(rows):
        total=[]; count=[]; spatial=[]
        with torch.no_grad():
            for selected in rows.split(256):
                mu,logits=predict(selected)
                losses=endpoint_loss(mu,logits,y[str(lead)]['count'][selected],y[str(lead)]['recruitment'][selected],readout.log_dispersion,args.view)
                for dest,val in zip((total,count,spatial),losses): dest.append(val)
        return {'total':torch.cat(total),'count':torch.cat(count),'recruitment':torch.cat(spatial)}
    inner_rows=torch.arange(fit_end,inner_end,device=device)
    initial_inner=float(score(inner_rows)['total'].mean())
    best=initial_inner; selected_step=0; stale=0
    best_state=(copy.deepcopy(observer.state_dict()),copy.deepcopy(readout.state_dict()))
    logs=[]; first_gradient=None; reason='BUDGET_LIMIT'
    for step in range(1,args.max_steps+1):
        rows=torch.randint(fit_end,(min(args.batch_size,fit_end),),generator=rng,device=device)
        optimizer.zero_grad(set_to_none=True)
        mu,logits=predict(rows,True)
        losses=endpoint_loss(mu,logits,y[str(lead)]['count'][rows],y[str(lead)]['recruitment'][rows],readout.log_dispersion,args.view)
        loss=losses[0].mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training objective')
        loss.backward()
        gradients={name:float(p.grad.norm()) for name,p in observer.named_parameters() if p.grad is not None}
        if first_gradient is None: first_gradient=gradients
        norm=torch.nn.utils.clip_grad_norm_(list(observer.parameters())+list(readout.parameters()),2.,error_if_nonfinite=True)
        optimizer.step()
        if step%args.eval_every==0 or step==args.max_steps:
            inner=float(score(inner_rows)['total'].mean())
            logs.append(dict(step=step,fit_batch=float(loss.detach()),inner=inner,gradient_norm=float(norm),lr=args.lr))
            if inner < best-1e-5:
                best=inner; selected_step=step; stale=0
                best_state=(copy.deepcopy(observer.state_dict()),copy.deepcopy(readout.state_dict()))
            else: stale+=1
            atomic_json(output.with_name('progress.json'),dict(status='RUNNING',step=step,selected_step=selected_step,inner=inner,elapsed_seconds=time.time()-start))
            if stale>=args.patience:
                reason='INNER_PATIENCE'; break
    observer.load_state_dict(best_state[0]); readout.load_state_dict(best_state[1])
    selection=score(torch.arange(inner_end,len(x),device=device))
    check=output.with_suffix('.pt')
    torch.save(dict(observer=best_state[0],readout=best_state[1],config=vars(args),truth=data['truth']),check)
    scores=output.with_suffix('.npz')
    np.savez_compressed(scores,**{k:v.cpu().numpy() for k,v in selection.items()},episode_id=np.arange(inner_end,len(x)))
    delta={name:float((value-initial[name]).abs().max()) for name,value in observer.state_dict().items()}
    credit={}
    if args.family!='C':
        old=x[inner_end:inner_end+4].detach().clone().requires_grad_(True)
        state=observer.scan(old,dt[inner_end:inner_end+4],checkpoint_chunk=32)
        if args.forecast=='rollout': state=observer.advance(state,lead)
        mu,logits=readout(state,context[inner_end:inner_end+4],lead)
        losses=endpoint_loss(mu,logits,y[str(lead)]['count'][inner_end:inner_end+4],y[str(lead)]['recruitment'][inner_end:inner_end+4],readout.log_dispersion,args.view)[0]
        grad=torch.autograd.grad(losses.mean(),old)[0].abs().sum((0,2))
        credit={'last_2h':float(grad[-24:].sum()),'2_to_6h':float(grad[-72:-24].sum()),'6_to_8h':float(grad[:-72].sum())}
    source_paths=[Path(__file__),ROOT/'src/topic5_group_event_state/v039/transition.py',ROOT/'src/topic5_group_event_state/v039/synthetic.py']
    card=dict(status='COMPLETE',format='v039_transition_instrument_v1',case=args.case,family=args.family,seed=args.seed,
              config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},truth=data['truth'],
              initial_inner=initial_inner,selected_inner=best,selected_step=selected_step,steps_run=step,stop_reason=reason,
              optimization_limited=reason=='BUDGET_LIMIT',first_step_transition_gradients=first_gradient,selected_parameter_deltas=delta,
              selected_parameter_inventory={name:dict(shape=list(value.shape),numel=value.numel()) for name,value in observer.named_parameters()},
              parameter_count=sum(p.numel() for p in observer.parameters())+sum(p.numel() for p in readout.parameters()),
              held_out={k:float(v.mean()) for k,v in selection.items()},held_out_event_gradient=credit,training_curve=logs,
              checkpoint=str(check),checkpoint_sha256=hashlib.sha256(check.read_bytes()).hexdigest(),
              scores=str(scores),scores_sha256=hashlib.sha256(scores.read_bytes()).hexdigest(),
              source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              input_sha256=hashlib.sha256(data['inputs'].tobytes()+data['context'].tobytes()).hexdigest(),
              development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,human_data_read=False,
              elapsed_seconds=time.time()-start)
    atomic_json(output,card);print(json.dumps({k:card[k] for k in ('status','case','family','selected_step','steps_run','held_out','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=CASES,required=True);parser.add_argument('--family',choices=('C','F','L','N'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=20260905);parser.add_argument('--width',type=int,default=16)
    parser.add_argument('--lr',type=float,default=.003);parser.add_argument('--max-steps',type=int,default=480)
    parser.add_argument('--eval-every',type=int,default=40);parser.add_argument('--patience',type=int,default=6)
    parser.add_argument('--batch-size',type=int,default=128);parser.add_argument('--episodes',type=int,default=3072)
    parser.add_argument('--lead',type=int,choices=(0,2,6),default=2);parser.add_argument('--strength',type=float,default=1.)
    parser.add_argument('--forecast',choices=('rollout','direct'),default='direct')
    parser.add_argument('--view',choices=('count','recruitment','joint'),default='joint')
    run(parser.parse_args())

#!/usr/bin/env python3
"""Freeze a selected single-view observer, then fit identical low-capacity probes.

Raw latent transfer and transfer through the trained task readout are kept
separate: unused latent coordinates can retain a second independent cause.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition,FutureReadout
from src.topic5_group_event_state.v039.synthetic import generate
from src.topic5_group_event_state.v035.contracts import atomic_json


def ridge(x,y,fit_end,inner_end):
    center=x[:fit_end].mean(0);scale=x[:fit_end].std(0);scale=np.where(scale>1e-6,scale,1.)
    features=np.concatenate((np.ones((len(x),1)),(x-center)/scale),axis=1)
    fit=features[:fit_end];target=y[:fit_end];lhs=fit.T@fit/fit_end;rhs=fit.T@target/fit_end
    candidates=[]
    for alpha in (.01,.1,1.,10.):
        penalty=np.eye(fit.shape[1])*alpha;penalty[0,0]=0
        weights=np.linalg.solve(lhs+penalty,rhs)
        inner=float(np.mean((features[fit_end:inner_end]@weights-y[fit_end:inner_end])**2))
        candidates.append((inner,alpha,weights))
    score,alpha,weights=min(candidates,key=lambda r:(r[0],r[1]))
    error=(features[inner_end:]@weights-y[inner_end:])**2
    if error.ndim>1:error=error.mean(axis=-1)
    return dict(inner_mse=score,alpha=alpha,weights=weights,center=center,scale=scale,held_out_squared_error=error)


def run(source,output,device='cpu'):
    start=time.time();torch.set_num_threads(1)
    if output.exists():raise FileExistsError(output)
    card=json.loads(source.read_text());cfg=card['config'];weight_path=Path(card['checkpoint'])
    if hashlib.sha256(weight_path.read_bytes()).hexdigest()!=card['checkpoint_sha256']:raise ValueError('Upstream checkpoint changed')
    data=generate(cfg['case'],n=cfg['episodes'],strength=cfg['strength'])
    if hashlib.sha256(data['inputs'].tobytes()+data['context'].tobytes()).hexdigest()!=card['input_sha256']:raise ValueError('Input replay mismatch')
    state=torch.load(weight_path,map_location='cpu',weights_only=False)
    observer=EventTransition(5,cfg['family'],width=cfg['width'],seed=cfg['seed']);observer.load_state_dict(state['observer']);observer.requires_grad_(False)
    initial=EventTransition(5,cfg['family'],width=cfg['width'],seed=cfg['seed']);initial.requires_grad_(False)
    history=EventTransition(5,'F');history.requires_grad_(False)
    readout=FutureReadout(observer.width,2,2);readout.load_state_dict(state['readout']);readout.requires_grad_(False)
    observer.to(device);initial.to(device);history.to(device);readout.to(device)
    x=torch.from_numpy(data['inputs']).to(device);dt=torch.from_numpy(data['dt_hours']).to(device);context=torch.from_numpy(data['context']).to(device)
    raw=[];null=[];fixed=[];functional=[]
    with torch.no_grad():
        for start_row in range(0,len(x),256):
            rows=slice(start_row,start_row+256);s=observer.scan(x[rows],dt[rows],checkpoint_chunk=0)
            raw.append(s.cpu().numpy());null.append(initial.scan(x[rows],dt[rows],checkpoint_chunk=0).cpu().numpy());fixed.append(history.scan(x[rows],dt[rows],checkpoint_chunk=0).cpu().numpy())
            mu,logits=readout(s,context[rows],cfg['lead'])
            functional.append((mu[:,None] if cfg['view']=='count' else logits).cpu().numpy())
    context=context.cpu().numpy().astype(float)
    features={'background':context,'trained_raw_latent':np.c_[np.concatenate(raw),context],
              'initialized_raw_latent':np.c_[np.concatenate(null),context],
              'full_fixed_history':np.c_[np.concatenate(fixed),context],
              'trained_view_readout':np.c_[np.concatenate(functional),context]}
    target=data['targets'][str(cfg['lead'])]
    targets={'untrained_fine_expression':target['fine'].astype(float)}
    if cfg['view']=='recruitment':targets['cross_count_log1p']=np.log1p(target['count']).astype(float)
    # Keep contact identity. Oppositely modulated contacts can have a constant
    # mean while their spatial distribution is highly predictable.
    else:targets['cross_recruitment_distribution']=target['recruitment'].astype(float)
    output.parent.mkdir(parents=True,exist_ok=True);arrays={};results={}
    for name,y in targets.items():
        cells={arm:ridge(values,y,data['fit_end'],data['inner_end']) for arm,values in features.items()}
        results[name]={}
        for arm,cell in cells.items():
            for key,value in cell.items():arrays[name+'__'+arm+'__'+key]=np.asarray(value)
            gains=cells['background']['held_out_squared_error']-cell['held_out_squared_error']
            results[name][arm]=dict(selected_alpha=cell['alpha'],inner_mse=cell['inner_mse'],held_out_mse=float(cell['held_out_squared_error'].mean()),
                                   gain_over_background=float(gains.mean()),n_independent_episodes=len(gains))
    saved=output.with_suffix('.npz');np.savez_compressed(saved,**arrays)
    if hashlib.sha256(weight_path.read_bytes()).hexdigest()!=card['checkpoint_sha256']:raise ValueError('Probe modified upstream')
    result=dict(status='COMPLETE',case=cfg['case'],family=cfg['family'],view=cfg['view'],seed=cfg['seed'],truth=data['truth'],results=results,
                upstream_source_card=str(source),upstream_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint_sha256=card['checkpoint_sha256'],
                upstream_optimization_limited=card['optimization_limited'],upstream_training_steps=card['selected_step'],
                probe_fit='FIT-only standardisation and ridge; INNER selects ridge only after upstream freezing',
                interpretation='Transfer of raw latent coordinates alone is not evidence of one common physiological cause; compare trained task readout and independent-state truth.',
                weights_and_errors=str(saved),weights_and_errors_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),device=device,
                development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,human_data_read=False,elapsed_seconds=time.time()-start)
    atomic_json(output,result);print(json.dumps({k:result[k] for k in ('status','case','family','view','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args();run(args.source,args.output,args.device)

#!/usr/bin/env python
"""Morphology information world: P_stats/P_marks x old/new targets, two realisations."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch

from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311.synthetic import make_world
from src.topic5_group_event_state.v0311.prepare import Prepared
from src.topic5_group_event_state.v0311.train import (RunConfig,build_run,episode_plan,filter_episodes,
    rollout_from_queries,combined_loss,evaluate_starts,main_views_for)
from src.topic5_group_event_state.v0311.objective import VIEWS
from src.topic5_group_event_state.v0310.trainer import build_param_groups,PlateauSchedule

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')


def build_split_synthetic(payload,seed=20260906):
    """No seizures and full exposure; the S-E forward rule is otherwise identical."""
    sp=D.build_split.__wrapped__ if hasattr(D.build_split,'__wrapped__') else None
    support=payload['observed_support'];pk=payload['packets']
    start=float(support[0,0]);end=float(payload['phase_boundaries']['80pct'])
    cutoff=start+0.5*(end-start);fit_end=start+0.6*(cutoff-start)
    rng=np.random.default_rng(seed)
    rel=np.array([b['release'] for b in payload['blocks']])
    gap=float(rel[rel>cutoff].min())
    grid=np.arange(start,end,1800.)
    inner=D._scatter_blocks(grid[(grid>=start+4*3600.)&(grid+7200.<=cutoff)],rng)
    forward=grid[(grid>=gap)&(grid+7200.<=end)]
    return dict(support_start=start,support_end=end,se_cutoff=cutoff,se_gap_start=gap,fit_end=fit_end,
                train_packet=pk['end']<=cutoff,inner_starts=inner,forward_starts=forward,
                inner_target_times=np.concatenate([inner+h for h in (60.,300.,1800.,7200.)]),
                excluded_intervals=np.empty((0,2)),seizures=[],seed=int(seed),
                contract='synthetic S-E: forward segment after the 50% cutoff')


def fit_one(payload,inputs,targets,device,updates,seed=20260906):
    split=build_split_synthetic(payload,seed)
    px,pt,_=D.packet_tables(payload,split);scaling=D.fit_scaling(payload,split,px,pt)
    dev=torch.device(device);prep=Prepared(payload,split,scaling,dev)
    cfg=RunConfig(subject=payload['subject'],inputs=inputs,family='I-L-G1',device=device,
                  max_updates=updates,extended_updates=updates,seed=seed,
                  target_ablation=(targets=='old'))
    model,ref=build_run(cfg,payload,split,scaling,prep)
    groups,_=build_param_groups([('m',model)])
    opt=torch.optim.AdamW(groups,lr=cfg.lr,betas=(0.9,0.999),eps=1e-8)
    plateau=PlateauSchedule(cfg.lr,eval_every=cfg.eval_every)
    cand,q=episode_plan(prep,split,cfg)
    rng=np.random.default_rng(seed);gen=torch.Generator(device=dev).manual_seed(seed)
    mv=main_views_for(cfg)
    best=dict(score=float('inf'),state=None,updates=-1)
    for u in range(1,updates+1):
        starts=rng.choice(cand,size=cfg.batch_episodes,replace=False)
        opt.zero_grad(set_to_none=True)
        pl,pu,snap=filter_episodes(model,prep,starts,q,cfg,gen,training=True)
        rl,ru,_=rollout_from_queries(model,prep,snap,cfg,gen,training=True)
        loss=combined_loss(pl,pu,rl,ru,views=mv)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        for g in opt.param_groups:g['lr']=plateau.lr
        opt.step()
        if u%cfg.eval_every==0:
            ev=evaluate_starts(model,prep,split,cfg,gen,split['inner_starts'])
            if ev['selection'] is not None and ev['selection']<best['score']:
                best=dict(score=ev['selection'],state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},updates=u)
            if plateau.observe(u,ev['selection'] or float('inf'))=='stop':break
    if best['state'] is not None:model.load_state_dict(best['state'])
    out=evaluate_starts(model,prep,split,cfg,gen,split['forward_starts'],paths=cfg.eval_paths)
    return dict(inputs=inputs,targets=targets,inner=best['score'],selected_updates=best['updates'],
                outer=out['per_horizon'],n_outer=out['n_eligible'])


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--updates',type=int,default=400)
    ap.add_argument('--realisations',type=int,default=2)
    ap.add_argument('--mode',default='morphology',choices=('morphology','identity','zero_effect'))
    a=ap.parse_args()
    rows=[];t0=time.time()
    for r in range(a.realisations):
        kw=dict(seed=100+r)
        if a.mode=='identity':kw.update(morph_gain=0.0,identity_gain=1.5)
        if a.mode=='zero_effect':kw.update(zero_effect=True)
        world=make_world(**kw)
        for inputs in ('P_stats','P_marks'):
            for targets in ('old','new'):
                r0=fit_one(world,inputs,targets,a.device,a.updates)
                r0['realisation']=r;r0['mode']=a.mode
                rows.append(r0)
                print(json.dumps({k:r0[k] for k in ('realisation','inputs','targets','inner','n_outer')},default=str),flush=True)
    p=ROOT/'synthetic'/f'{a.mode}.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(dict(mode=a.mode,updates=a.updates,seconds=round(time.time()-t0,1),rows=rows,
        contract=('rate and coarse composition are constant by construction; only shape and propagation '
                  'follow the slow state. This is an interface and discrimination screen, not a '
                  'calibrated power analysis.')),indent=1,default=str))

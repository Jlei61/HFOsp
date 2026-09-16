#!/usr/bin/env python
"""Measure real per-update cost and peak memory before setting the ETA."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from src.topic5_group_event_state.v0311.train import (RunConfig,build_run,episode_plan,filter_episodes,
    rollout_from_queries,combined_loss,evaluate_starts,main_views_for)
from src.topic5_group_event_state.v0311.prepare import Prepared
from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0310.trainer import build_param_groups

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--subject',default='epilepsiae_1125');ap.add_argument('--inputs',default='P_marks')
    ap.add_argument('--family',default='I-L-G1');ap.add_argument('--arm',default='state')
    ap.add_argument('--updates',type=int,default=30);ap.add_argument('--batch-episodes',type=int,default=8)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--split',default='S-E')
    ap.add_argument('--out',default=None)
    a=ap.parse_args()
    cfg=RunConfig(subject=a.subject,inputs=a.inputs,family=a.family,arm=a.arm,
                  batch_episodes=a.batch_episodes,device=a.device,split=a.split)
    payload=torch.load(f'{cfg.packets_root}/{cfg.subject}.pt',weights_only=False)
    split=D.build_split(payload,cfg.subject,cfg.seed) if cfg.split=='S-E' else D.build_split_id(payload,cfg.subject,cfg.seed)
    px,pt,_=D.packet_tables(payload,split);scaling=D.fit_scaling(payload,split,px,pt)
    dev=torch.device(cfg.device);prep=Prepared(payload,split,scaling,dev)
    model,ref=build_run(cfg,payload,split,scaling,prep)
    module=model if model is not None else ref
    groups,_=build_param_groups([('m',module)])
    opt=torch.optim.AdamW(groups,lr=cfg.lr)
    cand,q=episode_plan(prep,split,cfg)
    rng=np.random.default_rng(0);gen=torch.Generator(device=dev).manual_seed(0)
    mv=main_views_for(cfg)
    torch.cuda.reset_peak_memory_stats(dev)
    times=[]
    for u in range(a.updates):
        t=time.time()
        starts=rng.choice(cand,size=cfg.batch_episodes,replace=False)
        opt.zero_grad(set_to_none=True)
        pl,pu,snap=filter_episodes(model,prep,starts,q,cfg,gen,training=True,reference=ref)
        rl,ru,_=rollout_from_queries(model,prep,snap,cfg,gen,training=True,reference=ref)
        loss=combined_loss(pl,pu,rl,ru,views=mv)
        loss.backward();torch.nn.utils.clip_grad_norm_(module.parameters(),2.0);opt.step()
        torch.cuda.synchronize(dev);times.append(time.time()-t)
    t=time.time()
    ev=evaluate_starts(model,prep,split,cfg,gen,split['inner_starts'],reference=ref)
    torch.cuda.synchronize(dev);inner_s=time.time()-t
    t=time.time()
    out=evaluate_starts(model,prep,split,cfg,gen,split['forward_starts'],reference=ref)
    torch.cuda.synchronize(dev);outer_s=time.time()-t
    n_par=sum(p.numel() for p in module.parameters())
    r=dict(subject=a.subject,inputs=a.inputs,family=a.family,arm=a.arm,split=a.split,
           batch_episodes=a.batch_episodes,n_candidates=int(len(cand)),
           median_update_s=float(np.median(times[3:])) if len(times)>3 else float(np.median(times)),
           first_update_s=round(times[0],2),peak_gib=round(torch.cuda.max_memory_allocated(dev)/2**30,2),
           inner_eval_s=round(inner_s,1),outer_eval_s=round(outer_s,1),
           n_inner=int(len(split['inner_starts'])),n_forward=int(len(split['forward_starts'])),
           inner_eligible=ev['n_eligible'],outer_eligible=out['n_eligible'],
           outer_not_estimable=out['not_estimable'],n_parameters=int(n_par),
           grad_state_fraction=snap['grad_state_fraction'],
           est_3200_minutes=round((np.median(times[3:])*3200+64*inner_s)/60,1))
    print(json.dumps(r,default=str))
    if a.out:Path(a.out).write_text(json.dumps(r,indent=1,default=str))

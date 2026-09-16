#!/usr/bin/env python
"""Run one v0.3.11 fit (state arm or reference arm)."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.topic5_group_event_state.v0311.train import RunConfig,run_cell

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k,v in RunConfig.__dataclass_fields__.items():
        t=v.type if isinstance(v.type,str) else v.type.__name__
        if k=='subject':ap.add_argument('--subject',required=True);continue
        arg='--'+k.replace('_','-')
        if t=='bool':ap.add_argument(arg,action='store_true')
        elif t=='int':ap.add_argument(arg,type=int,default=None)
        elif t=='float':ap.add_argument(arg,type=float,default=None)
        else:ap.add_argument(arg,type=str,default=None)
    a=ap.parse_args()
    # `0` is a legal value for warm_packets and q_first; `x not in (None, False)`
    # would silently drop it because 0 == False.
    kw={}
    for k,f in RunConfig.__dataclass_fields__.items():
        v=getattr(a,k,None)
        t=f.type if isinstance(f.type,str) else f.type.__name__
        if v is None:continue
        if t=='bool' and v is False:continue
        kw[k]=v
    cfg=RunConfig(**kw)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    t=time.time()
    def progress(u,s,lr,act):
        print(f'[{cfg.subject} {cfg.inputs} {cfg.family} {cfg.arm}] u={u} inner={s} lr={lr:.2e} {act} '
              f'({time.time()-t:.0f}s)',flush=True)
    card=run_cell(cfg,progress=progress)
    print(json.dumps({k:card.get(k) for k in ('status','updates','stop_reason','inner_selection',
                                              'outer_eligible','peak_gpu_gib','seconds')},default=str))

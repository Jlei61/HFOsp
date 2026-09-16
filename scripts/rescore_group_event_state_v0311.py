#!/usr/bin/env python
"""Re-score frozen checkpoints on the held-out segment with the corrected input mask.

Training is unaffected: episodes always lay inside the fitting region, where the
two masks agree. Only the held-out scoring pass changes, so this rewrites the
outer table without retraining anything.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch

from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311.prepare import Prepared
from src.topic5_group_event_state.v0311.train import (RunConfig,build_run,evaluate_starts,
                                                      apply_ablations)

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
FIX=('held-out scoring previously multiplied the readable-input mask by train_packet, '
     'so every forward-segment query ran on a prior-only state; inputs are now read '
     'up to the query while parameter fitting and target scoring stay restricted')


EVAL_SEED=20260906


def rescore(card_path,device,paths=64):
    card=json.loads(Path(card_path).read_text())
    cfg=RunConfig(**{**card['config'],'device':device,'eval_paths':paths})
    payload=torch.load(f'{cfg.packets_root}/{cfg.subject}.pt',weights_only=False)
    split=(D.build_split(payload,cfg.subject,cfg.seed) if cfg.split=='S-E'
           else D.build_split_id(payload,cfg.subject,cfg.seed))
    px,pt,_=D.packet_tables(payload,split);scaling=D.fit_scaling(payload,split,px,pt)
    dev=torch.device(device);prep=Prepared(payload,split,scaling,dev)
    apply_ablations(prep,cfg)
    model,reference=build_run(cfg,payload,split,scaling,prep)
    module=model if model is not None else reference
    ck=torch.load(Path(card_path).with_suffix('').with_suffix('.ckpt.pt'),
                  weights_only=False,map_location=dev)
    module.load_state_dict(ck['state_dict']);module.eval()
    # A common evaluation seed for every card: otherwise a seed replicate would also
    # redraw the Monte-Carlo paths and the measured spread would mix optimiser noise
    # with sampling noise.
    gen=torch.Generator(device=dev).manual_seed(EVAL_SEED)
    with torch.no_grad():
        inner=evaluate_starts(model,prep,split,cfg,gen,split['inner_starts'],reference=reference)
        outer=evaluate_starts(model,prep,split,cfg,gen,split['forward_starts'],reference=reference,
                              collect=True)
    new=dict(card)
    new['outer_prefix_bug']=dict(outer=card['outer'],outer_units=card['outer_units'],
                                 note='superseded held-out table, kept for the record')
    new.update(outer=outer['per_horizon'],outer_units=outer['units'],
               outer_eligible=outer['n_eligible'],outer_requested=outer['n_requested'],
               outer_not_estimable=outer['not_estimable'],
               outer_effective_history_hours=outer['effective_history_hours'],
               inner_final=inner['per_horizon'],inner_eligible=inner['n_eligible'],
               rescored=dict(when=time.strftime('%Y-%m-%dT%H:%M:%S%z'),fix=FIX,eval_paths=paths,
                             common_evaluation_seed=EVAL_SEED))
    return new,outer


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cpu');ap.add_argument('--paths',type=int,default=64)
    ap.add_argument('--dirs',nargs='+',default=['runs','runs_extended','runs_reference_variants'])
    ap.add_argument('--only',default=None)
    a=ap.parse_args()
    done=0
    for d in a.dirs:
        src=ROOT/d;out=ROOT/f'{d}_rescored';out.mkdir(parents=True,exist_ok=True)
        for f in sorted(src.glob('*.card.json')):
            if a.only and a.only not in f.name:continue
            tgt=out/f.name
            if tgt.exists() and json.loads(tgt.read_text()).get('rescored'):continue
            try:
                new,o=rescore(f,a.device,a.paths)
            except Exception as e:
                print(f'FAILED {f.name}: {type(e).__name__}: {e}',flush=True);continue
            tmp=tgt.with_suffix('.tmp')
            tmp.write_text(json.dumps(new,indent=1,default=str));tmp.rename(tgt)
            np.savez_compressed(out/f.name.replace('.card.json','.outer_detail.npz'),
                                **{f"{r['horizon']}_{i}_{k}":v for i,r in enumerate(o['detail'])
                                   for k,v in r.items() if isinstance(v,np.ndarray)})
            done+=1
            print(f"{f.name} n_outer={o['n_eligible']} h30_count="
                  f"{o['per_horizon'][30]['count']:.4f}",flush=True)
    print(json.dumps(dict(rescored=done)))

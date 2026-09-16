"""Paired development summaries from the registered queue, preserving physical units."""
import json
from pathlib import Path
import numpy as np
import torch
from .train import atomic_json,tag,RunConfig


def flatten(result,view):
    out={}
    for r in result['rows']:
        for i,t in enumerate(r['packet']):
            key=(int(r['horizon']),int(t));u=float(r['units'][view][i])
            if u<=0:continue
            if key in out:raise ValueError('physical target/horizon scored twice')
            out[key]=(float(r['logp'][view][i]),u)
    return out


def paired(a,b,view):
    aa=flatten(a,view);bb=flatten(b,view);out=[]
    for h in (1,5,30,120):
        keys=sorted(k for k in aa.keys()&bb.keys() if k[0]==h)
        if not keys:
            out.append(dict(horizon=h,status='NOT_ESTIMABLE'));continue
        if any(aa[k][1]!=bb[k][1] for k in keys):raise ValueError('paired target units differ')
        den=sum(aa[k][1] for k in keys)
        gain=sum(aa[k][0]-bb[k][0] for k in keys)/den
        # Packet grid starts at the same frozen epoch for all arms. Hour blocks
        # quantify temporal replication; optimization seeds are never patients.
        blocks=sorted(set(k[1]//60 for k in keys));rng=np.random.default_rng(827)
        grouped=np.array([[sum(aa[k][0]-bb[k][0] for k in keys if k[1]//60==g),sum(aa[k][1] for k in keys if k[1]//60==g)] for g in blocks])
        ci=None
        if len(blocks)>=3:
            draws=grouped[rng.integers(len(blocks),size=(1000,len(blocks)))].sum(1);ci=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist()
        out.append(dict(horizon=h,status='DEVELOPMENT',gain_a_over_b=gain,units=den,n_targets=len(keys),n_hour_blocks=len(blocks),
            descriptive_hour_block_ci=ci,lost_a=len([k for k in aa if k[0]==h])-len(keys),lost_b=len([k for k in bb if k[0]==h])-len(keys)))
    return out


def summarize(plan_path,out_path):
    plan=json.loads(Path(plan_path).read_text());root=Path(plan['root']);tasks=[];runs=[]
    for task in plan['tasks']:
        p=root/'queue_state'/(task['id']+'.json');status=json.loads(p.read_text()) if p.exists() else dict(status='PENDING')
        tasks.append(dict(id=task['id'],kind=task['kind'],**status))
        if task['kind']=='train' and status['status']=='COMPLETE':
            cfg=RunConfig(**task['config']);d=Path(cfg.out_dir)/tag(cfg)
            card=json.loads((d/'card.json').read_text())
            runs.append(dict(task=task['id'],directory=str(d),config=task['config'],card=card))
    pairs=[];floor=[]
    outers=[r for r in runs if r['config']['stage'] in ('outer','sid')]
    def same(a,b):return all(a['config'][k]==b['config'][k] for k in ('subject','protocol','stage','seed','history_hours','crossview','old_targets'))
    for a in outers:
        if a['config']['inputs']!='P_marks' or a['config']['arm']!='state':continue
        for b in outers:
            if not same(a,b):continue
            if b is a:continue
            x=torch.load(Path(a['directory'])/'predictions.pt',weights_only=False,map_location='cpu');y=torch.load(Path(b['directory'])/'predictions.pt',weights_only=False,map_location='cpu')
            for v in ('count','spatial','morphology','load'):
                pairs.append(dict(a=a['task'],b=b['task'],seed=a['config']['seed'],view=v,rows=paired(x,y,v),
                    learned_a=a['card']['selected_updates']>0,learned_b=b['card']['selected_updates']>0))
        # Whole-endpoint reference floor, not an impossible per-target oracle.
        refs=[r for r in outers if same(a,r) and r['config']['arm']!='state']
        if refs:
            x=torch.load(Path(a['directory'])/'predictions.pt',weights_only=False,map_location='cpu')
            for v in ('count','spatial','morphology','load'):
                fa=flatten(x,v);ff=[flatten(torch.load(Path(r['directory'])/'predictions.pt',weights_only=False,map_location='cpu'),v) for r in refs]
                ref_labels=[r['task'] for r in refs]
                candidates=sorted((root/'baselines').glob('*.fixed_distributions.pt')) if a['config']['stage']=='outer' else []
                if candidates:
                    ff.append(flatten(torch.load(candidates[0],weights_only=False,map_location='cpu'),v));ref_labels.append('independent_FIT_distribution')
                if v=='count':
                    independent=sorted((root/'baselines').glob('*.json')) if a['config']['stage']=='outer' else []
                    if independent:
                        base=json.loads(independent[0].read_text())
                        for name,mm in base['models'].items():
                            if mm['fit_success']:
                                ff.append({(int(r['horizon']),int(r['packet'])):(r['logp'],1.) for r in mm['rows']});ref_labels.append(name)
                for h in (1,5,30,120):
                    common=set(fa)
                    for f in ff:common&=set(f)
                    keys=sorted(k for k in common if k[0]==h)
                    if not keys:continue
                    den=sum(fa[k][1] for k in keys);sa=-sum(fa[k][0] for k in keys)/den
                    ss=[-sum(f[k][0] for k in keys)/den for f in ff]
                    floor.append(dict(task=a['task'],view=v,horizon=h,state_nll=sa,reference_floor=min(ss),gain_over_reference_floor=min(ss)-sa,
                        reference_tasks=ref_labels,reference_scores=ss,units=den,scope='conservative benchmark, not OUTER-selected deployed model'))
    result=dict(status='SNAPSHOT',source_digest=plan['source_digest'],tasks=tasks,n_complete=sum(t['status']=='COMPLETE' for t in tasks),n_total=len(tasks),
                paired=pairs,reference_floors=floor,
                scientific_status='OPEN: optimization completion and nonzero gradients cannot establish IED physiology or clinical prediction')
    atomic_json(result,out_path);return result

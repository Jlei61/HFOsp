"""Resumable training of the repaired v0311 model, with physical-target sampling."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from . import data as D
from .prepare import Prepared
from .engine import build_model,apply_ablations,infer_asof,predict,score_predictions,wrong_time_control
from .objective import VIEWS,SECONDARY

ROOT='/data/hfosp_group_event_state_observable_state_validation_v0312'
PACKETS='/data/hfosp_group_event_state_rich_event_identification_v0311/packets'
HORIZONS=(1,5,30,120)


@dataclass
class RunConfig:
    subject:str='epilepsiae_1125'
    protocol:str='S-E'
    stage:str='inner0'
    inputs:str='P_marks'
    family:str='I-L-G1'
    arm:str='state'
    seed:int=20260906
    split_seed:int=20260906
    sampler_seed:int=20260916
    eval_seed:int=20260926
    history_hours:float|None=None
    grad_hours:float=2.
    batch_size:int=32
    microbatch:int=32
    train_paths:int=4
    eval_paths:int=64
    lr:float=1e-3
    dynamics_lr:float=3e-4
    max_updates:int=3200
    extended_updates:int=6400
    eval_every:int=50
    eval_stride:int=30
    eval_chunk:int=24
    checkpoint_every:int=25
    crossview:bool=False
    old_targets:bool=False
    activation_checkpoint:bool=True
    packets_root:str=PACKETS
    out_dir:str=ROOT+'/runs'
    recipe_path:str=''
    device:str='cuda:0'


def json_safe(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
    raise TypeError(type(v).__name__)


def atomic_json(data,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with open(tmp,'w') as f:
        json.dump(data,f,indent=2,default=json_safe);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_torch(data,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(data,tmp);os.replace(tmp,path)


def source_digest():
    root=Path(__file__).parent
    files={str(p.relative_to(root.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.glob('*.py'))}
    # Stable formulas still imported from older modules must be in the lineage too.
    for p in (root.parent/'v0311/data.py',root.parent/'v0311/packets.py',root.parent/'v039/human_data.py'):
        files[str(p.relative_to(root.parents[2]))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((root.parents[2]/'scripts').glob('*group_event_state_v0312*.py')):
        files[str(p.relative_to(root.parents[2]))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return D.digest(files),files


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def config_identity(cfg):
    c=asdict(cfg)
    for k in ('device','out_dir','microbatch','eval_chunk','checkpoint_every'):c.pop(k,None)
    return D.digest(c)


def tag(cfg):
    h='persistent' if cfg.history_hours is None else f'h{cfg.history_hours:g}'
    suffix='__nodelay' if cfg.crossview else '__oldtargets' if cfg.old_targets else ''
    return f'{cfg.subject}__{cfg.protocol}__{cfg.stage}__{cfg.inputs}__{cfg.family}__{cfg.arm}__{h}__seed{cfg.seed}{suffix}'


def load_run(cfg):
    payload=torch.load(Path(cfg.packets_root)/f'{cfg.subject}.pt',weights_only=False,map_location='cpu')
    split=D.build_split(payload,cfg.subject,cfg.split_seed,cfg.stage,cfg.protocol)
    x,t,_=D.packet_tables(payload,split);sc=D.fit_scaling(payload,split,x,t)
    prep=Prepared(payload,split,sc,torch.device(cfg.device));transform=apply_ablations(prep,cfg.crossview)
    prep.scaling=dict(prep.scaling,transform_id=D.digest((sc['transform_id'],transform)))
    model=build_model(prep,cfg.inputs,cfg.family,cfg.arm,cfg.seed)
    return model,prep


def target_units(prep,ids):
    """Data-only denominators, fixed before all microbatches of an update."""
    ids=np.asarray(ids,int);pk=prep.payload['packets'];out={k:0. for k in VIEWS+SECONDARY}
    out['count']=float(len(ids))
    out['spatial']=out['load']=float((prep.count[torch.as_tensor(ids,device=prep.device)]>0).sum())
    for i in ids:
        a,b=pk['event_lo'][i],pk['event_hi'][i]
        out['morphology']+=float(prep.band_ratio_valid[a:b].sum()+prep.xlag_valid[a:b].sum()+prep.iqr_valid[a:b].sum())
    return out


def main_views(cfg):return ('count','spatial') if cfg.old_targets else VIEWS


def training_table(prep):
    if not hasattr(prep,'_training_table'):
        prep._training_table=D.target_table(prep.payload,prep.split,'fit',stride=1)
    return prep._training_table


def training_ids(prep):
    # A short valid prediction must not be deleted because a 120-minute origin
    # would cross a clinical reset. Horizons have their own legal support.
    return np.unique(training_table(prep)[:,0])


def training_normalizers(prep,batch_size):
    table=training_table(prep);n=len(training_ids(prep));result={}
    for h in HORIZONS:
        units=target_units(prep,np.unique(table[table[:,1]==h,0]))
        result[h]={k:batch_size*v/max(n,1) for k,v in units.items()}
    return result


def loss_for_targets(model,prep,cfg,ids,normalizers,update,credit_starts=None):
    ids=np.asarray(ids,int)
    legal=training_table(prep);legal=legal[np.isin(legal[:,0],ids)]
    queries=np.unique(legal[:,2])
    state=infer_asof(model,prep,queries,'fit',cfg.history_hours,training=True,
                     grad_hours=cfg.grad_hours,activation_checkpoint=cfg.activation_checkpoint,
                     grad_start_by_episode=credit_starts)
    predictions=predict(model,prep,state,tuple(np.unique(legal[:,1])),paths=cfg.train_paths,seed=cfg.seed+1000003*update)
    rows=score_predictions(model,prep,state,predictions,'fit',ids)
    loss=prep.stats.new_zeros(())
    for row in rows:
        h=row['horizon'];weight=.5/len(HORIZONS)+(.5 if h==1 else 0.)
        for v in main_views(cfg)+SECONDARY:
            den=normalizers[h][v]
            if den<=0:continue
            vw=1/len(main_views(cfg)) if v in main_views(cfg) else .25
            loss-=weight*vw*(row['logp'][v]*(row['units'][v]>0)).sum()/den
    return loss


def optimizer_for(model,cfg):
    groups={}
    for name,p in model.named_parameters():
        slow=name.startswith('dynamics.') or 'eta_head' in name
        decay=p.ndim>=2 and not slow
        key=(slow,decay);groups.setdefault(key,[]).append(p)
    params=[dict(params=ps,lr=cfg.dynamics_lr if slow else cfg.lr,
                 initial_lr=cfg.dynamics_lr if slow else cfg.lr,
                 weight_decay=1e-4 if decay else 0.,name=f'{"dynamics_cov" if slow else "network"}_{"decay" if decay else "nodecay"}')
            for (slow,decay),ps in groups.items()]
    return torch.optim.AdamW(params,betas=(.9,.999),eps=1e-8)


def update(model,prep,cfg,optimizer,ids,normalizers,step,microbatch):
    optimizer.zero_grad(set_to_none=True);total=0.
    legal=training_table(prep);q=np.unique(legal[np.isin(legal[:,0],ids),2]);starts=prep.split['episode_start'][q]
    credit={int(s):int(q[starts==s].min()-round(cfg.grad_hours*60)) for s in np.unique(starts)}
    for a in range(0,len(ids),microbatch):
        loss=loss_for_targets(model,prep,cfg,ids[a:a+microbatch],normalizers,step,credit)
        if not torch.isfinite(loss):raise FloatingPointError('non-finite loss; no update permitted')
        loss.backward();total+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('non-finite gradient; no update permitted')
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('non-finite parameter after update')
    return total,float(norm)


def aggregate_rows(rows,cfg):
    sums={h:{v:[0.,0.] for v in VIEWS+SECONDARY} for h in HORIZONS}
    for row in rows:
        for v in VIEWS+SECONDARY:
            u=row['units'][v];lp=row['logp'][v]
            sums[row['horizon']][v][0]+=float((lp*(u>0)).sum())
            sums[row['horizon']][v][1]+=float(u.sum())
    scores={h:{v:(-x[0]/x[1] if x[1] else None) for v,x in vs.items()} for h,vs in sums.items()}
    def total(vs):
        v=[vs[k] for k in main_views(cfg) if vs[k] is not None]
        return (sum(v)/len(v)+(.25*vs['load'] if vs['load'] is not None else 0.)) if v else None
    per=[total(scores[h]) for h in HORIZONS];have=[x for x in per if x is not None]
    selection=None if not have else .5*(per[0] if per[0] is not None else sum(have)/len(have))+.5*sum(have)/len(have)
    return dict(scores=scores,units={h:{v:x[1] for v,x in vs.items()} for h,vs in sums.items()},selection=selection)


@torch.no_grad()
def evaluate(model,prep,cfg,role='inner',collect=False,rule='EVOLVE',reference=None,paths=None,seed=None):
    table=D.target_table(prep.payload,prep.split,role,cfg.eval_stride)
    if not len(table):return dict(status='NOT_ESTIMABLE',selection=None,scores={},units={},rows=[])
    queries=np.unique(table[:,2]);rows=[];meta=[]
    for a in range(0,len(queries),cfg.eval_chunk):
        qq=queries[a:a+cfg.eval_chunk]
        state=infer_asof(model,prep,qq,role,cfg.history_hours,producer_hash=tensor_hash(model.state_dict()))
        if rule=='WRONGTIME':
            state=wrong_time_control(state,reference)
            if state is None:continue
        pred=predict(model,prep,state,HORIZONS,paths=paths or cfg.eval_paths,
                     seed=cfg.eval_seed if seed is None else seed,rule='EVOLVE' if rule=='WRONGTIME' else rule,reference=reference)
        rr=score_predictions(model,prep,state,pred,role,np.unique(table[:,0]))
        # A target/horizon pair must have its own registered origin, exactly once.
        rows.extend(rr);meta.append(state.metadata())
    result=aggregate_rows(rows,cfg)
    result.update(status='COMPLETE' if rows else 'NOT_ESTIMABLE',n_targets=len(np.unique(table[:,0])),n_queries=len(queries),
                  target_table_digest=D.digest(table),rows=rows if collect else [],query_metadata=meta if collect else [])
    return result


class Plateau:
    def __init__(self):self.best=math.inf;self.bad=0;self.drops=0
    def observe(self,score):
        if score<self.best-1e-3:self.best=score;self.bad=0;return 'improve'
        self.bad+=1
        if self.drops<2 and self.bad>=6:self.bad=0;self.drops+=1;return 'drop'
        if self.drops>=2 and self.bad>=8:return 'stop'
        return 'wait'


def interpret_training(stage,selected_updates,stop_reason,plateau):
    """Bound what a stopping record says about optimization and learning."""
    if int(selected_updates)==0:
        return 'origin checkpoint selected; this fit supplies no learned component'
    if stage=='outer' and stop_reason=='fixed_inner_recipe':
        return 'OUTER followed a frozen INNER update count; nonzero refit steps are not convergence evidence'
    drops=int((plateau or {}).get('drops',0))
    if stop_reason=='plateau' and drops>=2:
        return 'two-stage learning-rate patience plateau; local optimization evidence, not global convergence proof'
    if stop_reason=='budget':
        return 'budget edge reached; convergence remains unresolved'
    if stop_reason in ('paused','resource_pressure'):
        return f'{stop_reason}; exact continuation is required before optimization is interpreted'
    return f'{stop_reason}; optimization adequacy remains unresolved'


def run_cell(cfg,*,stop_after=None,deadline=None,progress=None):
    """Resume exact optimizer/sampler/scheduler state; outer runs require an INNER recipe."""
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(cfg.device);torch.cuda.reset_peak_memory_stats(cfg.device)
    begin=time.time();out=Path(cfg.out_dir)/tag(cfg);out.mkdir(parents=True,exist_ok=True)
    source,files=source_digest();identity=config_identity(cfg)
    packets_hash=file_hash(Path(cfg.packets_root)/f'{cfg.subject}.pt')
    model,prep=load_run(cfg);opt=optimizer_for(model,cfg);plateau=Plateau()
    ids=training_ids(prep)
    if len(ids)<cfg.batch_size:raise ValueError(f'only {len(ids)} training targets, batch={cfg.batch_size}')
    # Fixed unit expectations make accumulated gradients independent of microbatch boundaries.
    normalizers=training_normalizers(prep,cfg.batch_size)
    rng=np.random.default_rng(cfg.sampler_seed);curve=[];updates=0;micro=cfg.microbatch
    budget=cfg.max_updates;paused=False;incidents=[]
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    if cfg.stage=='outer':
        if not cfg.recipe_path:raise ValueError('outer refit requires a frozen nested-INNER recipe')
        recipe=json.loads(Path(cfg.recipe_path).read_text())
        if recipe['source_digest']!=source:raise ValueError('recipe/source mismatch')
        for k in ('subject','inputs','family','arm','history_hours','crossview','old_targets'):
            if recipe['config'][k]!=getattr(cfg,k):raise ValueError(f'recipe mismatch: {k}')
        budget=int(recipe['updates'])
    else:recipe=None
    best=dict(score=math.inf,updates=0,state=None)
    last=out/'last.pt';stop_flag=[False]
    old_handlers={}
    for sig in (signal.SIGTERM,signal.SIGINT):
        old_handlers[sig]=signal.signal(sig,lambda *args:stop_flag.__setitem__(0,True))
    def save():
        atomic_torch(dict(config=asdict(cfg),identity=identity,source_digest=source,source_files=files,packets_sha256=packets_hash,
            model=model.state_dict(),optimizer=opt.state_dict(),plateau=vars(plateau),sampler=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if cfg.device.startswith('cuda') else [],
            updates=updates,best=best,curve=curve,budget=budget,microbatch=micro,incidents=incidents,
            initial=initial,split=prep.split,scaling=prep.scaling),last)
        atomic_json(dict(status='RUNNING',updates=updates,budget=budget,source_digest=source,
                         pid=os.getpid(),heartbeat=time.time(),microbatch=micro),out/'progress.json')
    try:
        if last.exists():
            old=torch.load(last,weights_only=False,map_location=cfg.device)
            if old.get('packets_sha256')!=packets_hash:raise ValueError('resume measurement packets changed')
            if old['identity']!=identity or old['source_digest']!=source:raise ValueError('resume config/source changed')
            if old['split']['split_id']!=prep.split['split_id'] or old['scaling']['transform_id']!=prep.scaling['transform_id']:
                raise ValueError('resume data contract changed')
            model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);plateau.__dict__.update(old['plateau'])
            rng.bit_generator.state=old['sampler'];torch.set_rng_state(old['torch_rng'].cpu())
            if old['cuda_rng']:torch.cuda.set_rng_state_all([r.cpu() for r in old['cuda_rng']])
            updates=old['updates'];best=old['best'];curve=old['curve'];budget=old['budget'];micro=old['microbatch'];incidents=old['incidents'];initial=old['initial']
            del old
        else:
            baseline=evaluate(model,prep,cfg,'inner') if cfg.stage!='outer' else None
            if baseline and baseline['selection'] is None:raise ValueError('no INNER score; cannot select checkpoint')
            best=dict(score=baseline['selection'] if baseline else None,updates=0,
                      state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            curve.append(dict(update=0,inner=None if baseline is None else baseline['selection'],lrs=[g['lr'] for g in opt.param_groups]))
            if baseline:plateau.best=baseline['selection']
            save()
        stop_reason='budget'
        while updates<budget:
            if stop_flag[0] or (stop_after is not None and updates>=stop_after) or (deadline is not None and time.time()>=deadline):
                paused=True;stop_reason='paused';break
            if cfg.device.startswith('cuda') and torch.cuda.mem_get_info(cfg.device)[0]<6*2**30:
                paused=True;stop_reason='resource_pressure';break
            anchor=int(rng.integers(len(ids)));batch=ids[(anchor+np.arange(cfg.batch_size))%len(ids)]
            backup={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            opt_backup=copy.deepcopy(opt.state_dict())
            t=time.time()
            while True:
                try:
                    loss,norm=update(model,prep,cfg,opt,batch,normalizers,updates,micro);break
                except torch.OutOfMemoryError:
                    model.load_state_dict(backup);opt.load_state_dict(opt_backup);opt.zero_grad(set_to_none=True)
                    import gc;gc.collect();torch.cuda.empty_cache()
                    if micro<=1:raise
                    micro=max(1,micro//2);incidents.append(dict(update=updates,event='OOM_RETRY_SAME_BATCH',microbatch=micro))
            updates+=1
            if recipe and updates in recipe.get('lr_drop_updates',[]):
                for g in opt.param_groups:g['lr']*=.3
            row=dict(update=updates,train_loss=loss,gradient_norm=norm,step_seconds=time.time()-t,
                     lrs=[g['lr'] for g in opt.param_groups],microbatch=micro)
            action='train'
            if cfg.stage!='outer' and updates%cfg.eval_every==0:
                ev=evaluate(model,prep,cfg,'inner');score=ev['selection'];row.update(inner=score,scores=ev['scores'])
                if score is None:raise ValueError('INNER denominator disappeared')
                if score<best['score']:best=dict(score=score,updates=updates,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                action=plateau.observe(score)
                if action=='drop':
                    for g in opt.param_groups:g['lr']*=.3
                if action=='stop':stop_reason='plateau'
                if updates>=budget and action=='improve' and cfg.extended_updates>budget:
                    budget=cfg.extended_updates
            curve.append(row)
            if updates%cfg.checkpoint_every==0 or action in ('drop','stop'):save()
            if progress:progress(row,action)
            if action=='stop':break
        save()
        if paused:
            atomic_json(dict(status='PAUSED',updates=updates,budget=budget,source_digest=source),out/'progress.json')
            return dict(status='PAUSED',updates=updates,directory=str(out))
        last_hash=tensor_hash(model.state_dict())
        if cfg.stage=='outer':
            stop_reason='fixed_inner_recipe'
            best=dict(score=None,updates=updates,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        model.load_state_dict(best['state'])
        role='outer' if cfg.stage in ('outer','sid') else 'inner'
        final=evaluate(model,prep,cfg,role,collect=True)
        atomic_torch(dict(config=asdict(cfg),state_dict=model.state_dict(),split=prep.split,scaling=prep.scaling,
                          source_digest=source,packets_sha256=packets_hash,selected_updates=best['updates']),out/'selected.pt')
        atomic_torch(final,out/'predictions.pt')
        inventory=[]
        for n,p in model.named_parameters():
            p0=initial[n].to(p);delta=float((p-p0).norm());den=float(p0.norm())
            inventory.append(dict(name=n,shape=list(p.shape),n_parameters=p.numel(),initial_norm=den,
                                  final_norm=float(p.norm()),update_norm=delta,relative_update=delta/den if den>0 else None))
        card=dict(status='COMPLETE',scope='development',config=asdict(cfg),source_digest=source,source_files=files,packets_sha256=packets_hash,
            split_id=prep.split['split_id'],transform_id=prep.scaling['transform_id'],updates=updates,
            selected_updates=best['updates'],stop_reason=stop_reason,plateau=vars(plateau),
            inner_selection=best['score'],score_role=role,scores=final['scores'],units=final['units'],
            target_table_digest=final.get('target_table_digest'),selected_parameter_hash=tensor_hash(model.state_dict()),
            last_parameter_hash=last_hash,parameter_inventory=inventory,curve=curve,incidents=incidents,
            seconds=time.time()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cfg.device.startswith('cuda') else 0.,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cfg.device.startswith('cuda') else 0.,
            batch_size=cfg.batch_size,actual_microbatch=micro,recipe=recipe,
            training_interpretation=interpret_training(cfg.stage,best['updates'],stop_reason,vars(plateau)),
            credit_semantics='persistent batches share a credit boundary two hours before their earliest query; later queries can have a longer credit span; strict H differentiates the entire H',
            stopping_precision='fixed 1e-3 common-noise plateau tolerance; post-freeze MC audit is required and plateau is not proof of convergence')
        atomic_json(card,out/'card.json');atomic_json(dict(status='COMPLETE',updates=updates),out/'progress.json')
        return card
    except Exception as exc:
        atomic_json(dict(status='FAILED',updates=updates,error=f'{type(exc).__name__}: {exc}',source_digest=source),out/'progress.json')
        raise
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)

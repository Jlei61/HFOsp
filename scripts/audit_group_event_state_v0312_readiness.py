#!/usr/bin/env python3
"""Actual CUDA profiling and destructive-to-own-child crash/resume audit only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NVIDIA_TF32_OVERRIDE','0')
import argparse,copy,json,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.topic5_group_event_state.v0312.train import *
from src.topic5_group_event_state.v0312.engine import infer_asof,predict,score_predictions
from src.topic5_group_event_state.v0312.frozen import run_bundle


def profile(out,device,subject='epilepsiae_1125',seconds=1800):
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    cfg=RunConfig(subject=subject,device=device,batch_size=32,microbatch=32,eval_paths=64)
    loaded_source=source_digest();m,p=load_run(cfg);opt=optimizer_for(m,cfg);ids=training_ids(p)
    normalizers=training_normalizers(p,cfg.batch_size)
    anchor=int(np.argmax(ids-p.split['episode_start'][ids]));batch=ids[np.minimum(anchor+np.arange(32),len(ids)-1)]
    # Choose distinct physical targets near the longest legal prefix.
    anchor=min(max(0,anchor-16),len(ids)-32);batch=ids[anchor:anchor+32]
    rows=[];begin=time.time();step=0;phase=0
    for history in (None,.5,2.,8.):
        cfg.history_hours=history
        t=time.time();loss,norm=update(m,p,cfg,opt,batch,normalizers,step,cfg.microbatch);step+=1
        rows.append(dict(phase='history_stress',history_hours=history,seconds=time.time()-t,loss=loss,gradient_norm=norm,
                         peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))
        print(json.dumps(rows[-1]),flush=True)
    cfg.history_hours=None;rng=np.random.default_rng(727)
    steady_begin=time.time()
    while time.time()-steady_begin<seconds:
        anchor=int(rng.integers(len(ids)-32));batch=ids[anchor:anchor+32]
        t=time.time();loss,norm=update(m,p,cfg,opt,batch,normalizers,step,cfg.microbatch);step+=1
        rows.append(dict(phase='steady',seconds=time.time()-t,loss=loss,gradient_norm=norm))
        if step%20==0:print(json.dumps(dict(step=step,elapsed=time.time()-begin)),flush=True)
    t=time.time();ev=evaluate(m,p,cfg,'inner');eval_seconds=time.time()-t
    # Long credit uses real events and a held-ahead FIT target, never an OUTER tuning loss.
    cfg.history_hours=8.;q=int(ids[np.argmax(ids-p.split['episode_start'][ids])]-1)
    p.tokens.requires_grad_(True)
    st=infer_asof(m,p,[q],'fit',8.,training=True,activation_checkpoint=False)
    pred=predict(m,p,st,(1,),paths=4,seed=991);rr=score_predictions(m,p,st,pred,'fit',[q+1])[0]
    loss=sum(-rr['logp'][k].sum()/rr['units'][k].sum().clamp(min=1) for k in ('count','spatial','morphology'))
    loss.backward();grad=p.tokens.grad;et=p.payload['event_time'];age=(p.packet_end[q]-et)/3600
    credit=[]
    for lo,hi in ((0,.5),(.5,2),(2,6),(6,8)):
        ii=np.flatnonzero((age>=lo)&(age<hi));val=float(grad[ii].norm()) if len(ii) else 0.
        credit.append(dict(age_hours=[lo,hi],n_events=len(ii),gradient_norm=val))
    mask=(age>=6)&(age<8);ii=np.flatnonzero(mask)
    fd=dict(status='NOT_ESTIMABLE',reason='no 6-8h events with nonzero derivative')
    if len(ii) and float(grad[ii].norm())>0:
        gi=grad[ii];direction=gi/gi.abs().max();g=float((gi*direction).sum());original=p.tokens[ii].detach().clone();eps=.02
        def score():
            state=infer_asof(m,p,[q],'fit',8.);pp=predict(m,p,state,(1,),paths=4,seed=991);r=score_predictions(m,p,state,pp,'fit',[q+1])[0]
            return sum(-r['logp'][k].sum()/r['units'][k].sum().clamp(min=1) for k in ('count','spatial','morphology'))
        with torch.no_grad():
            p.tokens[ii]=original+eps*direction;plus=float(score());p.tokens[ii]=original-eps*direction;minus=float(score());p.tokens[ii]=original
        observed=(plus-minus)/(2*eps);fd=dict(status='COMPLETE',direction='gradient-aligned perturbation of readable 6-8h events',autograd=g,finite_difference=observed,absolute_error=abs(g-observed),epsilon=eps)
    p.tokens.requires_grad_(False)
    result=dict(status='COMPLETE',kind='engineering_profile_not_scientific_fit',subject=subject,device=device,updates=step,seconds=time.time()-begin,
        source_digest=loaded_source[0],source_files=loaded_source[1],source_unchanged=loaded_source[0]==source_digest()[0],rows=rows,
        median_update_seconds=float(np.median([r['seconds'] for r in rows if r['phase']=='steady'])),eval_seconds=eval_seconds,eval_units=ev['units'],
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
        credit=credit,finite_difference=fd,scope='throughput, finite training and FIT gradient reach; no validation-based scientific selection')
    atomic_json(result,out);return result


def recovery(root,device):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=RunConfig(device=device,batch_size=4,microbatch=4,train_paths=2,eval_paths=4,max_updates=4,extended_updates=4,
        eval_every=2,eval_stride=180,eval_chunk=8,checkpoint_every=1)
    paths=[]
    for name in ('uninterrupted','crashed'):
        cfg=copy.copy(base);cfg.out_dir=str(root/name);path=root/(name+'.json');atomic_json(asdict(cfg),path);paths.append((cfg,path))
    script=Path(__file__).with_name('run_group_event_state_v0312_cell.py');logs=[]
    for cfg,path in paths:
        log=open(root/(Path(cfg.out_dir).name+'.log'),'w');cmd=[sys.executable,str(script),'--config',str(path)]
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(script.parents[1]));killed=False
        if Path(cfg.out_dir).name=='crashed':
            progress=Path(cfg.out_dir)/tag(cfg)/'progress.json';until=time.time()+180
            while child.poll() is None and time.time()<until:
                if progress.exists():
                    try:r=json.loads(progress.read_text())
                    except json.JSONDecodeError:r={}
                    if r.get('updates',0)>=2:
                        child.kill();child.wait();killed=True;break
                time.sleep(.01)
            if not killed:raise RuntimeError('failed to kill own child after a valid checkpoint')
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(script.parents[1]))
        if child.wait()!=0:raise RuntimeError(f'recovery subprocess failed: {path}')
        log.close()
    a,b=[torch.load(Path(c.out_dir)/tag(c)/'last.pt',weights_only=False,map_location='cpu') for c,_ in paths]
    def compare(x,y):
        if isinstance(x,torch.Tensor):return torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(compare(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return len(x)==len(y) and all(compare(i,j) for i,j in zip(x,y))
        if isinstance(x,np.ndarray):return np.array_equal(x,y)
        return x==y
    result=dict(status='COMPLETE',model_identical=compare(a['model'],b['model']),optimizer_identical=compare(a['optimizer'],b['optimizer']),
        sampler_identical=compare(a['sampler'],b['sampler']),scheduler_identical=compare(a['plateau'],b['plateau']),
        weight_max_abs=max(float((a['model'][k].double()-b['model'][k].double()).abs().max()) for k in a['model']),source_digest=source_digest()[0])
    if not all(result[k] for k in ('model_identical','optimizer_identical','sampler_identical','scheduler_identical')):raise AssertionError(result)
    atomic_json(result,root/'result.json');return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['profile','recovery','bundle']);ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--subject',default='epilepsiae_1125');ap.add_argument('--seconds',type=float,default=1800);ap.add_argument('--selected');a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    if a.mode=='profile':r=profile(a.out,a.device,a.subject,a.seconds)
    elif a.mode=='recovery':r=recovery(a.out,a.device)
    else:r=run_bundle(a.selected,a.out,a.device,quick=True)
    print(json.dumps({k:r.get(k) for k in ('status','seconds','updates','model_identical','producer_unchanged')}),flush=True)

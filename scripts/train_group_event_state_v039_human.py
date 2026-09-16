#!/usr/bin/env python3
"""Finite FIT/INNER training with a frozen background foundation and event residual."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition, FutureReadout, endpoint_loss
from src.topic5_group_event_state.v035.contracts import atomic_json


def run(args):
    start=time.time();torch.set_num_threads(1);torch.manual_seed(args.seed)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data=torch.load(args.data,map_location='cpu',weights_only=False);samples=data['samples'];device=torch.device(args.device)
    key=str(float(args.history_hours));n=len(samples);steps=max(len(s['histories'][key][0]) for s in samples)
    x=np.zeros((n,steps,data['input_dim']),np.float32);dt=np.zeros((n,steps),np.float32)
    for i,s in enumerate(samples):
        values,elapsed=s['histories'][key];x[i,:len(values)]=values;dt[i,:len(elapsed)]=elapsed
    x=torch.from_numpy(x).to(device);dt=torch.from_numpy(dt).to(device)
    context=torch.tensor(np.array([s['context'] for s in samples]),dtype=torch.float32,device=device)
    if args.event_only:context=context[:,-5:]
    counts=torch.tensor(np.array([[t[0] for t in s['targets']] for s in samples]),dtype=torch.float32,device=device)
    recruitment=torch.tensor(np.array([[t[1] for t in s['targets']] for s in samples]),dtype=torch.float32,device=device)
    valid=torch.tensor(np.array([[t[2] for t in s['targets']] for s in samples]),dtype=torch.bool,device=device)
    spatial_valid=torch.tensor(np.array([[t[3] for t in s['targets']] for s in samples]),dtype=torch.bool,device=device)
    phases=np.array([s['phase'] for s in samples]);fit=torch.tensor(np.flatnonzero(phases=='FIT'),device=device)
    inner=torch.tensor(np.flatnonzero((phases=='INNER')&valid[:,1].cpu().numpy()),device=device)
    if args.view=='recruitment':inner=inner[spatial_valid[inner,1]]
    selection=torch.tensor(np.flatnonzero(phases=='SELECTION'),device=device)
    if len(fit)<16 or len(inner)<4 or not len(selection) or not spatial_valid[fit].any():
        atomic_json(args.output,dict(status='NOT_ESTIMABLE',subject=data['subject'],reason='insufficient paired FIT/INNER/SELECTION anchors',
                    development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False));return
    baseline=FutureReadout(0,data['n_recruitment'],context.shape[1]).to(device)
    with torch.no_grad():
        output=baseline.layers[-1];output.weight.zero_()
        y=counts[fit][valid[fit]];mean=y.mean();variance=y.var(unbiased=False)
        output.bias[0]=mean.clamp_min(1e-3).log()
        p=recruitment[fit][spatial_valid[fit]].mean(0).clamp(.01,.99)
        output.bias[1:]=torch.logit(p)
        r=(mean.square()/(variance-mean).clamp_min(.1)).clamp(.05,100)
        baseline.log_dispersion.copy_(torch.log(torch.expm1(r.clamp_max(30))))
    rng=torch.Generator(device=device).manual_seed(args.seed+13)
    leads=(0.,2.,6.)
    def base_predictions(rows,lead):return baseline(context.new_empty((len(rows),0)),context[rows],lead)
    def vector_loss(mu,logits,rows,hi,dispersion):
        _,count,spatial=endpoint_loss(mu,logits,counts[rows,hi],recruitment[rows,hi],dispersion,'joint')
        losses=count if args.view=='count' else spatial if args.view=='recruitment' else count+spatial*spatial_valid[rows,hi]
        use=spatial_valid[rows,hi] if args.view=='recruitment' else valid[rows,hi]
        return losses,use,count,spatial
    def train_stage(parameters,predict,dispersion,max_steps,patience,lr,name,snapshot,restore):
        optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=1e-4)
        def inner_score():
            # Two-hour lead is the registered model-selection objective.
            with torch.no_grad():
                values=[]
                for rows in inner.split(256):
                    mu,logits=predict(rows,2.,False);loss,use,_,_=vector_loss(mu,logits,rows,1,dispersion())
                    values.append(loss[use])
                joined=torch.cat(values)
                return float(joined.mean()) if joined.numel() else math.inf
        best=inner_score();initial=best;best_state=snapshot();selected=0;stale=0;curve=[];peak_grad=0.;peak_module_grad={};updates=0
        batch_hash=hashlib.sha256();min_batch=n;max_batch=0;min_row=n;max_row=-1
        for step in range(1,max_steps+1):
            rows=fit[torch.randint(len(fit),(min(args.batch_size,len(fit)),),generator=rng,device=device)]
            hi=int((step-1)%3);lead=leads[hi];use=valid[rows,hi] if args.view!='recruitment' else spatial_valid[rows,hi]
            if not use.any():continue
            rows=rows[use]
            # Preserve the registered CUDA RNG stream, but validate and bind the
            # actual sampled indices before they enter the observer.
            checked_rows=rows.detach().cpu().numpy()
            if checked_rows.min()<0 or checked_rows.max()>=n or not np.all(phases[checked_rows]=='FIT'):
                raise IndexError(f'{name} invalid FIT batch at iteration {step}')
            batch_hash.update(checked_rows.astype('<i8',copy=False).tobytes())
            min_batch=min(min_batch,len(checked_rows));max_batch=max(max_batch,len(checked_rows))
            min_row=min(min_row,int(checked_rows.min()));max_row=max(max_row,int(checked_rows.max()))
            optimizer.zero_grad(set_to_none=True)
            mu,logits=predict(rows,lead,True);loss,use,_,_=vector_loss(mu,logits,rows,hi,dispersion());loss=loss[use].mean()
            if not torch.isfinite(loss):raise FloatingPointError(f'{name} nonfinite objective')
            loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(parameters,2.,error_if_nonfinite=True));peak_grad=max(peak_grad,gradient)
            if name=='event':
                for key,p in observer.named_parameters():
                    if p.grad is not None:peak_module_grad[key]=max(peak_module_grad.get(key,0.),float(p.grad.norm()))
            optimizer.step()
            updates+=1
            if step%25==0 or step==max_steps:
                score=inner_score();curve.append(dict(step=step,fit_batch=float(loss.detach()),inner=score,gradient_norm=gradient))
                if score<best-1e-5:best=score;selected=step;stale=0;best_state=snapshot()
                else:stale+=1
                atomic_json(args.output.with_name('progress.json'),dict(status='RUNNING',stage=name,step=step,selected_step=selected,inner=score,elapsed_seconds=time.time()-start))
                if stale>=patience:break
        restore(best_state)
        with torch.no_grad():
            fit_scores=[]
            for rows in fit.split(256):
                mu,logits=predict(rows,2.,False);loss,use,_,_=vector_loss(mu,logits,rows,1,dispersion())
                fit_scores.append(loss[use])
            joined=torch.cat(fit_scores)
            fit_score=float(joined.mean()) if joined.numel() else None
        return dict(initial_inner=initial,selected_inner=best,selected_step=selected,steps_run=step,
                    optimizer_updates=updates,selected_fit_loss_at_2h=fit_score,
                    actual_batch_size_min=min_batch if updates else 0,actual_batch_size_max=max_batch,
                    sampled_fit_row_min=min_row if updates else None,sampled_fit_row_max=max_row if updates else None,
                    sampled_fit_indices_sha256=batch_hash.hexdigest(),sampled_indices_checked=True,
                    stop_reason='INNER_PATIENCE' if stale>=patience else 'BUDGET_LIMIT',optimization_limited=stale<patience,
                    peak_gradient_norm=peak_grad,peak_transition_gradient_norms=peak_module_grad,curve=curve)
    base_stage=train_stage(list(baseline.parameters()),lambda rows,lead,training:base_predictions(rows,lead),lambda:baseline.log_dispersion,
        args.foundation_steps,8,.003,'background',lambda:copy.deepcopy(baseline.state_dict()),baseline.load_state_dict)
    baseline.requires_grad_(False)
    # Same seed and same event-write initialisation for L and N.
    torch.manual_seed(args.seed+100)
    observer=EventTransition(data['input_dim'],args.family,width=args.width,seed=args.seed).to(device)
    residual=FutureReadout(observer.width,data['n_recruitment'],0).to(device)
    with torch.no_grad():
        residual.layers[-1].weight.zero_();residual.layers[-1].bias.zero_()
        residual.log_dispersion.copy_(baseline.log_dispersion)
    original=copy.deepcopy(observer.state_dict())
    def state_predictions(rows,lead,training):
        state=observer.scan(x[rows],dt[rows],checkpoint_chunk=32 if training else 0)
        if args.forecast=='rollout':state=observer.advance(state,lead)
        dm,dl=residual(state,state.new_empty((len(rows),0)),lead)
        with torch.no_grad():bm,bl=base_predictions(rows,lead)
        return bm+dm,bl+dl
    def snapshot():return (copy.deepcopy(observer.state_dict()),copy.deepcopy(residual.state_dict()))
    def restore(states):observer.load_state_dict(states[0]);residual.load_state_dict(states[1])
    event_stage=train_stage(list(observer.parameters())+list(residual.parameters()),state_predictions,lambda:residual.log_dispersion,
        args.max_steps,12,args.lr,'event',snapshot,restore)
    with torch.no_grad():
        fit_mean=sum(observer.scan(x[rows],dt[rows],checkpoint_chunk=0).sum(0) for rows in fit.split(256))/len(fit)
        selection_states=torch.cat([observer.scan(x[rows],dt[rows],checkpoint_chunk=0) for rows in selection.split(256)])
    torch.manual_seed(args.seed+200)
    constant=FutureReadout(observer.width,data['n_recruitment'],0).to(device)
    with torch.no_grad():
        constant.layers[-1].weight.zero_();constant.layers[-1].bias.zero_();constant.log_dispersion.copy_(baseline.log_dispersion)
    def constant_predictions(rows,lead,training):
        with torch.no_grad():
            state=fit_mean[None].expand(len(rows),-1)
            if args.forecast=='rollout':state=observer.advance(state,lead)
            bm,bl=base_predictions(rows,lead)
        dm,dl=constant(state,state.new_empty((len(rows),0)),lead)
        return bm+dm,bl+dl
    constant_stage=train_stage(list(constant.parameters()),constant_predictions,lambda:constant.log_dispersion,
        args.foundation_steps,8,.003,'refitted_constant',lambda:copy.deepcopy(constant.state_dict()),constant.load_state_dict)
    selection_times=np.array([samples[int(i)]['anchor'] for i in selection])
    donor=np.full(len(selection),-1,int)
    breaks=np.r_[0,np.flatnonzero(np.diff(selection_times)>300.0001)+1,len(selection)]
    for left,right in zip(breaks[:-1],breaks[1:]):
        if right-left>=4:donor[left:right]=np.roll(np.arange(left,right),(right-left)//2)
    donor_tensor=torch.tensor(np.maximum(donor,0),device=device)
    selection_position=torch.full((n,),-1,dtype=torch.long,device=device);selection_position[selection]=torch.arange(len(selection),device=device)
    predictions={};metrics={}
    with torch.no_grad():
        for hi,lead in enumerate(leads):
            collected={k:[] for k in ('state_loss','baseline_loss','state_count','baseline_count','state_recruitment','baseline_recruitment','valid','spatial_valid','refitted_constant_loss','shifted_loss')}
            for rows in selection.split(256):
                mu,logits=state_predictions(rows,lead,False);loss,use,count,spatial=vector_loss(mu,logits,rows,hi,residual.log_dispersion)
                bm,bl=base_predictions(rows,lead);b_loss,_,b_count,b_spatial=vector_loss(bm,bl,rows,hi,baseline.log_dispersion)
                cm,cl=constant_predictions(rows,lead,False);c_loss,_,_,_=vector_loss(cm,cl,rows,hi,constant.log_dispersion)
                position=selection_position[rows];wrong=selection_states[donor_tensor[position]]
                if args.forecast=='rollout':wrong=observer.advance(wrong,lead)
                dm,dl=residual(wrong,wrong.new_empty((len(rows),0)),lead)
                wrong_loss,_,_,_=vector_loss(bm+dm,bl+dl,rows,hi,residual.log_dispersion)
                wrong_loss[torch.tensor(donor[position.cpu().numpy()]<0,device=device)]=float('nan')
                for key,value in zip(collected,(loss,b_loss,count,b_count,spatial,b_spatial,use,spatial_valid[rows,hi],c_loss,wrong_loss)):collected[key].append(value.cpu().numpy())
            values={k:np.concatenate(v) for k,v in collected.items()};mask=values['valid'];smask=values['spatial_valid']
            metrics[str(int(lead))]=dict(n_selection_anchors=int(mask.sum()),n_spatial_anchors=int(smask.sum()),
                gain_over_frozen_background=float((values['baseline_loss']-values['state_loss'])[mask].mean()) if mask.any() else None,
                count_gain=float((values['baseline_count']-values['state_count'])[mask].mean()) if mask.any() else None,
                recruitment_gain=float((values['baseline_recruitment']-values['state_recruitment'])[smask].mean()) if smask.any() else None,
                gain_over_refitted_constant_floored=float(min((values['refitted_constant_loss']-values['state_loss'])[mask].mean(),(values['baseline_loss']-values['state_loss'])[mask].mean())) if mask.any() else None,
                correct_over_shifted=float((values['shifted_loss']-values['state_loss'])[mask&np.isfinite(values['shifted_loss'])].mean()) if (mask&np.isfinite(values['shifted_loss'])).any() else None)
            predictions.update({f'{int(lead)}h_{k}':v for k,v in values.items()})
    predictions['anchor_time']=np.array([samples[int(i)]['anchor'] for i in selection])
    scores=args.output.with_suffix('.npz');np.savez_compressed(scores,**predictions)
    weights=args.output.with_suffix('.pt');torch.save(dict(observer=observer.state_dict(),residual=residual.state_dict(),baseline=baseline.state_dict(),constant=constant.state_dict(),fit_state_mean=fit_mean,
        config=vars(args),data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),input_dim=data['input_dim'],context_dim=context.shape[1],n_recruitment=data['n_recruitment']),weights)
    card=dict(status='COMPLETE',subject=data['subject'],family=args.family,seed=args.seed,config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        stages=dict(background=base_stage,event=event_stage,refitted_constant=constant_stage),metrics=metrics,selected_transition_deltas={k:float((v-original[k]).abs().max()) for k,v in observer.state_dict().items()},
        wrong_time_control=dict(eligible_anchors=int((donor>=0).sum()),median_offset_hours=float(np.median(np.abs(selection_times[donor[donor>=0]]-selection_times[donor>=0]))/3600) if (donor>=0).any() else None,
                               rule='half circular shift within each contiguous scored 5-minute anchor run; background/clock remain at correct time'),
        model_inventory={name:{k:dict(shape=list(v.shape),parameters=v.numel(),requires_grad=v.requires_grad) for k,v in module.named_parameters()} for name,module in [('observer',observer),('residual',residual),('background',baseline),('constant',constant)]},
        observer_buffers={k:dict(shape=list(v.shape),elements=v.numel()) for k,v in observer.named_buffers()},
        state_width=observer.width,input_dim=data['input_dim'],context_dim=context.shape[1],
        training_contract=dict(optimizer='AdamW',weight_decay=1e-4,clip_global_norm=2.,batch_size=args.batch_size,
            max_effective_batch_size=min(args.batch_size,len(fit)),n_fit_anchors=len(fit),n_inner_anchors=len(inner),
            event_lr=args.lr,foundation_lr=.003,normalization='frozen FIT median/MAD and count-per-hour scale from data bundle; no BatchNorm',
            integration='RK4 <= 5 minutes; full history BPTT with checkpoint chunks of 32, no detach',
            initialization='A negative diagonal; event B normal(0,0.1/sqrt(input_dim)); N U/V normal(0,0.05), bias 0; event residual output zero',
            selection='own trained view at 2-hour lead on INNER, including initial checkpoint; SELECTION never chooses weights'),
        data_path=str(args.data),data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),checkpoint=str(weights),checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        scores=str(scores),scores_sha256=hashlib.sha256(scores.read_bytes()).hexdigest(),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/topic5_group_event_state/v039/transition.py']},
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,elapsed_seconds=time.time()-start)
    atomic_json(args.output,card);print(json.dumps({k:card[k] for k in ('status','subject','family','metrics','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=('F','L','N'),required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=20260905)
    p.add_argument('--history-hours',type=float,choices=(.5,8.),default=8.);p.add_argument('--forecast',choices=('direct','rollout'),default='direct')
    p.add_argument('--view',choices=('count','recruitment','joint'),default='joint');p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--width',type=int,default=16);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--max-steps',type=int,default=800)
    p.add_argument('--foundation-steps',type=int,default=800);p.add_argument('--event-only',action='store_true')
    run(p.parse_args())

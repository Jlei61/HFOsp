#!/usr/bin/env python3
"""Replay held-out loss from individual real event tokens and audit long credit."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition,FutureReadout,endpoint_loss
from src.topic5_group_event_state.v035.contracts import atomic_json


def replay_events(data,sample,hours):
    """Independent reconstruction from event-level cumsums, before aggregation."""
    anchor=sample['anchor'];left=anchor-hours*3600;last=left;dt=[];tokens=[];times=[];rows=[]
    for block in data['event_replay_blocks']:
        if not left<block['release']<=anchor:continue
        begin=np.searchsorted(block['times'],left);values=np.diff(block['cumsum'],axis=0)[begin:]
        delta=(block['release']-last)/3600;pieces=max(1,int(np.ceil(delta/(1/12))))
        dt.extend([delta/pieces]*pieces);rows.extend([len(dt)-1]*len(values));tokens.extend(values);times.extend(block['times'][begin:]);last=block['release']
    delta=(anchor-last)/3600;pieces=max(1,int(np.ceil(delta/(1/12))));dt.extend([delta/pieces]*pieces)
    return np.asarray(tokens,float).reshape(-1,data['input_dim']),np.asarray(times,float),np.asarray(rows,int),np.asarray(dt,float)


def audit(source,output):
    torch.set_num_threads(1);card=json.loads(source.read_text());cfg=card['config']
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(card['checkpoint'])!=card['checkpoint_sha256'] or sha(card['data_path'])!=card['data_sha256']:raise ValueError('Frozen upstream changed')
    checkpoint=torch.load(card['checkpoint'],map_location='cpu',weights_only=False);data=torch.load(card['data_path'],map_location='cpu',weights_only=False)
    observer=EventTransition(data['input_dim'],cfg['family'],width=cfg['width'],seed=cfg['seed']).double();observer.load_state_dict(checkpoint['observer'])
    residual=FutureReadout(observer.width,data['n_recruitment'],0).double();residual.load_state_dict(checkpoint['residual'])
    baseline=FutureReadout(0,data['n_recruitment'],checkpoint['context_dim']).double();baseline.load_state_dict(checkpoint['baseline'])
    for model in (observer,residual,baseline):model.requires_grad_(False)
    eligible=[s for s in data['samples'] if s['phase']=='SELECTION' and s['targets'][1][2]
              and (cfg['view']!='recruitment' or s['targets'][1][3])]
    picked=[]
    for sample in eligible:
        if not picked or sample['anchor']>=picked[-1]['anchor']+1800:picked.append(sample)
        if len(picked)==3:break
    if not picked:raise ValueError('No held-out 2-hour target for gradient audit')
    records=[]
    for sample in picked:
        tokens,times,rows,elapsed=replay_events(data,sample,cfg['history_hours'])
        token=torch.tensor(tokens,dtype=torch.float64,requires_grad=True);index=torch.tensor(rows);dt=torch.tensor(elapsed)[None]
        context=torch.tensor(np.asarray(sample['context']),dtype=torch.float64)[None]
        if cfg['event_only']:context=context[:,-5:]
        count,recruit,_,spatial=sample['targets'][1];count=torch.tensor([count],dtype=torch.float64);recruit=torch.tensor(np.asarray(recruit),dtype=torch.float64)[None]
        def pool(values):return values.new_zeros((len(elapsed),data['input_dim'])).index_add(0,index,values)[None]
        def score(x,duration):
            state=observer.scan(x,duration,checkpoint_chunk=32)
            if cfg['forecast']=='rollout':state=observer.advance(state,2.)
            dm,dl=residual(state,state.new_empty((1,0)),2.);bm,bl=baseline(state.new_empty((1,0)),context,2.)
            _,nb,bce=endpoint_loss(bm+dm,bl+dl,count,recruit,residual.log_dispersion)
            loss=nb if cfg['view']=='count' else bce if cfg['view']=='recruitment' else nb+bce*float(spatial)
            return loss.sum(),state
        x=pool(token);loss,state=score(x,dt)
        gradient=torch.autograd.grad(loss,token)[0].detach().numpy()
        cached,cached_dt=sample['histories'][str(float(cfg['history_hours']))]
        input_delta=float(np.max(np.abs(x.detach().numpy()[0]-cached)))
        with torch.no_grad():
            cached_loss,cached_state=score(torch.tensor(cached,dtype=torch.float64)[None],torch.tensor(cached_dt,dtype=torch.float64)[None])
            half=torch.zeros((1,2*len(elapsed),data['input_dim']),dtype=torch.float64);half[:,1::2]=x.detach()
            half_dt=dt.repeat_interleave(2,dim=1)/2;half_loss,half_state=score(half,half_dt)
        old=(sample['anchor']-times>=6*3600)&(sample['anchor']-times<8*3600)
        direction=tokens*old[:,None];derivative=float((gradient*direction).sum());eps=.001
        with torch.no_grad():
            plus,_=score(pool(token.detach()+eps*torch.tensor(direction)),dt);minus,_=score(pool(token.detach()-eps*torch.tensor(direction)),dt)
        finite_difference=float((plus-minus)/(2*eps));gap=abs(finite_difference-derivative)
        age=(sample['anchor']-times)/3600;bands={}
        for lo,hi,label in ((0,2,'0_to_2h'),(2,6,'2_to_6h'),(6,8,'6_to_8h')):
            keep=(age>=lo)&(age<hi);g=gradient[keep];effect=(gradient*tokens)[keep]
            bands[label]=dict(n_real_events=int(keep.sum()),gradient_l1=float(np.abs(g).sum()),event_scale_sensitivity_l1=float(np.abs(effect.sum(-1)).sum()),
                feature_gradient_l1={group:float(np.abs(g[:,[name.startswith(group) for name in data['feature_names']]]).sum()) for group in ('count','participation','delay','first_group','band_','cross_band','detector_','bipolar_','shaft_car_','coupled','available_')})
        records.append(dict(anchor=sample['anchor'],n_real_events=len(times),bands=bands,replayed_loss=float(loss.detach()),cached_loss=float(cached_loss),
            input_max_abs_difference=input_delta,loss_abs_difference=abs(float(loss.detach()-cached_loss)),
            replay_pass=bool(input_delta<1e-4 and abs(float(loss.detach()-cached_loss))<1e-4),
            half_step_state_max_abs_difference=float((state.detach()-half_state).abs().max()),half_step_loss_difference=abs(float(loss.detach()-half_loss)),
            long_history_directional_autograd=derivative,long_history_directional_finite_difference=finite_difference,
            finite_difference_pass=bool(gap<=1e-7+1e-4*max(abs(derivative),abs(finite_difference)))))
    result=dict(status='COMPLETE',subject=card['subject'],family=card['family'],seed=card['seed'],history_hours=cfg['history_hours'],
        selected_step=card['stages']['event']['selected_step'],records=records,replay_pass=all(r['replay_pass'] for r in records),
        finite_difference_pass=all(r['finite_difference_pass'] for r in records),
        source_card=str(source),source_card_sha256=sha(source),checkpoint_sha256=card['checkpoint_sha256'],data_sha256=card['data_sha256'],source_sha256=sha(__file__),
        interpretation='Actual held-out NB/recruitment loss gradients to individually reconstructed real event tokens. Gradients describe the delayed observer, not physiological event feedback; zero gradient at a selected background fallback is allowed.',
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False)
    atomic_json(output,result);print(json.dumps({k:result[k] for k in ['status','subject','family','replay_pass','finite_difference_pass']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.source,a.output)

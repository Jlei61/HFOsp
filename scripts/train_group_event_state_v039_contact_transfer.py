#!/usr/bin/env python3
"""Frozen-state contact transfer, with exact prefix/size-conditioned scoring."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from src.topic5_group_event_state.v039.frozen_transfer import (
    exact_subset_nll,first_unknown_group,branching_strata,LogitAdapter,standardise_anchor_features,equal_anchor_mean)
from src.topic5_group_event_state.v035.contracts import atomic_json


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    start=time.time();torch.set_num_threads(1);device=torch.device(args.device);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    fm=json.loads(args.features.with_suffix('.json').read_text());prefix=args.root/'frozen_prefix'/f"{fm['subject']}.npz"
    pm=json.loads(prefix.with_suffix('.json').read_text());target=Path(pm['transfer_data_path'])
    tm=json.loads(target.with_suffix('.json').read_text())
    if fm['human_data_sha256']!=tm['human_data_sha256']:raise ValueError('Expression and observer data versions differ')
    upstream=Path(fm['source_card']);upstream_card=json.loads(upstream.read_text())
    if sha(upstream)!=fm['source_card_sha256'] or upstream_card['checkpoint_sha256']!=fm['checkpoint_sha256']:
        raise ValueError('Frozen export does not match its upstream card')
    if sha(args.features)!=fm['export_sha256'] or sha(prefix)!=pm['data_sha256'] or sha(target)!=pm['transfer_data_sha256']:raise ValueError('Frozen transfer inputs changed')
    with np.load(args.features) as z:features={k:z[k].copy() for k in z.files}
    with np.load(target) as z:ranks=z['ranks'];phase=z['phase'];anchor=z['anchor_position'];times=z['event_time'];block=z['block'];anchor_time=z['anchor_time']
    with np.load(prefix) as z:
        if not np.array_equal(z['event_time'],times) or not np.array_equal(z['anchor_position'],anchor):raise ValueError('Prefix cache event alignment failed')
        raw=np.c_[z['logits'],z['stops']]
    if not np.array_equal(features['anchor_time'][anchor],anchor_time):raise ValueError('Frozen state anchor alignment failed')
    y,available,stop,keys,_=first_unknown_group(ranks);branch,branch_keys,_=branching_strata(ranks,phase)
    fit=np.flatnonzero(phase=='FIT');inner=np.flatnonzero(phase=='INNER');selection=np.flatnonzero(phase=='SELECTION')
    if min(len(fit),len(inner),len(selection))<5:
        atomic_json(args.output,dict(status='NOT_ESTIMABLE',subject=fm['subject'],reason='insufficient expression partitions',development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False));return
    fit_anchors=np.unique(anchor[fit]);c=y.shape[-1];raw=torch.tensor(raw,dtype=torch.float32,device=device)
    y=torch.tensor(y,device=device);available=torch.tensor(available,device=device);stop=torch.tensor(stop,device=device)
    groups={int(a):np.flatnonzero((anchor==a)&(phase=='FIT')) for a in fit_anchors}
    inverse_inner=torch.tensor(np.unique(anchor[inner],return_inverse=True)[1],device=device)
    inner=torch.tensor(inner,device=device)
    def loss_vectors(logits,rows):
        subset=exact_subset_nll(logits[:,:c],y[rows],available[rows]);stop_loss=F.binary_cross_entropy_with_logits(logits[:,c],stop[rows].float(),reduction='none')
        return subset+stop_loss,subset,stop_loss
    def mean_inner(predict):
        with torch.no_grad():
            values=[]
            for rows in inner.split(512):values.append(loss_vectors(predict(rows),rows)[0])
            loss=torch.cat(values);count=torch.bincount(inverse_inner);sums=loss.new_zeros(len(count)).index_add(0,inverse_inner,loss)
            return float((sums/count).mean())
    stages={};normalizers={};weights={};predictions={}
    def train(name,parameters,predict,snapshot,restore,budget,lr,patience):
        optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=1e-4);rng=np.random.default_rng(args.seed+701)
        best=mean_inner(predict);initial=best;saved=snapshot();selected=0;stale=0;curve=[];peak=0.
        for step in range(1,budget+1):
            picked=rng.choice(fit_anchors,size=args.batch_size,replace=True)
            rows=torch.tensor([rng.choice(groups[int(a)]) for a in picked],device=device)
            optimizer.zero_grad(set_to_none=True);loss=loss_vectors(predict(rows),rows)[0].mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite frozen contact probe loss')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(parameters,2.,error_if_nonfinite=True));peak=max(peak,norm);optimizer.step()
            if step%25==0 or step==budget:
                value=mean_inner(predict);curve.append(dict(step=step,fit_batch=float(loss.detach()),inner=value,gradient_norm=norm))
                if value<best-1e-5:best=value;selected=step;stale=0;saved=snapshot()
                else:stale+=1
                atomic_json(args.output.with_name('progress.json'),dict(status='RUNNING',stage=name,step=step,selected_step=selected))
                if stale>=patience:break
        restore(saved)
        stages[name]=dict(initial_inner=initial,selected_inner=best,selected_step=selected,optimizer_updates=step,
            optimization_limited=stale<patience,stop_reason='INNER_PATIENCE' if stale>=patience else 'BUDGET_LIMIT',curve=curve,peak_gradient_norm=peak)
    bias=torch.nn.Parameter(torch.zeros(c+1,device=device))
    train('prefix_recalibration',[bias],lambda rows:raw[rows]+bias,lambda:bias.detach().clone(),lambda value:bias.data.copy_(value),args.foundation_steps,.003,10)
    bias.requires_grad_(False)
    background,bnorm=standardise_anchor_features(features['background'],fit_anchors);normalizers['background']=bnorm
    background=torch.tensor(background,device=device);event_anchor=torch.tensor(anchor,device=device)
    torch.manual_seed(args.seed+301);bg=LogitAdapter(background.shape[1],c).to(device)
    train('background',list(bg.parameters()),lambda rows:raw[rows]+bias+bg(background[event_anchor[rows]]),
        lambda:copy.deepcopy(bg.state_dict()),bg.load_state_dict,args.foundation_steps,.003,10)
    bg.requires_grad_(False)
    with torch.no_grad():parent=raw+bias+bg(background[event_anchor])
    predictions['frozen_prefix']=raw.detach().cpu().numpy();predictions['background']=parent.cpu().numpy()
    weights['prefix_bias']=bias.detach().cpu();weights['background']=bg.state_dict();models={};tensors={}
    feature_arms={k:features[k] for k in ('state','initialized','fixed_history','functional')}
    feature_arms['constant']=np.zeros_like(features['state'])
    for name,values in feature_arms.items():
        equal=next((other for other,previous in feature_arms.items() if other in models and np.array_equal(values,previous)),None)
        if equal:
            models[name]=models[equal];tensors[name]=tensors[equal];predictions[name]=predictions[equal].copy();stages[name]=dict(reused_identical_features=equal,**stages[equal]);normalizers[name]=normalizers[equal];weights[name]=weights[equal];continue
        values,norm=standardise_anchor_features(values,fit_anchors);normalizers[name]=norm;tensors[name]=torch.tensor(values,device=device)
        torch.manual_seed(args.seed+401);model=LogitAdapter(values.shape[1],c).to(device);models[name]=model
        train(name,list(model.parameters()),lambda rows:parent[rows]+model(tensors[name][event_anchor[rows]]),
            lambda:copy.deepcopy(model.state_dict()),model.load_state_dict,args.max_steps,.001,12)
        model.requires_grad_(False);weights[name]=model.state_dict()
        with torch.no_grad():predictions[name]=(parent+model(tensors[name][event_anchor])).cpu().numpy()
    # Deterministic same-prefix-and-size donors, with >=2h offset. A seizure
    # between donor and recipient disqualifies the pair; background stays put.
    index=json.loads((Path('/data/hfosp_group_event_state_v0_1/dataset')/fm['subject']/'index.json').read_text())
    onsets=np.array([s['onset_epoch'] for s in index['seizures']]);donor=np.full(len(times),-1,int)
    for key in np.unique(keys[selection]):
        members=selection[keys[selection]==key]
        for i in members:
            others=members[np.abs(times[members]-times[i])>=7200]
            if len(others):
                others=others[np.array([not np.any((onsets>min(times[i],times[j]))&(onsets<=max(times[i],times[j]))) for j in others])]
            if len(others):donor[i]=others[np.argmax(np.abs(times[others]-times[i]))]
    shifted=predictions['state'].copy();paired=np.flatnonzero(donor>=0)
    if len(paired):
        with torch.no_grad():shifted[paired]=(parent[paired]+models['state'](tensors['state'][event_anchor[torch.tensor(donor[paired],device=device)]])).cpu().numpy()
    predictions['shifted_state']=shifted
    scores={};metrics={}
    for name,values in predictions.items():
        with torch.no_grad():total,subset,stop_loss=loss_vectors(torch.tensor(values,device=device),torch.arange(len(times),device=device))
        scores[name+'_total']=total.cpu().numpy();scores[name+'_subset']=subset.cpu().numpy();scores[name+'_stop']=stop_loss.cpu().numpy()
        metrics[name]={}
        for endpoint,mask in [('exact_next_subset',(phase=='SELECTION')&branch),('next_subset_all',(phase=='SELECTION')&~stop.cpu().numpy()),('stop',(phase=='SELECTION'))]:
            keyscore=name+('_stop' if endpoint=='stop' else '_subset');v=scores[keyscore]
            if name=='shifted_state':mask=mask&(donor>=0)
            per_anchor=equal_anchor_mean(v[mask],anchor[mask]) if mask.any() else np.empty(0)
            metrics[name][endpoint]=dict(loss=float(per_anchor.mean()) if len(per_anchor) else None,n_events=int(mask.sum()),n_anchors=len(per_anchor),n_raw_blocks=len(np.unique(block[mask])))
    arrays=dict(event_time=times,anchor_time=anchor_time,phase=phase,block=block,branch_mask=branch,same_prefix_size_key=keys,shift_donor=donor,**scores)
    args.output.parent.mkdir(parents=True,exist_ok=True);score_path=args.output.with_suffix('.npz');np.savez_compressed(score_path,**arrays)
    checkpoint=args.output.with_suffix('.pt');torch.save(dict(weights=weights,normalizers=normalizers,config=vars(args)),checkpoint)
    result=dict(status='COMPLETE',subject=fm['subject'],family=fm['family'],view=fm['view'],history_hours=fm['history_hours'],seed=args.seed,
        stages=stages,metrics=metrics,n_fit_branching_strata=len(branch_keys),branching_rule='first two exact contact groups and observed third-group size; >=5 FIT events and >=2 distinct FIT next sets; held-out diversity cannot qualify a stratum',
        frozen_feature_card=str(args.features.with_suffix('.json')),frozen_feature_card_sha256=sha(args.features.with_suffix('.json')),prefix_card=str(prefix.with_suffix('.json')),prefix_card_sha256=sha(prefix.with_suffix('.json')),
        upstream_card=str(upstream),upstream_card_sha256=sha(upstream),upstream_checkpoint_sha256=fm['checkpoint_sha256'],
        scores=str(score_path),scores_sha256=sha(score_path),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
        training_contract=dict(optimizer='AdamW',background_lr=.003,adapter_lr=.001,weight_decay=1e-4,clip=2.,batch_size=args.batch_size,adapter_hidden_width=8,
            sampling='anchors uniform, one event drawn per anchor; each event appears at one anchor',upstream_and_decoder_frozen=True,
            objective='exact third-group subset NLL conditional on size + STOP BCE after two observed groups',primary_endpoint='same-prefix-and-size branching subset NLL'),
        model_inventory={name:{key:dict(shape=list(p.shape),parameters=p.numel()) for key,p in model.named_parameters()} for name,model in models.items()},
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/topic5_group_event_state/v039/frozen_transfer.py']},
        config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,seizure_annotations_used='donor pairing excludes intervening seizures only',elapsed_seconds=time.time()-start)
    atomic_json(args.output,result);print(json.dumps(dict(status='COMPLETE',subject=fm['subject'],family=fm['family'],primary_n=metrics['state']['exact_next_subset']['n_events'],elapsed_seconds=time.time()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260905);p.add_argument('--device',default='cuda:0');p.add_argument('--max-steps',type=int,default=600);p.add_argument('--foundation-steps',type=int,default=400);p.add_argument('--batch-size',type=int,default=128)
    run(p.parse_args())

#!/usr/bin/env python3
"""Frozen transfer to untrained expression families and single-view cross tasks."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.ridge_transfer import fit_candidates,select_probe,fit_supported_response_columns
from src.topic5_group_event_state.v039.frozen_transfer import equal_anchor_mean
from src.topic5_group_event_state.v035.contracts import atomic_json


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    start=time.time();torch.set_num_threads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    fm=json.loads(args.features.with_suffix('.json').read_text());subject=fm['subject'];source=Path(fm['source_card'])
    if sha(source)!=fm['source_card_sha256'] or sha(args.features)!=fm['export_sha256']:raise ValueError('Changed frozen state export')
    upstream=json.loads(source.read_text());target=args.root/'transfer_data'/f'{subject}.npz';prefix=args.root/'frozen_prefix'/f'{subject}.npz'
    tm=json.loads(target.with_suffix('.json').read_text());pm=json.loads(prefix.with_suffix('.json').read_text())
    if fm['human_data_sha256']!=tm['human_data_sha256'] or sha(target)!=tm['data_sha256'] or sha(prefix)!=pm['data_sha256']:raise ValueError('Transfer data version mismatch')
    with np.load(args.features) as z:features={k:z[k].copy() for k in z.files}
    with np.load(target) as z:
        event_time=z['event_time'];positions=z['anchor_position'];phase=z['phase'];anchor_time=z['anchor_time'];block=z['block']
        targets={k.removeprefix('target_'):z[k].copy() for k in z.files if k.startswith('target_')}
    with np.load(prefix) as z:
        if not np.array_equal(z['event_time'],event_time):raise ValueError('Frozen prefix event order changed')
        prefix_hidden=z['hidden'].copy()
    if not np.array_equal(features['anchor_time'][positions],anchor_time):raise ValueError('State/expression time mismatch')
    base=np.c_[prefix_hidden,features['background'][positions]]
    states={k:features[k][positions] for k in ('state','initialized','fixed_history','functional')}
    arrays=dict(event_time=event_time,anchor_time=anchor_time,phase=phase,block=block);weights={};results={}
    def one(name,y,baseline,states,phases,anchors,prefix_key):
        if y.ndim==1:y=y[:,None]
        original_components=y.shape[1]
        component_mask=fit_supported_response_columns(y,phases,anchors)
        if not component_mask.any():
            results[name]=dict(status='NOT_ESTIMABLE',reason='no response component measured at three distinct FIT anchors',n_input_components=original_components);return
        y=y[:,component_mask]
        valid=np.isfinite(y).all(-1)&np.isfinite(baseline).all(-1);rows=np.flatnonzero(valid)
        x=baseline[rows];yy=np.asarray(y[rows],float);ph=phases[rows];aa=anchors[rows]
        fit=np.flatnonzero(ph=='FIT');inner=np.flatnonzero(ph=='INNER');selection=np.flatnonzero(ph=='SELECTION')
        if min(len(fit),len(inner),len(selection))<3 or len(np.unique(aa[fit]))<3:
            results[name]=dict(status='NOT_ESTIMABLE',reason='insufficient complete FIT/INNER/SELECTION response vectors');return
        # Equal-anchor target scale. No held-out response sets a normalizer.
        mu=equal_anchor_mean(yy[fit],aa[fit]).mean(0)
        variance=equal_anchor_mean((yy[fit]-mu)**2,aa[fit]).mean(0);scale=np.sqrt(variance);scale=np.where(scale>1e-6,scale,1.)
        z=(yy-mu)/scale
        candidates,mean=fit_candidates(x,z,fit,aa)
        candidates.append(dict(alpha=0.,prediction=np.broadcast_to(mean,z.shape).copy(),constant_intercept=mean))
        basefit=select_probe(candidates,z,inner,aa,allow_zero=False);parent=basefit['prediction']
        predictions={'prefix_background':parent};fitted={'prefix_background':basefit['model']}
        # Refit a constant residual under exactly the same FIT weighting.
        intercept=equal_anchor_mean((z-parent)[fit],aa[fit]).mean(0)
        const=select_probe([dict(alpha=0.,prediction=np.broadcast_to(intercept,z.shape).copy(),constant_intercept=intercept)],z,inner,aa,parent)
        predictions['refitted_constant']=const['prediction'];fitted['refitted_constant']=const['model']
        for arm,values in states.items():
            same=next((other for other,previous in states.items() if other in predictions and np.array_equal(values,previous)),None)
            if same:
                predictions[arm]=predictions[same];fitted[arm]={**fitted[same],'reused_identical_features':same};continue
            candidates,_=fit_candidates(values[rows],z-parent,fit,aa,pool_fit_by_anchor=True)
            selected=select_probe(candidates,z,inner,aa,parent)
            predictions[arm]=selected['prediction'];fitted[arm]=selected['model']
        cell={};error={arm:((prediction-z)**2).mean(-1) for arm,prediction in predictions.items()}
        for arm,err in error.items():
            per_anchor=equal_anchor_mean(err[selection],aa[selection]);bg=equal_anchor_mean(error['prefix_background'][selection],aa[selection])
            constant=equal_anchor_mean(error['refitted_constant'][selection],aa[selection])
            cell[arm]=dict(selected_alpha=fitted[arm].get('alpha'),selected_parent_fallback=fitted[arm].get('zero_residual',False),
                held_out_normalized_mse=float(per_anchor.mean()),gain_over_parent=float((bg-per_anchor).mean()),
                gain_over_refitted_constant_floored=float(min((bg-per_anchor).mean(),(constant-per_anchor).mean())))
            full=np.full(len(y),np.nan);full[rows]=err;arrays[prefix_key+'__'+name+'__'+arm]=full
        weights[name]=dict(target_center=mu,target_scale=scale,response_component_indices=np.flatnonzero(component_mask),models={arm:{k:v for k,v in model.items() if k!='prediction'} for arm,model in fitted.items()})
        results[name]=dict(status='COMPLETE',n_components=y.shape[1],arms=cell,
            n_input_components=original_components,response_component_indices=np.flatnonzero(component_mask).tolist(),
            response_component_selection='at least three measured FIT anchors; unsupported/missing bands cannot discard supported components',
            support={p:dict(events_or_anchors=int((ph==p).sum()),anchors=len(np.unique(aa[ph==p]))) for p in ['FIT','INNER','SELECTION']})
        atomic_json(args.output.with_name('progress.json'),dict(status='RUNNING',completed_endpoint=name,elapsed_seconds=time.time()-start))
    for name,y in targets.items():one(name,y,base,states,phase,positions,'event')
    if fm['view'] in ('count','recruitment'):
        # This response did not train a recruitment-only observer. Every
        # eligible anchor is included, including zero-event future windows.
        data=torch.load(upstream['data_path'],map_location='cpu',weights_only=False)
        if fm['view']=='recruitment':
            y=np.array([np.log1p(s['targets'][1][0]) if s['targets'][1][2] else np.nan for s in data['samples']])[:,None]
            name='cross_future_count_log1p_2h'
        else:
            y=np.array([s['targets'][1][1] if s['targets'][1][3] else np.full(data['n_recruitment'],np.nan) for s in data['samples']])
            name='cross_future_recruitment_vector_2h'
        aa=np.arange(len(y));ph=np.array([s['phase'] for s in data['samples']]);states={k:features[k] for k in ('state','initialized','fixed_history','functional')}
        one(name,y,features['background'],states,ph,aa,'anchor')
        arrays['cross_anchor_time']=features['anchor_time'];arrays['cross_anchor_phase']=ph
    args.output.parent.mkdir(parents=True,exist_ok=True);score=args.output.with_suffix('.npz');np.savez_compressed(score,**arrays)
    checkpoint=args.output.with_suffix('.pt');torch.save(weights,checkpoint)
    result=dict(status='COMPLETE',subject=subject,family=fm['family'],history_hours=fm['history_hours'],view=fm['view'],seed=fm['seed'],results=results,
        upstream_card=str(source),upstream_card_sha256=sha(source),upstream_checkpoint_sha256=fm['checkpoint_sha256'],
        frozen_feature_card=str(args.features.with_suffix('.json')),frozen_feature_card_sha256=sha(args.features.with_suffix('.json')),prefix_card=str(prefix.with_suffix('.json')),prefix_card_sha256=sha(prefix.with_suffix('.json')),
        training_contract='FIT weighted standardisation, unpenalised intercept, closed-form ridge 0.01/0.1/1/10 selected on INNER. Freeze prefix/background prediction before residual state/history/initial/functional probes; exact zero residual remains selectable.',
        target_scope='Delay and coupling on mapped decoder contacts; bands and waveform statistics on the full new FIT sensor universe. All are untrained future expression targets of the upstream state.',
        interpretation='Raw latent transfer alone can retain independent causes. Functional trained-view readout and initialized/history controls must be interpreted together; overlapping anchors are not independent samples.',
        scores=str(score),scores_sha256=sha(score),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/topic5_group_event_state/v039/ridge_transfer.py',ROOT/'src/topic5_group_event_state/v039/frozen_transfer.py']},
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,elapsed_seconds=time.time()-start)
    atomic_json(args.output,result);print(json.dumps(dict(status='COMPLETE',subject=subject,family=fm['family'],endpoints=len(results),elapsed_seconds=time.time()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

#!/usr/bin/env python3
"""Read-only CPU audit of v0311 artifacts; writes a separate review directory.

No training, source repair, access to sealed data, or changes to old results.
The fitted NB comparator is a diagnostic on already-open development data.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import gammaln

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311.prepare import Prepared
from src.topic5_group_event_state.v0311.train import RunConfig, query_indices, build_run, filter_episodes
from src.topic5_group_event_state.v0311 import frozen as FZ
from src.topic5_group_event_state.v0311.numerics import propagate_moments

R = Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
O = Path('/data/hfosp_group_event_state_v0311_independent_review')


def nb_nll(par, y, exposure_hours):
    mu = np.exp(par[0]) * exposure_hours
    size = np.exp(par[1])
    return -(gammaln(y+size)-gammaln(size)-gammaln(y+1)
             + size*(np.log(size)-np.log(size+mu))
             + y*(np.log(np.maximum(mu,1e-100))-np.log(size+mu)))


def safe(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)


def main():
    torch.set_num_threads(1)
    O.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'existing development artifacts only; CPU diagnostics, no producer refit',
              'patients': {}, 'inventory': {}, 'evidence_hashes': {}}
    for folder in ('runs','runs_rescored','runs_extended','runs_extended_rescored',
                   'runs_history','runs_reference_variants','runs_reference_variants_rescored',
                   'runs_replication','runs_replication_rescored'):
        files = list((R/folder).rglob('*.card.json'))
        result['inventory'][folder] = {'cards': len(files)}
    all_cards = []
    # Rescored successors supersede their source cards, without changing histories.
    for folder in ('runs_rescored','runs_extended_rescored','runs_reference_variants_rescored',
                   'runs_replication_rescored','runs_history/h2_rescored','runs_history/h8_rescored'):
        all_cards += [(p, json.loads(p.read_text())) for p in (R/folder).rglob('*.card.json')]
    for subject in ('epilepsiae_1125','epilepsiae_1096','epilepsiae_253'):
        payload = torch.load(R/'packets'/f'{subject}.pt', weights_only=False, map_location='cpu')
        pk = payload['packets']
        seeds = (20260906,20260907,20260908)
        splits = [D.build_split(payload,subject,s) for s in seeds]
        split = splits[0]
        px,pt,_ = D.packet_tables(payload,split)
        sc = D.fit_scaling(payload,split,px,pt)
        prep = Prepared(payload,split,sc,torch.device('cpu'))
        pairs=[]
        for a,b in ((0,1),(0,2),(1,2)):
            aa,bb=map(set,(splits[a]['inner_starts'],splits[b]['inner_starts']))
            pairs.append({'seeds':[seeds[a],seeds[b]],'n_inner':[len(aa),len(bb)],
                          'intersection':len(aa&bb),'jaccard':len(aa&bb)/max(1,len(aa|bb)),
                          'outer_identical':np.array_equal(splits[a]['forward_starts'],splits[b]['forward_starts'])})
        sid = D.build_split_id(payload,subject,seeds[0])
        fit = (pk['end']<=sid['fit_end']) & pt['valid']
        withheld = fit & ~sid['train_packet']
        # Change only hidden S-ID packet values in a copy of the scaler input.
        altered=px.copy();altered[withheld,0]+=100.
        sid_sc=D.fit_scaling(payload,sid,px,pt)
        sid_alt=D.fit_scaling(payload,sid,altered,pt)
        usable=(pk['end']<=split['se_cutoff']) & pt['valid'] & ~prep.seizure_masked_np & ~prep.inner_target_held
        # Fixed literal rate plus dispersion, with exposure offset, FIT labels only.
        y=pt['count'][usable];e=pt['exposure'][usable]/3600.
        init=np.array([np.log(y.sum()/e.sum()),0.])
        fits=[minimize(lambda p:nb_nll(p,y,e).mean(),init+np.array([0.,d]),method='L-BFGS-B',
                       bounds=[(-20,20),(-12,15)]) for d in (-3.,0.,3.)]
        best=min(fits,key=lambda f:f.fun)
        outer={}
        for h in (1,5,30,120):
            details=[];card=None
            for path,c in all_cards:
                cfg=c['config']
                if (cfg['subject']==subject and cfg['split']=='S-E' and cfg['seed']==seeds[0]
                    and cfg['arm']=='intercept' and 'runs_rescored' in str(path)):
                    detail=np.load(path.with_name(path.name.replace('.card.json','.outer_detail.npz')))
                    for key in detail.files:
                        if key.startswith(f'{h}_') and key.endswith('_packet'):
                            prefix=key[:-len('packet')]
                            ids=detail[key];units=detail[prefix+'units_count']
                            details.extend(ids[units>0].tolist())
                    card=c;break
            if details:
                ids=np.asarray(details,int)
                outer[h]={'n':len(ids),'diagnostic_nb_nll':float(nb_nll(best.x,pt['count'][ids],pt['exposure'][ids]/3600.).mean()),
                          'delivered_intercept_nll':card['outer'][str(h)]['count']}
        # Existing query times, actual availability, strict 30-minute occurrence history.
        qidx=np.searchsorted(pk['end'],split['forward_starts']-1e-6)
        legal=[];ages=[];badrel=0;wronglast=0;recent=[];unpublished=[]
        for i in qidx:
            if i<179 or i>=len(pk['end']) or not pt['valid'][i] or prep.seizure_masked_np[i]:continue
            t=pk['end'][i]
            mask=(pk['start']>=t-1800.-1e-6)&(pk['end']<=t+1e-6)&(pk['release']<=t+1e-6)&pt['valid']&~prep.seizure_masked_np
            legal.append({'packet':int(i),'query_time':float(t),'readable_30min_packets':int(mask.sum()),
                          'readable_30min_events':int(pt['count'][mask].sum())})
            start=i-179
            k,lag,ok=query_indices(prep,[start],np.array([179]))
            if ok[0,0]:ages.append(int(lag[0,0]))
            rr=pk['release'][start:i+1]
            badrel+=int(np.any(rr[1:]<rr[:-1]))
            last=np.flatnonzero(rr<=t)
            wronglast+=int(bool(len(last)) and int(k[0,0])!=int(last[-1]))
            rrids=np.arange(max(0,i-30),i)
            counts=pt['count'][rrids]
            recent.append(float(counts.sum()))
            unpublished.append(float(counts[pk['release'][rrids]>t].sum()))
        res={'seed_changes_inner':pairs,
             'inner_hidden_as_training_input':int((prep.inner_target_held&split['train_packet']&pt['valid']&~prep.seizure_masked_np).sum()),
             'sid_scaler_hidden_packets':int(withheld.sum()),'sid_scaler_fit_packets':int(fit.sum()),
             'sid_scaler_perturbation':{'only_hidden_packet_count_input_changed':True,
               'center_before':float(sid_sc['packet_center'][0]),'center_after':float(sid_alt['packet_center'][0]),
               'scale_before':float(sid_sc['packet_scale'][0]),'scale_after':float(sid_alt['packet_scale'][0])},
             'fit_nb':{'success':bool(best.success),'n_fit':len(y),'mean_fit_count':float(y.mean()),
                       'rate_per_hour':float(np.exp(best.x[0])),'size':float(np.exp(best.x[1])),
                       'fit_nll':float(best.fun),'outer_diagnostic':outer},
             'half_hour_history':{'n_queries':len(legal),'queries_with_readable_packets':sum(x['readable_30min_packets']>0 for x in legal),
                                  'queries_with_readable_events':sum(x['readable_30min_events']>0 for x in legal),
                                  'examples':next(([x] for x in legal if x['readable_30min_events']>0),[])},
             'actual_source_age_minutes':{'n':len(ages),'min':min(ages) if ages else None,'max':max(ages) if ages else None},
             'nonmonotonic_release_prefixes':badrel,
             'last_released_lookup_mismatches':wronglast,
             'recent_30m_count_unpublished_fraction':float(sum(unpublished)/max(1,sum(recent)))}
        # The actual shuffle applies across the entire payload, including outer.
        donor=torch.randperm(len(payload['event_time']),generator=torch.Generator().manual_seed(seeds[0]+991)).numpy()
        in_fit=payload['event_time']<=split['se_cutoff']
        res['shuffle_fit_recipients_with_outer_donors']=int((in_fit & ~in_fit[donor]).sum())
        res['shuffle_fit_recipients']=int(in_fit.sum())
        # Exact reproduction of the control exclusion's timestamp-set construction.
        on=sorted(float(s['onset_epoch']) for s in split['seizures'])
        clusters=[]
        for t in on:
            if clusters and t-clusters[-1][-1]<7200:clusters[-1].append(t)
            else:clusters.append([t])
        pre_set=set()
        for c in clusters:
            for lo,hi in ((-7200.,-1800.),(-1800.,-300.)):
                for t in np.arange(c[0]+lo-1800,c[0]+hi+1800,1800):
                    pre_set.add(int(np.searchsorted(pk['end'],t-1e-6)))
        ctrl=[]
        for t in np.arange(split['support_start'],split['support_end'],1800):
            i=int(np.searchsorted(pk['end'],t-1e-6))
            if i>=179 and i<len(pk['end']) and pt['valid'][i] and not prep.seizure_masked_np[i] and i not in pre_set:
                ctrl.append(i)
        res['seizure_control_interval_check']={'n_controls':len(ctrl),
            'controls_inside_actual_preictal_intervals':sum(any(c[0]-7200<=pk['end'][i]<c[0]-300 for c in clusters) for i in ctrl)}
        # Frozen identity training samples targets one minute after FIT grid queries.
        grid=np.arange(split['support_start']+180*60,split['fit_end'],600.)
        ii=np.searchsorted(pk['end'],grid-1e-6)+1
        ii=ii[ii<len(pk['end'])]
        res['identity_fit_targets_also_inner_targets']=int(prep.inner_target_held[ii].sum())
        sid_fit_grid=np.arange(sid['support_start']+180*60,sid['fit_end'],600.)
        si=np.searchsorted(pk['end'],sid_fit_grid-1e-6)+1
        si=si[si<len(pk['end'])]
        res['sid_identity_fit_target_packets_in_outer_core']=int((~sid['train_packet'][si]).sum())
        res['se_forward_query_calendar_span_hours']=float(np.ptp(split['forward_starts'])/3600.)
        if subject=='epilepsiae_1125':
            # An actual frozen producer: verify export is the source posterior,
            # then propagate that posterior by its measured age on CPU.
            stem=f'{subject}__S-E__P_marks__I-L-G1__state__seed20260906'
            card=json.loads((R/'runs'/f'{stem}.card.json').read_text())
            cfg=RunConfig(**{**card['config'],'device':'cpu'})
            model,_=build_run(cfg,payload,split,sc,prep)
            ck=torch.load(R/'runs'/f'{stem}.ckpt.pt',weights_only=False,map_location='cpu')
            model.load_state_dict(ck['state_dict']);model.eval()
            i=legal[0]['packet'];L=cfg.warm_packets+cfg.grad_packets
            with torch.no_grad():
                m,P,kept=FZ.states_at_queries(model,prep,cfg,np.array([i]))
                _,_,snap=filter_episodes(model,prep,[i-L+1],np.array([L-1]),cfg,
                                         torch.Generator().manual_seed(cfg.seed),training=False)
                age=int(snap['lag'][0,0])
                mt,Pt=propagate_moments(model.dynamics,m,P,age/60.)
            res['actual_checkpoint_frozen_time_test']={
                'query_packet':i,'source_age_minutes':age,
                'export_minus_source_mean_max_abs':float((m-snap['m'][0]).abs().max()),
                'query_propagation_mean_change_max_abs':float((mt-m).abs().max()),
                'query_propagation_covariance_change_max_abs':float((Pt-P).abs().max()),
                'note':'moment propagation demonstrates omitted elapsed time; not a corrected clinical score'}
            del model,ck
        result['patients'][subject]=res
        print(subject,json.dumps(res,default=safe),flush=True)
        del prep,payload
    result['training_status']={}
    for _,c in all_cards:
        key=c.get('stop_reason','MISSING')
        result['training_status'][key]=result['training_status'].get(key,0)+1
    result['canonical_cards_scanned']=len(all_cards)
    for p in [W/'src/topic5_group_event_state/v0311'/f for f in ('data.py','train.py','frozen.py')]+[Path(__file__)]:
        result['evidence_hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (O/'independent_checks.json').write_text(json.dumps(result,indent=2,default=safe)+'\n')


if __name__=='__main__':main()

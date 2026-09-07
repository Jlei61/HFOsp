#!/usr/bin/env python3
"""Locate patient-pattern residuals across completed common-seed geometries."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_topic4_joint_xy_kernel_search as run
from src.topic4_joint_xy import joint_features


def describe(t,xy,groups):
    mask=np.isfinite(t);n=t.shape[1]
    f=joint_features(t,xy,groups);rank=f[:,n:2*n]*4*np.sqrt(n)-1
    counts=mask.sum(axis=0)
    mean=np.divide(np.where(mask,rank,0.).sum(axis=0),counts,out=np.full(n,np.nan),where=counts>0)
    pair_counts=np.zeros((n,n),int);order=np.full((n,n),np.nan);lag=np.full((n,n),np.nan)
    for i in range(n):
        for j in range(n):
            if i==j:continue
            ok=mask[:,i]&mask[:,j];pair_counts[i,j]=ok.sum()
            if ok.any():
                d=t[ok,j]-t[ok,i];order[i,j]=np.mean(d>1e-9)+.5*np.mean(abs(d)<=1e-9);lag[i,j]=np.median(d)
    return {'n_events':len(t),'participation':mask.mean(axis=0),'participating_mean_rank':mean,
            'pair_joint_counts':pair_counts,'pair_i_before_j_probability':order,'pair_median_j_minus_i_ms':lag}


def main():
    out=run.OUT/'replicated_residual_audit';out.mkdir(exist_ok=True)
    obj=run.KernelObjective(run.v1,run.OUT,run.KERNEL);plan=run.read(run.CONFIG)
    stage=run.OUT/'rounds/000';ids=run.read(stage/'race_nomination.json')['candidate_ids']
    paths=[stage/f'combined_{cid}.json' for cid in ids]
    patient=describe(obj.patient,obj.xy,obj.groups);models=[]
    for path in paths:
        row=run.read(path)['candidates'][0]
        if len(row['units'])!=8:raise RuntimeError('common-seed expansion incomplete')
        tables=[run.v1.read_worker(Path(u['worker_path']),obj,plan)[0] for u in row['units']]
        t=np.concatenate(tables)
        if len(t)!=row['n_events']:raise RuntimeError('observation count changed')
        m=describe(t,obj.xy,obj.groups)
        models.append({'candidate_id':row['candidate_id'],'centers_mm':row['candidate']['node_field']['centers_mm'],
                       **m,'participation_residual':m['participation']-patient['participation'],
                       'rank_residual':m['participating_mean_rank']-patient['participating_mean_rank']})
    participation=np.stack([r['participation_residual'] for r in models]);rank=np.stack([r['rank_residual'] for r in models])
    channels=[]
    for i,name in enumerate(obj.training['contact_names']):
        channels.append({'contact':name,'xy_mm':obj.xy[i], 'patient_participation':patient['participation'][i],
            'model_participation_min_max':[min(r['participation'][i] for r in models),max(r['participation'][i] for r in models)],
            'patient_participating_rank':patient['participating_mean_rank'][i],
            'model_rank_min_max':[min(r['participating_mean_rank'][i] for r in models),max(r['participating_mean_rank'][i] for r in models)],
            'all_six_participation_below_patient':bool(np.all(participation[:,i]<0)),
            'all_six_rank_later_than_patient':bool(np.all(rank[:,i]>0)),
            'all_six_rank_earlier_than_patient':bool(np.all(rank[:,i]<0))})
    pairs=[];names=obj.training['contact_names'];n=len(names)
    for i in range(n):
        for j in range(i+1,n):
            if min(r['pair_joint_counts'][i,j] for r in models)<10:continue
            delta=[r['pair_median_j_minus_i_ms'][i,j]-patient['pair_median_j_minus_i_ms'][i,j] for r in models]
            pdelta=[r['pair_i_before_j_probability'][i,j]-patient['pair_i_before_j_probability'][i,j] for r in models]
            if min(delta)>0 or max(delta)<0:
                pairs.append({'contacts':[names[i],names[j]],'patient_median_j_minus_i_ms':patient['pair_median_j_minus_i_ms'][i,j],
                    'all_six_delay_residual_min_max_ms':[min(delta),max(delta)],
                    'minimum_absolute_residual_ms':min(abs(x) for x in delta),
                    'patient_i_before_j_probability':patient['pair_i_before_j_probability'][i,j],
                    'order_probability_residual_min_max':[min(pdelta),max(pdelta)]})
    pairs.sort(key=lambda r:r['minimum_absolute_residual_ms'],reverse=True)
    result={'status':'REPLICATED_GEOMETRY_RESIDUAL_AUDIT_COMPLETE','patient':patient,'models':models,'contacts':channels,
        'consistent_pair_delay_residuals':pairs,'n_geometries':len(models),'n_networks_per_geometry':8,
        'contact_names':names,'groups':obj.groups,'heldout_opened':False,'live_search_changed':False,
        'claim_boundary':'Descriptive residuals in six selected geometries of one patient. Common seeds and contacts are dependent; not a cohort test or proof that all XY lack capacity.',
        'source_hashes':{str(p):run.sha(p) for p in paths+[Path(__file__),Path(obj.training['path']),run.CONFIG]}}
    run.write(out/'summary.json',result)
    print('Contacts below patient participation across all geometries:',[c['contact'] for c in channels if c['all_six_participation_below_patient']])
    print('Later rank across all geometries:',[c['contact'] for c in channels if c['all_six_rank_later_than_patient']])
    print('Earlier rank across all geometries:',[c['contact'] for c in channels if c['all_six_rank_earlier_than_patient']])
    print('Largest persistent pair residuals:',pairs[:5])


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Prospective kernel qualification on training-only paired perturbation controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_topic4_xy_research import training_contract,read,write,sha
from src.topic4_joint_xy_kernel import event_kernel_features,fit_kernel_maps,kernel_map,mapped_distance


def main():
    out=ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search/kernel_qualification'
    out.mkdir(exist_ok=True)
    training,_=training_contract();t=training['onsets_ms'];mask=np.isfinite(t)
    xy=np.asarray(read(ROOT/'results/topic4_sef_hfo/vth_dual_core_xy_research/direction_objective_v2.json')['contact_xy_mm'])
    first=np.nanmin(t,axis=1)[:,None];lag=t-first
    scale=float(np.median(lag[lag>0]));rng=np.random.default_rng(2026090606)
    shuffle=t.copy()
    for row in shuffle:
        idx=np.flatnonzero(np.isfinite(row));row[idx]=rng.permutation(row[idx])
    drop=t.copy();drop[rng.random(t.shape)<.1]=np.nan
    variants={'true':t,'rank_shuffle':shuffle,'time_stretch_1_5':first+1.5*lag,
              'reverse_time':first+np.nanmax(lag,axis=1)[:,None]-lag,'drop_10pct_contacts':drop}
    features=event_kernel_features(t,xy,training['groups'],scale);maps=fit_kernel_maps(features)
    write(out/'kernel_contract.json',{'time_scale_ms':scale,'maps':maps,'patient_training_sha256':training['sha256'],
        'fitted_on_patient_training_only':True,'geometry_candidates_used_to_choose_kernels':False,
        'source_hashes':{str(p):sha(p) for p in [Path(__file__),ROOT/'src/topic4_joint_xy_kernel.py',ROOT/'src/topic4_joint_xy.py']}})
    mapped={name:{k:kernel_map(v,maps[k]) for k,v in event_kernel_features(a,xy,training['groups'],scale).items()}
            for name,a in variants.items()}
    rows=[];blocks=training['block_ids'];unique=np.unique(blocks)
    for draw in range(48):
        left=np.isin(blocks,rng.choice(unique,len(unique)//2,replace=False))
        reference={k:v[~left].mean(axis=0,dtype=float) for k,v in mapped['true'].items()}
        for n in (16,64,256):
            idx=rng.choice(np.flatnonzero(left),n,replace=False)
            for name,values in mapped.items():
                rows.append({'draw':draw,'n':n,'perturbation':name,
                             **{k:mapped_distance(v[idx],reference[k]) for k,v in values.items()}})
    summary=[]
    for n in (16,64,256):
        for metric in maps:
            original=np.array([r[metric] for r in rows if r['n']==n and r['perturbation']=='true'])
            q=float(np.quantile(original,.95))
            for name in variants:
                if name=='true':continue
                altered=np.array([r[metric] for r in rows if r['n']==n and r['perturbation']==name])
                summary.append({'n':n,'metric':metric,'perturbation':name,'patient_q95':q,
                    'true_median':float(np.median(original)),'altered_median':float(np.median(altered)),
                    'paired_delta_median':float(np.median(altered-original)),
                    'fraction_detected_at_empirical_q95':float(np.mean(altered>q))})
    write(out/'paired_draws.json',{'rows':rows})
    write(out/'summary.json',{'status':'PROSPECTIVE_TRAINING_KERNEL_QUALIFICATION_COMPLETE','rows':summary,
        'heldout_opened':False,'model_results_used':False,'running_search_changed':False,
        'inference':'Descriptive paired control detection; threshold estimated from the same true draws, not independent inferential power.'})
    print([(r['metric'],r['perturbation'],r['fraction_detected_at_empirical_q95']) for r in summary
           if r['n']==64 and r['perturbation'] in ('rank_shuffle','time_stretch_1_5')])


if __name__=='__main__':main()

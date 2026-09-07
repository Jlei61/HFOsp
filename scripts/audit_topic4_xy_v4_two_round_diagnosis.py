#!/usr/bin/env python3
"""Fixed two-round review: loss approximation, network retrieval, and residuals.

Descriptive diagnostics on already selected candidates, never a replacement loss
or independent validation. No simulation, acceptance change, or heldout patient.
"""
from pathlib import Path
import sys
import csv
import time
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_topic4_joint_xy_component_search as run
from src.topic4_joint_xy_kernel import event_kernel_features
from src.topic4_xy_replicated_anchors import anchor_eligible


def kernel_mean(x,y,bandwidth):
    total=0.
    yy=np.sum(y*y,axis=1)
    for start in range(0,len(x),64):
        a=x[start:start+64]
        square=np.maximum(np.sum(a*a,axis=1)[:,None]+yy[None,:]-2*a@y.T,0.)
        total+=np.exp(-square/(2*bandwidth**2)).sum()
    return float(total/(len(x)*len(y)))


def main():
    out=run.OUT/'two_round_diagnosis';out.mkdir(exist_ok=True)
    plan=run.read(run.CONFIG);obj=run.KernelObjective(run.v1,run.OUT,run.KERNEL)
    cal=run.read(run.OUT/'patient_calibration.json');sources={}
    def read(p):
        p=Path(p);sources[str(p)]=run.sha(p);return run.read(p)
    pool=read(run.OUT/'baseline_scores.json')['candidates'];baseline=list(pool)
    additions=[];screens={};coverage=[]
    for n in (1,2):
        stage=run.OUT/'rounds'/f'{n:03d}';design=read(stage/'design.json')
        screen=read(stage/'scores.json')['candidates'];pool+=screen
        nomination=read(stage/'race_nomination.json')
        expected=run.select_racers(pool,n,plan)
        assert [r['candidate_id'] for r in expected]==nomination['candidate_ids']
        coverage.append({'round':n,'global':sum(r['proposal']=='fresh_uniform_restart' for r in design['candidates']),
                         'local':sum(r['proposal']=='random_multi_anchor_local' for r in design['candidates']),
                         'anchors':design['local_anchor_candidates']})
        for r in expected:
            screens[r['candidate_id']]=r
            expanded=read(stage/f"combined_{r['candidate_id']}.json")['candidates'][0]
            assert len(expanded['units'])==8
            additions.append(expanded)
            pool=[expanded if p['candidate_id']==r['candidate_id'] else p for p in pool]
        for phase,jobs in [(f'round_{n:03d}',48),(f'race_{n:03d}',36)]:
            done=read(run.OUT/'execution'/phase/'completion.json')
            assert done['jobs']==jobs and done['status']=='ALL_WORKERS_COMPLETE'
            snap=run.OUT/'execution'/phase/'runtime_snapshot.json'
            assert run.sha(snap)==done['runtime_snapshot_sha256'];sources[str(snap)]=run.sha(snap)
    previous=min((r for r in baseline if anchor_eligible(r,plan)),key=lambda r:r['exploration_score'])
    selected=[previous]+additions;tables={};seeds={};mapped={};features={};details=[]
    reference=event_kernel_features(obj.patient,obj.xy,obj.groups,obj.time_scale)['joint']
    bandwidth=float(obj.maps['joint']['bandwidth'])
    # Omit the common patient-patient term: sufficient for exact-kernel ranking,
    # but these signed scores must NOT be compared against qualification gates.
    names=list(obj.training['contact_names']);i=names.index('SCL9');j=names.index('ICL4')
    for r in selected:
        cid=r['candidate_id'];base=run.OLD_V3 if cid==previous['candidate_id'] else run.OUT
        folder=base/'raw_propagation_audit'/cid
        marker=read(folder/'automated_diagnostic_completion.json')
        for p,h in marker['outputs'].items():assert run.sha(p)==h
        summary=read(folder/'summary.json')
        for p,h in summary['source_hashes'].items():assert run.sha(p)==h
        sources.update(summary['source_hashes'])
        cp=folder/'all_event_contacts.csv';sources[str(cp)]=run.sha(cp)
        records=list(csv.DictReader(cp.open()));eventkeys=sorted({(int(x['seed']),int(x['event_index'])) for x in records})
        ei={k:q for q,k in enumerate(eventkeys)};t=np.full((len(eventkeys),len(names)),np.nan)
        seen=set()
        for row in records:
            key=(int(row['seed']),int(row['event_index']));column=names.index(row['contact']);entry=(ei[key],column)
            assert entry not in seen;seen.add(entry)
            if row['participant']=='True':t[entry]=float(row['readout_centroid_ms'])
        assert len(seen)==t.size and len(t)==r['n_events']
        metric=obj.metrics(t)
        for key in ('joint_distance','D_support','D_order','D_lag','direction_distance'):
            assert np.isclose(metric[key],r[key],rtol=1e-8,atol=1e-10),(cid,key)
        tables[cid]=t;seeds[cid]=np.array([key[0] for key in eventkeys]);mapped[cid]=obj.map(t)['joint']
        f=event_kernel_features(t,obj.xy,obj.groups,obj.time_scale)['joint'];features[cid]=f
        exact=kernel_mean(f,f,bandwidth)-2*kernel_mean(f,reference,bandwidth)
        first=np.nanmin(t,axis=1);scaled=first[:,None]+.67*(t-first[:,None]);s=obj.metrics(scaled)
        ok=np.isfinite(t[:,i])&np.isfinite(t[:,j]);delta=t[ok,j]-t[ok,i]
        detail={'candidate_id':cid,'n_events':len(t),'joint_distance':r['joint_distance'],
            'exact_joint_kernel_ranking_score_without_patient_constant':exact,
            'median_contact_span_ms':float(np.nanmedian(np.nanmax(t,axis=1)-first)),
            'native_timing_kernel':r['kernel_distances']['timing_space'],
            'posthoc_scale_067_timing_kernel':s['kernel_distances']['timing_space'],
            'posthoc_scale_067_joint':s['joint_distance'],'posthoc_scale_067_all_checks':run.assess({**r,**s},cal,plan)['pass'],
            'SCL9_ICL4_common_events':int(ok.sum()),'SCL9_before_ICL4_probability':float(np.mean(delta>0)),
            'median_ICL4_minus_SCL9_ms':float(np.median(delta)),
            'kernel_components':r['kernel_distances'],'D_support':r['D_support'],'D_order':r['D_order'],'D_lag':r['D_lag']}
        if cid in screens:
            detail['screen_joint']=screens[cid]['joint_distance'];detail['screen_exploration_score']=screens[cid]['exploration_score']
        details.append(detail);print('audited',cid,flush=True)
    ids=list(tables);retrieval=[]
    for label,a,b in [('screen2_vs_extra6',[2511,2512],list(range(2513,2519))),
                       ('first4_vs_last4',list(range(2511,2515)),list(range(2515,2519)))]:
        for target in ids:
            ref=mapped[target][np.isin(seeds[target],b)].mean(axis=0,dtype=float)
            values={cid:float(np.sum((mapped[cid][np.isin(seeds[cid],a)].mean(axis=0,dtype=float)-ref)**2)) for cid in ids}
            order=sorted(values,key=values.get)
            retrieval.append({'split':label,'target_geometry':target,'same_geometry_rank':order.index(target)+1,
                              'nearest_geometry':order[0],'same_geometry_distance':values[target],
                              'nearest_geometry_distance':values[order[0]],'candidate_count':len(ids)})
    # Monte Carlo calibration draws are in-sample threshold descriptions, not new power estimates.
    calibration_checks={}
    for size,item in cal['samples'].items():
        flags=[all(d[k]<=q for k,q in item['q95'].items()) and
               all(d['kernel_distances'][k]<=q for k,q in item['kernel_q95'].items()) for d in item['draws']]
        calibration_checks[size]={'joint_conjunction_pass_fraction_on_calibration_draws':float(np.mean(flags))}
    full=[r for r in pool if len(r['units'])==8];eligible=[r for r in full if anchor_eligible(r,plan)]
    smaller=cal['samples']['64'];relaxed=[]
    for r in additions:
        fits=all(r[k]<=v for k,v in smaller['q95'].items()) and all(r['kernel_distances'][k]<=v for k,v in smaller['kernel_q95'].items())
        relaxed.append({'candidate_id':r['candidate_id'],'all_distribution_checks_using_64_bin':fits})
    xy=np.concatenate([r['candidate']['node_field']['centers_mm'] for r in pool])
    patient_ok=np.isfinite(obj.patient[:,i])&np.isfinite(obj.patient[:,j]);pd=obj.patient[patient_ok,j]-obj.patient[patient_ok,i]
    result={'status':'TWO_COMPLETED_ROUNDS_CAUSE_DIAGNOSIS_NOT_QUALIFICATION','cutoff_round':2,'updated_unix':time.time(),
        'n_geometries_scored':len(pool),'n_full_eight_network':len(full),'n_anchor_eligible':len(eligible),
        'n_new_expanded':len(additions),'event_records_new_expanded':sum(r['n_events'] for r in additions),
        'coverage':coverage,'all_center_coordinate_min':xy.min(axis=0),'all_center_coordinate_max':xy.max(axis=0),
        'new_expanded_all_qualified':sum(run.assess(r,cal,plan)['pass'] for r in additions),
        'candidate_details':details,'exact_vs_RFF_joint_ranking_spearman':float(spearmanr(
            [d['joint_distance'] for d in details],[d['exact_joint_kernel_ranking_score_without_patient_constant'] for d in details]).statistic),
        'screen_vs_eight_raw_joint_spearman_selected12':float(spearmanr([screens[r['candidate_id']]['joint_distance'] for r in additions],[r['joint_distance'] for r in additions]).statistic),
        'known_geometry_retrieval':retrieval,'calibration_conjunction':calibration_checks,
        'less_strict_64_bin_sensitivity_no_qualification':relaxed,
        'patient':{'n_events':len(obj.patient),'median_contact_span_ms':float(np.median(np.nanmax(obj.patient,axis=1)-np.nanmin(obj.patient,axis=1))),
                   'SCL9_before_ICL4_probability':float(np.mean(pd>0)),'median_ICL4_minus_SCL9_ms':float(np.median(pd))},
        'patient_heldout_opened':False,'live_search_changed':False,
        'limitations':['Selected 13 geometries and shared network seeds; descriptive, not an exhaustive optimization benchmark.',
            'Known-geometry target and candidate network sets are disjoint, but all geometries were previously selected on patient training data.',
            'Fixed patient-trained kernel maps used for synthetic-model retrieval; this probes the current pipeline, not every possible loss.',
            'Exact-kernel scores omit a common reference term and only test ranking, never qualification.',
            'Timing scaling is an offline diagnostic, not a simulated tau intervention or accepted result.'],
        'source_hashes':{**sources,**{str(p):run.sha(p) for p in [Path(__file__),run.CONFIG,run.OUT/'patient_calibration.json',Path(obj.training['path']),run.KERNEL]}}}
    run.write(out/'summary.json',result)
    print({k:v for k,v in result.items() if k not in ('source_hashes','candidate_details','known_geometry_retrieval')})
    for split in ('screen2_vs_extra6','first4_vs_last4'):
        ranks=[r['same_geometry_rank'] for r in retrieval if r['split']==split]
        print(split,'ranks',ranks,'top1',sum(r==1 for r in ranks),'median',np.median(ranks))


if __name__=='__main__':main()

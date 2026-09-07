#!/usr/bin/env python3
"""Post-round development review: paired, matched-count and native-field checks."""
from pathlib import Path
import argparse,json,pickle,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.qualify_topic4_observation_repair import OUT as INPUT,OLD,read,write,sha
from scripts.report_topic4_observation_repair import table,compact
from scripts import audit_topic4_xy_raw_propagation_video as native_producer
OUT=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_round1_review'
ANCHORS=['historical','support_rank','old_joint']


def state(name,**kw):
    write(OUT/'status.json',{'status':name,'updated_unix':time.time(),
          'automatic_next_round':False,'final_substrate_frozen':False,'fig5_released':False,**kw})


def wait_for_parent(wait):
    while True:
        parent=read(INPUT/'status.json');latest=read(INPUT/'latest_analysis.json')
        if parent['status']=='ROUND1_COMPLETE_PENDING_SCIENTIFIC_REVIEW' and latest['n_completed_jobs']==200:return
        if parent['status'].startswith(('ERROR','FAILURE')):raise RuntimeError('parent paused: '+str(parent))
        if not wait:raise RuntimeError('full 200-job analysis is not yet complete')
        state('WAITING_FOR_FULL_ROUND',parent_status=parent['status'],complete=parent.get('complete'),latest_analyzed=latest['n_completed_jobs'])
        time.sleep(30)


def interval(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return np.quantile(a,[.05,.5,.95]).tolist() if len(a) else None


def matched_counts(rows,workers,ev):
    by={}
    for w in workers:by.setdefault(w['candidate_id'],{})[w['seed']]=table(w)
    contrasts=[]
    for ci,r in enumerate(rows):
        cid=r['candidate_id'];anchor=cid.split('__')[0]
        if anchor not in ANCHORS or cid.endswith('__baseline'):continue
        baseline=by[anchor+'__baseline'];candidate=by[cid];counts={s:min(len(baseline[s]),len(candidate[s])) for s in sorted(baseline)}
        rng=np.random.default_rng(202609071100+ci);draws=[]
        for it in range(32):
            pair={}
            for name,source in [('baseline',baseline),('candidate',candidate)]:
                tt=[];ss=[]
                for s,n in counts.items():
                    if not n:continue
                    t=source[s];tt.append(t if n==len(t) else t[rng.choice(len(t),n,replace=False)]);ss.extend([s]*n)
                pair[name]=compact(ev.metrics(np.concatenate(tt),ss,detail=False)) if tt else None
            draws.append(pair)
        delta={}
        for key in ['joint_distance','supported_fraction','unsupported_fraction','direction_distance']:
            delta[key]=interval([x['candidate'][key]-x['baseline'][key] for x in draws if x['candidate'] is not None and x['baseline'][key] is not None])
        delta['coverage_by_mode']=[interval([x['candidate']['coverage_by_mode'][m]-x['baseline']['coverage_by_mode'][m] for x in draws if x['candidate'] is not None]) for m in range(ev.k)]
        row={'candidate_id':cid,'baseline_id':anchor+'__baseline','matched_counts_by_seed':counts,'n_draws':32,
             'candidate_minus_baseline_q05_median_q95':delta,'draws':draws,
             'interpretation':'Conditional event subsampling only; intervals are not patient-level confidence intervals or equivalence tests.'}
        contrasts.append(row);print('matched-count review',cid,flush=True)
    write(OUT/'matched_count_parameter_effects.json',{'comparisons':contrasts,'all_single_factor_arms_included':True})
    return contrasts


def native_audit(workers,ev):
    from src.topic4_observation_repaired import order_distribution
    aggregate={};contacts={};records=[]
    for w in workers:
        meta=read(w['worker_path']);arr=Path(meta['arrays']['path'])
        if sha(w['worker_path'])!=w['worker_sha256'] or sha(arr)!=w['arrays_sha256']:raise RuntimeError('worker changed')
        cid=w['candidate_id'];events=w['observation'];t=table(w);indices=events['primary_event_indices']
        with np.load(arr) as z:
            field=z['sheet_activity_counts'];dt=float(z['sheet_activity_frame_ms']);xy=z['contact_xy_mm'];names=z['contact_names'].astype(str)
            cell=np.floor(xy).astype(int);local=field[:,cell[:,1],cell[:,0]].astype(float).T;times=(np.arange(len(field))+.5)*dt
            assert dt==2. and field.shape[1:]==(20,20)
            for ei,(index,row) in enumerate(zip(indices,t)):
                start,stop=events['windows_ms'][index];use=(times>=start)&(times<stop);weights=local[:,use];den=weights.sum(1)
                tn=np.divide(weights@times[use],den,out=np.full(len(names),np.nan),where=den>0);tn[~np.isfinite(row)]=np.nan
                rec={'candidate_id':cid,'seed':w['seed'],'event_index':index,
                     **native_producer.comparison(row,tn),'median_absolute_contact_centroid_difference_ms':float(np.nanmedian(abs(tn-row)))}
                records.append(rec);aggregate.setdefault(cid,[]).append(rec)
        labels,_,_=ev.classify(t)
        for m in range(ev.k):
            cur=t[labels==m]
            if len(cur):contacts.setdefault((cid,m),[]).append(cur)
    summaries=[]
    for cid,rr in aggregate.items():
        summaries.append({'candidate_id':cid,'n_events':len(rr),
            **{k:interval([r[k] for r in rr if r[k] is not None]) for k in ['order_discordance','pair_lag_mae_ms','median_absolute_contact_centroid_difference_ms']}})
    residuals=[]
    for (cid,m),tables in contacts.items():
        t=np.concatenate(tables);ref=ev.fit[ev.fit_labels==m]
        residuals.append({'candidate_id':cid,'mode':m,'n_model_events':len(t),
            'model_participation':np.isfinite(t).mean(0).tolist(),'patient_participation':np.isfinite(ref).mean(0).tolist(),
            'participation_model_minus_patient':(np.isfinite(t).mean(0)-np.isfinite(ref).mean(0)).tolist()})
    write(OUT/'native_readout_audit.json',{'all_primary_events':records,'candidates':summaries,
        'role':'Containing 1 mm cell count centroid versus density readout, same window and participation mask; sensitivity diagnostic, not independent neuronal onset validation.',
        'native_quantity':'distinct active E neurons per cell per 2 ms; unadjusted counts, not spikes or HFO'})
    write(OUT/'mode_contact_residuals.json',{'contact_names':names.tolist(),'rows':residuals,'patient_reference':'frozen FIT modes; all model-assigned events, including unsupported events'})
    return summaries,residuals


def figures(rows,contrasts,residuals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    arms=[r['candidate_id'].split('__')[1] for r in rows if r['candidate_id'].startswith('historical__') and not r['candidate_id'].endswith('__baseline')]
    lookup={r['candidate_id']:r for r in contrasts}
    fig,axes=plt.subplots(1,4,figsize=(17,8),layout='constrained')
    for ax,key,title in zip(axes,['joint_distance','unsupported_fraction',0,1],['Joint distance change','OOD fraction change','Mode 0 coverage change','Mode 1 coverage change']):
        values=np.full((len(arms),3),np.nan)
        for i,arm in enumerate(arms):
            for j,anchor in enumerate(ANCHORS):
                d=lookup[anchor+'__'+arm]['candidate_minus_baseline_q05_median_q95'];v=d['coverage_by_mode'][key] if isinstance(key,int) else d[key]
                if v is not None:values[i,j]=v[1]
        vmax=max(float(np.nanmax(abs(values))),.01)
        im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        for i in range(len(arms)):
            for j in range(3):ax.text(j,i,f'{values[i,j]:+.3f}',ha='center',va='center',fontsize=7)
        ax.set(xticks=range(3),xticklabels=['Historical','Support/rank','Old joint'],yticks=range(len(arms)),yticklabels=[s.replace('_weight_scale','').replace('_ms','') for s in arms],title=title)
        ax.tick_params(axis='y',labelsize=7);ax.tick_params(axis='x',rotation=25,labelsize=8)
        fig.colorbar(im,ax=ax,shrink=.6,label='Candidate − its own baseline')
    fig.suptitle('Same event count within each seed: 32-draw median changes\nBlue helps distance/OOD; red helps coverage. Development diagnostics, no significance claim.',fontsize=12)
    fig.savefig(figs/'matched_count_parameter_effects.png',dpi=160);fig.savefig(figs/'matched_count_parameter_effects.pdf');plt.close(fig)
    selected=['support_rank__baseline','support_rank__vth_low','old_joint__baseline','old_joint__tau_d_GABA_ms_high']
    rr={(r['candidate_id'],r['mode']):r for r in residuals};names=read(OUT/'mode_contact_residuals.json')['contact_names']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for m,ax in enumerate(axes):
        values=np.array([rr[(cid,m)]['participation_model_minus_patient'] for cid in selected])
        im=ax.imshow(values,cmap='RdBu_r',vmin=-.5,vmax=.5,aspect='auto');ax.set(xticks=range(15),xticklabels=names,yticks=range(4),yticklabels=selected,title=f'Patient mode {m}: participation residual')
        ax.tick_params(axis='x',rotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=7);fig.colorbar(im,ax=ax,label='Model − patient',shrink=.7)
    fig.savefig(figs/'candidate_contact_participation.png',dpi=160);fig.savefig(figs/'candidate_contact_participation.pdf');plt.close(fig)
    (figs/'README.md').write_text('### matched_count_parameter_effects.png\n每个网络种子内匹配事件数，32 次子抽样后的参数效应中位数；PDF 为同版导出。前三组位置各自与自己的基线比较。**关注点**：降低联合距离/OOD 与覆盖两模式是否一致，颜色不表示统计显著性。\n\n### candidate_contact_participation.png\n两个待审参数改动及各自基线相对患者各模式的逐通道参与偏差，PDF 为同版导出。患者参考来自冻结 FIT，模型包含全部被分配到该模式的事件。**关注点**：联合分数改善后，是否仍存在固定通道缺失或过度参与。\n')


def movies(workers):
    if sha(native_producer.VIDEO)!=read(native_producer.META)['sha256']:raise RuntimeError('patient video changed')
    if sha(native_producer.PATIENT)!='d2fd193cbb30c7cd30a8c173c955ed300d08e657bac8eb43f0d29c7b9a0a9a8b':raise RuntimeError('patient reference changed')
    with np.load(native_producer.PATIENT) as z:patient={k:z[k] for k in z.files}
    design={r['candidate_id']:r for r in read(OLD/'design.json')['candidates']};reports=[]
    for cid in ['support_rank__baseline','support_rank__vth_low','old_joint__baseline','old_joint__tau_d_GABA_ms_high']:
        w=min((r for r in workers if r['candidate_id']==cid),key=lambda r:r['seed']);obs=w['observation'];indices=obs['primary_event_indices']
        if not indices:reports.append({'candidate_id':cid,'status':'NO_PRIMARY_EVENT_IN_LOWEST_SEED'});continue
        folder=OUT/'raw_previews'/cid;(folder/'figures').mkdir(parents=True,exist_ok=True)
        meta=read(w['worker_path'])
        with np.load(meta['arrays']['path']) as z:arrays={k:z[k] for k in ['contact_names','contact_xy_mm','sheet_activity_counts','sheet_activity_frame_ms','contact_envelope','contact_envelope_dt_ms','topology_seed']}
        r=native_producer.render_movie(folder,arrays,np.asarray(obs['centroid_ms'],float),obs,patient,design[cid],indices[0]);r['candidate_id']=cid;r['selection']='lowest seed, first primary isolated event; posthoc conditions, no movie resemblance selection';reports.append(r)
        (folder/'figures/README.md').write_text('### first_chronological_event_raw_comparison.gif\n本条件最低网络种子的首个孤立事件，与 Supplementary Video 1 的两个未配对人体示例比较。上方包含未做 readout 平滑的原生 1 mm / 2 ms 活动。**关注点**：首发区域、扩展路径、接触点外活动及缺失通道；不能用一个示例代替全事件分布验证，待人工检查。\n')
        write(OUT/'raw_previews.json',{'movies':reports,'human_acceptance':False})
        print('raw preview',cid,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);wait_for_parent(args.wait)
    inputs=['before_after_parameter_comparison.json','reobserved_workers.json','paired_parameter_comparison.json','actual_N_patient_reference.json','observation_contract.json','evaluator.pkl']
    sources={str(INPUT/n):sha(INPUT/n) for n in inputs};rows=read(INPUT/'before_after_parameter_comparison.json')['candidates'];workers=read(INPUT/'reobserved_workers.json')['workers']
    if len(rows)!=50 or len(workers)!=200 or any(len(r['seeds'])!=4 for r in rows):raise RuntimeError('incomplete round')
    with open(INPUT/'evaluator.pkl','rb') as f:ev=pickle.load(f)
    if sha(INPUT/'evaluator.pkl')!=read(INPUT/'qualification.json')['evaluator_sha256']:raise RuntimeError('patient evaluator changed')
    state('MATCHED_COUNT_ANALYSIS',n_candidates=50,n_workers=200)
    contrasts=matched_counts(rows,workers,ev)
    state('NATIVE_FIELD_AND_CONTACT_ANALYSIS');native,residuals=native_audit(workers,ev);figures(rows,contrasts,residuals)
    state('RENDERING_RAW_PREVIEWS');movies(workers)
    for p,h in sources.items():
        if sha(p)!=h:raise RuntimeError('analysis input changed during review')
    write(OUT/'review_summary.json',{'status':'COMPUTATIONAL_REVIEW_COMPLETE_PENDING_AGENT_VISUAL_AND_SCIENTIFIC_REVIEW',
        'sources':sources,'review_script_sha256':sha(__file__),'n_candidates':50,'n_simulations':200,
        'runaway_networks':sum(r['runaway'] for r in workers),'no_post_burnin_networks':sum(r['observation_status']=='NOT_ESTIMABLE_RUNAWAY_BEFORE_BURNIN' for r in workers),
        'contrasts':len(contrasts),'candidate_native_summaries':native,'primary_patient':'E10 / Epilepsiae 1146',
        'limits':['single-patient reused development data','four shared network seeds','matched-N intervals reflect event subsampling, not independent confirmation',
                  'native readout check shares windows and contact masks','posthoc shortlisted movies are unpaired examples, not distribution qualification'],
        'final_model_acceptance':None,'automatic_next_round':False,'fig5_released':False})
    state('COMPUTATIONAL_REVIEW_COMPLETE_PENDING_AGENT_REVIEW',n_candidates=50,n_workers=200)


if __name__=='__main__':
    try:main()
    except Exception as exc:state('ERROR',error=repr(exc));raise

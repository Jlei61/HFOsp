#!/usr/bin/env python3
"""All-candidate development report and deterministic raw-field review package."""
from pathlib import Path
import csv
import json
import os
import pickle
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_topic4_multidimensional_pilot import OUT,SEEDS,read,write,sha
from scripts.run_topic4_xy_research import training_contract,PYTHON,ENV
from src.topic4_joint_xy import observable_groups


def self_reference(ev,counts,seed):
    """Actual-N block-resampling envelope; descriptive, not a joint pass rule."""
    rng=np.random.default_rng(seed);blocks=ev.partition['PROBE'];samples=[]
    for _ in range(24):
        tables=[];units=[]
        for network,n in enumerate(counts):
            if not n:continue
            chosen=rng.choice(blocks);source=ev.patient[ev.blocks==chosen]
            # A pseudo-network is one recording block. Sampling with replacement
            # only when a small block has fewer events than the actual model unit.
            tables.append(source[rng.choice(len(source),n,replace=n>len(source))])
            units.extend([network]*n)
        if not tables:continue
        m=ev.metrics(np.concatenate(tables),units,detail=False)
        samples.append({k:m[k] for k in ('joint_distance','supported_fraction','unsupported_fraction','operational_mode_coverage','direction_distance')})
    return {'actual_unit_event_counts':counts,'draws':samples,
            'range_q05_q95':{k:np.quantile([r[k] for r in samples if r[k] is not None],[.05,.95]).tolist()
                            for k in samples[0] if any(r[k] is not None for r in samples)} if samples else {},
            'role':'descriptive patient-block envelope at actual model N; 24 draws, no equivalence claim'}


def main():
    manifest=read(OUT/'evaluation_manifest.json')
    if sha(OUT/'evaluator.pkl')!=manifest['evaluator_sha256']:raise RuntimeError('evaluator changed')
    with open(OUT/'evaluator.pkl','rb') as f:ev=pickle.load(f)
    training,obj=training_contract();design=read(OUT/'design.json')
    observation=read(ROOT/'config/topic4_joint_xy_kernel_v4.json')['observation']
    rows=[];csvrows=[];folder=OUT/'rounds/001';folder.mkdir(parents=True,exist_ok=True)
    worker_dir=OUT/'execution/paired_round1/workers'
    for ci,candidate in enumerate(design['candidates']):
        cid=candidate['candidate_id'];destination=folder/f'combined_{cid}.json'
        if destination.exists():
            saved=read(destination)['candidates'][0]
            for u in saved['units']:
                if sha(u['worker_path'])!=u['worker_sha256']:raise RuntimeError('saved worker changed')
            rows.append(saved);continue
        tables=[];units=[];seed_labels=[]
        for seed in SEEDS:
            wp=worker_dir/f'{cid}_seed_{seed}.json';meta=read(wp);array=Path(meta['arrays']['path'])
            if sha(array)!=meta['arrays']['sha256']:raise RuntimeError('worker arrays changed')
            with np.load(array) as z:
                if list(z['contact_names'].astype(str))!=training['contact_names']:raise RuntimeError('contact identity mismatch')
                table,obs=observable_groups(z['contact_envelope'],float(z['contact_envelope_dt_ms']),**observation)
                geometry=meta['xy_geometry_audit']
                if not np.isfinite(z['active_fraction']).all():raise RuntimeError('nonfinite simulation')
            unit={'seed':seed,'worker_path':str(wp),'worker_sha256':sha(wp),'arrays_sha256':sha(array),
                  'observation':obs,'n_events':len(table),'runaway':meta['simulation']['runaway_early_stop_ms'] is not None,
                  'geometry':geometry,'parameter_audit':meta['multidimensional_parameter_audit'],
                  'evaluation':ev.metrics(table,np.full(len(table),seed))}
            units.append(unit);tables.append(table);seed_labels.extend([seed]*len(table))
        pooled=np.concatenate(tables);metrics=ev.metrics(pooled,seed_labels)
        old=obj.component_vector(pooled,training['reference'],training['groups'],training['pairs'],training['embedding'],
                                  composite=False,components=('D_support','D_order','D_lag')) if len(pooled) else {}
        record={'candidate':candidate,'candidate_id':cid,'units':units,**metrics,
                'legacy_diagnostics':old,'patient_self_at_actual_N':self_reference(ev,[len(t) for t in tables],2026090700+ci),
                'runaway_networks':sum(u['runaway'] for u in units),
                'scientific_status':'DEVELOPMENT_COMPARISON_NOT_QUALIFIED',
                'final_acceptance':None,'fig5_released':False}
        write(destination,{'candidates':[record]});rows.append(record)
        print('evaluated',cid,len(pooled),flush=True)
    lookup={r['candidate_id']:r for r in rows}
    endpoints=('joint_distance','supported_fraction','unsupported_fraction','operational_mode_coverage','direction_distance')
    paired=[]
    for row in rows:
        c=row['candidate'];baseline=lookup[c['anchor']+'__baseline']
        deltas=[]
        for a,b in zip(row['units'],baseline['units']):
            assert a['seed']==b['seed']
            deltas.append({'seed':a['seed'],**{k:a['evaluation'].get(k)-b['evaluation'].get(k)
                 if a['evaluation'].get(k) is not None and b['evaluation'].get(k) is not None else None for k in endpoints}})
        paired.append({'candidate_id':row['candidate_id'],'baseline':baseline['candidate_id'],'seed_differences':deltas})
        csvrows.append({'candidate_id':row['candidate_id'],'anchor':c['anchor'],'arm':c['arm'],'n_events':row['n_events'],
            'runaway_networks':row['runaway_networks'],**{k:row.get(k) for k in endpoints}})
    write(OUT/'all_candidate_evaluation.json',{'candidates':rows,'mode_manifest':manifest,'observation':observation,
        'all_candidates_reported':True,'parameters_selected_by_new_metrics':False,'new_qualification_pass':False})
    write(OUT/'paired_parameter_effects.json',{'comparisons':paired,'inference':'within-network seed differences; four reused development networks; no causal necessity or clinical equivalence claim'})
    with open(OUT/'candidate_metrics.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,11),layout='constrained')
    for ax,key in zip(axes,['joint_distance','supported_fraction','unsupported_fraction']):
        values=[r.get(key,np.nan) if r.get(key) is not None else np.nan for r in rows]
        ax.scatter(values,np.arange(len(rows)),s=22)
        ax.set(xlabel=key,yticks=np.arange(len(rows)),yticklabels=[r['candidate_id'] for r in rows] if ax is axes[0] else [])
        ax.tick_params(axis='y',labelsize=6);ax.grid(axis='x',alpha=.2);ax.invert_yaxis()
    fig.suptitle('All prespecified arms | development diagnostics | no model accepted')
    fig.savefig(figs/'all_candidate_metrics.png',dpi=160);fig.savefig(figs/'all_candidate_metrics.pdf');plt.close(fig)
    (figs/'README.md').write_text('### all_candidate_metrics.png / .pdf\n按预先固定的候选顺序展示所有位置和参数组合，不按成绩删选。联合距离、患者支持比例、不支持比例分别呈现；支持与不支持之间还有不能判定的部分。\n**关注点**：同一位置下参数改变能否同时改善多个指标；需结合每 seed 配对差和患者自对照范围，不凭单项最低值验收。\n')
    # Same accepted raw-video producer, routed to this versioned output root.
    # Complete-event CSV and first chronological movie are generated for EVERY
    # candidate with observed events, independent of its score.
    raw=[]
    for row in rows:
        cid=row['candidate_id'];summary=OUT/'raw_propagation_audit'/cid/'summary.json'
        if row['n_events']==0:
            raw.append({'candidate_id':cid,'status':'NO_OBSERVED_EVENTS_FULL_TRAJECTORY_RETAINED'});continue
        if not summary.exists():
            subprocess.run([PYTHON,str(ROOT/'scripts/render_topic4_multidimensional_raw.py'),'--candidate',cid,'--round','1'],
                           cwd=ROOT,env=ENV,check=True)
        raw.append({'candidate_id':cid,'summary':str(summary),'sha256':sha(summary)})
    write(OUT/'raw_review_manifest.json',{'candidates':raw,'status':'GENERATED_PENDING_HUMAN_AND_AGENT_VISUAL_REVIEW',
         'gold_standard_reference':'Supplementary Video 1 actual contact envelopes and full patient training distribution',
         'visual_qa_complete':False,'fig5_released':False})


if __name__=='__main__':main()

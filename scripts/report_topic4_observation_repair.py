#!/usr/bin/env python3
"""Report isolated-event fits alongside crowded activity and native-field movies."""
from pathlib import Path
import argparse,csv,json,pickle,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.qualify_topic4_observation_repair import OUT,OLD,read,write,sha


def table(worker):
    obs=worker['observation']
    return np.asarray(obs['centroid_ms'],float).reshape(-1,15)[obs['primary_event_indices']]


def compact(m):
    return {**{k:m.get(k) for k in ('n_events','joint_distance','supported_fraction','unsupported_fraction','mode_presence_fraction','direction_distance')},
            'coverage_by_mode':[r['patient_probe_neighborhood_coverage'] for r in m['modes']]}


def number(value):
    return np.nan if value is None else float(value)


def coverage(metric,mode):
    return number(metric['modes'][mode]['patient_probe_neighborhood_coverage']) if len(metric['modes'])>mode else np.nan


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gif',action='store_true');args=ap.parse_args()
    with open(OUT/'evaluator.pkl','rb') as f:ev=pickle.load(f)
    if sha(OUT/'evaluator.pkl')!=read(OUT/'qualification.json')['evaluator_sha256']:raise RuntimeError('evaluator changed')
    data=read(OUT/'reobserved_workers.json');rows=read(OUT/'before_after_parameter_comparison.json')['candidates']
    workers=data['workers'];by={}
    for w in workers:by.setdefault(w['candidate_id'],{})[w['seed']]=w
    # Sampling CAL blocks avoids comparing sampled PROBE events with themselves.
    # These remain reused development blocks, not independent clinical validation.
    references=[];paired=[]
    for row in rows:
        cid=row['candidate_id'];units=by[cid];counts=[len(table(w)) for s,w in sorted(units.items())]
        seed=2026090720+next(i for i,c in enumerate(read(OLD/'design.json')['candidates']) if c['candidate_id']==cid)
        rng=np.random.default_rng(seed);draws=[]
        for _ in range(24):
            tt=[];ss=[]
            for unit,n in enumerate(counts):
                if not n:continue
                block=rng.choice(ev.partition['CAL']);p=ev.patient[ev.blocks==block]
                tt.append(p[rng.choice(len(p),n,replace=n>len(p))]);ss.extend([unit]*n)
            if tt:draws.append(compact(ev.metrics(np.concatenate(tt),ss,detail=False)))
        refs={'candidate_id':cid,'network_event_counts':counts,'draws':draws,
              'sampling':'24 CAL-block draws at each actual network event count; PROBE is query-only; descriptive development reference'}
        references.append(refs)
        base=by.get(cid.split('__')[0]+'__baseline',{});common=sorted(set(base)&set(units))
        individual=[]
        for s in common:
            individual.append({'seed':s,'baseline':compact(ev.metrics(table(base[s]),detail=False)),
                               'candidate':compact(ev.metrics(table(units[s]),detail=False))})
        pair={'candidate_id':cid,'common_seeds':common,'per_seed':individual}
        for label,source in [('baseline',base),('candidate',units)]:
            if common:
                ts=[table(source[s]) for s in common];ids=np.concatenate([np.full(len(t),s) for t,s in zip(ts,common)])
                pair[label+'_common_seed_pooled']=compact(ev.metrics(np.concatenate(ts),ids,detail=False))
        paired.append(pair)
        print('reported',cid,flush=True)
    write(OUT/'actual_N_patient_reference.json',{'candidates':references,'clinical_equivalence_test':False})
    write(OUT/'paired_parameter_comparison.json',{'candidates':paired,'event_count_difference_not_removed':True,
          'interpretation':'Common network seeds; read actual-N reference too. These are development comparisons, not significance or mechanism necessity tests.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted(rows,key=lambda r:(not r['candidate_id'].endswith('__baseline'),r['candidate_id']))
    labels=[r['candidate_id'].replace('historical__','').replace('_weight_scale','').replace('E_to_','E→').replace('_',' ')+'\n'+str(len(r['seeds']))+' seeds' for r in rows]
    x=np.arange(len(rows));fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    a,b,c,d=axes.ravel()
    a.bar(x-.17,[number(r['metrics']['old']['joint_distance']) for r in rows],.34,label='Old observer',color='.6')
    a.bar(x+.17,[number(r['metrics']['1.0']['joint_distance']) for r in rows],.34,label='Repaired, isolated',color='#2878a5')
    a.set(title='A  Same saved trajectories, repaired observation',ylabel='Joint distance (lower is closer)');a.legend(fontsize=8)
    b.bar(x,[r['selection_audit']['n_primary'] for r in rows],label='Isolated primary events',color='#2878a5')
    b.bar(x,[r['selection_audit']['n_detected']-r['selection_audit']['n_primary'] for r in rows],bottom=[r['selection_audit']['n_primary'] for r in rows],label='Retained only in activity audit',color='#dbb46d')
    b.set(title='B  Detection yield and window selection',ylabel='Events in available simulations');b.legend(fontsize=8)
    for row,label in zip(rows,labels):c.plot([.75,1.,1.25],[number(row['metrics'][k]['joint_distance']) for k in ('.75','1.0','1.25')],'.-',label=label.replace('\n',' / '))
    c.set(title='C  Fixed threshold sensitivity',xlabel='Multiplier of reference threshold increment',ylabel='Joint distance');c.legend(fontsize=7)
    for mode,color in [(0,'#7b3294'),(1,'#008837')]:
        d.bar(x+(mode-.5)*.32,[coverage(r['metrics']['1.0'],mode) for r in rows],.32,color=color,label=f'Patient mode {mode}')
    d.set(title='D  Patient neighborhood coverage, not mode presence',ylabel='Fraction of supported patient PROBE events',ylim=(0,1));d.legend(fontsize=8)
    for ax in (a,b,d):ax.set_xticks(x,labels,fontsize=8)
    if len(rows)>8:
        # Expand to readable horizontal rows as the full 50-condition round fills.
        plt.close(fig);fig,axes=plt.subplots(1,4,figsize=(21,max(8,len(rows)*.40)),layout='constrained',sharey=True)
        a,b,c,d=axes
        labels=[r['candidate_id'].replace('_weight_scale','').replace('E_to_','E→')+f" ({len(r['seeds'])} seeds; runaway {r.get('simulation_audit',{}).get('runaway_networks',0)})" for r in rows]
        for key,offset,color,label in [('old',-.17,'.6','Old'),('1.0',.17,'#2878a5','Repaired')]:
            a.barh(x+offset,[number(r['metrics'][key]['joint_distance']) for r in rows],.34,color=color,label=label)
        primary=[r['selection_audit']['n_primary'] for r in rows]
        b.barh(x,primary,color='#2878a5',label='Primary')
        b.barh(x,[r['selection_audit']['n_detected']-r['selection_audit']['n_primary'] for r in rows],left=primary,color='#dbb46d',label='Audit only')
        for key,color in [('.75','#7570b3'),('1.0','#2878a5'),('1.25','#d95f02')]:
            c.scatter([number(r['metrics'][key]['joint_distance']) for r in rows],x,c=color,s=18,label='threshold × '+key)
        for mode,color in [(0,'#7b3294'),(1,'#008837')]:
            d.barh(x+(mode-.5)*.32,[coverage(r['metrics']['1.0'],mode) for r in rows],.32,color=color,label=f'Mode {mode}')
        for ax,title in zip(axes,['A  Joint distance','B  Event counts','C  Threshold sensitivity','D  Patient coverage']):
            ax.set_title(title);ax.legend(fontsize=7,loc='lower right');ax.set_yticks(x,labels,fontsize=7)
        a.invert_yaxis();d.set_xlim(0,1)
    fig.suptitle('Observation repair audit — development only; no substrate accepted',fontsize=14)
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    fig.savefig(figs/'observation_repair_comparison.png',dpi=170);fig.savefig(figs/'observation_repair_comparison.pdf');plt.close(fig)
    with open(OUT/'candidate_metrics.csv','w') as f:
        records=[]
        for r in rows:
            m=r['metrics']['1.0'];records.append({'candidate_id':r['candidate_id'],'n_networks':len(r['seeds']),
                'runaway_networks':r.get('simulation_audit',{}).get('runaway_networks',0),
                'no_post_burnin_networks':r.get('simulation_audit',{}).get('no_post_burnin_networks',0),
                **r['selection_audit'],**{k:m[k] for k in ('joint_distance','supported_fraction','unsupported_fraction','mode_presence_fraction')}})
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if args.gif:
        from scripts import audit_topic4_xy_raw_propagation_video as producer
        if sha(producer.VIDEO)!=read(producer.META)['sha256']:raise RuntimeError('reference GIF changed')
        if sha(producer.PATIENT)!='d2fd193cbb30c7cd30a8c173c955ed300d08e657bac8eb43f0d29c7b9a0a9a8b':raise RuntimeError('patient reference changed')
        cid='historical__baseline';w=by[cid][min(by[cid])];obs=w['observation'];indices=obs['primary_event_indices']
        if not indices:raise RuntimeError('first network has no isolated event for prespecified preview')
        meta=read(w['worker_path']);path=Path(meta['arrays']['path'])
        if sha(path)!=w['arrays_sha256']:raise RuntimeError('raw trajectory changed')
        with np.load(path) as z:arrays={k:z[k] for k in ['contact_names','contact_xy_mm','sheet_activity_counts','sheet_activity_frame_ms','contact_envelope','contact_envelope_dt_ms','topology_seed']}
        with np.load(producer.PATIENT) as z:patient={k:z[k] for k in z.files}
        candidate=next(c for c in read(OLD/'design.json')['candidates'] if c['candidate_id']==cid)
        report=producer.render_movie(OUT,arrays,np.asarray(obs['centroid_ms'],dtype=float),obs,patient,candidate,indices[0])
        report.update(selection='baseline, lowest seed, first chronological PRIMARY isolated event; no resemblance selection',observer_contract_sha256=sha(OUT/'observation_contract.json'),worker_sha256=w['worker_sha256'])
        write(OUT/'raw_preview.json',report)
    (figs/'README.md').write_text('### observation_repair_comparison.png\n同一批已保存轨迹的观测修复前后比较，PDF 为同版导出。联合距离只比较孤立事件，拥挤窗口另行报告；网络数和事件数不同，图中的条高不表示显著性。**关注点**：阈值敏感性、被排除活动与真正的患者邻域覆盖。\n\n### first_chronological_event_raw_comparison.gif\n基线最低网络种子的首个孤立事件，直接显示原生二维活动、虚拟通道与 Supplementary Video 1 的 TA/TB 参考。两个人体示例并非与模型配对的验证样本；底层活动是每格每帧活跃 E 神经元数。**关注点**：原生传播是否与读出一致，时间顺序是否支持分布指标；待用户目视验收。\n' if (figs/'first_chronological_event_raw_comparison.gif').exists() else '### observation_repair_comparison.png\n同一批轨迹的观测修复前后比较，PDF 为同版导出。孤立事件与拥挤窗口分别报告，均为开发诊断。**关注点**：阈值敏感性与患者邻域覆盖，不能只看两个模式是否出现。\n')


if __name__=='__main__':main()

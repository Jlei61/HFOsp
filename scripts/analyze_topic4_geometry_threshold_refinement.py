"""Patient Fig2C alongside existing/new geometry pilots; full readout, no warp."""
import argparse,sys,json,csv,copy,warnings
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from PIL import Image
from scripts import run_topic4_geometry_threshold_refinement as run
from scripts.paper_figures import plot_topic4_core_extent_pilot as p
rt=run.rt
MAIN=Path('/home/honglab/leijiaxin/HFOsp')
META=MAIN/'results/paper-ready-figure/fig2/fig2_panelc_metadata.json'
CACHE=MAIN/'results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz'


def path_for(c,s):
    root=run.OLD if c.get('reference') else run.OUT
    return root/'duration_90000/units'/c['id']/str(s)/'workers/trajectory.json'


def patient_examples():
    meta=rt.read(META);result={}
    with np.load(CACHE) as z:
        for lab in ['TA','TB']:
            ex=meta['exemplar'][lab];part=z[lab+'_participant'].astype(bool)
            assert int(z[lab+'_event_pos'])==ex['event_pos'] and str(z[lab+'_block'])==ex['block']
            offset=min(c['time_within_event_sec'] for c in ex['fig1a_centroid_alignment']['centroids'] if part[c['channel_index']])*1000
            time=z[lab+'_envelope_time_from_first_centroid_ms']+offset;keep=(time>=-1e-6)&(time<250-1e-6)
            result[lab]=dict(mass=np.maximum(z[lab+'_envelope_robust_z'][:,keep],0),time=time[keep],mask=part,names=z['contact_order'].astype(str),event_id=ex['event_pos'],block=ex['block'])
    return result


def label(c):
    names={'baseline':'原半径','expand_A_2.5':'仅左核 2.5 mm','expand_B_2.5':'仅右核 2.5 mm','expand_AB_2.5':'双核 2.5 mm','expand_A_4':'仅左核 4 mm','expand_B_4':'仅右核 4 mm','expand_AB_4':'双核 4 mm',
    'weighted_base':'加权中心·原半径','weighted_AB25':'加权中心·双核 2.5 mm','xy_A_y_minus':'左核下移 0.75 mm','xy_A_y_plus':'左核上移 0.75 mm','xy_B_x_minus':'右核左移 0.75 mm','xy_B_x_plus':'右核右移 0.75 mm'}
    if c['id'] in names:return names[c['id']]
    name=c['id'].replace('AB25_','双核 2.5 mm·').replace('A4_','左核 4 mm·')
    for x,y in [('dose_preserved','保持阈值总量'),('mean_raise025','平均阈值升 0.25 mV'),('dispersion150','阈值标准差 ×1.5'),('state_extent','扩大 I 输入调制范围')]:name=name.replace(x,y)
    return name


def analyze(existing_only=False,native_review=False,observations_only=False):
    plan=rt.read(run.OUT/'plan.json');cs=[dict(c,reference=True) for c in plan['references']]+([] if existing_only else plan['candidates']);seeds=plan['training_seeds']
    out=run.OUT/('running_observations' if observations_only else 'existing_patient_review' if existing_only else 'analysis');figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    p.ANALYSIS_END_MS=90000;p.p.path_for=path_for
    ev=rt.load_evaluator(plan['parent_design']);pat=np.asarray(ev.fit);pl=np.asarray(ev.fit_labels);units={};rows=[];events=[]
    for c in cs:
        for s in seeds:
            unit=p.load_unit(c,s)
            if unit is None:continue
            units[(c['id'],s)]=unit;r,ar,ids=unit;names=ar['contact_names'];scl=np.char.startswith(names,'SCL');icl=np.char.startswith(names,'ICL');table=ar['centroid_ms'][ids]
            for m in [0,1]:
                mm=ar['event_mode'][ids]==m;t=table[mm];ref=pat[pl==m];mr=p.avg(p.ranks(t)) if len(t) else np.full(len(names),np.nan);pr=p.avg(p.ranks(ref));ok=np.isfinite(mr)&np.isfinite(pr)
                rows.append(dict(candidate=c['id'],label=label(c),parent_id=c.get('parent_id','baseline'),seed=s,mode=m,n=len(t),fraction=float(mm.mean()) if len(ids) else None,
                    no_SCL=float((~np.isfinite(t)[:,scl].any(1)).mean()) if len(t) else None,no_ICL=float((~np.isfinite(t)[:,icl].any(1)).mean()) if len(t) else None,
                    participation_MAE=float(abs(np.isfinite(t).mean(0)-np.isfinite(ref).mean(0)).mean()) if len(t) else None,rank_rho=float(spearmanr(mr[ok],pr[ok]).statistic) if ok.sum()>2 else None,physical_status=r['physical_status'],OOD_fraction=float(np.mean([r['events'][int(i)]['support']==-1 for i in ids[mm]])) if len(t) else None,patient_supported_fraction=float(np.mean([r['events'][int(i)]['support']==1 for i in ids[mm]])) if len(t) else None))
            for i in ids:
                lo,hi=r['events'][i]['window_ms'];env=ar['contact_envelope'][int(lo/2):int(hi/2)];cum=np.cumsum(env,axis=0);mass=cum[-1];mask=np.isfinite(ar['centroid_ms'][i])&(mass>0);q=np.stack([np.argmax(cum>=x*mass,axis=0)*2 for x in [.1,.5,.9]])
                events.append(dict(candidate=c['id'],seed=s,mode=int(ar['event_mode'][i]),event_index=int(i),time_ms=r['events'][i]['event_time_ms'],local_width_ms=float(np.median((q[2]-q[0])[mask])),recruitment_span_ms=float(np.ptp(q[0,mask])),centroid_span_ms=float(np.ptp(ar['centroid_ms'][i,mask])),n_contacts=int(mask.sum())))
    p.csv_write(out/'per_run_observations.csv',rows);p.csv_write(out/'per_event_observations.csv',events)
    p.csv_write(out/'run_event_counts.csv',[
        dict(candidate=cid,seed=seed,physical_status=result['physical_status'],actual_duration_ms=result['actual_duration_ms'],
             detected=result.get('n_detected'),observer_primary=len(arrays['primary_event_indices']),
             analysis_after_burnin_and_window=len(ids))
        for (cid,seed),(result,arrays,ids) in units.items()])
    rt.write(out/'observation_exclusions.json',[
        dict(candidate=cid,seed=seed,detected=result.get('n_detected'),observer_primary=len(arrays['primary_event_indices']),
             reason_counts=dict(Counter(reason for event in result['events'] for reason in event['exclusion_reasons'])))
        for (cid,seed),(result,arrays,ids) in units.items()])
    if observations_only:
        # Completion order does not nominate a candidate or change the fixed batch.
        metrics=['participation_MAE','rank_rho','no_SCL','fraction','OOD_fraction']
        paired=[]
        lookup={(r['candidate'],r['seed'],r['mode']):r for r in rows}
        for c in cs:
            parent=c.get('parent_id','baseline')
            if c['id']==parent:continue
            for mode in [0,1]:
                keys=[(cid,s,mode) for cid in [c['id'],parent] for s in seeds]
                if any(key not in lookup or lookup[key]['physical_status']!='COMPLETE_NO_RUNAWAY' for key in keys):continue
                for metric in metrics:
                    if any(lookup[key][metric] is None for key in keys):continue
                    values=[lookup[(c['id'],s,mode)][metric] for s in seeds]
                    reference=[lookup[(parent,s,mode)][metric] for s in seeds]
                    paired.append(dict(candidate=c['id'],label=label(c),parent=parent,mode='TA' if mode==1 else 'TB',metric=metric,
                        seeds=seeds,values=values,parent_values=reference,deltas=[a-b for a,b in zip(values,reference)],
                        mean_delta=float(np.mean(values)-np.mean(reference))))
        rt.write(out/'paired_responses.json',paired)
        rt.write(out/'status.json',dict(status='PARTIAL_OBSERVATIONS_ONLY',readable_units=len(units),expected_units=len(cs)*len(seeds),
            complete_conditions=[c['id'] for c in cs if all((c['id'],s) in units for s in seeds)],
            missing_units=[dict(candidate=c['id'],seed=s) for c in cs for s in seeds if (c['id'],s) not in units],
            nomination=None,scientific_acceptance=False))
        (out/'README.md').write_text('# 运行中的配对观测快照\n\n仅描述已经可读的完整物理输出，不按完成速度提名，也不替换整轮正式分析。每条噪声是一个配对重演；事件是运行内部观测，两个噪声方向相同不等于统计显著或跨拓扑泛化。`paired_responses.json` 仅包含候选与直接出发点均有全部两条噪声的项目，保留两条差值。差值为候选减直接出发点；参与误差、缺 SCL 和邻域外比例通常越低越好，平均 rank 相关越高越好；模式比例没有统一越大越好的方向。TA=M1、TB=M0。\n')
        print(json.dumps(dict(status='PARTIAL_OBSERVATIONS_ONLY',output=str(out),n_runs=len(units))),flush=True)
        return
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(figdir/f'{name}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    # Candidate selection is explicit development display, not automatic acceptance.
    candidate_summary=[]
    for c in cs:
        rr=[r for r in rows if r['candidate']==c['id']];means=[]
        for m in [0,1]:
            r0=[r for r in rr if r['mode']==m]
            if len(r0)!=len(seeds) or any(r['participation_MAE'] is None or r['rank_rho'] is None or r['physical_status']!='COMPLETE_NO_RUNAWAY' for r in r0):break
            means.append({k:float(np.mean([r[k] for r in r0])) for k in ['participation_MAE','rank_rho','no_SCL','fraction']})
        if len(means)==2:candidate_summary.append(dict(candidate=c,worst_participation=max(x['participation_MAE'] for x in means),worst_rank=min(x['rank_rho'] for x in means),means=means))
    rt.write(out/'candidate_observations.json',candidate_summary)
    if not candidate_summary:raise RuntimeError('no readable two-label candidate for display; observation support missing')
    aa=min(candidate_summary,key=lambda x:x['worst_participation']);bb=max(candidate_summary,key=lambda x:x['worst_rank'])
    if bb['candidate']['id']==aa['candidate']['id'] and len(candidate_summary)>1:bb=max([x for x in candidate_summary if x!=aa],key=lambda x:x['worst_rank'])
    chosen=[aa['candidate'],bb['candidate']];rt.write(out/'display_selection.json',dict(candidates=chosen,rule='first: smallest worse-mode participation MAE; second: largest worse-mode mean-rank correlation among different conditions; paired-run means; diagnostic display only, not accepted best propagation',label_mapping={'TA':1,'TB':0},mapping_source=str(MAIN/'.worktrees/topic4-substrate-autapse-fix/scripts/paper_figures/plot_topic4_fig2c_contact_envelope_comparison.py')))
    pats=patient_examples();displaynames=[f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)];data={};selections=[]
    for lab,m in [('TA',1),('TB',0)]:
        d=pats[lab];order=[list(d['names']).index(n) for n in displaynames];data[(lab,0)]=dict(mass=d['mass'][order],mask=d['mask'][order],id=d['event_id'])
        for col,c in enumerate(chosen,1):
            r,ar,ids=units[(c['id'],seeds[0])];ix=ids[ar['event_mode'][ids]==m]
            if not len(ix):raise RuntimeError('first paired run has no display mode')
            phi=ar['event_phi'][ix];i=int(ix[np.argmin(((phi-phi.mean(0))**2).sum(1))]);lo,hi=r['events'][i]['window_ms'];order=[list(ar['contact_names']).index(n) for n in displaynames]
            data[(lab,col)]=dict(mass=ar['contact_envelope'][int(lo/2):int(hi/2)][:,order].T,mask=np.isfinite(ar['centroid_ms'][i])[order],id=i)
            selections.append(dict(candidate=c['id'],seed=seeds[0],mode=m,event_index=i,window_ms=[lo,hi]))
    rt.write(out/'representative_events.json',selections)
    for scale in ['event','contact']:
        fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for row,lab in enumerate(['TA','TB']):
            for col in range(3):
                d=data[(lab,col)];mass=d['mass'];den=max(float(mass.max()),1e-20) if scale=='event' else np.maximum(mass.max(1,keepdims=True),1e-20)
                ax=axs[row,col];im=ax.imshow(mass/den,aspect='auto',extent=[0,250,14.5,-.5],cmap='magma',vmin=0,vmax=1,interpolation='nearest');ax.axhline(3.5,color='#66bbbb',lw=.7)
                ax.set(title=('患者 Fig2C' if col==0 else label(chosen[col-1]))+f' · {lab} · 事件 {d["id"]}',yticks=range(15),yticklabels=[n+(' *' if not ok else '') for n,ok in zip(displaynames,d['mask'])],xlabel='原始事件窗口时间 (ms)');ax.tick_params(axis='y',labelsize=8)
        fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.65,label='包络 / '+('整事件峰值' if scale=='event' else '各接触点峰值'))
        fig.suptitle('患者两类事件与两个诊断候选；相同触点顺序及 250 ms 时间尺度，不拉伸、不按 rank 排列通道\n左：80–250 Hz HFO 包络；右：完整神经元发放密度包络；* 表示未参与，仍保留信号')
        save(fig,'patient_vs_two_models_'+scale+'_scale')
    # Paired response panels, natural units. Every candidate kept, labels are explanatory.
    metrics=[('participation_MAE','参与概率误差 ↓'),('rank_rho','平均质心顺序相关 ↑'),('no_SCL','整条 SCL 缺失比例 ↓'),('fraction','模式占比'),('OOD_fraction','患者特征邻域外比例 ↓')]
    for section,subset in [('references',[c for c in cs if c.get('reference')]),('new_parameters',[c for c in cs if not c.get('reference')])]:
        if not subset:continue
        fig,axs=plt.subplots(2,5,figsize=(max(18,len(subset)*1.3),8),layout='constrained')
        for m in [0,1]:
            for ax,(key,title) in zip(axs[m],metrics):
                for seed,color in zip(seeds,['#277da8','#e47832']):
                    vals=[next((r[key] for r in rows if r['candidate']==c['id'] and r['seed']==seed and r['mode']==m),None) for c in subset];ax.plot(range(len(subset)),[np.nan if x is None else x for x in vals],'o-',label=f'噪声 {seed}',color=color)
                ax.set(title=f'{"TA" if m==1 else "TB"}（M{m}）· {title}',xticks=range(len(subset)),xticklabels=[label(c) for c in subset]);ax.tick_params(axis='x',rotation=75,labelsize=8);ax.grid(alpha=.15)
                if key=='no_SCL':ax.axhline((~np.isfinite(pat[pl==m])[:,np.char.startswith(names,'SCL')].any(1)).mean(),color='black',ls='--')
                if key=='fraction':ax.axhline((pl==m).mean(),color='black',ls='--')
        axs[0,0].legend();save(fig,section+'_parameter_observations')
    # Match each intervention to its own parent using paired random seeds.
    # I support changes also change the per-step coupling-stream dimension.
    deltas=[];indices={(r['candidate'],r['seed'],r['mode']):r for r in rows}
    for c in cs:
        if c.get('reference') or not c.get('parent_id'):continue
        for seed in seeds:
            for m in [0,1]:
                current=indices.get((c['id'],seed,m));parent=indices.get((c['parent_id'],seed,m))
                if current is None or parent is None:continue
                d=dict(candidate=c['id'],parent=c['parent_id'],label=label(c),seed=seed,mode=m,n=current['n'],parent_n=parent['n'])
                for k in ['participation_MAE','rank_rho','no_SCL','fraction','OOD_fraction']:
                    d[k+'_delta']=float(current[k]-parent[k]) if current[k] is not None and parent[k] is not None else None
                deltas.append(d)
    if deltas:
        p.csv_write(out/'paired_parameter_changes.csv',deltas);subset=[c for c in cs if not c.get('reference')]
        fig,axs=plt.subplots(2,5,figsize=(21,9),layout='constrained')
        for m in [0,1]:
            for ax,(key,title) in zip(axs[m],metrics):
                ax.axhline(0,color='black',ls='--',lw=1)
                for seed,color in zip(seeds,['#277da8','#e47832']):
                    vals=[next((d[key+'_delta'] for d in deltas if d['candidate']==c['id'] and d['seed']==seed and d['mode']==m),None) for c in subset]
                    ax.plot(range(len(subset)),[np.nan if x is None else x for x in vals],'o-',label=f'噪声 {seed}',color=color)
                ax.set(title=f'{"TA" if m==1 else "TB"} · '+title+'的变化',xticks=range(len(subset)),xticklabels=[label(c) for c in subset]);ax.tick_params(axis='x',rotation=80,labelsize=8)
        axs[0,0].legend();fig.suptitle('每个条件减去其直接出发点；同一网络、配对随机种子；抑制投影扩展同时改变局部调制输入\n零线表示该观测不变；参与误差、缺杆、OOD 为负通常改善，顺序相关为正通常改善；占比须对照患者')
        save(fig,'paired_parameter_changes')
    # Distribution of physical timing observables; raw readout, no patient-HFO equivalence.
    fig,axs=plt.subplots(2,3,figsize=(max(15,len(cs)*.8),9),layout='constrained')
    timing_summary=[]
    for m in [0,1]:
        for ax,(key,title) in zip(axs[m],[('local_width_ms','局部 q10–q90 宽度'),('recruitment_span_ms','跨触点 q10 招募跨度'),('centroid_span_ms','跨触点质心跨度')]):
            for j,c in enumerate(cs):
                for u,seed in enumerate(seeds):
                    vals=[e[key] for e in events if e['candidate']==c['id'] and e['seed']==seed and e['mode']==m]
                    if not vals:continue
                    q=np.percentile(vals,[5,50,95]);xx=j+(u-.5)*.2;ax.plot([xx,xx],[q[0],q[2]],color=['#277da8','#e47832'][u],alpha=.6);ax.scatter(xx,q[1],color=['#277da8','#e47832'][u],s=15)
                    timing_summary.append(dict(candidate=c['id'],seed=seed,mode=m,observable=key,n=len(vals),mean=float(np.mean(vals)),variance=float(np.var(vals)),q05=float(q[0]),median=float(q[1]),q95=float(q[2])))
            ax.set(title=f'{"TA" if m==1 else "TB"} · {title}',ylabel='ms',xticks=range(len(cs)),xticklabels=[label(c) for c in cs]);ax.tick_params(axis='x',rotation=80,labelsize=7)
    fig.suptitle('每条噪声：点为事件中位数，线为事件 5–95% 范围；不是置信区间\n来自完整模型发放密度包络；局部宽度不直接等同患者 HFO 宽度')
    save(fig,'parameter_timing_distributions');p.csv_write(out/'timing_distribution_summary.csv',timing_summary)
    # Explicit per-contact patient and model distributions, with run spread.
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained');order=[list(names).index(n) for n in displaynames]
    for m in [0,1]:
        for row in [0,1]:
            ref=pat[pl==m];rv=np.isfinite(ref).mean(0) if row==0 else p.avg(p.ranks(ref));ax=axs[row,m];ax.plot(rv[order],'ko-',label='患者 FIT',ms=4)
            for c,color in zip(chosen,['#277da8','#e47832']):
                vals=[]
                for seed in seeds:
                    _,ar,ids=units[(c['id'],seed)];t=ar['centroid_ms'][ids][ar['event_mode'][ids]==m];vals.append(np.isfinite(t).mean(0) if row==0 else p.avg(p.ranks(t)))
                vals=np.array(vals)[:,order];ax.plot(p.avg(vals),'o-',label=label(c),color=color);ax.fill_between(range(15),np.nanmin(vals,0),np.nanmax(vals,0),color=color,alpha=.15)
            ax.set(title=f'{"TA" if m==1 else "TB"} · '+['触点参与概率','参与后的平均质心 rank'][row],xticks=range(15),xticklabels=displaynames,ylim=(-.03,1.03));ax.tick_params(axis='x',rotation=60)
    axs[0,0].legend(fontsize=9);fig.suptitle('患者与模型：曲线为运行等权均值，阴影为两条噪声范围，非置信区间');save(fig,'patient_model_contact_profiles')
    # Full patient-vs-model event cloud; one event is not a distribution.
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#aaa');fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for row,c in enumerate([None]+chosen):
        for col,(lab,m) in enumerate([('TA',1),('TB',0)]):
            if c is None:t=pat[pl==m];t=t[np.linspace(0,len(t)-1,100).astype(int)];name='患者 FIT';bounds=[]
            else:
                ts=[ar['centroid_ms'][ix][ar['event_mode'][ix]==m] for seed in seeds for _,ar,ix in [units[(c['id'],seed)]]];t=np.concatenate(ts);bounds=np.cumsum([len(x) for x in ts])[:-1];name=label(c)
            ax=axs[row,col];ax.imshow(p.ranks(t)[:,order].T,aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
            for b in bounds:ax.axvline(b-.5,color='white',lw=1)
            ax.set(title=f'{name} · {lab} · {len(t)} 事件',yticks=range(15),yticklabels=displaynames,xlabel='事件');ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('紫色质心早，黄色晚，灰色未参与；患者均匀取 100 例，模型全部事件；白线为噪声边界');save(fig,'patient_model_all_events')
    checks=[]
    for path in figdir.glob('*.png'):
        with Image.open(path) as im:im.load();checks.append(dict(file=path.name,size=list(im.size)))
    rt.write(out/'checks.json',dict(figures=checks,patient_meta_sha=rt.sha(META),patient_cache_sha=rt.sha(CACHE),model_events=selections,common_window_ms=250,no_time_warp=True,full_contact_readout=True,manual_acceptance='PENDING'))
    (figdir/'README.md').write_text('\n\n'.join('### '+x['file']+'\n'+('左列严格使用 Fig2C 锁定的 TA 6344、TB 937 原始包络，右侧来自两个诊断候选自身模式均值附近的事件。所有触点和 250 ms 窗口固定；完整保留未参与信号，患者 HFO 与模型发放密度幅度不可直接比较。' if 'scale' in x['file'] else '图中保留每条噪声或全部事件，并与冻结患者 FIT 参考比较。具体观测及单位见坐标标签；均值、类别标签和事件数量不能单独代表完整传播恢复。')+'\n**关注点**：SCL/ICL 的参与和时序是否同时恢复，而非仅有相反斜率。' for x in checks)+'\n')
    (out/'scientific_report.md').write_text('# 患者—模型传播审阅\n\n两个展示候选分别根据两模式中较差的参与误差、较差的平均顺序相关选取，规则见 display_selection.json；不是已通过视觉验证的最佳模型。患者两例是 Fig2C 既定示例，不代表自然频率或完整方差。模型代表事件取自身类均值附近者；另附全部事件图。\n\n原冻结 L_off 不作为本批次成功或自动晋级标准。完整结果按 per_run_observations.csv 和 per_event_observations.csv 的参与、模式比例、局部宽度与招募时差分别解释。全部条件为单拓扑、两条噪声，配对响应不是患者泛化。\n\n配对表示同一网络与相同种子。普通几何/阈值条件保留原外部驱动和固定 I 投影；抑制投影扩展条件改变 I 目标集合、加载和调制均匀数流长度，不能视为每个细胞始终接受相同调制随机数。其差异包括作用域和局部调制输入实现变化，见上级 paired_input_semantics.md；新完整轨迹仍须依据输入哈希复核。\n\n图件等待用户目视审阅，未启动后续 EE/II 或输入干预。\n')
    if not existing_only:
        from scripts.plot_topic4_all_condition_time_review import render
        print(json.dumps(render(units,cs,seeds,pats,out,label),ensure_ascii=False),flush=True)
    if not existing_only or native_review:
        # Reuse canonical montage/native multi-event producer only for displayed candidates.
        nativeout=out/'native_review';nativeout.mkdir(exist_ok=True);nativefig=nativeout/'figures';nativefig.mkdir(exist_ok=True)
        local=copy.deepcopy(plan);local['candidates']=chosen;local['observation_seeds']=list(seeds);rt.write(nativeout/'plan.json',local);rt.write(nativeout/'all_candidates.json',chosen)
        from scripts import analyze_topic4_core_extent_long as olda
        def scoring(c,seeds0):
            obj=rt.load_objective(plan['parent_design']);uu=[]
            for s in seeds0:
                u=units.get((c['id'],s));q=obj.score_network(u[1]['centroid_ms'][u[2]]) if u is not None else dict(loss_off=None,status='MISSING');uu.append(dict(seed=s,**q))
            return dict(candidate=c,units=uu,loss_off=float(np.mean([u['loss_off'] for u in uu])) if all(u['loss_off'] is not None for u in uu) else None)
        p.OUT=nativeout;p.F=nativefig;p.SHOW_ALL_GEOMETRIES=True;p.RUN_ROLE='paired_development';p.MODE_NAMES={0:'TB (M0)',1:'TA (M1)'};p.MODE_ORDER=[1,0];p.p.score_candidate=scoring;sys.argv=['plot'];p.main()
    print(json.dumps(dict(status='REVIEW_READY',output=str(out),display_candidates=[c['id'] for c in chosen],n_runs=len(units)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--existing-only',action='store_true');ap.add_argument('--native-review',action='store_true');ap.add_argument('--observations-only',action='store_true');args=ap.parse_args();analyze(args.existing_only,args.native_review,args.observations_only)

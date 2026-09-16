"""Overnight recovery review. Frozen primary and all-detection diagnostics stay separate.

Labels organize comparisons; they are not evidence of restored patient pathways.
Core cumulative-mass times describe timing, not causal origins.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import analyze_topic4_core_connectivity_search as an
from scripts import run_topic4_propagation_recovery_night as night
rt=an.rt

LABELS={
 'endpoint__baseline':'端点几何原位',
 'up4p5__baseline':'左核上移 4.5 mm',
 'near_upper__baseline':'左核靠近上部 SCL',
 'up4p5__EE_core_to_out_scale_1.25':'上移左核；离核 EE 权重 ×1.25',
 'near_upper__EE_kernel_perp_scale_1.5':'靠近 SCL；EE 横向范围 ×1.5',
 'recovery_up_out_transverse':'离核输出增强＋横向范围 ×1.25',
 'recovery_up_out_extent_depth':'离核输出增强＋左核扩大、减弱降阈值',
 'recovery_up_out_EI':'离核输出增强＋核内 E→I ×0.85',
 'recovery_up_out_IE':'离核输出增强＋核内 I→E ×1.2',
 'recovery_upper_wide_bias_B':'横向范围扩大＋相对增强右核易激性',
 'recovery_upper_wide_bias_A':'横向范围扩大＋相对增强左核易激性',
 'recovery_upper_wide_recurrence':'横向范围扩大＋核内 E→E ×0.75',
 'recovery_upper_output_transverse':'离核输出 ×1.25＋横向范围 ×1.25',
}

def display(c):return LABELS.get(c.get('base_id',c['id']),c.get('display_name',c['id']))

def core_timing(r,a,i):
    lo,hi=r['events'][i]['window_ms'];tt=a['trace_time_ms'];sel=(tt>=lo)&(tt<hi)
    qs=[];masses=[]
    for g in ['coreAE','coreBE']:
        x=a['trace_'+g+'_spikes'][sel].astype(float);mass=x.sum();masses.append(float(mass))
        qs.append(np.interp(np.array([.1,.5,.9])*mass,np.cumsum(x),tt[sel]) if mass>0 else np.full(3,np.nan))
    return dict(B_minus_A_t10_ms=float(qs[1][0]-qs[0][0]),B_minus_A_t50_ms=float(qs[1][1]-qs[0][1]),
        coreA_width_ms=float(qs[0][2]-qs[0][0]),coreB_width_ms=float(qs[1][2]-qs[1][0]),
        coreA_mass=masses[0],coreB_mass=masses[1])

def stage_cases(phase):
    plan=rt.read(night.OUT/'plan.json');old=rt.read(an.run.OUT/'plan.json')
    spec=rt.read(night.OUT/f'{phase}_units.json')
    lookup={c['id']:c for c in old['candidates']}
    lookup.update({c['id']:c for c in plan['wave1']['candidates']})
    cases=[]
    for cid,t,s in spec['units']:
        key=f'{cid}@{t}'
        if any(c['id']==key for c in cases):continue
        if cid not in lookup:lookup[cid]=rt.read(an.run.OUT/'candidates'/f'{cid}.json')
        cases.append(dict(lookup[cid],id=key,base_id=cid,topology=int(t),output_stage=spec['stage']))
    for cid in ['endpoint__baseline','up4p5__baseline','near_upper__baseline']+list(dict.fromkeys(c['parent_id'] for c in cases if 'parent_id' in c)):
        if any(c['base_id']==cid and c['topology']==2511 for c in cases):continue
        if cid not in lookup:lookup[cid]=rt.read(an.run.OUT/'candidates'/f'{cid}.json')
        c=lookup[cid]
        cases.append(dict(c,id=f'{cid}@2511',base_id=cid,topology=2511,output_stage=c.get('stage','screen')))
    return old,plan,spec,cases

def main(phase,figures=False):
    old,plan,spec,cases=stage_cases(phase);out=night.OUT/('analysis_'+phase);out.mkdir(exist_ok=True)
    F=out/'figures';F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    parent=rt.read(an.run.PARENT);ev=rt.load_evaluator(parent);obj=rt.load_objective(parent)
    patient=np.asarray(ev.fit);plabels=np.asarray(ev.fit_labels)
    maskref=an.jm.MaskReference(np.isfinite(patient))
    cal=rt.read(an.run.OUT/'analysis/mask_score_calibration.json');maskref.a_mask=cal['a_mask']
    names=np.asarray(rt.load_observation_contract(rt.read(an.run.PARENT))['contact_names'])
    rows=[];event_rows=[];counts=[];missing=[];units={};completed_cases=[];native_selections=[];score_records=[]
    for c in cases:
      seeds=sorted({int(s) for cid,t,s in spec['units'] if cid==c['base_id'] and int(t)==c['topology']}) or old['seeds']
      available=False
      for seed in seeds:
        path=an.run.result_path(c['output_stage'],c['base_id'],c['topology'],seed)
        unit=an.load_unit(path,old['analysis']['burnin_ms'])
        if unit is None:
            missing.append(dict(candidate=c['id'],seed=seed,path=str(path)));continue
        available=True;r,a,primary=unit;allids=an.all_detected_ids(r,old['analysis']['burnin_ms'])
        root=dict(candidate=c['id'],base_id=c['base_id'],display_name=display(c),topology_seed=c['topology'],seed=seed)
        counts.append(dict(root,primary=len(primary),physical_status=r['physical_status'],duration_ms=r['actual_duration_ms'],**an.detection_layer(r,a,old['analysis']['burnin_ms']),**an.core_activity(a,old['analysis']['burnin_ms'],r['actual_duration_ms'])))
        times={}
        for i in allids:
            z=dict(root,event=int(i),event_time_ms=float(a['event_time_ms'][i]),mode='TA' if a['event_mode'][i]==1 else 'TB',primary=bool(i in primary),**core_timing(r,a,i))
            et=an.event_timing(r,a,i,names)
            if et:z.update(et)
            nd=an.native_diag(r,a,i,len(a['group_coreAE'])+len(a['group_coreBE']),len(a['group_surroundE']))
            if nd:z.update(nd)
            times[int(i)]=z;event_rows.append(z)
        for layer,ids in [('primary',primary),('all_detected',allids)]:
          for mode,label in [('ALL',None),('TA',1),('TB',0)]:
            selected=ids if label is None else ids[a['event_mode'][ids]==label]
            x=a['centroid_ms'][selected];ref=patient if label is None else patient[plabels==label]
            z=dict(root,mode=mode,layer=layer,**an.measures(x,ref,names))
            pp=an.pair_table(ref);errs=[];supported=[]
            for (i,j),(pn,p) in pp.items():
                ok=np.isfinite(x[:,i])&np.isfinite(x[:,j]);d=x[ok,j]-x[ok,i]
                if len(d) and p is not None:errs.append(abs(np.mean((d>0)+.5*(d==0))-p));supported.append(len(d))
            z.update(pair_order_probability_mae=float(np.mean(errs)) if errs else None,pair_order_supported_pairs=len(errs),pair_support_median=float(np.median(supported)) if supported else None)
            selected_times=[times[int(i)] for i in selected]
            for key in ['local_width_ms','recruitment_span_ms','B_minus_A_t10_ms','B_minus_A_t50_ms','first_10pct_mass_core_share','first_10pct_core_to_outside_per_neuron_density','peak_largest_component_fraction']:
                values=np.asarray([t.get(key,np.nan) for t in selected_times],float);values=values[np.isfinite(values)]
                for q,suffix in [(.05,'q05'),(.5,'median'),(.95,'q95')]:z[key+'_'+suffix]=float(np.quantile(values,q)) if len(values) else None
            lag=np.asarray([t['B_minus_A_t10_ms'] for t in selected_times]);lag=lag[np.isfinite(lag)]
            z['coreA_t10_earlier_fraction']=float((lag>0).mean()) if len(lag) else None
            if mode=='ALL' and layer=='primary':
                sc=obj.score_network(x) if len(x) else {};ms=an.jm.score_times(x,maskref) if len(x) else {}
                z.update(L_off=sc.get('loss_off'),L_D16=sc.get('loss_D16'),L_off_A=sc.get('loss_off_A_component'),L_off_B=sc.get('loss_off_B_subtraction'),D_mask_off=ms.get('D_mask_off'),L_search=an.jm.combined_search_loss(sc.get('loss_off'),ms.get('D_mask_off'),maskref.a_mask))
                score_records.append(dict(root,primary_n=len(x),distribution_score=sc,joint_participation_score=ms,combined_search_loss=z['L_search']))
            rows.append(z)
        if figures:
            an.continuous_raster(c,seed,r,a,primary,F,display(c),names)
            # Separate all-detection figure population; keep the original primary score untouched.
            if seed==seeds[0]:
                native_selections.extend(an.native_gif(c,seed,r,a,allids,F,display(c)+'；全部检测的开发诊断'))
            units[(c['id'],seed)]=(r,a,allids)
      if available:completed_cases.append(c)
    an.writecsv(out/'run_observations.csv',rows);an.writecsv(out/'event_timing.csv',event_rows);an.writecsv(out/'counts.csv',counts)
    rt.write(out/'training_scores.json',dict(scores=score_records,primary_rule_unchanged=True,negative_off_scores_not_clipped=True))
    rt.write(out/'review_state.json',dict(phase=phase,complete_runs=len(counts),missing=missing,primary_population='unchanged frozen patient-matched isolated-window rule',
        all_detected_population='developmental diagnostic only; not the frozen patient-matched training population',
        interpretation='Labels organize comparison, cumulative core mass timing is not causal origin, no propagation acceptance implied',
        parameter_units='One candidate x topology x dynamics replay is a unit; within-replay events are not independent networks'))
    if figures and units:
        # Each topology has its own comparison page to avoid combining network identities.
        for topo in sorted({c['topology'] for c in completed_cases}):
            cc=[c for c in completed_cases if c['topology']==topo];ss=sorted({s for cid,s in units if cid in {c['id'] for c in cc}})
            an.render(units,cc,ss,an.patient_examples(),out/f'black_comparison_{topo}',display,event_population='全部检测的开发诊断；原 primary 评分未变')
            an.render(units,cc,ss,an.patient_examples(),out/f'black_comparison_{topo}',display,event_population='全部检测的开发诊断；原 primary 评分未变',support_only=True)
        plot_timing(rows,completed_cases,F)
        rt.write(out/'native_selection.json',dict(event_population='all_detected developmental only',selection=native_selections))
        entries=[]
        for f in sorted(F.iterdir()):
            if f.suffix not in ('.png','.pdf','.gif'):continue
            if '_native_' in f.name:
                body='完整原生2 ms活动，每类取最早三个可读检测事件，保留核外活动及真实电极布局；所有帧在该运行内共用色标。该图消费全部检测的开发诊断集合，未替换原primary评分。**关注点**：同时亮区、传播连续性及核间时序；一个分类标签不证明一条因果路径。'
            elif f.name.endswith('_continuous.png'):
                body='本运行完整连续接触点读出，保留全部15触点和真实时间；标题中的合格事件数仍指原primary集合。**关注点**：被事件筛选遗漏的活动、节律与持续活动，不能只看入选事件。'
            else:
                body='按参数条件、拓扑与噪声分别显示SCL/ICL参与和两核10%累计活动时间，TA红色、TB蓝色，圆/方为两条噪声；点线为事件中位数与5–95%范围。全部检测仅作开发诊断。**关注点**：招募改善是否伴随两类事件保留不同的时间过程；累计质量时间不是因果起源。'
            entries.append(f'### {f.name}\n\n{body}')
        (F/'README.md').write_text('\n\n'.join(entries)+'\n')
    print(json.dumps(dict(output=str(out),complete=len(counts),missing=len(missing)),ensure_ascii=False),flush=True)

def plot_timing(rows,cases,F):
    fig,axes=plt.subplots(1,3,figsize=(15,8),layout='constrained',sharey=True)
    for ci,c in enumerate(cases):
      seeds=sorted({r['seed'] for r in rows if r['candidate']==c['id']})
      for si,seed in enumerate(seeds):
       for mode,shift in [('TA',-.16),('TB',.16)]:
        r=next((r for r in rows if r['candidate']==c['id'] and r['seed']==seed and r['mode']==mode and r['layer']=='all_detected'),None)
        if r is None or not r['n']:continue
        y=ci+shift+(si-(len(seeds)-1)/2)*.13;color=an.MODE_COLOR[mode];marker='o' if si==0 else 's'
        for ax,key in zip(axes[:2],['SCL_upper_participation','ICL_contact_participation']):ax.plot(r[key],y,marker=marker,ms=4,color=color)
        m=r['B_minus_A_t10_ms_median'];lo=r['B_minus_A_t10_ms_q05'];hi=r['B_minus_A_t10_ms_q95']
        if m is not None:
            axes[2].plot([lo,hi],[y,y],color=color,lw=1);axes[2].plot(m,y,marker=marker,ms=4,color=color)
            axes[2].text(hi+2,y,f"{r['n']}",fontsize=6,color=color,va='center')
    for ax in axes[:2]:ax.set_xlim(-.03,1.03)
    axes[0].set(yticks=range(len(cases)),yticklabels=[display(c) for c in cases],xlabel='SCL9/8 参与概率的平均')
    axes[1].set_xlabel('ICL 各触点参与概率的平均');axes[2].set_xlabel('右核 − 左核的 10% 累计活动时间 (ms)')
    axes[2].axvline(0,c='gray',lw=.8);axes[0].invert_yaxis()
    fig.suptitle('招募与两核时间关系分别检查｜红 TA，蓝 TB；圆/方为两条噪声\n全部检测的开发诊断；横线为事件 5–95% 范围，数字为实际事件数；正值为左核较早，并非因果起源')
    for ext in ['png','pdf']:fig.savefig(F/f'participation_and_core_timing.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',default='wave1');ap.add_argument('--figures',action='store_true');args=ap.parse_args();main(args.phase,args.figures)

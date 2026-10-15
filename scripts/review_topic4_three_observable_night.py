"""Bounded overnight status, scientific summaries and compact paired figures.

Reads immutable simulations. Does not change the objective, propose physics,
mark native review passed, or infer acceptance from score alone.
"""
from pathlib import Path
import argparse,datetime,fcntl,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import control_topic4_three_observable_bo as ctl
a=ctl.analysis;run=ctl.run;rt=run.rt;OUT=run.OUT;NIGHT=OUT/'overnight_20260914'

def snapshot():
    import psutil
    rows=a.records();physical=[]
    for p in OUT.glob('*/units/*/*/workers/trajectory.json'):
        r=rt.read(p);physical.append(dict(stage=p.parents[4].name,candidate=p.parents[2].name,topology=r['job']['topology_seed'],noise=r['job']['dynamics_seed'],status=r['status'],physical_status=r['physical_status'],raw_primary=r['n_primary'],source=str(p)))
    stages={}
    for stage in sorted({r['stage'] for r in physical}):
        q=[r for r in physical if r['stage']==stage];sc=[r for r in rows if r['stage']==stage]
        stages[stage]=dict(complete=sum(r['status']=='COMPLETE' for r in q),runaway=sum(r['physical_status']=='RUNAWAY' for r in q),scored=len(sc),eligible_events=sum(r['N'] for r in sc))
    processes=[]
    for p in OUT.glob('*_process.json'):
        r=rt.read(p)
        try:
            q=psutil.Process(r['pid']);live=q.is_running() and q.status()!=psutil.STATUS_ZOMBIE
            processes.append(dict(role=p.stem,pid=r['pid'],alive=live,cmd=q.cmdline() if live else []))
        except psutil.NoSuchProcess:processes.append(dict(role=p.stem,pid=r['pid'],alive=False))
    state=rt.read(OUT/'status.json');active=[]
    for r in state.get('active',[]):
        c,t,n=r['unit'];p=run.unit_path(state['stage'],c,t,n).with_name('trajectory.progress.json')
        try:
            q=psutil.Process(r['pid'])
            if q.is_running():active.append(dict(**r,progress=rt.read(p) if p.exists() else None))
        except psutil.NoSuchProcess:pass
    failures=[dict(path=str(p),record=rt.read(p)) for p in list((OUT/'analysis').glob('*failure.json'))+list(OUT.glob('*failure.json'))]
    window=rt.read(NIGHT/'window.json');now=time.time()
    payload=dict(time=datetime.datetime.now().astimezone().isoformat(),unix=now,stages=stages,
        physical_complete=len(physical),new_physical_complete=len(physical)-len(rt.read(OUT/'plan.json')['reused_units']),
        active=active,processes=processes,failures=failures,available_GiB=psutil.virtual_memory().available/2**30,
        optimizer=rt.read(OUT/'optimizer_status.json'),native_review_pending=(OUT/'g3_ready_for_native_review.json').exists() and not (OUT/'analysis/g3_agent_native_review.json').exists(),
        window_end_reached=now>=window['latest_review_unix'],remaining_hours=max(0,(window['latest_review_unix']-now)/3600))
    rt.write(NIGHT/'status.json',payload)
    with (NIGHT/'status_history.jsonl').open('a') as f:f.write(json.dumps(payload,ensure_ascii=False)+'\n')
    return payload

def report():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from src.topic4_pdf_font_guard import install
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    state=snapshot();rows=a.records();conditions=ctl.training_data();good=[r for r in conditions if r['scorable']];good.sort(key=lambda r:r['J'])
    from scripts.decompose_topic4_three_observable_scores import decompose
    decomposition=decompose()
    from scripts import reference_topic4_three_observable_blocks as blockref
    blockref.figure(blockref.update())
    plan=rt.read(OUT/'plan.json');ref=next(r for r in conditions if r['candidate']==plan['reference_id']);norms=rt.read(a.A/'objective_frozen.json')['scales']
    ev,names,_=a.patient();from scripts.report_topic4_three_observable_raw import observables
    patient={m:observables(ev.fit[ev.fit_labels==k],names)[0] for m,k in [('TA',1),('TB',0)]}
    summary=[]
    for r in conditions:
        rr=sorted([x for x in rows if x['candidate']==r['candidate'] and x['stage'] in ['initial','adaptive']],key=lambda x:x['noise'])
        s=dict(r,label=a.point_label(r['candidate']),runs=[])
        for x in rr:
            s['runs'].append(dict(noise=x['noise'],N=x['N'],mode_counts=x['mode_counts'],J=x['J'],components=x['components'],
                A_scaled_mean=float(np.mean([v['A']/norms[g] for g,v in x['groups'].items()])) if x['groups'] else None,
                B_scaled_mean=float(np.mean([v['B']/norms[g] for g,v in x['groups'].items()])) if x['groups'] else None,
                raw=x['raw']))
        summary.append(s)
    rt.write(NIGHT/'condition_summary.json',dict(patient=patient,conditions=summary,statistical_unit='one fixed condition-specific graph and dynamics realization; equal run weights; conditions share a topology seed but angle changes rebuild EE edges and delays',physical_state=state))
    F=NIGHT/'figures';F.mkdir(exist_ok=True)
    chosen=[ref]+[r for r in good if r['candidate']!=ref['candidate']][:3];legend=['固定参考']+[a.plot_label(r['candidate']) for r in chosen[1:]]
    def average_parts(cid):
        selected=[r for r in decomposition if r['candidate']==cid and r['topology']==2511 and r['noise'] in [847401,847402]]
        return {k:float(np.mean([r['contributions_to_J'][k] for r in selected])) for k in selected[0]['contributions_to_J']}
    ref_parts=average_parts(ref['candidate']);best_parts=average_parts(good[0]['candidate'])
    fig,axes=plt.subplots(2,3,figsize=(15,8));mapping={r['candidate']:r for r in summary}
    for i,r in enumerate(chosen):
        s=mapping[r['candidate']]
        for j in range(3):
            for x in s['runs']:axes[0,j].scatter(i,x['components'][j],marker='o' if x['noise']==847401 else '^',color='#3276a8' if x['noise']==847401 else '#db7c36',s=42)
            axes[0,j].scatter(i,r['components'][j],marker='_',color='black',s=160)
        for x in s['runs']:
            for j,v in enumerate([x['raw']['TA'].get('rank_correlation'),x['raw']['TB'].get('rank_correlation'),x['raw']['TB'].get('SCL_minus_ICL_lag_median_ms')]):
                if v is not None:axes[1,j].scatter(i,v,marker='o' if x['noise']==847401 else '^',color='#3276a8' if x['noise']==847401 else '#db7c36',s=42)
    titles=['整体rank分布误差 ↓','杆内/杆间时序分布误差 ↓','参与结构分布误差 ↓','TA平均rank模板相关 ↑','TB平均rank模板相关 ↑','TB：SCL−ICL质心中位差 (ms)']
    for ax,title in zip(axes.ravel(),titles):ax.set(xticks=range(len(chosen)),xticklabels=legend,title=title);ax.tick_params(axis='x',rotation=20,labelsize=8);ax.grid(alpha=.18)
    for ax in axes[1,:2]:ax.axhline(1,color='black',ls='--',lw=1);ax.set_ylim(0,1.05)
    axes[1,2].axhline(patient['TB']['rod_lag_ms']['median'],color='black',ls='--',label='患者TB中位数');axes[1,2].legend(fontsize=8)
    fig.suptitle('当前完整条件：三组分布分数改善，并不等于TB传播已恢复\n蓝圆=噪声847401，橙三角=847402；拓扑种子2511，条件间图可不同；黑横线为两运行均值',fontsize=13);fig.tight_layout(rect=(0,.22,1,.92));a.parameter_table(fig,[c['candidate'] for c in chosen],height=.17)
    for ext in ['png','pdf']:fig.savefig(F/f'current_conditions_and_TB_gap.{ext}',dpi=140)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6.4));partkeys=list(ref_parts);xx=np.arange(len(chosen));positive=np.zeros(len(chosen));negative=np.zeros(len(chosen))
    for key,color,label in zip(partkeys,['#7c8b96','#d4aa4b','#4d84b1','#b46b83'],['全事件特征','模式频率常数项','TB频率加权特征','TA频率加权特征']):
        vals=np.array([average_parts(c['candidate'])[key] for c in chosen]);axes[0].bar(xx,vals,bottom=np.where(vals>=0,positive,negative),color=color,label=label);positive+=np.maximum(vals,0);negative+=np.minimum(vals,0)
    axes[0].scatter(xx,[c['J'] for c in chosen],color='black',marker='d',s=24,label='实际J（含负项）')
    axes[0].set(xticks=xx,xticklabels=legend,ylabel='对冻结J的代数贡献',title='总分降低来自哪些项');axes[0].tick_params(axis='x',rotation=20,labelsize=8);axes[0].legend(fontsize=8)
    for i,c in enumerate(chosen):
        selected=[r for r in decomposition if r['candidate']==c['candidate'] and r['topology']==2511]
        for j,g in enumerate(['rank_pattern','local_and_interrod_timing','participation']):
            val=np.mean([r['conditional_feature_distances'][g]['TB']['D_off'] for r in selected])
            axes[1].scatter(j+i*.06,val,label=legend[i] if j==0 else None,color=['#252525','#d37c37','#487fa9','#519665'][i])
    axes[1].set(xticks=[.09,1.09,2.09],xticklabels=['TB rank','TB杆内/杆间时序','TB参与'],ylabel='仅TB事件的条件特征距离 ↓',title='条件分布另看：消除模型模式比例的直接混入');axes[1].legend(fontsize=8);axes[1].grid(alpha=.15)
    fig.suptitle('代数分解不是独立方差来源；模式特征块仍带频率权重\n右图使用已保存的逐模式条件距离，仅作诊断，不更换优化目标',fontsize=12);fig.tight_layout(rect=(0,.25,1,.9));a.parameter_table(fig,[c['candidate'] for c in chosen],height=.2)
    for ext in ['png','pdf']:fig.savefig(F/f'score_frequency_and_TB_shape.{ext}',dpi=140)
    plt.close(fig)
    # All initial one-scalar interventions, presented as a compact causal response
    # summary. Arrows/changes are paired within the same topology and noise.
    init=rt.read(OUT/'plan.json')['initial_meta'];base=mapping[ref['candidate']];ref_runs={r['noise']:r for r in base['runs']}
    effect=[]
    for m in init:
        if m['axis'] is None:continue
        s=mapping[m['candidate']]
        for r in s['runs']:
            b=ref_runs[r['noise']]
            effect.append(dict(candidate=m['candidate'],label=s['label'],axis=m['axis'],noise=r['noise'],delta_J=r['J']-b['J'],
                delta_groups=(np.array(r['components'])-b['components']).tolist(),
                delta_TA_rank=r['raw']['TA'].get('rank_correlation')-b['raw']['TA'].get('rank_correlation'),
                delta_TB_rank=r['raw']['TB'].get('rank_correlation')-b['raw']['TB'].get('rank_correlation'),
                delta_TB_lag_error=abs(r['raw']['TB'].get('SCL_minus_ICL_lag_median_ms')-patient['TB']['rod_lag_ms']['median'])-abs(b['raw']['TB'].get('SCL_minus_ICL_lag_median_ms')-patient['TB']['rod_lag_ms']['median'])))
    rt.write(NIGHT/'initial_paired_effects.json',dict(rows=effect,reference=ref['candidate'],definition='paired differences; same topology and dynamics seed'))
    with PdfPages(F/'night_progress_report.pdf') as pdf:
        from PIL import Image
        pages=[F/'current_conditions_and_TB_gap.png',F/'score_frequency_and_TB_shape.png',a.F/'patient_model_rank_templates.png',a.F/'patient_model_raw_observables.png',blockref.DEST/'figures/patient_actual_N_reference.png',a.F/'raw_paired_response_TA.png',a.F/'raw_paired_response_TB.png',a.F/'optimization_progress.png',NIGHT/'core_timing_tradeoff/figures/TB_lag_participation_core_timing.png',a.A/'native_review/g2_b01_p04/2511_847401/g2_b01_p04_patient_spectra_model_envelopes.png']
        pages += sorted(NIGHT.glob('batch*_review/figures/*.png'))
        pages += [a.A/'native_review/g2_b02_p04/2511_847401/g2_b02_p04_patient_spectra_model_envelopes.png']
        for path in pages:
            if not path.exists():continue
            with Image.open(path) as im:
                fig,ax=plt.subplots(figsize=(16,11));ax.imshow(im);ax.axis('off');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    (F/'README.md').write_text('### current_conditions_and_TB_gap.png\n三个训练分项与TA/TB模板相关、TB杆间毫秒差并列。每个点是一次固定拓扑与噪声运行，黑横线为两噪声等权平均；患者时间参考在TB图中显示。\n**关注点**：分布分数降低与TB传播恢复是不同问题，不能用上排替代下排及原生图。\n\n### night_progress_report.pdf\n汇集当前条件、患者模板、原量对照、单参数响应及训练进展，随完整评分条件更新。截图仅作报告打包，原始PDF/数据保存在本轮analysis。\n**关注点**：确认尚未完成时，不能把单拓扑训练响应升级为稳定效应。\n')
    with (F/'README.md').open('a') as f:f.write('\n### score_frequency_and_TB_shape.png\n左侧精确拆分冻结J的代数贡献；右侧独立展示TB条件特征距离，仍以两条运行等权。模式特征块也含频率影响，分解不能当作独立方差来源或机制归因。\n**关注点**：模式比例变好是否掩盖了TB条件传播分布仍未改善；不据此中途改变loss。\n')
    best=good[0];best_s=mapping[best['candidate']];lags=[r['raw']['TB']['SCL_minus_ICL_lag_median_ms'] for r in best_s['runs']]
    note=f'''# 夜间优化进展与科学判断

更新时间：{state['time']}。原始问题是：固定背景下，core位置、核向外EE强度和EE轴方向能否恢复患者两类事件的rank、杆内/杆间时差和参与分布；本轮不以一个事件同时匹配两类。

目前完整物理单元{state['physical_complete']}个，其中本轮新增{state['new_physical_complete']}个；实时阶段详见status.json。{len(good)}个条件已完成两噪声联合评分，仍按运行等权。患者标签参与训练评分，患者模板图不是独立验证。

当前最低J条件为“{a.point_label(best['candidate'])}”：J={best['J']:.3f}，固定参考J={ref['J']:.3f}，相对下降{(1-best['J']/ref['J'])*100:.1f}%。该条件TB杆间质心差两噪声为{lags[0]:.2f}/{lags[1]:.2f} ms，患者TB中位数{patient['TB']['rod_lag_ms']['median']:.2f} ms。不能仅以J降低判定TB已经恢复。A/B减项、逐模式原量、事件数和配对差均展开保存。

将当前最低J与参考的差精确拆开，单是模式频率常数项就从{ref_parts['mode_frequency']:.3f}降至{best_parts['mode_frequency']:.3f}；全事件特征项从{ref_parts['global_features']:.3f}变为{best_parts['global_features']:.3f}。这是评分函数的代数贡献，不是解释患者总体方差的比例。模式频率加权特征块还混有比例影响，因此另外给TB条件特征距离；首批“右核左移0.75mm”的TB时序条件距离实际上从参考约0.302增至0.318，支持“首批低分尚未修复TB时序”的判断。此诊断不修改本轮训练排序。

首批的方向性结论是：左核进一步左移改善TA平均rank相似度；右核左移改善三组联合分布分数但未缩短TB杆间延迟；右核上移使TB延迟略短，但两噪声的综合收益不一致。这些是共用拓扑种子的配对参数响应，尚待新的拓扑与噪声确认；角度改变会重建EE边和时延。SCL内部先后、ICL过度参与及TB集中在约35ms的残差必须继续看原量和原生场，不能归因于某个唯一机制。

另以每条模型的实际事件数N，抽取同一患者CAL数据块内连续N个事件作为描述性参照，保留自然模式比例及块内时序。以首批右核左移为例，N=144/145时患者窗口的TB杆间时差中位数5–95%范围约为−8至7ms，模型却仍为35.5–35.9ms；TA杆间时差中位数较接近，但模型TA事件间散布明显较窄。这项参照提示中心位置和散布是不同的恢复缺口。它不是置信区间或独立验证：CAL已用于开发，短于N的块不贡献完整窗，并且患者这些窗口多为数分钟至数十分钟，与模型60秒并非匹配时长。原始draw、实际N、每模式计数和逐触点/触点对统计保存在analysis/patient_count_block_reference，不加入loss或新门槛。

第一批联合探索点4提供了另一种取舍：两噪声TB杆间差降到8.27/5.52ms，但只有31/53、30/46个TB同时参与两杆；两杆可读的事件偏向两核接近同时达峰。其平均J约1.142仍差于参考1.069，不是当前最佳。这个六标量联合改变不能归因于某一参数，尚不能接受为患者双模式恢复；见[参与—时差—core峰值分解](core_timing_tradeoff/scientific_note.md)。该诊断及真实患者/模型图随附于PDF末尾，不进入BO或替代G3验收。

夜间按预定四批BO完成逐批选点和确认，使用相同三组目标及六标量硬范围；不根据当前图改变loss。G3之后实际审阅多事件原生图与患者频谱，满足既定数值和传播条件才开展最多40条参数响应补充。若仍不满足，交付具体能力边界和已完成的初始单因素响应，不无限加轮或宣称双模式恢复。

[核心比较图](figures/current_conditions_and_TB_gap.png) · [报告PDF](figures/night_progress_report.pdf) · [条件原量](condition_summary.json) · [配对影响](initial_paired_effects.json)。Agent已对启动时首批参考/候选显示布局进行目视核查；未来G3传播审阅与用户目视验收分别保存。
'''
    if state['window_end_reached']:
        note += ('\n**自主探索窗口已经结束。** 新增条件性G4扩展不再派发；已派发的完整轨迹继续收尾，'
                 '随后完成既定确认分析和真实原生图审阅。窗口结束不等于确认通过或全部目标完成；'
                 '实际完成量与仍在运行的单元以上方状态及窗口交付记录为准。\n')
    reviewed=sorted(NIGHT.glob('batch*_review/scientific_review.md'))
    if reviewed:
        note += '\n已完成批次的具体科学审阅：'+ ' · '.join(f'[{p.parent.name}]({p.relative_to(NIGHT)})' for p in reviewed)+'。各批新图和患者模板随附于同一PDF；独立批次图不更改冻结训练目标。\n'
    if (NIGHT/'initial_response_interpretation.md').exists():
        note += '\n[已有单参数实验的具体作用方向](initial_response_interpretation.md)：核向外EE增强使TA/TB的SCL内部顺序误差向相反方向变化；平均rank改善也可能伴随杆内顺序变差。共用拓扑种子不等于各角度条件使用完全相同EE边集合；方向变化的效应包含按规则重建的EE边与时延。\n'
    (NIGHT/'scientific_progress.md').write_text(note)
    rt.write(NIGHT/'last_report.json',dict(scored_units=len(rows),time=time.time()))

def monitor():
    with (NIGHT/'monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            s=snapshot();old=rt.read(NIGHT/'last_report.json') if (NIGHT/'last_report.json').exists() else {}
            if len(a.records())!=old.get('scored_units'):report()
            if s['window_end_reached'] or (OUT/'optimization_complete.json').exists():report();return
            time.sleep(60)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['report','monitor','snapshot']);arg=p.parse_args()
    {'report':report,'monitor':monitor,'snapshot':snapshot}[arg.action]()

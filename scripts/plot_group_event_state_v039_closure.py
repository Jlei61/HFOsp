#!/usr/bin/env python3
"""Reproduce pilot audit figures from the frozen machine summary and CSV."""
import argparse,csv,hashlib,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch


COLORS={'F':'#687880','L':'#187a9b','N':'#bf6533'}
SUBJECTS=['epilepsiae_1096','epilepsiae_1125','epilepsiae_253']


def plot(folder):
    summary=json.loads((folder/'summary_main.json').read_text());figures=folder/'figures'
    if figures.exists():raise FileExistsError(figures)
    figures.mkdir();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
    outputs=[];descriptions=[]
    def save(fig,name,text,focus):
        for ext in ['png','pdf','svg']:
            p=figures/(name+'.'+ext);fig.savefig(p,bbox_inches='tight');outputs.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        plt.close(fig);descriptions.append(f'### {name}.png\n\n{text}\n\n**关注点**：{focus}\n')
    with (folder/'paired_window_scores.csv').open() as stream:windows=list(csv.DictReader(stream))
    fig,axs=plt.subplots(3,3,figsize=(16.5,10.2),layout='constrained')
    for row,s in enumerate(SUBJECTS):
        selected=[r for r in windows if r['subject']==s and float(r['history_hours'])==8 and r['nonoverlapping_target']=='True']
        values=defaultdict(dict)
        for r in selected:values[(float(r['anchor_time']),int(r['seed']))][r['family']]=float(r['state_loss'])
        times=sorted({t for t,seed in values});origin=min(times);ax=axs[row,0]
        for family in ['L','N']:
            for seed in [20260905,20260906,20260907]:
                gain=[values[(t,seed)]['F']-values[(t,seed)][family] for t in times]
                ax.plot((np.array(times)-origin)/3600,gain,'o',color=COLORS[family],alpha=.22,ms=4)
            mean=[np.mean([values[(t,seed)]['F']-values[(t,seed)][family] for seed in [20260905,20260906,20260907]]) for t in times]
            ax.plot((np.array(times)-origin)/3600,mean,'o-',color=COLORS[family],ms=4,lw=1,label=family+' over F')
        ax.axhline(0,color='#555',lw=.8);ax.set_title(s.replace('epilepsiae_','E')+f': {len(times)} nonoverlapping target windows')
        ax.set_ylabel('Paired score gain over F');ax.set_xlabel('Time since first scored window (h)');ax.legend(frameon=False)
        ax=axs[row,1]
        for i,family in enumerate('FLN'):
            rows=[r for r in summary['main_rows'] if r['subject']==s and r['family']==family and r['history_hours']==8 and r['lead_hours']==2 and r['support_subset']=='all_anchors']
            ax.scatter(np.full(len(rows),i),[r['gain_over_constant_floored'] for r in rows],color=COLORS[family],s=35)
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks(range(3),list('FLN'));ax.set_ylabel('Net event score gain')
        ax.set_title('Beyond background / refitted constant')
        ax=axs[row,2]
        for i,family in enumerate('FLN'):
            rows=[r for r in summary['paired_comparisons'] if r['subject']==s and r['contrast']==family+':H8_over_H0.5' and r['lead_hours']==2 and r['support_subset']=='all_anchors']
            active=[r for r in summary['paired_comparisons'] if r['subject']==s and r['contrast']==family+':H8_over_H0.5' and r['lead_hours']==2 and r['support_subset']=='recent_input_available']
            ax.scatter(np.full(len(rows),i)-.08,[r['gain'] for r in rows],color=COLORS[family],s=35)
            ax.scatter(np.full(len(active),i)+.08,[r['gain'] for r in active],facecolors='none',edgecolors=COLORS[family],s=35)
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks(range(3),list('FLN'));ax.set_ylabel('H8 score gain over H0.5')
        ax.set_title('History gain: all anchors / recent-input subset')
        if row==0:
            ax.plot([],[],'o',color='#555',label='All eligible anchors');ax.plot([],[],'o',mfc='none',color='#555',label='Recent token available');ax.legend(frameon=False)
    fig.suptitle('Design pilot: history length and transition family\nLead = 2 h; target width = 0.5 h. Dots repeat optimization seeds, not patients.',fontsize=13)
    save(fig,'event_state_history_and_transition','左列逐个非重叠物理目标窗比较L/N与完整固定历史F；浅色点是三个优化seed，实线是这些seed的均值。中列直接显示超过背景和重拟合常数封底的事件净收益；右列分别显示全部anchor和近期确有已发布事件输入的长短历史配对收益。','相对F名次较好不等于事件净贡献为正；非重叠目标仍可共享历史，不能按点数推出独立患者证据。')
    contact_endpoints=['exact_next_subset','next_subset_all','stop']
    expression_endpoints=['band_centroid','band_log_energy','band_log_peak','cross_band_lag','waveform_statistics','propagation_span_s','next_group_delay_s','contact_coupling']
    label={'exact_next_subset':'Same-prefix next set','next_subset_all':'All next sets','stop':'STOP','band_centroid':'Band time centroid','band_log_energy':'Band energy','band_log_peak':'Band peak',
        'cross_band_lag':'Cross-band delay','waveform_statistics':'Waveform statistics','propagation_span_s':'Propagation span','next_group_delay_s':'Next-group delay','contact_coupling':'Contact coupling'}
    fig,axs=plt.subplots(3,2,figsize=(13,11.8),layout='constrained');column=[(f,a) for f in 'FLN' for a in ['state','functional']]
    for i,s in enumerate(SUBJECTS):
        for j,(name,endpoints) in enumerate([('contact_rows',contact_endpoints),('expression_rows',expression_endpoints)]):
            matrix=np.full((len(endpoints),6),np.nan)
            for row,endpoint in enumerate(endpoints):
                for col,(family,arm) in enumerate(column):
                    values=[r['gain_over_constant_floored'] for r in summary[name] if r['subject']==s and r['family']==family and r['history_hours']==8 and r['view']=='joint' and r['endpoint']==endpoint and r.get('arm')==arm and r.get('gain_over_constant_floored') is not None]
                    if values:matrix[row,col]=np.median(values)
            finite=np.abs(matrix[np.isfinite(matrix)]);limit=max(float(np.max(finite)) if len(finite) else 0,1e-4)
            ax=axs[i,j];im=ax.imshow(matrix,cmap='RdBu',norm=TwoSlopeNorm(0,-limit,limit),aspect='auto')
            ax.set_xticks(range(6),[f+' '+('latent' if a=='state' else 'task') for f,a in column],rotation=35,ha='right');ax.set_yticks(range(len(endpoints)),[label[e] for e in endpoints])
            ax.set_title(s.replace('epilepsiae_','E')+(' | Contact NLL gain' if j==0 else ' | Normalized expression MSE gain'))
            for row in range(len(endpoints)):
                for col in range(6):
                    value=matrix[row,col];text='NE' if not np.isfinite(value) else '0.000' if abs(value)<.0005 else f'{value:.3f}'
                    ax.text(col,row,text,ha='center',va='center',fontsize=7.5,color='white' if np.isfinite(value) and abs(value)>.65*limit else '#202020')
            fig.colorbar(im,ax=ax,shrink=.6,pad=.02)
    fig.suptitle('Same frozen H8 state, previously untrained event-expression tasks\nMedian across 3 optimization seeds; gains capped by the parent and refitted constant.',fontsize=13)
    save(fig,'frozen_event_expression_transfer','每个病例都用同一批冻结的F/L/N状态评价contact集合/STOP和未训练的细事件表达。latent是原始状态，task是上游已训练任务的功能读出；各格为三个seed的收益中位数，并已相对父基线和重拟合常数封底。','不同面板使用各自色标；多端点正值不自动证明一个共同生理原因，还需对照完整历史、初始化表征和单视图迁移。')
    fig,axs=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,s in zip(axs,SUBJECTS):
        for i,(view,endpoint,title) in enumerate([('count','cross_future_recruitment_vector_2h','Count → recruitment'),('recruitment','cross_future_count_log1p_2h','Recruitment → count')]):
            for off,arm,marker in [(-.09,'state','o'),(.09,'functional','s')]:
                rows=[r for r in summary['expression_rows'] if r['subject']==s and r['view']==view and r['endpoint']==endpoint and r.get('arm')==arm]
                ax.scatter(np.full(len(rows),i)+off,[r['gain_over_constant_floored'] for r in rows],color='#187a9b' if arm=='state' else '#bf6533',marker=marker,s=35,label=arm if i==0 else None)
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks([0,1],['Count → recruitment','Recruitment → count'],rotation=18,ha='right')
        ax.set_title(s.replace('epilepsiae_','E')+' | '+summary['single_view_family_selection']['family'][s]);ax.set_ylabel('Frozen cross-task normalized MSE gain');ax.legend(frameon=False)
    fig.suptitle('Single-view transfer: family fixed using joint-task INNER only\nCross-task gains can exist at initialization; own-task and full controls are reported separately.',fontsize=12)
    save(fig,'single_view_frozen_cross_prediction','上游分别只训练未来计数或粗招募，冻结后再预测没有用于塑造该状态的另一个完整响应。每个点是一条优化seed，两个符号分别为latent与已训练任务功能读出；家族在任何迁移结果之前由joint H8的INNER确定。','E1096最大的latent正值来自第0步，与初始化相同；必须结合正文的自身任务与完整对照，不能只凭跨任务正值宣称统一状态。')
    cases=['fixed_history','linear_nonlinear_observation','nonlinear_transition','background_only','separate_states','no_event_feedback']
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    rows=[r for r in summary['instruments'] if r['experiment']=='joint'];values={(r['case'],r['family']):r['held_out']['total'] for r in rows}
    for i,f in enumerate('FLN'):
        gain=[values[(case,'C')]-values[(case,f)] for case in cases];axs[0].plot(range(6),gain,'o-',color=COLORS[f],label=f)
    axs[0].axhline(0,color='#555',lw=.8);axs[0].set_xticks(range(6),[s.replace('_',' ') for s in cases],rotation=35,ha='right');axs[0].set_ylabel('Held-out score gain over fitted C');axs[0].legend(frameon=False);axs[0].set_title('Six known-truth instruments: one training seed')
    for i,case in enumerate(['separate_states','no_event_feedback']):
        p=next(r for r in summary['synthetic_frozen_probes'] if r['case']==case and r['family']=='N' and r['view']=='count')
        outcomes=p['results']['cross_recruitment_distribution']
        for off,arm,color in [(-.15,'trained_raw_latent','#187a9b'),(.15,'trained_view_readout','#bf6533')]:
            axs[1].bar(i+off,outcomes[arm]['gain_over_background'],.28,color=color,label=arm.replace('_',' ') if i==0 else None)
    axs[1].axhline(0,color='#555',lw=.8);axs[1].set_xticks([0,1],['Separate causes','No event feedback']);axs[1].set_ylabel('Cross-recruitment MSE gain');axs[1].legend(frameon=False);axs[1].set_title('Prediction and observer updates do not prove feedback')
    save(fig,'known_truth_and_independent_cause_controls','左图完整列出六类已知真值反例，不要求非线性N处处获胜；每类的学习率只由INNER选择。右图用count-view的非线性观察器展示独立原因与无事件反馈反例中的冻结跨任务信息。','这一层使用独立合成episode和一个训练seed，是仪器诊断；不能替代少量真实记录窗及发作分母下的全流程功效校准。')
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.6),layout='constrained')
    for s,row in zip(SUBJECTS,summary['window_support_calibration']['rows']):
        axs[0].plot([r['signal_sd_per_fit_factor_sd'] for r in row['results']],[r['fraction_positive_heldout_gain'] for r in row['results']],'o-',label=s.replace('epilepsiae_','E'))
    axs[0].set_xlabel('Synthetic signal / noise SD');axs[0].set_ylabel('Fraction with positive held-out gain');axs[0].set_ylim(0,1.03);axs[0].legend(frameon=False);axs[0].set_title('Known factor supplied to readout (optimistic diagnostic)')
    x=np.arange(3)
    for i,(p,color) in enumerate(zip(['FIT','INNER','SELECTION'],['#687880','#187a9b','#bf6533'])):
        axs[1].bar(x+(i-1)*.24,[r['onsets_by_phase'][p] for r in summary['seizure_transfer']['rows']],.23,color=color,alpha=.3,label=p)
        axs[1].bar(x+(i-1)*.24,[r['risk']['observed_seizures_by_phase'][p] for r in summary['seizure_transfer']['rows']],.11,color=color)
    axs[1].set_xticks(x,['E1096','E1125','E253']);axs[1].set_ylabel('Distinct annotated onsets');axs[1].set_title('Raw / supported next onsets\nAll three designs: NOT ESTIMABLE')
    handles,labels=axs[1].get_legend_handles_labels();axs[1].legend(handles+[Patch(color='#444',alpha=.3),Patch(color='#444')],labels+['Raw upper bound','Past-supported next onset'],frameon=False,fontsize=7.5)
    save(fig,'physical_support_and_seizure_denominators','左图保留真实查询窗口与缺口，在不同合成效应强度下直接把已知因子交给读出器，展示500次相关噪声重复中的正收益比例。右图浅色宽柱为原始FIT/INNER/留出发作数，深色窄柱为满足过去历史支持的查询实际可对应的下一次发作数，三例均未达到可估条件。','左图不是完整模型训练或临床发作功效；右图不能用原始或聚集发作数代替可估分母，NOT_ESTIMABLE也不是发作关系阴性。')
    (figures/'README.md').write_text('# v0.3.9设计pilot审计图\n\n这些图对应修正背景输入的有限实验；不作为独立患者确认或临床预测验证。\n\n'+'\n'.join(descriptions))
    (figures/'metadata.json').write_text(json.dumps(dict(source_summary=str(folder/'summary_main.json'),source_sha256=hashlib.sha256((folder/'summary_main.json').read_bytes()).hexdigest(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outputs=outputs,visual_review='PENDING'),indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',figures=len(descriptions),formats=['png','pdf','svg'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);plot(p.parse_args().reports)

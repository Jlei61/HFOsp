"""Compact scientific summary of the completed screen; no simulation or scoring changes."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from scripts import analyze_topic4_core_connectivity_search as an
from scripts import run_topic4_propagation_recovery_night as night

def main():
    A=an.run.OUT/'analysis';O=night.OUT/'initial_review';F=O/'figures';F.mkdir(parents=True,exist_ok=True)
    read=lambda p:list(csv.DictReader(p.open()))
    obs=read(A/'run_mode_observations.csv');ref=json.loads((A/'patient_observation_reference.json').read_text());plan=json.loads((an.run.OUT/'plan.json').read_text())
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
    families=['radius_A_mm','EE_same_core_scale','EI_same_core_scale','EE_kernel_perp_scale']
    titles=['左核半径','核内 E→E 权重','核内 E→I 权重','E→E 横向范围']
    keys=['SCL_upper_participation','ICL_contact_participation','TA_fraction']
    ylabs=['SCL9/8 平均参与概率','ICL 平均参与概率','TA 标签占比']
    fig,axes=plt.subplots(3,4,figsize=(12.8,8),sharey='row',layout='constrained');records=[]
    for col,(fam,title) in enumerate(zip(families,titles)):
      cases=[c for c in plan['candidates'] if c['layout']=='up4p5' and c['changed_parameter'] in ['baseline',fam]]
      cases.sort(key=lambda c:c['parameters'][fam]);xs=[c['parameters'][fam] for c in cases]
      for si,seed in enumerate(plan['seeds']):
        values=[]
        for c in cases:
            rr=[r for r in obs if r['candidate']==c['id'] and r['seed']==str(seed) and r['layer']=='all_detected'];x=next(r for r in rr if r['mode']=='ALL');ta=next(r for r in rr if r['mode']=='TA')
            values.append([float(x[k]) if x[k] else np.nan for k in keys[:2]]+[float(ta['n'])/float(x['n']) if float(x['n']) else np.nan])
            records.append(dict(candidate=c['id'],parameter=fam,value=c['parameters'][fam],seed=seed,n=int(x['n']),TA=int(ta['n']),SCL=float(x['SCL_upper_participation']) if x['SCL_upper_participation'] else None,ICL=float(x['ICL_contact_participation']) if x['ICL_contact_participation'] else None))
        for row,key in enumerate(keys):
            ax=axes[row,col];ax.plot(xs,np.asarray(values)[:,row],color='#255b84',ls='-' if si==0 else '--',marker='o' if si==0 else 's',lw=1.5,ms=5)
      for row,key in enumerate(keys):
        ax=axes[row,col];target=ref['ALL'][key] if key!='TA_fraction' else ref['TA']['n']/ref['ALL']['n'];ax.axhline(target,c='#a04a45',ls=':',lw=1)
        ax.axvline(plan['baseline_parameters'][fam],c='gray',lw=.5,alpha=.7);ax.set_ylim(-.03,1.03);ax.set_xticks(xs,[f'{x:.2f}' if fam=='radius_A_mm' else f'{x:g}' for x in xs]);ax.spines[['right','top']].set_visible(False)
        if col==0:ax.set_ylabel(ylabs[row])
        if row==0:ax.set_title(title)
        if row==2:ax.set_xlabel('半径 (mm)' if fam=='radius_A_mm' else '相对基线的倍数')
    handles=[Line2D([0],[0],c='#255b84',marker='o',label='噪声重演 1'),Line2D([0],[0],c='#255b84',ls='--',marker='s',label='噪声重演 2'),Line2D([0],[0],c='#a04a45',ls=':',label='患者 FIT 描述参考')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('同一上移布局：哪些参数增加招募，哪些改变两类标签的占比？\n全部检测事件的开发诊断；患者参考采用原合格事件表，两者尚不能作正式分布验收',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(F/f'parameter_recruitment_tradeoff.{ext}',dpi=200)
    plt.close(fig)
    data=read(an.run.OUT/'rapid_audit_20260911/core_event_timing.csv');ids=['endpoint__baseline','up4p5__EE_core_to_out_scale_1.25','near_upper__EE_kernel_perp_scale_1.5'];labels=['端点原位','左核上移＋离核输出增强','靠近上部 SCL＋横向范围扩大']
    fig,axes=plt.subplots(1,3,figsize=(12.8,4),layout='constrained',sharey=True);timing_records=[]
    for ax,cid,title in zip(axes,ids,labels):
      for mi,mode in enumerate(['TA','TB']):
       for si,seed in enumerate(plan['seeds']):
        x=np.array([float(r['B_minus_A_t10_ms']) for r in data if r['candidate']==cid and r['seed']==f'2511_{seed}' and r['mode']==mode]);x=x[np.isfinite(x)]
        at=mi+[-.15,.15][si]
        if len(x):
            q=np.quantile(x,[.05,.5,.95]);ax.plot([at,at],[q[0],q[2]],color=an.MODE_COLOR[mode],lw=2);ax.plot(at,q[1],marker='o' if si==0 else 's',c=an.MODE_COLOR[mode],ms=6);ax.text(at,94,f'n={len(x)}',ha='center',fontsize=8)
            timing_records.append(dict(candidate=cid,seed=seed,mode=mode,n=len(x),q05=q[0],median=q[1],q95=q[2],A_earlier_fraction=float((x>0).mean())))
        else:ax.text(at,94,'n=0',ha='center',fontsize=8,color='gray')
      ax.axhline(0,c='gray',lw=.8);ax.set(xticks=[0,1],xticklabels=['TA 标签','TB 标签'],title=title,ylim=(-80,108),xlim=(-.5,1.5));ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('右核 − 左核的 10% 累计活动时间 (ms)')
    fig.suptitle('两种标签，不一定对应不同的领先核\n正值：左核较早；点为中位数、线为事件 5–95% 范围；圆/方为两条噪声',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(F/f'core_timing_mode_separation.{ext}',dpi=200)
    plt.close(fig)
    an.writecsv(O/'parameter_effect_records.csv',records);an.writecsv(O/'core_timing_records.csv',timing_records)
    (F/'README.md').write_text('''### parameter_recruitment_tradeoff.png
固定左核上移4.5 mm的布局和拓扑2511，四列分别改变左核半径、核内EE、核内EI和EE横向范围；每条线保留一个动力学噪声重演。三行分别是SCL9/8参与、ICL参与、TA标签占比；红点线仅是患者FIT原合格事件表的描述参考。全部模型检测与患者合格表的窗口资格不同，不能据此做正式分布验收。**关注点**：招募增加与两种传播恢复是不同问题；半径会改变核成员和随机输入总支持，范围改变实际邻接。

### parameter_recruitment_tradeoff.pdf
上图同源PDF，数值来自同一CSV，没有重新聚合或选点。**关注点**：这是开发诊断，不是已接受的论文阳性主图。

### core_timing_mode_separation.png
三个事后定位条件的全部检测事件，分别计算两核在250 ms观察窗内累积发放质量达到10%的时间。圆/方对应两个噪声，点线为事件中位数与5–95%范围，无事件显示n=0；相邻模型窗口可重叠。**关注点**：正值表示左核较早积累活动，不证明因果起源；增加SCL参与后，两个标签可能仍主要由同一核较早活动。

### core_timing_mode_separation.pdf
上图同源PDF，保留分模式、分运行及实际事件数。**关注点**：没有将事件当作独立网络，也没有以领先核定义TA/TB。
''')
    print(str(F))

if __name__=='__main__':main()

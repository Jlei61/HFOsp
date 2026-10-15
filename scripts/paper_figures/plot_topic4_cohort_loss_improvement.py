"""Cohort loss trajectories and per-patient FIT/TEST improvements.

Selection remains frozen and training-only. No simulation, refitting, or new
held-out selection is performed. Missing baselines are retained as unavailable.
"""
from pathlib import Path
import csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/topic4_cohort_optimization'))
from common import OUT, read, write, sha

DEST=OUT/'cohort_loss_improvement_20260923'
TRAIN='#286FA8'
TEST='#C77B2E'

def collect():
    rows=read(OUT/'cohort_review.json')['subjects'];data=[]
    for i,r in enumerate(rows,1):
        sid=r['subject'];d=OUT/'subjects'/sid
        scores=[read(p) for p in sorted((d/'scores').glob('*.json'))]
        assert len(scores)==28
        scores={s['candidate']:s for s in scores};base=scores['initial_00']['J']
        stage_ids=[];stage_loss=[]
        for stage in (0,1,2):
            good=[s for s in scores.values() if s['stage']<=stage and s['J'] is not None]
            best=min(good,key=lambda s:(s['J'],s['candidate']))
            stage_ids.append(best['candidate']);stage_loss.append(best['J'])
        nomination=read(d/'nomination.json')['selected']
        assert stage_ids[-1]==nomination==r['arms']['nominated']['candidate']
        train_final=r['arms']['nominated']['J']
        np.testing.assert_allclose(stage_loss[-1],train_final,rtol=0,atol=1e-12)
        test_base=r['heldout_distribution_scores']['baseline']['J']
        test_final=r['heldout_distribution_scores']['nominated']['J']
        paired=base is not None and test_base is not None
        if paired:
            # Signed D_off can in general be negative. Relative reduction is
            # valid here only because these particular baseline losses are >0.
            assert base>0 and test_base>0
            curve=np.r_[base,stage_loss]/base
            assert np.all(np.diff(curve)<=1e-12)
            fit_reduction=100*(1-train_final/base)
            heldout_reduction=100*(1-test_final/test_base)
        else:
            curve=None;fit_reduction=None;heldout_reduction=None
        data.append(dict(patient_number=i,subject=sid,selected_candidate=nomination,paired=paired,
            baseline_primary_events=r['arms']['baseline']['N'],selected_primary_events=r['arms']['nominated']['N'],
            train_baseline=base,train_selected=train_final,test_baseline=test_base,test_selected=test_final,
            train_reduction_percent=fit_reduction,test_reduction_percent=heldout_reduction,
            batch_end_selected_ids=stage_ids,training_loss_at_counts=[base,*stage_loss],normalized_training_loss=curve,
            baseline_status=scores['initial_00']['status'],selected_order_support_complete=r['arms']['nominated']['heldout_metrics'].get('full_contact_pair_support',False),
            source_review_sha256=sha(d/'review.json'),source_nomination_sha256=sha(d/'nomination.json')))
    return data

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':20,'axes.labelsize':22,'axes.titlesize':22,
        'xtick.labelsize':17,'ytick.labelsize':18,'legend.fontsize':16,'axes.linewidth':1.,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})

def trajectory(ax,data,*,legend_loc='lower left',legend_frame=False):
    x=np.array([1,12,20,28]);curves=np.array([r['normalized_training_loss'] for r in data if r['paired']])
    for y in curves:ax.plot(x,y,color='#71A2C8',lw=1,alpha=.50,zorder=1)
    ax.plot(x,np.median(curves,axis=0),'o-',color=TRAIN,lw=3,ms=7,zorder=4)
    ax.set(xlim=(.2,28.8),ylim=(0,1.06),xticks=x,yticks=[0,.25,.5,.75,1.],
        xlabel='Evaluated conditions',ylabel='Best training loss / baseline')
    ax.set_yticklabels(['0','0.25','0.50','0.75','1.00'])
    ax.grid(axis='y',alpha=.15,lw=.7)
    legend=ax.legend(handles=[Line2D([],[],color='#71A2C8',lw=1,label='Each patient'),Line2D([],[],color=TRAIN,lw=3,label='Patient median')],
        loc=legend_loc,frameon=legend_frame,borderaxespad=.3,handlelength=1.6,
        fancybox=False,framealpha=1,facecolor='white',edgecolor='.35')
    legend.get_frame().set_linewidth(.8)

def reductions(ax,data):
    paired=[r for r in data if r['paired']];x=np.array([r['patient_number'] for r in paired]);w=.36
    fit=np.array([r['train_reduction_percent'] for r in paired]);test=np.array([r['test_reduction_percent'] for r in paired])
    ax.bar(x-w/2,fit,width=w,color=TRAIN,zorder=3)
    ax.bar(x+w/2,test,width=w,color=TEST,zorder=3)
    for r in data:
        if not r['paired']:
            pos=r['patient_number'];ax.axvspan(pos-.44,pos+.44,facecolor='.96',edgecolor='.84',hatch='///',lw=.4,zorder=0)
            ax.text(pos,.035,'NA',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=12,color='.35')
    # Do not force apparent improvement if future artifact values are negative.
    lower=min(0,float(np.floor(min(fit.min(),test.min())/20)*20))
    upper=max(100,float(np.ceil(max(fit.max(),test.max())/20)*20))
    ax.set(xlim=(.35,len(data)+.65),ylim=(lower,upper),xticks=range(1,len(data)+1),
        xlabel='Patient',ylabel='Loss reduction (%)')
    ax.set_yticks(np.arange(lower,upper+1,20));ax.tick_params(axis='x',labelsize=15,length=3,pad=7)
    ax.grid(axis='y',alpha=.15,lw=.7)
    ax.legend(handles=[Patch(facecolor=TRAIN,label='Training'),Patch(facecolor=TEST,label='Held-out records')],
        loc='upper left',bbox_to_anchor=(0,1.15),ncol=2,frameon=False,borderaxespad=0,handlelength=1.2,columnspacing=1.6)

def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight',pad_inches=.10)
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',pad_inches=.10)
    plt.close(fig)

def main():
    style();data=collect();DEST.mkdir(parents=True,exist_ok=True);f=DEST/'figures';f.mkdir(exist_ok=True)
    paired=[r for r in data if r['paired']]
    train=np.array([r['train_reduction_percent'] for r in paired]);test=np.array([r['test_reduction_percent'] for r in paired])
    curves=np.array([r['normalized_training_loss'] for r in paired])
    summary=dict(n_run_subjects=len(data),n_paired=len(paired),n_train_improved=int((train>0).sum()),n_test_improved=int((test>0).sum()),
        median_within_patient_train_reduction_percent=float(np.median(train)),median_within_patient_test_reduction_percent=float(np.median(test)),
        train_reduction_range_percent=[float(train.min()),float(train.max())],test_reduction_range_percent=[float(test.min()),float(test.max())],
        median_normalized_training_loss=np.median(curves,axis=0),condition_counts=[1,12,20,28],
        initial_to_adaptive_improved=int(np.sum(curves[:,-1]<curves[:,1]-1e-12)),
        first_to_second_adaptive_improved=int(np.sum(curves[:,-1]<curves[:,2]-1e-12)),
        nomination='Fixed final training-only minimum; the same candidate is evaluated on held-out patient records',
        relative_loss_formula='J / own baseline J; all 24 finite denominators verified strictly positive',
        reduction_formula='100 * (J_baseline - J_selected) / J_baseline, separately for FIT and TEST',
        limitation='Training incumbent curves are nonincreasing by construction; held-out endpoint reductions supply separate evidence. No matched-budget optimizer comparison or new-noise/topology confirmation.',
        records=data)
    write(DEST/'figure_data.json',summary)
    with (DEST/'patient_loss_table.csv').open('w') as handle:
        fields=['patient_number','subject','selected_candidate','paired','baseline_primary_events','selected_primary_events','train_baseline','train_selected','test_baseline','test_selected','train_reduction_percent','test_reduction_percent']
        w=csv.DictWriter(handle,fieldnames=fields);w.writeheader()
        for r in data:w.writerow({k:r[k] for k in fields})
    fig,(a,b)=plt.subplots(1,2,figsize=(17,6),gridspec_kw={'width_ratios':[1,1.85]})
    fig.subplots_adjust(left=.068,right=.99,bottom=.18,top=.86,wspace=.31)
    trajectory(a,data);reductions(b,data)
    for ax,letter in [(a,'A'),(b,'B')]:ax.text(-.10,1.16,letter,transform=ax.transAxes,fontsize=24,fontweight='bold',va='top')
    save(fig,f/'cohort_loss_improvement')
    fig,ax=plt.subplots(figsize=(6.6,6));fig.subplots_adjust(left=.20,right=.97,bottom=.17,top=.97);trajectory(ax,data,legend_loc='upper right',legend_frame=True);save(fig,f/'cohort_loss_trajectory')
    fig,ax=plt.subplots(figsize=(11.2,6));fig.subplots_adjust(left=.115,right=.99,bottom=.17,top=.86);reductions(ax,data);save(fig,f/'cohort_loss_reduction_by_patient')
    (f/'README.md').write_text('''### cohort_loss_improvement.png
左图显示同一24位可配对患者在基线、12个初始条件完成、20个及28个条件完成时的已观测最低训练loss，以各自基线归一化；细线为患者，粗线为中位数。右图逐患者比较训练与留出记录的最终loss降幅，25位患者均保留编号，第17位基线没有合格事件，标为NA而不画成0。留出记录始终评价训练选出的同一工作点，没有按留出分数重新挑选。
**关注点**：24/24可配对患者两种loss均下降，患者内降幅中位数约55%和50%；训练最佳值曲线按定义不增，跨记录改善的证据来自右侧留出结果，不能由左侧单独证明优化器优越或患者机制恢复。

### cohort_loss_trajectory.png
合图A的独立无角标版本，图例位于右上角并加边框；只在完整批次边界连线，不按worker完成顺序虚构更新，也不把曲线中间线段称为实测评估。横轴为已评估条件数，1是端点先验基线，12含该基线，20和28分别包含第一、第二批自适应提案。
**关注点**：24位可配对患者中，22位在初始12点以后仍找到更低训练loss；这不是相同预算随机搜索的对照实验。

### cohort_loss_reduction_by_patient.png
合图B的独立无角标版本，降幅为100×(基线loss−训练选优工作点loss)/基线loss，分别对训练和留出数据计算；这里24个基线分母均为正。患者编号固定沿本轮队列次序，完整身份映射、原始loss和合格事件数见上级patient_loss_table.csv。
**关注点**：第17位不可计算基线降幅；第12位虽缺少部分杆内触点对观测，整体分布loss仍可计算，因此本图的配对人数24与此前三个描述性指标图的完整支持人数23不同。
''')
    text=f'''# 跨患者 loss 改善图说明

本图补充“同一优化流程是否能跨患者降低事件分布拟合误差”，与已有三指标取舍图回答不同问题。25位执行患者中，24位基线与最终工作点的loss均可估计；24/24训练loss下降，24/24对未参与优化的患者记录重新计算的loss也下降。患者内相对降幅的中位数分别为{np.median(train):.1f}%与{np.median(test):.1f}%，不是两个群体中位数之比。

第17位程帅的基线268个检测窗全部因重叠未进入primary，不能计算基线loss；其最终工作点有119个primary事件，单独报告，不将缺值填成0。E590（第12位）整体分布loss可评分，虽然部分杆内触点对无法单独估计；所以本图24人配对与之前描述性指标23人完整配对的分母不同。

左图按真实批次边界显示最佳已观测训练分数：端点几何基线、12点初始探测、20点、28点。最终提名与已冻结训练最小值逐一核对；不按TEST改名次。TEST使用同一套FIT核特征、同一CAL正尺度和同一模型事件，只将患者目标均值嵌入换为完整留出记录。

建议结果表述：**同一几何先验约束下的优化流程在本轮全部24位可配对患者中均降低了事件分布拟合损失，且改善在未参与优化的患者记录上保留。** 这支持本轮可运行队列内的跨患者适用性；“普适地恢复患者机制”或“优于同预算的其他搜索方法”仍没有对应证据。当前仍是固定拓扑/噪声实现，也不抵消杆内相对顺序在部分患者变差的结论。

未产生新仿真、未修改训练目标或提名、未替换论文正式图。输出为候选PNG/SVG；等待用户目视验收。
'''
    (DEST/'scientific_note.md').write_text(text)
    print({k:v for k,v in summary.items() if k not in ['records']})

if __name__=='__main__':main()

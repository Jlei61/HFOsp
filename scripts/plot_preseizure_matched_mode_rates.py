#!/usr/bin/env python3
"""Re-express existing mode-specific rate contrasts with a symmetric comparator."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic5_preseizure_template_association_broadband/cohort_20260909'
OUT=BASE/'matched_mode_rate_comparison'


def main():
    OUT.mkdir(exist_ok=True)
    old=pd.read_csv(BASE/'all_patient_association_tests.csv')
    rows=[]
    for p in sorted((BASE/'per_subject').glob('*/association/window_observations.csv')):
        sid=int(p.parts[-3].split('_')[-1]);d=pd.read_csv(p)
        for window,g in d.groupby('window'):
            for mode in ['TA','TB']:
                x=(g['n_'+mode.lower()]/g.observed_hours).to_numpy();same=(g.label==mode).to_numpy()
                valid=np.isfinite(x);x=x[valid];same=same[valid];n1=int(same.sum());n0=int((~same).sum())
                effect=2*(rankdata(x)[same].sum()-n1*(n1+1)/2)/(n1*n0)-1 if n1 and n0 else np.nan
                ref=old[(old.subject==sid)&(old.feature==window+'_log_rate_'+mode.lower())]
                row=dict(subject=sid,window=window,interictal_mode=mode,n_same=n1,n_other=n0,
                    median_same=float(np.median(x[same])) if n1 else np.nan,
                    median_other=float(np.median(x[~same])) if n0 else np.nan,
                    rank_effect_same_higher=effect,primary_geometry=sid!=139,
                    comparison_status='DESCRIPTIVE_ONLY_TOO_FEW' if min(n1,n0)<2 else 'EXISTING_EXPLORATORY_CONTRAST')
                if len(ref) and min(n1,n0)>=2:
                    r=ref.iloc[0];sign=-1 if mode=='TA' else 1
                    np.testing.assert_allclose(effect,sign*r.rank_biserial,atol=1e-12)
                    bounds=sorted([sign*r.leave_one_out_min,sign*r.leave_one_out_max])
                    row.update(p=r.p,q_bh22=r.q_bh22,leave_one_out_min=bounds[0],leave_one_out_max=bounds[1])
                rows.append(row)
    allrows=pd.DataFrame(rows);allrows.to_csv(OUT/'same_vs_other_mode_rates.csv',index=False)
    balanced=[]
    for (sid,window),g in allrows.groupby(['subject','window']):
        if len(g)==2 and g.rank_effect_same_higher.notna().all():
            balanced.append(dict(subject=sid,window=window,
                balanced_same_higher_probability=float(((g.rank_effect_same_higher+1)/2).mean()),
                balanced_rank_effect=float(g.rank_effect_same_higher.mean()),
                minimum_seizures_per_source=int(g[['n_same','n_other']].min().min()),
                definition='Equal weight for TA and TB; within each mode compare its rate before same vs other source; ties count 0.5. Descriptive, not held-out prediction accuracy.'))
    pd.DataFrame(balanced).to_csv(OUT/'label_symmetric_rate_summary.csv',index=False)
    d=allrows[(allrows.window=='pre15')&allrows.subject.isin([1146,590,635])]
    fig,ax=plt.subplots(figsize=(10,5.2));fig.subplots_adjust(left=.2,right=.95,top=.79,bottom=.28)
    subjects=[1146,590,635];colors={'TA':'#b2182b','TB':'#2166ac'}
    for mode,dy in [('TA',-.13),('TB',.13)]:
        q=d[d.interictal_mode==mode].set_index('subject').loc[subjects]
        yy=np.arange(3)+dy
        ax.hlines(yy,q.leave_one_out_min,q.leave_one_out_max,color=colors[mode],alpha=.35,lw=3)
        ax.scatter(q.rank_effect_same_higher,yy,color=colors[mode],s=80,label=mode+' interictal events',zorder=3)
    b=pd.DataFrame(balanced).query("window=='pre15'").set_index('subject').loc[subjects]
    ax.scatter(b.balanced_rank_effect,np.arange(3),c='black',s=45,marker='D',label='TA/TB equally weighted',zorder=4)
    ax.axvline(0,c='#888888',ls='--',lw=1)
    ax.set(yticks=np.arange(3),yticklabels=['E1146  (10 TA / 2 TB)','E590  (2 TA / 3 TB)','E635  (4 TA / 4 TB)'],
        xlim=(-1.08,1.08),ylim=(2.5,-.55),xticks=[-1,-.5,0,.5,1],xlabel='Rank effect: same-source versus other-source seizures')
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,fontsize=10)
    fig.suptitle('Does the corresponding interictal mode become more frequent?',fontsize=16,y=.97)
    fig.text(.2,.13,'Negative: more frequent before the OTHER source type.\nPositive: more frequent before the SAME source type.',fontsize=11,linespacing=1.5)
    fig.text(.2,.02,'Preceding ≤15 min; dots = 2 × AUC − 1. Lines = leave-one-seizure-out range, not confidence intervals.\nExisting exploratory contrasts; each seizure is one observation.',fontsize=10,color='#444444')
    fp=OUT/'figures';fp.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(fp/f'matched_mode_pre15_rate.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (fp/'README.md').write_text('### matched_mode_pre15_rate.png\n对TA、TB分别比较该间期模式在同型发作前和异型发作前的事件率；正值表示在同型发作前更高，零表示组间秩效应为零。彩色点为2×AUC−1，细线为逐一删除发作后的效应范围而非置信区间，黑菱形为两种模式等权平均，PDF为同图矢量版。读数直接复用原候选分析，将TA事件率的效应反号以统一同型/异型语义；等权汇总为新描述，不进行新的显著性检验，也不选择每患者较有利的模式。\n**关注点**：主图只显示两种source各至少2次的二维患者；其余患者的描述结果保留在CSV，单一标签患者无法作患者内对照。模式等权的同型较高概率不等于留出预测准确率，当前综合方向较弱，图待人工检查。\n',encoding='utf-8')
    print(d[['subject','interictal_mode','n_same','n_other','median_same','median_other','rank_effect_same_higher','p']].to_string(index=False))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Standalone descriptive display of the existing broadband-qualified contrast."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.explore_e1146_seizure_interictal_association import exact_rank_test
BASE=ROOT/'results/topic5_preseizure_template_association_broadband/cohort_20260909'
SOURCE=BASE/'per_subject/epilepsiae_1146/association'
OUT=BASE/'e1146_pre15_tb_rate'


def main():
    windows=pd.read_csv(SOURCE/'window_observations.csv')
    d=windows[windows.window=='pre15'].copy().sort_values('sz')
    d['tb_events_per_hour']=d.n_tb/d.observed_hours
    d['observed_minutes']=60*d.observed_hours
    d['nominal_minutes']=(d.end_epoch-d.start_epoch)/60
    assert len(d)==12 and sum(d.label=='TB')==2
    np.testing.assert_allclose(d.tb_events_per_hour,np.expm1(d.log_rate_tb),rtol=1e-12)
    test=exact_rank_test(d.tb_events_per_hour,d.label=='TB')
    original=pd.read_csv(SOURCE/'association_screen.csv').set_index('feature').loc['pre15_log_rate_tb']
    assert np.isclose(test['p'],original.p) and test['auc_tb']==1
    med=d.groupby('label').tb_events_per_hour.median().to_dict()
    OUT.mkdir(exist_ok=True);fp=OUT/'figures';fp.mkdir(exist_ok=True)
    d.to_csv(OUT/'plotted_seizures.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.2,7.8));fig.subplots_adjust(left=.15,right=.96,top=.9,bottom=.26)
    colors={'TA':'#b2182b','TB':'#2166ac'}
    for x,label in enumerate(['TA','TB']):
        g=d[d.label==label].copy().sort_values(['tb_events_per_hour','sz'])
        offsets=np.array([-.30,-.10,.10,.30,-.19,.16,-.12,.22,-.22,.12]) if label=='TA' else np.array([-.09,.09])
        for offset,(_,r) in zip(offsets,g.iterrows()):
            xx=x+offset;yy=r.tb_events_per_hour;short=not r.complete
            ax.scatter(xx,yy,s=80,edgecolor=colors[label],facecolor='white' if short else colors[label],lw=1.8,zorder=4,clip_on=False)
            if yy==0:
                ax.annotate(f'SZ{int(r.sz)}',(xx,yy),xytext=(0,-10),textcoords='offset points',ha='center',va='top',fontsize=11)
            elif r.sz==12:
                ax.annotate('SZ12',(xx,yy),xytext=(-8,0),textcoords='offset points',ha='right',va='center',fontsize=11)
            else:
                ax.annotate(f'SZ{int(r.sz)}',(xx,yy),xytext=(6,6),textcoords='offset points',fontsize=11)
        ax.hlines(med[label],x-.36,x+.36,color=colors[label],lw=3,zorder=2)
        ax.text(x+.43,med[label],f'{med[label]:.1f}',color=colors[label],va='center',ha='left',fontsize=15,fontweight='bold')
    ax.plot([0,0,1,1],[285,293,293,285],c='black',lw=1)
    ax.text(.5,298,f'Two-sided exact permutation p = {test["p"]:.4f}',ha='center',va='bottom',fontsize=13)
    ax.set(xlim=(-.55,1.65),ylim=(0,320),xticks=[0,1],xticklabels=['TA-source seizure\n(n = 10)','TB-source seizure\n(n = 2)'],yticks=[0,50,100,150,200,250,300],ylabel='Preseizure TB interictal events / hour')
    ax.tick_params(axis='x',pad=34,length=0)
    ax.set_title('E1146 · Broadband-qualified seizures',pad=18,fontsize=17)
    fig.text(.15,.12,'Each dot: one seizure; horizontal line: group median.\nHollow dots: window shortened by the preceding seizure.\nRate = TB count / observed hours within the preceding ≤15 min.',fontsize=11,va='top',linespacing=1.5)
    fig.text(.15,.025,f'Exploratory contrast selected from 22 readouts; within-patient BH q = {original.q_bh22:.3f}.',fontsize=10.5,color='#444444')
    for ext in ['png','pdf','svg']:fig.savefig(fp/f'e1146_pre15_tb_rate.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    metadata=dict(subject='E1146',n_ta=10,n_tb=2,median_raw_rate=med,raw_rate_test=test,
        within_patient_bh22_q=float(original.q_bh22),source_table=str(SOURCE/'window_observations.csv'),
        original_statistic='log1p(rate) rank test; monotone transform preserves ranks; plotted medians are computed on raw rates',
        window='[max(previous seizure offset, current seizure onset - 900 s), current seizure onset]',
        unit='one broadband-qualified, clear signed-source seizure with a previous seizure',
        observation='available labelled-event artifact coverage; no-event observations remain zero, absent coverage is not zero',
        signed_label='fixed shared Timing+Space geometry, energy correlated with source=-rank; no absolute r or mirror selection',
        broadband='clinical 0-10s matched-window 5/6 bands, at least 2 low and 2 fast bands; additional baseline/window eligibility',
        blocked6_n_mixed_strata=int(original.block6_n_mixed_time_blocks),blocked12_p=float(original.block12_p),
        short_window_seizures=d.loc[~d.complete,'sz'].astype(int).tolist(),user_visual_acceptance='pending')
    (OUT/'figure_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (fp/'README.md').write_text('### e1146_pre15_tb_rate.png\n展示E1146宽频合格发作按signed source分为TA型（10次）和TB型（2次）后，发作前最多15分钟的间期TB事件率；PDF和SVG为同图矢量版。每点为一次发作并标注编号，横线为原始事件率的组内中位数，空心点表示受到上一次发作边界截短的窗口。图中双侧精确标签置换p沿用已有22项候选分析，原始率和log1p率的秩相同；单独展示不改变多重检验家族。\n**关注点**：两次TB型发作前的事件率均高于十次TA型，但TB组仅2次；每患者22项校正q=0.333，6小时分层无混合标签层，不能据此认定可推广的预测关系。图待人工检查。\n',encoding='utf-8')
    (OUT/'CAPTION.md').write_text('E1146中宽频增强合格且空间能量模式具有明确signed source标签的发作。统计窗为本次发作开始前最多15分钟，起点不早于上次发作结束；间期TB事件须完整落入窗口，事件率以实际有事件标签产物覆盖的小时数为分母。每点代表一次发作，横线为组内中位数：TA型18.0次/小时（n=10），TB型183.8次/小时（n=2）。SZ19为67个TB事件/15.0分钟（268.0次/小时）；SZ22为23个/13.85分钟（99.6次/小时）。SZ9、11、13窗口分别为11.54、4.35、13.55分钟；SZ11、13在可观测窗口内没有间期事件，零值保留。双侧精确秩置换检验枚举66种标签分配，p=0.01515；原22项候选家族内BH q=0.333。该图单独展示探索性方向，不能把事后选择的一项视作新的单次验证；6小时分层无混合标签层，无法排除记录时段差异。\n',encoding='utf-8')
    print(json.dumps({'medians':med,'p':test['p'],'q':float(original.q_bh22),'figure':str(fp/'e1146_pre15_tb_rate.png')},ensure_ascii=False))


if __name__=='__main__':main()

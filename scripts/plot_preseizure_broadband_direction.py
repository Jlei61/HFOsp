#!/usr/bin/env python3
"""Show the existing interval-half contrast without introducing a new test."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic5_preseizure_template_association_broadband/cohort_20260909'


def main():
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True,layout='constrained')
    for ax,sid in zip(axes,[1146,590,635]):
        path=OUT/'per_subject'/f'epilepsiae_{sid}'/'association'
        f=pd.read_csv(path/'seizure_features.csv')
        s=pd.read_csv(path/'feature_support_counts.csv')
        d=f[['sz','label','halves_p_ta']].merge(s[['sz','halves_p_ta']],on='sz',suffixes=('_change','_support')).dropna()
        d['delta_tb_pp']=-100*d.halves_p_ta_change
        stats=pd.read_csv(path/'association_screen.csv').set_index('feature').loc['halves_p_ta']
        labels=[]
        for x,(label,color) in enumerate([('TA','#b2182b'),('TB','#2166ac')]):
            g=d[d.label==label].sort_values('delta_tb_pp');labels.append(f'{label} source\n(n={len(g)})')
            xx=x+np.linspace(-.11,.11,len(g))
            for j,(p,(_,row)) in enumerate(zip(xx,g.iterrows())):
                ax.scatter(p,row.delta_tb_pp,s=55,facecolor=color if row.halves_p_ta_support>=20 else 'white',edgecolor=color,lw=1.5,zorder=3)
                ax.annotate(f'SZ{int(row.sz)}',(p,row.delta_tb_pp),xytext=(5,-12 if j%2==0 else 6),textcoords='offset points',fontsize=8)
            ax.plot([x-.23,x+.23],[g.delta_tb_pp.median()]*2,color=color,lw=2.5)
        ax.axhline(0,color='#999999',lw=.8,ls='--');ax.set(xticks=[0,1],xticklabels=labels,xlim=(-.4,1.5),title=f'E{sid} | effect = {-stats.rank_biserial:.2f}\nunadjusted p = {stats.p:.3f}')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('TB fraction: second half − first half\n(percentage points)')
    fig.suptitle('Does interictal composition shift toward TB before a TB-source seizure?',fontsize=13)
    fig.supxlabel('Each dot: one broadband-qualified seizure; line: median. Hollow: <20 events in either half.',fontsize=10)
    fp=OUT/'figures'
    for ext in ['png','pdf']:fig.savefig(fp/f'interval_half_tb_direction.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    readme=fp/'README.md';text=readme.read_text();entry='\n### interval_half_tb_direction.png\n每点为一次宽频合格且source明确、两个半程均有事件的发作；横线为组内中位数，空心点表示任一半程不足20个事件。纵轴为从上次发作结束至本次发作开始按实际时长等分后，后半程减前半程的TB占比变化，PDF为同图矢量版；统计直接读取原22项候选中的halves_p_ta并反转符号，没有新增检验。\n**关注点**：三个患者均有同向组间趋势，保留每半程至少20事件时方向仍一致；这不消除时段、覆盖缺口和少量发作的限制，图中p未作多重校正，未建立跨患者可推广关联。图待人工检查。\n'
    if '### interval_half_tb_direction.png' not in text:readme.write_text(text+entry)


if __name__=='__main__':main()

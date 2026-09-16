#!/usr/bin/env python3
"""Separate total interictal rate, composition, and subsequent source label."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic5_preseizure_template_association_broadband/cohort_20260909'
FIELDS=ROOT/'results/interictal_propagation_masked/template_gradient_fields_all_events_timing_plus_space/per_subject'
OUT=BASE/'total_rate_label_preference'


def main():
    OUT.mkdir(exist_ok=True)
    manifest=pd.read_csv(BASE/'cohort_manifest.csv',dtype={'subject':str}).set_index('subject')
    original=pd.read_csv(BASE/'all_patient_association_tests.csv')
    points=[];groups=[];correlations=[];audits=[]
    for path in sorted((BASE/'per_subject').glob('*/association/window_observations.csv')):
        sid=path.parts[-3].split('_')[-1];folder=path.parents[1]
        source=FIELDS/f'epilepsiae_{sid}.json';record=json.loads(source.read_text())
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest==manifest.loc[sid,'field_sha256']
        discovery=record['template_discovery']
        assert discovery['method']=='timing_plus_space_all_events_missing_view_v1'
        z=np.load(folder/'event_index.npz');idx=z['source_event_index'];labels=np.asarray(discovery['event_labels'])
        np.testing.assert_array_equal(np.sort(idx),np.arange(len(labels)))
        np.testing.assert_array_equal(z['template_label'],labels[idx])
        audited=0
        for row in pd.read_csv(path).itertuples():
            m=z['nonseizure_eligible']&(z['event_abs_time']>=row.start_epoch-1e-6)&(z['event_end_time']<=row.end_epoch+1e-6)
            assert int(sum(m&(z['template_label']==0)))==row.n_ta
            assert int(sum(m&(z['template_label']==1)))==row.n_tb
            audited+=1
        audits.append(dict(subject=sid,field=str(source),sha256=digest,n_frozen_events=len(labels),n_window_counts_verified=audited))
        d=pd.read_csv(path);d['subject']=int(sid);d['total_rate']=d.n_events/d.observed_hours
        d['tb_fraction']=d.n_tb/d.n_events.replace(0,np.nan);d['ta_rate']=d.n_ta/d.observed_hours;d['tb_rate']=d.n_tb/d.observed_hours
        points.append(d)
        for (window,label),g in d.groupby(['window','label']):
            groups.append(dict(subject=int(sid),window=window,seizure_source=label,n_seizures=len(g),n_with_events=int(g.tb_fraction.notna().sum()),
                median_total_rate=g.total_rate.median(),median_ta_rate=g.ta_rate.median(),median_tb_rate=g.tb_rate.median(),median_tb_fraction=g.tb_fraction.median() if g.tb_fraction.notna().any() else np.nan))
        for window,g in d.groupby('window'):
            for min_events in [1,20]:
                v=g[(g.n_events>=min_events)&np.isfinite(g.total_rate)&np.isfinite(g.tb_fraction)]
                rho=spearmanr(v.total_rate,v.tb_fraction).statistic if len(v)>=4 and v.total_rate.nunique()>1 and v.tb_fraction.nunique()>1 else np.nan
                correlations.append(dict(subject=int(sid),window=window,min_events=min_events,n=len(v),rho_total_rate_vs_tb_fraction=rho,
                    status='DESCRIPTIVE_NO_INFERENTIAL_P' if np.isfinite(rho) else 'NOT_ESTIMABLE',
                    note='One seizure-window per observation; no-event windows have no defined composition; threshold sensitivity is not a new primary analysis.'))
    allpoints=pd.concat(points,ignore_index=True);allpoints.to_csv(OUT/'seizure_window_readouts.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'group_summaries.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'rate_composition_correlations.csv',index=False)
    keep=original.feature.isin(['whole_log_rate_all','pre60_log_rate_all','pre15_log_rate_all'])
    original[keep].to_csv(OUT/'existing_total_rate_source_tests.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(field_family='all-event Timing+Space',checks=audits,statistical_unit='seizure within patient',
        population='broadband-qualified clear-source seizures with a preceding seizure',
        source_label='fixed common spatial operator: signed r(early energy, -rank), no abs r and no mirror optimization',
        observations='event start/end containment; no parent-block exclusion or 50 percent coverage threshold',
        label_rule='TA=more events and TB=fewer events within each patient; no common anatomical identity across patients',
        scope='Preseizure windows only; not a test of the entire interictal recording',user_visual_acceptance='pending'),ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,5.5),sharey=True)
    fig.subplots_adjust(left=.075,right=.98,bottom=.27,top=.76,wspace=.13)
    colors={'TA':'#b2182b','TB':'#2166ac'}
    for ax,sid in zip(axes,[1146,590,635]):
        d=allpoints[(allpoints.subject==sid)&(allpoints.window=='pre15')];v=d.dropna(subset=['tb_fraction'])
        for _,r in v.iterrows():
            ax.scatter(r.total_rate,100*r.tb_fraction,s=70,edgecolor=colors[r.label],facecolor=colors[r.label] if r.n_events>=20 else 'white',lw=1.5,clip_on=False)
            if sid==1146 and r.sz in [8,19,22]:
                ax.annotate(f'SZ{int(r.sz)}',(r.total_rate,100*r.tb_fraction),xytext=(-4,8),ha='center',textcoords='offset points',fontsize=10)
        if sid==1146:ax.annotate('SZ3, SZ9',(8,0),xytext=(12,10),textcoords='offset points',fontsize=9)
        ax.set(xlim=(0,float(v.total_rate.max())*1.1),ylim=(0,70),xlabel='Total interictal events / hour',title=f'E{sid} · {len(v)} nonempty windows')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('TB fraction among interictal events (%)')
    fig.legend(handles=[Line2D([],[],marker='o',color='none',markerfacecolor=colors[k],markeredgecolor=colors[k],label=k+'-source seizure',markersize=7) for k in ['TA','TB']],loc='upper center',bbox_to_anchor=(.5,.88),ncol=2,frameon=False)
    fig.suptitle('Total event rate and template composition are separate observations',fontsize=15,y=.97)
    fig.text(.075,.11,'Preceding ≤15 min; one dot per broadband-qualified seizure. Hollow: fewer than 20 interictal events.\nTwo zero-event TA-source windows in E1146 remain in rate tests but have no defined TB fraction.',fontsize=10,linespacing=1.5)
    fp=OUT/'figures';fp.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(fp/f'total_rate_vs_tb_fraction.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (fp/'README.md').write_text('### total_rate_vs_tb_fraction.png\n横轴为发作前最多15分钟TA+TB的总事件率，纵轴为该窗口间期TB事件占比，颜色表示随后发作的signed source类型；PDF为同图矢量版。全部标签已逐事件核对为空间信息参与的all-event Timing+Space冻结聚类，空心点表示少于20事件。E1146两个零事件窗口在总率比较中保留，但其组成比例未定义，不能画成0% TB。\n**关注点**：E1146、E635的TB型发作前总率中位数较高，不等于总率与TB占比存在稳定单调联系；相关性对稀疏窗口敏感，且本图只覆盖合格发作前窗口，不能外推整个记录。图待人工检查。\n',encoding='utf-8')
    report=['# 整体事件率、间期标签组成与发作source偏好','','本次没有重新聚类：逐事件验证空间信息参与的Timing+Space冻结标签、当前输入哈希以及每个时间窗口TA/TB计数。TA/TB为患者内多数/少数模板，不代表跨患者共同解剖标签。','',
        '整体率定义为(TA事件数+TB事件数)/实际观测小时数；组成定义为TB/(TA+TB)。前者可以在零事件窗口计算为0，后者不能。分析人群沿用宽频资格及明确signed source标签，统计单位为发作；新计算的总率—组成Spearman仅作描述，不把大量事件当成独立统计样本。','',
        '前15分钟（可被上次发作截短）结果：E1146 TA/TB型前总率中位数102.1/579.8次每小时；E635为136.1/272.1；E590为84.1/52.0。对应原有总率检验未校正p分别为0.167、0.486、0.800，均不构成已确立的标签偏好。','',
        '间期TB占比中位数：E1146 TA/TB型前12.1%/31.9%，E635为36.6%/44.5%，E590为50.6%/53.8%。按全体非空窗口，总率与TB占比的Spearman rho分别为0.426、0.024、-0.051；要求每窗至少20事件后，E1146为-0.119（8窗）、E635为0.600（6窗），E590仅2窗不作估计。因此不能把“较高总率”当作稳定的TB标签读出。','',
        'E1146 SZ8总率612/h、TB占比11.1%；SZ22总率307.6/h、TB占比32.4%。这个实例说明较高TB事件率可能同时包含总活动强度和组成偏移两个因素，不能仅凭总率辨认间期模式。这里的乘法分解在逐窗口层面成立，不能将分组中位数任意相乘还原另一中位数。','',
        '限制：只看合格发作前窗口，未检验整段间期记录的高率状态；发作标签在记录时段上的聚集尚不能排除。不同事件阈值下的相关性变化为敏感性观察，不选择其中较有利的结果作为主结论。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Verified subjects/windows:',len(audits),sum(a['n_window_counts_verified'] for a in audits))


if __name__=='__main__':main()

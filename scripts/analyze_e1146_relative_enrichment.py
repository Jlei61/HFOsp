#!/usr/bin/env python3
"""Descriptive source-matched enrichment over explicit retrospective baselines."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_e1146_event_time_correspondence import OUT as SOURCE, contained, outside_seizures, label_permutation
from scripts.analyze_e1146_source_signed_correspondence import BASE,FIELD,RED,BLUE
from scripts.analyze_e1146_preseizure_template_share import inventory,ARTIFACT,coverage,write_json,sha
OUT=SOURCE/'relative_enrichment'


def matched_excess(p_a,q_a,is_a):
    return np.where(is_a,1.,-1.)*(np.asarray(p_a)-np.asarray(q_a))


def balanced_enrichment(p_a,q_a,is_a):
    e=matched_excess(p_a,q_a,is_a);is_a=np.asarray(is_a,bool)
    return .5*(e[is_a].mean()+e[~is_a].mean()) if is_a.any() and (~is_a).any() else np.nan


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def run():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(SOURCE/'interval_correspondence.csv');original=json.loads((SOURCE/'summary.json').read_text())
    d=d[d.observed_pair].copy();rows=[];stats=[]
    for post in (0,60):
        b=original['baseline_by_post_exclusion'][str(post)];q=b['ta_share']
        t=d[d.exclude_post_minutes==post].copy()
        t['q_ta_global']=q
        t['q_ta_leave_window_out']=(b['n_ta']-t.n_ta)/(b['n_events']-t.n_events)
        t['matched_excess']=matched_excess(t.ta_share,q,t.label.eq('TA'))
        t['matched_excess_leave_window_out']=matched_excess(t.ta_share,t.q_ta_leave_window_out,t.label.eq('TA'))
        # Removing a window from the baseline cannot flip the enrichment sign.
        np.testing.assert_array_equal(np.sign(t.matched_excess),np.sign(t.matched_excess_leave_window_out))
        rows.append(t)
        for win,g in t.groupby('window'):
            for pool in ('all_observed','at_least_20_events'):
                v=g if pool=='all_observed' else g[g.n_events>=20]
                is_a=v.label.eq('TA');score=balanced_enrichment(v.ta_share,q,is_a)
                delta=v.loc[is_a,'ta_share'].mean()-v.loc[~is_a,'ta_share'].mean()
                np.testing.assert_allclose(score,delta/2)
                groups={lab:dict(n=int((v.label==lab).sum()),n_enriched=int(((v.label==lab)&(v.matched_excess>0)).sum()),mean_matched_excess=v.loc[v.label==lab,'matched_excess'].mean()) for lab in ('TA','TB')}
                previous=next(x for x in original['comparisons'] if x['window']==win and x['exclude_post_minutes']==post and x['pool']==pool)
                stats.append(dict(window=win,exclude_post_minutes=post,pool=pool,groups=groups,
                    balanced_enrichment=score,balanced_fraction_enriched=.5*sum(g['n_enriched']/g['n'] for g in groups.values()),
                    exact_two_sided_p=previous['exact_two_sided_p'],circular_shift_two_sided_p=previous['circular_label_shift_two_sided_p']))
    enriched=pd.concat(rows,ignore_index=True);enriched.to_csv(OUT/'relative_enrichment.csv',index=False)
    # Independent exact-event partition: earlier interval vs complete last hour.
    sql,inv=inventory();rec=json.loads(FIELD.read_text());z=np.load(BASE/'event_index.npz')
    starts=z['event_abs_time'];ends=np.full(len(starts),np.nan);labs=np.asarray(rec['template_discovery']['event_labels'])
    assert np.array_equal(rec['template_discovery']['sampled_event_indices'],z['source_event_index'])
    bm={f'{b["recording_id"]}_{b["block_no"]:04d}':b for b in sql['blocks']};ranges=[]
    for i,stem in enumerate(z['source_record_names']):
        packed=np.load(ARTIFACT/f'{stem}_packedTimes_withFreqCent.npy');m=z['source_block_id']==i;b=bm[stem]
        np.testing.assert_allclose(starts[m],b['begin_epoch']+packed[:,0],atol=1e-5,rtol=0)
        ends[m]=b['begin_epoch']+packed[:,1];ranges.append((b['begin_epoch'],b['end_epoch']))
    eligible=outside_seizures(starts,ends,inv);local=[]
    whole=enriched[(enriched.window=='whole')&(enriched.exclude_post_minutes==0)]
    for _,r in whole.iterrows():
        cut=r.end_epoch-3600
        if cut<=r.start_epoch:continue
        early=eligible&contained(starts,ends,r.start_epoch,cut);late=eligible&contained(starts,ends,cut,r.end_epoch)
        ne,nl=int(early.sum()),int(late.sum())
        pe=float(np.mean(labs[early]==0)) if ne else np.nan;pl=float(np.mean(labs[late]==0)) if nl else np.nan
        local.append(dict(sz=r.sz,label=r.label,n_early=ne,n_last60=nl,p_ta_early=pe,p_ta_last60=pl,
            delta_ta=pl-pe,matched_local_change=float(matched_excess(pl,pe,r.label=='TA')),
            earlier_hours=(cut-r.start_epoch)/3600,earlier_coverage=coverage(ranges,r.start_epoch,cut)/(cut-r.start_epoch),
            last60_coverage=coverage(ranges,cut,r.end_epoch)/3600))
    local=pd.DataFrame(local);local.to_csv(OUT/'within_interval_last_hour_change.csv',index=False)
    local_stats=[]
    for minimum in (1,20):
        v=local[(local.n_early>=minimum)&(local.n_last60>=minimum)]
        s=label_permutation(v.delta_ta,v.label.eq('TA'))
        s.update(min_events_each_side=minimum,n=len(v),n_ta=int(v.label.eq('TA').sum()),n_tb=int(v.label.eq('TB').sum()),
                 mean_matched_local_change_by_label={lab:v.loc[v.label.eq(lab),'matched_local_change'].mean() for lab in ('TA','TB')})
        local_stats.append(s)
    write_json(OUT/'summary.json',dict(global_reference=original['baseline_by_post_exclusion'],comparisons=stats,local_change_sensitivity=local_stats))
    write_json(OUT/'contract.json',dict(input=str(SOURCE/'interval_correspondence.csv'),input_sha256=sha(SOURCE/'interval_correspondence.csv'),
        primary_readout='signed matched excess: TA-source p_A-q_A; TB-source q_A-p_A, units proportion or percentage points',
        reference='event-count-weighted non-seizure patient background; retrospective, includes other times and initially own window; not a prospective or time-matched control',
        leave_window_out='same reference excluding exact target window events; sensitivity only; sign provably unchanged',
        balance='mean excess separately within each seizure label, then average the two label means; algebraically half of between-label mean TA share difference',
        inferential_reuse='constant baseline cancels from balanced contrast; original permutation and circular-shift p values unchanged',
        local_sensitivity='actual events wholly within earlier portion and last full 60min of same interval; omit windows with no observed earlier events, report n>=20 both sides separately',
        temporal_limit='within-interval change reduces a constant local level but does not control arbitrary temporal/postictal trends; observational single-patient analysis',
        labels_unchanged=True,parent_block_exclusion=False,coverage_threshold=None,user_visual_acceptance='pending'))
    plot(whole,pd.read_csv(SOURCE/'seizure_source_labels.csv'),original['baseline_by_post_exclusion']['0']['ta_share'],local)
    report(whole,stats,local_stats)
    print(json.dumps([s for s in stats if s['window']=='whole' and s['exclude_post_minutes']==0],indent=2))
    print(json.dumps(local_stats,indent=2))


def plot(w,labels,q,local):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,8.5),layout='constrained')
    w=w.sort_values(['label','sz']);y=np.arange(len(w))
    for i,(_,r) in enumerate(w.iterrows()):
        color=RED if r.label=='TA' else BLUE
        ax.barh(i,100*r.matched_excess,color=color,alpha=.85 if r.n_events>=20 else .28,height=.65)
        ax.text(100*r.matched_excess+(0.6 if r.matched_excess>=0 else -.6),i,f'{100*r.matched_excess:+.1f}',ha='left' if r.matched_excess>=0 else 'right',va='center',fontsize=10)
    ax.axvline(0,c='black',lw=.8);ax.axhline(w.label.eq('TA').sum()-.5,c='#bbbbbb',lw=.8)
    ax.set(yticks=y,yticklabels=[f'SZ{r.sz}  {r.label}  (n={r.n_events:,})' for _,r in w.iterrows()],
        ylim=(len(w)-.4,-.6),xlim=(-24,38),xlabel='Matching-template excess over its patient baseline (percentage points)',
        title=f'E1146: enrichment relative to background\nTA baseline {q:.1%}; TB baseline {1-q:.1%}')
    ax.text(.5,-.1,'Negative: depleted relative to background       Positive: enriched relative to background',transform=ax.transAxes,ha='center',fontsize=10)
    save(fig,'source_matched_relative_enrichment')
    fig,axes=plt.subplots(2,1,figsize=(13,6.8),sharex=True,layout='constrained')
    origin=labels.onset_epoch.min()
    for _,r in w.iterrows():
        l=labels[labels.sz==r.sz].iloc[0];x=(l.onset_epoch-origin)/3600
        axes[0].scatter(x,100*r.ta_share,c=RED if r.label=='TA' else BLUE,s=30+4*np.log1p(r.n_events),alpha=1 if r.n_events>=20 else .3)
        axes[0].annotate(str(r.sz),(x,100*r.ta_share),xytext=(2,5),textcoords='offset points',fontsize=8)
    axes[0].axhline(q*100,c='black',ls=':',lw=.9);axes[0].set(ylabel='Whole-interval TA share (%)',ylim=(45,108),title='Chronology: template proportions and seizure source preference')
    for _,r in labels.iterrows():
        if not np.isfinite(r.r_a):continue
        x=(r.onset_epoch-origin)/3600;color=RED if r.label=='TA' else BLUE if r.label=='TB' else '#888888'
        axes[1].scatter(x,r.r_a-r.r_b,c=color,s=35)
    axes[1].axhline(0,c='black',lw=.7);axes[1].set(xlabel='Hours since first observed seizure',ylabel='Seizure energy: r(TA) − r(TB)')
    fig.legend(handles=[Line2D([],[],marker='o',lw=0,c=RED,label='TA-source seizure'),Line2D([],[],marker='o',lw=0,c=BLUE,label='TB-source seizure')],loc='outside lower center',ncol=2,frameon=False)
    save(fig,'chronology_and_source_preference')
    (OUT/'figures/README.md').write_text('''# 相对基线富集

### source_matched_relative_enrichment.png / .pdf
每一行是一次发作，横轴为前一发作结束至本次发作前的同型间期比例减去患者同型基线；TA与TB分别使用67.7%与32.3%的非发作事件背景。正值表示相对富集，负值表示相对减少；不足20个事件显示浅色，不因覆盖率不足50%删去。
**关注点**：TB不到50%仍可以富集，TA超过50%仍可以减少。点估计不等于显著变化，尤其SZ9仅1个事件。

### chronology_and_source_preference.png / .pdf
上图按真实发作小时数显示每段TA比例，下图为同次发作早期能量的signed r差，颜色表示固定source标签。横轴保留实际时间间距，各点不连线，TA基线用点线标出。
**关注点**：TB型发作集中在中间一段时间，状态趋势可能共同影响发作标签与间期比例；患者全时段基线不是时间匹配对照。图已Agent自查，待用户目视检查。
''',encoding='utf-8')


def report(w,stats,local):
    primary=next(s for s in stats if s['window']=='whole' and s['exclude_post_minutes']==0 and s['pool']=='all_observed')
    lines=['# E1146：相对基线的同型间期事件富集','','只看是否超过50%会混淆患者TA本来占多数与发作类型对应。这里保留连续比例，先减去模式自己的基线，再按发作标签分别汇总。signed source发作标签、空间事件模板、逐事件时间归属和低覆盖率保留均不改变。','','## 读出、基线与统计单位','','对TA-source发作，富集量为 p(TA)−q(TA)；对TB-source发作，为 p(TB)−q(TB)。q取患者所有非发作事件的模式比例：TA=29,973/44,282=67.69%，TB=32.31%。单位为百分点；不按事件数制造统计重复。这是回顾性患者背景，不能当成未来预测的无泄漏基线，也不等于时间匹配对照。另给出删除当前窗口事件后的基线敏感性，方向与原基线一致。','','先在TA和TB发作内分别平均，再对两组等权，可避免13次TA、7次TB的不平衡。对于共同基线，此平衡富集量在代数上恰为两类发作前平均TA比例之差的一半，因此不会产生新的显著性：整段双侧置换p仍为0.376，循环标签平移p仍为0.75。只改变解释和展示，不把换指标当成新证据。','','## 整段结果','','|发作标签|次数|相对基线富集次数|平均同型变化|','|---|---:|---:|---:|']
    for lab,g in primary['groups'].items():lines.append(f"|{lab}|{g['n']}|{g['n_enriched']}/{g['n']}|{g['mean_matched_excess']*100:+.2f}个百分点|")
    lines += ['',f"两类等权的平均同型富集为{primary['balanced_enrichment']*100:+.2f}个百分点；两类等权的正富集比例为{primary['balanced_fraction_enriched']:.1%}。当前尚无明确的类别对应证据。患者基线本身为描述性参照，不对每个事件做二项显著性检验。",'', '例子：SZ21为TB-source，TB比例37.04%，虽然不到50%，却比32.31%基线高4.72个百分点；SZ26为TA-source，TA比例56.88%，虽然过半，却比67.69%基线低10.80个百分点。SZ17的TB仅高1.72个百分点，SZ19仅高0.16个百分点，不能把这些小正差自动解释成可靠富集。','','## 时间趋势敏感性','','TB-source发作在时间上集中。进一步比较同一间隔较早部分与最后完整60分钟，仅纳入前后均有事件的间隔；这是临近发作的相对变化，和整段相对患者背景是不同问题。用实际事件起止分别归属两段，不把跨分界事件硬分到某边。','','|每段最少事件数|TA / TB发作数|TA型同型变化|TB型同型变化|组间TA变化差的双侧置换p|','|---|---:|---:|---:|---:|']
    for s in local:
        g=s['mean_matched_local_change_by_label'];lines.append(f"|{s['min_events_each_side']}|{s['n_ta']} / {s['n_tb']}|{100*g['TA']:+.2f}个百分点|{100*g['TB']:+.2f}个百分点|{s['exact_two_sided_p']:.3f}|")
    lines += ['','局部参照能去掉同一间隔的固定背景差，但不能消除发作后恢复、昼夜趋势等所有时间变化；间隔过短或较早部分无事件的发作不能进入该比较。这里不据此声称前瞻预测或机制。','','## 交付与验证','','`relative_enrichment.csv`保留所有窗口、全患者与去掉当前窗口两套基线；`within_interval_last_hour_change.csv`给出局部前后事件数、覆盖率和变化；`summary.json`与`contract.json`保存定义和统计。图为候选版本，待用户目视检查。','','核对共同基线下平衡富集=组间比例差/2；核对移除当前窗口后富集方向不变；局部比较再次从原始packed起止时间按SQL绝对时间重建。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':run()

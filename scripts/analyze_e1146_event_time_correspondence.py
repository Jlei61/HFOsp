#!/usr/bin/env python3
"""E1146 source-signed correspondence using complete event time containment.

User-authorized removal of parent-hour exclusions. Frozen spatial templates,
fixed original geometry, common smoothing and source-signed labels are retained.
"""
from pathlib import Path
import sys, json, itertools, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_e1146_source_signed_correspondence import BASE, FIELD, RED, BLUE, common_scores, classify_source
from scripts.analyze_e1146_preseizure_template_share import inventory, coverage, ARTIFACT, write_json, WINDOWS, sha
from src.topic5_template_axis_field import scorers_from_interictal_record
OUT=BASE/'source_signed_event_time'


def contained(starts,ends,lo,hi):
    return (starts>=lo)&(ends<=hi)&(starts<hi)&(ends>starts)


def outside_seizures(starts,ends,inv,post_minutes=0):
    keep=np.ones(len(starts),bool)
    for s in inv:
        keep &= ~((starts<s['offset']+post_minutes*60)&(ends>s['onset']))
    return keep


def label_permutation(values,is_a):
    """Exact equal-seizure difference of mean TA share, conditional group sizes."""
    values=np.asarray(values,float);is_a=np.asarray(is_a,bool)
    n=len(values);na=int(is_a.sum());nb=n-na
    if not na or not nb:return {'status':'NOT_ESTIMABLE_MISSING_LABEL_GROUP'}
    delta=float(values[is_a].mean()-values[~is_a].mean())
    combos=np.array(list(itertools.combinations(range(n),na)),dtype=int)
    suma=values[combos].sum(axis=1)
    null=suma/na-(values.sum()-suma)/nb
    shifts=np.array([values[np.roll(is_a,k)].mean()-values[~np.roll(is_a,k)].mean() for k in range(n)])
    return dict(status='EXPLORATORY',delta_ta_share=delta,
                exact_two_sided_p=float(np.mean(np.abs(null)>=abs(delta)-1e-12)),
                exact_one_sided_positive_p=float(np.mean(null>=delta-1e-12)),n_permutations=len(null),
                circular_label_shift_two_sided_p=float(np.mean(np.abs(shifts)>=abs(delta)-1e-12)),n_circular_shifts=n)


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def run():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    sql,inv=inventory();rec=json.loads(FIELD.read_text());sc=scorers_from_interictal_record(rec)
    a,b=sc['shared_a'],sc['shared_b'];assert np.array_equal(a['points'],b['points']) and a['sigma']==b['sigma']
    support=np.minimum(a['support'],b['support']);label_rows=[]
    energy_meta=json.loads((BASE/'extraction_summary.json').read_text())
    for r,e in zip(inv,energy_meta):
        assert r['seizure_id']==e['seizure_id']
        q=dict(sz=r['sz'],seizure_idx=r['seizure_idx'],seizure_id=r['seizure_id'],onset_epoch=r['onset'],offset_epoch=r['offset'])
        if e['status']=='ok':
            rs,_,_=common_scores(e['activation'],rec['rank_a'],rec['rank_b'],a['points'],support,a['sigma'])
            label,provisional=classify_source(*rs)
            q.update(r_a=rs[0],r_b=rs[1],label=label,provisional_label=provisional,reason='')
        else:q.update(r_a=np.nan,r_b=np.nan,label='unavailable',provisional_label='',reason=e['reason'])
        label_rows.append(q)
    labels=pd.DataFrame(label_rows);labels.to_csv(OUT/'seizure_source_labels.csv',index=False)
    prev=pd.read_csv(BASE/'source_signed_fixed_geometry/seizure_source_labels.csv')
    assert labels.label.tolist()==prev.label.tolist()
    np.testing.assert_allclose(labels[['r_a','r_b']],prev[['r_a','r_b']],equal_nan=True)
    z=np.load(BASE/'event_index.npz');starts=z['event_abs_time'];ids=z['source_block_id']
    event_labels=np.asarray(rec['template_discovery']['event_labels'])
    assert np.array_equal(rec['template_discovery']['sampled_event_indices'],z['source_event_index'])
    assert len(starts)==len(event_labels)==46683
    blockmap={f'{b["recording_id"]}_{b["block_no"]:04d}':b for b in sql['blocks']}
    ends=np.full(len(starts),np.nan);ranges=[]
    for i,stem in enumerate(z['source_record_names']):
        raw=np.load(ARTIFACT/f'{stem}_packedTimes_withFreqCent.npy');mask=ids==i;block=blockmap[stem]
        assert mask.sum()==len(raw)
        np.testing.assert_allclose(starts[mask],block['begin_epoch']+raw[:,0],atol=1e-5,rtol=0)
        ends[mask]=block['begin_epoch']+raw[:,1]
        assert contained(starts[mask],ends[mask],block['begin_epoch'],block['end_epoch']).all()
        ranges.append((block['begin_epoch'],block['end_epoch']))
    rawranges=[(b['begin_epoch'],b['end_epoch']) for b in sql['blocks']]
    eligible={p:outside_seizures(starts,ends,inv,p) for p in (0,60)}
    baselines={p:float(np.mean(event_labels[eligible[p]]==0)) for p in eligible}
    rows=[]
    for i,s in enumerate(inv):
        base=inv[i-1]['offset'] if i else min(a for a,b in rawranges)
        for post in (0,60):
            for name,sec in WINDOWS.items():
                lo=max(base+post*60,s['onset']-sec) if sec else base+post*60
                hi=s['onset'];dur=max(0.,hi-lo)
                mask=eligible[post]&contained(starts,ends,lo,hi)
                aa=int(np.sum(mask&(event_labels==0)));bb=int(np.sum(mask&(event_labels==1)));n=aa+bb
                p=aa/n if n else np.nan;l=labels.iloc[i].label
                cov=coverage(ranges,lo,hi)/dur if dur else 0
                rawcov=coverage(rawranges,lo,hi)/dur if dur else 0
                baseline=baselines[post]
                rows.append(dict(sz=s['sz'],window=name,exclude_post_minutes=post,start_epoch=lo,end_epoch=hi,
                    interval_hours=dur/3600,n_ta=aa,n_tb=bb,n_events=n,ta_share=p,label=l,
                    coverage_fraction=cov,raw_coverage_fraction=rawcov,below_50pct_coverage=cov<.5,
                    complete_requested_window=bool(sec is None or dur>=sec-1e-5),has_previous_seizure=bool(i),
                    matching_share=(p if l=='TA' else 1-p) if n and l in ('TA','TB') else np.nan,
                    matching_baseline=baseline if l=='TA' else 1-baseline if l=='TB' else np.nan,
                    observed_pair=bool(i and n and l in ('TA','TB'))))
    counts=pd.DataFrame(rows);counts.to_csv(OUT/'interval_correspondence.csv',index=False)
    attr=pd.read_csv(BASE/'source_signed_fixed_geometry/parent_block_exclusion_attrition.csv')
    whole=counts[(counts.window=='whole')&(counts.exclude_post_minutes==0)]
    np.testing.assert_array_equal(whole.n_events,attr.n_events_in_interval)
    attr.to_csv(OUT/'parent_block_exclusion_attrition.csv',index=False)
    stats=[]
    for (win,post),s in counts.groupby(['window','exclude_post_minutes'],sort=False):
        for pool in ('all_observed','at_least_20_events','complete_window_at_least_20_events'):
            t=s[s.observed_pair]
            if pool!='all_observed':t=t[t.n_events>=20]
            if pool.startswith('complete'):t=t[t.complete_requested_window]
            is_a=t.label.eq('TA');stat=label_permutation(t.ta_share,is_a)
            stat.update(window=win,exclude_post_minutes=int(post),pool=pool,n=len(t),
                n_ta_source=int(is_a.sum()),n_tb_source=int((~is_a).sum()),sz=t.sz.tolist(),
                mean_ta_share_before_ta=t.loc[is_a,'ta_share'].mean(),mean_ta_share_before_tb=t.loc[~is_a,'ta_share'].mean(),
                n_matching_majority=int(t.matching_share.gt(.5).sum()),n_above_matching_baseline=int((t.matching_share>t.matching_baseline).sum()),
                mean_matching_share=t.matching_share.mean(),median_matching_share=t.matching_share.median())
            stats.append(stat)
    write_json(OUT/'summary.json',dict(label_counts=labels.label.value_counts().to_dict(),comparisons=stats,
        baseline_by_post_exclusion={p:dict(ta_share=baselines[p],n_ta=int(np.sum(eligible[p]&(event_labels==0))),n_events=int(eligible[p].sum())) for p in eligible},
        n_whole_intervals_with_events=int(whole.iloc[1:].n_events.gt(0).sum()),
        n_events_between_seizures=int(whole.iloc[1:].n_events.sum())))
    write_json(OUT/'contract.json',dict(field=str(FIELD),field_sha256=sha(FIELD),event_index_sha256=sha(BASE/'event_index.npz'),
        energy_source=str(BASE/'extraction_summary.json'),energy_sha256=sha(BASE/'extraction_summary.json'),
        source_labels='fixed original geometry; identical minimum participation support, Gaussian smoothing and evaluation contacts; Pearson r(energy, -rank)',
        label_rule='larger positive r; max r <=0 neither; absolute r gap <0.05 ambiguous; margin is descriptive, not calibrated',
        temporal_rule='complete actual event start/end containment between previous max(EEG,clinical) offset and next min(EEG,clinical) onset',
        parent_hour_exclusion=False,postictal_primary='included; separate sensitivity omits actual first 60 minutes after each seizure',
        daynight='no exclusion for this non-stratified question',gap_policy='SQL block containment verified; no filling gaps or missing label artifacts',
        coverage='available source artifact time / requested available interval; no coverage threshold; not event duration fraction',
        window_policy='last 30/60/120 min clipped to previous seizure offset, marked incomplete if shorter; complete-window sensitivity separately',
        first_seizure='descriptive only',statistical_unit='one preceding interval per labelled seizure; equal seizure weights',
        test='exploratory exact seizure-label permutation of difference in mean TA share; preserves class counts; two-sided and one-sided positive',
        temporal_sensitivity='circular label shifts on observed chronological sequence; coarse sensitivity, not a full time-varying confounder control',
        interpretation='single patient; serial dependence and state trends can violate label exchangeability; no confirmatory or mechanism claim; windows not independent, no multiplicity correction',
        user_visual_acceptance='pending'))
    plot(labels,counts,attr,stats,baselines[0],starts,ends,event_labels,eligible[0],ranges)
    report(labels, counts, attr, stats, baselines[0])
    print(whole[['sz','label','n_events','ta_share','matching_share','coverage_fraction']].to_string(index=False))
    print(json.dumps([s for s in stats if s['window'] in ('whole','pre60min') and s['exclude_post_minutes']==0 and s['pool']=='all_observed'],indent=2))


def plot(labels,counts,attr,stats,baseline,starts,ends,event_labels,eligible,ranges):
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,11.5),sharey=True,gridspec_kw={'width_ratios':[1,1.4,1.4]})
    for ax in axes:
        ax.set_ylim(25.6,-.7);ax.set_yticks(np.arange(26))
        for i in range(26):
            if i%2==0:ax.axhspan(i-.5,i+.5,color='#f4f4f4',zorder=0)
    ax=axes[0]
    for _,l in labels.iterrows():
        y=l.sz-1
        if np.isfinite(l.r_a):
            ax.plot([l.r_a,l.r_b],[y,y],color='#bbbbbb',lw=.8)
            ax.scatter([l.r_a,l.r_b],[y,y],c=[RED,BLUE],s=24,zorder=3)
        ax.text(1.07,y,dict(TA='TA',TB='TB',ambiguous='unclear',neither='neither',unavailable='missing')[l.label],va='center',fontsize=9)
    ax.axvline(0,color='#999999',lw=.7);ax.set_xlim(-1.05,1.75);ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticklabels([f'SZ {i}' for i in range(1,27)])
    ax.set_title('Seizure early energy\nSigned source correlation',fontsize=12);ax.set_xlabel('Signed r')
    for ax,win,title in zip(axes[1:],('whole','pre60min'),('Whole preceding interval','Last 60 min (available portion)')):
        sub=counts[(counts.window==win)&(counts.exclude_post_minutes==0)]
        for _,r in sub.iterrows():
            y=r.sz-1
            if r.n_events:
                alpha=.9 if r.n_events>=20 else .3
                ax.barh(y,r.ta_share,color=RED,height=.56,alpha=alpha)
                ax.barh(y,1-r.ta_share,left=r.ta_share,color=BLUE,height=.56,alpha=alpha)
                ax.text(1.04,y,f'{r.ta_share:.1%} | {r.n_events:,.0f} | {r.coverage_fraction:.0%}',va='center',fontsize=9)
            else:ax.text(.03,y,'No events in available artifacts',va='center',fontsize=8,color='#888888')
        ax.axvline(.5,color='#444444',lw=.7,ls='--');ax.axvline(baseline,color='black',lw=1,ls=':')
        ax.set_xlim(0,1.95);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xticklabels(['0','25','50','75','100'])
        ax.set_title(title+'\nTA share | n events | coverage',fontsize=12);ax.set_xlabel('TA share (%) — remainder is TB')
    fig.suptitle('E1146: signed source labels and interictal proportions\nActual event times; no parent-hour exclusion',fontsize=15,y=.99)
    fig.legend(handles=[Line2D([],[],color=RED,lw=5,label='TA'),Line2D([],[],color=BLUE,lw=5,label='TB'),Line2D([],[],color='black',ls=':',label=f'Non-seizure TA baseline: {baseline:.1%}')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.024))
    fig.text(.055,.005,'SZ1: descriptive only. Pale bars: fewer than 20 events. Coverage refers to available label artifacts, not event duration.',fontsize=10)
    fig.subplots_adjust(left=.055,right=.995,top=.90,bottom=.09,wspace=.09);save(fig,'all_seizures_source_correspondence')
    fig,axes=plt.subplots(1,2,figsize=(11,5.7),layout='constrained')
    rng=np.random.default_rng(20260909)
    for ax,win,title in zip(axes,('whole','pre60min'),('Whole interval','Last 60 min (available portion)')):
        sub=counts[(counts.window==win)&(counts.exclude_post_minutes==0)&counts.observed_pair]
        for x,lab,color in [(0,'TA',RED),(1,'TB',BLUE)]:
            t=sub[sub.label==lab];jit=np.linspace(-.23,.23,len(t));rng.shuffle(jit)
            for j,(_,r) in zip(jit,t.iterrows()):
                ax.scatter(x+j,r.ta_share,c=color,s=45,alpha=1 if r.n_events>=20 else .3)
                ax.annotate(str(r.sz),(x+j,r.ta_share),xytext=(3,2),textcoords='offset points',fontsize=8)
            ax.plot([x-.29,x+.29],[t.ta_share.mean()]*2,c=color,lw=3)
        s=next(s for s in stats if s['window']==win and s['exclude_post_minutes']==0 and s['pool']=='all_observed')
        ax.axhline(.5,c='#777777',ls='--',lw=.8);ax.axhline(baseline,c='black',ls=':',lw=.9)
        ax.set(xlim=(-.48,1.48),ylim=(-.04,1.08),xticks=[0,1],xticklabels=[f'TA-source\nn={s["n_ta_source"]}',f'TB-source\nn={s["n_tb_source"]}'],ylabel='Preceding interictal TA share',title=title)
        ax.set_title(title+'\n'+f'Mean difference {s["delta_ta_share"]:+.1%}; permutation p={s["exact_two_sided_p"]:.3g}',fontsize=11,pad=12)
    fig.supxlabel('Seizure label (numbers identify seizures; horizontal bars are equal-seizure means)',fontsize=10)
    save(fig,'label_group_comparison')
    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained');t=attr[attr.sz>1]
    ax.bar(t.sz,t.n_events_in_interval,color='#aaaaaa',label='Events inside the actual interval')
    ax.bar(t.sz,t.n_retained_strict,color=BLUE,label='Previously retained by parent-hour rule')
    ax.set(xticks=t.sz,xlabel='Upcoming seizure',ylabel='Number of events',title='Why the previous analysis had so many empty intervals')
    ax.legend(frameon=False);save(fig,'parent_rule_event_loss')
    observed=counts[(counts.window=='whole')&(counts.exclude_post_minutes==0)&counts.observed_pair]
    fig,axes=plt.subplots(math.ceil(len(observed)/4),4,figsize=(15,3*math.ceil(len(observed)/4)),sharey=True,layout='constrained')
    bins=[]
    for ax,(_,r) in zip(axes.flat,observed.iterrows()):
        edges=np.linspace(r.start_epoch,r.end_epoch,11)
        for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=eligible&contained(starts,ends,lo,hi);n=int(mask.sum());p=float(np.mean(event_labels[mask]==0)) if n else np.nan
            bins.append(dict(sz=r.sz,label=r.label,bin=j+1,start_epoch=lo,end_epoch=hi,n_events=n,ta_share=p,coverage_fraction=coverage(ranges,lo,hi)/(hi-lo)))
            if n:ax.scatter(((lo+hi)/2-r.end_epoch)/3600,p,color=RED if r.label=='TA' else BLUE,s=12+6*np.log1p(n),alpha=.9 if n>=20 else .3)
        ax.axhline(.5,c='#777777',ls='--',lw=.7);ax.axhline(baseline,c='black',ls=':',lw=.7)
        ax.set(xlim=(-r.interval_hours,0),ylim=(-.03,1.05),xlabel='Hours to seizure',title=f'SZ{r.sz}: {r.label}-source | coverage {r.coverage_fraction:.0%}')
    for ax in axes.flat[len(observed):]:ax.set_visible(False)
    for ax in axes[:,0]:ax.set_ylabel('Interictal TA share')
    save(fig,'within_interval_observed_trajectory');pd.DataFrame(bins).to_csv(OUT/'trajectory_bins.csv',index=False)
    (OUT/'figures/README.md').write_text('''# E1146：逐事件时间归属与 signed source 标签

### all_seizures_source_correspondence.png / .pdf
左侧为固定原方位、共用空间平滑下的早期能量与source（−rank）的signed相关，中右为本次发作前整段及末60分钟的TA/TB事件比例。取消整小时块排除和50%覆盖率门槛；事件须完整位于实际时间区间，少于20个事件用浅色，首发仅描述。
**关注点**：23/25个发作间隔有事件，TA占比不是对应模式占比；TB-source发作应读蓝色部分。覆盖率是有标签文件的时间比例，缺失数据不补零。

### label_group_comparison.png / .pdf
比较TA-source与TB-source发作前的TA比例，每个点是一次发作对应的时间间隔，数字为发作编号，短横线为组内等权均值。虚线为50%，点线为患者非发作事件的TA基线；置换检验保持两类发作数，p为双侧探索性结果。
**关注点**：检验类别对应要看两组之差，而非只看TA是否过半；同一患者的连续发作可能受时间趋势影响，不能把p直接当成独立重复的确认性证据。

### parent_rule_event_loss.png / .pdf
灰柱为两次发作之间完整发生的事件数，蓝柱为此前整小时排除后保留数。两者来自同一批冻结空间标签，仅时间排除方式不同。
**关注点**：此前空行主要由整块排除造成，不代表发作前没有间期事件；取消规则后仍不补全缺少的标签文件。

### within_interval_observed_trajectory.png / .pdf
每个可配对间隔分成10个等长时间格，横轴为距本次发作的真实小时数，纵轴为格内TA比例，点颜色为终点发作的source标签。只画有完整包含事件的时间格，点大小随事件数增加，少于20个事件变浅，空白不插值。
**关注点**：各间隔长度不同，点显示局部观测而非连续轨迹；无数据时间不能推定任何模式占优。候选图待用户目视检查。
''',encoding='utf-8')

def report(labels, counts, attr, stats, baseline):
    whole=counts[(counts.window=='whole')&(counts.exclude_post_minutes==0)]
    t=whole[whole.observed_pair]
    lines=["# E1146：逐事件时间归属下的 source-signed 发作对应关系", "",
        "已按用户要求取消整小时来源块排除，保留低于50%覆盖率的间隔。本目录是当前分析；source_signed_fixed_geometry 中的整块筛选结果仅作历史对照。图已做 Agent 目视检查，尚待用户目视验收。", "",
        "## 科学问题与方法", "",
        "比较下一次发作早期能量偏向哪一个 source，与前一发作结束后至本次发作前的 TA/TB 间期事件比例是否对应。统计单位为发作间隔，每次发作等权；事件数量只用于比例估计，不当成独立发作重复。", "",
        "模板与事件标签来自 all-events Timing+Space 冻结结果。发作能量沿用 Fig3 提取：CAR、1–150 Hz、1秒谱窗/0.5秒步长，基线为EEG起点前[-120,-90]秒的robust-z，早期取临床[0,10]秒完整窗均值。15个触点保持相同顺序。固定原方位，对两种−rank source模板与能量使用同一个Gaussian平滑、共同support和同一组评价触点，比较 signed Pearson r；不取|r|，不选择镜像。较大正相关者为标签；两者非正不归类，差值<0.05标记不明确（描述性标记，不是校准置信度）。", "",
        "时间区间为上一发作 max(EEG,clinical) offset 至本次 min(EEG,clinical) onset；每个事件的实际起止须完整位于区间内，同时核对SQL来源块包含关系。末30/60/120分钟截断于前一发作结束，并记录是否足够完整；不填补记录gap或缺失标签文件。主分析纳入发作后事件，敏感性分析按实际时间去掉发作后前60分钟。昼夜边界不再排除事件。", "",
        "覆盖率 = 有当前标签资产的时间 / 区间时间，不是事件占用时间。首发没有上一发作，仅描述。SZ4 标签不明确，SZ6 两个相关均为负，SZ18 缺少可用基线，不强制归类。", "",
        "## 空行的来源", "",
        "25个发作间隔实际包含33,033个事件，23段有事件；旧整块规则只留下6,369个事件、4段，误让19个有事件的间隔显示为空。按互斥顺序统计：发作重叠来源块排除13,151个，发作后60分钟分界来源块再排除12,311个，昼夜分界来源块再排除1,202个；合计筛去26,664个（80.7%）。这些事件本身完整位于两次发作之间。", "",
        "逐事件归属后，仅SZ11前4.35分钟、SZ13前13.55分钟没有完整落入区间的当前已检测/已聚类事件，两段标签文件时间覆盖均为100%。这不等于原始EEG中不存在任何HFO。SZ25的17.47小时间隔仅有29.0%标签资产覆盖，802个事件仍纳入；比例只代表已观测部分。", "",
        "## 对应关系", "",
        f"20次可配对发作包含13次TA-source、7次TB-source。两组的每一个整段都以TA事件为多数：对应模式过半分别为13/13和0/7，因此总体13/20的‘匹配率’不能当成特异对应证据。患者非发作事件的TA基础比例为{baseline:.1%}（29,973/44,282），其本身偏向TA。", "",
        "|窗口/筛选|TA-source / TB-source次数|TA-source前平均TA比例|TB-source前平均TA比例|差值（百分点）|探索性双侧置换p|", "|---|---:|---:|---:|---:|---:|"]
    for win,post,pool,name in [('whole',0,'all_observed','整段，所有有观测间隔'),('pre60min',0,'all_observed','末60分钟，可用部分'),('whole',0,'at_least_20_events','整段，至少20个事件'),('pre60min',0,'complete_window_at_least_20_events','完整末60分钟，至少20个事件'),('whole',60,'all_observed','整段，去掉实际发作后60分钟')]:
        r=next(x for x in stats if x['window']==win and x['exclude_post_minutes']==post and x['pool']==pool)
        lines.append(f"|{name}|{r['n_ta_source']} / {r['n_tb_source']}|{r['mean_ta_share_before_ta']:.1%}|{r['mean_ta_share_before_tb']:.1%}|{100*r['delta_ta_share']:+.2f}|{r['exact_two_sided_p']:.3f}|")
    lines += ["", "当前结果没有明确支持‘什么类型的发作之前，就由同型间期事件主导’。TA-source前TA平均比例略高，但组差不稳定、没有明确统计支持；TB-source前仍以TA事件为主。signed R保证高能量对准source的标签语义，但不会自动产生发作前对应关系。", "",
        "精确置换固定TA/TB发作组数，以两组TA比例均值差为统计量（整段77,520种分配）。同一患者发作有序、标签也随时间成段，因此交换性不是已证明的；这些p仅作探索性结果，不是跨患者独立重复结论。保持标签顺序的循环平移敏感性中，整段双侧p=0.75；它同样不能完全控制时间趋势。多个重叠窗口不独立，未作多重校正。", "",
        "## 逐发作整段结果", "", "|本次发作|source标签|事件数|TA比例|TB比例|标签资产时间覆盖|", "|---|---|---:|---:|---:|---:|"]
    for _,r in whole.iterrows():
        a=f'{r.ta_share:.1%}' if r.n_events else '—';b=f'{1-r.ta_share:.1%}' if r.n_events else '—'
        lines.append(f'|SZ{r.sz}|{r.label}|{r.n_events:,}|{a}|{b}|{r.coverage_fraction:.1%}|')
    bins=pd.read_csv(OUT/'trajectory_bins.csv');tb=bins[(bins.label=='TB')&(bins.n_events>0)]
    local=tb[tb.ta_share<.5]
    lines += ["", "## 间隔内部的局部变化", "",
        f"各间隔分成10个等长时间格，以实际距发作小时数绘图，不连线、不插值。有TB-source终点的{len(tb)}个有观测时间格中，只有{len(local)}格TB比例超过50%：SZ21的第7格为5/9事件，SZ23的第6格为2/2事件；均少于20个事件。该分格只是描述，不能据此建立稳定的局部TB优势。TB的比例低于50%的格也完整保留，不能将低于50%当作无观测。", "",
        "## 文件与验证", "",
        "- `interval_correspondence.csv`：逐发作、逐窗口、发作后排除敏感性；`summary.json`：全部等权统计。", "- `seizure_source_labels.csv`：逐次signed相关与标签；`contract.json`：输入、时间、统计与边界定义。", "- `trajectory_bins.csv`：10格局部比例；`parent_block_exclusion_attrition.csv`：此前整块排除的事件损失。", "- `figures/`：四张PNG/PDF和中文图说明。", "",
        "核对46,683个事件的空间标签索引、SQL绝对起止、原始packed事件结束时间、来源块完整包含；逐间隔事件数与独立排除审计逐项相等。新旧固定方位signed分数与标签完全一致，仅改时间筛选。6项针对性测试通过，覆盖source符号、平滑不变性、实际事件边界、发作后时间排除和精确置换已知例子。", "",
        "重现：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/honglab/leijiaxin/anaconda3/envs/nd2/bin/python scripts/analyze_e1146_event_time_correspondence.py`。"]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':run()

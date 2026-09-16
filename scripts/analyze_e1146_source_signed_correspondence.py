#!/usr/bin/env python3
"""Fixed-orientation source/energy labels; retain low-coverage observed intervals.

Single-patient exploratory readout. Existing parent-block exclusions remain in
force; the 50% coverage filter is removed, as requested by the user.
"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.topic5_contact_similarity import kernel_smooth_at_contacts
from src.topic5_template_axis_field import scorers_from_interictal_record
from scripts.analyze_e1146_preseizure_template_share import write_json, inventory, coverage, ARTIFACT

BASE=ROOT/'results/topic5_preseizure_template_share/epilepsiae_1146'
OUT=BASE/'source_signed_fixed_geometry'
FIELD=ROOT/'results/interictal_propagation_masked/template_gradient_fields_all_events_timing_plus_space/per_subject/epilepsiae_1146.json'
GAP_FLAG=.05  # descriptive separation flag, not calibrated confidence
RED,BLUE='#B2182B','#2166AC'


def classify_source(ra,rb):
    if not np.isfinite([ra,rb]).all():return 'unavailable',''
    if max(ra,rb)<=0:return 'neither',''
    winner='TA' if ra>rb else 'TB'
    if abs(ra-rb)<GAP_FLAG:return 'ambiguous',winner
    return winner,winner


def common_scores(activation,rank_a,rank_b,points,support,sigma):
    """Use one identical operator for energy and both -rank templates."""
    smooth=lambda v:kernel_smooth_at_contacts(v,points,points,support,sigma)
    energy=smooth(activation)
    templates=[smooth(-np.asarray(r,float)) for r in (rank_a,rank_b)]
    return np.array([np.corrcoef(t,energy)[0,1] for t in templates]),energy,templates


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def run():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    rec=json.loads(FIELD.read_text());sc=scorers_from_interictal_record(rec)
    a,b=sc['shared_a'],sc['shared_b']
    assert np.array_equal(a['points'],b['points']) and a['sigma']==b['sigma']
    points=a['points'];support=np.minimum(a['support'],b['support']);sigma=a['sigma']
    assert np.all(support>0)
    _,inv=inventory()
    energy_metadata=json.loads((BASE/'extraction_summary.json').read_text())
    label_rows=[];vectors={}
    for r,e in zip(inv,energy_metadata):
        assert r['seizure_id']==e['seizure_id']
        q=dict(sz=r['sz'],seizure_idx=r['seizure_idx'],seizure_id=r['seizure_id'],
               onset_epoch=r['onset'],offset_epoch=r['offset'],status=e['status'])
        if e['status']!='ok':
            q.update(label='unavailable',provisional_label='',reason=e['reason'])
        else:
            assert e['extraction']['n_finite_contacts']==len(rec['names'])==15
            (ra,rb),energy,tpls=common_scores(e['activation'],rec['rank_a'],rec['rank_b'],points,support,sigma)
            label,provisional=classify_source(ra,rb)
            q.update(r_a=ra,r_b=rb,margin=abs(ra-rb),label=label,provisional_label=provisional,
                     reason='uncalibrated gap <0.05' if label=='ambiguous' else ('both correlations nonpositive' if label=='neither' else ''))
            vectors[str(r['sz'])]=dict(raw_energy=e['activation'],common_smoothed_energy=energy)
        label_rows.append(q)
    labels=pd.DataFrame(label_rows);labels.to_csv(OUT/'seizure_source_labels.csv',index=False)
    write_json(OUT/'scoring_vectors.json',dict(contact_order=rec['names'],support=support,sigma=sigma,
               points=points,template_a=tpls[0],template_b=tpls[1],seizures=vectors))
    z=np.load(BASE/'event_index.npz');times=z['event_abs_time'];eligible=z['strict_eligible'];blockids=z['source_block_id']
    ix=np.asarray(rec['template_discovery']['sampled_event_indices']);event_labels=np.asarray(rec['template_discovery']['event_labels'])
    assert np.array_equal(ix,z['source_event_index']) and len(event_labels)==len(times)==46683
    ends=np.full(len(times),np.nan)
    for i,stem in enumerate(z['source_record_names']):
        packed=np.load(ARTIFACT/f'{stem}_packedTimes_withFreqCent.npy');mask=blockids==i
        assert mask.sum()==len(packed)
        # Infer the common block offset from the already SQL-verified starts,
        # and independently require every event to give the identical offset.
        offsets=times[mask]-packed[:,0]
        assert np.ptp(offsets)<1e-5
        ends[mask]=offsets[0]+packed[:,1]
    assert np.all(ends>times)
    total_a=int(np.sum(eligible&(event_labels==0)));total_n=int(eligible.sum())
    baseline=total_a/total_n;full_baseline=np.mean(event_labels==0)
    old=pd.read_csv(BASE/'template_label_audit/corrected_strict_interval_shares.csv')
    rows=[]
    for _,r in old.iterrows():
        l=labels[labels.sz.eq(r.sz)].iloc[0]
        mask=eligible&(times>=r.start_epoch)&(ends<=r.end_epoch)&(times<r.end_epoch)
        aa=int(np.sum(mask&(event_labels==0)));bb=int(np.sum(mask&(event_labels==1)));n=aa+bb
        p=aa/n if n else np.nan
        q=dict(sz=r.sz,seizure_idx=r.seizure_idx,window=r.window,exclude_post_minutes=r.exclude_post_minutes,
               interval_hours=r.interval_hours,coverage_fraction=r.strict_coverage_fraction,
               start_epoch=r.start_epoch,end_epoch=r.end_epoch,n_ta=aa,n_tb=bb,n_events=n,
               ta_share=p,tb_share=1-p if n else np.nan,label=l.label,r_a=l.get('r_a'),r_b=l.get('r_b'),
               has_previous_seizure=r.sz>1,complete_requested_window=r.complete_requested_window,
               below_50pct_coverage=r.strict_coverage_fraction<.5,low_count=n<20,
               matching_share=(p if l.label=='TA' else 1-p) if n and l.label in ('TA','TB') else np.nan,
               observed_pair=bool(n and r.sz>1 and l.label in ('TA','TB')))
        outside_a=total_a-aa;outside_n=total_n-n
        q['leave_window_out_ta_baseline']=outside_a/outside_n if outside_n else np.nan
        q['ta_excess_over_same_filter_baseline']=p-baseline if n else np.nan
        q['ta_excess_over_full_discovery_baseline']=p-full_baseline if n else np.nan
        rows.append(q)
    counts=pd.DataFrame(rows);counts.to_csv(OUT/'interval_correspondence.csv',index=False)
    summaries=[]
    for (window,post),s in counts.groupby(['window','exclude_post_minutes'],sort=False):
        s=s[s.observed_pair]
        for pool in ('all_observed','at_least_20_events'):
            t=s if pool=='all_observed' else s[s.n_events>=20]
            na=int(t.label.eq('TA').sum());nb=int(t.label.eq('TB').sum())
            summaries.append(dict(window=window,exclude_post_minutes=int(post),pool=pool,n=int(len(t)),n_ta_source=na,n_tb_source=nb,
                                  sz=t.sz.tolist(),n_matching_majority=int(t.matching_share.gt(.5).sum()),
                                  mean_matching_share=t.matching_share.mean(),median_matching_share=t.matching_share.median(),
                                  mean_ta_share=t.ta_share.mean(),n_above_same_filter_baseline=int(t.ta_share.gt(baseline).sum()),
                                  n_above_full_discovery_baseline=int(t.ta_share.gt(full_baseline).sum()),
                                  specificity_status='NOT_ESTIMABLE_MISSING_LABEL_GROUP' if not na or not nb else 'requires_test',
                                  label_specificity_p=None))
    assert all(s['specificity_status']=='NOT_ESTIMABLE_MISSING_LABEL_GROUP' for s in summaries)
    write_json(OUT/'summary.json',dict(label_counts=labels.label.value_counts().to_dict(),
                 ambiguous_is_provisional_not_confidence_interval=True,
                 same_filter_baseline=dict(n_ta=total_a,n_tb=total_n-total_a,ta_share=baseline),
                 discovery_baseline=dict(n_ta=int(sum(event_labels==0)),n_tb=int(sum(event_labels==1)),ta_share=full_baseline),
                 comparisons=summaries))
    write_json(OUT/'contract.json',dict(question='Does source-aligned seizure energy correspond to preceding interictal TA/TB proportions?',
                 field=str(FIELD),field_sha256=hashlib.sha256(FIELD.read_bytes()).hexdigest(),
                 fixed_original_geometry=True,mirror_selection=False,absolute_correlation_selection=False,
                 source_scalar='negative frozen dense rank',common_support='elementwise minimum of TA and TB participation support',
                 common_operator='shared-plane Gaussian kernel evaluated at the same 15 contacts; frozen shared sigma',
                 classification='larger positive r; both nonpositive -> neither; abs difference <0.05 -> ambiguous with provisional winner',
                 separation_flag='0.05 is descriptive, not calibrated confidence',
                 coverage_filter_removed=True,count_display='all observed events including n<20; n>=20 sensitivity separately',
                 first_seizure='descriptive only, excluded from pair summaries',
                 parent_block_policy='unchanged existing strict exclusions; low coverage does not restore excluded blocks',
                 pooling='equal seizure weights; no event-level binomial pseudoreplication',
                 association_test='not estimable without observed TA-source and TB-source groups',
                 baseline_definition='all events passing the SAME parent-block exclusions; discovery-universe baseline separately reported',
                 user_visual_acceptance='pending'))

    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    # All seizures are retained so missing observations cannot hide TB cases.
    fig,axes=plt.subplots(1,3,figsize=(14.8,11.5),sharey=True,gridspec_kw={'width_ratios':[1,1.3,1.3]})
    for ax in axes:
        ax.set_ylim(25.6,-.7);ax.set_yticks(np.arange(26))
        for i in range(26):
            if i%2==0:ax.axhspan(i-.5,i+.5,color='#f4f4f4',zorder=0)
    ax=axes[0]
    for _,l in labels.iterrows():
        y=l.sz-1
        if l.status=='ok':
            ax.plot([l.r_a,l.r_b],[y,y],color='#bbbbbb',lw=.8)
            ax.scatter([l.r_a,l.r_b],[y,y],c=[RED,BLUE],s=22,zorder=3)
        ax.text(1.08,y,{'TA':'TA','TB':'TB','ambiguous':'unclear','neither':'neither','unavailable':'missing'}[l.label],va='center',fontsize=8)
    ax.axvline(0,color='#999999',lw=.7);ax.set_xlim(-1.05,1.72);ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticklabels([f'SZ {i}' for i in range(1,27)])
    ax.set_title('Fixed original orientation\nSigned source–energy correlation',fontsize=12);ax.set_xlabel('Signed r')
    for ax,win,title in zip(axes[1:],('whole','pre60min'),('Whole preceding interval','Last 60 min (available portion)')):
        sub=counts[(counts.window==win)&(counts.exclude_post_minutes==0)]
        for _,r in sub.iterrows():
            y=r.sz-1
            if r.n_events:
                alpha=.9 if r.n_events>=20 else .28
                ax.barh(y,r.ta_share,color=RED,height=.56,alpha=alpha)
                ax.barh(y,1-r.ta_share,left=r.ta_share,color=BLUE,height=.56,alpha=alpha)
                ax.text(1.04,y,f'{r.ta_share:.1%} | {r.n_events:,.0f} | {r.coverage_fraction:.0%}',va='center',fontsize=8)
            else:ax.text(.03,y,'No admissible observations',va='center',fontsize=8,color='#888888')
        ax.axvline(.5,color='#444444',lw=.7,ls='--');ax.axvline(baseline,color='black',lw=1,ls=':')
        ax.set_xlim(0,1.9);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xticklabels(['0','25','50','75','100'])
        ax.set_title(title+'\nTA share | n events | coverage',fontsize=12);ax.set_xlabel('TA share (%) — remainder is TB')
    fig.suptitle('E1146: source-aligned seizure labels versus interictal templates\nLow-coverage intervals retained',fontsize=15,y=.99)
    fig.legend(handles=[Line2D([],[],color=RED,lw=5,label='TA'),Line2D([],[],color=BLUE,lw=5,label='TB'),
                        Line2D([],[],color='black',ls=':',label=f'Same-filter TA baseline: {baseline:.1%}')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.022))
    fig.text(.055,.005,'SZ1 has no previous seizure. Pale bars have fewer than 20 events. Blank rows are missing observations, not a 0% template share.',fontsize=9)
    fig.subplots_adjust(left=.055,right=.995,top=.90,bottom=.09,wspace=.09)
    save(fig,'all_seizures_source_correspondence')

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,win,title in zip(axes,('whole','pre60min'),('Whole interval','Last 60 min')):
        s=counts[(counts.window==win)&(counts.exclude_post_minutes==0)&counts.observed_pair]
        for _,r in s.iterrows():
            x=list(s.sz).index(r.sz)
            ax.scatter(x,r.matching_share,c=RED if r.label=='TA' else BLUE,s=70,alpha=1 if r.n_events>=20 else .3)
            ax.annotate(f'{r.matching_share:.1%}\nn={r.n_events}',(x,r.matching_share),xytext=(0,11 if r.matching_share<.9 else -35),textcoords='offset points',ha='center',fontsize=10,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':.7})
        ax.axhline(.5,color='#777777',lw=.8,ls='--');ax.axhline(baseline,color='black',ls=':',lw=1)
        ax.set(xticks=range(len(s)),xticklabels=[f'SZ{v}\n{c:.0%} coverage' for v,c in zip(s.sz,s.coverage_fraction)],
               ylim=(0,1.1),xlim=(-.5,len(s)-.5),ylabel='Matching interictal template share',title=title)
    fig.legend(handles=[Line2D([],[],color='#777777',ls='--',label='50% majority'),
                        Line2D([],[],color='black',ls=':',label=f'Same-filter TA baseline: {baseline:.1%}')],
               loc='upper center',bbox_to_anchor=(.5,1.075),ncol=2,frameon=False,fontsize=10)
    save(fig,'observed_intervals_matching_share')
    blocks=pd.read_csv(BASE/'block_coverage_audit.csv')
    ranges=list(zip(blocks.loc[blocks.strict_eligible,'start'],blocks.loc[blocks.strict_eligible,'end']))
    observed_whole=counts[(counts.window=='whole')&(counts.exclude_post_minutes==0)&counts.observed_pair]
    fig,axes=plt.subplots(1,len(observed_whole),figsize=(14,3.8),sharey=True,layout='constrained')
    bins=[]
    for ax,(_,r) in zip(np.atleast_1d(axes),observed_whole.iterrows()):
        edges=np.linspace(r.start_epoch,r.end_epoch,11)
        for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=eligible&(times>=lo)&(times<hi)&(ends<=hi)
            n=int(mask.sum());p=float(np.mean(event_labels[mask]==0)) if n else np.nan
            cov=coverage(ranges,lo,hi)/(hi-lo)
            bins.append(dict(sz=r.sz,bin=j+1,start_epoch=lo,end_epoch=hi,n_events=n,ta_share=p,coverage_fraction=cov))
            if n:
                x=((lo+hi)/2-r.end_epoch)/3600
                ax.scatter(x,p,color=RED,s=14+7*np.log1p(n),alpha=.85 if n>=20 else .25)
        ax.axhline(.5,color='#777777',ls='--',lw=.7);ax.axhline(baseline,color='black',ls=':',lw=.8)
        ax.set(xlim=(-r.interval_hours,0),ylim=(0,1.05),xlabel='Hours to next seizure',title=f'SZ{r.sz}: TA-source\n{r.coverage_fraction:.0%} coverage')
    axes[0].set_ylabel('TA share in observed time bins')
    save(fig,'within_interval_observed_trajectory')
    pd.DataFrame(bins).to_csv(OUT/'trajectory_bins.csv',index=False)
    (OUT/'figures/README.md').write_text('''# 固定原方位 signed R 的 E1146 对应关系

### all_seizures_source_correspondence.png / .pdf
左侧为统一support和平滑、固定原方位下的TA/TB source–energy相关；中右为正确空间聚类标签在整段及末60分钟的比例。时间覆盖不足50%的区间保留，浅色条仅表示少于20个事件，缺失行不伪造零比例。
**关注点**：目前有观测的4段终点发作全属TA-source；TB-source行缺失，不能据此推断两类特异对应。首发只作描述。

### observed_intervals_matching_share.png / .pdf
直接列出4段可观测间隔的对应模式比例、事件数与覆盖率，虚线是50%，点线为同样排除规则下的患者TA基础比例。右图SZ24只有1个事件，100%仅是该单事件的事实。
**关注点**：占多数和超过患者基础比例不同，四次发作等权描述，不把数千事件当成独立发作重复；候选图待用户目视验收。

### within_interval_observed_trajectory.png / .pdf
四个可观测间隔各分为10个等时长格，横轴为距本次发作的实际小时数；只显示有完整包含事件的时间格，空白不插值。点大小随事件数增加，少于20个事件的点变浅，点线是同规则患者TA基础比例。
**关注点**：只代表已观测片段，不能把稀疏点连成整段连续演化；格边界处跨格事件不进入分格比例，整段计数仍保留。
''',encoding='utf-8')
    print(labels[['sz','r_a','r_b','label']].to_string(index=False))
    print(counts[(counts.window=='whole')&(counts.exclude_post_minutes==0)&counts.observed_pair][['sz','n_ta','n_tb','ta_share','coverage_fraction','label']].to_string(index=False))


if __name__=='__main__':run()

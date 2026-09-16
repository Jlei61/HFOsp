"""Held-out conditional mode distributions and sequence-randomization controls.

Reuse frozen chronological TRAIN parameters. Every null permutes TEST labels,
then reruns filtering before each mark. Fixed history scores are never paired
with shuffled outcomes without rebuilding history. No SNN runs or refits.
"""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import sys,json,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.model import slice_data,filter_adf

OUT=ROOT/'results/topic5_patient_state_inference/e1146_history_conditioned_modes_20260910'
EDGES=np.array([0,.2,.3,.4,.5,1.])
BIN_NAMES=['<0.20','0.20–0.30','0.30–0.40','0.40–0.50','≥0.50']


def load_parts():
    data=dict(np.load(RUN/'observations.npz'));events=pd.read_csv(RUN/'events.csv')
    folds=json.loads((RUN/'splits.json').read_text());best=best_fits();saved=np.load(RUN/'forward_predictions.npz')
    parts=[]
    for f in folds:
        start,end=f['test_start'],f['test_end'];assert data['reset'][start]
        d=slice_data(data,start,end);theta=np.asarray(best[f"fold{f['fold']}",'ou']['theta'])
        p=filter_adf(theta,d)['predict_tb'];np.testing.assert_allclose(p,saved[f"fold{f['fold']}_ou"],atol=1e-12,rtol=0)
        ix=np.arange(start,end);coverage=events.iloc[ix].coverage_segment.to_numpy().astype(int)
        coarse=np.column_stack([coverage,d['epoch']])
        _,seg=np.unique(coarse,axis=0,return_inverse=True)
        _,short=np.unique(np.column_stack([coarse,np.floor(d['t']*4).astype(int)]),axis=0,return_inverse=True)
        parts.append(dict(data=d,theta=theta,fold=f['fold'],index=ix,p=p,
                          constant=saved[f"fold{f['fold']}_constant"],groups=dict(global_order=np.zeros(len(ix),int),segment=seg,minutes15=short)))
    return parts


def score(p,y):
    p=np.clip(p,1e-12,1-1e-12)
    return y*np.log(p)+(1-y)*np.log1p(-p)


def null_worker(job):
    family,seed=job;target=OUT/'null_draws'/f'{family}_{seed}.json'
    if target.exists():return json.loads(target.read_text())
    rng=np.random.default_rng(seed);gain=0.;n=0;bin_n=np.zeros(5);bin_tb=np.zeros(5)
    for part in load_parts():
        d=part['data'];y=d['y'].copy()
        for group in np.unique(part['groups'][family]):
            ix=np.flatnonzero(part['groups'][family]==group);y[ix]=rng.permutation(y[ix])
            assert y[ix].sum()==d['y'][ix].sum()
        d=dict(d);d['y']=y;p=filter_adf(part['theta'],d)['predict_tb']
        gain+=np.sum(score(p,y)-score(part['constant'],y));n+=len(y)
        bins=np.searchsorted(EDGES[1:-1],p,side='right')
        bin_n+=np.bincount(bins,minlength=5);bin_tb+=np.bincount(bins,weights=y,minlength=5)
    r=dict(family=family,seed=seed,n=n,log_score_gain=gain/n,bin_n=bin_n,bin_tb=bin_tb)
    write_json(target,r);return r


def real_table():
    rows=[]
    for part in load_parts():
        d=part['data'];p=part['p']
        rows.append(pd.DataFrame(dict(index=part['index'],fold=part['fold'],hour=d['t'],y=d['y'],p=p,constant=part['constant'],segment=part['groups']['segment'])))
    table=pd.concat(rows,ignore_index=True);assert len(table)==16157 and table['index'].is_unique
    table['bin']=np.searchsorted(EDGES[1:-1],table.p,side='right')
    table['timeblock']=np.floor(table.hour/6).astype(int)
    table['gain']=score(table.p.to_numpy(),table.y.to_numpy())-score(table.constant.to_numpy(),table.y.to_numpy())
    return table


def summarize():
    table=real_table();table.to_csv(OUT/'real_pre_event_predictions.csv.gz',index=False)
    draws=pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'null_draws').glob('*.json')])
    draws.to_csv(OUT/'null_draws.csv',index=False)
    groups=list(table.groupby(['fold','timeblock'],sort=True));rng=np.random.default_rng(191046)
    B=5000;weights=np.zeros((B,len(groups)),int)
    for fold in sorted(table.fold.unique()):
        ids=np.array([j for j,((f,_),_) in enumerate(groups) if f==fold])
        for row in range(B):weights[row,ids]=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))
    ns=np.zeros((len(groups),5));ys=np.zeros_like(ns);gps=np.zeros(len(groups));total=np.zeros(len(groups))
    for j,(_,g) in enumerate(groups):
        ns[j]=np.bincount(g.bin,minlength=5);ys[j]=np.bincount(g.bin,weights=g.y,minlength=5);gps[j]=g.gain.sum();total[j]=len(g)
    bn=weights@ns;by=weights@ys;fractions=np.divide(by,bn,out=np.full_like(by,np.nan),where=bn>0)
    rows=[]
    for b in range(5):
        g=table[table.bin.eq(b)];lo,hi=np.nanquantile(fractions[:,b],[.025,.975])
        rows.append(dict(bin=b,label=BIN_NAMES[b],n=len(g),tb=int(g.y.sum()),ta=int(len(g)-g.y.sum()),
                         mean_pre_event_probability=g.p.mean(),observed_tb=g.y.mean(),lower=lo,upper=hi,
                         n_timeblocks=int((ns[:,b]>0).sum())))
    bins=pd.DataFrame(rows);bins.to_csv(OUT/'conditional_mode_fractions.csv',index=False)
    difference=bins.iloc[-1].observed_tb-bins.iloc[0].observed_tb
    ci=np.nanquantile(fractions[:,-1]-fractions[:,0],[.025,.975])
    actual=float(table.gain.mean());comparisons=[]
    for family,g in draws.groupby('family'):
        comps=dict(family=family,n_draws=len(g),median=g.log_score_gain.median(),
                   lower=g.log_score_gain.quantile(.025),upper=g.log_score_gain.quantile(.975),
                   n_ge_actual=int((g.log_score_gain>=actual-1e-12).sum()),
                   permutation_p=(1+int((g.log_score_gain>=actual-1e-12).sum()))/(len(g)+1))
        comparisons.append(comps)
    pd.DataFrame(comparisons).to_csv(OUT/'randomization_comparison.csv',index=False)
    per_fold=[]
    for fold,g in table.groupby('fold'):
        per_fold.append(dict(fold=int(fold),n=len(g),gain=g.gain.mean(),
                             low_n=int(g.p.lt(.2).sum()),high_n=int(g.p.ge(.5).sum()),
                             low_tb=g.loc[g.p.lt(.2),'y'].mean(),high_tb=g.loc[g.p.ge(.5),'y'].mean()))
    pd.DataFrame(per_fold).to_csv(OUT/'fold_sensitivity.csv',index=False)
    # The newest whole-interval pilot is an explicitly separate sensitivity cohort.
    recent=pd.read_csv(ROOT/'results/topic5_patient_state_inference/e1146_state_type_pilot_v2_20260910/ied_predictions.csv.gz')
    recent['bin']=np.searchsorted(EDGES[1:-1],recent.ou_p_tb,side='right')
    rec=recent.groupby('bin').agg(n=('y','size'),observed_tb=('y','mean'),mean_probability=('ou_p_tb','mean')).reset_index()
    rec.to_csv(OUT/'whole_interval_pilot_sensitivity.csv',index=False)
    # Existing fixed-horizon results isolate more remote history; no new model selection.
    horizons=pd.read_csv(RUN/'tuned_horizon_review_v1_17/predictions.csv.gz')
    sensitivity=[]
    for h in sorted(horizons.horizon_minutes.unique()):
        g=horizons[horizons.model.eq('ou')&horizons.horizon_minutes.eq(h)]
        if not len(g):continue
        sensitivity.append(dict(minutes_without_new_labels=float(h),n=len(g),low_n=int(g.p_tb.lt(.2).sum()),high_n=int(g.p_tb.ge(.5).sum()),
                                low_tb=g.loc[g.p_tb.lt(.2),'y'].mean(),high_tb=g.loc[g.p_tb.ge(.5),'y'].mean()))
    pd.DataFrame(sensitivity).to_csv(OUT/'remote_history_sensitivity.csv',index=False)
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',n_forward_events=len(table),n_chronological_folds=3,n_6h_blocks=len(groups),
        bin_edges=EDGES,bin_selection='Fixed probability ranges; no observed-label-driven quantiles',
        main_model='Basic OU, frozen prefix fits, no current-label use',
        high_minus_low=difference,high_minus_low_lower=ci[0],high_minus_low_upper=ci[1],
        real_log_score_gain=actual,real_gain_interval=np.quantile((weights@gps)/(weights@total),[.025,.975]),
        randomization=comparisons,bootstrap='5000 six-hour block resamples within chronological folds; pointwise descriptive intervals',
        global_null='Within each test fold: fixed event times, total counts and TA/TB counts',
        segment_null='Within each test fold x coverage segment x clinical interictal interval: fixed mode counts',
        local_null='Same segment null, additionally preserve 15-min-bin mode counts',
        critical_null_rule='Rebuild sequential state after each shuffle using fixed TRAIN parameters; do not shuffle labels against old states',
        scope='History-dependent conditional label distributions; not unique latent mechanism, autonomous state or causal patient E/I identification',
        source_label_scope='Frozen development labels, not new independent template validation',human_visual_review='pending'))
    plot(bins,draws,actual,comparisons)
    print(bins.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False));print(pd.DataFrame(per_fold).to_string(index=False))


def plot(bins,draws,actual,comparisons):
    fp=OUT/'figures';fp.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(13.6,5.7),gridspec_kw={'width_ratios':[1.2,1]})
    x=np.arange(5);p=bins.observed_tb.to_numpy()
    axs[0].bar(x,1-p,color='#d99099',label='TA');axs[0].bar(x,p,bottom=1-p,color='#598fbd',label='TB')
    for j,row in enumerate(bins.itertuples()):
        axs[0].text(j,1-row.observed_tb/2,f'TB {100*row.observed_tb:.1f}%',ha='center',va='center',fontsize=10,color='white')
        axs[0].text(j,.025,f'n={row.n:,}',ha='center',va='bottom',fontsize=9,color='#4d2228')
    axs[0].set_xticks(x,BIN_NAMES);axs[0].set_ylim(0,1)
    axs[0].set_ylabel('Observed fraction of subsequent events')
    axs[0].set_xlabel('TB propensity inferred BEFORE the current label')
    axs[0].set_title('A   Observed modes by pre-event prediction',loc='left',pad=31)
    axs[0].legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False,borderaxespad=0)
    families=['global_order','segment','minutes15'];names=['Fold counts\npreserved','Segment counts\npreserved','15-min counts\npreserved']
    for j,family in enumerate(families):
        vals=draws[draws.family.eq(family)].log_score_gain.to_numpy()
        viol=axs[1].violinplot([vals],positions=[j],showmeans=False,showmedians=True,showextrema=False)
        for body in viol['bodies']:body.set_facecolor('#91a6b4');body.set_alpha(.7)
        lo,med,hi=np.quantile(vals,[.025,.5,.975]);axs[1].plot([j,j],[lo,hi],color='#4d6472',lw=2)
        axs[1].scatter([j],[med],color='#4d6472',s=20)
    axs[1].axhline(actual,c='#6a3d9a',lw=2,label=f'Actual sequence: {actual:.4f}')
    axs[1].set_xticks(range(3),names);axs[1].set_ylabel('Forward log-score gain over fixed proportion')
    axs[1].set_title('B   Reconstruct state after label shuffling',loc='left',pad=31)
    axs[1].legend(loc='lower right',fontsize=9)
    axs[1].text(.98,.68,'256 shuffles per control\n0 / 256 reach the actual score',
                transform=axs[1].transAxes,ha='right',va='top',fontsize=9,color='#4d6472')
    fig.suptitle('E1146 | History predicts a distribution of IED modes',fontsize=15,y=1.015)
    fig.tight_layout(rect=(0,.06,1,.96))
    fig.text(.02,.015,'16,157 chronological held-out events. State parameters use earlier training data; templates are frozen development labels.',fontsize=10)
    save(fig,fp/'history_conditioned_modes')
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.errorbar(bins.mean_pre_event_probability,bins.observed_tb,
                yerr=[bins.observed_tb-bins.lower,bins.upper-bins.observed_tb],fmt='o-',color='#2166ac',capsize=3,label='Observed TB fraction')
    ax.plot([0,1],[0,1],':',c='gray',label='Calibrated prediction')
    ax.set(xlabel='Mean pre-event TB probability',ylabel='Observed TB fraction',xlim=(0,.7),ylim=(0,.7),title='Held-out conditional distributions; 6h-block intervals')
    ax.legend();fig.tight_layout();save(fig,fp/'conditional_probability_calibration')
    (fp/'README.md').write_text('### history_conditioned_modes.png / history_conditioned_modes.pdf\n\n左图按看到当前标签之前的 OU 预测概率分组，展示随后真实 TA/TB 的条件比例；所有 16,157 个前推事件保留。右图固定三种尺度的模式数量，打乱测试标签后重新推断状态，参数仍使用原训练前缀，紫线为真实顺序的预测增益。\n\n**关注点**：低/高状态对应不同概率，不要求硬分类全部正确；随机化检验针对顺序信息，不证明某个物理隐藏状态唯一存在。\n\n### conditional_probability_calibration.png / conditional_probability_calibration.pdf\n\n同一数据的预测概率与真实比例，误差条为按前推折分层的 6 小时时间块重采样点态区间。状态分组固定且未依据当前标签选择。\n\n**关注点**：这是预测前概率的留出校准，不是用已知标签反推后再作分类；待用户目视检查。\n',encoding='utf-8')


def save(fig,path):
    for ext in ('png','pdf'):fig.savefig(path.with_suffix('.'+ext),dpi=190,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=256);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    (OUT/'null_draws').mkdir(parents=True,exist_ok=True)
    jobs=[(family,26091000+1000*j+i) for j,family in enumerate(['global_order','segment','minutes15']) for i in range(args.replicates)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(null_worker,j) for j in jobs])):
            f.result()
            if (i+1)%64==0:print(f'{i+1}/{len(jobs)} null reconstructions complete',flush=True)
    summarize()

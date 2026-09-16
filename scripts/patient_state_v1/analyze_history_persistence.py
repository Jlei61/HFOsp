"""Causal history summaries, fixed-origin future marks and online probability traces.

Reuses the frozen chronological OU fits. New predictive fusion weights use only
the same earlier TRAIN prefix. No future marks update a fixed-origin forecast.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import ROOT, RUN, write_json
from scripts.patient_state_v1.model import filter_adf, slice_data
from scripts.patient_state_v1.round2 import ewma
from scripts.patient_state_v1.plot_history_conditioned_modes import load_parts, score

OUT = ROOT/'results/topic5_patient_state_inference/e1146_history_persistence_20260910'
COLORS = ['#356eae', '#6e98bb', '#96979a', '#bf8d85', '#b7474b']
LABELS = ['<0.20', '0.20–0.30', '0.30–0.40', '0.40–0.50', '≥0.50']


def block_interval(table, value, group=None, seed=260910, B=5000):
    """Paired 6h-block resampling within chronological folds; preserve overlaps."""
    t=table.copy();t['block']=np.floor(t.hour/6).astype(int)
    if group is None:t['_group']=0;group='_group'
    levels=sorted(t[group].unique());blocks=list(t.groupby(['fold','block']))
    counts=np.zeros((len(blocks),len(levels)));sums=np.zeros_like(counts)
    for j,(_,g) in enumerate(blocks):
        for k,lev in enumerate(levels):
            v=g.loc[g[group].eq(lev),value].dropna()
            counts[j,k]=len(v);sums[j,k]=v.sum()
    rng=np.random.default_rng(seed);w=np.zeros((B,len(blocks)),int)
    for fold in sorted(t.fold.unique()):
        ids=np.array([j for j,((f,_),_) in enumerate(blocks) if f==fold])
        for b in range(B):w[b,ids]=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))
    den=w@counts;num=w@sums
    draws=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    rows=[]
    for k,lev in enumerate(levels):
        vals=t.loc[t[group].eq(lev),value].dropna()
        rows.append(dict(group=int(lev),n=len(vals),mean=vals.mean(),
                         lower=np.nanquantile(draws[:,k],.025),upper=np.nanquantile(draws[:,k],.975)))
    return pd.DataFrame(rows),draws


def logistic_calibration(features, y):
    X=np.column_stack([np.ones(len(y)),features])
    def fun(b):
        eta=X@b
        # Fixed weak ridge, never selected on TEST.
        return np.logaddexp(0,eta).sum()-y@eta+.5*np.dot(b[1:],b[1:]), X.T@(expit(eta)-y)+np.r_[0,b[1:]]
    fit=minimize(fun,np.r_[0,np.full(X.shape[1]-1,1/(X.shape[1]-1))],jac=True,method='BFGS',options={'gtol':1e-6})
    assert np.max(np.abs(fit.jac))<1e-3,fit.message
    return fit.x


def fusion(parts):
    data=dict(np.load(RUN/'observations.npz'));controls=pd.read_csv(RUN/'round2_controls.csv')
    old=pd.read_csv(RUN/'all_forward_predictions.csv.gz')
    rows=[];params=[]
    for part in parts:
        fold=part['fold'];start=part['index'][0];end=part['index'][-1]+1
        ou=filter_adf(part['theta'],slice_data(data,0,end))['predict_tb']
        er=controls[controls.fold.eq(fold)&controls.model.eq('ewma')].iloc[0]
        memory=ewma(data['y'][:end],data['n'][:end],data['dt'][:end],data['reset'][:end],er.tau_hours,er.strength,er.p0)
        np.testing.assert_allclose(ou[start:end],part['p'],atol=1e-12,rtol=0)
        reference=old[old.fold.eq(fold)&old.model.eq('ewma')].sort_values('index')
        np.testing.assert_allclose(memory[start:end],reference.p_tb,atol=1e-12,rtol=0)
        X=logit(np.clip(np.column_stack([ou,memory]),1e-8,1-1e-8));y=data['y'][:start]
        prediction=dict(ou=ou[start:end],ewma=memory[start:end],constant=part['constant'])
        prediction['ou_history']=old[old.fold.eq(fold)&old.model.eq('ou_history')].sort_values('index').p_tb.to_numpy()
        for name,cols in [('ou_calibrated',[0]),('ewma_calibrated',[1]),('ou_ewma_fusion',[0,1])]:
            coef=logistic_calibration(X[:start,cols],y)
            prediction[name]=expit(coef[0]+X[start:end,cols]@coef[1:])
            params.append(dict(fold=fold,model=name,coefficients=coef,train_end=int(start),ridge=1.0))
        base=dict(fold=fold,index=part['index'],hour=part['data']['t'],y=part['data']['y'])
        rows.append(pd.DataFrame({**base,**prediction}))
    pred=pd.concat(rows,ignore_index=True);pred.to_csv(OUT/'model_predictions.csv.gz',index=False)
    write_json(OUT/'fusion_parameters.json',params)
    models=['constant','ewma','ou','ou_history','ou_calibrated','ewma_calibrated','ou_ewma_fusion'];metrics=[];comparisons=[]
    for m in models:
        metrics.append(dict(model=m,n=len(pred),brier=np.mean((pred[m]-pred.y)**2),log_score=np.mean(score(pred[m].to_numpy(),pred.y.to_numpy()))))
    for m,b in [('ou','ewma'),('ou_history','ou'),('ou_ewma_fusion','ou'),('ou_ewma_fusion','ewma'),
                ('ou_ewma_fusion','ou_calibrated'),('ou_ewma_fusion','ewma_calibrated')]:
        t=pred.copy();t['gain']=score(t[m].to_numpy(),t.y.to_numpy())-score(t[b].to_numpy(),t.y.to_numpy())
        st,_=block_interval(t,'gain');comparisons.append(dict(model=m,baseline=b,**st.iloc[0].to_dict()))
    pd.DataFrame(metrics).to_csv(OUT/'model_scores.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'model_comparisons.csv',index=False)
    corr=[]
    for fold,g in pred.groupby('fold'):
        corr.append(dict(fold=int(fold),n=len(g),pearson=g.ou.corr(g.ewma),spearman=g.ou.corr(g.ewma,method='spearman')))
    pd.DataFrame(corr).to_csv(OUT/'ou_memory_agreement.csv',index=False)
    return metrics,comparisons


def make_tables(parts):
    past=[];future=[];pairs=[];triples=[]
    for part in parts:
        d=part['data'];y=d['y'];t=d['t'];seg=part['groups']['segment'];n=len(y)
        assert np.all(d['n']==1)
        # The group IDs are ordered contiguous coverage x clinical intervals.
        cut=np.r_[True,seg[1:]!=seg[:-1]];start=np.maximum.accumulate(np.where(cut,np.arange(n),0))
        count=np.r_[0,np.cumsum(y)]
        for i in range(n):
            base=dict(fold=part['fold'],index=int(part['index'][i]),hour=t[i])
            if i-start[i]>=10:
                c=int(count[i]-count[i-10]);past.append(dict(**base,tb_past10=c,group=min(c//2,4),
                    seconds_past10=(t[i]-t[i-10])*3600,y=y[i],p=part['p'][i]))
            if i+9<n and start[i+9]<=i:
                rec=dict(**base,p=part['p'][i],group=int(np.searchsorted([.2,.3,.4,.5],part['p'][i],side='right')))
                for k in (1,2,5,10):
                    rec[f'y{k}']=y[i+k-1];rec[f'mean{k}']=y[i:i+k].mean();rec[f'seconds{k}']=(t[i+k-1]-t[i])*3600
                future.append(rec)
            if i>0 and start[i]<=i-1:
                pairs.append(dict(**base,previous=int(y[i-1]),y=int(y[i]),gap_seconds=(t[i]-t[i-1])*3600))
            if i>0 and i+1<n and start[i+1]<=i-1 and y[i]==1:
                triples.append(dict(**base,isolated=int(y[i-1]==0 and y[i+1]==0),
                                    both_gaps_under60=int(max(t[i]-t[i-1],t[i+1]-t[i])*3600<=60)))
    tables=[pd.DataFrame(x) for x in (past,future,pairs,triples)]
    for t,name in zip(tables,['past10_predictions','fixed_origin_future_events','adjacent_events','tb_neighbor_status']):
        t.to_csv(OUT/f'{name}.csv.gz',index=False)
    return tables


def cluster_stat(parts, labels):
    nums=np.zeros(8);past_num=np.zeros(5);past_den=np.zeros(5)
    for part,y in zip(parts,labels):
        seg=part['groups']['segment'];n=len(y);t=part['data']['t']
        pair=np.flatnonzero(seg[1:]==seg[:-1])+1
        pair=pair[(t[pair]-t[pair-1])*3600<=60]
        for prev in (0,1):
            a=pair[y[pair-1]==prev];nums[2*prev]+=y[a].sum();nums[2*prev+1]+=len(a)
        ix=np.arange(1,n-1);ix=ix[(seg[ix-1]==seg[ix])&(seg[ix+1]==seg[ix])]
        ix=ix[(np.maximum(t[ix]-t[ix-1],t[ix+1]-t[ix])*3600<=60)&(y[ix]==1)]
        nums[4]+=np.sum((y[ix-1]==0)&(y[ix+1]==0));nums[5]+=len(ix)
        ix=np.arange(10,n);ix=ix[seg[ix]==seg[ix-10]]
        cs=np.r_[0,np.cumsum(y)];h=cs[ix]-cs[ix-10];b=np.minimum(h//2,4).astype(int)
        past_num+=np.bincount(b,weights=y[ix],minlength=5);past_den+=np.bincount(b,minlength=5)
    return dict(tb_after_ta=nums[0]/nums[1],tb_after_tb=nums[2]/nums[3],
                transition_difference=nums[2]/nums[3]-nums[0]/nums[1],isolated_tb=nums[4]/nums[5],
                n_pairs=int(nums[1]+nums[3]),n_tb_with_neighbors=int(nums[5]),
                past10_fraction=np.divide(past_num,past_den,out=np.full(5,np.nan),where=past_den>0))


def cluster_nulls(parts):
    real=cluster_stat(parts,[p['data']['y'] for p in parts]);rows=[]
    for family in ('segment','minutes15'):
        grouped=[[np.flatnonzero(p['groups'][family]==g) for g in np.unique(p['groups'][family])] for p in parts]
        rng=np.random.default_rng(2609101 if family=='segment' else 2609102)
        for rep in range(512):
            labels=[]
            for p,groups in zip(parts,grouped):
                y=p['data']['y'].copy()
                for ix in groups:y[ix]=rng.permutation(y[ix])
                labels.append(y)
            rows.append(dict(family=family,replicate=rep,**cluster_stat(parts,labels)))
    write_json(OUT/'cluster_statistics.json',real);write_json(OUT/'cluster_null_draws.json',rows)
    summary=[]
    for family in ('segment','minutes15'):
        sub=[r for r in rows if r['family']==family]
        for measure in ('transition_difference','isolated_tb'):
            vals=np.array([r[measure] for r in sub]);n_extreme=int(np.sum(vals>=real[measure])) if measure=='transition_difference' else int(np.sum(vals<=real[measure]))
            summary.append(dict(family=family,measure=measure,actual=real[measure],median=np.median(vals),
                                lower=np.quantile(vals,.025),upper=np.quantile(vals,.975),n_extreme=n_extreme,n_draws=len(vals),p=(1+n_extreme)/(len(vals)+1)))
    pd.DataFrame(summary).to_csv(OUT/'cluster_null_summary.csv',index=False)
    return real,rows,summary


def summarize_tables(past,future,pairs,triples):
    h,_=block_interval(past,'y','group');h.to_csv(OUT/'past10_conditional_fractions.csv',index=False)
    fs=[];contrasts=[]
    for k in (1,2,5,10):
        for kind in ('y','mean'):
            s,draws=block_interval(future,f'{kind}{k}','group');s['k']=k;s['outcome']=kind;fs.append(s)
            contrast=draws[:,-1]-draws[:,0]
            contrasts.append(dict(k=k,outcome=kind,difference=s.iloc[-1]['mean']-s.iloc[0]['mean'],
                                  lower=np.nanquantile(contrast,.025),upper=np.nanquantile(contrast,.975)))
    f=pd.concat(fs,ignore_index=True);f.to_csv(OUT/'fixed_origin_future_summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'fixed_origin_contrasts.csv',index=False)
    scope=dict(n_past10=len(past),past10_seconds_quantiles=np.quantile(past.seconds_past10,[.1,.5,.9,.95]),
               n_common_next10_origins=len(future),next10_seconds_quantiles=np.quantile(future.seconds10,[.1,.5,.9,.95]),
               n_event_pairs=len(pairs),n_pairs_within60=int(pairs.gap_seconds.le(60).sum()),n_tb_two_neighbors=len(triples))
    return h,f,scope


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot_summary(h,f,nullrows,future):
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(13,5.2))
    x=np.arange(5);v=h['mean'].to_numpy()
    axs[0].errorbar(x,v,yerr=[v-h.lower,h.upper-v],fmt='o-',c='#245f91',capsize=3,label='Actual held-out sequence')
    vals=np.array([r['past10_fraction'] for r in nullrows if r['family']=='minutes15'])
    med=np.nanmedian(vals,axis=0);lo,hi=np.nanquantile(vals,[.025,.975],axis=0)
    axs[0].plot(x,med,'s--',c='#999999',label='Shuffle within 15-min segments')
    axs[0].fill_between(x,lo,hi,color='#bbbbbb',alpha=.3)
    for j,row in h.iterrows():axs[0].annotate(f'n={int(row.n):,}',(j,row['mean']),xytext=(0,13),textcoords='offset points',ha='center',fontsize=8)
    axs[0].set(xticks=x,xticklabels=['0–1','2–3','4–5','6–7','8–10'],xlabel='TB count among the previous 10 events',
               ylabel='Observed probability that the next event is TB',ylim=(0,.85),title='A   Direct history: no latent-state model needed')
    axs[0].legend(loc='upper left',fontsize=9)
    for group in range(5):
        s=f[f.group.eq(group)&f.outcome.eq('y')].sort_values('k')
        axs[1].plot(s.k,s['mean'],'o-',c=COLORS[group],label=LABELS[group])
        axs[1].fill_between(s.k,s.lower,s.upper,color=COLORS[group],alpha=.08)
    axs[1].set(xlabel='Future event number from the fixed prediction origin',ylabel='Observed TB fraction',
               xticks=[1,2,5,10],ylim=(0,.75),title='B   Freeze the starting probability; no label updates')
    axs[1].legend(title='Initial OU TB probability',fontsize=8,title_fontsize=9,loc='upper right',ncol=3)
    fig.suptitle('E1146 | TB clustering and persistence of historical mode preference',fontsize=14)
    fig.tight_layout(rect=(0,.075,1,.97))
    fig.text(.02,.018,f'Chronological held-out events. A: prior 10 events in the same segment. B: {len(future):,} common origins with 10 future events.\nIntervals: 6-hour block resampling; overlapping event windows are not independent replicates.',fontsize=9)
    save(fig,'history_and_future_modes')


def probability_trace(parts):
    data=dict(np.load(RUN/'observations.npz'));origin=float(data['origin_epoch']);events=pd.read_csv(RUN/'events.csv')
    exposure=pd.read_csv(RUN/'exposure.csv');nodes,weights=hermgauss(32);weights/=np.sqrt(np.pi)
    pieces=[];meta=[]
    # Use real contiguous exposure; do not draw a data-supported line across gaps.
    for part in parts:
        d=part['data'];filtered=filter_adf(part['theta'],d);b,lt,ls=part['theta'];tau,sd=np.exp([lt,ls])
        end_times=(events.iloc[part['index']].end_epoch.to_numpy()-origin)/3600
        for j,r in exposure.iterrows():
            lo=max((r.start_epoch-origin)/3600,d['t'][0]);hi=min((r.end_epoch-origin)/3600,d['t'][-1]+.25/3600)
            if hi<=lo:continue
            query=np.arange(lo,hi,2/3600);ix=np.searchsorted(end_times,query,side='right')-1
            m=np.zeros(len(query));v=np.full(len(query),sd*sd);ok=ix>=0
            # Exclude observations from an earlier clinical epoch at a reset boundary.
            epoch_ix=np.maximum(np.searchsorted(d['t'],query,side='right')-1,0)
            ok&=d['epoch'][np.maximum(ix,0)]==d['epoch'][epoch_ix]
            ii=ix[ok];a=np.exp(-(query[ok]-d['t'][ii])/tau)
            m[ok]=a*filtered['mean'][ii];v[ok]=a*a*filtered['variance'][ii]+sd*sd*(1-a*a)
            prob=(expit(b+m[:,None]+np.sqrt(2*v[:,None])*nodes)*weights).sum(axis=1)
            pieces.append(pd.DataFrame(dict(fold=part['fold'],exposure=int(j),hour=query,p_tb=prob)))
            meta.append(dict(fold=part['fold'],exposure=int(j),start=lo,end=hi,duration=hi-lo))
    trace=pd.concat(pieces,ignore_index=True);trace.to_csv(OUT/'online_probability_2s.csv.gz',index=False)
    # Representative section chosen solely by coverage length; tie -> earliest.
    eligible=sorted(meta,key=lambda r:(-r['duration'],r['start']))
    chosen=eligible[0];lo=chosen['start'];hi=chosen['end']
    part=next(p for p in parts if p['fold']==chosen['fold']);q=trace[trace.fold.eq(chosen['fold'])&trace.exposure.eq(chosen['exposure'])&trace.hour.between(lo,hi)]
    take=(part['data']['t']>=lo)&(part['data']['t']<=hi);t=part['data']['t'][take];y=part['data']['y'][take]
    fig,axs=plt.subplots(3,1,figsize=(14,8.5),gridspec_kw={'height_ratios':[1.2,1,.6]})
    for _,g in trace.groupby(['fold','exposure']):axs[0].plot(g.hour,g.p_tb,c='#235f93',lw=.75)
    axs[0].set(xlabel='Hours from record origin',ylabel='Online TB probability',ylim=(0,.8),title='A   All held-out coverage; gaps left blank')
    axs[0].axvspan(lo,hi,color='#d4bd96',alpha=.35)
    axs[1].plot((q.hour-lo)*60,q.p_tb,c='#235f93',lw=1.1,label='Online probability (2-second display grid)')
    axs[1].scatter((t-lo)*60,part['p'][take],s=4,c='#222222',alpha=.35,label='Prediction before each event label')
    axs[1].set(xlim=(0,(hi-lo)*60),ylim=(0,.8),ylabel='TB probability',title='B   Longest held-out coverage interval (chosen by duration)')
    axs[1].legend(loc='upper right',fontsize=9)
    for lab,color,level,name in [(0,'#c97883',0,'TA'),(1,'#497da8',1,'TB')]:
        times=(t[y==lab]-lo)*60;axs[2].vlines(times,level-.23,level+.23,color=color,lw=.8)
    axs[2].set(xlim=(0,(hi-lo)*60),ylim=(-.5,1.5),yticks=[0,1],yticklabels=['TA','TB'],xlabel='Minutes from section start',ylabel='Actual labels')
    fig.suptitle('E1146 | A continuous-time, history-inferred probability of TB',fontsize=15)
    fig.tight_layout(rect=(0,.07,1,.98))
    fig.text(.015,.015,'Only completed earlier labels update the curve. Between events: OU propagation; at new observations: filtering updates.\nThe latent process is continuous; its online estimate can jump. No future-label smoothing; no seizure-time prediction.',fontsize=9)
    save(fig,'online_tb_probability')
    return dict(**chosen,display_end=hi,selection='Entire longest actual held-out contiguous exposure, independent of labels and predictive effect',n_display_events=len(t))


def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True);parts=load_parts()
    events=pd.read_csv(RUN/'events.csv')
    assert np.all(events.end_epoch.to_numpy()[:-1] <= events.start_epoch.to_numpy()[1:]+1e-8), 'Previous label window must finish before current prediction'
    metrics,comparisons=fusion(parts);print('Prediction fusion completed',flush=True)
    past,future,pairs,triples=make_tables(parts);h,f,scope=summarize_tables(past,future,pairs,triples)
    real,nulls,ns=cluster_nulls(parts);print('1024 count-preserving clustering controls completed',flush=True)
    plot_summary(h,f,nulls,future);trace=probability_trace(parts)
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',n_forward_events=16157,scope=scope,trace=trace,
        fusion='Three earlier TRAIN-prefix calibrations with fixed ridge 1; paired single-input calibration controls included',
        history='Prior 10 completed events within a contiguous coverage x clinical interval; current label excluded',
        future='Freeze pre-label group at origin; outcomes 1,2,5,10 and means of next 1,2,5,10; no intervening label updates',
        cluster='Adjacent eligible events with <=60s gaps, keep local 15-min/coverage counts in nulls; no assertion of few isolated events',
        limitations='Single patient, frozen development label bank, exploratory comparisons, 14 uncertainty blocks; mark clustering is not unique hidden-state causality',
        human_visual_review='pending'))
    (OUT/'figures/README.md').write_text('### history_and_future_modes.png / history_and_future_modes.pdf\n\n左图展示同覆盖段内前 10 个已完成事件的 TB 数量与随后 TB 比例，并用保留 15 分钟内数量的随机顺序作对照。右图固定当前标签读取前的 OU 概率组，展示未来第 1、2、5、10 个事件的真实比例，不以中间标签更新分组。\n\n**关注点**：两图回答直接历史与固定起点的持续信息；重叠预测窗口不是独立样本，误差范围来自按折分层的 6 小时时间块。\n\n### online_tb_probability.png / online_tb_probability.pdf\n\n上图展示全部前推覆盖时段的在线 TB 概率，空白为未观测间隙；下两图固定选择完整的最长连续覆盖段，配对概率和真实标签。曲线仅在先前标签窗口完成后更新，不使用未来平滑。\n\n**关注点**：连续时间的隐过程与可以跳变的在线估计是两个概念；局部示例按覆盖选取，证据来自全部留出事件。\n',encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False));print(h.to_string(index=False));print(f.to_string(index=False));print(real);print(pd.DataFrame(ns).to_string(index=False));print(scope)


if __name__=='__main__':main()

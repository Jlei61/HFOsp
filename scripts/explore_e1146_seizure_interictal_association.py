#!/usr/bin/env python3
"""Fixed 22-feature exploratory association screen; seizure is the sample unit."""
from pathlib import Path
import sys,json,itertools,math
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_e1146_event_time_correspondence import OUT as SOURCE,contained,outside_seizures
from scripts.analyze_e1146_source_signed_correspondence import BASE,FIELD,RED,BLUE
from scripts.analyze_e1146_preseizure_template_share import inventory,ARTIFACT,coverage,write_json,sha
OUT=SOURCE/'association_exploration'
WINDOWS={'whole':None,'pre60':3600.,'pre15':900.}
META={}
for win in WINDOWS:
    for key,title,family,units in [('p_ta','TA share','composition','fraction'),('log_rate_ta','TA event rate','rate','log1p(events/hour)'),('log_rate_tb','TB event rate','rate','log1p(events/hour)'),('log_rate_all','Total event rate','activity_control','log1p(events/hour)')]:
        META[f'{win}_{key}']=dict(window=win,readout=title,family=family,units=units)
for win in ('whole','pre60'):META[f'{win}_repeat_excess']=dict(window=win,readout='Same-label adjacency excess',family='sequence',units='probability excess')
META['last20_p_ta']=dict(window='whole',readout='TA share in last 20 events',family='recency',units='fraction')
META['pre60_recency_bias']=dict(window='pre60',readout='TA-relative event recency',family='recency',units='log[(age TB +0.25s)/(age TA +0.25s)]')
for kind in ('halves','last60_change'):
    for key,title,units in [('p_ta','TA share change','fraction'),('log_rate_ta','TA rate change','delta log1p(events/hour)'),('log_rate_tb','TB rate change','delta log1p(events/hour)')]:
        META[f'{kind}_{key}']=dict(window=kind,readout=title,family='change',units=units)
assert len(META)==22


def bh_adjust(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);rank=np.arange(1,len(p)+1)
    adjusted=np.minimum.accumulate((p[order]*len(p)/rank)[::-1])[::-1]
    out=np.empty(len(p));out[order]=np.clip(adjusted,0,1);return out


def auc_from_ranks(ranks,is_tb):
    n1=int(np.sum(is_tb));n0=len(ranks)-n1
    return (np.asarray(ranks)[is_tb].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan


@lru_cache(maxsize=32)
def combinations(n,k):return np.array(list(itertools.combinations(range(n),k)),dtype=np.int16)


def exact_rank_test(x,is_tb):
    x=np.asarray(x,float);is_tb=np.asarray(is_tb,bool);n1=int(is_tb.sum());n0=len(x)-n1
    if min(n0,n1)<2:return dict(status='NOT_ESTIMABLE',p=1.)
    ranks=rankdata(x);au=auc_from_ranks(ranks,is_tb);effect=2*au-1
    space=math.comb(len(x),n1);exact=space<=200000
    if exact:sums=ranks[combinations(len(x),n1)].sum(axis=1)
    else:
        rng=np.random.default_rng(20260909)
        picks=np.argpartition(rng.random((19999,len(x))),n1-1,axis=1)[:,:n1]
        sums=ranks[picks].sum(axis=1)
    null=2*(sums-n1*(n1+1)/2)/(n0*n1)-1
    extreme=int(np.sum(np.abs(null)>=abs(effect)-1e-12))
    p=float(extreme/len(null) if exact else (extreme+1)/(len(null)+1))
    loo=[2*auc_from_ranks(rankdata(np.delete(x,i)),np.delete(is_tb,i))-1 for i in range(len(x))]
    return dict(status='EXPLORATORY',p=p,auc_tb=au,rank_biserial=effect,n_permutations=len(null),permutation_exact=exact,
                leave_one_out_min=float(np.nanmin(loo)),leave_one_out_max=float(np.nanmax(loo)))


def blocked_rank_test(x,is_tb,time_hours,width):
    """Exact centered rank sum conditional on class counts in fixed time strata."""
    x=np.asarray(x,float);is_tb=np.asarray(is_tb,bool);ranks=rankdata(x)
    bins=np.floor(np.asarray(time_hours)/width).astype(int);null=np.array([0.]);center=0.;mixed=0
    space=math.prod(math.comb(int(np.sum(bins==b)),int(np.sum(is_tb[bins==b]))) for b in np.unique(bins))
    if space>200000:
        rng=np.random.default_rng(20260909+int(width));null=np.zeros(19999)
        for block in np.unique(bins):
            r=ranks[bins==block];k=int(is_tb[bins==block].sum());n=len(r);center+=k*float(np.mean(r))
            if 0<k<n:mixed+=1
            if k:
                picks=np.argpartition(rng.random((19999,n)),k-1,axis=1)[:,:k]
                null+=r[picks].sum(axis=1)
        observed=float(ranks[is_tb].sum());p=float((np.sum(np.abs(null-center)>=abs(observed-center)-1e-12)+1)/(len(null)+1))
        return dict(p=p,n_permutations=len(null),n_mixed_time_blocks=mixed,rank_sum_residual=observed-center,permutation_exact=False)
    for block in np.unique(bins):
        m=bins==block;r=ranks[m];k=int(is_tb[m].sum());n=len(r)
        center+=k*float(np.mean(r))
        if k and k<n:mixed+=1
        possible=r[combinations(n,k)].sum(axis=1) if k else np.array([0.])
        null=(null[:,None]+possible[None,:]).ravel()
    observed=float(ranks[is_tb].sum());p=float(np.mean(np.abs(null-center)>=abs(observed-center)-1e-12))
    return dict(p=p,n_permutations=len(null),n_mixed_time_blocks=mixed,rank_sum_residual=observed-center)


def repeat_excess(mark,segments):
    """Permutation expectation preserves each continuous segment's label counts."""
    numerator=expected=denom=0.
    for seg in np.unique(segments):
        labels=np.asarray(mark)[segments==seg];n=len(labels)
        if n<2:continue
        a=int(np.sum(labels==0));b=n-a;pairs=n-1
        numerator+=np.sum(labels[1:]==labels[:-1]);denom+=pairs
        expected+=pairs*(a*(a-1)+b*(b-1))/(n*(n-1))
    return (numerator-expected)/denom if denom>=2 and len(mark)>=20 else np.nan


def load_events():
    sql,inv=inventory();rec=json.loads(FIELD.read_text());z=np.load(BASE/'event_index.npz')
    names=z['source_record_names'];starts=z['event_abs_time'];ends=np.full(len(starts),np.nan);labs=np.asarray(rec['template_discovery']['event_labels'])
    assert np.array_equal(rec['template_discovery']['sampled_event_indices'],z['source_event_index'])
    bm={f'{b["recording_id"]}_{b["block_no"]:04d}':b for b in sql['blocks']};ranges=[]
    for i,stem in enumerate(names):
        raw=np.load(ARTIFACT/f'{stem}_packedTimes_withFreqCent.npy');mask=z['source_block_id']==i;b=bm[stem]
        np.testing.assert_allclose(starts[mask],b['begin_epoch']+raw[:,0],atol=1e-5,rtol=0)
        ends[mask]=b['begin_epoch']+raw[:,1]
        assert contained(starts[mask],ends[mask],b['begin_epoch'],b['end_epoch']).all()
        ranges.append((b['begin_epoch'],b['end_epoch']))
    assert len(list(ARTIFACT.glob('*_packedTimes_withFreqCent.npy')))==len(names)==80
    merged=[]
    for lo,hi in sorted(ranges):
        if merged and lo<=merged[-1][1]+1e-6:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    segments=np.full(len(starts),-1,int)
    for i,(lo,hi) in enumerate(merged):segments[contained(starts,ends,lo,hi)]=i
    assert (segments>=0).all()
    order=np.argsort(starts,kind='stable')
    return inv,starts[order],ends[order],labs[order],segments[order],ranges


def build_features(events=None,labels=None,prior=None,out_dir=None):
    original_run=events is None
    out_dir=OUT if out_dir is None else Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    inv,starts,ends,labs,segments,ranges=load_events() if events is None else events
    eligible=outside_seizures(starts,ends,inv)
    labels=pd.read_csv(SOURCE/'seizure_source_labels.csv') if labels is None else labels
    rows=[];windows=[];support=[]
    origin=inv[0]['onset']
    def read(lo,hi):
        ix=np.flatnonzero(eligible&contained(starts,ends,lo,hi));seconds=coverage(ranges,lo,hi) if hi>lo else 0
        na=int(np.sum(labs[ix]==0));nb=len(ix)-na;n=na+nb
        values=dict(n_events=n,n_ta=na,n_tb=nb,observed_hours=seconds/3600,
                    coverage_fraction=seconds/(hi-lo) if hi>lo else 0,
                    p_ta=na/n if n else np.nan,
                    log_rate_ta=np.log1p(na*3600/seconds) if seconds else np.nan,
                    log_rate_tb=np.log1p(nb*3600/seconds) if seconds else np.nan,
                    log_rate_all=np.log1p(n*3600/seconds) if seconds else np.nan)
        return values,ix
    for i,r in enumerate(inv):
        l=labels.iloc[i]
        if not i or l.label not in ('TA','TB'):continue
        lo=inv[i-1]['offset'];hi=r['onset'];v=dict(sz=r['sz'],label=l.label,is_tb=l.label=='TB',time_hours=(hi-origin)/3600,
              interval_hours=(hi-lo)/3600,seizure_score=l.r_a-l.r_b)
        nrow=dict(sz=r['sz'])
        for win,seconds in WINDOWS.items():
            begin=max(lo,hi-seconds) if seconds else lo
            f,ix=read(begin,hi)
            windows.append(dict(sz=r['sz'],label=l.label,window=win,start_epoch=begin,end_epoch=hi,
                                complete=seconds is None or hi-lo>=seconds,**f))
            for key in ('p_ta','log_rate_ta','log_rate_tb','log_rate_all'):
                v[f'{win}_{key}']=f[key];nrow[f'{win}_{key}']=f['n_events']
            if win in ('whole','pre60'):
                v[f'{win}_repeat_excess']=repeat_excess(labs[ix],segments[ix]);nrow[f'{win}_repeat_excess']=len(ix)
            if win=='whole':
                v['last20_p_ta']=float(np.mean(labs[ix[-20:]]==0)) if len(ix)>=20 else np.nan;nrow['last20_p_ta']=min(len(ix),20)
            if win=='pre60':
                have_a=ix[labs[ix]==0];have_b=ix[labs[ix]==1]
                valid=len(have_a) and len(have_b) and coverage(ranges,hi-60,hi)>=59.9 and hi-lo>=60
                v['pre60_recency_bias']=np.log((hi-starts[have_b[-1]]+.25)/(hi-starts[have_a[-1]]+.25)) if valid else np.nan
                nrow['pre60_recency_bias']=len(ix)
        for kind,cut in [('halves',(lo+hi)/2),('last60_change',hi-3600)]:
            f0,_=read(lo,cut) if cut>lo else ({'n_events':0,**{k:np.nan for k in ('p_ta','log_rate_ta','log_rate_tb')}},[])
            f1,_=read(cut,hi) if cut>lo else ({'n_events':0,**{k:np.nan for k in ('p_ta','log_rate_ta','log_rate_tb')}},[])
            for key in ('p_ta','log_rate_ta','log_rate_tb'):
                v[f'{kind}_{key}']=f1[key]-f0[key];nrow[f'{kind}_{key}']=min(f0['n_events'],f1['n_events'])
        rows.append(v);support.append(nrow)
    features=pd.DataFrame(rows);support=pd.DataFrame(support);windows=pd.DataFrame(windows)
    prior=pd.read_csv(SOURCE/'interval_correspondence.csv') if prior is None and original_run else prior
    for _,w in (windows[windows.window=='whole'].iterrows() if prior is not None and len(windows) else []):
        p=prior[(prior.sz==w.sz)&(prior.window=='whole')&(prior.exclude_post_minutes==0)].iloc[0]
        assert w.n_events==p.n_events and abs(w.coverage_fraction-p.coverage_fraction)<1e-9
    if original_run:assert len(features)==22 and features.is_tb.sum()==7
    features.to_csv(out_dir/'seizure_features.csv',index=False);support.to_csv(out_dir/'feature_support_counts.csv',index=False)
    windows.to_csv(out_dir/'window_observations.csv',index=False)
    return features,support,windows


def screen(features,support,out_dir=None):
    out_dir=OUT if out_dir is None else Path(out_dir)
    results=[];yall=features.is_tb.to_numpy(bool);time=features.time_hours.to_numpy(float)
    for key,meta in META.items():
        xall=features[key].to_numpy(float);valid=np.isfinite(xall);x=xall[valid];y=yall[valid]
        result=exact_rank_test(x,y);result.update(feature=key,**meta,n=len(x),n_ta=int((~y).sum()),n_tb=int(y.sum()),
            median_ta=float(np.median(x[~y])) if (~y).any() else np.nan,median_tb=float(np.median(x[y])) if y.any() else np.nan)
        if result['status']=='NOT_ESTIMABLE':
            result.update(auc_tb=np.nan,rank_biserial=np.nan,leave_one_out_min=np.nan,leave_one_out_max=np.nan,spearman_source_score=np.nan,circular_shift_p=np.nan,n_valid_circular_shifts=0,n20_n=0,n20_p=1.,n20_rank_biserial=np.nan,common_event_cohort_n=0,common_event_cohort_p=1.,common_event_cohort_rank_biserial=np.nan)
            for width in (6,12):result.update({f'block{width}_p':1.,f'block{width}_n_permutations':0,f'block{width}_n_mixed_time_blocks':0,f'block{width}_rank_sum_residual':np.nan})
            results.append(result);continue
        result['spearman_source_score']=spearmanr(x,features.seizure_score.to_numpy()[valid]).statistic if len(x)>2 and np.ptp(x)>0 else np.nan
        for width in (6,12):
            b=blocked_rank_test(x,y,time[valid],width)
            result.update({f'block{width}_{k}':v for k,v in b.items()})
        ranks=rankdata(x);shifts=[]
        for k in range(len(yall)):
            yy=np.roll(yall,k)[valid]
            if min(yy.sum(),len(yy)-yy.sum())>=2:shifts.append(2*auc_from_ranks(ranks,yy)-1)
        result['circular_shift_p']=float(np.mean(np.abs(shifts)>=abs(result.get('rank_biserial',np.inf))-1e-12)) if shifts else np.nan
        result['n_valid_circular_shifts']=len(shifts)
        good=valid&(support[key].to_numpy()>=20);st=exact_rank_test(xall[good],yall[good])
        result.update(n20_n=int(good.sum()),n20_p=st['p'],n20_rank_biserial=st.get('rank_biserial',np.nan))
        common=valid&np.isfinite(features.whole_p_ta.to_numpy())
        cs=exact_rank_test(xall[common],yall[common])
        result.update(common_event_cohort_n=int(common.sum()),common_event_cohort_p=cs['p'],common_event_cohort_rank_biserial=cs.get('rank_biserial',np.nan))
        results.append(result)
        print(key,'n',len(x),'effect',result.get('rank_biserial'),'p',result['p'],'block6',result['block6_p'],flush=True)
    stats=pd.DataFrame(results)
    for p,q in [('p','q_bh22'),('block6_p','block6_q_bh22'),('block12_p','block12_q_bh22'),('n20_p','n20_q_bh22'),('common_event_cohort_p','common_event_cohort_q_bh22')]:stats[q]=bh_adjust(stats[p])
    stats.to_csv(out_dir/'association_screen.csv',index=False)
    write_json(out_dir/'summary.json',dict(n_candidates=len(META),n_seizures=len(features),n_ta=int((~yall).sum()),n_tb=int(yall.sum()),
        n_naive_bh05=int((stats.q_bh22<.05).sum()),n_block6_bh05=int((stats.block6_q_bh22<.05).sum()),n_block12_bh05=int((stats.block12_q_bh22<.05).sum()),
        candidates=stats.sort_values('p').to_dict('records')))
    return stats


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert (OUT/'PLAN.md').exists()
    write_json(OUT/'feature_contract.json',dict(features=META,plan_sha256=sha(OUT/'PLAN.md'),field_sha256=sha(FIELD),
       seizure_label_sha256=sha(SOURCE/'seizure_source_labels.csv'),positive_class='TB-source',rank_effect='2*AUC(TB higher than TA)-1; descriptive, no trained predictive model',
       event_time='actual start and end containment, no parent-hour exclusion',rate_denominator='hours covered by available labelled-event artifacts, including zero-event portions of present artifacts',
       sequence='adjacent time-sorted events within continuous observed segments only; expected same-label fraction conditioned on segment label counts',
       temporal_strata='fixed 6h from first seizure; 12h sensitivity; permutation strata only, never remove event parent blocks',
       blocked_test='rank sum centered at its conditional expectation under fixed label counts in time strata',
       multiple_testing='BH over all 22 prespecified exploratory readouts separately for unconstrained/6h/12h; not a correction for historical analysis choices',
       status='EXPLORATORY_SINGLE_PATIENT',user_visual_acceptance='pending'))
    f,s,w=build_features();stats=screen(f,s)
    print(stats[['feature','n','rank_biserial','p','q_bh22','block6_p','block6_q_bh22','block12_p','circular_shift_p']].sort_values('p').to_string(index=False))

if __name__=='__main__':main()

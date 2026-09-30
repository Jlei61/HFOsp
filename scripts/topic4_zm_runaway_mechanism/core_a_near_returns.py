"""Find individual recurrence seeds, not an average-lag periodicity claim.

The random projection only screens candidate pairs. Every reported pair is
ranked again with all E/I group rates, lagged rates and recorded dynamic M.
An exact full-state replay and shooting correction are still required.
"""
from common import OUT, np, read, write, log, model
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse

DEST=OUT/'core_a_bifurcation_type_20260924/near_returns'


def main(label,high_core_A=False,min_time=0.,max_time=float('inf'),min_period=20.,max_period=2000.,name=None):
    folder=DEST/(name or (label+('_high_A' if high_core_A else '')));folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'result.json').exists(),'Preserve previous recurrence screens'
    if label.startswith('mid_'):
        side=label[4:];root=OUT/'core_a_transition_continuation_20260924'
        paths=[Path(x) for x in read(root/f'mid1_{side}_30s_history.json')['sources']]
    elif label=='below60s':
        audit=read(DEST.parent/'sustained_A_long_window/joined60s_audit.json')
        assert audit['status']=='AUDIT_PASS'
        paths=[Path(x) for x in audit['sources']]
    elif label=='above70s':
        audit=read(DEST.parent/'censoring_controls/above_SN/joined70s_audit.json')
        assert audit['status']=='AUDIT_PASS'
        paths=[Path(x) for x in audit['sources']]
    elif label in ['below','above']:
        paths=sorted((DEST.parent/'fold_attractor_contrast'/label).glob('block*.npz'))
    else:
        paths=sorted((DEST.parent/label).glob('block*.npz'))
    blocks=[np.load(x) for x in paths];R=np.concatenate([x['group_rate_hz'] for x in blocks]).astype(float)
    # Separate continuation segments store segment-local clocks. The source
    # list is the audited chronological concatenation; use that physical
    # ordering, never searchsorted on its reset clocks.
    T=np.arange(1,len(R)+1,dtype=float);M=np.concatenate([x['M_current'] for x in blocks])
    MT=np.arange(10,10*len(M)+1,10,dtype=float);assert len(R)==10*len(M);s=model(40)
    W=np.array([s.sizes*(s.E&(s.geo['group_region']==i)) for i in range(3)])
    W=W/W.sum(1)[:,None];regional=R@W.T
    sm_A=uniform_filter1d(regional[:,0],10,mode='nearest')
    below=np.r_[0,np.cumsum(sm_A<50)]
    def permitted(a,b):
        return T[a]>=min_time and T[b]<=max_time and (not high_core_A or
            (sm_A[a]>200 and sm_A[b]>200 and below[b+1]==below[a]))
    w=s.sizes/s.sizes.sum();wroot=np.sqrt(w)
    rv=np.sum(np.var(R,axis=0)*w);mv=np.sum(np.var(M,axis=0)*w)
    assert rv>0 and mv>0
    rng=np.random.default_rng(92404);proj=rng.normal(size=(s.P,64))/8
    rp=(R*wroot)@proj/np.sqrt(rv)
    mp=(M*wroot)@proj/np.sqrt(mv)
    lag=np.array([0,1,2,4,8,16,32,64,128]);indices=np.arange(129,len(T),10)
    indices=indices[(T[indices]>=min_time)&(T[indices]<=max_time)]
    if high_core_A:indices=indices[sm_A[indices]>200]
    assert len(indices)>=100
    def mi(i):return np.clip(np.searchsorted(MT,T[i]),0,len(MT)-1)
    features=np.hstack([rp[indices-k]/np.sqrt(len(lag)) for k in lag]+[mp[mi(indices)]])
    nearest=NearestNeighbors(n_neighbors=100,n_jobs=2).fit(features)
    dd,jj=nearest.kneighbors(features);pairs=set()
    for i,nn in enumerate(jj):
        for j in nn:
            a,b=sorted([int(indices[i]),int(indices[j])]);period=T[b]-T[a]
            if min_period<=period<=max_period and permitted(a,b):
                excursion=rp[(a+b)//2]-rp[a]
                if excursion@excursion>.05:pairs.add((a,b))
    pairs=list(pairs)
    def score(a,b):
        dif=R[a-lag]-R[b-lag]
        rr=float(np.mean(dif*dif@w)/rv)
        dm=M[mi(a)]-M[mi(b)];mm=float(dm*dm@w/mv)
        return rr+mm,rr,mm
    rows=[]
    for a,b in pairs:
        total,rr,mm=score(a,b);rows.append((total,a,b,rr,mm))
    rows.sort();seeds=[]
    # Refine separate time neighborhoods in the original1ms records.
    for _,a,b,_,_ in rows:
        if any(abs(a-x[1])<40 and abs(b-x[2])<40 for x in seeds):continue
        choices=[]
        for da in range(-9,10):
            for db in range(-9,10):
                aa=a+da;bb=b+db
                excursion=rp[(aa+bb)//2]-rp[aa] if aa>=128 and bb<len(T) else np.zeros(1)
                if aa>=128 and bb<len(T) and min_period<=T[bb]-T[aa]<=max_period and excursion@excursion>.05 and permitted(aa,bb):
                    total,rr,mm=score(aa,bb);choices.append((total,aa,bb,rr,mm))
        seeds.append(min(choices))
        if len(seeds)>=24:break
    seeds.sort()
    output=[]
    for total,a,b,rr,mm in seeds:
        output.append(dict(time1_ms=float(T[a]),time2_ms=float(T[b]),period_ms=float(T[b]-T[a]),
            score=total,rate_history_squared_error_over_variance=rr,M_squared_error_over_variance=mm,
            regional_rates_A_B_S_1=regional[a].tolist(),regional_rates_A_B_S_2=regional[b].tolist()))
    write(folder/'result.json',dict(status='RECURRENCE_SEEDS_ONLY',label=label,sources=[str(x) for x in paths],
        selection=dict(high_core_A=high_core_A,min_time_ms=min_time,max_time_ms=max_time,
            min_period_ms=min_period,max_period_ms=max_period,
            high_A_rule='10ms-smoothed Core A >200Hz at both endpoints and never below50Hz between them; this only targets the sustained-A orbit seed, not a new state classifier'),
        definition='E/I original-cell weighted rate-history mismatch over nine lags0..128ms normalized by total rate temporal variance, plus M mismatch normalized by M temporal variance. Random64projection screens only; final scores use every original group. Recorded1ms rates and10ms M are not full-state closure.',
        candidates=output,model_promoted=False))
    log('LOCAL NEAR RETURNS',label,output[:3])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--high-core-A',action='store_true')
    p.add_argument('--min-time',type=float,default=0.);p.add_argument('--max-time',type=float,default=float('inf'))
    p.add_argument('--min-period',type=float,default=20.);p.add_argument('--max-period',type=float,default=2000.);p.add_argument('--name')
    a=p.parse_args();assert 0<a.min_period<a.max_period
    main(a.label,a.high_core_A,a.min_time,a.max_time,a.min_period,a.max_period,a.name)

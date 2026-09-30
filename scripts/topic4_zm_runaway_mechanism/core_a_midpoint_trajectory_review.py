"""Independent events and full-group near returns for midpoint trajectories."""
from common import np,read,write,model
from onset_state_continuation import regional_weights
from audit_core_a_natural_entry_step import intervals
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse


def main(source,discard=2000):
    out=Path(source).resolve();assert read(out/'jobs.json')['status']=='COMPLETE'
    s=model(40);W=regional_weights(s);z=np.load(out/'trajectory.npz');r=z['group_rate_hz'].astype(float)
    regional=r@W.T;bound=abs(np.spacing(z['group_rate_hz']).astype(float))@W.T
    assert np.all(abs(regional-z['regional_rate_hz'])<=bound+1e-10)
    assert np.isfinite(r).all() and r.min()>=0
    original=read(out/'result.json');sm=uniform_filter1d(regional,10,axis=0,mode='nearest');rows=[]
    for j,name in enumerate(['Global E','Core A','Core B','Surround']):
        quiet=[(a,b) for a,b in intervals(sm[:,j]<5) if b-a>=20]
        assert quiet==[tuple(q) for q in original['rows'][j]['quiet_intervals_ms']]
        edge=[(0,0),*quiet,(len(r),len(r))]
        activities=[dict(start_ms=b,end_ms=c,duration_ms=c-b,left_censored=b==0,right_censored=c==len(r))
                    for (_,b),(c,_) in zip(edge[:-1],edge[1:]) if c>b]
        assert activities==original['rows'][j]['activities']
        complete=[a for a in activities if not a['left_censored'] and not a['right_censored']]
        rows.append(dict(region=name,quiet_fraction=sum(b-a for a,b in quiet)/len(r),
            complete_count=len(complete),max_complete_ms=max((a['duration_ms'] for a in complete),default=None),
            complete_long=[a for a in complete if a['duration_ms']>=1000],activities=activities))
    contract=read(out/'contract.json');final=np.load(out/'final_state.npz');initial=np.load(contract['source'])
    assert np.array_equal(z['Z'],final['syn'][5])
    assert np.all(final['parameters'][19]==0) and np.all(final['parameters'][20]==1)
    assert np.array_equal(z['Z'][~(s.E&(s.geo['group_region']==0))],initial['syn'][5,~(s.E&(s.geo['group_region']==0))])
    t=z['time_ms'];mt=z['M_time_ms'];m=z['M_current'];ix=np.flatnonzero((sm[:-1,1]<100)&(sm[1:,1]>=100))
    cross=t[ix]+(100-sm[ix,1])/(sm[ix+1,1]-sm[ix,1]);cross=cross[cross>discard]
    def sample(data,grid,tm):
        j=np.clip(np.searchsorted(grid,tm)-1,0,len(grid)-2);f=(tm-grid[j])/(grid[j+1]-grid[j])
        return data[j]*(1-f)[...,None]+data[j+1]*f[...,None]
    pairs=[];best=[]
    if len(cross)>1:
        h=sample(r,t,cross[:,None]-np.arange(36.)[None,:]);v=sample(m,mt,cross)
        w=s.sizes/s.sizes.sum();hs=np.mean(np.sum(h*h*w,axis=-1));ms=np.mean(np.sum(v*v*w,axis=-1))
        for i in range(len(cross)):
            for j in range(i+1,min(i+19,len(cross))):
                hr=float(np.sqrt(np.mean(np.sum((h[j]-h[i])**2*w,axis=-1))/hs))
                mr=float(np.sqrt(np.sum((v[j]-v[i])**2*w)/ms))
                pairs.append(dict(index1=i,index2=j,burst_count=j-i,time1_ms=float(cross[i]),time2_ms=float(cross[j]),
                    period_ms=float(cross[j]-cross[i]),history_relative_rms=hr,M_relative_rms=mr,score=float(np.hypot(hr,mr)/np.sqrt(2))))
        pairs.sort(key=lambda q:q['score'])
        best=[min((q for q in pairs if q['burst_count']==n),key=lambda q:q['score']) for n in sorted({q['burst_count'] for q in pairs})]
    result=dict(status='INDEPENDENT_READOUT_PASS',source=str(out),Z_A=float(z['Z']@W[1]),D_A=float(1-z['Z']@W[1]),
        regions=rows,discard_ms=discard,crossings_ms=cross.tolist(),best_by_burst_count=best,candidates=pairs[:20],
        recurrence_scope='A near return of saved all-group rate history and M nominates full-state replay. It is not a solved cycle, full-state closure, stability or a bifurcation.',
        target_entry_type='NOT_ESTABLISHED',model_promoted=False)
    write(out/'independent_audit.json',result)
    print(out.name,'ZA',result['Z_A'],'maxcomplete',rows[1]['max_complete_ms'],'tail',rows[1]['activities'][-1:],flush=True)
    print('near returns',pairs[:6],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--discard-ms',type=float,default=2000)
    a=p.parse_args();main(a.source,a.discard_ms)

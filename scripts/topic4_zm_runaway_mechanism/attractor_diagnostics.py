"""Describe finite-time post-transition activity without labelling it a periodic orbit."""
from common import *
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import argparse


def diagnose(path):
    z=np.load(path);g=z['global_E_hz'];R=z['group_rate_hz'];n=len(g)
    start=max(0,n-6000);x=uniform_filter1d(g,10,mode='nearest')[start:]
    d=np.diff(np.r_[0,(x>=5).astype(int),0]);beg=np.flatnonzero(d==1);end=np.flatnonzero(d==-1)
    events=[(int(a+start),int(b+start)) for a,b in zip(beg,end) if a>0 and b<len(x) and b-a>=20 and x[a:b].max()>20]
    iei=np.diff([v[0] for v in events]);dur=[b-a for a,b in events]
    rr=R[start:].astype(float)
    # Field recurrence measured on all group rates, not on the global average alone.
    # Restrict to 80--1500 ms, covering self-limited bursts rather than fast within-event peaks.
    gx=x-x.mean();ac=np.correlate(gx,gx,'full')[len(gx)-1:]
    ac/=np.maximum(np.arange(len(gx),0,-1),1);ac/=max(ac[0],1e-20)
    pk=find_peaks(ac[80:min(1501,len(ac))])[0]+80
    lags=sorted(pk,key=lambda j:ac[j],reverse=True)[:5];rec=[]
    denom=max(float(np.mean((rr-rr.mean(0))**2)),1e-20)
    for lag in lags:
        error=float(np.sqrt(np.mean((rr[lag:]-rr[:-lag])**2)/denom))
        rec.append(dict(lag_ms=int(lag),global_autocorrelation=float(ac[lag]),field_recurrence_error=error))
    blocks=g[start:start+(n-start)//1000*1000].reshape(-1,1000).mean(1)
    q=dict(source=str(path),tail_start_ms=start,tail_length_ms=n-start,events=events,
           duration_ms=dur,interval_ms=iei.tolist(),interval_cv=float(iei.std()/iei.mean()) if len(iei)>1 else None,
           block_mean_hz=blocks.tolist(),recurrences=rec,
           interpretation='Finite-time event and field-recurrence diagnostics; no Floquet or chaos classification inferred')
    return q


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');a=p.parse_args();rows=[]
    for path in a.paths:
        q=diagnose(path);rows.append(q);log(Path(path).parent.name,'events',q['events'],'IEI CV',q['interval_cv'],'field returns',q['recurrences'])
    write(OUT/'attractor_diagnostics.json',rows)

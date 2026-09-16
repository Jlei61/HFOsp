"""Paired timing-weight audit. No detection changes or HFO equivalence claim."""
import numpy as np


def reweight_centroids(envelope,dt_ms,windows_ms,mask,*,power=1.,baseline=None):
    env=np.asarray(envelope,float);mask=np.asarray(mask,bool)
    if mask.shape!=(len(windows_ms),len(env)) or power<=0:raise ValueError('invalid paired observation')
    if baseline is None:baseline=np.zeros(len(env))
    baseline=np.asarray(baseline,float)
    if baseline.shape!=(len(env),):raise ValueError('baseline/contact mismatch')
    t=np.full(mask.shape,np.nan)
    for i,(start,stop) in enumerate(windows_ms):
        a,b=int(round(start/dt_ms)),int(round(stop/dt_ms))
        weight=np.maximum(env[:,a:b]-baseline[:,None],0.)**power
        total=weight.sum(axis=1)
        valid=mask[i]&(total>0)
        t[i,valid]=weight[valid]@(np.arange(a,b)*dt_ms)/total[valid]
    return t


def paired_order_change(a,b):
    c=a.shape[1];i,j=np.triu_indices(c,1)
    da=a[:,i]-a[:,j];db=b[:,i]-b[:,j]
    ok=np.isfinite(da)&np.isfinite(db)&(abs(da)>1e-9)&(abs(db)>1e-9)
    return {'n_comparable_event_pairs':int(ok.sum()),
            'strict_order_reversal_fraction':float(np.mean((da[ok]*db[ok])<0)) if ok.any() else None}

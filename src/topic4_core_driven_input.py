"""Core-led stochastic input; geometry only, no event or patient-mode input."""
import numpy as np


def lowering_only(delta):
    """Retain existing lowering heterogeneity; clip only positive shifts."""
    delta=np.asarray(delta,float)
    if not np.isfinite(delta).all():raise ValueError('nonfinite threshold field')
    return np.minimum(delta,0.)


class MaskedSpatialDrive:
    def __init__(self, base, mask):
        self.base=base;self.mask=np.asarray(mask,bool).copy()
        self.last_t=None;self.cached=None
    def step(self,time_ms):
        values=self.base.step(time_ms)
        if values.shape!=self.mask.shape:raise ValueError('drive/mask mismatch')
        # Do not renormalize, recenter, or reassign removed noise outside the core.
        return np.where(self.mask,values,0.)


class RateAudit:
    """Audit actual rates before Poisson sampling, after both OU contributions."""
    def __init__(self,groups,n_total,dt):
        self.groups={k:np.asarray(v,int) for k,v in groups.items()}
        self.n_total=n_total;self.stride=max(1,round(1./dt));self.rows=[]
        self.max_outside_deviation=0.;self.calls=0
    def observe(self,t,tm,signal,xi,rates):
        self.calls+=1
        outside=self.groups['surroundE']
        self.max_outside_deviation=max(self.max_outside_deviation,float(np.max(np.abs(rates[outside]-max(signal,0)),initial=0)))
        if t%self.stride:return
        row=[tm,signal,xi]
        for index in self.groups.values():row.extend([float(rates[index].mean()),float(rates[index].std())])
        self.rows.append(row)
    def arrays(self):
        names=['time_ms','tonic_signal_per_ms','global_xi_per_ms']
        for name in self.groups:names.extend([name+'_mean_rate_per_ms',name+'_sd_rate_per_ms'])
        return {'columns':np.asarray(names),'values':np.asarray(self.rows,float)}


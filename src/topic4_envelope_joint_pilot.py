"""Joint contact timing/shape objective for a bounded physical development pilot.

The new statistic is an off-diagonal empirical kernel distance, not a claim
of unbiasedness for correlated events. Patient sampling weights correct the
balanced morphology packet; they do not rebalance generated events.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist

VERSION='contact_timing_shape_joint_pilot_v1'


def aligned_packet_event(path, names):
    with np.load(path) as z:
        raw=z['contact_names'].astype(str).tolist()
        if len(set(raw))!=len(raw) or set(raw)!=set(names):
            raise ValueError('patient/model contact identity mismatch')
        order=[raw.index(n) for n in names]
        window=z['packed_window_mask'].astype(bool)
        mass=z['positive_envelope_mass'][order][:,window].astype(float)
        time=z['time_ms'][window].astype(float)
        mask=z['participation_mask'][order].astype(bool)
    return envelope_descriptor(mass,time,mask)


def envelope_descriptor(mass,time,mask):
    mass=np.asarray(mass,float);time=np.asarray(time,float);mask=np.asarray(mask,bool)
    if mass.shape!=(len(mask),len(time)) or not np.isfinite(mass).all() or np.any(mass<0):
        raise ValueError('finite nonnegative contact envelopes required')
    mu=np.full(len(mask),np.nan);q=np.full((len(mask),3),np.nan)
    for i in np.flatnonzero(mask):
        total=mass[i].sum()
        if total<=0:raise ValueError('participating contact has no positive mass')
        mu[i]=mass[i]@time/total
        q[i]=time[np.minimum(np.searchsorted(np.cumsum(mass[i])/total,[.1,.5,.9]),len(time)-1)]
    # Translation invariant and contact-resolved; never sort/relabel contacts.
    center=float(np.nanmedian(q[:,1]))
    values=np.column_stack([mask,np.nan_to_num(q[:,1]-center),
                            np.nan_to_num(q[:,1]-q[:,0]),np.nan_to_num(q[:,2]-q[:,1]),
                            np.nan_to_num(mu-q[:,1])])
    ix=np.flatnonzero(mask);i,j=np.triu_indices(len(ix),1);qq=q[ix];mm=mu[ix]
    overlap=np.maximum(0,np.minimum(qq[i,2],qq[j,2])-np.maximum(qq[i,0],qq[j,0]))
    union=np.maximum(qq[i,2],qq[j,2])-np.minimum(qq[i,0],qq[j,0])
    delta=mm-qq[:,0];dmu=mm[j]-mm[i];d10=qq[j,0]-qq[i,0]
    resolved=(np.abs(dmu)>2)&(np.abs(d10)>2)
    stats=dict(contact_width_ms=float(np.median(qq[:,2]-qq[:,0])),centroid_span_ms=float(np.ptp(mm)),
               overlap=float(np.median(overlap/np.maximum(union,1e-12))),
               shape_lag_ms=float(np.median(np.abs(delta[j]-delta[i]))),
               reversal=float(np.mean(dmu[resolved]*d10[resolved]<0)) if resolved.any() else None,
               n_contacts=int(mask.sum()))
    return dict(values=values,centroid=mu,quantiles=q,statistics=stats)


def model_events(worker_path, observe_function):
    worker_path=Path(worker_path);worker=json.loads(worker_path.read_text())
    _,op,metadata=observe_function(worker_path)
    with np.load(op) as z:
        indices=z['primary_event_indices'];centroids=z['centroid_ms'][indices].astype(float)
    with np.load(worker['arrays']['path']) as z:
        envelope=z['contact_envelope'].astype(float);dt=float(z['contact_envelope_dt_ms'])
        names=z['contact_names'].astype(str).tolist()
    descriptors=[];info=[]
    for k,idx in enumerate(indices):
        event=metadata['events'][int(idx)];lo,hi=np.rint(np.asarray(event['window_ms'])/dt).astype(int)
        mass=np.maximum(envelope[:,lo:hi]-np.asarray(event['local_baseline'])[:,None],0)
        d=envelope_descriptor(mass,(np.arange(lo,hi)+.5)*dt,np.isfinite(centroids[k]))
        if np.nanmax(np.abs(d['centroid']-centroids[k]))>=.002:
            raise RuntimeError('saved centroid differs beyond absolute float32 precision')
        descriptors.append(d['values'])
        info.append(dict(event_id=int(idx),**d['statistics']))
    return dict(descriptors=np.asarray(descriptors).reshape(-1,len(names),5),centroids=centroids,
                info=info,names=names,worker=worker,metadata=metadata)


class TemporalDistributionObjective:
    def __init__(self, values, labels, proportions, names, seed=820801):
        values=np.asarray(values,float);labels=np.asarray(labels,int)
        self.names=list(names)
        self.proportions=np.asarray(proportions,float)
        self.patient_weights=np.array([self.proportions[k]/np.sum(labels==k) for k in labels])
        assert np.isclose(self.patient_weights.sum(),1.)
        # Four temporal blocks retain physical units; robust unit scales are
        # obtained from TRAIN contacts only, and have a 10 ms floor.
        active=values[:,:,0].astype(bool)
        self.scales=np.array([1.]+[max(10.,float(np.median(np.abs(values[:,:,i][active])))) for i in range(1,5)])
        self.reference=self.features(values)
        dd=cdist(self.reference,self.reference,'euclidean');positive=dd[np.triu_indices(len(dd),1)]
        self.bandwidth=max(1e-3,float(np.median(positive[positive>0])))
        self.k_ref=self.kernel(self.reference,self.reference)
        self.reference_constant=float(self.patient_weights@self.k_ref@self.patient_weights)
        # A strictly positive unit scale, not a confidence interval or IID
        # reference test: finite-m=16 diagonal distances on the TRAIN packet.
        rng=np.random.default_rng(seed);cal=[]
        for _ in range(256):
            ii=rng.choice(len(values),size=16,replace=True,p=self.patient_weights)
            cal.append(self.diagonal(self.reference[ii]))
        self.normalizer=float(np.median(cal))
        if not self.normalizer>0:raise RuntimeError('positive temporal scale required')

    def features(self,values):
        return (np.asarray(values,float)/self.scales).reshape(len(values),-1)/np.sqrt(len(self.names)*5)

    def kernel(self,x,y):
        d=cdist(np.asarray(x,float),np.asarray(y,float),'sqeuclidean')
        return sum(np.exp(-d/(2*(self.bandwidth*m)**2)) for m in [.5,1.,2.])/3

    def diagonal(self,x):
        xx=self.kernel(x,x);xy=self.kernel(x,self.reference)
        return float(xx.mean()-2*np.mean(xy@self.patient_weights)+self.reference_constant)

    def score(self,values):
        n=len(values)
        if n<16:return dict(status='INSUFFICIENT_EVENTS',loss=None,N=n)
        x=self.features(values);xx=self.kernel(x,x);xy=self.kernel(x,self.reference)
        a=float(xx.mean()-2*np.mean(xy@self.patient_weights)+self.reference_constant)
        v=float(np.mean(np.diag(xx))-xx.mean());b=v/(n-1)
        off=float((xx.sum()-np.trace(xx))/(n*(n-1))-2*np.mean(xy@self.patient_weights)+self.reference_constant)
        assert np.isclose(off,a-b,atol=1e-12)
        return dict(status='ESTIMABLE',N=n,A=a,B=b,D_off=off,D16=a+(n-16)/(16*(n-1))*v,
                    normalizer=self.normalizer,loss=off/self.normalizer)

    def population_distance(self, values, weights):
        x=self.features(values);w=np.asarray(weights,float)
        return float(w@self.kernel(x,x)@w-2*w@self.kernel(x,self.reference)@self.patient_weights+self.reference_constant)


def combined_score(old_objective,temporal,model):
    old=old_objective.score_network(model['centroids'])
    new=temporal.score(model['descriptors'])
    eligible=(model['worker'].get('physical_status')!='RUNAWAY' and old['loss_off'] is not None and new['loss'] is not None)
    return dict(status='ESTIMABLE' if eligible else ('RUNAWAY' if model['worker'].get('physical_status')=='RUNAWAY' else 'INSUFFICIENT_EVENTS'),
                N=len(model['centroids']),mode_counts=old.get('mode_counts'),old=old,temporal=new,
                loss=.5*old['loss_off']+.5*new['loss'] if eligible else None,
                physical_status=model['worker'].get('physical_status'),execution_status=model['worker'].get('execution_status'))

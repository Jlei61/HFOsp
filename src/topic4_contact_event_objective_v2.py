"""Observable-separated multi-event objective, development revision 2.

Four envelope blocks plus the existing full-FIT centroid feature embedding.
The four envelope blocks combine a linear mean term and a nonlinear
distribution term. No model TA/TB classifier is used.

All U corrections are retained, including negative values. The finite-sample
statistic is not claimed unbiased for correlated continuous SNN events.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d

VERSION='contact_event_distribution_v2_development'
QUANTILES=np.arange(1,20,dtype=float)/20
TIME_GRID_MS=np.arange(-250.,250.01,2.)
BLOCKS=('participation','local_shape','recruitment','joint_envelope')


def event_representation(mass,time_ms,participation):
    a=np.asarray(mass,float);t=np.asarray(time_ms,float);m=np.asarray(participation,bool)
    if a.ndim!=2 or a.shape!=(len(m),len(t)) or len(t)<2 or not m.any():
        raise ValueError('contact x time envelope and nonempty participation required')
    if not np.isfinite(a).all() or np.any(a<0) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('finite positive envelope and increasing time required')
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-4):raise ValueError('regular input sampling required')
    if t[-1]-t[0]+dt>250.01:raise ValueError('input exceeds frozen 250 ms observation window')
    mass_by_contact=a.sum(1)*dt
    if np.any(mass_by_contact[m]<=0) or mass_by_contact.sum()<=0:raise ValueError('participating contacts require positive mass')
    q=np.zeros((len(m),len(QUANTILES)))
    for i in np.flatnonzero(m):
        cum=np.cumsum(a[i])*dt/mass_by_contact[i]
        q[i]=np.interp(QUANTILES,np.r_[0.,cum],np.r_[t[0]-dt,t])
    med=q[:,9];event_mid=float(np.median(med[m]))
    local=np.where(m[:,None],q-med[:,None],0.)
    recruitment=np.where(m,med-event_mid,0.)[:,None]
    # The joint view includes EVERY contact, even those absent from the event
    # participation mask. One event-wide normalization avoids boosting each
    # weak contact independently. One event-wide shift preserves recruitment.
    pooled=np.cumsum(a.sum(0))*dt/mass_by_contact.sum()
    anchor=float(np.interp(.5,np.r_[0.,pooled],np.r_[t[0]-dt,t]))
    cdf=np.cumsum(a,axis=1)*dt/mass_by_contact.sum()
    joint=np.array([np.interp(TIME_GRID_MS+anchor,np.r_[t[0]-dt,t],np.r_[0.,row],left=0.,right=row[-1]) for row in cdf])
    assert np.isclose(joint[:,-1].sum(),1.) and np.all(joint[:,0]<1e-7)
    centroid=np.full(len(m),np.nan)
    centroid[m]=(a[m]@t)/a[m].sum(1)
    return dict(participation=m.astype(float),local_shape=local,recruitment=recruitment,centroid=centroid,
                joint_envelope=joint.reshape(-1),quantiles=q,anchor_ms=anchor,
                local_width_ms=float(np.median((q[:,17]-q[:,1])[m])))


def stack_events(events):
    if not events:raise ValueError('events required')
    return {k:np.stack([e[k] for e in events]) for k in (*BLOCKS,'centroid')}


def patient_event(path,names):
    with np.load(path) as z:
        raw=z['contact_names'].astype(str).tolist()
        if len(set(raw))!=len(raw) or set(raw)!=set(names):raise ValueError('contact identity mismatch')
        ix=[raw.index(n) for n in names];window=z['packed_window_mask'].astype(bool)
        a=z['positive_envelope_mass'][ix][:,window].astype(float)
        t=z['time_ms'][window].astype(float);m=z['participation_mask'][ix].astype(bool)
    return event_representation(a,t,m)


def worker_events(path,local_smoothing_sd_ms=0.):
    path=Path(path);op=path.parent.parent/'repaired_observation'/path.name
    metadata=json.loads(op.read_text());worker=json.loads(path.read_text())
    if metadata['worker_lineage_onsets_used_for_training']:raise ValueError('full contact observer required')
    with np.load(op.with_suffix('.npz')) as z:
        indices=z['primary_event_indices'];mu=z['centroid_ms'][indices]
    with np.load(worker['arrays']['path']) as z:
        a=z['contact_envelope'].astype(float);dt=float(z['contact_envelope_dt_ms'])
        names=z['contact_names'].astype(str).tolist()
    rows=[];details=[]
    for idx,centroid in zip(indices,mu):
        e=metadata['events'][int(idx)];lo,hi=np.rint(np.asarray(e['window_ms'])/dt).astype(int)
        mass=np.maximum(a[:,lo:hi]-np.array(e['local_baseline'])[:,None],0.)
        t=(np.arange(lo,hi)+.5)*dt;retained=1.
        if local_smoothing_sd_ms!=0.:
            if local_smoothing_sd_ms<0:raise ValueError('nonnegative smoothing SD required')
            # Offline counterfactual ONLY: masks/windows remain frozen. This
            # does not simulate a network or rerun the event detector.
            total=mass.sum()
            mass=gaussian_filter1d(mass,local_smoothing_sd_ms/dt,axis=1,mode='constant',cval=0.,truncate=5.)
            retained=float(mass.sum()/total)
        try:r=event_representation(mass,t,np.isfinite(centroid))
        except ValueError as exc:raise ValueError(f'{path.name} event {idx}, smoothing SD {local_smoothing_sd_ms}: {exc}') from exc
        if local_smoothing_sd_ms==0 and np.nanmax(np.abs(r['centroid']-centroid))>=.002:
            raise ValueError('full-envelope centroids disagree with frozen observer')
        rows.append(r)
        details.append(dict(event_id=int(idx),local_width_ms=r['local_width_ms'],retained_window_mass_fraction=retained))
    return (stack_events(rows) if rows else None),details,names,worker


def u_components(kxx,cross_mean,target_constant,minimum_events=16):
    n=len(kxx)
    if n<minimum_events:return dict(status='INSUFFICIENT_EVENTS',N=n,D_off=None)
    a=float(kxx.mean()-2*np.mean(cross_mean)+target_constant)
    b=float((np.trace(kxx)/n-kxx.mean())/(n-1))
    off=float((kxx.sum()-np.trace(kxx))/(n*(n-1))-2*np.mean(cross_mean)+target_constant)
    if not np.isclose(off,a-b,atol=1e-10):raise RuntimeError('U identity failed')
    return dict(status='ESTIMABLE',N=n,A=a,B=b,D_off=off)


class VectorKernel:
    """Equal linear + multi-bandwidth RBF, each in patient-variance units."""
    def __init__(self,reference,weights):
        self.reference=np.asarray(reference,float);self.weights=np.asarray(weights,float)
        if len(self.reference)!=len(self.weights) or not np.isclose(self.weights.sum(),1) or np.any(self.weights<0):raise ValueError('reference weights must be a probability measure')
        self.dim=self.reference.shape[1]
        self.mean=self.weights@self.reference
        centered=self.reference-self.mean
        self.linear_variance=float(self.weights@np.mean(centered**2,axis=1))
        self.linear_scale=max(self.linear_variance,1e-6)
        self.bandwidth=float(np.sqrt(max(2*self.linear_variance,1e-6)))
        rr=self.parts(self.reference,self.reference,normalize=False)['nonlinear']
        self.nonlinear_variance=float(1-self.weights@rr@self.weights)
        self.nonlinear_scale=max(self.nonlinear_variance,1e-6)
        self.reference_parts=self.parts(self.reference,self.reference)
        self.constants={k:float(self.weights@v@self.weights) for k,v in self.reference_parts.items()}

    def parts(self,x,y,normalize=True):
        x=np.asarray(x,float);y=np.asarray(y,float)
        d=cdist(x,y,'sqeuclidean')/self.dim
        nonlinear=sum(np.exp(-d/(2*(self.bandwidth*h)**2)) for h in [.5,1.,2.])/3
        linear=(x-self.mean)@(y-self.mean).T/self.dim
        if normalize:
            linear/=self.linear_scale;nonlinear/=self.nonlinear_scale
        return dict(mean=linear,nonlinear=nonlinear)

    def score(self,x):
        xx=self.parts(x,x);xy=self.parts(x,self.reference);out={}
        for k in xx:out[k]=u_components(xx[k],xy[k]@self.weights,self.constants[k])
        if len(x)<16:return dict(status='INSUFFICIENT_EVENTS',N=len(x),D_off=None,components=out)
        return dict(status='ESTIMABLE',N=len(x),D_off=float(sum(v['D_off'] for v in out.values())/2),
                    A=float(sum(v['A'] for v in out.values())/2),B=float(sum(v['B'] for v in out.values())/2),components=out)

    def population(self,x,weights):
        weights=np.asarray(weights,float);xx=self.parts(x,x);xy=self.parts(x,self.reference)
        return float(sum(weights@xx[k]@weights-2*weights@xy[k]@self.weights+self.constants[k] for k in xx)/2)


class ParticipationKernel:
    """Exact mask-pattern frequency plus contact participation means."""
    def __init__(self,fit_masks):
        x=np.asarray(fit_masks,float);self.c=x.shape[1];self.mean=x.mean(0)
        self.powers=(1<<np.arange(self.c)).astype(np.int64)
        codes=x.astype(np.int64)@self.powers
        self.probabilities=np.bincount(codes,minlength=1<<self.c)/len(x)
        self.linear_scale=max(float(np.mean(self.mean*(1-self.mean))),1e-6)
        self.pattern_scale=max(float(1-np.sum(self.probabilities**2)),1e-6)
        self.pattern_constant=float(np.sum(self.probabilities**2)/self.pattern_scale)

    def score(self,x):
        x=np.asarray(x,float);codes=x.astype(np.int64)@self.powers;xc=x-self.mean
        lin=xc@xc.T/self.c/self.linear_scale
        pattern=(codes[:,None]==codes[None,:]).astype(float)/self.pattern_scale
        out=dict(mean=u_components(lin,np.zeros(len(x)),0.),
                 nonlinear=u_components(pattern,self.probabilities[codes]/self.pattern_scale,self.pattern_constant))
        if len(x)<16:return dict(status='INSUFFICIENT_EVENTS',N=len(x),D_off=None,components=out)
        return dict(status='ESTIMABLE',N=len(x),D_off=float(sum(v['D_off'] for v in out.values())/2),
                    A=float(sum(v['A'] for v in out.values())/2),B=float(sum(v['B'] for v in out.values())/2),components=out)

    def population(self,x,weights):
        x=np.asarray(x,float);w=np.asarray(weights,float)
        codes=x.astype(np.int64)@self.powers
        p=np.bincount(codes,weights=w,minlength=1<<self.c)
        return float(.5*(np.mean((w@x-self.mean)**2)/self.linear_scale+
                          np.sum((p-self.probabilities)**2)/self.pattern_scale))


class FrozenCentroidAnchor:
    """Reuse full-FIT unlabelled features; do not invoke the old classifier."""
    def __init__(self,old_objective,fit_times):
        self.old=old_objective
        features=self.old.embedding(fit_times)
        self.target=features.mean(0)
        if not np.allclose(self.target,self.old.target_global):raise ValueError('full FIT source changed')
        self.scale=max(float(np.mean(np.sum((features-self.target)**2,axis=1))),1e-6)
        self.n_reference=len(features)

    def score(self,times):
        x=(self.old.embedding(times)-self.target)/np.sqrt(self.scale)
        return u_components(x@x.T,np.zeros(len(x)),0.)


class ContactEventObjectiveV2:
    def __init__(self,patient,weights,fit_masks,names,centroid_anchor=None):
        self.names=list(names);self.weights=np.asarray(weights,float);self.minimum_events=16
        self.participation=ParticipationKernel(fit_masks)
        self.patient_participation=self.participation.mean
        self.marked={};self.conditional_reference_support={}
        for block in ['local_shape','recruitment']:
            kernels=[];support=[]
            for c,p in enumerate(self.patient_participation):
                m=patient['participation'][:,c]>0
                if not m.any():raise ValueError(f'no patient shape support for {self.names[c]}')
                cw=self.weights[m]/self.weights[m].sum()
                # Match large FIT participation frequency, using the waveform
                # packet only for the contact-conditional temporal distribution.
                v=patient[block][m,c]/50.
                reference=np.vstack([np.zeros((1,v.shape[1]+1)),np.column_stack([np.ones(len(v)),v])])
                w=np.r_[1-p,p*cw]
                kernels.append(VectorKernel(reference,w));support.append(int(m.sum()))
            self.marked[block]=kernels;self.conditional_reference_support[block]=support
        self.joint=VectorKernel(patient['joint_envelope'],self.weights)
        self.centroid_anchor=centroid_anchor

    def marked_values(self,table,block,c):
        m=table['participation'][:,c]
        return np.column_stack([m,table[block][:,c]/50.*m[:,None]])

    def score(self,table,physical_status=None):
        n=0 if table is None else len(table['participation'])
        if n<16 or physical_status=='RUNAWAY':return dict(status='RUNAWAY' if physical_status=='RUNAWAY' else 'INSUFFICIENT_EVENTS',N=n,loss=None)
        if table['participation'].shape[1]!=len(self.names):raise ValueError('contact count changed')
        scores={'participation':self.participation.score(table['participation'])}
        for block,kernels in self.marked.items():
            cc=[k.score(self.marked_values(table,block,c)) for c,k in enumerate(kernels)]
            scores[block]={key:float(np.mean([r[key] for r in cc])) for key in ['D_off','A','B']}
            scores[block]['contacts']={name:r for name,r in zip(self.names,cc)}
        scores['joint_envelope']=self.joint.score(table['joint_envelope'])
        if self.centroid_anchor is not None:scores['centroid_structure']=self.centroid_anchor.score(table['centroid'])
        return dict(status='ESTIMABLE',N=n,loss=float(np.mean([s['D_off'] for s in scores.values()])),
                    A=float(np.mean([s['A'] for s in scores.values()])),B=float(np.mean([s['B'] for s in scores.values()])),
                    blocks=scores,weights={k:1/len(scores) for k in scores},model_mode_classifier_used=False,
                    native_causal_regularization='NOT_INCLUDED_NOT_YET_VALIDATED',version=VERSION)

    def population(self,table,weights):
        """Exact four-block reference-measure diagnostic; not a model ranking."""
        scores={'participation':self.participation.population(table['participation'],weights)}
        for block,kernels in self.marked.items():
            scores[block]=float(np.mean([k.population(self.marked_values(table,block,c),weights) for c,k in enumerate(kernels)]))
        scores['joint_envelope']=self.joint.population(table['joint_envelope'],weights)
        if self.centroid_anchor is not None:
            x=self.centroid_anchor.old.embedding(table['centroid'])
            scores['centroid_structure']=float(np.sum((np.asarray(weights)@x-self.centroid_anchor.target)**2)/self.centroid_anchor.scale)
        return scores

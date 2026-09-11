"""Patient-derived event-distribution objective, evaluated per fixed network.

No named propagation route, source-region rule, native movie or validation
metric enters this module. Patient modes and kernel maps are fitted elsewhere
on FIT and frozen before candidate evaluation.
"""
import numpy as np
from src.topic4_joint_xy_kernel import event_kernel_features, kernel_map
from src.topic4_interictal_repaired_evaluation import rank_features


def matched_mean_distance(features, target, sample_count):
    """Exact mean squared distance over all size-m subsets, without replacement.

    This removes the arbitrary Monte Carlo draw from matched-count scoring.
    It is an algebraic finite-sample identity, not an iid confidence interval.
    """
    x=np.asarray(features,float);target=np.asarray(target,float);m=int(sample_count)
    if x.ndim!=2 or target.shape!=(x.shape[1],) or not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError('finite events x features and matching target required')
    if m<2:raise ValueError('matched count must be at least two')
    n=len(x)
    if n<m:return None
    mean=x.mean(0);population_variance=np.mean(np.sum((x-mean)**2,axis=1))
    return float(np.sum((mean-target)**2)+(n-m)/(m*(n-1))*population_variance)


class MultieventDistributionObjective:
    def __init__(self, frozen_evaluator, *, matched_count=16):
        ev=frozen_evaluator
        self.km=ev.km;self.xy=ev.xy;self.groups=ev.groups;self.scale=ev.scale;self.maps=ev.maps
        self.k=ev.k;self.matched_count=int(matched_count)
        self.proportions=np.bincount(ev.fit_labels,minlength=self.k)/len(ev.fit_labels)
        if np.any(self.proportions<=0):raise ValueError('empty patient FIT mode')
        phi=self.embedding(ev.fit)
        self.target_global=phi.mean(0)
        self.target_modes=np.concatenate([np.r_[1.,phi[ev.fit_labels==k].mean(0)] for k in range(self.k)])/np.sqrt(self.k)
        self.normalizers=None

    def embedding(self,times):
        return kernel_map(event_kernel_features(times,self.xy,self.groups,self.scale)['joint'],self.maps['joint']).astype(float)

    def balanced_embedding(self,phi,labels):
        # The indicator mass measures the patient's mixture, not a forced 50/50.
        z=np.column_stack([np.ones(len(phi)),phi])
        return np.column_stack([z*((labels==k)/self.proportions[k])[:,None] for k in range(self.k)])/np.sqrt(self.k)

    def components(self,times):
        t=np.asarray(times,float)
        if t.ndim!=2 or t.shape[1]!=len(self.xy) or np.isinf(t).any():raise ValueError('events x contacts required')
        if len(t)<self.matched_count:return {'status':'INSUFFICIENT_EVENTS','n_events':len(t),'global':None,'balanced_modes':None}
        if np.any(np.isfinite(t).sum(1)<2):raise ValueError('unreadable event in training table')
        phi=self.embedding(t);labels=self.km.predict(rank_features(t))
        return {'status':'ESTIMABLE','n_events':len(t),
                'global':matched_mean_distance(phi,self.target_global,self.matched_count),
                'balanced_modes':matched_mean_distance(self.balanced_embedding(phi,labels),self.target_modes,self.matched_count)}

    def score_network(self,times):
        out=self.components(times)
        if out['status']!='ESTIMABLE':return {**out,'loss':None}
        if self.normalizers is None:raise RuntimeError('patient-only calibration not frozen')
        loss=.5*out['global']/self.normalizers['global']+.5*out['balanced_modes']/self.normalizers['balanced_modes']
        return {**out,'loss':float(loss)}

    def score_candidate(self,networks):
        units={str(seed):self.score_network(t) for seed,t in networks.items()}
        eligible=bool(units) and all(v['loss'] is not None for v in units.values())
        return {'status':'ESTIMABLE' if eligible else 'INSUFFICIENT_EVENTS','per_network':units,
                'loss':float(np.mean([v['loss'] for v in units.values()])) if eligible else None,
                'aggregation':'equal weight per fixed network; no pooling across network realizations'}

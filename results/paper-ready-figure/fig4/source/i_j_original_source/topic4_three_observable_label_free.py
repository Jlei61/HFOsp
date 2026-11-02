"""Offline label-free counterpart: input is event x contact centroid times only.

Reuse only the original label-free feature maps. No classifier, class proportions,
mode target, class weighting, or per-mode calibration is retained.
"""
from copy import deepcopy
import numpy as np
from src.topic4_three_observable_objective import ThreeObservableObjective, GROUPS, validate, signed_statistic

class LabelFreeThreeObservableObjective:
    version='three_observable_label_free_offline_v1'
    raw_features=ThreeObservableObjective.raw_features
    mapped=ThreeObservableObjective.mapped
    features=ThreeObservableObjective.features

    def __init__(self, feature_source):
        for key in ['names','scl','icl','pairs','time_scales','maps','n_fourier','seed']:
            setattr(self,key,deepcopy(getattr(feature_source,key)))
        self.targets={};self.scales=None

    def feature_moments(self,t,device='cpu'):
        t=validate(t);s={}
        for start in range(0,len(t),256):
            for g,x in self.features(t[start:start+256],device).items():
                if g not in s:s[g]=dict(N=0,total=np.zeros(x.shape[1]),norm=0.)
                s[g]['N']+=len(x);s[g]['total']+=x.sum(0);s[g]['norm']+=float((x*x).sum())
        return s

    def fit_targets(self,t,device='cpu'):
        self.targets={g:s['total']/s['N'] for g,s in self.feature_moments(t,device).items()}

    def calibrate(self,cal,draws,device='cpu'):
        features={g:[] for g in GROUPS}
        for start in range(0,len(cal),256):
            for g,x in self.features(cal[start:start+256],device).items():features[g].append(x)
        features={g:np.concatenate(x) for g,x in features.items()};values={g:[] for g in GROUPS}
        for d in draws:
            ids=d['cal_indices'];assert len(ids)==16
            for g,x in features.items():values[g].append(float(np.sum((x[ids].mean(0)-self.targets[g])**2)))
        self.scales={g:max(float(np.median(v)),1e-8) for g,v in values.items()}
        return dict(scales=self.scales,A_values=values,draws=draws,label_usage=False)

    def score(self,t,device='cpu'):
        t=validate(t);n=len(t)
        if n<16:return dict(N=n,status='INSUFFICIENT_EVENTS',J=None,groups={})
        groups={}
        for g,s in self.feature_moments(t,device).items():
            q=signed_statistic(s['total'],s['norm'],n,self.targets[g]);q['scaled']=q['D_off']/self.scales[g];groups[g]=q
        return dict(N=n,status='SCORABLE',J=float(np.mean([groups[g]['scaled'] for g in GROUPS])),groups=groups)

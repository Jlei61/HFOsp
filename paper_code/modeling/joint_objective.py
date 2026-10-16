"""Same-event joint distance used for the paper's core-position experiment.

Concatenate rank, timing and participation BEFORE the nonlinear Fourier map.
The marginal objective J remains the core-count comparison's training target.
"""
import numpy as np
from modeling.src.topic4_three_observable_objective import signed_statistic


class JointObjective:
    version = 'same_event_joint_v1'

    def __init__(self, marginal, seed=2026091601):
        self.old = marginal
        self.keys = ['rank', 'scl_pairs', 'icl_pairs', 'rod_lag', 'mask']
        self.weights = dict(rank=1/3, scl_pairs=1/9, icl_pairs=1/9, rod_lag=1/9, mask=1/3)
        dim = sum(marginal.maps[k]['weights'].shape[0] for k in self.keys)
        rng = np.random.default_rng(seed)
        self.W = np.column_stack([rng.normal(size=(dim, 2048))/s for s in [.5, 1., 2.]])
        self.phase = rng.uniform(0, 2*np.pi, self.W.shape[1])

    def features(self, times):
        raw = self.old.raw_features(times)
        x = np.column_stack([raw[k]*np.sqrt(self.weights[k])/self.old.maps[k]['bandwidth'] for k in self.keys])
        return np.cos(x@self.W+self.phase)*np.sqrt(2/self.W.shape[1])

    def fit(self, fit, cal, draws):
        total = np.zeros(self.W.shape[1])
        for start in range(0, len(fit), 128):
            total += self.features(fit[start:start+128]).sum(0)
        self.target = total/len(fit)
        values = np.concatenate([self.features(cal[i:i+128]) for i in range(0, len(cal), 128)])
        self.scale = float(np.median([np.sum((values[ix].mean(0)-self.target)**2) for ix in draws]))
        if self.scale <= 0:
            raise ValueError('A positive independent CAL scale is required')
        return self

    def score(self, times):
        if len(times) < 16:
            return dict(J_joint=None, joint={})
        features = self.features(times)
        q = signed_statistic(features.sum(0), float(np.square(features).sum()), len(times), self.target)
        q['scaled'] = q['D_off']/self.scale
        return dict(J_joint=q['scaled'], joint=q)

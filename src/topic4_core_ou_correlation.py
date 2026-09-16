"""Optional core-wide OU correlation assay; not enabled by existing workers.

For each core k, eta_k = sqrt(rho)*xi_global + sqrt(1-rho)*xi_k.
The independent xi_k use the same OU law and zero initial condition as the
existing global process. Marginal variance and time constant are preserved;
rho changes cross-core conditional-rate correlation. Poisson innovations still
differ neuron by neuron. No patient label, event detection or spatial route is
an input to this class. Outside-core E receives exactly zero added rate.
"""
import numpy as np


class CoreOUMixture:
    def __init__(self,core_index,signal_per_ms,mean_scale,rho,dt_ms,tau_ms,sigma_n,dynamics_seed):
        self.core_index=np.asarray(core_index,int)
        if not np.isin(self.core_index,[-1,0,1]).all():raise ValueError('expected two core identities')
        if not (0<=rho<=1 and 0<mean_scale<=1 and dt_ms>0 and tau_ms>0 and sigma_n>=0):
            raise ValueError('invalid core OU parameters')
        self.rho=float(rho);self.mean_scale=float(mean_scale);self.signal=float(signal_per_ms);self.dt=float(dt_ms)
        self.common_loading=float(np.sqrt(rho));self.independent_loading=float(np.sqrt(1-rho))
        self.a=float(np.exp(-dt_ms/tau_ms));self.stationary_std=float(sigma_n*1e-3*np.sqrt(tau_ms/2))
        self.b=self.stationary_std*np.sqrt(1-self.a*self.a)
        self.seed_namespace=[int(dynamics_seed),20260911,201]
        self.rng=np.random.default_rng(np.random.SeedSequence(self.seed_namespace))
        self.state=np.zeros(2);self.values=np.zeros(len(self.core_index));self.members=[np.flatnonzero(self.core_index==k) for k in [0,1]]
        self.n_steps=0

    def step(self,time_ms):
        if not np.isclose(time_ms,self.n_steps*self.dt,rtol=0,atol=1e-7):
            raise ValueError('core OU must advance exactly once per physical step, beginning at zero')
        self.state=self.a*self.state+self.b*self.rng.standard_normal(2)
        for k,idx in enumerate(self.members):
            self.values[idx]=(self.mean_scale-1)*self.signal+self.independent_loading*self.state[k]
        self.n_steps+=1
        return self.values

    def metadata(self):
        return dict(rho=self.rho,common_loading=self.common_loading,independent_loading=self.independent_loading,
            stationary_std_per_ms=self.stationary_std,discrete_a=self.a,discrete_b=self.b,
            independent_seed_namespace=self.seed_namespace,initial_state=[0.,0.],mean_scale=self.mean_scale,
            law='Same marginal OU law, different between-core correlation, before nonnegative-rate clipping.',
            no_input='No patient labels, detections, core activation times or routes are consumed.')

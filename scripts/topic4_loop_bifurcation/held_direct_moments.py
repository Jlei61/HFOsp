"""Exact held-state input map at K9.35; no transported response derivative."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy import sparse
from campaign import ROOT,read
from conditional_density_inputs import OPS


class HeldInputs:
    def __init__(self):
        self.geo=dict(np.load(OPS/'geometry.npz'));self.params=read(OPS/'prepared.json')['params']
        self.group=self.geo['cell_group'];self.sizes=self.geo['group_size'];self.N=len(self.group);self.P=len(self.sizes)
        self.E=np.arange(self.N)<32000;self.groupE=self.geo['population']==0
        self.S=sparse.coo_matrix((1/self.sizes[self.group],(self.group,np.arange(self.N))),shape=(self.P,self.N)).tocsr()
        self.weights=np.where(self.groupE,self.sizes/32000,0.)
        self.tm=np.where(self.E,self.params['tau_m_E'],self.params['tau_m_I'])
        self.jext=np.where(self.E,self.params['J_ext_E'],self.params['J_ext_I'])
        self.area=np.array([.1/(self.params[n]*(-np.expm1(-.1/self.params[n]))) for n in ['tau_r_AMPA','tau_r_GABA']])
        self.causal=.1/(15*(-np.expm1(-.1/15)))
        self.reference=dict(np.load(ROOT/'held_exit_phase_stationarity_K9p35/inputs.npz'))
        self.Z=self.reference['Z'];self.K=self.reference['K'];self.external=self.reference['external_per_ms']
        self.W=[sparse.load_npz(ROOT/'target_stationary_response_audit'/f'{name}_dc.npz').tocsr()
            for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]

    def moments(self,source_Hz,M):
        assert source_Hz.min()>=0 and M.min()>=0 and np.all(M[~self.E]==0)
        R=float(self.causal*(self.weights@source_Hz));G=float(30*np.clip((R-200)/300,0,1))
        g=self.E*self.Z*G+self.K;h=1+g
        a,b,qa,qb=[w@(source_Hz/1000) for w in self.W]
        IE=self.tm*self.area[0]*(a+self.jext*self.external);II=self.tm*self.area[1]*b
        physical=np.c_[(IE-self.Z*II-.0005*M+self.E*self.Z*G*(-17.662847938268442)-30*self.K)/h,
            self.tm*self.area[0]**2*(qa+self.jext**2*self.external)/h**2,
            self.tm*(self.Z*self.area[1])**2*qb/h**2]
        assert np.isfinite(physical).all() and physical[:,1:].min()>=0
        return physical,g,G,R

    def aggregate_sem(self,sem):
        return np.sqrt(np.bincount(self.group,weights=sem**2,minlength=self.P))/self.sizes

    def target_action(self,chi,direction):
        h=1+self.reference['g'];D=1+.0005*self.E*chi[:,0]/h
        assert D.min()>.5
        C=np.array([chi[:,0]*self.tm*self.area[0]/(1000*h),-chi[:,0]*self.Z*self.tm*self.area[1]/(1000*h),
            chi[:,1]*self.tm*self.area[0]**2/(1000*h**2),chi[:,2]*self.tm*(self.Z*self.area[1])**2/(1000*h**2)])/D
        R=float(self.reference['stationary_R_Hz'])
        dG=.1*self.causal*float(self.weights@direction) if 200<R<500 else 0.
        return sum(c*(w@direction) for c,w in zip(C,self.W))+self.E*self.Z*chi[:,3]/D*dG,D

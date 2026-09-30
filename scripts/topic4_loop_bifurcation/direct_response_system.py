"""Measured local DC linearisation and exact stationary-input construction.

This is a local numerical predictor. Nonlinear responses are always evaluated
with the original density cell updates, never with an extrapolated rate fit.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,gmres
from campaign import ROOT,read
from conditional_density_inputs import OPS

EG=-17.662847938268442
EK=-30.


class DirectDC:
    def __init__(self):
        self.geo=dict(np.load(OPS/'geometry.npz'));self.p=read(OPS/'prepared.json')['params']
        self.group=self.geo['cell_group'];self.sizes=self.geo['group_size'];self.P=len(self.sizes);self.N=len(self.group)
        self.E=np.arange(self.N)<32000;self.groupE=self.geo['population']==0
        self.S=sparse.coo_matrix((1/self.sizes[self.group],(self.group,np.arange(self.N))),shape=(self.P,self.N)).tocsr()
        self.weights=np.where(self.groupE,self.sizes/32000,0.)
        self.tm=np.where(self.E,self.p['tau_m_E'],self.p['tau_m_I'])
        self.jext=np.where(self.E,self.p['J_ext_E'],self.p['J_ext_I'])
        self.area=np.array([.1/(self.p[n]*(-np.expm1(-.1/self.p[n]))) for n in ['tau_r_AMPA','tau_r_GABA']])
        self.causal=.1/(15*(-np.expm1(-.1/15)))
        with np.load(ROOT/'target_stationary_response_audit/response.npz') as z:
            self.Z=z['Z'];self.Kshape=z['K']/9.;self.reference_g=z['g'];self.external=z['mean_external_rate_per_ms']
            self.reference_physical=z['physical'];self.reference_M=z['M']
            self.reference_source=z['mean_source_rate_per_ms'].astype(float)*1000
        self.W=[sparse.load_npz(ROOT/'target_stationary_response_audit'/f'{name}_dc.npz').tocsr()
            for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
        with np.load(ROOT/'all_target_dc_direct/measured_dc.npz') as z:
            self.chi=z['gain'][:,:,0];self.blocks=z['replicate_block_gain'][:,:,0]
        assert np.isfinite(self.chi).all()
        self.C,self.u,self.D,self.fk=self.coefficients(self.chi)
        self.dG=.1*self.causal*self.weights
        self.J=LinearOperator((self.P,self.P),matvec=lambda v:self.S@self.target_action(v)-v,dtype=float)
        diagonal=-np.ones(self.P)
        for c,w in zip(self.C,self.W):diagonal+=self.S@(c*np.asarray(w[np.arange(self.N),self.group]).ravel())
        diagonal+=(self.S@self.u)*self.dG
        diagonal=np.where(abs(diagonal)>.05,diagonal,-1.)
        self.pre=LinearOperator((self.P,self.P),matvec=lambda v:v/diagonal,dtype=float)

    def coefficients(self,chi):
        h=1+self.reference_g
        D=1+.0005*self.E*chi[:,0]/h
        assert D.min()>.5
        C=np.array([chi[:,0]*self.tm*self.area[0]/(1000*h),-chi[:,0]*self.Z*self.tm*self.area[1]/(1000*h),
            chi[:,1]*self.tm*self.area[0]**2/(1000*h**2),chi[:,2]*self.tm*(self.Z*self.area[1])**2/(1000*h**2)])/D
        u=self.E*self.Z*chi[:,3]/D
        fk=self.Kshape*(chi[:,3]+chi[:,0]*(EK-EG)/h)/D
        return C,u,D,fk

    def target_action(self,direction,delta_K=0.,block=None):
        C,u,D,fk=(self.C,self.u,self.D,self.fk) if block is None else self.coefficients(self.blocks[:,:,block])
        return sum(c*(w@direction) for c,w in zip(C,self.W))+u*(self.dG@direction)+fk*delta_K

    def solve(self,rhs):
        history=[]
        v,info=gmres(self.J,rhs,M=self.pre,rtol=1e-7,atol=1e-9,restart=100,maxiter=3,
            callback=lambda r:history.append(float(r)),callback_type='pr_norm')
        return v,dict(gmres_info=int(info),iterations=len(history),linear_max_residual_Hz=float(abs(self.J@v-rhs).max()))

    def moments(self,source_Hz,M,Kmean):
        assert source_Hz.min()>=0 and M.min()>=0
        R=float(self.causal*(self.weights@source_Hz));G=30*np.clip((R-200)/300,0,1)
        K=self.Kshape*Kmean;g=self.E*self.Z*G+K;h=1+g
        a,b,qa,qb=[w@(source_Hz/1000) for w in self.W]
        IE=self.tm*self.area[0]*(a+self.jext*self.external);II=self.tm*self.area[1]*b
        physical=np.c_[(IE-self.Z*II-.0005*M+self.E*self.Z*G*EG+K*EK)/h,
            self.tm*self.area[0]**2*(qa+self.jext**2*self.external)/h**2,
            self.tm*(self.Z*self.area[1])**2*qb/h**2]
        assert np.isfinite(physical).all() and physical[:,1:].min()>=0
        return physical,g,float(G),R

    def field(self,cellrate):
        from coupled_density_exit import ADAPTED
        geo=np.load(ADAPTED/'geometry.npz');cell=geo['group_cell'][self.group[self.E]]
        count=np.bincount(cell,minlength=400)
        return np.bincount(cell,weights=cellrate[self.E],minlength=400)/np.maximum(count,1)

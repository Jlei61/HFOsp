"""Candidate closure: fast AMPA transfer averaged over slow GABA fluctuations.

This changes an approximation, not the biological network. Its quasi-static
GABA assumption and network dynamics require validation; no Hopf claim follows
from defining this class.
"""
from topic4_e_only_z_rate import EOnlySystem
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import erfcx
import numpy as np


class MixedTimescaleSystem(EOnlySystem):
    def __init__(self,grid=10,tau=20.611550480127335,quadrature=5):
        super().__init__(grid=grid,tau=tau)
        self.gh_nodes,self.gh_weights=hermgauss(quadrature)
        self.gh_weights/=np.sqrt(np.pi)
        self.lg_nodes,self.lg_weights=leggauss(16)
        a=np.exp(-self.dt/self.rg);b=np.exp(-self.dt/self.tau)
        tm=np.r_[np.full(self.n,self.m.tau_mem_e_ms),np.full(self.n,self.m.tau_mem_i_ms)]
        factor=self.dt*(tm/self.rg*(1-b)/(a-b))**2*(a*a/(1-a*a)+b*b/(1-b*b)-2*a*b/(1-a*b))
        self.white_to_current_variance=factor/tm
        self.tm=tm

    def _lif(self,mu,sigma,threshold,tm,tref):
        mu,sigma,threshold=np.broadcast_arrays(mu,sigma,threshold)
        lower=(self.m.v_reset_mv-mu)/sigma;upper=(threshold-mu)/sigma
        sample=(.5*(lower+upper))[...,None]+(.5*(upper-lower))[...,None]*self.lg_nodes
        integral=.5*(upper-lower)*np.sum(self.lg_weights*erfcx(-sample),axis=-1)
        with np.errstate(over='ignore',invalid='ignore'):
            denominator=tref+tm*np.sqrt(np.pi)*integral
            result=np.divide(1.,denominator,out=np.zeros_like(denominator),where=np.isfinite(denominator))
        return np.clip(result,0,1/tref)

    def phi(self,mu,ex,inh):
        sigma=np.sqrt(np.maximum(inh*self.white_to_current_variance,0))
        shifted=mu-2.065/2*np.sqrt(np.maximum(ex*(self.ra+self.ta)/self.tm,1e-16))
        means=shifted[None,:]+np.sqrt(2)*sigma[None,:]*self.gh_nodes[:,None]
        fast_sigma=np.sqrt(np.maximum(ex,1e-12));n=self.n;m=self.m
        pe=self._lif(means[:,:n,None],fast_sigma[None,:n,None],m.threshold_nodes_e[None,:,:],m.tau_mem_e_ms,m.tau_ref_e_ms)
        pi=self._lif(means[:,n:],fast_sigma[None,n:],m.v_threshold_i_mv,m.tau_mem_i_ms,m.tau_ref_i_ms)
        return np.r_[np.sum(self.gh_weights[:,None]*np.sum(pe*m.threshold_weights_e[None,:,:],axis=-1),axis=0),
                     np.sum(self.gh_weights[:,None]*pi,axis=0)]

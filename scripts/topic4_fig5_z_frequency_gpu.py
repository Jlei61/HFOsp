"""Exact delay Fourier sums, evaluated with GPU sparse matrix products."""
import numpy as np
import time
import torch
from scipy import sparse


class Frequency:
    def __init__(self,coos,vcoos,n,D,device=0):
        torch.set_num_threads(1);self.device=f'cuda:{device}';self.n=n;self.matrices={}
        for tag,collection in [('',coos),('v',vcoos)]:
            for key,c in collection.items():
                a=sparse.coo_matrix((c.data,(c.row*n+c.col%n,c.col//n)),shape=(n*n,D)).tocsr()
                self.matrices[tag+key]=a
    def evaluate(self,key,delays,omega):
        phase=delays[:,None]*omega[None,:];c=np.cos(phase);s=np.sin(phase)
        X=torch.tensor(np.c_[c,-s,phase*s,phase*c],device=self.device)
        matrix=torch.tensor(self.matrices[key].toarray(),device=self.device)
        ans=(matrix@X).cpu().numpy();K=len(omega);n=self.n
        w=np.ascontiguousarray((ans[:,:K]+1j*ans[:,K:2*K]).T.reshape(K,n,n))
        wp=np.ascontiguousarray((ans[:,2*K:3*K]+1j*ans[:,3*K:]).T.reshape(K,n,n))
        return w,wp


def install(o,device=0):
    from topic4_fig5_z_frozen_v1 import filters,rate_filters
    engine=Frequency(o.coos,o.vcoos,o.m.n,o.m.D,device)
    def kernels(T):
        if o.cached is not None and T==o.cached[0]:return o.cached[1:5]
        m=o.m;om=2*np.pi*o.k/T;lam=1j*om;H={};HP={}
        for key in o.coos:
            me,mp,va,vp=filters(m,lam,key[-1]=='e')
            for tag,f,fp in [(key,me,mp),('v'+key,va,vp)]:
                begin=time.time()
                w,wp=engine.evaluate(tag,(np.arange(m.D)+1)*m.dt,om)
                H[tag]=w*f[:,None,None];HP[tag]=wp*f[:,None,None]+w*fp[:,None,None]
                print('GPU DELAY KERNEL',tag,T,time.time()-begin,flush=True)
        le,li,lm,lep,lip,lmp=rate_filters(m,lam)
        o.cached=(T,H,le,li,lm,HP,lep,lip,lmp)
        return o.cached[1:5]
    o.kernels=kernels
    return engine

"""Recover local filter states and full delay history from rate harmonics."""


def recover(o,kernels,cf,D,dt):
    cp=o.cp;s=o.s;K=o.K;lam=2j*cp.pi*cp.arange(K)[:,None]/o.recovery_period
    tm,ref,th,alpha,tf,ts,E=o.gpars;ops=kernels[0];H=kernels[-2]
    a,b,qa,qb=[(op@cf.ravel()).reshape(K,s.P) for op in ops]
    target=cf/H;xa=target/(1+lam*tf);xb=target/(1+lam*ts)
    qav=tm*s.area[0]*a/(1+lam*s.rise[0]);iav=qav/(1+lam*s.decay[0])
    qgv=tm*s.area[1]*b/(1+lam*s.rise[1]);igv=qgv/(1+lam*s.decay[1])
    va=tm*s.area[0]**2*qa/(1+lam*s.tau[0]/2);vg=tm*s.area[1]**2*qb/(1+lam*s.tau[1]/2)
    m=.5*E*cf/(1+lam*1000);factor=cp.full((K,1),2.);factor[0]=1;factor[-1]=1
    local=cp.stack([cp.sum(v*factor,axis=0).real for v in [xa,xb,qav,iav,qgv,igv,va,vg,m]])
    phase=cp.exp(-cp.arange(1,D+1)[:,None]*dt*lam[:,0][None,:])
    history=(phase@(cf*factor)).real
    return cp.r_[local.ravel(),history.ravel()].get()


def recover_bounded(s,cf,period,J,D,dt,device=0,block_size=64):
    """Identical nine-state/history reconstruction in harmonic blocks.

    Retain every harmonic, delayed edge and population. Only the temporary
    sparse operators and filter arrays are bounded; no modal truncation or
    approximation is introduced. This also avoids building a 2N harmonic
    index bank for an antiperiodic mode.
    """
    from rate_periodic import Periodic,np
    o=Periodic(s,2*block_size,device);cp=o.cp
    cf=np.asarray(cf);K=len(cf)
    assert cf.ndim==2 and cf.shape[1]==s.P and period>0 and dt>0
    local=cp.zeros((9,s.P));history=cp.zeros((D,s.P))
    tm,ref,th,alpha,tf,ts,E=o.gpars
    ages=cp.arange(1,D+1)[:,None]*dt
    for first in range(0,K,block_size):
        last=min(K,first+block_size);count=last-first
        lam=2j*cp.pi*cp.arange(first,last)[:,None]/period
        edge_phase=cp.exp(-cp.asarray(s.delays)[:,None]*lam[:,0])
        v=cp.asarray(cf[first:last]);arrivals=[]
        for kind,(d,mask,index,ptr) in enumerate(o.raw):
            scale=cp.where(mask,J**(1 if kind==0 else 2),1.) if kind in (0,2) else 1.
            data=(d@edge_phase).T*scale;edges=d.shape[0]
            op=o.cs.csr_matrix((data.ravel(),index[:count*edges],ptr[:count*s.P+1]),
                shape=(count*s.P,count*s.P))
            arrivals.append((op@v.ravel()).reshape(count,s.P))
        a,b,qa,qb=arrivals
        H=alpha/(1+lam*tf)+(1-alpha)/(1+lam*ts)
        target=v/H;xa=target/(1+lam*tf);xb=target/(1+lam*ts)
        qav=tm*s.area[0]*a/(1+lam*s.rise[0]);iav=qav/(1+lam*s.decay[0])
        qgv=tm*s.area[1]*b/(1+lam*s.rise[1]);igv=qgv/(1+lam*s.decay[1])
        va=tm*s.area[0]**2*qa/(1+lam*s.tau[0]/2)
        vg=tm*s.area[1]**2*qb/(1+lam*s.tau[1]/2)
        m=.5*E*v/(1+lam*1000)
        factor=cp.full((count,1),2.)
        if first==0:factor[0]=1.
        if last==K:factor[-1]=1.
        for i,state in enumerate([xa,xb,qav,iav,qgv,igv,va,vg,m]):
            local[i]+=cp.sum(state*factor,axis=0).real
        phase=cp.exp(-ages*lam[:,0][None,:])
        history+=(phase@(v*factor)).real
    result=cp.r_[local.ravel(),history.ravel()].get()
    o.cache=None
    return result

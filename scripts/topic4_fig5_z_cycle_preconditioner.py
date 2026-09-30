"""Mean-gain Fourier preconditioner; nonlinear model and residual are unchanged."""
import numpy as np
from scipy.linalg import lu_factor,lu_solve


class MeanGainInverse:
    def __init__(self,o,ge,gi):
        self.o=o;m=o.m;n=m.n;K=m.K;self.U=o.U
        _,self.H,self.le,self.li,lm,*_=o.cached
        self.gu,self.gv,self.gw=[g.mean(0) for g in ge]
        ui,vi,wi=[g.mean(0) for g in gi]
        self.den=1+self.le[:,None]*m.eta_M*self.gu[None,:]*lm[:,None]
        def avg(x):return (x*m.w_u).reshape(len(o.k),n,K).sum(2)
        am=avg(self.le[:,None]*self.gu/self.den);av=avg(self.le[:,None]*self.gv/self.den)
        az=avg(self.le[:,None]*self.gu*o.z/self.den);aw=avg(self.le[:,None]*self.gw*o.z2/self.den)
        self.lus=[]
        for k in range(min(len(o.k),getattr(o,'preconditioner_modes',len(o.k)))):
            h={name:a[k] for name,a in self.H.items()}
            A=np.empty((2*n,2*n),complex)
            A[:n,:n]=np.eye(n)-m.te*(am[k,:,None]*h['ee']+av[k,:,None]*h['vee'])
            A[:n,n:]=m.te*(az[k,:,None]*h['ei']-aw[k,:,None]*h['vei'])
            A[n:,:n]=-self.li[k]*m.ti*(ui[:,None]*h['ie']+vi[:,None]*h['vie'])
            A[n:,n:]=np.eye(n)+self.li[k]*m.ti*(ui[:,None]*h['ii']-wi[:,None]*h['vii'])
            self.lus.append(lu_factor(A,check_finite=False))

    def __call__(self,rhs):
        o=self.o;m=o.m;n=m.n;K=m.K;U=self.U
        q=np.fft.rfft(rhs,axis=0);u=q[:,:U]/self.den
        macro=np.c_[(u*m.w_u).reshape(len(o.k),n,K).sum(2),q[:,U:]]
        sol=macro.copy()
        sol[:len(self.lus)]=np.array([lu_solve(lu,macro[k],check_finite=False) for k,lu in enumerate(self.lus)])
        e,i=sol[:,:n],sol[:,n:]
        def conv(key,r):return np.einsum('kij,kj->ki',self.H[key],r)
        mean=np.repeat(conv('ee',e),K,axis=1)-o.z*np.repeat(conv('ei',i),K,axis=1)
        va=np.repeat(conv('vee',e),K,axis=1);vg=o.z2*np.repeat(conv('vei',i),K,axis=1)
        du=u+self.le[:,None]*m.te*(self.gu*mean+self.gv*va+self.gw*vg)/self.den
        return np.fft.irfft(np.c_[du,i],n=o.N,axis=0)


def bordered_inverse(o,jr,ft,fs,pp,ap,arc_t=0.,arc_s=0.):
    base=MeanGainInverse(o,*jr.transfer_gains)
    vt=base(ft);vs=base(fs)
    B=np.array([[np.sum(vt*pp),np.sum(vs*pp)],[np.sum(vt*ap)-arc_t,np.sum(vs*ap)-arc_s]])
    if not np.isfinite(B).all() or np.linalg.cond(B)>1e12:
        raise ValueError('Singular mean-gain border')
    def apply(x):
        r=base(x[:-2].reshape(o.N,-1))
        par=np.linalg.solve(B,np.r_[np.sum(r*pp)-x[-2],np.sum(r*ap)-x[-1]])
        r-=vt*par[0]+vs*par[1]
        return np.r_[r.ravel(),par]
    return apply

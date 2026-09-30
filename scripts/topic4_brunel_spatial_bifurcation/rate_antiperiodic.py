"""Locate multiplier -1 through the full antiperiodic variational BVP.

u(t+T)=-u(t). The full delay operator is evaluated at odd harmonics of 2T.
A zero eigenvalue of this operator means a Floquet multiplier -1, not +1.
"""
from rate_periodic import *
from scipy.sparse.linalg import LinearOperator,eigs
import gc


class ReferenceAntiperiodic:
    def __init__(self,s,path,N=256,device=0):
        z=np.load(path);self.s=s;self.N=N;self.J=float(z['J']);self.T=float(z['T'])
        r=resample(z['r'],N,axis=0);o=Periodic(s,2*N,device);self.o=o;cp=o.cp;self.cp=cp
        self.kernels=o.kernels(2*self.T,self.J);r=cp.asarray(np.r_[r,r])
        mom=o.moments(r,self.kernels)+o.private[:,None,:];gains=[]
        for k in range(3):
            step=1e-5*cp.maximum(cp.abs(mom[k]),1.);hi=mom.copy();lo=mom.copy();hi[k]+=step;lo[k]-=step
            gains.append((o.phi(hi)-o.phi(lo))/(2*step))
        self.gains=cp.stack(gains);self.calls=0
        cp.get_default_memory_pool().free_all_blocks()

    def apply(self,x):
        cp=self.cp;o=self.o;u=x.reshape(self.N,self.s.P);full=cp.concatenate([u,-u],axis=0)
        dphi=cp.sum(self.gains*o.moments(full,self.kernels),axis=0)
        return (full-o.filt(dphi,self.kernels[-2]))[:self.N].ravel()

    def compute(self,k=3):
        from cupyx.scipy.sparse.linalg import LinearOperator as CL,gmres
        cp=self.cp;dim=self.N*self.s.P;shift=1e-4
        cg=CL((dim,dim),matvec=lambda x:self.apply(x)-shift*x,dtype=float)
        def mv(x):return self.apply(cp.asarray(x)).get()
        def inverse(x):
            y,info=gmres(cg,cp.asarray(x),tol=1e-9,atol=1e-12,restart=160,maxiter=2400)
            error=float(cp.linalg.norm(cg@y-cp.asarray(x))/cp.linalg.norm(cp.asarray(x)))
            self.calls+=1
            if self.calls%10==0:print('ANTIPERIODIC',self.calls,info,error,flush=True)
            if error>1e-6:raise RuntimeError(f'antiperiodic inversion residual {error}')
            return y.get()
        op=LinearOperator((dim,dim),matvec=mv,dtype=float);inv=LinearOperator((dim,dim),matvec=inverse,dtype=float)
        vals,vec=eigs(op,k=k,sigma=shift,OPinv=inv,ncv=16,tol=2e-7,maxiter=60)
        errors=[]
        for i,v in enumerate(vals):
            av=mv(vec[:,i].real)+1j*mv(vec[:,i].imag);errors.append(float(np.linalg.norm(av-v*vec[:,i])))
        return vals,vec,errors


class Antiperiodic(ReferenceAntiperiodic):
    """Exactly the same operator using only odd harmonics of the 2T extension.

    Omitting identically zero even harmonics and unused parameter derivatives
    saves memory without dropping spatial groups, delays or temporal modes.
    """
    def __init__(self,s,path,N=256,device=0,low_memory=False,harmonic_chunk_size=0,stream_harmonics=False):
        assert N%2==0
        z=np.load(path);self.s=s;self.N=N;self.J=float(z['J']);self.T=float(z['T'])
        base=Periodic(s,N,device);base.low_memory=True;base.harmonic_chunk_size=harmonic_chunk_size
        base.stream_harmonics=stream_harmonics
        base.derivative_chunk_size=harmonic_chunk_size;cp=base.cp;self.cp=cp
        r=cp.asarray(resample(z['r'],N,axis=0))
        mom=base.moments(r,base.kernels(self.T,self.J))+base.private[:,None,:]
        gains=[]
        for k in range(3):
            step=1e-5*cp.maximum(cp.abs(mom[k]),1.);hi=mom.copy();lo=mom.copy();hi[k]+=step;lo[k]-=step
            gains.append((base.phi(hi)-base.phi(lo))/(2*step))
        self.gains=cp.stack(gains)
        # Derivative operators close over their Periodic owner. Break that
        # cache cycle before constructing the antiperiodic operator bank.
        base.cache=None;del base,mom,hi,lo,r,gains
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        o=Periodic(s,N-2,device);self.o=o;self.K=N//2
        zz=1j*np.pi*cp.arange(1,N,2)/self.T;lam=zz[:,None]
        phase=cp.exp(-cp.asarray(s.delays)[:,None]*zz);self.ops=[]
        self.low_memory=low_memory or stream_harmonics;self.delay_phase=phase
        self.harmonic_chunk_size=harmonic_chunk_size;self.stream_harmonics=stream_harmonics
        for k,(d,mask,index,ptr) in enumerate(o.raw):
            if self.low_memory:continue
            scale=cp.where(mask,self.J**(1 if k==0 else 2),1.) if k in (0,2) else 1.
            if harmonic_chunk_size:
                vals=cp.empty((self.K,d.shape[0]),complex)
                for first in range(0,self.K,harmonic_chunk_size):
                    last=min(self.K,first+harmonic_chunk_size)
                    vals[first:last]=(d@phase[:,first:last]).T*scale
            else:vals=(d@phase).T.copy()*scale
            self.ops.append(o.cs.csr_matrix((vals.ravel(),index,ptr),shape=(self.K*s.P,self.K*s.P)))
        self.fil=(1/((1+lam*s.rise[0])*(1+lam*s.decay[0])),
                  1/((1+lam*s.rise[1])*(1+lam*s.decay[1])),
                  1/(1+lam*s.tau[0]/2),1/(1+lam*s.tau[1]/2),1/(1+lam*1000))
        tm,ref,th,alpha,tf,ts,E=o.gpars
        self.H=alpha/(1+lam*tf)+(1-alpha)/(1+lam*ts);self.calls=0

    def transform(self,u):
        return self.cp.fft.rfft(self.cp.concatenate([u,-u],axis=0),axis=0)[1::2]

    def inverse_transform(self,v):
        cp=self.cp;cf=cp.zeros((self.N+1,self.s.P),dtype=complex);cf[1::2]=v
        return cp.fft.irfft(cf,n=2*self.N,axis=0)[:self.N]

    def apply(self,x):
        cp=self.cp;s=self.s;u=x.reshape(self.N,s.P);uf=self.transform(u);v=uf.ravel()
        if self.low_memory:
            arrivals=[]
            for k,(d,mask,index,ptr) in enumerate(self.o.raw):
                scale=cp.where(mask,self.J**(1 if k==0 else 2),1.) if k in (0,2) else 1.
                if self.stream_harmonics:
                    result=cp.empty_like(v);edges=d.shape[0];block=self.harmonic_chunk_size or 64
                    for first in range(0,self.K,block):
                        last=min(self.K,first+block)
                        vals=(d@self.delay_phase[:,first:last]).T*scale
                        ii=index[first*edges:last*edges]-first*s.P
                        pp=ptr[first*s.P:last*s.P+1]-first*edges
                        op=self.o.cs.csr_matrix((vals.ravel(),ii,pp),shape=((last-first)*s.P,)*2)
                        result[first*s.P:last*s.P]=op@v[first*s.P:last*s.P]
                    arrivals.append(result.reshape(self.K,s.P));continue
                vals=(d@self.delay_phase).T.copy()*scale
                op=self.o.cs.csr_matrix((vals.ravel(),index,ptr),shape=(self.K*s.P,self.K*s.P))
                arrivals.append((op@v).reshape(self.K,s.P))
                del vals,op
            a,b,qa,qb=arrivals
        else:a,b,qa,qb=[(op@v).reshape(self.K,s.P) for op in self.ops]
        ha,hg,hva,hvg,hm=self.fil;tm,ref,th,alpha,tf,ts,E=self.o.gpars
        cm=[tm*(s.area[0]*ha*a-s.area[1]*hg*b)-.5*E*hm*uf,
            tm*s.area[0]**2*hva*qa,tm*s.area[1]**2*hvg*qb]
        dphi=sum(self.gains[k]*self.inverse_transform(cm[k]) for k in range(3))
        return (u-self.inverse_transform(self.transform(dphi)*self.H)).ravel()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--N',type=int,default=256);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();s=RateField();o=Antiperiodic(s,a.path,a.N,a.device);vals,v,errors=o.compute()
    dest=PERIODIC_OUT/'antiperiodic';dest.mkdir(exist_ok=True)
    row=dict(orbit=a.path,J_EE_core=o.J,T_ms=o.T,N=a.N,eigenvalues=vals,residuals=errors,
        meaning='Only a zero crossing establishes a candidate multiplier -1 bifurcation; these eigenvalues are not Floquet multipliers')
    write(dest/f'{Path(a.path).stem}_N{a.N}.json',row);save_periodic_array(dest/f'{Path(a.path).stem}_N{a.N}.npz',eigenvalues=vals,vectors=v)
    print('ANTIPERIODIC RESULT',row,flush=True)

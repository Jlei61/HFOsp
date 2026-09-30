"""Locate multiplier -1 through the full antiperiodic variational BVP.

u(t+T)=-u(t). The full delay operator is evaluated at odd harmonics of 2T.
A zero eigenvalue of this operator means a Floquet multiplier -1, not +1.
"""
from rate_periodic import *
from scipy.sparse.linalg import LinearOperator,eigs


class Antiperiodic:
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
            y,info=gmres(cg,cp.asarray(x),tol=1e-9,atol=1e-12,restart=80,maxiter=480)
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--N',type=int,default=256);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();s=RateField();o=Antiperiodic(s,a.path,a.N,a.device);vals,v,errors=o.compute()
    dest=PERIODIC_OUT/'antiperiodic';dest.mkdir(exist_ok=True)
    row=dict(orbit=a.path,J_EE_core=o.J,T_ms=o.T,N=a.N,eigenvalues=vals,residuals=errors,
        meaning='Only a zero crossing establishes a candidate multiplier -1 bifurcation; these eigenvalues are not Floquet multipliers')
    write(dest/f'{Path(a.path).stem}_N{a.N}.json',row);np.savez_compressed(dest/f'{Path(a.path).stem}_N{a.N}.npz',eigenvalues=vals,vectors=v)
    print('ANTIPERIODIC RESULT',row,flush=True)

"""Complex Floquet exponents from the periodic delay characteristic BVP.

delta r(t)=exp(lambda*t)*u(t), u(t+T)=u(t). Eliminate the same linear
filters at lambda+i*k*omega; retain all 935 spatial groups. Use bordered
Newton corrections, with exponents seeded by core equilibrium modes or
continued periodic eigenfunctions. Monodromy is the independent check.
"""
from periodic_zm import *


class SpectralFloquet:
    def __init__(self,s,path,N=64,device=0):
        z=np.load(path);self.s=s;self.N=N;self.J=1.;self.depletion=float(z['D']);s.set_D(self.depletion);self.T=float(z['T'])
        r=resample(z['r'],N,axis=0);base=ZMPeriodic(s,N,device);cp=base.cp;self.cp=cp
        mom=base.moments(cp.asarray(r),base.kernels(self.T,self.depletion))+base.private[:,None,:];gains=[]
        for k in range(3):
            step=1e-5*cp.maximum(cp.abs(mom[k]),1.);hi=mom.copy();lo=mom.copy();hi[k]+=step;lo[k]-=step
            gains.append((base.phi(hi)-base.phi(lo))/(2*step))
        self.gains=cp.stack(gains);del base,mom,hi,lo
        self.o=ZMPeriodic(s,2*(N-1),device);self.o.N=N
        self.z=cp.asarray(s.Z);self.last=None;self.cache=None;cp.get_default_memory_pool().free_all_blocks()

    def kernels(self,lam):
        if self.last==lam:return self.cache
        cp=self.cp;s=self.s;o=self.o;self.cache=None
        zz=lam+2j*np.pi*cp.fft.fftfreq(self.N)*self.N/self.T;z=zz[:,None]
        phase=cp.exp(-cp.asarray(s.delays)[:,None]*zz);phasep=-cp.asarray(s.delays)[:,None]*phase
        ops=[]
        for k,(d,mask,index,ptr) in enumerate(o.raw):
            scale=cp.where(mask,self.J**(1 if k==0 else 2),1.) if k in (0,2) else 1.
            make=lambda x:o.cs.csr_matrix((x.ravel(),index,ptr),shape=(self.N*s.P,self.N*s.P))
            ops.append(make((d@phase).T.copy()*scale))
        ha=1/((1+z*s.rise[0])*(1+z*s.decay[0]));hg=1/((1+z*s.rise[1])*(1+z*s.decay[1]))
        hva=1/(1+z*s.tau[0]/2);hvg=1/(1+z*s.tau[1]/2);hm=1/(1+z*1000)
        hap=-ha*(s.rise[0]/(1+z*s.rise[0])+s.decay[0]/(1+z*s.decay[0]))
        hgp=-hg*(s.rise[1]/(1+z*s.rise[1])+s.decay[1]/(1+z*s.decay[1]))
        hvap=-hva**2*s.tau[0]/2;hvgp=-hvg**2*s.tau[1]/2;hmp=-hm**2*1000
        tm,ref,th,alpha,tf,ts,E=o.gpars
        H=alpha/(1+z*tf)+(1-alpha)/(1+z*ts);Hp=-alpha*tf/(1+z*tf)**2-(1-alpha)*ts/(1+z*ts)**2
        self.last=lam;self.cache=(ops,phasep,(ha,hg,hva,hvg,hm),(hap,hgp,hvap,hvgp,hmp),H,Hp);return self.cache

    def moments(self,u,kernels,derivative=False):
        cp=self.cp;s=self.s;ops,phasep,fil,filp,H,Hp=kernels;uf=cp.fft.fft(u,axis=0);v=uf.ravel();tm,ref,th,alpha,tf,ts,E=self.o.gpars
        def apply(operators,filters,adapt=True):
            a,b,qa,qb=[(op@v).reshape(self.N,s.P) for op in operators];ha,hg,hva,hvg,hm=filters
            return cp.stack([tm*(s.area[0]*ha*a-self.z*s.area[1]*hg*b)-(.5*E*hm*uf if adapt else 0),
                tm*s.area[0]**2*hva*qa,tm*(self.z*s.area[1])**2*hvg*qb])
        if derivative:
            terms=[]
            for d,mask,index,ptr in self.o.raw:
                values=(d@phasep).T.copy()
                op=self.o.cs.csr_matrix((values.ravel(),index,ptr),shape=(self.N*s.P,self.N*s.P))
                terms.append((op@v).reshape(self.N,s.P))
                del values,op
                cp.get_default_memory_pool().free_all_blocks()
            a,b,qa,qb=terms;ha,hg,hva,hvg,hm=fil
            out=cp.stack([tm*(s.area[0]*ha*a-self.z*s.area[1]*hg*b),
                          tm*s.area[0]**2*hva*qa,tm*(self.z*s.area[1])**2*hvg*qb])+apply(ops,filp)
        else:out=apply(ops,fil)
        return cp.fft.ifft(out,axis=1)

    def filt(self,u,H):return self.cp.fft.ifft(self.cp.fft.fft(u,axis=0)*H,axis=0)

    def refine(self,lam,u):
        from cupyx.scipy.sparse.linalg import LinearOperator,gmres
        cp=self.cp;u=cp.asarray(u,dtype=cp.complex128).reshape(self.N,self.s.P);pivot=int(cp.argmax(cp.abs(u)));u/=u.ravel()[pivot]
        dim=u.size;history=[]
        for it in range(18):
            kernels=self.kernels(lam);H,Hp=kernels[-2:]
            def apply(v):
                v=v.reshape(u.shape);return v-self.filt(cp.sum(self.gains*self.moments(v,kernels),axis=0),H)
            f=apply(u);err=float(cp.linalg.norm(f)/cp.linalg.norm(u));history.append(err)
            print('FLOQUET BVP',it,lam,np.exp(lam*self.T),err,flush=True)
            if err<2e-9:break
            dl=-self.filt(cp.sum(self.gains*self.moments(u,kernels),axis=0),Hp)-self.filt(cp.sum(self.gains*self.moments(u,kernels,True),axis=0),H)
            def mv(x):return cp.r_[(apply(x[:-1])+dl*x[-1]*.01).ravel(),x[pivot]]
            A=LinearOperator((dim+1,dim+1),matvec=mv,dtype=complex);rhs=cp.r_[-f.ravel(),0j]
            dy,info=gmres(A,rhs,tol=1e-7,atol=1e-12,restart=80,maxiter=640)
            linerr=float(cp.linalg.norm(A@dy-rhs)/cp.linalg.norm(rhs));print('linear',info,linerr,flush=True)
            if linerr>1e-4:break
            step=1.
            if abs(complex(dy[-1])*.01)>.02:step=.02/abs(complex(dy[-1])*.01)
            u+=step*dy[:-1].reshape(u.shape);lam+=step*complex(dy[-1])*.01
            del A,dl,kernels;cp.get_default_memory_pool().free_all_blocks()
        return lam,u.get(),history

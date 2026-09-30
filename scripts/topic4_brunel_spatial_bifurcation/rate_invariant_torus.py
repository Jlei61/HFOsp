"""Two-angle Fourier invariant-torus BVP for the unchanged spatial rate DDE.

Every harmonic uses its physical frequency k*omega+l*nu in all synapses,
rate filters, adaptation and delays. The extra angle resolves slow modulation
directly; no long-time activity, spatial mode truncation, or SNN forcing is used.
"""
from rate_periodic import *


class SeparableTorusSubspace:
    """Exact tensor-product preconditioner space, without dense lifted vectors."""
    def __init__(self,o,modes,limit):
        cp=o.cp;self.cp=cp;self.shape=o.shape;self.size=np.prod(o.shape)+3
        raw=cp.stack([m.ravel()/cp.linalg.norm(m) for m in modes],axis=1)
        U,S,V=cp.linalg.svd(raw,full_matrices=False)
        rank=int(cp.sum(S>cp.max(S)*1e-10));self.B=U[:,:rank].T.reshape(rank,o.nt,o.s.P)
        angle=2*cp.pi*cp.arange(o.np)/o.np;waves=[]
        for l in range(min(o.np//2,limit)+1):
            wave=cp.cos(l*angle);waves.append(wave/cp.linalg.norm(wave))
            if 0<l<o.np//2:
                wave=cp.sin(l*angle);waves.append(wave/cp.linalg.norm(wave))
        self.W=cp.stack(waves);self.m=rank;self.l=len(waves);self.dimension=rank*self.l+3

    def project(self,v):
        cp=self.cp;coeff=cp.einsum('mtp,tsp->ms',self.B,v[:-3].reshape(self.shape))@self.W.T
        return cp.r_[coeff.ravel(),v[-3:]]

    def lift(self,v):
        cp=self.cp;coeff=v[:-3].reshape(self.m,self.l)@self.W
        return cp.r_[cp.einsum('mtp,ms->tsp',self.B,coeff).ravel(),v[-3:]]

    def column(self,j):
        v=self.cp.zeros(self.dimension);v[j]=1.;return self.lift(v)


class Torus:
    def __init__(self,s,nt=64,np_=8,device=0):
        self.s=s;self.nt=nt;self.np=np_;self.N=nt*np_;self.K=nt*(np_//2+1)
        self.base=Periodic(s,2*(self.K-1),device);self.base.N=self.N
        self.cp=cp=self.base.cp
        self.kt=cp.fft.fftfreq(nt)*nt;self.kp=cp.arange(np_//2+1)
        self.shape=(nt,np_,s.P);self.last=None;self.cache=None

    def fft(self,r):
        return self.cp.fft.rfft2(r.reshape(self.shape),axes=(0,1)).reshape(self.K,self.s.P)

    def inverse(self,f):
        return self.cp.fft.irfft2(f.reshape(self.nt,self.np//2+1,self.s.P),s=(self.nt,self.np),axes=(0,1))

    def kernels(self,T,nu,J):
        key=(T,nu,J)
        if key==self.last:return self.cache
        self.last=None;self.cache=None;cp=self.cp;s=self.s;o=self.base
        lam=1j*(self.kt[:,None]*2*np.pi/T+self.kp[None,:]*nu).ravel()[:,None]
        phase=cp.exp(-cp.asarray(s.delays)[:,None]*lam[:,0]);ops=[]
        for k,(d,mask,index,ptr) in enumerate(o.raw):
            scale=cp.where(mask,J**(1 if k==0 else 2),1.) if k in (0,2) else 1.
            data=(d@phase).T.copy()*scale
            ops.append(o.cs.csr_matrix((data.ravel(),index,ptr),shape=(self.K*s.P,self.K*s.P)))
        ha=1/((1+lam*s.rise[0])*(1+lam*s.decay[0]));hg=1/((1+lam*s.rise[1])*(1+lam*s.decay[1]))
        va=1/(1+lam*s.tau[0]/2);vg=1/(1+lam*s.tau[1]/2);m=1/(1+1000*lam)
        tm,ref,th,alpha,tf,ts,E=o.gpars
        H=alpha/(1+lam*tf)+(1-alpha)/(1+lam*ts)
        self.last=key;self.cache=(ops,(ha,hg,va,vg,m),H);return self.cache

    def moments(self,r,k):
        cp=self.cp;s=self.s;rf=self.fft(r);ops,filters,H=k
        a,b,qa,qb=[(op@rf.ravel()).reshape(self.K,s.P) for op in ops]
        ha,hg,va,vg,m=filters;tm,ref,th,alpha,tf,ts,E=self.base.gpars
        ff=[tm*(s.area[0]*ha*a-s.area[1]*hg*b)-.5*E*m*rf,
            tm*s.area[0]**2*va*qa,tm*s.area[1]**2*vg*qb]
        return cp.stack([self.inverse(v).reshape(self.N,s.P) for v in ff])

    def filt(self,r,H):return self.inverse(self.fft(r)*H)

    def projection(self,r,q):
        first=self.cp.fft.fft(r,axis=1)[:,1,:]/self.np
        return self.cp.vdot(q,first)/self.cp.vdot(q,q)

    def evaluate(self,y,ref,phase,q,amp,nu0,derivative=False,arc=None):
        from cupyx.scipy.sparse.linalg import LinearOperator
        cp=self.cp;s=self.s
        r=y[:-3].reshape(self.shape)*.001;T=float(cp.exp(y[-3]));nu=float(y[-2])*nu0;J=float(y[-1])*.001
        k=self.kernels(T,nu,J);H=k[-1]
        mom=self.moments(r,k)+self.base.private[:,None,:]
        phi=self.base.phi(mom).reshape(self.shape)
        f=(r-self.filt(phi,H))/.001;z=self.projection(r,q)
        constraints=[cp.sum((r-ref)*phase)/.001,z.imag/amp,z.real/amp-1]
        if arc is not None:
            pred,tangent,weight=arc
            constraints[-1]=cp.sum((y-pred)*tangent*weight**2)
        F=cp.r_[f.ravel(),cp.asarray(constraints)]
        if not derivative:return F
        gains=[]
        for i in range(3):
            h=1e-5*cp.maximum(cp.abs(mom[i]),1.);a=mom.copy();b=mom.copy();a[i]+=h;b[i]-=h
            gains.append((self.base.phi(a)-self.base.phi(b))/(2*h))
        gains=cp.stack(gains);cols=[]
        if getattr(self,'low_memory_borders',True):
            # Release the baseline operator bank before finite-differencing
            # the three scalar borders. Rebuild it afterwards for the spatial
            # Jacobian. This retains every harmonic while avoiding two full
            # delayed-operator banks at the same time.
            del k
            self.last=None;self.cache=None
            cp.get_default_memory_pool().free_all_blocks()
        # Only three border columns use finite differences. The large spatial
        # Jacobian acts through the exact linear filters and local Phi gains.
        for j,h in enumerate([1e-5,1e-4,1e-3]):
            a=y.copy();b=y.copy();a[-3+j]+=h;b[-3+j]-=h
            cols.append((self.evaluate(a,ref,phase,q,amp,nu0,arc=arc)-self.evaluate(b,ref,phase,q,amp,nu0,arc=arc))/(2*h))
        if getattr(self,'low_memory_borders',True):k=self.kernels(T,nu,J);H=k[-1]
        else:self.last=None;self.cache=None
        def mv(dy):
            dr=dy[:-3].reshape(self.shape)*.001
            dphi=cp.sum(gains*self.moments(dr,k),axis=0).reshape(self.shape)
            df=(dr-self.filt(dphi,H))/.001;dz=self.projection(dr,q)
            out=cp.r_[df.ravel(),cp.asarray([cp.sum(dr*phase)/.001,dz.imag/amp,dz.real/amp])]
            if arc is not None:out[-1]=cp.sum(dy[:-3]*tangent[:-3]*weight[:-3]**2)
            for j in range(3):out+=cols[j]*dy[-3+j]
            return out
        return F,LinearOperator((len(y),len(y)),matvec=mv,dtype=float)

    def solve(self,r,T,nu,J,q,amp,nu0,maxiter=16,tol=1e-10,arc=None):
        from cupyx.scipy.sparse.linalg import gmres,LinearOperator
        cp=self.cp;r=cp.asarray(r);q=cp.asarray(q);ref=r.copy()
        dr=cp.fft.ifft(1j*self.kt[:,None,None]*cp.fft.fft(ref,axis=0),axis=0).real
        phase=dr/cp.sum(dr*dr)*.001
        y=cp.r_[(r/.001).ravel(),cp.asarray([np.log(T),nu/nu0,J/.001])];history=[];start=time.time()
        if arc is not None:arc=tuple(cp.asarray(v) for v in arc)
        # Resolve the near-resonant phase/Floquet directions explicitly in the
        # preconditioner. This changes only the linear solver: A and the final
        # nonlinear residual retain every group and every two-angle grid value.
        modes=[dr.mean(1),q.real,q.imag];limit=getattr(self,'precondition_harmonics',self.np//2)
        count=getattr(self,'snapshot_modes',0)
        if count:
            snapshots=cp.transpose(r-r.mean(1,keepdims=True),(0,2,1)).reshape(self.nt*self.s.P,self.np)
            U,S,V=cp.linalg.svd(snapshots,full_matrices=False)
            count=min(count,int(cp.sum(S>S[0]*1e-10)))
            modes += [U[:,j].reshape(self.nt,self.s.P) for j in range(count)]
            del snapshots,U,S,V
        if getattr(self,'separable_preconditioner',False):
            subspace=SeparableTorusSubspace(self,modes,limit)
            project,lift,column,nsub=subspace.project,subspace.lift,subspace.column,subspace.dimension
        else:
            basis=[];angle=2*cp.pi*cp.arange(self.np)/self.np
            for l in range(min(self.np//2,limit)+1):
                waves=[cp.cos(l*angle)]+([cp.sin(l*angle)] if 0<l<self.np//2 else [])
                for mode in modes:
                    for wave in waves:
                        v=cp.r_[(mode[:,None,:]*wave[None,:,None]).ravel(),cp.zeros(3)]
                        initial_norm=float(cp.linalg.norm(v))
                        for _ in range(2):
                            for u in basis:v-=u*cp.dot(u,v)
                        norm=float(cp.linalg.norm(v))
                        if norm>max(1e-12,initial_norm*1e-8):basis.append(v/norm)
            for j in range(3):
                v=cp.zeros_like(y);v[-3+j]=1.;basis.append(v)
            Q=cp.stack(basis,axis=1);del basis
            project=lambda v:Q.T@v
            lift=lambda v:Q@v
            column=lambda j:Q[:,j]
            nsub=Q.shape[1]
        for it in range(maxiter):
            F,A=self.evaluate(y,ref,phase,q,amp,nu0,True,arc=arc);error=float(cp.max(cp.abs(F)));history.append(error)
            print('TORUS BVP',self.nt,self.np,it,'J',float(y[-1])*.001,'T',float(cp.exp(y[-3])),
                'nu',float(y[-2])*nu0,'error',error,'sec',round(time.time()-start,1),flush=True)
            if error<tol:break
            small=cp.stack([project(A@column(j)) for j in range(nsub)],axis=1);inverse=cp.linalg.inv(small)
            def precondition(v):
                c=project(v);return v+lift(inverse@c-c)
            M=LinearOperator(A.shape,matvec=precondition,dtype=float)
            # Normalize only the linear right-hand side. This avoids an
            # absolute Krylov stopping floor as the nonlinear defect becomes
            # tiny; the full nonlinear equations/tolerance stay unchanged.
            rhs_norm=cp.linalg.norm(F)
            dy,info=gmres(A,-F/rhs_norm,M=M,tol=min(.02,max(2e-5,error*.02)),atol=0.,restart=getattr(self,'krylov_restart',120),maxiter=2400)
            dy*=rhs_norm
            linerr=float(cp.linalg.norm(A@dy+F)/cp.linalg.norm(F));print('TORUS LINEAR',info,linerr,flush=True)
            if linerr>.1:break
            del A,M
            cp.get_default_memory_pool().free_all_blocks()
            for back in range(14):
                yy=y+dy*2.**-back
                if abs(float(yy[-3]-y[-3]))>.1 or float(yy[-2])<=0:continue
                ff=self.evaluate(yy,ref,phase,q,amp,nu0,arc=arc)
                if float(cp.linalg.norm(ff))<float(cp.linalg.norm(F)):y=yy;break
            else:break
        error=float(cp.max(cp.abs(self.evaluate(y,ref,phase,q,amp,nu0,arc=arc))))
        return y[:-3].reshape(self.shape).get()*.001,float(cp.exp(y[-3])),float(y[-2])*nu0,float(y[-1])*.001,error,history


def main(a):
    s=RateField();critical_source=PERIODIC_OUT/f'{a.critical}_N128.json'
    critical=read(critical_source);base=np.load(critical['orbit'])
    mode=np.load(PERIODIC_OUT/f'{a.critical}_mode_N128.npz');T=float(base['T']);J=float(base['J'])
    lam=complex(mode['lam']);omega=2*np.pi/T;shift=round(lam.imag/omega);nu0=lam.imag-shift*omega
    q=resample(mode['u'],a.nt,axis=0)*np.exp(2j*np.pi*shift*np.arange(a.nt)/a.nt)[:,None]
    if nu0<0:q=q.conj();nu0=-nu0
    # A common amplitude coordinate across temporal meshes.
    q/=abs(resample(mode['u'],64,axis=0)).max();parent=resample(base['r'],a.nt,axis=0);nu=nu0
    o=Torus(s,a.nt,a.np,a.device);dest=PERIODIC_OUT/'tori';dest.mkdir(exist_ok=True)
    o.precondition_harmonics=getattr(a,'precondition_harmonics',a.np//2);o.krylov_restart=getattr(a,'krylov_restart',120)
    o.separable_preconditioner=getattr(a,'separable_preconditioner',False)
    o.snapshot_modes=getattr(a,'snapshot_modes',0)
    previous=None
    if a.from_torus:
        seed=np.load(a.from_torus);r=resample(resample(seed['r'],a.nt,axis=0),a.np,axis=1)
        T=float(seed['T']);nu=float(seed['nu']);J=float(seed['J']);previous=float(seed['amplitude_hz'])/1000
    for amp_hz in a.amplitudes:
        amp=amp_hz/1000
        if previous is None:r=parent[:,None,:]+2*amp*np.real(q[:,None,:]*np.exp(2j*np.pi*np.arange(a.np)/a.np)[None,:,None])
        else:r=r.mean(1,keepdims=True)+(r-r.mean(1,keepdims=True))*amp/previous
        r,T,nu,J,error,history=o.solve(r,T,nu,J,q,amp,nu0,tol=a.tol)
        tag=f'{a.label}_{a.branch_tag}_a{amp_hz:.6f}Hz_N{a.nt}x{a.np}'
        np.savez_compressed(dest/(tag+'.npz'),r=r,T=T,nu=nu,J=J,q=q,amplitude_hz=amp_hz,residual=error,history=history)
        row=dict(status='CONVERGED' if error<a.tol else 'NOT_CONVERGED',J_EE_core=J,T_ms=T,
            modulation_period_ms=2*np.pi/nu,modulation_frequency_per_ms=nu,amplitude_hz=amp_hz,
            N_theta=a.nt,N_psi=a.np,residual=error,parent_critical_source=str(critical_source),
            source=str(dest/(tag+'.npz')),stability='NOT_COMPUTED',
            amplitude_reference='Critical mode normalized on a common Ntheta=64 mesh',requested_tolerance=a.tol,
            limitation='Two-angle collocation solution; mesh convergence and small-amplitude approach to TR required before classifying criticality.')
        write(dest/(tag+'.json'),row);print('TORUS SAVED',row,flush=True)
        if error>=a.tol:break
        previous=amp


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nt',type=int,default=64);p.add_argument('--np',type=int,default=8)
    p.add_argument('--amplitudes',type=float,nargs='+',default=[.005,.01,.02]);p.add_argument('--device',type=int,default=1)
    p.add_argument('--label',default='deflated',help='Output prefix; earlier trial files are retained')
    p.add_argument('--from-torus');p.add_argument('--tol',type=float,default=1e-10)
    p.add_argument('--critical',default='TR_A_B');p.add_argument('--branch-tag',default='TR')
    p.add_argument('--precondition-harmonics',type=int,default=8);p.add_argument('--krylov-restart',type=int,default=60)
    p.add_argument('--separable-preconditioner',action='store_true')
    p.add_argument('--snapshot-modes',type=int,default=0,help='Extra fast-space snapshot directions in the linear preconditioner only')
    a=p.parse_args();main(a)

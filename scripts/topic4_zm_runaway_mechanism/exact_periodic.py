"""Analytic period and Z columns for the unchanged spatial Fourier-Galerkin BVP.

Differentiating the delay phases and every LTI filter removes finite-difference
error from the exceptionally small parameter turns. The rate-direction Jacobian
uses the same independently audited local response derivatives.
"""
from streaming_periodic import StreamGalerkin,HarmonicOperator
from periodic_v3 import RS,TAU_M
from native_path import *


class ExactGalerkin(StreamGalerkin):
    def sample_input_harmonics(self,harmonics):
        return self.interpolate(self.cp.fft.irfft(harmonics,n=self.N,axis=1),self.M,axis=1)

    def linear_inputs(self,dr,T,Z):
        # The Jacobian is homogeneous. Do not add and then subtract the large
        # external moments: Fourier interpolation of that affine offset creates
        # a roundoff floor that can prevent small Newton corrections converging.
        cp=self.cp;s=self.s;K=self.K;P=s.P
        ops,(ha,hg,hva,hvg,hm),(fs,fE,fI,fvE,fvI),lam=self.kernels(T)
        rf=cp.fft.rfft(dr,axis=0);v=rf.ravel()
        a,b,qa,qb=[(x@v).reshape(K,P) for x in ops]
        tm=self.gp[0];E=self.gp[2];Z=cp.asarray(Z)
        mh=tm*(s.area[0]*ha*a-Z*s.area[1]*hg*b)-.5*E*hm*rf
        eh=tm*s.area[0]**2*hva*qa;ih=Z*Z*tm*s.area[1]**2*hvg*qb
        harmonics=cp.stack([mh,eh,ih,fs*mh,fE*eh,fI*ih,fvE*eh,fvI*ih])
        return self.sample_input_harmonics(harmonics)

    def preconditioner(self,dim):
        from cupyx.scipy.sparse.linalg import LinearOperator
        cp=self.cp;scale=cp.ones(dim);scale[self.N*self.s.P]=1e-3
        return LinearOperator((dim,dim),matvec=lambda x:x*scale,dtype=np.float64)

    def solve(self,*args,**kwargs):
        # Right scaling only; CuPy GMRES returns the unscaled solution M*x.
        import cupyx.scipy.sparse.linalg as krylov
        original=krylov.gmres
        def scaled_gmres(A,b,**kw):
            kw['M']=self.preconditioner(A.shape[0])
            cap=getattr(self,'linear_tolerance_cap',None)
            if cap is not None:kw['tol']=min(kw.get('tol',1e-5),cap)
            floor=getattr(self,'linear_tolerance_floor',None)
            if floor is not None:kw['tol']=max(kw.get('tol',1e-5),floor)
            if getattr(self,'log_linear_progress',False):
                kw['callback_type']='pr_norm'
                kw['callback']=lambda residual:log('GMRES restart residual',float(residual))
            if getattr(self,'host_krylov',False):
                from host_krylov import host_gmres
                return host_gmres(A,b,**kw)
            return original(A,b,**kw)
        krylov.gmres=scaled_gmres
        try:return super().solve(*args,**kwargs)
        finally:krylov.gmres=original

    def output_differential(self,di,inp,ph):
        cp=self.cp
        mu,vE,vI,mus,vEf,vIf,vEv,vIv=inp
        dmu,dvE,dvI,dmus,dvEf,dvIf,dvEv,dvIv=di
        w=ph[5:10];gr=ph[10:25].reshape(5,3,self.M,self.s.P).copy()
        gr[:,1]*=(vE>=0)[None];gr[:,2]*=(vI>=0)[None]
        dw=cp.einsum('pcnj,cnj->pnj',gr,cp.stack([dmu,dvE,dvI]))
        dmueff=w[0]*dmu+(1-w[0])*dmus+(mu-mus)*dw[0]+w[3]*(dvE-dvEf)+(vE-vEf)*dw[3]+w[4]*(dvI-dvIf)+(vI-vIf)*dw[4]
        dve=w[1]*dvE+(1-w[1])*dvEv+(vE-vEv)*dw[1]
        dvi=w[2]*dvI+(1-w[2])*dvIv+(vI-vIv)*dw[2]
        return self.interpolate(ph[1]*dmueff+ph[2]*dve+ph[3]*dvi,self.N)

    def parameter_inputs(self,r,T,Z,Zprime=None):
        cp=self.cp;s=self.s;K=self.K;P=s.P
        ops,(ha,hg,hva,hvg,hm),(fs,fE,fI,fvE,fvI),lam=self.kernels(T)
        rf=cp.fft.rfft(r,axis=0);v=rf.ravel()
        a,b,qa,qb=[(x@v).reshape(K,P) for x in ops]
        tm=cp.asarray(s.tm);E=cp.asarray(s.E);Z=cp.asarray(Z)
        mh=tm*(s.area[0]*ha*a-Z*s.area[1]*hg*b)-.5*E*hm*rf
        eh=tm*s.area[0]**2*hva*qa;ih=Z*Z*tm*s.area[1]**2*hvg*qb
        if Zprime is None:
            phase=cp.exp(-cp.asarray(s.delays)[:,None]*lam[:,0])
            dphase=phase*cp.asarray(s.delays)[:,None]*lam[:,0]
            da,db,dqa,dqb=[(HarmonicOperator(self,raw,dphase,cache=False)@v).reshape(K,P) for raw in self.raw]
            def fder(f,tau):return f*lam*tau/(1+lam*tau)
            dha=ha*(lam*s.rise[0]/(1+lam*s.rise[0])+lam*s.decay[0]/(1+lam*s.decay[0]))
            dhg=hg*(lam*s.rise[1]/(1+lam*s.rise[1])+lam*s.decay[1]/(1+lam*s.decay[1]))
            dhva=fder(hva,s.tau[0]/2);dhvg=fder(hvg,s.tau[1]/2);dhm=fder(hm,TAU_M)
            tf,ts,tE,tI,tvE,tvI=[cp.asarray(x) for x in s.poles]
            dfs=fder(fs,ts);dfE=fder(fE,tE);dfI=fder(fI,tI);dfvE=fder(fvE,tvE);dfvI=fder(fvI,tvI)
            dm=tm*(s.area[0]*(dha*a+ha*da)-Z*s.area[1]*(dhg*b+hg*db))-.5*E*dhm*rf
            de=tm*s.area[0]**2*(dhva*qa+hva*dqa)
            di=Z*Z*tm*s.area[1]**2*(dhvg*qb+hvg*dqb)
        else:
            dz=cp.asarray(Zprime);dm=-dz*tm*s.area[1]*hg*b
            de=cp.zeros_like(eh);di=2*Z*dz*tm*s.area[1]**2*hvg*qb
            dfs=dfE=dfI=dfvE=dfvI=0.
        harmonics=cp.stack([dm,de,di,dfs*mh+fs*dm,dfE*eh+fE*de,
                            dfI*ih+fI*di,dfvE*eh+fvE*de,dfvI*ih+fvI*di])
        return self.sample_input_harmonics(harmonics)

    def evaluate(self,y,reference,phase,D,amplitude=None,arc=None,derivative=False):
        from cupyx.scipy.sparse.linalg import LinearOperator
        cp=self.cp;s=self.s;n=self.N*s.P;extra=2 if amplitude is not None or arc is not None else 1
        # Same zero set, unit-norm phase row instead of an almost-zero row.
        phase=phase/cp.linalg.norm(phase)
        r=y[:n].reshape(self.N,s.P)*RS;T=float(cp.exp(y[-extra]));Dv=float(y[-1]*1e-3) if extra==2 else D
        s.set_D(Dv);Z=s.Z.copy();F,inp,ph=self.residual(r,T,Z,True)
        constraints=[cp.sum((r-reference)*phase)/RS]
        if amplitude is not None:
            q,target=amplitude;projection=cp.vdot(q,cp.fft.rfft(r,axis=0)[1]/self.N)/cp.vdot(q,q);constraints.append(projection.real-target)
        if arc is not None:
            yp,tan,weight=arc;constraints.append(cp.sum((y-yp)*tan*weight**2))
        res=cp.concatenate([F.ravel(),cp.asarray(constraints)])
        if not derivative:return res
        if getattr(self,'iteration_checkpoint',None):
            dest=Path(self.iteration_checkpoint);tmp=dest.with_name(f'{dest.stem}.{os.getpid()}.tmp.npz')
            np.savez_compressed(tmp,r=r.get(),T=T,D=Dv,Z=Z,y=y.get(),
                                residual=float(cp.max(abs(res))),status='ITERATE_ONLY')
            tmp.replace(dest)
        colT=(-self.output_differential(self.parameter_inputs(r,T,Z),inp,ph)/RS).ravel()
        colD=None
        if extra==2:
            # Both paths are piecewise linear in per-group Z within this segment.
            dz=path_Z_derivative(s,Dv)
            colD=(-self.output_differential(self.parameter_inputs(r,T,Z,dz),inp,ph)/RS*.001).ravel()
        def matvec(dy):
            dr=dy[:n].reshape(self.N,s.P)*RS;di=self.linear_inputs(dr,T,Z)
            out=((dr-self.output_differential(di,inp,ph))/RS).ravel()+colT*dy[-extra]
            if extra==2:out+=colD*dy[-1]
            cc=[cp.sum(dr*phase)/RS]
            if amplitude is not None:cc.append((cp.vdot(q,cp.fft.rfft(dr,axis=0)[1]/self.N)/cp.vdot(q,q)).real)
            if arc is not None:cc.append(cp.sum(dy*tan*weight**2))
            return cp.concatenate([out,cp.asarray(cc)])
        return res,LinearOperator((len(y),len(y)),matvec=matvec,dtype=np.float64),dict(inputs=inp,phi=ph)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--M',type=int,default=8192)
    p.add_argument('--family',choices=['rate','native'],default='rate');p.add_argument('--device',type=int,default=1)
    p.add_argument('--D',type=float);p.add_argument('--label');a=p.parse_args()
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    z=np.load(a.orbit);D=float(z['D']);T=float(z['T']);o=ExactGalerkin(s,len(z['r']),a.M,a.device);o.cache_mean_operators=False
    cp=o.cp;r=cp.asarray(z['r']);s.set_D(D);Z=s.Z.copy();checks=[]
    for kind in ['logT','D']:
        if kind=='D':
            s.set_D(D+1e-7);zp=s.Z.copy();s.set_D(D-1e-7);zm=s.Z.copy();s.set_D(D)
            di=o.parameter_inputs(r,T,Z,(zp-zm)/(2e-7))
        else:di=o.parameter_inputs(r,T,Z)
        inp=o.inputs(r,T,Z);ph=o.phi(inp);analytic=o.output_differential(di,inp,ph)
        for h in [1e-5,1e-6,1e-7,1e-8,1e-9]:
            if kind=='logT':plus=o.residual(r,T*np.exp(h),Z);minus=o.residual(r,T*np.exp(-h),Z)
            else:
                s.set_D(D+h);plus=o.residual(r,T,s.Z);s.set_D(D-h);minus=o.residual(r,T,s.Z);s.set_D(D)
            fd=-(plus-minus)*RS/(2*h)
            row=dict(column=kind,h=h,relative_error=float(cp.linalg.norm(fd-analytic)/cp.linalg.norm(analytic)))
            if h==1e-7:
                if kind=='logT':ip=o.inputs(r,T*np.exp(h),Z);im=o.inputs(r,T*np.exp(-h),Z)
                else:
                    s.set_D(D+h);ip=o.inputs(r,T,s.Z);s.set_D(D-h);im=o.inputs(r,T,s.Z);s.set_D(D)
                row['input_derivative_relative_error']=float(cp.linalg.norm((ip-im)/(2*h)-di)/cp.linalg.norm(di))
                row['largest_output_error_phase_group']=list(map(int,np.unravel_index(int(cp.argmax(abs(fd-analytic))),fd.shape)))
                del ip,im
            checks.append(row);log('EXACT COLUMN',row)
        del di,inp,ph,analytic,fd,plus,minus;cp.get_default_memory_pool().free_all_blocks()
    write(OUT/f'exact_parameter_columns_{Path(a.orbit).stem}.json',checks)
    assert all(min(q['relative_error'] for q in checks if q['column']==kind)<1e-6 for kind in ['logT','D'])
    if a.D is not None:
        from native_cycles import save
        sol=o.solve(z['r'],T,a.D,maxiter=18,tol=2e-8,restart=40)
        save(s,sol,OUT/'periodic',a.label)

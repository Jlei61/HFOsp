"""Same Galerkin Newton operator with precontracted eight input gains.

The 25 spline/response arrays and eight operating inputs are only needed while
forming the derivative. Keeping eight contracted gains avoids retaining both
full derivative workspaces across Newton iterations, without changing equations.
"""
from exact_periodic import *


class CompactExactGalerkin(ExactGalerkin):
    fft_population_block=64
    def sample_input_harmonics(self,harmonics):
        # N is odd and M >= 2N: all retained positive-frequency harmonics
        # have a full conjugate partner. Direct padding avoids an exactly
        # cancelling odd-N irfft/rfft pair and its prime-size workspace.
        cp=self.cp;out=cp.empty((len(harmonics),self.M,self.s.P))
        for k in range(len(harmonics)):
            # Identical component FFTs. Population blocks avoid a full-P
            # zero-padded FFT workspace beside the retained local gains.
            for lo in range(0,self.s.P,self.fft_population_block):
                hi=min(lo+self.fft_population_block,self.s.P)
                out[k,:,lo:hi]=cp.fft.irfft(harmonics[k,:,lo:hi].T,n=self.M,axis=1).T*(self.M/self.N)
        return out

    def inputs(self,r,T,Z):
        out=self.linear_inputs(r,T,Z)
        for k in (0,3):out[k]+=self.gp[3]
        for k in (1,4,6):out[k]+=self.gp[4]
        return out

    def gains(self,inp,ph):
        cp=self.cp;mu,ve,vi,mus,vEf,vIf,vEv,vIv=inp
        samples=inp.shape[1]
        w=ph[5:10];gr=ph[10:25].reshape(5,3,samples,self.s.P).copy()
        # Newton predictors can leave the positive-variance domain. Response
        # weights use sqrt(max(v,0)); only their variance derivative is zero
        # for v<0. The direct effective-variance derivative is retained below.
        gr[:,1]*=(ve>=0)[None];gr[:,2]*=(vi>=0)[None]
        gain=cp.empty((8,samples,self.s.P))
        for c in range(3):
            direct=w[0] if c==0 else w[3] if c==1 else w[4]
            gain[c]=ph[1]*(direct+(mu-mus)*gr[0,c]+(ve-vEf)*gr[3,c]+(vi-vIf)*gr[4,c])
            gain[c]+=ph[2]*((w[1] if c==1 else 0.)+(ve-vEv)*gr[1,c])
            gain[c]+=ph[3]*((w[2] if c==2 else 0.)+(vi-vIv)*gr[2,c])
        gain[3]=ph[1]*(1-w[0]);gain[4]=-ph[1]*w[3];gain[5]=-ph[1]*w[4]
        gain[6]=ph[2]*(1-w[1]);gain[7]=ph[3]*(1-w[2])
        return gain

    def compact_residual(self,r,T,Z,derivative):
        cp=self.cp;P=self.s.P;inp=self.inputs(r,T,Z)
        output=cp.empty((self.M,P));gain=cp.empty((8,self.M,P)) if derivative else None
        for lo in range(0,self.M,1024):
            hi=min(lo+1024,self.M);n=(hi-lo)*P
            args=[cp.ascontiguousarray(x[lo:hi].ravel()) for x in inp]
            ph=cp.empty((25,n))
            self.phik(((n+127)//128,),(128,),(*args,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,ph,np.int32(n)))
            ph=ph.reshape(25,hi-lo,P);output[lo:hi]=ph[0]
            if derivative:gain[:,lo:hi]=self.gains(inp[:,lo:hi],ph)
        F=(r-self.interpolate(output,self.N))/RS
        return F,gain

    def contracted(self,di,gain):
        cp=self.cp;out=cp.zeros((self.M,self.s.P))
        for i in range(8):out+=gain[i]*di[i]
        return self.interpolate(out,self.N)

    def evaluate(self,y,reference,phase,D,amplitude=None,arc=None,derivative=False):
        from cupyx.scipy.sparse.linalg import LinearOperator
        cp=self.cp;s=self.s;n=self.N*s.P;extra=2 if amplitude is not None or arc is not None else 1
        phase=phase/cp.linalg.norm(phase)
        r=y[:n].reshape(self.N,s.P)*RS;T=float(cp.exp(y[-extra]));Dv=float(y[-1]*1e-3) if extra==2 else D
        s.set_D(Dv);Z=s.Z.copy();F,gain=self.compact_residual(r,T,Z,derivative)
        constraints=[cp.sum((r-reference)*phase)/RS]
        if amplitude is not None:
            q,target=amplitude;projection=cp.vdot(q,cp.fft.rfft(r,axis=0)[1]/self.N)/cp.vdot(q,q);constraints.append(projection.real-target)
        if arc is not None:
            yp,tan,weight=arc;constraints.append(cp.sum((y-yp)*tan*weight**2))
        res=cp.concatenate([F.ravel(),cp.asarray(constraints)])
        if not derivative:return res
        if getattr(self,'iteration_checkpoint',None):
            dest=Path(self.iteration_checkpoint);tmp=dest.with_name(f'{dest.stem}.{os.getpid()}.tmp.npz')
            np.savez_compressed(tmp,r=r.get(),T=T,D=Dv,Z=Z,y=y.get(),residual=float(cp.max(abs(res))),status='ITERATE_ONLY')
            tmp.replace(dest)
        del F;cp.get_default_memory_pool().free_all_blocks()
        colT=(-self.contracted(self.parameter_inputs(r,T,Z),gain)/RS).ravel();colD=None
        if extra==2:
            dz=path_Z_derivative(s,Dv)
            colD=(-self.contracted(self.parameter_inputs(r,T,Z,dz),gain)/RS*.001).ravel()
            self.fixed_D_column_norm=float(cp.linalg.norm(colD))
        def matvec(dy):
            dr=dy[:n].reshape(self.N,s.P)*RS;di=self.linear_inputs(dr,T,Z)
            out=((dr-self.contracted(di,gain))/RS).ravel()+colT*dy[-extra]
            if extra==2:out+=colD*dy[-1]
            cc=[cp.sum(dr*phase)/RS]
            if amplitude is not None:cc.append((cp.vdot(q,cp.fft.rfft(dr,axis=0)[1]/self.N)/cp.vdot(q,q)).real)
            if arc is not None:cc.append(cp.sum(dy*tan*weight**2))
            return cp.concatenate([out,cp.asarray(cc)])
        # Do not retain obsolete operating inputs in the caller's '_' local.
        return res,LinearOperator((len(y),len(y)),matvec=matvec,dtype=np.float64),None


if __name__=='__main__':
    from scipy.signal import resample
    import argparse,gc
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args()
    s=model();attach_rate_entry_path(s);z=np.load(OUT/'periodic/rate_seed_N1024.npz');N=1025;M=4096
    ExactGalerkin.harmonic_block=33;CompactExactGalerkin.harmonic_block=33
    base=ExactGalerkin(s,N,M,a.device);base.cache_mean_operators=False;cp=base.cp
    cp.fft.config.get_plan_cache().set_size(0);r=cp.asarray(resample(z['r'],N,axis=0));n=r.size
    y=cp.r_[(r/RS).ravel(),np.log(float(z['T'])),float(z['D'])*1000]
    phase=cp.fft.irfft(2j*np.pi*cp.arange(base.K)[:,None]*cp.fft.rfft(r,axis=0),n=N,axis=0)
    c=cp.zeros(n+2);c[-2]=1000;arc=(y.copy(),c,cp.ones(n+2))
    f,A,_=base.evaluate(y,r,phase,float(z['D']),arc=arc,derivative=True)
    v=cp.asarray(np.random.default_rng(1942).normal(size=len(y)));v[-2]*=1e-4;v[-1]*=.001
    expected=A@v;expected=expected.get();ff=f.get();del A,base,_;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    compact=CompactExactGalerkin(s,N,M,a.device);compact.cache_mean_operators=False
    g,B,_=compact.evaluate(y,r,phase,float(z['D']),arc=arc,derivative=True);actual=(B@v).get()
    err=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));res=float(abs(g.get()-ff).max())
    # Independent FFT execution can differ by a few 1e-12 in Hz-scaled
    # residuals. This remains 200-fold below the 2e-8 root tolerance.
    assert err<1e-11 and res<1e-10,(err,res)
    q=dict(status='PASS',relative_JVP_error=err,max_residual_difference=res,N=N,M=M,model_change=False,
           residual_roundoff_tolerance=1e-10,root_tolerance=2e-8,
           input_harmonics='direct zero-padding onto M nodes, algebraically identical for odd N',
           response_schedule='same phi_batch kernel in blocks of 1024 temporal nodes',
           inverse_fft_population_block=compact.fft_population_block,
           initial_stricter_check='Residual difference 2.998e-12 exceeded 1e-12; JVP parity passed')
    write(OUT/'compact_periodic_check.json',q);log('COMPACT CHECK',q)

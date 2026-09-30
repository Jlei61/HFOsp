"""Same Galerkin BVP with large-quadrature work arrays kept off the GPU.

Retain the eight exact local gains on the host. Reconstruct operating inputs there,
then contract variational harmonics one component/population block at a time.
This changes memory scheduling only, and is checked against CompactExactGalerkin.
"""
from compact_periodic import *
from scipy.fft import irfft as host_irfft
import gc


class LargeQuadratureGalerkin(CompactExactGalerkin):
    population_block=32
    gain_cache_gb=0.

    def sample_input_harmonics(self,harmonics):
        if getattr(self,'capture_harmonics',False):return harmonics
        return super().sample_input_harmonics(harmonics)

    def harmonics(self,function,*args):
        assert not getattr(self,'capture_harmonics',False)
        self.capture_harmonics=True
        try:return function(*args)
        finally:self.capture_harmonics=False

    def project(self,values):
        cp=self.cp;out=cp.empty((self.N,self.s.P))
        for lo in range(0,self.s.P,self.population_block):
            hi=min(lo+self.population_block,self.s.P)
            h=cp.fft.rfft(values[:,lo:hi].T,axis=1)[:,:self.K]
            out[:,lo:hi]=cp.fft.irfft(h,n=self.N,axis=1).T*(self.N/self.M)
        return out

    def contract_harmonics(self,harmonics,gain):
        cp=self.cp;out=cp.empty((self.N,self.s.P))
        for lo in range(0,self.s.P,self.population_block):
            hi=min(lo+self.population_block,self.s.P)
            local_gain=gain[lo] if isinstance(gain,dict) else gain[:,:,lo:hi]
            if isinstance(local_gain,np.ndarray):local_gain=cp.asarray(np.ascontiguousarray(local_gain))
            temporal=cp.zeros((self.M,hi-lo))
            for k in range(8):
                sampled=cp.fft.irfft(harmonics[k,:,lo:hi].T,n=self.M,axis=1).T
                temporal+=local_gain[k]*sampled*(self.M/self.N)
            h=cp.fft.rfft(temporal.T,axis=1)[:,:self.K]
            out[:,lo:hi]=cp.fft.irfft(h,n=self.N,axis=1).T*(self.N/self.M)
        return out

    def compact_residual(self,r,T,Z,derivative):
        cp=self.cp;P=self.s.P
        h=self.harmonics(self.linear_inputs,r,T,Z)
        inp=np.empty((8,self.M,P))
        for k in range(8):
            hh=np.ascontiguousarray(h[k].get().T)
            inp[k]=host_irfft(hh,n=self.M,axis=1,workers=2).T*(self.M/self.N)
        del h,hh;cp.get_default_memory_pool().free_all_blocks()
        for k in (0,3):inp[k]+=self.s.private_mu
        for k in (1,4,6):inp[k]+=self.s.private_ve
        output=cp.empty((self.M,P));gain=np.empty((8,self.M,P)) if derivative else None
        for lo in range(0,self.M,1024):
            hi=min(lo+1024,self.M);block=cp.asarray(inp[:,lo:hi]);n=(hi-lo)*P
            args=[cp.ascontiguousarray(x.ravel()) for x in block]
            ph=cp.empty((25,n))
            self.phik(((n+127)//128,),(128,),
                      (*args,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,ph,np.int32(n)))
            ph=ph.reshape(25,hi-lo,P);output[lo:hi]=ph[0]
            if derivative:gain[:,lo:hi]=self.gains(block,ph).get()
        F=(r-self.project(output))/RS
        if derivative and self.gain_cache_gb>0:
            # Copy host blocks once; cache only the explicitly budgeted prefix
            # on the GPU. Exact float64 values and contraction are unchanged.
            blocks={};used=0;budget=int(self.gain_cache_gb*1024**3)
            for lo in range(0,P,self.population_block):
                hi=min(lo+self.population_block,P)
                block=np.ascontiguousarray(gain[:,:,lo:hi])
                if used+block.nbytes<=budget:
                    blocks[lo]=cp.asarray(block);used+=block.nbytes
                else:blocks[lo]=block
            gain=blocks
        return F,gain

    def evaluate(self,y,reference,phase,D,amplitude=None,arc=None,derivative=False):
        from cupyx.scipy.sparse.linalg import LinearOperator
        assert amplitude is None,'This memory implementation supports fixed-D/period/arc correctors only'
        cp=self.cp;s=self.s;n=self.N*s.P;extra=2 if arc is not None else 1
        phase=phase/cp.linalg.norm(phase)
        r=y[:n].reshape(self.N,s.P)*RS;T=float(cp.exp(y[-extra]));Dv=float(y[-1]*.001) if extra==2 else D
        s.set_D(Dv);Z=s.Z.copy();F,gain=self.compact_residual(r,T,Z,derivative)
        constraints=[cp.sum((r-reference)*phase)/RS]
        if arc is not None:
            yp,tan,weight=arc;constraints.append(cp.sum((y-yp)*tan*weight**2))
        res=cp.r_[F.ravel(),cp.asarray(constraints)]
        if not derivative:return res
        if getattr(self,'iteration_checkpoint',None):
            dest=Path(self.iteration_checkpoint);tmp=dest.with_name(f'{dest.stem}.{os.getpid()}.tmp.npz')
            np.savez_compressed(tmp,r=r.get(),T=T,D=Dv,Z=Z,y=y.get(),
                                residual=float(cp.max(abs(res))),status='ITERATE_ONLY')
            tmp.replace(dest)
        del F;cp.get_default_memory_pool().free_all_blocks()
        h=self.harmonics(self.parameter_inputs,r,T,Z)
        colT=(-self.contract_harmonics(h,gain)/RS).ravel();del h
        colD=None
        if extra==2:
            h=self.harmonics(self.parameter_inputs,r,T,Z,path_Z_derivative(s,Dv))
            colD=(-self.contract_harmonics(h,gain)/RS*.001).ravel();del h
            self.fixed_D_column_norm=float(cp.linalg.norm(colD))
        cp.get_default_memory_pool().free_all_blocks()
        def matvec(dy):
            dr=dy[:n].reshape(self.N,s.P)*RS
            h=self.harmonics(self.linear_inputs,dr,T,Z)
            out=((dr-self.contract_harmonics(h,gain))/RS).ravel()+colT*dy[-extra]
            if extra==2:out+=colD*dy[-1]
            cc=[cp.sum(dr*phase)/RS]
            if arc is not None:cc.append(cp.sum(dy*tan*weight**2))
            return cp.r_[out,cp.asarray(cc)]
        return res,LinearOperator((len(y),len(y)),matvec=matvec,dtype=np.float64),None


if __name__=='__main__':
    from scipy.signal import resample
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);a=p.parse_args()
    s=model();attach_rate_entry_path(s);z=np.load(OUT/'periodic/rate_seed_N1024.npz')
    N=513;M=2048;CompactExactGalerkin.harmonic_block=33;LargeQuadratureGalerkin.harmonic_block=33
    old=CompactExactGalerkin(s,N,M,a.device);old.cache_mean_operators=False;cp=old.cp
    cp.fft.config.get_plan_cache().set_size(0);r=cp.asarray(resample(z['r'],N,axis=0));n=r.size
    phase=cp.fft.irfft(2j*np.pi*cp.arange(old.K)[:,None]*cp.fft.rfft(r,axis=0),n=N,axis=0)
    y=cp.r_[(r/RS).ravel(),np.log(float(z['T'])),float(z['D'])*1000]
    c=cp.zeros(n+2);c[-2]=1000.;arc=(y.copy(),c,cp.ones(n+2))
    v=cp.asarray(np.random.default_rng(219).normal(size=len(y)));v[-2]*=.0001;v[-1]*=.001
    f,A,_=old.evaluate(y,r,phase,float(z['D']),arc=arc,derivative=True)
    expected=(A@v).get();ff=f.get();del A,old;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    new=LargeQuadratureGalerkin(s,N,M,a.device);new.cache_mean_operators=False
    g,B,_=new.evaluate(y,r,phase,float(z['D']),arc=arc,derivative=True)
    actual=(B@v).get();relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    residual=float(np.max(abs(g.get()-ff)))
    assert relative<1e-11 and residual<1e-9,(relative,residual)
    row=dict(status='PASS',N=N,M=M,relative_JVP_error=relative,max_residual_difference_hz=residual,
             residual_roundoff_limit_hz=1e-9,root_tolerance_hz=2e-8,model_change=False,
             change='Host operating inputs and exact gains; population-block Fourier contraction only')
    write(OUT/'large_quadrature_periodic_check.json',row);log('LARGE QUADRATURE PARITY',row)

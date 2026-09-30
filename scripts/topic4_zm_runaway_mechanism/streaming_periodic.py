"""Bound memory by evaluating unchanged delay operators in harmonic blocks."""
from galerkin_cycles import Galerkin
from periodic_v3 import PeriodicV3,TAU_M


class HarmonicOperator:
    def __init__(self,o,raw,phase,cache=False):
        self.o=o;self.raw=raw;self.phase=phase;self.block=o.harmonic_block
        self.cache={} if cache else None

    def __matmul__(self,v):
        o=self.o;cp=o.cp;P=o.s.P;d,index,ptr=self.raw
        nnz=len(index)//self.block;out=cp.empty_like(v)
        for lo in range(0,o.K,self.block):
            hi=min(o.K,lo+self.block);b=hi-lo
            if self.cache is not None and lo in self.cache:A=self.cache[lo]
            else:
                values=(d@self.phase[:,lo:hi]).T.copy()
                A=o.cs.csr_matrix((values.ravel(),index[:b*nnz],ptr[:b*P+1]),shape=(b*P,b*P))
                A.has_sorted_indices=True;A.has_canonical_format=True
                if self.cache is not None:self.cache[lo]=A
            out[lo*P:hi*P]=A@v[lo*P:hi*P]
        return out


class StreamKernels:
    harmonic_block=129
    cache_mean_operators=True
    def check_pattern(self):
        import numpy as np
        for row,col,_ in self.s.raw:
            assert np.all(np.diff(row)>=0)
            assert np.all(np.diff(col)[row[1:]==row[:-1]]>0)

    def kernels(self,T):
        if self.cache_key==T:return self.cache
        cp=self.cp;s=self.s;lam=2j*cp.pi*cp.arange(self.K)[:,None]/T
        phase=cp.exp(-cp.asarray(s.delays)[:,None]*lam[:,0])
        ops=[HarmonicOperator(self,raw,phase,cache=self.cache_mean_operators and i<2) for i,raw in enumerate(self.raw)]
        ha=1/((1+lam*s.rise[0])*(1+lam*s.decay[0]));hg=1/((1+lam*s.rise[1])*(1+lam*s.decay[1]))
        hva=1/(1+lam*s.tau[0]/2);hvg=1/(1+lam*s.tau[1]/2);hm=1/(1+lam*TAU_M)
        tf,ts,tE,tI,tvE,tvI=[cp.asarray(x) for x in s.poles]
        fs=1/(1+lam*ts[None,:]);fE=1/(1+lam*tE[None,:]);fI=1/(1+lam*tI[None,:])
        fvE=1/(1+lam*tvE[None,:]);fvI=1/(1+lam*tvI[None,:])
        self.cache_key=T;self.cache=(ops,(ha,hg,hva,hvg,hm),(fs,fE,fI,fvE,fvI),lam)
        return self.cache


class StreamPeriodic(StreamKernels,PeriodicV3):
    def __init__(self,s,N,device=0):
        super().__init__(s,2*self.harmonic_block-1,device);self.N=N;self.K=N//2+1;self.check_pattern()


class StreamGalerkin(StreamKernels,Galerkin):
    def __init__(self,s,N,M,device=0):
        assert N%2==1 and M>=2*N
        super().__init__(s,2*self.harmonic_block-1,M,device);self.N=N;self.K=N//2+1;self.check_pattern()


if __name__=='__main__':
    from native_path import *
    from scipy.signal import resample
    s=model();attach_native_path(s);z=np.load(OUT/'periodic/seed_N512.npz');s.set_D(float(z['D']))
    a=PeriodicV3(s,257,0);b=StreamPeriodic(s,257,0);r=a.cp.asarray(resample(z['r'],257,axis=0))
    ia=a.inputs(r,float(z['T']),s.Z);ib=b.inputs(r,float(z['T']),s.Z)
    err=float(a.cp.max(abs(ia-ib)));rel=float(a.cp.linalg.norm(ia-ib)/a.cp.linalg.norm(ia))
    fa=a.residual(r,float(z['T']),s.Z);fb=b.residual(r,float(z['T']),s.Z)
    fe=float(a.cp.max(abs(fa-fb)));assert rel<1e-12 and fe<1e-8
    write(OUT/'streaming_operator_check.json',dict(status='PASS',input_max_error=err,input_relative_error=rel,
        response_max_error_hz=fe,N=257,operator='same physical delay matrices',harmonic_block=b.harmonic_block))
    log('STREAM PARITY',err,rel,fe)

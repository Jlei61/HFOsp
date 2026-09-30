"""Separate quadrature phase locking from truncation without full GPU gains.

Same LTI harmonics and local derivatives; CPU inverse FFT and temporal chunks
allow nonlinear quadrature refinement without increasing the retained orbit.
"""
from compact_periodic import *
from scipy.fft import irfft,rfft
from scipy.signal import resample
import argparse,gc


class HostInputs(CompactExactGalerkin):
    def sample_input_harmonics(self,harmonics):
        assert self.N%2==1 and self.M>=2*self.N
        out=np.empty((len(harmonics),self.M,self.s.P))
        for k,h in enumerate(harmonics):
            hh=np.ascontiguousarray(h.get().T)
            out[k]=irfft(hh,n=self.M,axis=1,workers=2).T*(self.M/self.N)
        return out

    def inputs(self,r,T,Z):
        out=self.linear_inputs(r,T,Z)
        for k in (0,3):out[k]+=self.s.private_mu
        for k in (1,4,6):out[k]+=self.s.private_ve
        return out


def main(a):
    s=model();{'rate':attach_rate_entry_path,'fine':attach_fine_rate_entry_path,
               'native':attach_native_path}[a.family](s);z=np.load(a.orbit)
    assert float(z['residual'])<2e-8
    r=z['r'];N=len(r);T=float(z['T']);D=float(z['D']);s.set_D(D)
    if 'Z' in z:assert np.max(abs(z['Z']-s.Z))<1e-12
    phase=irfft(2j*np.pi*np.arange(N//2+1)[:,None]*rfft(r,axis=0),n=N,axis=0)
    rows=[]
    for M in a.M:
        HostInputs.harmonic_block=33;o=HostInputs(s,N,M,a.device);cp=o.cp
        o.cache_mean_operators=False;cp.fft.config.get_plan_cache().set_size(0)
        inp=o.inputs(cp.asarray(r),T,s.Z);cp.get_default_memory_pool().free_all_blocks()
        di=o.linear_inputs(cp.asarray(phase),T,s.Z);cp.get_default_memory_pool().free_all_blocks()
        pred=np.empty((M,s.P));derivative=np.empty_like(pred)
        for lo in range(0,M,1024):
            hi=min(lo+1024,M);block=cp.asarray(inp[:,lo:hi]);direction=cp.asarray(di[:,lo:hi])
            original=o.M;o.M=hi-lo
            try:ph=o.phi(block);gains=o.gains(block,ph)
            finally:o.M=original
            pred[lo:hi]=ph[0].get();derivative[lo:hi]=cp.sum(gains*direction,axis=0).get()
        retained=resample(pred,N,axis=0);dretained=resample(derivative,N,axis=0)
        pointwise=resample(r,M,axis=0)-pred
        q=dict(N=N,M=M,D=D,T_ms=T,
               retained_residual_max_hz=float(np.max(abs(r-retained))*1000),
               phase_JVP_relative=float(np.linalg.norm(phase-dretained)/np.linalg.norm(phase)),
               phase_JVP_max_hz_per_cycle_phase=float(np.max(abs(phase-dretained))*1000),
               unprojected_rms_hz=float(np.sqrt(np.mean(pointwise**2))*1000),
               unprojected_max_hz=float(np.max(abs(pointwise))*1000),
               unprojected_relative_rate_L2=float(np.linalg.norm(pointwise)/np.linalg.norm(pred)))
        rows.append(q);log('STREAMED PHASE QUADRATURE',q)
        write(Path(a.orbit).with_name(Path(a.orbit).stem+'_streamed_phase_quadrature.json'),
              dict(status='RUNNING',source=a.orbit,rows=rows,
                   scope='Same orbit evaluated on different quadrature grids; not re-corrected and not a stability result'))
        del o,inp,di,pred,derivative,retained,dretained,pointwise,block,direction,ph,gains
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(Path(a.orbit).with_name(Path(a.orbit).stem+'_streamed_phase_quadrature.json'),
          dict(status='DIAGNOSTIC_COMPLETE',source=a.orbit,rows=rows,
               scope='Same orbit evaluated on different quadrature grids; not re-corrected and not a stability result'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--M',nargs='+',type=int,default=[65536,131072,262144])
    p.add_argument('--family',choices=['rate','fine','native'],default='rate')
    p.add_argument('--device',type=int,default=1);main(p.parse_args())

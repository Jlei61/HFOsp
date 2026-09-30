"""Separate orbit truncation from nonlinear quadrature phase locking."""
from compact_periodic import *
import argparse,gc


def main(a):
    s=model();attach_rate_entry_path(s);z=np.load(a.orbit)
    assert float(z['residual'])<2e-8
    rows=[]
    for M in a.M:
        CompactExactGalerkin.harmonic_block=33
        o=CompactExactGalerkin(s,len(z['r']),M,a.device);o.cache_mean_operators=False
        cp=o.cp;cp.fft.config.get_plan_cache().set_size(0)
        r=cp.asarray(z['r']);T=float(z['T']);D=float(z['D']);s.set_D(D)
        phase=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(r,axis=0),n=o.N,axis=0)
        y=cp.r_[(r/RS).ravel(),np.log(T)]
        f,A,_=o.evaluate(y,r,phase,D,derivative=True)
        v=cp.r_[(phase/RS).ravel(),0.];av=A@v
        q=dict(N=o.N,M=M,D=D,T_ms=T,
               retained_residual_max_hz=float(cp.max(abs(f[:-1]))),
               phase_JVP_relative=float(cp.linalg.norm(av[:-1])/cp.linalg.norm(v[:-1])),
               phase_JVP_max_hz_per_cycle_phase=float(cp.max(abs(av[:-1]))))
        del A,av,v;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        # compact_residual supplies only projected output; obtain unprojected
        # rates in temporal chunks without retaining the 25-column derivative.
        inputs=o.inputs(r,T,s.Z);pred=cp.empty((M,s.P))
        for lo in range(0,M,1024):
            hi=min(lo+1024,M);block=inputs[:,lo:hi]
            original=o.M;o.M=hi-lo
            try:phi=o.phi(block);pred[lo:hi]=phi[0]
            finally:o.M=original
            del phi
        error=(o.interpolate(r,M)-pred)*1000
        q['unprojected_rms_hz']=float(cp.sqrt(cp.mean(error**2)))
        q['unprojected_max_hz']=float(cp.max(abs(error)))
        q['unprojected_relative_rate_L2']=float(cp.linalg.norm(error)/(1000*cp.linalg.norm(pred)))
        rows.append(q);log('GALERKIN PHASE DEFECT',q)
        del o,inputs,pred,error,r,phase,y,f;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    dest=Path(a.orbit).parent/(Path(a.orbit).stem+'_galerkin_phase_defect.json')
    if dest.exists():
        rows=sorted([q for q in read(dest)['rows'] if q['M'] not in a.M]+rows,key=lambda q:q['M'])
    write(dest,dict(status='DIAGNOSTIC_COMPLETE',source=a.orbit,rows=rows,
                   claim='Closure/quadrature diagnostics, not a stability classification'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--M',type=int,nargs='+',default=[32768,65536])
    p.add_argument('--device',type=int,default=0);main(p.parse_args())

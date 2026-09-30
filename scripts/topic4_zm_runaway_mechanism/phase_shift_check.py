"""A physical autonomous cycle cannot depend on the numerical time origin."""
from galerkin_cycles import *
import gc


def main(a):
    s=model();attach_native_path(s);rows=[]
    for method,path in [('collocation',a.collocation),('galerkin',a.galerkin)]:
        z=np.load(path);r=z['r'];T=float(z['T']);D=float(z['D']);N=len(r)
        if a.stream:
            from streaming_periodic import StreamGalerkin,StreamPeriodic
            o=StreamGalerkin(s,N,a.M,a.device) if method=='galerkin' else StreamPeriodic(s,N,a.device)
            o.cache_mean_operators=False
        else:o=Galerkin(s,N,a.M,a.device) if method=='galerkin' else PeriodicV3(s,N,a.device)
        cp=o.cp;rr=cp.asarray(r);rf=cp.fft.rfft(rr,axis=0);s.set_D(D)
        for frac in [0.,1/N,.5/N,.173]:
            moved=cp.fft.irfft(rf*cp.exp(2j*np.pi*cp.arange(o.K)[:,None]*frac),n=N,axis=0)
            f=o.residual(moved,T,s.Z)
            q=dict(method=method,orbit=path,N=N,phase_fraction=frac,max_residual_hz=float(cp.max(abs(f))),rms_residual_hz=float(cp.sqrt(cp.mean(f*f))))
            rows.append(q);log(q)
        del o,rr,rf,moved,f;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(OUT/f'phase_shift_invariance_{Path(a.galerkin).stem}_M{a.M}.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('collocation');p.add_argument('galerkin');p.add_argument('--device',type=int,default=1)
    p.add_argument('--M',type=int,default=2048)
    p.add_argument('--stream',action='store_true')
    main(p.parse_args())

"""Shape-preserving numerical predictor for sharp periodic waveforms.

Per-group time shifts are only a Newton initial guess. They are never accepted
as solutions or used to redefine network coupling, data or phase comparisons.
The final full-network BVP residual and phase condition remain unchanged.
"""
import numpy as np


def predict(reference,increment):
    n=len(reference);freq=np.arange(n//2+1)[:,None]
    spectrum=np.fft.rfft(reference,axis=0)
    slope=np.fft.irfft(spectrum*(2j*np.pi*freq),n=n,axis=0)
    denom=np.sum(slope*slope,axis=0)
    shifts=np.divide(np.sum(increment*slope,axis=0),denom,
        out=np.zeros(reference.shape[1]),where=denom>1e-20)
    # Keep very flat groups from acquiring a large arbitrary phase. The
    # untransported remainder preserves exactly the first-order increment.
    shifts=np.clip(shifts,-.02,.02)
    remainder=increment-slope*shifts
    moved=np.fft.irfft(spectrum*np.exp(2j*np.pi*freq*shifts),n=n,axis=0)
    return moved+remainder,shifts


def main(a):
    from native_path import model,attach_native_path,OUT,write,log
    from large_quadrature_periodic import LargeQuadratureGalerkin
    z=np.load(a.orbit);assert float(z['residual'])<2e-8
    s=model();attach_native_path(s)
    r=z['r'];T=float(z['T']);D=float(z['D']);h=np.log(a.period/T)
    v=z['tangent'];assert abs(v[-2]-1)<1e-6
    increment=.001*h*v[:r.size].reshape(r.shape)
    guessed_D=D+.001*h*v[-1]
    transported,shifts=predict(r,increment)
    LargeQuadratureGalerkin.harmonic_block=129
    o=LargeQuadratureGalerkin(s,len(r),a.M,a.device);o.cache_mean_operators=False
    cp=o.cp;cp.fft.config.get_plan_cache().set_size(8);rows=[]
    for name,rr,dd in [('constant',r,D),('linear',r+increment,guessed_D),
                       ('transport',transported,guessed_D)]:
        s.set_D(dd);f,_=o.compact_residual(cp.asarray(rr),a.period,s.Z,False)
        q=dict(predictor=name,D=dd,max_residual_hz=float(cp.max(abs(f))),
            residual_L2=float(cp.linalg.norm(f)),min_group_rate_hz=float(rr.min()*1000))
        rows.append(q);log('PREDICTOR CHECK',q)
        del f;cp.get_default_memory_pool().free_all_blocks()
    write(OUT/'periodic'/a.label/'result.json',dict(status='PREDICTOR_DIAGNOSTIC_COMPLETE',
        source=a.orbit,target_period_ms=a.period,N=len(r),M=a.M,rows=rows,
        maximum_group_shift_ms=float(abs(shifts).max()*T),
        scope='Initial guesses only. No branch point, stability, model change or bifurcation accepted.'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--period',type=float,required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--M',type=int,default=65536)
    p.add_argument('--label',required=True);main(p.parse_args())

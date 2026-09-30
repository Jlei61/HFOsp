"""Compare frequency-block actions with the retained full harmonic banks."""
from rate_antiperiodic import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    a=p.parse_args();s=RateField();rows=[];rng=np.random.default_rng(91018)
    orbit=PERIODIC_OUT/'orbits/PD_A_return_eval_J1.07580677267_N1024_accuracy_N2048.npz'
    z=np.load(orbit)
    for N in (128,256):
        r=resample(z['r'],N,axis=0)+rng.normal(0,1e-5,(N,s.P))
        reference=Periodic(s,N,a.device);reference.low_memory=True
        reference.harmonic_chunk_size=32;reference.derivative_chunk_size=32
        streamed=Periodic(s,N,a.device);streamed.stream_harmonics=True
        streamed.harmonic_chunk_size=31;cp=reference.cp
        for T,J in [(float(z['T']),float(z['J'])),(1531.73,.944731)]:
            ka=reference.kernels(T,J);kb=streamed.kernels(T,J)
            for mode in ('normal','T','J'):
                x=reference.moments(cp.asarray(r),ka,mode)
                y=streamed.moments(cp.asarray(r),kb,mode)
                rel=float(cp.linalg.norm(x-y)/cp.linalg.norm(x))
                rows.append(dict(kind='periodic_moments',N=N,T_ms=T,J=J,mode=mode,
                                 relative_error=rel,maximum_absolute_error=float(cp.max(cp.abs(x-y)))))
        reference.cache=None;streamed.cache=None
        del ka,kb,reference,streamed;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        cached=Antiperiodic(s,orbit,N,a.device,harmonic_chunk_size=32)
        block=Antiperiodic(s,orbit,N,a.device,harmonic_chunk_size=31,stream_harmonics=True)
        v=cp.asarray(rng.normal(size=N*s.P));x=cached.apply(v);y=block.apply(v)
        rows.append(dict(kind='antiperiodic_independently_differenced_gains',N=N,
                         relative_error=float(cp.linalg.norm(x-y)/cp.linalg.norm(x)),
                         gain_relative_difference=float(cp.linalg.norm(cached.gains-block.gains)/cp.linalg.norm(cached.gains)),
                         maximum_absolute_error=float(cp.max(cp.abs(x-y)))))
        # Roundoff in moments is amplified by the 1e-5 finite difference
        # used for local gains. Test the linear action separately using the
        # same gain coefficients, then retain the end-to-end discrepancy.
        block.gains=cached.gains.copy();y=block.apply(v)
        rows.append(dict(kind='antiperiodic_same_gains',N=N,
                         relative_error=float(cp.linalg.norm(x-y)/cp.linalg.norm(x)),
                         maximum_absolute_error=float(cp.max(cp.abs(x-y)))))
        del cached,block,v,x,y;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    actions=[q for q in rows if q['kind']!='antiperiodic_independently_differenced_gains']
    gains=[q for q in rows if q['kind']=='antiperiodic_independently_differenced_gains']
    passed=max(q['relative_error'] for q in actions)<1e-12 and max(q['relative_error'] for q in gains)<1e-10
    result=dict(status='PASS' if passed else 'FAIL',rows=rows,
        criteria=dict(identical_coefficient_action_relative_error=1e-12,
                      independently_finite_differenced_gain_action_relative_error=1e-10),
        scope='Same full 935-group delay operators, including log-period and J derivatives; numerical storage/action equivalence only, not orbit or bifurcation validation.')
    write(PERIODIC_OUT/'streamed_harmonic_operator_check.json',result);print(result,flush=True)
    assert result['status']=='PASS'


if __name__=='__main__':main()

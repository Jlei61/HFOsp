"""Independent cubic-delay and coupled refinement checks for the RK4 flow."""
from native_path import *
from streaming_periodic import StreamPeriodic
from cached_monodromy import CachedMonodromy
from rk4_monodromy import RK4Monodromy
import argparse,gc


def main(a):
    s=model();attach_rate_entry_path(s);z=np.load(OUT/'periodic/rate_seed_N1024.npz')
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']));o=StreamPeriodic(s,1024,a.device);o.cache_mean_operators=False
    rng=np.random.default_rng(1981);Y=rng.normal(size=(14,s.P));Y[11]=0.
    aa=rng.normal(size=s.P);bb=rng.normal(size=s.P);cc=rng.normal(size=s.P)
    def history(t):return aa+np.sin(t[:,None]/17)*bb+np.cos(t[:,None]/13)*cc
    sampled=[];rows=[];checks=[]
    for method,dt in [('rk4',.05),('rk4',.025),('heun',.0125)]:
        cls=RK4Monodromy if method=='rk4' else CachedMonodromy
        m=cls(s,o,sol,dtmax=dt,device=a.device);cp=m.cp
        if not checks:
            # Each group has a distinct amplitude. A cubic in physical time is
            # reproduced exactly at every heterogeneous delay and RK stage.
            coeff=rng.normal(size=(4,s.P));times=-np.arange(m.depth)*m.dt
            hist=sum(coeff[k]*(times[:,None]/40)**k for k in range(4))
            canonical=np.empty_like(hist);canonical[(-np.arange(m.depth))%m.depth]=hist
            m.hist[:]=cp.asarray(canonical);m.offset.fill(0)
            for stage in [0.,.5,1.]:
                m.k['delayed_cubic']((s.P,),(128,),(*m.ops,m.hist,m.arr,m.offset,np.int32(0),stage,np.int32(m.depth),m.dt))
                actual=m.arr.get();expected=[]
                delays=(np.arange(len(s.delays))*.1+.1)
                tt=stage*m.dt-delays
                values=sum(coeff[k][None,:]*(tt[:,None]/40)**k for k in range(4)).ravel()
                for kind in ['ampa','gaba']:
                    for moment in ['mean','variance']:
                        from scipy import sparse
                        expected.append(sparse.load_npz(s.folder/f'{moment}_{kind}.npz')@values)
                expected=np.asarray(expected)[[0,2,1,3]]
                err=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));assert err<1e-12,err
                checks.append(dict(stage=stage,cubic_delay_relative_error=err))
        x=np.r_[Y.ravel(),history(-np.arange(1,m.Dd+1)*m.dt).ravel()]
        m.release_full_orbit();tic=time.time();out=m.matvec(x);elapsed=time.time()-tic
        # Compare endpoint states and the same physical history lags. Use cubic
        # interpolation solely for this cross-grid diagnostic.
        from scipy.interpolate import CubicSpline
        rates=out[14*s.P:].reshape(m.Dd,s.P)
        sample=CubicSpline(np.arange(1,m.Dd+1)*m.dt,rates,axis=0)(np.arange(1,358)*.1)
        sampled.append(np.r_[out[:14*s.P],sample.ravel()]);rows.append(dict(method=method,dt_ms=m.dt,seconds=elapsed))
        log('RK4 CHECK FLOW',rows[-1]);del m;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    fine=sampled[-1];e0=float(np.linalg.norm(sampled[0]-fine)/np.linalg.norm(fine));e1=float(np.linalg.norm(sampled[1]-fine)/np.linalg.norm(fine))
    # The reference is second order; this compares the same continuous operator,
    # rather than requiring bitwise equality between different discretizations.
    ok=bool(e0<.02 and e1<.005 and e1<e0)
    q=dict(status='PASS' if ok else 'NEEDS_REFINEMENT',delay_checks=checks,flows=rows,
           coarse_vs_fine_heun_relative_error=e0,fine_vs_fine_heun_relative_error=e1,
           model_change=False,claim='Cubic delay polynomial identity and coupled-flow convergence; orbit phase gates remain mandatory')
    write(OUT/'rk4_monodromy_check.json',q);log('RK4 CHECK',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args())

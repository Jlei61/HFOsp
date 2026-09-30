"""Independent full-state parity and timing of exact derivative caching."""
from common import np,read,write,log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn,CubicSectionDerivative
from onset_cached_tangent import CachedCubicSectionDerivative
from fine_rate_frozen_Z_fields import capture,restore
from pathlib import Path
import argparse,time,os


def main(a):
    out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True);assert not(out/'jobs.json').exists()
    write(out/'contract.json',dict(source=str(Path(a.source).resolve()),seed_period_ms=a.period,dt_ms=a.dt,
        comparison='Same original full-state Poincare derivative versus float64 local-Jacobian caching along exactly the same nominal flow. Entire delayed history, all M, private variance and transient response retained. No reduced rank or altered network.',
        gates='Nominal terminal state bitwise identical; every declared full-state derivative product relative error<1e-9, phase and return-time derivative included. Timing reported, not presumed.',directions=a.directions,model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs)
    try:
        e=build(a.device,a.dt);base=dict(np.load(a.source));A=CubicSectionReturn(base,e,a.period,min(8.,a.period*.2))
        x=A.xref;y,meta=A(x);slope=A.last_time_slope.copy();T=meta['period_ms']
        J=CubicSectionDerivative(A,x,T,slope)
        begin=time.time();C=CachedCubicSectionDerivative(A,x,T,slope);construction=time.time()-begin
        restore(e,A.state(x));n=C.t.steps;whole,tail=divmod(n,round(10/e.dt))
        for _ in range(whole):e.chunk()
        for _ in range(tail):e.step()
        e.cp.cuda.get_current_stream().synchronize();terminal=capture(e)
        nominal={k:bool(np.array_equal(v,C.t.nominal_terminal[k])) for k,v in terminal.items()}
        write(out/'nominal_parity.json',nominal);assert all(nominal.values())
        rng=np.random.default_rng(92444);rows=[]
        for index in range(a.directions):
            v=y-x if index==0 else rng.standard_normal(x.size)*(x if index==1 else 1.)
            v.reshape(-1,e.s.P)[4,~e.s.E]=0
            v-=A.normal*(A.normal@v);v/=np.linalg.norm(v)
            begin=time.time();original=J(v);oldtime=time.time()-begin;oldT=J.last_return_time_derivative
            begin=time.time();cached=C(v);newtime=time.time()-begin;newT=C.last_return_time_derivative
            error=float(np.linalg.norm(cached-original)/np.linalg.norm(original))
            row=dict(direction=index,relative_full_state_error=error,
                relative_return_time_error=float(abs(newT-oldT)/max(abs(oldT),1.)),
                original_seconds=oldtime,cached_seconds=newtime,speedup=oldtime/newtime)
            rows.append(row);write(out/'progress.json',rows);log('CACHED TANGENT PARITY',row)
            assert error<1e-9 and row['relative_return_time_error']<1e-9
        result=dict(status='PASS',period_ms=T,dt_ms=e.dt,cache_bytes=C.t.cache_bytes,
            cache_construction_seconds=construction,nominal_terminal_bitwise=True,rows=rows,
            scope='Derivative evaluation implementation only. No new orbit, stability or bifurcation evidence.',model_promoted=False)
        write(out/'result.json',result);jobs.update(status='COMPLETE');write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--period',type=float,required=True)
    p.add_argument('--destination',required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--dt',type=float,default=.05);p.add_argument('--directions',type=int,choices=[1,2,3],default=3);main(p.parse_args())

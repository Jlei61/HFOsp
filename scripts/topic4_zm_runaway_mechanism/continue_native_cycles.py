"""Use audited spatial BVP continuation on the explicitly defined native Z path."""
from native_path import *
import continue_periodic_v3 as continuation
from types import SimpleNamespace
import argparse


def main(a):
    if a.linear_tol_cap:
        import cupyx.scipy.sparse.linalg as krylov
        gmres=krylov.gmres
        def bounded_gmres(*args,**kw):
            kw['tol']=min(kw.get('tol',1e-5),a.linear_tol_cap)
            return gmres(*args,**kw)
        krylov.gmres=bounded_gmres
    def load(device):
        s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s);return s
    continuation.load_model=load
    continuation.PERIODIC_OUT=OUT/'periodic'
    original=continuation.PeriodicV3
    def make_solver(s,N,device):
        if a.exact_columns:
            from exact_periodic import ExactGalerkin
            assert a.galerkin_M
            o=ExactGalerkin(s,N,a.galerkin_M,device)
        elif a.galerkin_M:
            if a.stream:
                from streaming_periodic import StreamGalerkin
                o=StreamGalerkin(s,N,a.galerkin_M,device)
            else:
                from galerkin_cycles import Galerkin
                o=Galerkin(s,N,a.galerkin_M,device)
        elif a.stream:
            from streaming_periodic import StreamPeriodic
            o=StreamPeriodic(s,N,device)
        else:o=original(s,N,device)
        if a.no_operator_cache:o.cache_mean_operators=False
        solve=o.solve
        o.solve=lambda *args,**kw:solve(*args,**dict({'restart':a.restart},**kw))
        return o
    continuation.PeriodicV3=make_solver
    args=SimpleNamespace(first=a.first,second=a.second,label=a.label,N=a.N,device=a.device,
        steps=a.steps,ds=a.ds,max_ds=a.max_ds,w_logT=100.,w_D=1.,w_r=1.,
        D_min=.14 if a.family=='rate' else .20,D_max=.15 if a.family=='rate' else .24,maxiter=14,tol=2e-8,resume=a.resume)
    continuation.main(args)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--label',default='native_up512')
    p.add_argument('--N',type=int,default=512);p.add_argument('--device',type=int,default=1);p.add_argument('--steps',type=int,default=80)
    p.add_argument('--ds',type=float,default=.4);p.add_argument('--max-ds',type=float,default=1.)
    p.add_argument('--resume',action='store_true');p.add_argument('--family',choices=['native','rate'],default='native')
    p.add_argument('--galerkin-M',type=int,default=0);p.add_argument('--stream',action='store_true')
    p.add_argument('--no-operator-cache',action='store_true')
    p.add_argument('--restart',type=int,default=120);p.add_argument('--linear-tol-cap',type=float,default=0.)
    p.add_argument('--exact-columns',action='store_true')
    main(p.parse_args())

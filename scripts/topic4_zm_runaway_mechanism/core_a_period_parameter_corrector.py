"""Period-parameterized periodic shooting in the unchanged full spatial model.

Allowing physical D_A to vary removes the fixed-D coordinate singularity at
a possible cycle fold. A solver root or a turning D alone is not a certificate.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_poincare_corrector import SectionReturn
from onset_shooting_newton import SectionDerivative,gmres_step
from onset_period_return import dynamical_state,errors
from core_a_equilibrium_branch import Family
from core_a_positive_newton_coordinates import retract
from pathlib import Path
import argparse,os,time

ROOT=OUT/'core_a_bifurcation_type_20260924/period_parameter_corrector'


def main(source,target,device,name,iterations):
    out=ROOT/name;out.mkdir(parents=True,exist_ok=True);assert not (out/'jobs.json').exists()
    raw={k:v.copy() for k,v in np.load(source).items()};np.savez_compressed(out/'source_snapshot.npz',**raw)
    write(out/'contract.json',dict(question='Does a physical periodic orbit near the recurrent burst exist when period, rather than D_A, parameterizes the branch?',
        source=str(Path(source).resolve()),target_period_ms=target,dt_ms=.05,
        equations='Unchanged full3479-group spatial conditional drift; only native Core A Z family varies and outside-Core-A Z remains native9s. All Z fixed per orbit, all M dynamic.',
        method='Bordered Newton: [I-DP,-P_D; T_X,T_D] with full actual return derivative and two-mesh finite-difference parameter derivative. Positive numerical retraction with exact Core-A-rate phase plane. No model-flow clipping or parameter refit.',
        limits=dict(iterations=iterations,Krylov_per_step=20,D_A_interval=[.26,.36],linesearch_returns=4),
        interpretation='A closed periodic root is not a fold certificate. Need continuation both sides, unit Floquet crossing, nondegeneracy and time-step verification.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),stage='setup');write(out/'jobs.json',jobs);start=time.time()
    e=build(device);family=Family(e.s);D=float(1-np.average(raw['syn'][5,family.A],weights=e.s.sizes[family.A]));family.set(D)
    assert np.max(abs(raw['syn'][5]-e.s.Z))<2e-12
    A=SectionReturn(raw,e,target,20.);weights=e.s.sizes*family.A;weights=weights/weights.sum()
    normal=np.zeros_like(A.xref);normal.reshape(-1,e.s.P)[47]=weights*A.c.scale[47]/A.c.weight[0];A.normal=normal/np.linalg.norm(normal)
    x=A.xref.copy();Wd=100.;Ts=10.;rows=[];status='ITERATION_LIMIT_NOT_AN_ORBIT'
    def field(d):
        tm=family.set(d);A.base['syn'][5]=e.s.Z.copy();return tm
    def score(xx,dd):
        field(dd);y,meta=A(xx);f=np.r_[y-xx,(target-meta['period_ms'])/Ts]
        err=errors(dynamical_state(A.state(xx)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
        return f,meta,err,A.last_time_slope.copy(),y
    try:
        for it in range(iterations):
            step=out/f'iteration{it:02d}';step.mkdir(exist_ok=True)
            f,meta,err,slope,y=score(x,D)
            row=dict(iteration=it,D_A=D,Z_A=1-D,**meta,**err,augmented_residual=float(np.linalg.norm(f)),target_period_ms=target)
            rows.append(row);write(out/'iterations.json',rows);log('CORE A PERIOD PARAMETER',row)
            np.savez_compressed(out/'latest_state.npz',**A.state(x));write(out/'latest_coordinates.json',dict(D_A=D,period_ms=meta['period_ms']))
            if err['combined_relative_rms']<1e-7 and max(v['relative_rms'] for v in err['blocks'].values())<1e-6 and abs(meta['period_ms']-target)<1e-7:
                status='NUMERICAL_PERIODIC_ROOT';break
            param=[]
            for h in ([1e-6,5e-7] if it==0 else [5e-7]):
                values=[];times=[]
                for sign in [-1,1]:
                    field(D+sign*h);v,info=A(x);values.append(v);times.append(info['period_ms'])
                pd=(values[1]-values[0])/(2*h);td=(times[1]-times[0])/(2*h)
                param.append(dict(h=h,P_D=pd,T_D=td))
            field(D);pd=param[-1]['P_D'];td=param[-1]['T_D']
            if it==0:
                check=dict(relative_P_D_error=float(np.linalg.norm(param[0]['P_D']-pd)/np.linalg.norm(pd)),
                    relative_T_D_error=float(abs(param[0]['T_D']-td)/max(abs(td),1e-12)),T_D_ms_per_D=td)
                write(out/'parameter_derivative_check.json',check);assert max(check['relative_P_D_error'],check['relative_T_D_error'])<1e-3
            J=SectionDerivative(A,x,meta['period_ms'],slope)
            if it==0:
                v=f[:-1];jv=J(v);tv=J.last_return_time_derivative;tests=[]
                for eps in [1e-4,2e-5]:
                    assert A.admissible(x+eps*v)
                    q,inf=A(x+eps*v);fd=(q-y)/eps
                    qa=dict(epsilon=eps,relative_P_derivative_error=float(np.linalg.norm(fd-jv)/np.linalg.norm(jv)),
                        relative_T_derivative_error=float(abs((inf['period_ms']-meta['period_ms'])/eps-tv)/max(abs(tv),1e-12)))
                    tests.append(qa);write(out/'state_derivative_check.json',tests)
                    if max(qa['relative_P_derivative_error'],qa['relative_T_derivative_error'])<1e-3:break
                assert max(tests[-1]['relative_P_derivative_error'],tests[-1]['relative_T_derivative_error'])<1e-3
            def operator(v):
                jv=J(v[:-1]);tv=J.last_return_time_derivative;dd=v[-1]/Wd
                return np.r_[v[:-1]-jv-pd*dd,(tv+td*dd)/Ts]
            jobs.update(stage='bordered_GMRES',iteration=it);write(out/'jobs.json',jobs)
            delta,linear=gmres_step(lambda v:v-operator(v),f,20,step)
            np.savez_compressed(step/'direction.npz',delta=delta,source=x,D_A=D,normal=A.normal)
            accepted=False;trials=[];attempts=0
            for alpha in 2.**-np.arange(14):
                dd=D+alpha*delta[-1]/Wd
                if not .26<dd<.36:continue
                xx=retract(A,x,delta[:-1],alpha)
                if xx is None:continue
                if attempts>=4:break
                attempts+=1
                try:
                    ff,inf,ee,ss,vv=score(xx,dd);ratio=float(np.linalg.norm(ff)/np.linalg.norm(f))
                    trial=dict(alpha=float(alpha),D_A=float(dd),period_ms=inf['period_ms'],ratio=ratio,combined_relative_rms=ee['combined_relative_rms'])
                except RuntimeError as exc:ratio=np.inf;trial=dict(alpha=float(alpha),D_A=float(dd),error=str(exc))
                trials.append(trial);write(step/'line_search.json',trials);log('CORE A PERIOD PARAMETER LINE',trial)
                if ratio<1:
                    x=xx;D=float(dd);A.period=inf['period_ms'];accepted=True
                    np.savez_compressed(step/'accepted_state.npz',**A.state(x));write(step/'accepted_coordinates.json',dict(D_A=D,period_ms=A.period));break
            row['linear_last']=linear[-1];row['line_search']=trials;write(out/'iterations.json',rows)
            if not accepted:status='STEP_NOT_ACCEPTED_NOT_A_BIFURCATION';break
        write(out/'result.json',dict(status=status,iterations=rows,target_period_ms=target,elapsed_seconds=time.time()-start,model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status);write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--period',type=float,required=True);p.add_argument('--device',type=int,default=1);p.add_argument('--name',required=True);p.add_argument('--iterations',type=int,default=6)
    a=p.parse_args();main(a.source,a.period,a.device,a.name,a.iterations)

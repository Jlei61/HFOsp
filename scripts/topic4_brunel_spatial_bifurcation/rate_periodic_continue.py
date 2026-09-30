"""Pseudo-arclength continuation through folds of full spatial periodic BVPs."""
from rate_periodic import *


def encode(z,N):return np.r_[(resample(z['r'],N,axis=0)*1000).ravel(),np.log(float(z['T'])),float(z['J'])*1000]


def continuation_state(a):
    """Recover the last accepted secants without changing the total budget.

    The positional seeds still identify the original segment.  A resumed
    worker appends to its saved rows; it never rewrites an accepted orbit.
    Process handoff is the caller's responsibility.
    """
    first,second=[np.load(f) for f in [a.first,a.second]]
    x0,x1=[encode(z,a.N) for z in [first,second]]
    js=[float(first['J']),float(second['J'])]
    rows=[];previous=None;ds=a.ds
    destination=PERIODIC_OUT/(a.label+'_continuation.json')
    if a.resume_existing:
        record=read(destination)
        assert record['method']=='pseudo-arclength' and record['N']==a.N
        assert record['status'] in ['CONTINUED','PAUSED_FOR_CRITICAL_POINT_REFINEMENT']
        rows=list(record['rows']);assert len(rows)>=3
        assert len(rows)<=a.steps
        if a.require_filter_positivity:
            assert record['constituent_filter_check']=='EACH_PROFILE_PASSED'
        for i,row in enumerate(rows,a.start_index):
            path=Path(row['path'])
            assert path.stem==f'{a.label}_{i:04d}_N{a.N}' and path.exists()
            meta=read(path.with_suffix('.json'))
            assert meta['status']=='CONVERGED' and meta['residual_hz']<a.tol
            js.append(meta['J_EE_core'])
        previous,x0,x1=[encode(np.load(row['path']),a.N) for row in rows[-3:]]
        history=np.load(rows[-1]['path'])['history']
        ds=rows[-1]['ds']
        if len(history)<5:ds=min(ds*1.2,a.ds*2)
        if len(history)>7:ds*=.7
    else:
        assert not destination.exists(),'Use an explicit checkpoint resume for an existing segment'
    return x0,x1,previous,ds,rows,js


def screened_curvature_guess(o,previous,x0,x1,ds,weight):
    """Improve the initial guess within the original arc and phase planes.

    Both candidates are measured by the identical nonlinear BVP residual.
    The corrected solution must still pass the ordinary Newton tolerance.
    """
    h1=np.linalg.norm((x1-x0)*weight);tangent=(x1-x0)/h1
    linear=x1+ds*tangent
    if previous is None:return linear,linear,tangent,dict(method='LINEAR_FIRST_STEP')
    h0=np.linalg.norm((x0-previous)*weight)
    bend=ds*(ds+h1)/(h0+h1)*(tangent-(x0-previous)/h0)
    bend-=tangent*np.sum(bend*tangent*weight**2)
    candidate=linear+bend
    cp=o.cp;reference=cp.asarray(linear[:-2].reshape(o.N,o.s.P)/1000)
    dr=cp.fft.irfft(cp.fft.rfft(reference,axis=0)*(2j*np.pi*cp.arange(o.K))[:,None],n=o.N,axis=0)
    phase=dr/cp.sum(dr*dr)*.001
    arc=tuple(cp.asarray(v) for v in [linear,tangent,weight]);errors=[]
    for y in [linear,candidate]:
        f=o.evaluate(cp.asarray(y),reference,phase,float(linear[-1]/1000),arc=arc)
        errors.append(dict(maximum=float(cp.max(cp.abs(f))),L2=float(cp.linalg.norm(f))))
    accepted=all(np.isfinite(list(errors[1].values()))) and all(errors[1][k]<errors[0][k] for k in ['maximum','L2'])
    return candidate if accepted else linear,linear,tangent,dict(
        method='CURVATURE_INITIAL_GUESS' if accepted else 'LINEAR_RESIDUAL_FALLBACK',
        residuals=errors,arc_plane_displacement=float(np.sum((candidate-linear)*tangent*weight**2)),
        weighted_correction_norm=float(np.linalg.norm(bend*weight)),
        scope='Initial guess only; the same arc plane, phase reference, full equations and tolerances are used.')


def main():
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--N',type=int,default=64)
    p.add_argument('--steps',type=int,default=60);p.add_argument('--ds',type=float,default=.2);p.add_argument('--label',required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--low-memory',action='store_true');p.add_argument('--krylov-restart',type=int,default=160)
    p.add_argument('--linear-normalize',action='store_true')
    p.add_argument('--host-krylov',action='store_true')
    p.add_argument('--stream-harmonics',action='store_true',help='Exact frequency-block actions; retain every spatial population and delay')
    p.add_argument('--quadratic-predictor',action='store_true',help='Use a full-residual-screened curvature initial guess in the unchanged continuation planes')
    p.add_argument('--resume-existing',action='store_true',help='Append from the last accepted checkpoint, retaining the original total step and turn limits')
    p.add_argument('--require-filter-positivity',action='store_true',
                   help='Withhold and stop at a profile whose constituent rate filters need finer resolution')
    p.add_argument('--linear-rtol-cap',type=float,default=.02,help='Maximum inexact-Newton linear relative tolerance')
    p.add_argument('--tol',type=float,default=2e-8,help='Periodic BVP residual tolerance in Hz')
    p.add_argument('--stop-after-turns',type=int,default=0,
                   help='Pause after this many sampled J reversals for separate root refinement; zero disables')
    a=p.parse_args();assert 0<a.linear_rtol_cap<=.02
    x0,x1,previous,ds,rows,js=continuation_state(a)
    s=RateField();o=Periodic(s,a.N,a.device)
    o.low_memory=a.low_memory;o.krylov_restart=a.krylov_restart
    o.harmonic_chunk_size=64;o.derivative_chunk_size=64;o.host_krylov=a.host_krylov
    o.stream_harmonics=a.stream_harmonics
    o.normalize_linear_rhs=a.linear_normalize
    o.linear_rtol_cap=a.linear_rtol_cap
    o.linear_target_aware=a.require_filter_positivity
    weight=np.r_[np.full(a.N*s.P,1/np.sqrt(a.N*s.P)),50.,1.]
    if a.quadratic_predictor:
        assert read(PERIODIC_OUT/'curvature_predictor_same_orbit_check.json')['status']=='PASS'
    changes=np.diff(js)
    if a.stop_after_turns and int(np.sum(changes[:-1]*changes[1:]<0))>=a.stop_after_turns:
        print('EXISTING TURN LIMIT REACHED',flush=True);return
    for i in range(a.start_index+len(rows),a.start_index+a.steps):
        tangent=x1-x0;tangent/=np.linalg.norm(tangent*weight)
        for retry in range(7):
            pred=x1+ds*tangent;guess=pred;info=dict(method='LINEAR')
            if a.quadratic_predictor:
                guess,pred,tangent,info=screened_curvature_guess(o,previous,x0,x1,ds,weight)
                print('PREDICTOR',info,flush=True)
            r=guess[:-2].reshape(a.N,s.P)/1000;T=np.exp(guess[-2]);J=guess[-1]/1000
            rr,TT,JJ,err,history=o.solve(r,T,J,arc=(pred,tangent,weight),maxiter=12,tol=a.tol,
                phase_reference=pred[:-2].reshape(a.N,s.P)/1000 if a.quadratic_predictor else None)
            if err<a.tol:break
            ds*=.5
        else:
            print('CONTINUATION FAILED',i,ds,err,flush=True);break
        if a.require_filter_positivity:
            from audit_rate_filter_states import filter_state_minima
            physical=filter_state_minima(s,rr,TT)
            if not physical['positive']:
                path=save_orbit(s,rr,TT,JJ,err,history,f'diagnostic_{a.label}_{i:04d}_N{a.N}')
                write(PERIODIC_OUT/f'{a.label}_continuation.json',dict(status='PHYSICAL_REFINEMENT_REQUIRED',
                    rows=rows,method='pseudo-arclength',N=a.N,failed_index=i,withheld_orbit=str(path),filter_state_check=physical))
                print('PHYSICAL REFINEMENT REQUIRED',i,physical,flush=True);return
        path=save_orbit(s,rr,TT,JJ,err,history,f'{a.label}_{i:04d}_N{a.N}');rows.append(dict(path=str(path),ds=ds,predictor=info))
        previous=x0;x0=x1;x1=np.r_[(rr*1000).ravel(),np.log(TT),JJ*1000]
        if len(history)<5:ds=min(ds*1.2,a.ds*2)
        if len(history)>7:ds*=.7
        write(PERIODIC_OUT/f'{a.label}_continuation.json',dict(status='CONTINUED',rows=rows,method='pseudo-arclength',N=a.N,
              constituent_filter_check='EACH_PROFILE_PASSED' if a.require_filter_positivity else 'NOT_CHECKED'))
        js.append(JJ)
        changes=np.diff(js);turns=int(np.sum(changes[:-1]*changes[1:]<0))
        if a.stop_after_turns and turns>=a.stop_after_turns:
            write(PERIODIC_OUT/f'{a.label}_continuation.json',dict(status='PAUSED_FOR_CRITICAL_POINT_REFINEMENT',
                rows=rows,method='pseudo-arclength',N=a.N,candidate_parameter_turns=turns,
                constituent_filter_check='EACH_PROFILE_PASSED' if a.require_filter_positivity else 'NOT_CHECKED'))
            break
        if JJ<.5 or JJ>2.1 or TT>5000 or TT<5:break


if __name__=='__main__':main()

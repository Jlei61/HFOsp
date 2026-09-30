"""Trace the conditional high-rate equilibrium from the physical D=1 endpoint.

The full-Z/M high root supplies only a seed. Conditional stability is separate.
Use the already defined native path, including its explicitly synthetic endpoint
extensions. Interior path joins are nonsmooth and cannot certify bifurcations.
"""
from native_path import *
import equilibria_v3 as arc
import argparse


def parameter_column(s, r, h=None):
    _, b, _, qb = s.matrices()
    g = s.phi(*s.moments(r))
    zd = path_Z_derivative(s, s.D)
    ig = s.tm * s.area[1] * (b @ r)
    vg = s.tm * s.area[1]**2 * (qb @ r)
    return -g['d_mu'] * ig * zd + g['d_vi'] * 2 * s.Z * vg * zd


def check_column(s, r):
    rows = []
    for D in [.999, .8, .5, .32, .29, .25, .23, .21]:
        s.set_D(D)
        expected = parameter_column(s, r)
        errors = []
        for h in [1e-6, 3e-7]:
            s.set_D(D+h); fp = s.residual(r)
            s.set_D(D-h); fm = s.residual(r)
            s.set_D(D)
            errors.append(float(np.linalg.norm((fp-fm)/(2*h)-expected) /
                                max(np.linalg.norm(expected), 1e-14)))
        rows.append(dict(D=D, relative_errors=errors))
    assert max(q['relative_errors'][-1] for q in rows) < 1e-5, rows
    return rows


def main(a):
    s=model(); path=attach_native_path(s); arc.RS=.1; arc.DS=.1
    out=OUT/'equilibria'/a.label; out.mkdir(parents=True, exist_ok=True)
    contract=dict(question=a.question,
        Z='held spatial native path', M='dynamic; equilibrium m=.5 E r',
        path=path, seed=a.resume or str(OUT/'full_ZM_equilibria/high.npz'),
        endpoint_extension='D above the last observed knot is linear from native10370 to E Z=0; not a later observed Z trajectory',
        max_points=a.steps, max_turns=a.max_turns, resume=a.resume,
        cross_known_path_joins=a.cross_known_path_joins,
        target_D=a.target, target_direction=a.target_direction, root_tolerance_hz=2e-8,
        acceptance='Physical converged equilibria only; no stability inheritance; tangent reversals are candidates, and path joins are not smooth bifurcations')
    write(out/'contract.json', contract)
    if a.resume:
        source=np.load(a.resume);r=source['r'].copy();D=float(source['D'])
        s.set_D(D)
        assert np.max(abs(s.Z-source['Z']))<1e-13
        assert abs(s.residual(r)).max()*1000<2e-8
        checks=check_column(s,r)
        arc.param_derivative=parameter_column
        t=arc.tangent(s,r,D,source['tangent'])
        assert t@source['tangent']>.99999
    else:
        source=np.load(contract['seed']); r=source['r'].copy()
        s.set_D(1.); r,ok,tr=s.solve(r); assert ok, tr[-1]
        np.savez_compressed(out/'endpoint.npz',r=r,D=1.,Z=s.Z)
        checks=check_column(s,r)
        arc.param_derivative=parameter_column
        D=1.-1e-5; s.set_D(D); r,ok,tr=s.solve(r); assert ok,tr[-1]
        t=arc.tangent(s,r,D,direction=-1)
    write(out/'parameter_column_check.json',dict(status='PASS',rows=checks))
    ds=.05; rows=[]; brackets=[]; joins=[]; status='RUNNING'
    for k in range(a.steps):
        s.set_D(D); p=out/f'point{k:04d}.npz'
        np.savez_compressed(p,r=r,D=D,Z=s.Z,tangent=t)
        row=dict(index=k,path=str(p),D=D,global_E_hz=s.global_rate(r),
            regional_hz=s.regional_rates(r),residual_hz=float(abs(s.residual(r)).max()*1000),
            tangent_D=float(t[-1]),ds=ds,stability='NOT_ESTABLISHED',
            observed_path=bool(path['observed_D_interval'][0]<=D<=path['observed_D_interval'][1]))
        assert row['residual_hz']<2e-8 and r.min()>=0 and np.all(r<1/s.ref)
        if rows and t[-1]*rows[-1]['tangent_D']<0:
            crossed_knots=[float(v) for v in s.path_D_knots[1:-1]
                if min(D,rows[-1]['D'])<=v<=max(D,rows[-1]['D'])]
            brackets.append(dict(indices=[k-1,k],path_knots=crossed_knots,
                status='NONSMOOTH_JOIN' if crossed_knots else 'UNREFINED_TURN'))
        rows.append(row)
        write(out/'result.json',dict(status=status,rows=rows,turn_brackets=brackets,path_joins=joins))
        log(a.label,k,'D',D,'mean',row['global_E_hz'],'ds',ds)
        if (D<=a.target if a.target_direction=='below' else D>=a.target):status='TARGET_D_REACHED';break
        knot=float(s.path_D_knots[1:-1][np.argmin(abs(s.path_D_knots[1:-1]-D))])
        at_join_toward_other_segment=abs(D-knot)<1e-10 and (D-knot)*t[-1]<=0
        if ds<1e-10 or (a.cross_known_path_joins and at_join_toward_other_segment):
            if a.cross_known_path_joins and abs(knot-D)<1e-8:
                from types import SimpleNamespace
                from equilibrium_path_join import main as cross_join
                assert read(OUT/'equilibria/native_high_join9870/result.json')['status']=='JOIN_CONTINUITY_AND_ONE_SIDED_CHECK_PASS'
                assert read(OUT/'equilibria/native_high_join9420/result.json')['status']=='JOIN_CONTINUITY_AND_ONE_SIDED_CHECK_PASS'
                label=f'{a.label}_join{len(joins):02d}'
                direction=1 if t[-1]>0 else -1
                try:
                    cross_join(SimpleNamespace(source=str(p),label=label,direction=direction))
                    audit=read(OUT/'equilibria'/label/'result.json')
                    assert audit['status']=='JOIN_CONTINUITY_AND_ONE_SIDED_CHECK_PASS'
                except Exception as exc:
                    status='PATH_JOIN_AUDIT_FAILED';log(status,repr(exc));break
                follow=np.load(audit['continuation_seed'])
                r=follow['r'];D=float(follow['D']);t=follow['tangent'];ds=.05
                joins.append(dict(after_index=k,audit=str(OUT/'equilibria'/label/'result.json'),
                    direction=direction,continuation_seed=audit['continuation_seed']))
                continue
            status='NUMERICAL_STEP_STAGNATION';break
        if len(brackets)>=a.max_turns:status='TURN_BUDGET_REACHED';break
        x=np.r_[r/arc.RS,D/arc.DS]
        for attempt in range(22):
            xx,ok,it=arc.correct(s,x+ds*t,t)
            if ok:
                nt=arc.tangent(s,xx[:-1]*arc.RS,float(xx[-1]*arc.DS),t)
                cosine=float(nt@t); corr=float(np.linalg.norm(xx-x-ds*t)/ds)
                if cosine>.90 and corr<.25:break
                ok=False
            ds*=.5
        if not ok:
            if a.cross_known_path_joins and min(abs(s.path_D_knots[1:-1]-D))<1e-8:
                ds=1e-11
                continue
            status='NONCONVERGENCE';break
        r=xx[:-1]*arc.RS;D=float(xx[-1]*arc.DS);t=nt
        if it<=4 and cosine>.98 and corr<.1:ds=min(.25,ds*1.2)
        elif it>=10:ds*=.7
    else:status='POINT_LIMIT_REACHED'
    write(out/'result.json',dict(status=status,rows=rows,turn_brackets=brackets,path_joins=joins,
        scope='Conditional equilibria with held spatial Z and dynamic M. No stability or onset type established by this continuation.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='native_high_descent')
    p.add_argument('--steps',type=int,default=800);p.add_argument('--target',type=float,default=.21)
    p.add_argument('--target-direction',choices=['below','above'],default='below')
    p.add_argument('--question',default='Does the conditional high-activity equilibrium extend into the observed native Z range, and where does it lose existence or stability?')
    p.add_argument('--resume');p.add_argument('--max-turns',type=int,default=3)
    p.add_argument('--cross-known-path-joins',action='store_true',help='At a known interpolation corner, run the independently verified one-sided equilibrium audit before continuing; no smoothing or stability inference')
    main(p.parse_args())

"""Transverse return constructed from iterates of the unchanged native map.

An invariant curve need not close in an integer number of 0.1-ms steps. We
interpolate nearby native iterates onto a fixed transverse hyperplane, then
measure the error as interpolation order increases. This is an approximate
generalized return, not an exact periodic point of the original discrete map.
"""
from cycle_monodromy import *
from scipy.optimize import brentq


def coefficients(alpha,order):
    return np.array([np.prod([(alpha-j)/(i-j) for j in range(order+1) if j!=i]) for i in range(order+1)])


def crossing(values,order):
    values=np.asarray(values)[:order+1]
    # The integer step is only a nearby anchor. Retain the next full step so
    # a return at half a native step is not cut off by an artificial boundary.
    return brentq(lambda alpha:coefficients(alpha,order)@values,-.5,1.5,xtol=2e-14)


def analytic_circle_check():
    # Known invariant circle of a smooth dissipative map. Its irrational
    # rotation has no finite exact return; interpolation error is measurable
    # against the known intersection (1,0), independent of this application.
    rows=[]
    for h in (.2,.1,.05):
        step=h*np.sqrt(2.);n=round(2*np.pi/step)
        values=np.array([[np.cos((n+k)*step),np.sin((n+k)*step)] for k in range(6)])
        for order in (1,3,5):
            alpha=crossing(values[:,1],order);point=coefficients(alpha,order)@values[:order+1]
            rows.append(dict(step_angle=step,order=order,intersection_error=float(np.linalg.norm(point-[1.,0.])),
                             fitted_fraction=alpha,true_fraction=2*np.pi/step-n))
    for h in (.2,.1,.05):
        errors=[row['intersection_error'] for row in rows if abs(row['step_angle']-h*np.sqrt(2.))<1e-14]
        assert errors[2]<errors[1]<errors[0]
    return rows


def run(a):
    cfg=read(a.source/'config.json');ecfg=read(a.endpoint/'config.json')
    assert ecfg['resumed_from']==str(a.source.resolve())
    folder=OUT/'generalized_returns'/a.label;folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(a.source);coords=StateCoordinates(m);initial_step=m.step_index;x=coords.pack(m)
    m.advance_step();y=coords.pack(m);m.advance_step();z=coords.pack(m)
    normal=(-3*x+4*y-z)/(2*DT);normal/=cp.linalg.norm(normal);del y,z
    m.restore(a.endpoint);n=m.step_index-initial_step
    assert abs(n*DT-a.period)<1e-9
    delta=[];values=[]
    for k in range(6):
        if k:m.advance_step()
        v=coords.pack(m)-x;delta.append(v);values.append(float(cp.dot(normal,v).get()))
    rows=[];returns={}
    for order in (1,3,5):
        alpha=crossing(values,order);c=coefficients(alpha,order)
        residual=sum(float(ck)*dk for ck,dk in zip(c,delta));returns[order]=residual
        rows.append(dict(interpolation_order=order,phase_fraction=alpha,return_time_ms=(n+alpha)*DT,
            weighted_transverse_return_norm=float(cp.linalg.norm(residual).get()),
            section_residual=float(cp.dot(normal,residual).get())))
    differences={f'{lo}_to_{hi}':float(cp.linalg.norm(returns[lo]-returns[hi]).get()) for lo,hi in [(1,3),(3,5)]}
    result=dict(status='GENERALIZED_RETURN_INTERPOLATION_AUDIT_COMPLETE',source=str(a.source),endpoint=str(a.endpoint),
        D=cfg['D'],rows=rows,interpolation_order_differences=differences,
        analytic_circle_validation=analytic_circle_check(),native_map_diagnostics=m.diagnostics(),
        definition='Polynomial through native full-state iterates at n,...,n+p; intersect fixed hyperplane normal to initial native orbit chord',
        interpretation='Approximate invariant-curve return. A fixed point, its normal spectrum, and interpolation convergence still need correction/validation. This is not an exact n-step periodic point.',
        origin='Own implementation of a generalized return; motivated by Sanchez, Net and Simo (2010), DOI 10.1016/j.physd.2009.10.012. No model-specific claim follows from that reference.')
    write(folder/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--endpoint',type=Path,required=True)
    ap.add_argument('--period',type=float,required=True);ap.add_argument('--label',required=True);ap.add_argument('--device',type=int,default=0)
    run(ap.parse_args())

"""Separate a discrete sampling phase error from orbit-shape return error.

Native steps are unchanged. Interpolation is used only to diagnose a return
between two time samples; it is not passed off as an exact periodic solution
of the discrete map. Both linear and four-point cubic reconstructions are
tested in the complete density/current/M/ordered-delay state.
"""
from cycle_monodromy import StateCoordinates
from search_recurrence import *
from scipy.optimize import minimize_scalar


def run(args):
    source=Path(args.source);cfg=read(source/'config.json')
    folder=OUT/'fractional_returns'/args.label;folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);initial=capture(m);coords=StateCoordinates(m);x0=coords.pack(m)
    n=round(args.period/DT);initial_index=m.step_index
    write(folder/'config.json',dict(source=str(source.resolve()),D=cfg['D'],native_dt_ms=DT,
        center_return_steps=n,center_return_ms=n*DT,endpoint=str(args.endpoint) if args.endpoint else None,
        interpolation='Diagnostic only; native map unchanged'))
    frames=[];states=[];started=time.time();last=started
    if args.endpoint:
        m.restore(args.endpoint);assert m.step_index-initial_index==n
        frames.append(coords.pack(m));states.append(capture(m));first=n+1
    else:first=1
    for step in range(first,n+4):
        m.advance_step()
        if step>=n:frames.append(coords.pack(m));states.append(capture(m))
        if time.time()-last>20:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),elapsed_ms=step*DT,wall_s=time.time()-started))
            print('fractional return',args.label,step*DT,flush=True);last=time.time()
    # Forward Lagrange reconstruction uses an already saved native endpoint
    # when available; it does not require replaying thousands of native steps.
    def coeff(a,order):
        if order==1:return np.array([1-a,a,0.,0.])
        return np.array([np.prod([(a-j)/(i-j) for j in range(4) if j!=i]) for i in range(4)])
    # Four-by-four Gram matrix makes phase fitting inexpensive without losing
    # a single state component. All entries use the same fixed physical metric.
    delta=[f-x0 for f in frames]
    gram=np.array([[float(cp.dot(a,b).get()) for b in delta] for a in delta])
    weights=cp.asarray(m.geo['group_size']/40000.);rows=[]
    for order in (1,3):
        fit=minimize_scalar(lambda a:float(coeff(a,order)@gram@coeff(a,order)),bounds=(-.5,.5),method='bounded',options={'xatol':1e-12})
        c=coeff(fit.x,order);state={}
        for k in (*STATE_NAMES,'ordered_history'):
            if k=='history':continue
            state[k]=sum(float(ci)*s[k] for ci,s in zip(c,states))
        row=dict(order=order,phase_fraction=float(fit.x),period_ms=(n+fit.x)*DT,
                 weighted_return_norm=float(np.sqrt(max(fit.fun,0.))),**separation(initial,state,weights))
        rows.append(row)
    write(folder/'result.json',dict(status='COMPLETE',rows=rows,gram=gram,diagnostics=m.diagnostics(),wall_s=time.time()-started,
        interpretation='Interpolated whole-state return diagnostic; small error supplies an invariant-cycle initial guess, not proof of an exact periodic orbit or its stability.'))
    print('fractional return results',rows,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--period',type=float,required=True);ap.add_argument('--endpoint',type=Path)
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())

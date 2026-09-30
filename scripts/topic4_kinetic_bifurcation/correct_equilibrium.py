"""Correct an equilibrium predictor in the actual local/network density map."""
from equilibrium_predictor import *
from stationary_response import LocalStationaryDensity


def warm_start(model,problem,u):
    mode=read(problem.table_path/'config.json').get('basis_mode','legacy')
    assert mode==model.basis_mode, 'Stationary predictor PDF must use the same noise coordinates'
    path=problem.table_path/'density_state.npz'
    with np.load(path) as z:
        F=z['F']
    if F.shape[1:] != model.F.shape[1:]:return False
    C=len(problem.current_grid)
    at=np.clip(np.searchsorted(problem.current_grid,u)-1,0,C-2)
    w=np.clip((u-problem.current_grid[at])/(problem.current_grid[at+1]-problem.current_grid[at]),0.,1.)
    x=np.zeros(model.F.shape)
    for theta_index,tw in [(problem.lo,1-problem.w),(problem.hi,problem.w)]:
        for offset,uw in [(0,1-w),(1,w)]:
            x+=F[theta_index*C+at+offset]*(tw*uw)[:,None,None]
    model.F[:]=cp.asarray(x)
    return True


def run(args):
    suffix=('_'+args.label) if args.label else ('_refined' if args.resume else '')
    folder=OUT/'corrected_equilibria'/f'D{args.D:.8f}_degree{args.degree}_dv{args.dv:g}{suffix}'
    folder.mkdir(parents=True,exist_ok=False)
    problem=EquilibriumProblem(args.table,args.theta_interpolation)
    basis_mode=read(args.table/'config.json').get('basis_mode','legacy')
    resumed=None
    if args.resume:
        assert read(args.resume/'config.json').get('basis_mode','legacy')==basis_mode
        with np.load(args.resume/'stationary_local_state.npz') as f:
            resumed={k:f[k] for k in f.files}
        seed=resumed['rate_hz']
    elif args.predictor:
        with np.load(args.predictor) as f:
            at=np.argmin(abs(f['D']-args.D));seed=f['rate_hz'][at]
    else:
        seed=np.where(problem.e,.1,.3) if args.D<.2 else np.where(problem.e,450.,700.)
    if resumed is None:
        r,qa=problem.fixed_D(args.D,seed)
        assert qa['converged'],qa
    else:r=seed.copy()
    u=problem.equations(r,args.D)[-1]
    local=LocalStationaryDensity(problem.theta,problem.geo['population'],u,args.degree,args.dv,args.device,basis_mode=basis_mode)
    if resumed is not None:
        assert resumed['F'].shape==local.F.shape
        local.F[:]=cp.asarray(resumed['F']);initialized=True
    else:initialized=warm_start(local,problem,u)
    cfg=dict(D=args.D,degree=args.degree,voltage_dv=args.dv,dt_ms=DT,
        communication_operators=str(OPERATORS),table=str(args.table),warm_start=initialized,
        predictor_theta_interpolation=args.theta_interpolation,basis_mode=basis_mode,
        stable_marginal_projection=args.stable_marginal_projection,
        actual_response_secants=args.response_secant,
        signed_algebraic_rate_iterates=args.signed_rate_iterates,
        maximum_algebraic_rate_step_hz=args.maximum_rate_step,
        numerical_seed_source=str(args.resume.resolve()) if args.resume else None,
        maximum_outer_iterations=args.outer_iterations,
        object='Full conditional-density equilibrium; common OU at mean, Z frozen, M dynamic in stability')
    write(folder/'config.json',cfg)
    history=[];started=time.time();converged=False
    previous_u=None;previous_actual=None;last_slope=None
    if args.response_secant:
        z,_=problem.resource(args.D)
        actual_coupling=(problem.A-sparse.diags(z)@problem.G-problem.Mcoupling).tocsr()
        if resumed is not None:
            # The saved local PDF/response belongs to this recorded current;
            # its rate predictor may already be the next, unevaluated guess.
            # Retain the actual previous response pair for the first secant.
            previous_u=resumed['current_mv'].copy()
            previous_actual=resumed['actual_rate_hz'].copy()
            last_slope=problem.response(previous_u)[1]
    for outer in range(args.outer_iterations):
        _,J,_,u=problem.equations(r,args.D)
        local.drive[:]=cp.asarray(u)
        def progress(it,residual,rate):
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),outer=outer,
                inner=it,maximum_stationary_residual=float(residual.max()),wall_s=time.time()-started))
            print('correct D',args.D,'outer',outer,'inner',it,'pdf residual',residual.max(),flush=True)
        tolerance=1e-12 if resumed is not None or (history and history[-1]['rate_residual_hz']<1e-5) else 1e-10
        actual,inner_res,diag,_=local.solve(10000,progress,tolerance=tolerance,acceleration=args.acceleration,
            stable_marginal_projection=args.stable_marginal_projection)
        residual=r-actual
        item=dict(outer=outer,rate_residual_hz=float(np.max(abs(residual))),
            mean_E_hz=float(problem.eweights@actual),local=diag)
        history.append(item);print(item,flush=True)
        if diag['converged'] and np.max(abs(residual))<1e-7:
            converged=True;break
        if not diag['converged']:
            write(folder/'status.json',dict(status='LOCAL_DENSITY_SOLVE_INCOMPLETE',history=history))
            break
        if args.response_secant:
            # Only the nonlinear root solver changes. The response remains the
            # actual stationary density at every group, and acceptance below
            # still uses the unchanged complete native-map residual.
            slope=problem.response(u)[1] if last_slope is None else last_slope.copy()
            valid=np.zeros(problem.P,dtype=bool)
            if previous_u is not None:
                du=u-previous_u;resolved=abs(du)>1e-6
                sec=(actual-previous_actual)/np.where(resolved,du,1.)
                valid=resolved&np.isfinite(sec)&(sec>=0.)
                slope[valid]=sec[valid]
            J=problem.I-sparse.diags(slope)@actual_coupling
            last_slope=slope;previous_u=u.copy();previous_actual=actual.copy()
            item['actual_response_secant_groups']=int(valid.sum())
        delta=spsolve(J,-residual)
        if outer and item['rate_residual_hz']>history[-2]['rate_residual_hz']*1.1:delta*=.5
        scale=1.
        if args.maximum_rate_step is not None:
            assert args.maximum_rate_step>0
            scale=min(scale,args.maximum_rate_step/max(float(np.max(abs(delta))),1e-300))
        # These are algebraic rate guesses used to evaluate the stationary
        # residual, not physical trajectories. A nonnegativity restriction can
        # pin every network update at a nearly silent group. Signed guesses
        # are already used by EquilibriumProblem.fixed_D; an accepted root
        # must still equal the actual nonnegative density response and pass the
        # complete original-map check below.
        if not args.signed_rate_iterates:
            while (r+scale*delta).min() < 0:scale*=.5
        item['algebraic_update_scale']=scale
        item['algebraic_update_max_hz']=float(np.max(abs(scale*delta)))
        item['minimum_rate_predictor_hz']=float(r.min())
        write(folder/'outer_history.json',dict(status='RUNNING',history=history))
        r+=scale*delta
    write(folder/'correction.json',dict(converged=converged,history=history,wall_s=time.time()-started))
    np.savez_compressed(folder/'stationary_local_state.npz',F=cp.asnumpy(local.F),rate_hz=r,
        actual_rate_hz=actual,current_mv=u,theta=problem.theta,population=problem.geo['population'])
    if not converged:
        write(folder/'status.json',dict(status='NETWORK_CORRECTION_INCOMPLETE',history=history));return
    density=AutonomousDensity(args.D,args.degree,args.dv,args.device,basis_mode=basis_mode)
    density.F[:]=local.F
    density.qa[:]=cp.asarray(problem.A@r);density.ia[:]=density.qa
    density.qg[:]=cp.asarray(problem.G@r);density.ig[:]=density.qg
    density.M[:]=cp.asarray(r*problem.e)
    density.history[:]=cp.asarray(r[None,:]*DT/1000.)
    before={name:getattr(density,name).copy() for name in ('F','qa','ia','qg','ig','M','history')}
    density.advance_step()
    errors={name:float(cp.max(abs(getattr(density,name)-old)).get()) for name,old in before.items()}
    errors['spike_rate_hz']=float(cp.max(abs(density.activity*1000/DT-cp.asarray(r))).get())
    density.save(folder)
    qa=density.diagnostics()
    passed=(errors['F']<1e-11 and errors['spike_rate_hz']<1e-7 and
        max(errors[k] for k in ('qa','ia','qg','ig','M','history'))<1e-9 and qa['finite'] and qa['maximum_mass_error']<1e-9)
    write(folder/'status.json',dict(status='FULL_MAP_EQUILIBRIUM_CORRECTED' if passed else 'FULL_MAP_CHECK_FAILED',D=args.D,
        mean_E_hz=float(problem.eweights@r),one_step_absolute_residual=errors,
        density_diagnostics=qa,stability='NOT_COMPUTED',
        numerical_mesh_acceptance='PENDING',source='Corrected local stationary PDFs plus exact recurrent/delay/M equilibrium constraints'))
    print(folder,errors,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--D',type=float,required=True)
    ap.add_argument('--degree',type=int,default=6);ap.add_argument('--dv',type=float,default=.125)
    ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--table',type=Path,default=OUT/'stationary_response/degree6_dv0.125')
    ap.add_argument('--predictor',type=Path);ap.add_argument('--resume',type=Path)
    ap.add_argument('--label');ap.add_argument('--acceleration',action='store_true')
    ap.add_argument('--stable-marginal-projection',action='store_true')
    ap.add_argument('--outer-iterations',type=int,default=12)
    ap.add_argument('--response-secant',action='store_true',help='Update the root-solver Jacobian with measured local stationary response secants; no change to density dynamics or root acceptance')
    ap.add_argument('--signed-rate-iterates',action='store_true',help='Permit signed algebraic rate predictors while retaining all physical density and complete-root acceptance gates')
    ap.add_argument('--maximum-rate-step',type=float,help='Bound the largest algebraic rate update in Hz; a numerical trust radius, not a model parameter')
    ap.add_argument('--theta-interpolation',choices=['linear','pchip'],default='linear');run(ap.parse_args())

"""Full-density equilibrium continuation with global E rate as local coordinate.

Fixing a rate section regularizes continuation near a D turning point. D remains
an unknown in [0,1]. The section is a numerical coordinate, not a rate clamp in
the physical model; the delivered checkpoint uses the original dynamic map.
"""
from correct_equilibrium import *


def bordered_step(problem,r,D,target,residual=None,response_slope=None):
    f,J,fd,u=problem.equations(r,D)
    if response_slope is not None:
        z,dz=problem.resource(D)
        B=problem.A-sparse.diags(z)@problem.G-problem.Mcoupling
        J=problem.I-sparse.diags(response_slope)@B
        fd=response_slope*dz*(problem.G@r)
    if residual is not None:f=residual
    matrix=sparse.vstack([sparse.hstack([J,fd[:,None]]),
        sparse.csr_matrix(np.r_[problem.eweights,0.][None,:])]).tocsr()
    delta=spsolve(matrix,-np.r_[f,problem.eweights@r-target])
    return delta[:-1],delta[-1],u


def run(args):
    name=f'rE{args.rate:.8f}_degree{args.degree}_dv{args.dv:g}'
    if args.label:name+='_'+args.label
    folder=OUT/'corrected_rate_sections'/name
    folder.mkdir(parents=True,exist_ok=False)
    write(folder/'progress.json',dict(status='INITIALIZING_PREDICTOR',pid=os.getpid(),rate_section_hz=args.rate,degree=args.degree,table=str(args.table)))
    p=EquilibriumProblem(args.table,'pchip')
    basis_mode=read(args.table/'config.json').get('basis_mode','legacy')
    resumed=None
    if args.resume:
        cfg=read(args.resume/'config.json')
        assert cfg.get('basis_mode','legacy')==basis_mode
        with np.load(args.resume/'stationary_local_state.npz') as z:resumed={k:z[k] for k in z.files}
        r=resumed['rate_hz'].copy();D=float(cfg['D'])
    else:
        with np.load(args.predictor) as z:
            rates=z['rate_hz'];at=np.argmin(abs(rates@p.eweights-args.rate));r=rates[at].copy();D=float(z['D'][at])
    for it in range(0 if resumed is not None else 25):
        residual=p.equations(r,D,False)
        if np.max(abs(residual))<1e-8 and abs(p.eweights@r-args.rate)<1e-10:break
        dr,dd,_=bordered_step(p,r,D,args.rate);error=np.max(abs(residual))+abs(p.eweights@r-args.rate)*100
        accepted=False
        for a in 2.**-np.arange(15):
            nd=D+a*dd;nr=r+a*dr
            if not 0<=nd<=1:continue
            nf=p.equations(nr,nd,False)
            if np.max(abs(nf))+abs(p.eweights@nr-args.rate)*100<error:
                r,D=nr,nd;accepted=True;break
        if not accepted:raise RuntimeError('Rate-section predictor did not converge')
    if resumed is None:assert np.max(abs(p.equations(r,D,False)))<1e-7
    u=p.equations(r,D)[-1]
    local=LocalStationaryDensity(p.theta,p.geo['population'],u,args.degree,args.dv,args.device,basis_mode=basis_mode)
    if resumed is None or resumed['F'].shape!=local.F.shape:initialized=warm_start(local,p,u)
    else:local.F[:]=cp.asarray(resumed['F']);initialized=True
    history=[];started=time.time();converged=False;previous_u=None;previous_actual=None;last_slope=None
    if args.response_seed:
        assert resumed is not None and args.secant_jacobian
        assert args.response_seed.parent.resolve()==args.resume.resolve(), 'Use the measured response at this actual seed root'
        assert read(args.response_seed/'status.json')['status']=='LOCAL_STATIONARY_SLOPE_COMPLETE'
        with np.load(args.response_seed/'susceptibility.npz') as measured:last_slope=measured['static_derivative_hz_per_mv'].copy()
        assert last_slope.shape==r.shape and np.isfinite(last_slope).all()
        previous_u=resumed['current_mv'].copy();previous_actual=resumed['actual_rate_hz'].copy()
    for outer in range(args.outer_iterations):
        u=p.equations(r,D)[-1];local.drive[:]=cp.asarray(u)
        last=history[-1]['rate_residual_hz'] if history else .1
        tolerance=max(1e-12,min(1e-8,last*1e-9))
        def progress(it,residual,rate):
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),outer=outer,D=D,
                mean_E_hz=float(p.eweights@r),inner=it,pdf_residual=float(residual.max()),wall_s=time.time()-started))
            print('section',args.rate,'D',D,'outer',outer,'inner',it,'residual',residual.max(),flush=True)
        actual,inner,diag,_=local.solve(10000,progress,tolerance=tolerance,acceleration=True,
            stable_marginal_projection=args.stable_marginal_projection)
        residual=r-actual
        used_secants=0
        if args.secant_jacobian:
            slope=p.response(u)[1] if last_slope is None else last_slope.copy()
            if previous_u is not None:
                delta_u=u-previous_u;resolved=abs(delta_u)>1e-5
                secant=(actual-previous_actual)/np.where(resolved,delta_u,1.)
                valid=resolved&np.isfinite(secant)&(secant>=0.)
                slope[valid]=secant[valid];used_secants=int(valid.sum())
            last_slope=slope
        else:slope=None
        row=dict(outer=outer,D=D,rate_residual_hz=float(np.max(abs(residual))),
            mean_rate_constraint_error=float(p.eweights@r-args.rate),local=diag,
            local_response_secants=used_secants)
        history.append(row);print(row,flush=True)
        if diag['converged'] and np.max(abs(residual))<1e-7 and diag['max_stationary_residual']<2e-12 and abs(p.eweights@r-args.rate)<1e-10:
            converged=True;break
        if not diag['converged']:break
        dr,dd,_=bordered_step(p,r,D,args.rate,residual,response_slope=slope)
        previous_u=u.copy();previous_actual=actual.copy()
        scale=.5 if outer and row['rate_residual_hz']>1.2*history[-2]['rate_residual_hz'] else 1.
        if args.maximum_rate_step is not None:
            assert args.maximum_rate_step>0
            scale=min(scale,args.maximum_rate_step/max(float(np.max(abs(dr))),1e-300))
        if args.maximum_D_step is not None:
            assert args.maximum_D_step>0
            scale=min(scale,args.maximum_D_step/max(abs(float(dd)),1e-300))
        while not 0<=D+scale*dd<=1:scale*=.5
        row.update(algebraic_update_scale=scale,maximum_rate_update_hz=float(np.max(abs(scale*dr))),
                   parameter_update_D=float(scale*dd))
        write(folder/'outer_history.json',dict(status='RUNNING',history=history))
        r+=scale*dr;D+=scale*dd
    cfg=dict(D=D,degree=args.degree,voltage_dv=args.dv,dt_ms=DT,communication_operators=str(OPERATORS),
        global_rate_section_hz=args.rate,table=str(p.table_path),predictor_theta_interpolation='pchip',
        secant_jacobian=args.secant_jacobian,basis_mode=basis_mode,
        stable_marginal_projection=args.stable_marginal_projection,
        maximum_outer_iterations=args.outer_iterations,maximum_algebraic_rate_step_hz=args.maximum_rate_step,
        maximum_algebraic_D_step=args.maximum_D_step,
        measured_response_seed=str(args.response_seed.resolve()) if args.response_seed else None,
        object='Original dynamic density equilibrium; rate is a continuation coordinate only, M remains dynamic')
    write(folder/'config.json',cfg);write(folder/'correction.json',dict(converged=converged,history=history,wall_s=time.time()-started))
    extra=dict(last_numerical_response_slope=last_slope) if last_slope is not None else {}
    np.savez_compressed(folder/'stationary_local_state.npz',F=cp.asnumpy(local.F),rate_hz=r,
        actual_rate_hz=actual,current_mv=u,theta=p.theta,population=p.geo['population'],**extra)
    if not converged:
        write(folder/'status.json',dict(status='NETWORK_CORRECTION_INCOMPLETE',history=history));return
    assert r.min()>-1e-7
    density=AutonomousDensity(D,args.degree,args.dv,args.device,basis_mode=basis_mode);density.F[:]=local.F
    density.qa[:]=cp.asarray(p.A@r);density.ia[:]=density.qa
    density.qg[:]=cp.asarray(p.G@r);density.ig[:]=density.qg
    density.M[:]=cp.asarray(r*p.e);density.history[:]=cp.asarray(r[None,:]*DT/1000.)
    before={name:getattr(density,name).copy() for name in ('F','qa','ia','qg','ig','M','history')}
    density.advance_step()
    errors={name:float(cp.max(abs(getattr(density,name)-old)).get()) for name,old in before.items()}
    errors['spike_rate_hz']=float(cp.max(abs(density.activity*1000/DT-cp.asarray(r))).get())
    density.save(folder)
    qa=density.diagnostics()
    passed=(errors['F']<1e-11 and errors['spike_rate_hz']<1e-7 and
        max(errors[k] for k in ('qa','ia','qg','ig','M','history'))<1e-9 and qa['finite'] and qa['maximum_mass_error']<1e-9)
    write(folder/'status.json',dict(status='FULL_MAP_EQUILIBRIUM_CORRECTED' if passed else 'FULL_MAP_CHECK_FAILED',D=D,mean_E_hz=float(p.eweights@r),
        one_step_absolute_residual=errors,density_diagnostics=qa,
        stability='NOT_COMPUTED',bifurcation_type='NOT_CLASSIFIED',numerical_mesh_acceptance='PENDING'))
    print(folder,D,errors,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--rate',type=float,required=True)
    ap.add_argument('--predictor',type=Path,default=OUT/'equilibrium_predictors/low/branch.npz')
    ap.add_argument('--device',type=int,default=0);ap.add_argument('--resume',type=Path)
    ap.add_argument('--degree',type=int,default=6);ap.add_argument('--dv',type=float,default=.125)
    ap.add_argument('--table',type=Path,default=OUT/'stationary_response/degree6_dv0.125')
    ap.add_argument('--label');ap.add_argument('--secant-jacobian',action='store_true')
    ap.add_argument('--stable-marginal-projection',action='store_true')
    ap.add_argument('--outer-iterations',type=int,default=15)
    ap.add_argument('--maximum-rate-step',type=float)
    ap.add_argument('--maximum-D-step',type=float)
    ap.add_argument('--response-seed',type=Path,help='Measured complete stationary susceptibility at the resumed root; seeds only the numerical Jacobian')
    run(ap.parse_args())

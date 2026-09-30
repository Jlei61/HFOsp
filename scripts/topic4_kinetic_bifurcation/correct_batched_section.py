"""Memory-bounded full-density equilibrium correction at a mean-rate section.

Only independent local stationary PDF solves are batched. The coupled network
Newton system and final delay/M/transport check use the complete original model.
No subset of spatial connections or populations is dropped.
"""
from correct_rate_section import *
import shutil


def run(args):
    folder=OUT/'corrected_batched_sections'/args.label;folder.mkdir(parents=True,exist_ok=False)
    resumed=read(args.resume_from/'config.json') if args.resume_from else None
    if resumed:
        source_status=read(args.resume_from/'status.json')
        assert source_status['status']=='LOCAL_SOLVE_INCOMPLETE' and source_status['outer']==1
        args.table=Path(resumed['table']);args.rate=resumed['global_rate_section_hz']
    else:assert args.table is not None and args.rate is not None and args.predictor is not None
    p=EquilibriumProblem(args.table,'pchip');tcfg=read(args.table/'config.json')
    degree=tcfg['degree'];dv=tcfg['voltage_dv'];basis=tcfg.get('basis_mode','legacy')
    previous_u=None;previous_actual=None;last_slope=None;history=[];outer_start=0
    if resumed:
        history=read(args.resume_from/'correction.json')['history'];assert len(history)==1 and history[0]['outer']==0
        with np.load(args.resume_from/'latest_rates.npz') as z:
            r=z['rate_hz'].copy();D=float(z['D']);previous_actual=z['actual_rate_hz'].copy();previous_u=z['current_mv'].copy()
        last_slope=p.response(previous_u)[1]
        dr,dd,_=bordered_step(p,r,D,args.rate,r-previous_actual,response_slope=last_slope)
        r+=dr;D+=dd;outer_start=1
        assert abs(D-read(args.resume_from/'progress.json')['D'])<1e-13
    else:
        with np.load(args.predictor) as z:
            at=np.argmin(abs(z['rate_hz']@p.eweights-args.rate));r=z['rate_hz'][at].copy();D=float(z['D'][at])
        for _ in range(20):
            f=p.equations(r,D,False)
            if np.max(abs(f))<1e-8 and abs(p.eweights@r-args.rate)<1e-11:break
            dr,dd,_=bordered_step(p,r,D,args.rate);r+=dr;D+=dd
        assert 0<=D<=1 and np.max(abs(p.equations(r,D,False)))<1e-7
    cfg=dict(D=D,degree=degree,voltage_dv=dv,dt_ms=DT,basis_mode=basis,communication_operators=str(OPERATORS),
        global_rate_section_hz=args.rate,table=str(args.table.resolve()),batch_size=args.batch_size,
        resumed_from=str(args.resume_from.resolve()) if args.resume_from else None,
        stable_marginal_projection=args.stable_marginal_projection,
        object='Complete spatial conditional-density equilibrium with dynamic M; rate section is a numerical coordinate',
        local_solver='Batches of independent constant-current stationary PDF solves; global recurrent coupling is unchanged')
    write(folder/'config.json',cfg)
    table_path=args.table/'density_coefficients.npy'
    if table_path.exists():table_F=np.load(table_path,mmap_mode='r')
    else:
        with np.load(args.table/'density_state.npz') as z:table_F=z['F']
    if resumed:
        shutil.copyfile(args.resume_from/'density_coefficients.npy',folder/'density_coefficients.npy')
        Fdisk=np.load(folder/'density_coefficients.npy',mmap_mode='r+')
    else:
        Fdisk=np.lib.format.open_memmap(folder/'density_coefficients.npy',mode='w+',dtype=np.float64,
                                      shape=(p.P,*table_F.shape[1:]))
    started=time.time();converged=False
    for outer in range(outer_start,args.outer_iterations):
        u=p.equations(r,D)[-1];actual=np.empty(p.P);local_qa=[]
        np.savez_compressed(folder/'outer_start.npz',rate_hz=r,D=D,current_mv=u,outer=outer,
            previous_u=previous_u if previous_u is not None else np.empty(0),
            previous_actual=previous_actual if previous_actual is not None else np.empty(0),
            last_slope=last_slope if last_slope is not None else np.empty(0))
        tolerance=1e-10 if not history else max(2e-12,min(1e-10,history[-1]['rate_residual_hz']*1e-9))
        for left in range(0,p.P,args.batch_size):
            right=min(p.P,left+args.batch_size);sel=slice(left,right)
            m=LocalStationaryDensity(p.theta[sel],p.geo['population'][sel],u[sel],degree,dv,args.device,basis_mode=basis)
            if outer==0:
                C=len(p.current_grid);at=np.clip(np.searchsorted(p.current_grid,u[sel])-1,0,C-2)
                w=np.clip((u[sel]-p.current_grid[at])/(p.current_grid[at+1]-p.current_grid[at]),0.,1.)
                x=np.zeros(m.F.shape)
                for ti,tw in [(p.lo[sel],1-p.w[sel]),(p.hi[sel],p.w[sel])]:
                    for off,uw in [(0,1-w),(1,w)]:
                        x+=table_F[ti*C+at+off,:,:m.width]*(tw*uw)[:,None,None]
                m.F[:]=cp.asarray(x);del x
            else:m.F[:]=cp.asarray(Fdisk[sel,:,:m.width])
            def progress(it,residual,rate):
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),outer=outer,D=D,left=left,right=right,
                    iteration=it,pdf_residual=float(residual.max()),wall_s=time.time()-started))
                print('batched correction',args.label,outer,left,it,residual.max(),flush=True)
            rate,res,diag,_=m.solve(10000,progress,tolerance=tolerance,acceleration=True,
                stable_marginal_projection=args.stable_marginal_projection)
            actual[sel]=rate;Fdisk[sel]=0.;Fdisk[sel,:,:m.width]=cp.asnumpy(m.F);Fdisk.flush()
            local_qa.append(dict(left=left,right=right,**diag))
            write(folder/f'outer_{outer:02d}_batches.json',local_qa)
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),outer=outer,D=D,completed_groups=right,
                total_groups=p.P,maximum_stationary_residual=max(q['max_stationary_residual'] for q in local_qa),wall_s=time.time()-started))
            del m;cp.get_default_memory_pool().free_all_blocks()
            if not diag['converged']:
                write(folder/'status.json',dict(status='LOCAL_SOLVE_INCOMPLETE',outer=outer,batch=local_qa[-1]));return
        residual=r-actual
        row=dict(outer=outer,D=D,mean_E_hz=float(p.eweights@r),rate_residual_hz=float(np.max(abs(residual))),
            mean_rate_constraint_error=float(p.eweights@r-args.rate),maximum_stationary_residual=max(q['max_stationary_residual'] for q in local_qa))
        history.append(row);write(folder/'correction.json',dict(converged=False,history=history,wall_s=time.time()-started))
        np.savez_compressed(folder/'latest_rates.npz',rate_hz=r,D=D,actual_rate_hz=actual,current_mv=u)
        print('coupled correction',args.label,row,flush=True)
        if np.max(abs(residual))<1e-7 and row['maximum_stationary_residual']<3e-12 and abs(row['mean_rate_constraint_error'])<1e-10:
            converged=True;break
        slope=p.response(u)[1] if last_slope is None else last_slope.copy()
        if previous_u is not None:
            du=u-previous_u;resolved=abs(du)>1e-6;sec=(actual-previous_actual)/np.where(resolved,du,1.)
            valid=resolved&np.isfinite(sec)&(sec>=0.);slope[valid]=sec[valid]
        last_slope=slope;dr,dd,_=bordered_step(p,r,D,args.rate,residual,response_slope=slope)
        previous_u=u.copy();previous_actual=actual.copy()
        scale=.5 if outer and row['rate_residual_hz']>1.2*history[-2]['rate_residual_hz'] else 1.
        while not 0<=D+scale*dd<=1:scale*=.5
        r+=scale*dr;D+=scale*dd
    write(folder/'correction.json',dict(converged=converged,history=history,wall_s=time.time()-started))
    cfg['D']=D;write(folder/'config.json',cfg)
    if not converged:
        write(folder/'status.json',dict(status='NETWORK_CORRECTION_INCOMPLETE',history=history));return
    if args.batched_check:
        from batched_equilibrium_check import check
        checked=check(folder,args.device,args.batch_size)
        errors=checked['one_step_absolute_residual'];qa=checked['density_diagnostics']
        passed=(errors['F']<1e-11 and errors['spike_rate_hz']<1e-7 and
                max(errors[k] for k in ('qa','ia','qg','ig','M','history'))<1e-9 and
                qa['finite'] and qa['maximum_mass_error']<1e-9)
        write(folder/'status.json',dict(status='FULL_MAP_EQUILIBRIUM_CORRECTED' if passed else 'FULL_MAP_CHECK_FAILED',
            D=D,mean_E_hz=float(p.eweights@r),one_step_absolute_residual=errors,density_diagnostics=qa,
            checkpoint_layout='Stationary PDF in density_coefficients.npy; rates and D in latest_rates.npz; currents reconstructed from exact original operators',
            verification='All recurrent connections together, followed by independent local PDF/observe batches',
            stability='NOT_COMPUTED',bifurcation_type='NOT_CLASSIFIED'))
        return
    # A stationary local solve is not enough: check one step of the full
    # original network, including the actual CUDA delayed recurrence.
    density=AutonomousDensity(D,degree,dv,args.device,basis_mode=basis)
    density.F[:]=cp.asarray(Fdisk);density.qa[:]=cp.asarray(p.A@r);density.ia[:]=density.qa
    density.qg[:]=cp.asarray(p.G@r);density.ig[:]=density.qg
    density.M[:]=cp.asarray(r*p.e);density.history[:]=cp.asarray(r[None,:]*DT/1000.)
    before={name:getattr(density,name).copy() for name in ('qa','ia','qg','ig','M','history')}
    density.advance_step()
    errors={name:float(cp.max(abs(getattr(density,name)-old)).get()) for name,old in before.items()}
    errors['F']=max(float(cp.max(abs(density.F[left:left+args.batch_size]-cp.asarray(Fdisk[left:left+args.batch_size]))).get())
                    for left in range(0,p.P,args.batch_size))
    errors['spike_rate_hz']=float(cp.max(abs(density.activity*1000/DT-cp.asarray(r))).get())
    density.save(folder)
    write(folder/'status.json',dict(status='FULL_MAP_EQUILIBRIUM_CORRECTED',D=D,mean_E_hz=float(p.eweights@r),
        one_step_absolute_residual=errors,density_diagnostics=density.diagnostics(),
        stability='NOT_COMPUTED',bifurcation_type='NOT_CLASSIFIED'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--rate',type=float)
    ap.add_argument('--table',type=Path);ap.add_argument('--predictor',type=Path)
    ap.add_argument('--resume-from',type=Path,help='Resume the saved coupled update after a local failure in outer 1, preserving prior results')
    ap.add_argument('--stable-marginal-projection',action='store_true')
    ap.add_argument('--batch-size',type=int,default=128);ap.add_argument('--outer-iterations',type=int,default=20)
    ap.add_argument('--batched-check',action='store_true',help='Verify the exact full native step with independent PDF batches to bound GPU memory')
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())

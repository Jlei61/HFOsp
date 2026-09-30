"""Bounded Newton--GMRES repair of one stalled local stationary PDF solve.

Reconstructs the actual outer-1 current from the saved coupled update. It does
not treat the mixed outer-0/outer-1 disk array as a network equilibrium and never
changes the model, residual tolerance, or prior output. A single worst group is
used to decide whether this numerical repair merits a full continuation.
"""
from correct_batched_section import *
from network_tangent import CODE as TANGENT_CODE
from scipy.sparse.linalg import LinearOperator,gmres


def run(a):
    cfg=read(a.source/'config.json');status=read(a.source/'status.json')
    assert status['status']=='LOCAL_SOLVE_INCOMPLETE' and status['outer']==1 and status['batch']['left']==0
    folder=OUT/'local_stationary_polish'/a.label;folder.mkdir(parents=True,exist_ok=False)
    p=EquilibriumProblem(Path(cfg['table']),'pchip')
    with np.load(a.source/'latest_rates.npz') as z:
        r=z['rate_hz'];D=float(z['D']);actual=z['actual_rate_hz'];u=z['current_mv']
    dr,dd,_=bordered_step(p,r,D,cfg['global_rate_section_hz'],r-actual,response_slope=p.response(u)[1])
    D1=D+dd;u1=p.equations(r+dr,D1)[-1]
    progress=read(a.source/'progress.json');assert abs(D1-progress['D'])<1e-13
    count=status['batch']['right'];disk=np.load(a.source/'density_coefficients.npy',mmap_mode='r')
    m=LocalStationaryDensity(p.theta[:count],p.geo['population'][:count],u1[:count],
        cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.F[:]=cp.asarray(disk[:count,:,:m.width]);m.map();res=m.residual();gid=int(np.argmax(res.max(1)))
    measured=res.tolist();del m;cp.get_default_memory_pool().free_all_blocks()
    m=LocalStationaryDensity(p.theta[gid:gid+1],p.geo['population'][gid:gid+1],u1[gid:gid+1],
        cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.F[:]=cp.asarray(disk[gid:gid+1,:,:m.width]);shape=m.F.shape
    if a.warm_start:
        with np.load(a.warm_start/'local_state.npz') as z:
            assert int(z['group'])==gid and abs(float(z['D'])-D1)<1e-13
            assert abs(float(z['current_mv'])-u1[gid])<1e-13
            m.F[:]=cp.asarray(z['F'])
    mass=cp.asnumpy(m.mass);marginal=cp.asnumpy(m.stationary_noise_marginal);A=cp.asnumpy(m.A)
    basis_qa=dict(mass_stationarity=float(np.max(abs(A@mass-mass))),
        marginal_stationarity=float(np.max(abs(A@marginal-marginal))),
        mass_marginal_difference=float(np.max(abs(mass-marginal))))
    write(folder/'config.json',dict(source=str(a.source.resolve()),reconstructed_outer1_D=D1,group=gid,
        current_mv=float(u1[gid]),theta_mv=float(p.theta[gid]),population=int(p.geo['population'][gid]),
        initial_batch_residuals=measured,basis_diagnostics=basis_qa,
        warm_start=str(a.warm_start.resolve()) if a.warm_start else None,
        gmres_restart=a.restart,gmres_cycles=a.cycles,
        marginal_projection='Shared physical voltage PDF' if a.stable_marginal_projection else 'Each signed modal row normalized by its marginal',
        scope='One local stationary root repair; no whole-network equilibrium or stability claim'))
    mod=cp.RawModule(code=TANGENT_CODE,options=('--fmad=false',),name_expressions=['voltage_tangent'])
    tangent=mod.get_function('voltage_tangent');dout=cp.empty_like(m.F);flux=cp.empty_like(m.flux);zero=cp.zeros(1)
    rows=[];start=time.time()
    for outer in range(a.iterations):
        rate=m.map();before_components=m.residual().tolist();before=float(m.residual().max());x=m.F.copy();rhs=(m.Q-x).copy()
        if a.stable_marginal_projection:
            physical=cp.maximum(cp.einsum('k,gkv->gv',m.mass,x),0.)
            distribution=(physical/physical.sum(1)[:,None])[:,None,:]
        else:distribution=x/cp.sum(x,axis=2)[:,:,None]
        def project(v):return v-v.sum(2)[:,:,None]*distribution
        projection_change=float(cp.linalg.norm(rhs-project(rhs)).get())
        rhs=project(rhs);reference_moved=cp.ascontiguousarray(m.A@x)
        calls=[0]
        def apply(v):
            v=project(cp.asarray(v).reshape(shape));moved=cp.ascontiguousarray(m.A@v)
            tangent((m.P*m.K,),(128,),(reference_moved,moved,dout,flux,m.de,m.dc,m.dw,m.nodes,m.ratio,
                m.decay,m.drive,zero,m.refs,np.int32(m.K),np.int32(m.nv),np.int32(m.width)))
            calls[0]+=1
            return cp.asnumpy(project(v-dout)).ravel()
        operator=LinearOperator((m.F.size,m.F.size),matvec=apply,dtype=np.float64)
        linear_res=[]
        def callback(value):
            linear_res.append(float(value))
            if len(linear_res)%20==0:
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),outer=outer,
                    native_residual_before=before,linear_iterations=len(linear_res),linear_relative_residual=float(value),
                    wall_s=time.time()-start))
        delta,info=gmres(operator,cp.asnumpy(rhs).ravel(),rtol=1e-3,atol=0.,restart=a.restart,maxiter=a.cycles,
                         callback=callback,callback_type='pr_norm')
        delta=project(cp.asarray(delta).reshape(shape));trials=[];best=before;best_state=x
        predicted=cp.asarray(apply(cp.asnumpy(delta).ravel())).reshape(shape)
        m.F[:]=x;m.map();base_map=m.Q.copy()
        m.F[:]=x+delta;m.map()
        independent_linear_error=float(cp.linalg.norm((delta-(m.Q-base_map))-predicted).get())
        for scale in (1.,.5,.25,.125):
            m.F[:]=x+scale*delta;m.map();value=float(m.residual().max())
            trials.append(dict(scale=scale,residual=value))
            if value<best:best=value;best_state=m.F.copy()
        m.F[:]=best_state
        row=dict(iteration=outer,native_residual_before=before,native_residual_after=best,
            before_residual_components=before_components,rhs_projection_change=projection_change,
            nonlinear_increment_linearization_error=independent_linear_error,
            linear_info=int(info),linear_iterations=len(linear_res),linear_history=linear_res,
            matvecs=calls[0],trials=trials,update_norm=float(cp.linalg.norm(delta).get()))
        rows.append(row);write(folder/'iterations.json',rows);print(row,flush=True)
        if best<2e-12 or best>=before*.99:break
    m.map();final=float(m.residual().max())
    np.savez_compressed(folder/'local_state.npz',F=cp.asnumpy(m.F),group=gid,current_mv=u1[gid],D=D1)
    physical=cp.einsum('k,gkv->gv',m.mass,m.F)
    write(folder/'result.json',dict(status='LOCAL_POLISH_PASS' if final<2e-12 else 'LOCAL_POLISH_INCOMPLETE',
        group=gid,residual=final,rows=rows,basis_diagnostics=basis_qa,
        maximum_mass_error=float(cp.max(abs(physical.sum(1)-1)).get()),
        maximum_negative_probability=float(cp.maximum(-physical,0).sum(1).max().get()),
        wall_s=time.time()-start,network_equilibrium='NOT_ACCEPTED_FROM_ONE_GROUP'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--iterations',type=int,default=3);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--warm-start',type=Path);ap.add_argument('--restart',type=int,default=40);ap.add_argument('--cycles',type=int,default=6)
    ap.add_argument('--stable-marginal-projection',action='store_true')
    run(ap.parse_args())

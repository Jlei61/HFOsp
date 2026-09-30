"""Local susceptibility of a complete, memory-bounded equilibrium.

All groups are retained. Batching is only over independent local PDF maps at
the fixed equilibrium current; delays, synapses and dynamic M are assembled
later. The integrated analytic impulse is compared to independent stationary
plus/minus current solves. A selected-group pilot is explicitly not a complete
network susceptibility and cannot be consumed by the network spectrum code.
"""
from stationary_response import *
from equilibrium_predictor import EquilibriumProblem
from network_tangent import CODE as TANGENT_CODE


def run(a):
    source=a.source;cfg=read(source/'config.json')
    assert read(source/'status.json')['status']=='FULL_MAP_EQUILIBRIUM_CORRECTED'
    folder=source/a.label;folder.mkdir(parents=True,exist_ok=False)
    p=EquilibriumProblem(Path(cfg['table']),'pchip')
    if (source/'latest_rates.npz').exists():
        with np.load(source/'latest_rates.npz') as z:
            r=z['rate_hz'];u=z['current_mv'];D=float(z['D'])
        disk=np.load(source/'density_coefficients.npy',mmap_mode='r')
    else:
        with np.load(source/'stationary_local_state.npz') as z:
            r=z['rate_hz'];u=z['current_mv'];disk=z['F']
        D=float(cfg['D'])
    if a.pilot:
        # Include large-response groups and spatial/population coverage.
        fp=p.response(u)[1]
        selected=np.unique(np.r_[np.argsort(-fp)[:8],np.linspace(0,p.P-1,9,dtype=int)])
    else:selected=np.arange(p.P)
    cfgout=dict(source=str(source.resolve()),D=D,degree=cfg['degree'],voltage_dv=cfg['voltage_dv'],
        basis_mode=cfg.get('basis_mode','legacy'),epsilon_mv=a.epsilon,duration_ms=a.duration,
        stationary_residual_tolerance=a.stationary_tolerance,
        group_indices=selected,complete_network=not a.pilot,batch_size=a.batch_size,
        scope='Local PDF susceptibility only; network coupling and dynamic M remain to be assembled')
    write(folder/'config.json',cfgout)
    nsteps=round(a.duration/DT);h=np.empty((nsteps,len(selected)));static=np.empty(len(selected))
    fwd=np.empty_like(static);back=np.empty_like(static);qa_batches=[];started=time.time();last=started
    cp.cuda.Device(a.device).use()
    module=cp.RawModule(code=TANGENT_CODE,options=('--fmad=false',),name_expressions=['voltage_tangent'])
    kernel=module.get_function('voltage_tangent')
    for left in range(0,len(selected),a.batch_size):
        ids=selected[left:left+a.batch_size];right=left+len(ids)
        m=LocalStationaryDensity(p.theta[ids],p.geo['population'][ids],u[ids],cfg['degree'],cfg['voltage_dv'],
            a.device,basis_mode=cfg.get('basis_mode','legacy'))
        reference=cp.asarray(disk[ids,:,:m.width]);m.F[:]=reference
        reference_moved=cp.ascontiguousarray(cp.matmul(m.A,reference))
        zero=cp.zeros(m.P);one=cp.ones(m.P);m.F.fill(0.)
        max_marginal=0.
        for step in range(nsteps):
            moved=cp.ascontiguousarray(cp.matmul(m.A,m.F))
            kernel((m.P*m.K,),(128,),(reference_moved,moved,m.Q,m.flux,m.de,m.dc,m.dw,m.nodes,
                m.ratio,m.decay,m.drive,one if step==0 else zero,m.refs,
                np.int32(m.K),np.int32(m.nv),np.int32(m.width)))
            h[step,left:right]=cp.asnumpy(m.flux@m.mass*1000/DT)
            if step==0:initial_tangent=m.Q.copy()
            m.F,m.Q=m.Q,m.F
            if (step+1)%100==0:max_marginal=max(max_marginal,float(cp.max(abs(m.F.sum(2))).get()))
            if time.time()-last>20:
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),stage='analytic_impulse',
                    completed_groups=left,current_right=right,elapsed_ms=(step+1)*DT,wall_s=time.time()-started))
                print(a.label,'impulse',left,(step+1)*DT,flush=True);last=time.time()
        # Independent one-step input finite difference checks the analytic
        # current term, including active transport boundaries and resets.
        m.F[:]=reference;m.drive[:]=cp.asarray(u[ids]+a.epsilon);m.map();plus=m.Q.copy()
        m.F[:]=reference;m.drive[:]=cp.asarray(u[ids]-a.epsilon);m.map()
        difference=(plus-m.Q)/(2*a.epsilon)-initial_tangent
        impulse_error=float((cp.linalg.norm(difference.ravel())/cp.maximum(cp.linalg.norm(initial_tangent.ravel()),1e-30)).get())
        del plus,initial_tangent,difference,reference_moved,moved
        rates=[];solveqa=[]
        for sign in (1,-1):
            m.F[:]=reference;m.drive[:]=cp.asarray(u[ids]+sign*a.epsilon)
            def progress(it,residual,rate):
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),stage='stationary_finite_difference',
                    completed_groups=left,current_right=right,sign=sign,iteration=it,
                    maximum_stationary_residual=float(residual.max()),wall_s=time.time()-started))
                print(a.label,'static',left,sign,it,residual.max(),flush=True)
            rate,_,diag,_=m.solve(a.max_iterations,progress,tolerance=a.stationary_tolerance,acceleration=True)
            rates.append(rate);solveqa.append(diag)
            if not diag['converged']:
                write(folder/'status.json',dict(status='STATIONARY_DIFFERENCE_INCOMPLETE',left=left,sign=sign,qa=diag));return
        static[left:right]=(rates[0]-rates[1])/(2*a.epsilon)
        fwd[left:right]=(rates[0]-r[ids])/a.epsilon;back[left:right]=(r[ids]-rates[1])/a.epsilon
        qa_batches.append(dict(left=left,right=right,group_indices=ids,impulse_relative_error=impulse_error,
            maximum_noise_marginal_tangent_error=max_marginal,stationary_solves=solveqa))
        write(folder/'batches.json',qa_batches)
        del m,reference;cp.get_default_memory_pool().free_all_blocks()
    integral=h.sum(0);area=abs(h).sum(0);tail=abs(h[-round(50/DT):]).sum(0)
    valid=abs(static)>1e-4;error=abs(integral[valid]-static[valid])/abs(static[valid])
    maxerror=float(error.max(initial=0.));maxtail=float(np.max(tail/np.maximum(area,1e-12)))
    impulse_error=max(x['impulse_relative_error'] for x in qa_batches)
    np.savez_compressed(folder/'susceptibility.npz',kernel_hz_per_mv=h,time_s=np.arange(nsteps)*DT/1000.,
        group_indices=selected,static_derivative_hz_per_mv=static,integrated_derivative_hz_per_mv=integral,
        forward_derivative_hz_per_mv=fwd,backward_derivative_hz_per_mv=back,tail_absolute_area=tail)
    qa=dict(max_relative_dc_error=maxerror,median_relative_dc_error=float(np.median(error)),
        max_tail_fraction_of_absolute_response=maxtail,max_analytic_input_impulse_relative_error=impulse_error,
        maximum_noise_marginal_tangent_error=max(x['maximum_noise_marginal_tangent_error'] for x in qa_batches),
        epsilon_mv=a.epsilon,complete_network=not a.pilot,
        usable_for_dynamic_spectrum=bool(not a.pilot and maxerror<.01 and maxtail<.01 and impulse_error<.01),
        derivative='Analytic original transport Jacobian; independent stationary symmetric current difference')
    write(folder/'status.json',dict(status='PILOT_LOCAL_RESPONSE_COMPLETE' if a.pilot else 'LOCAL_TANGENT_RESPONSE_COMPLETE',
        qa=qa,wall_s=time.time()-started));print(a.label,qa,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--epsilon',type=float,default=.0005);ap.add_argument('--duration',type=float,default=200.)
    ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--max-iterations',type=int,default=8000)
    ap.add_argument('--stationary-tolerance',type=float,default=1e-12)
    ap.add_argument('--pilot',action='store_true');ap.add_argument('--device',type=int,default=0);run(ap.parse_args())

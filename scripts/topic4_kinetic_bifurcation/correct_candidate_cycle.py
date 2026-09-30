"""Correct a candidate integer-period orbit of the original density map.

A small Ritz subspace supplies a quasi-Newton preconditioner. Every accepted
step is evaluated by an independent complete native-time map iteration; no
interpolated time or added feedback is used in the physical equations.
Convergence proves a numerical map orbit, not discretization-independent SNN
correspondence or a bifurcation type.
"""
from cycle_monodromy import *


def run(args):
    assert not args.M_block_preconditioner or args.arnoldi_preconditioner
    source=Path(args.source);cfg=read(source/'config.json');spectrum=read(args.spectrum/'result.json')
    spec_cfg=read(args.spectrum/'config.json')
    for key in ('D','degree','voltage_dv'):assert spec_cfg[key]==cfg[key]
    folder=OUT/'corrected_cycles'/args.label;folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'))
    # Raw Arnoldi vectors use the fixed metric at the reference used to build
    # the spectrum. Keep that metric when restarting at an updated iterate.
    m.restore(Path(spec_cfg['source']) if args.arnoldi_preconditioner else source)
    coords=StateCoordinates(m)
    if args.arnoldi_preconditioner:m.restore(source)
    base_index=m.step_index;t=NetworkTangent(m)
    steps=spec_cfg['steps'];initial=coords.pack(m);x=initial.copy()
    vals=np.array(spectrum['final']['ritz_real'])+1j*np.array(spectrum['final']['ritz_imag'])
    modes=[];inverse=[];mode_info=[]
    for i,value in enumerate(vals if not args.arnoldi_preconditioner else []):
        path=args.spectrum/f'mode_{i:02d}_real.npz'
        if not path.exists():continue
        # The current preconditioner uses real eigenmodes only. Complex pairs
        # remain in the full residual and are handled by validated iterations.
        if abs(value.imag)>1e-8:continue
        if abs(1-value.real)<args.minimum_denominator:continue
        with np.load(path) as z:
            for k in STATE_NAMES:
                if k=='history':t.history[(m.step_index-np.arange(m.D))%m.D]=cp.asarray(z[k])
                else:getattr(t,k)[:]=cp.asarray(z[k])
        q=coords.pack(t);q/=cp.linalg.norm(q);modes.append(q)
        inverse.append(value.real/(1-value.real));mode_info.append(dict(index=i,eigenvalue=float(value.real)))
    if args.arnoldi_preconditioner:
        H=np.load(args.spectrum/'arnoldi.npz')['H'];dimension=H.shape[1]
        assert H.shape==(dimension+1,dimension)
        basis_files=[Path(spec_cfg['krylov_storage'])/f'q{i:03d}.npy' for i in range(dimension+1)]
        assert all(p.exists() for p in basis_files)
        Mfactor=0.
        G=np.eye(dimension,dimension+1)
        if args.M_block_preconditioner:
            beta=(1-DT/1000.)**steps;Mfactor=beta/(1-beta)
            qm=np.stack([np.asarray(np.load(p,mmap_mode='r')[coords.slices['M']]) for p in basis_files])
            G+=Mfactor*(qm[:-1]@qm.T)
        small=G[:,:dimension]-G@H;inv_small=np.linalg.inv(small)
        preconditioner=dict(method='Full Arnoldi low-rank inverse, including non-normal couplings',
            dimension=dimension,small_system_condition_number=float(np.linalg.cond(small)),
            equation='delta = residual + Q_(m+1) H_m (I-H_top)^(-1) Q_m^T residual',
            metric_reference=spec_cfg['source'])
        if args.M_block_preconditioner:
            preconditioner.update(M_native_decay_over_period=beta,M_inverse_diagonal=1+Mfactor,
                equation='Woodbury inverse of I-J0-(JQ-J0Q)Q^T; J0 is native M decay and zero on other coordinates',
                scope='M decay is used only to precondition the residual solve; original M remains dynamic throughout every shooting evaluation')
    else:
        assert modes, 'Need saved real Ritz modes for this preconditioner'
        gram=np.array([[float(cp.dot(a,b).get()) for b in modes] for a in modes])
        pinv=np.linalg.pinv(gram,rcond=1e-10)
        preconditioner=dict(method='Real-mode projection',real_modes_used=mode_info)
    config=dict(cfg,source=str(source.resolve()),period_steps=steps,period_ms=steps*DT,
        duration_ms=steps*DT,precision='FP64',
        period_scope='Integer-period orbit of the native discrete map; continuous-cycle interpretation remains separate',
        preconditioner_source=str(args.spectrum.resolve()),preconditioner=preconditioner,
        initial_ms=base_index*DT,M='dynamic',Z='frozen physical spatial field')
    config['numerical_candidate_gates']=dict(maximum_mass_error=1e-8,maximum_negative_voltage_probability=5e-4,
        minimum_step_spike_probability=-1e-6)
    write(folder/'config.json',config);started=time.time();last=started;history=[];call_count=0
    count=np.bincount(m.geo['group_cell'],weights=np.where(m.geo['population']==0,m.geo['group_size'],0),minlength=1600)
    def evaluate(state,record=False):
        nonlocal last,call_count
        call_count+=1;m.step_index=base_index;coords.unpack(state,m)
        for k in ('masserror','maxneg','minflux'):getattr(m,k).fill(0.)
        rates=[];fields=[];block=cp.zeros(m.P)
        for step in range(steps):
            rate=m.advance_step()*1000/DT
            if record:
                rates.append(cp.asnumpy(cp.r_[m.e_weights@rate,m.region_weights@rate]));block+=rate
                if (step+1)%10==0:
                    fields.append(cp.asnumpy(cp.bincount(m.observable_cell,weights=(block/10)*m.e_sizes,minlength=1600))/np.maximum(count,1))
                    block.fill(0.)
            if time.time()-last>20:
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),map_evaluations=call_count,
                    elapsed_ms=(step+1)*DT,latest=history[-1] if history else None,wall_s=time.time()-started))
                print('cycle correction map',args.label,call_count,(step+1)*DT,flush=True);last=time.time()
        mapped=coords.pack(m);res=mapped-state;norm=float(cp.linalg.norm(res).get())
        errors={k:float(cp.max(abs(res[coords.slices[k]].reshape(coords.shapes[k])/coords.scales[k])).get()) for k in STATE_NAMES}
        qa=m.diagnostics()
        valid=qa['finite'] and qa['maximum_mass_error']<1e-8 and qa['maximum_negative_voltage_probability']<5e-4 and qa['minimum_step_spike_probability']>-1e-6
        write(folder/f'evaluation_{call_count:03d}.json',dict(weighted_residual=norm,physical_numerical_gate=bool(valid),
            max_absolute_by_component=errors,diagnostics=qa))
        if not valid:norm=float('inf')
        return res,norm,errors,(np.asarray(rates),np.asarray(fields))
    residual,error,components,_=evaluate(x);status='ITERATION_LIMIT'
    for it in range(args.iterations):
        row=dict(iteration=it,weighted_residual=error,max_absolute_by_component=components,map_evaluations=call_count)
        history.append(row);write(folder/'correction.json',dict(status='RUNNING',history=history))
        # Preserve the current complete shooting state before a potentially
        # long native-map replay. A new correction run may restart here using
        # the same numerical preconditioner, with its provenance retained.
        checkpoint=folder/'iteration_state';checkpoint.mkdir(exist_ok=True)
        m.step_index=base_index;coords.unpack(x,m);m.save(checkpoint);write(checkpoint/'config.json',config)
        print('cycle correction',row,flush=True)
        if error<args.tolerance and components['F']<1e-8 and max(components[k] for k in ('M','qa','ia','qg','ig'))<1e-6:
            status='FULL_MAP_PERIODIC_ORBIT_CORRECTED';break
        delta=residual.copy()
        if args.arnoldi_preconditioner:
            if Mfactor:delta[coords.slices['M']]*=1+Mfactor
            rhs=[]
            for path in basis_files[:-1]:
                q=cp.asarray(np.load(path,mmap_mode='r'));rhs.append(float(cp.dot(q,delta).get()));del q
            coef=inv_small@np.asarray(rhs);c=H@coef
            for i,(path,a) in enumerate(zip(basis_files,c)):
                q=cp.asarray(np.load(path,mmap_mode='r'));delta+=a*q
                if Mfactor:delta[coords.slices['M']]+=Mfactor*(a-(coef[i] if i<dimension else 0.))*q[coords.slices['M']]
                del q
        else:
            rhs=np.array([float(cp.dot(v,residual).get()) for v in modes]);c=pinv@rhs
            for v,a,b in zip(modes,c,inverse):delta+=v*(a*b)
        # Backtracking protects against switches of the minmod/threshold
        # branches. A failed correction is recorded, never called an orbit.
        accepted=False
        for alpha in (1.,.5,.25,.125):
            trial=x+alpha*delta
            rnew,enew,cnew,_=evaluate(trial)
            if np.isfinite(enew) and enew<error:
                x=trial;residual,error,components=rnew,enew,cnew;accepted=True
                row['accepted_step']=alpha;break
        if not accepted:status='NO_RESIDUAL_DESCENT';break
    m.step_index=base_index;coords.unpack(x,m);m.save(folder)
    write(folder/'correction.json',dict(status=status,history=history,final_weighted_residual=error,
        final_max_absolute_by_component=components,map_evaluations=call_count,wall_s=time.time()-started))
    if status=='FULL_MAP_PERIODIC_ORBIT_CORRECTED':
        residual,error,components,trace=evaluate(x,record=True)
        np.savez_compressed(folder/'trajectory.npz',rate_0p1ms=trace[0],field_1ms=trace[1],count_e=count)
        write(folder/'cycle_observables.json',dict(mean_rate_hz=trace[0].mean(0),maximum_rate_hz=trace[0].max(0),
            minimum_rate_hz=trace[0].min(0),period_ms=steps*DT,
            final_independent_replay_residual=error,final_max_absolute_by_component=components))
    write(folder/'status.json',dict(status=status,final_weighted_residual=error,diagnostics=m.diagnostics(),
        stability='RECOMPUTE_AT_CORRECTED_ORBIT',physical_SNN_bifurcation='NOT_CLASSIFIED',wall_s=time.time()-started))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--spectrum',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--iterations',type=int,default=6);ap.add_argument('--tolerance',type=float,default=1e-8)
    ap.add_argument('--minimum-denominator',type=float,default=1e-4);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--arnoldi-preconditioner',action='store_true',help='Use the saved complete Arnoldi projection instead of a few real eigenvectors')
    ap.add_argument('--M-block-preconditioner',action='store_true',help='Also precondition slow M outside the Arnoldi subspace using its exact native decay')
    run(ap.parse_args())

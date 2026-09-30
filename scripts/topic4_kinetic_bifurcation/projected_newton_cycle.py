"""Test a Newton direction using only already measured full-map products.

The correction lies in the saved Krylov space. Therefore (I-J)delta is known
from the full native tangent products, with no unmeasured identity complement.
Small nonlinear trials audit this derivative over an entire burst period.
This is a numerical solver diagnostic; it does not certify an orbit.
"""
from cycle_monodromy import *


def run(args):
    spec=read(args.spectrum/'config.json');source=Path(spec['source']);cfg=read(source/'config.json')
    endcfg=read(args.endpoint/'config.json')
    assert endcfg['resumed_from']==str(source.resolve())
    folder=OUT/'projected_newton_steps'/args.label;folder.mkdir(parents=True,exist_ok=False)
    H=np.load(args.spectrum/'arnoldi.npz')['H'];rows,dim=H.shape
    assert rows==dim+1
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);coords=StateCoordinates(m);start=capture(m);start_index=m.step_index;x=coords.pack(m)
    m.restore(args.endpoint);assert m.step_index-start_index==spec['steps']
    residual=coords.pack(m)-x;initial_norm=float(cp.linalg.norm(residual).get())
    basis_files=[Path(spec['krylov_storage'])/f'q{i:03d}.npy' for i in range(rows)]
    projections=[]
    for p in basis_files:
        q=cp.asarray(np.load(p,mmap_mode='r'));projections.append(float(cp.dot(q,residual).get()));del q
    projections=np.asarray(projections);B=np.eye(rows,dim)-H
    c,_,rank,singular=np.linalg.lstsq(B,projections,rcond=1e-12)
    direction=cp.zeros_like(x);action=cp.zeros_like(x);bc=B@c
    for i,p in enumerate(basis_files):
        q=cp.asarray(np.load(p,mmap_mode='r'))
        if i<dim:direction+=c[i]*q
        action+=bc[i]*q;del q
    predicted=residual-action
    quality=dict(initial_residual=initial_norm,dimension=dim,rank=int(rank),singular_values=singular,
        unrepresented_residual_norm=np.sqrt(max(0.,initial_norm**2-projections@projections)),
        predicted_full_linear_residual=float(cp.linalg.norm(predicted).get()),
        relative_linear_residual=float(cp.linalg.norm(predicted).get())/initial_norm,
        step_norm=float(cp.linalg.norm(direction).get()),measured_action_norm=float(cp.linalg.norm(action).get()),
        method='Least squares with full (I-J)Q=Qplus(Ibar-H), using only measured physical tangent products')
    write(folder/'direction.json',quality)
    np.savez_compressed(folder/'linear_step.npz',coefficients=c,residual_projection=projections,H=H)
    write(folder/'config.json',dict(source=str(source),endpoint=str(args.endpoint),spectrum=str(args.spectrum),
        D=cfg['D'],steps=spec['steps'],period_ms=spec['steps']*DT,small_trials=args.small_trials,
        fresh_baseline=args.fresh_baseline,symmetric_audit=args.symmetric_audit,
        cached_baseline='Same complete source and original FP64 endpoint; step difference checked exactly',
        scope='Whole-period directional derivative audit and projected Newton trial; no orbit acceptance'))
    started=time.time();last=started;trials=[]
    def reset(state):
        for k,v in start.items():
            if k not in ('ordered_history','step_index'):getattr(m,k)[:]=v
        m.step_index=start_index;coords.unpack(state,m)
        for k in ('masserror','maxneg','minflux'):getattr(m,k).fill(0.)
        for k in ('positive_emitted','negative_emitted','maximum_lower_mass'):getattr(m,k).fill(0.)
        m.minimum_drive_bound.fill(cp.inf);m.diagnostic_start_step=start_index
    def evaluate(alpha,kind):
        nonlocal last
        trial=x+alpha*direction;reset(trial)
        for step in range(spec['steps']):
            m.advance_step()
            if time.time()-last>20:
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),kind=kind,alpha=alpha,
                    elapsed_ms=(step+1)*DT,wall_s=time.time()-started,completed_trials=trials))
                print('projected Newton',args.label,kind,alpha,(step+1)*DT,flush=True);last=time.time()
        r=coords.pack(m)-trial;norm=float(cp.linalg.norm(r).get())
        discrepancy=r-(residual-alpha*action)
        relative=(float(cp.linalg.norm(discrepancy).get())/(abs(alpha)*max(quality['measured_action_norm'],1e-30))) if alpha else None
        qa=m.diagnostics();valid=qa['finite'] and qa['maximum_mass_error']<1e-8 and qa['maximum_negative_voltage_probability']<5e-4 and qa['minimum_step_spike_probability']>-1e-6
        row=dict(kind=kind,alpha=alpha,weighted_residual=norm,
            relative_directional_linearization_error=relative,predicted_residual=float(cp.linalg.norm(residual-alpha*action).get()),
            full_map_numerical_gate=bool(valid),diagnostics=qa)
        trials.append(row);write(folder/'trials.json',dict(status='RUNNING',trials=trials))
        print('projected Newton trial',row,flush=True)
        return trial,row,r
    if args.fresh_baseline:
        _,row,new_residual=evaluate(0.,'roundtrip_baseline')
        baseline=dict(weighted_difference_from_cached_endpoint=float(cp.linalg.norm(new_residual-residual).get()),
            cached_residual=initial_norm,fresh_residual=row['weighted_residual'],
            scope='Repeat after the same coordinate pack/unpack used by shooting; measures its numerical baseline floor')
        write(folder/'baseline_repeat.json',baseline)
        residual=new_residual;initial_norm=row['weighted_residual']
        if not row['full_map_numerical_gate']:
            write(folder/'result.json',dict(status='BASELINE_NUMERICAL_GATE_FAILED',trials=trials));return
    audit_pass=False
    for alpha in args.small_trials:
        _,row,rplus=evaluate(alpha,'directional_audit')
        if args.symmetric_audit:
            _,minus,rminus=evaluate(-alpha,'directional_audit_minus')
            residual_fd=(rplus-rminus)/(2*alpha)
            error=residual_fd+action
            relative=float((cp.linalg.norm(error)/cp.linalg.norm(action)).get())
            relative_map=float((cp.linalg.norm(error)/cp.linalg.norm(direction-action)).get())
            row=dict(kind='symmetric_directional_audit',alpha=alpha,
                relative_directional_linearization_error=relative,
                relative_full_map_JVP_error=relative_map,
                residual_derivative_condition_ratio=float((cp.linalg.norm(direction-action)/cp.linalg.norm(action)).get()),
                full_map_numerical_gate=bool(row['full_map_numerical_gate'] and minus['full_map_numerical_gate']))
            trials.append(row);write(folder/'trials.json',dict(status='RUNNING',trials=trials))
            del rminus,residual_fd,error
        if row['full_map_numerical_gate'] and row['relative_directional_linearization_error']<.01:
            audit_pass=True;break
    status='DIRECTIONAL_AUDIT_FAILED';accepted_state=None
    if audit_pass and args.try_step:
        status='NO_PROJECTED_DESCENT'
        for alpha in (1.,.5,.25,.125,.0625):
            trial,row,_=evaluate(alpha,'Newton_step')
            if row['full_map_numerical_gate'] and row['weighted_residual']<initial_norm:
                reset(trial);accepted_state=folder/'accepted_state';accepted_state.mkdir()
                m.save(accepted_state)
                # This checkpoint is the accepted shooting initial state, not
                # the endpoint of its return. Replays must retain that meaning.
                newcfg=dict(cfg,initial_ms=start_index*DT,precision='FP64',source=str(source),
                    numerical_initial_state='Accepted projected Newton shooting update',
                    scientific_acceptance='Uncorrected periodic-point candidate')
                write(accepted_state/'config.json',newcfg)
                status='PROJECTED_NEWTON_STEP_ACCEPTED';break
    elif audit_pass:status='DIRECTIONAL_AUDIT_PASSED'
    write(folder/'result.json',dict(status=status,direction=quality,trials=trials,wall_s=time.time()-started,
        accepted_state=str(accepted_state) if accepted_state else None,orbit_acceptance='NOT_ESTABLISHED'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spectrum',type=Path,required=True)
    ap.add_argument('--endpoint',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--device',type=int,default=1);ap.add_argument('--small-trials',type=float,nargs='+',default=[.001,.0001])
    ap.add_argument('--fresh-baseline',action='store_true');ap.add_argument('--symmetric-audit',action='store_true')
    ap.add_argument('--try-step',action='store_true');run(ap.parse_args())

"""Anderson correction of a transverse generalized return.

Physical evolutions use only original FP64 native steps. Polynomial return
interpolation and Anderson combinations are numerical root-finding operations,
not a change to the model or evidence of an integer-period orbit. The final
return is compared with a second interpolation order before acceptance.
"""
from generalized_return_audit import *


def run(a):
    cfg=read(a.source/'config.json');folder=OUT/'generalized_corrections'/a.label
    folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(a.source);coords=StateCoordinates(m);initial=capture(m);initial_step=m.step_index
    section_x=coords.pack(m);x=section_x.copy()
    m.advance_step();y=coords.pack(m);m.advance_step();z=coords.pack(m)
    normal=(-3*x+4*y-z)/(2*DT);normal/=cp.linalg.norm(normal);del y,z
    if a.restart_from:
        previous=read(a.restart_from/'config.json')
        assert Path(previous['source']).resolve()==a.source.resolve(), 'Preserve the original metric and section'
        normal=cp.asarray(np.load(a.restart_from/'section_normal.npy'))
        m.restore(a.restart_from/'best_state');assert m.step_index==initial_step
        x=coords.pack(m)
    n=round(a.period/DT);assert abs(n*DT-a.period)<1e-9
    comparison_order=3 if a.order==5 else 5
    config=dict(source=str(a.source.resolve()),D=cfg['D'],integer_steps=n,initial_integer_steps=n,
        interpolation_order=a.order,comparison_order=comparison_order,tolerance=a.tolerance,maximum_iterations=a.iterations,
        restart_from=str(a.restart_from.resolve()) if a.restart_from else None,return_search_radius_ms=a.search_radius,
        safeguard_initial_pdf=a.safeguard_initial_pdf,
        backtrack_failed_native_replay=a.backtrack_invalid,
        periodic_M_balance_preconditioner=a.balance_M,
        phase_section='Fixed hyperplane through original source, normal to its second-order native-step chord in fixed StateCoordinates metric',
        algorithm='Anderson3 on generalized transverse return; all physical replays FP64, Z fixed, M dynamic',
        object='Approximate fixed point of a generalized return on a native-map invariant curve; not an exact integer-period point')
    write(folder/'config.json',config)
    np.save(folder/'section_normal.npy',cp.asnumpy(normal))
    started=time.time();last=started;records=[];dx=[];df=[];previous_x=None;previous_f=None;best=float('inf')
    accepted=False;last_comparison=None;consecutive_invalid=0
    proposals=[]
    def initial_physical_qa(v):
        f=v[coords.slices['F']].reshape(coords.shapes['F'])/coords.scales['F']
        physical=cp.einsum('k,gkv->gv',m.mass,f)
        slow=v[coords.slices['M']].reshape(coords.shapes['M'])/coords.scales['M']
        hist=v[coords.slices['history']].reshape(coords.shapes['history'])/coords.scales['history']
        return dict(maximum_negative_probability=float(cp.maximum(-physical,0.).sum(1).max().get()),
            maximum_mass_error=float(cp.max(abs(physical.sum(1)-1.)).get()),
            minimum_M=float(slow.min().get()),minimum_delay_probability=float(hist.min().get()))
    def reset(v):
        for k,value in initial.items():
            if k not in ('ordered_history','step_index'):getattr(m,k)[:]=value
        m.step_index=initial_step;coords.unpack(v,m)
        for k in ('masserror','maxneg','minflux','positive_emitted','negative_emitted','maximum_lower_mass'):
            getattr(m,k).fill(0.)
        m.minimum_drive_bound.fill(cp.inf);m.diagnostic_start_step=initial_step
    for iteration in range(a.iterations):
        reset(x);delta=[];values=[];radius=round(a.search_radius/DT)
        lower=max(1,n-radius);upper=n+radius;anchor=None if radius else n
        previous_delta=None;previous_value=None;search_samples=[]
        for step in range(1,upper+6):
            m.advance_step()
            if step>=lower:
                d=coords.pack(m)-section_x;value=float(cp.dot(normal,d).get())
                search_samples.append(dict(step=step,time_ms=step*DT,section_value=value))
                if anchor is None:
                    if previous_value is not None and previous_value<=0<value:
                        anchor=step-1;delta=[previous_delta,d];values=[previous_value,value]
                    previous_delta=d;previous_value=value
                elif step>=anchor:
                    delta.append(d);values.append(value)
                if anchor is not None and len(delta)==6:break
            if time.time()-last>20:
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),iteration=iteration,
                    elapsed_ms=step*DT,wall_s=time.time()-started,latest=records[-1] if records else None))
                print(a.label,iteration,step*DT,flush=True);last=time.time()
        if anchor is None or len(delta)!=6:
            failure=dict(iteration=iteration,expected_steps=n,
                search_radius_ms=a.search_radius,samples=search_samples,diagnostics=m.diagnostics(),
                interpretation='No accepted forward crossing in the declared window; this is not branch disappearance')
            write(folder/'return_window_failure.json',failure)
            endpoint=folder/f'failed_window_endpoint_{iteration:02d}';endpoint.mkdir(exist_ok=True);m.save(endpoint)
            write(endpoint/'window_samples.json',failure)
            write(endpoint/'config.json',dict(cfg,source=str(a.source.resolve()),initial_ms=m.step_index*DT,
                numerical_initial_state='Last original state of a failed return-window search; not a corrected orbit'))
            if a.backtrack_invalid and previous_x is not None and consecutive_invalid<3:
                consecutive_invalid+=1;x=previous_x+.5*(x-previous_x);dx=[];df=[]
                write(endpoint/'backtrack.json',dict(fraction_toward_last_accepted_state=.5,
                    consecutive_invalid=consecutive_invalid,next_iteration=iteration+1,
                    acceptance='No root accepted from a failed window; next initial guess is a smaller numerical update'))
                continue
            status='RETURN_OUTSIDE_DECLARED_WINDOW';break
        n=anchor;config['integer_steps']=n;write(folder/'config.json',config)
        alpha=crossing(values,a.order);coeff=coefficients(alpha,a.order)
        gx=section_x+sum(float(c)*d for c,d in zip(coeff,delta));f=gx-x
        norm=float(cp.linalg.norm(f).get());qa=m.diagnostics()
        valid=qa['finite'] and qa['maximum_mass_error']<1e-8 and qa['maximum_negative_voltage_probability']<5e-4 and qa['minimum_step_spike_probability']>-1e-6
        alpha3=crossing(values,comparison_order);g3=section_x+sum(float(c)*d for c,d in zip(coefficients(alpha3,comparison_order),delta))
        comparison_delta=g3-gx
        comparison=float(cp.linalg.norm(comparison_delta).get());last_comparison=comparison
        row=dict(iteration=iteration,weighted_return_residual=norm,phase_fraction=alpha,
            effective_return_time_ms=(n+alpha)*DT,integer_steps=n,comparison_order_return_difference=comparison,
            comparison_order_phase_fraction=alpha3,
            comparison_order_return_time_ms=(n+alpha3)*DT,
            comparison_order_difference_by_coordinate_block={k:float(cp.linalg.norm(comparison_delta[s]).get()) for k,s in coords.slices.items()},
            return_window_section_values=values,
            native_residual_by_coordinate_block={k:float(cp.linalg.norm(f[s]).get()) for k,s in coords.slices.items()},
            section_residual=float(cp.dot(normal,x-section_x).get()),full_map_numerical_gate=bool(valid),diagnostics=qa)
        del comparison_delta
        records.append(row);write(folder/'iterations.json',dict(status='RUNNING',history=records))
        print('generalized correction',row,flush=True)
        if norm<best and valid:
            best=norm;reset(x);saved=folder/'best_state';saved.mkdir(exist_ok=True);m.save(saved)
            write(saved/'config.json',dict(cfg,initial_ms=initial_step*DT,precision='FP64',source=str(a.source.resolve()),
                numerical_initial_state='Shooting state for generalized return; effective period need not be an integer number of native steps',
                scientific_acceptance='UNCLASSIFIED invariant-curve candidate'))
        if not valid:
            reset(x);rejected=folder/f'rejected_initial_state_{iteration:02d}';rejected.mkdir(exist_ok=True);m.save(rejected)
            write(rejected/'config.json',dict(cfg,initial_ms=initial_step*DT,source=str(a.source.resolve()),
                numerical_initial_state='Rejected shooting iterate; its subsequent native replay failed the probability gate'))
            write(rejected/'initial_qa.json',initial_physical_qa(x))
            if a.backtrack_invalid and previous_x is not None and consecutive_invalid<3:
                # A positive marginal at time zero need not make a high-order
                # PDF extrapolation admissible through its entire replay.
                # Backtrack the numerical proposal against the last accepted
                # initial state and recheck the unchanged physical trajectory.
                consecutive_invalid+=1;x=previous_x+.5*(x-previous_x);dx=[];df=[]
                write(rejected/'backtrack.json',dict(fraction_toward_last_accepted_state=.5,
                    consecutive_invalid=consecutive_invalid,next_iteration=iteration+1,
                    acceptance='Rejected replay is excluded; the next native replay must pass the original gate'))
                del delta,gx,g3
                continue
            status='NATIVE_MAP_NUMERICAL_GATE_FAILED';break
        consecutive_invalid=0
        if norm<a.tolerance:
            accepted=True;status='GENERALIZED_RETURN_CORRECTED';break
        if a.balance_M:
            # M is linear in its carried value for a specified spike history:
            # G_M = a_M M + weighted spike input. This preconditions the root
            # iteration toward its periodic balance without freezing M during
            # any physical replay or changing the roots of G(x)-x.
            bare_M=float(coefficients(alpha,a.order)@(1-DT/1000.)**(n+np.arange(a.order+1)))
            assert 0<bare_M<1
            f[coords.slices['M']]/=1-bare_M
        if previous_x is not None:
            dx.append(x-previous_x);df.append(f-previous_f)
            if len(dx)>3:dx.pop(0);df.pop(0)
        proposal=x+f if a.balance_M else gx.copy()
        if df:
            gram=np.array([[float(cp.dot(aa,bb).get()) for bb in df] for aa in df])
            rhs=np.array([float(cp.dot(aa,f).get()) for aa in df])
            gamma=np.linalg.solve(gram+np.eye(len(df))*max(float(np.trace(gram)),1e-300)*1e-12,rhs)
            if np.isfinite(gamma).all() and np.sum(abs(gamma))<100:
                for c,s,t in zip(gamma,dx,df):proposal-=float(c)*(s+t)
        proposal-=normal*cp.dot(normal,proposal-section_x)
        if a.safeguard_initial_pdf:
            baseline_qa=initial_physical_qa(x)
            negative_limit=max(1e-4,baseline_qa['maximum_negative_probability']*1.01)
            assert negative_limit<5e-4, 'Restart from a numerically admissible density'
            accepted_proposal=None;trials=[]
            for scale in 2.**-np.arange(17):
                candidate=x+float(scale)*(proposal-x);candidate_qa=initial_physical_qa(candidate)
                okay=(candidate_qa['maximum_negative_probability']<=negative_limit and
                    candidate_qa['maximum_mass_error']<1e-8 and candidate_qa['minimum_M']>=-1e-8 and
                    candidate_qa['minimum_delay_probability']>=-1e-6)
                trials.append(dict(scale=float(scale),pass_check=bool(okay),**candidate_qa))
                if okay:accepted_proposal=candidate;break
            proposals.append(dict(iteration=iteration,negative_probability_limit=negative_limit,trials=trials))
            write(folder/'proposal_safeguards.json',proposals)
            if accepted_proposal is None:status='NO_ADMISSIBLE_ANDERSON_PROPOSAL';break
            proposal=accepted_proposal
        previous_x=x;previous_f=f;x=proposal
        del delta,gx,g3
    else:status='GENERALIZED_RETURN_CORRECTION_INCOMPLETE'
    write(folder/'result.json',dict(status=status,converged=accepted,best_weighted_return_residual=best,
        interpolation_order=a.order,comparison_order=comparison_order,comparison_order_difference=last_comparison,
        interpolation_acceptance='NOT_ESTABLISHED; compare corrected solutions and normal spectra at multiple orders',
        history=records,wall_s=time.time()-started,stability='NOT_COMPUTED',bifurcation_type='NOT_CLASSIFIED',
        original_discrete_periodicity='NOT_CLAIMED'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--period',type=float,required=True);ap.add_argument('--order',type=int,choices=[3,5],default=5)
    ap.add_argument('--iterations',type=int,default=10);ap.add_argument('--tolerance',type=float,default=1e-8)
    ap.add_argument('--restart-from',type=Path,help='Warm start from a previous correction with the exact same original metric and section')
    ap.add_argument('--search-radius',type=float,default=0.,help='Bounded ms window about expected return; use first forward section crossing, keeping native time steps unchanged')
    ap.add_argument('--safeguard-initial-pdf',action='store_true',help='Backtrack only the numerical Anderson proposal to retain admissible initial probability, M and delays; native map and full-replay gate are unchanged')
    ap.add_argument('--backtrack-invalid',action='store_true',help='Retry at most three half proposals after a full native replay fails its probability gate; every retry counts toward the iteration budget')
    ap.add_argument('--balance-M',action='store_true',help='Precondition only the numerical return-root iteration using exact exponential M balance; M stays dynamic in the unchanged physical replays')
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())

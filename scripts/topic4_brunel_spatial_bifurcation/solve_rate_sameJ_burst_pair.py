"""Compare the two nearby burst families at exactly the same J, not a projection crossing."""
from complete_rate_positive_stability import *
import subprocess

ACTIVE_WORKER = None


def incremental_correction(o, s, z, target, maximum_step, minimum_step, folder, name, status):
    """Bound fixed-J corrections; a failed path does not prove branch absence."""
    from audit_rate_filter_states import filter_state_minima
    assert 0 < minimum_step <= maximum_step
    weights=s.geo['group_size']/s.geo['group_size'].sum()
    r,T,current=z['r'].copy(),float(z['T']),float(z['J'])
    assert current != target, 'Incremental correction requires a distinct target J'
    step=maximum_step;accepted=[];attempts=[]
    evidence=folder/(name+'_correction_steps.json')
    while current != target:
        trial=target if abs(target-current)<=step else current+np.sign(target-current)*step
        status('INCREMENTAL_SAME_J_CORRECTION',from_J=current,target_J=trial,
               requested_J=target,accepted_steps=len(accepted),maximum_J_step=step)
        candidate,period,actual,error,history=o.solve(r,T,trial,tol=2e-11,maxiter=12)
        record=dict(from_J=current,target_J=trial,residual_hz=error,newton_history=history)
        passed=error<2e-11 and abs(actual-trial)<1e-14 and np.isfinite(period) and period>0
        if passed:
            distance,_=distances((r*1000)[:,None,:],candidate*1000,weights)
            amplitude=np.sqrt(np.mean(np.sum(((r-r.mean(0))*1000)**2*weights,axis=1)))
            drift=float(distance[0]/amplitude)
            period_change=abs(period/T-1)
            physical=filter_state_minima(s,candidate,period)
            record.update(relative_waveform_change=drift,relative_period_change=period_change,
                          filter_state_check=physical)
            passed=drift<.05 and period_change<.01 and physical['positive']
        record['accepted']=bool(passed);attempts.append(record)
        if passed:
            orbit=save_orbit(s,candidate,period,trial,error,history,
                             f'{name}_increment{len(accepted):03d}_N{len(candidate)}')
            accepted.append(dict(**record,orbit=str(orbit)))
            r,T,current=candidate,period,trial
            step=min(maximum_step,step*1.5)
        else:
            step*=.5
        payload=dict(status='TARGET_REACHED' if current==target else 'CORRECTION_IN_PROGRESS',
            seed_J_EE_core=float(z['J']),target_J_EE_core=target,reached_J_EE_core=current,
            accepted=accepted,attempts=attempts,
            scope='Finite small-parameter-step Newton corrections with phase-aligned jump guards and positive filters. The final orbit still requires an independent off-grid check. This path does not certify interval stability or exclude folds or other branches.')
        if not passed and step<minimum_step:
            payload['status']='FIXED_J_PATH_NOT_REACHED'
            write(evidence,payload)
            raise RuntimeError(f'Fixed-J corrections cannot reach target at the allowed step; see {evidence}. No branch-absence inference.')
        if current!=target and len(accepted)>=64:
            payload['status']='FIXED_J_PATH_STEP_LIMIT'
            write(evidence,payload)
            raise RuntimeError(f'Bounded correction path reached 64 accepted steps; see {evidence}.')
        write(evidence,payload)
    return r,T,actual,error,history,str(evidence)


def main():
    global ACTIVE_WORKER
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--min-free-gib',type=float,default=3.5)
    p.add_argument('--source-index',type=int,help='A point in the physically checked B-leading prefix')
    p.add_argument('--output-name',default='sameJ_pair')
    p.add_argument('--after-pids',type=int,nargs='*',default=[])
    p.add_argument('--maximum-J-step',type=float,
                   help='Use bounded small fixed-J corrections instead of one large parameter jump')
    p.add_argument('--minimum-J-step',type=float,default=1e-5)
    a=p.parse_args();assert Path(a.output_name).name==a.output_name
    if a.source_index is not None:assert a.output_name!='sameJ_pair', 'Preserve the original comparison'
    folder=DEST/'Bleading_extension';worker=folder/(a.output_name+'_worker.json')
    ACTIVE_WORKER=worker
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw));print(state,kw,flush=True)
    def identity(pid):
        try:return Path(f'/proc/{pid}/cmdline').read_bytes()
        except (FileNotFoundError,ProcessLookupError):return None
    dependencies={pid:v for pid in a.after_pids if (v:=identity(pid)) is not None}
    while dependencies:
        for pid,value in list(dependencies.items()):
            if identity(pid)!=value:dependencies.pop(pid)
        if dependencies:status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    q=read(folder/'nearest_target_refinement.json');assert q['status']=='COMPLETE'
    target=next(v for v in q['rows'] if v['family']=='single')
    assert target['same_branch_refinement_pass'] and target['resolution']['filter_state_check']['positive']
    source=Path(target['original_pair']['first_orbit']);seed=Path(target['corrected_target'])
    accepted=read(PERIODIC_OUT/'arcBleadingConnection_20260920_accuracy.json')
    if a.source_index is not None:
        source=PERIODIC_OUT/'orbits'/f'arcBleadingConnection_20260920_{a.source_index:04d}_N4096.npz'
    assert str(source) in accepted['included_orbits']
    assert read(PERIODIC_OUT/'bounded_harmonic_index_monodromy_check.json')['status']=='PASS'
    source_z=np.load(source);z=np.load(seed);J=float(source_z['J']);N=len(z['r'])
    source_index=16 if a.source_index is None else a.source_index
    name=f'Aleading_sameJ_Bleading{source_index:04d}_N{N}';output=PERIODIC_OUT/'orbits'/(name+'.npz')
    if a.maximum_J_step is not None:
        assert 0<a.minimum_J_step<=a.maximum_J_step
        name+='_'+a.output_name;output=PERIODIC_OUT/'orbits'/(name+'.npz')
    correction_source=None
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    if not output.exists():
        required=max(a.min_free_gib,3. if N<=2048 else 6.)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=required*1024:break
            status('WAITING_RESOURCE',free_mib=free,required_gib=required);time.sleep(30)
        status('CORRECTING_AT_EXACT_SAME_J',seed=str(seed),J_EE_core=J,N=N)
        o=Periodic(s,N,a.device,harmonic_capacity=64)
        o.low_memory=True;o.host_krylov=True;o.normalize_linear_rhs=True;o.linear_target_aware=True
        o.derivative_chunk_size=64
        if a.maximum_J_step is None:
            r,T,actual_J,err,history=o.solve(z['r'],float(z['T']),J,tol=2e-11,maxiter=24)
        else:
            r,T,actual_J,err,history,correction_source=incremental_correction(
                o,s,z,J,a.maximum_J_step,a.minimum_J_step,folder,a.output_name,status)
        assert err<2e-11 and abs(actual_J-J)<1e-14
        output=save_orbit(s,r,T,J,err,history,name)
        o.cache=None;del o;release(a.device)
    elif a.maximum_J_step is not None:
        correction_source=str(folder/(a.output_name+'_correction_steps.json'))
        assert read(correction_source)['status']=='TARGET_REACHED'
    actual,check=prepare(output,a.device,max_N=4096,host_krylov=True,
        check_filter_states=True,harmonic_chunk_size=32,stream_harmonics=True)
    assert check['status']=='RESOLUTION_CHECKED' and check['maximum_group_defect_Hz']<.001
    after=np.load(actual);x,y=[resample(v['r']*1000,4096,axis=0) for v in [source_z,after]]
    old=resample(z['r']*1000,4096,axis=0)
    amplitude=lambda r:float(np.sqrt(np.mean(np.sum((r-r.mean(0))**2*weights,axis=1))))
    change,_=distances(old[:,None,:],y,weights)
    branch_change=float(change[0]/amplitude(old));period_change=abs(float(after['T']/z['T'])-1)
    if correction_source is None:
        assert branch_change<.05 and period_change<.01,(branch_change,period_change)
    else:
        # The same guards were applied to every small parameter step.
        # Check temporal refinement against the final pre-refinement orbit;
        # total change along a resolved parameter path need not be small.
        initial=np.load(output)
        terminal=resample(initial['r']*1000,4096,axis=0)
        final_drift,_=distances(terminal[:,None,:],y,weights)
        assert float(final_drift[0]/amplitude(terminal))<.02
        assert abs(float(after['T']/initial['T'])-1)<.001
    d,phase=distances(x[:,None,:],y,weights)
    frequencies=np.fft.fftfreq(len(y))*len(y)
    aligned=np.fft.ifft(np.fft.fft(y,axis=0)*np.exp(-2j*np.pi*frequencies*phase[0])[:,None],axis=0).real
    squared=np.mean((x-aligned)**2,axis=0)*weights
    energy={}
    for label,mask in [('Core_A_E',s.E&(s.geo['group_region']==0)),
                       ('Core_B_E',s.E&(s.geo['group_region']==1)),
                       ('Surround_E',s.E&(s.geo['group_region']==2)),('All_I',~s.E)]:
        energy[label]=float(squared[mask].sum()/squared.sum())
    rows=[]
    for label,path,zz,r in [('Bleading',source,source_z,x),('Aleading',actual,after,y)]:
        rg=np.array([s.regional_rates(v/1000) for v in r]);T=float(zz['T'])
        peak_times=np.argmax(rg,axis=0)*T/len(r)
        rows.append(dict(family=label,orbit=str(path),J_EE_core=float(zz['J']),T_ms=T,
            regional_mean_Hz=rg.mean(0),regional_peak_Hz=rg.max(0),
            B_minus_A_peak_lag_ms=float((peak_times[1]-peak_times[0]+T/2)%T-T/2)))
    result=dict(status='TWO_PHYSICAL_SAME_J_PERIODIC_SOLUTIONS',rows=rows,
        seed=str(seed),seed_J_EE_core=float(z['J']),relative_seed_waveform_change=branch_change,
        relative_seed_period_change=period_change,corrected_profile_check=check,
        source_profile_check=next(v for v in accepted['checks'] if Path(v['orbit']).resolve()==source.resolve()),
        full_group_RMS_difference_Hz=float(d[0]),relative_waveform_difference=float(d[0]/max(amplitude(x),amplitude(y))),
        common_phase_cycles=float(phase[0]),squared_distance_contributions=energy,
        energy_definition='Neuron-weighted squared waveform difference at one common optimal phase; Core/Surround groups here are E only. All I groups are aggregated without a spatial-region claim.',
        stability='NOT_CLASSIFIED_FOR_THIS_EXACT_PAIR',global_connection_confirmed=False,
        scope='Two phase-fixed, physically checked periodic solutions at identical J. Distinct solutions do not establish coexistence of stable attractors, an intervening bifurcation or exclusion of a connection elsewhere.')
    result['source_continuation_index']=source_index
    result['parameter_correction_steps']=correction_source
    write(folder/(a.output_name+'.json'),result);status('SAME_J_PAIR_COMPLETE',result=result)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if ACTIVE_WORKER is not None:
            previous=read(ACTIVE_WORKER) if ACTIVE_WORKER.exists() else {}
            write(ACTIVE_WORKER,{**previous,'status':'COMPUTATION_FAILED',
                'error':repr(exc),'timestamp':time.time()})
        raise

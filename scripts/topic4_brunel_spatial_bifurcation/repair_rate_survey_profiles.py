"""Repair selected survey profiles before reclassifying full-DDE stability.

The frozen survey indices do not change. Negative Fourier interpolation is
repaired by temporal refinement, never by clipping. A leading unstable mode
needs paired physical-time steps; a stable verdict needs filtered spectrum
coverage at both steps. All earlier verdicts are retained in attempt history.
"""
from rate_stability_coverage import *
from compare_rate_torus_periodic_targets import distances


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',type=int,nargs='+',required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--min-free-gib',type=float,default=10.)
    p.add_argument('--wait-pids',type=int,nargs='*',default=[])
    p.add_argument('--full-spectrum-indices',type=int,nargs='*',default=[])
    p.add_argument('--stream-harmonics',action='store_true',help='Exact frequency blocks for finer physical profiles')
    a=p.parse_args();plan=read(COVER/'plan.json')['rows'];rows=[]
    worker=COVER/('profile_repair_worker_'+'_'.join(map(str,a.indices))+'.json')
    def status(stage,**kw):
        write(worker,dict(status=stage,pid=os.getpid(),rows=rows,**kw));print(stage,kw,flush=True)
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.wait_pids
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:dependencies.pop(pid)
        if dependencies:
            status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    def memory(stage,index,N=0):
        # Larger orbit refinements use host Krylov storage but still need
        # room for Fourier/edge operators. Keep shared-GPU reserve explicit.
        required=max(a.min_free_gib,14. if index in [105,111,114,118] else 0.,
                     4+N/(1024 if a.stream_harmonics else 512) if N else 0.)
        import cupy as cp
        cp.cuda.Device(a.device).use();gc.collect()
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=1024*required:return
            status('WAITING_GPU_RESOURCE',index=index,next_stage=stage,free_mib=free,required_free_gib=required)
            time.sleep(30)
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    for index in a.indices:
        item=plan[index];tag=Path(item['orbit']).stem;dest=COVER/(tag+'.json')
        prior=read(dest) if dest.exists() else dict(status='PENDING',orbit=item['orbit'])
        if (prior.get('positive_profile_followup') and prior['status'] in ['UNSTABLE','NUMERICALLY_STABLE']
            and prior.get('resolution',{}).get('filter_state_check',{}).get('positive',False)):
            rows.append(dict(index=index,status=prior['status'],source=str(dest)));continue
        source=Path(prior.get('analyzed_orbit',item['orbit']))
        memory('ORBIT_REFINEMENT',index);status('ORBIT_REFINEMENT',index=index,orbit=str(source))
        try:
            # Resume an already persisted correction without repeating it.
            # Positivity/continuous residual are still rechecked below.
            cached=[read(f) for f in (PERIODIC_OUT/'orbits').glob(source.stem+'_accuracy_N*.json')]
            cached=[q for q in cached if q['status']=='CONVERGED' and
                    abs(q['J_EE_core']-item['J_EE_core'])<1e-12]
            start=Path(max(cached,key=lambda q:q['N'])['path']) if cached else source
            # These two weak cycles have enormous transverse growth. Reduce
            # their underlying temporal aliasing before comparing exponents.
            # The N512 solve at 61 and N1024 solve at 113 stagnated.
            # Correct on a finer temporal mesh directly; keep the same J,
            # equations, accuracy thresholds and orbit-identity check.
            initial_N=1024 if index==61 else 2048 if index==113 else 256 if index in [77,81] else 0
            actual,accuracy=prepare(start,a.device,max_N=8192,host_krylov=True,min_N=initial_N,
                check_filter_states=True,harmonic_chunk_size=64,before_mesh=lambda N,stage:memory(stage,index,N),
                stream_harmonics=a.stream_harmonics)
            detail=dict(index=index,original_orbit=item['orbit'],analyzed_orbit=str(actual),resolution=accuracy)
            write(COVER/(tag+'_positive_followup_progress.json'),detail)
            if accuracy['status']!='RESOLUTION_CHECKED':
                detail.update(status='UNRESOLVED',reason='Temporal profile has not passed residual and positivity checks')
                write(COVER/(tag+'_positive_followup.json'),detail);rows.append(detail);continue
            # Refinement at fixed J must preserve the local orbit identity.
            before=np.load(source);after=np.load(actual);N=max(len(before['r']),len(after['r']))
            x=resample(before['r']*1000,N,axis=0);y=resample(after['r']*1000,N,axis=0)
            d,phase=distances(x[:,None,:],y,weights)
            rms=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1)))
            drift=float(d[0]/rms);period_drift=abs(float(after['T'])/float(before['T'])-1)
            assert abs(float(after['J'])-float(before['J']))<1e-12
            assert drift<.02 and period_drift<.01,('Refinement changed orbit identity',drift,period_drift)
            detail.update(relative_waveform_refinement_change=drift,relative_period_change=period_drift)
            gc.collect()
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            full=(index in [103,104,105] or index in a.full_spectrum_indices)
            checks=[]
            for dt in ([.05,.025] if full else [.1,.05]):
                memory('FLOQUET',index);status('FLOQUET',index=index,dt_ms=dt,full_spectrum=full)
                label=tag+('_positive_full' if full else '_positive_leading')
                f=PERIODIC_OUT/'floquet'/f'{label}_dt{dt:g}.json'
                cached=read(f) if f.exists() else None
                q=cached if cached and Path(cached['orbit']).resolve()==actual.resolve() else compute(
                    actual,dt,8 if full else 1,a.device,full,24 if full else 8,output_label=label,
                    stream_harmonics=a.stream_harmonics)
                v=np.asarray(q['multipliers']);v=v[:,0]+1j*v[:,1] if v.ndim==2 else v.astype(complex)
                k=int(np.argmax(abs(v)))
                checks.append(dict(source=str(f),multiplier=v[k],relative_residual=q['residuals'][k]/max(1,abs(v[k])),
                    phase_defect=q['phase_tangent_relative_defect'],assessment=assess(q) if full else None))
                gc.collect();cp.get_default_memory_pool().free_all_blocks()
            change=float(abs(checks[-1]['multiplier']-checks[0]['multiplier'])/abs(checks[-1]['multiplier']))
            if full:
                verdict='UNRESOLVED'
                if all(q['assessment']['status']=='NUMERICALLY_STABLE' for q in checks):verdict='NUMERICALLY_STABLE'
                elif change<.01 and all(q['assessment']['status']=='UNSTABLE' for q in checks):verdict='UNSTABLE'
            else:
                verified=(change<.01 and all(abs(q['multiplier'])>1.1 and q['relative_residual']<1e-6 for q in checks))
                verdict='UNSTABLE' if verified else 'UNRESOLVED'
            detail.update(status=verdict,checks=checks,relative_multiplier_step_change=change,
                scope='Positive, refined full spatial periodic orbit at the same J. Paired physical-time DDE monodromy. A leading-mode instability verdict does not supply a complete spectrum or locate a bifurcation.')
            output=COVER/(tag+'_positive_followup.json');write(output,detail)
            history=COVER/'attempt_history'/f'{tag}_before_positive_reclassification_{time.time_ns()}.json';write(history,prior)
            current=dict(prior,status=verdict,resolution=accuracy,analyzed_orbit=str(actual),
                J_EE_core=item['J_EE_core'],T_ms=float(after['T']),memberships=item['memberships'],
                positive_profile_followup=str(output),positivity_review_required=False,
                classification_basis=detail['scope'],floquet_source=checks[-1]['source'])
            if verdict=='UNSTABLE':current.update(reliable_outside_count=1,margin=.1)
            write(dest,current);rows.append(dict(index=index,status=verdict,source=str(output)))
            print('REPAIRED SITE',rows[-1],flush=True)
        except Exception as exc:
            failure=dict(index=index,status='FOLLOWUP_FAILED',error=repr(exc),
                scope='No new stability verdict. The existing unresolved site is retained.')
            write(COVER/(tag+'_positive_followup_failure.json'),failure);rows.append(failure)
            if type(exc).__name__=='OutOfMemoryError' or isinstance(exc,OSError):
                status('STOPPED_WITH_ERROR',index=index,error=repr(exc));raise
            status('SITE_FAILED',index=index,error=repr(exc))
    status('BATCH_FINISHED')


if __name__=='__main__':main()

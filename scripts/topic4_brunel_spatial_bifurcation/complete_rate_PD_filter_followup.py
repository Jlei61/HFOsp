"""Resolve a PD parent at unchanged J and recheck its prolonged null mode.

This validates a previously located root on a finer temporal mesh. It does
not manufacture a new root or inherit the critical mode solely from a
positive corrected waveform. The fine antiperiodic residual and independent
full-state monodromy must both pass before the parent can be accepted.
"""
from rate_periodic_accuracy import *
from rate_antiperiodic import Antiperiodic
from compare_rate_torus_periodic_targets import distances
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('label',choices=['PD_double_low','PD_double_upper'])
    p.add_argument('--reuse-physical',action='store_true',help='Resume a persisted same-root physical and null-mode check')
    p.add_argument('--N',type=int,default=8192);p.add_argument('--device',type=int,default=0)
    p.add_argument('--min-free-gib',type=float,default=13.)
    p.add_argument('--wait-pids',type=int,nargs='*',default=[]);a=p.parse_args()
    label=a.label;s=RateField();scripts=Path(__file__).parent;N=a.N
    worker=PERIODIC_OUT/(label+'_filter_followup_worker.json')
    def record(status,**kw):
        write(worker,dict(status=status,pid=os.getpid(),N=N,**kw));print(status,kw,flush=True)
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.wait_pids if Path(f'/proc/{pid}/cmdline').exists()}
    def gate(stage):
        import cupy as cp
        cp.cuda.Device(a.device).use();gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
        while True:
            active=[]
            for pid,identity in deps.items():
                try:
                    if Path(f'/proc/{pid}/cmdline').read_bytes()==identity:active.append(pid)
                except FileNotFoundError:pass
            if active:record('WAITING_DEPENDENCIES',stage=stage,dependencies=active);time.sleep(30);continue
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            record('WAITING_GPU_RESOURCE',stage=stage,free_mib=free);time.sleep(30)
    root=max((read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')),key=lambda q:q['N'])
    original=Path(root['orbit']);z=np.load(original)
    assert root['N']<=N and root['antiperiodic_relative_residual']<1e-7
    source_mode=PERIODIC_OUT/f'{label}_mode_N{root["N"]}.npz'
    parent=PERIODIC_OUT/f'{label}_filter_parent_N{N}.json'
    profile=PERIODIC_OUT/'orbits'/f'{label}_filter_parent_N{N}.npz'
    suffix=''
    if profile.exists() and float(np.load(profile)['J'])!=root['J_EE_core']:
        suffix=f'_rootN{root["N"]}'
        profile=PERIODIC_OUT/'orbits'/f'{label}_filter_parent{suffix}_N{N}.npz'
    # Reuse a persisted fixed-J physical correction if it is exactly this
    # critical parameter; none of these arrays is called a newly found root.
    cached=[]
    for f in PERIODIC_OUT.glob('filter_state_repair_*.json'):
        for row in read(f).get('rows',[]):
            if row.get('label')==label and row.get('resolution',{}).get('status')=='RESOLUTION_CHECKED':
                candidate=Path(row['refined_orbit']);zz=np.load(candidate)
                if len(zz['r'])==N and float(zz['J'])==root['J_EE_core']:cached.append(candidate)
    if profile.exists():cached.insert(0,profile)
    if cached:profile=cached[0]
    else:
        gate('PARENT_CORRECTION');record('PARENT_CORRECTION',root_N=root['N'])
        o=Periodic(s,N,a.device);o.low_memory=True;o.stream_harmonics=True
        o.host_krylov=True;o.normalize_linear_rhs=True;o.linear_target_aware=True
        o.harmonic_chunk_size=64;o.derivative_chunk_size=64
        r,T,J,err,hist=o.solve(resample(z['r'],N,axis=0),float(z['T']),float(z['J']),tol=2e-11,maxiter=24)
        assert err<2e-11,err
        profile=save_orbit(s,r,T,J,err,hist,f'{label}_filter_parent{suffix}_N{N}')
        o.cache=None;del o;gc.collect()
    output=PERIODIC_OUT/(label+'_filter_state_followup.json')
    if a.reuse_physical:
        result=read(output);zz=np.load(profile);mode=Path(result['mode'])
        assert result['status'] in ['INDEPENDENT_MONODROMY_PENDING','FILTER_AND_CRITICAL_MODE_RECHECKED']
        assert Path(result['orbit']).resolve()==profile.resolve() and result['N']==N
        assert Path(result['source_root_orbit']).resolve()==original.resolve() and result['source_root_N']==root['N']
        assert result['J_EE_core']==float(zz['J']) and result['T_ms']==float(zz['T'])
        assert result['continuous_check']['filter_state_check']['positive'] and result['antiperiodic_relative_residual']<1e-7
        previous=np.load(mode)
        assert previous['u'].shape==(N,s.P) and float(previous['J'])==float(zz['J']) and float(previous['T'])==float(zz['T'])
        record('REUSING_COMPLETED_PHYSICAL_AND_NULL_MODE_CHECK',source=str(output))
    else:
        gate('PHYSICAL_CHECK');record('PHYSICAL_CHECK',orbit=str(profile))
        check=defect(s,profile,a.device,harmonic_chunk_size=64,stream_harmonics=True)
        zz=np.load(profile);check['filter_state_check']=filter_state_minima(s,zz['r'],float(zz['T']))
        assert float(zz['J'])==float(z['J'])
        old=resample(z['r']*1000,N,axis=0);fine=zz['r']*1000
        weights=s.geo['group_size']/s.geo['group_size'].sum()
        distance,_=distances(old[:,None,:],fine,weights)
        wave_change=float(distance[0]/np.sqrt(np.mean(np.sum((old-old.mean(0))**2*weights,axis=1))))
        period_change=abs(float(zz['T'])/float(z['T'])-1)
        assert wave_change<1e-3 and period_change<1e-4,(wave_change,period_change)
        result=dict(label=label,N=N,orbit=str(profile),source_root_orbit=str(original),source_root_N=root['N'],
            J_EE_core=float(zz['J']),T_ms=float(zz['T']),continuous_check=check,
            relative_waveform_change=wave_change,relative_period_change=period_change)
        output=PERIODIC_OUT/(label+'_filter_state_followup.json')
        physical=(check['filter_state_check']['positive'] and check['minimum_rate_Hz']>=-1e-9
                  and check['maximum_group_defect_Hz']<.1 and max(check['regional_defect_Hz'])<.001)
        if not physical:
            result['status']='FINER_PHYSICAL_PROFILE_REQUIRED';write(output,result);record(result['status']);return
        gate('ANTIPERIODIC_RESIDUAL');record('ANTIPERIODIC_RESIDUAL')
        source=np.load(source_mode)['u'];u=resample(np.r_[source,-source],2*N,axis=0)[:N]
        anti=Antiperiodic(s,profile,N,a.device,low_memory=True,harmonic_chunk_size=64,stream_harmonics=True)
        cp=anti.cp;v=cp.asarray(u.ravel());residual=float(cp.linalg.norm(anti.apply(v))/cp.linalg.norm(v))
        result['antiperiodic_relative_residual']=residual
        anti.o.cache=None;del anti,v;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        if residual>=1e-7:
            result['status']='FINE_CRITICAL_MODE_CORRECTION_REQUIRED';write(output,result);record(result['status'],residual=residual);return
        mode=PERIODIC_OUT/f'{label}_filter_mode{suffix}_N{N}.npz'
        if mode.exists():
            previous=np.load(mode)
            assert previous['u'].shape==u.shape and np.allclose(previous['u'],u,rtol=1e-13,atol=1e-15), 'Persisted mode differs'
            assert float(previous['J'])==float(zz['J']) and float(previous['T'])==float(zz['T'])
        else:
            save_periodic_array(mode,u=u,J=float(zz['J']),T=float(zz['T']))
        write(output,dict(result,status='INDEPENDENT_MONODROMY_PENDING',mode=str(mode)))
    write(parent,dict(orbit=str(profile),J_EE_core=float(zz['J']),T_ms=float(zz['T']),
        source_root_orbit=str(original),meaning='Same-J fine parent for independent mode validation; not a newly located root'))
    checks=[];prefix=label+'_filter'+suffix
    for dt in [.05,.025,.0125]:
        dest=PERIODIC_OUT/f'{prefix}_monodromy_check_N{N}_dt{dt:g}.json'
        cached=read(dest) if dest.exists() else None
        if not (cached and Path(cached['orbit']).resolve()==profile.resolve() and cached['mode']==str(mode)):
            gate('MONODROMY');log=PERIODIC_OUT/f'{prefix}_N{N}_dt{dt:g}_{time.time_ns()}.log'
            with log.open('w') as out:
                child=subprocess.Popen([sys.executable,'-u',str(scripts/'verify_rate_PD_monodromy.py'),
                    str(parent),str(mode),'--label',prefix,'--dt',str(dt),'--device',str(a.device),'--stream-harmonics'],
                    stdout=out,stderr=subprocess.STDOUT)
                record('MONODROMY',dt=dt,child_pid=child.pid,log=str(log));code=child.wait()
            assert code==0,('Mode check failed',code,str(log))
        checks.append(read(dest))
    errors=np.array([q['minus_one_relative_defect'] for q in checks])
    passed=errors[-1]<1e-4 and np.all(errors[:-1]/errors[1:]>3)
    result.update(status='FILTER_AND_CRITICAL_MODE_RECHECKED' if passed else 'MONODROMY_REFINEMENT_REQUIRED',
        mode=str(mode),monodromy_checks=checks,successive_error_reduction=errors[:-1]/errors[1:],
        scope='Same previously located J, refined physical parent, prolonged antiperiodic mode independently residual-checked on that parent, and full nine-state plus delay-history monodromy at three time steps. Root slope and inter-mesh J agreement remain sourced from the located roots; no new root or child criticality inferred.')
    write(output,result)
    if passed:
        cmd=[sys.executable,str(scripts/'validate_rate_period_doubling.py')]
        if label=='PD_double_upper':cmd+=['--label',label,'--display-label','PD2','--monodromy-label','PD_double_upper','--child-classification','PD_upper_child_classification.json']
        subprocess.run(cmd,check=True)
    record(result['status'],source=str(output))


if __name__=='__main__':main()

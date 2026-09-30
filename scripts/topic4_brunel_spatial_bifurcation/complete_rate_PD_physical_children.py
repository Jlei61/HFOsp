"""Recompute PD departure on the accepted fine parent and physical 2T cycles.

The original coarse children are preserved. The fine parent's J and mode
define the switch coordinate; old mesh-relative parameter shifts are never
mixed with the corrected root. A local child verdict is not global coverage.
"""
from complete_rate_positive_stability import *
from rate_periodic_accuracy import defect
from audit_rate_filter_states import filter_state_minima
import subprocess


def coarse_child_seed(parent,label,amplitude):
    """Use a solved coarse child only as an initial guess for a fine solve.

    Measure its amplitude against the *accepted fine* parent and null mode.
    No positivity, stability, or bifurcation label is inherited from it.
    """
    prefix='PDupperchild' if label=='PD_double_upper' else 'PDchild'
    candidates=list((PERIODIC_OUT/'orbits').glob(f'{prefix}_a{amplitude:.5f}_N*.npz'))
    if not candidates:return None
    N=parent['N'];base=np.load(parent['orbit']);u=np.load(parent['mode'])['u']
    assert len(base['r'])==N and u.shape==base['r'].shape
    u=resample(np.r_[u,-u],2*N,axis=0);u/=np.max(abs(u))
    reference=np.r_[base['r'],base['r']]
    candidates.sort(key=lambda f:len(np.load(f)['r']),reverse=True)
    for path in candidates:
        z=np.load(path)
        if float(z['residual'])>=2e-8:continue
        if abs(float(z['J'])-parent['J_EE_core'])>1e-5:continue
        if abs(float(z['T'])/(2*parent['T_ms'])-1)>.001:continue
        r=resample(z['r'],2*N,axis=0)
        measured=float(np.sum((r-reference)*1000*u)/np.sum(u*u))
        if not .5*amplitude<=measured<=2*amplitude:continue
        return dict(orbit=str(path),source_N=len(z['r']),target_N=2*N,
            projection_on_fine_parent_mode=measured,target_amplitude=amplitude,
            purpose='INITIAL_GUESS_ONLY',
            scope='The requested fine BVP, amplitude constraint, off-grid equation, both filter states, and child stability must be recomputed.')
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',choices=['PD_double_low','PD_double_upper'],required=True)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=8.)
    a=p.parse_args();folder=DEST/'physical_children'/a.label;folder.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).parent;worker=folder/'worker.json'
    def status(stage,**kw):
        write(worker,dict(status=stage,pid=os.getpid(),**kw));print(stage,kw,flush=True)
    extra_dependencies=[]
    if a.label=='PD_double_upper':
        # Both fine 2T Newton solves exceed half of GPU memory. Preserve
        # existing live PD1 work and serialize the later PD2 child stage.
        for file,token in [(DEST/'physical_children/PD_double_low/worker.json','complete_rate_PD_physical_children.py'),
                           (DEST/'PD1_parent_witnesses/worker.json','complete_rate_PD1_parent_witnesses.py')]:
            if not file.exists():continue
            pid=read(file).get('pid');proc=Path(f'/proc/{pid}/cmdline')
            if proc.exists() and token.encode() in proc.read_bytes():extra_dependencies.append(pid)
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in [*a.after_pid,*extra_dependencies]
          if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:deps.pop(pid)
        if deps:status('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    parent=read(PERIODIC_OUT/(a.label+'_filter_state_followup.json'))
    if parent['status']!='FILTER_AND_CRITICAL_MODE_RECHECKED':
        status('PARENT_RECHECK_REQUIRED',parent_status=parent['status']);return
    def memory(required_free,stage):
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=required_free*1024:return
            status('WAITING_GPU_RESOURCE',free_mib=free,required_free_gib=required_free,next_stage=stage);time.sleep(30)
    N=parent['N'];upper=a.label=='PD_double_upper'
    prefix=('PD2' if upper else 'PD1')+'_physical_20260920'
    amplitudes=[5.,10.,20.] if upper else [10.,20.,40.]
    branchfile=PERIODIC_OUT/f'{prefix}_branch_N{2*N}.json'
    previous=read(branchfile) if branchfile.exists() else []
    remaining=[v for v in amplitudes if not any(abs(v-q['amplitude_hz'])<1e-12 for q in previous)]
    if remaining:
        memory(max(a.min_free_gib,15. if 2*N>=16384 else a.min_free_gib),'CHILD_BVP')
        log=folder/'fine_branch.log'
        command=[sys.executable,'-u',str(scripts/'rate_period_doubled_branch.py'),
            str(PERIODIC_OUT/f'{a.label}_filter_parent_N{N}.json'),parent['mode'],
            '--N',str(N),'--device',str(a.device),'--label',prefix,'--amplitudes',*map(str,remaining),
            '--linear-normalize','--host-krylov','--stream-harmonics',
            '--derivative-chunk-size','64','--tol','2e-11']
        if previous:command+=['--from-orbit',previous[-1]['orbit']]
        else:
            seed=coarse_child_seed(parent,a.label,remaining[0])
            if seed:
                command+=['--from-orbit',seed['orbit']]
                write(folder/'coarse_child_initial_guess.json',seed)
        with log.open('a') as out:
            child=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
            status('FINE_CHILD_BRANCH',child_pid=child.pid,log=str(log));code=child.wait()
        if code:status('CHILD_SOLVE_FAILED',exit_code=code,log=str(log));return
    s=RateField();rows=read(branchfile);checks=[]
    for row in rows:
        path=Path(row['orbit']);destination=folder/(path.stem+'_physical.json')
        if destination.exists():q=read(destination)
        else:
            memory(a.min_free_gib,'CONTINUOUS_CHILD_CHECK');status('CONTINUOUS_CHILD_CHECK',orbit=str(path))
            q=defect(s,path,a.device,harmonic_chunk_size=64,stream_harmonics=True)
            z=np.load(path);q['filter_state_check']=filter_state_minima(s,z['r'],float(z['T']))
            write(destination,q)
        good=(q['filter_state_check']['positive'] and q['minimum_rate_Hz']>=-1e-9
            and q['maximum_group_defect_Hz']<.1 and max(q['regional_defect_Hz'])<.001)
        checks.append(dict(**row,physical_check=q,physical_pass=good))
    result=dict(parent=parent,rows=checks,branch_source=str(branchfile),
        scope='Fine-parent local branch switch, physical child waveforms, and exact child Poincare stability. No global branch connection or irregular-attractor claim.')
    if not all(q['physical_pass'] for q in checks):
        result['status']='CHILD_TEMPORAL_REFINEMENT_REQUIRED'
        write(folder/'result.json',result);status(result['status']);return
    target=Path(rows[-1]['orbit']);attempts=[]
    for nev,steps in [(2,[.1,.05]),(6,[.05,.025]),(8,[.025,.0125])]:
        pair=[]
        for dt in steps:
            release(a.device);tag=f'{prefix}_child_k{nev}'
            source=PERIODIC_OUT/'poincare_floquet'/f'{tag}_dt{dt:g}.json'
            if not source.exists():
                required=a.min_free_gib
                if 2*N>=16384:
                    bounded_check=PERIODIC_OUT/'bounded_harmonic_index_monodromy_check.json'
                    bounded=bounded_check.exists() and read(bounded_check)['status']=='PASS'
                    required=max(required,6. if bounded else 15.)
                memory(required,f'CHILD_POINCARE_k{nev}_dt{dt:g}')
            status('CHILD_POINCARE',nev=nev,dt_ms=dt,orbit=str(target))
            q=read(source) if source.exists() else compute_return(target,dt,nev,a.device,
                max(10,2*nev+4),stream_harmonics=True,output_label=tag)
            assert Path(q['orbit']).resolve()==target.resolve();pair.append(q)
        verdict=paired_modes(*pair);attempts.append(dict(steps_ms=steps,classification=verdict,spectra=pair))
        if verdict['status']!='UNRESOLVED':break
    coeff=np.array([row['J_shift_over_amplitude_squared'] for row in rows])
    spread=float(np.ptp(coeff)/abs(np.mean(coeff)))
    # Preserve the original direction convention only after all new
    # parameter shifts refer to the accepted fine parent and share a sign.
    geometry=bool(np.all(coeff>0) and spread<.1)
    expected='NUMERICALLY_STABLE' if upper else 'UNSTABLE'
    result.update(status='PHYSICAL_CHILD_BRANCH_CHECKED' if geometry and verdict['status']==expected
        else 'CRITICALITY_RECHECK_REQUIRED',full_physical_child_checks=True,
        coefficient_relative_spread=spread,child_classification=verdict,attempts=attempts,
        proposed_local_criticality='SUPERCRITICAL_PD' if upper else 'SUBCRITICAL_PD',
        promotion_requirement='Parent-side witnesses and crossing direction must be matched to physical corrected cycles before updating the canonical criticality.')
    write(folder/'result.json',result);status(result['status'])


if __name__=='__main__':main()

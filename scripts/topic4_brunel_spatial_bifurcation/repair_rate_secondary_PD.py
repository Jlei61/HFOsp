"""Bounded mesh and full-state follow-up of the secondary -1 candidate."""
from rate_periodic_accuracy import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--wait-pids',type=int,nargs='*',default=[])
    a=p.parse_args();label='PD_upper_child_next_amplitude';prefix='PD2_secondary'
    worker=PERIODIC_OUT/'PD2_secondary_refinement_worker.json';rows=[]
    scripts=Path(__file__).parent
    def record(status,**kw):write(worker,dict(status=status,pid=os.getpid(),rows=rows,**kw))
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.wait_pids if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:deps.pop(pid)
        if deps:record('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    assert read(PERIODIC_OUT/'streamed_harmonic_operator_check.json')['status']=='PASS'
    def gate():
        import cupy as cp
        cp.cuda.Device(a.device).use();gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=14*1024:return
            record('WAITING_RESOURCE',free_mib=free,required_free_gib=14);time.sleep(30)
    def run(stage,cmd):
        gate();log=PERIODIC_OUT/f'{prefix}_{stage}_{time.time_ns()}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',*map(str,cmd)],stdout=out,stderr=subprocess.STDOUT)
            record('RUNNING',stage=stage,child_pid=child.pid,log=str(log));code=child.wait()
        if code:record('STOPPED_STAGE_FAILURE',stage=stage,exit_code=code,log=str(log));raise RuntimeError(stage)
    previous=read(PERIODIC_OUT/f'{label}_N2048.json')
    for N in [4096,8192]:
        source=PERIODIC_OUT/f'{label}_N{N}.json'
        if not source.exists():
            center=previous['coordinate_value'];oldN=previous['N']
            run(f'root_N{N}',[scripts/'rate_child_secondary_PD.py',previous['orbit'],previous['orbit'],
                '--parent',PERIODIC_OUT/'PD_double_upper_N2048.json',
                '--parent-mode',PERIODIC_OUT/'PD_double_upper_mode_N2048.npz',
                '--seed',PERIODIC_OUT/f'{label}_mode_N{oldN}.npz','--N',N,'--device',a.device,
                '--stream-harmonics','--host-krylov','--amplitude-bracket',center-2,center+2,
                '--cache-glob',f'{label}_eval_*_N{N}.npz'])
        q=read(source);gate();record('CONTINUOUS_AND_FILTER_CHECK',N=N)
        check=defect(RateField(),q['orbit'],a.device,harmonic_chunk_size=64,stream_harmonics=True)
        z=np.load(q['orbit']);check['filter_state_check']=filter_state_minima(RateField(),z['r'],float(z['T']))
        continuous=PERIODIC_OUT/f'{prefix}_continuous_N{N}.json';write(continuous,check)
        dj=abs(q['J_EE_core']-previous['J_EE_core'])
        passed=(check['filter_state_check']['positive'] and check['minimum_rate_Hz']>=-1e-9 and
            check['maximum_group_defect_Hz']<.1 and max(check['regional_defect_Hz'])<.001 and dj<1e-7)
        rows.append(dict(N=N,root=str(source),J_mesh_difference=dj,continuous_check=str(continuous),
                         physical_profile_and_mesh_pass=passed))
        previous=q
        if not passed:record('FINER_MESH_REQUIRED',N=N);continue
        checks=[]
        for dt in [.05,.025,.0125]:
            output=PERIODIC_OUT/f'{prefix}_monodromy_check_N{N}_dt{dt:g}.json'
            if not output.exists():
                run(f'mode_N{N}_dt{dt:g}',[scripts/'verify_rate_PD_monodromy.py',source,
                    PERIODIC_OUT/f'{label}_mode_N{N}.npz','--label',prefix,'--device',a.device,
                    '--dt',dt,'--stream-harmonics'])
            checks.append(read(output))
        errors=np.array([v['minus_one_relative_defect'] for v in checks])
        mode_pass=errors[-1]<1e-4 and np.all(errors[:-1]/errors[1:]>3)
        phase_pass=checks[-1]['phase_relative_defect']<1e-4
        result=dict(status='VALIDATED_SECONDARY_MINUS_ONE_CROSSING' if mode_pass and phase_pass else 'MODE_REFINEMENT_REQUIRED',
            root=q,mesh_checks=rows,continuous_profile=check,full_state_mode_checks=checks,
            error_reduction_factors=errors[:-1]/errors[1:],criticality='NOT_COMPUTED',child_branch='NOT_COMPUTED',
            scope='The same frozen full spatial delay model. Independent -1 mode validation does not establish a new stable doubled child, a full Floquet spectrum, irregular bursting, or native-SNN equivalence.')
        write(PERIODIC_OUT/'PD2_secondary_validation.json',result)
        record('FOLLOWUP_FINISHED',scientific_status=result['status']);return
    record('MAXIMUM_MESH_REACHED_WITHOUT_ACCEPTANCE',scope='No promotion and no claim of absence of a bifurcation; inspect the unresolved numerical evidence.')


if __name__=='__main__':main()

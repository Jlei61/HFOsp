"""Independently test the later PD4-child -1 candidate with an anti-periodic BVP.

Spectral endpoint evidence selects a numerical starting point only. The
bounded job stops after locating one root on N=1024; mesh agreement and
independent full-history null-mode checks remain required for promotion.
"""
from complete_rate_positive_stability import *
from rate_antiperiodic import Antiperiodic
import subprocess


FOLDER=DEST/'H2_local_PD/later_flip_root'
LABEL='PD4_child_later_flip_20260920'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=1)
    parser.add_argument('--after-pids',type=int,nargs='*',default=[])
    parser.add_argument('--min-free-gib',type=float,default=8.)
    a=parser.parse_args();assert a.min_free_gib>=8.
    FOLDER.mkdir(exist_ok=True)
    def status(stage,**kw):
        write(FOLDER/'worker.json',dict(status=stage,pid=os.getpid(),timestamp=time.time(),**kw))
        print(stage,kw,flush=True)
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pids
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            f=Path(f'/proc/{pid}/cmdline')
            try:live=f.read_bytes()==identity
            except FileNotFoundError:live=False
            if not live:dependencies.pop(pid)
        if dependencies:status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    def gate(stage):
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            status('WAITING_GPU_RESOURCE',free_mib=free,required_free_gib=a.min_free_gib,next_stage=stage)
            time.sleep(30)
    def run(command,stage):
        gate(stage)
        log=FOLDER/f'{stage}_{time.time_ns()}.log'
        with log.open('w') as output:
            p=subprocess.Popen([sys.executable,'-u',*map(str,command)],stdout=output,stderr=subprocess.STDOUT)
            status(stage,child_pid=p.pid,log=str(log))
            code=p.wait()
        if code:raise RuntimeError(f'{stage} failed: {log}')
    profiles=read(DEST/'H2_local_PD/physical_children.json')['rows'][1:]
    assert [q['amplitude_hz'] for q in profiles]==[.04,.08]
    endpoints=[]
    for index,profile in zip([1,2],profiles):
        assert profile['physical_check']['filter_state_check']['positive']
        assert profile['physical_check']['maximum_group_defect_Hz']<1e-6
        paths=[PERIODIC_OUT/f'poincare_floquet/PD4_child_index{index}_full_k6_20260920_dt{dt}.json'
               for dt in ['0.05','0.025']]
        pair=[read(path) for path in paths]
        assert all(Path(q['orbit']).resolve()==Path(profile['orbit']).resolve() for q in pair)
        verdict=paired_modes(*pair);mu=values(pair[-1]);ids=np.flatnonzero((abs(mu.imag)<1e-8)&(mu.real<0))
        assert len(ids)==2
        weak=ids[np.argmin(abs(mu[ids]))]
        assert verdict['reliable_mode_mask'][weak]
        assert pair[-1]['phase_overlap'][weak]<1e-6
        endpoints.append(dict(amplitude_Hz=profile['amplitude_hz'],J_EE_core=profile['J_EE_core'],
            weak_multiplier=float(mu[weak].real),weak_margin=float(verdict['per_mode_margin'][weak]),
            block_at_minus_one=float(np.prod(-1-mu[ids].real)),sources=list(map(str,paths))))
    left,right=endpoints
    assert left['weak_multiplier'] < -1-left['weak_margin']
    assert -1+right['weak_margin'] < right['weak_multiplier'] < 0
    assert left['block_at_minus_one']>0>right['block_at_minus_one']
    fraction=left['block_at_minus_one']/(left['block_at_minus_one']-right['block_at_minus_one'])
    amplitude=float(np.sqrt((1-fraction)*.04**2+fraction*.08**2))
    write(FOLDER/'initial_guess.json',dict(status='SEED_ONLY',amplitude_Hz=amplitude,endpoints=endpoints,
        rule='Interpolate the selected two-mode characteristic evaluated at -1 linearly in squared child amplitude.',
        scope='Numerical initial guess only; not a located root or a certified continuation of this spectral block.'))
    parent=read(PERIODIC_OUT/'PD_H2_after_LPC13_validation.json')
    assert parent['full_acceptance'] and parent['criticality']=='SUBCRITICAL_PD'
    scripts=Path(__file__).resolve().parent
    seed_label=LABEL+'_seed'
    branch=PERIODIC_OUT/f'{seed_label}_branch_N1024.json'
    if not branch.exists():
        run([scripts/'rate_period_doubled_branch.py',PERIODIC_OUT/'PD_H2_after_LPC13_N512.json',
            parent['accepted_mode'],'--N','512','--amplitudes',repr(amplitude),
            '--from-orbit',profiles[0]['orbit'],'--device',a.device,'--label',seed_label,
            '--linear-normalize','--stream-harmonics','--host-krylov','--tol','2e-11'],'SEED_BVP')
    seed_rows=read(branch);assert len(seed_rows)==1
    candidate=Path(seed_rows[0]['orbit'])
    gate('SEED_PHYSICAL_CHECK')
    actual,physical=prepare(candidate,a.device,max_N=2048,check_filter_states=True,
        harmonic_chunk_size=64,stream_harmonics=True,adaptive_memory=True,host_krylov=True,
        before_mesh=lambda N,stage:gate(f'SEED_{stage}_N{N}'))
    assert physical['status']=='RESOLUTION_CHECKED' and physical['maximum_group_defect_Hz']<1e-6
    assert physical['filter_state_check']['positive']
    write(FOLDER/'seed_physical_check.json',dict(orbit=str(actual),physical=physical))
    seed=PERIODIC_OUT/f'{LABEL}_antiperiodic_seed_N512.npz'
    if not seed.exists():
        gate('ANTIPERIODIC_SEED')
        status('ANTIPERIODIC_SEED',orbit=str(actual),N=512)
        anti=Antiperiodic(RateField(),actual,N=512,device=a.device,low_memory=True,
                           harmonic_chunk_size=64,stream_harmonics=True)
        vals,vectors,errors=anti.compute(k=3)
        selected=int(np.argmin(abs(vals)))
        write(FOLDER/'antiperiodic_seed.json',dict(orbit=str(actual),N=512,eigenvalues=vals,
            residuals=errors,selected_index=selected,
            scope='Eigenvalues of the antiperiodic boundary operator, not Floquet multipliers; used only to seed the bordered root solve.'))
        if abs(vals[selected].imag)>1e-7 or errors[selected]>1e-6:
            status('ANTIPERIODIC_SEED_UNRESOLVED');return
        save_periodic_array(seed,u=vectors[:,selected].real.reshape(512,935),J=anti.J,T=anti.T)
        del anti;release(a.device)
    root=PERIODIC_OUT/f'{LABEL}_N1024.json'
    if not root.exists():
        run([scripts/'rate_child_secondary_PD.py',profiles[0]['orbit'],profiles[1]['orbit'],
            '--parent',PERIODIC_OUT/'PD_H2_after_LPC13_N512.json','--parent-mode',parent['accepted_mode'],
            '--seed',seed,'--N','1024','--device',a.device,'--label',LABEL,
            '--stream-harmonics','--host-krylov','--tol','2e-11','--amplitude-bracket','.04','.08'],
            'ANTIPERIODIC_ROOT_N1024')
    q=read(root)
    assert q['antiperiodic_relative_residual']<1e-7 and q['half_period_relative_mismatch']>1e-5
    assert profiles[0]['J_EE_core']<q['J_EE_core']<profiles[1]['J_EE_core']
    write(FOLDER/'result.json',dict(status='ROOT_LOCATED_CHECKS_PENDING',root_source=str(root),
        root=q,full_acceptance=False,
        scope='Independent antiperiodic root on one temporal mesh. Requires finer mesh, positive full-orbit checks, nonzero crossing verification and independent full-history mode checks before a new PD is added to the figure.'))
    status('ROOT_LOCATED_CHECKS_PENDING',J_EE_core=q['J_EE_core'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        FOLDER.mkdir(exist_ok=True)
        write(FOLDER/'worker.json',dict(status='COMPUTATION_FAILED',pid=os.getpid(),error=repr(exc)))
        raise

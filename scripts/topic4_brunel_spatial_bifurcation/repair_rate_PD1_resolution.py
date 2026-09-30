"""Resume the same PD1 root at finer temporal resolution, then verify it."""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--wait-pids',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=14.)
    p.add_argument('--N',type=int,default=3072)
    p.add_argument('--stream-harmonics',action='store_true')
    a=p.parse_args();label='PD_double_low';scripts=Path(__file__).parent
    worker=PERIODIC_OUT/'PD1_resolution_repair_worker.json'
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.wait_pids if Path(f'/proc/{pid}/cmdline').exists()}
    def record(status,**kw):write(worker,dict(status=status,pid=os.getpid(),**kw))
    def gate():
        import gc,cupy as cp
        cp.cuda.Device(a.device).use();gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
        while True:
            active=[]
            for pid,identity in deps.items():
                try:
                    if Path(f'/proc/{pid}/cmdline').read_bytes()==identity:active.append(pid)
                except FileNotFoundError:pass
            if active:record('WAITING_DEPENDENCIES',dependencies=active);time.sleep(30);continue
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            record('WAITING_GPU_RESOURCE',free_mib=free,required_free_gib=a.min_free_gib);time.sleep(30)
    def run(stage,args):
        gate();log=PERIODIC_OUT/'remaining_turn_refinements'/f'PD1_{stage}_{time.time_ns()}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],stdout=out,stderr=subprocess.STDOUT)
            record('RUNNING',stage=stage,child_pid=child.pid,log=str(log));code=child.wait()
        if code:record('STOPPED_STAGE_FAILURE',stage=stage,exit_code=code,log=str(log));raise RuntimeError(stage)
    N=a.N
    earlier=[read(f) for f in PERIODIC_OUT.glob(f'{label}_N*.json') if read(f)['N']<N]
    assert earlier,'A lower-mesh critical root is required'
    old=max(earlier,key=lambda q:q['N']);path=old['orbit'];J=old['J_EE_core']
    root=PERIODIC_OUT/f'{label}_N{N}.json'
    if not root.exists():
        run(f'root_N{N}',[scripts/'rate_period_doubling.py',path,path,'--seed',PERIODIC_OUT/f'{label}_mode_N{old["N"]}.npz',
             '--label',label,'--N',N,'--device',a.device,'--low-memory','--linear-normalize',
             *(['--stream-harmonics','--host-krylov'] if a.stream_harmonics else ['--cached-antiperiodic','--gpu-antiperiodic']),
             '--cache-glob',f'{label}_eval_J*_N{N}.npz','--parameter-bracket',J-2e-7,J+2e-7])
    q=read(root);continuous=PERIODIC_OUT/f'{label}_continuous_defect_N{N}.json'
    if not continuous.exists():
        gate();record('CONTINUOUS_ORBIT_CHECK',N=N)
        from rate_periodic_accuracy import defect
        check=defect(RateField(),q['orbit'],a.device,harmonic_chunk_size=64,stream_harmonics=a.stream_harmonics);write(continuous,check)
        import gc,cupy as cp
        gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
    check=read(continuous)
    from audit_rate_filter_states import filter_state_minima
    z=np.load(q['orbit']);check['filter_state_check']=filter_state_minima(RateField(),z['r'],float(z['T']))
    write(continuous,check)
    if not (check['minimum_rate_Hz']>=-1e-9 and check['maximum_group_defect_Hz']<.1
            and max(check['regional_defect_Hz'])<.001 and check['filter_state_check']['positive']):
        record('FINER_TIME_MESH_REQUIRED',check=check);return
    for dt in [.05,.025,.0125]:
        output=PERIODIC_OUT/f'PD_monodromy_check_N{N}_dt{dt:g}.json'
        if not output.exists():run(f'mode_N{N}_dt{dt:g}',[scripts/'verify_rate_PD_monodromy.py',root,PERIODIC_OUT/f'{label}_mode_N{N}.npz','--device',a.device,'--dt',dt,
            *(['--stream-harmonics'] if a.stream_harmonics else [])])
    direct=[read(PERIODIC_OUT/f'PD_monodromy_check_N{N}_dt{dt:g}.json') for dt in [.05,.025,.0125]]
    errors=np.array([v['minus_one_relative_defect'] for v in direct])
    assert errors[-1]<1e-4 and np.all(errors[:-1]/errors[1:]>3),errors
    run('acceptance',[scripts/'validate_rate_period_doubling.py'])
    record('PARENT_RECHECK_FINISHED',source=str(PERIODIC_OUT/f'{label}_validation.json'),
           scope='Parent root only; child criticality remains a separate resolution review.')


if __name__=='__main__':main()

"""Relocate PD2 when prolonging its coarse null mode fails on the fine parent."""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=13.)
    p.add_argument('--reuse-physical',action='store_true',help='Resume the persisted fine parent and null-mode checks after an independent-check failure')
    a=p.parse_args();label='PD_double_upper';scripts=Path(__file__).parent
    folder=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/PD2_fine_root')
    folder.mkdir(parents=True,exist_ok=True);worker=folder/'worker.json'
    def record(state,**kw):write(worker,dict(status=state,pid=os.getpid(),**kw))
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid
          if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:deps.pop(pid)
        if deps:record('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    while True:
        free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
        if free>=a.min_free_gib*1024:break
        record('WAITING_GPU_RESOURCE',free_mib=free);time.sleep(30)
    def run(stage,args):
        log=folder/f'{stage}_{time.time_ns()}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],stdout=out,stderr=subprocess.STDOUT)
            record(stage,child_pid=child.pid,log=str(log));code=child.wait()
        if code:record('STAGE_FAILED',stage=stage,exit_code=code);raise RuntimeError(str(log))
    old=read(PERIODIC_OUT/f'{label}_N2048.json');J=old['J_EE_core']
    followup=PERIODIC_OUT/(label+'_filter_state_followup.json')
    if followup.exists() and not (folder/'preceding_filter_followup.json').exists():
        write(folder/'preceding_filter_followup.json',read(followup))
    root=PERIODIC_OUT/f'{label}_N4096.json'
    if not root.exists():
        run('ROOT_N4096',[scripts/'rate_period_doubling.py',old['orbit'],old['orbit'],
            '--seed',PERIODIC_OUT/f'{label}_mode_N2048.npz','--label',label,'--N','4096',
            '--device',a.device,'--low-memory','--linear-normalize','--host-krylov',
            '--cached-antiperiodic','--parameter-bracket',J-2e-7,J+2e-7,
            '--cache-glob',f'{label}_eval_J*_N4096.npz'])
    q=read(root);continuous=PERIODIC_OUT/f'{label}_continuous_defect_N4096.json'
    if not continuous.exists():
        from rate_periodic_accuracy import defect
        record('CONTINUOUS_ROOT_CHECK')
        write(continuous,defect(RateField(),q['orbit'],a.device,harmonic_chunk_size=64))
    run('FINE_PARENT_AND_MODE',[scripts/'complete_rate_PD_filter_followup.py',label,
        '--N','8192','--device',a.device,'--min-free-gib','8',
        *(['--reuse-physical'] if a.reuse_physical else [])])
    if read(followup)['status']!='FILTER_AND_CRITICAL_MODE_RECHECKED':
        record('FURTHER_MODE_REFINEMENT_REQUIRED',followup=str(followup));return
    run('PHYSICAL_CHILDREN',[scripts/'complete_rate_PD_physical_children.py',
        '--label',label,'--device',a.device])
    record('STAGES_FINISHED',validation=str(PERIODIC_OUT/(label+'_validation.json')))


if __name__=='__main__':main()

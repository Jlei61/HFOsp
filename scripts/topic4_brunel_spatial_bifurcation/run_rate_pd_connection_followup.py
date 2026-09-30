"""Serial GPU queue: regular PD-child coordinate, H1 extension, H2 repair."""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args();scripts=Path(__file__).parent
    state=PERIODIC_OUT/'pd_connection_followup_queue.json';rows=[]
    def record(status,**kw):write(state,dict(status=status,pid=os.getpid(),rows=rows,**kw))
    proc=Path(f'/proc/{a.after_pid}/cmdline');identity=proc.read_bytes() if proc.exists() else None
    while identity:
        try:active=proc.read_bytes()==identity
        except FileNotFoundError:active=False
        if not active:break
        record('WAITING_DEPENDENCY',dependency=a.after_pid);time.sleep(30)
    def run(stage,cmd,specific=None):
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=14*1024:break
            record('WAITING_RESOURCE',stage=stage,free_mib=free);time.sleep(30)
        log=PERIODIC_OUT/f'pd_connection_followup_{stage}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',*map(str,cmd)],stdout=out,stderr=subprocess.STDOUT)
            record('RUNNING',stage=stage,child_pid=child.pid,log=str(log))
            if specific:
                write(specific,dict(pid=child.pid,worker_pid=os.getpid(),status='AMPLITUDE_COORDINATE_ROOT_SOLVE',
                    log=str(log),source=str(PERIODIC_OUT/'PD2_child_extension_stability.json'),
                    scope='Secondary -1 search in the established child amplitude; no new PD label without mesh and independent full-state verification.'))
            code=child.wait()
        rows.append(dict(stage=stage,exit_code=code,log=str(log)));record('STAGE_FINISHED',stage=stage)
        if specific:
            q=read(specific);q.update(status='BVP_ROOT_REQUIRES_MESH_AND_MONODROMY_VALIDATION' if code==0 else 'STOPPED_CHECK_RESULTS',exit_code=code);write(specific,q)
        return code
    probe=read(PERIODIC_OUT/'PD_upper_child_next_amplitude_scan_N2048.json')
    second=next(v['orbit'] for v in probe if abs(v['coordinate_value']-30)<1e-9)
    label='PD_upper_child_next_amplitude'
    if not (PERIODIC_OUT/(label+'_N2048.json')).exists():
        run('PD_amplitude_root',[scripts/'rate_child_secondary_PD.py',
            PERIODIC_OUT/'orbits/PDupperchild_a10.00000_N4096.npz',second,
            '--parent',PERIODIC_OUT/'PD_double_upper_N2048.json',
            '--parent-mode',PERIODIC_OUT/'PD_double_upper_mode_N2048.npz',
            '--seed',PERIODIC_OUT/'PD2_child40_negative_mode_seed.npz','--N','2048','--device',a.device,
            '--cache-glob','PD_upper_child_next*_eval_*_N2048.npz'],PERIODIC_OUT/'PD2_child_next_crossing_worker.json')
    run('H1_continuation',[scripts/'continue_rate_checked_endpoints.py','--device',a.device,'--steps','160'])
    run('H2_refinement',[scripts/'repair_rate_extension_segment.py','arcBconnectionStage3',
        '--device',a.device,'--min-N','1024','--max-N','4096','--min-free-gib','14'])
    record('BATCH_FINISHED',scope='Each stage has its own scientific acceptance status; process completion is not bifurcation completeness.')


if __name__=='__main__':main()

"""Finish a refined H2 fold using independent curvature and full-state checks."""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('label')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    a=p.parse_args();scripts=Path(__file__).parent
    folder=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/H2_folds')
    folder.mkdir(parents=True,exist_ok=True);worker=folder/(a.label+'_worker.json')
    def status(state,**kw):write(worker,dict(status=state,pid=os.getpid(),label=a.label,**kw))
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid
          if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:deps.pop(pid)
        if deps:status('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    def run(stage,command):
        log=folder/f'{a.label}_{stage}_{time.time_ns()}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',*map(str,command)],stdout=out,stderr=subprocess.STDOUT)
            status(stage,child_pid=child.pid,log=str(log));code=child.wait()
        if code:raise RuntimeError(f'{stage}: {log}')
    try:
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>5*1024:break
            status('WAITING_GPU_RESOURCE',free_mib=free);time.sleep(30)
        # This reuses recorded whole-period checks, keeping their failures.
        run('ROOT_PROFILE_AND_CURVATURE',[scripts/'validate_rate_mean_fold.py',a.label,'--device',a.device])
        cpu=PERIODIC_OUT/(a.label+'_independent_variational_BVP_e0.125.json')
        if not cpu.exists():
            run('INDEPENDENT_CPU_RHS',[scripts/'check_rate_fold_full_rhs_tangent.py',
                a.label,'--factor','4','--epsilon-scale','.125'])
        seg=PERIODIC_OUT/(a.label+'_segmented_mode_checks.json')
        if not seg.exists() or read(seg)['status']!='DIAGNOSTIC_COMPLETE':
            run('FULL_STATE_SEGMENTED_CHECKS',[scripts/'check_rate_segmented_fold_mode.py',a.label,'--device',a.device])
        run('VALIDATE',[scripts/'validate_rate_segmented_fold.py',a.label])
        result=read(PERIODIC_OUT/(a.label+'_validation.json'))
        status('BATCH_FINISHED',scientific_status=result['status'],J_EE_core=result['J_EE_core'])
    except Exception as exc:
        status('STOPPED_WITH_ERROR',error=repr(exc));raise


if __name__=='__main__':main()

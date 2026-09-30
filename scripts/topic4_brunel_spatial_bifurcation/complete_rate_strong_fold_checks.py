"""Bounded sequential independent checks for already refined fold candidates."""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('labels',nargs='+')
    p.add_argument('--device',type=int,default=0);a=p.parse_args();rows=[]
    output=PERIODIC_OUT/'strong_fold_checks_worker.json'
    scripts=Path(__file__).parent
    def status(stage,**kw):
        write(output,dict(status=stage,pid=os.getpid(),rows=rows,**kw));print(stage,kw,flush=True)
    try:
        for label in a.labels:
            cpu=PERIODIC_OUT/(label+'_independent_variational_BVP_e0.125.json')
            if not cpu.exists():
                status('INDEPENDENT_CPU_RHS',label=label)
                subprocess.run([sys.executable,'-u',str(scripts/'check_rate_fold_full_rhs_tangent.py'),label,'--factor','4','--epsilon-scale','.125'],check=True)
            seg=PERIODIC_OUT/(label+'_segmented_mode_checks.json')
            if not seg.exists() or read(seg)['status']!='DIAGNOSTIC_COMPLETE':
                while True:
                    free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                    if free>9*1024:break
                    status('WAITING_GPU_RESOURCE',label=label,free_mib=free);time.sleep(30)
                status('ALL_SEGMENT_VARIATIONAL_MATCHING',label=label)
                subprocess.run([sys.executable,'-u',str(scripts/'check_rate_segmented_fold_mode.py'),label,'--device',str(a.device)],check=True)
            status('SCIENTIFIC_VALIDATION',label=label)
            subprocess.run([sys.executable,'-u',str(scripts/'validate_rate_segmented_fold.py'),label],check=True)
            q=read(PERIODIC_OUT/(label+'_validation.json'))
            rows.append(dict(label=label,scientific_status=q['status'],J_EE_core=q['J_EE_core']))
        status('BATCH_FINISHED')
    except Exception as exc:
        status('STOPPED_WITH_ERROR',error=repr(exc));raise


if __name__=='__main__':main()

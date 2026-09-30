"""Bounded independent full-state audit of previously BVP-located LPCs.

No new continuation is added here. Older located points are checked by the
same variational DDE used for the newer folds; failures remain unpromoted.
"""
from rate_periodic import *
from validate_rate_mean_fold import validation_matches_latest_root
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--labels',nargs='+',default=['LPC_A1','LPC_B1']+
        [f'LPC_A{i}' for i in range(2,6)]+[f'LPC_B{i}' for i in range(2,9)]+
        ['LPC_Bleading_low','LPC_burst_high','LPC_double_high','LPC_double_low'])
    args=parser.parse_args();folder=PERIODIC_OUT/'legacy_fold_mode_audit';folder.mkdir(exist_ok=True)
    status=folder/'worker.json';rows=[]
    for label in args.labels:
        validation=PERIODIC_OUT/(label+'_validation.json')
        if validation_matches_latest_root(label):
            rows.append(dict(label=label,status='VALIDATED_CYCLE_FOLD',source=str(validation)));continue
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(args.device),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>9*1024:break
            write(status,dict(pid=os.getpid(),status='WAITING_RESOURCE',label=label,rows=rows,free_gpu_mib=free));time.sleep(30)
        log=folder/(label+f'_{time.time_ns()}.log')
        with log.open('w') as output:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('validate_rate_mean_fold.py')),
                label,'--device',str(args.device),'--analytic-seed'],stdout=output,stderr=subprocess.STDOUT)
            write(status,dict(pid=os.getpid(),status='INDEPENDENT_MODE_CHECK',label=label,child_pid=child.pid,rows=rows,log=str(log)))
            code=child.wait()
        verdict=read(validation)['status'] if code==0 and validation.exists() else 'COMPUTATION_FAILED'
        rows.append(dict(label=label,status=verdict,exit_code=code,log=str(log),source=str(validation) if validation.exists() else None))
        print('LEGACY FOLD',rows[-1],flush=True)
    write(status,dict(pid=os.getpid(),status='BATCH_FINISHED',rows=rows,
        scope='Completed numerical checks are distinct from passed critical-mode validation. No claim of complete adjacent Floquet spectra.'))


if __name__=='__main__':main()

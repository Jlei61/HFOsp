"""Refine and independently validate the two remaining log-period folds."""
from rate_periodic import *
from audit_rate_filter_states import filter_state_minima
import subprocess


DEST=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')
CASES=[('LPC_burst_high',1024,[2048,4096,8192]),
       ('LPC_double_secondary4',4096,[8192,16384])]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--validation-free-gib',type=float,
        help='Resource gate for the exact streamed validation, independent of Newton basis storage')
    a=p.parse_args()
    folder=DEST/'period_coordinate_folds';folder.mkdir(exist_ok=True)
    worker=folder/'worker.json';rows=[];scripts=Path(__file__).parent
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),rows=rows,timestamp=time.time(),**kw))
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            path=Path(f'/proc/{pid}/cmdline')
            if not path.exists() or path.read_bytes()!=identity:dependencies.pop(pid)
        if dependencies:
            status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    verification=DEST/'qa/period_host_stream_equivalence.json'
    assert verification.exists() and read(verification)['status']=='PASS'
    s=RateField()
    def run(command,label,stage,N):
        required=8 if N<=4096 else 11 if N<=8192 else 16
        if stage=='FULL_VALIDATION' and N<=2048 and a.validation_free_gib is not None:
            required=a.validation_free_gib
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=required*1024:break
            status('WAITING_GPU_RESOURCE',label=label,stage=stage,free_mib=free,
                   required_free_gib=required);time.sleep(30)
        log=folder/f'{label}_{stage}_{time.time_ns()}.log'
        with log.open('w') as output:
            child=subprocess.Popen([sys.executable,'-u',*map(str,command)],stdout=output,stderr=subprocess.STDOUT)
            status(stage,label=label,N=N,child_pid=child.pid,log=str(log));code=child.wait()
        if code:raise RuntimeError(f'{label} {stage}: exit {code}; {log}')
    for label,base,resolutions in CASES:
        try:
            prior=PERIODIC_OUT/(label+'_validation.json')
            from validate_rate_mean_fold import validation_matches_latest_root
            if validation_matches_latest_root(label):
                accepted=read(prior)
                assert accepted['continuous_defect']['filter_state_check']['positive']
                rows.append(dict(label=label,status='VALIDATED_CYCLE_FOLD',source=str(prior),reused=True))
                continue
            if prior.exists() and not (folder/(label+'_previous_validation.json')).exists():
                write(folder/(label+'_previous_validation.json'),read(prior))
            latest=base
            for N in resolutions:
                root=PERIODIC_OUT/f'{label}_N{N}.json'
                source=PERIODIC_OUT/'orbits'/f'{label}_N{latest}.npz'
                if not root.exists():
                    run([scripts/'rate_cycle_folds.py',source,source,'--label',label,'--N',N,
                        '--device',a.device,'--low-memory','--linear-normalize','--host-krylov',
                        '--stream-harmonics','--tangent-predictor','--radius','.05','--tol','2e-11'],
                        label,'ROOT',N)
                q=read(root);z=np.load(q['orbit'])
                assert abs(q['dJ_dlogT'])<1e-7 and abs(q['d2J_dlogT2'])>1e-8
                profile=filter_state_minima(s,z['r'],float(z['T']))
                write(folder/f'{label}_filter_N{N}.json',dict(orbit=q['orbit'],**profile))
                latest=N
                if profile['positive']:break
            run([scripts/'validate_rate_mean_fold.py',label,'--device',a.device,
                '--analytic-seed','--stream-harmonics','--dt','.05','.025','.0125'],
                label,'FULL_VALIDATION',latest)
            validation=read(prior)
            write(folder/(label+'_validation_snapshot.json'),validation)
            rows.append(dict(label=label,status=validation['status'],source=str(prior)))
        except Exception as exc:
            rows.append(dict(label=label,status='COMPUTATION_FAILED',error=repr(exc)))
            status('SITE_FAILED',label=label,error=repr(exc))
    status('BATCH_FINISHED')


if __name__=='__main__':main()

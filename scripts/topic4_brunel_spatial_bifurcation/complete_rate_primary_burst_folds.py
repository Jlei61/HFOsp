"""Finish physical-profile checks at the two lower primary burst folds."""
from rate_periodic import *
from audit_rate_filter_states import filter_state_minima
import subprocess

DEST=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')
CASES=[('LPC_Bleading_low','A',2048),('LPC_burst_low','B',1024)]
EXTRA_CASES=[('LPC_A_return_recruitment','A',512),
             ('LPC_double_low','B',3072),('LPC_double_high','A',3072),
             ('LPC_double_secondary1','B',4096),
             ('LPC_double_secondary2','B',2048),
             ('LPC_double_secondary3','B',2048),
             ('LPC_A_large_return1','A',1024),
             ('LPC_A_large_return2','A',1024),
             ('LPC_A_large_return3','B',1024),
             ('LPC_single_upper1','B',1024),
             ('LPC_single_upper2','B',1024),
             ('LPC_single_upper3','B',1024),
             ('LPC_single_upper4','B',1024),
             ('LPC_A_global_turn1','A',1024)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--worker-label',default='')
    p.add_argument('--min-free-gib',type=float,default=0.)
    p.add_argument('--labels',nargs='+',choices=[q[0] for q in CASES+EXTRA_CASES],
                   default=[q[0] for q in CASES])
    p.add_argument('--resolutions',nargs='+',type=int,help='Explicit increasing refinement meshes, above the stored starting mesh')
    p.add_argument('--max-N',type=int,default=4096)
    a=p.parse_args();s=RateField();scripts=Path(__file__).parent
    folder=DEST/'primary_folds';folder.mkdir(parents=True,exist_ok=True)
    worker=folder/('worker'+('_'+a.worker_label if a.worker_label else '')+'.json');rows=[]
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),rows=rows,**kw))
    def process_identity(pid):
        try:return Path(f'/proc/{pid}/cmdline').read_bytes()
        except (FileNotFoundError,ProcessLookupError):return None
    dependencies={pid:identity for pid in a.after_pid if (identity:=process_identity(pid)) is not None}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            if process_identity(pid)!=identity:dependencies.pop(pid)
        if dependencies:
            status('WAITING_DEPENDENCY',dependencies=list(dependencies));time.sleep(30)
    def run(command,label,stage):
        if a.min_free_gib:
            while True:
                free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free>=a.min_free_gib*1024:break
                status('WAITING_GPU_RESOURCE',label=label,stage=stage,free_mib=free,
                    required_free_gib=a.min_free_gib);time.sleep(30)
        log=folder/f'{label}_{stage}_{time.time_ns()}.log'
        with log.open('w') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,command)],stdout=f,stderr=subprocess.STDOUT)
            status('RUNNING',label=label,stage=stage,log=str(log),child_pid=child.pid)
            result=child.wait()
        if result:raise RuntimeError(f'{label}: {stage} failed, see {log}')
    # The old same-J alternating-burst comparison included an invalid
    # 0.2 ms Heun step. Reuse its valid 0.1 ms result and add 0.05 ms.
    alternate=PERIODIC_OUT/'orbits/refined_J0.942000000_N1536_accuracy_N3072_accuracy_N6144.npz'
    run([scripts/'rate_floquet_poincare.py',alternate,'--device',a.device,
         '--dt','.1','.05','--nev','2','--ncv','10'],'sameJ_c','valid_delay_steps')
    for label,core,base in [q for q in CASES+EXTRA_CASES if q[0] in a.labels]:
        try:
            original=PERIODIC_OUT/(label+'_validation.json')
            if original.exists() and not (folder/(label+'_previous_validation.json')).exists():
                write(folder/(label+'_previous_validation.json'),read(original))
            latest=base
            # A non-power-of-two starting mesh (e.g. 3072) must still
            # reach the authorized cap 8192 if 6144 fails positivity.
            # This changes temporal refinement only, not spatial nodes.
            resolutions=a.resolutions or sorted({n for n in [base*2,base*4,a.max_N]
                                                  if base<n<=a.max_N})
            assert resolutions, f'No refinement mesh above {base} within max-N={a.max_N}'
            assert resolutions==sorted(set(resolutions)) and min(resolutions)>base
            for N in resolutions:
                if N>a.max_N:break
                source=PERIODIC_OUT/'orbits'/f'{label}_N{latest}.npz'
                root=PERIODIC_OUT/f'{label}_N{N}.json'
                if not root.exists():
                    run([scripts/'rate_cycle_fold_mean.py',source,source,'--label',label,
                         '--N',N,'--core',core,'--device',a.device,'--radius','.002',
                         '--low-memory','--linear-normalize','--tol','2e-11',
                         '--tangent-predictor',*(['--cached-host-krylov'] if N>=4096 else [])],
                        label,f'root_N{N}')
                q=read(root);z=np.load(q['orbit'])
                physical=filter_state_minima(s,z['r'],float(z['T']))
                write(folder/f'{label}_filter_N{N}.json',dict(orbit=q['orbit'],**physical))
                latest=N
                if physical['positive']:break
            run([scripts/'validate_rate_mean_fold.py',label,'--device',a.device,
                 '--analytic-seed','--dt','.05','.025','.0125'],label,'full_validation')
            result=read(original)
            write(folder/(label+'_validation_snapshot.json'),result)
            rows.append(dict(label=label,status=result['status'],source=str(original)))
        except Exception as exc:
            rows.append(dict(label=label,status='COMPUTATION_FAILED',error=repr(exc)))
            status('SITE_FAILED',label=label,error=repr(exc))
            if isinstance(exc,OSError):raise
    status('BATCH_FINISHED')


if __name__=='__main__':main()

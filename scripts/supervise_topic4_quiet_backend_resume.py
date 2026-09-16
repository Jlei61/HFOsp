#!/usr/bin/env python3
"""Adopt the active log-M batch, safely resume six quiet jobs at checkpoints."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import json,time,sys,subprocess,signal,fcntl
from pathlib import Path
import psutil
import run_topic4_fig5_log_m_scan as old
ROOT=old.ROOT;OUT=old.OUT;AUDIT=ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914'
def alive(pid):
    try:return psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
def main():
    qa=old.base.read(AUDIT/'quiet_backend_benchmark.json')
    assert qa['status']=='PASS' and qa['entire_checkpoint_recursive_bitwise']
    initial=old.base.read(OUT/'status.json');protocol=old.prepare()
    supervisor=psutil.Process(initial['pid']);assert 'supervise' in supervisor.cmdline() and 'run_topic4_fig5_log_m_scan.py' in ' '.join(supervisor.cmdline())
    assert initial['pending']==0 and not initial['failed']
    # Original supervisor has no pending jobs. Its worker sessions survive it.
    supervisor.terminate();supervisor.wait(timeout=10)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs={j['name']:j for j in protocol['jobs']};pids=initial['running'].copy();children={};failed=[]
    targets={n for n in pids if n.startswith('eta1_tau') and jobs[n]['tau_M_s']>=10}
    assert len(targets)<=6
    migrations=[];last=-1
    old.base.write(OUT/'backend_migration.json',dict(status='ADOPTED_WAITING_FOR_40S_CHECKPOINTS',original_supervisor_pid=initial['pid'],new_supervisor_pid=os.getpid(),
        benchmark=str(AUDIT/'quiet_backend_benchmark.json'),targets=sorted(targets),adopted=pids,
        preserved='Original jobs, parameters, outcomes and saved observations are unchanged. Ordered CPU and GPU continuation have identical full spike matrix and entire checkpoint. Only uncheckpointed replay after a saved boundary is discarded.',migrations=migrations))
    while pids:
        for n,pid in list(pids.items()):
            folder=OUT/'runs'/n
            if n in children:children[n].poll()
            if (folder/'result.json').exists():
                if not alive(pid):pids.pop(n)
                continue
            if not alive(pid):
                failed.append(dict(name=n,pid=pid));pids.pop(n);continue
            if n not in targets:continue
            cp=folder/'checkpoint.pkl'
            if not cp.exists():continue
            prog=old.base.read(folder/'progress.json')
            if prog.get('time_s',0)<40:continue
            state=old.base.load_pickle(cp);step=int(state['engine']['step'])
            if step<400000:continue
            assert state['job']==jobs[n] and state['identity']==protocol['identity']
            # Pause other dispatcher during the brief worker replacement gap.
            r2=AUDIT/'fast_threshold_round2/status.json';dispatch_pid=None
            if r2.exists():
                q=old.base.read(r2);possible=q.get('pid')
                if possible and alive(possible) and 'supervise_topic4_fast_threshold_recovery.py' in ' '.join(psutil.Process(possible).cmdline()):
                    dispatch_pid=possible;os.kill(dispatch_pid,signal.SIGSTOP)
            try:
                proc=psutil.Process(pid);assert n in proc.cmdline() and 'worker' in proc.cmdline()
                proc.suspend()
                # Read again after suspension: the most recent atomic checkpoint wins.
                state=old.base.load_pickle(cp);step=int(state['engine']['step'])
                assert state['job']==jobs[n]
                import numpy as np
                ends=[]
                for ch in sorted((folder/'chunks').glob('*.npz')):
                    if '.tmp.' not in ch.name:
                        with np.load(ch) as a:ends.append(int(a['end_step']))
                if max(ends)!=step:
                    # A newer chunk may have been published just before capture
                    # writes its paired checkpoint. Let that boundary finish.
                    proc.resume();continue
                cp_sha=old.base.sha(cp)
                proc.kill();proc.wait(timeout=10)
                with (folder/'worker.log').open('a') as log:
                    log.write(f'\nBACKEND_RESUME ordered CPU at saved step {step}; previous GPU pid {pid}\n');log.flush()
                    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_topic4_quiet_cpu_resume.py'),'worker','--name',n,'--producer-script',str(ROOT/'scripts/run_topic4_fig5_log_m_scan.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[n]=child;pids[n]=child.pid;targets.remove(n)
                record=dict(name=n,old_pid=pid,new_pid=child.pid,checkpoint_step=step,checkpoint_sha256=cp_sha,
                    previously_observed_progress_s=prog.get('time_s'),time=time.time(),no_state_reset=True)
                migrations.append(record);old.base.write(folder/'backend_resume.json',record)
                report=old.base.read(OUT/'backend_migration.json');report.update(status='MIGRATING' if targets else 'MIGRATED_MONITORING',migrations=migrations)
                old.base.write(OUT/'backend_migration.json',report)
            finally:
                if alive(pid) and psutil.Process(pid).status()==psutil.STATUS_STOPPED:psutil.Process(pid).resume()
                if dispatch_pid and alive(dispatch_pid):os.kill(dispatch_pid,signal.SIGCONT)
        complete=sum((OUT/'runs'/n/'result.json').exists() for n in jobs)
        old.base.write(OUT/'status.json',dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',pid=os.getpid(),updated_at=time.time(),completed_new=complete,total_new=protocol['new_jobs'],reused=protocol['reused'],running=pids,pending=0,failed=failed,backend='mixed_existing_GPU_and_validated_quiet_CPU'))
        if complete!=last:old.analyze();last=complete
        if pids:time.sleep(15)
    old.analyze()
    old.base.write(OUT/'status.json',dict(status='COMPLETE_PENDING_HUMAN_REVIEW' if not failed else 'FAILED_REVIEW',completed_new=complete,total_new=protocol['new_jobs'],reused=protocol['reused'],running={},pending=0,failed=failed))
if __name__=='__main__':main()

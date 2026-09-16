#!/usr/bin/env python3
"""Queue only the nine already-authorized interrupted followups after the pilot."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import psutil
import run_topic4_weaker_M_onset_pilot as pilot
import supervise_topic4_m_modes_overnight as old


def main():
    queue=pilot.OUT/'previous_followup_queue.json'
    lock=(pilot.OUT/'previous_followup_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not old.discover(),'Existing old workers must not be duplicated'
    protocol=pilot.core.read(old.OUT/'protocol.json')
    pending=[j['name'] for j in protocol['jobs'] if not (old.OUT/'runs'/j['name']/'result.json').exists()]
    assert len(pending)<=9
    record=dict(status='QUEUED_AFTER_PILOT',jobs=pending,new_conditions=0,pid=os.getpid(),
                reason='Previously authorized followups interrupted by host restart; current six-condition pilot has priority.',
                old_status_at_queue=pilot.core.read(old.OUT/'status.json'))
    pilot.core.write(queue,record)
    pilot.core.write(old.OUT/'status.json',dict(status='QUEUED_AFTER_WEAKER_M_PILOT',completed=40-len(pending),
        total=40,running={},pending=len(pending),failed=[],queue=str(queue),updated_at=time.time()))
    while True:
        complete=all((pilot.OUT/'runs'/j['name']/'result.json').exists() for j in pilot.core.read(pilot.OUT/'protocol.json')['jobs'])
        if complete:break
        if (pilot.OUT/'supervisor_failure.json').exists():
            record.update(status='WAITING_AFTER_PILOT_FAILURE',updated_at=time.time());pilot.core.write(queue,record);return
        time.sleep(20)
    pilot.check_sources();assert not old.discover()
    policy_path=old.OUT/'observation_tail_amendment_20260913.json';policy=pilot.core.read(policy_path)
    record['prior_adopted_records']={n:policy['adopted_jobs'][n] for n in pending}
    for name in pending:
        folder=old.OUT/'runs'/name;saved=pilot.core.load_pickle(folder/'checkpoint.pkl')
        assert saved['job']==pilot.core.read(old.OUT/'jobs'/(name+'.json'))
        assert saved['identity']==protocol['identity']
        step=int(saved['engine']['step']);cursor=0
        for path in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            lo,hi=map(int,path.stem.split('_'))
            if hi>step:continue
            assert lo==cursor,(name,path);cursor=hi
        assert cursor==step
        policy['adopted_jobs'][name]=dict(policy['adopted_jobs'][name],resume_step=step,
                                        checkpoint_sha256=pilot.core.sha(folder/'checkpoint.pkl'))
    policy.setdefault('resume_history',[]).append(dict(reason='Resume pre-existing followups after paired weaker-M pilot',time=time.time(),queue=str(queue)))
    pilot.core.write(policy_path,policy)
    record['started']={}
    for name in pending:
        folder=old.OUT/'runs'/name
        with (folder/'after_weaker_M_pilot.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(old.ROOT/'scripts/run_topic4_m_modes_short_tail.py'),
                str(old.RUNNER),'worker','--job',str(old.OUT/'jobs'/(name+'.json'))],cwd=old.ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['started'][name]=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time())
        time.sleep(.25)
    with (old.OUT/'after_weaker_M_pilot_controller.log').open('a') as log:
        controller=subprocess.Popen([sys.executable,'-u',str(old.ROOT/'scripts/supervise_topic4_m_modes_lookup.py')],
            cwd=old.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(status='STARTED_EXISTING_FOLLOWUPS',controller_pid=controller.pid,updated_at=time.time())
    pilot.core.write(queue,record)


if __name__=='__main__':main()

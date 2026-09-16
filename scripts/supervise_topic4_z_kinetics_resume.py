#!/usr/bin/env python3
"""Adopt the unchanged Z-kinetics workers after an exact-source repair."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
import run_topic4_m_on_z_kinetics as core

ROOT, OUT = core.ROOT, core.OUT
RUNNER = Path(core.__file__).resolve()


def discover():
    active = {}
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        cmd = process.info['cmdline'] or []
        if 'worker' not in cmd or '--job' not in cmd: continue
        if not any(Path(a).name == RUNNER.name for a in cmd): continue
        path = Path(cmd[cmd.index('--job')+1])
        if path.parent.resolve() != (OUT/'jobs').resolve(): continue
        name = core.read(path)['name']
        active[name] = dict(pid=process.pid, create_time=process.info['create_time'])
    return active


def alive(record):
    try:
        process = psutil.Process(record['pid'])
        return process.create_time() == record['create_time'] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error: return False


def main():
    lock = (OUT/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = core.read(OUT/'protocol.json'); core.check_sources(protocol)
    assert core.read(OUT/'qa.json')['status'] == 'PASS'
    points = [(3,3)]; todo = [(x,y) for x in range(7) for y in range(7) if (x,y)!=(3,3)]
    while todo:
        point = max(todo,key=lambda q:min((q[0]-v[0])**2+(q[1]-v[1])**2 for v in points))
        points.append(point); todo.remove(point)
    jobs = sorted(protocol['jobs'],key=lambda j:(points.index((j['x'],j['y']))//4,
        core.SEEDS.index(j['seed']),points.index((j['x'],j['y']))))
    active = discover(); owned = {}; streams = {}; failed = []
    completed = {j['name'] for j in jobs if (OUT/'runs'/j['name']/'result.json').exists()}
    pending = [j for j in jobs if j['name'] not in active and j['name'] not in completed]
    core.write(OUT/'adoption_after_source_repair.json',dict(time=time.time(),active=active,
        new_controller_pid=os.getpid(),worker_restart=False,source_hashes_restored=True,
        cause_record=str(ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/source_repair/incident.json')))
    started = time.time(); last_analysis = 0; previous_n = -1
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib')
    while pending or active:
        for name, record in list(active.items()):
            if alive(record): continue
            if name in owned: owned.pop(name).poll()
            if name in streams: streams.pop(name).close()
            del active[name]
            if not (OUT/'runs'/name/'result.json').exists(): failed.append(dict(name=name,error='Worker exited without result'))
        if not failed:
            try: core.check_sources(protocol)
            except Exception as exc: failed.append(dict(error=repr(exc)))
        available = psutil.virtual_memory().available/2**30; reserve = 0
        for record in active.values():
            try: reserve += max(0,4-psutil.Process(record['pid']).memory_info().rss/2**30)
            except psutil.Error: pass
        while (pending and not failed and len(active)<protocol['max_workers']
               and available-reserve>64 and psutil.cpu_percent(interval=1)<85):
            if shutil.disk_usage(OUT).free/2**30<20: break
            job = pending.pop(0); name = job['name']; stream = (OUT/'logs'/(name+'.log')).open('a')
            child = subprocess.Popen([sys.executable,'-u',str(RUNNER),'worker','--job',str(OUT/'jobs'/(name+'.json'))],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            owned[name] = child; streams[name] = stream
            active[name] = dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time()); reserve += 4
        completed = {j['name'] for j in jobs if (OUT/'runs'/j['name']/'result.json').exists()}
        core.write(OUT/'status.json',dict(status='DRAINING_AFTER_FAILURE' if failed else 'RUNNING',
            pid=os.getpid(),total=147,completed=len(completed),running={n:r['pid'] for n,r in active.items()},
            pending=len(pending),failed=failed,wall_s_since_controller_adoption=time.time()-started,
            available_memory_GiB=available,max_workers=protocol['max_workers'],adopted_existing_workers=True))
        if len(completed)!=previous_n and time.time()-last_analysis>60:
            with (OUT/'analysis.log').open('a') as stream:
                code = subprocess.call([sys.executable,str(ROOT/'scripts/analyze_topic4_m_on_z_kinetics.py')],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if code: failed.append(dict(error='Analysis failed',returncode=code))
            previous_n=len(completed); last_analysis=time.time()
        if failed and not active: break
        time.sleep(15)
    with (OUT/'analysis.log').open('a') as stream:
        code = subprocess.call([sys.executable,str(ROOT/'scripts/analyze_topic4_m_on_z_kinetics.py')],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    core.write(OUT/'status.json',dict(status='FAILED' if failed or code else 'COMPLETE_PENDING_SCIENTIFIC_REVIEW',
        completed=len(completed),total=147,pending=len(pending),running={},failed=failed,analysis_exit_code=code,
        adopted_existing_workers=True))


if __name__ == '__main__': main()

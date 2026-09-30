"""Handoff one live continuation only after its next accepted checkpoint."""
from pathlib import Path
import argparse,json,os,signal,subprocess,time


ROOT=Path(__file__).resolve().parents[2]
DEST=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')
PER=ROOT/'results/topic4_sef_hfo/interictal_spatial_rate_only_20260917/periodic_completion'


def identity(pid):
    path=Path(f'/proc/{pid}/cmdline')
    return path.read_bytes() if path.exists() else b''


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=int,required=True)
    p.add_argument('--child',type=int,required=True);p.add_argument('--minimum-points',type=int,required=True)
    a=p.parse_args();folder=DEST/'Bleading_extension'
    worker=folder/'curvature_handoff_worker.json'
    def status(state,**kw):
        worker.write_text(json.dumps(dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw),indent=2)+'\n')
        print(state,kw,flush=True)
    assert read(DEST/'qa/Bleading_checkpoint_resume_check.json')['status']=='PASS'
    assert read(PER/'curvature_predictor_same_orbit_check.json')['status']=='PASS'
    parent,child=identity(a.parent),identity(a.child)
    assert b'extend_rate_Bleading_connection.py' in parent
    assert b'rate_periodic_continue.py' in child and b'arcBleadingConnection_20260920' in child
    assert b'--quadratic-predictor' not in child and b'--resume-existing' not in child
    record=folder/'worker.json';old=read(record)
    assert old['pid']==a.parent and old['child_pid']==a.child and old['status']=='CONTINUATION'
    assert int(Path(f'/proc/{a.child}/stat').read_text().split()[3])==a.parent
    checkpoint=PER/'arcBleadingConnection_20260920_continuation.json'
    status('WAITING_NEXT_ACCEPTED_CHECKPOINT',parent=a.parent,child=a.child,minimum_points=a.minimum_points)
    deadline=time.monotonic()+3600
    while True:
        assert identity(a.parent)==parent and identity(a.child)==child,'Original worker exited or changed before handoff'
        try:q=read(checkpoint)
        except json.JSONDecodeError:time.sleep(.2);continue
        if len(q['rows'])>=a.minimum_points:break
        if time.monotonic()>deadline:raise TimeoutError('No next accepted checkpoint within bounded handoff wait; original calculation preserved')
        time.sleep(1)
    stopped=[]
    try:
        for pid,expected in [(a.parent,parent),(a.child,child)]:
            assert identity(pid)==expected
            os.kill(pid,signal.SIGSTOP);stopped.append(pid)
            until=time.monotonic()+5
            while Path(f'/proc/{pid}/stat').read_text().split()[2]!='T':
                assert time.monotonic()<until
                time.sleep(.05)
        q=read(checkpoint)
        assert len(q['rows'])>=a.minimum_points and q['status']=='CONTINUED'
        assert q['N']==4096 and q['constituent_filter_check']=='EACH_PROFILE_PASSED'
        for row in q['rows'][-3:]:
            f=Path(row['path']);meta=read(f.with_suffix('.json'))
            assert f.exists() and meta['status']=='CONVERGED' and meta['residual_hz']<2e-11
        audit=dict(timestamp=time.time(),previous_parent=a.parent,previous_child=a.child,
            previous_parent_cmdline=parent.decode().replace('\0',' '),
            previous_child_cmdline=child.decode().replace('\0',' '),
            saved_points=len(q['rows']),last_accepted_orbit=q['rows'][-1]['path'],
            checkpoint=q,original_budget=80,scope='Same full spatial model, continuation planes, temporal mesh and acceptance criteria. Preserve every accepted point; replace only the Newton initial guess after an accepted checkpoint. Full-equation same-orbit predictor and checkpoint recovery checks both passed.')
        (folder/'curvature_checkpoint_handoff.json').write_text(json.dumps(audit,indent=2)+'\n')
        os.kill(a.child,signal.SIGTERM);os.kill(a.child,signal.SIGCONT);stopped.remove(a.child)
        until=time.monotonic()+30
        while identity(a.child):
            assert time.monotonic()<until
            time.sleep(.1)
        os.kill(a.parent,signal.SIGCONT);stopped.remove(a.parent)
        until=time.monotonic()+30
        while identity(a.parent):
            assert time.monotonic()<until
            time.sleep(.1)
    finally:
        for pid in stopped:
            if identity(pid):os.kill(pid,signal.SIGCONT)
    command=[v.decode() for v in parent.split(b'\0') if v]+['--quadratic-predictor','--resume-existing']
    log=folder/'curvature_resumed_worker.log'
    with log.open('w') as stream:
        resumed=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status('RESUMED_WITH_CHECKED_PREDICTOR',parent_pid=resumed.pid,
               saved_points=len(q['rows']),log=str(log),command=command)
        code=resumed.wait()
    status('RESUMED_BATCH_FINISHED' if code==0 else 'RESUMED_BATCH_FAILED',returncode=code,log=str(log))
    assert code==0


if __name__=='__main__':main()

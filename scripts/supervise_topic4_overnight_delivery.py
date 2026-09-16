#!/usr/bin/env python3
"""Finish the already-dispatched night package; never launch new simulations."""
import os,subprocess,sys,time
from pathlib import Path
import snapshot_topic4_autonomous_exploration as snapshot

BASE=snapshot.BASE;ROOT=snapshot.ROOT

def write(value):
    import json
    p=BASE/'overnight_review/delivery_status.json';p.parent.mkdir(exist_ok=True)
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
    import json
    while True:
        snapshot.main();s=json.loads((BASE/'latest_review_snapshot.json').read_text())
        ready=s['total_completed']==48 and all(
            (BASE/name/'continuation_complete.json').exists()
            for name in ['nativeZ_continuation_to60','revisedZ_continuation_to60'])
        if ready:break
        write(dict(status='WAITING_FOR_SAVED_ENDPOINTS',updated_at=time.time(),pid=os.getpid(),
            completed=s['total_completed'],expected=48,simulations_dispatched_by_this_script=0))
        if time.time()>1789433430+600:
            write(dict(status='REVIEW_REQUIRED',reason='Some saved endpoints missing after the bounded night deadline',
                completed=s['total_completed'],expected=48,updated_at=time.time()))
            return
        time.sleep(30)
    commands=[['scripts/summarize_topic4_autonomous_overnight.py','--final'],
              ['scripts/analyze_topic4_long_recovery_contrast.py','--require-complete']]
    for name in ['resource_rho0_k200_tau10_s9108401','resource_rho0.25_k200_tau10_s9108401',
                 'resource_rho0.25_k50_tau10_s9108401']:
        commands.append(['scripts/audit_topic4_native_energy_baselines.py','--root',
            str(BASE/'native_field_candidates_recurrence'/name),'--name',name])
    for command in commands:
        write(dict(status='FINAL_ANALYSIS',command=command,updated_at=time.time(),pid=os.getpid()))
        with (BASE/'overnight_review/final_analysis.log').open('ab') as log:
            r=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            write(dict(status='ANALYSIS_FAILED_REVIEW',command=command,returncode=r.returncode,updated_at=time.time()))
            return
    write(dict(status='READY_FOR_FINAL_SCIENTIFIC_REVIEW',updated_at=time.time(),pid=os.getpid(),
        completed=48,simulations_dispatched_by_this_script=0,
        human_review='PENDING',next='Read final ledger and long contrast; inspect final figure; finish scientific report and user delivery. No automatic freeze or goal completion.'))

if __name__=='__main__':main()

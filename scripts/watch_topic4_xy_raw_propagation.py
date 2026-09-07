#!/usr/bin/env python3
"""Serial diagnostics for every completed common-seed nominee from round 6.

Writes review artifacts only. It never releases the Fig5 hold or qualifies a model.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v3'
STATE=OUT/'raw_propagation_audit'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)


def main():
    STATE.mkdir(exist_ok=True)
    guard=(STATE/'watcher.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=[Path(__file__),ROOT/'scripts/audit_topic4_xy_raw_propagation_video.py',ROOT/'scripts/audit_topic4_xy_native_boundary.py',ROOT/'src/topic4_joint_xy.py',ROOT/'config/topic4_joint_xy_kernel_v3.json']
    main_root=Path('/home/honglab/leijiaxin/HFOsp')
    sources.extend(main_root/p for p in ('results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz','results/paper-ready-figure/supplementary-video-1.gif','results/paper-ready-figure/supplementary-video-1_metadata.json'))
    lock=STATE/'watcher_source_lock.json';snapshot={str(p):sha(p) for p in sources}
    if lock.exists():
        if json.loads(lock.read_text())!=snapshot:raise RuntimeError('diagnostic watcher source changed')
    else:write(lock,snapshot)
    while True:
        if any(sha(p)!=h for p,h in snapshot.items()):raise RuntimeError('diagnostic source changed while running')
        pending=[]
        for analysis in sorted((OUT/'rounds').glob('*/analysis.json')):
            number=int(analysis.parent.name)
            if number<6:continue
            report=json.loads(analysis.read_text())
            for item in report['expanded']:
                cid=item['candidate_id'];target=STATE/cid/'automated_diagnostic_completion.json'
                if target.exists():
                    done=json.loads(target.read_text())
                    if done['analysis_sha256']!=sha(analysis) or any(sha(p)!=h for p,h in done['outputs'].items()):raise RuntimeError('completed diagnostic changed')
                else:pending.append((number,cid,analysis,target))
        for number,cid,analysis,target in pending:
            write(STATE/'watcher_status.json',{'status':'RENDERING_SERIAL_DIAGNOSTIC','round':number,'candidate_id':cid,'upstream_changed':False})
            for script in ('audit_topic4_xy_raw_propagation_video.py','audit_topic4_xy_native_boundary.py'):
                subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--round',str(number),'--candidate',cid],cwd=ROOT,check=True)
            folder=STATE/cid
            outputs={str(p):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='automated_diagnostic_completion.json'}
            write(target,{'status':'DIAGNOSTIC_GENERATED_PENDING_VISUAL_QA','round':number,'candidate_id':cid,'analysis_sha256':sha(analysis),'outputs':outputs,'scientific_qualification':False,'fig5_hold_released':False})
        upstream=subprocess.run(['systemctl','--user','is-active','codex-t4-joint-replicated-xy-20260906.service'],capture_output=True,text=True).stdout.strip()
        status='WAITING_COMPLETED_ROUND' if upstream in ('active','activating') else 'AVAILABLE_DIAGNOSTICS_DRAINED'
        write(STATE/'watcher_status.json',{'status':status,'upstream_service':upstream,'upstream_changed':False,'fig5_hold_released':False,'updated_unix':time.time()})
        if status=='AVAILABLE_DIAGNOSTICS_DRAINED':return
        time.sleep(30)


if __name__=='__main__':main()

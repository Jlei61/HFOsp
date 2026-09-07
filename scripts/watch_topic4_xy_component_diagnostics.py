#!/usr/bin/env python3
"""Serial complete-round figures and raw movies; never qualify or release Fig5."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4'
SERVICE='codex-t4-component-xy-20260906.service'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    state=OUT/'raw_propagation_audit';state.mkdir(exist_ok=True)
    guard=(state/'watcher.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=[Path(__file__),ROOT/'scripts/render_topic4_xy_component_diagnostics.py',
           ROOT/'scripts/audit_topic4_xy_raw_propagation_video.py',
           ROOT/'scripts/paper_figures/plot_topic4_joint_replicated_expanded.py',
           ROOT/'src/topic4_joint_xy.py',ROOT/'config/topic4_joint_xy_kernel_v3.json',
           ROOT/'config/topic4_joint_xy_kernel_v4.json',OUT/'objective_contract.json']
    main=Path('/home/honglab/leijiaxin/HFOsp')
    files.extend(main/p for p in ('results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz',
        'results/paper-ready-figure/supplementary-video-1.gif','results/paper-ready-figure/supplementary-video-1_metadata.json'))
    snapshot={str(p):sha(p) for p in files};lock=state/'watcher_source_lock.json'
    if lock.exists() and json.loads(lock.read_text())!=snapshot:raise RuntimeError('diagnostic source drift')
    if not lock.exists():write(lock,snapshot)
    while True:
        if any(sha(p)!=h for p,h in snapshot.items()):raise RuntimeError('diagnostic source changed')
        for analysis in sorted((OUT/'rounds').glob('*/analysis.json')):
            stage=analysis.parent;number=int(stage.name);report=json.loads(analysis.read_text())
            tasks=[('figure',None,stage/'figure_generation.json')]
            tasks.extend(('raw',r['candidate_id'],state/r['candidate_id']/'automated_diagnostic_completion.json') for r in report['expanded'])
            for kind,cid,marker in tasks:
                if marker.exists():
                    done=json.loads(marker.read_text())
                    if done['analysis_sha256']!=sha(analysis) or any(sha(p)!=h for p,h in done['outputs'].items()):
                        raise RuntimeError('completed diagnostic changed')
                    continue
                write(state/'watcher_status.json',{'status':'RENDERING_SERIAL_DIAGNOSTIC','round':number,'kind':kind,'candidate_id':cid,'fig5_hold_released':False})
                cmd=[sys.executable,str(ROOT/'scripts/render_topic4_xy_component_diagnostics.py'),kind,'--round',str(number)]
                if cid:cmd+=['--candidate',cid]
                subprocess.run(cmd,cwd=ROOT,check=True)
                folder=stage/'figures' if kind=='figure' else state/cid
                outputs={str(p):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='visual_qa.json'}
                write(marker,{'status':'DIAGNOSTIC_GENERATED_PENDING_VISUAL_QA','round':number,'candidate_id':cid,
                    'analysis_sha256':sha(analysis),'outputs':outputs,'producer_source_hashes':snapshot,
                    'scientific_qualification':False,'fig5_hold_released':False})
        service=subprocess.run(['systemctl','--user','is-active',SERVICE],capture_output=True,text=True).stdout.strip()
        active=service in ('active','activating')
        write(state/'watcher_status.json',{'status':'WAITING_COMPLETED_ROUND' if active else 'AVAILABLE_DIAGNOSTICS_DRAINED',
             'upstream_service':service,'updated_unix':time.time(),'fig5_hold_released':False})
        if not active:return
        time.sleep(30)


if __name__=='__main__':main()

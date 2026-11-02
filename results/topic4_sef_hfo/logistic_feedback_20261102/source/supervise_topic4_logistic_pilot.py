#!/usr/bin/env python3
"""Stop only this task's no-entry pilots after a complete 60-s checkpoint.

Trajectories that enter high activity continue to the original120-s horizon.
A stopped pilot is explicitly censored, keeps its native checkpoint, and can
resume with its original command. No native COMPLETE result is fabricated.
"""
import json
from pathlib import Path
import pickle
import signal
import time
import psutil

ROOT=Path('/data/hfosp/topic4_sef_hfo/logistic_feedback_20261102')


def main():
    pending={75:'dispatch.json',50:'dispatch_k50.json',40:'dispatch_k40.json'}
    while pending:
        for width,dispatch in list(pending.items()):
            name=f'logistic_r350_k{width}_s9108404';run=ROOT/'runs'/name
            if (run/'result.json').exists() or (run/'pilot_stop.json').exists():
                del pending[width];continue
            f=run/'progress.json'
            if not f.exists():continue
            progress=json.loads(f.read_text())
            # A high entry changes the question to termination and recovery.
            if progress.get('entries'):
                del pending[width];continue
            chunks=sorted((run/'chunks').glob('*.npz'))
            if not chunks or int(chunks[-1].stem.split('_')[-1])<600000:continue
            dispatch_record=json.loads((ROOT/dispatch).read_text())
            process=psutil.Process(dispatch_record['pid'])
            assert process.cmdline()==dispatch_record['command']
            assert abs(process.create_time()-dispatch_record['started_epoch'])<5
            # Wait until the atomic native checkpoint also contains this chunk.
            with (run/'checkpoint.pkl').open('rb') as handle:state=pickle.load(handle)
            step=int(state['engine']['step'])
            if step<int(chunks[-1].stem.split('_')[-1]):continue
            process.send_signal(signal.SIGTERM);process.wait(timeout=30)
            with (run/'checkpoint.pkl').open('rb') as handle:state=pickle.load(handle)
            step=int(state['engine']['step'])
            assert step==max(int(p.stem.split('_')[-1]) for p in (run/'chunks').glob('*.npz'))
            record=dict(status='PILOT_STOPPED_NO_HIGH_ENTRY',observed_s=step/10000,
                original_planned_horizon_s=120.,first_60s_no_high_entry=True,
                reason='No high entry during the completed60-s pilot; trajectory cannot demonstrate a full cycle in this pilot.',
                no_claim_about_later_entry=True,checkpoint_preserved=True,
                resume_command=dispatch_record['command'],pid=process.pid,stopped_epoch=time.time())
            (run/'pilot_stop.json').write_text(json.dumps(record,indent=2)+'\n')
            # Keep the public progress marker truthful after the worker exits;
            # the resumable native checkpoint and planned job are unchanged.
            progress.update(status=record['status'],observed_s=record['observed_s'],
                            pilot_stop='pilot_stop.json')
            temporary=f.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(progress,indent=2)+'\n');temporary.replace(f)
            print(json.dumps(dict(name=name,**record)),flush=True)
            del pending[width]
        if pending:time.sleep(10)


if __name__=='__main__':main()

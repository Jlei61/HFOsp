#!/usr/bin/env python
"""Run frozen export and seizure scoring for every state checkpoint as it appears."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
HERE=Path(__file__).resolve().parent

def pending(dirs):
    out=[]
    for d in dirs:
        for f in sorted(Path(d).glob('*.card.json')):
            try:c=json.loads(f.read_text())
            except Exception:continue
            if c.get('config',{}).get('arm')!='state' or c.get('status')!='COMPLETE':continue
            tag=str(f.parent.name)
            suf='' if tag=='runs' else f'.{tag}'
            fz=ROOT/'frozen'/f.name.replace('.card.json',f'{suf}.frozen.json')
            sz=ROOT/'seizure'/f.name.replace('.card.json',f'{suf}.seizure.json')
            cb=ROOT/'calibration'/f.name.replace('.card.json',f'{suf}.calibration.json')
            if not (fz.exists() and sz.exists() and cb.exists()):out.append((f,fz,sz,cb,tag))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cpu');ap.add_argument('--deadline-epoch',type=float,default=None)
    ap.add_argument('--dirs',nargs='+',default=[str(ROOT/'runs'),str(ROOT/'runs_extended')])
    a=ap.parse_args()
    seen=set()
    while True:
        if a.deadline_epoch and time.time()>a.deadline_epoch:
            print('postprocess deadline reached',flush=True);break
        todo=[t for t in pending(a.dirs) if str(t[0]) not in seen]
        if not todo:
            time.sleep(60);continue
        for f,fz,sz,cb,tag in todo:
            seen.add(str(f))
            for script,out in (('export_group_event_state_v0311_frozen.py',fz),
                               ('score_group_event_state_v0311_seizure.py',sz),
                               ('calibrate_group_event_state_v0311.py',cb)):
                if out.exists():continue
                default=out.with_name(out.name.replace(f'.{tag}.',f'.')) if tag!='runs' else out
                cmd=[sys.executable,str(HERE/script),'--card',str(f),'--device',a.device]
                r=subprocess.run(cmd,capture_output=True,text=True)
                print(f'{script} {f.name} [{tag}] rc={r.returncode} {r.stdout.strip()[:200]}',flush=True)
                if r.returncode!=0:print(r.stderr.strip()[-600:],flush=True)
                # A run from another results directory must not overwrite the main one.
                if default.exists() and default!=out:default.rename(out)

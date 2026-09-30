"""Bounded CPU batch for the selected kinetic model, separate result directory."""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/kinetic_fixed_D_response_20260916'
DS=(0.,.15,.20,.225,.25,.30,.50,1.)


def worker(d,h):
    name=f'D{d:.6f}_h{h}'
    with (OUT/'logs'/f'{name}.log').open('w') as f:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/topic4_kinetic_D_response.py'),
            'run','--D',str(d),'--history',str(h)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        print(name,'pid',p.pid,flush=True)
        rc=p.wait()
    print(name,'exit',rc,flush=True)
    return dict(D=d,history=h,exit_code=rc)


if __name__=='__main__':
    assert all(json.loads((OUT/'canary/qa.json').read_text()).values())
    (OUT/'logs').mkdir(exist_ok=True)
    (OUT/'status.json').write_text(json.dumps(dict(status='RUNNING',stage='initial_16',started=time.time()),indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs=[pool.submit(worker,d,h) for d in DS for h in (8000,10370)]
        results=[j.result() for j in jobs]
    (OUT/'initial_jobs.json').write_text(json.dumps(results,indent=2)+'\n')
    ok=all(j['exit_code']==0 for j in results)
    (OUT/'status.json').write_text(json.dumps(dict(status='INITIAL_COMPLETE' if ok else 'INITIAL_ERRORS',jobs=results),indent=2)+'\n')
    sys.exit(0 if ok else 1)

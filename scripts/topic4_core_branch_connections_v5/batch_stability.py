"""Evaluate the leading transverse multiplier along completed arc segments."""
from common import *
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--names',nargs='+',required=True);ap.add_argument('--stride',type=int,default=8);ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();tasks=[]
    for name in a.names:
        rows=read(OUT/'arcs'/name/'progress.json');indices=set(range(0,len(rows),a.stride));indices.add(len(rows)-1)
        for x,y in zip(rows,rows[1:]):
            if x['tangent_g']*y['tangent_g']<0:indices.update([x['index'],y['index']])
        for i in sorted(indices):
            path=Path(rows[i]['source']);dest=OUT/'poincare'/path.parent.name/path.stem/'rk4_orthogonal_dt0.05.json'
            if not dest.exists():tasks.append(path)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def run(path):
        log=logs/f'stability_{path.parent.name}_{path.stem}.log'
        with log.open('w') as f:result=subprocess.run([sys.executable,str(Path(__file__).with_name('poincare.py')),str(path),'--method','rk4','--dt','.05','--nev','1'],stdout=f,stderr=subprocess.STDOUT)
        return dict(path=str(path),exit_code=result.returncode,log=str(log))
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,path) for path in tasks]
        for f in as_completed(futures):
            row=f.result();results.append(row);print('STABILITY_TASK',json.dumps(row),flush=True)
    write('stability_batch_'+a.names[0]+'.json',results)

if __name__=='__main__':main()

from common import *
import numpy as np,argparse,subprocess
from scipy.signal import resample
from orbits import save_solve
CASES=[(6,1.14,0),(7,1.17623,0),(8,1.17632,1),(9,1.24,1),(10,1.25,1),(11,1.34,1),('12a',1.355,1),('12b',1.355,2),(13,1.37,2),(14,1.375,2),('15a',1.38,2),('15b',1.38,3),('16a',1.385,2),('16b',1.385,3),(17,1.395,3),(18,1.45,3)]
def one(number,g,family):
    dest=OUT/'periodic'/f'condition_{number}'
    if list(dest.glob('*.npz')):return
    seq=read(V6/'displayed_curve_sequences.json')[family]
    seed=min(seq,key=lambda x:abs(x['g']-g));z=np.load(seed['path'])
    save_solve(System(),g,resample(z['r'],2048,axis=0),float(z['T']),2048,dict(source=seed['path'],number=number,family=family),f'condition_{number}')
def batch():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    def job(c):
        with (OUT/'solver_logs'/f'condition_{c[0]}.log').open('w') as f:
            rc=subprocess.run([sys.executable,__file__,'--number',str(c[0]),'--g',str(c[1]),'--family',str(c[2])],stdout=f,stderr=subprocess.STDOUT).returncode
        return dict(number=c[0],returncode=rc)
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(job,c) for c in CASES]):rows.append(f.result());write('condition_orbit_status.json',rows);print(rows[-1],flush=True)
    assert all(x['returncode']==0 for x in rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',action='store_true');p.add_argument('--number');p.add_argument('--g',type=float);p.add_argument('--family',type=int);a=p.parse_args()
    if a.batch:batch()
    else:one(a.number,a.g,a.family)

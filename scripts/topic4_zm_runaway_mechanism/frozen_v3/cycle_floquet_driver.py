"""Run Floquet (floquet_v3) on selected orbits of a periodic continuation: nearest orbits to the requested
D values plus the two orbits around each cycle-fold bracket. Sequential (GPU)."""
from common_v3 import *
import argparse,subprocess,sys
PY=sys.executable
def main(a):
    info=read(DEST/'periodic'/a.label/'continuation.json');rows=info['rows'];D=np.array([r['D'] for r in rows]);picks=set()
    for d in a.D:picks.add(int(np.argmin(abs(D-d))))
    for i,j in info['brackets']:picks.update([i,j])
    for k in sorted(picks):
        orbit=rows[k]['orbit'];out=DEST/'periodic/floquet'/(Path(orbit).stem+f'_dt{a.dt}.json')
        if out.exists():log('exists',out);continue
        log('FLOQUET on',orbit,'D',rows[k]['D']);subprocess.run([PY,'-u',str(Path(__file__).parent/'floquet_v3.py'),orbit,'--dt',str(a.dt),'--nev',str(a.nev)],check=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--D',type=float,nargs='*',default=[]);p.add_argument('--dt',type=float,default=.1);p.add_argument('--nev',type=int,default=4);main(p.parse_args())

"""Refine every cycle-fold bracket of a periodic continuation with cycle_fold_v3 (global-mean coordinate)."""
from common_v3 import *
import argparse,subprocess,sys
def main(a):
    info=read(DEST/'periodic'/a.label/'continuation.json');rows=info['rows']
    for n,(i,j) in enumerate(info['brackets']):
        lab=f'LPC_{a.label}_{n+1}'
        if (DEST/'periodic'/f'{lab}_N{a.N}.json').exists():log('exists',lab);continue
        subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'cycle_fold_v3.py'),rows[i]['orbit'],rows[j]['orbit'],'--label',lab,'--N',str(a.N),'--core',a.core],check=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--N',type=int,default=128);p.add_argument('--core',default='G');main(p.parse_args())

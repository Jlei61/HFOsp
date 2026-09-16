from common import *
from poincare import compute
import argparse
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('paths',nargs='+');ap.add_argument('--dt',type=float,default=.025);ap.add_argument('--nev',type=int,default=2);a=ap.parse_args()
    for path in a.paths:compute(path,a.dt,'orthogonal','rk4',a.nev)

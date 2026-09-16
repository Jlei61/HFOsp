from common import *
from orbits import save_solve
from scipy.signal import resample
import numpy as np,argparse
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--name',required=True);p.add_argument('--N',type=int,default=4096);a=p.parse_args();z=np.load(a.source)
save_solve(System(),float(z['g']),resample(z['r'],a.N,axis=0),float(z['T']),a.N,dict(source=a.source),a.name)

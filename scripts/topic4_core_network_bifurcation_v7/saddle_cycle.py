from common import *
from orbits import save_solve
from scipy.signal import resample
import numpy as np
rows=read(V5/'arcs/surround_unstable/progress.json');z=np.load(rows[69]['source']);g=1.17626
path,row=save_solve(System(),g,resample(z['r'],2048,axis=0),float(z['T']),2048,dict(source=rows[69]['source']),'candidate_saddle_cycle')
from poincare import compute
compute(path,.025,'orthogonal','rk4',3)

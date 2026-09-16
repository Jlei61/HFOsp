from common import *
from orbits import save_solve
from scipy.signal import resample
import numpy as np
rows=read(OUT/'arcs/low_fast/progress.json');g=1.12183
row=min([r for r in rows if r['tangent_g']>0],key=lambda r:abs(r['g']-g));z=np.load(row['source'])
save_solve(System(),g,resample(z['r'],4096,axis=0),float(z['T']),4096,dict(source=row['source']),'condition_20b')

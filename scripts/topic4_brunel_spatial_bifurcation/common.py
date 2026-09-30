"""Current topology-6101 spatial mean-field analysis in the Brunel lineage."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys, json, time
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917'
PROJECTED=ROOT/'results/topic4_sef_hfo/spatial_rate_dynamics_20260917/operators'
def read(p): return json.loads(Path(p).read_text())
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,complex): return [x.real,x.imag]
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)

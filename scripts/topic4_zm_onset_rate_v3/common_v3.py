"""Shared paths and helpers for the v3 Z/M spatial rate model (colored-noise MC transfer)."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEST=ROOT/'results/topic4_sef_hfo/fig5_zm_rate_v3_20260918'
OLDV2=ROOT/'results/topic4_sef_hfo/fig5_zm_rate_synchronized_20260917'
OPERATORS=ROOT/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators'
NATIVE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401'
NATIVE2=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108402'
def read(p): return json.loads(Path(p).read_text())
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,complex): return [x.real,x.imag]
    if isinstance(x,float) and not np.isfinite(x): return None
    if isinstance(x,Path): return str(x)
    return x
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(*a):
    print(time.strftime('%H:%M:%S'),*a,flush=True)

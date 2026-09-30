"""Independent spatial-reduction validation, before any continuation."""
from pathlib import Path
import os, sys, json
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/spatial_reduction_validation_20260916'
SOURCE=ROOT/'.worktrees/topic4-continuous-core-state-r1'
V10=ROOT/'results/topic4_sef_hfo/core_spatial_readout_v10_20260916'
sys.path[:0]=[str(ROOT/'scripts/topic4_burst_regime'),str(SOURCE),str(SOURCE/'src/snn_engine')]
import runtime
from src.topic4_observation_repaired import observe
from scripts.report_topic4_label_free_dense import describe,metrics
J=1.355
DURATION=12000.
SEEDS=(848101,848102,848103)

def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def partition(positions,region,grid):
    region=np.asarray(region,dtype=np.int64)
    xy=np.clip((positions/20*grid).astype(int),0,grid-1)
    cell=xy[:,1]*grid+xy[:,0]
    labels,group=np.unique(region*grid*grid+cell,return_inverse=True)
    reg=labels//(grid*grid)
    assert np.array_equal(reg[group],region)
    assert np.all((reg>=0)&(reg<6))
    return group,reg,labels%(grid*grid)

def smooth_contacts(raw):
    x=np.arange(-8,9);k=np.exp(-x*x/(2*2.5**2));k/=k.sum()
    return np.stack([np.convolve(row,k,mode='same') for row in raw.T],axis=1)

def observations(env,duration=DURATION):
    c=read(V10/'native/a/observation_contract.json')
    ob=observe(env.T,2.,c)
    ids=np.array([i for i in ob['primary_event_indices'] if ob['events'][i]['window_ms'][0]>=2000 and ob['events'][i]['window_ms'][1]<=duration],int)
    mu=np.asarray(ob['centroid_ms']).reshape(-1,15)
    return ob,ids,mu,describe(mu[ids],c['contact_names'])

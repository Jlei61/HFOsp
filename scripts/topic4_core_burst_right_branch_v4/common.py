"""Unchanged v2 equations and graph; a separate right-branch result lineage."""
from pathlib import Path
import os, sys, json
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/topic4_core_bifurcation_v2'))
from model import System, OUT as V2
V3=ROOT/'results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915'
OUT=ROOT/'results/topic4_sef_hfo/core_burst_right_branch_v4_20260915'
OUT.mkdir(parents=True,exist_ok=True)

def write(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

def read(path):
    return json.loads(path.read_text())

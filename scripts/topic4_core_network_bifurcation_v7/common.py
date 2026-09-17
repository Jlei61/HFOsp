"""Keep the v2-v5 frozen model; resolve projections and the recruitment region."""
from pathlib import Path
import os,sys,json
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
for name in ['topic4_core_bifurcation_v2','topic4_core_burst_right_branch_v4','topic4_core_branch_connections_v5']:
    sys.path.append(str(ROOT/'scripts'/name))
from model import System,OUT as V2
V3=ROOT/'results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915'
V4=ROOT/'results/topic4_sef_hfo/core_burst_right_branch_v4_20260915'
V5=ROOT/'results/topic4_sef_hfo/core_branch_connections_v5_20260915'
V6=ROOT/'results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915'
OUT=ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916'
OUT.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(Path(path).read_text())
def write(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')

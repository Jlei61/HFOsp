"""Mean-preserving Core A heterogeneity in the frozen six-population DDE."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT/'results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915'
OUT = ROOT/'results/topic4_sef_hfo/core_heterogeneity_bifurcation_20260918'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/'scripts/topic4_core_bifurcation_v2'))
from model import System as FrozenSystem


class System(FrozenSystem):
    def __init__(self, heterogeneity=1., groups=32):
        super().__init__(groups=groups)
        self.heterogeneity = float(heterogeneity)
        if not 0 <= self.heterogeneity <= 1:
            raise ValueError('This experiment contracts the existing Core A distribution: 0 <= h <= 1.')
        with np.load(V2/'projected_graph.npz') as z:
            self.original_thresholds = z['vtheta'][z['region']==0].copy()
        self.mean_A = float(self.original_thresholds.mean())
        self.original_std_A = float(self.original_thresholds.std())
        self.threshold[0] = self.mean_A + self.heterogeneity*(self.threshold[0]-self.mean_A)
        self.actual_thresholds_A = self.mean_A + self.heterogeneity*(self.original_thresholds-self.mean_A)


def write(path, obj):
    path = Path(path)
    if not path.is_absolute(): path = OUT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    path = Path(path)
    if not path.is_absolute(): return ROOT/path
    old = Path('/home/honglab/leijiaxin/HFOsp')
    try: return ROOT/path.relative_to(old)
    except ValueError: return path

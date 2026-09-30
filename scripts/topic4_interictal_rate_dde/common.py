"""Isolated interictal spatial rate DDE paths and original physical units."""
import os,sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/topic4_interictal_surrogate'))
from interictal_common import read,write,NE,NI,DT,SOURCE,KIN,OUT as REFERENCE
BASE=ROOT/'results/topic4_sef_hfo/spatial_rate_dynamics_20260917'
FIELD=ROOT/'results/topic4_sef_hfo/interictal_spatial_population_density_6101_20260916'
HAZARD=ROOT/'results/topic4_sef_hfo/interictal_escape_hazard_closure_20260917'
PY='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'

"""Frozen E1146 core-count comparison; no changes to historical experiments."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np
REPO = CODE_ROOT / 'modeling'
WT = CODE_ROOT / 'modeling'
HERE = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT / 'modeling' / 'common'
PARENT = Path(str(INPUT_ROOT / 'modeling/topic4_sef_hfo/core_position_overnight_expansion_20260918'))
ANCHOR = PARENT / 'candidates/dense_axis3_12.json'
PY = __import__('sys').executable
BO_PY = __import__('sys').executable
ENV = dict(os.environ, **{k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}, TOPIC4_CORE_COUNT_OUT=str(OUT), PYTHONUNBUFFERED='1', MPLCONFIGDIR=str(OUT / 'tmp/mpl'))
GROUPS = ('rank_pattern', 'local_and_interrod_timing', 'participation')

def native_imports():
    pass

def read(path):
    return json.loads(Path(path).read_text())

def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list, np.ndarray)):
        return [safe(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x

def write(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(safe(x), indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()

def save_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)

def run_dir(k, restart):
    return OUT / 'per_subject/epilepsiae_1146' / f'core{k}' / f'restart{restart:02d}'

def bounds(k):
    plan = read(OUT / 'plan.json')
    radius = plan['radius_mm']
    return (np.r_[np.full(2 * k, radius + 0.05), plan['outward_bounds'][0], plan['angle_bounds_deg'][0]], np.r_[np.full(2 * k, 20 - radius - 0.05), plan['outward_bounds'][1], plan['angle_bounds_deg'][1]])

def valid(k, vector):
    x = np.asarray(vector)
    lo, hi = bounds(k)
    if x.shape != lo.shape or not np.isfinite(x).all() or np.any(x < lo) or np.any(x > hi):
        return False
    xy = x[:2 * k].reshape(k, 2)
    minimum = 2 * read(OUT / 'plan.json')['radius_mm'] + 0.05
    return all((np.linalg.norm(xy[i] - xy[j]) >= minimum for i in range(k) for j in range(i)))

def candidate(k, restart, vector, cid, epoch, stage, role):
    import copy
    anchor = read(OUT / 'anchor.json')
    x = np.asarray(vector, float)
    c = {name: copy.deepcopy(anchor[name]) for name in ('parameters', 'shape', 'outgoing', 'core_mean_rate_scale', 'core_ou_correlation')}
    c.update(id=cid, core_count=k, restart=restart, vector=x.tolist(), epoch=epoch, stage=stage, role=role, centers_mm=x[:2 * k].reshape(k, 2).tolist(), radii_mm=[anchor['radii_mm'][0]] * k, depth_scales=[1.0] * k)
    c['parameters'].update(EE_core_to_out_scale=float(x[-2]), EE_angle_offset_deg=float(x[-1]))
    return c

def verify_sources():
    for path, digest in read(OUT / 'source_snapshot.json')['files'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen dependency changed: ' + path)

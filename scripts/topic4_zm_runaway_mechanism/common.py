"""Private, frozen model and output paths for the overnight mechanism analysis."""
import sys
import faulthandler
import signal
import pickle
import hashlib
import os
from pathlib import Path
faulthandler.register(signal.SIGUSR1,all_threads=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
BASE = ROOT / 'results/topic4_sef_hfo/fig5_zm_rate_v3_20260918'
sys.path.insert(0, str(HERE / 'frozen_v3'))
from dynamics_v3 import DynamicModel, ResponseParams, Integrator
from common_v3 import read, write, log, np, time


def model(grid=20):
    cache=OUT/f'frozen_data/model_g{grid}.pkl'
    sources=[HERE/'frozen_v3'/f for f in ['model_v3.py','dynamics_v3.py','transfer_spline.py','response_tables.py']]
    sources+=list((OUT/'frozen_data/transfer_table').glob('*.npz'))+list((OUT/'frozen_data/response_closure').glob('*'))
    fingerprint={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    op=ROOT/f'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g{grid}'
    fingerprint['operator_files']={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in op.glob('*') if p.is_file()}
    if cache.exists():
        with cache.open('rb') as f:payload=pickle.load(f)
        assert payload['fingerprint']==fingerprint,'Frozen model cache no longer matches its source'
        return payload['model']
    log('MODEL loading frozen v3, grid',grid)
    s=DynamicModel(grid=grid, table=OUT/'frozen_data/transfer_table',
                   resp=ResponseParams(OUT/'frozen_data/response_closure/closure.json'), quiet=True)
    log('MODEL ready',s.P,'groups')
    tmp=cache.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:pickle.dump(dict(model=s,fingerprint=fingerprint),f,protocol=5)
    tmp.replace(cache)
    return s


def checkpoint_initial(path, Z=None):
    z = np.load(path)
    state = z['state'].copy() if 'state' in z else z['final_state'].copy()
    history = z['history'] if 'history' in z else z['final_history']
    tick = int(z['tick']) if 'tick' in z else int(z['final_tick'])
    # New slot -lag must equal old slot tick-lag at the same physical time.
    history = np.roll(history, -(tick % len(history)), axis=0).copy()
    if Z is not None:
        state[11] = Z
    return state, history

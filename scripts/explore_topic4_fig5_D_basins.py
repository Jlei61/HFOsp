"""Bounded basin exploration of the unchanged frozen-D filtered-v1 map.

Batching only shares sparse products; ReducedModel.step remains the updater.
No native-SNN runs and no assertion of asymptotic attractors from a time window.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from topic4_fig5_z_filtered_guide import make
from topic4_fig5_z_branch_dynamics import accelerate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_D_separatrix_exploration_20260916'
OLD = ROOT / 'results/topic4_sef_hfo/fig5_z_branch_extension_20260915/deterministic'
D3 = 0.228844760565


def write(name, value):
    tmp = OUT / (name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(OUT / name)


class CachedProduct:
    def __init__(self, batch, key, moment, member):
        self.batch, self.key, self.moment, self.member = batch, key, moment, member

    def __matmul__(self, vector):
        return self.batch.products[self.key][self.moment, :, self.member]


class Batch:
    def __init__(self, eq, states, device):
        self.device = f'cuda:{device}'
        self.models = []
        self.products = {}
        self.matrices = {}
        torch.set_num_threads(1)
        for key in eq.m.ops:
            a = sparse.vstack([eq.m.ops[key], eq.m.vops[key]], format='csr')
            self.matrices[key] = torch.sparse_csr_tensor(
                torch.as_tensor(a.indptr, device=self.device),
                torch.as_tensor(a.indices, device=self.device),
                torch.as_tensor(a.data, device=self.device), size=a.shape)
        for i, state in enumerate(states):
            m = copy.copy(eq.m)
            m.load_state_dict(state)
            assert m.pending is None and not m.freeze_m
            m.ops = {k: CachedProduct(self, k, 0, i) for k in self.matrices}
            m.vops = {k: CachedProduct(self, k, 1, i) for k in self.matrices}
            accelerate(m)
            self.models.append(m)
        self.nu = np.full(eq.m.n, eq.m.nu_sig)

    def step(self):
        histories = {
            'E': torch.as_tensor(np.stack([m.hE.ravel() for m in self.models], 0), device=self.device).T,
            'I': torch.as_tensor(np.stack([m.hI.ravel() for m in self.models], 0), device=self.device).T}
        for key, mat in self.matrices.items():
            h = histories['E' if key in ('ee', 'ie') else 'I']
            self.products[key] = torch.sparse.mm(mat, h).cpu().numpy().reshape(2, -1, len(self.models))
        for m in self.models:
            m.step(self.nu, m.nu_sig)


def state_error(a, b):
    errors = {}
    for key in a:
        if isinstance(a[key], np.ndarray):
            errors[key] = float(np.max(np.abs(a[key] - b[key])))
        elif key == 'y':
            errors[key] = max(float(np.max(np.abs(a[key][k] - b[key][k]))) for k in a[key])
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--duration-s', type=float, default=6.)
    p.add_argument('--qa-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    eq = make(False)
    sources = {'burst': OLD/'s0_extension_end.pkl', 'tonic': OLD/'s0.228845_end.pkl'}
    specs = [dict(name=f'D{D:.6f}_{history}', D=D, history=history, source=str(sources[history]))
             for D in (.1, .16, D3) for history in sources]
    states = []
    for spec in specs:
        file = OUT / (spec['name']+'_end.pkl') if a.resume else Path(spec['source'])
        with file.open('rb') as f:
            state = pickle.load(f)
        state['z_u'], state['z2_u'] = eq.z(spec['D'])
        state['freeze_m'] = False
        states.append(state)
    batch = Batch(eq, states, a.gpu)
    if not a.resume:
        refs = []
        for state in states:
            ref = copy.copy(eq.m)
            ref.load_state_dict(state)
            accelerate(ref)
            refs.append(ref)
        start = time.time()
        for _ in range(30):
            batch.step()
            for ref in refs:
                ref.step(batch.nu, ref.nu_sig)
        errors = [state_error(m.state_dict(), ref.state_dict()) for m, ref in zip(batch.models, refs)]
        maxerr = max(max(e.values()) for e in errors)
        assert maxerr < 1e-8, maxerr
        start = time.time()
        for _ in range(200):
            batch.step()
        benchmark = (time.time()-start)/200
        write('batch_qa.json', dict(steps=30, conditions=specs, full_state_max_errors=errors,
                                   max_error=maxerr, seconds_per_batch_step=benchmark,
                                   time_step_ms=eq.m.dt, update='original ReducedModel.step',
                                   arithmetic='float64 sparse matrix-matrix products'))
        print('QA_PASS', maxerr, 'seconds_per_batch_step', benchmark, flush=True)
        for m, state in zip(batch.models, states):
            m.load_state_dict(state)
        if a.qa_only:
            return
    names = ['global_E', 'core_A', 'core_B', 'surround', 'global_I', 'mean_M', 'active_E_fraction', 'spatial_rate_sd']
    samples = [[] for _ in specs]
    cells = [[] for _ in specs]
    start = time.time()
    steps = round(a.duration_s*1000/eq.m.dt)
    write('status.json', dict(status='RUNNING', pid=os.getpid(), conditions=specs,
                              duration_s=a.duration_s, resume=a.resume))
    def save(elapsed):
        for spec, m, rows, spatial in zip(specs, batch.models, samples, cells):
            file = OUT/(spec['name']+'.npz')
            data = np.asarray(rows)
            rates = np.asarray(spatial)
            offset = 0.
            if a.resume and file.exists():
                old = np.load(file)
                offset = float(old['time_s'][-1])
                data = np.concatenate([old['readouts'], data])
                rates = np.concatenate([old['cell_E_hz'], rates])
            temp = OUT/(spec['name']+'.tmp.npz')
            np.savez_compressed(temp, D=spec['D'], readouts=data, readout_names=names,
                                time_s=np.arange(1, len(data)+1)*.001, cell_E_hz=rates,
                                cell_time_s=np.arange(1, len(rates)+1)*.01)
            temp.replace(file)
            with (OUT/(spec['name']+'_end.pkl')).open('wb') as f:
                pickle.dump(m.state_dict(), f)
        return offset
    for k in range(steps):
        batch.step()
        if k % 10 == 9:
            for m, rows in zip(batch.models, samples):
                re = m.cell_rate_e()*1000
                global_r = np.average(re, weights=m.count_e)
                rows.append([global_r, *[m.region_rate(re, f'175_{j}') for j in range(3)],
                             np.average(m.r_i, weights=m.count_i)*1000,
                             np.average(m.m_u, weights=m.unit_count),
                             np.average(re > 5., weights=m.count_e),
                             np.sqrt(np.average((re-global_r)**2, weights=m.count_e))])
        if k % 100 == 99:
            for m, rows in zip(batch.models, cells):
                rows.append((m.cell_rate_e()*1000).astype(np.float32))
        if k % 5000 == 4999:
            elapsed = (k+1)*eq.m.dt/1000
            latest = [np.asarray(rows)[-500:, 0].mean() for rows in samples]
            print('PROGRESS', elapsed, 's', 'wall_s', round(time.time()-start, 1),
                  'last_500ms_mean_Hz', np.round(latest, 3).tolist(), flush=True)
            write('progress.json', dict(simulated_s=elapsed, wall_s=time.time()-start,
                                        names=[s['name'] for s in specs], last_500ms_mean_Hz=latest))
            if not a.resume:
                save(elapsed)
    save(a.duration_s)
    write('status.json', dict(status='EXECUTION_COMPLETE', active_processes=[], conditions=specs,
                              duration_added_s=a.duration_s, resume=a.resume,
                              wall_s=time.time()-start, scientific_status='ANALYSIS_PENDING'))
    print('COMPLETE', time.time()-start, flush=True)


if __name__ == '__main__':
    main()

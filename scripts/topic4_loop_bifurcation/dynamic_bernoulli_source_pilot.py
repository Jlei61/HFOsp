#!/usr/bin/env python3
"""One physical-time test of conditional Bernoulli recurrent variance.

At dt=0.1ms each native source can spike at most once. For an empirical density
probability p, independent source variance is p(1-p), not p. Cross-source and
cross-time residual covariances remain omitted, and external noise stays as in
the paired previous pilot. This is not a new native biological equation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from campaign import ROOT, read, write, sha
import dynamic_individual_source_pilot as base
import target_density_exit as target

OUT = ROOT/'dynamic_bernoulli_source_pilot'
CODE = target.CODE.split('extern "C" __global__ void target_particles')[0]
assert 'q+=va[i]*r;' in CODE and 'v+=vb[i]*r;' in CODE
CODE = CODE.replace('q+=va[i]*r;', 'q+=va[i]*r*fmax(0.,1.-.1*r);')
CODE = CODE.replace('v+=vb[i]*r;', 'v+=vb[i]*r*fmax(0.,1.-.1*r);')


class BernoulliNetwork(base.DynamicNetwork):
    def __init__(self, replicas, device):
        super().__init__('individual_source', replicas, device)
        self.bernoulli_module = self.cp.RawModule(code=CODE, options=('--fmad=false',), name_expressions=['target_delayed'])
        self.k['target_delayed'] = self.bernoulli_module.get_function('target_delayed')


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    result = read(base.OUT/'analysis/result.json')
    assert all(not r['development_relevance_retained'] for r in result['rows'])
    write(OUT/'contract.json', dict(status='REGISTERED_BEFORE_ONE_VARIANCE_ASSAY', created_epoch=time.time(),
        question='Does replacing Poisson recurrent variance by the conditional independent Bernoulli variance repair the spurious surround recruitment and G activation in the full-source physical dynamics?',
        evidence='Both completed source-identity arms retained core rates but drifted to causalR>200 and spuriously activatedG; individual source tail fieldRMS18.84Hz. Original exact source identity and initial state alone were insufficient.',
        design='Exactly one1s trajectory from the same complete72s high state, same64 replicas, thresholds, fixedpercellnu, randomstreams, heldZ/K and dynamicM/G. Mean arrivals unchanged. Replace only squared-weight rate r by r*(1-dt*r) in recurrent variance, because source probability p=dt*r is0/1 perstep. Reuse completed individual-Poisson arm; no biological parameter adjustment.',
        limitations='Variance belongs to empirical particle probability, not a claim of exact finite-network covariance. Independent Gaussian residuals still omit source/time correlations; external Gaussian input remains unchanged. Finite64 particle error needs separate convergence if relevant. Not an autonomous loop or stable branch.',
        validation='GPU delayed means versus same CPU operators; variance versus W2*(r*(1-dt*r)); nonnegativeand<=Poisson variance, per-cell/group count conservation, eager/captured bitwise. Membrane kernel unchanged from completed paired pilot.',
        decision='Same registered one-second native relevance guards; stop after one trajectory, no automatic root or parameter/horizon extension. Use failure to localize remaining approximation, not relax guards.',
        duration_ms=base.DURATION_MS, replicas=base.REPLICAS, producer_sha256=sha(__file__),
        base_sha256=sha(base.__file__), target_sha256=sha(target.__file__), formal_bifurcation_allowed=False))
    shutil.copy2(__file__, OUT/'producer.py')


def qa(e):
    cp = e.cp;rng = np.random.default_rng(930187)
    h = rng.uniform(0, 10, size=e.source_history.shape);e.source_history[:] = cp.asarray(h);tick = e.depth-1;e.clock[0] = tick;e.arrivals()
    x = h[(tick-np.arange(1, e.depth)) % e.depth].ravel();variance = x*(1-.1*x)
    expected = np.array([e.ops_cpu[0][0]@x, e.ops_cpu[1][0]@x,
        e.ops_cpu[0][1]@variance, e.ops_cpu[1][1]@variance])
    error = float(abs(expected-e.arr.get()).max());assert error < 1e-8
    oldvariance = np.array([e.ops_cpu[0][1]@x, e.ops_cpu[1][1]@x])
    assert (expected[2:] >= 0).all() and (expected[2:] <= oldvariance+1e-12).all()
    e.reset()
    for _ in range(100):e.step()
    keys = ['state', 'ref', 'rng', 'history', 'source_history', 'clock', 'global_state',
        'output', 'global_output', 'source_output', 'source_accumulator', 'accumulator']
    previous = {k: getattr(e, k).get() for k in keys}
    source = previous['source_output'];group = previous['output'][:, 0];g = e.geo['cell_group']
    projected = np.stack([np.bincount(g, weights=a, minlength=e.P)/e.sizes for a in source])
    count_error = float(abs(projected-group).max());assert count_error < 1e-10
    e.graph();e.chunk()
    eq = {k: np.array_equal(v, getattr(e, k).get()) for k, v in previous.items()};assert all(eq.values())
    e.reset()
    return dict(status='PASS', mean_and_Bernoulli_variance_operator_max_error=error,
        variance_nonnegative_and_le_Poisson=True, per_cell_group_count_max_error=count_error,
        captured_bitwise=eq, unchanged_membrane_kernel=True)


def run(device):
    c = read(OUT/'contract.json')
    assert c['producer_sha256'] == sha(__file__) and c['base_sha256'] == sha(base.__file__) and c['target_sha256'] == sha(target.__file__)
    assert not (OUT/'supervisor.json').exists()
    write(OUT/'supervisor.json', dict(status='RUNNING_ONE_VARIANCE_ASSAY', pid=os.getpid(), updated_epoch=time.time()))
    started = time.time();e = BernoulliNetwork(base.REPLICAS, device)
    write(OUT/'implementation_qa.json', qa(e));print('BERNOULLI PHYSICAL QA PASS', flush=True)
    e.graph();groups = [];globals_ = [];cells = []
    for offset in range(0, base.DURATION_MS, 10):
        groups.append(e.chunk().astype('f4'));globals_.append(e.global_output.get());cells.append(e.source_output.get().astype('f4'))
        if (offset+10) % 100 == 0:
            write(OUT/'progress.json', dict(status='RUNNING', pid=os.getpid(), elapsed_simulation_ms=offset+10,
                elapsed_wall_s=time.time()-started, updated_epoch=time.time()))
            print('BERNOULLI PHYSICAL', offset+10, flush=True)
    value, global_, source = np.concatenate(groups), np.concatenate(globals_), np.concatenate(cells)
    assert int(e.clock.get()[0]) == base.DURATION_MS*10
    assert np.isfinite(value).all() and np.isfinite(global_).all()
    assert np.array_equal(e.state.get()[:, :, 6:8], e.initial_state.get()[:, :, 6:8])
    np.savez_compressed(OUT/'trajectory.npz', elapsed_time_ms=np.arange(1, base.DURATION_MS+1), group_output=value,
        global_R_Hz=global_[:, 0], global_s=global_[:, 1], cell_rate_Hz=source)
    np.savez_compressed(OUT/'final_state.npz', state=e.state.get(), ref=e.ref.get(), rng=e.rng.get(),
        source_history=e.source_history.get(), history=e.history.get(), clock=e.clock.get(), global_state=e.global_state.get())
    result = dict(status='COMPLETE_ONE_BERNOULLI_VARIANCE_ASSAY_ANALYSIS_PENDING', duration_ms=base.DURATION_MS,
        replicas=base.REPLICAS, held_fields_bitwise=True, elapsed_s=time.time()-started, formal_bifurcation_allowed=False)
    write(OUT/'result.json', result);write(OUT/'supervisor.json', dict(status='COMPLETE', updated_epoch=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'run']);p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    if a.command == 'prepare':prepare()
    else:
        try:run(a.device)
        except Exception:
            write(OUT/'supervisor.json', dict(status='FAILED', pid=os.getpid(), updated_epoch=time.time()));raise

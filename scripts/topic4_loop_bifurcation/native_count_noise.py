#!/usr/bin/env python3
"""Two native continuations changing only Poisson counts, with paired OU path.

The original generator still consumes its own Poisson draw, so all subsequent
global OU increments remain identical. Only the returned external count is
replaced. No neural, slow-state, connectivity or mean-input equation changes.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import pickle
import shutil
import time
import numpy as np
from campaign import ROOT, NATIVE, read, write, sha
from reconstruct_loop_future_drive import SOURCE, INITIAL
import run_topic4_loop_zk_conditional as native
import run_topic4_loop_cuda_override as gpu
from run_topic4_recovery_window import assert_same_state

OUT = ROOT / 'native_count_noise'
SEEDS = [928701, 928702]


class CountRng:
    def __init__(self, original, seed):
        self.original = original
        self.count_rng = np.random.default_rng(seed) if seed is not None else None
        self.calls = 0
        self.expected_sum = 0.
        self.observed_sum = 0.

    def __getattr__(self, key):
        return getattr(self.original, key)

    def poisson(self, lam, size=None):
        reference = self.original.poisson(lam, size=size)
        value = reference if self.count_rng is None else self.count_rng.poisson(lam, size=size)
        self.calls += 1
        self.expected_sum += float(np.broadcast_to(lam, value.shape).sum())
        self.observed_sum += float(value.sum())
        return value


def check_rng():
    tests = []
    for seed in [None, SEEDS[0]]:
        a = np.random.default_rng(928700)
        proxy = CountRng(np.random.default_rng(928700), seed)
        different = 0
        for tick in range(250):
            assert a.standard_normal() == proxy.standard_normal()
            lam = np.linspace(.1, 2., 40000)
            x, y = a.poisson(lam), proxy.poisson(lam)
            assert a.bit_generator.state == proxy.bit_generator.state
            if seed is None:
                assert np.array_equal(x, y)
            else:
                different += int(not np.array_equal(x, y))
        error = (proxy.observed_sum - proxy.expected_sum) / np.sqrt(proxy.expected_sum)
        assert abs(error) < 8.
        assert seed is None or different == 250
        tests.append(dict(seed=seed, original_rng_after_counts_bitwise=True,
                          subsequent_normals_bitwise=True, changed_count_arrays=different,
                          total_count_standardized_error=error))
    return dict(status='PASS', tests=tests, scope='RNG routing and count mean only; no dynamical acceptance.')


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'contract.json').exists(), 'Do not redefine a frozen experiment'
    assert read(ROOT / 'coupled_density_exit/forcing_qa.json')['status'] == 'PASS'
    qa = check_rng()
    write(OUT / 'rng_routing_qa.json', qa)
    write(OUT / 'contract.json', dict(
        status='REGISTERED_BEFORE_NEW_NATIVE_TRAJECTORIES', created_epoch=time.time(),
        question='Can finite external-count noise alone move the first autonomous exit around the 13.7s dip, with identical native10s fullstate and external OU mean path?',
        motivation='Coupled density927671 matches native10-12s mean rate/Z/K/G closely, but exits13.750s versus original16.868s and reactivates. Two new nativecount realizations distinguish strong conditional native variability from a systematic candidate-coupling discrepancy; they do not estimate an ensemble distribution precisely.',
        design='Exactlytwo originalnative10-20s continuations. Same graph/cells/fulljointinitialstate/pendingdelays, same0.1ms globalandspatialOU path. Only Poissonexternalcounts replaced, seeds928701/928702. Allnetwork,G/R,Z/M/K states autonomous.',
        method='Originalrng still consumes anddiscards originalPoissondraw; separatecountgenerator supplies newdraw with identicallambda. OriginalnormalincrementsandspatialOU remainpaired. No simulator source edits.',
        checks='RNG wrapper deterministic unitcheck; exact100 sparse originalinputrecords and completefinalglobalrng,xi,spatialOU field/cache/RNG/clocks; owncounterstate retainedat checkpoints. Verify countdrawcount100000 and matchedsumexpectedintensity.',
        readouts='Absolute10-20s traces, first100ms causalR<=5, first100ms joint10msallE/coreA/coreB<=5, G resource-blockrelease, 2s meanfields andZ/K endpoints; samecollector asdensity, no phasealignment.',
        decisions='Nativeexit near13.7s supports countnoise sensitivity, but doesnot certify density or bifurcation. Bothnativecontinuations nearoriginal16.9s strengthens coupled-approximation concern. No automatic seed/grid/horizon expansion or refit.',
        statistical_unit='Same nativecondition with two new externalcountstreams; original8405 third countstream. Not three independentOU trajectories, autonomouscoldstarts,or patientreplicates.',
        source_checkpoint=str(INITIAL), source_sha256=sha(INITIAL), seeds=SEEDS,
        producer_sha256=sha(__file__), native_engine_unchanged=True,
        formal_bifurcation_allowed=False))
    protocol = copy.deepcopy(read(NATIVE / 'protocol.json'))
    protocol.update(stage='PAIRED_OU_NATIVE_COUNT_NOISE', created_epoch=time.time(),
                    deadline_epoch=time.time()+86400)
    write(OUT / 'protocol.json', protocol)
    shutil.copy2(NATIVE / 'geometry.npz', OUT / 'geometry.npz')
    native.OUT = OUT
    native.prepare = lambda: protocol
    for seed in SEEDS:
        name = f'count{seed}'
        native.make_job(name, str(INITIAL), 10., clamp=False, common_input=False)
    write(OUT / 'queue.json', dict(names=[f'count{s}' for s in SEEDS], total=2,
          job_sha256={f'count{s}': sha(OUT / 'jobs' / f'count{s}.json') for s in SEEDS},
          bounded=True, no_automatic_restart=True))


def worker(seed, device):
    assert seed in SEEDS
    contract = read(OUT / 'contract.json')
    assert sha(__file__) == contract['producer_sha256']
    assert sha(INITIAL) == contract['source_sha256']
    name = f'count{seed}'
    folder = OUT / 'runs' / name
    assert not (folder / 'count_noise_progress.json').exists(), 'No silent restart'
    assert sha(OUT / 'jobs' / f'{name}.json') == read(OUT / 'queue.json')['job_sha256'][name]
    sparse = {}
    for path in sorted((SOURCE / 'chunks').glob('*.npz')):
        a, b = map(int, path.stem.split('_'))
        if b <= 100000 or a >= 200000:
            continue
        with np.load(path) as z:
            for row in z['inputs']:
                sparse[round(row[0]*10)] = row.copy()
    backend0 = gpu.cuda_backend.wrap_simulator
    runtime = {}

    def paired_backend(original, device_index):
        fast = backend0(original, device_index=device_index)
        def simulate(params, net, *args, **kw):
            assert kw['resume_state']['step'] == 100000
            proxy = CountRng(net['rng'], seed)
            paired_net = dict(net)
            paired_net['rng'] = proxy
            observer = kw.get('input_observer')
            seen = 0

            def inputs(tm, nu, xi):
                nonlocal seen
                if observer is not None:
                    observer(tm, nu, xi)
                tick = round(tm*10)
                if tick in sparse:
                    actual = np.array([tm, xi, nu[:32000].mean(), nu[32000:].mean()])
                    assert np.array_equal(actual, sparse[tick]), (tick, actual-sparse[tick])
                    seen += 1
            kw['input_observer'] = inputs
            import checkpoint
            capture0 = checkpoint.capture

            def capture(*aa, **kk):
                state = capture0(*aa, **kk)
                state['paired_external_count_noise'] = dict(seed=seed,
                    rng_state=copy.deepcopy(proxy.count_rng.bit_generator.state),
                    calls=proxy.calls, expected_sum=proxy.expected_sum,
                    observed_sum=proxy.observed_sum, paired_sparse_records=seen)
                return state
            checkpoint.capture = capture
            try:
                return fast(params, paired_net, *args, **kw)
            finally:
                checkpoint.capture = capture0
                runtime.update(count_calls=proxy.calls, sparse_records=seen,
                               expected_count_sum=proxy.expected_sum,
                               observed_count_sum=proxy.observed_sum)
                write(folder / 'count_noise_runtime.json', runtime)
        return simulate

    gpu.cuda_backend.wrap_simulator = paired_backend
    write(folder / 'count_noise_progress.json', dict(status='RUNNING', pid=os.getpid(),
          seed=seed, device=device, interval_s=[10, 20], started_epoch=time.time()))
    try:
        gpu.worker(OUT, name, device)
    finally:
        gpu.cuda_backend.wrap_simulator = backend0
    assert read(folder / 'result.json')['status'] == 'COMPLETE'
    with (folder / 'checkpoint.pkl').open('rb') as f:
        state = pickle.load(f)['engine']
    with (SOURCE / 'states/t20s.pkl').open('rb') as f:
        reference = pickle.load(f)['engine']
    for key in ['rng_state', 'xi', 'external_drive']:
        assert_same_state(state[key], reference[key])
    assert runtime['count_calls'] == 100000 and runtime['sparse_records'] == 100, runtime
    countstate = state['paired_external_count_noise']
    assert countstate['calls'] == 100000 and countstate['seed'] == seed
    error = (runtime['observed_count_sum']-runtime['expected_count_sum'])/np.sqrt(runtime['expected_count_sum'])
    assert abs(error) < 8., error
    audit = dict(status='PASS', count_seed=seed, unchanged_expected_input=True,
        final_global_rng_xi_bitwise=True, final_complete_spatial_OU_bitwise=True,
        sparse_original_input_records_bitwise=100, count_draws=100000,
        standardized_count_error=error, counts_state_saved=True,
        formal_bifurcation_allowed=False)
    write(folder / 'paired_input_audit.json', audit)
    write(folder / 'count_noise_progress.json', dict(status='COMPLETE_INPUT_QA_PASS',
          seed=seed, ended_epoch=time.time()))
    print('PAIRED COUNT NOISE COMPLETE', seed, audit, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'worker'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        worker(args.seed, args.device)

#!/usr/bin/env python3
"""Exact common external mean input for the existing native exit-field cuts."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import pickle
import time
import numpy as np
from campaign import ROOT, read, write, sha
from reconstruct_loop_future_drive import SOURCE
from coupled_density_exit import ADAPTED
import run_topic4_loop_zk_conditional as native
from checkpoint import restore_external_drive
from run_topic4_recovery_window import assert_same_state

OUT = ROOT / 'exit_branch_density'
INITIAL = SOURCE / 'states/t50s.pkl'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'forcing_contract.json').exists()
    write(OUT / 'forcing_contract.json', dict(status='REGISTERED_BEFORE_RECONSTRUCTION', created_epoch=time.time(),
        question='Pair the first10s externalmeaninput of allfour existing actual-exitfield nativecuts, which share originalsource50s externalstate.',
        scope='Exactlysource50-60s, 0.1msfloat64means for3479g40groups; no neural simulation. Preserve originalOU/Poisson RNGconsumption as in validated10-20s reconstruction.',
        checks='Every100ms originalsourceglobal/inputmeans bitwise, complete60s globalRNG/xi andspatialOU state/cache/RNG/clocks bitwise, pairednativecut sparseinputs withcorrectabsoluteclock shifts.',
        source=str(INITIAL), initial_sha256=sha(INITIAL), producer_sha256=sha(__file__)))
    start = time.time()
    setup, trace, _, identity = native.base.old.setup(9108405)
    p = setup.params
    assert identity == read(ROOT / 'native_slices/protocol.json')['identity']
    geo = dict(np.load(ADAPTED / 'geometry.npz'))
    assert np.array_equal(geo['original_positions'], setup.net['pos'])
    with INITIAL.open('rb') as handle:
        initial = pickle.load(handle)['engine']
    assert initial['step'] == 500000
    rng = np.random.default_rng()
    rng.bit_generator.state = initial['rng_state']
    xi = initial['xi']
    spatial = native.base.old.make_external_drive(setup, trace['spatial_ou'], 9108405)
    restore_external_drive(initial, spatial)
    nu = p.nu_ext_ratio * native.base.old.simulate_kick.__globals__['compute_nu_theta'](p)[0]
    aa = np.exp(-p.dt/p.tau_n)
    sigma = p.sigma_n*1e-3*np.sqrt(p.tau_n/2.)
    bb = sigma*np.sqrt(1-aa*aa)
    drive = np.lib.format.open_memmap(OUT/'drive_0p1ms.npy', mode='w+', dtype='f8', shape=(100000, len(geo['group_size'])))
    source_records = {}
    for path in sorted((SOURCE/'chunks').glob('*.npz')):
        a, b = map(int, path.stem.split('_'))
        if b <= 500000 or a >= 600000:
            continue
        with np.load(path) as z:
            for row in z['inputs']:
                source_records[round(row[0]*10)] = row.copy()
    cut_records = {}
    for k in [9, 12]:
        for history in ['high', 'recovery']:
            name = f'exit_z0.21_k{k}_fields16p7_{history}'
            folder = ROOT/'exit_return_probes/runs'/name
            job = read(ROOT/'exit_return_probes/jobs'/f'{name}.json')
            origin = job['branch_start_s']*1000
            records = {}
            for path in sorted((folder/'chunks').glob('*.npz')):
                with np.load(path) as z:
                    for row in z['inputs']:
                        offset = round((row[0]-origin)*10)
                        if 0 <= offset < 100000:
                            records[offset] = row[1:].copy()
            assert len(records) == 100, (name, len(records))
            cut_records[name] = records
    checked = 0
    for offset in range(100000):
        tick = 500000+offset
        tm = tick*.1
        xi = aa*xi + bb*rng.standard_normal()
        vec = np.full(40000, max(0., nu+xi))
        vec[:32000] = np.maximum(vec[:32000]+spatial.step(tm), 0.)
        rng.poisson(vec*.1, size=40000)
        drive[offset] = np.bincount(geo['cell_group'], weights=vec, minlength=len(geo['group_size']))/geo['group_size']
        if tick in source_records:
            actual = np.array([tm, xi, vec[:32000].mean(), vec[32000:].mean()])
            assert np.array_equal(actual, source_records[tick]), tick
            for name, rows in cut_records.items():
                assert np.array_equal(actual[1:], rows[offset]), (name, tick)
            checked += 1
        if (offset+1) % 5000 == 0:
            write(OUT/'forcing_progress.json', dict(status='RUNNING', pid=os.getpid(), source_time_s=(tick+1)*.0001, elapsed_s=time.time()-start))
    with (SOURCE/'states/t60s.pkl').open('rb') as handle:
        target = pickle.load(handle)['engine']
    assert xi == target['xi'] and rng.bit_generator.state == target['rng_state']
    saved = target['external_drive']
    assert np.array_equal(spatial._state, saved['field_state'])
    assert np.array_equal(spatial._cached, saved['cached'])
    assert_same_state(spatial._rng.bit_generator.state, saved['rng_state'])
    assert spatial._next_step == saved['next_step'] and spatial._last_step == saved['last_step']
    assert checked == 100
    drive.flush()
    result = dict(status='PASS', source_interval_s=[50, 60], candidate_elapsed_s=[0, 10],
        groups=len(geo['group_size']), samples=100000, sampling_ms=.1, dtype='float64',
        source_sparse_input_records_bitwise=checked, native_cut_records_bitwise=4*checked,
        complete60s_global_rng_xi_bitwise=True, complete60s_spatial_OU_bitwise=True,
        neural_simulations=0, elapsed_s=time.time()-start, producer_sha256=sha(__file__))
    write(OUT/'forcing_qa.json', result)
    write(OUT/'forcing_progress.json', result)
    print('EXIT BRANCH FORCING PASS', result, flush=True)


if __name__ == '__main__':
    main()

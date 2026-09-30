#!/usr/bin/env python3
"""Two bounded cold-start spatial-axis controls of the unchanged autonomous law."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy import sparse
import run_topic4_loop_zk_conditional as native
from src.topic4_multidimensional_parameters import sparse_digest
from src.topic4_rev20_dual_core_mechanism import _invalidate_ampa_caches
from src.topic4_cuda_ordered_scatter import wrap_simulator

ROOT = native.OUT/'axis_controls'
OUT = ROOT/'native_runs'


def graph_folder(condition):
    return ROOT/'reference' if condition == 'reference' else ROOT/'angular_reassignment'/condition


def prepare(condition, qa=False):
    folder = OUT/condition
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'protocol.json'
    if path.exists():
        return native.base.read(path)
    audit = native.base.read(graph_folder(condition)/'audit.json')
    if condition != 'reference':
        assert audit['status'] == 'STATIC_CONTROL_PASS'
    p = copy.deepcopy(native.base.read(native.OUT/'protocol.json'))
    p['identity']['ampa_topology_sha256'] = audit['ampa_topology_sha256']
    p['identity']['ampa_values_sha256'] = audit['ampa_values_sha256']
    p.update(stage='AUTONOMOUS_AXIS_CONTROL', axis_condition=condition,
        created_epoch=time.time(), deadline_epoch=time.time()+7*86400,
        axis_runner_sha256=native.base.sha(__file__),
        graph_audit=str(graph_folder(condition)/'audit.json'),
        graph_audit_sha256=native.base.sha(graph_folder(condition)/'audit.json'),
        question='Does changing the spatial EE axis alter native interictal propagation and autonomous entry/exit/return under the same fatigue/feedback law?',
        initial_jobs=[], branch_jobs=[], max_workers=1,
        diagnostic_only=False, counts_as_autonomous_loop=True,
        finite_horizon_s=.2 if qa else 120.,
        source_identity=audit['source_identity'],
        paired_noise='Same cold-start dynamics seed9108405 and original exogenous drive; verify external records. Endogenous trajectory may differ.',
        matched_control=audit.get('matched', ['Exact original graph reloaded from the audited cache']),
        changed_control=audit.get('changed', []),
        caveat='Rotation achieved88.3deg centrally but reduced axis ratio1.936 to1.577; outgoing degree is not matched. These are model structure controls, not patient connectome estimates.',
        acceptance='Report original0-8s interictal spatial events, high entry, autonomous exit, nativeZ recovery and subsequent brief core recruitment separately. No requirement that all structural conditions cycle. No automatic parameter expansion.',
        inference='Only two new120s trajectories, one fixed seed per structural condition. Finite-time transitions and spatial observations; not a certified bifurcation.')
    old = native.base.read(native.SOURCE/'runs'/native.NAME/'result.json')['job']
    job = copy.deepcopy(old)
    job.update(name='reference_loader_qa' if qa else f'{condition}_s9108405',
        horizon_s=.2 if qa else 120., checkpoint_s=.2 if qa else 2.,
        device=1, backend='cpu' if qa else 'cuda_ordered', stage='axis_control',
        qa=False, stop_after_second_entry=False, conditional_clamp=False,
        common_exogenous_input=False, branch_start_s=0., diagnostic_only=False,
        axis_condition=condition)
    p['initial_jobs'] = [job]
    native.write(folder/'jobs'/f'{job["name"]}.json', job)
    shutil.copy2(native.SOURCE/'geometry.npz', folder/'geometry.npz')
    native.write(path, p)
    return p


def worker(condition):
    p = prepare(condition, qa=condition == 'reference')
    assert p['axis_runner_sha256'] == native.base.sha(__file__)
    assert p['graph_audit_sha256'] == native.base.sha(p['graph_audit'])
    job = p['initial_jobs'][0]
    folder = OUT/condition
    setup0 = native.carrier.base.old.setup
    def setup(seed):
        s, tr, frozen, identity = setup0(seed)
        assert identity == p['source_identity']
        bins = [sparse.load_npz(path) for path in sorted((graph_folder(condition)/'ampa_by_delay').glob('*.npz'))]
        assert len(bins) == len(s.net['ampa_by_delay'])
        assert sparse_digest(bins) == p['identity']['ampa_values_sha256']
        assert sparse_digest(bins, topology=True) == p['identity']['ampa_topology_sha256']
        # All E->I edges must be unchanged, in addition to GABA and thresholds.
        for old, new in zip(s.net['ampa_by_delay'], bins):
            a, b = old.tocsr()[s.n_e:], new.tocsr()[s.n_e:]
            assert np.array_equal(a.indptr,b.indptr) and np.array_equal(a.indices,b.indices)
            assert np.array_equal(a.data,b.data)
        s.net = dict(s.net)
        s.net['ampa_by_delay'] = bins
        removed = _invalidate_ampa_caches(s.net)
        native.write(folder/'runs'/job['name']/'graph_loader_qa.json',
            dict(status='PASS', identity=p['identity'], all_nonEE_unchanged=True,
                 invalidated_caches=removed, cold_start=True, no_state_transfer=True))
        return s, tr, frozen, copy.deepcopy(p['identity'])
    native.carrier.base.old.setup = setup
    native.OUT = folder
    native.prepare = lambda: p
    if job['backend'] == 'cuda_ordered':
        fixed0 = native.fixed.worker
        def gpu_worker(name):
            native.carrier.wrap_simulator = wrap_simulator
            return fixed0(name)
        native.fixed.worker = gpu_worker
    native.worker(job['name'])
    result = native.base.read(folder/'runs'/job['name']/'result.json')
    result.update(diagnostic_only=condition == 'reference', counts_as_autonomous_loop=condition != 'reference',
                  no_external_intervention=True, display_stop_s=result['end_s'],
                  axis_control=True, cold_start=True)
    if result['status'] == 'COMPLETE':
        result['tracker']['stop_reason'] = 'SIMULATION_HORIZON'
    native.write(folder/'runs'/job['name']/'result.json', result)
    native.write(folder/'runs'/job['name']/'progress.json', result)
    if condition == 'reference':
        native.qa_compare(job['name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'worker'])
    parser.add_argument('condition', choices=['reference', 'rotated', 'isotropic'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.condition, qa=args.condition == 'reference')
    else:
        worker(args.condition)

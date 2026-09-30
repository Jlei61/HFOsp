#!/usr/bin/env python3
"""Three local native cuts; reuses the frozen spatial conditional engine."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import shutil
import time
import numpy as np
from campaign import REPO, ROOT, PREVIOUS, NATIVE, read, write, sha
import run_topic4_loop_zk_conditional as native
import run_topic4_loop_cuda_override as gpu
from run_topic4_recovery_window import assert_same_state

CUTS = [
    dict(label='entry', points=[dict(Z=z, K=.0002) for z in [.65, .70, .74, .78, .85]],
         histories=[('high', 'entry1_checkpoint.pkl'), ('interictal', 't50s.pkl')]),
    dict(label='exit', points=[dict(Z=.21, K=k) for k in [6., 9., 12., 15., 18.]],
         histories=[('high', 'entry1_checkpoint.pkl'), ('recovery', 't30s.pkl')]),
    dict(label='return', points=[dict(Z=.995, K=k) for k in [.003, .01, .03, .1, .3]],
         histories=[('recovery', 't30s.pkl'), ('interictal', 't50s.pkl')]),
]
STREAMS = ['chunks', 'actual_current_chunks', 'conditional_drift_chunks',
           'feedback_chunks', 'global_response_chunks', 'intrinsic_adaptation_chunks',
           'mechanism_chunks', 'regional_chunks']


def protocol(root, qa=False):
    path = root / 'protocol.json'
    if path.exists():
        return read(path)
    root.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(read(PREVIOUS / 'protocol.json'))
    for source, digest in p['source_hashes'].items():
        assert sha(source) == digest, source
    assert sha(native.__file__) == p['runner_sha256']
    p.update(created_epoch=time.time(), deadline_epoch=time.time() + 7*86400,
             authorization='2026-09-27 user: 把这条线设置goal继续做完，做到能出新的分岔图，能有完整的机制解释。注意时刻反思不要走偏',
             stage='LOCAL_ENTER_EXIT_RETURN_CUTS' if not qa else 'EXISTING_ENGINE_ROUTE_QA',
             question='Locate conditional changes near actual entry, termination and short-event return; distinguish history/G dependence and stochastic excitability from certified bifurcations.',
             cuts=CUTS, maximum_grid_runs=30 if not qa else 2,
             max_workers=4, initial_jobs=[], branch_jobs=[],
             analysis_contract=str(ROOT / 'execution_contract.md'),
             local_preparer_sha256=sha(__file__),
             scientific_template_limit='Same native t20 spatial Z/K template as previous grid; it is a declared conditional family, not every natural trajectory field. Actual-stage spatial-template and G-history checks are needed before interpreting these cuts as a complete natural transition boundary.',
             backend='Previously validated original CUDA ordered scatter, execution override device 0 or 1; unchanged physical updates.',
             counts_as_autonomous_loop=False)
    write(path, p)
    shutil.copy2(PREVIOUS / 'geometry.npz', root / 'geometry.npz')
    return p


def configure(root):
    p = protocol(root, root.name == 'route_qa')
    native.OUT = root
    native.prepare = lambda: p
    return p


def prepare(qa=False):
    root = ROOT / 'route_qa' if qa else NATIVE
    if (root / 'queue.json').exists():
        return read(root / 'queue.json')
    if not qa:
        assert read(ROOT / 'route_qa/gate.json')['status'] == 'PASS'
    p = configure(root)
    names = []
    if qa:
        rows = [('qa_high', 'entry1_checkpoint.pkl', .75, 2., 'qa', 'high'),
                ('qa_interictal', 't50s.pkl', .75, 2., 'qa', 'interictal')]
    else:
        # Interleave the three scientific questions in the dispatch order.
        rows = []
        for index in [2, 0, 4, 1, 3]:
            for cut in CUTS:
                point = cut['points'][index]
                for history, state in cut['histories']:
                    z, k = point['Z'], point['K']
                    rows.append((f"{cut['label']}_z{z:g}_k{k:g}_{history}", state,
                                 z, k, cut['label'], history))
    for name, state, z, k, cut, history in rows:
        job = native.make_job(name, state, .2 if qa else 30., True, z, k, common_input=True)
        checkpoint = root / 'runs' / name / 'checkpoint.pkl'
        saved = native.read_pickle(checkpoint)
        job.update(local_cut=cut, source_history=history, counts_as_autonomous_loop=False,
                   field_template=p['field_template'], native_conditional_only=True)
        saved['job'] = job
        native.base.save_pickle(checkpoint, saved)
        write(root / 'jobs' / f'{name}.json', job)
        zz, kk = native.fields(z, k)
        assert np.array_equal(saved['engine']['slow']['z'][:32000], zz)
        assert np.array_equal(saved['engine']['termination_mechanism']['sahp_g'], kk)
        assert saved['identity'] == p['identity']
        names.append(name)
    queue = dict(names=names, total=len(names), bounded=True, diagnostic_only=True,
                 job_sha256={name: sha(root / 'jobs' / f'{name}.json') for name in names})
    write(root / 'queue.json', queue)
    return queue


def worker(name, device, qa=False):
    root = ROOT / 'route_qa' if qa else NATIVE
    p = configure(root)
    assert sha(__file__) == p['local_preparer_sha256']
    assert sha(root / 'jobs' / f'{name}.json') == read(root / 'queue.json')['job_sha256'][name]
    gpu.worker(root, name, device)


def verify():
    rows = []
    for name, old in [('qa_high', 'clamp_mechanism_qa'), ('qa_interictal', 'clamp_input_qa')]:
        new = ROOT / 'route_qa/runs' / name
        previous = PREVIOUS / 'runs' / old
        assert read(new / 'result.json')['status'] == 'COMPLETE'
        assert_same_state(native.read_pickle(new / 'checkpoint.pkl')['engine'],
                          native.read_pickle(previous / 'checkpoint.pkl')['engine'])
        arrays = 0
        for stream in STREAMS:
            left, right = sorted((new / stream).glob('*.npz')), sorted((previous / stream).glob('*.npz'))
            assert [p.name for p in left] == [p.name for p in right], stream
            for a, b in zip(left, right):
                with np.load(a) as x, np.load(b) as y:
                    assert set(x.files) == set(y.files)
                    for key in x.files:
                        if x[key].dtype.kind in 'fc':
                            assert np.array_equal(x[key], y[key], equal_nan=True), (stream, key)
                        else:
                            assert np.array_equal(x[key], y[key]), (stream, key)
                        arrays += 1
        rows.append(dict(name=name, full_engine_bitwise=True, arrays_bitwise=arrays,
                         runtime=read(new / 'runtime_backend.json')))
    gate = dict(status='PASS', rows=rows, science_runner_sha256=sha(native.__file__),
                preparer_sha256=sha(__file__), verified_epoch=time.time())
    write(ROOT / 'route_qa/gate.json', gate)
    print(json.dumps(gate), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'worker', 'verify'])
    parser.add_argument('--qa', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('--device', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.qa)), flush=True)
    elif args.command == 'worker':
        worker(args.name, args.device, args.qa)
    else:
        verify()

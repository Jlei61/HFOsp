#!/usr/bin/env python3
"""Bounded native conditional Z/K slice, separate from autonomous-loop evidence."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import run_topic4_global_feedback_response as response

SOURCE = response.OUT
NAME = 'G30_response0.5_s9108405'
OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
fixed, parent, base = response.fixed, response.parent, response.base
carrier = fixed.carrier
write = response.write


def read_pickle(path):
    with path.open('rb') as f:
        return pickle.load(f)


def fields(zbar, kbar):
    state = read_pickle(SOURCE/'runs'/NAME/'states/t20s.pkl')['engine']
    z = state['slow']['z'][:32000]
    # Monotone bounded transformation preserves spatial ordering, not its amplitude.
    logits = np.log(z)-np.log1p(-z)
    lo, hi = -40., 40.
    for _ in range(80):
        mid = (lo+hi)/2
        if np.mean(1/(1+np.exp(-(logits+mid)))) < zbar:
            lo = mid
        else:
            hi = mid
    zz = 1/(1+np.exp(-(logits+(lo+hi)/2)))
    kk = state['termination_mechanism']['sahp_g'].copy()
    kk *= kbar/kk.mean()
    assert abs(zz.mean()-zbar) < 1e-13 and abs(kk.mean()-kbar) < 1e-13
    return zz, kk


class ConditionalSlow(response.GlobalResponseSlow):
    clamp = False
    target_z = None
    target_k = None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.conditional_records = []
        self.drift_sum = np.zeros((4, 2))
        self.drift_steps = 0

    def step(self, spk, labels, dt):
        if not self.clamp:
            return super().step(spk, labels, dt)
        # Read the counterfactual native drift before advancing the unclamped states.
        target = self._I_I_last[:self.NE] < self.cfg.I_th_EI
        dz = (target.astype(float)-self.z[:self.NE])/self.cfg.tau_z*1000.
        tau = self.off_tau_ms if self.r_global <= self.retention_below_Hz else self.sahp_tau_ms
        dk = self.g_k*(np.exp(-dt/tau)-1.)
        dk[spk[:self.NE]] += .01*self.sahp_gain*self.gate
        self.drift_sum[:, 0] += self.means(dz)
        self.drift_sum[:, 1] += self.means(dk*1000./dt)
        self.drift_steps += 1
        q = self.gate
        # Original M, refractory-independent slow bookkeeping and causal R_G.
        # Z is then restored to the prescribed field; K is held, not reset by a timer.
        parent.NativeTerminationStep(self, spk, labels, dt)
        self.z[:self.NE] = self.target_z
        self.g_k[:] = self.target_k
        decay = np.exp(-dt/self.global_tau_ms)
        self.global_state = self.global_state*decay+q*(1.-decay)
        if self._step_index % 200 == 0:
            self.conditional_records.append((self._step_index*.1, self.drift_sum/self.drift_steps))
            self.drift_sum.fill(0.)
            self.drift_steps = 0


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        return base.read(path)
    p = copy.deepcopy(base.read(SOURCE/'protocol.json'))
    for file, digest in p['source_hashes'].items():
        assert base.sha(file) == digest, file
    p.update(created_epoch=time.time(), deadline_epoch=time.time()+7*86400,
        max_workers=2, stage='NATIVE_CONDITIONAL_ZK', initial_jobs=[], branch_jobs=[],
        runner_sha256=base.sha(__file__), authorization='2026-09-24 user: 设置goal开始',
        question='At fixed spatial Z/K fields, do high and interictal histories converge to different sustained conditional states?',
        field_template=str(SOURCE/'runs'/NAME/'states/t20s.pkl'),
        field_rule='One common observed full spatial template; Z logit-shifted to prescribed E mean; K scaled to prescribed E mean. No uniform replacement.',
        Z_values=[.25, .75, .95], K_values=[.02, 2., 8.],
        histories={'high': 'entry1_checkpoint.pkl', 'interictal': 't50s.pkl'},
        finite_horizon_s=30., maximum_grid_runs=18,
        paired_input='Common future RNG and complete exogenous OU state from t50s; endogenous delay/current/voltage/M/G histories retained; verify future input traces.',
        classification='Last10s: sustained high, recurrent brief population events, quiet, mixed/transition; report both histories and censoring. Never call finite-horizon coexistence a certified bifurcation.',
        rate_gate='Only pursue formal branches after conductance-aware local response and native spatial/event correspondence. G and M remain dynamic.',
        backend='CPU native serial ordered scatter; must first reproduce saved CUDA trajectory bitwise.',
        diagnostic_only=True, counts_as_autonomous_loop=False,
        later_axis_stage='Current axis, +90deg axis, isotropic with matched incoming EE degree/weight and audited distance/delay; define concrete matched graph before dispatch.',
        human_review='PENDING')
    shutil.copy2(SOURCE/'geometry.npz', OUT/'geometry.npz')
    write(path, p)
    return p


def make_job(name, history, duration, clamp=False, z=.75, k=2., common_input=False):
    prepare()
    folder = OUT/'runs'/name
    jobfile = OUT/'jobs'/f'{name}.json'
    if jobfile.exists():
        return base.read(jobfile)
    source = SOURCE/'runs'/NAME/'states'/history
    saved = read_pickle(source)
    step = int(saved['engine']['step'])
    job = copy.deepcopy(saved['job'])
    job.update(name=name, horizon_s=step*.0001+duration, checkpoint_s=min(2., duration),
               stage='conditional' if clamp else 'resume_qa', qa=False, stop_after_second_entry=False,
               conditional_clamp=clamp, target_Z=z, target_K=k, backend='cpu',
               branch_start_s=step*.0001, source_checkpoint=str(source),
               common_exogenous_input=common_input, diagnostic_only=True)
    engine = saved['engine']
    engine['slow']['kind'] = 'ConditionalSlow'
    if clamp:
        zz, kk = fields(z, k)
        engine['slow']['z'][:32000] = zz
        engine['termination_mechanism']['sahp_g'][:] = kk
    if common_input:
        common = read_pickle(SOURCE/'runs'/NAME/'states/t50s.pkl')['engine']
        for key in ['rng_state', 'xi', 'external_drive']:
            engine[key] = copy.deepcopy(common[key])
        offset = step-common['step']
        for key in ['next_step', 'last_step']:
            engine['external_drive'][key] += offset
    saved['job'] = job
    saved['tracker'] = carrier.fresh_tracker()
    folder.mkdir(parents=True, exist_ok=True)
    base.save_pickle(folder/'checkpoint.pkl', saved)
    write(jobfile, job)
    return job


def worker(name):
    p = prepare()
    assert base.sha(__file__) == p['runner_sha256'], 'New runner changed after protocol freeze'
    job = base.read(OUT/'jobs'/f'{name}.json')
    folder = OUT/'runs'/name
    cls = ConditionalSlow
    cls.C_R = 0.; cls.feedback_form = 'conductance'; cls.sahp_gain = job['sahp_gain']
    cls.sahp_tau_ms = 500.; cls.off_tau_ms = 5000.; cls.retention_below_Hz = 5.
    cls.global_tau_ms = 500.; cls.record_regions = True; cls.z_gate_off = False; cls.k_freeze = False
    cls.clamp = job['conditional_clamp']
    if cls.clamp:
        cls.target_z, cls.target_k = fields(job['target_Z'], job['target_K'])
    import checkpoint
    capture0, restore0 = checkpoint.capture, checkpoint.restore_slow
    def capture(**kw):
        state = capture0(**kw)
        state['global_feedback_response'] = response.extra_state(kw['slow'])
        return state
    def restore(state, obj):
        restore0(state, obj)
        response.restore_extra(state, obj)
    sink0 = fixed.observation_sink
    def factory(sink, j, deadline):
        full = sink0(sink, j, deadline)
        def observe(step, state):
            try:
                return full(step, state)
            finally:
                parent.flush(folder, step)
                obj = response.budget.RecoveryBudgetSlow.instance
                for attr, subdir in [('global_response_records', 'global_response_chunks'),
                                     ('conditional_records', 'conditional_drift_chunks')]:
                    rec = getattr(obj, attr)
                    if not rec:
                        continue
                    dest = folder/subdir
                    dest.mkdir(exist_ok=True)
                    if attr == 'conditional_records':
                        times = np.array([row[0] for row in rec]); values = np.stack([row[1] for row in rec])
                        arrays = dict(time_ms=times, values=values, variables=np.array(['dZ_per_s', 'dK_per_s']))
                    else:
                        arr = np.asarray(rec); times = arr[:, 0]
                        arrays = dict(time_ms=times, q=arr[:, 1], s=arr[:, 2], G_raw=arr[:, 3])
                    np.savez_compressed(dest/f'{round(times[0]*10):010d}_{step:010d}.npz', **arrays)
                    rec.clear()
                # These fields must stay fixed even when natural drift is nonzero.
                if cls.clamp:
                    assert np.array_equal(obj.z[:obj.NE], cls.target_z)
                    assert np.array_equal(obj.g_k, cls.target_k)
        return observe
    checkpoint.capture, checkpoint.restore_slow = capture, restore
    fixed.OUT, fixed.prepare, fixed.TerminationSlow, fixed.observation_sink = OUT, lambda: p, cls, factory
    # Original serial CPU scatter has the same accumulation order as the CUDA implementation.
    carrier.wrap_simulator = lambda fn, device_index: fn
    fixed.worker(name)
    result = base.read(folder/'result.json')
    result.update(no_external_intervention=not cls.clamp and not job['common_exogenous_input'],
                  diagnostic_only=True, counts_as_autonomous_loop=False,
                  clamp_Z_and_K=cls.clamp, endogenous_G_and_M_dynamic=True)
    write(folder/'result.json', result)
    write(folder/'progress.json', result)


def qa_compare(name):
    job = base.read(OUT/'jobs'/f'{name}.json')
    start = round(job['branch_start_s']*10000)
    end = round(job['horizon_s']*10000)
    expected = {}
    for path in sorted((SOURCE/'runs'/NAME/'chunks').glob('*.npz')):
        a, b = map(int, path.stem.split('_'))
        if a <= start and b >= end:
            with np.load(path) as f:
                for key, stride in [('spikes_1ms', 10), ('regions_1ms', 10), ('field_5ms', 50),
                                    ('raster', 1), ('Z', 200), ('M', 200), ('currents', 200), ('inputs', 1000)]:
                    expected[key] = f[key][(start-a)//stride:(end-a)//stride]
            break
    assert expected, 'QA must lie within one saved source chunk'
    paths = list((OUT/'runs'/name/'chunks').glob('*.npz'))
    assert len(paths) == 1
    with np.load(paths[0]) as f:
        checks = {key: bool(np.array_equal(f[key], value)) for key, value in expected.items()}
    write(OUT/'qa'/f'{name}.json', dict(checks=checks, status='PASS' if all(checks.values()) else 'FAIL'))
    assert all(checks.values()), checks
    print(json.dumps(checks), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('qa')
    w = sub.add_parser('worker'); w.add_argument('name')
    sub.add_parser('prepare-grid')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'qa':
        for name, state in [('resume_high_cpu', 'entry1_checkpoint.pkl'), ('resume_interictal_cpu', 't50s.pkl')]:
            make_job(name, state, .2)
            # Each worker is a fresh interpreter; its monkeypatches cannot leak across runs.
            import subprocess, sys
            subprocess.run([sys.executable, __file__, 'worker', name], check=True)
            qa_compare(name)
        name = 'clamp_mechanism_qa'
        make_job(name, 'entry1_checkpoint.pkl', .2, True, common_input=True)
        subprocess.run([sys.executable, __file__, 'worker', name], check=True)
        write(OUT/'qa/clamp_mechanism.json', dict(status='PASS', fields_constant=True,
                                                G_M_dynamic=True, native_drift_recorded=True))
    elif args.command == 'prepare-grid':
        for key in ['resume_high_cpu', 'resume_interictal_cpu', 'clamp_mechanism']:
            assert base.read(OUT/'qa'/f'{key}.json')['status'] == 'PASS'
        jobs = []
        for z in [.25, .75, .95]:
            for k in [.02, 2., 8.]:
                for history, state in [('high', 'entry1_checkpoint.pkl'), ('interictal', 't50s.pkl')]:
                    name = f'z{z:g}_k{k:g}_{history}'
                    make_job(name, state, 30., True, z, k, common_input=True)
                    jobs.append(name)
        write(OUT/'queue.json', dict(names=jobs, bounded=True, maximum=18))
    else:
        worker(args.name)


if __name__ == '__main__':
    main()

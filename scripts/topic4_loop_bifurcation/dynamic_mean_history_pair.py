#!/usr/bin/env python3
"""Two bounded ten-second physical correspondence tests of the leading mean."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import pickle
import shutil
import time
import numpy as np
from campaign import ROOT, read, write, sha
import dynamic_mean_input_pilot as leading
import dynamic_individual_source_pilot as base

OUT = ROOT/'dynamic_mean_history_pair'
CASES = {
    'high': (ROOT/'native_K9p35_held_history/runs/exit_z0.21_k9.35_fields16p7_held_K9_history/checkpoint.pkl', 720000, 'high_history_constant_background'),
    'asymmetric': (ROOT/'native_exit_K_bracket/runs/exit_z0.21_k9.35_fields16p7_high/checkpoint.pkl', 420000, 'asymmetric_history_constant_background'),
}


class HistoryNetwork(leading.MeanInputNetwork):
    def __init__(self, case, replicas, device):
        super().__init__(replicas, device)
        path, tick, _ = CASES[case]
        with path.open('rb') as f:
            saved = pickle.load(f)
        assert saved['identity'] == self.prep['graph_identity']
        s = saved['engine'];assert s['step'] == tick
        k = np.zeros(self.N);k[:32000] = s['termination_mechanism']['sahp_g']
        original = np.stack([s['V'], s['s_E'], s['I_E'], s['s_I'], s['I_I'], s['slow']['m'], s['slow']['z'], k], axis=1)
        assert np.array_equal(original[:, 6:8], self.native_initial[:, 6:8])
        self.native_initial = original;self.native_ref = s['ref']
        self.initial_state = self.cp.asarray(np.repeat(original[:, None, :], replicas, axis=1))
        self.initial_ref = self.cp.asarray(np.repeat(self.native_ref[:, None], replicas, axis=1), dtype='i4')
        self.initial_global = np.array([s['termination_mechanism']['r_global'], s['global_feedback_response']['global_state']])
        order = (tick+np.arange(self.depth)) % self.depth
        self.pending_cpu = np.stack([s['ring_sE'][order], s['ring_sI'][order]])
        self.pending = self.cp.asarray(self.pending_cpu);self.reset()
        assert np.array_equal(self.state.get()[:, 0], original)
        assert np.array_equal(self.ref.get()[:, 0], s['ref'])


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(leading.OUT/'analysis/result.json')['development_relevance_retained']
    dependencies = {p: sha(p) for p in [__file__, leading.__file__, base.__file__, leading.previous.__file__]}
    write(OUT/'contract.json', dict(status='REGISTERED_TWO_LONGER_HISTORY_TESTS', created_epoch=time.time(),
        question='Does the leading recurrent mean retain native spatial recruitment, core activity, resource balance and feedback over10s from both high and asymmetric complete native histories at identical held Z/K?',
        design='Exactly two10s physical trajectories,40000 targets x64 numericalreplicas. Native full72s and42s states with all membrane/ref/synaptic/M/global states and pending pulses. Same individualthresholds, graphweights/full delays, fixedpercell externalmeans and Poissonlaw. Z/K held; G/M/recurrent rates free. No periodforcing or biological parameter fitting.',
        rationale='The preceding1s high-state result passed unchanged spatial/core/feedback relevance guards after omitting an independently approximated recurrent residual whose white temporal structure was shown to overstate filtered variance. This pair tests persistence and an independent native history, not another noise fit.',
        references={k: dict(initial=str(v[0]), initial_sha256=sha(v[0]), initial_step=v[1], native=str(base.MATCHED/'runs'/v[2])) for k, v in CASES.items()},
        guards='For each0-5s and5-10s interval: spatial fieldRMS<=10Hz, eachcore absolute meanrate error<=10Hz, eachcore Zdrift error<=.01/s; whole causalR<200 and Graw<.1 as both native references remainfeedback-off. Same thresholds across histories. Passing is conditional development correspondence, not branch/stability certification.',
        outputs='Continuous1ms groupoutputs plus globalR/G; final fullmodelstate and numericalRNG. Compare native5ms space,1msR/G,20msdrift without temporal phase alignment. Replicas are not native seeds.',
        stop='Stop each after10s and analyze both. No automatic extension, scan, root or bifurcation calculation. If one fails, inspect the discrepancy before changing closure.',
        dependencies=dependencies, formal_bifurcation_allowed=False))
    shutil.copy2(__file__, OUT/'producer.py')


def run(case, device):
    c = read(OUT/'contract.json');assert all(sha(p) == h for p, h in c['dependencies'].items())
    assert sha(CASES[case][0]) == c['references'][case]['initial_sha256']
    dest = OUT/case;dest.mkdir(exist_ok=True);assert not (dest/'progress.json').exists()
    started = time.time();write(dest/'progress.json', dict(status='INITIALIZING', pid=os.getpid(), updated_epoch=time.time()))
    e = HistoryNetwork(case, 64, device);write(dest/'implementation_qa.json', leading.qa(e))
    np.savez_compressed(dest/'initial_state.npz', state=e.native_initial, ref=e.native_ref, global_state=e.initial_global)
    chunks = dest/'chunks';chunks.mkdir();e.graph();groups = [];globals_ = []
    for offset in range(0, 10000, 10):
        groups.append(e.chunk().astype('f4'));globals_.append(e.global_output.get())
        if (offset+10) % 100 == 0:
            value = np.concatenate(groups);glob = np.concatenate(globals_)
            assert np.isfinite(value).all() and np.isfinite(glob).all()
            np.savez_compressed(chunks/f'{offset-90:05d}_{offset+10:05d}.npz', group_output=value, global_R_Hz=glob[:, 0], global_s=glob[:, 1])
            groups.clear();globals_.clear()
            write(dest/'progress.json', dict(status='RUNNING', pid=os.getpid(), elapsed_simulation_ms=offset+10, elapsed_wall_s=time.time()-started, updated_epoch=time.time()))
            if (offset+10) % 1000 == 0:print(case, offset+10, flush=True)
    assert int(e.clock.get()[0]) == 100000
    assert np.array_equal(e.state.get()[:, :, 6:8], e.initial_state.get()[:, :, 6:8])
    np.savez_compressed(dest/'final_state.npz', state=e.state.get(), ref=e.ref.get(), rng=e.rng.get(), external_rng=e.external_rng.get(),
        source_history=e.source_history.get(), history=e.history.get(), clock=e.clock.get(), global_state=e.global_state.get())
    write(dest/'result.json', dict(status='COMPLETE_TEN_SECOND_HISTORY', case=case, replicas=64, duration_ms=10000, held_fields_bitwise=True,
        complete_initial_state_exact=True, elapsed_wall_s=time.time()-started, formal_bifurcation_allowed=False))
    write(dest/'progress.json', dict(status='COMPLETE', updated_epoch=time.time()));print(case, 'COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'run']);p.add_argument('--case', choices=list(CASES));p.add_argument('--device', type=int, default=0);a = p.parse_args()
    if a.command == 'prepare':prepare()
    else:
        try:run(a.case, a.device)
        except Exception:
            write(OUT/a.case/'progress.json', dict(status='FAILED', pid=os.getpid(), updated_epoch=time.time()));raise

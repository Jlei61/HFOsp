"""Fixed native LIF threshold ablation; no parameter or example search.

Run with the historical cuda_env Python, -B, and single-thread BLAS.
All writes (including caches) are confined to a NEW --output directory.
"""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import math
import os
import shutil
import sys
import time

REPO = Path('/home/honglab/leijiaxin/HFOsp')
WT = REPO / '.worktrees/topic4-continuous-core-state-r1'
HISTORY = Path('/data/hfosp/topic4_sef_hfo/e1146_core_count_optimization_20260928')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    def safe(x):
        if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [safe(v) for v in x]
        if hasattr(x, 'tolist'): return safe(x.tolist())
        if isinstance(x, Path): return str(x)
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(safe(obj), indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration-ms', type=float, default=10000.)
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(WT): raise ValueError('Output must be in the authorized worktree')
    out.mkdir(parents=True, exist_ok=False)
    os.environ['TOPIC4_CORE_COUNT_OUT'] = str(out)
    os.environ['MPLCONFIGDIR'] = str(out / 'mpl')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(REPO / 'scripts/topic4_core_count_optimization'), str(WT), str(WT / 'src/snn_engine')]
    import numpy as np
    import physics
    from params import compute_nu_theta
    from lfp import LFPRecorder
    from src.topic4_streaming_spike_readout import simulate_streaming
    from src.topic4_observation_repaired import observe

    source = HISTORY / 'canary_candidates/core1.json'
    candidate = json.loads(source.read_text())
    old_applied = json.loads((HISTORY / 'canary/core1/applied_physics.json').read_text())
    old_cache = Path(old_applied['graph']['global_cache'])
    (out / 'global_graph_cache').mkdir()
    shutil.copyfile(old_cache, out / 'global_graph_cache' / old_cache.name)
    shutil.copyfile(source, out / 'candidate.json')
    shutil.copyfile(HISTORY / 'observer.json', out / 'contact_observer.json')
    design = dict(schema='fig4.native_de.v1', topology_seed=2511, dynamics_seed=847401,
                  duration_ms=a.duration_ms, burnin_ms=1500.,
                  source_candidate=str(source), source_candidate_sha256=digest(source),
                  selection='Pre-existing K=1 physical canary, before inspecting its long response; no search.',
                  contrast='Only E threshold lowering removed in D; graph, weights, delays, initial voltage, stochastic support and all external realizations held fixed.',
                  D='No lowered-threshold core; retains identical localized stochastic afferent patch and region-specific graph weights.',
                  E='Original one-core canary physics; no kick, prescribed events, Z/M or slow I.',
                  statistical_unit='One paired 10 s realization on one topology; events are not independent network replicates.',
                  operational_burst='Non-overlapping 10 ms bins: >=1% distinct E cells globally or >=10% distinct E cells in fixed input patch; adjacent supra-threshold bins merged across <=100 ms gaps; report duration, do not equate with patient HFO.',
                  sparse_definition='Descriptive E mean <1 Hz and mean distinct-E fraction per 10 ms <1%; report values and events separately.',
                  graph_cache_source=str(old_cache), graph_cache_sha256=digest(old_cache))
    write(out / 'design.json', design)
    started = time.time()
    print('BUILDING', flush=True)
    sub, groups, loading, deterministic, applied, (ec, ic) = physics.build_unit(candidate, 2511, 847401)
    for key in ('positions_E_sha256', 'vtheta_float64_sha256', 'ampa_values_sha256', 'gaba_values_sha256'):
        if applied['identity'][key] != old_applied['identity'][key]: raise RuntimeError('Historical identity mismatch: ' + key)
    write(out / 'applied_physics.json', applied)
    write(out / 'params.json', dataclasses.asdict(sub.params))
    print('BUILT', round(time.time() - started, 1), 'seconds', flush=True)
    sub.params.T = a.duration_ms
    ne, ni = sub.n_e, sub.n_i
    dt = sub.params.dt
    nt = ne + ni
    signal = float(sub.params.nu_ext_ratio * compute_nu_theta(sub.params)[0])
    patch = np.flatnonzero(ec >= 0)
    sites = np.asarray([candidate['centers_mm'][0], np.asarray(candidate['centers_mm'][0]) + [2., 0.]])
    np.savez_compressed(out / 'substrate.npz', positions_E=sub.positions_e, positions_I=sub.positions_i,
                        vtheta_E_condition=sub.vtheta, input_patch_E=ec, input_patch_I=ic,
                        lfp_sites_mm=sites, deterministic_external_mask=deterministic)

    class Recorder:
        def __init__(self, arm, drive):
            self.arm, self.drive = arm, drive
            self.steps, self.ids = [], []
            self.stream = hashlib.sha256()
            self.groups = {'allE': np.arange(ne), 'patchE': patch, 'outsideE': np.flatnonzero(ec < 0), 'allI': np.arange(ne, nt)}
            self.trace = []
            self.sum_spikes = np.zeros(nt, np.int64)
            self.last_report = time.time()
            self.outside_max_error = 0.

        def observe(self, t, tm, xi, nu_now, ext, delta_rate, V, IE, II, spk):
            # Digest every stochastic arrival, global OU and external drive state.
            # All remaining entries are checked against their deterministic expectation.
            self.stream.update(np.asarray([xi, nu_now, self.drive.state[0]], '<f8').tobytes())
            self.stream.update(np.asarray(ext[patch], '<f8').tobytes())
            self.outside_max_error = max(self.outside_max_error, float(np.max(np.abs(ext[deterministic] - signal * dt), initial=0.)))
            ids = np.flatnonzero(spk).astype(np.uint32)
            if ids.size:
                self.steps.append(np.full(ids.size, t, np.uint32)); self.ids.append(ids)
            self.sum_spikes += spk
            if t % round(1. / dt) == 0:
                row = [tm, float(self.drive.state[0]), max(candidate['core_mean_rate_scale'] * signal + self.drive.state[0], 0.)]
                for idx in self.groups.values():
                    e, i = IE[idx], II[idx]
                    row += [float(e.mean()), float(i.mean()), float(np.mean(np.abs(e-i))), float(V[idx].mean())]
                self.trace.append(row)
            if (t+1) % round(1000. / dt) == 0:
                print(self.arm, 'simulated_ms', round(tm+dt, 2), 'wall_s', round(time.time()-started, 1), flush=True)

    def run_arm(arm):
        d = out / arm; d.mkdir()
        drive = physics.CoreCountOU(ec, signal, candidate['core_mean_rate_scale'], 0., dt, sub.params.tau_n, sub.params.sigma_n, 847401)
        recorder = Recorder(arm, drive)
        sub.net['rng'] = np.random.default_rng(847401)
        threshold = sub.vtheta.copy() if arm == 'E' else np.full(nt, sub.engine['v_base'])
        lfp = LFPRecorder(sub.params, sub.net['pos'], sub.net['labels'], sites=sites)
        t0 = time.time()
        result = simulate_streaming(physics.base.simulate_kick, sub.params, sub.net,
            positions=sub.positions_e, montage=sub.montage, KICK_BOOST=0., t_kick=1e9,
            V_th_per_neuron=threshold, slow=None, early_stop_runaway=True, **physics.base.RUNAWAY,
            global_ou_loading=np.zeros(nt), deterministic_external_mask=deterministic,
            step_observer=recorder, external_e_rate_drive=drive, lfp_recorder=lfp,
            dump_pathway_trace=True)
        spikes = result.pop('E_spk_bool')
        movie = spikes.native(); env, envdt, _ = spikes.envelope()
        observed = observe(env, float(envdt), json.loads((out/'contact_observer.json').read_text()))
        trace_names = ['time_ms', 'patch_ou_per_ms', 'patch_afferent_rate_per_ms']
        for group in recorder.groups:
            trace_names += [group+'_'+x for x in ['I_E', 'I_I', 'abs_net_I', 'V']]
        steps = np.concatenate(recorder.steps) if recorder.steps else np.array([], np.uint32)
        ids = np.concatenate(recorder.ids) if recorder.ids else np.array([], np.uint32)
        np.savez_compressed(d/'trajectory.npz', spike_steps=steps, spike_ids=ids,
            dt_ms=dt, duration_ms=len(result['rate_E'])*dt, rate_E=result['rate_E'], rate_I=result['rate_I'],
            lfp_proxy=result['lfp_trace'], trace=np.asarray(recorder.trace), trace_columns=np.asarray(trace_names),
            contact_envelope=env, contact_envelope_dt_ms=envdt,
            contact_names=np.asarray(sub.contact_names), contact_xy_mm=sub.contact_xy,
            sheet_activity_counts=movie['activity_counts'], sheet_frame_ms=movie['frame_ms'],
            vtheta=threshold, **{f'pathway_{k}': v for k, v in result.get('pathway_trace', {}).items()})
        write(d/'contact_events.json', observed)
        parity = None
        if arm == 'E':
            with np.load(HISTORY/'canary/core1/trajectory.npz') as h:
                parity = {k: bool(np.array_equal(np.asarray(result[k][:len(h[k])], np.float32), h[k])) for k in ['rate_E', 'rate_I']}
                parity['sheet_activity_counts'] = bool(np.array_equal(movie['activity_counts'][:len(h['sheet_activity_counts'])], h['sheet_activity_counts']))
            if not all(parity.values()): raise RuntimeError('Historical canary prefix mismatch')
        write(d/'execution.json', dict(status='COMPLETE', actual_duration_ms=len(result['rate_E'])*dt,
            runaway_early_stop_ms=result.get('runaway_early_stop_ms'), wall_seconds=time.time()-t0,
            n_spikes=int(len(ids)), input_stream_sha256=recorder.stream.hexdigest(),
            deterministic_max_error=recorder.outside_max_error, historical_prefix_parity=parity,
            final_rng_sha256=hashlib.sha256(json.dumps(sub.net['rng'].bit_generator.state, sort_keys=True).encode()).hexdigest(),
            trajectory_sha256=digest(d/'trajectory.npz')))

    processes = [mp.get_context('fork').Process(target=run_arm, args=(arm,)) for arm in ['D', 'E']]
    for process in processes: process.start()
    for process in processes: process.join()
    if any(p.exitcode for p in processes): raise RuntimeError('Arm failed; retained partial outputs')
    records = [json.loads((out/arm/'execution.json').read_text()) for arm in ['D', 'E']]
    pair = dict(same_external_stream=records[0]['input_stream_sha256']==records[1]['input_stream_sha256'],
                same_final_rng=records[0]['final_rng_sha256']==records[1]['final_rng_sha256'],
                deterministic_input_exact=all(r['deterministic_max_error']==0 for r in records),
                complete_duration=all(r['actual_duration_ms']==a.duration_ms for r in records),
                elapsed_seconds=time.time()-started)
    write(out/'paired_validation.json', pair)
    snap = out/'source_snapshot'; snap.mkdir()
    files = {Path(__file__).resolve(), source, physics.base.PARENT, Path(physics.__file__)}
    for module in list(sys.modules.values()):
        f = getattr(module, '__file__', None)
        if f and str(f).endswith('.py') and Path(f).resolve().is_relative_to(REPO): files.add(Path(f).resolve())
    for info in json.loads(physics.base.PARENT.read_text())['sources'].values():
        f = Path(info['path'])
        if f.suffix == '.json': files.add(f)
    manifest = {}
    for i, f in enumerate(sorted(files)):
        target = snap/f'{i:03d}_{f.name}'; shutil.copyfile(f, target)
        manifest[str(f)] = dict(sha256=digest(f), snapshot=target.name)
    write(out/'source_manifest.json', manifest)
    if not all(pair[k] for k in ['same_external_stream', 'same_final_rng', 'deterministic_input_exact', 'complete_duration']):
        raise RuntimeError('Pair did not satisfy full pairing contract')
    print('COMPLETE', pair, flush=True)


if __name__ == '__main__':
    main()

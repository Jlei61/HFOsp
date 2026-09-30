"""Trace the missing increasing-J direction from the checked B-leading seed.

The existing arc095Down -> BleadingConnection route goes around LPC7 and
does not cover this side of the original J=.95 seed. Stability is known at
the seed only; continuation never inherits its stability classification.
"""
from complete_rate_positive_stability import (
    DEST, PERIODIC_OUT, RateField, Path, np, read, write, prepare,
    best_cached, distances, release, paired_modes, compute_return)
import argparse
import os
import subprocess
import sys
import time
from scipy.signal import resample


LABEL = 'arcBleadingStableUp_20260920'
FOLDER = DEST/'Bleading_stable_side'


def seed_contract():
    anchor = read(DEST/'sites/085.json')
    assert anchor['status'] == 'NUMERICALLY_STABLE'
    assert abs(anchor['J_EE_core']-.95) < 1e-12
    first = PERIODIC_OUT/'orbits/branch095_J0.949990000_N512.npz'
    second = Path(anchor['analyzed_orbit'])
    rows = [read(f.with_suffix('.json')) for f in [first, second]]
    assert all(q['status'] == 'CONVERGED' for q in rows)
    assert rows[0]['J_EE_core'] < rows[1]['J_EE_core']
    return dict(label=LABEL, first=str(first), second=str(second),
        seed_stability_source=str(DEST/'sites/085.json'),
        initial_direction='Increasing J from the original J=.95 seed',
        populations=935, spatial_cells=400, local_states=8415,
        original_physical_delays=True,
        scope='The anchor is numerically stable. New profiles require their own stability checks. This segment is not appended to the returning end of the older B-leading family.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--after-pids', type=int, nargs='*', default=[])
    parser.add_argument('--min-free-gib', type=float, default=8.)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--ds', type=float, default=.24)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    assert args.min_free_gib >= 8 and 0 < args.steps <= 80 and 0 < args.ds <= .24
    FOLDER.mkdir(parents=True, exist_ok=True)
    contract = seed_contract()
    write(FOLDER/'seed_direction_contract.json', contract)
    if args.preflight_only:
        print('SEED DIRECTION CHECKED', contract, flush=True)
        return
    destination = PERIODIC_OUT/(LABEL+'_continuation.json')
    assert not destination.exists(), 'Never overwrite or restart a saved segment implicitly'

    def status(stage, **kwargs):
        write(FOLDER/'worker.json', dict(status=stage, pid=os.getpid(),
            label=LABEL, device=args.device, requested_steps=args.steps, **kwargs))
        print(stage, kwargs, flush=True)

    dependencies = {pid:Path(f'/proc/{pid}/cmdline').read_bytes()
                    for pid in args.after_pids if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid, identity in list(dependencies.items()):
            try:
                active = Path(f'/proc/{pid}/cmdline').read_bytes() == identity
            except (FileNotFoundError, ProcessLookupError):
                active = False
            if not active:
                dependencies.pop(pid)
        if dependencies:
            status('WAITING_DEPENDENCIES', dependencies=list(dependencies))
            time.sleep(30)

    def gate(mesh=0, stage='STAGE'):
        release(args.device)
        while True:
            free = float(subprocess.check_output(['nvidia-smi', '-i', str(args.device),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
            if free >= args.min_free_gib*1024:
                return
            status('WAITING_GPU_RESOURCE', free_mib=free,
                   required_free_gib=args.min_free_gib, N=mesh, next_stage=stage)
            time.sleep(30)

    scripts = Path(__file__).parent

    def run(stage, command):
        gate(stage=stage)
        log = FOLDER/f'{stage}_{time.time_ns()}.log'
        with log.open('w') as output:
            child = subprocess.Popen([sys.executable, '-u', *map(str, command)],
                stdout=output, stderr=subprocess.STDOUT)
            status(stage, child_pid=child.pid, log=str(log))
            code = child.wait()
        if code:
            raise RuntimeError(f'{stage} exited {code}: {log}')

    model = RateField()
    assert model.P == 935
    weights = model.geo['group_size']/model.geo['group_size'].sum()
    rows = []
    for key in ['first', 'second']:
        source = Path(contract[key])
        meta = read(source.with_suffix('.json'))
        start = best_cached(dict(orbit=str(source), J_EE_core=meta['J_EE_core'], T_ms=meta['T_ms']))
        status('REFINING_DIRECTION_SEED', source=str(source))
        actual, check = prepare(start, args.device, min_N=4096, max_N=8192,
            host_krylov=True, check_filter_states=True, harmonic_chunk_size=64,
            stream_harmonics=True, adaptive_memory=True, before_mesh=gate)
        if check['status']=='RESOLUTION_CHECKED' and check['maximum_group_defect_Hz']>=.001:
            actual, check = prepare(actual, args.device, min_N=2*check['N'], max_N=8192,
                host_krylov=True, check_filter_states=True, harmonic_chunk_size=64,
                stream_harmonics=True, adaptive_memory=True, before_mesh=gate)
        if check['status']!='RESOLUTION_CHECKED' or check['maximum_group_defect_Hz']>=.001:
            status('SEED_PROFILE_UNRESOLVED', source=str(source), check=check)
            return
        old, new = np.load(source), np.load(actual)
        mesh = max(len(old['r']), len(new['r']))
        before, after = [resample(z['r']*1000, mesh, axis=0) for z in [old, new]]
        difference, phase = distances(before[:,None,:], after, weights)
        scale = np.sqrt(np.mean(np.sum((before-before.mean(0))**2*weights, axis=1)))
        drift = float(difference[0]/scale)
        period_drift = abs(float(new['T']/old['T'])-1)
        assert abs(float(new['J']-old['J']))<1e-12 and drift<.02 and period_drift<.01
        rows.append(dict(source=str(source), orbit=str(actual), resolution=check,
            relative_waveform_change=drift, relative_period_change=period_drift))
        write(FOLDER/'seeds.json', dict(status='SEEDS_CHECKED' if len(rows)==2 else 'RUNNING', rows=rows))
    paths = [q['orbit'] for q in rows]
    seeds = [np.load(f) for f in paths]
    assert float(seeds[0]['J']) < float(seeds[1]['J'])
    mesh = max(len(z['r']) for z in seeds)
    # The current continuation allocator has an audited N4096 footprint;
    # a higher seed mesh requires a separately controlled storage change.
    if mesh > 4096:
        status('CONTINUATION_STORAGE_REVIEW_REQUIRED', N=mesh, seeds=paths)
        return
    run('CONTINUATION', [scripts/'rate_periodic_continue.py', *paths,
        '--N', mesh, '--steps', args.steps, '--ds', args.ds,
        '--label', LABEL, '--device', args.device, '--low-memory',
        '--host-krylov', '--linear-normalize', '--require-filter-positivity',
        '--stream-harmonics', '--quadratic-predictor', '--tol', '2e-11',
        '--stop-after-turns', '2'])
    continuation = read(destination)
    if not continuation['rows']:
        status('NO_ACCEPTED_NEW_POINTS', continuation_status=continuation['status'])
        return
    initial = read(Path(continuation['rows'][0]['path']).with_suffix('.json'))
    assert initial['J_EE_core'] > .95, 'The continuation left in the wrong direction'
    run('ALL_PROFILE_CHECKS', [scripts/'check_rate_extension.py', LABEL,
        '--device', args.device, '--stride', '1', '--check-filter-states',
        '--stream-harmonics', '--maximum-group-defect', '.001',
        '--prior-segment', 'Bleading_original_seed', '--prior-orbits', *paths])
    checked = read(PERIODIC_OUT/(LABEL+'_accuracy.json'))
    write(FOLDER/'checked_segment.json', checked)
    if checked['status'] != 'SAMPLED_PASS':
        status('NEW_SEGMENT_PROFILE_UNRESOLVED', evidence=str(PERIODIC_OUT/(LABEL+'_accuracy.json')))
        return
    # Bounded witnesses on the new arm. They do not classify every interval.
    indices = sorted({0, len(checked['included_orbits'])//2, len(checked['included_orbits'])-1})
    spectra = []
    for index in indices:
        orbit = Path(checked['included_orbits'][index])
        attempts = []
        for nev, steps in [(6, [.05,.025]), (10, [.025,.0125])]:
            pair, sources = [], []
            for dt in steps:
                gate(stage='POINCARE')
                label = f'{LABEL}_witness{index:04d}_k{nev}'
                target = PERIODIC_OUT/'poincare_floquet'/f'{label}_dt{dt:g}.json'
                status('POINCARE', index=index, nev=nev, dt_ms=dt, orbit=str(orbit))
                q = read(target) if target.exists() else compute_return(orbit, dt, nev,
                    args.device, max(16,2*nev+4), stream_harmonics=True, output_label=label)
                assert Path(q['orbit']).resolve()==orbit.resolve()
                pair.append(q)
                sources.append(str(target))
            verdict = paired_modes(*pair)
            attempts.append(dict(sources=sources, classification=verdict))
            if verdict['status']!='UNRESOLVED':
                break
        record = dict(index=index, orbit=str(orbit), status=verdict['status'], attempts=attempts)
        write(FOLDER/f'witness_{index:04d}.json', record)
        spectra.append(record)
    status('BOUNDED_EXTENSION_FINISHED', continued_points=len(checked['included_orbits']),
        sampled_stability=spectra, interval_completeness=False, global_connection_confirmed=False)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        FOLDER.mkdir(parents=True, exist_ok=True)
        path = FOLDER/'worker.json'
        previous = read(path) if path.exists() else {}
        write(path, {**previous, 'status':'COMPUTATION_FAILED', 'error':repr(exc)})
        raise

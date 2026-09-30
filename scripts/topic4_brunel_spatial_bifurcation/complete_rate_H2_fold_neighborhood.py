"""Resolve spectra on a bounded, continuous neighborhood of the H2 fold.

The branch coordinate is Core B mean rate. Both sides of a cycle fold can
have the same J, so they are kept separate. A dominant-multiplier sign change
is never promoted to a period-doubling without a continued critical mode.
"""
from complete_rate_positive_stability import *
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--after-pid', type=int, nargs='*', default=[])
    p.add_argument('--min-free-gib', type=float, default=3.)
    p.add_argument('--offsets', type=float, nargs='+', default=[-.003, -.001, .001, .003])
    p.add_argument('--worker-label', default='', help='Separate a subsequent bounded batch from the original worker record')
    a = p.parse_args()
    folder = DEST / 'H2_fold_neighborhood'
    folder.mkdir(exist_ok=True)
    suffix='_'+a.worker_label if a.worker_label else ''
    worker = folder / ('worker'+suffix+'.json')
    rows = []
    def status(stage, **kw):
        write(worker, dict(status=stage, pid=os.getpid(), rows=rows,
                          timestamp=time.time(), **kw))
    dependencies = {pid: Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid
                    if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid, identity in list(dependencies.items()):
            proc = Path(f'/proc/{pid}/cmdline')
            if not proc.exists() or proc.read_bytes() != identity:
                dependencies.pop(pid)
        if dependencies:
            status('WAITING_DEPENDENCIES', dependencies=list(dependencies))
            time.sleep(30)
    def gate():
        release(a.device)
        while True:
            free = float(subprocess.check_output(['nvidia-smi', '-i', str(a.device),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
            if free >= a.min_free_gib * 1024:
                return
            status('WAITING_GPU_RESOURCE', free_mib=free)
            time.sleep(30)
    label = 'LPC_B2'
    validation = read(PERIODIC_OUT / (label + '_validation.json'))
    assert validation['status'] == 'VALIDATED_CYCLE_FOLD'
    critical = validation['mesh_checks'][-1]
    assert critical['coordinate'] == 'core_B_mean_Hz'
    base = critical['coordinate_value']
    missing = [v for v in a.offsets if not (PERIODIC_OUT / 'orbits' /
        f'{label}_local_{v:+.5f}_N{critical["N"]}.npz').exists()]
    if missing:
        gate()
        log = folder / f'profiles_{time.time_ns()}.log'
        with log.open('w') as output:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).parent /
                'sample_rate_fold_neighborhood.py'), label, '--device', str(a.device),
                '--tol', '2e-11', '--offsets', *map(str, missing)],
                stdout=output, stderr=subprocess.STDOUT)
            status('SOLVING_NEIGHBORS', child_pid=child.pid, log=str(log))
            code = child.wait()
        if code:
            status('PROFILE_SOLVE_FAILED', exit_code=code, log=str(log))
            return
    s = RateField()
    w = s.geo['group_size'] * s.E * (s.geo['group_region'] == 1)
    w /= w.sum()
    for offset in sorted(a.offsets, key=abs):
        output = folder / f'offset_{offset:+.5f}.json'
        if output.exists() and read(output).get('status') == 'SPECTRUM_CHECKED':
            rows.append(read(output))
            continue
        orbit = PERIODIC_OUT / 'orbits' / f'{label}_local_{offset:+.5f}_N{critical["N"]}.npz'
        gate()
        status('CHECKING_PROFILE', offset=offset, orbit=str(orbit))
        actual, physical = prepare(orbit, a.device, max_N=1024, check_filter_states=True)
        z = np.load(actual)
        measured = float(z['r'].mean(0) @ w * 1000)
        assert abs(measured - (base + offset)) < 1e-6
        row = dict(offset=offset, coordinate='core_B_mean_Hz', coordinate_value=measured,
            original_orbit=str(orbit), orbit=str(actual), physical=physical,
            J_EE_core=float(z['J']), T_ms=float(z['T']))
        if physical['status'] != 'RESOLUTION_CHECKED' or physical['maximum_group_defect_Hz'] > .001:
            row['status'] = 'PROFILE_REVIEW_PENDING'
            write(output, row)
            rows.append(row)
            continue
        spectra = []
        sources = []
        for dt in [.1, .05]:
            prefix = f'H2_neighborhood_{offset:+.5f}_{actual.stem}'
            source = PERIODIC_OUT / 'poincare_floquet' / f'{prefix}_dt{dt:g}.json'
            gate()
            status('COMPUTING_SPECTRUM', offset=offset, dt_ms=dt, orbit=str(actual))
            spectrum = read(source) if source.exists() else compute_return(
                actual, dt, 6, a.device, 16, stream_harmonics=True, output_label=prefix)
            assert Path(spectrum['orbit']).resolve() == actual.resolve()
            spectra.append(spectrum)
            sources.append(str(source))
        row.update(status='SPECTRUM_CHECKED', sources=sources,
                   classification=paired_modes(*spectra))
        write(output, row)
        rows.append(row)
        status('POINT_FINISHED', offset=offset)
    release(a.device)
    write(folder / ('summary'+suffix+'.json'), dict(status='BOUNDED_SPECTRAL_SAMPLING_FINISHED',
        rows=sorted(rows, key=lambda q: q['coordinate_value']),
        fold_validation=str(PERIODIC_OUT / (label + '_validation.json')),
        global_connection_confirmed=False,
        scope='Physically checked cycle samples and paired-step numerical spectra. '
              'No additional crossing is located or classified by this sampling alone.'))
    status('BATCH_FINISHED')


if __name__ == '__main__':
    main()

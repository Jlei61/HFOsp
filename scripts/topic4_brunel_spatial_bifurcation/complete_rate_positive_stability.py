"""Paired-step full-delay stability on physically checked spatial cycles.

Preserves the frozen survey and all earlier artifacts. Each new verdict uses
the exact same-J orbit after a positivity and off-grid equation check. The
survey samples branches; it never declares that an interval has no hidden
crossings just because both endpoints have the same stability.
"""
from rate_floquet_poincare import *
from rate_periodic_accuracy import prepare
from compare_rate_torus_periodic_targets import distances
import gc
import os

DEST = Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')


def paired_modes(coarse, fine):
    """A large growing mode must not mask a smaller verified growing mode."""
    for q in [coarse, fine]:
        assert q['dt_ms'] <= q.get('minimum_occupied_delay_ms', .1)*(1+1e-12), 'Invalid Heun delay step'
    a, b = values(coarse), values(fine)
    ii, jj = linear_sum_assignment(abs(a[:, None] - b[None, :]))
    change = np.full(len(b), np.inf)
    change[jj] = abs(a[ii] - b[jj])
    margins = np.maximum(2e-5, np.maximum(6 * change,
        4 * max(coarse['phase_tangent_relative_defect'], fine['phase_tangent_relative_defect'])))
    fine_res = np.asarray(fine['residuals']) / np.maximum(1, abs(b))
    coarse_res = np.full(len(b), np.inf)
    coarse_res[jj] = np.asarray(coarse['residuals'])[ii] / np.maximum(1, abs(a[ii]))
    reliable = (fine_res < 1e-6) & (coarse_res < 1e-6)
    outside = reliable & (abs(b) > 1 + margins)
    inside = reliable & (abs(b) < 1 - margins)
    covered = all(filter_spectrum_covered(q) for q in [coarse, fine])
    section_ok = all(max(q['phase_overlap']) < 1e-6 for q in [coarse, fine])
    status = 'UNSTABLE' if outside.any() else (
        'NUMERICALLY_STABLE' if covered and inside.all() and section_ok else 'UNRESOLVED')
    return dict(status=status, multipliers=b, per_mode_margin=margins,
        reliable_mode_mask=reliable, outside_unit_disk_mask=outside,
        paired_absolute_changes=change, reliable_outside_count=int(outside.sum()),
        numerical_unstable_dimension=int(outside.sum()) if covered and
        (inside | outside).all() and section_ok else None,
        filter_coverage=covered, section_projection_checked=section_ok,
        maximum_returned_modulus=float(max(abs(b))),
        scope='Paired-step numerical full-history Poincare spectrum at one cycle; not a rigorous enclosure or interval-wide completeness proof.')


def release(device):
    import cupy as cp
    cp.cuda.Device(device).use()
    gc.collect()
    cp.fft.config.get_plan_cache().clear()
    cp.get_default_memory_pool().free_all_blocks()


def best_cached(item):
    origin = Path(item['orbit'])
    candidates = [origin]
    candidates.extend(Path(q['path']) for f in (PERIODIC_OUT/'orbits').glob(origin.stem + '_accuracy_*.json')
        if (q := read(f)).get('status') == 'CONVERGED' and
        abs(q['J_EE_core'] - item['J_EE_core']) < 1e-12 and
        abs(q['T_ms']/item['T_ms'] - 1) < .01 and q['residual_hz'] < 2e-8)
    return max(candidates, key=lambda p: len(np.load(p)['r']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indices', type=int, nargs='+', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--require-dimension', action='store_true',
        help='Continue spectrum refinement until all returned modes and spectral coverage support an unstable-dimension count')
    p.add_argument('--min-free-gib', type=float, default=0.)
    p.add_argument('--worker-label', help='Separate status filename for an independent bounded batch')
    p.add_argument('--after-pids',type=int,nargs='*',default=[],
        help='Wait for these exact live process identities before starting the bounded batch')
    a = p.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'sites').mkdir(exist_ok=True)
    frozen = read(PERIODIC_OUT/'stability_coverage/plan.json')
    planfile = DEST/'frozen_survey_plan.json'
    if planfile.exists():
        assert read(planfile) == frozen, 'Frozen survey changed'
    else:
        write(planfile, frozen)
    s = RateField()
    weights = s.geo['group_size']/s.geo['group_size'].sum()
    if a.worker_label:assert Path(a.worker_label).name==a.worker_label
    worker = DEST/(a.worker_label+'.json') if a.worker_label else DEST/f'worker_gpu{a.device}.json'
    completed = []
    def progress(state, **kw):
        write(worker, dict(status=state, pid=os.getpid(), device=a.device,
              requested_indices=a.indices, completed=completed, timestamp=time.time(), **kw))
        print(state, kw, flush=True)
    def memory(stage, index):
        if a.min_free_gib<=0:return
        import subprocess
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            progress('WAITING_GPU_RESOURCE',stage=stage,index=index,free_mib=free)
            time.sleep(30)
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pids
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            proc=Path(f'/proc/{pid}/cmdline')
            if not proc.exists() or proc.read_bytes()!=identity:dependencies.pop(pid)
        if dependencies:
            progress('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    for index in a.indices:
        item = frozen['rows'][index]
        output = DEST/'sites'/f'{index:03d}.json'
        if (output.exists() and read(output)['status'] in ['UNSTABLE', 'NUMERICALLY_STABLE']
            and (not a.require_dimension or read(output).get('classification',{}).get('numerical_unstable_dimension') is not None)):
            completed.append(dict(index=index, status=read(output)['status'], source=str(output)))
            continue
        try:
            memory('PROFILE_CHECK', index)
            release(a.device)
            start = best_cached(item)
            progress('PROFILE_CHECK', index=index, orbit=str(start))
            actual, accuracy = prepare(start, a.device, max_N=8192,
                check_filter_states=True, harmonic_chunk_size=32, adaptive_memory=True,
                stream_harmonics=a.require_dimension,host_krylov=a.require_dimension)
            detail = dict(index=index, original_orbit=item['orbit'], analyzed_orbit=str(actual),
                memberships=item['memberships'], resolution=accuracy)
            if accuracy['status'] != 'RESOLUTION_CHECKED':
                detail.update(status='PROFILE_UNRESOLVED')
                write(output, detail)
                completed.append(dict(index=index, status=detail['status'], source=str(output)))
                continue
            before, after = np.load(item['orbit']), np.load(actual)
            N = max(len(before['r']), len(after['r']))
            x, y = [resample(z['r']*1000, N, axis=0) for z in [before, after]]
            d, phase = distances(x[:, None, :], y, weights)
            rms = np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights, axis=1)))
            drift = float(d[0]/rms)
            period_drift = abs(float(after['T'])/float(before['T'])-1)
            assert abs(float(after['J'])-float(before['J'])) < 1e-12
            assert drift < .02 and period_drift < .01, (drift, period_drift)
            detail.update(J_EE_core=float(after['J']), T_ms=float(after['T']),
                relative_waveform_refinement_change=drift, relative_period_change=period_drift,
                status='FLOQUET_PENDING', attempts=[])
            write(output, detail)
            # Increase eigenmode coverage and integration resolution only when
            # the first pair cannot establish a numerical stability verdict.
            schedule=([(6,[.05,.025]),(10,[.025,.0125]),(16,[.0125,.00625])]
                      if a.require_dimension else [(2,[.1,.05]),(6,[.05,.025]),(8,[.025,.0125])])
            for nev, steps in schedule:
                rows, sources = [], []
                for dt in steps:
                    release(a.device)
                    memory('FLOQUET', index)
                    label = f'coverage20260920_{index:03d}_k{nev}_{actual.stem}'
                    dest = PERIODIC_OUT/'poincare_floquet'/f'{label}_dt{dt:g}.json'
                    progress('FLOQUET', index=index, nev=nev, dt_ms=dt, orbit=str(actual))
                    q = read(dest) if dest.exists() else compute_return(actual, dt, nev,
                        a.device, max(10, 2*nev+4), stream_harmonics=(a.require_dimension or N>=8192), output_label=label)
                    assert Path(q['orbit']).resolve() == actual.resolve()
                    rows.append(q)
                    sources.append(str(dest))
                result = paired_modes(*rows)
                detail['attempts'].append(dict(nev=nev, requested_steps_ms=steps,
                    sources=sources, classification=result))
                detail.update(status=result['status'], classification=result,
                    unstable_dimension_requested=a.require_dimension,
                    numerical_dimension_status='RESOLVED_AT_THIS_SAMPLE' if result['numerical_unstable_dimension'] is not None else 'INCOMPLETE')
                write(output, detail)
                print('PAIRED CLASSIFICATION', index, result, flush=True)
                if result['status'] != 'UNRESOLVED' and (not a.require_dimension or result['numerical_unstable_dimension'] is not None):
                    break
            completed.append(dict(index=index, status=detail['status'], source=str(output)))
        except Exception as exc:
            failure = dict(index=index, original_orbit=item['orbit'], status='COMPUTATION_FAILED', error=repr(exc))
            write(DEST/'sites'/f'{index:03d}_failure.json', failure)
            completed.append(failure)
            progress('SITE_FAILED', index=index, error=repr(exc))
            if type(exc).__name__ == 'OutOfMemoryError' or isinstance(exc, OSError):
                raise
    release(a.device)
    progress('BATCH_FINISHED')


if __name__ == '__main__':
    main()

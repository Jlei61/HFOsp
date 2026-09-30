#!/usr/bin/env python3
"""Native Euler-Z upper bound: whether a low-activity episode can restore Z."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from campaign import ROOT, read, write, sha
from analyze_feedback_tail import SOURCE, load


def main():
    rows = []
    dt_s = .0001
    decay = 1. - dt_s / 5.
    for seed in [9108402, 9108403, 9108405]:
        name = f'G30_response0.5_s{seed}'
        folder = SOURCE / 'runs' / name
        result = read(folder / 'result.json')
        assert result['status'] == 'COMPLETE'
        analysis = read(SOURCE / 'analysis' / f'{name}.json')
        rate = load(folder, 'mechanism_chunks', ['time_ms', 'global_E_rate_Hz', 'global_raw_conductance_ratio'])
        budget = load(folder, 'z_budget_chunks', ['time_ms', 'values'])
        adapt = load(folder, 'intrinsic_adaptation_chunks', ['time_ms', 'sahp_mean_conductance_ratio'])
        assert np.array_equal(rate['time_ms'], adapt['time_ms'])
        assert np.max(abs(budget['values'][:, :, 5])) < 1e-11
        t = rate['time_ms'] / 1000.; tb = budget['time_ms'] / 1000.
        Z = budget['values'][:, 1:3, 1]
        ref = np.asarray(analysis['absolute_Z_recovery']['reference_core_Z'])
        g_limit = result['job']['threshold'] / (18 - result['job']['global_reversal_mV'])
        entries = np.asarray([e['onset_s'] for e in analysis['primary']['entries']])
        for i, ep in enumerate(analysis['absolute_Z_recovery']['episodes']):
            start, end = ep['exit_start_s'], ep['window_end_s']
            take = np.flatnonzero((t >= start) & (t < end) & (rate['global_E_rate_Hz'] <= 5))
            if not len(take):
                rows.append(dict(seed=seed, episode=i + 1, status='NO_SAMPLED_R5_IN_EPISODE'))
                continue
            low = int(take[0])
            take = np.flatnonzero((t >= t[low]) & (t < end) & (rate['global_raw_conductance_ratio'] < g_limit))
            if not len(take):
                rows.append(dict(seed=seed, episode=i + 1, status='G_BLOCK_NOT_RELEASED_IN_EPISODE'))
                continue
            unblocked = int(take[0])
            anchor = int(np.searchsorted(tb, t[unblocked]))
            z0 = Z[anchor]
            n = np.maximum(0, np.ceil(np.log((1 - ref) / (1 - z0)) / np.log(decay))).astype(int)
            earliest_each = n * dt_s
            earliest = float(earliest_each.max())
            mask = (tb >= tb[anchor]) & (tb < end)
            steps = np.rint((tb[mask] - tb[anchor]) / dt_s).astype(int)
            upper = 1 - (1 - z0) * decay ** steps[:, None]
            max_excess = float((Z[mask] - upper).max())
            assert max_excess < 1e-9, (seed, i, max_excess)
            recovered = np.flatnonzero(mask & (Z >= ref).all(1))
            first_ref = float(tb[recovered[0]]) if len(recovered) else None
            if first_ref is not None:
                assert first_ref - tb[anchor] >= earliest - .00010001
            entry_ends = bool(np.any(abs(entries - end) < 1e-8))
            available = float(end - tb[anchor])
            r_above = np.flatnonzero((t > t[low]) & (t < end) & (rate['global_E_rate_Hz'] > 5))
            rows.append(dict(seed=seed, episode=i + 1, status='MEASURED', exit_start_s=start,
                first_sampled_R5_s=float(t[low]), first_G_below_limit_s=float(t[unblocked]),
                budget_anchor_s=float(tb[anchor]), anchor_core_Z=z0.tolist(), reference_core_Z=ref.tolist(),
                fastest_possible_time_to_reference_each_core_s=earliest_each.tolist(),
                fastest_possible_both_reference_s=earliest,
                observed_both_reference_s=first_ref,
                interval_end_s=end, interval_ends_with_new_entry=entry_ends,
                time_available_until_end_s=available,
                physically_too_short_before_next_entry=bool(entry_ends and available < earliest),
                maximum_observed_Z_above_ideal_upper_bound=max_excess,
                first_R_above5_after_low_s=float(t[r_above[0]]) if len(r_above) else None,
                K_at_low=float(adapt['sahp_mean_conductance_ratio'][low]),
                K_at_R_rebound=float(adapt['sahp_mean_conductance_ratio'][r_above[0]]) if len(r_above) else None,
                original_full_return=ep['sustained_return_after_absolute_recovery']))
    dest = ROOT / 'recovery_time_bound'
    write(dest / 'analysis.json', dict(status='COMPLETE_EXISTING_NATIVE_RECORDS', rows=rows,
        question='Can the low-activity interval possibly restore both core resources before the next entry, even with perfect recovery eligibility?',
        equation='For native EulerZ, meanZ_n <= 1-(1-meanZ_0)*(1-dt/5s)^n because0<=p<=1. Fastest time uses the ceiling of the logarithmic crossing step. Bound applies independently to each original observer core.',
        anchor='First20msbudget endpoint at/afterGraw falls below the necessary2.6694threshold following sampledR<=5 inside the originalexit interval. Actual measuredZ at this endpoint is used.',
        statistical_unit='Exit episodes nested within3existing native seeds; no new simulation and no independent-episode inference.',
        interpretation='Too-short episodes cannot reach the originalinterictalZreference regardless of subsequentinput. Sufficienttime is only a necessary opportunity, not proof of recovery or brief-event return. The Zreference is not a certified bifurcation threshold.',
        producer_sha256=sha(__file__), formal_bifurcation_allowed=False))
    print([{k: r.get(k) for k in ['seed', 'episode', 'fastest_possible_both_reference_s',
        'time_available_until_end_s', 'physically_too_short_before_next_entry', 'observed_both_reference_s',
        'original_full_return']} for r in rows], flush=True)


if __name__ == '__main__':
    main()

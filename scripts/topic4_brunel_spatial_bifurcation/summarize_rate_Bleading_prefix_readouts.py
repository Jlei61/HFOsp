"""Compare the frozen three contact metrics on a newly checked branch segment.

Each sample is one periodic solution. Repeated cycles and bin origins only
check the deterministic readout; they are not independent realizations.
"""
from pathlib import Path
import argparse
import numpy as np

from complete_rate_positive_stability import DEST, read, write
from audit_rate_survey_filter_states import fingerprint


def differences(first, second):
    results = {}
    for metric in ['mean_rank', 'within_shaft_order_probability', 'participation']:
        x, y = (np.asarray(row[metric], dtype=float) for row in [first, second])
        assert x.shape == y.shape
        common = np.isfinite(x) & np.isfinite(y)
        results[metric] = dict(
            shared_defined_entries=int(common.sum()),
            defined_mask_changes=int(np.count_nonzero(np.isfinite(x) != np.isfinite(y))),
            max_absolute_change=float(np.max(np.abs(x[common]-y[common]))) if common.any() else None)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-prefix', type=Path, required=True)
    parser.add_argument('--current-prefix', type=Path, required=True)
    args = parser.parse_args()
    before, after = [read(path) for path in [args.previous_prefix, args.current_prefix]]
    assert before['status'] == after['status'] == 'SAMPLED_PASS'
    old, new = before['included_orbits'], after['included_orbits']
    assert new[:len(old)] == old and len(new) > len(old)
    summary = read(DEST/'SCL_branch_scan/summary.json')
    assert summary['status'] == 'CHECKED_POINT_OBSERVER_SCAN_COMPLETE'
    lookup = {str(Path(row['orbit']).resolve()): row for row in summary['rows']
              if row['family'] == 'Bleading'}

    def observation(orbit):
        source = lookup[str(Path(orbit).resolve())]['source']
        q = read(source)
        assert Path(q['orbit']).resolve() == Path(orbit).resolve()
        assert q['profile_fingerprint'] == fingerprint(orbit)
        records = {row['bin_origin_ms']: row for row in q['records']}
        assert len(records) == 4
        return source, q, records

    baseline_source, baseline, baseline_bins = observation(old[-1])
    rows = []
    for index in range(len(old), len(new)):
        source, q, bins = observation(new[index])
        assert bins.keys() == baseline_bins.keys()
        check = after['checks'][index]
        assert Path(check['orbit']).resolve() == Path(q['orbit']).resolve()
        assert check['filter_state_check']['positive']
        assert check['maximum_group_defect_Hz'] < .001
        comparisons = []
        for origin, row in sorted(bins.items()):
            reference = baseline_bins[origin]
            comparisons.append(dict(bin_origin_ms=origin,
                reference_qualified_events=reference['qualified_events'],
                qualified_events=row['qualified_events'],
                SCL_qualified_events=row['SCL_qualified_events'],
                individual_SCL_contacts=row['sustained_SCL_contact_names'],
                metrics=differences(reference['metrics'], row['metrics'])))
        rows.append(dict(index=index, orbit=q['orbit'], source=source,
            J_EE_core=q['J_EE_core'], T_ms=q['T_ms'], comparisons=comparisons))
    output = DEST/'Bleading_extension'/f'prefix{len(new)}_readout_change.json'
    result = dict(status='FROZEN_PREFIX_READOUT_COMPARISON_COMPLETE',
        previous_prefix=str(args.previous_prefix), current_prefix=str(args.current_prefix),
        baseline_source=baseline_source, baseline_index=len(old)-1,
        baseline_J_EE_core=baseline['J_EE_core'], rows=rows,
        all_defined_metrics_unchanged=all(
            metric['defined_mask_changes'] == 0 and
            metric['max_absolute_change'] is not None and metric['max_absolute_change'] < 1e-12
            for row in rows for comparison in row['comparisons']
            for metric in comparison['metrics'].values()),
        stability_inferred=False, interval_completeness=False,
        scope='Compare each new physically checked solution with the previous prefix endpoint using the frozen observer and matching bin origins. Missing ranks and unsupported pairs remain undefined. Metric agreement does not establish identical millisecond timing, orbital stability, absence of interior bifurcations, or a complete SCL window. Repeated cycles and bin origins are not independent samples.')
    write(output, result)
    print('PREFIX READOUT', len(rows), result['all_defined_metrics_unchanged'], str(output), flush=True)


if __name__ == '__main__':
    main()

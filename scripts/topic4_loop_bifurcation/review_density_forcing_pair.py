#!/usr/bin/env python3
"""Compare the completed one-variable input repair with its fixed baseline."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import time
import numpy as np
from campaign import ROOT, read, write, sha


def main():
    base = ROOT / 'density_spatial_grouping'
    repair = ROOT / 'density_fine_forcing'
    before, after = [read(p / 'comparison.json') for p in [base, repair]]
    assert before['status'] == after['status'] == 'COMPLETE'
    assert read(repair / 'implementation_check.json')['status'] == 'PASS'
    assert read(base / 'contract.json')['engine_sha256'] == read(repair / 'contract.json')['engine_sha256']
    assert read(repair / 'result.json')['engine_sha256_unchanged']
    # Fixed draws per particle per step make this a numerical-noise paired test.
    with np.load(base / 'final_state.npz') as a, np.load(repair / 'final_state.npz') as b:
        paired = {key: bool(np.array_equal(a[key], b[key])) for key in ['rng', 'clock', 'particle_count', 'seed']}
    assert all(paired.values()), paired
    fields = dict(onset_ms=lambda r: r['summary']['high_onset_ms'],
        D9870=lambda r: r['D9870'], event_count=lambda r: r['windows']['1000-9420']['n'],
        median_duration_ms=lambda r: r['windows']['1000-9420']['median_duration_ms'],
        median_area=lambda r: r['windows']['1000-9420']['median_area'],
        quiet_fraction=lambda r: r['quiet_by_window']['1000-9420'])
    summaries = {}
    for label, row in [('coarse_input', before['dynamics']), ('fine_input', after['dynamics']),
                       *[(name, r) for name, r in after['reference_dynamics'].items() if name.startswith('native')]]:
        summaries[label] = {name: fn(row) for name, fn in fields.items()}
    contact = {name: dict(N=data['contacts']['summary']['N'], direction_counts=data['direction']['counts'],
                         comparisons=data['contact_comparisons']) for name, data in [('coarse_input', before), ('fine_input', after)]}
    out = dict(status='COMPLETE_PAIRED_INPUT_DIAGNOSTIC', summaries=summaries, contacts=contact,
        final_numerical_stream_and_clock_bitwise=paired,
        original_A4_before=before['dynamics']['original_A4_checks'],
        original_A4_after=after['dynamics']['original_A4_checks'],
        comparison='Only externalfinegroupdrive changes; nativegraph,densitykernel,initialstates,dt andfullnumericalrandomstreamfixed. Original1msinputholding remains.',
        acceptance='UnchangedoriginalA4checks arediagnostic. No numericalconvergence, actualG/Kspatialcorrespondence or formalbifurcation followsfrom thispair.',
        producer_sha256=sha(__file__), native_correspondence_certified=False,
        formal_bifurcation_allowed=False, human_review='PENDING')
    write(repair / 'paired_review.json', out)
    print(dict(status=out['status'], summaries=summaries, A4=out['original_A4_after'], paired=paired), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--wait', action='store_true'); args = p.parse_args()
    while args.wait and not (ROOT / 'density_fine_forcing/comparison.json').exists():
        time.sleep(20)
    main()

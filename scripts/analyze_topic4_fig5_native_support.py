#!/usr/bin/env python3
"""Compare observables on matched baseline and spatial support, without simulation."""
import json
import numpy as np
from analyze_topic4_fig5_early_spatial import OUT, correlation, write


def main():
    a = np.load(OUT / 'analysis_arrays.npz')
    replay = np.load(OUT / 'native_current_replay.npz')
    early = (replay['t10180_time_s'] >= 10.48) & (replay['t10180_time_s'] < 10.73)
    contact_power = (replay['t10180_readout'][early] ** 2).mean(0) - (replay['t8000_readout'] ** 2).mean(0)
    native_power = replay['native_current_delta_power']
    # The directly containing 1-mm cell; no Gaussian interpolation or fitting.
    xy_cell = np.minimum(a['contact_xy'].astype(int), 19)
    cell = xy_cell[:, 1] * 20 + xy_cell[:, 0]
    sampled_power = native_power[cell]
    groups = {}
    for k, family in enumerate(['A', 'B']):
        contact_rank = a['template_rank'][k]
        native_rank = a['native_template_rank'][k]
        sampled_rank = native_rank[cell]
        groups[family] = dict(
            contact_order_contact_power_same_baseline=correlation(-contact_rank, contact_power),
            native_order_native_power_full_grid=correlation(-native_rank, native_power),
            native_order_native_power_at_contacts=correlation(-sampled_rank, sampled_power),
            contact_order_native_power_at_contacts=correlation(-contact_rank, sampled_power),
            native_order_contact_power_at_contacts=correlation(-sampled_rank, contact_power),
            contact_vs_native_order_at_contacts=correlation(contact_rank, sampled_rank),
            valid_native_cells_at_contacts=int(np.isfinite(sampled_rank).sum()),
            valid_native_cells_full_grid=int(np.isfinite(native_rank).sum()))
    result = dict(
        baseline_window_s=[8., 8.25], early_window_s=[10.48, 10.73],
        power_definition='For both observables: mean(signal squared) early minus baseline, without quiet-bin centering. Not the primary contact panel definition.',
        support='Native 1-mm cells directly containing each of the 15 contact positions; no smoothing, interpolation or spatial refitting.',
        groups=groups, contact_cell_indices=cell.tolist(),
        contact_power_vs_native_power_at_contacts=correlation(contact_power, sampled_power),
        interpretation='The B-leading correspondence survives without electrode weighting at contact locations. Its reversal on the full grid exposes a spatial-support limitation; smoothing alone cannot explain the discrepancy. Recruitment definitions still differ between contact and spike observables.',
        statistical_scope='One realization and one transition; descriptive correlations, not independent-cell or patient inference.')
    write('matched_support_audit.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()

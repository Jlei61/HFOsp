"""Recheck existing positive root associations without temporal downsampling.

This verifies the selected continuation segment, not an exhaustive root
search. Original and physically corrected profiles remain distinguished.
"""
from plot_rate_periodic_completion import RateField, families, read, write
from compare_rate_torus_periodic_targets import distances
from audit_rate_survey_filter_states import fingerprint
from scipy.signal import resample
from pathlib import Path
import numpy as np
import os


DATA = Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')


def segment_distance(root, left, right, weights, mesh):
    target, first, second = [resample(x, mesh, axis=0) for x in (root, left, right)]
    aligned, shifts = [], []
    for wave in (first, second):
        _, shift = distances(target[:, None, :], wave, weights)
        aligned.append(np.fft.ifft(np.fft.fft(wave, axis=0) *
            np.exp(-2j*np.pi*np.fft.fftfreq(mesh)*mesh*shift[0])[:, None], axis=0).real)
        shifts.append(float(shift[0]))
    first, second = aligned
    delta = second-first
    inner = lambda a, b: float(np.mean(np.sum(a*b*weights, axis=1)))
    step = np.sqrt(inner(delta, delta))
    assert step > 0
    fraction = inner(target-first, delta) / step**2
    error = target-first-np.clip(fraction, 0., 1.)*delta
    residual = np.sqrt(inner(error, error))
    return dict(fraction=float(fraction), RMS_distance_Hz=float(residual),
        distance_over_local_waveform_step=float(residual/step),
        local_waveform_step_Hz=float(step), common_phase_shifts_cycles=shifts,
        within_sampled_polyline_neighborhood=bool(0 <= fraction <= 1 and residual/step < .2))


def main():
    source = DATA/'current_interval_root_associations.json'
    previous = read(source)
    mapping_source = DATA/'SCL_branch_scan/manifest.json'
    mapping = {str(Path(p['original_orbit']).resolve()): p
               for p in read(mapping_source)['points']}
    fs = families()
    model = RateField()
    assert model.P == 935
    weights = model.geo['group_size']/model.geo['group_size'].sum()
    rows = []
    for bracket in previous['rows']:
        for root in bracket['nearby_roots']:
            if not root['within_sampled_polyline_neighborhood']:
                continue
            index = root['nearest_segment']['left_index']
            originals = [Path(fs[bracket['family']][i]['path']) for i in [index, index+1]]
            corrected = [mapping.get(str(p.resolve())) for p in originals]
            paths = [Path(root['orbit'])] + [Path(q['orbit']) if q else p
                       for p, q in zip(originals, corrected)]
            stamps = [fingerprint(p) for p in paths]
            waves = []
            for p in paths:
                with np.load(p) as z:
                    assert z['r'].shape[1] == 935
                    waves.append(z['r']*1000)
            mesh = 2*max(512, *(len(w) for w in waves))
            write(DATA/'interval_root_temporal_verification_worker.json',
                dict(status='FULL_WAVEFORM_SEGMENT_CHECK', pid=os.getpid(),
                     label=root['display_label'], phase_samples=mesh, completed=len(rows)))
            coarse = segment_distance(*waves, weights, 512)
            fine = segment_distance(*waves, weights, mesh)
            assert stamps == [fingerprint(p) for p in paths]
            row = dict(family=bracket['family'], label=root['display_label'],
                internal_label=root['internal_label'],
                site_indices=[e['site_index'] for e in bracket['ends']],
                selected_left_continuation_index=index, root_validation_status=root['validation_status'],
                orbits=[str(p) for p in paths], profile_fingerprints=stamps,
                physical_branch_profile_evidence=[q['profile_evidence'] if q else None for q in corrected],
                temporal_meshes=[len(w) for w in waves], phase_samples=mesh,
                coarse=coarse, fine=fine,
                association_retained=bool(fine['within_sampled_polyline_neighborhood']),
                scope='The previously selected adjacent segment is tested at preserved temporal bandwidth. '
                      'A match neither validates unchecked branch profiles nor excludes additional roots in the bracket.')
            rows.append(row)
            print(root['display_label'], mesh, fine, flush=True)
    result = dict(status='SELECTED_ASSOCIATIONS_RECHECKED', source=str(source),
        corrected_profile_map_source=str(mapping_source), rows=rows,
        all_selected_associations_retained=all(r['association_retained'] for r in rows),
        temporal_downsampling_in_fine_check=False,
        scope='All previously positive associations, one common phase per full spatial waveform, '
              '935 populations and neuron-count-weighted RMS. This is not an exhaustive search, '
              'root count, physical validation of every interior sample, or interval completeness proof.')
    write(DATA/'interval_root_temporal_verification.json', result)
    write(DATA/'interval_root_temporal_verification_worker.json',
        dict(status='COMPLETE', pid=os.getpid(), completed=len(rows),
             all_selected_associations_retained=result['all_selected_associations_retained']))


if __name__ == '__main__':
    main()

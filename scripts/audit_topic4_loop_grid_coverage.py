#!/usr/bin/env python3
"""Locate the observed first loop relative to the fixed nine-point Z/K grid."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
SOURCE = Path('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923/runs/G30_response0.5_s9108405')


def load(folder, keys):
    parts = {key: [] for key in keys}
    for path in sorted((SOURCE / folder).glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as data:
            for key in keys:
                parts[key].append(data[key])
    return {key: np.concatenate(value) for key, value in parts.items()}


def main():
    data = load('chunks', ['slow_time_ms', 'Z'])
    feedback = load('feedback_chunks', ['time_ms', 'K_mean'])
    assert np.array_equal(data['slow_time_ms'], feedback['time_ms'])
    time = data['slow_time_ms'] / 1000
    use = time < 60
    time, z, k = time[use], data['Z'][use, 0], feedback['K_mean'][use]
    assert np.allclose(np.diff(time), .02)
    outside = (z < .25) | (z > .95) | (k < .02) | (k > 8)
    landmarks = []
    for label, target in [('entry_observer', 9.94), ('high_state_checkpoint', 12.),
                           ('exit_observer', 16.7), ('joint_low_activity', 16.83),
                           ('spatial_template_checkpoint', 20.), ('recovery', 35.),
                           ('first_returned_event', 49.31), ('interictal_checkpoint', 50.),
                           ('second_entry', 59.74)]:
        index = max(0, np.searchsorted(time, target, side='right') - 1)
        landmarks.append(dict(label=label, target_time_s=target, recorded_time_s=float(time[index]),
            mean_Z=float(z[index]), mean_K_over_gL=float(k[index]),
            outside_grid_rectangle=bool(outside[index])))
    result = dict(status='GRID_DOES_NOT_ENCLOSE_OBSERVED_LOOP', source=str(SOURCE),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observation='Uninterpolated20ms native samples,0<=t<60s; landmark uses last sample at or before target.',
        grid_Z=[.25, .75, .95], grid_K_over_gL=[.02, 2., 8.],
        native_Z_range=[float(z.min()), float(z.max())],
        native_K_range=[float(k.min()), float(k.max())], landmarks=landmarks,
        sampled_duration_outside_grid_rectangle_s=float(outside.sum() * .02),
        interpretation='Nine held-field points probe finite conditional responses. They cannot locate the complete entry or exit boundary of the observed loop. A rectangle enclosure would still not establish sampled spatial-field, G/M or basin equivalence.',
        statistical_unit='One selected successful trajectory; samples describe time coverage, not independent trials.',
        changes_to_existing_scientific_queue=False, new_simulations=0)
    (ROOT / 'trajectory_grid_coverage.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

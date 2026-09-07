#!/usr/bin/env python3
"""Freeze a patient-training-only direction amendment before candidate rescoring."""
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_xy_research as base
from src.topic4_xy_direction import (onset_directions, direction_histogram,
    direction_summary, direction_distance, N_BINS, MIN_CONTACTS, MAX_CONDITION,
    KERNEL_WIDTH)

PATH = base.OUT / 'direction_objective_v2.json'
CONTACTS = base.ART / 'results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/contact_shaft_contract.json'


def prepare():
    if PATH.exists():
        result = base.read(PATH)
        if base.sha(CONTACTS) != result['contact_contract']['sha256']:
            raise RuntimeError('direction coordinate contract changed')
        if base.sha(result['patient_training_contract']['path']) != result['patient_training_contract']['sha256']:
            raise RuntimeError('direction training input changed')
        if base.sha(ROOT / 'src/topic4_xy_direction.py') != result['direction_module_sha256']:
            raise RuntimeError('frozen direction objective implementation changed')
        return result
    training, objective = base.training_contract()
    contacts = base.read(CONTACTS)['contacts']
    if [r['contact_name'] for r in contacts] != training['contact_names']:
        raise RuntimeError('training and geometry contact order differ')
    xy = np.array([r['sheet_xy_mm'] for r in contacts])
    patient = training['onsets_ms']
    view = onset_directions(patient, xy)
    hist = direction_histogram(view)
    summary = direction_summary(view)
    rotated = {**view, 'angle_rad': view['angle_rad'] + np.pi / 2}
    reverse = {**view, 'angle_rad': view['angle_rad'] + np.pi}
    dir_scale = direction_distance(hist, direction_histogram(rotated))
    if summary['axial_angle_deg'] is None or dir_scale < 0.01:
        raise RuntimeError('patient direction target is too weak for this objective')
    rng = np.random.default_rng(20260905)
    cloud_nulls = []
    # Destroy temporal order within the existing participation mask. This
    # leaves event counts and per-contact missingness intact, and uses no SNN.
    masks, inv = np.unique(np.isfinite(patient), axis=0, return_inverse=True)
    for unused in range(4):
        null = patient.copy()
        for mi, mask in enumerate(masks):
            rows = np.flatnonzero(inv == mi); cols = np.flatnonzero(mask)
            values = patient[np.ix_(rows, cols)]
            order = np.argsort(rng.random(values.shape), axis=1)
            null[np.ix_(rows, cols)] = np.take_along_axis(values, order, axis=1)
        vec = objective.component_vector(null, training['reference'], training['groups'],
            training['pairs'], training['embedding'], composite=True, components=())
        cloud_nulls.append(vec['D_cloud_composite'])
    cloud_scale = float(np.median(cloud_nulls))
    if not np.isfinite(cloud_scale) or cloud_scale <= 0:
        raise RuntimeError('invalid training-only cloud calibration')
    blocks = np.unique(training['block_ids']); split_distances = []
    for unused in range(32):
        chosen = rng.permutation(blocks)[:len(blocks)//2]
        mask = np.isin(training['block_ids'], chosen)
        def sub(m):
            return {k: v[m] for k, v in view.items()}
        split_distances.append(direction_distance(direction_histogram(sub(mask)), direction_histogram(sub(~mask))))
    result = {
        'status': 'TRAINING_DIRECTION_OBJECTIVE_FIXED_BEFORE_CANDIDATE_RESCORING',
        'version': 'xy_direction_v2', 'created_unix': time.time(),
        'patient_training_contract': {'path': str(training['path']), 'sha256': training['sha256']},
        'contact_contract': {'path': str(CONTACTS), 'sha256': base.sha(CONTACTS)},
        'direction_module_sha256': base.sha(ROOT / 'src/topic4_xy_direction.py'),
        'preparer_sha256': base.sha(Path(__file__)),
        'contact_names': training['contact_names'], 'contact_xy_mm': xy.tolist(),
        'patient_direction_histogram': hist.tolist(), 'patient_direction_summary': summary,
        'definition': {'direction': 'positive onset gradient, earlier to later; signed circle, not absolute cosine',
            'n_bins': N_BINS, 'minimum_contacts': MIN_CONTACTS, 'maximum_geometry_condition': MAX_CONDITION,
            'coherence': 'clipped adjusted R2, with two spatial predictors and intercept',
            'unresolved_mass': 'one minus coherent mass; includes nonplanar, synchronous and unestimable events',
            'distance': 'Gaussian-kernel MMD on signed unit circle plus distinct unresolved state',
            'kernel_width': KERNEL_WIDTH, 'event_denominator': 'all returned families, including unreadable'},
        'normalization': {'D_cloud': cloud_scale, 'D_direction': dir_scale,
            'cloud_training_within_event_permutation_distances': cloud_nulls,
            'direction_90_degree_rotation_distance': dir_scale,
            'direction_180_degree_rotation_distance': direction_distance(hist, direction_histogram(reverse)),
            'source': 'patient training only; fixed null responsiveness, not candidate score ranges'},
        'training_block_split_direction_distances': split_distances,
        'primary_objective': 'J = D_cloud / scale_cloud + D_direction / scale_direction',
        'primary_core_prior_weight': 0., 'weak_prior_weight': 0.1,
        'prior_sensitivity_weights': [0., 0.05, 0.1, 0.2],
        'prior_penalty': 'sin(core-line angle minus training behavioral axis angle)^2',
        'prior_role': 'secondary sensitivity shortlist, no forced endpoints or midpoint; primary remains unconstrained',
        'structural_ee_axis_deg': base.THETA, 'structural_ee_aspect_ratio': 2.,
        'structural_axis_role': 'existing patient-informed fixed input; not independently rediscovered',
        'parameters_searched': ['VTH core x1', 'VTH core y1', 'VTH core x2', 'VTH core y2'],
        'EE_EtoI_ZM_learning': 'off', 'VTH_depth_parameters': 'fixed during XY search',
        'patient_heldout_opened': False, 'ictal_opened': False,
        'full_sheet_behavior': 'additional source-onset gradient diagnostic, not a replacement for contact-matched patient scoring',
        'limitation': 'planar observation summary, not anatomical direction recovery; legacy registration remains patient-informed',
        'final_substrate_frozen': False,
    }
    base.write(PATH, result)
    return result


if __name__ == '__main__':
    result = prepare()
    print({'path': str(PATH), 'status': result['status'],
           'patient_direction_summary': result['patient_direction_summary'],
           'normalization': result['normalization']})

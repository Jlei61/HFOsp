#!/usr/bin/env python3
"""Freeze the selected SNN reference for reduction validation; run no simulation."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.worktrees/topic4-substrate-autapse-fix'
SEARCH = SOURCE / 'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
EXECUTION = SEARCH / 'execution/confirmation_24s'
OUT = ROOT / 'results/topic4_sef_hfo/rate_model_dynamics_validation_v1'
CID = 'v2_1_pop1_de_b_002'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path, expected=None):
    digest = sha(path)
    if expected is not None and digest != expected:
        raise RuntimeError(f'Input changed: {path}')
    return {'path': str(path.resolve()), 'sha256': digest}


def main():
    scores = read(SEARCH / 'g3_scores.json')
    ranked = sorted((c for c in scores['candidates'] if c['ranking_eligible']),
                    key=lambda c: (c['score']['loss_off'], c['candidate_id']))
    assert ranked[0]['candidate_id'] == CID, 'Revisit selection; do not silently change base'
    candidate = next(c for c in read(SEARCH / 'g3_candidates.json')['candidates']
                     if c['candidate_id'] == CID)
    snapshot_path = EXECUTION / 'runtime_snapshot.json'
    snapshot = read(snapshot_path)
    # Verify every historically frozen source before preserving its exact bytes.
    source_records = {rel: record(SOURCE / rel, digest)
                      for rel, digest in snapshot['source_hashes'].items()}
    inputs = [record(Path(path), digest)
              for path, digest in snapshot['input_hashes'].items()]
    runs = []
    for dyn in (7101, 7102):
        path = EXECUTION / 'workers' / f'{CID}_topo_6101_dyn_{dyn}.json'
        worker = read(path)
        assert worker['candidate_id'] == CID and worker['topology_seed'] == 6101
        assert worker['dynamics_seed'] == dyn
        assert worker['simulation']['actual_duration_ms'] == 24000
        assert worker['mechanism_freeze']['Z_M'] == 'off'
        assert worker['mechanism_freeze']['edge_coefficients_all_zero']
        record(snapshot_path, worker['provenance']['source_hash_snapshot']['sha256'])
        runs.append({'dynamics_seed': dyn, 'worker': record(path),
                     'arrays': record(Path(worker['arrays']['path']), worker['arrays']['sha256']),
                     'static_array_identity': worker['static_array_identity'],
                     'score': ranked[0]['units'][f'topo_6101_dyn_{dyn}']['score']})
    assert runs[0]['static_array_identity'] == runs[1]['static_array_identity']
    reference = read(Path(runs[0]['worker']['path']))
    graph = reference['network_cache_source']
    graph_record = record(Path(graph['path']), graph['sha256'])
    config = read(EXECUTION / 'execution_config.json')
    transition = config['inputs']['transition_config']
    transition_path = SOURCE / transition['path']
    inputs.append(record(transition_path, transition['sha256']))
    evidence = [record(SEARCH / name) for name in (
        'g3_scores.json', 'g3_candidates.json', 'scientific_review.md',
        'g3_confirmation_analysis.json', 'g3_six_second_segments.json')]
    OUT.mkdir(parents=True, exist_ok=True)
    for rel, entry in source_records.items():
        dest = OUT / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry['path'], dest)
        entry['snapshot'] = record(dest, entry['sha256'])
    for entry in inputs:
        source = Path(entry['path'])
        dest = OUT / 'input_snapshot' / source.relative_to(SOURCE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        entry['snapshot'] = record(dest, entry['sha256'])
    manifest = {
        'schema': 'topic4_rate_model_dynamics_validation_base_v1',
        'status': 'REFERENCE_IDENTITY_FROZEN_DYNAMICS_VALIDATION_NOT_RUN',
        'date': '2026-09-08', 'candidate_id': CID,
        'authorization': 'User accepted one fixed data-driven dual-core reference for reduction validation.',
        'selection': {'rule': 'Lowest completed G3 equal-run L_off; candidate is development-selected using G3.',
                      'loss_off': ranked[0]['score']['loss_off'],
                      'global_optimality_claim': False,
                      'new_independent_test_claim': False},
        'topology_seed': 6101,
        'topology_choice': 'Smallest confirmation topology ID, not selected by per-network score.',
        'candidate': candidate,
        'realized_geometry': reference['xy_geometry_audit'],
        'realized_ellipse': {k: v for k, v in reference['mechanism_freeze']['ellipse_audit']['weighted_geometry_after'].items()
                            if k != 'weighted_distance_histogram'},
        'static_array_identity': reference['static_array_identity'],
        'base_graph': graph_record,
        'transformed_graph_reconstruction_verified_this_freeze': False,
        'reconstruction_requirement': 'Apply historical worker mapping and compare all static_array_identity hashes before any new simulation; graph cache is before candidate transformations.',
        'reference_runs': runs, 'source_records': source_records,
        'runtime_snapshot': record(snapshot_path), 'inputs': inputs, 'evidence': evidence,
        'physical_contract': {
            'sheet_mm': 20, 'n_e': 32000, 'n_i': 8000, 'dt_ms': 0.1,
            'autapses': False, 'baseline_Z_M': 'off',
            'preserve': ['realized positions and boundaries', 'empirical per-neuron thresholds',
                         'all transformed pathway weights and realized delay bins',
                         'synaptic rise and decay with original jump and impulse-area convention',
                         'external input mean, OU law, spatial covariance and clipping order'],
            'tau_gaba_intervention': 'Audit corrected: fixed jump/rise preserves infinite-horizon normalized-current impulse area; decay changes peak and timing. Finite-window and closed-loop total inhibition can change.',
            'new_Z_M': 'Separate matched extension only after baseline reduction validation; never inherit old GIF-base settings silently.'},
        'validation_state': {'reference_files_verified': True,
                             'n_frozen_source_files': len(source_records),
                             'same_graph_reference_identity_verified': True,
                             'new_simulations_run': 0,
                             'interictal_rate_model_pass': False,
                             'sustained_oscillation_rate_model_pass': False,
                             'bifurcation_analysis_released': False},
    }
    target = ROOT / 'config/topic4_rate_model_dynamics_validation_v1.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'manifest': str(target), 'source_files': len(source_records),
                      'candidate': CID, 'topology': 6101, 'references_verified': len(runs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

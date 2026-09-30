#!/usr/bin/env python3
"""Final bounded-campaign delivery audit; never certify a bifurcation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import psutil
import numpy as np
import analyze_topic4_loop_zk_conditional as conditional
import analyze_topic4_loop_axis_native as native

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
CANDIDATE = REPO / 'results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924'
STREAMS = ['chunks', 'actual_current_chunks', 'conditional_drift_chunks',
           'feedback_chunks', 'global_response_chunks', 'intrinsic_adaptation_chunks',
           'mechanism_chunks', 'regional_chunks']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def json_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported audit value: {type(value).__name__}')


def recorded_files(value, base):
    found = []
    if isinstance(value, list):
        for item in value:
            found += recorded_files(item, base)
    elif isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            path = Path(value['path'])
            if not path.is_absolute():
                path = base / path
            assert sha(path) == value['sha256'], path
            found.append(str(path))
        else:
            for item in value.values():
                found += recorded_files(item, base)
    return found


def visual(folder, result_path):
    review = read(folder / 'review.json')
    assert review['result_sha256'] == sha(result_path)
    assert review['field_spike_integrity'] == 'PASS'
    approved = read(folder / 'agent_visual_review.json')
    assert approved['status'] == 'AGENT_VISUAL_REVIEWED'
    files = recorded_files(approved['files'], folder)
    expected = [str(folder / 'figures/full_rate_context.png')]
    expected += [g['example']['file'] for g in review['groups'] if g['example']]
    assert set(expected) <= set(files), (folder, 'Unreviewed native images')
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, help='Final reproducibility label')
    args = parser.parse_args()
    structure = ROOT / 'axis_controls/conditional_runs'
    native_root = ROOT / 'axis_controls/native_runs'
    for path, stage in [(ROOT / 'status.json', 'COMPLETE'),
                        (structure / 'status.json', 'COMPLETE'),
                        (native_root / 'status.json', 'COMPLETE'),
                        (ROOT / 'postprocessing/status.json', 'COMPLETE_ANALYSIS_CANDIDATES')]:
        status = read(path)
        assert status['stage'] == stage, (path, status['stage'])
        assert not status.get('failed', status.get('failures', status.get('failed_analysis', [])))
        pid = status.get('supervisor_pid', status.get('watcher_pid'))
        assert not pid or not psutil.pid_exists(pid), (path, 'Supervisor still live', pid)
    frozen = {}; cases = []; reference_input = None
    for graph, root in [('current', ROOT), ('rotated', structure / 'rotated'),
                        ('isotropic', structure / 'isotropic')]:
        protocol = read(root / 'protocol.json')
        for name, digest in protocol['source_hashes'].items():
            assert sha(Path(name)) == digest, name
            frozen[name] = digest
        for path, key in [(REPO / 'scripts/run_topic4_loop_zk_conditional.py', 'runner_sha256')]:
            assert sha(path) == protocol[key]; frozen[str(path)] = protocol[key]
        if graph != 'current':
            path = REPO / 'scripts/run_topic4_loop_axis_conditional.py'
            assert sha(path) == protocol['axis_conditional_sha256']
            frozen[str(path)] = protocol['axis_conditional_sha256']
            assert sha(Path(protocol['graph_audit'])) == protocol['graph_audit_sha256']
        names = read(root / 'queue.json')['names']
        assert len(names) == (18 if graph == 'current' else 8)
        old = conditional.OUT; conditional.OUT = root
        try:
            for name in names:
                folder = root / 'runs' / name; result_path = folder / 'result.json'
                result = read(result_path); job = read(root / 'jobs' / f'{name}.json')
                assert result['status'] == 'COMPLETE' and result['job'] == job
                assert result['identity'] == protocol['identity']
                assert result['clamp_Z_and_K'] and result['diagnostic_only']
                assert not result['counts_as_autonomous_loop']
                assert result['endogenous_G_and_M_dynamic']
                assert result['end_s'] == job['horizon_s']
                start = round(job['branch_start_s'] * 10000)
                stop = round(job['horizon_s'] * 10000)
                assert stop - start == 300000
                streams = {}
                for stream in STREAMS:
                    last = start; count = 0
                    for path in sorted((folder / stream).glob('*.npz')):
                        assert '.tmp.' not in path.name, path
                        lo, hi = map(int, path.stem.split('_'))
                        assert lo == last + (200 if stream == 'conditional_drift_chunks' else 0), path
                        assert hi > lo
                        if stream == 'chunks':
                            with np.load(path) as z:
                                assert int(z['start_step']) == lo and int(z['end_step']) == hi
                        last = hi; count += 1
                    assert last == stop and count == 15, (name, stream, last, count)
                    streams[stream] = dict(blocks=count, end_step=last)
                row, inputs = conditional.analyze(name)
                assert row['duration_s'] == 30. and row['full_horizon']
                if reference_input is not None:
                    assert np.array_equal(inputs, reference_input), (graph, name)
                reference_input = inputs
                images = visual(root / 'spatial_review' / name, result_path)
                cases.append(dict(graph=graph, name=name, duration_s=30.,
                    identity_matches_protocol=True, future_input_values_exact=True,
                    native_count_field_and_held_Z_checks='PASS', streams=streams,
                    state=row['finite_window_state'], mean_Hz=row['tail_mean_Hz'],
                    reviewed_images=images, result_sha256=sha(result_path)))
        finally:
            conditional.OUT = old
    assert len(cases) == 34
    native_rows = [native.one('current', True)] + [native.one(c) for c in ['rotated', 'isotropic']]
    for row in native_rows:
        assert row['full120s'] and row['spike_field_integrity'] == 'PASS'
        assert row['paired_exogenous_records_exact']
        if row['condition'] != 'current':
            root = native_root / row['condition']; result_path = root / 'runs' / row['name'] / 'result.json'
            result = read(result_path); protocol = read(root / 'protocol.json')
            assert result['status'] == 'COMPLETE' and result['end_s'] == 120.
            assert result['identity'] == protocol['identity']
            assert result['no_external_intervention'] and not result['clamp_Z_and_K']
            assert not result['Z_reset'] and not result['M_reset']
            assert sha(REPO / 'scripts/run_topic4_loop_axis_native.py') == protocol['axis_runner_sha256']
            visual(root / 'spatial_review' / row['name'], result_path)
    metadata_paths = [CANDIDATE / 'metadata.json', CANDIDATE / 'conditional_figure_metadata.json',
                      CANDIDATE / 'conditional_spatial_metadata.json', structure / 'figure_metadata.json',
                      native_root / 'full_comparison/metadata.json'] + [
                      structure / 'event_comparison' / h / 'metadata.json' for h in ['high', 'interictal']]
    figures = []
    for path in metadata_paths:
        meta = read(path)
        reviews = [v for k, v in meta.items() if k.startswith('agent_') and 'review' in k and isinstance(v, str)]
        assert any(v.startswith('PASS') for v in reviews), (path, reviews)
        assert meta['human_review'] == 'PENDING'
        if 'source_sha256' in meta:
            assert sha(Path(meta['source'])) == meta['source_sha256'], path
        if 'analysis_sha256' in meta:
            assert sha(Path(meta['analysis'])) == meta['analysis_sha256'], path
        files = recorded_files(meta['files'], path.parent)
        assert any(x.endswith('.png') for x in files) and any(x.endswith('.pdf') for x in files)
        figures.append(dict(metadata=str(path), sha256=sha(path), checked_files=files))
    assert read(structure / 'figure_metadata.json')['completed'] == 24
    qa = read(CANDIDATE / 'publication_qa.json')
    assert qa['status'] == 'PASS_TECHNICAL_AND_AGENT_VISUAL'
    for path, digest in qa['source_dependency_hashes'].items():
        assert sha(Path(path)) == digest, path
    gate = read(ROOT / 'conductance_response/static_calibration_v1/gate_review.json')
    assert gate['status'] == 'REJECTED_FOR_BIFURCATION_THIS_ROUND'
    assert not gate['static']['formal_bifurcation_allowed']
    assert not gate['static']['spatial_validated'] and not gate['static']['transient_validated']
    snap = ROOT / 'reproducibility' / args.snapshot
    manifest = read(snap / 'manifest.json')
    assert manifest['status'] == 'PASS_ARCHIVED_CODE_AND_CONFIGURATION'
    for item in manifest['files']:
        assert sha(snap / item['snapshot']) == item['sha256']
        assert sha(Path(item['original'])) == item['sha256'], item['original']
    for filename in ['final_scientific_review.md', 'delivery_index.md']:
        assert (ROOT / filename).stat().st_size > 1000
    payload = dict(status='PASS_BOUNDED_DELIVERY_FORMAL_BIFURCATION_NOT_ESTABLISHED',
        verified_utc=datetime.now(timezone.utc).isoformat(), conditional_cases=cases,
        conditional_total=34, original18_reused_in_structural_map=8,
        native_common_window_s=120, native_rows=native_rows, frozen_sources=frozen,
        figures=figures, reproducibility_snapshot=str(snap), snapshot_files=len(manifest['files']),
        rate_closure='STATIC_VALIDATION_FAIL; no transient/spatial certification or formal continuation',
        human_review='PENDING', formal_bifurcation='NOT_ESTABLISHED',
        scope='Completed the authorized bounded native experiment and candidate-delivery contract, including its explicit failed-rate-gate fallback. Does not certify all seeds cycle, a stable attractor, pure axis-angle causality, clinical equivalence or a complete transition boundary.')
    (ROOT / 'completion_audit.json').write_text(json.dumps(payload, indent=2, default=json_native) + '\n')
    print(json.dumps({k: payload[k] for k in ['status', 'conditional_total', 'snapshot_files', 'human_review', 'formal_bifurcation']}))


if __name__ == '__main__':
    main()

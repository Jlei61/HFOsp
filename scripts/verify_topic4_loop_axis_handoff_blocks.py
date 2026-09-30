#!/usr/bin/env python3
"""Check the first committed conditional block after each device handoff."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
OUT = ROOT / 'axis_controls/conditional_runs'
ARCHIVE = ROOT / 'backend_handoffs/axis_cpu_gpu'
STREAMS = ['chunks', 'actual_current_chunks', 'conditional_drift_chunks',
           'feedback_chunks', 'global_response_chunks', 'intrinsic_adaptation_chunks',
           'mechanism_chunks', 'regional_chunks']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(folder):
    return sorted(p for p in folder.glob('*.npz') if '.tmp.' not in p.name)


def check(path):
    transfer = json.loads(path.read_text())
    condition, name, start = transfer['condition'], transfer['name'], transfer['step']
    folder = OUT / condition / 'runs' / name
    answer = dict(condition=condition, name=name, handoff=str(path), checkpoint_start_step=start)
    resumed = []
    for block in files(folder / 'chunks'):
        with np.load(block) as data:
            if int(data['start_step']) >= start:
                resumed.append(block)
    if not resumed:
        return dict(answer, status='PENDING_FIRST_RESUMED_BLOCK')
    first = resumed[0]
    with np.load(first) as data:
        assert int(data['start_step']) == start, ('Gap after handoff', first)
        end = int(data['end_step'])
    streams = {}
    for stream in STREAMS:
        matching = [p for p in files(folder / stream) if int(p.stem.split('_')[-1]) == end]
        if not matching:
            return dict(answer, status='PENDING_FIRST_BLOCK_STREAM_FLUSH', missing_stream=stream)
        assert len(matching) == 1
        # Drift filenames begin at the first20ms right endpoint, unlike the
        # main observer's block start. Compare each stream with its own paired
        # reference, preserving the native sampling convention.
        reference_path = ROOT / 'runs' / name / stream / matching[0].name
        assert reference_path.exists(), reference_path
        with np.load(matching[0]) as actual, np.load(reference_path) as reference:
            assert set(actual.files) == set(reference.files), stream
            for key in actual.files:
                assert actual[key].shape == reference[key].shape, (stream, key)
            times = [key for key in actual.files if key.endswith('time_ms')]
            assert times, ('No timestamp arrays in stream', stream)
            for key in times:
                assert np.array_equal(actual[key], reference[key]), (stream, key)
        streams[stream] = dict(path=str(matching[0]), sha256=sha(matching[0]),
                               filename_matches_reference_stream=True, all_array_shapes_match=True,
                               timestamp_arrays_exact=times)
    references = [p for p in files(ROOT / 'runs' / name / 'chunks')
                  if int(p.stem.split('_')[-1]) == end]
    assert len(references) == 1
    with np.load(first) as data, np.load(references[0]) as reference:
        assert int(reference['start_step']) == start
        assert np.array_equal(data['inputs'], reference['inputs'])
        assert np.array_equal(data['Z'][:, :8], reference['Z'][:, :8])
        assert np.array_equal(data['regions_1ms'][:, :3].sum(1), data['spikes_1ms'][:, 0])
        assert np.array_equal(data['regions_1ms'][:, 3:].sum(1), data['spikes_1ms'][:, 1])
        assert np.array_equal(data['field_5ms'].sum(1), data['spikes_1ms'][:, 0].reshape(-1, 5).sum(1))
        assert len(data['time_ms']) == (end - start) // 10
        for key in ['time_ms', 'slow_time_ms', 'field_time_ms', 'lfp_time_ms']:
            assert np.array_equal(data[key], reference[key])
    reference_fb = [p for p in files(ROOT / 'runs' / name / 'feedback_chunks')
                    if int(p.stem.split('_')[-1]) == end]
    assert len(reference_fb) == 1
    with np.load(streams['feedback_chunks']['path']) as data, np.load(reference_fb[0]) as reference:
        assert np.array_equal(data['time_ms'], reference['time_ms'])
        assert np.array_equal(data['K_mean'], reference['K_mean'])
    assert sha(Path(transfer['backup'])) == transfer['checkpoint_sha256']
    answer.update(status='PASS_FIRST_RESUMED_BLOCK_CONTINUITY', first_committed_end_step=end,
                  streams=streams, saved_checkpoint_backup_intact=True,
                  future_input_records_exact_to_original_graph=True,
                  held_Z_summary_columns0_to7_exact=True, held_K_mean_exact=True,
                  native_spike_field_conservation=True, native_sample_times_exact=True,
                  stream_definition='Eight clamped-branch streams; conditional drift replaces autonomous Zbudget.',
                  scope='First resumed block only. Whole30s analysis remains required; different graphs need not have matching spikes.')
    (path.parent / 'first_block_verification.json').write_text(json.dumps(answer, indent=2) + '\n')
    return answer


def main():
    rows = [check(path) for path in sorted(ARCHIVE.glob('*/*/*/handoff.json'))]
    pending = sum(row['status'].startswith('PENDING') for row in rows)
    result = dict(status='PENDING_BLOCKS' if pending else 'PASS_FIRST_BLOCKS', transfers=rows,
                  pending=pending, producer_sha256=sha(Path(__file__)))
    (ARCHIVE / 'first_blocks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], pending=pending,
                          rows=[{k: row.get(k) for k in ['condition', 'name', 'status', 'first_committed_end_step']} for row in rows])))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Both histories and every native stream must match before GPU1 handoff."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import numpy as np
import run_topic4_loop_axis_conditional as run

QA = run.PRIMARY / 'qa/axis_cuda_device1_route'
REPO = Path(__file__).resolve().parents[1]
STREAMS = ['chunks', 'actual_current_chunks', 'conditional_drift_chunks',
           'feedback_chunks', 'global_response_chunks', 'intrinsic_adaptation_chunks',
           'mechanism_chunks', 'regional_chunks']


def main():
    run.OUT = QA
    run.reference_qa()  # Includes complete saved engine state, not just readouts.
    rows = []
    for name, reference in [('qa_high', 'clamp_mechanism_qa'),
                            ('qa_interictal', 'clamp_input_qa')]:
        folder = QA / 'reference/runs' / name
        target = run.PRIMARY / 'runs' / reference
        result = run.native.base.read(folder / 'result.json')
        assert result['status'] == 'COMPLETE'
        assert result['runtime_backend']['actual_device'] == 1
        checks = {}
        for stream in STREAMS:
            paths = sorted((folder / stream).glob('*.npz'))
            old = sorted((target / stream).glob('*.npz'))
            assert len(paths) == len(old) == 1, stream
            assert paths[0].name == old[0].name, stream
            with np.load(paths[0]) as a, np.load(old[0]) as b:
                assert set(a.files) == set(b.files), (name, stream, a.files, b.files)
                for key in a.files:
                    assert a[key].dtype == b[key].dtype, (name, stream, key)
                    assert np.array_equal(a[key], b[key],
                                          equal_nan=a[key].dtype.kind in 'fc'), (name, stream, key)
                checks[stream] = dict(status='PASS_ALL_ARRAYS_EXACT', keys=a.files)
        rows.append(dict(name=name, full_engine_bitwise=True, streams=checks))
    gate = run.native.base.read(QA / 'reference/route_qa.json')
    gate.update(device=1, checks=rows,
        wrapper_sha256=run.native.base.sha(REPO / 'scripts/run_topic4_loop_axis_cuda_override.py'),
        backend_sha256=run.native.base.sha(REPO / 'src/topic4_cuda_ordered_scatter.py'),
        verifier_sha256=run.native.base.sha(__file__),
        scope='Both original-graph full clamped histories,0.2s each,device1. '
              'Complete engine states and every array in all eight observation streams '
              'must equal the existing serial reference. Execution-only qualification; no scientific job added.')
    run.native.write(QA / 'gate.json', gate)
    print(json.dumps(dict(status=gate['status'], device=1, histories=2,
                          complete_engine_exact=True, streams_exact_per_history=len(STREAMS))))


if __name__ == '__main__':
    main()

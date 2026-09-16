"""Read-only source/data checks plus reproducibility and figure-file snapshots."""
import sys, time, hashlib, platform, importlib.metadata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import pandas as pd
from PIL import Image
from scripts.patient_state_v1.common import ROOT, RUN, write_json
from scripts.patient_state_v1.prepare import load_events, outside_seizures

def main():
    original = json.loads((RUN/'data_audit.json').read_text())
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in original['sources'].items()}
    assert all(hashes.values())
    inv, starts, ends, labels, segments, ranges = load_events()
    keep = outside_seizures(starts, ends, inv)
    frozen = pd.read_csv(RUN/'events.csv')
    for name, values in [('start_epoch', starts[keep]), ('end_epoch', ends[keep])]:
        assert np.allclose(frozen[name], values, rtol=0, atol=1e-6)
    assert np.array_equal(frozen.label_tb, labels[keep])
    assert np.array_equal(frozen.coverage_segment, segments[keep])
    assert inv == json.loads((RUN/'seizures.json').read_text())
    assert np.allclose(ends-starts, .25, rtol=0, atol=1e-6)
    assert np.all(starts[1:] >= ends[:-1]-1e-6)
    exposure = pd.read_csv(RUN/'exposure.csv')
    assert np.all(exposure.end_epoch > exposure.start_epoch)
    assert np.all(exposure.start_epoch.to_numpy()[1:] >= exposure.end_epoch.to_numpy()[:-1]-1e-6)
    for row in exposure.itertuples():
        assert all(row.end_epoch <= s['onset']+1e-6 or row.start_epoch >= s['offset']-1e-6 for s in inv)
    windows = pd.read_csv(RUN/'frozen_seizure_windows.csv')
    for row in windows.itertuples():
        selected = (starts[keep] >= row.start_epoch) & (ends[keep] <= row.end_epoch) & (starts[keep] < row.end_epoch)
        assert int(labels[keep][selected].sum()) == row.n_tb
        assert int(selected.sum()-labels[keep][selected].sum()) == row.n_ta
    write_json(RUN/'input_delivery_audit.json', dict(status='PASS', created_unix=time.time(), n_source_events=len(starts), n_interictal_events=int(keep.sum()), n_ta=int((labels[keep] == 0).sum()), n_tb=int(labels[keep].sum()), n_source_artifact_blocks=len(ranges), n_continuous_coverage_segments=len(np.unique(segments)), n_frozen_windows=len(windows), source_hash_checks=hashes, raw_artifact_times_reloaded_and_matched=True, clinical_exclusions_reloaded_and_matched=True, window_duration_250ms_and_no_overlap=True, exposure_disjoint_and_outside_ictal=True, scope='All available fixed-label artifacts, not all raw monitoring or detector completeness'))
    packages = {}
    for name in ['numpy', 'pandas', 'scipy', 'numba', 'arviz', 'matplotlib', 'cupy-cuda12x', 'cupy-cuda11x']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts/patient_state_v1').glob('*.py'))}
    write_json(RUN/'runtime_and_source_snapshot.json', dict(created_unix=time.time(), python=sys.version, executable=sys.executable, platform=platform.platform(), packages=packages, source_hashes=source_hashes))
    readme = (RUN/'figures/README.md').read_text(); figures = []
    for path in sorted((RUN/'figures').glob('*.png')):
        with Image.open(path) as im:
            im.load(); size = im.size
        entry = dict(filename=path.name, pixels=size, decoded=True, pdf_exists=path.with_suffix('.pdf').exists(), readme_entry=('### '+path.name) in readme)
        assert entry['pdf_exists'] and entry['readme_entry']
        figures.append(entry)
    write_json(RUN/'figure_file_audit.json', dict(status='PASS', created_unix=time.time(), figures=figures, scope='File decoding, companion PDF and README only; this is neither scientific validation nor human visual acceptance'))
    print(json.dumps(dict(status='PASS', events=int(keep.sum()), source_blocks=len(ranges), figures=len(figures), scripts=len(source_hashes))))

if __name__ == '__main__':
    main()

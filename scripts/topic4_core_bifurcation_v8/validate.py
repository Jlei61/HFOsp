"""Check display artifacts and the few numerical readouts recomputed in v8."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/topic4_sef_hfo/core_bifurcation_types_v8_20260916'
V7 = ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916'
FIG = OUT/'figures'
read = lambda p: json.loads(p.read_text())
manifest = read(OUT/'figure_manifest.json')
display = read(OUT/'display_validation.json')
assert hashlib.sha256((V7/'displayed_curve_sequences.json').read_bytes()).hexdigest() == display['source_curve_sha256']
assert len(manifest) == 7 and display['numbered_native_markers'] == 0
assert .79 < display['fraction_vertical_range_below_100_hz'] < .81
assert all(r['axes'] == 1 for r in manifest if r['name'].startswith(('00_', '01_')))

checked = []
readme = (FIG/'README.md').read_text()
for row in manifest:
    with Image.open(FIG/(row['name']+'.png')) as im:
        im.load()
        assert list(im.size) == row['pixels']
    assert '### '+row['name'] in readme
for path in sorted(FIG.glob('*.pdf')):
    xml = subprocess.check_output(['pdftotext', '-bbox', str(path), '-'], text=True)
    pages = ET.fromstring(xml).findall('.//{*}page')
    assert len(pages) == (7 if path.name == 'core_bifurcation_types.pdf' else 1)
    words = []
    for page in pages:
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        for word in page.findall('.//{*}word'):
            a = word.attrib
            assert 0 <= float(a['xMin']) <= float(a['xMax']) <= width, (path, word.text)
            assert 0 <= float(a['yMin']) <= float(a['yMax']) <= height, (path, word.text)
            words.append(word.text or '')
    assert 'Numbered native' not in ' '.join(words)
    checked.append(dict(file=path.name, pages=len(pages), word_count=len(words)))

metrics = read(OUT/'period_doubling_waveform_metrics.json')
for row in metrics:
    assert Path(row['source']).exists()
    assert max(row['max_half_difference_hz']) > .1
    if row['label'] in ('PD1', 'PD3'):
        assert row['max_transverse'] < 1
    else:
        assert row['max_transverse'] > 1
    if row['A_burst_peak_intervals_ms'] is not None:
        assert abs(sum(row['A_burst_peak_intervals_ms'])-row['full_repeat_ms']) < 1e-9
        assert all(.49 < v/row['full_repeat_ms'] < .51 for v in row['A_burst_peak_intervals_ms'])
    else:
        assert min(row['min_hz'][:2]) > 100
for row in read(OUT/'critical_mode_components.json'):
    assert np.isclose(sum(row['right_percent']), 100)
    assert np.isclose(sum(row['left_percent']), 100)

report = dict(status='PASS', png_count=len(manifest), pdf_checks=checked,
    main_axes_are_single=True, source_periodic_curves_unchanged=True,
    waveform_period_and_stability_checks='PASS', mode_normalization='PASS',
    human_acceptance='PENDING',
    scope='Display and saved-orbit readouts only; inherited numerical bifurcation validation remains v2-v7.')
(OUT/'figure_validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))

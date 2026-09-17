"""Validate composition artifacts, source identity, and orbit correspondence."""
import hashlib
import json
import sys
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/topic4_core_bifurcation_v2'))
from rate_paths import saved_path
OUT=ROOT/'results/topic4_sef_hfo/core_bifurcation_composite_v9_20260916'
FIG=OUT/'figures'
meta=json.loads((OUT/'metadata.json').read_text())
assert hashlib.sha256(saved_path(meta['source_curve_path']).read_bytes()).hexdigest()==meta['source_curve_sha256']
assert meta['curve_sequences']==8 and meta['periodic_curve_points']==841
rows={r['letter']:r for r in meta['source_conditions']}
assert list(rows)==list('abcde')
for left,right in meta['coexistence_pairs']:
    assert abs(rows[left]['g']-rows[right]['g'])<1e-11
assert rows['a']['kind']=='Stable equilibrium'
assert all(rows[k]['kind']=='Stable periodic orbit' for k in 'bcde')
readouts=[]
for letter,row in rows.items():
    assert saved_path(row['path']).exists()
    if row['kind']=='Stable periodic orbit':
        z=np.load(saved_path(row['path']))
        assert abs(float(z['g'])-row['g'])<1e-11
        assert abs(float(z['T'])-row['T'])<1e-8
        difference=float(np.max(abs(z['r'].mean(0)*1000-row['mean'])))
        assert difference<1e-8
        readouts.append(dict(letter=letter,mean_rate_error_hz=difference,period_ms=float(z['T'])))
assert rows['d']['lo'][0]<.01 and rows['e']['lo'][0]>200
assert rows['d']['lo'][1]>200 and rows['e']['lo'][1]>200

for item in meta['figures']:
    with Image.open(FIG/(item['name']+'.png')) as im:
        im.load()
        assert list(im.size)==item['pixels']
    assert '### '+item['name'] in (FIG/'README.md').read_text()
checks=[]
for f in sorted(FIG.glob('*.pdf')):
    xml=subprocess.check_output(['pdftotext','-bbox',str(f),'-'],text=True)
    pages=ET.fromstring(xml).findall('.//{*}page')
    assert len(pages)==(2 if f.name.endswith('_comparison.pdf') else 1)
    for page in pages:
        width,height=float(page.attrib['width']),float(page.attrib['height'])
        for word in page.findall('.//{*}word'):
            a=word.attrib
            assert 0<=float(a['xMin'])<=float(a['xMax'])<=width,(f,word.text)
            assert 0<=float(a['yMin'])<=float(a['yMax'])<=height,(f,word.text)
    checks.append(dict(file=f.name,pages=len(pages),all_text_within_page=True))
answer=dict(status='PASS',png_count=2,pdf_checks=checks,
    original_bifurcation_curves_unchanged=True,source_readouts=readouts,
    matched_coexistence_parameters=True,model_changed=False,
    human_acceptance='PENDING',scope='Composition and inherited orbit correspondence; no new dynamical proof.')
(OUT/'validation.json').write_text(json.dumps(answer,indent=2)+'\n')
print(json.dumps(answer,indent=2))

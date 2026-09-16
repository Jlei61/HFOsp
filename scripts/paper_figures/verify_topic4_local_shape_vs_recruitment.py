"""Independent delivery checks for the frozen-output local-shape diagnostic."""
from pathlib import Path
import csv, json, hashlib, subprocess
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1/local_shape_vs_recruitment_audit'
F=P/'figures';Q=P/'qa';Q.mkdir(exist_ok=True)
audit={'png':[],'pdf':[],'gif':[]}
for path in sorted(F.glob('*.png')):
    with Image.open(path) as im:
        im.load();audit['png'].append(dict(path=str(path),size=im.size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
for path in sorted(F.glob('*.pdf')):
    subprocess.run(['pdfinfo',str(path)],check=True,capture_output=True,text=True)
    audit['pdf'].append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
for path in sorted(F.glob('*.gif')):
    with Image.open(path) as im:
        n=im.n_frames
        for i in range(n):im.seek(i);im.convert('RGB').load()
        canvas=Image.new('RGB',(1800,960),'white')
        for ui in range(4):
            for j,offset in enumerate([14,16,20]):
                im.seek(ui*41+offset);tile=im.convert('RGB');tile.thumbnail((600,240));canvas.paste(tile,(j*600,ui*240))
        canvas.save(Q/(path.stem+'_frames.png'))
        assert n==164
        audit['gif'].append(dict(path=str(path),frames=n,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
subprocess.run(['pdftoppm','-scale-to','1800','-png','-singlefile',str(F/'c1_factorial_distributions.pdf'),str(Q/'c1_pdf_render')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
read=lambda path:list(csv.DictReader(path.open()))
r=read(P/'event_observables.csv');s=read(P/'score_components.csv')
assert sum(x['candidate']=='0' for x in r)==64
assert sum(x['candidate']!='0' and x['arm']=='original' for x in r)==1082
assert len(s)==112
for ci in range(1,8):
    for unit in {x['unit'] for x in s if x['candidate']==str(ci)}:
        a={x['arm']:float(x['loss']) for x in s if x['candidate']==str(ci) and x['unit']==unit}
        assert abs(a['original']-a['width'])<1e-7 and abs(a['timing']-a['both'])<1e-7
old=read(P.parent/'timing_capacity_diagnostics/event_median_contact_widths.csv')
m=json.loads((P/'manifest.json').read_text());cands=m['candidate_ids']
lookup={(x['source'],x['event_id']):float(x['median_contact_width_ms']) for x in old}
matched=0
for x in r:
    if x['candidate'] in ['1','2'] and x['arm']=='original':
        key=(cands[int(x['candidate'])-1]+'/'+x['unit'],x['event_id'])
        assert np.isclose(float(x['contact_width_ms']),lookup[key],atol=1e-7);matched+=1
# Verify source identity and channel order; model arrays themselves were decoded
# by the producer, and their historical hashes are retained in the manifest.
reference_names=np.array(m['display_contact_names'])
for source in m['patient_sources']:
    path=Path(source['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
    with np.load(path) as z:names=z['contact_names'].astype(str)
    assert np.array_equal(names,source['source_contact_names'])
    assert np.array_equal(names[source['display_index']],reference_names)
for source in m['model_sources']:
    with np.load(Path(source['worker_path']).with_suffix('.npz')) as z:
        assert np.array_equal(z['contact_names'].astype(str),reference_names)
for path,digest in m['source_files'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
assert len(audit['png'])==len(audit['pdf'])==12 and len(audit['gif'])==4
audit.update(patient_events=64,model_events=1082,model_units=28,arms=4,original_widths_reproduced=matched,
             min_displayed_mass_fraction=min(min(x['plotted_mass_fraction']) for x in m['clips']),
             min_animated_mass_fraction=min(min(x['animated_mass_fraction']) for x in m['clips']),
             max_centroid_error_ms=m['max_centroid_error_ms'],max_relative_mass_error=m['max_relative_mass_error'],
             human_visual_acceptance='PENDING',agent_visual_review='PENDING',
             loss_invariant_to_width_in_all_28_units=True)
(P/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ['png','pdf','gif']},indent=2))

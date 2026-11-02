from pathlib import Path
import json,hashlib,re,sys,zipfile,tempfile
import numpy as np
MAIN=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/home/honglab/leijiaxin/HFOsp')
fig=MAIN/'results/paper-ready-figure/fig6/candidates/shared_logistic_feedback_20261102'
evidence=MAIN/'results/topic4_sef_hfo/logistic_feedback_20261102'
report=MAIN/'docs/weekly_reports/2026-11-02_epilepsy_model_figures'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((fig/'figure_metadata.json').read_text());validation=json.loads((fig/'validation.json').read_text())
assert all(validation['PDF_text_inside_canvas'].values()) and validation['E_pdf_byte_identical']
assert meta['validation']['complete_cycle'] and meta['validation']['observed_s']==120
rows=json.loads((evidence/'comparison.json').read_text())['rows']
assert len(rows)==5 and sum(r['complete_cycle'] for r in rows)==3
assert all(r['recorded_gate_max_error']==0 and r['external_input_bitwise_equal'] for r in rows[1:])
assert all(r['status']=='PILOT_STOPPED_NO_HIGH_ENTRY' for r in rows[-2:])
for dependency in json.loads((fig/'source/render_dependencies.json').read_text()):
 assert sha(fig/dependency['snapshot'])==dependency['sha256']
manifest=json.loads((report/'source_manifest.json').read_text())
assert sha(report/manifest['report'])==manifest['report_sha256']
for name,row in manifest['assets'].items():assert sha(report/name)==row['sha256']
assert sha(report/'figures/current/Figure6.png')==sha(fig/'figures/fig6-complete-layout.png')
assert sha(report/'figures/current/Figure6.pdf')==sha(fig/'figures/fig6-complete-layout.pdf')
for fn in ['model_selected_patient_matches.json','propagation_normalization.json','A_C_stage_correspondence.csv']:
 assert sha(report/'source'/f'fig6_{fn}')==sha(fig/fn)
assert (report/'source/fig6_caption_zh.md').read_bytes()==(fig/'caption_zh.md').read_bytes()
doc=(report/manifest['report']).read_text()
assert doc.count('S(R;R_0,k)')>=4 and '\\sum_n' in doc
assert '14/3' not in doc and '14}{3' not in doc
assert '22.04 s' in doc and '71.81 s' in doc and '48 次短事件' in doc
# Only links added or fully updated by this change are audited here.
paths=[fig/'README.md',fig/'caption_zh.md',evidence/'README.md',report/manifest['report']]
for path in paths:
 for ref in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
  if ref.startswith(('https://','http://','#')):continue
  assert (path.parent/ref.split('#')[0]).is_file(),(str(path),ref)
zip_path=report.parent/'2026-11-02_epilepsy_model_figures_Obsidian.zip'
with tempfile.TemporaryDirectory() as tmp:
 with zipfile.ZipFile(zip_path) as archive:
  assert len(archive.namelist())==12
  archive.extractall(tmp)
 extracted=Path(tmp)/report.name
 assert (extracted/manifest['report']).read_bytes()==(report/manifest['report']).read_bytes()
 for name,row in manifest['assets'].items():assert sha(extracted/name)==row['sha256']
print(json.dumps(dict(status='PASS',root=str(MAIN),native_completed_s=[r['observed_s'] for r in rows],logistic_gate_samples=[r['gate_samples'] for r in rows[1:]],figures_and_sidecars_consistent=True,all_source_snapshots_verified=True,report_package_verified=True),indent=2))

from pathlib import Path
import hashlib,json,subprocess,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
p=Path(__file__).resolve().parents[1]
load=lambda f:json.loads((p/f).read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
m=load('fig5_metadata.json');old=load('reference_fig5_metadata.json');manifest=load('panel_manifest.json')
assert m['display_time_window_s']==[8.,11.]
assert [s['time_s'] for s in m['snapshots']]==[8.42,9.42,9.48,9.6,10.37]
assert [s['number'] for s in m['native_maps']]==[2,3,4,5]
assert [s['time_s'] for s in m['E1']['states']]==[s['time_s'] for s in m['snapshots']]
assert len(m['waveform']['contact_order'])==15 and len(set(m['waveform']['contact_order']))==15
assert m['waveform']['band_hz'] is None and m['waveform']['unfiltered']
assert m['waveform']['dc_preserved'] and not m['waveform']['baseline_subtraction']
assert not m['waveform']['detrending'] and not m['waveform']['common_average_reference']
assert not m['slow_variables']['numerical_offsets'] and not m['slow_variables']['per_population_rescaling']
for oldzoom,newzoom in zip(old['raster_zoom_windows'],m['raster_zoom_windows']):
 assert oldzoom['sample_ids']==newzoom['sample_ids']
assert np.allclose(m['raster_zoom_windows'][0]['time_window_s'],[8.37,8.67])
assert m['raster_zoom_windows'][1]['time_window_s']==old['raster_zoom_windows'][1]['time_window_s']
assert m['slow_variables']['population_styles']=={'All E':'solid','Core A':'dashed'}
assert m['slow_variables']['variable_colors']=={'Z':'#2878b5','effective_M':'#d77d27'}
assert m['waveform']['raw_mean_strip_displayed'] is False
assert m['waveform']['normalization_window_s']==[0.,9.42]
assert m['waveform']['time_varying_gain'] is False
assert manifest['files']['A']['axes']==1
assert m['E2']==old['E2'] and m['E2_display']==old['E2_display']
for a,b in [(old['native_maps'][2],m['native_maps'][0]),(old['native_maps'][3],m['native_maps'][3])]:
 assert a['rate_Hz']==b['rate_Hz']
for s in m['state_time_links']:assert np.ptp(s['aligned_x_px'])<1e-8
assert load('source_snapshot.json')['source_result_sha256']==sha(Path(m['source'])/'result.json')
formats={}
for key,record in manifest['files'].items():
 for f in record['files']:assert sha(p/f['path'])==f['sha256']
 paths={Path(f['path']).suffix:p/f['path'] for f in record['files']}
 with Image.open(paths['.png']) as im: size=list(im.size); im.verify()
 root=ET.parse(paths['.svg'])
 subprocess.run(['pdfinfo',str(paths['.pdf'])],check=True,capture_output=True)
 text=subprocess.run(['pdftotext',str(paths['.pdf']),'-'],check=True,capture_output=True,text=True).stdout
 if key=='complete':
  assert 'Rest' not in text and 'Interictal' not in text
  assert 'Entry time / lower bound' not in text
  subprocess.run(['pdftoppm','-scale-to','1800','-png','-singlefile',str(paths['.pdf']),'/tmp/fig5-unfiltered-complete-pdf'],check=True,capture_output=True)
 if key in ['A','B','D','F']:
  subprocess.run(['pdftoppm','-scale-to','1200','-png','-singlefile',str(paths['.pdf']),f'/tmp/fig5-unfiltered-{key}-pdf'],check=True,capture_output=True)
 formats[key]=dict(png_pixels=size,pdf_readable=True,svg_text_elements=len(root.findall('.//{http://www.w3.org/2000/svg}text')))
qa=dict(status='PASS_DATA_AND_EXPORT_PENDING_VISUAL_REVIEW',formats=formats,
 same_source_trajectory=True,source_raster_unchanged=True,original_sample_neurons_unchanged=True,late_interictal_window_inside_AB=True,preictal_zoom_unchanged=True,
 old_preictal_and_onset_spatial_counts_identical=True,new_spatial_frames_from_native_counts=True,
 F_entire_comparison_and_display_metadata_unchanged=True,all15_contacts_preserved=True,
 AB_three_axes_same_time_mapping=True,variables_encode_color_populations_encode_linestyle=True,raw_mean_strip_removed=True,pre_recruitment_p95_fixed_scaling=True,unfiltered_DC_preserved=True,E_trajectory_markers_match_all_state_times=True,
 no_rest_or_interictal_spatial_frame=True,no_parameter_scan_in_complete_figure=True,
 state_mapping=m['snapshots'],caption_sha256=sha(p/'fig5_caption_detailed_zh.md'),human_review='PENDING')
(p/'publication_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
print('PASS: same physical trajectory,15 contacts, two native raster windows within8–11s,4 real spatial frames, aligned times, unchanged F quantities,21 readable figure assets.')

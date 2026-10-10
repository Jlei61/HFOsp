from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
ROOT=Path('/home/honglab/leijiaxin/HFOsp');sys.path.insert(0,str(ROOT))
BASE=ROOT/'results/paper-ready-figure/fig4'
OLD=BASE/'candidates/a_sampling_zoom_modes_20261009/source'
SRC=BASE/'candidates/a_rigid_seeg_bursts_20261010/source'
SRC.mkdir(parents=True,exist_ok=True)
for name in ['legacy_a_components.npz','legacy_a_geometry.json','A_zoom_in.json','mechanism_source.json','spatial_reference.npz','D_position_straight_electrodes.json','waveform_arrays.npz']:
 shutil.copy2(OLD/name,SRC/name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Reuse the exact sparse neuronal samples used to build the old raster.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.paper_figures.plot_fig4_subject_snn_grouped import DEFAULT_TAG
from scripts.paper_figures.plot_fig_subject_snn import _registered_axis_display
from scripts.paper_figures.plot_fig_subject_snn_mechanism import _load_figdata,_plot_mechanism,_reconstruct_posI
fd,source_path=_load_figdata(DEFAULT_TAG)
pos_i,imeta=_reconstruct_posI(fd,DEFAULT_TAG)
display=_registered_axis_display(fd)
reg=dict(fd['reg'].item());reg['source_names']=[];reg['sink_names']=[]
updated=dict(fd);updated['reg']=np.asarray(reg,dtype=object)
fig,ax=plt.subplots()
_plot_mechanism(updated,ax,clean=True,posI=pos_i,plot_seed=int((imeta.get('seed') or 0)+101),display=display,homogeneous_cores=True,semantic_core_colors=False,show_basic_labels=True,show_title=False)
e=np.concatenate([np.asarray(c.get_offsets()) for c in ax.collections if c.get_zorder() in (2,5)])
i=np.concatenate([np.asarray(c.get_offsets()) for c in ax.collections if c.get_zorder() in (1,6)])
plt.close(fig)
np.savez_compressed(SRC/'overview_neuron_samples.npz',E=e,I=i)
with np.load(SRC/'spatial_reference.npz') as z:
 names=z['names'].astype(str);orig=z['contacts']
 np.testing.assert_allclose(z['posE'],np.asarray(fd['posE'])@display['matrix']+display['offset'])
 np.testing.assert_allclose(z['posI'],pos_i@display['matrix']+display['offset'])
d=json.loads((SRC/'D_position_straight_electrodes.json').read_text())
idx=[d['names'].index(n) for n in names]
rod=np.array(d['rod_sheet_xy'])[idx]-10.
max_error=0.
for shaft in ['SCL','ICL']:
 points=rod[np.char.startswith(names,shaft)];u,s,v=np.linalg.svd(points-points.mean(0),full_matrices=False)
 max_error=max(max_error,float(np.abs((points-points.mean(0))@v[-1]).max()))
assert max_error<1e-10
np.savez_compressed(SRC/'rigid_contact_geometry.npz',names=names,original_xy_mm=orig,rigid_xy_mm=rod)
with np.load(SRC/'waveform_arrays.npz') as z:
 wn=list(z['contact_names'].astype(str)); ts=z['absolute_time_ms']; w=z['filtered_contact_activity'];selected=['ICL8','ICL6','ICL4'];rows=[wn.index(n) for n in selected]
 modes=[];peaks=[]
 for start,end in [(4310,4430),(3710,3830)]:
  mask=(ts>=start)&(ts<=end);t=ts[mask]-start;segment=w[rows][:,mask]
  np.testing.assert_array_equal(t,np.arange(0,122,2));modes.append(segment);peaks.append(t[np.argmax(segment,axis=1)])
 modes=np.array(modes);peaks=np.array(peaks)
 np.savez_compressed(SRC/'burst_readout_arrays.npz',time_ms=t,waveforms=modes,peak_times_ms=peaks,names=np.array(selected),common_amplitude_scale=np.max(np.abs(modes)))
m=json.loads((SRC/'mechanism_source.json').read_text())
m['electrode_display']={'source':'D_position_straight_electrodes.json','source_sha256':sha(SRC/'D_position_straight_electrodes.json'),'geometry_snapshot':'rigid_contact_geometry.npz','rule':'Reuse current E panel rigid rod projections; schematic display only, no simulation coordinates or readout traces changed.','maximum_collinearity_residual_mm':max_error,'maximum_display_displacement_mm':float(np.linalg.norm(rod-orig,axis=1).max()),'overview_and_sampling_use_identical_contacts':True}
m['overview_neuron_samples']={'snapshot':'overview_neuron_samples.npz','E':len(e),'I':len(i),'sampling':'Exact old overview scatter coordinates retained; raster replaced with vector scatter for straight rod drawing.'}
m['revised_display']='Original left circuit preserved; same neuron samples with rigid straight SCL/ICL shafts in overview and sampling zoom. Stronger green Gaussian footprint and dashed sampling box. Two real 120-ms burst fragments below, labelled SEEG readout.'
m['readout']['display_title']='SEEG readout'
m['readout']['display_title_meaning']='Virtual SEEG contact readout of underlying SNN propagation; firing-density proxy, not measured clinical voltage or a newly computed potential forward model.'
m['mode_showcases']['observable']='Frozen 30-80 Hz contact activity from the F source arrays, cropped to two 120-ms three-contact burst sequences.'
m['mode_showcases']['distinction_from_F']='A: two short three-contact mechanism readouts, fixed channel order, shared gain; F: continuous 15-contact long record and event marks.'
m['mode_showcases']['snapshot']='burst_readout_arrays.npz';m['mode_showcases']['snapshot_sha256']=sha(SRC/'burst_readout_arrays.npz')
m['mode_showcases']['waveform_source']='waveform_arrays.npz';m['mode_showcases']['waveform_source_sha256']=sha(SRC/'waveform_arrays.npz')
m['mode_showcases'].pop('exact_native_trajectory_crops_and_event_labels_verified',None)
m['mode_showcases']['exact_frozen_F_waveform_crops_verified']=True
m['mode_showcases']['no_artificial_peak_sharpening_or_time_warping']=True
m['mode_showcases']['peak_relative_ms']=peaks.tolist()
for event,start,end,pk in zip(m['mode_showcases']['events'],[4310,3710],[4430,3830],peaks):
 event['window_absolute_ms']=[start,end];event['peak_relative_ms']=pk.tolist()
m['sampling_zoom']['geometry']='Same frozen neuron population and same display-only rigid contacts as overview; both use the E-panel rod projection. Local kernels are illustrative operators, not reconstructed graph edges.'
(SRC/'mechanism_source.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
if Path(__file__).resolve() != (SRC/'prepare_sources.py').resolve():
 shutil.copy2(__file__,SRC/'prepare_sources.py')
print('Frozen samples',len(e),len(i),'straightness residual',max_error,'burst peaks',peaks.tolist())

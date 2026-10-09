from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np
from scipy.signal import butter,sosfiltfilt
root=Path.cwd();sys.path.insert(0,str(root))
from scripts.paper_figures import build_fig4_compact_ai as c
import matplotlib.pyplot as plt
out=root/'results/paper-ready-figure/fig4/candidates/a_circuit_sampling_sequence_20261009';src=out/'source';src.mkdir(parents=True,exist_ok=True)
c.plate.W,c.plate.H=c.W,c.H;c.plate.SOURCE=src;c.plate.GROUPS.clear();c.plate.CHECKS.clear()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
fig=plt.figure(figsize=(300/25.4,232/25.4),dpi=160);c.draw_a(fig);g=c.plate.GROUPS['A'];fig.canvas.draw();geometry={'axes':[],'texts':[],'connectors':[],'status':'Exact v4 first two components, including the left raster circuit and registered substrate'};arrays={}
for n,ax in enumerate(g['axes']):
 im=ax.images[0];arrays[f'image_{n}']=np.asarray(im.get_array());geometry['axes'].append(dict(position=ax.get_position().bounds,image_extent=im.get_extent(),xlim=ax.get_xlim(),ylim=ax.get_ylim(),aspect=ax.get_aspect()))
geometry['local_frame']=list(g['axes'][0].patches[0].get_bbox().bounds)
for t in g['texts']:geometry['texts'].append(dict(position=t.get_position(),text=t.get_text(),fontsize=t.get_fontsize(),ha=t.get_ha(),va=t.get_va(),weight=t.get_weight()))
for a in g['artists']:geometry['connectors'].append(dict(xy1=list(a.xy1),xy2=list(a.xy2)))
np.savez_compressed(src/'legacy_a_components.npz',**arrays);(src/'legacy_a_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n');plt.close(fig)
old=root/'results/paper-ready-figure/fig4/candidates/a_spatial_readout_20261009/source'
shutil.copy2(old/'mechanism_source.json',src/'mechanism_source.json')
p=Path('/data/hfosp/topic4_sef_hfo/xy_block_joint_observation_pilot_20260916/xy_block/units/xy_left_20/2511_847401')
d=json.loads((p/'workers/trajectory.json').read_text());physics=json.loads((p/'applied_physics.json').read_text());z=np.load(p/'workers/trajectory.npz');wave_names=[f'ICL{i}' for i in range(5,0,-1)];names=z['contact_names'].astype(str);ids=[list(names).index(n) for n in wave_names]
t=np.arange(z['contact_envelope'].shape[1])*float(z['contact_envelope_dt_ms']);traces=sosfiltfilt(butter(4,[30,80],btype='bandpass',fs=500,output='sos'),z['contact_envelope'],axis=1)
lo,hi=d['events'][10]['qualifying_interval_ms'];window=(t>=lo)&(t<=hi);peaks=t[window][traces[ids][:,window].argmax(axis=1)];assert np.all(np.diff(peaks)>0)
sel=(t>=4342)&(t<=4422);scale=float(np.load(root/'results/paper-ready-figure/fig4/source/waveform_arrays.npz')['common_amplitude_scale'])
np.savez_compressed(src/'sequence_arrays.npz',time_ms=t[sel]-4342,filtered_activity=traces[ids][:,sel],names=wave_names,peak_times_ms=peaks-4342,peak_absolute_ms=peaks,recruitment_absolute_ms=z['recruitment_ms'][10,ids],common_amplitude_scale=scale,contacts=z['contact_xy_mm'][ids],positions_E=z['positions_E'],positions_I=z['positions_I'])
meta=json.loads((src/'mechanism_source.json').read_text());meta['sequence']={'event':10,'same_event_as_F_MTA':True,'workpoint':'xy_left_20','topology_seed':2511,'dynamics_seed':847401,'names':wave_names,'selection':'Use the already displayed F-panel MTA event 10 and contiguous ICL5 through ICL1 contacts; no new event search or fit.','absolute_window_ms':[4342,4422],'peak_absolute_ms':peaks.tolist(),'recruitment_absolute_ms':z['recruitment_ms'][10,ids].tolist(),'no_artificial_time_shift':True,'no_per_contact_amplitude_scaling':True,'source_trajectory':str(p/'workers/trajectory.json'),'source_arrays_sha256':hashlib.sha256((p/'workers/trajectory.npz').read_bytes()).hexdigest()};meta['combined_physical_kernel']=physics['graph']['kernel'];meta['revised_display']='Original circuit and full-network components preserved. Third component combines the actual workpoint kernel, nearby neurons, contacts, and Gaussian readout. No formula or lower explanatory text.'
(src/'mechanism_source.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print('geometry:',json.dumps(geometry,indent=2));print('peaks',peaks)

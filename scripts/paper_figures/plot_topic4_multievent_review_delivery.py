"""Frozen-output visual delivery: continuous event movies and parameter/metric atlas."""
from pathlib import Path
import sys,json,csv,hashlib,argparse,pickle
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from src.topic4_interictal_repaired_evaluation import rank_features
R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT=R/'visual_review_delivery'; F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
BEST='v2_1_pop1_de_b_002'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
MAN={'source_role':'frozen_output_visual_review_only_no_simulation_or_ranking_change','figures':[],'movies':[]}
def read(p):return json.loads(Path(p).read_text())
def savefig(fig,name):
 fig.savefig(F/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig);MAN['figures'].append(name)
def val(x):return np.nan if x is None else float(x)
def avg(x):
 x=np.asarray(x,float);return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan
NAMES=['Core 1 x (mm)','Core 1 y (mm)','Core 2 x (mm)','Core 2 y (mm)','Threshold offset gain','E to E weight','E to I weight','I to E weight','GABA decay (ms)','EE axis offset (deg)','EE aspect ratio']
SHORT={'v2_anchor_historical__baseline':'A: widely spaced','v2_anchor_old_joint__baseline':'C: near-horizontal','v2_anchor_support_rank__baseline':'B: upper, closer','v2_anchor_support_rank__vth_low':'B: threshold gain 0.7','v2_anchor_old_joint__tau_d_GABA_ms_high':'C: GABA 24 ms','v2_1_pop0_de_a_001':'Joint-search model 1',BEST:'Selected joint-fit model'}
def metrics():
 train=read(R/'g2_all_64_scores.json')['candidates'];g3=read(R/'g3_scores.json')['candidates'];runs=read(R/'g3_confirmation_analysis.json')['runs']
 # All 11 dimensions: descriptive scatter, failed conditions in separate bottom rail.
 fields=[('loss_off','Joint L_off'),('loss_off_A_component','Mean-distance A'),('loss_off_B_subtraction','Finite-event B'),('support','Supported fraction'),('mix','M0 fraction'),('N','Primary events / run')]
 table=[]
 for c in train:
  units=list(c['units'].values()); row={'candidate_id':c['candidate_id'],'eligible':c['ranking_eligible'],'stage':'DE-B' if '_de_b_' in c['candidate_id'] else ('DE-A' if '_de_a_' in c['candidate_id'] else 'Initial')}
  row.update({f'p{i}':p for i,p in enumerate(c['parameters'])});row.update({k:val(c['score'].get(k)) for k,_ in fields[:3]})
  row['support']=avg([sum(u['mode_supported_counts'])/u['n_primary_events'] if u['n_primary_events'] else np.nan for u in units]);row['mix']=avg([u['mode_counts'][0]/u['n_primary_events'] if u['n_primary_events'] else np.nan for u in units]);row['N']=avg([u['n_primary_events'] for u in units]);row['runaway']=sum(u.get('physical_status')=='RUNAWAY' for u in units)
  table.append(row)
 for page,inds in [('spatial',range(4)),('physiology',range(4,11))]:
  fig,axes=plt.subplots(len(inds),6,figsize=(17,2.0*len(inds)),squeeze=False)
  for rr,i in enumerate(inds):
   for j,(k,title) in enumerate(fields):
    ax=axes[rr,j]
    for stage,color,marker in [('Initial','#8b95a5','o'),('DE-A','#267ea6','s'),('DE-B','#cf6535','^')]:
     sub=[r for r in table if r['stage']==stage and r['eligible']];ax.scatter([r[f'p{i}'] for r in sub],[r[k] for r in sub],s=21,color=color,marker=marker,alpha=.8)
    failed=[r for r in table if not r['eligible']];ax.scatter([r[f'p{i}'] for r in failed],[-.075]*len(failed),transform=ax.get_xaxis_transform(),marker='x',s=14,c='#c35757',clip_on=False)
    best=next(r for r in table if r['candidate_id']==BEST);ax.scatter(best[f'p{i}'],best[k],s=120,marker='*',c='#172332',zorder=5)
    if k=='mix':ax.axhline(6605/19770,color='black',ls='--',lw=.8);ax.set_ylim(0,1)
    if k=='support':ax.set_ylim(0,1)
    ax.set_xlabel(NAMES[i],fontsize=9);ax.tick_params(labelsize=8);ax.tick_params(axis="x",pad=13);ax.grid(alpha=.15)
    if rr==0:ax.set_title(title,fontsize=11)
  fig.suptitle('64-condition joint search: parameter associations ('+page+')',fontsize=16,y=1.005)
  fig.text(.5,-.016,'Gray: initial | blue: DE-A | orange: DE-B | star: confirmation-best candidate | red rail: unranked conditions\nOther parameters vary simultaneously. No connecting lines, causal effects or optimized-response curves are implied.',ha='center',fontsize=10)
  fig.tight_layout();savefig(fig,'search_parameter_metrics_'+page)
 with (OUT/'search_parameter_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 # G3 all requested independent endpoints and actual 4 paired units.
 keys=[('loss','L_off (lower)'),('participation','Participation MAE (lower)'),('order','Order TV, 2 ms (lower)'),('lag','Lag Wasserstein, ms (lower)'),('direction','Direction distance (lower)'),('coverage0','M0 coverage (higher)'),('coverage1','M1 coverage (higher)'),('support','Supported fraction (higher)'),('N','Primary events')]
 cids=[c['candidate_id'] for c in sorted(g3,key=lambda c:c['score']['loss_off'])]; rows=[]
 for c in g3:
  for unitname,u in c['units'].items():
   a=next(x for x in runs if x['candidate_id']==c['candidate_id'] and f"topo_{x['topology_seed']}_dyn_{x['dynamics_seed']}"==unitname);ms=a['mode_metrics'];row={'candidate_id':c['candidate_id'],'unit':unitname,'loss':u['score']['loss_off'],'N':a['n_primary'],'support':sum(a['supported_counts'])/a['n_primary']}
   for k,field in [('participation','participation_mae'),('order','order_TV_at_2ms'),('lag','signed_lag_wasserstein_ms'),('direction','direction_distance')]:row[k]=avg([val(m[field]) for m in ms])
   for mo in [0,1]:row['coverage'+str(mo)]=val(ms[mo]['patient_probe_neighborhood_coverage'])
   rows.append(row)
 fig,axs=plt.subplots(3,3,figsize=(17,12));paircolors=['#194e78','#579ccc','#9c4c24','#df9e69'];pairs=sorted({r['unit'] for r in rows})
 for ax,(k,title) in zip(axs.flat,keys):
  for pi,pair in enumerate(pairs):
   ys=[next(r[k] for r in rows if r['candidate_id']==cid and r['unit']==pair) for cid in cids];ax.scatter(np.arange(len(cids))+(pi-1.5)*.07,ys,s=28,color=paircolors[pi])
  means=[avg([r[k] for r in rows if r['candidate_id']==cid]) for cid in cids];ax.plot(range(len(cids)),means,'_',color='black',ms=15,mew=2)
  ax.set_title(title);ax.set_xticks(range(len(cids)),[SHORT[c] for c in cids],rotation=35,ha='right',fontsize=8);ax.grid(axis='y',alpha=.15)
 fig.suptitle('Confirmation: each parameter condition across 2 graphs x 2 noise replays',fontsize=16)
 fig.text(.5,.007,'Dots: individual runs (blue: topology 6101; brown: 6102). Black: mean. Mode-conditional errors average M0 and M1 equally.\nCoverage uses the frozen patient PROBE neighborhoods; these are reused development data. Four runs are not four patients.',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.05,1,.97));savefig(fig,'confirmation_metrics')
 with (OUT/'confirmation_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # Seven confirmed geometries plus actual parameter table.
 fig=plt.figure(figsize=(17,8));gs=fig.add_gridspec(2,7,height_ratios=[1,1.2]);ps=np.array([next(c['parameters'] for c in g3 if c['candidate_id']==cid) for cid in cids])
 for j,cid in enumerate(cids):
  ax=fig.add_subplot(gs[0,j]);xy=ps[j,:4].reshape(2,2);ax.scatter(xy[:,0],xy[:,1],c=['#286da6','#a24640'],s=70);ax.plot(xy[:,0],xy[:,1],color='.65',lw=1)
  theta=np.deg2rad(-22.80538396505847+ps[j,9]);center=xy.mean(0);dv=3*np.array([np.cos(theta),np.sin(theta)]);ax.plot([center[0]-dv[0],center[0]+dv[0]],[center[1]-dv[1],center[1]+dv[1]],'k--',lw=1)
  ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',title=SHORT[cid]);ax.tick_params(labelsize=8);ax.set_title(SHORT[cid],fontsize=9);ax.set_xlabel('x (mm)');
  if j==0:ax.set_ylabel('y (mm)')
 ax=fig.add_subplot(gs[1,:]);ax.axis('off');cell=[[f'{p:.2f}' for p in ps[:,i]] for i in range(4,11)];tab=ax.table(cellText=cell,rowLabels=NAMES[4:],colLabels=[SHORT[c] for c in cids],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.6)
 fig.suptitle('Actual confirmation parameters and two-core center locations',fontsize=16);fig.text(.5,.015,'Dashed axis: nominal EE weight redistribution axis; it is not an observed propagation direction. Every core position is shown on the same 20 mm sheet.',ha='center',fontsize=10);fig.tight_layout(rect=(.07,.05,1,.97));savefig(fig,'confirmation_parameters')

def old_oat():
 old=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2';records=read(old/'paired_parameter_comparison.json')['candidates'];by={r['candidate_id']:r for r in records}
 manifest=read(ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/execution/paired_round1/candidate_manifest.json')['candidates'];cm={r['candidate_id']:r for r in manifest}
 pars=[('vth','Threshold offset gain',['vth_low','baseline','vth_high'],lambda c:c['node_mapping']['node_gain']),('EE','E to E weight',['E_to_E_weight_scale_low','baseline','E_to_E_weight_scale_high'],lambda c:c['dynamic_parameters'].get('E_to_E_weight_scale',1.)),('EI','E to I weight',['E_to_I_weight_scale_low','baseline','E_to_I_weight_scale_high'],lambda c:c['dynamic_parameters'].get('E_to_I_weight_scale',1.)),('IE','I to E weight',['I_to_E_weight_scale_low','baseline','I_to_E_weight_scale_high'],lambda c:c['dynamic_parameters'].get('I_to_E_weight_scale',1.)),('tau','GABA decay (ms)',['tau_d_GABA_ms_low','baseline','tau_d_GABA_ms_high'],lambda c:c['dynamic_parameters'].get('tau_d_GABA_ms',18.)),('theta','EE axis offset (deg)',['EE_axis_minus20','baseline','EE_axis_plus20'],lambda c:c['mechanisms']['ellipse_angle_deg']-c['mechanisms']['ellipse_reference_angle_deg']),('AR','EE weight aspect ratio',['EE_weight_AR1','baseline'],lambda c:c['mechanisms']['ellipse_aspect_ratio'])]
 mets=[('joint_distance','Legacy joint distance (lower)',None),('unsupported_fraction','OOD fraction (lower)',None),('direction_distance','Direction distance (lower)',None),('coverage_by_mode','M0 coverage (higher)',0),('coverage_by_mode','M1 coverage (higher)',1),('n_events','Primary events',None)]
 raw=[]
 for page,subset in [(1,pars[:4]),(2,pars[4:])]:
  fig,axes=plt.subplots(len(subset),6,figsize=(17,2.35*len(subset)),squeeze=False)
  for i,(par,label,arms,fun) in enumerate(subset):
   for j,(metric,title,mo) in enumerate(mets):
    ax=axes[i,j]
    for anchor,color in [('historical','#6f7782'),('support_rank','#237baa'),('old_joint','#d2763c')]:
     xs=[fun(cm[anchor+'__'+arm]) for arm in arms];ys=[]
     for seed in [2511,2512,2513,2514]:
      sv=[]
      for arm,x in zip(arms,xs):
       rr=by[anchor+'__'+arm];pr=next((p for p in rr['per_seed'] if p['seed']==seed),None);v=np.nan if pr is None else pr['candidate'].get(metric);v=val(v[mo] if mo is not None and v is not None else v);sv.append(v);raw.append({'anchor':anchor,'parameter':par,'value':x,'seed':seed,'metric':title,'result':v})
      ax.plot(xs,sv,color=color,alpha=.17,lw=.7);ys.append(sv)
     med=np.nanmedian(ys,axis=0);ax.plot(xs,med,'o-',color=color,lw=1.8,ms=4,label={'historical':'A: widely spaced cores','support_rank':'B: upper, closer cores','old_joint':'C: near-horizontal cores'}[anchor])
    ax.set_xlabel(label,fontsize=9);ax.tick_params(labelsize=8);ax.tick_params(axis="x",pad=13);ax.grid(alpha=.15)
    if i==0:ax.set_title(title,fontsize=10)
    if 'fraction' in metric or mo is not None:ax.set_ylim(0,1)
  handles, labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.947),ncol=3,fontsize=10,frameon=False);fig.suptitle('Previous paired single-parameter screen: response curves',fontsize=16)
  fig.text(.5,.003,'Three fixed core geometries; thin lines: four legacy seeds; thick lines: medians. Previous 12 s trajectories and legacy evaluation scale.\nThese curves are not the new D_off objective. Missing values stay missing; event support is shown alongside fit and OOD.',ha='center',fontsize=10)
  fig.tight_layout(rect=(0,.055,1,.90));savefig(fig,f'paired_parameter_response_{page}')
 with (OUT/'paired_parameter_response.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)

def model(dyn,cid=BEST):
 stem=f'{cid}_topo_6101_dyn_{dyn}';g=R/'execution/confirmation_24s';record=read(g/'workers'/f'{stem}.json')
 with np.load(record['arrays']['path']) as z: data={k:np.array(z[k]) for k in ['contact_names','contact_xy_mm','contact_envelope','contact_envelope_dt_ms','sheet_activity_counts','sheet_activity_frame_ms']}
 with np.load(g/'repaired_observation'/f'{stem}.npz') as z:data.update({k:np.array(z[k]) for k in ['centroid_ms','primary_event_indices','windows_ms']})
 with (ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl').open('rb') as f:e=pickle.load(f)
 data['labels']=np.full(len(data['windows_ms']),-1,int);ix=data['primary_event_indices'];data['labels'][ix]=e.km.predict(rank_features(data['centroid_ms'][ix]));data['stem']=stem
 data['vmax']=max(float(np.quantile(data['sheet_activity_counts'],.9995)),1.);data['envmax']=max(float(np.quantile(data['contact_envelope'],.999)),1e-9)
 return data

def contact_axes(ax,xy,names,title):
 ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',title=title);ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20]);ax.tick_params(labelsize=8)
 for (x,y),name in zip(xy,names):ax.annotate(name,(x,y),xytext=(4,1),textcoords='offset points',fontsize=6,color='#56606b')
 return ax.scatter(xy[:,0],xy[:,1],s=62,c=np.zeros(len(xy)),cmap='magma',vmin=0,vmax=1,edgecolors='#a8abb0',linewidths=.6,zorder=3)

def writegif(frames,name,record,duration=60):
 p=F/(name+'.gif');frames[0].save(p,save_all=True,append_images=frames[1:],duration=duration,loop=0,optimize=False,disposal=2)
 with Image.open(p) as im:
  for i in range(im.n_frames):im.seek(i);im.load()
  record.update(path=str(p),frames=im.n_frames,decoded_all_frames=True,frame_display_ms=duration,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 MAN['movies'].append(record)
 for i in [0,len(frames)//3,len(frames)//2]:frames[i].convert('RGB').save(F/f'{name}_preview_{i:03d}.png')
 print('movie',name,len(frames),round(p.stat().st_size/1024**2,1),'MiB',flush=True)

def chrono(dyn):
 d=model(dyn);xy=d['contact_xy_mm'];names=d['contact_names'];selected=list(range(8));frames=[];fig,axes=plt.subplots(1,3,figsize=(12,4.5),dpi=95,gridspec_kw={'width_ratios':[1,1,1.15]});fig.subplots_adjust(top=.77,bottom=.24,wspace=.30)
 im=axes[0].imshow(d['sheet_activity_counts'][0],origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=d['vmax'],interpolation='nearest');axes[0].set(title='Native network activity',xticks=[0,10,20],yticks=[0,10,20]);sc=contact_axes(axes[1],xy,names,'Dynamic contact envelope')
 hm=axes[2].imshow(np.zeros((15,125)),aspect='auto',origin='lower',extent=(0,250,-.5,14.5),cmap='magma',vmin=0,vmax=d['envmax']);axes[2].set(title='All contact envelopes',xlabel='Time from window start (ms)',yticks=range(15),yticklabels=names);axes[2].tick_params(axis='y',labelsize=7);line=axes[2].axvline(0,c='#42c7c7',lw=1.3)
 head=fig.suptitle('',fontsize=14,y=.98);sub=fig.text(.5,.85,'',ha='center',fontsize=11)
 fig.text(.5,.075,'First 8 detected windows, chronological; excluded windows retained. t=0 is detection-window start, not ignition.\nNative counts and contact amplitudes have separate fixed scales. No spatial smoothing or time warping; movie sampled every 4 ms.',ha='center',fontsize=9)
 fig.colorbar(im,ax=axes[0],fraction=.04,pad=.03,label='spikes / 2 ms bin');fig.colorbar(hm,ax=axes[2],fraction=.04,pad=.02)
 for ii in selected:
  start,stop=d['windows_ms'][ii];t=np.arange(start,stop,4.);j0=int(start/2);j1=int(stop/2);hm.set_data(d['contact_envelope'][:,j0:j1]);hm.set_extent((0,stop-start,-.5,14.5));axes[2].set_xlim(0,stop-start)
  mode=d['labels'][ii];tag=f'M{mode} primary' if mode>=0 else 'excluded from training (overlap / qualification)';head.set_text(f'DE-B candidate | same graph 6101 | noise {dyn}\nEvent {ii+1}/8: {tag}')
  for tt in t:
   j=min(int(tt/float(d['sheet_activity_frame_ms'])),len(d['sheet_activity_counts'])-1);im.set_data(d['sheet_activity_counts'][j]);en=d['contact_envelope'][:,int(tt/float(d['contact_envelope_dt_ms']))];sc.set_array(en/d['envmax']);line.set_xdata([tt-start]*2);sub.set_text(f'Record time {tt/1000:.3f} s   |   window +{tt-start:.0f} ms');fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
 plt.close(fig);writegif(frames,f'continuous_first8_noise{dyn}',{'candidate':BEST,'topology':6101,'dynamics':dyn,'detected_event_indices':selected,'mode_labels':d['labels'][selected].tolist(),'selection':'first eight detected windows including excluded','window_alignment':'frozen detection-window start','sample_step_ms':4,'native_vmax':d['vmax'],'model_envelope_vmax':d['envmax']})

def comparison(cid=BEST):
 d=model(7101,cid);packet=read(R/'patient_time_packet/packet_manifest.json');patients={};models={};pmax=[]
 for mode in [1,0]:
  rows=sorted([r for r in packet['readable_events'] if r['mode']==mode],key=lambda r:r['selection_order'])[:4];patients[mode]=[];models[mode]=np.flatnonzero(d['labels']==mode)[:4].tolist();assert len(models[mode])==4
  for row in rows:
   with np.load(row['arrays_path']) as z:
    names=z['contact_names'].astype(str).tolist();ind=[names.index(n) for n in d['contact_names']];mask=z['packed_window_mask'];time=z['time_ms'][mask];env=np.maximum(z['envelope_background_robust_z'][ind][:,mask],0);patients[mode].append({'time':time-time[0],'env':env,'id':row['raw_global_event_index'],'source':row['arrays_path'],'state':row['status']});pmax.append(np.quantile(env,.99))
 cap=max(pmax);fig,axs=plt.subplots(2,3,figsize=(11,7.2),dpi=95);fig.subplots_adjust(top=.84,bottom=.12,hspace=.42,wspace=.22);arts=[]
 for ri,mo in enumerate([1,0]):
  ps=contact_axes(axs[ri,0],d['contact_xy_mm'],d['contact_names'],f'Patient M{mo}');ms=contact_axes(axs[ri,1],d['contact_xy_mm'],d['contact_names'],f'Model M{mo}');im=axs[ri,2].imshow(d['sheet_activity_counts'][0],origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=d['vmax'],interpolation='nearest');axs[ri,2].set(title=f'Model M{mo}: native field',xticks=[0,10,20],yticks=[0,10,20]);arts.append((ps,ms,im))
 head=fig.suptitle('',fontsize=14);fig.text(.5,.04,'Time zero: each frozen detection window start, not neuronal ignition. No event-wise time stretching or route matching.\nPatient / model contact brightness uses separate fixed scales. Contact layout is a projection; only the native model panels show a tissue field.',ha='center',fontsize=9)
 frames=[]
 for eventno in range(4):
  for rel in np.arange(0,250,4):
   for ri,mo in enumerate([1,0]):
    p=patients[mo][eventno];ii=models[mo][eventno];start,stop=d['windows_ms'][ii];tt=start+rel;ps,ms,im=arts[ri];j=int(np.argmin(abs(p['time']-rel)));pen=p['env'][:,j] if rel<=p['time'][-1] else np.full(15,np.nan);ps.set_array(pen/cap);ms.set_array(d['contact_envelope'][:,int(tt/2)]/d['envmax']);im.set_data(d['sheet_activity_counts'][int(tt/2)]);axs[ri,0].set_title(f'Patient M{mo} | raw {p["id"]}',fontsize=10);axs[ri,1].set_title(f'Model M{mo} | event {ii+1}',fontsize=10)
   head.set_text(f'{SHORT[cid]} vs patient | event pair {eventno+1}/4 | window +{rel:.0f} ms\nFirst four primary events per mode; no selection by similarity');fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
 plt.close(fig);name='patient_model_modes' if cid==BEST else 'patient_oldjoint_modes';writegif(frames,name,{'candidate':cid,'topology':6101,'dynamics':7101,'model_detected_indices':models,'patient_raw_ids':{mo:[p['id'] for p in patients[mo]] for mo in patients},'selection':'first four primary model events per mode; first four readable frozen patient selections per mode, no match-based filtering','alignment':'each frozen detection window start, 0..248 ms','sample_step_ms':4,'patient_cap':cap,'model_cap':d['envmax'],'native_cap':d['vmax']})

def geometry_key():
 manifest=read(ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/execution/paired_round1/candidate_manifest.json')['candidates'];by={c['candidate_id']:c for c in manifest}
 fig,axes=plt.subplots(1,3,figsize=(10,4.6));records=[]
 for ax,(anchor,color,title) in zip(axes,[('historical','#6f7782','A: widely spaced cores'),('support_rank','#237baa','B: upper, closer cores'),('old_joint','#d2763c','C: near-horizontal cores')]):
  c=by[anchor+'__baseline'];xy=np.asarray(c['node_field']['centers_mm']);ax.scatter(xy[:,0],xy[:,1],s=100,color=color);ax.plot(xy[:,0],xy[:,1],color=color,lw=1.5)
  for k,(x,y) in enumerate(xy):ax.annotate(f'{k+1}: ({x:.1f}, {y:.1f})',(x,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
  ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=title);ax.set_xticks([0,5,10,15,20]);ax.set_yticks([0,5,10,15,20]);ax.grid(alpha=.15);records.append({'figure_label':title,'legacy_id':anchor,'centers_mm':xy.tolist()})
 fig.suptitle('Three starting geometries in the single-parameter curves',fontsize=14)
 fig.text(.5,.04,'All three baselines: same threshold gain (1), EE / EI / IE gains (1), GABA decay (18 ms), and EE axis / aspect ratio.\nWithin each colored curve, only the parameter named on the x-axis changes; the core centers stay fixed.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.14,1,.92));savefig(fig,'three_core_geometries');(OUT/'curve_label_mapping.json').write_text(json.dumps(records,indent=2)+'\n')

def posthoc_errors():
 train=read(R/'g2_all_64_scores.json')['candidates'];cache=OUT/'posthoc_metric_cache.json'
 if cache.exists():rows=read(cache)
 else:
  with (ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl').open('rb') as f:e=pickle.load(f)
  rows=[]
  for c in train:
   row={'candidate_id':c['candidate_id'],'parameters':c['parameters'],'eligible':c['ranking_eligible']};units=[]
   if c['ranking_eligible']:
    for key,u in c['units'].items():
     op=Path(u['repaired_observation_path']).with_suffix('.npz')
     with np.load(op) as z:t=z['centroid_ms'][z['primary_event_indices']]
     m=e.metrics(t,detail=True);md=m['modes'];d={'OOD':val(m['unsupported_fraction'])}
     for k,fld in [('participation','participation_mae'),('order','order_TV_at_2ms'),('lag','signed_lag_wasserstein_ms'),('direction','direction_distance')]:
      vv=[val(mm.get(fld)) for mm in md];d[k]=float(np.mean(vv)) if len(vv)==2 and np.isfinite(vv).all() else np.nan
     units.append(d)
   for k in ['participation','order','lag','direction','OOD']:
    vv=[u[k] for u in units];row[k]=float(np.mean(vv)) if len(vv)==2 and np.isfinite(vv).all() else None
   rows.append(row)
  cache.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
 fields=[('participation','Participation MAE'),('order','Order TV, 2 ms'),('lag','Lag Wasserstein (ms)'),('direction','Direction distance'),('OOD','OOD fraction')]
 for page,inds in [('spatial',range(4)),('physiology',range(4,11))]:
  fig,axes=plt.subplots(len(inds),5,figsize=(16,2.1*len(inds)),squeeze=False)
  for rr,i in enumerate(inds):
   for j,(k,title) in enumerate(fields):
    ax=axes[rr,j]
    for row in rows:
     x=row['parameters'][i]
     if row[k] is None:ax.scatter(x,-.065,transform=ax.get_xaxis_transform(),marker='x',c='#bc5353',s=15,clip_on=False);continue
     cid=row['candidate_id'];color='#cf6535' if '_de_b_' in cid else ('#267ea6' if '_de_a_' in cid else '#8b95a5');ax.scatter(x,row[k],s=100 if cid==BEST else 22,marker='*' if cid==BEST else 'o',color='black' if cid==BEST else color,zorder=5 if cid==BEST else 2)
    ax.set_xlabel(NAMES[i],fontsize=9);ax.grid(alpha=.15);ax.tick_params(labelsize=8);ax.tick_params(axis="x",pad=13)
    if rr==0:ax.set_title(title+' (lower)',fontsize=10)
    if k=='OOD':ax.set_ylim(0,1)
  fig.suptitle('Post-fit diagnostic metrics across the 64-condition search ('+page+')',fontsize=15,y=1.004)
  fig.text(.5,-.017,'Gray: initial; blue: DE-A; orange: DE-B; star: DE-B selected in confirmation. Red rail: unranked / not estimable.\nEach dot averages two complete training runs, with M0/M1 equally weighted. This is a post-fit association; it did not drive nomination.',ha='center',fontsize=10)
  fig.tight_layout();savefig(fig,'search_diagnostic_metrics_'+page)
 with (OUT/'search_diagnostic_metrics.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['candidate_id','eligible']+[f'p{i}' for i in range(11)]+[k for k,_ in fields]);writer.writeheader()
  for row in rows:writer.writerow({**{k:row[k] for k in ['candidate_id','eligible']+[k for k,_ in fields]},**{f'p{i}':x for i,x in enumerate(row['parameters'])}})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['plots','movies','diagnostics','all'],default='all');a=ap.parse_args()
 if a.part in ['plots','all']:metrics();old_oat();geometry_key()
 if a.part in ['diagnostics','all']:posthoc_errors()
 if a.part in ['movies','all']:chrono(7101);chrono(7102);comparison();comparison('v2_anchor_old_joint__baseline')
 (OUT/f'manifest_{a.part}.json').write_text(json.dumps(MAN,indent=2,allow_nan=False)+'\n')
 print('DONE',a.part,flush=True)

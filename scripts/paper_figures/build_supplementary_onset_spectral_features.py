#!/usr/bin/env python3
"""S6: early-seizure spectral phenotypes and phenotype-matched energy features.

Reuses the saved cohort classification and the native Fig3 raw/TFR/band-power
readout. The original E20 example has zero-based index 7, hence public SZ8.
"""
from pathlib import Path
import sys,json,hashlib
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, ScaledTranslation
from scripts.paper_figures import plot_fig3_raw_spectral_context as native

OUT=ROOT/'results/paper-ready-figure/supplementary_onset_spectral_features'
STATE=native.PHENOTYPE_STATE_CSV
# Published packages replay the same classification from their frozen copy.
STATE_INPUT=STATE if STATE.is_file() else OUT/'data/per_seizure_spectral_overlap_state.csv'
SUMMARY=STATE_INPUT.parent/'cohort_spectral_simple_summary.csv'
BASELINE=(-120.,-90.);DISPLAY=((-110.,-90.),(-10.,20.));XWINDOW=(-120.,20.)
COLORS={'Broadband':'#596DA5','Gamma-dominant':'#269783'}
DISPLAY_LABELS={'Gamma-dominant':'Gamma'}
CASES=[('Y15','yuquan_litengsheng',2,'D3','Broadband'),('E20','epilepsiae_635',7,'HRB1','Gamma-dominant')]
PHENOTYPE_ORDER=['broadband_1_150','gamma_nonbroadband','low_frequency_only','other']
PHENOTYPE_LABELS=['Broadband','Gamma-dominant','Low-frequency only','Other']
PHENOTYPE_COLORS=[COLORS['Broadband'],COLORS['Gamma-dominant'],'#A5A5A5','#D4D4D4']
PIE_SUBJECT='epilepsiae_635'
EPISODES=ROOT/'results/topic5_extended_type_association/yuquan_episode_review_20261017/episode_map.csv'

def aligned_window(subject,idx):
 # The legacy Yuquan inventory stores SZ markers under eeg_onset_epoch.
 # Use the independently reviewed, explicit clinical annotation for this case.
 sw=native.extract_seizure_window(subject.replace('_','/',1),idx,pre_sec=140.,post_sec=40.,results_root=ROOT/'results',reference='car')
 anchor={'time_origin':'clinical onset'}
 if subject.startswith('yuquan_'):
  episodes=pd.read_csv(EPISODES);episode=episodes[(episodes.subject==subject)&(episodes.sz==idx+1)].iloc[0]
  assert episode.anchor_kind=='clinical' and np.isfinite(episode.clinical_onset_epoch)
  delta=float(episode.clinical_onset_epoch)-sw.clin_onset_epoch
  anchor.update({'clinical_anchor_source':str(EPISODES.relative_to(ROOT)),'original_marker_minus_clinical_sec':-delta,'original_annotation':episode.clinical_annotation,'episode_id':episode.episode_id})
  sw=replace(sw,clin_onset_epoch=float(episode.clinical_onset_epoch),t_axis=sw.t_axis-delta)
 return sw,anchor

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def extract():
 (OUT/'data').mkdir(parents=True,exist_ok=True)
 states=pd.read_csv(STATE_INPUT);summ=pd.read_csv(SUMMARY);combined=summ[summ.dataset=='combined_descriptive']
 assert len(states)==324 and states.subject.nunique()==23
 for _,r in combined.iterrows():assert int((states.simple_phenotype==r.phenotype).sum())==int(r.n_seizures)
 cases=[]
 for public,subject,idx,contact,kind in CASES:
  path=OUT/'data'/f'{public}_SZ{idx+1}_spectral_readout.npz';meta=path.with_suffix('.json')
  if path.exists() and meta.exists() and json.loads(meta.read_text()).get('raw_display_sampling')=='native':cases.append((np.load(path),json.loads(meta.read_text())));continue
  state=native._phenotype_row(subject,idx)
  assert state['simple_phenotype']==('broadband_1_150' if kind=='Broadband' else 'gamma_nonbroadband')
  sw,anchor=aligned_window(subject,idx)
  selected,source,_=native._select_channels(sw,subject,idx,15,None,channel_source='lagpat')
  aliases=native._alias_index(sw.ch_names);spectral_idx=aliases[contact]
  freqs,t,db,psd,base=native._channel_tfr(sw,spectral_idx,XWINDOW,BASELINE)
  curves=native._band_enhancement(freqs,t,psd,BASELINE)
  # Six evenly spaced source contacts, including the fixed TFR contact.
  shown=[selected[i] for i in np.linspace(0,len(selected)-1,6).round().astype(int)]
  if spectral_idx not in shown:shown[-1]=spectral_idx
  assert len(set(shown))==6
  mask=(sw.t_axis>=-120)&(sw.t_axis<=20);indices=np.flatnonzero(mask)
  np.savez_compressed(path,raw_times=sw.t_axis[indices],raw_signal=sw.signal[np.asarray(shown)][:,indices],raw_names=np.array([sw.ch_names[i] for i in shown]),tfr_times=t,tfr_freqs=freqs,tfr_db=db,band_names=np.array(list(curves)),band_curves=np.stack(list(curves.values())),band_curves_smoothed=np.stack([native._smooth_curve(v,t,smooth_sec=2) for v in curves.values()]))
  m={'public_label':public,'public_seizure_label':f'SZ{idx+1}','subject':subject,'zero_based_seizure_idx':idx,'seizure_id':sw.seizure_id,'spectral_channel':contact,'kind':kind,'reference':'car','fs_hz':sw.fs,'selected_source_contacts':[sw.ch_names[i] for i in selected],'shown_raw_contacts':[sw.ch_names[i] for i in shown],'channel_source':source,'clinical_onset_epoch':sw.clin_onset_epoch,'eeg_onset_relative_sec':native._eeg_rel_sec(sw),'baseline_sec':list(BASELINE),'early_window_sec':[0,10],'phenotype':state['simple_phenotype'],'phenotype_band_hits':native._phenotype_detection_counts(state),'mean_band_gain_0_10_db':native._window_mean_by_band(curves,t,(0,10)),'cache_sha256':sha(path)}
  if public=='E20':
   old=json.loads((ROOT/'results/paper-ready-figure/supp_fig5_early_seizure_phenotypes/figures/epilepsiae_635_seizure_07_raw_spectral_context_summary.json').read_text())
   assert old['seizure_id']==sw.seizure_id
   diffs={k:m['mean_band_gain_0_10_db'][k]-old['band_enhancement_mean_db']['clinical_0_10'][k] for k in curves}
   m['old_gamma_readout_differences_db']=diffs
   assert max(abs(v) for v in diffs.values())<1e-9
   m['label_correction']='Previous caption used index 7 as SZ7; the original record 63500800102 is public SZ8. Same seizure and channel retained.'
  m.update(anchor);m['raw_display_sampling']='native'
  write(meta,m);cases.append((np.load(path),m));print(public,sw.seizure_id,m['mean_band_gain_0_10_db'],flush=True)
 return cases,combined

def shade(ax,compressed=False):
 if compressed:
  native._shade_compact_interval(ax,BASELINE,DISPLAY,color='#4C78A8',alpha=.14)
  native._shade_compact_interval(ax,(0,10),DISPLAY,color='#B2182B',alpha=.12)
  x0=native._compressed_position(0,DISPLAY)
  ax.axvline(x0,color='black',ls='--',lw=1)
 else:
  ax.axvspan(*BASELINE,color='#4C78A8',alpha=.14);ax.axvspan(0,10,color='#B2182B',alpha=.12);ax.axvline(0,color='black',ls='--',lw=1)

def compact_axis(ax):
 native._configure_compressed_time_axis(ax,DISPLAY)
 ax.tick_params(labelsize=18,length=3)

def within_patient_distribution():
 states=pd.read_csv(STATE_INPUT);rows=states.loc[states.subject==PIE_SUBJECT].copy()
 assert len(rows)>0 and not rows.duplicated(['subject','seizure_idx']).any()
 assert rows.simple_phenotype.isin(PHENOTYPE_ORDER).all()
 counts=rows.simple_phenotype.value_counts().reindex(PHENOTYPE_ORDER,fill_value=0)
 assert int(counts.sum())==len(rows) and (counts.iloc[:2]>0).all()
 public=native.public_patient_label(*PIE_SUBJECT.split('_',1))
 rows.insert(0,'public_patient_label',public)
 rows.insert(1,'public_seizure_label',rows.seizure_idx.map(lambda i:f'SZ{int(i)+1}'))
 rows[['public_patient_label','public_seizure_label','analysis_version','simple_phenotype']].to_csv(OUT/'data/panel_b_patient_seizures.csv',index=False)
 summary=pd.DataFrame({'public_patient_label':public,'phenotype':PHENOTYPE_ORDER,'label':PHENOTYPE_LABELS,'n_seizures':counts.to_numpy(),'fraction':counts.to_numpy()/len(rows)})
 summary.to_csv(OUT/'data/panel_b_patient_distribution.csv',index=False)
 return {'public_label':public,'n_seizures':len(rows),'counts':{k:int(counts[k]) for k in PHENOTYPE_ORDER},'denominator':'All seizures for this patient in the frozen classification table; no phenotype exclusions.','statistical_unit':'Seizure within one patient','selection_reason':'The existing gamma example patient in panel D also has broadband seizures; both are the joint most frequent categories.','broadband_and_gamma_fraction':float(counts.iloc[:2].sum()/len(rows))}

def draw_patient_pie(fig,bounds,patient):
 """Draw the same full-denominator patient distribution in both exports."""
 x,y,w,h=bounds
 ax=fig.add_axes([x,y,w*.40,h])
 nums=np.array([patient['counts'][k] for k in PHENOTYPE_ORDER]);keep=nums>0
 wedges,_,texts=ax.pie(nums[keep],colors=np.array(PHENOTYPE_COLORS)[keep],startangle=90,counterclock=False,radius=1.10,autopct=lambda pct:f'{pct:.1f}%',pctdistance=.56,textprops={'fontsize':14},wedgeprops={'edgecolor':'white','linewidth':1.6})
 for text,color in zip(texts,np.array(PHENOTYPE_COLORS)[keep]):text.set_color('#333333' if color in ('#A5A5A5','#D4D4D4') else 'white')
 ax.set_aspect('equal')
 legend=fig.add_axes([x+w*.45,y,w*.55,h]);legend.axis('off')
 for y0,label,n,color in zip([.80,.50,.20],np.array(PHENOTYPE_LABELS)[keep],nums[keep],np.array(PHENOTYPE_COLORS)[keep]):
  legend.add_patch(Rectangle((0,y0-.04),.065,.08,facecolor=color,edgecolor='none',transform=legend.transAxes))
  legend.text(.10,y0+.035,DISPLAY_LABELS.get(label,label),fontsize=18,va='center',transform=legend.transAxes)
  legend.text(.10,y0-.09,f'n = {int(n)}',fontsize=14,color='#666666',va='center',transform=legend.transAxes)
 # The complete percentage text box, including a small margin, must fit
 # inside its own sector in both the composite and standalone export.
 fig.canvas.draw();renderer=fig.canvas.get_renderer();checks=[]
 for wedge,text in zip(wedges,texts):
  box=text.get_window_extent(renderer).padded(2)
  xx,yy=np.meshgrid(np.linspace(box.x0,box.x1,5),np.linspace(box.y0,box.y1,5))
  inside=wedge.get_path().transformed(wedge.get_transform()).contains_points(np.column_stack([xx.ravel(),yy.ravel()])).all()
  assert inside,f'Pie percentage exceeds its sector: {text.get_text()}'
  checks.append({'text':text.get_text(),'font_pt':text.get_fontsize(),'text_box_inside_sector':bool(inside)})
 return checks

def main():
 cases,counts=extract();(OUT/'figures').mkdir(exist_ok=True);patient=within_patient_distribution()
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':20,'axes.labelsize':22,'xtick.labelsize':18,'ytick.labelsize':18,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig=plt.figure(figsize=(18,12.7),facecolor='white')
 # A. Directly labelled cohort distribution.
 ax=fig.add_axes([.20,.81,.25,.14]);order=['broadband_1_150','gamma_nonbroadband','low_frequency_only','other'];labels=['','','Low bands','Other'];cc=[COLORS['Broadband'],COLORS['Gamma-dominant'],'#A5A5A5','#D4D4D4']
 data=counts.set_index('phenotype');nums=[int(data.loc[k,'n_seizures']) for k in order];fractions=np.array(nums)/324
 ax.barh(np.arange(4),fractions*100,color=cc,height=.58);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,50);ax.set_xticks([0,20,40]);ax.set_xlabel('Seizures (%)',fontsize=20,labelpad=2)
 ax.tick_params(axis='y',length=0,pad=8,labelsize=18)
 for row,kind,band in [(0,'Broadband','1–150 Hz'),(1,'Gamma','30–80 Hz')]:
  for label,offset,size,color in [(kind,6,18,'#222222'),(band,-10,14,'#666666')]:
   trans=ax.get_yaxis_transform()+ScaledTranslation(-8/72,offset/72,fig.dpi_scale_trans)
   ax.text(0,row,label,transform=trans,ha='right',va='center',fontsize=size,color=color)
 for i,(num,frac) in enumerate(zip(nums,fractions)):ax.text(frac*100+.8,i,f'{num} ({frac*100:.1f}%)',va='center',fontsize=18)
 fig.text(.035,.965,'A',weight='bold',fontsize=28);fig.text(.20,.965,'324 seizures  |  23 patients',fontsize=22,weight='bold')
 # B. All classified seizures from the same patient shown in panel D.
 fig.text(.50,.965,'B',weight='bold',fontsize=28)
 fig.text(.55,.965,f"{patient['public_label']}  |  {patient['n_seizures']} seizures",fontsize=20,weight='bold')
 composite_pie_checks=draw_patient_pie(fig,(.54,.79,.42,.17),patient)
 for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'supp_fig6_panels_ab.{ext}',dpi=220,bbox_inches='tight',pad_inches=.10)
 # C/D. Aligned raw signal and TFR examples.
 tfr_images=[];tfr_axes=[];tfr_colorbars=[]
 for i,(arrays,m) in enumerate(cases):
  x=.075+i*.475;w=.370 if i==0 else .385;kind=m['kind'];color=COLORS[kind]
  fig.text(x-.045,.735,chr(67+i),weight='bold',fontsize=28);fig.text(x,.735,f"{m['public_label']} | {m['public_seizure_label']}   {DISPLAY_LABELS.get(kind,kind)}",color=color,fontsize=24,weight='bold')
  raw=fig.add_axes([x,.55,w,.15]);t=arrays['raw_times'];v=arrays['raw_signal'];v=v-np.median(v,axis=1,keepdims=True);scale=max(np.finfo(float).eps if m['subject'].startswith('yuquan_') else 40,float(np.nanpercentile(abs(v),95))*3)
  if m['subject'].startswith('yuquan_'):
   visible=np.logical_or.reduce([(t>=lo)&(t<=hi) for lo,hi in DISPLAY])
   scale=max(np.finfo(float).eps,2.2*float(np.nanmax(abs(v[:,visible]))))
  for j,row in enumerate(v):
   for mask,mapped in native._compressed_segments(t,DISPLAY):raw.plot(mapped,row[mask]/scale+j,color='#252525',lw=.45,rasterized=True)
  compact_axis(raw);shade(raw,True);raw.set_ylim(-.65,5.65);raw.invert_yaxis();raw.set_yticks(range(6),[str(n) for n in arrays['raw_names']],fontsize=16);raw.tick_params(axis='y',length=0);raw.set_xticklabels([])
  shown_width=native._compressed_position(DISPLAY[-1][-1],DISPLAY)
  raw.text(native._compressed_position(-100,DISPLAY)/shown_width,1.02,'Baseline',transform=raw.transAxes,color='#3C648C',fontsize=18,ha='center');raw.text(native._compressed_position(5,DISPLAY)/shown_width,1.02,'Onset',transform=raw.transAxes,color='#B2182B',fontsize=18,ha='center')
  tfr=fig.add_axes([x,.365,w,.125]);tfr_axes.append(tfr)
  for mask,mapped in native._compressed_segments(arrays['tfr_times'],DISPLAY):im=tfr.pcolormesh(mapped,arrays['tfr_freqs'],arrays['tfr_db'][:,mask],cmap='RdBu_r',vmin=-14,vmax=14,shading='auto',rasterized=True)
  compact_axis(tfr);tfr.set_ylim(1,150);tfr.set_yticks([1,50,100,150]);tfr.set_ylabel('Frequency (Hz)',fontsize=20);tfr.set_xlabel('Time from clinical onset (s)',fontsize=20,labelpad=3);tfr.set_title(m['spectral_channel'],fontsize=19,loc='left',pad=4)
  tfr.axvline(native._compressed_position(0,DISPLAY),color='black',ls='--',lw=1);tfr_images.append(im)
 for im,cbar_x in zip(tfr_images,[.451,.951]):
  cax=fig.add_axes([cbar_x,.365,.009,.125]);cb=fig.colorbar(im,cax=cax,ticks=[-14,0,14]);cb.ax.set_title('TFR (dB)',fontsize=16,pad=12);cb.ax.tick_params(labelsize=17,length=3);tfr_colorbars.append(cb)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 right_frequency_label=tfr_axes[1].yaxis.label.get_window_extent(renderer).padded(2)
 assert all(not text.get_window_extent(renderer).overlaps(right_frequency_label) for text in [*tfr_colorbars[0].ax.get_yticklabels(),tfr_colorbars[0].ax.title])
 assert all(cb.mappable.get_clim()==(-14,14) for cb in tfr_colorbars)
 tfr_bounds=Bbox.union([ax.get_tightbbox(renderer) for ax in [*tfr_axes,*[cb.ax for cb in tfr_colorbars]]]).transformed(fig.dpi_scale_trans.inverted()).padded(.10)
 for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'supp_fig6_tfr_panels.{ext}',dpi=220,bbox_inches=tfr_bounds)
 # E. The same two channels on continuous time axes in four frequency bands.
 fig.text(.03,.285,'E',weight='bold',fontsize=28)
 power_axes=[];power_legends=[];power_handles=[]
 for j,(band,lo,hi) in enumerate(native.DISPLAY_BANDS):
  ax=fig.add_axes([.075+j*.232,.065,.188,.16]);power_axes.append(ax);shade(ax);allv=[];handles=[]
  for a,m in cases:
   k=a['band_names'].tolist().index(band);v=a['band_curves_smoothed'][k];line,=ax.plot(a['tfr_times'],v,color=COLORS[m['kind']],lw=1.6,label=f"{m['public_label']}/{m['public_seizure_label']}  {DISPLAY_LABELS.get(m['kind'],m['kind'])}");allv.append(v)
   handles.append(line)
  v=np.concatenate(allv);assert np.nanmin(v)>-10 and np.nanmax(v)<30
  ax.set(xlim=XWINDOW,ylim=(-10,30),xticks=[-120,-60,0,20],yticks=[-10,0,10,20,30]);ax.axhline(0,color='.4',lw=.65);ax.set_title(f"{band.capitalize()}\n{lo:g}–{hi:g} Hz",fontsize=20,pad=8);ax.set_xlabel('Time (s)',fontsize=20);ax.set_ylabel('Power gain (dB)' if j==0 else '',fontsize=20)
  power_legends.append(ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.02,1),ncol=1,frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=12,handlelength=1.0,handletextpad=.35,labelspacing=.15,borderpad=.08,borderaxespad=.15))
  power_handles.append(handles)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for ax,legend,handles in zip(power_axes,power_legends,power_handles):
  legend_box=legend.get_window_extent(renderer);axis_box=ax.get_window_extent(renderer)
  assert axis_box.contains(legend_box.x0,legend_box.y0) and axis_box.contains(legend_box.x1,legend_box.y1)
  assert not any(line.get_path().transformed(line.get_transform()).intersects_bbox(legend_box.padded(3),filled=False) for line in handles)
  assert legend_box.x1+3 < ax.transData.transform((0,0))[0]
 power_bounds=Bbox.union([ax.get_tightbbox(renderer) for ax in power_axes]).transformed(fig.dpi_scale_trans.inverted()).padded(.10)
 for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'supp_fig6_panele_power_curves.{ext}',dpi=220,bbox_inches=power_bounds)
 for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'supp_fig6_onset_spectral_features.{ext}',dpi=220,bbox_inches='tight',pad_inches=.10)
 plt.close(fig)
 panel=plt.figure(figsize=(8,3),facecolor='white')
 panel.text(.04,.88,f"{patient['public_label']}  |  {patient['n_seizures']} seizures",fontsize=20,weight='bold')
 standalone_pie_checks=draw_patient_pie(panel,(.02,.04,.96,.76),patient)
 for ext in ('png','pdf'):panel.savefig(OUT/'figures'/f'supp_fig6_panelb_patient_distribution.{ext}',dpi=220,bbox_inches='tight',pad_inches=.10)
 plt.close(panel)
 metadata={'asset_id':'early_seizure_spectral_features','paper_slot':'FigS6','status':'PENDING_AUTHOR_VISUAL_REVIEW','producer':str(Path(__file__).relative_to(ROOT)),'cohort':{'state_source':str(STATE.relative_to(ROOT)),'source_sha256':sha(STATE_INPUT),'n_seizures':324,'n_patients':23,'counts':dict(zip(order,nums))},'examples':[m for a,m in cases],'panels':{'A':'Cohort spectral phenotype distribution','B':'Within-patient spectral phenotype distribution across all 17 classified seizures of E20','C':'Yuquan Y15/SZ3 broadband raw and D3 TFR','D':'E20/SZ8 gamma-dominant raw and HRB1 TFR','E':'Four frequency-band power trajectories from the same representative contacts'},'display':{'raw_TFR_windows':DISPLAY,'band_curve_window':XWINDOW,'baseline_sec':BASELINE,'early_window_sec':[0,10],'tfr_limits_db':[-14,14],'spectral_window_sec':1,'spectral_step_sec':.1,'band_curve_smoothing_sec':2},'feature_contract':'Broadband uses 1-150 Hz, non-broadband gamma uses 30-80 Hz. Downstream fields use contact-wise baseline-standardized log-band-power averaged over 0-10 s; displayed example traces are dB gains, not the field z-scores.','classification_contract':'At least 5 of 6 band hits, at least 2 low and 2 fast, defines broadband; otherwise gamma hit defines gamma-dominant. Exact frozen detection rule and source retained in methods note.','claim_scope':'Descriptive early spectra and frequency-band choice; no onset-detection performance estimate.'}
 metadata['display']['spectral_step_sec']=float(np.median(np.diff(cases[0][0]['tfr_times'])))
 metadata['layout_version']='dual_tfr_colorbars_20261103'
 metadata['tfr_colorbars']={'panels':['C','D'],'count':2,'label':'TFR (dB)','limits_db':[-14,14],'ticks_db':[-14,0,14],'colormap':'RdBu_r','one_colorbar_per_patient':True,'left_colorbar_clear_of_right_frequency_label':True}
 metadata['panel_E_legend']={'axes':'all four frequency-band subplots','count':len(power_legends),'location':'upper left inside each axes','columns':1,'font_pt':12,'bbox_anchor_axes':[.02,1],'background':'compact borderless white','inside_axes':True,'no_intersection_with_data_curves_with_3px_margin':True,'onset_and_early_window_unobscured':True}
 metadata['display_label_mapping']=DISPLAY_LABELS
 metadata['panels']['A']='Cohort spectral phenotype distribution with broadband 1-150 Hz and gamma 30-80 Hz readout bands'
 metadata['panel_A_band_labels_hz']={'Broadband':[1,150],'Gamma':[30,80]}
 metadata['panel_A_display_label_mapping']={'low_frequency_only':'Low bands'}
 metadata['pie_typography']={'title_pt':20,'legend_pt':18,'percentage_pt':14,'composite_checks':composite_pie_checks,'standalone_checks':standalone_pie_checks}
 metadata['label_display']={'A':'Type on first line; readout band on smaller grey second line','B':'Type on first line; seizure count on smaller grey second line','secondary_text_pt':14}
 metadata['within_patient_distribution']=patient
 metadata['claim_scope']='Descriptive early spectra, within-patient coexistence and predominance of broadband/gamma types in E20, and frequency-band choice; no cohort-wide within-patient prevalence or onset-detection performance estimate.'
 metadata['display']['spectral_step_sec_by_case']={m['public_label']:float(np.median(np.diff(a['tfr_times']))) for a,m in cases}
 metadata['display']['band_curve_limits_db']=[-10,30]
 metadata['display']['raw_sampling_hz']={m['public_label']:m['fs_hz'] for a,m in cases}
 metadata['display']['raw_amplitude_scaling']='Within-case scaling; native raw units retained, no between-patient amplitude comparison.'
 write(OUT/'metadata.json',metadata)
 description='A 展示 23 名患者、324 次发作的早期频谱类型，并在宽带型与 gamma 优势型标签中注明 1–150 Hz 和 30–80 Hz 能量读出频带。B 展示 E20 全部 17 次已分类发作的患者内分布：宽带型和 gamma 优势型各 6 次，其他 5 次。C、D 为玉泉 Y15/SZ3（D3）和 Epilepsiae E20/SZ8（HRB1）的脑电及时频谱，E 比较同一示例触点的四频带功率轨迹；两例均以临床起始为零点。**关注点**：E20 内宽带型和 gamma 优势型均存在，且是占比最高的两类。'
 panel_description='E20 的患者内类型占比，分母为原分类表中该患者全部 17 次发作。宽带型与 gamma 优势型各占 35.3%，其他占 29.4%；仅低频型为 0 次，不绘制扇区。**关注点**：两种主要类型合计占 70.6%，未剔除其他类型后重新归一化。'
 ab_description='完整图 A/B 的同状态局部预览：A 将类型和频段分为两行；B 将类型和发作数分为两行。频段及发作数用较小灰字呈现，百分比保留在对应扇区内。**关注点**：缩短横向标签，保留频段、数量及百分比。'
 e_description='Y15/SZ3 与 E20/SZ8 示例触点的四频带功率变化，复用完整图中的曲线和坐标。每幅子图左上角均竖排两例的图例，使用 12 号字和紧凑白底，类型简称为 Broadband 和 Gamma。**关注点**：各子图可独立读取图例，且不遮挡曲线、起始线及 0–10 秒区域。'
 tfr_description='C、D 的代表触点时频谱局部图，与完整图保持同一状态。两位患者各有独立 TFR (dB) 色条，范围均为 −14 至 14 dB，刻度为 −14、0、14。**关注点**：左侧患者的颜色读出可直接由其右侧色条解释，两例使用相同色标。'
 (OUT/'figures/README.md').write_text('\n\n'.join(f'### {name}.{ext}\n\n{desc}' for name,desc in [('supp_fig6_onset_spectral_features',description),('supp_fig6_panelb_patient_distribution',panel_description),('supp_fig6_panels_ab',ab_description),('supp_fig6_panele_power_curves',e_description),('supp_fig6_tfr_panels',tfr_description)] for ext in ('png','pdf'))+'\n')
 print(OUT/'figures/supp_fig6_onset_spectral_features.png',flush=True)
if __name__=='__main__':main()

"""Frozen G3 outputs: parameter maps, contact timing and unsmoothed native movies."""
from pathlib import Path
import sys, json, pickle, hashlib, warnings
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import rankdata
from PIL import Image
from src.topic4_interictal_repaired_evaluation import rank_features
R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT=R/'candidate_spatial_intuition';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
read=lambda p:json.loads(Path(p).read_text())
score=read(R/'g3_scores.json');selected=sorted(score['candidates'],key=lambda c:c['score']['loss_off'])[:3]
with (ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl').open('rb') as f: evaluator=pickle.load(f)
MAN={'selection':'lowest three G3 equal-unit L_off; descriptive post-confirmation display, not independent validation','candidates':[],'figures':[]}
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=160,bbox_inches='tight')
 plt.close(fig);MAN['figures'].append(name)
def axes(ax,title):
 ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',title=title,xlabel='x (mm)',ylabel='y (mm)');ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
def contacts(ax,xy,names,labels=False):
 for shaft,color in [('SCL','#dc883d'),('ICL','#21a3b1')]:
  ix=[i for i,n in enumerate(names) if n.startswith(shaft)];ix=sorted(ix,key=lambda i:xy[i,0]);ax.plot(xy[ix,0],xy[ix,1],c=color,lw=1.1,zorder=4)
 ax.scatter(*xy.T,s=20,c='white',edgecolors='#444444',linewidths=.5,zorder=5)
 if labels:
  for i in [0,7,1,14]:ax.annotate(names[i],xy[i],xytext=(0,6),textcoords='offset points',ha='center',fontsize=7,zorder=10)
def field_grid(pos,values):
 bins=np.linspace(0,20,81);n=np.histogram2d(*pos.T,bins=(bins,bins))[0];s=np.histogram2d(*pos.T,bins=(bins,bins),weights=values)[0]
 return np.divide(s,n,out=np.zeros_like(s),where=n>0).T
def ranks(x):
 out=np.full_like(x,np.nan,dtype=float)
 for i,row in enumerate(x):
  ok=np.isfinite(row)
  if ok.sum()>1:out[i,ok]=(rankdata(row[ok])-1)/(ok.sum()-1)
 return out
def summary(x):
 rr=ranks(x)
 with warnings.catch_warnings():warnings.simplefilter('ignore');mean=np.nanmean(rr,axis=0)
 part=np.isfinite(x).mean(0);return mean,part
def timing(ax,xy,names,x,title,background=None):
 axes(ax,title)
 if background is not None:ax.imshow(background,origin='lower',extent=(0,20,0,20),cmap='Greys',vmin=0,vmax=1,alpha=.23,interpolation='nearest')
 contacts(ax,xy,names)
 mean,part=summary(x);ok=np.isfinite(mean)
 ax.scatter(xy[ok,0],xy[ok,1],c=mean[ok],s=25+100*part[ok],cmap='viridis',vmin=0,vmax=1,edgecolors='white',lw=.7,zorder=6)
 # A mean arrow is deliberately omitted: it can hide branching and event variability.
 ax.text(.03,.97,f'n = {len(x)} events',transform=ax.transAxes,va='top',fontsize=9)
 return {'mean_normalized_centroid_rank':mean.tolist(),'participation':part.tolist(),'n':len(x)}
datasets=[]
for number,c in enumerate(selected,1):
 cid=c['candidate_id'];allx={0:[],1:[]};static=None;unitcounts={};first=None
 for unit,u in c['units'].items():
  wp=Path(u['worker_path']);rec=read(wp)
  with np.load(wp.with_suffix('.npz')) as z:
   if static is None:static={k:np.array(z[k]) for k in ['positions_E','h','delta_vtheta','contact_xy_mm','contact_names']};audit=rec
   if unit=='topo_6101_dyn_7101':first={k:np.array(z[k]) for k in ['sheet_activity_counts','sheet_activity_frame_ms','contact_xy_mm','contact_names']}
  op=wp.parent.parent/'repaired_observation'/wp.with_suffix('.npz').name
  with np.load(op) as z:x=z['centroid_ms'][z['primary_event_indices']];windows=z['windows_ms'][z['primary_event_indices']]
  lab=evaluator.km.predict(rank_features(x));assert np.bincount(lab,minlength=2).tolist()==u['mode_counts']
  unitcounts[unit]=np.bincount(lab,minlength=2).tolist()
  for mode in [0,1]:allx[mode].append(x[lab==mode])
  if unit=='topo_6101_dyn_7101':first.update(x=x,windows=windows,labels=lab)
 allx={m:np.concatenate(v) for m,v in allx.items()};p=np.array(c['parameters']);xy=static['contact_xy_mm'];names=static['contact_names'].astype(str)
 d={'c':c,'number':number,'static':static,'audit':audit,'x':allx,'first':first,'unitcounts':unitcounts};datasets.append(d)
 fields=[('Threshold shift (local)','RdBu_r',field_grid(static['positions_E'],static['delta_vtheta']),-1.8,1.8,'Mean applied shift (mV)'),('E to E strength (global)','Purples',np.full((2,2),p[5]),.8,1.25,'Weight / baseline'),('E to I strength (global)','Greens',np.full((2,2),p[6]),.8,1.25,'Weight / baseline'),('I to E strength (global)','Blues',np.full((2,2),p[7]),.8,1.25,'Weight / baseline'),('GABA decay (global)','Oranges',np.full((2,2),p[8]),12,26,'Decay time (ms)')]
 fig,axs=plt.subplots(2,4,figsize=(15,9));fig.subplots_adjust(hspace=.55,wspace=.40,top=.85,bottom=.19)
 for ax,(title,cmap,f,lo,hi,label) in zip(axs.flat,fields):
  im=ax.imshow(f,origin='lower',extent=(0,20,0,20),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');axes(ax,title);contacts(ax,xy,names)
  cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.19,orientation='horizontal');cb.set_label(label,fontsize=8);cb.ax.tick_params(labelsize=8)
  if title.startswith('Threshold'):
   ax.scatter(*p[:4].reshape(2,2).T,marker='+',c='black',s=65);ax.text(.04,.04,f'Offset gain = {p[4]:.2f}',transform=ax.transAxes,fontsize=9)
  else:
   val=p[8] if 'GABA' in title else p[{'E to E strength (global)':5,'E to I strength (global)':6,'I to E strength (global)':7}[title]]
   ax.text(.04,.04,f'{val:.2f}'+(' ms' if 'GABA' in title else ' x'),transform=ax.transAxes,fontsize=10)
 ax=axs[1,1];axes(ax,'E to E directional geometry');contacts(ax,xy,names)
 actual=audit['mechanism_freeze']['ellipse_audit']['weighted_geometry_after'];ang=actual['major_axis_deg_unoriented'];ratio=actual['sqrt_eigenvalue_ratio']
 for y in [3,9,15]:
  for xx in [3,10,17]:ax.add_patch(Ellipse((xx,y),4,4/ratio,angle=ang,facecolor='#9673b8',alpha=.25,edgecolor='#633b8a',lw=1.2))
 ax.text(.03,.96,f'Applied weighted axis: {ang:.1f} deg\nAxis ratio: {ratio:.2f}',transform=ax.transAxes,va='top',fontsize=8)
 hh=field_grid(static['positions_E'],static['h']);summ={}
 for ax,m,name in [(axs[1,2],1,'Model TA label'),(axs[1,3],0,'Model TB label')]:summ[m]=timing(ax,xy,names,allx[m],name,hh)
 fig.suptitle(f'Candidate {number} | confirmation loss {c["score"]["loss_off"]:.3f}\n'+['Upper cores + joint parameter changes','Lower, near-horizontal cores','Upper, closer cores + smaller threshold offsets'][number-1],fontsize=15)
 fig.text(.5,.035,'Same SEEG layout in every panel. Parameter color scales are fixed across candidates; uniform maps denote global controls.\nTA/TB panels: purple = earlier, yellow = later activity centroid; dot size = participation; gray = core mask. All 4 confirmation runs pooled.\nLabels organize comparison; these are not proven patient-like routes. Ellipses summarize weighted edge geometry, not local connection density.',ha='center',fontsize=9)
 save(fig,f'candidate_{number}_parameter_and_modes')
 MAN['candidates'].append({'number':number,'candidate_id':cid,'parameters':p.tolist(),'loss':c['score']['loss_off'],'per_unit_mode_counts':unitcounts,'contact_summaries':summ,'static_display_unit':next(iter(c['units'])),'actual_weighted_EE_axis_deg':ang,'actual_weighted_EE_axis_ratio':ratio})
 print('candidate',number,'maps done',flush=True)

# Compact side-by-side overview: applied local field and observed contact orders.
fig,axs=plt.subplots(3,3,figsize=(10,12));fig.subplots_adjust(top=.91,bottom=.16,hspace=.45,wspace=.28)
for row,d in enumerate(datasets):
 s=d['static'];xy=s['contact_xy_mm'];names=s['contact_names'].astype(str);p=d['c']['parameters'];ax=axs[row,0]
 im=ax.imshow(field_grid(s['positions_E'],s['delta_vtheta']),origin='lower',extent=(0,20,0,20),cmap='RdBu_r',vmin=-1.8,vmax=1.8,interpolation='nearest');contacts(ax,xy,names,True);axes(ax,f'Candidate {row+1}: threshold shift')
 ax.text(.03,.96,f'Loss {d["c"]["score"]["loss_off"]:.3f}',transform=ax.transAxes,va='top')
 for col,m in [(1,1),(2,0)]:timing(axs[row,col],xy,names,d['x'][m],'TA label' if m else 'TB label',field_grid(s['positions_E'],s['h']))
fig.suptitle('Top three confirmed conditions: location and event ordering',fontsize=16)
fig.text(.5,.035,'Threshold: blue = lower, red = higher; range -1.8 to +1.8 mV (0.25 mm bin means).\nContact color: purple = earlier, yellow = later centroid rank. Larger dots = higher participation.\nAll events from 2 topologies x 2 noise replays; means can conceal different individual paths.',ha='center',fontsize=10)
save(fig,'top3_spatial_overview')

# Native fields: earliest two primary events per mode, no path-based selection.
for d in datasets:
 first=d['first'];counts=first['sheet_activity_counts'];frames=[];info=[];fig,axs=plt.subplots(4,5,figsize=(13,10));fig.subplots_adjust(top=.87,bottom=.18,hspace=.55,wspace=.25)
 chosen=[(m,int(i)) for m in [1,0] for i in np.flatnonzero(first['labels']==m)[:2]]
 cap=max(1,float(np.quantile(counts,.9995)))
 for row,(m,i) in enumerate(chosen):
  start,stop=first['windows'][i];info.append({'mode':m,'primary_index':i,'window_ms':[float(start),float(stop)]})
  for col,rel in enumerate([20,60,100,140,180]):
   ax=axs[row,col];j=int((start+rel)/float(first['sheet_activity_frame_ms']));im=ax.imshow(counts[j],origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=cap,interpolation='nearest');contacts(ax,first['contact_xy_mm'],first['contact_names'].astype(str));axes(ax,f'+{rel} ms')
   if col==0:ax.set_ylabel(('TA' if m else 'TB')+f' event {i+1}\ny (mm)')
 fig.suptitle(f'Candidate {d["number"]}: native activity, first two events per label\nSame topology 6101 / noise 7101; 2 ms frames without spatial smoothing',fontsize=14)
 cbax=fig.add_axes([.32,.095,.36,.012]);fig.colorbar(im,cax=cbax,orientation='horizontal',label='Spikes per 1 mm cell / 2 ms')
 fig.text(.5,.025,'Time is relative to the frozen detection window, not ignition. Fixed within-candidate activity scale; no trajectory similarity selection.',ha='center',fontsize=9)
 save(fig,f'candidate_{d["number"]}_native_event_sequence');MAN['candidates'][d['number']-1]['native_preview_selection']=info

MAN['producer']=str(Path(__file__).resolve());MAN['producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(MAN,indent=2)+'\n')
lines=[]
for name in MAN['figures']:
 if 'parameter_and_modes' in name:desc='同一候选的五类实际参数空间图、实际加权EE轴与两类事件的接触点质心顺序。各参数使用独立色系，同一种参数在三个候选之间保持同一色标；阈值来自保存的逐神经元偏移的0.25 mm分箱均值，其余强度及时间常数是全局设置。';focus='对照局部双核与全局参数；跨候选同时变化多个参数，不能归因于单一参数。'
 elif 'native' in name:desc='同图同噪声下，每个标签最先两次primary事件的原生二维活动；显示检测窗后20/60/100/140/180 ms的实际2 ms帧。保持SEEG布局，无空间平滑，不按患者相似度选事件。';focus='检查聚合平均是否掩盖多热点或不同路径；检测窗零点不代表起燃。'
 else:desc='确认损失最低的三个条件按排名排列，左列为实际阈值偏移，右两列为全部四次确认运行中各模式的参与和质心顺序。接触点颜色为逐事件归一化rank的条件均值，大小表示参与率，灰色底图仅标示双核。';focus='最优候选已移至上部；两类标签和平均rank不等于真实传播路径恢复。'
 lines.append(f'### {name}.png / .pdf\n\n{desc}\n\n**关注点**：{focus}\n')
(F/'README.md').write_text('\n'.join(lines));print('DONE',flush=True)

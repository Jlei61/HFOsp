#!/usr/bin/env python3
"""Same-trajectory M-on Z-refill figure, with twin-axis Z/M and native readouts."""
from pathlib import Path
import argparse,json,pickle,time,os
import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.colors import Normalize,PowerNorm,TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plot_topic4_fig5_m_return as native
import plot_topic4_fig5_manual_release_layout_v4 as geometry
from plot_topic4_fig5_manual_release_layout_v3 import ACTIVITY_CMAP

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'weak_fast_z_refill_v1';FIG=OUT/'figures'
NAME='weak_fast_z_refill'
ORDER=['SCL9','SCL8','SCL7','SCL6','ICL11','ICL10','ICL9','ICL8','ICL7','ICL6','ICL5','ICL4','ICL3','ICL2','ICL1']
REGIONS=[(0,'#654080','All E'),(5,'#b23a6b','Core A'),(6,'#157da2','Core B')]
COLORS=['#2576a0','#cf7b13','#ba253d','#238f79','#39765f']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':18,'axes.titlesize':19,
 'xtick.labelsize':16,'ytick.labelsize':16,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(fig,name):
 if hasattr(fig,'_slow_legend_layout'):
  za,legend,heading=fig._slow_legend_layout
  fig.canvas.draw();renderer=fig.canvas.get_renderer()
  low=heading.get_window_extent(renderer).y1
  high=za.xaxis.label.get_window_extent(renderer).y0
  x=za.get_position().x0+za.get_position().width/2
  y=fig.transFigure.inverted().transform((0,(low+high)/2))[1]
  legend.set_bbox_to_anchor((x,y),transform=fig.transFigure)
 FIG.mkdir(exist_ok=True);fig.savefig(FIG/(name+'.png'),dpi=155,bbox_inches='tight',pad_inches=.15)
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.15);plt.close(fig)

def snapshots(metric,r):
 events=metric['events'];release=r['release_ms']/1000
 finite=[v for v in events if 68.05<v['peak_s']<70]
 one=min(finite,key=lambda v:abs(v['peak_s']-68.575))['peak_s']
 after=[v for v in events if release+.15<v['peak_s']<release+12]
 four=after[0]['peak_s'] if after else release+2.
 later=[v for v in events if max(four+1,metric['duration_s']-6)<v['peak_s']<metric['duration_s']-.05]
 five=later[-1]['peak_s'] if later else metric['duration_s']-.25
 times=[one,72.8,r['restore_start_ms']/1000-.15,four,five]
 labels=['Self-limited','Entry','High rate','After refill' if after else 'Post-refill','Later event' if later else 'Late activity']
 out=[]
 for n,(t,label) in enumerate(zip(times,labels),1):
  lo=round(t*1000)-25
  out.append(dict(number=n,time_s=t,label=label,lo_ms=lo,hi_ms=lo+50,color=COLORS[n-1]))
 return out

def left(fig,spec,a,r,metric,snaps,start=68.):
 end=metric['duration_s'];gs=spec.subgridspec(5,1,height_ratios=[1.8,1.18,.95,.10,.79],hspace=.36)
 ids=np.array([list(a['contact_names']).index(n) for n in ORDER]);lt=a['lfp_time_ms']/1000
 baseline=(lt>=.5)&(lt<1);raw=a['lfp_effective'][:,ids]
 # One amplitude gain throughout this run; no stage-specific normalization.
 x=raw-np.median(raw[baseline],axis=0)
 early=(lt>=68)&(lt<71)
 gain=float(np.max(np.quantile(x[early],.995,axis=0)-np.quantile(x[early],.005,axis=0)))
 sel=(lt>=start)&(lt<end);x=x[sel]/gain*.8+np.arange(15)[::-1]
 ax=fig.add_subplot(gs[0]);axes=[ax]
 for j in range(15):ax.plot(lt[sel],x[:,j],c='#285c70' if j<4 else '#955d2b',lw=.75,rasterized=True)
 ax.set(yticks=np.arange(15)[::-1],yticklabels=ORDER,ylim=(-.4,15.2),ylabel='Virtual SEEG\ncurrent proxy (a.u.)')
 ax.set_title('A  Unfiltered contact readout',loc='left',weight='bold',pad=35)
 ax.text(1,1.025,'ηM = 0.02, τM = 2 s',transform=ax.transAxes,fontsize=16,ha='right')
 ra=fig.add_subplot(gs[1]);axes.append(ra)
 chosen=np.r_[np.arange(0,60,3),np.arange(60,120,3),np.arange(120,240,6),np.arange(240,300,3)]
 lo=round(start*10000);hi=round(end*10000);ti,ni=np.where(a['sample_spikes'][lo:hi,chosen]);ti=(ti+lo)*.0001
 for l,h,col in [(0,60,'#245e80'),(60,80,'#a26425')]:
  mask=(ni>=l)&(ni<h);ra.scatter(ti[mask],ni[mask],s=1.7,c=col,marker='.',lw=0,rasterized=True)
 for y in [19.5,39.5,59.5]:ra.axhline(y,color='#bbbbbb',lw=.6)
 ra.set(ylim=(-1,80),yticks=[10,30,50,70],yticklabels=['Core A E','Core B E','Other E','I'])
 ra.set_title('B  Continuous spike raster',loc='left',weight='bold',pad=13)
 za=fig.add_subplot(gs[2]);ma=za.twinx();ma.spines['right'].set_visible(True);axes.append(za)
 zt=a['z_time_ms']/1000;q=(zt>=start)&(zt<end);zt=zt[q];zs=a['z_stats'][q];ms=a['m_stats'][q]*r['eta_m']
 za.fill_between(zt,zs[:,2],zs[:,4],color=REGIONS[0][1],alpha=.12,lw=0)
 for idx,col,label in REGIONS:
  za.plot(zt,zs[:,idx],c=col,lw=1.45)
  ma.plot(zt,ms[:,idx],c=col,lw=1.5,ls='--')
 za.set(ylabel='Resource Z',ylim=(0,1.07),yticks=[0,.5,1],xlabel='Time, t (s)')
 ma.set(ylabel='ηM × M (mV equiv.)',ylim=(0,max(1,float(ms[:,[0,5,6]].max())*1.17)))
 ma.yaxis.labelpad=11
 za.set_title('C  Inhibition resource and adaptation',loc='left',weight='bold',pad=14)
 handles=[Line2D([],[],color=c,label=l,lw=2) for _,c,l in REGIONS]
 handles += [Line2D([],[],color='#303030',lw=1.5,label='Z'),Line2D([],[],color='#303030',ls='--',lw=1.5,label='ηM × M')]
 legend=za.legend(handles=handles,ncol=5,loc='center',bbox_to_anchor=(.5,-.39),frameon=False,fontsize=13,handlelength=1.8,columnspacing=1.1)
 refill=r['restore_start_ms']/1000;release=r['release_ms']/1000
 ticks=np.arange(70,end+.01,5)
 for axis in axes:
  axis.set(xlim=(start,end),xticks=ticks)
  for l,h in metric['high_intervals_s']:axis.axvspan(l,h,color='#b32e43',alpha=.105,lw=0)
  axis.axvspan(refill,release,color='#359f86',alpha=.13,lw=0)
  for t in [refill,release]:axis.axvline(t,c='#27886f',ls='--',lw=1.)
  for s in snaps:axis.axvline(s['time_s'],c=s['color'],ls=':',lw=.9,alpha=.8)
  if axis is not za:axis.tick_params(labelbottom=False)
 ma.set_xlim(start,end)
 ax.annotate('Refill Z',xy=(refill,1.025),xycoords=ax.get_xaxis_transform(),xytext=(-12,0),textcoords='offset points',ha='right',fontsize=15,color='#27886f')
 ax.annotate('Release Z',xy=(release,1.025),xycoords=ax.get_xaxis_transform(),xytext=(12,0),textcoords='offset points',ha='left',fontsize=15,color='#27886f')
 for s in snaps:ax.text(s['time_s'],.99,str(s['number']),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=16,weight='bold',color=s['color'],bbox=dict(fc='white',ec='none',alpha=.8,pad=.3))
 header=fig.add_subplot(gs[3]);header.axis('off');heading=header.text(0,.35,'D  Spatial activity (50-ms windows)',fontsize=18,weight='bold')
 fig._slow_legend_layout=(za,legend,heading)
 maps=gs[4].subgridspec(1,5,wspace=.29)
 for k,s in enumerate(snaps):
  m=fig.add_subplot(maps[k]);field=a['field_e_count_1ms'][s['lo_ms']:s['hi_ms']].sum(0)/a['cell_e_counts']/.05
  im=m.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],interpolation='nearest',cmap=ACTIVITY_CMAP,norm=PowerNorm(.6,vmin=0,vmax=500))
  for xy in a['centers_mm']:m.add_patch(Circle(xy,1.5,fc='none',ec='#34cfce',lw=1.3))
  m.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)');m.tick_params(labelsize=15)
  if k==0:m.set_ylabel('y (mm)')
  else:m.set_yticklabels([])
  m.set_title(f'{k+1}  {s["label"]}\n{s["time_s"]:.2f} s',color=s['color'],fontsize=15,pad=10)
  if k==4:
   ca=m.inset_axes([1.05,0,.045,1]);cb=fig.colorbar(im,cax=ca,ticks=[0,250,500]);cb.set_label('E rate (Hz)',fontsize=16);ca.tick_params(labelsize=14)
 bounds=np.array([[p.get_position().x0,p.get_position().x1] for p in axes+[ma]])
 assert np.allclose(bounds,bounds[0])
 return dict(time_limits_s=[start,end],same_time_axis=True,C_Z_and_M_same_coordinate_frame=True,M_display='eta_M times mean spike-triggered M, dashed, right y-axis',raster_sample_n=80,raster_spikes=len(ti),readout_gain=gain,contact_order=ORDER)

def trajectory(fig,spec,a,r,snaps,start=68.):
 ax=fig.add_subplot(spec,projection='3d');ax.computed_zorder=False
 rate=gaussian_filter1d(a['rate_e_hz'].reshape(-1,50).mean(1),1)
 t=(np.arange(len(rate))+.5)*.005;zt=a['z_time_ms']/1000
 xyz=np.column_stack([np.interp(t,zt,a['z_stats'][:,0]),np.interp(t,zt,gaussian_filter1d(a['currents_5ms'][:,2],1)),rate])
 end=snaps[-1]['time_s']+.05;q=(t>=start)&(t<=end);tt=t[q];xx=xyz[q]
 norm=Normalize(start,end);cmap=plt.get_cmap('viridis');rs=r['restore_start_ms']/1000;re=r['release_ms']/1000
 # Carry every 5-ms bin; dense low-speed samples may overlap, never bridge a missing interval.
 for l,h,width,zorder in [(start,rs,1.15,3),(rs,re,3.1,7),(re,end,1.3,5)]:
  st,pts=geometry.exact_path(t,xyz,l,h)
  if l==rs:ax.plot(*pts.T,c='#616975',lw=4.4,alpha=.65,zorder=6)
  lines=Line3DCollection(np.stack([pts[:-1],pts[1:]],axis=1),colors=cmap(norm((st[:-1]+st[1:])/2)),linewidths=width,zorder=zorder)
  ax.add_collection3d(lines)
  # Local arrows along the actual curve, emphasizing the externally forced return.
  distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts/np.array([.5,1000,500]),axis=0),axis=1))]
  for frac in ([.15,.35,.65,.88] if l==rs else [.25,.7]):
   k=min(np.searchsorted(distance,distance[-1]*frac),len(pts)-2);j=min(np.searchsorted(distance,distance[k]+.04),len(pts)-1)
   if j>k:geometry.arrow(ax,pts[k],pts[j],cmap(norm(st[k])),size=14 if l==rs else 10)
 for s in snaps:
  pt=geometry.point_at(t,xyz,s['time_s']);ax.scatter(*pt,s=38,fc='white',ec='#3e4650',lw=1.2,depthshade=False,zorder=20)
  offsets={1:(-15,22),2:(-25,-19),3:(16,15),4:(18,18),5:(-22,-23)}
  geometry.label(ax,str(s['number']),pt,offsets[s['number']],'#3e4650')
 for tm in [rs,re]:
  pt=geometry.point_at(t,xyz,tm);ax.scatter(*pt,s=32,marker='s',color=cmap(norm(tm)),edgecolor='#525963',depthshade=False,zorder=21)
 upper=float(np.ceil(xx[:,1].max()/200)*200)
 ax.set(xlim=(min(.55,float(xx[:,0].min())-.025),1.025),ylim=(-.035*upper,1.04*upper),zlim=(-12,max(420,float(xx[:,2].max())*1.06)),
  xticks=[.6,.8,1],yticks=[0,upper/2,upper],zticks=[0,200,400],xlabel='Mean Z',ylabel=r'$H_E$ (mV equiv.)',zlabel=r'E rate, $r_E$ (Hz)')
 ax.view_init(elev=24,azim=-125);ax.set_box_aspect((1.2,1,1.05));ax.grid(True,alpha=.3)
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.label.set_fontsize(17);axis.labelpad=14;axis.pane.set_facecolor('#f3f6fa')
 ax.tick_params(labelsize=15);ax.set_title('E1  State trajectory',loc='left',weight='bold',pad=15)
 ca=ax.inset_axes([1.06,.19,.029,.64]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ca);cb.set_label('Time, t (s)',rotation=270,labelpad=22,fontsize=17);ca.tick_params(labelsize=15)
 return dict(time_s=tt,coordinates_Z_H_E=xx)

def early_comparison(fig,spec,a):
 # Same estimator, fixed clinical SZ13. Recompute model values and correlation without selecting another seizure.
 fs=2000.;lt=a['lfp_time_ms']/1000
 def pwr(x):
  f,_,p=spectrogram(np.asarray(x,float),fs=fs,nperseg=2000,noverlap=1000,scaling='density',mode='psd',axis=0)
  return p[(f>=1)&(f<=150)].sum(0).T
 b=(lt>=.5)&(lt<8);e=(lt>=73)&(lt<74)
 bp=pwr(a['lfp_effective'][b]).mean(0);ep=pwr(a['lfp_effective'][e])[0];db=10*np.log10(ep/bp)
 old=np.load(BASE/'patient_readout_v1/comparison_arrays.npz');index=[list(old['contact_names']).index(n) for n in a['contact_names']]
 patient=old['patient_robust_z'][index];assert read(BASE/'patient_readout_v1/summary.json')['selected']['public_seizure']=='SZ13'
 rho=float(spearmanr(db,patient).statistic)
 gs=spec.subgridspec(3,1,height_ratios=[.17,1,.16],hspace=.15)
 h=fig.add_subplot(gs[0]);h.axis('off');h.text(0,.4,'E2  Early energy · 1–150 Hz',weight='bold',fontsize=20)
 maps=gs[1].subgridspec(1,5,width_ratios=[1,.045,.20,1,.045],wspace=.15)
 xy=a['contact_xy'];grid=np.linspace(0,20,201);xx,yy=np.meshgrid(grid,grid);w=np.exp(-((xx[...,None]-xy[:,0])**2+(yy[...,None]-xy[:,1])**2)/(2*2.5**2));support=w.sum(-1);alpha=np.clip(support/(.32*support.max()),0,1)
 for k,(values,title,label) in enumerate([(db,'Model readout\n73–74 s','Change (dB)'),(patient,'E1146 · SZ13\n0–10 s from onset','Power (robust z)')]):
  ax=fig.add_subplot(maps[0 if k==0 else 3]);vmax=max(1,float(np.ceil(abs(values).max())))
  norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax) if values.min()<0 else Normalize(0,vmax)
  # Keep blue = enhancement in both panels; red shows model power suppression.
  cmap='RdBu' if values.min()<0 else 'Blues'
  field=(w*values).sum(-1)/support
  ax.imshow(field,origin='lower',extent=[0,20,0,20],cmap=cmap,norm=norm,alpha=alpha,interpolation='nearest')
  ax.scatter(xy[:,0],xy[:,1],c=values,cmap=cmap,norm=norm,s=47,edgecolors='#222222',lw=.8)
  if k==0:
   for xy0 in a['centers_mm']:ax.add_patch(Circle(xy0,1.5,fc='none',ec='#dd743c',lw=1.2))
  ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (model mm)',ylabel='y (model mm)' if k==0 else '')
  if k==1:ax.set_yticklabels([])
  ax.set_title(title,fontsize=17,pad=10)
  ca=fig.add_subplot(maps[1 if k==0 else 4]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ca);cb.set_label(label,fontsize=16);ca.tick_params(labelsize=14)
 foot=fig.add_subplot(gs[2]);foot.axis('off');foot.text(.5,.1,f'Same SZ13 example · 15 contacts · ρ = {rho:.2f}',ha='center',fontsize=16)
 return dict(model_window_s=[73,74],baseline_s=[.5,8],patient='E1146 SZ13',patient_window_s=[0,10],rho=rho,model_dB=db.tolist(),model_positive_contacts=int((db>0).sum()),patient_robust_z=patient.tolist(),band_hz=[1,150],new_patient_selection=False,amplitudes_comparable=False)

def render():
 with np.load(OUT/'runs'/f'{NAME}.npz') as f:a={k:f[k] for k in f.files}
 r=read(OUT/'runs'/f'{NAME}.json');metric=native.analyze(a,r);assert not metric['no_intervention']
 metric['activity_pattern_classification']=metric['classification']
 metric['classification']='EXTERNAL_REFILL_'+metric['classification']
 metric['return_mechanism']='External one-second Z refill; M was not reset'
 metric['autonomous_return_demonstrated']=False
 snaps=snapshots(metric,r);start=r['restore_start_ms'];release=r['release_ms']
 with (OUT/'parent_state.pkl').open('rb') as f:parent=pickle.load(f)
 n=parent['engine']['step'];obs=parent['observations'];assert n==755000
 qa=dict(prefix_rates_bitwise=np.array_equal(a['rate_e_hz'][:n],obs['rates'][:,0]),prefix_raster_bitwise=np.array_equal(a['sample_spikes'][:n],obs['raster']),
 prefix_Z_M_bitwise=np.array_equal(a['z_stats'][:len(obs['zs'])],np.asarray(obs['zs'])) and np.array_equal(a['m_stats'][:len(obs['mstats'])],np.asarray(obs['mstats'])),
 M_reconstruction_max_error=metric['M_reconstruction_error'],M_not_reset=metric['M_reconstruction_error']<1e-8,
 spatial_count_conservation=r['spatial_count_conservation'],no_I_cell_M=bool(np.all(parent['engine']['slow']['m'][32000:]==0)))
 tz=a['z_time_ms'];i=np.flatnonzero(tz==start)[0];j=np.flatnonzero(tz==release)[0]
 frac=((tz[i:j+1]-start)/1000)[:,None];pred=a['z_field_5ms'][i]+frac*(1-a['z_field_5ms'][i])
 qa['Z_refill_linearity_error']=float(abs(pred-a['z_field_5ms'][i:j+1]).max());qa['Z_refill_end_all_one']=bool(np.all(a['z_field_5ms'][j]==1))
 qa['Z_native_after_release']=bool(np.any(a['z_stats'][j+1:,0]<.999));assert qa['Z_refill_linearity_error']<1e-12
 assert all(qa[k] for k in ['prefix_rates_bitwise','prefix_raster_bitwise','prefix_Z_M_bitwise','M_not_reset','spatial_count_conservation','Z_refill_end_all_one','Z_native_after_release'])
 for s in snaps:
  m=(tz>=s['lo_ms'])&(tz<s['hi_ms']);s['mean_Z']=float(a['z_stats'][m,0].mean());s['mean_adaptation_current']=float(.02*a['m_stats'][m,0].mean());s['mean_E_hz_50ms']=float(a['rate_e_hz'][s['lo_ms']*10:s['hi_ms']*10].mean())
 fig=plt.figure(figsize=(15.5,18));gs=fig.add_gridspec(1,1,left=.14,right=.85,top=.94,bottom=.07)
 layout=left(fig,gs[0],a,r,metric,snaps);save(fig,'weak_fast_Zreset_left')
 fig=plt.figure(figsize=(27.5,18));outer=fig.add_gridspec(1,2,width_ratios=[1.5,1],left=.075,right=.93,top=.95,bottom=.065,wspace=.34)
 left(fig,outer[0],a,r,metric,snaps)
 right=outer[1].subgridspec(2,1,height_ratios=[1.2,1],hspace=.33)
 phase=trajectory(fig,right[0],a,r,snaps);comparison=early_comparison(fig,right[1],a)
 save(fig,'weak_fast_Zreset_assembled')
 np.savez_compressed(OUT/'trajectory_arrays.npz',**phase)
 write(OUT/'analysis_summary.json',metric);write(OUT/'figure_metadata.json',dict(source_run=str(OUT/'runs'/f'{NAME}.npz'),eta_m=.02,tau_M_s=2.,refill_s=[start/1000,release/1000],windows=snaps,left=layout,E2=comparison,
 omitted_F='Old7x7x3 Z-kinetics sweep was M-off and cannot be relabeled as M-on. No new latency sweep in this one-trajectory intervention.',
 trajectory='Observed native Z / applied GABA current / E rate; continuous time colors; externally forced return is not an autonomous limit cycle.'))
 write(OUT/'artifact_qa.json',dict(**qa,agent_visual_review='PENDING',human_acceptance='PENDING_USER_REVIEW'))
 (FIG/'README.md').write_text('### weak_fast_Zreset_left.png / .pdf\n固定手放双核、ηM=0.02与τM=2秒，从75.5秒完整状态继续，75.5–76.5秒只把E细胞Z补到1，再释放原生Z演化至96.5秒。A为原生15触点读出，B为连续80细胞抽样raster，C左轴实线Z、右轴虚线ηM×M，D为同一时间序列的50ms空间窗。\n**关注点**：M和所有快状态连续保留；返回由外部补回Z触发，不能标为M自主终止。编号⑤按实际晚期活动标注，不强行要求第二次runaway。\n\n### weak_fast_Zreset_assembled.png / .pdf\n同一M-on轨迹拼接左图、E1三维实际轨迹和E2电极早期能量。E2模型73–74秒重新计算1–150Hz增强，患者固定原SZ13例，不重新挑选高相关发作；旧M-off的F扫描未沿用。\n**关注点**：模型dB与患者robust-z幅度不能直接等同，插值仅供空间展示，相关在原始15触点计算；此候选仍待用户目视审阅。\n')
 write(OUT/'status.json',dict(status='FIGURES_READY_PENDING_VISUAL_REVIEW',figure=str(FIG/'weak_fast_Zreset_assembled.png'),duration_s=metric['duration_s'],first_low_return_s=metric['first_low_return_s']))
 print(json.dumps(dict(metric={k:v for k,v in metric.items() if k!='events'},windows=snaps,E2=comparison,qa=qa),indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
 if args.watch:
  while not (OUT/'runs'/f'{NAME}.json').exists():
   progress=OUT/'progress'/f'{NAME}.json'
   if progress.exists() and read(progress).get('status')=='FAILED':raise RuntimeError(read(progress))
   time.sleep(20)
 render()
if __name__=='__main__':
 try:main()
 except Exception as e:write(OUT/'figure_status.json',dict(status='FAILED',error=repr(e)));raise

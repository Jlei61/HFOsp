"""R1 saved-trajectory figures, using the accepted Figure 1E rank painter.

No simulation, fitting, patient-distance selection, or change to the R1 contract.
"""
from pathlib import Path
import sys, json, warnings
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse,Patch
from matplotlib.lines import Line2D
from scipy.stats import rankdata,spearmanr
from src import topic4_initial_state_runtime as rt
from scripts import plot_interictal_propagation as pp
from scripts.paper_figures.plot_fig1_interictal_hfo_temporal_scaffold import _draw_fig1e_cluster_row
from scripts.review_topic4_same_network_events import group_times

D=rt.read(ROOT/'config/topic4_continuous_core_state_r1.json');BASE=Path(D['output_root'])
OUT=BASE/'representative_review';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
COL={0:'#277DA1',1:'#C43C39'};SHAFT={'ICL':'#E67E22','SCL':'#159EAE'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
C=rt.candidate_record(D);centers=np.asarray(C['node_field']['centers_mm'])
EV=rt.load_evaluator(D)
records=[];inputs=[]
for j in D['jobs']:
 if j['kind']!='ou':continue
 p=BASE/'workers'/(j['id']+'.json');r=rt.read(p)
 assert rt.sha(p.with_suffix('.npz'))==r['arrays_sha256']
 with np.load(p.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
 ids=np.array([e['event_index'] for e in r['events'] if e['primary_eligible'] and e['window_ms'][0]>=1500 and e['window_ms'][1]<=30000],int)
 records.append((r,a,ids));inputs.append({'path':str(p),'sha256':rt.sha(p),'arrays_sha256':r['arrays_sha256']})
records.sort(key=lambda x:x[0]['job']['id']);r,a,ids=records[0]
names=a['contact_names'].astype(str).tolist();xy=a['contact_xy_mm'];N=len(names)
assert names==rt.load_observation_contract(D)['contact_names']
def ranks(x):
 result=np.full_like(x,np.nan,dtype=float)
 for i,row in enumerate(x):
  ok=np.isfinite(row)
  if ok.sum()>1:result[i,ok]=(rankdata(row[ok])-1)/(ok.sum()-1)
 return result
pr=ranks(EV.fit);order=pp._fixed_channel_order(pr.T,np.isfinite(pr.T));ordered=[names[i] for i in order]
def mean(x):
 count=np.isfinite(x).sum(0)
 return np.divide(np.nansum(x,axis=0),count,out=np.full(x.shape[1],np.nan),where=count>0)
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
def plane(ax,title):
 ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',title=title,xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])
def contacts(ax):
 for shaft,col in SHAFT.items():
  ix=sorted([i for i,n in enumerate(names) if n.startswith(shaft)],key=lambda i:int(names[i][3:]))
  ax.plot(xy[ix,0],xy[ix,1],color=col,lw=1.4,zorder=5)
 ax.scatter(*xy.T,s=24,fc='white',ec='.5',lw=.65,zorder=6)
def cores(ax):
 for k,(x,y) in enumerate(centers):
  ax.add_patch(Circle((x,y),r['radius_mm'],fc='none',ec='#a44545',lw=1.05,zorder=7))
  ax.text(x,y+r['radius_mm']+.5,chr(65+k),ha='center',fontsize=9,color='#973b3b',zorder=8)
def hgrid():
 bins=np.linspace(0,20,81);pos=a['positions_E'];count=np.histogram2d(*pos.T,bins=(bins,bins))[0]
 total=np.histogram2d(*pos.T,bins=(bins,bins),weights=a['h'])[0]
 return np.divide(total,count,out=np.zeros_like(total),where=count>0).T
h=hgrid();chosen=[]
for m in [0,1]:
 take=ids[a['event_mode'][ids]==m];phi=a['event_phi'][take]
 distance=((phi-phi.mean(0))**2).sum(1);i=int(take[np.argmin(distance)]);e=r['events'][i]
 chosen.append(e)
assert len({e['event_index'] for e in chosen})==2
selected=[dict(e,job=r['job']['id'],within_run_mode_count=int(np.sum(a['event_mode'][ids]==e['mode'])),
               routes=group_times(a['centroid_ms'][e['event_index']],names)) for e in chosen]

# A: same geometry and same continuous replay, complete 15-contact train.
fig=plt.figure(figsize=(18.4,6.1));gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,2.1],left=.045,right=.98,bottom=.32,top=.83,wspace=.36)
ax=fig.add_subplot(gs[0]);im=ax.imshow(h,origin='lower',extent=(0,20,0,20),cmap='plasma',vmin=0,vmax=1,interpolation='nearest')
contacts(ax);cores(ax);plane(ax,'Fixed substrate')
ax.add_patch(Ellipse((14,16),4,2,angle=-22.8053839651,fc='none',ec='white',lw=1.3));ax.text(14,18.3,'EE axis',color='white',ha='center',fontsize=8)
fig.text(.12,.115,'s > 0: I input A increases, B decreases\nFixed projection; total expected I input conserved',ha='center',fontsize=8)
fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.04,pad=.18,ticks=[0,1],label='Fixed core field h')
span=max(np.nanmax(a['centroid_ms'][e['event_index']])-np.nanmin(a['centroid_ms'][e['event_index']]) for e in chosen)
mode_axes=[]
for k,e in enumerate(chosen):
 ax=fig.add_subplot(gs[k+1]);mode_axes.append(ax);ax.imshow(h,origin='lower',extent=(0,20,0,20),cmap='Greys',vmin=0,vmax=1,alpha=.14)
 contacts(ax);cores(ax);mu=a['centroid_ms'][e['event_index']];good=np.isfinite(mu)
 dots=ax.scatter(*xy[good].T,c=(mu-np.nanmin(mu))[good],s=75,cmap='viridis',vmin=0,vmax=span,ec='white',lw=.8,zorder=9)
 plane(ax,f'M{e["mode"]} example');ax.text(.03,.96,f'event {e["event_index"]+1}\ns = {e["z_at_detection_start"]:.2f}',transform=ax.transAxes,va='top',fontsize=9)
axbar=fig.add_axes([.31,.205,.20,.017]);fig.colorbar(dots,cax=axbar,orientation='horizontal',label='Contact activity centroid (ms from earliest)')
right=gs[3].subgridspec(2,1,height_ratios=[1,4],hspace=.18);az=fig.add_subplot(right[0]);at=fig.add_subplot(right[1],sharex=az)
time=np.arange(len(a['state_z']))*float(a['state_dt_ms'])/1000
az.plot(time[::100],a['state_z'][::100],color='.2',lw=.9);az.axhline(0,c='.65',lw=.6);az.set(ylabel='s',xlim=(0,30));az.tick_params(labelbottom=False)
env=a['contact_envelope'];tt=np.arange(env.shape[1])*float(a['contact_envelope_dt_ms'])/1000
scale=max(np.quantile(env,.9999),1e-12)
for k,i in enumerate(order):at.plot(tt,k+.72*env[i]/scale,c='.2',lw=.6)
for i in ids:
 e=r['events'][i];lo,hi=np.array(e['window_ms'])/1000
 at.axvspan(lo,hi,fc=COL[e['mode']],alpha=.13,lw=0)
for e in chosen:
 lo,hi=np.array(e['window_ms'])/1000
 for aa in [at,az]:aa.axvspan(lo,hi,fc=COL[e['mode']],alpha=.20,ec=COL[e['mode']],lw=.9)
 az.text((lo+hi)/2,az.get_ylim()[1],f'M{e["mode"]}',color=COL[e['mode']],ha='center',va='bottom',fontsize=9)
at.set(yticks=np.arange(N),yticklabels=ordered,ylim=(-.6,N-.2),xlabel='Time (s)');at.tick_params(axis='y',labelsize=7)
at.legend(handles=[Patch(fc=COL[m],alpha=.35,label=f'M{m}') for m in [0,1]],frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,-.20),fontsize=8)
fig.suptitle('Continuous-state R1 | same network, first OU replay | patient-compatible propagation NOT established',fontsize=13,y=.98)
fig.text(.5,.025,'Examples nearest their own model-mode mean, without patient-distance selection. Hollow contacts did not participate.\nRight: complete unfiltered firing-density envelopes (model units, one common scale); shading denotes frozen labels, not verified forward/reverse waves.',ha='center',fontsize=9)
save(fig,'same_network_state_and_propagation')

# B: actual unsmoothed 2 ms native frames alongside complete individual envelopes.
fig=plt.figure(figsize=(16,7.2));gs=fig.add_gridspec(2,5,width_ratios=[1.8,1,1,1,1],left=.06,right=.99,bottom=.21,top=.86,hspace=.43,wspace=.24)
cap=max(float(a['sheet_activity_counts'][round(e['window_ms'][0]/2):round(e['window_ms'][1]/2)].max()) for e in chosen)
frame_records=[]
for row,e in enumerate(chosen):
 lo,hi=[round(v/2) for v in e['window_ms']];local=env[:,lo:hi];offset=int(np.argmax(local.sum(0)))
 peak=(lo+offset)*2+1;frame_indices=[int(np.clip(lo+offset+d,lo,hi-1)) for d in [-15,-5,5,15]]
 frame_records.append(dict(event_index=e['event_index'],contact_sum_peak_ms=peak,frame_center_ms=[i*2+1 for i in frame_indices]))
 ax=fig.add_subplot(gs[row,0]);st=(np.arange(lo,hi)*2+1)-peak
 for k,i in enumerate(order):
  ax.plot(st,k+.72*local[i]/scale,c='.3',lw=.8)
  mu=a['centroid_ms'][e['event_index'],i]
  if np.isfinite(mu):ax.scatter(mu-peak,k+.20,s=12,c=COL[e['mode']],zorder=4)
 for f in frame_indices:ax.axvline(f*2+1-peak,color='.8',lw=.65)
 ax.set(yticks=np.arange(N),yticklabels=ordered,ylim=(-.6,N-.2),xlabel='Time from envelope peak (ms)',title=f'M{e["mode"]}, event {e["event_index"]+1}; s={e["z_at_detection_start"]:.2f}');ax.tick_params(axis='y',labelsize=7)
 for col,f in enumerate(frame_indices,1):
  ax=fig.add_subplot(gs[row,col]);im=ax.imshow(a['sheet_activity_counts'][f],origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=cap,interpolation='nearest')
  contacts(ax);cores(ax);plane(ax,f'{f*2+1-peak:+.0f} ms')
  if col>1:ax.set_ylabel('')
cb=fig.add_axes([.48,.115,.35,.015]);fig.colorbar(im,cax=cb,orientation='horizontal',label='Active E cells per 1 mm square / 2 ms (shared scale)')
fig.suptitle('Two typical events from the same continuous run | native activity and contact readout',fontsize=14,y=.98)
fig.text(.5,.015,'Frame times fixed at envelope peak -30 / -10 / +10 / +30 ms; no spatial smoothing. Circles mark fixed cores A/B.\nContact dots show activity centroids, not ignition times. These are the same examples as in the four-column figure.',ha='center',fontsize=9)
save(fig,'representative_native_frames')

# C: Figure 1E shared painter; frozen patient labels, not a new unsupervised fit.
mu=np.concatenate([aa['centroid_ms'][ii] for rr,aa,ii in records]);labels=np.concatenate([aa['event_mode'][ii] for rr,aa,ii in records]);xr=ranks(mu)*(N-1)
patient=np.array([mean(pr[EV.fit_labels==m])*(N-1) for m in [0,1]])
model=np.array([mean(xr[labels==m]) for m in [0,1]])
arr={'ranks':xr.T,'bools':np.isfinite(xr.T),'channel_order':order,'ordered_names':ordered,'clustered_events_all':np.argsort(labels,kind='stable'),'clustered_labels_all':np.sort(labels,kind='stable'),'valid_events':np.arange(len(mu)),'labels':labels,'channel_names':names}
fig=plt.figure(figsize=(17,5.5));gs=fig.add_gridspec(1,5,width_ratios=[3.5,.12,1,1.6,1.3],left=.05,right=.975,bottom=.22,top=.82,wspace=.42)
draw=_draw_fig1e_cluster_row(fig,gs,0,arr,column_indices=(0,1,3),gap_half_width_events=0,cluster_label_names=['M0','M1'],cluster_colors=[COL[0],COL[1]],mean_profile_label_names=['M0','M1'],heatmap_ytick_fontsize=9,cluster_label_fontsize=11,mean_label_fontsize=10)
ah=draw['axes']['heatmap'];ap=draw['axes']['mean_rank'];ah.set_title('');ah.set_ylabel('Contact')
rg=gs[2].subgridspec(2,1,height_ratios=[20,1],hspace=.06);ad=fig.add_subplot(rg[0])
pp._plot_rank_histogram(ad,xr.T,np.isfinite(xr.T),np.arange(len(mu)),order,names,title='Rank distribution',show_ylabels=False,label_fontsize=9,title_fontsize=11,xtick_fontsize=8,ridge_spacing=1.,smooth_sigma_bins=.72,smooth_ridge_height=.7)
for collection in list(ap.collections):collection.remove()
for m in [0,1]:ap.plot(patient[m,order],np.arange(N),'--',c=COL[m],lw=1.5)
for aa in [ap,ad]:aa.set_ylim(ah.get_ylim());aa.set_yticks(ah.get_yticks());aa.set_yticklabels([])
ap.set_title('Model / patient FIT',fontsize=11);ap.set_xlabel('Mean rank position',fontsize=9)
ap.legend(handles=[Line2D([0],[0],color=COL[m],ls=ls,label=f'{source} {m}') for source,ls in [('Model','-'),('FIT','--')] for m in [0,1]],loc='upper center',bbox_to_anchor=(.5,-.15),frameon=False,ncol=2,fontsize=8)
mat=np.array([[spearmanr(model[i,np.isfinite(model[i])&np.isfinite(patient[j])],patient[j,np.isfinite(model[i])&np.isfinite(patient[j])]).statistic for j in [0,1]] for i in [0,1]])
am=fig.add_subplot(gs[4]);im=am.imshow(mat,cmap='RdBu_r',vmin=-1,vmax=1);am.set(xticks=[0,1],yticks=[0,1],xticklabels=['FIT M0','FIT M1'],yticklabels=['Model M0','Model M1'],title='Mean-rank correlation')
for i in range(2):
 for j in range(2):am.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if abs(mat[i,j])>.65 else '.15',fontsize=13)
fig.colorbar(im,ax=am,fraction=.046,pad=.04,label='Spearman rho')
fig.suptitle(f'All {len(mu)} primary OU events | one topology, three noise replays | no refit of labels or mode count',fontsize=13,y=.985)
fig.text(.5,.06,'Gray cells: nonparticipation. All supported and unsupported primary events retained. Counts are descriptive pooling, not independent networks.\nDashed profiles use previously opened patient FIT only. Correlations summarize mean rank; timing, participation and full-distribution recovery remain separate.',ha='center',fontsize=9)
scl=[i for i,n in enumerate(names) if n.startswith('SCL')]
missing_model=int(np.sum(~np.isfinite(mu[labels==1][:,scl]).any(1)))
missing_patient=int(np.sum(~np.isfinite(EV.fit[EV.fit_labels==1][:,scl]).any(1)))
fig.text(.5,.012,f'M1 without SCL: model {missing_model}/{sum(labels==1)} ({missing_model/sum(labels==1):.1%}); patient FIT {missing_patient}/{sum(EV.fit_labels==1):,} ({missing_patient/sum(EV.fit_labels==1):.1%}).',ha='center',fontsize=9,color='#8f342e')
save(fig,'all_events_patient_comparison')

manifest=dict(scientific_status='DIAGNOSTIC_REVIEW_PATIENT_PROPAGATION_NOT_ESTABLISHED',sources=inputs,producer=str(Path(__file__).resolve()),producer_sha256=rt.sha(__file__),selected=selected,selection='First OU replay by fixed job id; per frozen label choose smallest squared distance to that replay model-mode feature mean; no patient score or support filtering',frame_rule=frame_records,channel_order=ordered,rank_painter='scripts/paper_figures/plot_fig1_interictal_hfo_temporal_scaffold.py::_draw_fig1e_cluster_row',label_contract='Frozen FIT labels retained as M0/M1; no new KMeans or TA/TB renaming',n_ou_events=len(mu),mode_counts=np.bincount(labels,minlength=2),model_patient_mean_rank_spearman=mat,model_participation=[np.isfinite(mu[labels==m]).mean(0) for m in [0,1]],patient_participation=[np.isfinite(EV.fit[EV.fit_labels==m]).mean(0) for m in [0,1]],human_visual_review_pending=True)
manifest['files']=[dict(path=str(p),sha256=rt.sha(p)) for p in sorted(F.iterdir()) if p.suffix in ['.png','.pdf']]
rt.write(OUT/'manifest.json',manifest)
(F/'README.md').write_text(f'''### same_network_state_and_propagation.png / .pdf

四列依次展示固定双 core 场、同一 OU 重演的 M0/M1 单事件接触质心时序，以及完整 30 秒状态和全部 15 个触点的未滤波发放密度包络。示例取首个 OU 重演中最接近各自模型模式特征均值的事件，不用患者距离或观测支持选图；空心触点为不参与。
**关注点**：同一网络能出现不同标签，但质心时间不等于起始时间，图中不添加未验证的传播箭头。

### representative_native_frames.png / .pdf

上述两例的完整接触包络与原生 2 ms 活动帧，统一取接触总包络峰前后 -30/-10/+10/+30 ms。帧间及事件间共享活动色标，无空间平滑；圆圈标固定 core，触点颜色沿用 ICL 橙、SCL 青。
**关注点**：实际活动如何扩散、是否多处同时招募，不能只看标签或平均图判为完整患者传播。

### all_events_patient_comparison.png / .pdf

三个 OU 重演全部 {len(mu)} 个合格事件使用 Figure 1E 原 painter 展示 rank 热图、分布、模型/患者 FIT 均值剖面及描述性相关矩阵。保留不支持的事件和未参与灰格，使用冻结患者标签 M0/M1，不重新 KMeans，也不沿用旧 TA/TB 名称冒充对应已验证。
**关注点**：均值 rank 相关不是完整毫秒时序、参与或分布恢复；这是固定基底状态支线的诊断图，待人工科学目视检查。
''')
print(json.dumps(rt.json_safe({'output':str(F),'selected':selected,'n':len(mu),'matrix':mat})),flush=True)

"""Frozen-output timing sensitivity and additional observable distributions.

Positive time rescaling is a scoring diagnostic, never a physical simulation,
new event detection, readout calibration, or reranking of actual candidates.
"""
from pathlib import Path
import sys,json,csv,pickle,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.topic4_interictal_repaired_evaluation import rank_features
from src.topic4_joint_xy_kernel import event_kernel_features
from src.topic4_xy_direction import onset_directions,direction_histogram,direction_summary

R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1';OUT=R/'timing_capacity_diagnostics';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
EP=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl'
with EP.open('rb') as f:ev=pickle.load(f)
with (R/'training_objective_v2_1.pkl').open('rb') as f:obj=pickle.load(f)
cs=sorted(json.loads((R/'g3_scores.json').read_text())['candidates'],key=lambda c:c['score']['loss_off'])
with np.load(Path(next(iter(cs[0]['units'].values()))['worker_path']).with_suffix('.npz')) as z:names=z['contact_names'].astype(str).tolist();xy=z['contact_xy_mm'].copy()
colors=['#256a9c','#76b7dd','#975426','#d5a173'];labels=['Network 1 / noise 1','Network 1 / noise 2','Network 2 / noise 1','Network 2 / noise 2']
handles=[Line2D([],[],color='black',lw=2,label='Patient')]+[Line2D([],[],color=c,lw=2,label=l) for c,l in zip(colors,labels)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
stats=[];sweep=[];angular=[];sources=[];alltables=[];files=[]

def dist(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,**{k:None for k in ['mean','median','sd','variance','q05','q25','q75','q95']})
    q=np.quantile(a,[.05,.25,.5,.75,.95])
    return dict(n=len(a),mean=float(a.mean()),median=float(q[2]),sd=float(a.std()),variance=float(a.var()),q05=float(q[0]),q25=float(q[1]),q75=float(q[3]),q95=float(q[4]))

def record(source,stratum,key,a):stats.append(dict(source=source,stratum=stratum,observable=key,**dist(a)))
def writecsv(name,rows):
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name,text):
    for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=155,bbox_inches='tight')
    plt.close(fig);files.append((name,text))

def observations(t):
    v=onset_directions(t,xy);h=direction_histogram(v);groups={}
    for key,nn in [('upper_SCL',['SCL9','SCL8','SCL7','SCL6']),('right_ICL',['ICL1','ICL2']),('left_ICL',['ICL9','ICL10','ICL11'])]:
        a=t[:,[names.index(n) for n in nn]];groups[key]=np.array([np.median(r[np.isfinite(r)]) if np.isfinite(r).any() else np.nan for r in a])
    return dict(direction=h,view=v,coherence=v['coherence'],SCL_to_right_ms=groups['right_ICL']-groups['upper_SCL'],right_to_left_ms=groups['left_ICL']-groups['right_ICL'])

def strata(t,source):
    ll=ev.km.predict(rank_features(t));out={}
    for s,k in [('ALL',None),('TA',1),('TB',0)]:
        a=t if k is None else t[ll==k];o=observations(a);out[s]=o
        for key in ['coherence','SCL_to_right_ms','right_to_left_ms']:record(source,s,key,o[key])
        angular.append(dict(source=source,stratum=s,**direction_summary(o['view'])))
    return out

patient=strata(ev.patient,'patient_ALL');patient_span=np.nanmax(ev.patient,axis=1)-np.nanmin(ev.patient,axis=1)
scales=np.array([.3,.4,.5,.6,.7,.8,.9,1.,1.1,1.25,1.5])
for ci,c in enumerate(cs,1):
    rr=[]
    for unit,u in sorted(c['units'].items()):
        wp=Path(u['worker_path']);op=wp.parent.parent/'repaired_observation'/wp.with_suffix('.npz').name
        with np.load(op) as z:t=np.asarray(z['centroid_ms'][z['primary_event_indices']],float)
        source=c['candidate_id']+'/'+unit;rr.append(dict(unit=unit,obs=strata(t,source)))
        mask=np.isfinite(t);xrank=rank_features(t);basefeatures=event_kernel_features(t,xy,ev.groups,ev.scale)
        origin=np.nanmin(t,axis=1)[:,None];span=np.nanmax(t,axis=1)-origin[:,0]
        for scale in scales:
            tt=origin+scale*(t-origin)
            assert np.array_equal(mask,np.isfinite(tt)) and np.allclose(rank_features(tt),xrank,atol=1e-10)
            ff=event_kernel_features(tt,xy,ev.groups,ev.scale);assert np.allclose(ff['rank_space'],basefeatures['rank_space'],atol=1e-10)
            score=obj.score_network(tt)
            assert score['mode_counts']==u['mode_counts']
            if scale==1:assert np.isclose(score['loss_off'],u['score']['loss_off'],atol=1e-10)
            sweep.append(dict(candidate=ci,candidate_id=c['candidate_id'],unit=unit,time_multiplier=scale,loss=score['loss_off'],A=score['loss_off_A_component'],B=score['loss_off_B_subtraction'],global_half=.5*score['D_off']['global']/obj.normalizers['global'],modes_half=.5*score['D_off']['balanced_modes']/obj.normalizers['balanced_modes'],span_mean_ms=float(np.mean(scale*span)),span_wasserstein_ms=float(wasserstein_distance(scale*span,patient_span))))
    alltables.append(rr);print('Timing diagnostic computed C'+str(ci),flush=True)

fig,axs=plt.subplots(2,3,figsize=(15,8));fig.subplots_adjust(top=.87,bottom=.16,hspace=.45,wspace=.3)
for ri,ci in enumerate([1,2]):
    for j,(key,title) in enumerate([('loss','Frozen total loss'),('A','Mean-feature mismatch term A'),('span_wasserstein_ms','Centroid-span distribution distance (ms)')]):
        ax=axs[ri,j]
        for z,unit in enumerate(sorted(cs[ci-1]['units'])):
            d=[r for r in sweep if r['candidate']==ci and r['unit']==unit];ax.plot(scales,[r[key] for r in d],'.-',color=colors[z],lw=1)
        ax.axvline(1,color='black',ls=':',lw=1);ax.set(xlabel='Hypothetical within-event time multiplier',title=title,ylabel=f'C{ci}');ax.grid(alpha=.15)
fig.suptitle('Does the frozen objective reward different timing with exactly the same participation and ranks?\nC1: best joint candidate; C2: reference placement A',fontsize=13)
fig.legend(handles=handles[1:],loc='lower center',bbox_to_anchor=(.5,.04),ncol=4,frameon=False)
fig.text(.5,.005,'Offline transformation of saved centroids only. No network dynamics, event count, labels, or physical outputs were changed.',ha='center',fontsize=9)
save(fig,'fixed_rank_time_scale_sensitivity','固定每次事件的参与、顺序和标签，仅将相对最早质心的时差乘正系数，重算冻结损失、均值嵌入差异A及跨度分布距离。每次运行分别作图，竖线1为真实输出。**关注点**：这是评分敏感性诊断，不能把缩放后的数据称为网络产生的新结果，也不能证明物理可达性。')

for ci,rr in enumerate(alltables,1):
    fig,axs=plt.subplots(3,4,figsize=(17,9.8));fig.subplots_adjust(top=.9,bottom=.14,hspace=.5,wspace=.28)
    for ri,s in enumerate(['ALL','TA','TB']):
        dd=[patient[s]]+[r['obs'][s] for r in rr]
        for z,o in enumerate(dd):
            col='black' if z==0 else colors[z-1]
            axs[ri,0].plot(np.arange(25)*15,np.r_[o['direction'][:-1],o['direction'][0]],color=col,lw=1.8 if z==0 else 1)
            for j,key in enumerate(['coherence','SCL_to_right_ms','right_to_left_ms'],1):
                a=o[key];a=np.sort(a[np.isfinite(a)]);axs[ri,j].step(a,np.arange(1,len(a)+1)/len(a),where='post',color=col,lw=1.8 if z==0 else 1)
        axs[ri,0].set(xlim=(0,360),xticks=[0,90,180,270,360],xlabel='Early-to-late angle (deg)',ylabel=s+'\nCoherence-weighted event mass',title='Observed direction distribution')
        axs[ri,1].set(xlim=(0,1),ylim=(0,1),xlabel='Clipped adjusted R2',title='How well does a plane describe timing?')
        axs[ri,2].set(ylim=(0,1),xlabel='right ICL minus upper SCL (ms)',title='Upper SCL to right ICL delay')
        axs[ri,3].set(ylim=(0,1),xlabel='left ICL minus right ICL (ms)',title='Right ICL to left ICL delay')
        for j in [2,3]:axs[ri,j].axvline(0,color='gray',ls=':',lw=.7)
    for j in range(4):
        lim=(min(a.get_xlim()[0] for a in axs[:,j]),max(a.get_xlim()[1] for a in axs[:,j]))
        for a in axs[:,j]:a.set_xlim(lim);a.grid(alpha=.12)
    fig.suptitle(f'C{ci}: additional observed spatial and timing structure\nAll events first; mode-conditioned rows are supplementary diagnostics',fontsize=14)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=5,frameon=False)
    fig.text(.5,.006,'Direction uses centroid times, not ignition times; nonplanar/unresolved mass is retained in the coherence column and CSV.\nGroup delays use median centroid time among participating contacts; only events with both groups observed contribute. Negative delay reverses order.',ha='center',fontsize=9)
    save(fig,f'c{ci}_additional_observables',f'C{ci}补充观察方向分布、平面时间梯度拟合程度以及SCL—右ICL、右ICL—左ICL的有符号时差分布。第一行全部事件，下两行模式条件；缺失组不填零，实际有效数在CSV。**关注点**：方向概率质量按平面一致程度加权，未解释质量未被删除；这些稀疏接触点关系不是连续传播路线或速度。')

# Envelope packet is deliberately balanced; never use its pooled distribution as a natural-frequency reference.
packet=json.loads((R/'patient_time_packet/packet_manifest.json').read_text())
events=list(csv.DictReader((R/'g3_all_primary_events.csv').open()))
env={};edge=[]
for s,k in [('TA',1),('TB',0)]:
    # The frozen readable packet includes predeclared reserve replacements for unreadable primaries.
    selected=[e for e in packet['readable_events'] if e['mode']==k]
    for e in selected:
        assert e['status'] in ['complete','edge_mass_suspected']
        edge.append(dict(source='patient_packet',stratum=s,left_mass=e['left_10ms_mass_fraction'],right_mass=e['right_10ms_mass_fraction'],status=e['status']))
    env[('patient_packet',s)]=np.array([e['t10_t50_t90_ms'] for e in selected],float)
    assert len(selected)==32
    for ci,c in enumerate(cs,1):
        for unit in sorted(c['units']):
            topo,dyn=[int(v) for v in unit.replace('topo_','').split('_dyn_')]
            rows=[e for e in events if e['candidate_id']==c['candidate_id'] and int(e['topology_seed'])==topo and int(e['dynamics_seed'])==dyn and int(e['mode'])==k]
            source=c['candidate_id']+'/'+unit;env[(source,s)]=np.array([json.loads(e['t10_t50_t90_ms']) for e in rows],float)
            for e in rows:edge.append(dict(source=source,stratum=s,left_mass=float(e['left_10ms_mass_fraction']),right_mass=float(e['right_10ms_mass_fraction']),status='window_conditional_model'))
for (source,s),a in env.items():
    assert a.ndim==2 and a.shape[1]==3 and np.all(np.diff(a,axis=1)>=0)
    for key,v in [('mass_t10_ms',a[:,0]),('mass_t50_ms',a[:,1]),('mass_t90_ms',a[:,2]),('mass_width_10_90_ms',a[:,2]-a[:,0])]:record(source,s,key,v)
fig,axs=plt.subplots(2,4,figsize=(17,10));fig.subplots_adjust(left=.14,right=.985,top=.86,bottom=.13,hspace=.4,wspace=.35)
for ri,s in enumerate(['TA','TB']):
    for j,(key,title) in enumerate([('mass_t10_ms','10% mass time (ms)'),('mass_t50_ms','50% mass time (ms)'),('mass_t90_ms','90% mass time (ms)'),('mass_width_10_90_ms','10-90% mass width (ms)')]):
        ax=axs[ri,j];p=next(r for r in stats if r['source']=='patient_packet' and r['stratum']==s and r['observable']==key)
        ax.axvspan(p['q25'],p['q75'],color='gray',alpha=.2);ax.axvline(p['median'],color='black',lw=1.7)
        for ci,c in enumerate(cs,1):
            for z,unit in enumerate(sorted(c['units'])):
                d=next(r for r in stats if r['source']==c['candidate_id']+'/'+unit and r['stratum']==s and r['observable']==key);y=ci+(z-1.5)*.14
                ax.plot([d['q25'],d['q75']],[y,y],color=colors[z],lw=1.5);ax.plot(d['median'],y,'.',color=colors[z],ms=6)
        ax.set(title=title,ylim=(7.6,.4),yticks=range(1,8),yticklabels=[f'C{i}' for i in range(1,8)] if j==0 else []);ax.grid(axis='x',alpha=.15)
        if j==0:ax.set_ylabel(s,fontsize=14)
fig.suptitle('Envelope timing is different from the spread of contact centroid times\nPatient: 32 events per mode; model: every primary event, four runs shown separately',fontsize=14)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=5,frameon=False)
fig.text(.5,.006,'Black line / gray band: patient median / IQR. Colored dots / lines: model median / IQR.\nTimes are relative to the earliest participating contact centroid; mass quantiles are conditional on saved windows. Balanced packet is not a frequency reference.',ha='center',fontsize=9)
save(fig,'envelope_timing_all_candidates','用已冻结的32+32患者包与全部模型primary事件，对比累计包络质量10%、50%、90%时刻及10–90%宽度。患者黑线/灰带为中位数/IQR，模型四次运行分别显示；全部时间以最早参与接触点质心为零。**关注点**：这是不同观测信号的包络时间表征，不能与质心跨度混称一个速度；保留窗口边界质量，平衡患者包不能估计自然比例。')
writecsv('time_rescaling_score_diagnostic.csv',sweep);writecsv('additional_observable_statistics.csv',stats);writecsv('direction_summaries.csv',angular);writecsv('envelope_boundary_mass.csv',edge)
summary=[]
for ci,c in enumerate(cs,1):
    dd=[r for r in sweep if r['candidate']==ci]
    means={float(a):{k:float(np.mean([r[k] for r in dd if r['time_multiplier']==a])) for k in ['loss','A','B','span_wasserstein_ms']} for a in scales}
    best=min(means,key=lambda a:means[a]['loss']);summary.append(dict(candidate=ci,candidate_id=c['candidate_id'],best_hypothetical_scale=best,at_1=means[1.],at_best=means[best],per_run_best_scales=[min([r for r in dd if r['unit']==u],key=lambda r:r['loss'])['time_multiplier'] for u in sorted(c['units'])]))
(OUT/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'audit.json').write_text(json.dumps(dict(role='frozen_output_diagnostic_only',scoring_sensitivity_changes_physics=False,rank_mask_labels_preserved=True,original_28_losses_reproduced=True,patient_packet_n=64,patient_packet_natural_frequency_reference=False,figure_count=len(files),patient_FIT_time_scale_ms=float(ev.scale),kernel_input_dimensions=obj.maps['joint']['weights'].shape[0],kernel_output_dimensions=obj.maps['joint']['weights'].shape[1],sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EP,R/'training_objective_v2_1.pkl',R/'g3_all_primary_events.csv',R/'patient_time_packet/packet_manifest.json']}),indent=2)+'\n')
(F/'README.md').write_text('\n\n'.join('### '+n+'.png / .pdf\n\n'+d for n,d in files)+'\n')
print('DONE',OUT,flush=True)

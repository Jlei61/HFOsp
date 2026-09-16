"""Frozen-output factorial diagnostic; never produces new physical SNN runs.

Gaussian convolution changes local shape, not centroid differences. Fractional
translations compress centroid spacing, not local shape (2 ms discretization).
Patient packet is conditional morphology reference, not a frequency estimate.
"""
from pathlib import Path
import sys, json, pickle, csv, hashlib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from PIL import Image
from src.topic4_interictal_repaired_evaluation import rank_features

R = ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT = R/'local_shape_vs_recruitment_audit'
F = OUT/'figures'
F.mkdir(parents=True, exist_ok=True)
SIGMA, ALPHA = 30., .6
ARMS = ['original', 'width', 'timing', 'both']
TITLES = ['Original output', 'Broaden local envelope', 'Compress contact timing', 'Both changes']
COLORS = ['#3076a6', '#da8b24', '#7d52aa', '#168e7c']
CNAMES = ['Best joint candidate', 'Reference placement A', 'Placement B / smaller threshold shift',
          'Reference placement B', 'Placement A / longer GABA decay', 'Second joint candidate', 'Historical placement']
METRICS = {
    'contact_width_ms':'Local 10-90% width (ms)',
    'centroid_span_ms':'Between-contact centroid span (ms)',
    'overlap':'Contact-pair temporal overlap (0-1)',
    'shape_lag_ms':'Pairwise shape contribution to lag (ms)',
    'reversal':'10%-time / centroid order reversals',
    'nonoverlap':'Non-overlapping contact pairs',
}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42,
                     'axes.spines.top':False, 'axes.spines.right':False})
with (ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl').open('rb') as f:
    ev = pickle.load(f)
with (R/'training_objective_v2_1.pkl').open('rb') as f:
    objective = pickle.load(f)
cs = sorted(json.loads((R/'g3_scores.json').read_text())['candidates'], key=lambda c:c['score']['loss_off'])
packet = json.loads((R/'patient_time_packet/packet_manifest.json').read_text())
reference_worker=Path(next(iter(cs[0]['units'].values()))['worker_path']).with_suffix('.npz')
with np.load(reference_worker) as z:
    canonical_names=z['contact_names'].astype(str)
assert len(set(canonical_names))==len(canonical_names)
rows, contact_rows, scores, checks, sources, figures, clips = [], [], [], [], [], [], []
patient, exemplars = {1:[], 0:[]}, {}

def csvout(name, data):
    with (OUT/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def measures(a, time, mask):
    ix = np.flatnonzero(mask)
    total = a[ix].sum(1)
    assert np.all(total > 0)
    mu = a[ix] @ time / total
    q = np.array([time[np.minimum(np.searchsorted(np.cumsum(v)/v.sum(), [.1,.5,.9]),len(time)-1)] for v in a[ix]])
    i,j = np.triu_indices(len(ix), 1)
    overlap = np.maximum(0, np.minimum(q[i,2],q[j,2])-np.maximum(q[i,0],q[j,0]))
    union = np.maximum(q[i,2],q[j,2])-np.minimum(q[i,0],q[j,0])
    dmu, d10 = mu[j]-mu[i], q[j,0]-q[i,0]
    resolved = (np.abs(dmu)>2) & (np.abs(d10)>2)
    delta = mu-q[:,0]
    out = dict(contact_width_ms=float(np.median(q[:,2]-q[:,0])), centroid_span_ms=float(np.ptp(mu)),
               overlap=float(np.median(overlap/np.maximum(union,1e-12))),
               shape_lag_ms=float(np.median(np.abs(delta[j]-delta[i]))),
               reversal=float(np.mean(dmu[resolved]*d10[resolved]<0)) if resolved.any() else np.nan,
               nonoverlap=float(np.mean(overlap==0)), resolved_pairs=int(resolved.sum()), n_contacts=len(ix))
    return out, mu, q

def record(candidate, unit, event_id, mode, arm, a, time, mask, names):
    obs,mu,q = measures(a,time,mask)
    base = dict(candidate=candidate, unit=unit, event_id=event_id, mode='TA' if mode else 'TB', arm=arm)
    rows.append(dict(**base, **obs))
    for k,ch in enumerate(np.flatnonzero(mask)):
        contact_rows.append(dict(**base, contact=str(names[ch]), centroid_ms=float(mu[k]),
                                 t10_ms=float(q[k,0]), t50_ms=float(q[k,1]), t90_ms=float(q[k,2])))
    return obs,mu,q

def translate(a, shifts, dt):
    # Linear deposition on an ample zero-padded grid preserves area and first moment.
    x = np.arange(a.shape[1],dtype=float)
    return np.array([np.interp(x-s/dt, x, v, left=0.,right=0.) for v,s in zip(a,shifts)])

for e in packet['readable_events']:
    p = Path(e['arrays_path'])
    with np.load(p) as z:
        m=z['packed_window_mask'].astype(bool); a=z['positive_envelope_mass'][:,m].astype(float)
        time=z['time_ms'][m].astype(float); mask=z['participation_mask'].astype(bool)
        names=z['contact_names'].astype(str)
    source_names=names.copy()
    assert len(set(names))==len(names) and set(names)==set(canonical_names)
    permutation=np.array([names.tolist().index(n) for n in canonical_names])
    a=a[permutation];mask=mask[permutation];names=names[permutation]
    assert np.array_equal(names,canonical_names)
    a[~mask]=0
    _,mu,_=measures(a,time,mask)
    # Align the displayed signals themselves, not a different spectral centroid.
    time=time-mu.min()
    record(0,'patient',int(e['raw_global_event_index']),e['mode'],'patient',a,time,mask,names)
    patient[e['mode']].append(dict(a=a,time=time,mask=mask,names=names,event_id=int(e['raw_global_event_index'])))
    sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_contact_names=source_names.tolist(),
                        display_index=permutation.tolist(),display_contact_names=names.tolist()))

for ci,c in enumerate(cs,1):
    for ui,(unit,u) in enumerate(sorted(c['units'].items())):
        wp=Path(u['worker_path']); op=wp.parent.parent/'repaired_observation'/wp.with_suffix('.npz').name
        meta=json.loads(op.with_suffix('.json').read_text())
        with np.load(op) as z:
            primary=z['primary_event_indices']; times=z['centroid_ms'][primary].astype(float)
        labs=ev.km.predict(rank_features(times))
        with np.load(wp.with_suffix('.npz')) as z:
            env=z['contact_envelope'];dt=float(z['contact_envelope_dt_ms']); names=z['contact_names'].astype(str)
            xy=z['contact_xy_mm'].copy()
        assert np.array_equal(names,canonical_names)
        assert np.array_equal(xy,ev.xy) or np.allclose(xy,ev.xy)
        all_mu={arm:[] for arm in ARMS}; count={0:0,1:0}
        for local,idx in enumerate(primary):
            e=meta['events'][int(idx)];lo,hi=np.rint(np.asarray(e['window_ms'])/dt).astype(int)
            a=np.maximum(env[:,lo:hi]-np.asarray(e['local_baseline'])[:,None],0)
            mask=np.isfinite(times[local]);a[~mask]=0
            original_time=(np.arange(lo,hi)+.5)*dt
            _,mu,_=measures(a,original_time,mask);anchor=mu.min()
            pad=int(np.ceil(600/dt));a=np.pad(a,((0,0),(pad,pad)))
            time=(np.arange(a.shape[1])-pad+lo+.5)*dt-anchor
            _,mu0,q0=measures(a,time,mask)
            # Historical arrays store absolute centroids as float32 (up to 24 s).
            assert np.max(np.abs(mu-times[local,mask]))<.002, 'observer centroid mismatch beyond float32 precision'
            shift=np.zeros(len(mask));shift[mask]=(ALPHA-1)*mu0
            broad=gaussian_filter1d(a,SIGMA/dt,axis=1,mode='constant',truncate=5.)
            arms=dict(original=a,width=broad,timing=translate(a,shift,dt),both=translate(broad,shift,dt))
            for arm,aa in arms.items():
                obs,mm,qq=record(ci,unit,int(idx),int(labs[local]),arm,aa,time,mask,names)
                expected=mu0*(ALPHA if arm in ['timing','both'] else 1)
                merr=float(np.max(np.abs(mm-expected)))
                masserr=float(np.max(np.abs(aa[mask].sum(1)/a[mask].sum(1)-1)))
                assert merr<1e-6 and masserr<1e-10
                # Width quantiles can move by one bin under fractional translation.
                if arm=='timing': assert np.max(np.abs((qq[:,2]-qq[:,0])-(q0[:,2]-q0[:,0])))<=2*dt+1e-7
                full=np.full(len(mask),np.nan);full[mask]=mm;all_mu[arm].append(full)
                checks.append(dict(candidate=ci,unit=unit,event_id=int(idx),arm=arm,
                                   max_centroid_error_ms=merr,max_relative_mass_error=masserr))
            mode=int(labs[local])
            # Earliest event per label in each of the four units; no match-quality selection.
            if ci<=2 and count[mode]==0:
                exemplars[(ci,mode,ui)]=dict(arms=arms,time=time,mask=mask,names=names,event_id=int(idx),unit=unit)
            count[mode]+=1
        stored_score=objective.score_network(times)['loss_off']
        original_score=objective.score_network(np.asarray(all_mu['original']))['loss_off']
        assert abs(stored_score-original_score)<.001
        for arm,mm in all_mu.items():
            mm=np.asarray(mm)
            assert np.array_equal(ev.km.predict(rank_features(mm)),labs)
            sc=objective.score_network(mm)
            if arm in ['original','width']: assert abs(sc['loss_off']-original_score)<1e-7
            scores.append(dict(candidate=ci,unit=unit,arm=arm,loss=sc['loss_off'],stored_original_loss=stored_score,
                               A=sc['loss_off_A_component'],B=sc['loss_off_B_subtraction']))
        print(f'Computed candidate {ci}, {unit}: {len(primary)} events',flush=True)

def stats(vals):
    a=np.asarray(vals,float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,mean=np.nan,median=np.nan,sd=np.nan,variance=np.nan,q05=np.nan,q25=np.nan,q75=np.nan,q95=np.nan)
    q=np.quantile(a,[.05,.25,.5,.75,.95])
    return dict(n=len(a),mean=float(a.mean()),median=float(q[2]),sd=float(a.std()),variance=float(a.var()),
                q05=float(q[0]),q25=float(q[1]),q75=float(q[3]),q95=float(q[4]))

summary=[]
for ci in range(8):
    units=['patient'] if ci==0 else sorted(cs[ci-1]['units'])
    for unit in units:
        for arm in (['patient'] if ci==0 else ARMS):
            for mode in (['TA','TB'] if ci==0 else ['ALL','TA','TB']):
                rr=[r for r in rows if r['candidate']==ci and r['unit']==unit and r['arm']==arm and (mode=='ALL' or r['mode']==mode)]
                for key in METRICS:
                    summary.append(dict(candidate=ci,unit=unit,arm=arm,mode=mode,observable=key,**stats([r[key] for r in rr])))
csvout('event_observables.csv',rows);csvout('contact_times.csv',contact_rows)
csvout('observable_summary.csv',summary);csvout('score_components.csv',scores);csvout('transformation_checks.csv',checks)
csvout('candidate_parameters.csv',[dict(candidate=i+1,label=CNAMES[i],candidate_id=c['candidate_id'],parameters=json.dumps(c['parameters'])) for i,c in enumerate(cs)])

def save(fig,name,description):
    for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=155,bbox_inches='tight')
    plt.close(fig);figures.append((name,description))

limits={k:(1. if k in ['overlap','reversal','nonoverlap'] else 1.05*max(s['q95'] for s in summary if s['observable']==k and np.isfinite(s['q95']))) for k in METRICS}
for ci in range(1,8):
    fig,axs=plt.subplots(2,6,figsize=(19,8.5));fig.subplots_adjust(left=.12,right=.99,top=.81,bottom=.17,wspace=.36,hspace=.32)
    units=sorted(cs[ci-1]['units'])
    for ri,mode in enumerate(['TA','TB']):
        for j,key in enumerate(METRICS):
            ax=axs[ri,j]
            for ai,arm in enumerate(['patient']+ARMS):
                for ui,unit in enumerate(['patient'] if ai==0 else units):
                    s=next(s for s in summary if s['candidate']==(0 if ai==0 else ci) and s['mode']==mode and s['unit']==unit and s['arm']==arm and s['observable']==key)
                    y=ai if ai==0 else ai+(ui-1.5)*.15
                    color='black' if ai==0 else COLORS[ai-1]
                    ax.plot([s['q05'],s['q95']],[y,y],c=color,lw=.6,alpha=.65)
                    ax.plot([s['q25'],s['q75']],[y,y],c=color,lw=2.8)
                    ax.plot(s['median'],y,marker=['o','s','^','D'][ui],c=color,ms=4)
                    ax.plot(s['mean'],y,marker='|',c=color,ms=8,mew=1.3)
            ax.set(ylim=(4.6,-.5),yticks=range(5),yticklabels=(['Patient n=32']+TITLES if j==0 else []),xlabel=METRICS[key])
            if ri==0:ax.set_title(METRICS[key].split(' (')[0],fontsize=10)
            if j==0:ax.set_ylabel(mode,fontsize=13,fontweight='bold')
            ax.set_xlim(0,limits[key])
            ax.grid(axis='x',alpha=.15)
    for j in range(6):
        lim=max(axs[0,j].get_xlim()[1],axs[1,j].get_xlim()[1]);axs[0,j].set_xlim(0,lim);axs[1,j].set_xlim(0,lim)
    fig.suptitle(f'{CNAMES[ci-1]} | local envelope shape versus contact timing\nOffline output changes: Gaussian width = 30 ms; contact timing multiplier = 0.6',fontsize=15)
    fig.text(.5,.045,'Four rows within each colored arm: network 1 / noise 1, network 1 / noise 2, network 2 / noise 1, network 2 / noise 2.\nSymbol: median; vertical tick: mean; thick interval: IQR; thin interval: 5-95% of events (not confidence intervals).\nPatient packet is a conditional morphology reference. Unconditioned model statistics retain natural event frequencies in the CSV.\nThese are transformed contact readouts, not new neural trajectories. Event membership and TA/TB labels are held fixed.',ha='center',fontsize=9)
    save(fig,f'c{ci}_factorial_distributions','逐事件统计局部宽度、接触点质心跨度、时间重叠、局部形状对时差的贡献和顺序冲突，患者TA/TB各32事件；模型四次运行分别显示。点为中位数、竖线为均值、粗线为IQR、细线为5–95%事件范围。**关注点**：四组颜色代表离线操作，不是新物理参数；各条件真实参数另见candidate_parameters.csv。')

# Explicitly show the objective's invariance to convolution alongside local width.
fig,axs=plt.subplots(1,2,figsize=(12,4.8));fig.subplots_adjust(bottom=.23,top=.83,wspace=.35)
for ci in range(1,8):
    for ai,arm in enumerate(ARMS):
        values=[r['loss'] for r in scores if r['candidate']==ci and r['arm']==arm]
        x=ci+(ai-1.5)*.13
        axs[0].scatter([x]*4,values,color=COLORS[ai],s=14,alpha=.6)
        axs[0].plot(x,np.mean(values),'_',c=COLORS[ai],ms=13,mew=2)
        ww=[s['median'] for s in summary if s['candidate']==ci and s['mode']=='ALL' and s['arm']==arm and s['observable']=='contact_width_ms']
        axs[1].scatter([x]*4,ww,color=COLORS[ai],s=14,alpha=.6)
for ax,title in zip(axs,['Current distribution loss (lower is better)','Median local envelope width (ms)']):
    ax.set(xlabel='Confirmed parameter condition',xticks=range(1,8),xticklabels=[f'C{i}' for i in range(1,8)],ylabel=title);ax.grid(axis='y',alpha=.15)
fig.suptitle('Local waveform broadening is invisible to the current centroid-based objective',fontsize=14)
fig.legend([plt.Line2D([],[],color=c,lw=3) for c in COLORS],TITLES,loc='lower center',ncol=4,bbox_to_anchor=(.5,.065))
fig.text(.5,.015,'All 28 confirmation units; paired offline transformations. Loss is recomputed from transformed centroids, with frozen feature maps.',ha='center',fontsize=9)
save(fig,'loss_vs_local_width','同一批28次确认运行，比较四种离线操作的原损失和局部包络宽度。每点是一次运行，横线是该条件四次运行的损失均值。**关注点**：展宽可以改变波形和重叠而不改变当前损失；这不是物理优化成功。')

def normalized_shape(a):
    return np.divide(a,a.max(1)[:,None],out=np.zeros_like(a),where=a.max(1)[:,None]>0)

order=sorted(range(len(names)),key=lambda i:(0 if names[i].startswith('SCL') else 1,-int(''.join(filter(str.isdigit,names[i])))))
grid=np.linspace(0,20,100);gx,gy=np.meshgrid(grid,grid)
footprints=np.exp(-((gx[None]-xy[:,0,None,None])**2+(gy[None]-xy[:,1,None,None])**2)/(2*.5**2))

def display(ci,mode,ui):
    ex=exemplars[(ci,mode,ui)];pe=patient[mode][ui]
    arrays=[pe['a']]+[ex['arms'][a] for a in ARMS]
    tt=[pe['time']]+[ex['time']]*4;masks=[pe['mask']]+[ex['mask']]*4
    return ex,pe,[normalized_shape(a) for a in arrays],tt,masks

for ci in [1,2]:
    for mode in [1,0]:
        mode_name='TA' if mode else 'TB'; frames=[]
        fig,axs=plt.subplots(2,5,figsize=(15,6),gridspec_kw={'height_ratios':[1.35,1]})
        fig.subplots_adjust(left=.055,right=.94,top=.78,bottom=.17,wspace=.20,hspace=.35)
        title=fig.suptitle('',fontsize=14);maps=[];dots=[];heat=[];cursors=[]
        plot_times=np.arange(-140,261,2.)
        for j in range(5):
            ax=axs[0,j];ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)' if j==0 else '',xticks=[0,10,20],yticks=[0,10,20])
            maps.append(ax.imshow(np.zeros((100,100)),origin='lower',extent=(0,20,0,20),cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=1)))
            for shaft in ['SCL','ICL']:
                ix=[i for i,n in enumerate(names) if n.startswith(shaft)];ix=sorted(ix,key=lambda i:xy[i,0])
                ax.plot(xy[ix,0],xy[ix,1],c='white',lw=.8,alpha=.7)
            dots.append(ax.scatter(xy[:,0],xy[:,1],c=np.zeros(15),s=25,cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=1),edgecolors='white',lw=.5))
            ax.set_title((['Patient reference']+TITLES)[j],color='black' if j==0 else COLORS[j-1],fontsize=11)
            heat.append(axs[1,j].imshow(np.zeros((15,len(plot_times))),origin='upper',aspect='auto',extent=(-141,261,14.5,-.5),cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=1)))
            axs[1,j].set(xlabel='Time (ms)',yticks=range(15),yticklabels=[names[i] for i in order] if j==0 else [],xticks=[-100,0,100,200]);axs[1,j].tick_params(axis='y',labelsize=7)
            cursors.append(axs[1,j].axvline(0,color='#5be1ee',lw=1))
        cb=fig.colorbar(maps[0],cax=fig.add_axes([.96,.30,.012,.37]));cb.set_label('Local envelope / own full-window peak',fontsize=9)
        fig.text(.5,.025,'Same 20 x 20 mm SEEG layout. Halos visualize contacts only, not native neural activity. Gray dots: nonparticipating contacts.\nColor is local relative amplitude, fixed over time; absolute HFO energy and firing density are not equated.\nZero = earliest envelope-mass centroid in each original event. Patient/model examples are not matched pairs. Bottom: contact time courses in the displayed window.',ha='center',fontsize=9)
        for ui in range(4):
            ex,pe,aa,tt,masks=display(ci,mode,ui)
            for j in range(5):
                hm=np.array([np.interp(plot_times,tt[j],v,left=0,right=0) for v in aa[j]])
                heat[j].set_data(hm[order])
            for t in np.arange(-140,261,10):
                title.set_text(f'{CNAMES[ci-1]} | {mode_name} label | event {ui+1}/4 | {t:+.0f} ms\nPatient {pe["event_id"]}; model {ex["event_id"]}; network {ui//2+1} / noise {ui%2+1}')
                for j in range(5):
                    v=np.array([np.interp(t,tt[j],vv,left=0,right=0) for vv in aa[j]])
                    maps[j].set_data(np.max(footprints*v[:,None,None],axis=0))
                    dots[j].set_array(np.ma.array(v,mask=~masks[j]));dots[j].cmap.set_bad('#9b9b9b')
                    cursors[j].set_xdata([t,t])
                fig.canvas.draw()
                im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=128)
                frames.append(im)
                if t==0 and ui==0:
                    for ext in ['png','pdf']:fig.savefig(F/f'c{ci}_{mode_name.lower()}_contact_movie_keyframe.{ext}',dpi=155,bbox_inches='tight')
            clips.append(dict(candidate=ci,mode=mode_name,unit=ex['unit'],model_event_id=ex['event_id'],patient_event_id=pe['event_id'],
                              selection='first primary event of this label per unit; first four readable patient events in manifest order',
                              plotted_mass_fraction=[float(a[:,(t>=-140)&(t<=260)].sum()/a.sum()) for a,t in zip([pe['a']]+[ex['arms'][arm] for arm in ARMS],tt)],
                              animated_mass_fraction=[float(a[:,(t>=-140)&(t<=260)].sum()/a.sum()) for a,t in zip([pe['a']]+[ex['arms'][arm] for arm in ARMS],tt)]))
        name=f'c{ci}_{mode_name.lower()}_four_event_comparison'
        frames[0].save(F/f'{name}.gif',save_all=True,append_images=frames[1:],duration=110,loop=0,optimize=False)
        plt.close(fig)
        figures.append((f'c{ci}_{mode_name.lower()}_contact_movie_keyframe','多事件GIF的第一事件零时刻索引图，上排保持20×20毫米SEEG布局，下排显示指定时间范围内的接触点波形。颜色表示每个接触点包络除以该接触点自身整段峰值，各操作使用相同0–1色标；没有逐帧归一化。**关注点**：局部小光斑只表示接触点，非神经元原生场；患者与模型只按模式组织，不是事件一一配对。'))
        print('Rendered',name,flush=True)

manifest=dict(role='offline_observation_diagnostic_not_new_SNN',sigma_ms=SIGMA,timing_multiplier=ALPHA,
              width_choice='one diagnostic setting, not calibrated to patient widths',timing_choice='previous frozen-output sensitivity minimum; development diagnostic',
              patient_reference='64 frozen readable events, conditional TA/TB only; no natural frequency pooling',
              centroid_alignment='within-window positive-envelope mass centroid, same operational definition on both sides',
              membership='fixed primary event windows, participants, labels; detector is not rerun',
              shape_operation='symmetric normalized Gaussian convolution sigma30ms truncation5sigma, zero padding600ms',
              timing_operation='fractional area-conserving translation per contact; alpha0.6 centroid spacing; local width tolerance4ms',
              limitations=['positive envelope mass is not physiological onset or duration','convolution has acausal support; offline shape diagnostic only',
                           'zero padding does not recover missing patient/model tails','display divides each contact by its own peak and discards relative amplitude',
                           'mode classifier was used in training; conditional plots are not label-blind validation'],
              n_patient=64,
              display_contact_names=canonical_names.tolist(),candidate_ids=[c['candidate_id'] for c in cs],clips=clips,patient_sources=sources)
manifest['n_model']=sum(1 for r in rows if r['candidate']>0 and r['arm']=='original')
manifest['max_centroid_error_ms']=max(c['max_centroid_error_ms'] for c in checks)
manifest['max_relative_mass_error']=max(c['max_relative_mass_error'] for c in checks)
manifest['source_files']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), R/'g3_scores.json', R/'training_objective_v2_1.pkl', R/'patient_time_packet/packet_manifest.json']}
manifest['model_sources']=[]
for c in cs:
    for unit,u in sorted(c['units'].items()):
        wp=Path(u['worker_path']);op=wp.parent.parent/'repaired_observation'/wp.name
        md=json.loads(op.read_text())
        manifest['model_sources'].append(dict(candidate_id=c['candidate_id'],unit=unit,worker_path=str(wp),observation_path=str(op),
                                              source_array_sha256=md['arrays_sha256'],observation_array_sha256=md['observation_arrays_sha256']))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(F/'README.md').write_text('# 局部波形与跨接触点时间的离线诊断\n\n蓝色为原输出，橙色只展宽包络，紫色只压缩接触点时差，绿色两者同时。模型物理参数和原始输出均未改变。\n\n'+'\n\n'.join(f'### {n}.png / .pdf\n\n{d}' for n,d in figures)+'\n\n'+ '\n\n'.join(f'### c{ci}_{m}_four_event_comparison.gif\n\n连续展示四个模型事件，分别来自两张网络各两次噪声重演，每个单元取该标签最早primary事件；患者取冻结小包中该类前四事件。每个事件横向比较患者、原输出、只展宽、只缩时差、两者组合。\n\n**关注点**：时间按真实毫秒显示，未作逐帧增强；事件选择不按相似度，改造输出不代表网络产生了这些传播。' for ci in [1,2] for m in ['ta','tb'])+'\n')
print('DONE',json.dumps({k:manifest[k] for k in ['n_patient','n_model','max_centroid_error_ms','max_relative_mass_error']}),flush=True)

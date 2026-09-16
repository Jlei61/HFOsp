"""Full-output paired parameter -> observation response for the core_connectivity_v2 screen.

Every run (candidate x noise) is a sampling unit; events are samples within a run. No
condition is dropped or reordered by score, label count or patient similarity. Old-physics
outputs of the three layouts are loaded read-only as the input bridge.
"""
import sys,json,csv,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import label as component_labels
from scipy.stats import spearmanr,rankdata
from PIL import Image,ImageDraw
from scripts import run_topic4_core_connectivity_search as run
from scripts.analyze_topic4_geometry_threshold_refinement import patient_examples
from scripts.plot_topic4_all_condition_time_review import render
from src import topic4_joint_participation_mask as jm
from src.topic4_d6_natural_kmeans import normalize_event_ranks
rt=run.rt
DISPLAY=[f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)]
FAMILIES=['EE_same_core_scale','EE_core_to_out_scale','EE_out_to_out_scale','EI_same_core_scale','IE_same_core_scale','II_same_core_scale',
          'depth_A_scale','depth_B_scale','radius_A_mm','radius_B_mm','EE_core_to_out_degree_scale','EE_kernel_perp_scale','EE_kernel_parallel_scale','EE_angle_offset_deg']
PARAM_ZH={'EE_same_core_scale':'同核 E→E 权重 ×','EE_core_to_out_scale':'核→外 E→E 权重 ×','EE_out_to_out_scale':'外→外 E→E 权重 ×','EI_same_core_scale':'同核 E→I 权重 ×',
          'IE_same_core_scale':'同核 I→E 权重 ×','II_same_core_scale':'同核 I→I 权重 ×','depth_A_scale':'A核降阈值幅度 ×','depth_B_scale':'B核降阈值幅度 ×',
          'radius_A_mm':'A核半径 (mm)','radius_B_mm':'B核半径 (mm)','EE_core_to_out_degree_scale':'核→外 E→E 实际入度 ×','EE_kernel_perp_scale':'E→E 横向核尺度 ×',
          'EE_kernel_parallel_scale':'E→E 纵向核尺度 ×','EE_angle_offset_deg':'E→E 轴方向偏移 (°)'}
LAYOUT_ZH={'endpoint':'端点几何原位','up4p5':'原位上移 4.5 mm','near_upper':'靠近上部 SCL'}
MODE_COLOR={'ALL':'black','TA':'#d62728','TB':'#1f77b4'};SEED_STYLE={0:'-',1:'--'}
OBSERVABLES=[('SCL_upper_participation','SCL9/8 平均参与'),('ICL_contact_participation','ICL 各触点平均参与'),('both_rods','两杆联合参与'),
             ('pair_order_probability_mae','成对顺序概率误差'),('SCL_minus_ICL_lag_median_ms','SCL−ICL 质心时差中位 (ms)'),('local_width_median_ms','局部 10–90% 宽度中位 (ms)'),
             ('recruitment_span_median_ms','跨触点 t10 招募跨度中位 (ms)'),('n','合格事件数 N')]


def writecsv(path,rows):
    if rows:
        with Path(path).open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def avg(x):
    n=np.isfinite(x).sum(0);return np.divide(np.nansum(x,0),n,out=np.full(x.shape[1],np.nan),where=n>0)
def finite_rho(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    v=float(spearmanr(x,y).statistic);return v if np.isfinite(v) else None
def ranks(x):
    out=np.full_like(x,np.nan)
    for i,row in enumerate(x):
        ok=np.isfinite(row);out[i,ok]=rankdata(row[ok])
    return normalize_event_ranks(out)


def analysis_ids(r,a,burnin):
    return np.asarray([i for i in a['primary_event_indices'] if r['events'][i]['window_ms'][0]>=burnin and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)


def all_detected_ids(r,burnin):
    return np.asarray([i for i,e in enumerate(r['events']) if e['window_ms'][0]>=burnin and e['window_ms'][1]<=r['actual_duration_ms']],int)


def detection_layer(r,a,burnin):
    ids=all_detected_ids(r,burnin);ev=r['events']
    starts=np.asarray([ev[i]['qualifying_interval_ms'][0] if 'qualifying_interval_ms' in ev[i] else ev[i]['qualifying_start_ms'] for i in ids],float)
    gaps=np.diff(np.sort(starts)) if len(starts)>1 else np.array([])
    durs=np.asarray([ev[i]['qualifying_interval_ms'][1]-ev[i]['qualifying_interval_ms'][0] for i in ids if 'qualifying_interval_ms' in ev[i]],float)
    reasons={}
    for i in ids:
        for x in r['events'][i]['exclusion_reasons']:reasons[x]=reasons.get(x,0)+1
    return dict(n_all_detected=len(ids),n_TA_all=int((a['event_mode'][ids]==1).sum()) if len(ids) else 0,n_TB_all=int((a['event_mode'][ids]==0).sum()) if len(ids) else 0,
                detection_rate_per_s=float(len(ids)/max((r['actual_duration_ms']-burnin)/1000.,1e-9)),detection_interval_median_ms=float(np.median(gaps)) if len(gaps) else None,
                detection_interval_min_ms=float(gaps.min()) if len(gaps) else None,detection_interval_cv=float(gaps.std()/gaps.mean()) if len(gaps)>1 and gaps.mean()>0 else None,
                detection_interval_below_250ms_fraction=float((gaps<250).mean()) if len(gaps) else None,qualifying_duration_median_ms=float(np.median(durs)) if len(durs) else None,
                excluded_overlapping=reasons.get('overlapping_window',0),excluded_prolonged=reasons.get('prolonged_activity',0),excluded_insufficient=reasons.get('insufficient_estimable_centroids',0))


def load_unit(path,burnin):
    if not path.exists():return None
    r=rt.read(path)
    if r.get('status')!='COMPLETE':return None
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    assert rt.sha(path.with_suffix('.npz'))==r['arrays_sha256']
    a['contact_envelope']=a['contact_envelope'].T if a['contact_envelope'].shape[0]==15 else a['contact_envelope']
    return r,a,analysis_ids(r,a,burnin)


def measures(x,ref,names):
    scl=np.char.startswith(names,'SCL');icl=np.char.startswith(names,'ICL');upper=np.isin(names,['SCL9','SCL8'])
    n=len(x);ok=np.isfinite(x);pr=np.isfinite(ref).mean(0)
    m=dict(n=n,SCL_any=None,SCL_upper_participation=None,ICL_contact_participation=None,both_rods=None,participation_mae=None,rank_correlation=None,rank_shared_contacts=0,
           lag_joint_n=0,SCL_minus_ICL_lag_median_ms=None,SCL_minus_ICL_lag_q05_ms=None,SCL_minus_ICL_lag_q95_ms=None,mean_contacts_per_event=None)
    if not n:return m
    p=ok.mean(0);mr=avg(ranks(x));rref=avg(ranks(ref));common=np.isfinite(mr)&np.isfinite(rref);both=ok[:,scl].any(1)&ok[:,icl].any(1)
    m.update(SCL_any=float(ok[:,scl].any(1).mean()),SCL_upper_participation=float(p[upper].mean()),ICL_contact_participation=float(p[icl].mean()),both_rods=float(both.mean()),
             participation_mae=float(abs(p-pr).mean()),rank_correlation=finite_rho(mr[common],rref[common]),rank_shared_contacts=int(common.sum()),lag_joint_n=int(both.sum()),
             mean_contacts_per_event=float(ok.sum(1).mean()))
    if both.any():
        lag=np.nanmedian(x[both][:,scl],1)-np.nanmedian(x[both][:,icl],1);q=np.quantile(lag,[.05,.5,.95])
        m.update(SCL_minus_ICL_lag_median_ms=float(q[1]),SCL_minus_ICL_lag_q05_ms=float(q[0]),SCL_minus_ICL_lag_q95_ms=float(q[2]))
    return m


def pair_table(ref):
    c=ref.shape[1];out={}
    for i in range(c):
        for j in range(i+1,c):
            valid=np.isfinite(ref[:,i])&np.isfinite(ref[:,j]);delta=ref[valid,j]-ref[valid,i]
            out[(i,j)]=(int(valid.sum()),float(np.mean((delta>0)+.5*(delta==0))) if len(delta) else None)
    return out


def event_timing(r,a,i,names):
    """t10/t50/t90 from the full envelope inside the fixed window; centroid and first crossing kept apart."""
    lo,hi=r['events'][i]['window_ms'];dt=float(a['contact_envelope_dt_ms']);env=a['contact_envelope'][round(lo/dt):round(hi/dt)]
    cumulative=np.cumsum(env,axis=0);mass=cumulative[-1];part=np.isfinite(a['centroid_ms'][i])&(mass>0)
    if not part.any():return None
    q=np.stack([np.argmax(cumulative>=f*mass,axis=0)*dt for f in [.1,.5,.9]]);scl=np.char.startswith(names,'SCL');icl=np.char.startswith(names,'ICL')
    both=part[scl].any() and part[icl].any()
    row=dict(event=int(i),window_start_ms=lo,n_contacts=int(part.sum()),local_width_ms=float(np.median((q[2]-q[0])[part])),recruitment_span_ms=float(np.ptp(q[0,part])),
             centroid_span_ms=float(np.ptp(a['centroid_ms'][i,part])),t10_SCL_minus_ICL_ms=None,centroid_SCL_minus_ICL_ms=None,first_crossing_SCL_minus_ICL_ms=None)
    if both:
        row['t10_SCL_minus_ICL_ms']=float(np.median(q[0,part&scl])-np.median(q[0,part&icl]))
        row['centroid_SCL_minus_ICL_ms']=float(np.nanmedian(a['centroid_ms'][i][scl])-np.nanmedian(a['centroid_ms'][i][icl]))
        if 'recruitment_ms' in a:
            rec=a['recruitment_ms'][i];row['first_crossing_SCL_minus_ICL_ms']=float(np.nanmedian(rec[scl])-np.nanmedian(rec[icl]))
    return row


def native_diag(r,a,i,nc,no):
    lo,hi=r['events'][i]['window_ms'];tt=a['trace_time_ms'];mask=(tt>=lo)&(tt<hi)
    cc=a['trace_coreAE_spikes'][mask]+a['trace_coreBE_spikes'][mask];oo=a['trace_surroundE_spikes'][mask];total=cc+oo
    if not total.sum():return None
    end=np.searchsorted(np.cumsum(total),.1*total.sum())+1;denom=float(total[:end].sum());coremass=float(cc[:end].sum());outmass=float(oo[:end].sum())
    frames=a['sheet_activity_counts'][round(lo/2):round(hi/2)].astype(float);peak=frames[frames.sum((1,2)).argmax()];active=peak>=2
    labels,num=component_labels(active,structure=np.ones((3,3)));massc=np.bincount(labels.ravel(),weights=(peak*active).ravel())
    early_a=float(a['trace_coreAE_spikes'][mask][:end].sum());early_b=float(a['trace_coreBE_spikes'][mask][:end].sum())
    return dict(event=int(i),first_10pct_mass_core_share=coremass/denom,first_10pct_core_A_share=early_a/denom,first_10pct_core_B_share=early_b/denom,
                first_10pct_core_to_outside_per_neuron_density=(coremass/nc)/(outmass/no) if outmass>0 else None,
                peak_active_components=int(num),peak_largest_component_fraction=float(massc[1:].max(initial=0)/massc[1:].sum()) if massc[1:].sum() else None)


def core_activity(a,burnin,end):
    tt=a['trace_time_ms'];sel=(tt>=burnin)&(tt<end);sec=max(sel.sum()/1000.,1e-9);out={}
    for g in ['coreAE','coreBE','surroundE','allI']:
        n=len(a['group_'+g]);out[g+'_rate_hz']=float(a['trace_'+g+'_spikes'][sel].sum()/n/sec) if n else None
    return out


def native_gif(c,seed,r,a,ids,figdir,label):
    picks=sorted([int(i) for m in [1,0] for i in ids[a['event_mode'][ids]==m][:3]],key=lambda i:r['events'][i]['window_ms'][0])
    if not picks:return []
    vmax=max(max(float(a['sheet_activity_counts'][round(r['events'][i]['window_ms'][0]/2):round(r['events'][i]['window_ms'][1]/2)].max()) for i in picks),1)
    cmap=plt.get_cmap('inferno');frames=[];sel=[]
    fig,axes=plt.subplots(len(picks),6,figsize=(12,2*len(picks)),squeeze=False,layout='constrained')
    for row,i in enumerate(picks):
        lo,hi=r['events'][i]['window_ms'];mode='TA' if a['event_mode'][i]==1 else 'TB'
        sel.append(dict(candidate=c['id'],seed=seed,event=i,mode=mode,window_ms=[lo,hi],rule='first three eligible events per mode, chronological'))
        for col,offset in enumerate([0,40,80,120,160,200]):
            ax=axes[row,col];frame=a['sheet_activity_counts'][round((lo+offset)/2)]
            im=ax.imshow(frame,origin='lower',extent=[0,20,0,20],cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
            ax.scatter(*a['contact_xy_mm'].T,s=7,facecolors='none',edgecolors='cyan',linewidths=.5)
            for xy,rad in zip(c['centers_mm'],c['radii_mm']):ax.add_patch(Circle(xy,rad,fill=False,color='white',lw=.6))
            ax.set(xticks=[],yticks=[],title=f'{offset} ms' if row==0 else None)
            if col==0:ax.set_ylabel(f'{mode} · {i}')
        for offset in range(0,250,4):
            fr=a['sheet_activity_counts'][round((lo+offset)/2)];rgb=(cmap(np.clip(fr[::-1]/vmax,0,1))[:,:,:3]*255).astype('uint8')
            canvas=Image.new('RGB',(450,500),'white');canvas.paste(Image.fromarray(rgb).resize((400,400),Image.Resampling.NEAREST),(25,65));draw=ImageDraw.Draw(canvas)
            draw.text((15,10),f'{c["id"]} | seed {seed}',fill='black');draw.text((15,30),f'{mode} event {i} | t={lo+offset:.0f} ms | frame={offset} ms',fill='black')
            for xy,rad in zip(c['centers_mm'],c['radii_mm']):
                x,y=25+xy[0]*20,65+(20-xy[1])*20;rr=rad*20;draw.ellipse((x-rr,y-rr,x+rr,y+rr),outline='white',width=1)
            for x,y in a['contact_xy_mm']:
                px,py=25+x*20,65+(20-y)*20;draw.ellipse((px-2,py-2,px+2,py+2),outline='cyan',width=1)
            draw.text((15,478),'Native 2 ms activity; shared scale within this run',fill='black');frames.append(canvas)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.5,label='原生2ms活动神经元数 / 1mm网格');fig.suptitle(label+'：每类最早三个合格事件，无逐帧选优')
    fig.savefig(figdir/f'{c["id"]}_native_stills.png',dpi=120);plt.close(fig)
    frames[0].save(figdir/f'{c["id"]}_native_multievent.gif',save_all=True,append_images=frames[1:],duration=65,loop=0)
    return sel


def continuous_raster(c,seed,r,a,ids,figdir,label,names):
    order=[list(names).index(n) for n in DISPLAY];env=a['contact_envelope'][:,order].T
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    im=ax.imshow(env/max(env.max(),1e-20),aspect='auto',cmap='magma',extent=[0,env.shape[1]*float(a['contact_envelope_dt_ms'])/1000,14.5,-.5],vmin=0,vmax=1)
    ax.axhline(3.5,c='cyan',lw=.6);ax.set(yticks=range(15),yticklabels=DISPLAY,xlabel='时间 (s)',title=f'{label} | 噪声 {seed} | 完整读出；合格事件 {len(ids)} | {r["physical_status"]}')
    fig.colorbar(im,ax=ax,label='整段共同归一化包络');fig.savefig(figdir/f'{c["id"]}_{seed}_continuous.png',dpi=110);plt.close(fig)


def build_reference(plan):
    parent=rt.read(run.PARENT);ev=rt.load_evaluator(parent);objective=rt.load_objective(parent)
    patient=np.asarray(ev.fit);labels=np.asarray(ev.fit_labels);fit_blocks=np.asarray(ev.blocks)[ev.index['FIT']]
    maskref=jm.MaskReference(np.isfinite(patient));cal=jm.calibrate_scale(np.isfinite(patient),fit_blocks,maskref,sample_count=plan['analysis']['minimum_events_per_training_replay'],n_samples=128,seed=20260910)
    maskref.a_mask=cal['a_mask'];maskref.calibration=cal
    return ev,objective,patient,labels,maskref,cal


def main(stage='screen',out_root=None,cases=None,resolver=None,baseline_for=None,seeds=None,with_bridge=True,with_response=True):
    plan=rt.read(run.OUT/'plan.json');burnin=plan['analysis']['burnin_ms'];seeds=plan['seeds'] if seeds is None else list(seeds);topo=plan['topology_seed']
    OUT=Path(out_root) if out_root else run.OUT/'analysis';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    ev,objective,patient,plabels,maskref,cal=build_reference(plan)
    names=np.asarray(rt.load_observation_contract(rt.read(run.PARENT))['contact_names'])
    rt.write(OUT/'mask_score_calibration.json',dict(a_mask=cal['a_mask'],statistic=cal['statistic'],sample_count=cal['sample_count'],n_samples=cal['n_samples'],seed=cal['seed'],
        q05=cal['q05'],q95=cal['q95'],unique_patient_patterns=int(len(maskref.patterns)),patient_events=maskref.n_events,
        blocks_per_draw=[len(d['blocks']) for d in cal['draws']],events_available_per_draw=[d['n_events_available'] for d in cal['draws']],
        target='full patient FIT natural mask frequencies',score='L_search = 0.5*frozen L_off + 0.5*D_mask_off/a_mask; D_mask_off may be negative and is not clipped'))
    refs={};pairs={}
    for mode,label in [('ALL',None),('TA',1),('TB',0)]:
        ref=patient if label is None else patient[plabels==label];refs[mode]=measures(ref,ref,names);pairs[mode]=pair_table(ref)
        refs[mode]['contact_participation']=np.isfinite(ref).mean(0).tolist()
    rt.write(OUT/'patient_observation_reference.json',refs)
    cases=plan['candidates'] if cases is None else cases;units={};counts=[];obs=[];contacts=[];pair_rows=[];timing=[];native=[];activity=[];scores=[];selections=[];applied=[]
    def process(c,seed,unit,source):
        r,a,ids=unit;units[(c['id'],seed)]=unit
        counts.append(dict(candidate=c['id'],layout=c['layout'],source=source,changed_parameter=c['changed_parameter'],changed_value=c['changed_value'],seed=seed,
            physical_status=r['physical_status'],duration_ms=r['actual_duration_ms'],runaway_ms=r.get('runaway_early_stop_ms'),detected=r['n_detected'],primary=r['n_primary'],
            analysis_n=len(ids),n_TA=int((a['event_mode'][ids]==1).sum()),n_TB=int((a['event_mode'][ids]==0).sum()),**detection_layer(r,a,burnin)))
        act=core_activity(a,burnin,r['actual_duration_ms']);activity.append(dict(candidate=c['id'],layout=c['layout'],source=source,seed=seed,**act))
        ids_all=all_detected_ids(r,burnin)
        x_all=a['centroid_ms'][ids]
        sc=objective.score_network(x_all) if len(ids) else dict(status='INSUFFICIENT_EVENTS',loss_off=None)
        ms=jm.score_times(x_all,maskref) if len(ids) else dict(status='INSUFFICIENT_EVENTS',D_mask_off=None,D_mask_biased=None,D_mask_off_scaled=None)
        scores.append(dict(candidate=c['id'],layout=c['layout'],source=source,seed=seed,n=len(ids),loss_off=sc.get('loss_off'),D_off_global=(sc.get('D_off') or {}).get('global'),
            D_mask_off=ms['D_mask_off'],D_mask_biased=ms['D_mask_biased'],D_mask_off_scaled=ms['D_mask_off_scaled'],L_search=jm.combined_search_loss(sc.get('loss_off'),ms['D_mask_off'],maskref.a_mask),
            scorable=sc.get('loss_off') is not None and ms['D_mask_off'] is not None))
        tw=[];nd=[]
        nc=len(a['group_coreAE'])+len(a['group_coreBE']);no=len(a['group_surroundE'])
        for i in ids:
            t=event_timing(r,a,i,names)
            if t:t.update(candidate=c['id'],seed=seed,mode='TA' if a['event_mode'][i]==1 else 'TB');timing.append(t);tw.append(t)
            d=native_diag(r,a,i,nc,no)
            if d:d.update(candidate=c['id'],seed=seed,mode='TA' if a['event_mode'][i]==1 else 'TB');native.append(d);nd.append(d)
        for layer,base_ids in [('primary',ids),('all_detected',ids_all)]:
          for mode,label in [('ALL',None),('TA',1),('TB',0)]:
            sel=base_ids if label is None else base_ids[a['event_mode'][base_ids]==label];x=a['centroid_ms'][sel];ref=patient if label is None else patient[plabels==label]
            rec=dict(candidate=c['id'],layout=c['layout'],source=source,changed_parameter=c['changed_parameter'],changed_value=c['changed_value'],seed=seed,mode=mode,layer=layer,
                     parameter_value=(c['parameters'].get(c['changed_parameter']) if c['changed_parameter']!='baseline' else None),**measures(x,ref,names),**act)
            if layer=='all_detected':
                rec.update(pair_order_probability_mae=None,pair_order_supported_pairs=0);obs.append(rec)
                prob=np.isfinite(x).mean(0) if len(x) else np.full(15,np.nan)
                for name,p,pp in zip(names,prob,refs[mode]['contact_participation']):
                    contacts.append(dict(candidate=c['id'],layout=c['layout'],source=source,seed=seed,mode=mode,layer=layer,contact=name,n=len(x),participation=None if not np.isfinite(p) else float(p),patient_participation=float(pp)))
                continue
            ts=[t for t in tw if label is None or t['mode']==mode];rec['local_width_median_ms']=float(np.median([t['local_width_ms'] for t in ts])) if ts else None
            rec['recruitment_span_median_ms']=float(np.median([t['recruitment_span_ms'] for t in ts])) if ts else None
            t10=[t['t10_SCL_minus_ICL_ms'] for t in ts if t['t10_SCL_minus_ICL_ms'] is not None];rec['t10_SCL_minus_ICL_median_ms']=float(np.median(t10)) if t10 else None
            fc=[t['first_crossing_SCL_minus_ICL_ms'] for t in ts if t['first_crossing_SCL_minus_ICL_ms'] is not None];rec['first_crossing_SCL_minus_ICL_median_ms']=float(np.median(fc)) if fc else None
            ns=[d for d in nd if label is None or d['mode']==mode];rec['first_10pct_core_share_median']=float(np.median([d['first_10pct_mass_core_share'] for d in ns])) if ns else None
            errs=[]
            for (i,j),(pn,pp) in pairs[mode].items():
                valid=np.isfinite(x[:,i])&np.isfinite(x[:,j]);delta=x[valid,j]-x[valid,i]
                prob=float(np.mean((delta>0)+.5*(delta==0))) if len(delta) else None;err=abs(prob-pp) if prob is not None and pp is not None else None
                if err is not None:errs.append(err)
                pair_rows.append(dict(candidate=c['id'],seed=seed,mode=mode,contact_i=names[i],contact_j=names[j],model_joint_n=int(valid.sum()),patient_joint_n=pn,model_i_precedes_j=prob,patient_i_precedes_j=pp,absolute_probability_difference=err))
            rec.update(pair_order_probability_mae=float(np.mean(errs)) if errs else None,pair_order_supported_pairs=len(errs));obs.append(rec)
            prob=np.isfinite(x).mean(0) if len(x) else np.full(15,np.nan)
            for name,p,pp in zip(names,prob,refs[mode]['contact_participation']):
                contacts.append(dict(candidate=c['id'],layout=c['layout'],source=source,seed=seed,mode=mode,layer=layer,contact=name,n=len(x),participation=None if not np.isfinite(p) else float(p),patient_participation=float(pp)))
    for c in cases:
        ap=None
        for seed in seeds:
            path=resolver(c,seed) if resolver else run.result_path(stage,c['id'],topo,seed);unit=load_unit(path,burnin)
            if unit is None:counts.append(dict(candidate=c['id'],layout=c['layout'],source='v2',changed_parameter=c['changed_parameter'],changed_value=c['changed_value'],seed=seed,physical_status='MISSING'));continue
            process(c,seed,unit,'v2');r,a,ids=unit
            if ap is None:
                ap=rt.read(path.parent.parent/'applied_physics.json');th=ap['threshold'];g=ap['graph'];bl=g['stage_audits']['weights']['blocks']
                row=dict(candidate=c['id'],layout=c['layout'],changed_parameter=c['changed_parameter'],changed_value=c['changed_value'],members_A=th['members'][0],members_B=th['members'][1],
                    n_lowered=th['n_lowered'],total_lowering_mV=th['total_lowering_mV'],lowering_A_mV=th['total_lowering_per_core_mV'][0],lowering_B_mV=th['total_lowering_per_core_mV'][1],
                    floor_clipped=th['floor_clipped_count'],min_vtheta_mV=th['min_vtheta_mV'],I_members_A=ap['I_core_members'][0],I_members_B=ap['I_core_members'][1],
                    adjacency_changes=g['adjacency_changes'],stages='+'.join(g['stages']),max_delay_ms=g['max_delay_steps']*0.1,
                    **{f'{k}_edges':bl[k]['n_edges'] for k in bl},**{f'{k}_weight':bl[k]['weight_after'] for k in bl})
                if 'degree' in g['stage_audits']:
                    d=g['stage_audits']['degree'];row.update(degree_edges_added=d['n_edges_added'],degree_edges_removed=d['n_edges_removed'],degree_block_before=d['baseline_block_edges'],degree_block_after=d['final_block_edges'],degree_zero_baseline_targets=d['n_zero_baseline_targets'],degree_shortfall_targets=d['shortfall_targets'])
                if 'kernel' in g['stage_audits']:
                    k=g['stage_audits']['kernel'];row.update(kernel_l_par_mm=k['kernel']['l_par'],kernel_l_perp_mm=k['kernel']['l_perp'],kernel_theta_deg=k['kernel']['theta_deg'],kernel_spread_along_mm=k['partner_spread_along_axis_mm'],kernel_spread_across_mm=k['partner_spread_across_axis_mm'],kernel_mean_delay_ms=k['mean_delay_ms'])
                applied.append(row)
            continuous_raster(c,seed,r,a,ids,F,c['id'],names)
            if seed==seeds[0]:selections+=native_gif(c,seed,r,a,ids,F,c['id'])
    bridge_ids=dict(zip([l['id'] for l in plan['layouts']],[l['old_id'] for l in plan['layouts']]))
    bridge_cases=[]
    for layout,old in (bridge_ids.items() if with_bridge else []):
        c=dict(next(x for x in cases if x['id']==f'{layout}__baseline'));c=dict(c,id=f'{layout}__legacy_input',changed_parameter='legacy_input',changed_value=None,stage='bridge',source='v1_full_poisson')
        bridge_cases.append(c)
        for seed in seeds:
            unit=load_unit(Path(plan['historical_bridge']['root'])/'formal/units'/old/str(seed)/'workers/trajectory.json',burnin)
            if unit:process(c,seed,unit,'v1_full_poisson')
    writecsv(OUT/'per_run_counts.csv',counts);writecsv(OUT/'run_mode_observations.csv',obs);writecsv(OUT/'contact_participation.csv',contacts);writecsv(OUT/'contact_pair_order_probabilities.csv',pair_rows)
    writecsv(OUT/'per_event_timing.csv',timing);writecsv(OUT/'per_event_native_diagnostics.csv',native);writecsv(OUT/'core_activity.csv',activity);writecsv(OUT/'scores.csv',scores);writecsv(OUT/'applied_parameters.csv',applied)
    rt.write(OUT/'native_selection.json',selections)
    effects=[]
    for x in obs:
        if x['source']!='v2':continue
        bid=baseline_for(x) if baseline_for else f"{x['layout']}__baseline"
        b=next((v for v in obs if v['candidate']==bid and v['seed']==x['seed'] and v['mode']==x['mode'] and v['layer']==x['layer']),None)
        if b is None:continue
        for key,_ in OBSERVABLES:
            v0,v1=b.get(key),x.get(key)
            effects.append(dict(candidate=x['candidate'],layout=x['layout'],changed_parameter=x['changed_parameter'],changed_value=x['changed_value'],seed=x['seed'],mode=x['mode'],layer=x['layer'],observable=key,
                baseline_value=v0,value=v1,change=None if v0 is None or v1 is None else v1-v0,baseline_n=b['n'],n=x['n']))
    writecsv(OUT/'paired_parameter_effects.csv',effects)
    if with_response:
        figures_response(plan,[o for o in obs if o['layer']=='primary'],refs,F,'primary');figures_response(plan,[o for o in obs if o['layer']=='all_detected'],refs,F,'all_detected')
    figures_heatmaps(plan,contacts,obs,refs,F,cases,seeds);figures_scores(plan,scores,activity,obs,F,cases,seeds)
    if with_bridge:figures_bridge(plan,obs,counts,activity,F)
    summary_markdown(plan,obs,counts,effects,scores,applied,refs,OUT)
    for layout in bridge_ids:
        lc=[c for c in cases if c['layout']==layout]
        if lc:render(units,lc,seeds,patient_examples(),OUT/f'time_review_{layout}',lambda c:c['id'])
    readme(F)
    rt.write(OUT/'response_summary.json',dict(status='OBSERVATIONS_COMPLETE_PENDING_SCIENTIFIC_REVIEW',stage=stage,runs=sum(1 for x in counts if x['physical_status']!='MISSING' and x['source']=='v2'),
        missing=[dict(candidate=x['candidate'],seed=x['seed']) for x in counts if x['physical_status']=='MISSING'],runaways=[dict(candidate=x['candidate'],seed=x['seed'],ms=x['runaway_ms']) for x in counts if x.get('physical_status')=='RUNAWAY'],
        scorable_runs=sum(1 for s in scores if s['scorable'] and s['source']=='v2'),a_mask=maskref.a_mask,patient_reference=refs,user_accepted=False))
    return OUT


def fmt(v,digits=3):
    return '缺值' if v is None or (isinstance(v,float) and not np.isfinite(v)) else (f'{v:.{digits}f}' if isinstance(v,float) else str(v))


def summary_markdown(plan,obs,counts,effects,scores,applied,refs,OUT):
    """Descriptive tables only: what each parameter changed, per layout, both noises, ALL/TA/TB."""
    obs=[o for o in obs if o['layer']=='primary'];effects=[e for e in effects if e['layer']=='primary'];seeds=plan['seeds'];lines=['# 参数 → 观测：自动汇总（描述性，不含机制结论）','',
        f"患者 FIT 参考（不分模式）：SCL9/8 参与 {refs['ALL']['SCL_upper_participation']:.3f}，ICL 参与 {refs['ALL']['ICL_contact_participation']:.3f}，两杆联合 {refs['ALL']['both_rods']:.3f}，SCL−ICL 质心时差中位 {refs['ALL']['SCL_minus_ICL_lag_median_ms']:.2f} ms；TA/TB 事件数 {refs['TA']['n']}/{refs['TB']['n']}。",'',
        '## 每条运行的事件支持与物理状态','','|条件|噪声|物理状态|时长(ms)|原始检测|合格N|TA|TB|','|---|---:|---|---:|---:|---:|---:|---:|']
    for x in counts:
        if x['physical_status']=='MISSING':lines.append(f"|{x['candidate']}|{x['seed']}|缺失|||||");continue
        lines.append(f"|{x['candidate']}|{x['seed']}|{x['physical_status']}|{x['duration_ms']:.0f}|{x['detected']}（间隔中位 {fmt(x['detection_interval_median_ms'],0)} ms，{x['n_TA_all']}/{x['n_TB_all']}）|{x['analysis_n']}|{x['n_TA']}|{x['n_TB']}|")
    lines+=['','## 相对各布局新版基线的配对变化（两条噪声分别列出；同号且都超过阈值的记 ✔）','']
    keys=[('SCL_upper_participation',.05),('ICL_contact_participation',.05),('both_rods',.05),('pair_order_probability_mae',.02),('SCL_minus_ICL_lag_median_ms',2.),('n',4)]
    for layout in [l['id'] for l in plan['layouts']]:
        lines+=[f'### {LAYOUT_ZH[layout]}','']
        base={(o['seed'],o['mode']):o for o in obs if o['candidate']==f'{layout}__baseline'}
        lines.append('基线：'+'；'.join(f"噪声{s} {m}: N={base[(s,m)]['n']}, SCL9/8={fmt(base[(s,m)]['SCL_upper_participation'])}, ICL={fmt(base[(s,m)]['ICL_contact_participation'])}, 两杆={fmt(base[(s,m)]['both_rods'])}, 时差={fmt(base[(s,m)]['SCL_minus_ICL_lag_median_ms'],2)}" for s in seeds for m in ['ALL','TA','TB'] if (s,m) in base))
        lines+=['','|参数|值|模式|'+'|'.join(k for k,_ in keys)+'|','|---|---:|---|'+'|'.join('---' for _ in keys)+'|']
        for fam in FAMILIES:
            for value in sorted({e['changed_value'] for e in effects if e['layout']==layout and e['changed_parameter']==fam}):
                for mode in ['ALL','TA','TB']:
                    cells=[]
                    for key,thr in keys:
                        ch=[next((e['change'] for e in effects if e['layout']==layout and e['changed_parameter']==fam and e['changed_value']==value and e['seed']==s and e['mode']==mode and e['observable']==key),None) for s in seeds]
                        ok=all(c is not None for c in ch) and np.sign(ch[0])==np.sign(ch[1]) and min(abs(ch[0]),abs(ch[1]))>=thr
                        cells.append('/'.join(fmt(c,2) for c in ch)+(' ✔' if ok else ''))
                    lines.append(f'|{PARAM_ZH[fam]}|{value:g}|{mode}|'+'|'.join(cells)+'|')
        lines.append('')
    lines+=['## 分数（辅助排序，不是验收）','','|条件|噪声|N|L_off|D_mask_off/a_mask|L_search|','|---|---:|---:|---:|---:|---:|']
    for s in scores:
        if s['source']!='v2':continue
        lines.append(f"|{s['candidate']}|{s['seed']}|{s['n']}|{fmt(s['loss_off'])}|{fmt(s['D_mask_off_scaled'])}|{fmt(s['L_search'])}|")
    lines+=['','## 实际应用量（每条件）','','|条件|A/B成员|降阈值总量(mV)|截断数|邻接改变|阶段|核→外块边数|','|---|---|---:|---:|---|---|---:|']
    for a in applied:
        lines.append(f"|{a['candidate']}|{a['members_A']}/{a['members_B']}|{a['total_lowering_mV']:.1f}|{a['floor_clipped']}|{a['adjacency_changes']}|{a['stages']}|{a.get('EE_core_to_out_scale_edges','')}|")
    lines+=['','阈值：参与类观测 0.05、成对顺序误差 0.02、时差 2 ms、事件数 4；✔ 只表示两条噪声同向且都超过阈值，不是显著性检验，也不是患者恢复。','']
    (OUT/'auto_summary.md').write_text('\n'.join(lines))


def stage_candidates(stage):
    out=[]
    for p in sorted((run.OUT/'candidates').glob('*.json')):
        c=rt.read(p)
        if c.get('stage')==stage:out.append(c)
    return out


def analyze_adaptive():
    cases=stage_candidates('adaptive')
    if not cases:return None
    plan=rt.read(run.OUT/'plan.json')
    # parents come from the screen; paired effects are computed against the layout's new baseline like the screen
    out=main('adaptive',out_root=run.OUT/'analysis_adaptive',cases=cases,with_bridge=False,with_response=False)
    figures_de(plan,out);return out


def figures_de(plan,out):
    import csv
    root=run.OUT/'adaptive';hist=[]
    for p in sorted(root.glob('batch*_selection.json')):hist=rt.read(p)
    if not hist:return
    def table(path):
        with Path(path).open() as f:return list(csv.DictReader(f))
    obs=[o for o in table(run.OUT/'analysis/run_mode_observations.csv')+table(out/'run_mode_observations.csv') if o.get('layer','primary')=='primary']
    scores=table(run.OUT/'analysis/scores.csv')+table(out/'scores.csv')
    keys=[('SCL_upper_participation','ALL','SCL9/8 参与'),('both_rods','ALL','两杆联合'),('n','TA','TA 事件数'),('n','TB','TB 事件数'),('SCL_minus_ICL_lag_median_ms','ALL','SCL−ICL 时差中位 (ms)')]
    fig,axes=plt.subplots(1,len(keys)+1,figsize=(3.4*(len(keys)+1),4.2),layout='constrained')
    for ax,(key,mode,title) in zip(axes,keys):
        for i,h in enumerate(hist):
            for si,seed in enumerate(plan['seeds']):
                p=next((o for o in obs if o['candidate']==h['parent'] and o['seed']==str(seed) and o['mode']==mode),None);c=next((o for o in obs if o['candidate']==h['child'] and o['seed']==str(seed) and o['mode']==mode),None)
                ys=[_value(p,key) if p else np.nan,_value(c,key) if c else np.nan];ys=[float(y) if y not in (None,'') else np.nan for y in ys]
                ax.plot([i-.12+.24*si]*2,ys,marker='o',ms=3,ls=SEED_STYLE[si],color='C0' if h['layout']==plan['layouts'][0]['id'] else 'C1')
                ax.plot([i-.12+.24*si],[ys[1]],marker='s',ms=4,color='C0' if h['layout']==plan['layouts'][0]['id'] else 'C1')
        ax.set(title=f'{title} ({mode})',xticks=range(len(hist)),xticklabels=[f"b{h['batch']} t{h['target']}" for h in hist]);ax.tick_params(axis='x',rotation=90,labelsize=6)
    ax=axes[-1]
    for i,h in enumerate(hist):
        for si,seed in enumerate(plan['seeds']):
            p=next((x for x in scores if x['candidate']==h['parent'] and x['seed']==str(seed)),None);c=next((x for x in scores if x['candidate']==h['child'] and x['seed']==str(seed)),None)
            ys=[float(p['L_search']) if p and p['L_search'] not in ('','None') else np.nan,float(c['L_search']) if c and c['L_search'] not in ('','None') else np.nan]
            ax.plot([i-.12+.24*si]*2,ys,marker='o',ms=3,ls=SEED_STYLE[si],color='C0' if h['layout']==plan['layouts'][0]['id'] else 'C1');ax.plot([i-.12+.24*si],[ys[1]],marker='s',ms=4,color='C0' if h['layout']==plan['layouts'][0]['id'] else 'C1')
    ax.set(title='L_search（圆=父代，方=后代）',xticks=range(len(hist)),xticklabels=[f"b{h['batch']} t{h['target']}" for h in hist]);ax.tick_params(axis='x',rotation=90,labelsize=6)
    fig.suptitle('联合阶段：每个后代与其目标父代的配对观测（两条噪声实线/虚线；不可评分留空）')
    for ext in ['png','pdf']:fig.savefig(out/'figures'/f'de_parent_child_pairs.{ext}',dpi=150)
    plt.close(fig)


def analyze_confirmation():
    root=run.OUT/'confirmation'
    if not (root/'nomination.json').exists():return None
    nom=rt.read(root/'nomination.json');freeze=rt.read(root/'seed_freeze.json');plan=rt.read(run.OUT/'plan.json')
    lookup={c['id']:c for c in plan['candidates']};lookup.update({c['id']:c for c in stage_candidates('adaptive')})
    cases=[]
    for cid in nom['conditions']:
        for t in freeze['topology_seeds']:
            base=lookup[cid];cases.append(dict(base,id=f'{cid}@{t}',base_id=cid,topology=int(t),stage='confirmation',changed_parameter='confirmation',changed_value=None))
    resolver=lambda c,seed:run.result_path('confirmation',c['base_id'],c['topology'],seed)
    baseline_for=lambda x:f"endpoint__baseline@{x['candidate'].split('@')[1]}"
    return main('confirmation',out_root=run.OUT/'analysis_confirmation',cases=cases,resolver=resolver,baseline_for=baseline_for,seeds=freeze['dynamics_seeds'],with_bridge=False,with_response=False)


def _value(o,key):
    v=o.get(key);return np.nan if v is None else v


def figures_response(plan,obs,refs,F,layer='primary'):
    layouts=[l['id'] for l in plan['layouts']];seeds=plan['seeds'];base=plan['baseline_parameters']
    OBSERVABLES=[(k,t) for k,t in globals()['OBSERVABLES'] if layer=='primary' or k in ('SCL_upper_participation','ICL_contact_participation','both_rods','SCL_minus_ICL_lag_median_ms','n')]
    if layer!='primary':OBSERVABLES=[(k,'全部检测事件数 N' if k=='n' else t) for k,t in OBSERVABLES]
    for fam in FAMILIES:
        fig,axes=plt.subplots(len(OBSERVABLES),len(layouts),figsize=(4.2*len(layouts),2.3*len(OBSERVABLES)),squeeze=False,layout='constrained')
        for col,layout in enumerate(layouts):
            rows=[o for o in obs if o['source']=='v2' and o['layout']==layout and (o['changed_parameter']==fam or o['changed_parameter']=='baseline')]
            xs=sorted({base[fam] if o['changed_parameter']=='baseline' else float(o['changed_value']) for o in rows})
            for ri,(key,title) in enumerate(OBSERVABLES):
                ax=axes[ri,col]
                for mode in ['ALL','TA','TB']:
                    for si,seed in enumerate(seeds):
                        pts=[]
                        for xv in xs:
                            o=next((o for o in rows if o['seed']==seed and o['mode']==mode and ((o['changed_parameter']=='baseline' and xv==base[fam]) or (o['changed_parameter']==fam and float(o['changed_value'])==xv))),None)
                            pts.append(np.nan if o is None else _value(o,key))
                        ax.plot(xs,pts,marker='o',ms=4,ls=SEED_STYLE[si],color=MODE_COLOR[mode],label=f'{mode} · 噪声 {seed}' if ri==0 and col==0 else None)
                        if key!='n':
                            for xv,y in zip(xs,pts):
                                o=next((o for o in rows if o['seed']==seed and o['mode']==mode and ((o['changed_parameter']=='baseline' and xv==base[fam]) or (o['changed_parameter']==fam and float(o['changed_value'])==xv))),None)
                                if o is not None and np.isfinite(y):ax.annotate(f"n={o['n']}",(xv,y),xytext=(2,4 if si==0 else -9),textcoords='offset points',fontsize=5,color=MODE_COLOR[mode])
                    if key in refs[mode] and refs[mode][key] is not None and key!='n':ax.axhline(refs[mode][key],color=MODE_COLOR[mode],ls=':',lw=.8,alpha=.7)
                ax.axvline(base[fam],color='gray',lw=.5)
                ax.set(xticks=xs,xticklabels=[f'{v:g}' for v in xs]);ax.tick_params(labelsize=7)
                if ri==0:ax.set_title(LAYOUT_ZH[layout],fontsize=9)
                if col==0:ax.set_ylabel(title,fontsize=7)
                if ri==len(OBSERVABLES)-1:ax.set_xlabel(PARAM_ZH[fam],fontsize=8)
                if key in ('SCL_upper_participation','ICL_contact_participation','both_rods'):ax.set_ylim(-.05,1.1)
        axes[0,0].legend(fontsize=6,ncol=2)
        adjacency='密度/范围/方向干预改变实际邻接（配对拓扑随机身份）' if fam in ('EE_core_to_out_degree_scale','EE_kernel_perp_scale','EE_kernel_parallel_scale','EE_angle_offset_deg') else '权重/阈值/半径干预保持缓存邻接与时延'
        layer_note='合格事件（冻结孤立250 ms窗规则，与患者表同规则）' if layer=='primary' else '全部检测（含重叠窗/延长活动；与患者孤立窗表不是同一规则，只作描述）'
        fig.suptitle(f'{PARAM_ZH[fam]} → 传播观测｜{layer_note}（同一拓扑 2511，配对噪声实线/虚线；黑=不分模式，红=TA，蓝=TB；点线=患者 FIT 参考；n=实际事件数）\n{adjacency}；缺值=该条件该模式无事件，不填零',fontsize=9)
        suffix='' if layer=='primary' else '_alldetected'
        for ext in ['png','pdf']:fig.savefig(F/f'response_{fam}{suffix}.{ext}',dpi=150)
        plt.close(fig)


def figures_heatmaps(plan,contacts,obs,refs,F,allcases=None,seeds=None):
    seeds=plan['seeds'] if seeds is None else seeds;allcases=plan['candidates'] if allcases is None else allcases
    contacts=[x for x in contacts if x.get('layer','primary')=='primary'];obs=[o for o in obs if o.get('layer','primary')=='primary']
    for layout in [l['id'] for l in plan['layouts']]:
        cases=[c for c in allcases if c['layout']==layout]
        if not cases:continue
        fig,axes=plt.subplots(1,3,figsize=(17,11),layout='constrained')
        for ax,mode in zip(axes,['ALL','TA','TB']):
            matrix=[refs[mode]['contact_participation']];labs=['患者 FIT']
            ref_order=[list(np.asarray([x['contact'] for x in contacts if x['mode']==mode][:15])).index(n) for n in DISPLAY]
            matrix=[list(np.asarray(refs[mode]['contact_participation'])[ref_order])]
            for c in cases:
                for seed in seeds:
                    rows=[x for x in contacts if x['candidate']==c['id'] and x['seed']==seed and x['mode']==mode]
                    if not rows:continue
                    byname={x['contact']:x['participation'] for x in rows};matrix.append([np.nan if byname[n] is None else byname[n] for n in DISPLAY])
                    labs.append(f"{c['changed_parameter']}={c['changed_value'] if c['changed_value'] is not None else '基线'} · {seed%100:02d} (n={rows[0]['n']})")
            cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e6e6e6');im=ax.imshow(np.asarray(matrix,float),cmap=cmap,vmin=0,vmax=1,aspect='auto')
            ax.set(xticks=range(15),xticklabels=DISPLAY,yticks=range(len(labs)),yticklabels=labs,title='不分模式' if mode=='ALL' else mode)
            ax.tick_params(axis='x',rotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=5);ax.axvline(3.5,color='white',lw=1);ax.axhline(.5,color='red',lw=.8)
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.5,label='触点参与概率');fig.suptitle(f'{LAYOUT_ZH[layout]}：全部条件逐触点参与；患者参考在首行；灰色=无可估计事件')
        for ext in ['png','pdf']:fig.savefig(F/f'contact_participation_{layout}.{ext}',dpi=160)
        plt.close(fig)


def figures_scores(plan,scores,activity,obs,F,allcases=None,seeds=None):
    seeds=plan['seeds'] if seeds is None else seeds;allcases=plan['candidates'] if allcases is None else allcases
    for layout in [l['id'] for l in plan['layouts']]:
        cases=[c for c in allcases if c['layout']==layout];xx=np.arange(len(cases))
        if not cases:continue
        fig,axes=plt.subplots(4,1,figsize=(16,11),layout='constrained',sharex=True)
        for si,seed in enumerate(seeds):
            for ax,key,title in zip(axes[:3],['loss_off','D_mask_off_scaled','L_search'],['冻结 L_off（旧分数）','联合参与掩膜项 D_mask_off / a_mask','L_search = 0.5·L_off + 0.5·D_mask_off/a_mask']):
                ys=[next((_value(s,key) for s in scores if s['candidate']==c['id'] and s['seed']==seed),np.nan) for c in cases]
                ax.plot(xx,ys,marker='o',ms=4,ls=SEED_STYLE[si],label=f'噪声 {seed}');ax.set_ylabel(title,fontsize=8);ax.axhline(0,color='gray',lw=.5)
            ax=axes[3]
            for g,color in [('coreAE_rate_hz','#dc655d'),('coreBE_rate_hz','#659aba'),('surroundE_rate_hz','gray')]:
                ys=[next((_value(s,g) for s in activity if s['candidate']==c['id'] and s['seed']==seed),np.nan) for c in cases]
                ax.plot(xx,ys,marker='o',ms=3,ls=SEED_STYLE[si],color=color,label=f'{g.replace("_rate_hz","")} 噪声 {seed}')
            ax.set_ylabel('1.5 s 后完整时段平均放电率 (Hz)',fontsize=8)
        axes[0].legend(fontsize=7);axes[3].legend(fontsize=6,ncol=3)
        axes[3].set(xticks=xx,xticklabels=[f"{c['changed_parameter']}={c['changed_value'] if c['changed_value'] is not None else '基线'}" for c in cases]);axes[3].tick_params(axis='x',rotation=90,labelsize=7)
        fig.suptitle(f'{LAYOUT_ZH[layout]}：逐运行分数与两核完整活动（N<16 的运行不可评分，留空不填零；负分数保留）')
        for ext in ['png','pdf']:fig.savefig(F/f'scores_and_core_activity_{layout}.{ext}',dpi=150)
        plt.close(fig)


def figures_bridge(plan,obs,counts,activity,F):
    obs=[o for o in obs if o.get('layer','primary')=='primary'];seeds=plan['seeds'];layouts=[l['id'] for l in plan['layouts']]
    keys=[('SCL_upper_participation','SCL9/8 平均参与'),('ICL_contact_participation','ICL 平均参与'),('both_rods','两杆联合参与'),('n','合格事件数'),('coreAE_rate_hz','左核放电率 (Hz)'),('surroundE_rate_hz','核外 E 放电率 (Hz)')]
    fig,axes=plt.subplots(3,len(keys),figsize=(3.2*len(keys),8),layout='constrained')
    for ri,mode in enumerate(['ALL','TA','TB']):
        for ci,(key,title) in enumerate(keys):
            ax=axes[ri,ci]
            for li,layout in enumerate(layouts):
                for si,seed in enumerate(seeds):
                    old=next((o for o in obs if o['candidate']==f'{layout}__legacy_input' and o['seed']==seed and o['mode']==mode),None)
                    new=next((o for o in obs if o['candidate']==f'{layout}__baseline' and o['seed']==seed and o['mode']==mode),None)
                    ys=[np.nan if old is None else _value(old,key),np.nan if new is None else _value(new,key)]
                    ax.plot([li-.15+.3*si,li-.15+.3*si],ys,color=f'C{li}',ls=SEED_STYLE[si],marker='o',ms=4)
                    ax.plot([li-.15+.3*si],[ys[1]],marker='s',color=f'C{li}',ms=5)
            ax.set(xticks=range(len(layouts)),xticklabels=[LAYOUT_ZH[l] for l in layouts],title=f'{"不分模式" if mode=="ALL" else mode} · {title}',ylim=(-.05,1.1) if key in ('SCL_upper_participation','ICL_contact_participation','both_rods') else None)
            ax.tick_params(axis='x',rotation=15,labelsize=7)
    fig.suptitle('输入基底桥：同布局、同噪声下，旧版（全场 Poisson＋核内 OU，圆点）→ 新版（核外确定期望输入，方点）\n这是输入改变的效应，不能归因于连接参数',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(F/f'input_bridge.{ext}',dpi=150)
    plt.close(fig)


def readme(F):
    entries=[]
    for f in sorted(F.iterdir()):
        if f.suffix not in ('.png','.gif','.pdf'):continue
        if f.suffix!='.pdf':
            with Image.open(f) as im:
                for frame in range(min(getattr(im,'n_frames',1),3)):im.seek(frame);im.load()
        stem=f.name
        if stem.startswith('response_'):d='一个参数的配对响应：三列为三个布局，行为参与、两杆联合、成对顺序误差、SCL−ICL质心时差、局部宽度、t10招募跨度与实际N；黑/红/蓝为不分模式/TA/TB，实线/虚线为两条配对噪声，点线为患者FIT参考，横轴写真实参数值。**关注点**：两条噪声是否同向、TA/TB是否出现取舍、改变邻接的干预（密度/范围/方向）与仅改权重的干预要分开读。'
        elif stem.startswith('contact_participation_'):d='该布局全部条件、两条噪声、三种模式的逐触点参与概率，与冻结患者FIT首行比较；灰格表示无可估计事件。**关注点**：SCL9/8是否补足、ICL是否过度招募，不能只看平均值。'
        elif stem.startswith('scores_and_core_activity_'):d='该布局逐运行的冻结L_off、联合参与掩膜项和L_search，以及不依赖事件的两核/核外完整放电率。**关注点**：分数只用于排序辅助，不可评分运行留空；核活动说明TA缺失是否等于核静默。'
        elif stem.startswith('input_bridge'):d='三布局旧版（全场Poisson）与新版（核外确定输入）基线的配对差异：参与、事件数与放电率。**关注点**：这是输入基底改变本身的效应，后续连接效应都相对新版基线解释。'
        elif stem.endswith('_native_multievent.gif') or stem.endswith('_native_stills.png'):d='同一网络原生2 ms活动的多事件动画/帧图：每类最早三个合格事件，叠加真实core与电极位置，不按患者相似度选例。**关注点**：核内外并行活动、两杆招募的空间路径与时序断裂。'
        elif stem.endswith('_continuous.png'):d='该运行完整20秒的15触点连续包络（固定SEEG顺序，整段共同归一化），包含未成为合格事件的所有活动。**关注点**：事件密度、runaway与未参与触点的背景。'
        else:d='本轮自动生成的观测图。**关注点**：见同名CSV。'
        entries.append(f'### {stem}\n\n{d}')
    (F/'README.md').write_text('\n\n'.join(entries)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='screen');ap.add_argument('--out');a=ap.parse_args()
    if a.stage=='adaptive':analyze_adaptive()
    elif a.stage=='confirmation':analyze_confirmation()
    else:main(a.stage,a.out)

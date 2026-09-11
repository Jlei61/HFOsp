#!/usr/bin/env python3
"""Read-only, all-event raw-sheet audit against Supplementary Video 1 inputs.

The two published exemplars are descriptive references, not a validation set.
Never assign simulated events to a template by their best-looking movie.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_joint_xy import observable_groups

MAIN = Path('/home/honglab/leijiaxin/HFOsp')
SEARCH = ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v3'
PATIENT = MAIN/'results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz'
VIDEO = MAIN/'results/paper-ready-figure/supplementary-video-1.gif'
META = VIDEO.with_name('supplementary-video-1_metadata.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison(a, b, tolerance_ms=2.):
    """Compare fixed contact pairs; no lag warping or closest-template selection."""
    a, b = np.asarray(a), np.asarray(b)
    common = np.isfinite(a) & np.isfinite(b)
    i, j = np.triu_indices(len(a), 1)
    ok = common[i] & common[j]
    da, db = a[j[ok]]-a[i[ok]], b[j[ok]]-b[i[ok]]
    resolved = (abs(da) > tolerance_ms) & (abs(db) > tolerance_ms)
    return {'common_contacts': int(common.sum()), 'common_pairs': len(da),
            'resolved_pairs': int(resolved.sum()),
            'order_discordance': float(np.mean(da[resolved]*db[resolved] < 0)) if resolved.any() else None,
            'pair_lag_mae_ms': float(np.mean(abs(da-db))) if len(da) else None,
            'participation_mismatch': int(np.sum(np.isfinite(a) != np.isfinite(b)))}


def relative(t):
    return t-np.nanmin(t)


def window_frames(n_frames, frame_dt, start, stop):
    """Half-open physical window, immune to float rounding at integer edges."""
    centers=(np.arange(n_frames)+.5)*frame_dt
    return np.flatnonzero((centers>=start)&(centers<stop))


def render_movie(out, z, table, obs, patient, candidate, index):
    """First chronological event in the lowest seed; selected before comparison."""
    xy=z['contact_xy_mm']; names=z['contact_names'].astype(str)
    movie=z['sheet_activity_counts']; frame_dt=float(z['sheet_activity_frame_ms'])
    env=z['contact_envelope']; dt=float(z['contact_envelope_dt_ms'])
    start, stop=obs['windows_ms'][index]; t0=float(np.nanmin(table[index]))
    frames=window_frames(len(movie),frame_dt,start,stop)
    times=(frames+.5)*frame_dt-t0
    pxy=patient['points_mm']; pn=patient['contact_order'].astype(str)
    fig, axes=plt.subplots(2,3,figsize=(13.5,8),layout='constrained')
    raw, modelpoints, trace, ta, tb, ranks=axes.ravel()
    fig.suptitle(f"Diagnostic only: {candidate['candidate_id']} | seed {int(z['topology_seed'])} | event {index+1}",fontsize=13)
    vmax=max(1,int(movie.max()))
    img=raw.imshow(movie[frames[0]],origin='lower',extent=(0,20,0,20),interpolation='nearest',vmin=0,vmax=vmax,cmap='magma')
    fig.colorbar(img,ax=raw,label='Active E neurons / 1 mm cell / 2 ms',shrink=.7)
    for center in candidate['node_field']['centers_mm']:
        raw.add_patch(Circle(center,candidate['geometry']['distance_cutoff_mm'],fill=False,color='cyan',lw=1))
    raw.scatter(*xy.T,s=10,facecolors='none',edgecolors='white',linewidths=.4)
    raw.set(title='Native sheet: no readout or spatial smoothing',xlabel='Model x (mm)',ylabel='Model y (mm)')
    evmax=max(float(env.max()),1e-12)
    points=modelpoints.scatter(*xy.T,c=env[:,frames[0]],vmin=0,vmax=evmax,cmap='magma',s=60)
    modelpoints.set(xlim=(0,20),ylim=(0,20),aspect='equal',title='Virtual contacts: original density proxy',xlabel='Model x (mm)',ylabel='Model y (mm)')
    for (x,y),name in zip(xy,names):modelpoints.text(x+.1,y+.15,name,fontsize=6)
    fig.colorbar(points,ax=modelpoints,label='Firing-density proxy',shrink=.7)
    a,b=int(round(start/dt)),int(round(stop/dt))
    heat=trace.imshow(env[:,a:b],aspect='auto',origin='lower',extent=(a*dt-t0,b*dt-t0,-.5,len(names)-.5),vmin=0,vmax=evmax,cmap='magma',interpolation='nearest')
    trace.scatter(table[index]-t0,np.arange(len(names)),s=13,c='cyan',marker='x')
    cursor=trace.axvline(times[0],c='white',ls='--',lw=1)
    trace.set(yticks=np.arange(len(names)),yticklabels=names,xlabel='Time from first model centroid (ms)',title='Entire 250 ms observed event window')
    trace.tick_params(axis='y',labelsize=7)
    ppoints=[]
    metadata=json.loads(META.read_text())
    for label,ax in zip(('TA','TB'),(ta,tb)):
        part=patient[label+'_participant']
        scale=metadata['movie_contract']['normalization_scales_robust_z'][label]
        ax.scatter(*pxy[~part].T,s=40,facecolors='none',edgecolors='.7')
        sc=ax.scatter(*pxy[part].T,c=np.zeros(part.sum()),s=65,cmap='GnBu',norm=PowerNorm(.5,0,1))
        ax.set(xlim=patient['display_xlim_mm'],ylim=patient['display_ylim_mm'],aspect='equal',title=f'Patient {label}: actual contacts, no interpolation',xlabel='Frozen shared axis (mm)',ylabel='Patient y (mm)')
        for (x,y),name in zip(pxy,pn):ax.text(x+.5,y+.5,name,fontsize=6)
        ppoints.append((label,sc,part,scale))
        fig.colorbar(sc,ax=ax,label='HFO envelope / fixed window q99',shrink=.7)
    order=[list(pn).index(n) for n in names]
    for label,color in [('TA','#b2182b'),('TB','#2166ac')]:
        t=np.where(patient[label+'_participant'],patient[label+'_fig1a_centroid_ms'],np.nan)[order]
        ranks.plot(np.arange(len(names)),relative(t),'.-',label=f'Patient {label}',c=color,ms=4)
    ranks.plot(np.arange(len(names)),relative(table[index]),'.-',c='black',label='Model',ms=4)
    ranks.set(xticks=np.arange(len(names)),xticklabels=names,ylabel='Relative centroid time (ms)',title='Fixed contact identities; unpaired exemplars')
    ranks.tick_params(axis='x',rotation=90,labelsize=7);ranks.legend(fontsize=8)
    title=raw.text(.02,.98,'',transform=raw.transAxes,va='top',c='white',fontsize=9)
    def update(i):
        time=times[i]; img.set_data(movie[frames[i]])
        points.set_array(env[:,min(int(frames[i]*frame_dt/dt),env.shape[1]-1)])
        cursor.set_xdata([time,time]);title.set_text(f'{time:+.1f} ms')
        for label,sc,part,scale in ppoints:
            pt=patient[label+'_envelope_time_from_first_centroid_ms']
            use=abs(pt-time)<=1.5
            if not use.any():raise ValueError('patient frame is outside recorded signal')
            value=np.maximum(patient[label+'_envelope_robust_z'][:,use].mean(axis=1),0)/scale
            sc.set_array(value[part])
        return []
    anim=FuncAnimation(fig,update,frames=len(frames),interval=80,blit=False)
    path=out/'figures'/'first_chronological_event_raw_comparison.gif'
    anim.save(path,writer=PillowWriter(fps=12.5),dpi=80)
    for number in (0,len(frames)//3,2*len(frames)//3,len(frames)-1):
        update(number);fig.savefig(path.with_name(f'comparison_frame_{number:03d}.png'),dpi=120)
    plt.close(fig)
    with Image.open(path) as gif:
        if gif.n_frames!=len(frames):raise RuntimeError('GIF lost frames')
        durations=[]
        for k in range(gif.n_frames):gif.seek(k);durations.append(gif.info['duration'])
        if set(durations)!={80}:raise RuntimeError('GIF time contract changed')
    return {'path':str(path),'sha256':sha(path),'n_frames':len(frames),'biological_step_ms':frame_dt,
            'frame_timestamp':'center of stored non-overlapping 2 ms bin','duration_ms_per_frame':80,
            'all_frames_decoded':True,'visual_qa':False,'selection':'lowest seed, first chronological event; no resemblance selection',
            'model_window_ms':[start,stop],'relative_frame_times_ms':times.tolist(),
            'patient_reference':'unpaired published examples; common relative ms without time rescaling',
            'raw_quantity':'number of distinct active E neurons per spatial cell per frame, not spike count or HFO',
            'native_grid_is_still_binned':True,'full_simulation_movie_retained':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',default='replicated_r005_020')
    ap.add_argument('--round',type=int,default=5);ap.add_argument('--no-gif',action='store_true');args=ap.parse_args()
    source=SEARCH/f'rounds/{args.round:03d}/combined_{args.candidate}.json'
    row=json.loads(source.read_text())['candidates'][0]
    out=SEARCH/'raw_propagation_audit'/args.candidate;out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    metadata=json.loads(META.read_text())
    if sha(VIDEO)!=metadata['sha256']:raise RuntimeError('patient GIF hash mismatch')
    with np.load(PATIENT) as z:patient={k:z[k] for k in z.files}
    plan=json.loads((ROOT/'config/topic4_joint_xy_kernel_v3.json').read_text())
    records=[];contacts=[];tables=[];native_tables=[];labels=[];hashes={str(p):sha(p) for p in [source,PATIENT,VIDEO,META,Path(__file__)]}
    preview=None
    for unit in sorted(row['units'],key=lambda u:u['seed']):
        wp=Path(unit['worker_path']);meta=json.loads(wp.read_text());npz=Path(meta['arrays']['path'])
        if sha(wp)!=unit['worker_sha256'] or sha(npz)!=meta['arrays']['sha256']:raise RuntimeError('worker changed')
        hashes[str(wp)]=sha(wp);hashes[str(npz)]=sha(npz)
        with np.load(npz) as data:
            z={k:data[k] for k in ['contact_names','contact_xy_mm','sheet_activity_counts','sheet_activity_frame_ms','source_bin_mm','contact_envelope','contact_envelope_dt_ms','topology_seed']}
        names=z['contact_names'].astype(str);pn=list(patient['contact_order'].astype(str))
        if len(set(names))!=len(names) or set(names)!=set(pn):raise RuntimeError('contact identities differ')
        order=[pn.index(n) for n in names]
        table,obs=observable_groups(z['contact_envelope'],float(z['contact_envelope_dt_ms']),**plan['observation'])
        if obs!=unit['observation']:raise RuntimeError('frozen observation differs')
        movie=z['sheet_activity_counts']; frame_dt=float(z['sheet_activity_frame_ms']);bin_mm=float(z['source_bin_mm'])
        if movie.ndim!=3 or frame_dt!=2. or bin_mm!=1.:raise RuntimeError('native grid contract differs')
        ij=np.floor(z['contact_xy_mm']/bin_mm).astype(int)
        if np.any(ij<0) or np.any(ij>=movie.shape[1]):raise RuntimeError('contact outside native sheet')
        local=movie[:,ij[:,1],ij[:,0]].astype(float).T
        native=np.full_like(table,np.nan)
        for e,(start,stop) in enumerate(obs['windows_ms']):
            tt=(np.arange(movie.shape[0])+.5)*frame_dt
            use=(tt>=start)&(tt<stop);weights=local[:,use];den=weights.sum(axis=1)
            native[e]=np.divide(weights@tt[use],den,out=np.full(len(names),np.nan),where=den>0)
            native[e,~np.isfinite(table[e])]=np.nan
            rec={'seed':unit['seed'],'event_index':e,'start_ms':start,'stop_ms':stop,'n_contacts':int(np.isfinite(table[e]).sum())}
            for prefix,reference in [('native',native[e])]+[(lab,np.where(patient[lab+'_participant'],patient[lab+'_fig1a_centroid_ms'],np.nan)[order]) for lab in ('TA','TB')]:
                rec.update({prefix+'_'+k:v for k,v in comparison(table[e],reference).items()})
            records.append(rec);labels.append(f"{unit['seed']}:{e+1}")
            for c,name in enumerate(names):contacts.append({'seed':unit['seed'],'event_index':e,'contact':name,'participant':bool(np.isfinite(table[e,c])),'readout_centroid_ms':float(table[e,c]),'native_cell_centroid_ms':float(native[e,c])})
        tables.extend(table);native_tables.extend(native)
        if preview is None and len(table):preview=(z,table,obs)
    if len(records)!=row['n_events']:raise RuntimeError('event denominator changed')
    for filename,data in [('all_event_comparisons.csv',records),('all_event_contacts.csv',contacts)]:
        with (out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    t=np.asarray(tables);n=np.asarray(native_tables)
    fig,axes=plt.subplots(1,2,figsize=(13,15),layout='constrained')
    im=axes[0].imshow(t-np.nanmin(t,axis=1)[:,None],aspect='auto',interpolation='nearest',vmin=0,vmax=250,cmap='viridis')
    fig.colorbar(im,ax=axes[0],label='Time after first contact centroid (ms)',shrink=.5)
    im=axes[1].imshow(n-t,aspect='auto',interpolation='nearest',vmin=-125,vmax=125,cmap='RdBu_r')
    fig.colorbar(im,ax=axes[1],label='Native cell minus readout centroid (ms)',shrink=.5)
    axes[0].set_title('All observed events: fixed contact order');axes[1].set_title('Readout sensitivity: same event and contact')
    for ax in axes:ax.set(xticks=np.arange(len(names)),xticklabels=names,yticks=np.arange(len(labels)),yticklabels=labels);ax.tick_params(axis='y',labelsize=6);ax.tick_params(axis='x',rotation=90,labelsize=8)
    fig.suptitle(f'{args.candidate}: {len(t)} events, {len(row["units"])} networks | NOT QUALIFIED',fontsize=13)
    fig.savefig(out/'figures/all_event_contact_timing.png',dpi=140);fig.savefig(out/'figures/all_event_contact_timing.pdf');plt.close(fig)
    animation=None if args.no_gif else render_movie(out,*preview,patient,row['candidate'],0)
    result={'status':'DIAGNOSTIC_COMPLETE_NOT_QUALIFIED','candidate_id':args.candidate,'n_events':len(records),'n_networks':len(row['units']),
            'all_observed_events_included':True,'raw_movie_resolution':{'spatial_mm':1,'temporal_ms':2},'raw_movie_not_single_neuron_spikes':True,
            'animation':animation,'source_hashes':hashes,'live_search_changed':False,'patient_heldout_evaluated':False,
            'patient_reference_confirmed_by_user':'Supplementary Video 1','new_qualification_pass':False,
            'centroid_estimators_differ':'Model positive density centroid; patient displayed primary-enhancement spectrogram centroid; not interchangeable with stored lagPatRaw.',
            'native_comparison':'Containing 1 mm cell activity centroid on same 250 ms window and readout participant mask. Sensitivity diagnostic, not a replacement detector or neuron onset.',
            'tolerance_ms':2,'metrics':records,
            'claim_boundary':'Two previously selected patient examples are not the full patient distribution or independent validation. No best-template assignment, rescaling, event deletion, or mechanism claim.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'figures/README.md').write_text('### all_event_contact_timing.png / .pdf\n全部网络、全部实际观察事件的逐触点时间和同窗原始网格敏感性；未根据方向或相似度筛选事件。左图为模型参与触点的相对质心时间，右图为对应 1 mm 网格与 readout 的质心差，白色保持缺失。\n**关注点**：逐点顺序和 readout 扭曲；不能把局部网格质心称为神经元首次放电。\n\n'+('### first_chronological_event_raw_comparison.gif\n固定选择最低 seed 的首个事件，展示完整 250 ms 窗、未经 readout 的原始网格、虚拟触点和两个患者代表事件；无空间插值或逐帧归一化。双方只共享相对毫秒，未配对、未拉伸时间，患者与模型幅度单位分别标注。\n**关注点**：全场是否出现被电极平滑掩盖的跳跃、广泛同步或多处独立激活；当前候选未通过患者分布门槛。\n\n### comparison_frame_*.png\n动画的预设等间隔抽帧，供阅读和目视 QA。最终验收仍须检查全部动画帧和全部事件。\n**关注点**：抽帧不能代替完整传播检查。\n' if animation else ''))
    print(json.dumps({'out':str(out),'n_events':len(records),'animation':animation,'native_discordance_median':float(np.median([r['native_order_discordance'] for r in records if r['native_order_discordance'] is not None]))}))


if __name__=='__main__':main()

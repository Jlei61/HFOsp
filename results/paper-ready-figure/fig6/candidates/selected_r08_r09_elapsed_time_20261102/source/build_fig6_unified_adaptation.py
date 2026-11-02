#!/usr/bin/env python3
"""Fig6 reference layout for one autonomous unified-current trajectory.

A-E use one new simulation. E fixes the existing patient example without
reselection. F is explicitly labeled as the previous model's cohort reference.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
MAIN=Path('/home/honglab/leijiaxin/HFOsp')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(MAIN/'scripts/paper_figures'),str(MAIN/'scripts')]
import analyze_topic4_unified_current_adaptation as analysis
import build_fig5_fig4f_historical_zm as native
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,PowerNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
try:import pymupdf as fitz
except ImportError:
    sys.path.insert(0,'/tmp/hfosp_fig5_pdf_tools')
    import pymupdf as fitz

ROOT=analysis.ROOT
REFERENCE=MAIN/'results/paper-ready-figure/fig6/candidates/reference_layout_20261011'
OUT=MAIN/'results/paper-ready-figure/fig6/candidates/unified_current_adaptation_20261030'
COLORS=['#267ba8','#b55b56','#d18422','#806a9e','#4e9a76','#87b946']
PHASES=[]


def write(path,data):
    native.write(path,data)


def save(fig,name):
    dest=OUT/'figures';dest.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fixed=name in ['fig6-panela','fig6-panelb']
        fig.savefig(dest/f'{name}.{ext}',dpi=220,bbox_inches=None if fixed else 'tight',
            pad_inches=0 if fixed else .08)
    plt.close(fig)


def load(name):
    folder=ROOT/'runs'/name
    a=analysis.load(folder,'chunks',['time_ms','spikes_1ms','regions_1ms','field_time_ms',
        'field_5ms','slow_time_ms','Z','M','raster'])
    a.update(dict(np.load(ROOT/'geometry.npz')))
    raw=analysis.load(folder,'dense_readout_chunks',['time_ms','contact_current'])
    a.update(lfp_native_time_ms=raw['time_ms'],lfp_native=raw['contact_current'])
    assert np.allclose(np.diff(raw['time_ms']),.1,atol=1e-8)
    return a


def samples(a,r):
    entry=r['temporal']['entries'][0]['onset_s']
    recovered_episode=next((e for e in r['episodes'] if e['both_core_Z_reference_s'] is not None),r['episodes'][0])
    episode=next((e for e in r['episodes'] if e['sustained_return']),recovered_episode)
    active_entry=max(e['onset_s'] for e in r['temporal']['entries'] if e['onset_s']<episode['exit_start_s'])
    rate=a['spikes_1ms'][:,0].reshape(-1,5).sum(1)/32000/.005
    t=(np.arange(len(rate))+.5)*.005
    def peak(lo,hi):
        idx=np.flatnonzero((t>=lo)&(t<hi));assert len(idx)
        return float(t[idx[np.argmax(rate[idx])]])
    first=r['initial_brief_events'][0]
    inter=peak(first['start_s'],first['end_s'])
    # Locate actual low-rate onset inside the observer's 2-s confirmation window.
    rr=np.c_[a['spikes_1ms'][:,0],a['regions_1ms'][:,:2]].reshape(-1,10,3).sum(1)
    rr=rr/np.r_[32000,a['region_counts'][:2]]/.01
    lo=max(active_entry+.2,episode['exit_start_s']-.5)
    quiet=np.all(rr<5.,axis=1)
    ids=np.flatnonzero(np.convolve(quiet.astype(int),np.ones(10,int),'valid')==10)
    ids=ids[(ids*.01>=lo)&(ids*.01<=episode['exit_confirmed_s'])]
    exit_t=float(ids[0]*.01) if len(ids) else episode['exit_start_s']
    high=peak(active_entry,exit_t)
    return_events=episode['returned_brief_events']
    recovered=episode['both_core_Z_reference_s']
    times=[inter,high,max(high+.02,exit_t-.05),exit_t+.05]
    labels=['Interictal','High activity','Termination','Low activity']
    ss=[dict(number=i+1,time_s=float(v),color=COLORS[i],label=labels[i]) for i,v in enumerate(times)]
    if recovered is not None:
        ss.append(dict(number=5,time_s=recovered,color=COLORS[4],label='Core Z at ref.'))
    if return_events:
        returned=peak(return_events[0]['start_s'],return_events[0]['end_s'])
        ss.append(dict(number=6,time_s=returned,color=COLORS[5],label='Returned event'))
    return ss,entry,exit_t,episode


def marks(ax,ss,onset):
    for phase in PHASES:
        ax.axvspan(phase['start_s'],phase['end_s'],
            fc='#ba263c' if phase['state']=='high' else '#547785',
            alpha=.075 if phase['state']=='high' else .045,lw=0)
    for s in ss:
        ax.axvline(s['time_s'],color=s['color'],ls='--' if s['number']==1 else ':',lw=.9,alpha=.8)
    ax.set_xlim(native.WINDOW);ax.xaxis.set_major_locator(MaxNLocator(5))


def draw_maps(a,ss):
    fig=plt.figure(figsize=(12.5,3.6));gs=fig.add_gridspec(1,4,left=.065,right=.92,bottom=.2,top=.78,wspace=.25)
    rows=[]
    ids=[2,3,4,6] if any(s['number']==6 for s in ss) else [1,2,3,4]
    selected=[next(s for s in ss if s['number']==n) for n in ids]
    for j,s in enumerate(selected):
        ax=fig.add_subplot(gs[j]);center=round(s['time_s']/.005)*.005
        start=int(round((center-.01)/.005));stop=start+4
        count=a['field_5ms'][start:stop].sum(0)
        hz=count/a['cell_e_counts']/.02
        assert count.sum()==a['spikes_1ms'][start*5:stop*5,0].sum()
        im=ax.imshow(hz.reshape(20,20),origin='lower',extent=[0,20,0,20],
            cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for k,xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.7))
            ax.text(xy[0],xy[1]+2.1,'AB'[k],color='#10656a',fontsize=12,ha='center',weight='bold',
                bbox=dict(fc='white',ec='none',pad=.2,alpha=.8))
        ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if j==0:ax.set_ylabel('y (mm)')
        else:ax.tick_params(labelleft=False)
        ax.set_title(f"{s['number']}  {s['label']}\n{center:.3f} s",color=s['color'],fontsize=15,pad=10)
        rows.append(dict(state=s['number'],time_window_s=[start*.005,stop*.005],E_rate_Hz=hz,
            all_E_rate_Hz=float(count.sum()/32000/.02),color_limits_Hz=[0,500],
            cells_above_display_max=int((hz>500).sum())))
        if j==3:
            cb=fig.colorbar(im,cax=ax.inset_axes([1.1,0,.06,1]),ticks=[0,250,500]);cb.set_label('E rate (Hz)')
    save(fig,'fig6-panelc');return rows


def draw_trajectory(a,ss,duration):
    # Retain the current Fig6 D coordinates, view and temporal color grammar.
    # Expand limits from the old onset-only crop to include every actual state.
    fig=plt.figure(figsize=(8.4,6.7));ax=fig.add_subplot(111,projection='3d');ax.computed_zorder=False
    st=a['slow_time_ms']/1000.
    rate=a['spikes_1ms'][:,0].reshape(-1,5).sum(1)/32000/.005
    t=(np.arange(len(rate))+.5)*.005
    xyz=np.c_[np.interp(t,st,a['Z'][:,0]),.0005*np.interp(t,st,a['M'][:,0]),gaussian_filter1d(rate,1)]
    seg=np.stack([xyz[:-1],xyz[1:]],axis=1)
    ax.add_collection3d(Line3DCollection(seg,cmap='viridis',norm=Normalize(0,1),
        array=(t[:-1]+t[1:])/(2*duration),linewidths=1.25,zorder=6))
    ax.set(xlim=(max(0,float(xyz[:,0].min())-.035),1.02),ylim=(0,float(xyz[:,1].max())*1.08),
        zlim=(0,max(100,float(xyz[:,2].max())*1.06)),xlabel='$Z$',ylabel=r'$\eta_M M$ (mV)',zlabel=r'$r_E$ (Hz)')
    ax.view_init(elev=26,azim=-125);ax.set_box_aspect((1.25,1,1.05))
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.labelpad=15;axis.pane.set_facecolor('#f2f5fa');axis.set_major_locator(MaxNLocator(4))
    fig.canvas.draw();boxes=[];states=[]
    for s in ss:
        pt=np.array([np.interp(s['time_s'],t,xyz[:,k]) for k in range(3)])
        ax.scatter(*pt,c=s['color'],s=45,edgecolor='white',depthshade=False,zorder=10)
        x,y,_=proj3d.proj_transform(*pt,ax.get_proj());anchor=ax.transData.transform((x,y));unit=fig.dpi/72
        for off in [(15,15),(-24,15),(22,-24),(-24,-24),(0,43),(44,0),(-44,0),(0,-44),(45,35),(-45,35)]:
            center=anchor+np.array(off)*unit
            box=(center[0]-11*unit,center[1]-11*unit,center[0]+11*unit,center[1]+11*unit)
            if all(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3] for b in boxes):
                boxes.append(box);break
        ax.annotate(str(s['number']),xy=(x,y),xytext=off,textcoords='offset points',ha='center',va='center',
            fontsize=14,weight='bold',bbox=dict(boxstyle='circle,pad=.2',fc='white',ec='#777'),
            arrowprops=dict(arrowstyle='-',color='#777',lw=.6),zorder=20)
        states.append(dict(**s,Z_etaM_rate=pt))
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),ax=ax,shrink=.65,pad=.12)
    cb.set_label('Normalized time');cb.set_ticks([0,.25,.5,.75,1])
    fig.legend(handles=[Line2D([],[],marker='o',ls='none',color=s['color'],
        label=f"{s['number']}  {s['label']}") for s in ss],loc='lower center',
        bbox_to_anchor=(.48,-.025),ncol=3,frameon=False,fontsize=11.5,
        handletextpad=.2,columnspacing=1.0)
    save(fig,'fig6-paneld')
    np.savez_compressed(OUT/'trajectory_data.npz',time_s=t,Z_etaM_rate=xyz)
    return dict(coordinates=['mean Z','eta_M mean M (mV)','all-E rate (Hz)'],time_s=[0,duration],
        rate_bin_ms=5,display_smoothing_sigma_ms=5,normalized_time='t / full duration',
        clipping=False,states=states,eta_M=.0005,
        interpretation='Native trajectory projection; neither a bifurcation diagram nor proof of a limit cycle.')


def onset_comparison(a,entry):
    # Fix the existing SZ26 example instead of optimizing the match to this run.
    energy=native.energy
    previous=json.loads((REFERENCE/'source/field_comparison.json').read_text())
    # The old patient_cases helper references the former Fig3C publication path.
    # Load the locked patient observation directly and check its bytes/geometry.
    record,_=energy.clinical._load_record('epilepsiae_1146')
    fz=energy.clinical.load_frozen('epilepsiae_1146')
    patient_source=Path(previous['patient_source'])
    assert hashlib.sha256(patient_source.read_bytes()).hexdigest()==previous['patient_source_sha256']
    case=json.loads(patient_source.read_text())
    assert fz['fingerprint']==previous['frozen_fingerprint']
    assert case['clinical_contact_names']==list(fz['names'])==previous['contact_names']
    assert np.array_equal(case['activation_clinical_order'],previous['patient_robust_z'])
    # The reused painter only needs the frozen field geometry from this getter.
    energy.patient_cases=lambda:(record,fz,[case])
    target=list(fz['names']);names=list(a['contact_names']);ids=[names.index(n) for n in target]
    raw=a['lfp_native'][::5];t=a['lfp_native_time_ms'][::5]/1000.
    sig=(raw-raw.mean(axis=1,keepdims=True))[:,ids]
    onset=round(entry*2000)/2000
    bp,bt=energy.segment_power(sig,t,2000.,(.5,3.5))
    ep,et=energy.segment_power(sig,t,2000.,(onset,onset+1.))
    robust=energy.clinical.distal_baseline_robust_z(np.c_[bp,ep],np.r_[bt,et],(.5,3.5),min_frames=5)
    model=robust['delta'][:,-1]
    assert np.isfinite(model).all()
    # Only carry forward patient observations and the explicit observation contract.
    # Baseline powers, correlations and model selection from the old model are not reused.
    c={k:copy.deepcopy(previous[k]) for k in ['contract','contact_names','selected',
        'patient_robust_z','patient_extraction','patient_source','patient_source_sha256',
        'patient_checkpoint_max_error','frozen_fingerprint','field_scoring','display_geometry']}
    c['contract'].update(model_onset='First all-E >=200 Hz for200ms, using10ms bins; PSD rounded to nearest0.5ms.',
        model_source='Historical manual-core recovery network with one new current-type adaptation; seed9108405.',
        patient='E1146 SZ26 fixed from reference Fig6, same cached CAR broadband activation and baseline.',
        selection='No new patient selection or seizure search.',
        secondary='Signed contact Pearson/Spearman for the fixed example only.',
        patient_selection='SZ26 fixed from reference Fig6; no selection against the new model',
        model_signal_sampling='Native10kHz pre-Z proxy sampled every fifth point for canonical2kHz estimator')
    c.update(name='unified_current_adaptation',model_robust_z=model.tolist(),model_entry_s=entry,
        model_baseline_s=[.5,3.5],model_early_s=[onset,onset+1],model_baseline_frames=5,
        model_early_frames=ep.shape[1],model_baseline_log_power=bp,model_early_log_power=ep,
        model_baseline_log_power_median=robust['baseline_log_power_median'],
        model_baseline_log_power_mad=robust['baseline_log_power_mad'],
        model_baseline_z_median=robust['baseline_z_center'],
        model_mean_z=float(model.mean()),model_positive_contacts=int((model>0).sum()),
        selection='Patient SZ26 fixed from reference Fig6; no new seizure selection',
        n_candidates=1,model_sampling_hz=2000.,exact_entry_s=entry,
        PSD_alignment_error_ms=(onset-entry)*1000,
        model_signal_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_source=str(ROOT/'runs'/RUN_NAME),human_review='PENDING')
    weights=energy.scorers_from_interictal_record(record)['shared_a']['weight_id']
    mf=energy._smooth_from_weights(model,weights)
    pf=energy._smooth_from_weights(np.asarray(c['patient_robust_z']),weights)
    c['selected']=dict(c['selected'],field_r=float(np.corrcoef(mf,pf)[0,1]),
        contact_r=float(np.corrcoef(model,c['patient_robust_z'])[0,1]),
        contact_rho=float(spearmanr(model,c['patient_robust_z']).statistic))
    write(OUT/'field_comparison.json',c)
    fig=plt.figure(figsize=(8.7,4.5));spec=fig.add_gridspec(1,1,left=.05,right=.94,top=.95,bottom=.18)[0]
    native.draw_onset_fields(fig,spec,a,c,'minmax_z');save(fig,'fig6-panele')
    fig=plt.figure(figsize=(8.7,4.5));spec=fig.add_gridspec(1,1,left=.05,right=.94,top=.95,bottom=.18)[0]
    native.draw_onset_fields(fig,spec,a,c,'raw_z');save(fig,'onset-field-signed-z')
    return c


def raster_and_overview(a,r,ss):
    duration=r['observed_s'];samples=a['sample_ids'];ne=32000
    d=np.linalg.norm(a['positions_e'][:,None]-a['centers_mm'][None],axis=2)
    group=np.full(ne,2);group[d[:,0]<1.75]=0;group[(d[:,1]<1.75)&(d[:,1]<d[:,0])]=1
    gr=np.array([group[x] if x<ne else 3 for x in samples])
    order=np.argsort(gr,kind='stable');colors=['#b73169','#1286a4','#315a7a','#bd7a2c']
    fig,axs=plt.subplots(3,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[2.2,1,1]})
    for j,k in enumerate(order):
        t=np.flatnonzero(a['raster'][:,k])*.0001
        axs[0].plot(t,np.full(len(t),j),'.',ms=1,c=colors[gr[k]],rasterized=True)
    centers=[];labels=[]
    for g,label in enumerate(['Core A E','Core B E','Other E','I']):
        pos=np.flatnonzero(gr[order]==g)
        if len(pos):centers.append(pos.mean());labels.append(label)
    axs[0].set(yticks=centers,yticklabels=labels,ylabel='Sampled cells')
    rate=a['spikes_1ms'][:,0].reshape(-1,10).sum(1)/32000/.01
    axs[1].plot((np.arange(len(rate))+.5)*.01,rate,c='#333',lw=.8)
    axs[1].set_ylabel('E rate (Hz)')
    st=a['slow_time_ms']/1000.
    for idx,label,style in [(0,'All E','-'),(5,'Core A','--'),(6,'Core B',':')]:
        axs[2].plot(st,a['Z'][:,idx],ls=style,label=label,c=native.layout.Z_COLOR,lw=1.4)
    axs[2].set(ylabel='Z',xlabel='Time (s)',ylim=(0,1.05));axs[2].legend(ncol=3,frameon=False)
    for ax in axs:ax.set_xlim(0,duration)
    fig.tight_layout();save(fig,'native-raster-overview')
    # Expose the actual within-event core timing that a 60-s overview cannot resolve.
    selected=[s for s in ss if s['number'] in [1,3,6]]
    fig,axs=plt.subplots(1,len(selected),figsize=(5*len(selected),3.4),squeeze=False)
    core_order=[k for k in order if gr[k] in [0,1]]
    zooms=[]
    for ax,s in zip(axs[0],selected):
        anchor=s['time_s'];rule='Centered on the displayed event peak.'
        if s['number']==3:
            end=round((s['time_s']+.05)*10000)
            start=max(0,end-10000)
            fired=np.flatnonzero(a['raster'][start:end,core_order].any(axis=1))
            assert len(fired),'No core spikes in the last second before global low activity.'
            anchor=(start+fired[-1])/10000
            lo=max(0.,anchor-.25)
            rule='Last sampled-core spike preceding global low-activity onset; cores may stop before the surrounding sheet.'
        else:lo=max(0.,anchor-.15)
        hi=min(duration,lo+.3)
        for j,k in enumerate(core_order):
            spikes=np.flatnonzero(a['raster'][round(lo*10000):round(hi*10000),k])/10000+lo
            ax.plot(spikes,np.full(len(spikes),j),'|',ms=3,c=colors[gr[k]])
        n_a=sum(gr[k]==0 for k in core_order);n_b=len(core_order)-n_a
        ax.axhline(n_a-.5,c='#bbb',lw=.8)
        ax.set(yticks=[(n_a-1)/2,n_a+(n_b-1)/2],yticklabels=['Core A E','Core B E'],
            xlim=(lo,hi),ylim=(-.5,len(core_order)-.5),xlabel='Time (s)')
        ax.set_title(s['label'],color=s['color']);ax.xaxis.set_major_locator(MaxNLocator(3))
        zooms.append(dict(state=s['number'],window_s=[lo,hi],anchor_time_s=anchor,
            selection=rule,sample_ids=a['sample_ids'][core_order]))
    fig.tight_layout();save(fig,'core-event-zooms');write(OUT/'core_event_zooms.json',zooms)


def adaptation_diagnostic(name):
    folder=ROOT/'runs'/name
    d=analysis.load(folder,'adaptation_chunks',['time_ms','tau_A_s','q','global_s','adaptation_mean_mV'])
    g=analysis.load(folder,'mechanism_chunks',['time_ms','global_E_rate_Hz'])
    fig,axs=plt.subplots(4,1,figsize=(12,8),sharex=True)
    t=d['time_ms']/1000.
    axs[0].plot(g['time_ms']/1000.,g['global_E_rate_Hz'],lw=.8,c='#333')
    axs[0].axhline(5,c='#888',lw=.8,ls=':');axs[0].set_ylabel('Causal R (Hz)')
    axs[1].plot(t,30*d['global_s'],c='#547a9d',lw=1);axs[1].set_ylabel(r'$G/g_L$')
    axs[2].plot(t,d['adaptation_mean_mV'],c=native.layout.M_COLOR,lw=1)
    axs[2].set_ylabel(r'$\eta_M M$ (mV)')
    axs[3].step(t,d['tau_A_s'],where='post',c='#806a9e',lw=1)
    axs[3].set(ylabel=r'$\tau_A$ (s)',xlabel='Time (s)',yticks=[.5,5],ylim=(0,5.5))
    for ax in axs:ax.set_xlim(native.WINDOW)
    fig.tight_layout();save(fig,'adaptation-mechanism')


def compose():
    doc=fitz.open();mm=72/25.4;page=doc.new_page(width=320*mm,height=238*mm)
    boxes={'A':(5,12,185,109.5),'B':(5,115,185,173.5),'C':(6,181,193,233),
        'D':(199,8,316,111),'E':(199,115,316,173),'F':(218,184,314,233)}
    records={}
    for label,box in boxes.items():
        source=OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        target=fitz.Rect(*(v*mm for v in box))
        with fitz.open(source) as src:page.show_pdf_page(target,src,0,keep_proportion=True)
        letter_x=4 if label in 'ABC' else 198
        letter_y={'A':10,'D':10,'B':113,'E':113,'C':179,'F':179}[label]
        page.insert_text((letter_x*mm,letter_y*mm),label,fontname='hebo',fontsize=16)
        records[label]=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            box_mm=box,letter_position_mm=[letter_x,letter_y])
    page.insert_text((219*mm,181*mm),'Previous-model cohort reference',fontsize=8.2,fontname='heit',color=(.35,.35,.35))
    doc.save(OUT/'figures/fig6-complete-layout.pdf',garbage=4,deflate=True)
    # Render the serialized publication PDF, including its final font resources.
    with fitz.open(OUT/'figures/fig6-complete-layout.pdf') as published:
        published[0].get_pixmap(matrix=fitz.Matrix(2.6,2.6),alpha=False).save(OUT/'figures/fig6-complete-layout.png')
        (OUT/'figures/fig6-complete-layout.svg').write_text(published[0].get_svg_image())
    write(OUT/'layout_manifest.json',dict(canvas_mm=[320,238],panels=records,
        A_B_time_axis_positions_mm=[5+.12*180,5+.95*180],
        A_B_alignment='Same12-inch canvas width, left0.12/right0.95 axes, exact proportional180mm placement.',
        F_scope='Frozen prior-model cohort context, not validation of the unified adaptation model'))


def main(args):
    global RUN_NAME,EXIT,RETURN,PHASES
    RUN_NAME=args.name
    r=analysis.analyze(RUN_NAME)
    assert r['run_status']=='COMPLETE','Full candidate requires complete native trajectory.'
    assert r['has_high_entry'] and r['has_autonomous_exit'],'No termination: retain diagnostic, do not manufacture a termination figure.'
    OUT.mkdir(parents=True,exist_ok=True)
    a=load(RUN_NAME);ss,entry,EXIT,episode=samples(a,r)
    RETURN=next((s['time_s'] for s in ss if s['number']==6),r['observed_s'])
    PHASES=[]
    entries=r['temporal']['entries'];exits=r['temporal']['low_activity_exits']
    for j,en in enumerate(entries):
        end=entries[j+1]['onset_s'] if j+1<len(entries) else r['observed_s']
        ex=next((x for x in exits if en['confirmation_s']<x['confirmation_s']<=end),None)
        stop=ex['start_s'] if ex else end
        PHASES.append(dict(state='high',start_s=en['onset_s'],end_s=stop))
        if ex:PHASES.append(dict(state='post_high',start_s=stop,end_s=end))
    native.ETA=.0005;native.WINDOW=(0.,r['observed_s']);native.RECRUIT=entry
    native.MAIN_BAND_HZ=(80.,1000.);native.GAIN_PERCENTILE=99.5;native.DISPLAY_GAIN=.68/1.48
    native.HIDDEN_STATE_NUMBERS={3,4};native.marks=marks
    plt.rcParams.update({'font.size':16,'axes.labelsize':18,'axes.titlesize':17,
        'xtick.labelsize':14,'ytick.labelsize':14,'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    fig=plt.figure(figsize=(12,6.5));ax=fig.add_axes([.12,.10,.83,.82])
    waveform=native.draw_a(ax,a,ss,entry);ax.set_xlabel('Time (s)');ax.tick_params(axis='x',labelbottom=True)
    save(fig,'fig6-panela')
    fig=plt.figure(figsize=(12,3.9));gs=fig.add_gridspec(2,1,left=.12,right=.95,bottom=.18,top=.86,hspace=.22)
    za=fig.add_subplot(gs[0]);ma=fig.add_subplot(gs[1],sharex=za)
    slow=native.draw_slow(za,ma,a,ss,entry);save(fig,'fig6-panelb')
    maps=draw_maps(a,ss);traj=draw_trajectory(a,ss,r['observed_s'])
    comparison=onset_comparison(a,entry)
    for ext in ['pdf','png','svg']:
        shutil.copy2(REFERENCE/'figures'/f'fig6-panelf.{ext}',OUT/'figures'/f'fig6-panelf.{ext}')
    raster_and_overview(a,r,ss);adaptation_diagnostic(RUN_NAME);compose()
    source=OUT/'source';source.mkdir(exist_ok=True)
    for f in [Path(__file__),Path(analysis.__file__),Path(analysis.__file__).with_name('run_topic4_unified_current_adaptation.py')]:
        if f.resolve()!=(source/f.name).resolve():shutil.copy2(f,source/f.name)
    for f in ['effective_model.json','equation_qa.json','route_qa.json','quiet_decay_qa.json']:
        shutil.copy2(ROOT/f,source/f)
    # The historical carrier keeps unused K/tau_M/dispatch fields in its metadata.
    # Archive those inputs explicitly; the candidate protocol only states executed rules.
    shutil.copy2(ROOT/'protocol.json',source/'carrier_metadata_at_dispatch.json')
    shutil.copy2(ROOT/'runs'/RUN_NAME/'applied_configuration.json',source/'carrier_configuration_at_dispatch.json')
    dispatched=json.loads((ROOT/'protocol.json').read_text())
    effective=json.loads((ROOT/'effective_model.json').read_text())
    job=json.loads((ROOT/'jobs'/f'{RUN_NAME}.json').read_text())
    write(source/'protocol.json',dict(model=effective['model'],effective_model=effective,
        source_network_identity=dispatched['identity'],source_hashes=dispatched['source_hashes'],
        execution_source_sha256=dispatched['new_runner_sha256'],
        authorization=dispatched['authorization'],
        run={k:job[k] for k in ['name','seed','horizon_s','backend','device','high_jump_mV']},
        reference='Same trajectory at8s; no assertion of exact prefix equivalence to the old two-state model.',
        original_figure_replaced=False,author_visual_acceptance='PENDING',
        archived_carrier_note='carrier_*_at_dispatch.json preserve inherited implementation inputs and inactive legacy metadata; they do not define the new adaptation law.'))
    for f in ['cohort_statistics.json','cohort_current_version.json']:
        shutil.copy2(REFERENCE/'source'/f,source/f'previous_{f}')
    write(OUT/'scientific_result.json',r)
    write(OUT/'figure_metadata.json',dict(model=r['model'],source_run=str(ROOT/'runs'/RUN_NAME),
        source_layout=str(REFERENCE),same_run_panels=list('ABCDE'),frozen_prior_model_panel='F',
        core_readout_radius_mm=1.75,nominal_core_display_radius_mm=float(a['core_radius_mm']),
        region_counts=a['region_counts'],
        states=ss,waveform=waveform,slow=slow,spatial_maps=maps,trajectory=traj,
        marked_episode=episode,
        state_selection='First sustained return if present, otherwise first both-core Z recovery, otherwise first exit; mark its observed peak/termination. Full trajectory and all episodes are retained.',
        shaded_intervals=PHASES,
        patient_example='SZ26 fixed, not reselected',field_r=comparison['selected']['field_r'],
        human_visual_acceptance='PENDING',current_pointer_replaced=False))
    descriptions={'fig6-complete-layout':'沿用Fig6两列A–F布局。A–E模型数据均来自同一条新仿真，E固定原患者SZ26，F原样保留并标注为旧模型队列参考。',
        'fig6-panela':'15触点80–1000 Hz读出，先滤完整10 kHz轨迹，再用进入前P99.5固定增益显示全时程。模型读出为突触电流代理，不是临床电压。',
        'fig6-panelb':'蓝色Z、橙色单变量适应电流eta_M M；总体E实线、Core A虚线，与A共用时间轴。没有独立K状态或K电导。',
        'fig6-panelc':'四个20 ms原生空间率窗口，按实际观察到的状态选择；统一magma色条及真实双核位置，未观察到返回时不添加返回状态。',
        'fig6-paneld':'同一完整轨迹的Z、eta_M M与E率三维投影，时间色标为t/T。展开全部实际范围，不沿用旧图Z>=0.7的裁剪。',
        'fig6-panele':'新模型进入后1 s与固定SZ26患者场对照，保留1–150 Hz CAR robust-z算法与共享空间绘图。主图0–1色标显示形状，原始正负z保留在field_comparison.json。',
        'fig6-panelf':'原Fig6旧模型的13人队列统计，逐字节保留，未重算。它只提供既有背景，不能作为新适应模型的队列验证。',
        'onset-field-signed-z':'E的同一模型及患者读出，使用有符号robust-z色条；保留相对基线的增强和减弱方向。',
        'native-raster-overview':'完整仿真的固定细胞raster、总体E率及两核Z。用于核对高活动终止、低活动持续及间期事件返回，而非只凭D投影判断。'}
    descriptions['core-event-zooms']='间期、核心终止及已观察到的首次返回事件各300 ms双核raster；每个窗口使用相同细胞和纵轴。终止窗锚定全片低活动前最后一次核心样本放电，核心可早于周围网络停止；用于核对实际先后顺序。'
    descriptions['adaptation-mechanism']='实际保存的15 ms因果全E率、G电导、总体适应电流及适应时间常数。展示R跨越5 Hz时的时间常数切换，并与高活动的终止和再进入对应。'
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### {k}.png / .pdf\n{v}\n**关注点**：新候选待作者目视检查；数值与来源见scientific_result.json和figure_metadata.json。' for k,v in descriptions.items())+'\n')
    (OUT/'caption_zh.md').write_text(f'''# Figure 6 单变量适应候选

同一固定双核空间SNN中的高活动终止与恢复。保留历史成功Z/G/K轨迹的网络、空间输入和输入种子9108405，采用新批准的电流型单变量适应：膜电流减去eta_M M，eta_M=0.0005 mV；每次E放电增加[0.0005×1 s/tau_A+7.68q]/eta_M，tau_A在全E因果率R<=5 Hz时为5 s，其余为0.5 s。q=clip((R−200)/300,0,1)，R的因果滤波为15 ms。G保留30s_G及0.5 s响应；Z按原抑制负荷方程演化，tau_Z=5 s。原独立M/K两状态被一个M替代，没有K电导、定时干预或资源重置。参考电压匹配不等于原电导模型的严格等价变换。

**A–B**，完整{r['observed_s']:g} s的触点读出、Z和实际适应电流，固定通道顺序及显示增益。**C**，同次轨迹四个20 ms空间窗口。**D**，完整轨迹在总体Z、eta_M mean(M_i)、总体E率上的投影；率5 ms分箱并作5 ms显示平滑，慢变量插值到分箱中心；颜色t/T。标记时刻与A–C一致，完整范围可见，不能将投影闭合解释为极限环或正式分岔。

核心读出沿用中心1.75 mm邻域（Core A E=754、Core B E=786）；C的圆沿用冻结网络1.5 mm名义核心半径。浅红底标记观察器的高活动段，灰底为退出至下次高活动或记录终点的区间；灰底本身不等于已恢复间期。

**E**，新模型进入后1 s与原Fig6固定患者SZ26对照；模型基线0.5–3.5 s，5个重叠谱窗，患者基线和窗口保持原定义。主图各场独立min–max显示，不支持绝对幅度比较；有符号数据与相关保留在field_comparison.json。固定示例的场相关r={comparison['selected']['field_r']:.3f}；模型{comparison['model_positive_contacts']}/15触点为正，平均z={comparison['model_mean_z']:.3f}。没有重新选择最相似发作。**F**，逐字节保留旧模型13人/140次发作的统计，仅为旧模型队列参考，并非新模型验证。

本次模型统计单位为一个固定网络的一条输入实现；模型高活动按200 Hz持续200 ms定义，退出沿用既有独立观察器：连续两个1 s窗中总体及双核均值低于50 Hz、低于5 Hz的时间比例均至少20%。返回另要求两核Z均达到本次8 s参照，随后至少10个完整20–200 ms短事件、事件跨度至少5 s。该仿真说明模型内部行为，不证明临床癫痫机制。人工验收待完成。
''')
    with (OUT/'caption_zh.md').open('a') as f:
        f.write('\n状态编号与实际时刻：\n\n')
        for s in ss:f.write(f"- {s['number']}：{s['label']}，{s['time_s']:.4f} s。\n")
        f.write(f"\n操作性高活动进入：{entry:.3f} s；低活动退出窗起点：{episode['exit_start_s']:.3f} s，确认：{episode['exit_confirmed_s']:.3f} s。")
        if episode['both_core_Z_reference_s'] is not None:
            f.write(f"两核Z均回到本次8 s参照：{episode['both_core_Z_reference_s']:.3f} s。")
        f.write(f"恢复后完整短事件：{episode['brief_count']}次，跨度{episode['brief_span_s']:.3f} s，其中{episode['both_core_event_count']}次两核均达到20 Hz。\n")
    outcome='已自主终止，并满足本次操作性间期返回判据。' if r['has_sustained_return'] else '已自主终止；观察终点尚未满足本次操作性间期返回判据。'
    entries=', '.join(f"{e['onset_s']:.2f}" for e in r['temporal']['entries'])
    exits=', '.join(f"{e['confirmation_s']:.2f}" for e in r['temporal']['low_activity_exits'])
    (OUT/'README.md').write_text(f'''# Fig6 单变量电流适应候选

{outcome} 完整记录{r['observed_s']:g} s；高活动进入时刻为{entries} s，低活动退出确认时刻为{exits} s。所有段均显示，未将第一次短暂退出截成稳定恢复。

- [完整图](figures/fig6-complete-layout.png) / [PDF](figures/fig6-complete-layout.pdf)
- [D：完整状态轨迹](figures/fig6-paneld.png) / [D PDF](figures/fig6-paneld.pdf)
- [完整raster与率](figures/native-raster-overview.png) / [双核事件放大](figures/core-event-zooms.png)
- [适应时间常数与G读出](figures/adaptation-mechanism.png)
- [逐panel说明](figures/README.md) / [详细图注](caption_zh.md)
- [科学读出](scientific_result.json) / [当前方程和参数](source/protocol.json)

A–E的模型读出来自同一条新仿真。E固定原SZ26示例，场相关r={comparison['selected']['field_r']:.3f}；F逐字节保留旧模型队列结果，并在完整图中标明参考身份。仿真沿用历史手工双核恢复网络；本次复用Fig6版式，不是对Fig4F xy_left_20网络完成同一替换。

M是唯一内源适应状态，电流为eta_M M；没有独立K状态或K电导。转换按18 mV参考电压及固定率下的平均电流匹配，不能解释为原两状态系统的严格等价变换。原始载体的未执行K字段仅封存在source/carrier_*_at_dispatch.json，当前方程以source/protocol.json及effective_model.json为准。

统计单位为一个固定网络的一条输入实现。CPU/GPU短程完整状态逐位一致，放电数守恒及5 s静默衰减已核对。候选待作者目视检查，未覆盖正式Fig6。

无需重跑仿真，可在本工作区运行：

```bash
python /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1/scripts/paper_figures/build_fig6_unified_adaptation.py
```

原始观测：{ROOT/'runs'/RUN_NAME}。执行脚本及来源快照在source/。
''')
    print(str(OUT/'figures/fig6-complete-layout.png'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='unified_ref48_s9108405')
    main(p.parse_args())

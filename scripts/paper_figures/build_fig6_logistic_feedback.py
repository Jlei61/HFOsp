#!/usr/bin/env python3
"""Rebuild Fig6 A-D from a qualified shared-logistic trajectory; retain E.

Uses the accepted contact order, filters, spatial estimator, clinical field
comparison and panel dimensions. Stage times are explicit native observations,
never inherited from the clip trajectory. Requires a completed cycle assay.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter,sosfiltfilt

MAIN=Path('/home/honglab/leijiaxin/HFOsp')
WORK=MAIN/'.worktrees/topic4-continuous-core-state-r1'
ROOT=Path('/data/hfosp/topic4_sef_hfo/logistic_feedback_20261102')
OLD=MAIN/'results/paper-ready-figure/fig6/candidates/selected_r08_r09_normalized_onsets_20261102'
OUT=MAIN/'results/paper-ready-figure/fig6/candidates/shared_logistic_feedback_20261102'
sys.path.insert(0,str(MAIN/'scripts'))
sys.path.insert(0,str(WORK/'scripts/paper_figures'))
sys.path.insert(0,str(WORK/'scripts'))
import build_fig6_unified_adaptation as base
import build_fig6_selected_r08_r09_matched_row as matching
import build_fig6_history_seizure_selection as wave_style
import build_fig6_unified_adaptation_entry_zoom as trajectory_style
import analyze_topic4_logistic_feedback as assay

COLORS=['#9b8d32','#bd6c35','#ab4255','#ba263c']
PHASES=['Local','Spread','Entry','Ictal']
TICK,LABEL,TITLE,LETTER=matching.TICK,matching.LABEL,matching.TITLE,matching.LETTER
MM=72/25.4


def write(path,obj):base.write(path,obj)


def export(fig,stem):
    pdf=OUT/'figures'/f'{stem}.pdf'
    fig.savefig(pdf,dpi=220,facecolor='white')
    fig.savefig(pdf.with_suffix('.svg'),dpi=220,facecolor='white')
    plt.close(fig)
    with base.fitz.open(pdf) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72,220/72),alpha=False).save(pdf.with_suffix('.png'))


def load_native(run):
    a=base.analysis.load(run,'chunks',['time_ms','spikes_1ms','regions_1ms',
        'field_time_ms','field_5ms','slow_time_ms','Z','M'])
    raw=base.analysis.load(run,'dense_readout_chunks',['time_ms','contact_current'])
    a.update(lfp_native_time_ms=raw['time_ms'],lfp_native=raw['contact_current'])
    with np.load(ROOT/'geometry.npz') as g:a.update(dict(g))
    assert a['spikes_1ms'].shape[0]==120000
    assert raw['contact_current'].shape==(1200000,15)
    assert a['spikes_1ms'][:,0].sum()==a['regions_1ms'][:,:3].sum()==a['field_5ms'].sum()
    return a


def stages(a,config):
    t,ft=a['time_ms']/1000,a['field_time_ms']/1000
    rows=[];maps=[];denominators=[]
    for j,sec in enumerate(config['stage_times_s']):
        assert abs(sec*200-round(sec*200))<1e-8
        lo,hi=sec-.01,sec+.01
        count=a['field_5ms'][(ft>=lo)&(ft<hi)].sum(0)
        regions=a['regions_1ms'][(t>=lo)&(t<hi),:3].sum(0)
        assert count.sum()==regions.sum()
        hz=count/a['cell_e_counts']/.02;maps.append(hz)
        rows.append(dict(number=j+1,label=PHASES[j%4],episode=j//4+1,time_s=sec,
            sample_window_s=[lo,hi],color=COLORS[j%4],native_spike_count=int(count.sum()),
            all_E_rate20ms_Hz=float(count.sum()/32000/.02),
            core_A_B_surround_rate20ms_Hz=regions/a['region_counts'][:3]/.02,
            active_grid_fraction_above20Hz=float(np.mean(hz>20))))
    arrays={}
    for row in range(2):
        lo,hi=config['stage_times_s'][row*4],config['stage_times_s'][row*4+3]
        centers=np.arange(round(lo*200),round(hi*200)+1)/200
        fields=np.array([a['field_5ms'][(ft>=sec-.01)&(ft<sec+.01)].sum(0)/a['cell_e_counts']/.02 for sec in centers])
        den=float(fields.max());assert den>0;denominators.append(den)
        for j in range(4):
            at=np.flatnonzero(np.isclose(centers,rows[row*4+j]['time_s'],rtol=0,atol=1e-10))
            assert len(at)==1
            np.testing.assert_array_equal(fields[at[0]],maps[row*4+j])
        arrays[f'row{row+1}_center_time_s']=centers
        arrays[f'row{row+1}_continuous_field_Hz']=fields
    maps=np.asarray(maps)
    normalized=maps/np.repeat(denominators,4)[:,None]
    np.savez_compressed(OUT/'stage_readouts.npz',time_s=config['stage_times_s'],
        field_rate20ms_Hz=maps,cell_e_counts=a['cell_e_counts'])
    np.savez_compressed(OUT/'normalized_propagation_fields.npz',raw_selected_fields_Hz=maps,
        normalized_selected_fields=normalized,denominators_Hz=denominators,**arrays)
    normalization=[]
    for row in range(2):
        centers=arrays[f'row{row+1}_center_time_s']
        fields=arrays[f'row{row+1}_continuous_field_Hz']
        normalization.append(dict(row=row+1,center_time_window_s=[centers[0],centers[-1]],
            native_support_window_s=[centers[0]-.01,centers[-1]+.01],sample_width_ms=20,
            center_step_ms=5,n_continuous_frames=len(centers),denominator_Hz=denominators[row],
            selected_frame_peak_normalized=normalized[row*4:row*4+4].max(axis=1),
            peak_center_time_s=centers[np.unravel_index(np.argmax(fields),fields.shape)[0]]))
    write(OUT/'propagation_normalization.json',dict(method='One maximum across spatial cells and all 20-ms frames in each continuous illustrated entry window.',
        denominators_Hz=denominators,rows=normalization,per_frame_normalization=False,
        between_row_absolute_amplitude_comparable=False,color_limits=[0,1],color_mapping='linear',
        colorbar_label='Norm. Firing rate',native_rates_retained=True))
    with (OUT/'A_C_stage_correspondence.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['number','episode','phase','A_time_s','C_time_s','window_start_s','window_end_s','native_spike_count'])
        for state in rows:
            writer.writerow([state['number'],state['episode'],state['label'],state['time_s'],state['time_s'],
                *state['sample_window_s'],state['native_spike_count']])
    return rows,normalized,denominators


def draw_a(a,states,config,temporal):
    names=list(a['contact_names']);ids=[names.index(n) for n in wave_style.ORDER]
    t=a['lfp_native_time_ms']/1000
    signal=sosfiltfilt(butter(4,[80,1000],btype='bandpass',fs=10000,output='sos'),a['lfp_native'][:,ids],axis=0)
    cutoff=temporal['entries'][0]['onset_s']
    scale=np.percentile(abs(signal[t<cutoff]),99.5,axis=0)
    gains=np.maximum(scale,.15*np.median(scale[scale>0]))
    windows=config['waveform_windows_s'];onsets=config['display_onsets_s']
    durations=np.diff(windows,axis=1)[:,0];widths=.84*durations/durations.sum()
    fig=plt.figure(figsize=(17,5.6))
    axes=[fig.add_axes([.075,.145,widths[0],.775]),fig.add_axes([.14+widths[0],.145,widths[1],.775])]
    limits=[];records=[]
    for j,(ax,window,onset) in enumerate(zip(axes,windows,onsets)):
        use=(t>=window[0])&(t<=window[1]);v=signal[use,:13]/gains[:13]*.65+np.arange(12,-1,-1)
        ep=dict(process_start_s=onset)
        limits.append(wave_style.draw(ax,dict(episode=ep,t=t[use],values=v,window_s=window),left=j==0))
        ax.lines[-1].remove()
        for patch in list(ax.patches):patch.remove()
        exits=[e['start_s'] for e in temporal['low_activity_exits'] if e['start_s']>onset]
        end=min(window[1],exits[0]) if exits else window[1]
        ax.axvspan(onset,end,fc='#ba263c',alpha=.065,lw=0,zorder=0)
        ax.tick_params(labelsize=13)
        ax.set_xticks(np.arange(np.ceil(window[0]),np.floor(window[1])+1))
        ss=states[j*4:j*4+4]
        anchors=np.asarray([s['time_s'] for s in ss])
        gap=(window[1]-window[0])*.032/max(widths[j],.1)
        for k in range(1,4):anchors[k]=max(anchors[k],anchors[k-1]+gap)
        for state,anchor in zip(ss,anchors):
            ax.axvline(state['time_s'],c=state['color'],ls=':',lw=.85)
            ax.annotate(str(state['number']),xy=(state['time_s'],1),xycoords=ax.get_xaxis_transform(),
                xytext=(anchor,1.028),textcoords=ax.get_xaxis_transform(),ha='center',va='bottom',
                fontsize=14,weight='bold',color=state['color'],annotation_clip=False,
                arrowprops=dict(arrowstyle='-',color=state['color'],lw=.5))
        records.append(dict(window_s=window,displayed_onset_s=onset,shading_s=[onset,end],samples=int(use.sum())))
        np.savez_compressed(OUT/f'onset{j+1}_display_data.npz',time_s=t[use],filtered_readout=signal[use],
                            gain_per_contact=gains,contact_names=wave_style.ORDER)
    ylim=[min(x[0] for x in limits),max(x[1] for x in limits)]
    for ax in axes:ax.set_ylim(ylim)
    p,q=[ax.get_position() for ax in axes];mid=(p.x1+q.x0)/2
    u=np.linspace(0,1,201);x=mid+(u-.5)*.025;y=p.y0+.010*np.sin(2*np.pi*u)*np.sin(np.pi*u)**2
    fig.add_artist(Line2D(np.r_[p.x1,x,q.x0],np.r_[p.y0,y,q.y0],transform=fig.transFigure,c='black',lw=.8))
    fig.text(.017,.982,'A',fontsize=20,weight='bold',va='top')
    fig.text(.5275,.087,'Time (s)',fontsize=16,ha='center',va='top')
    export(fig,'fig6-panela')
    return dict(segments=records,contact_order=wave_style.ORDER,display_contacts=wave_style.ORDER[:13],
        filter_Hz=[80,1000],filter_order=4,full_signal_filter_before_windowing=True,
        gain_percentile=99.5,gain_baseline_end_s=cutoff,gain_per_contact=gains,
        display_gain=.65,same_time_scale=True,same_gain_for_both_windows=True)


def curve(ax,t,xyz,limits):
    inside=np.all((xyz>=limits[:,0])&(xyz<=limits[:,1]),axis=1)
    keep=inside[:-1]&inside[1:]
    ax.add_collection3d(Line3DCollection(np.stack([xyz[:-1],xyz[1:]],axis=1)[keep],
        cmap='viridis',norm=Normalize(0,120),array=((t[:-1]+t[1:])/2)[keep],linewidths=.75))
    ax.set(xlim=limits[0],ylim=limits[1],zlim=limits[2])
    ax.view_init(elev=26,azim=-125);ax.set_box_aspect((1.6,1,1.25));ax.set_proj_type('ortho')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.set_facecolor('#f2f5fa')
    return int(keep.sum())


def draw_b(a,states,config):
    rate=a['spikes_1ms'][:,0].reshape(-1,5).sum(1)/32000/.005
    t=(np.arange(len(rate))+.5)*.005;st=a['slow_time_ms']/1000
    xyz=np.c_[np.interp(t,st,a['Z'][:,0]),.0005*np.interp(t,st,a['M'][:,0]),gaussian_filter1d(rate,1)]
    np.savez_compressed(OUT/'trajectory_data.npz',time_s=t,Z_etaM_rate=xyz)
    bounds=np.array([[0,1],[0,max(300,np.ceil(xyz[:,1].max()/100)*100)],[0,500]])
    fig=plt.figure(figsize=(139/25.4,124/25.4))
    ax=fig.add_axes([.065,.065,.75,.80],projection='3d',computed_zorder=False)
    n=curve(ax,t,xyz,bounds);assert n==len(t)-1
    ax.set(xlabel='$Z$',ylabel=r'$\eta_M M$ (mV)',xticks=[0,.5,1],yticks=np.linspace(*bounds[1],3),zticks=[0,250,500])
    ax.set_xticklabels(['', '0.5', '1.0'])  # Leave the shared origin label to the M axis.
    ax.tick_params(labelsize=TICK,pad=1)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.label.set_fontsize(LABEL);axis.labelpad=4
    fig.text(.008,.48,r'$r_E$ (Hz)',rotation=90,va='center',fontsize=LABEL)
    win=config['inset_window_s'];use=(t>=win[0])&(t<win[1]);zs=xyz[use]
    lower=zs.min(0);upper=zs.max(0);pad=np.maximum((upper-lower)*.05,[.001,.001,1])
    zoom_bounds=np.c_[lower-pad,upper+pad];zoom_bounds[1:,0]=np.maximum(zoom_bounds[1:,0],0)
    zoom=fig.add_axes([.565,.64,.37,.31],projection='3d',computed_zorder=False,facecolor='white')
    nn=curve(zoom,t[use],zs,zoom_bounds);assert nn==len(zs)-1
    zoom.set(xlabel='$Z$',ylabel=r'$\eta_M M$',zlabel=r'$r_E$')
    zoom.tick_params(labelsize=TICK,pad=-1)
    for axis in (zoom.xaxis,zoom.yaxis,zoom.zaxis):
        axis.set_major_locator(MaxNLocator(2));axis.label.set_fontsize(TITLE);axis.labelpad=-2
    fig.text(.74,.98,'Interictal → entry',fontsize=LABEL,ha='center',va='top')
    fig.text(.07,.97,'Full trajectory',fontsize=LABEL,va='top')
    # Shared state numbers refer to exactly the same native A/C sample times.
    labels=trajectory_style.label_states(zoom,states[:3],t,xyz,fontsize=TITLE)
    labels+=trajectory_style.label_states(ax,states[3:4],t,xyz,fontsize=TITLE)
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,120),cmap='viridis'),cax=fig.add_axes([.865,.19,.023,.40]),ticks=[0,60,120])
    cb.set_label('Time (s)',fontsize=LABEL,labelpad=4);cb.ax.tick_params(labelsize=TICK,pad=3)
    export(fig,'fig6-panelb')
    return dict(time_s=[0,120],rate_bin_ms=5,rate_display_smoothing_sigma_ms=5,
        limits=bounds,segments=n,inset_window_s=win,inset_bounds=zoom_bounds,inset_segments=nn,
        states=labels,color='elapsed time in seconds, same scale for both trajectories')


def draw_c(a,states,maps):
    old=json.loads((OLD/'figure_metadata.json').read_text())['C_display']
    fig=plt.figure(figsize=(177/25.4,110/25.4));axes=[]
    for j,(state,hz) in enumerate(zip(states,maps)):
        row,col=divmod(j,4);ax=fig.add_axes(old['field_axes_bounds'][j]);axes.append(ax)
        im=ax.imshow(hz.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=1,interpolation='nearest')
        for k,xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=.9))
            ax.text(xy[0],xy[1]+2.1,'AB'[k],color='#10656a',fontsize=TICK,ha='center',weight='bold',bbox=dict(fc='white',ec='none',pad=.12,alpha=.85))
        ax.set(xticks=[0,10,20],yticks=[0,10,20])
        ax.tick_params(labelsize=TICK,length=2,pad=2,labelbottom=row==1,labelleft=col==0)
        if row==1:ax.set_xlabel('x (mm)',fontsize=LABEL,labelpad=3)
        if row==1 and col>0:ax.set_xticklabels(['','10','20'])
        if col==0:ax.set_ylabel('y (mm)',fontsize=LABEL,labelpad=3)
        ax.set_title(f"{state['number']}  {state['label']}\n{state['time_s']:.2f} s",color=state['color'],fontsize=TITLE,pad=5)
    for row in range(2):
        cb=fig.colorbar(im,cax=fig.add_axes(old['colorbar_bounds'][row]),ticks=[0,.5,1])
        cb.set_label('Norm. Firing rate',fontsize=LABEL,labelpad=4);cb.ax.tick_params(labelsize=TICK,length=2,pad=2)
        pos=axes[row*4].get_position();fig.text(3.2/177,pos.y0+pos.height/2,f'Model Onset {row+1}',rotation=90,ha='center',va='center',fontsize=LABEL)
    export(fig,'fig6-panelc')


def draw_d(a,config):
    comparisons=[]
    for j,onset in enumerate(config['display_onsets_s']):
        base.OUT=OUT/'onset_comparisons'/f'entry{j+1}';base.OUT.mkdir(parents=True,exist_ok=True)
        c=base.onset_comparison(a,onset)
        c['contract'].update(model_source='Shared-logistic G/M, same continuous120s run, seed9108404',
            model_onset='Explicit reviewed onset in the native trace, fixed one-second energy window')
        comparisons.append(c)
    base.OUT=OUT;write(OUT/'field_comparisons.json',comparisons)
    matching.SOURCE=matching.OUT=OUT
    fz,matches=matching.find_matches()
    matches.update(model_fields_and_windows_unchanged=False,
                   model_source=str(ROOT/'runs'/config['run_name']),
                   new_model_fields_recomputed=True)
    write(OUT/'model_selected_patient_matches.json',matches)
    clinical=base.native.energy.clinical
    fields=[(f'Model Onset {j+1}',np.asarray(m['model_robust_z']),'black') for j,m in enumerate(matches['matches'])]
    fields.extend((f"E10 {m['patient']['public_seizure']}",np.asarray(m['patient']['values']),'#214e86') for m in matches['matches'])
    fig=plt.figure(figsize=(245/25.4,71/25.4))
    for j,(x,(title,value,color)) in enumerate(zip([15,66,119,170],fields)):
        ax=fig.add_axes([x/245,15/71,46/245,46/71])
        clinical._draw_field(ax,fz,clinical._normalize_minmax(value),np.asarray(fz['support_a']),cmap='Blues',colorbar_values=np.array([0,1]),title=title,title_color=color,show_y=j==0)
        ax.set(xticks=[-20,0,20],yticks=[-10,0,10,20],xlabel='')
        ax.tick_params(labelsize=TICK,length=2,pad=2);ax.yaxis.label.set_fontsize(LABEL)
        ax.yaxis.labelpad=2
        ax.title.set_fontsize(TITLE);ax.title.set_fontweight('bold')
    fig.text(118/245,8.5/71,'Shared TA axis (mm)',ha='center',va='top',fontsize=LABEL)
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='Blues'),cax=fig.add_axes([220/245,15/71,1.7/245,46/71]),ticks=[0,.5,1])
    cb.set_label('Normalized z field',fontsize=LABEL,labelpad=3);cb.ax.tick_params(labelsize=TICK,length=2,pad=2)
    export(fig,'fig6-paneld')
    return matches


def compose():
    doc=base.fitz.open();page=doc.new_page(width=320*MM,height=294*MM)
    boxes={'A':(0,1,320,108),'B':(3,102,142,226),'C':(140,109,317,219),'D':(3,220,248,291),'E':(255,220,317,291)}
    for label,box in boxes.items():
        with base.fitz.open(OUT/'figures'/f'fig6-panel{label.lower()}.pdf') as src:
            page.show_pdf_page(base.fitz.Rect(*(v*MM for v in box)),src,0,keep_proportion=True)
        if label!='A':
            y=106 if label in 'BC' else 220
            page.insert_text((box[0]*MM,y*MM),label,fontsize=LETTER,fontname='hebo')
    pdf=OUT/'figures/fig6-complete-layout.pdf';doc.save(pdf,deflate=True);doc.close()
    with base.fitz.open(pdf) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72,220/72),alpha=False).save(pdf.with_suffix('.png'))


def main(config_path):
    config=json.loads(Path(config_path).read_text())
    comparison=json.loads((ROOT/'comparison.json').read_text())
    result=next(r for r in comparison['rows'] if r['name']==config['run_name'])
    assert result['complete_cycle'], 'A full recovery/re-entry cycle is required'
    run=ROOT/'runs'/config['run_name']
    temporal=json.loads((ROOT/'analysis'/config['run_name']/'result.json').read_text())['temporal']
    pointer=MAIN/'results/paper-ready-figure/fig6/current_version.json';frozen=pointer.read_bytes()
    (OUT/'figures').mkdir(parents=True,exist_ok=True);(OUT/'source').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    base.ROOT=ROOT;base.RUN_NAME=config['run_name']
    a=load_native(run);states,maps,denominators=stages(a,config)
    wave=draw_a(a,states,config,temporal)
    trajectory=draw_b(a,states,config)
    draw_c(a,states,maps)
    matches=draw_d(a,config)
    for ext in ['pdf','png','svg']:shutil.copy2(OLD/'figures'/f'fig6-panele.{ext}',OUT/'figures'/f'fig6-panele.{ext}')
    compose()
    metadata=dict(status='PENDING_AUTHOR_VISUAL_REVIEW',source_run=str(run),source_simulation_rerun=True,
        feedback='shared parameterized logistic for G and M',same_run_panels=['A','B','C','D'],
        model_configuration=json.loads((run/'result.json').read_text())['job'],validation=result,
        effective_equations=json.loads((ROOT/'effective_protocol.json').read_text())['equations'],
        states=states,waveform=wave,trajectory=trajectory,normalization_denominators_Hz=denominators,
        C_normalization='Per-entry maximum across all20ms fields sampled every5ms from state1to4 or5to8; no per-frame normalization',
        D=dict(matches=[m['winner'] for m in matches['matches']],n_candidates=25),
        E_scope='Historical13patients140seizures; not rerun with this model',E_source=str(OLD/'figures/fig6-panele.pdf'),
        producer=str(Path(__file__).resolve()),formal_figure_pointer_modified=False)
    write(OUT/'figure_metadata.json',metadata);write(OUT/'display_configuration.json',config)
    for path in [Path(__file__),ROOT/'source/run_topic4_logistic_feedback.py',ROOT/'effective_protocol.json',ROOT/'equation_qa.json',ROOT/'comparison.json']:
        shutil.copy2(path,OUT/'source'/path.name)
    # Preserve the actually imported project helpers, including untracked older
    # renderers. Reproduction still requires the documented native/clinical data.
    dependencies=[]
    paths=sorted({Path(mod.__file__).resolve() for mod in sys.modules.values()
        if getattr(mod,'__file__',None) and str(mod.__file__).endswith('.py')
        and str(mod.__file__).startswith(str(MAIN))})
    for path in paths:
        destination=OUT/'source/python'/path.relative_to(MAIN)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        dependencies.append(dict(original=str(path),snapshot=str(destination.relative_to(OUT)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write(OUT/'source/render_dependencies.json',dependencies)
    checks={}
    for pdf in (OUT/'figures').glob('*.pdf'):
        with base.fitz.open(pdf) as doc:
            checks[pdf.name]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    write(OUT/'validation.json',dict(PDF_text_inside_canvas=checks,A_C_samples_match=True,
        complete_native_cycle=True,E_pdf_byte_identical=(OUT/'figures/fig6-panele.pdf').read_bytes()==(OLD/'figures/fig6-panele.pdf').read_bytes(),human_review='PENDING'))
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### fig6-{stem}.png / .pdf\n\n{text}\n**关注点**：共享 logistic 新连续仿真候选，待作者目视检查。' for stem,text in [
        ('complete-layout','完整 Figure 6。A–D 已用同一共享 logistic 连续仿真重算；E 保留历史队列结果。'),
        ('panela','同一网络的两次进入波形，共用通道顺序、滤波和增益。1–8 对应 C 的真实采样时刻。'),
        ('panelb','连续120秒的 Z、适应强度和放电率轨迹。小窗展示间期至首次进入。'),
        ('panelc','两次进入各四帧原生空间活动。每行用连续进入窗口的时空峰值统一归一化。'),
        ('paneld','两次模型进入窗口与 E10 患者发作场的描述性最佳匹配。'),
        ('panele','历史13患者140次发作统计，未以本次 logistic 模型重算。')])+'\n')
    print(json.dumps(dict(output=str(OUT),states=[s['time_s'] for s in states],D=metadata['D']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('configuration');args=ap.parse_args();main(args.configuration)

#!/usr/bin/env python3
"""Two horizontal entry sequences from one continuous unified-adaptation run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import build_fig6_unified_adaptation as base
import build_fig6_unified_adaptation_rearranged as layout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze_topic4_unified_adaptation_entry_switch as audit

COLORS = ['#9b8d32', '#bd6c35', '#ab4255', '#ba263c']
WINDOWS = [(5.5, 14.), (51., 58.)]
PHASES = ['Local', 'Spread', 'Entry', 'Ictal']


def draw_a(a, states, route, termination):
    native = base.native
    native.RECRUIT = route['episodes'][0]['high_threshold_onset_s']
    native.ETA = .0005
    native.MAIN_BAND_HZ = (80., 1000.)
    native.GAIN_PERCENTILE = 99.5
    native.DISPLAY_GAIN = .65
    native.HIDDEN_STATE_NUMBERS = set(range(1, 9))
    first, second = route['episodes'][:2]

    def marks(ax, ss, onset):
        for ep, end, color in [(first, termination, '#2166ac'), (second, 58., '#b2182b')]:
            ax.axvspan(ep['last_quiet_end_s'], end, fc=color, alpha=.065, lw=0)
        for s in ss:
            ax.axvline(s['time_s'], ls=':', color=s['color'], lw=.85)
        ax.set_xlim(native.WINDOW)
    native.marks = marks
    fig = plt.figure(figsize=(15.7, 6.3))
    gap, left, width = .035, .075, .905
    lengths = np.diff(np.array(WINDOWS), axis=1).ravel()
    widths = (width-gap)*lengths/lengths.sum()
    axes = [fig.add_axes([left, .13, widths[0], .69]),
            fig.add_axes([left+widths[0]+gap, .13, widths[1], .69])]
    waveforms = []
    for row, (ax, window) in enumerate(zip(axes, WINDOWS)):
        native.WINDOW = window
        ss = states[row*4:row*4+4]
        waveforms.append(native.draw_a(ax, a, ss, first['last_quiet_end_s']))
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('')
        ax.set_xticks([6, 8, 10, 12, 14] if row == 0 else [51, 53, 55, 57, 58])
        ax.spines[['top', 'right']].set_visible(False)
        if row:
            ax.set_ylabel(''); ax.tick_params(axis='y', left=False, labelleft=False)
            ax.spines['left'].set_visible(False)
        positions = [ss[0]['time_s']-.28, ss[1]['time_s']+.03, ss[2]['time_s']+.46,
                     min(ss[3]['time_s'], window[1]-.25)]
        for s, x in zip(ss, positions):
            ax.annotate(str(s['number']), xy=(s['time_s'], 1), xycoords=ax.get_xaxis_transform(),
                        xytext=(x, 1.035), textcoords=ax.get_xaxis_transform(), color=s['color'],
                        fontsize=15, weight='bold', ha='center', va='bottom', annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', lw=.7, color=s['color']))
        lead = 'B' if row == 0 else 'A'
        center = np.mean([first['last_quiet_end_s'], termination]) if row == 0 else 55.7
        ax.text(center, 1.17, f'Entry {row+1}: Core {lead} sustained first',
                transform=ax.get_xaxis_transform(), ha='center', fontsize=15,
                color='#2166ac' if row == 0 else '#b2182b')
        ax.text((window[0]+ss[0]['time_s'])/2, 1.17, 'Interictal',
                transform=ax.get_xaxis_transform(), ha='center', fontsize=14, color='#555')
    ylim = (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_ylim(ylim)
    for ax, x in [(axes[0], 1.), (axes[1], 0.)]:
        for y in [0., 1.]:
            ax.plot([x-.009, x+.009], [y-.012, y+.012], transform=ax.transAxes,
                    color='#555', lw=1.2, clip_on=False)
    fig.text(left+widths[0]+gap/2, .047, '14–51 s omitted', ha='center', fontsize=12, color='#555')
    fig.text(.53, .003, 'Time (s)', ha='center', va='bottom', fontsize=17)
    assert np.array_equal(waveforms[0]['gain_per_contact'], waveforms[1]['gain_per_contact'])
    base.save(fig, 'fig6-panela')
    return dict(segments=waveforms, visible_windows_s=WINDOWS, omitted_s=[14., 51.],
                equal_seconds_per_width=True, same_y_limits=ylim, same_contact_gains=True,
                filter_before_windowing=True, continuous_simulation=True,
                state_markers='1–4 first entry; 5–8 second entry; same four meanings')


def draw_c(a, states):
    fig = plt.figure(figsize=(8.8, 4.65))
    gs = fig.add_gridspec(2, 4, left=.095, right=.91, bottom=.12, top=.89,
                          wspace=.18, hspace=.57)
    records = []
    for j, s in enumerate(states):
        row, col = divmod(j, 4)
        ax = fig.add_subplot(gs[row, col])
        idx = round(s['time_s']/.005)
        count = a['field_5ms'][idx-2:idx+2].sum(0)
        hz = count/a['cell_e_counts']/.02
        assert count.sum() == a['spikes_1ms'][(idx-2)*5:(idx+2)*5, 0].sum()
        im = ax.imshow(hz.reshape(20, 20), origin='lower', extent=[0, 20, 0, 20],
                       cmap='magma', norm=PowerNorm(.6, 0, 500), interpolation='nearest')
        for k, xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy, float(a['core_radius_mm']), ec='#2dd4cd', fc='none', lw=1.1))
            ax.text(xy[0], xy[1]+2.1, 'AB'[k], color='#10656a', fontsize=9, ha='center',
                    weight='bold', bbox=dict(fc='white', ec='none', pad=.15, alpha=.85))
        ax.set(xticks=[0, 10, 20], yticks=[0, 10, 20])
        ax.tick_params(labelsize=9, length=2, pad=2, labelbottom=row==1, labelleft=col==0)
        if row == 1: ax.set_xlabel('x (mm)', fontsize=10, labelpad=2)
        if col == 0: ax.set_ylabel('y (mm)', fontsize=10, labelpad=2)
        ax.set_title(f"{s['number']}  {s['label']}\n{s['time_s']:.3f} s",
                     color=s['color'], fontsize=11, pad=5)
        records.append(dict(**s, E_rate_Hz=hz, time_window_s=[(idx-2)*.005, (idx+2)*.005],
                            native_spike_count=int(count.sum()), cells_above_display_max=int((hz>500).sum())))
    fig.text(.018, .70, 'Entry 1 · B leads', ha='center', va='center', rotation=90, fontsize=11, color='#2166ac')
    fig.text(.018, .25, 'Entry 2 · A leads', ha='center', va='center', rotation=90, fontsize=11, color='#b2182b')
    cb = fig.colorbar(im, cax=fig.add_axes([.94, .155, .018, .66]), ticks=[0, 250, 500])
    cb.set_label('E rate (Hz)', fontsize=11, labelpad=5); cb.ax.tick_params(labelsize=10)
    base.save(fig, 'fig6-panelc')
    return records


def draw_d(a, states, duration):
    fig = plt.figure(figsize=(4.7, 4.7))
    ax = fig.add_axes([.07, .22, .82, .73], projection='3d')
    ax.computed_zorder = False
    st = a['slow_time_ms']/1000
    rate = a['spikes_1ms'][:, 0].reshape(-1, 5).sum(1)/32000/.005
    t = (np.arange(len(rate))+.5)*.005
    xyz = np.c_[np.interp(t, st, a['Z'][:, 0]), .0005*np.interp(t, st, a['M'][:, 0]), gaussian_filter1d(rate, 1)]
    ax.add_collection3d(Line3DCollection(np.stack([xyz[:-1], xyz[1:]], axis=1),
        cmap='viridis', norm=Normalize(0, 1), array=(t[:-1]+t[1:])/(2*duration), linewidths=.75, zorder=6))
    ax.set(xlim=(max(0, xyz[:, 0].min()-.035), 1.02), ylim=(0, xyz[:, 1].max()*1.08),
           zlim=(0, xyz[:, 2].max()*1.06), xlabel='$Z$', ylabel=r'$\eta_M M$ (mV)', zlabel='')
    # Axes3D can omit its off-canvas z label from tight-bbox exports.
    fig.text(.015, .64, r'$r_E$ (Hz)', rotation=90, va='center', ha='left', fontsize=11)
    ax.view_init(elev=26, azim=-125); ax.set_box_aspect((1.25, 1, 1.05))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(11); axis.labelpad=5
        axis.pane.set_facecolor('#f2f5fa'); axis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=9, pad=0)
    fig.canvas.draw(); boxes=[]; records=[]
    for s in states:
        pt=np.array([np.interp(s['time_s'], t, xyz[:, k]) for k in range(3)])
        ax.scatter(*pt, c=s['color'], s=24, edgecolor='white', linewidth=.5, depthshade=False, zorder=10)
        x,y,_=proj3d.proj_transform(*pt, ax.get_proj()); anchor=ax.transData.transform((x,y)); unit=fig.dpi/72
        offsets=[(13,13),(-20,13),(18,-20),(-20,-20),(0,34),(34,0),(-34,0),(0,-34),(33,28),(-33,28),(45,-35),(-45,-35)]
        for off in offsets:
            center=anchor+np.array(off)*unit
            box=(center[0]-8*unit,center[1]-8*unit,center[0]+8*unit,center[1]+8*unit)
            if all(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3] for b in boxes):
                boxes.append(box); break
        else: raise RuntimeError('Unable to place trajectory state label.')
        ax.annotate(str(s['number']), xy=(x,y), xytext=off, textcoords='offset points', ha='center', va='center',
                    fontsize=10, weight='bold', bbox=dict(boxstyle='circle,pad=.15',fc='white',ec='#777',lw=.6),
                    arrowprops=dict(arrowstyle='-',color='#777',lw=.5), zorder=20)
        records.append(dict(**s, Z_etaM_rate=pt, label_offset_pt=off))
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),
                   cax=fig.add_axes([.18,.155,.55,.025]),orientation='horizontal',ticks=[0,.5,1])
    cb.set_label('Normalized time',fontsize=10,labelpad=2);cb.ax.tick_params(labelsize=9,pad=2,length=2)
    fig.legend(handles=[Line2D([],[],marker='o',ls='none',color=COLORS[i],
                  label=f'{i+1}, {i+5}  {PHASES[i]}',markersize=4) for i in range(4)],
               loc='lower center',bbox_to_anchor=(.48,-.045),ncol=2,frameon=False,fontsize=9,
               handletextpad=.1,columnspacing=.7)
    base.save(fig,'fig6-paneld')
    np.savez_compressed(base.OUT/'trajectory_data.npz',time_s=t,Z_etaM_rate=xyz)
    return dict(coordinates=['mean Z','eta_M mean M (mV)','all-E rate (Hz)'],time_s=[0,duration],
                rate_bin_ms=5,display_smoothing_sigma_ms=5,normalized_time='t / full duration',
                states=records,clipping=False,eta_M=.0005,
                interpretation='Full native trajectory projection; not a bifurcation diagram.')


def draw_e(a, comparisons):
    # Reuse the exact clinical field painter/geometry; the patient is common to both models.
    clinical=base.native.energy.clinical
    _,fz,_=base.native.energy.patient_cases()
    fig=plt.figure(figsize=(9.1,4.05))
    gs=fig.add_gridspec(1,3,left=.075,right=.91,bottom=.22,top=.80,wspace=.23)
    first,second=comparisons
    fields=[first['model_robust_z'],first['patient_robust_z'],second['model_robust_z']]
    titles=['Model onset 1\n8.40–9.40 s','Patient SZ26\nCommon reference','Model onset 2\n54.00–55.00 s']
    axes=[]
    for i,(values,title) in enumerate(zip(fields,titles)):
        ax=fig.add_subplot(gs[i]); axes.append(ax)
        clinical._draw_field(ax,fz,clinical._normalize_minmax(np.asarray(values)),np.asarray(fz['support_a']),
                             cmap='Blues',colorbar_values=np.array([0.,1.]),title=title,title_color='black',show_y=i==0)
        ax.set_xticks([-20,0,20]);ax.set_yticks([-10,0,10,20])
        ax.tick_params(labelsize=10);ax.xaxis.label.set_fontsize(11);ax.yaxis.label.set_fontsize(11)
        ax.title.set_fontsize(12);ax.title.set_fontweight('normal')
        if i>0:ax.tick_params(labelleft=False);ax.set_ylabel('')
        if i!=1:
            c=comparisons[0 if i==0 else 1]
            ax.text(.5,-.32,f"$r$ = {c['selected']['field_r']:+.3f}",transform=ax.transAxes,
                    ha='center',fontsize=12,color='#333')
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='Blues'),
                   cax=fig.add_axes([.945,.25,.018,.47]),ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=10);cb.set_label('Normalized z field',fontsize=11,labelpad=4)
    base.save(fig,'fig6-panele')
    return dict(layout='model onset 1 / same fixed patient / model onset 2',
                shared_patient=True,display='per-field min-max normalized robust z',
                correlation='signed Pearson r of canonical smoothed fields on the fixed geometry',
                correlations=[c['selected']['field_r'] for c in comparisons],patient='E1146 SZ26',
                model_windows_s=[c['model_early_s'] for c in comparisons])


def supplements(a,rates,route,states,termination):
    fullout=base.OUT; base.OUT=fullout/'supplementary_candidates'
    fig,axes=plt.subplots(2,1,figsize=(12,5.5),sharex=True)
    st=a['slow_time_ms']/1000
    for label,color,zi,mi in [('All E','#333333',0,0),('Core A','#b2182b',5,1),('Core B','#2166ac',6,2)]:
        axes[0].plot(st,a['Z'][:,zi],color=color,label=label,lw=1.3)
        axes[1].plot(st,.0005*a['M'][:,mi],color=color,lw=1.3)
    axes[0].set_ylabel('Z');axes[1].set(ylabel=r'$\eta_M M$ (mV)',xlabel='Time (s)')
    axes[0].legend(frameon=False,ncol=3)
    for ax in axes:
        ax.set_xlim(0,58);ax.spines[['top','right']].set_visible(False)
        for s in states:ax.axvline(s['time_s'],color=s['color'],ls=':',lw=.7)
    fig.tight_layout();base.save(fig,'slow-state-traces')
    fig,axes=plt.subplots(2,2,figsize=(11,6.4),sharey='row')
    sample_ids=a['sample_ids'];dist=np.linalg.norm(a['positions_e'][:,None]-a['centers_mm'][None],axis=2)
    core=[np.flatnonzero((sample_ids<32000)&np.array([dist[i,k]<1.75 if i<32000 else False for i in sample_ids])) for k in [0,1]]
    for col,ep in enumerate(route['episodes'][:2]):
        left=ep['last_quiet_end_s']-.1;right=ep['high_threshold_onset_s']+.2
        tt=(np.arange(len(rates))+.5)*.01;take=(tt>=left)&(tt<=right)
        for j,label,color in [(1,'Core A','#b2182b'),(2,'Core B','#2166ac'),(3,'Surround','#888888')]:
            axes[1,col].plot(tt[take],rates[take,j],color=color,label=label,lw=1.3)
        for j,color in [(1,'#b2182b'),(2,'#2166ac')]:
            axes[1,col].axvline(ep['sustained_200ms_onset_all_A_B_surround_s'][j],color=color,ls=':',lw=1)
        offset=0
        for ids,color in zip(core,['#b2182b','#2166ac']):
            block=a['raster'][round(left*10000):round(right*10000),ids]
            times,neurons=np.nonzero(block)
            axes[0,col].plot(times/10000+left,neurons+offset,'|',color=color,ms=2,rasterized=True);offset+=len(ids)
        axes[0,col].axhline(len(core[0])-.5,color='#bbb',lw=.7)
        axes[0,col].set(yticks=[len(core[0])/2,len(core[0])+len(core[1])/2],yticklabels=['Core A','Core B'],
            title=f"Entry {col+1}: Core {ep['sustained_200ms_core_leader']} sustained first")
        axes[1,col].set(xlabel='Time (s)',ylabel='E rate (Hz)' if col==0 else '')
        for ax in axes[:,col]:ax.set_xlim(left,right);ax.spines[['top','right']].set_visible(False)
    axes[1,1].legend(frameon=False,fontsize=10);fig.tight_layout();base.save(fig,'two-entry-core-recruitment')
    base.OUT=fullout


def compose(out):
    mm=72/25.4;doc=base.fitz.open();page=doc.new_page(width=320*mm,height=330*mm)
    boxes={'A':(3,8,317,134),'D':(3,141,108,242),'C':(112,141,317,242),
           'E':(3,252,218,327),'F':(225,252,317,327)}
    letters={'A':(3,7),'D':(3,141),'C':(111,141),'E':(3,251),'F':(224,251)}
    records={}
    for label,box in boxes.items():
        path=out/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(path) as src:
            page.show_pdf_page(base.fitz.Rect(*(v*mm for v in box)),src,0,keep_proportion=True)
        page.insert_text(tuple(v*mm for v in letters[label]),label,fontsize=16,fontname='hebo')
        records[label]=dict(source=str(path),box_mm=box)
    target=out/'figures/fig6-complete-layout.pdf'
    doc.save(target,garbage=4,deflate=True);doc.close();layout.export_pdf(target)
    base.write(out/'layout_manifest.json',dict(canvas_mm=[320,330],panels=records,
        C_layout='2 rows by4 columns; D spans both C rows at left',B='supplementary',
        E_layout='two model onset fields and their common patient reference'))


def main(seed):
    assert seed==9108406,'This revision explicitly selects the saved persistent B→A example.'
    name=f'seed{seed}';folder=audit.RAW/'runs'/name
    base.ROOT=base.analysis.ROOT=audit.RAW;base.RUN_NAME=name
    out=base.MAIN/f'results/paper-ready-figure/fig6/candidates/unified_current_adaptation_two_entry_rows_s{seed}_20261031'
    base.OUT=out;(out/'figures').mkdir(parents=True,exist_ok=True)
    a=base.load(name);rates,route=audit.recruitment(folder,a)
    assert route['sustained_200ms_core_order']=='BA'
    result=base.analysis.analyze(name);details=json.loads((audit.OUT/f'{name}_details.json').read_text())
    first,second=route['episodes'];duration=len(a['spikes_1ms'])/1000
    quiet=np.all(rates[:,:3]<5,axis=1)
    ids=np.flatnonzero(np.convolve(quiet.astype(int),np.ones(10,int),'valid')==10)
    ids=ids[(ids*.01>first['high_threshold_onset_s'])&(ids*.01<second['last_quiet_end_s'])]
    termination=float(ids[0]*.01)
    times=[]
    for ep,local,spread,end in [(first,8.62,8.78,termination),(second,54.12,54.32,duration-.01)]:
        high=ep['high_threshold_onset_s'];centers=np.arange(round(high/.005),round(end/.005))
        totals=np.array([a['field_5ms'][i-2:i+2].sum() for i in centers])
        peak=float(centers[np.argmax(totals)]*.005)
        assert ep['last_quiet_end_s']<local<spread<high<peak
        times.extend([local,spread,high,peak])
    states=[dict(number=i+1,label=PHASES[i%4],episode=i//4+1,time_s=t,color=COLORS[i%4]) for i,t in enumerate(times)]
    # Validate the Local frames against the native core rates, not appearance alone.
    for s,leader in [(states[0],2),(states[4],1)]:
        idx=round(s['time_s']/.01);r=rates[idx-1:idx+1].mean(0)
        assert r[leader]>100 and r[3-leader]<5,(s,r)
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':14,
        'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    waveform=draw_a(a,states,route,termination)
    maps=draw_c(a,states);trajectory=draw_d(a,states,duration)
    comparisons=[]
    for i,ep in enumerate(route['episodes']):
        base.OUT=out/'onset_comparisons'/f'entry{i+1}'
        base.OUT.mkdir(parents=True,exist_ok=True)
        c=base.onset_comparison(a,ep['last_quiet_end_s'])
        c['contract'].update(model_source=f'Unified-current adaptation; noise seed{seed}; continuous58s run.',
            model_onset='Last global-quiet end before this entry; same1s process window for both episodes; includes early transient spikes.')
        c.update(episode=i+1,high_threshold_onset_s=ep['high_threshold_onset_s'],
                 persistent_core_leader=ep['sustained_200ms_core_leader'],
                 calculation_producer=c['producer'],calculation_producer_sha256=c['producer_sha256'],
                 producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        assert np.allclose(c['model_robust_z'],details['energy_fields'][i]['robust_z'],atol=1e-10)
        base.write(base.OUT/'field_comparison.json',c);comparisons.append(c)
    base.OUT=out
    assert comparisons[0]['patient_source_sha256']==comparisons[1]['patient_source_sha256']
    assert np.array_equal(comparisons[0]['model_baseline_log_power'],comparisons[1]['model_baseline_log_power'])
    e_meta=draw_e(a,comparisons)
    for ext in ['pdf','png','svg']:
        shutil.copy2(layout.OUT/'figures'/f'fig6-panelf.{ext}',out/'figures'/f'fig6-panelf.{ext}')
    supplements(a,rates,route,states,termination);compose(out)
    base.write(out/'core_recruitment.json',route);base.write(out/'scientific_result.json',result)
    base.write(out/'field_comparisons.json',comparisons)
    base.write(out/'figure_metadata.json',dict(seed=seed,source_run=str(folder),same_run_panels=['A','C','D','E'],
        states=states,waveform=waveform,spatial_maps=maps,trajectory=trajectory,E=e_meta,
        termination_s=termination,termination_definition='All E and both core10ms rates<5Hz for100ms',
        original_observer_exit_s=result['episodes'][0]['exit_start_s'],continuous_no_reset=True,
        persistent_core_order=route['sustained_200ms_core_order'],initial_50ms_core_order=route['core_order'],
        initial_A_burst_preserved=True,energy_template_order=[f['best_abs_template'] for f in details['energy_fields']],
        energy_fields=details['energy_fields'],F_scope='Unchanged historical13-patient model cohort; footer removed previously.',
        human_visual_acceptance='PENDING'))
    checks={}
    for path in (out/'figures').glob('*.pdf'):
        with base.fitz.open(path) as doc:
            checks[path.name]=all(pg.rect.contains(base.fitz.Rect(s['bbox'])) for pg in doc
                for b in pg.get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans'])
    assert all(checks.values()),checks
    original_f=layout.OUT/'figures/fig6-panelf.pdf'
    assert (out/'figures/fig6-panelf.pdf').read_bytes()==original_f.read_bytes()
    base.write(out/'validation.json',dict(status='PASS',same_uninterrupted_simulation=True,
        eight_native_20ms_maps_spike_conservation=True,Local_leader_active_other_core_below5Hz=True,
        four_phase_semantics_repeated=True,broken_axis_explicit=True,seconds_per_width_equal=True,
        waveform_gains_identical=True,full_trajectory_filter_before_crop=True,
        common_patient_and_baseline=True,two_E_fields_match_independent_saved_analysis=True,
        F_PDF_byte_identical=True,PDF_text_inside=checks,human_visual_acceptance='PENDING'))
    (out/'source').mkdir(exist_ok=True)
    for file in [Path(__file__),Path(base.__file__),Path(layout.__file__),Path(audit.__file__)]:
        shutil.copy2(file,out/'source'/file.name)
    write_docs(out,seed,states,termination,comparisons)
    print(json.dumps(dict(output=str(out),states=states,similarities=e_meta['correlations'])),flush=True)


def write_docs(out,seed,states,termination,comparisons):
    (out/'caption_zh.md').write_text(f'''同一固定网络与统一适应模型的连续58 s仿真（噪声种子{seed}），没有状态重置或外加触发。A保留5.5–14 s与51–58 s，14–51 s由明确断轴省略；两段的每秒显示宽度、纵坐标、各触点增益均一致，80–1000 Hz滤波在完整轨迹上进行。左侧保留多个间期间歇事件，第一次高活动终止后再次进入。100 Hz持续200 ms的招募定义下，第一次B核8.57 s早于A核8.71 s，第二次A核54.02 s早于B核54.25 s。第一次8.40 s有先行的A核短暂放电，它保留在原始轨迹和补图中；因此B→A描述持续招募的核间先后，而非首次spike起源或能量场TA/TB类型。

C为两行横向四帧。1–4与5–8分别对应两次进入的Local、Spread、Entry、Ictal，Local分别为8.620 s与54.120 s；这两帧中领先核率大于100 Hz，另一核低于5 Hz。Spread为第二核已被招募但全局高活动阈值尚未达到的时刻；Entry为全体E率至少200 Hz并持续200 ms的起点；Ictal为该段已观测高活动内20 ms率峰值。所有图使用同一0–500 Hz色标，逐帧按实际细胞数换算20 ms率，溢出色标的格数保存在metadata。D保留同一完整0–58 s原生轨迹的Z、适应电流与E率投影，并对应全部八个编号。编号不包括额外的termination状态。

E在左、右分别显示8.40–9.40 s与54.00–55.00 s的模型onset能量场，中间为两者共同的固定E1146 SZ26患者参考。模型沿用同一0.5–3.5 s间期基线、CAR与1–150 Hz能量算子；两个场与患者场的带符号空间相关为{comparisons[0]['selected']['field_r']:+.6f}、{comparisons[1]['selected']['field_r']:+.6f}。显示采用逐场min-max归一化，仅比较空间形状；r在原冻结平滑场上计算，未重新挑选患者发作。另按原TA/TB最大绝对相关规则，两次模型场均偏TA，不能将核招募的B→A改称能量场TB→TA。

F保留历史模型13人队列及原p值，未使用本次种子搜索重算；图内小脚注已去掉。B的慢变量曲线以及两次进入的原生raster和区域率在supplementary候选中。首次进入低活动的100 ms严格判据起点为{termination:.2f} s，原observer记录的恢复段起点为13.24 s，二者定义不同。第二次高活动在58 s观测结束时仍持续。此例经噪声种子筛选，用于展示可能轨迹，不估计发生概率；正式Fig6入口未替换，待作者目视检查。
''')
    (out/'README.md').write_text(f'''# 两次进入、两行四帧 Figure 6 候选

[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [图注](caption_zh.md)。

顶部A为断轴连续仿真读出，中排左D与右C的两行等高，C每行横排四帧；底排E为两个模型onset与共同患者参考，F保留历史统计。种子{seed}按持续招募判据为B→A，但第一次有先行A核短暂放电，两个能量场均偏TA。

完整数值和来源见figure_metadata.json、core_recruitment.json与field_comparisons.json。正式入口不变，等待作者目视检查。复现：`python {Path(__file__).resolve()} --seed {seed}`。
''')
    descriptions={
        'fig6-complete-layout':'完整五面板排版：大A；中排左D、右侧两行四列C；底排E/F。所有新模型数据来自同一连续仿真。',
        'fig6-panela':'两段共用时间比例和增益的原生电极读出，明确省略14–51 s。左侧保留较长间期，1–4与5–8对应两次进入。',
        'fig6-panelc':'第一次B核持续招募领先、第二次A核领先，两行各四个20 ms原生率场。两行重复Local、Spread、Entry、Ictal语义。',
        'fig6-paneld':'完整0–58 s原生Z、适应电流、E率轨迹投影，标出两个进入过程共八个状态。时间颜色覆盖完整轨迹，未随A断轴。',
        'fig6-panele':'两个1 s模型onset能量场与同一患者SZ26参考，分别列出带符号空间相似度。患者参考与模型间期基线在两次比较中保持相同。',
        'fig6-panelf':'历史模型13人队列对照，数据与统计语法保持不变。沿用去除小脚注的版本。'}
    (out/'figures/README.md').write_text('\n\n'.join(f'### {name}.png / .pdf\n{text}\n**关注点**：'+(
        '历史统计不构成本次新模型的队列验证。' if name=='fig6-panelf' else
        'B→A指持续招募先后；首次短暂A核放电与能量场分类需要分别理解。') for name,text in descriptions.items())+'\n')
    (out/'supplementary_candidates/figures/README.md').write_text('''### slow-state-traces.png / .pdf
原B机制读出移入补图候选；显示完整连续58 s的全体E及双核Z、适应电流。没有删除A断轴省略的中间时段。
**关注点**：终止、恢复、再次进入之间没有变量重置。

### two-entry-core-recruitment.png / .pdf
同一批固定抽样神经元的raster与原生核心/周围区域率，竖线按持续200 ms定义定位。首次早期A核短暂放电仍完整显示。
**关注点**：区分最早短暂放电与持续招募，避免将B核领先说成B核最先出现spike。
''')
    for i in [1,2]:
        (out/f'onset_comparisons/entry{i}/figures/README.md').write_text(f'''### fig6-panele.png / .pdf
第{i}次onset的原双场renderer核对输出，模型与固定患者SZ26并排。主图E使用相同数值，合并两个模型与共同参考。
**关注点**：field_comparison.json记录完整窗口、基线、接触顺序及相关系数。

### onset-field-signed-z.png / .pdf
同一场的原robust-z色标核对输出，保留负值和幅度范围。归一化主图只用于比较空间形状。
**关注点**：不能由主图归一化颜色推断增强幅度相等。
''')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=9108406)
    main(parser.parse_args().seed)

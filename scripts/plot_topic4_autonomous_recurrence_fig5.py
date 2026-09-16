#!/usr/bin/env python3
"""Full Fig5 from a source-matched dense autonomous-recurrence trajectory.

Every panel is measured on the declared fixed manual-core substrate. The
parameter panel uses the same revised-Z equation, not the old native-Z M scan.
No fitted vector field, patient seizure map, or external-reset line is added.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter1d
import analyze_topic4_autonomous_recovery as observation
import plot_topic4_fig5_single_transition as old_figure

BASE = observation.OUT
SOURCE = BASE / 'preserved_global_gain_round7'
DISPLAY_FAMILY = 'auto'
COLORS = ['#657181', '#c48a28', '#d76834', '#b62a45', '#268771', '#a82b49']


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def dense(folder, keys):
    values = {key: [] for key in keys}
    for path in sorted((folder / 'dense_chunks').glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as chunk:
            for key in keys:
                values[key].append(chunk[key])
    return {key: np.concatenate(parts) for key, parts in values.items()}


def native_snapshot(data, geo, requested):
    # Fixed ten5ms bins: no floating-boundary ambiguity and no spatial blur.
    start = int(round((requested - .025) / .005))
    assert 0 <= start and start + 10 <= len(data['field_5ms'])
    counts = data['field_5ms'][start:start + 10].sum(0)
    rates = counts / geo['cell_e_counts'] / .05
    center = start * .005 + .025
    whole = counts.sum() / 32000 / .05
    assert np.isclose(np.average(rates, weights=geo['cell_e_counts']), whole)
    return dict(time_s=center, window_s=[start*.005, (start+10)*.005],
                rate_Hz=rates, all_E_Hz=whole, recruited_area=float((rates >= 20).mean()))


def snapshots(data, geo, result, audit, prior):
    times = list(prior['state_times_s'][:4])
    labels = ['Rest', 'Interictal', 'Entry', 'High']
    returned = audit['returns'][0]['finite_events_after_confirmation']
    if returned:
        times.append(returned[0]['peak_s'])
        labels.append('Return event')
    else:
        quiet = audit['temporal_audit']['recovery_timing_checks'][0][
            'strict_core_and_global_quiet_intervals_between_entries']
        interval = max(quiet, key=lambda v: v['duration_s'])
        times.append((interval['start_s'] + interval['end_s']) / 2)
        labels.append('Quiet return')
    second = result['tracker']['entries'][1]
    candidates = np.arange(np.ceil((second['onset_s']+.025)/.005)*.005,
                           second['confirmation_s']-.024, .005)
    assert len(candidates)
    time6 = max(candidates, key=lambda t: native_snapshot(data, geo, t)['all_E_Hz'])
    times.append(float(time6)); labels.append('Re-entry')
    snaps = []
    for number, (tm, label, color) in enumerate(zip(times, labels, COLORS), 1):
        snap = native_snapshot(data, geo, tm)
        snap.update(number=number, label=label, color=color)
        snaps.append(snap)
    assert np.all(np.diff([v['time_s'] for v in snaps]) > 0)
    assert snaps[0]['all_E_Hz'] < 1
    assert snaps[-1]['all_E_Hz'] >= 200
    return snaps


def matched_grid(horizon=30., kind='timescale'):
    r7=BASE/'preserved_global_gain_round7'
    if kind=='timescale':
        roots = [(BASE/'continuous_resource_recovery_round5', 'resource_rho0.25_k50_s9108401')]
        roots += [(r7, name) for name in ['resource_rho0.25_k200_tau2_s9108401',
                  'resource_rho0.25_k50_tau10_s9108401', 'resource_rho0.25_k200_tau10_s9108401']]
        y_values,y_parameter,y_label=[2,10],'pool_tau_s',r'$\tau_G$ (s)'
        fixed_parameter='recovery_ratio'
    else:
        assert kind=='resource'
        r8=BASE/'paired_recurrence_confirmation_round8'
        roots=[(r8,f'resource_rho0_k{gain}_tau10_s9108401') for gain in [50,200]]
        roots += [(r7,f'resource_rho0.25_k{gain}_tau10_s9108401') for gain in [50,200]]
        y_values,y_parameter,y_label=[0,.25],'recovery_ratio','Added recovery ρ'
        fixed_parameter='pool_tau_s'
    rows = []
    reference = None
    geometry_reference=None
    for root, name in roots:
        job = observation.read(root/'jobs'/f'{name}.json')
        data = observation.load(root/'runs'/name, ['spikes_1ms', 'regions_1ms'])
        if data is None or len(data['spikes_1ms']) < round(horizon*1000):
            raise RuntimeError(f'WAITING_COMMON_{horizon:g}S: {name}')
        geo = np.load(root/'geometry.npz')
        invariant = {key: job[key] for key in ['mode', 'gamma', 'eta_m', 'tau_M_s',
            'tau_Z_s', 'threshold', 'seed', 'pool_threshold_Hz', fixed_parameter]}
        if reference is None:
            reference = invariant
        assert invariant == reference
        geometry={key:geo[key] for key in ['region_counts','cell_e_counts','centers_mm','positions_e']}
        if geometry_reference is None:geometry_reference=geometry
        assert all(np.array_equal(value,geometry_reference[key]) for key,value in geometry.items())
        steps = round(horizon*1000); n = steps//10
        counts = data['spikes_1ms'][:steps, 0].reshape(n, 10).sum(1)
        regions = data['regions_1ms'][:steps, :3].reshape(n, 10, 3).sum(1)
        rates = np.c_[counts/320, regions/geo['region_counts'][:3]/.01]
        tracker = observation.run.fresh_tracker()
        for k, rr in enumerate(rates):
            observation.run.track(tracker, rr, (k+1)*.01)
        entries, returns = tracker['entries'], tracker['recoveries']
        category = 3 if len(entries)>=2 and returns else 2 if returns else 1 if entries else 0
        rows.append(dict(name=name, source=root/'runs'/name, job=job,
            matched_observation_s=horizon, category=category, entries=entries, returns=returns,
            late5s_all_A_B_surround_mean_Hz=rates[-500:].mean(0),
            late5s_all_A_B_surround_quiet_fraction=(rates[-500:]<5).mean(0)))
    return dict(rows=rows, horizon_s=horizon, invariant=reference,kind=kind,
        y_values=y_values,y_parameter=y_parameter,y_label=y_label,
        meaning='Categorical measured30s state outcomes on one fixed noise, not a continuous bifurcation surface or an onset-latency estimate. No-global-entry cells may have persistent local core firing.',
        categories=['No global high', 'High only', 'High then return', 'High / return / high'])


def letter(fig, spec, label):
    p = spec.get_position(fig)
    fig.text(p.x0-.008, p.y1+.014, label, fontsize=25, weight='bold')


def cores(ax, geo):
    for label, xy in zip('AB', geo['centers_mm']):
        ax.add_patch(Circle(xy, float(geo['core_radius_mm']), fill=False,
                            ec='#46d4d0', lw=1.6, zorder=6))
        ax.text(xy[0], xy[1]+2.05, label, color='#126871', fontsize=12,
                ha='center', weight='bold', zorder=7,
                bbox=dict(fc='white', ec='none', alpha=.85, pad=.3))


def draw_phase(fig, spec, state, counts, snaps, end):
    ax = fig.add_subplot(spec, projection='3d')
    ax.computed_zorder = False
    ax.set_anchor('N')
    st = state['state_time_ms']/1000
    keep = st < end
    st, values = st[keep], state['state'][keep]
    assert np.allclose(np.diff(st), .001)
    rate = gaussian_filter1d(counts[:, 0].astype(float)/32, 2)
    ct = (np.arange(len(counts))+.5)*.001
    xyz = np.c_[values[:, 0], values[:, 2], np.interp(st, ct, rate)]
    scale = np.maximum(np.ptp(xyz, axis=0), [1e-3, 1, 1])
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz/scale, axis=0), axis=1))]
    # Keep path geometry and at most50ms gaps, including slow motion at zero rate.
    selected = np.unique(np.r_[0, np.flatnonzero(np.diff(np.floor(length/.02))>0)+1,
                                np.arange(0, len(st), 50), len(st)-1])
    points, times = xyz[selected], st[selected]
    norm = Normalize(0, end)
    lc = Line3DCollection(np.stack([points[:-1], points[1:]], axis=1),
                          cmap='viridis', norm=norm, linewidth=1.05, alpha=.92, zorder=5)
    lc.set_array((times[:-1]+times[1:])/2)
    ax.add_collection3d(lc)
    ax.set(xlim=(max(0, xyz[:, 0].min()-.03), 1.025),
           ylim=(-15, xyz[:, 1].max()*1.05), zlim=(-12, xyz[:, 2].max()*1.08),
           xlabel='Mean Z', ylabel=r'$H_E$ (mV equiv.)', zlabel='E rate (Hz)')
    ax.view_init(elev=27, azim=-127)
    ax.set_box_aspect((1.2, 1, 1.05))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.labelpad=15; axis.set_major_locator(MaxNLocator(3))
        axis.pane.set_facecolor('#f3f6fa')
    fig.canvas.draw()
    annotations = []
    offsets = [(8, 27), (-23, 14), (18, -22), (-5, 24), (24, -18), (25, 14)]
    for snap, offset in zip(snaps, offsets):
        point = np.array([np.interp(snap['time_s'], st, xyz[:, j]) for j in range(3)])
        ax.scatter(*point, c=snap['color'], s=22, depthshade=False, zorder=9)
        px, py, _ = proj3d.proj_transform(*point, ax.get_proj())
        ax.annotate(str(snap['number']), (px, py), xytext=offset, textcoords='offset points',
                    ha='center', va='center', fontsize=14, weight='bold', zorder=15,
                    bbox=dict(boxstyle='circle,pad=.18', fc='white', ec='#777'),
                    arrowprops=dict(arrowstyle='-', color='#777', lw=.7))
        annotations.append(dict(number=snap['number'], time_s=snap['time_s'], coordinates=point))
    ca = ax.inset_axes([1.08, .15, .038, .70])
    fig.colorbar(lc, cax=ca, label='Time (s)')
    return dict(coordinates='Actual meanZ / mean_i[Zi*deliveredJi] / all-E rate',
        dense_state_dt_ms=1, rate_smoothing_sigma_ms=2, rate_source_dt_ms=1,
        preserved_points=len(selected), total_points=len(st),
        decimation='Normalized path-length threshold.02 plus at most50ms between points; all dynamics measured, no fitted arrows or nullclines.',
        view_elevation=27, view_azimuth=-127, state_points=annotations,
        trajectory_time_s=st, trajectory_coordinates=xyz)


def draw_grid(fig, spec, grid, working_job):
    ax = fig.add_subplot(spec)
    ax.set_anchor('N')
    values = np.full((2, 2), np.nan)
    for row in grid['rows']:
        job = row['job']; x=[50, 200].index(job['pool_gain']); y=grid['y_values'].index(job[grid['y_parameter']])
        values[y, x] = row['category']
    cmap = ListedColormap(['#dce0e6', '#bd5262', '#f2bd6e', '#5a9a89'])
    ax.imshow(values, origin='lower', cmap=cmap, vmin=-.5, vmax=3.5, aspect='equal')
    ax.set(xticks=[0, 1], xticklabels=['50', '200'], yticks=[0, 1],
           yticklabels=[f'{v:g}' for v in grid['y_values']],
           xlabel='Global gain κ', ylabel=grid['y_label'])
    for row in grid['rows']:
        job=row['job']; x=[50, 200].index(job['pool_gain']); y=grid['y_values'].index(job[grid['y_parameter']])
        if row['category']==3:
            label='High → return\n→ high'
        elif row['category']==0:
            label='No global high\nLocal activity'
        else:
            label=grid['categories'][row['category']]
        ax.text(x, y, label, ha='center', va='center', fontsize=16, color='#202830')
        if job['pool_gain']==working_job['pool_gain'] and job[grid['y_parameter']]==working_job[grid['y_parameter']]:
            ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,ec='#da8549',lw=3))
    ax.set_xticks([-.5,.5,1.5], minor=True); ax.set_yticks([-.5,.5,1.5], minor=True)
    ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor', length=0)
    ax.text(.5,-.24,'Same 30-s observation',ha='center',transform=ax.transAxes,fontsize=14)


def draw_energy(fig, spec, data, geo, folder, result):
    energy_folder=folder/'native_energy'
    spectral=np.load(energy_folder/'spectral_power.npz')
    method=observation.read(energy_folder/'metadata.json')
    observed=dense(folder,['contact_time_ms','effective_proxy'])
    # Template calculation reuses the established event/contact order observer.
    # It operates on actual post-Z contacts. Its unfiltered power is NOT plotted.
    templates=dict(data)
    templates['lfp_time_ms']=observed['contact_time_ms']
    templates['lfp_raw']=observed['effective_proxy']
    templates['region_counts']=geo['region_counts']
    comparison=old_figure.compare_families(templates,result['tracker']['entries'][0]['onset_s'],
        signal_name='Actual native10kHz post-Z current-magnitude contact proxy')
    contact_delta=spectral['delta_effective_proxy']
    field_delta=spectral['delta_native_effective_grid']
    all_families={family:dict(n_events=value['n_events'],rank=value['template_rank'],
        spearman_minus_rank_vs_band_change=old_figure.correlation(
            -np.asarray(value['template_rank'],float),contact_delta))
        for family,value in comparison['groups'].items()}
    if DISPLAY_FAMILY=='auto':
        estimable={key:value['spearman_minus_rank_vs_band_change'] for key,value in all_families.items()
            if np.isfinite(value['spearman_minus_rank_vs_band_change'])}
        family=max(estimable,key=estimable.get) if estimable else 'B'
    else:family=DISPLAY_FAMILY
    group=comparison['groups'][family]
    rank=np.asarray(group['template_rank'],float)
    rho=all_families[family]['spearman_minus_rank_vs_band_change']
    gs=spec.subgridspec(1,4,width_ratios=[1,.045,1,.045],wspace=.46)
    ax=fig.add_subplot(gs[0])
    ax.set_anchor('N')
    if np.isfinite(rank).any():
        old_figure.contact_field(ax,geo['contact_xy'],rank,geo['centers_mm'],
            float(geo['core_radius_mm']),Normalize(0,1),'viridis','',ylabel=True)
    else:
        ax.set(xlim=(0,20),ylim=(0,20),xlabel='x (mm)',ylabel='y (mm)')
        ax.text(.5,.5,f'{family} family not estimable',transform=ax.transAxes,ha='center')
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),cax=fig.add_subplot(gs[1]),ticks=[0,.5,1])
    cb.set_label(f'{family}-event contact rank')
    ax2=fig.add_subplot(gs[2])
    ax2.set_anchor('N')
    lim=max(float(np.abs(field_delta).max()),float(np.abs(contact_delta).max()),1e-12)
    exponent=int(np.floor(np.log10(lim))); divisor=10.**exponent
    limit=lim/divisor
    has_negative=min(field_delta.min(),contact_delta.min())<0
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit) if has_negative else Normalize(0,limit)
    cmap='RdBu_r' if has_negative else 'Blues'
    im=ax2.imshow((field_delta/divisor).reshape(20,20),origin='lower',extent=[0,20,0,20],
                  norm=norm,cmap=cmap,interpolation='nearest')
    ax2.scatter(*geo['contact_xy'].T,c=contact_delta/divisor,norm=norm,cmap=cmap,
                s=42,edgecolors='#333333',linewidths=.8,zorder=5)
    cores(ax2,geo)
    ax2.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20],yticklabels=[])
    cb2=fig.colorbar(im,cax=fig.add_subplot(gs[3]))
    cb2.set_label(f'ΔP, 1–150 Hz\n(mV equiv.)² ×10$^{{{exponent}}}$')
    fig.canvas.draw()
    for cb_axis,map_axis in [(cb.ax,ax),(cb2.ax,ax2)]:
        position=cb_axis.get_position();mp=map_axis.get_position()
        cb_axis.set_position([position.x0,mp.y0,position.width,mp.height])
    return dict(displayed_family=family,n_events=group['n_events'],rank=rank,
        all_family_correspondence=all_families,
        family_selection=('Largest signed rank/energy correlation among A and B in the fixed original baseline/early window. A selected within-model illustration, not independent validation; both scores retained, including negative scores. No temporal window optimization.'
            if DISPLAY_FAMILY=='auto' else 'Explicitly requested family; both family scores retained.'),
        template_observer={key:comparison[key] for key in ['event_definition','family_definition','template_definition']},
        model_contact_spearman_minus_rank_vs_band_change=rho,
        spectral_method=method,native_field_is_not_electrode_interpolation=True,
        rank_field_display='Gaussian2.5mm contact interpolation; contrast with direct1mm native energy field.',
        power_field=field_delta,contact_power=contact_delta,power_divisor=divisor,
        negative_values_clipped=False,patient_data_in_panel=False,
        scope='Illustrative selected within-model family order and first-entry band-energy correspondence; no patient match or independent validation claim.')


def main(name,grid_kind='timescale'):
    replay=BASE/'native_field_candidates_recurrence'/name
    qa=observation.read(replay/'observation_qa.json')
    assert qa['status']=='PASS' and qa['entire_checkpoint_bitwise']
    assert qa['actual_revised_Z_class_preserved'] and qa['all_added_resource_flux_and_pool_observations_bitwise']
    folder=replay/'runs'/name
    result=observation.read(folder/'result.json')
    source=SOURCE/'runs'/name
    assert result['no_external_intervention'] and not result['M_reset'] and not result['Z_reset']
    audit=observation.read(source/'finite_event_resource_audit.json')
    prior=observation.read(source/'figure_metadata.json')
    geo=dict(np.load(replay/'geometry.npz'))
    keys=['time_ms','spikes_1ms','regions_1ms','field_time_ms','field_5ms','raster','slow_time_ms','Z','M','currents']
    data=observation.load(folder,keys)
    state=dense(folder,['state_time_ms','state'])
    assert np.array_equal(data['spikes_1ms'][:,0],data['regions_1ms'][:,:3].sum(1))
    assert data['field_5ms'].sum()==data['spikes_1ms'][:,0].sum()
    end=float(result['display_stop_s']);job=result['job']
    assert np.isclose(end,result['tracker']['entries'][1]['confirmation_s']+2)
    snaps=snapshots(data,geo,result,audit,prior)
    grid=matched_grid(kind=grid_kind)
    plt.rcParams.update({'font.size':18,'axes.labelsize':20,'xtick.labelsize':17,'ytick.labelsize':17})
    fig=plt.figure(figsize=(30,19))
    outer=fig.add_gridspec(1,2,width_ratios=[1.18,1.05],left=.06,right=.948,
                           top=.95,bottom=.115,wspace=.24)
    left=outer[0].subgridspec(5,1,height_ratios=[2.5,1.,1.3,.12,1.25],hspace=.42)
    right=outer[1].subgridspec(2,1,height_ratios=[1.15,1],hspace=.42)
    upper=right[0].subgridspec(1,2,width_ratios=[1.28,1],wspace=.80)
    ra=fig.add_subplot(left[0]);letter(fig,left[0],'A')
    it,ix=np.nonzero(data['raster'][:round(end*10000)]);tt=it*.0001
    mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    row_colors=['#a86ca0','#357fa4','#225b7f','#c17730']
    for j,color in enumerate(row_colors):
        selected=(ix>=j*20)&(ix<(j+1)*20)
        ra.scatter(tt[selected],mapping[ix[selected]],s=7,c=color,lw=0,rasterized=True)
    for y in [34.5,70.5,85.5]:ra.axhline(y,c='#bbbbbb',lw=.7)
    ra.set(xlim=(0,end),ylim=(-2,101),yticks=[16.5,52.5,78,93],
           yticklabels=['Core A E','Core B E','Other E','I'])
    ra.tick_params(labelbottom=False)
    zooms=left[1].subgridspec(1,3,wspace=.42)
    zoom_records=[]
    for k,snap in enumerate([snaps[1],snaps[2],snaps[4]]):
        ax=fig.add_subplot(zooms[k]);lo=max(0,snap['time_s']-.05);hi=lo+.30
        for j,color in enumerate(row_colors[:2]):
            selected=(ix>=j*20)&(ix<(j+1)*20)&(tt>=lo)&(tt<hi)
            ax.scatter(tt[selected],ix[selected],s=17,marker='|',c=color,lw=.9,rasterized=True)
        ax.axhline(19.5,c='.7',lw=.6)
        ax.axvspan(snap['window_s'][0],snap['window_s'][1],color=snap['color'],alpha=.15)
        ax.set(xlim=(lo,hi),ylim=(-1,40),yticks=[9.5,29.5],
               yticklabels=['Core A E','Core B E'] if k==0 else [],xlabel='Time (s)')
        ax.set_xticks([lo+.05,lo+.15,lo+.25]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}'))
        ax.text(.02,.95,str(snap['number']),color=snap['color'],fontsize=18,weight='bold',transform=ax.transAxes,va='top',
                zorder=12,bbox=dict(facecolor='white',edgecolor='none',pad=1.5,alpha=.9))
        ra.add_patch(Rectangle((lo,-1),hi-lo,71,fill=False,ec=snap['color'],lw=1.6))
        zoom_records.append(dict(state=snap['number'],window_s=[lo,hi]))
    zm=fig.add_subplot(left[2],sharex=ra);ma=zm.twinx();letter(fig,left[2],'B')
    zt=data['slow_time_ms']/1000;take=zt<end
    zm.fill_between(zt[take],data['Z'][take,2],data['Z'][take,4],color='#814397',alpha=.13,lw=0)
    for column,color,label in [(0,'#814397','E mean Z'),(5,'#ce77ac','Core A Z'),(6,'#4b9ace','Core B Z')]:
        zm.plot(zt[take],data['Z'][take,column],c=color,lw=1.5,label=label)
    st=state['state_time_ms']/1000;use=st<end
    ma.plot(st[use],state['state'][use,3],c='#b66b26',lw=1.1,label='M current')
    zm.set(ylabel='Resource Z',xlabel='Time (s)',ylim=(0,1.035))
    ma.set_ylabel('M current (mV equiv.)',color='#b66b26');ma.set_ylim(bottom=0)
    handles,labels=zm.get_legend_handles_labels();h,l=ma.get_legend_handles_labels()
    zm.legend(handles+h,labels+l,loc='upper right',ncol=1,fontsize=12,framealpha=.94,edgecolor='none')
    zm.set_zorder(ma.get_zorder()+1);zm.patch.set_visible(False)
    for snap in snaps:
        for ax in [ra,zm]:ax.axvline(snap['time_s'],color=snap['color'],lw=1,ls=':')
        # Two adjacent early examples retain exact x positions; stagger text only.
        yy=1.045 if snap['number'] in [1,3] else .985
        ra.text(snap['time_s'],yy,str(snap['number']),transform=ra.get_xaxis_transform(),
                color=snap['color'],ha='center',va='top',fontsize=18,weight='bold',
                bbox=dict(fc='white',ec='none',alpha=.9,pad=.1))
    maps=left[4].subgridspec(1,6,wspace=.24);letter(fig,left[4],'C')
    map_axes=[]
    for k,snap in enumerate(snaps):
        ax=fig.add_subplot(maps[k]);map_axes.append(ax)
        im=ax.imshow(snap['rate_Hz'].reshape(20,20),origin='lower',extent=[0,20,0,20],
                     cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        cores(ax,geo)
        ax.set(xlabel='x (mm)',xticks=[0,20],yticks=[0,10,20])
        if k==0:ax.set_ylabel('y (mm)')
        else:ax.set_yticklabels([])
        ax.set_title(f'{snap["number"]}\n{snap["time_s"]:.2f} s',fontsize=16,color=snap['color'],pad=12)
        fig.add_artist(ConnectionPatch(xyA=(snap['time_s'],-.18),coordsA=zm.get_xaxis_transform(),
            xyB=(.5,1.26),coordsB=ax.transAxes,color=snap['color'],lw=1,alpha=.65,clip_on=False,zorder=-1))
    cbax=map_axes[0].inset_axes([0,-.47,1,.075])
    fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[0,250,500],label='E rate (Hz)')
    trajectory=draw_phase(fig,upper[0],state,data['spikes_1ms'],snaps,end);letter(fig,upper[0],'D')
    draw_grid(fig,upper[1],grid,job);letter(fig,upper[1],'E')
    energy=draw_energy(fig,right[1],data,geo,folder,result);letter(fig,right[1],'F')
    for ax in fig.axes:
        ax.tick_params(labelsize=17)
        for axis in [ax.xaxis,ax.yaxis]+([ax.zaxis] if hasattr(ax,'zaxis') else []):axis.label.set_fontsize(20)
    fig.canvas.draw()
    for snap in snaps:
        px=ra.get_xaxis_transform().transform((snap['time_s'],0))[0]
        qx=zm.get_xaxis_transform().transform((snap['time_s'],0))[0]
        assert abs(px-qx)<1e-6
    out=replay/'full_fig5';figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figures/f'fig5_autonomous_recurrence.{ext}',dpi=160,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)
    np.savez_compressed(out/'trajectory.npz',time_s=trajectory.pop('trajectory_time_s'),
                        coordinates=trajectory.pop('trajectory_coordinates'))
    write(out/'figure_metadata.json',dict(source=source,replay=replay,job=job,observation_qa=qa,
        display_time_window_s=[0,end],snapshots=snaps,raster_zoom_windows=zoom_records,
        parameter_grid=grid,trajectory=trajectory,early_energy=energy,
        autonomous_event_audit=audit,manual_intervention=False,
        panel_mapping={'A':'Native fixed80 raster; cores E only, I separately; exact boxed zooms2/3/5',
            'B':'NativeZ and actualM current, common time withA',
            'C':'Direct1mm E-rate50ms snapshots from the same trajectory',
            'D':'Actual1ms Z / effective inhibition / E-rate trajectory',
            'E':'Matched30s global-feedback gain with timescale or resource-equation control; see exact grid metadata',
            'F':'Selected model-family contact rank versus directly observed native early band-energy with contact readout; all family scores retained'},
        model_boundary=('Original nativeZ equation (rho0 branch) and nativeM, with added causal global feedback.'
                        if job['recovery_ratio']==0 else 'New concurrent recovery term rho.25 with added causal global feedback.')+
                       ' Neither model-only trajectory proves a clinical seizure or stable limit cycle.',
        agent_visual_review='PENDING',human_review='PENDING',producer=Path(__file__).resolve(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),updated_at=time.time()))
    equation_note='该候选保持原生Z方程，新增的是全局活动反馈。' if job['recovery_ratio']==0 else '该候选另含rho=.25资源恢复项。'
    (figures/'README.md').write_text('### fig5_autonomous_recurrence.png / .pdf\n同一手放双核、同一连续噪声轨迹，展示高活动、自主返回和再次越过高率判据；无Z/M reset，主图结束于第二次确认后2秒。A原生raster有2/3/5三个实际窗口放大，B显示Z与有效M电流，C是同一时刻原生细胞50ms场，D用精确重放的1ms状态；E用相同30秒的增益与时间常数或rho对照，具体轴和固定参数见metadata。\nF展示在固定窗口内与早期能量较相近的模型事件类别，色条标明A或B；两个类别的全部系数均保存，选择后的示例不能作独立验证。右侧背景直接来自细胞网格、圆点独立来自电极读出，均为原生10kHz计算的1–150Hz能量变化；不是患者Fig3图。**关注点**：返回到安静与恢复原先间期传播分布须分别判断；'+equation_note+'连续轨迹和参数格不构成Hopf或稳定极限环证明，仍待人工目视。\n')
    print(figures/'fig5_autonomous_recurrence.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--source-root',type=Path,default=SOURCE)
    parser.add_argument('--grid',choices=['timescale','resource'],default='timescale')
    parser.add_argument('--family',choices=['A','B','auto'],default='auto')
    args=parser.parse_args();SOURCE=args.source_root;DISPLAY_FAMILY=args.family
    main(args.name,args.grid)

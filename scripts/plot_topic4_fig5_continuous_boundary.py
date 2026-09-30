#!/usr/bin/env python3
"""Continuous display of the completed, fixed-seed first-entry parameter scan.

Interpolate log10(min(T_entry, 1000 s)) on a triangulation in log-parameter
coordinates. The boundary connects the log-midpoints of measured entry / no
entry brackets using shape-preserving cubic interpolation in log coordinates.
"""
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation
import analyze_topic4_fig5_boundary_refinement as refinement

MODE = 'continuous_log_bracket_boundary_v1'


def surface_only(grid):
    """Record the author's display choice without changing any scan values."""
    assert grid['display_mode'] == MODE
    grid['display_overlays'] = False
    grid['display_style'] = dict(
        author_request='2026-09-23: show continuous Fig5E surface without sampling points or boundary.',
        sampling_points=False, boundary_line=False, hatching=False,
        working_point=False, legend=False,
        producer=str(Path(__file__).resolve()),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return grid


def prepare(grid):
    assert grid['all_complete'] and grid['base_grid']['all_complete']
    records = grid['base_grid']['records'] + grid['refinement_records']
    records = sorted(records, key=lambda r: (r['job']['tau_M_s'], r['job']['eta_m']))
    assert len(records) == 59 and all(r['complete'] for r in records)
    assert {r['job']['seed'] for r in records} == {9108401}
    xy = np.array([[r['job']['tau_M_s'], r['job']['eta_m']] for r in records])
    assert len(np.unique(xy, axis=0)) == 59
    entered = np.array([r['event_observed'] for r in records], dtype=bool)
    times = np.array([r['first_entry']['confirmation_s'] if r['event_observed'] else r['followup_s'] for r in records])
    assert np.all(times[~entered] == 1000) and np.all(times[entered] <= 1000)
    logxy = np.log10(xy)
    tri = mtri.Triangulation(logxy[:, 0], logxy[:, 1])
    time_interpolator = mtri.LinearTriInterpolator(tri, np.log10(times))
    state_interpolator = mtri.LinearTriInterpolator(tri, entered.astype(float))
    time_error = float(np.max(abs(time_interpolator(*logxy.T) - np.log10(times))))
    state_error = float(np.max(abs(state_interpolator(*logxy.T) - entered)))
    assert time_error < 1e-10 and state_error < 1e-10
    brackets = []
    for tau in np.unique(xy[:, 0]):
        select = xy[:, 0] == tau
        levels, state = xy[select, 1], entered[select]
        # Audit monotonicity within each measured column, without constraining
        # the shape across tau (including the observed tau=1000 s depression).
        assert np.all(np.diff(state.astype(int)) <= 0)
        assert state.any() and (~state).any()
        low, high = float(levels[state].max()), float(levels[~state].min())
        brackets.append(dict(tau_M_s=float(tau), largest_entered_eta_M=low,
                             smallest_censored_eta_M=high, log_midpoint_eta_M=float(np.sqrt(low*high))))
    bt = np.log10([b['tau_M_s'] for b in brackets])
    be = np.log10([b['log_midpoint_eta_M'] for b in brackets])
    boundary = PchipInterpolator(bt, be, extrapolate=False)
    boundary_at_samples = boundary(logxy[:, 0])
    assert np.array_equal(logxy[:, 1] < boundary_at_samples, entered)
    bx = np.linspace(bt.min(), bt.max(), 1201)
    grid['display_mode'] = MODE
    grid['continuous_surface'] = dict(
        parameters=xy.tolist(), log10_parameters=logxy.tolist(), triangles=tri.triangles.tolist(),
        entry_observed=entered.tolist(), time_or_lower_bound_s=times.tolist(),
        source_records=[dict(name=r['job']['name'], source=r['source'], result_sha256=hashlib.sha256((Path(r['source'])/'result.json').read_bytes()).hexdigest()) for r in records],
        n_total=59, n_entered=int(entered.sum()), n_censored=int((~entered).sum()),
        entry_range_s=[float(times[entered].min()), float(times[entered].max())],
        boundary_brackets=brackets, interpolation_at_samples_max_error_log10_s=time_error,
        interpolation_at_samples_max_error_binary=state_error,
        boundary_classification_matches_all59_samples=True,
        boundary_curve_tau_M_s=(10**bx).tolist(), boundary_curve_eta_M=(10**boundary(bx)).tolist(),
        interpolation='Piecewise-linear barycentric interpolation of log10(min(T_entry,1000 s)) in log10(tau_M),log10(eta_M); no extrapolation outside the measured convex hull.',
        boundary='For each of8 measured tau columns, bracket the largest entered eta and smallest censored eta. Connect their geometric midpoints by shape-preserving PCHIP in log10(tau),log10(eta). All59 sample classifications are preserved. The line is an interpolated finite-window estimate, not a probability, confidence interval, exact separatrix, or asymptotic bifurcation.',
        censoring='All non-entry samples observed through1000 s. Their color is the1000 s cap/lower bound, not an observed entry at1000 s.',
        marker_key='Circle: entered. Open triangle: no entry by1000 s. Star: A-D/F model working point.',
        author_request='2026-09-23: replace discrete background plus added dots with a continuous parameter map and explicit boundary using the new points.',
        producer=str(Path(__file__).resolve()), producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    grid['interpretation'] = grid['continuous_surface']['interpolation'] + ' ' + grid['continuous_surface']['boundary']
    return surface_only(grid)


def collect():
    return prepare(refinement.collect())


def draw(fig, spec, grid, job=None):
    surface = grid['continuous_surface']
    logxy = np.asarray(surface['log10_parameters'])
    xy = np.asarray(surface['parameters'])
    entered = np.asarray(surface['entry_observed'], dtype=bool)
    values = np.asarray(surface['time_or_lower_bound_s'])
    tri = mtri.Triangulation(*logxy.T, triangles=np.asarray(surface['triangles']))
    gx = np.linspace(logxy[:, 0].min(), logxy[:, 0].max(), 601)
    gy = np.linspace(logxy[:, 1].min(), logxy[:, 1].max(), 601)
    xx, yy = np.meshgrid(gx, gy)
    logtime = mtri.LinearTriInterpolator(tri, np.log10(values))(xx, yy)
    boundary_y = None
    if grid.get('display_overlays', True):
        brackets = surface['boundary_brackets']
        boundary = PchipInterpolator(np.log10([b['tau_M_s'] for b in brackets]),
                                    np.log10([b['log_midpoint_eta_M'] for b in brackets]), extrapolate=False)
        boundary_y = boundary(gx)
    ax = fig.add_subplot(spec)
    ax.set_box_aspect(1)
    norm = LogNorm(1, 1000)
    im = ax.pcolormesh(10**gx, 10**gy, 10**logtime, shading='nearest',
                       cmap='viridis', norm=norm, rasterized=True, linewidth=0)
    if grid.get('display_overlays', True):
        # Retain the previous presentation for archived snapshots.
        ax.fill_between(10**gx, 10**boundary_y, 10**gy.max(), facecolor='none',
                        edgecolor='#77777766', hatch='//', linewidth=0)
        ax.plot(10**gx, 10**boundary_y, color='black', linewidth=1.8)
        ax.scatter(*xy[entered].T, c=values[entered], norm=norm, cmap='viridis',
                   s=20, edgecolors='white', linewidths=.6, zorder=6, clip_on=False)
        ax.scatter(*xy[~entered].T, facecolors='none', edgecolors='#333333',
                   marker='^', s=21, linewidths=.7, zorder=6, clip_on=False)
        if job is not None:
            ax.scatter([job['tau_M_s']], [job['eta_m']], marker='*', s=125,
                       facecolor='white', edgecolor='black', linewidth=1., zorder=8, clip_on=False)
    ax.set(xscale='log', yscale='log', xlim=(xy[:,0].min(),xy[:,0].max()),
           ylim=(xy[:,1].min(),xy[:,1].max()), xlabel=r'$\tau_M$ (s)', ylabel=r'$\eta_M$',
           xticks=[1,10,100,1000,10000], yticks=[.0001,.001,.01,.1,1,10])
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.minorticks_off()
    if grid.get('display_overlays', True):
        ax.legend(handles=[Line2D([],[],ls='-',color='black',lw=1.8,label='Entry boundary'),
                           Line2D([],[],ls='',marker='o',mfc='#31688e',mec='white',label='Entered'),
                           Line2D([],[],ls='',marker='^',mfc='none',mec='#333333',label='No entry by 1000 s')],
                  loc='upper right', fontsize=11, frameon=True, facecolor='white',
                  edgecolor='none', framealpha=.90, handlelength=1.4, borderpad=.5)
    slot=spec.get_position(fig)
    fig.text(slot.x0,slot.y1+.012,'E',weight='bold',fontsize=24,ha='left')
    cb=fig.colorbar(im,cax=ax.inset_axes([1.06,0,.055,1]))
    cb.set_label('Entry time / lower bound (s)')
    cb.set_ticks([1,10,100,1000])
    cb.formatter=LogFormatterSciNotation(base=10,labelOnlyBase=False,minor_thresholds=(np.inf,np.inf))
    cb.update_ticks();cb.ax.minorticks_off()
    ax._boundary_segments = [] if boundary_y is None else [np.c_[10**gx, 10**boundary_y].tolist()]
    return ax

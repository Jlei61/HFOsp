#!/usr/bin/env python3
"""Fig5 layout revision: physical axes alignment with frozen native painters."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch
from matplotlib.transforms import Bbox
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from scipy.stats import rankdata
try:
    import pymupdf as fitz
except ImportError:
    sys.path.insert(0, '/tmp/hfosp_fig5_pdf_tools')
    import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/paper-ready-figure/fig5/candidates/vertical_legend_20261027'
EXP = Path('/data/hfosp/topic4_sef_hfo/cohort_core_count_20261024')
RED, BLUE, PURPLE = '#C63D3A', '#287FA1', '#75628F'
DISPLAY = [f'SCL{i}' for i in range(9, 5, -1)] + [f'ICL{i}' for i in range(11, 0, -1)]
SHAFT = {'SCL': '#159CAC', 'ICL': '#E08229'}
W, H = 310., 154.
LABEL = TICK = CONTACT = LEGEND = 11.


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def write(p, data):
    Path(p).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def frozen_functions(path, names, **overrides):
    """Execute unchanged native function bodies, without legacy import side effects."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    namespace = dict(globals(), **overrides)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def rect(x, top, width, height):
    return [x/W, (H-top-height)/H, width/W, height/H]


def style_axis(ax, *, contacts=False):
    ax.tick_params(labelsize=TICK, pad=2.5, length=3)
    ax.xaxis.label.set_size(LABEL)
    ax.yaxis.label.set_size(LABEL)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 3
    if contacts:
        ax.tick_params(axis='y', labelsize=CONTACT, pad=2)


def native(fig, function, *args):
    before = set(fig.axes)
    with patch.object(plt, 'figure', return_value=fig):
        result = function(*args)
    return [ax for ax in fig.axes if ax not in before], result


def draw_abc_ef(fig, src, physics, previous, groups):
    original = frozen_functions(src/'build_fig5_propagation_split.py',
        ['schematic', 'core_outlines'], export=lambda *a: None)
    axes, info = native(fig, original['schematic'], src, src, physics, {})
    np.testing.assert_equal(info, previous['panel_a'])
    a = axes[0]
    a.set_position(rect(18, 18, 50, 50))
    style_axis(a)
    ellipse = next(p for p in a.patches if type(p) is Ellipse)
    theta = np.deg2rad(ellipse.angle)
    half_height = np.hypot(ellipse.width/2*np.sin(theta), ellipse.height/2*np.cos(theta))
    axis_label_position = [ellipse.center[0], ellipse.center[1]-half_height-.65]
    for t in a.texts:
        t.set_fontsize(LABEL)
        if t.get_text() == 'E→E axis':
            t.set_position(axis_label_position)
            t.set(ha='center', va='top')
    for t in a.legend_.texts:
        t.set_fontsize(LEGEND)
    a.legend_.set_title(None)
    groups['A'] = axes

    painter = frozen_functions(src/'build_fig4_geometry_patient_comparison.py',
        ['waveform_panel', 'rank_panel', 'matrix_panel'], MM=1/25.4,
        SOURCE=src, save=lambda *a: None)
    z = dict(np.load(src/'comparison_arrays.npz'))
    meta = read(src/'comparison_metadata.json')
    fields = ['contact_names', 'contact_envelope', 'contact_envelope_dt_ms', 'centroid_ms',
              'recruitment_ms', 'primary_event_indices', 'event_mode', 'event_time_ms']
    with np.load(src/'trajectory.npz') as raw:
        arrays = {k: raw[k] for k in fields}
    axes, wave = native(fig, painter['waveform_panel'], read(src/'trajectory.json'), arrays,
                        z['model_event_ids'], z['display_order'])
    np.testing.assert_equal(wave, meta['waveform'])
    with np.load(src/'accepted_waveform_arrays.npz') as old, np.load(src/'waveform_arrays.npz') as new:
        for k in old.files:
            np.testing.assert_array_equal(old[k], new[k])
    b = axes[0]
    b.set_position(rect(95, 18, 133, 50))
    style_axis(b, contacts=True)
    b.set_xticks([0, 200, 400, 600, 800])
    shaded = {s['event'] for s in wave['spans']}
    start = wave['absolute_window_ms'][0]
    extra = []
    # Accepted Fig4F includes every visible event, in addition to the selected pair.
    for event in wave['all_primary_ids_visible']:
        if event in shaded:
            continue
        mode = int(arrays['event_mode'][event])
        valid = arrays['recruitment_ms'][event]
        valid = valid[np.isfinite(valid)]
        lo, hi = float(valid.min()-start-12), float(valid.max()-start+12)
        b.axvspan(lo, hi, color=RED if mode == 1 else BLUE, alpha=.14, lw=0, zorder=-2)
        extra.append(dict(event=event, mode=mode, span_ms=[lo, hi]))
    b.legend_.remove()
    b.legend(handles=[Patch(fc=c, alpha=.22, label=s) for c, s in [(RED, 'MTA'), (BLUE, 'MTB')]],
             loc='upper right', ncol=2, fontsize=LEGEND, frameon=True, facecolor='white',
             edgecolor='none', framealpha=.9, borderpad=.2, borderaxespad=.2,
             handlelength=1.1, handletextpad=.4, columnspacing=.8)
    groups['B'] = axes

    axes, _ = native(fig, painter['rank_panel'], z['display_order'],
                     z['model_profiles_raw_mode_0_1'], z['patient_profiles_raw_mode_0_1'])
    c = axes[0]
    c.set_position(rect(255, 18, 50, 50))
    style_axis(c, contacts=True)
    c.set_xticks([0, 7, 14])
    fig.legends[-1].remove()
    c.legend(handles=[Line2D([], [], color=color, ls=ls, marker=m, lw=1.2, ms=2.8, label=s)
        for color, ls, m, s in [(RED, '-', 'o', 'MTA'), (RED, '--', None, 'TA'),
                               (BLUE, '-', 'o', 'MTB'), (BLUE, '--', None, 'TB')]],
        ncol=2, loc='lower right', bbox_to_anchor=(1, 1.005), fontsize=LEGEND,
        frameon=False, handlelength=1.3, handletextpad=.3, columnspacing=.7,
        labelspacing=.1, borderpad=0, borderaxespad=0)
    groups['C'] = axes

    axes, _ = native(fig, painter['matrix_panel'],
                     {'matrix': z['crossfit_matrix_raw_mode_0_1']}, meta['diagonal_tests'])
    e, cb = axes
    e.set_position(rect(174, 92, 50, 50))
    cb.set_position(rect(226, 92, 2, 50))
    style_axis(e)
    style_axis(cb)
    cb.set_ylabel('')
    cb.set_yticks([-1, 0, 1])
    e.set_title('Spearman ρ', fontsize=LABEL, pad=5)
    for t in e.get_xticklabels()+e.get_yticklabels():
        t.set_fontsize(LEGEND)
        t.set_fontweight('bold')
    for t in e.texts:
        t.set_fontsize(LABEL)
    groups['E'] = axes

    f = fig.add_axes(rect(255, 92, 50, 50))
    display_path = src/'cohort_display.json'
    display = read(display_path) if display_path.exists() else {}
    report_path = EXP/'template_recovery_summary.json'
    interim_path = src/'interim_core_count_summary.json'
    interim = interim_path.exists()
    complete = not interim and report_path.exists() and read(report_path)['status'] == 'TRAINING_COMPLETE_REVIEW_PENDING'
    if interim:
        report = read(interim_path)
        assert report['status'] == 'AUTHOR_REQUESTED_INTERIM' and not report['cohort_comparison_complete']
        assert len({r['training_used'] for r in report['subject_arms']}) == 1
        cohort = draw_completed_cohort(f, select_cohort(report, src), display)
    elif complete:
        cohort = draw_completed_cohort(f, select_cohort(read(report_path), src), display)
        shutil.copy2(report_path, src/'completed_core_count_summary.json')
    else:
        cohort_painter = frozen_functions(src/'build_fig4_template_recovery_by_mode.py', ['draw_modes'],
            COLORS=[RED, BLUE], LIGHT=['#EAB7B5', '#B0D3E1'])
        cohort = cohort_painter['draw_modes'](f, read(src/'template_recovery_summary.json'), legend_loc='upper right')
        for t in f.legend_.texts:
            t.set_fontsize(LEGEND)
    style_axis(f)
    f.tick_params(axis='x', labelsize=LEGEND, pad=2, length=0)
    groups['F'] = [f]
    return dict(waveform_exact_array_parity=True, waveform_metadata_exact_parity=True,
                A_axis_label_position_mm=axis_label_position,
                A_axis_label_gap_below_ellipse_mm=.65,
                additional_shading=extra, rank_profiles=z['model_profiles_raw_mode_0_1'],
                patient_rank_profiles=z['patient_profiles_raw_mode_0_1'],
                crossfit_matrix=z['crossfit_matrix_raw_mode_0_1'], cohort=cohort,
                cohort_comparison_complete=complete, cohort_interim=interim)


def select_cohort(report, src):
    """Apply a frozen patient eligibility list without reselecting any workpoint."""
    selection_path = src/'cohort_selection.json'
    if not selection_path.exists():
        return report
    selection = read(selection_path)
    subjects = selection['included_subjects']
    assert subjects and len(subjects) == len(set(subjects))
    lookup = {(r['subject'], r['core_count']): r for r in report['subject_arms']}
    assert len(lookup) == len(report['subject_arms'])
    rows = [lookup[sid, k] for sid in subjects for k in (1, 2)]
    filtered = dict(report, subject_arms=rows, subject_denominator=len(subjects),
                    cohort_selection=selection,
                    parent_subject_denominator=report['subject_denominator'])
    if 'confirmations' in report:
        filtered['parent_confirmation_count'] = len(report['confirmations'])
        filtered['confirmations'] = [r for r in report['confirmations'] if r['subject'] in subjects]
    filtered['statistics'] = []
    for k in (1, 2):
        for mode, name in enumerate(['TA–MTA', 'TB–MTB']):
            values = np.asarray([lookup[sid, k]['matched_rhos'][mode] for sid in subjects], float)
            finite = values[np.isfinite(values)]
            filtered['statistics'].append(dict(core_count=k, mode=name, n=len(finite),
                missing=len(values)-len(finite), median=float(np.median(finite)) if len(finite) else None,
                q25_q75=np.quantile(finite, [.25, .75]).tolist() if len(finite) else [None, None]))
    if 'training_used' in report:
        filtered['parent_training_used'] = report['training_used']
        filtered['training_used'] = sum(r['training_used'] for r in rows)
    write(src/'displayed_core_count_summary.json', filtered)
    columns = ['subject', 'core_count', 'training_used', 'training_completed', 'candidate',
               'selected_epoch', 'training_J', 'status', 'n_events', 'rho_TA_MTA', 'rho_TB_MTB']
    with (src.parent/'core_count_subjects.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k) for k in columns[:-2]},
                'rho_TA_MTA': row['matched_rhos'][0], 'rho_TB_MTB': row['matched_rhos'][1]})
    return filtered


def template_match_values(rhos, view):
    values = np.asarray(rhos, float)
    if view == 'named_templates':
        return values
    assert view == 'stronger_weaker'
    # A joint match requires both signed correlations; never skip a missing mode.
    if not np.isfinite(values).all():
        return np.full(3, np.nan)
    return np.asarray([values.max(), values.min(), abs(values[0]-values[1])])


def paired_core_statistics(report, view='named_templates'):
    """Patient-paired signed-rank tests, with exact tied-rank nulls and Holm correction."""
    rows = report['subject_arms']
    subjects = [r['subject'] for r in rows if r['core_count'] == 1]
    lookup = {(r['subject'], r['core_count']): r for r in rows}
    assert len(lookup) == len(rows) == 2*len(subjects)
    tests = []
    names = ['Stronger', 'Weaker', 'Gap'] if view == 'stronger_weaker' else ['TA–MTA', 'TB–MTB']
    for mode, name in enumerate(names):
        pairs = np.asarray([[template_match_values(lookup[sid, k]['matched_rhos'], view)[mode] for k in (1, 2)]
                            for sid in subjects], float)
        valid = np.isfinite(pairs).all(axis=1)
        # Remove floating point artifacts before ranking tied absolute differences.
        differences = np.round(pairs[valid, 1]-pairs[valid, 0], 12)
        nonzero = differences[differences != 0]
        weights = np.rint(2*rankdata(np.abs(nonzero), method='average')).astype(int)
        total = int(weights.sum())
        positive = int(weights[nonzero > 0].sum())
        # Dynamic programming enumerates all 2**n sign assignments, including ties.
        counts = [1]+[0]*total
        for weight in weights:
            for j in range(total, int(weight)-1, -1):
                counts[j] += counts[j-int(weight)]
        tail = min(positive, total-positive)
        p = min(1., 2*sum(counts[:tail+1])/2**len(nonzero)) if len(differences) else None
        tests.append(dict(mode=name, n_pairs=int(valid.sum()), missing_pairs=int((~valid).sum()),
            paired_subjects=np.asarray(subjects)[valid].tolist(), difference='2 cores minus 1 core',
            paired_differences=differences, median_paired_difference=float(np.median(differences)) if len(differences) else None,
            zero_differences=int((differences == 0).sum()), absolute_rank_ties=len(nonzero)-len(np.unique(np.abs(nonzero))),
            W=tail/2, p_raw=p, p_holm=None,
            rank_biserial=(2*positive-total)/total if total else None))
    order = sorted([i for i, t in enumerate(tests) if t['p_raw'] is not None], key=lambda i: tests[i]['p_raw'])
    previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(tests)-rank)*tests[i]['p_raw']))
        tests[i]['p_holm'] = previous
    return dict(test='Two-sided paired Wilcoxon signed-rank test',
        p_method='Exact conditional sign-permutation distribution of average ranks; dynamic programming',
        zero_method='wilcox: omit zero differences', difference_rounding_decimals=12,
        multiple_comparison='Holm correction across Stronger, Weaker, and Gap; only Stronger and Weaker are drawn' if view == 'stronger_weaker' else 'Holm correction across the two template comparisons',
        family_size=len(tests), statistical_unit='patient', view=view,
        interpretation='Post-hoc asymmetric selection-policy comparison; the same template correlations enter core1 eligibility and scoring; no independent validation' if report.get('selection_policy') else 'Post-hoc exploratory joint-template comparison; no sequential-testing adjustment' if view == 'stronger_weaker' else 'Exploratory fixed-snapshot comparison; no sequential-testing adjustment' if report['status'] == 'AUTHOR_REQUESTED_INTERIM' else 'Matched-budget cohort comparison',
        null='Patient-paired differences are symmetric about zero', comparisons=tests)


def draw_completed_cohort(ax, report, display=None):
    """Matched core-count readout on the common physical canvas."""
    rows = report['subject_arms']
    subjects = [r['subject'] for r in rows if r['core_count'] == 1]
    assert len(subjects) == len(set(subjects)) == report['subject_denominator']
    display = display or {}
    view = display.get('view', 'named_templates')
    joint = view == 'stronger_weaker'
    names = ['Stronger', 'Weaker', 'Gap'] if joint else ['TA–MTA', 'TB–MTB']
    colors = ['#52565A', PURPLE] if joint else [RED, BLUE]
    lookup = {(r['subject'], r['core_count']): template_match_values(r['matched_rhos'], view) for r in rows}
    jitter = np.random.default_rng(20261024).uniform(-.08, .08, len(subjects))
    for mode in (0, 1):
        color = colors[mode]
        for j, sid in enumerate(subjects):
            pair = [lookup[sid, k][mode] for k in (1, 2)]
            if all(v is not None and np.isfinite(v) for v in pair):
                ax.plot(np.array([mode*3, mode*3+1])+jitter[j], pair, color='.82', lw=.35, zorder=0)
        for k in (1, 2):
            values = np.array([lookup[sid, k][mode] for sid in subjects], float)
            x = mode*3+k-1
            ax.scatter(x+jitter, values, s=13, facecolors='white' if k == 1 else color, edgecolors=color, lw=.45)
            finite = values[np.isfinite(values)]
            if len(finite):
                q1, med, q3 = np.quantile(finite, [.25, .5, .75])
                ax.plot([x+.16, x+.16], [q1, q3], color=color, lw=1)
                ax.plot([x+.07, x+.25], [med, med], color=color, lw=1.3)
    ax.axhline(0, color='.8', ls='--', lw=.6)
    statistics = paired_core_statistics(report, view)
    for mode, result in enumerate(statistics['comparisons'][:2]):
        x = mode*3
        symbolic = display.get('significance_style') == 'stars'
        if (symbolic and not display.get('show_nonsignificant', True)
                and result['p_holm'] is not None and result['p_holm'] >= .05):
            continue
        finite_values = [lookup[sid, k][mode] for sid in subjects for k in (1, 2)
                         if lookup[sid, k][mode] is not None and np.isfinite(lookup[sid, k][mode])]
        bracket_bottom = max(display.get('significance_bracket_bottom', 1.06),
                             max(finite_values, default=0.)+.06)
        bracket_top = bracket_bottom+.06
        bracket, = ax.plot([x, x, x+1, x+1],
                           [bracket_bottom, bracket_top, bracket_top, bracket_bottom], color='.2', lw=.65)
        bracket.set_gid('cohort-significance-bracket')
        label = f"P = {result['p_holm']:.3f}" if result['p_holm'] is not None else 'Not estimable'
        if symbolic and result['p_holm'] is not None:
            label = '*' * sum(result['p_holm'] < t for t in (.05, .01, .001)) or 'n.s.'
        ax.text(x+.5, bracket_top+.04, label, ha='center', va='bottom',
                fontsize=15 if symbolic and '*' in label else LABEL,
                weight='bold' if symbolic and '*' in label else 'normal',
                gid='cohort-significance-label')
    ax.set(xlim=(-.55, 4.55), ylim=(-1, 1.43), yticks=[-1, 0, 1], xticks=[0, 1, 3, 4], xticklabels=['1', '2', '1', '2'],
           ylabel='Template similarity (ρ)', xlabel='cores number')
    if display.get('core_labels') == 'shared_legend':
        ax.set(xticks=[], xlabel='')
        handles = [Line2D([], [], marker='o', linestyle='none', markersize=4,
                          markerfacecolor='white' if k == 1 else '.3',
                          markeredgecolor='.3', markeredgewidth=.65,
                          label='1 core' if k == 1 else '2 cores') for k in (1, 2)]
        upper_right = display.get('core_legend_position') == 'upper_right'
        ax.legend(handles=handles, loc='upper right' if upper_right else 'upper center',
                  bbox_to_anchor=display.get('core_legend_anchor', (1., 1.105)) if upper_right else (.5, -.025),
                  ncol=display.get('core_legend_columns', 2), frameon=False, fontsize=LEGEND, borderaxespad=0,
                  labelspacing=.25,
                  borderpad=0 if upper_right else .4,
                  handlelength=.8, handletextpad=.35, columnspacing=1.)
    if display.get('template_label_position') == 'x_ticks':
        ax.set(xticks=[.5, 3.5], xticklabels=names[:2])
        for label, color in zip(ax.get_xticklabels(), colors):
            label.set_color(color)
    else:
        for x, text, color in zip([.5, 3.5], names[:2], colors):
            ax.text(x, 1.04, text, transform=ax.get_xaxis_transform(), ha='center', color=color, fontsize=LEGEND)
    interim = report['status'] == 'AUTHOR_REQUESTED_INTERIM'
    # Patient count and interim status remain in the caption and metadata.
    source = 'displayed_core_count_summary.json' if report.get('cohort_selection') else 'interim_core_count_summary.json' if interim else 'completed_core_count_summary.json'
    summaries = []
    for k in (1, 2):
        for mode, name in enumerate(names):
            values = np.asarray([lookup[sid, k][mode] for sid in subjects])
            finite = values[np.isfinite(values)]
            summaries.append(dict(core_count=k, mode=name, n=len(finite), missing=len(values)-len(finite),
                median=float(np.median(finite)) if len(finite) else None,
                q25_q75=np.quantile(finite, [.25, .75]).tolist() if len(finite) else [None, None]))
    if joint:
        with (OUT/'core_count_matching_subjects.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['subject', 'core_count', 'rho_stronger', 'rho_weaker', 'rho_gap'])
            for sid in subjects:
                for k in (1, 2):
                    writer.writerow([sid, k, *lookup[sid, k]])
    return dict(subjects=subjects, source=source, cohort_selection=report.get('cohort_selection'),
                common_budget=report.get('common_budget', 32), interim=interim,
                display_contract=display, displayed_metrics=names[:2],
                selection_policy=report.get('selection_policy'),
                statistics=summaries, paired_statistics=statistics,
                named_template_paired_statistics=paired_core_statistics(report))


def draw_d(fig, src, physics, previous, groups, notes):
    original = frozen_functions(src/'build_fig5_propagation_split.py', ['core_outlines'])
    cmap = LinearSegmentedColormap.from_list('model_activity', ['#fbfcfd', '#d4e4eb', '#83b3c9', '#397b9c', '#15364e'])
    norm = Normalize(vmin=0, vmax=1)
    with np.load(src/'rigid_contact_geometry.npz') as z:
        xy = z['rigid_xy_mm']
    with np.load(src/'trajectory.npz') as z:
        dt = float(z['sheet_activity_frame_ms'])
        assert dt == 2
        smooth = uniform_filter1d(z['sheet_activity_counts'].astype(float), size=5, axis=0, mode='nearest')
    # Keep the two historical events and the 1%-of-total-peak initiation rule.
    # Shared elapsed times; author requested an early local frame instead of decay.
    events = []
    for item in previous['panel_d']['events']:
        lo, hi = np.rint(np.asarray(item['window_ms'])/dt).astype(int)
        activity = smooth[lo:hi].sum((1, 2))
        active = np.flatnonzero(activity >= .01*activity.max())
        onset, end = lo+int(active[0]), lo+int(active[-1])+1
        assert onset*dt == item['first_frame_ms']
        events.append(dict(mode=item['mode'], event=item['event'], window_ms=item['window_ms'],
            first_frame_ms=onset*dt, onset_index=onset, end_relative_ms=(end-onset)*dt,
            end_index_exclusive=hi, peak_total_spikes=float(activity.max()),
            normalization_denominator=float(smooth[lo:hi].max())))
    relative_times = np.asarray([0., 16., 32., 64., 96.])
    offsets = np.rint(relative_times/dt).astype(int)
    np.testing.assert_array_equal(offsets*dt, relative_times)
    saved = {'relative_times_ms': relative_times}
    axes, text = [], []
    for row, item in enumerate(events):
        indices = item['onset_index']+offsets
        assert indices[-1] < item['end_index_exclusive']
        raw = smooth[indices]
        images = raw/item['normalization_denominator']
        assert images.min() >= 0 and images.max() <= 1
        np.testing.assert_allclose(images*item['normalization_denominator'], raw, rtol=0, atol=2e-14)
        item.update(relative_times_ms=relative_times, absolute_times_ms=indices*dt,
                    frame_indices=indices, total_spikes_at_frames=raw.sum((1, 2)),
                    final_to_peak_fraction=float(raw[-1].sum()/item['peak_total_spikes']))
        saved[item['mode']+'_raw_frames'] = raw
        saved[item['mode']+'_normalized_frames'] = images
        top = 92+row*27.5
        for col, (values, t) in enumerate(zip(images, relative_times)):
            ax = fig.add_axes(rect(13+col*27.5, top, 22.5, 22.5))
            im = ax.imshow(values, origin='lower', extent=[-10, 10, -10, 10], cmap=cmap, norm=norm, interpolation='nearest')
            original['core_outlines'](ax, physics, color='#775b83', lw=.55)
            ax.scatter(*xy.T, s=1.8, fc='white', ec='#6d767a', lw=.2, zorder=6)
            ax.set(xlim=(-10, 10), ylim=(-10, 10), xticks=[], yticks=[])
            ax.text(.035, .965, f'{t:g} ms', transform=ax.transAxes,
                    ha='left', va='top', fontsize=LEGEND,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=.2), zorder=8)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#ccd1d5')
                spine.set_linewidth(.45)
            if row == 1 and col == 0:
                ax.plot([-8, -3], [-8, -8], color='#333333', lw=1.1)
                ax.text(-4.8, -6.8, '5 mm', ha='center', fontsize=LABEL)
            axes.append(ax)
        text.append(fig.text(11/W, 1-(top+11.25)/H, item['mode'], ha='right', va='center',
                             fontsize=LABEL, color=RED if row == 0 else BLUE, weight='bold'))
    cax = fig.add_axes(rect(148, 92, 2, 50))
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.ax.set_yticklabels(['0', '0.5', '1'])
    cb.ax.set_title('Norm.\nFiring rate', fontsize=LABEL, pad=8, linespacing=1)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=TICK, length=2, pad=2)
    axes.append(cax)
    groups['D'] = axes
    notes['D'] = text
    np.savez_compressed(src/'shared_time_propagation_frames.npz', **saved)
    return dict(events=events, shared_relative_times_ms=relative_times,
        frame_intervals_ms=np.diff(relative_times), equally_spaced=False,
        native_bin_ms=dt, temporal_display_average_ms=10, spatial_bin_mm=1,
        spatial_interpolation='none', spatial_smoothing='none', color_range=[0, 1],
        normalization='Each event uses its maximum spatial-bin activity over its full frozen event window; one denominator per row, shared across all times.',
        normalization_denominators={e['mode']: e['normalization_denominator'] for e in events},
        per_frame_normalization=False, between_event_absolute_amplitude_comparable=False,
        color_mapping='linear', colorbar_label='Norm. Firing rate',
        selection='Same two historical events; author requested the 16-ms native activity frame instead of the 10-ms frame. Common elapsed milliseconds in both rows; nonuniform intervals explicitly labelled.',
        early_frame_scope='The 16-ms frame displays early propagation around the initiating core. Activity is not required to remain strictly inside the core disk.',
        end_of_event_shown=False,
        time_origin='Separate 1%-total-peak initiation times; identical relative elapsed milliseconds in corresponding columns; no time rescaling.',
        raw_frame_unit='E spikes per 2 ms per 1 mm², averaged across five consecutive native bins; omitted from normalized display.')


def export(fig, path, *, bbox=None):
    fig.savefig(path.with_suffix('.pdf'), bbox_inches=bbox, pad_inches=0, facecolor='white')
    fig.savefig(path.with_suffix('.svg'), bbox_inches=bbox, pad_inches=0, facecolor='white')
    with fitz.open(path.with_suffix('.pdf')) as doc:
        doc[0].get_pixmap(dpi=220, alpha=False).save(path.with_suffix('.png'))


def export_all(fig, figdir, groups, notes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = {k: [[v*scale for v, scale in zip(ax.get_position().bounds, [W, H, W, H])] for ax in axes]
              for k, axes in groups.items()}
    abc = np.array([bounds[k][0] for k in 'ABC'])
    np.testing.assert_allclose(abc[:, [1, 3]], [abc[0, [1, 3]]]*3, atol=1e-9)
    d = np.array(bounds['D'][:10])
    d_bottom, d_top = d[:, 1].min(), (d[:, 1]+d[:, 3]).max()
    np.testing.assert_allclose(d[:5, [0, 2, 3]], d[5:, [0, 2, 3]], atol=1e-9)
    for i in range(5):
        assert groups['D'][i].texts[0].get_text() == groups['D'][i+5].texts[0].get_text()
    np.testing.assert_allclose(abc[:, 3], [d_top-d_bottom]*3, atol=1e-9)
    np.testing.assert_allclose(bounds['C'][0][2], bounds['F'][0][2], atol=1e-9)
    assert bounds['B'][0][2] > 2*bounds['C'][0][2]
    for k in 'EF':
        x, y, width, height = bounds[k][0]
        np.testing.assert_allclose([y, y+height, width], [d_bottom, d_top, 50], atol=1e-9)
    top_xlabel_y = [groups[k][0].xaxis.label.get_window_extent(renderer).y0 for k in 'ABC']
    np.testing.assert_allclose(top_xlabel_y, [top_xlabel_y[0]]*3, atol=.1)
    outside = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.x1+.5 or b.y1 > fig.bbox.y1+.5:
            outside.append(t.get_text())
    assert not outside, outside
    f = groups['F'][0]
    significance = [t for t in f.texts if t.get_gid() == 'cohort-significance-label']
    legend = f.get_legend()
    legend_gap_mm = None
    if legend is not None and significance:
        lb = legend.get_window_extent(renderer)
        sb = Bbox.union([t.get_window_extent(renderer) for t in significance])
        assert not lb.overlaps(sb), 'F legend overlaps significance annotation'
        legend_gap_mm = (lb.y0-sb.y1)/fig.dpi*25.4
    export(fig, figdir/'fig5-complete-layout')
    with fitz.open(figdir/'fig5-complete-layout.pdf') as doc:
        doc[0].get_pixmap(dpi=125, alpha=False).save(figdir/'fig5-complete-layout-preview.png')
    positions = {ax: ax.get_position().frozen() for ax in fig.axes}
    figure_texts = list(fig.texts)
    for letter, axes in groups.items():
        for ax in fig.axes:
            ax.set_visible(ax in axes)
        for t in figure_texts:
            t.set_visible(t in notes.get(letter, []))
        fig.canvas.draw()
        boxes = [ax.get_tightbbox(renderer) for ax in axes]
        boxes += [t.get_window_extent(renderer) for t in notes.get(letter, [])]
        bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(.025)
        export(fig, figdir/f'fig5-panel{letter.lower()}', bbox=bbox)
        for ax, position in positions.items():
            ax.set_position(position)
    for ax in fig.axes:
        ax.set_visible(True)
    for t in figure_texts:
        t.set_visible(True)
    return dict(axes_bounds_mm_bottom_origin=bounds, top_row_axes_equal_height=True,
                both_rows_equal_height=True, C_and_F_equal_width=True,
                D_frame_columns_aligned=True, D_frame_count_per_row=5,
                D_identical_elapsed_ms_per_column=True, D_frame_time_labels='inside_each_frame',
                D_separate_time_axis=False, D_row_gap_mm=5,
                lower_row_plot_top_and_bottom_equal=True, top_row_xlabel_baseline_equal=True,
                text_outside_canvas=outside, label_font_pt=LABEL, tick_font_pt=TICK,
                contact_font_pt=CONTACT, legend_font_pt=LEGEND,
                D_title=None, gray_pending_caption=False, D_bottom_unit_label=None,
                F_sample_size_annotation=False, F_stage_annotation=False,
                F_xaxis_label=groups['F'][0].get_xlabel(),
                F_xaxis_tick_labels=[t.get_text() for t in groups['F'][0].get_xticklabels()],
                F_xaxis_tick_positions=groups['F'][0].get_xticks().tolist(),
                F_legend_significance_nonoverlap=True,
                F_legend_above_significance_gap_mm=legend_gap_mm,
                F_significance_label_positions=[t.get_position() for t in significance],
                F_legend_labels=[t.get_text() for t in groups['F'][0].get_legend().texts]
                    if groups['F'][0].get_legend() is not None else [])


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=OUT)
    OUT = ap.parse_args().output
    src, figdir = OUT/'source', OUT/'figures'
    figdir.mkdir(exist_ok=True)
    previous = read(src/'previous_figure5_manifest.json')
    physics = read(src/'applied_physics.json')
    assert sha(src/'trajectory.npz') == read(src/'trajectory.json')['arrays_sha256']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': TICK,
        'axes.labelsize': LABEL, 'axes.linewidth': .65, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(W/25.4, H/25.4))
    groups, notes = {}, {}
    checks = draw_abc_ef(fig, src, physics, previous, groups)
    if 'paired_statistics' in checks['cohort']:
        write(OUT/'core_count_statistics.json', checks['cohort']['paired_statistics'])
        write(OUT/'named_template_statistics.json', checks['cohort']['named_template_paired_statistics'])
    propagation = draw_d(fig, src, physics, previous, groups, notes)
    for letter, x, top in [('A', 2, 7), ('B', 78, 7), ('C', 238, 7),
                           ('D', 2, 82), ('E', 162, 82), ('F', 241, 82)]:
        fig.text(x/W, 1-top/H, letter, weight='bold', fontsize=15, va='top')
    layout = export_all(fig, figdir, groups, notes)
    plt.close(fig)
    complete = checks['cohort_comparison_complete']
    record = dict(figure='Fig5', asset_id=previous['asset_id'], version=OUT.name,
        status='PENDING_AUTHOR_VISUAL_REVIEW', human_visual_acceptance='PENDING',
        producer='scripts/paper_figures/build_fig5_aligned_layout.py', workpoint='xy_left_20',
        previous_version=previous['version'], panel_a=previous['panel_a'], panel_d=propagation,
        panel_map=previous['panel_map'], layout=layout, data_checks=checks,
        cohort_experiment=str(EXP), cohort_comparison_complete=complete,
        cohort_interim=checks['cohort_interim'],
        scientific_status='AUTHOR_REQUESTED_INTERIM' if checks['cohort_interim'] else 'COMPLETED_COHORT' if complete else 'HISTORICAL_COHORT',
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in src.iterdir() if p.is_file()},
        outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in figdir.iterdir() if p.suffix in ('.png', '.pdf', '.svg')})
    n_subjects = len(checks['cohort'].get('subjects', []))
    selection = checks['cohort'].get('cohort_selection')
    cohort_label = 'shared-axis ' if selection else ''
    if complete:
        record['panel_map']['F'] = f'Matched {n_subjects}-patient {cohort_label}1/2-core comparison'
    elif checks['cohort_interim']:
        record['panel_map']['F'] = f'Author-requested interim {n_subjects}-patient {cohort_label}1/2-core comparison at common completed search budget'
    if selection:
        record['cohort_selection'] = selection
    if checks['cohort'].get('display_contract', {}).get('view') == 'stronger_weaker':
        record['panel_map']['F'] += '; stronger/weaker signed template match at each original workpoint'
        record['analysis_status'] = 'POST_HOC_EXPLORATORY'
    if checks['cohort'].get('selection_policy'):
        record['panel_map']['F'] += '; TA-preferred constrained core1 versus unconstrained core2; true TA/TB labels'
        record['analysis_status'] = 'POST_HOC_ASYMMETRIC_SELECTION'
    record['panel_map']['D'] = 'Same-workpoint MTA/MTB, five shared elapsed-time frames, normalized within each event'
    status_path = EXP/'status.json'
    experiment_status = (read(status_path) if status_path.exists()
                         else previous.get('cohort_execution_at_render', {}))
    record['cohort_execution_status_source'] = ('live_execution_status' if status_path.exists()
                                                else 'frozen_previous_manifest')
    record['cohort_execution_at_render'] = {k: experiment_status.get(k) for k in
        ['status', 'train_complete', 'train_planned', 'confirmation_complete', 'confirmation_planned', 'updated_unix']}
    write(OUT/'figure5_manifest.json', record)
    write(OUT/'validation.json', dict(status='PASS', waveform_exact_array_parity=True,
        frozen_native_rank_matrix_painters=True, propagation_raw_frames_exact_native_sampling=True,
        cohort_interim=checks['cohort_interim'], cohort_comparison_complete=complete,
        propagation_normalization_roundtrip=True,
        new_simulations=0, **layout, human_visual_acceptance='PENDING'))
    print(json.dumps(dict(output=str(figdir), complete=complete, layout=layout), indent=2))


if __name__ == '__main__':
    main()

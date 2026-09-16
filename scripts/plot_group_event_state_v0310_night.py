#!/usr/bin/env python3
"""Night figures for v0.3.10. One panel answers one question (CLAUDE.md section 7).

Missing results are drawn as an explicit 'not yet run' panel; nothing is
zero-filled, interpolated or simulated.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.topic5_epi_prssm.figure_style import (MM, DOUBLE_COLUMN_MM, apply_style, panel_letter, zero_line)
from src.topic5_group_event_state.v0310 import audit
from src.topic5_group_event_state.v035.contracts import atomic_json

FAMILY_LABEL = {'F': 'fixed seven-kernel history', 'L': 'learned linear state', 'N': 'learned nonlinear state'}
FAMILY_COLOR = {'F': '#C58F3D', 'L': '#4C78A8', 'N': '#6A51A3'}
SUBJECT_LABEL = {'epilepsiae_1096': 'Patient A', 'epilepsiae_1125': 'Patient B', 'epilepsiae_253': 'Patient C'}
STOP_COLOR = {'PLATEAU_AFTER_LR_REDUCTION': '#3F8F5B', 'BUDGET_LIMIT': '#B33A3A',
              'WALL_TIME_LIMITED': '#8A8A8A'}
STOP_LABEL = {'PLATEAU_AFTER_LR_REDUCTION': 'plateau after two learning-rate drops',
              'BUDGET_LIMIT': 'stopped by the update budget', 'WALL_TIME_LIMITED': 'stopped by wall clock'}


def not_yet_run(ax, message):
    ax.text(.5, .5, 'not yet run\n' + message, ha='center', va='center', fontsize=7.5,
            color='#8A8A8A', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(False)


def save(fig, folder, name, provenance):
    folder.mkdir(parents=True, exist_ok=True)
    png, pdf = folder / f'{name}.png', folder / f'{name}.pdf'
    fig.savefig(png, dpi=600); fig.savefig(pdf); plt.close(fig)
    meta = dict(figure=name, timestamp=time.time(), producer=str(Path(__file__).relative_to(ROOT)),
                producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), **provenance)
    atomic_json(folder / f'{name}.metadata.json', meta)
    return meta


def figure_training(cards, folder, common_recipe=None):
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_MM * MM, 64 * MM),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, .9], wspace=.42))
    subjects = sorted({c['subject'] for c in cards})
    drawn = 0
    shown_recipe = {}
    for slot in range(3):
        ax = axes[slot]
        subject = subjects[slot] if slot < len(subjects) else None
        # One canonical comparison per patient: the shared capacity, the first
        # optimisation repeat, event-only, the main horizon. Without this the
        # panel overplots every seed, mode and horizon for that patient.
        rows = [c for c in cards
                if subject and c['subject'] == subject and c.get('stages', {}).get('event')
                and c['source_mode'] == 'event_only' and c['view'] == 'joint'
                and abs(c['history_hours'] - 8.0) < 1e-9
                and c['seed'] == min(x['seed'] for x in cards if x['subject'] == subject)] if subject else []
        if not rows:
            not_yet_run(ax, 'no fitted cell yet')
            ax.set_title(SUBJECT_LABEL.get(subject, subject) if subject else 'no patient', pad=4)
            continue
        # One recipe per patient, so the three families are compared like for like
        # and the fixed-history arm is not drawn once per capacity.
        counts = {}
        for card in rows:
            counts.setdefault(card['recipe'], set()).add(card['family'])
        recipe = (common_recipe or {}).get(subject)
        if recipe not in counts:
            recipe = max(counts, key=lambda r: (len(counts[r]), r))
        shown_recipe[subject] = recipe
        for card in sorted([c for c in rows if c['recipe'] == recipe], key=lambda c: c['family']):
            curve = card['stages']['event']['curve']
            if not curve:
                continue
            ax.plot([c['update'] for c in curve], [c['inner']['objective'] for c in curve],
                    color=FAMILY_COLOR[card['family']], lw=1.2, zorder=3)
            for level in card['stages']['event']['lr_levels'][1:]:
                ax.axvline(level['start_update'], color='#BFBFBF', lw=.6, ls=(0, (2, 2)), zorder=1)
            ax.plot([card['stages']['event']['selected_update']], [card['stages']['event']['selected_inner']],
                    marker='o', ms=3.2, mfc='white', mec=FAMILY_COLOR[card['family']], mew=1., zorder=4)
            drawn += 1
        ax.set_title(SUBJECT_LABEL.get(subject, subject), pad=4)
        ax.set_xlabel('effective optimiser updates')
        ax.text(.97, .04, 'capacity %s' % recipe.replace('R', ''), transform=ax.transAxes,
                fontsize=6.6, color='#4D4D4D', ha='right', va='bottom')
        ax.ticklabel_format(axis='y', useOffset=False, style='plain')
        ax.margins(x=.03)
        if slot == 0:
            ax.set_ylabel('held-in validation loss\n(lower is better)')
    ax = axes[3]
    counts = defaultdict(int)
    for card in cards:
        for stage, value in card.get('stages', {}).items():
            counts[(stage, value['stop_reason'])] += 1
    stages = ['background', 'event', 'refitted_constant']
    labels = ['background', 'event', 'constant']
    if counts:
        bottom = np.zeros(len(stages))
        for reason in ('PLATEAU_AFTER_LR_REDUCTION', 'BUDGET_LIMIT', 'WALL_TIME_LIMITED'):
            values = np.array([counts.get((s, reason), 0) for s in stages], float)
            if values.sum() == 0:
                continue
            ax.bar(np.arange(len(stages)), values, bottom=bottom, width=.62,
                   color=STOP_COLOR[reason], label=STOP_LABEL[reason], lw=0)
            bottom += values
        ax.set_xticks(np.arange(len(stages)))
        ax.set_xticklabels(labels, rotation=25, ha='right')
        ax.set_ylabel('fitted arms'); ax.set_title('why each fit stopped', pad=4)
        top = int(bottom.max()) if bottom.max() else 1
        ax.set_yticks(np.linspace(0, top, min(top + 1, 5)).round().astype(int))
    else:
        not_yet_run(ax, 'no completed stage')
        ax.set_title('why each fit stopped', pad=4)
    for letter, a in zip('AB', (axes[0], axes[3])):
        panel_letter(a, letter, dx=-0.24, dy=1.20)
    handles = [Line2D([], [], color=FAMILY_COLOR[f], lw=1.4, label=FAMILY_LABEL[f]) for f in ('F', 'L', 'N')]
    present = {r for (_, r), n in counts.items() if n}
    handles += [plt.Rectangle((0, 0), 1, 1, color=STOP_COLOR[r], label=STOP_LABEL[r])
                for r in ('PLATEAU_AFTER_LR_REDUCTION', 'BUDGET_LIMIT', 'WALL_TIME_LIMITED')
                if r in present]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.30), handlelength=1.3)
    fig.suptitle('Did the fits actually finish training?', y=1.10, fontsize=8.5)
    return save(fig, folder, 'fig1_training_sufficiency',
                dict(n_cards=len(cards), n_curves=drawn, recipe_shown_per_patient=shown_recipe,
                     question_a='does the held-in validation loss still improve at the budget?',
                     question_b='how often did the budget bind instead of a plateau?'))


def figure_capacity(cards, contrasts, folder):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_MM * MM, 62 * MM),
                             gridspec_kw=dict(width_ratios=[1.15, 1], wspace=.3))
    ax = axes[0]
    subjects = sorted({c['subject'] for c in cards})
    markers = dict(zip(subjects, 'os^'))
    any_point = False
    # Only the capacity-selection cells were run at all three capacities with
    # everything else held fixed; mixing seeds, horizons and modes here would
    # plot unrelated cells as if they were one curve.
    capacity = [c for c in cards if c.get('stages', {}).get('event') and c['view'] == 'joint'
                and c['source_mode'] == 'event_only' and abs(c['history_hours'] - 8.0) < 1e-9
                and c['seed'] == min(x['seed'] for x in cards)]
    for subject in subjects:
        for family in ('F', 'L', 'N'):
            rows = sorted([c for c in capacity if c['subject'] == subject and c['family'] == family],
                          key=lambda c: c['config']['readout_hidden'])
            if len(rows) < 2:
                continue
            x = [c['config']['readout_hidden'] for c in rows]
            y = [c['stages']['event']['selected_inner'] for c in rows]
            ax.plot(x, y, color=FAMILY_COLOR[family], lw=1., marker=markers[subject], ms=3.4,
                    mfc='white', mew=.9)
            any_point = True
    if any_point:
        ax.set_xscale('log', base=2); ax.set_xticks([32, 64, 128]); ax.set_xticklabels(['32', '64', '128'])
        ax.set_xlabel('readout width (state width 16 / 32 / 64)')
        ax.set_ylabel('held-in validation loss\n(lower is better)')
        ax.set_title('does more capacity help?', pad=3)
        ax.text(.97, .24, 'capacity-selection cells only', transform=ax.transAxes, fontsize=6.6,
                color='#4D4D4D', ha='right', va='center')
    else:
        not_yet_run(ax, 'need at least two capacities per family')
    ax = axes[1]
    pairs = [('N', 'L'), ('N', 'F'), ('L', 'F')]
    values = {p: [r[f'{p[0]}_over_{p[1]}'] for r in contrasts
                  if isinstance(r.get(f'{p[0]}_over_{p[1]}'), float)] for p in pairs}
    if any(values.values()):
        for i, pair in enumerate(pairs):
            v = np.asarray(values[pair], float)
            if not v.size:
                continue
            span = .62
            jitter = ((np.arange(len(v)) - (len(v) - 1) / 2) / max(len(v) - 1, 1)) * span
            colours = ['#B33A3A' if x < 0 else '#2F5D8A' for x in v]
            ax.scatter(i + jitter, v, s=11, c=colours, zorder=3, linewidths=0)
            ax.plot([i - .22, i + .22], [np.median(v)] * 2, color='#4D4D4D', lw=1.6, zorder=4)
        zero_line(ax)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(['nonlinear\nover linear', 'nonlinear\nover fixed', 'linear\nover fixed'])
        ax.set_ylabel('paired loss margin\n(positive = first arm better)')
        ax.set_title('does the transition family matter?', pad=3)
        ax.margins(x=.12)
    else:
        not_yet_run(ax, 'no complete family triple yet')
    for letter, a in zip('AB', axes):
        panel_letter(a, letter)
    handles = [Line2D([], [], color=FAMILY_COLOR[f], lw=1.4, label=FAMILY_LABEL[f]) for f in ('F', 'L', 'N')]
    handles += [Line2D([], [], color='#4D4D4D', marker=markers[s], ls='none', ms=3.4, mfc='white',
                       label=SUBJECT_LABEL.get(s, s)) for s in subjects]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.16), handlelength=1.2)
    fig.suptitle('Capacity and transition family', y=1.01, fontsize=8.5)
    return save(fig, folder, 'fig2_capacity_and_family',
                dict(n_cards=len(cards), question_a='does more capacity lower held-in loss?',
                     question_b='is there a paired margin between transition families?'))


def figure_holdout(cards, folder):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_MM * MM, 62 * MM), gridspec_kw=dict(wspace=.3))
    scored = [c for c in cards if c['metrics'].get('2', {}).get('status') == 'SCORED'
              and isinstance(c['metrics']['2'].get('gain_over_floored_control'), (int, float))]
    ax = axes[0]
    if scored:
        subjects = sorted({c['subject'] for c in scored})
        for i, subject in enumerate(subjects):
            rows = [c for c in scored if c['subject'] == subject]
            v = np.array([c['metrics']['2']['gain_over_floored_control'] for c in rows])
            colours = ['#B33A3A' if x < 0 else '#2F5D8A' for x in v]
            jitter = (np.arange(len(v)) - (len(v) - 1) / 2) * .05
            ax.scatter(i + jitter, v, s=13, c=colours, zorder=3, linewidths=0)
            ax.plot([i - .24, i + .24], [np.median(v)] * 2, color='#4D4D4D', lw=1.6, zorder=4)
        zero_line(ax)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([SUBJECT_LABEL.get(s, s) for s in subjects])
        ax.set_ylabel('held-out gain over the stronger control\n(positive = events add information)')
        ax.set_title('is there a net event increment?', pad=3)
        ax.margins(x=.15)
    else:
        not_yet_run(ax, 'no scored held-out cell')
    ax = axes[1]
    points = []
    for card in scored:
        donor = card['metrics']['2'].get('wrong_time_control') or {}
        if donor.get('window_equal_weight') is not None and donor.get('state_on_same_anchors') is not None:
            points.append((card['subject'], donor['window_equal_weight'] - donor['state_on_same_anchors'],
                           donor['n_eligible']))
    if points:
        subjects = sorted({p[0] for p in points})
        for i, subject in enumerate(subjects):
            v = np.array([p[1] for p in points if p[0] == subject])
            colours = ['#B33A3A' if x < 0 else '#2F5D8A' for x in v]
            jitter = (np.arange(len(v)) - (len(v) - 1) / 2) * .05
            ax.scatter(i + jitter, v, s=13, c=colours, zorder=3, linewidths=0)
            ax.plot([i - .24, i + .24], [np.median(v)] * 2, color='#4D4D4D', lw=1.6, zorder=4)
        zero_line(ax)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([SUBJECT_LABEL.get(s, s) for s in subjects])
        ax.set_ylabel('loss penalty when the state is taken\nfrom the wrong time')
        ax.set_title('is any increment time-specific?', pad=3)
        ax.margins(x=.15)
    else:
        not_yet_run(ax, 'no eligible wrong-time donor')
    for letter, a in zip('AB', axes):
        panel_letter(a, letter)
    fig.text(.5, -.1, 'Each point is one fitted arm; the reference is the lower loss of the frozen background '
                      'parent and the refitted constant, aggregated on two-hour recording windows.',
             ha='center', fontsize=6.8, color='#4D4D4D')
    fig.suptitle('Held-out increment from interictal event content', y=1.01, fontsize=8.5)
    return save(fig, folder, 'fig3_heldout_increment',
                dict(n_scored=len(scored), question_a='net gain over the stronger control?',
                     question_b='does a wrong-time state cost anything?'))


def figure_seizure(path, folder):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_MM * MM, 62 * MM),
                             gridspec_kw=dict(width_ratios=[1.25, 1], wspace=.3))
    payload = json.loads(path.read_text()) if path.exists() else None
    # The seizure layer now scores every family; the figure shows the family the
    # registered rule selected, and the metadata records that the others exist.
    every = [r for r in (payload or {}).get('per_cluster', []) if r['arm'] == 'state']
    rows = [r for r in every if r.get('is_primary_family', True)]
    ax = axes[0]
    def episode_label(row):
        return f"{SUBJECT_LABEL.get(row['subject'], row['subject'])[-1]}{row['cluster_id']}"

    if rows:
        cmap = plt.get_cmap('viridis')
        handles = []
        for i, row in enumerate(rows):
            traj = [(t['offset_hours'], t['mean_distance']) for t in row['trajectory']
                    if t['mean_distance'] is not None]
            if len(traj) < 2:
                continue
            x, y = zip(*traj)
            colour = cmap(i / max(len(rows) - 1, 1))
            ax.plot(x, y, color=colour, lw=1.1, marker='o', ms=2.2)
            handles.append(Line2D([], [], color=colour, lw=1.4, label=episode_label(row)))
        if handles:
            ax.legend(handles=handles, loc='upper left', ncol=2, handlelength=1.1,
                      columnspacing=.9, borderaxespad=.2)
        ax.axvspan(-2, -.5, color='#EAEAEA', zorder=0)
        ax.axvline(0, color='#B33A3A', lw=.9, ls=(0, (3, 2)))
        ax.set_xlabel('hours relative to seizure onset')
        ax.set_ylabel('state distance from the fitted-period centre')
        ax.set_title('what the state does around a seizure', pad=3)
        ax.margins(x=.02)
    else:
        not_yet_run(ax, 'seizure scoring has not produced a trajectory')
    ax = axes[1]
    if rows:
        labels, main, rate = [], [], []
        for row in rows:
            labels.append(episode_label(row))
            main.append(row['case_minus_control'])
            rate.append(row['case_minus_rate_matched'])
        idx = np.arange(len(labels))
        for offset, values, colour, label in ((-.16, main, '#2F5D8A', 'matched on time and coverage'),
                                              (.16, rate, '#A35E48', 'additionally matched on event load')):
            good = [(i + offset, v) for i, v in zip(idx, values) if v is not None]
            if good:
                ax.scatter(*zip(*good), s=16, color=colour, zorder=3, linewidths=0, label=label)
        zero_line(ax)
        ax.set_xticks(idx); ax.set_xticklabels(labels)
        ax.set_xlabel('seizure episode (patient / episode)')
        ax.set_ylabel('pre-seizure minus matched control')
        ax.set_title('per-episode difference', pad=3)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), handlelength=1.1)
        ax.margins(x=.12)
    else:
        not_yet_run(ax, 'no matched control available')
    for letter, a in zip('AB', axes):
        panel_letter(a, letter)
    degenerate = sorted({episode_label(r) for r in rows if r.get('learned_state_equals_initialisation')}) \
        if rows else []
    if degenerate:
        fig.text(.5, -.13, 'For %s the upstream fit selected its initial checkpoint, so the learned state '
                           'IS the initialised state; those episodes do not show a learned-state result.'
                 % ', '.join(degenerate), ha='center', fontsize=6.8, color='#B33A3A')
    fig.suptitle('Rare seizure episodes, one point per episode', y=1.04, fontsize=8.5)
    return save(fig, folder, 'fig4_rare_seizure',
                dict(n_clusters=len(rows), n_family_arms_scored=len(every), source=str(path),
                     families_shown=sorted({r['family'] for r in rows}),
                     families_scored=sorted({r['family'] for r in every}),
                     shown_rule='the family the registered rule selected on validation; the other families '
                                'are in the cluster score table as the registered sensitivity',
                     question_a='how does the state move approaching a seizure?',
                     question_b='does each episode differ from its own matched control?',
                     episodes_where_learned_state_equals_initialisation=degenerate,
                     limit='per-episode description only; repeated five-minute queries are not independent'))


ARM_LABEL = {'state': 'trained state', 'initialized': 'same shape, untrained',
             'fixed_history': 'fixed seven-kernel history'}
ARM_MARK = {'state': ('o', '#6A51A3'), 'initialized': ('^', '#8A8A8A'), 'fixed_history': ('s', '#C58F3D')}


def figure_drift(root, cards, folder):
    """Why the held-out predictions fail: the readout is extrapolating."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_MM * MM, 62 * MM),
                             gridspec_kw=dict(width_ratios=[1.15, 1], wspace=.34))
    path = root / 'final_reports' / 'state_drift.csv'
    rows = []
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row['source_mode'] != 'event_only':
                    continue
                try:
                    row['shift'] = float(row['selection_mean_shift_in_fitted_sd'])
                except (TypeError, ValueError):
                    continue
                rows.append(row)
    ax = axes[0]
    if rows:
        subjects = sorted({r['subject'] for r in rows})
        for i, subject in enumerate(subjects):
            for arm in ('state', 'initialized', 'fixed_history'):
                picked = [r for r in rows if r['subject'] == subject and r['arm'] == arm]
                if not picked:
                    continue
                marker, colour = ARM_MARK[arm]
                offset = {'state': -.2, 'initialized': 0., 'fixed_history': .2}[arm]
                ax.scatter([i + offset] * len(picked), [abs(r['shift']) for r in picked],
                           marker=marker, s=16, color=colour, zorder=3, linewidths=0)
        ax.axhline(1.0, color='#4D4D4D', lw=.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([SUBJECT_LABEL.get(s, s) for s in subjects])
        ax.set_ylabel('how far the held-out states sit outside\nthe fitted range (fitted standard deviations)')
        ax.set_title('is the readout extrapolating?', pad=4)
        ax.margins(x=.15)
    else:
        not_yet_run(ax, 'no frozen state exported yet')
    ax = axes[1]
    gains = {}
    for card in cards:
        metric = card['metrics'].get('2', {})
        if metric.get('status') == 'SCORED' and isinstance(metric.get('gain_over_floored_control'), float):
            gains[(card['subject'], card['seed'], card['family'], card['source_mode'])] = \
                metric['gain_over_floored_control']
    points = []
    for r in rows:
        if r['arm'] != 'state':
            continue
        key = (r['subject'], int(r['seed']), r['family'], r['source_mode'])
        if key in gains:
            points.append((abs(r['shift']), gains[key], r['subject']))
    if points:
        subjects = sorted({p[2] for p in points})
        markers = dict(zip(subjects, 'os^'))
        for subject in subjects:
            sel = [(x, y) for x, y, s in points if s == subject]
            ax.scatter([x for x, _ in sel], [y for _, y in sel], marker=markers[subject], s=18,
                       facecolor='white', edgecolor='#6A51A3', linewidths=1.0, zorder=3,
                       label=SUBJECT_LABEL.get(subject, subject))
        zero_line(ax)
        ax.set_xlabel('extrapolation (fitted standard deviations)')
        ax.set_ylabel('held-out gain over the stronger control')
        ax.set_title('does extrapolating cost accuracy?', pad=4)
        ax.legend(loc='lower left', handlelength=1.0, borderaxespad=.2)
        ax.margins(.12)
    else:
        not_yet_run(ax, 'no paired drift and held-out score yet')
    for letter, a in zip('AB', axes):
        panel_letter(a, letter, dx=-0.14, dy=1.14)
    handles = [Line2D([], [], color=ARM_MARK[a][1], marker=ARM_MARK[a][0], ls='none', ms=4,
                      label=ARM_LABEL[a]) for a in ('state', 'initialized', 'fixed_history')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.20), handlelength=1.2)
    fig.suptitle('The frozen state drifts out of the range its readout was fitted on', y=1.06, fontsize=8.5)
    return save(fig, folder, 'fig6_state_extrapolation',
                dict(n_states=len(rows), n_paired_points=len(points),
                     question_a='how far outside the fitted range do held-out queries land, and does '
                                'training make that worse than an untrained state of the same shape?',
                     question_b='does the amount of extrapolation track the held-out loss?',
                     reference_line='one fitted-period standard deviation'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    folder = args.root / 'final_reports' / 'figures'
    cards = audit.load_cards(args.root / 'human_v0310')
    def parse(value):
        # An empty cell means "not estimable"; it must become None, because a
        # bare '' passes an `is not None` filter and then breaks float().
        if value in ('', 'None', 'nan'):
            return None
        try:
            return float(value)
        except ValueError:
            return value

    contrasts = []
    path = args.root / 'final_reports' / 'paired_family_contrasts.csv'
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                contrasts.append({k: parse(v) for k, v in row.items()})
    summary_path = args.root / 'final_reports' / 'aggregation_summary.json'
    common_recipe = (json.loads(summary_path.read_text()).get('common_recipe')
                     if summary_path.exists() else {}) or {}
    metas = [figure_training(cards, folder, common_recipe), figure_capacity(cards, contrasts, folder),
             figure_holdout(cards, folder),
             figure_seizure(args.root / 'seizure' / 'seizure_scores_H2.0.json', folder),
             figure_drift(args.root, cards, folder)]
    atomic_json(folder / 'figure_index.json', dict(status='COMPLETE', timestamp=time.time(), figures=metas))
    print(json.dumps(dict(figures=[m['figure'] for m in metas], n_cards=len(cards))), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Per-patient example figures: the same prefix under different frozen states.

Spec section 6. The two example prefixes and the state quantile threshold are
both fixed on the fitted period; held-out events are then displayed as they
fall. Nothing is chosen because it looks good on the held-out set.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.topic5_epi_prssm.figure_style import MM, DOUBLE_COLUMN_MM, apply_style, panel_letter, zero_line
from src.topic5_group_event_state.v039.frozen_transfer import first_unknown_group, branching_strata, LogitAdapter
from src.topic5_group_event_state.v035.contracts import atomic_json

SUBJECT_LABEL = {'epilepsiae_1096': 'Patient A', 'epilepsiae_1125': 'Patient B', 'epilepsiae_253': 'Patient C'}
LOW, HIGH = '#4C78A8', '#A35E48'


def rebuild_logits(contact_json, features, transfer, prefix, arm):
    payload = json.loads(Path(contact_json).read_text())
    bundle = torch.load(payload['checkpoint'], map_location='cpu', weights_only=False)
    contacts = transfer['ranks'].shape[1]
    raw = torch.tensor(np.c_[prefix['logits'], prefix['stops']], dtype=torch.float32)
    total = raw + bundle['weights']['prefix_bias']
    anchor = transfer['anchor_position']
    for name in ('background', arm):
        values = features['background'] if name == 'background' else features[arm]
        norm = bundle['normalizers'][name]
        standard = np.clip((values - norm['center']) / norm['scale'], -8, 8).astype(np.float32)
        model = LogitAdapter(standard.shape[1], contacts)
        model.load_state_dict(bundle['weights'][name]); model.requires_grad_(False)
        with torch.no_grad():
            total = total + model(torch.tensor(standard))[torch.as_tensor(anchor)]
    return total[:, :contacts].numpy()


def figure_for_subject(subject, record, root, folder):
    features_path = Path(record['export'])
    with np.load(features_path) as z:
        features = {k: z[k] for k in z.files}
    meta = json.loads(Path(record['contact']).read_text())
    prefix_meta = json.loads((Path(meta['prefix_card'])).read_text())
    with np.load(prefix_meta['transfer_data_path']) as z:
        transfer = {k: z[k] for k in z.files}
    with np.load(Path(meta['prefix_card']).with_suffix('.npz')) as z:
        prefix = {k: z[k] for k in z.files}
    with np.load(meta['scores']) as z:
        scores = {k: z[k] for k in z.files}
    ranks, phase = transfer['ranks'], transfer['phase']
    target, available, stop, keys, _ = first_unknown_group(ranks)
    branch, eligible, _ = branching_strata(ranks, phase)
    anchor = transfer['anchor_position']
    # Outcome-blind state summary: first principal component of the fitted-period
    # states, split at the fitted-period median. No held-out value is consulted.
    fit_anchors = np.unique(anchor[phase == 'FIT'])
    state = features['state']
    centre = state[fit_anchors].mean(0)
    _, _, vt = np.linalg.svd(state[fit_anchors] - centre, full_matrices=False)
    score = (state - centre) @ vt[0]
    threshold = float(np.median(score[fit_anchors]))
    high = score[anchor] > threshold
    picked = []
    for key in eligible:
        rows = np.flatnonzero((keys == key) & branch & (phase == 'SELECTION'))
        if len(rows) >= 6 and 2 <= min(int(high[rows].sum()), int((~high[rows]).sum())):
            picked.append((len(rows), key, rows))
    picked.sort(key=lambda r: -r[0])
    picked = picked[:2]
    if not picked:
        # The designed display splits held-out repeats of one prefix by the
        # fitted-period state median. That is only possible if held-out states
        # straddle the median; here they do not, which is itself the finding.
        held = phase == 'SELECTION'
        counts = []
        for key in eligible:
            rows = np.flatnonzero((keys == key) & branch & held)
            if len(rows):
                counts.append((len(rows), int(high[rows].sum()), int((~high[rows]).sum())))
        counts.sort(reverse=True)
        return dict(subject=subject, figure=None, status='NOT_PRODUCIBLE',
                    reason='no branching prefix has held-out repeats on both sides of the fitted-period '
                           'state median, so the designed state-quantile comparison has nothing to '
                           'contrast; the frozen state drifts across the period boundary',
                    n_eligible_prefixes=len(eligible),
                    largest_prefixes_n_above_n_below=counts[:5],
                    held_out_fraction_above=float(high[held].mean()) if held.any() else None,
                    see='final_reports/state_drift.csv and fig6_state_extrapolation',
                    upstream_card=record['upstream_card'], family=record['family'], seed=record['seed'])
    logits = rebuild_logits(record['contact'], features, transfer, prefix, 'state')
    prefix_only = np.c_[prefix['logits']]
    names = [str(v) for v in transfer['contact_names']]
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_MM * MM, 62 * MM),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.05, .95], wspace=.62))
    width = .36
    index = np.arange(len(names))
    shift = logits - prefix_only
    top = max(max(target[r[2][~high[r[2]]]].mean(0).max(), target[r[2][high[r[2]]]].mean(0).max())
              for r in picked)
    for slot in range(2):
        ax = axes[slot]
        if slot >= len(picked):
            ax.text(.5, .5, 'only one prefix met the\nfitted-period support rule', ha='center',
                    va='center', fontsize=7, color='#8A8A8A', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for side in ('left', 'bottom'):
                ax.spines[side].set_visible(False)
            continue
        _, key, rows = picked[slot]
        low_rows, high_rows = rows[~high[rows]], rows[high[rows]]
        ax.bar(index - width / 2, target[low_rows].mean(0), width, color=LOW, lw=0)
        ax.bar(index + width / 2, target[high_rows].mean(0), width, color=HIGH, lw=0)
        ax.set_xticks(index); ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, top * 1.12)
        ax.set_title('example %d' % (slot + 1), pad=4)
        ax.text(.02, .97, '%d vs %d repeats' % (len(low_rows), len(high_rows)), transform=ax.transAxes,
                fontsize=6.6, color='#4D4D4D', va='top', ha='left')
        if slot == 0:
            ax.set_ylabel('fraction of held-out repeats\nwhere the contact appeared')
        else:
            ax.set_yticklabels([])
    ax = axes[2]
    for slot, (_, key, rows) in enumerate(picked):
        marker = 'o' if slot == 0 else '^'
        offset = (slot - .5) * .18
        ax.plot(index - width / 2 + offset, shift[rows[~high[rows]]].mean(0), color=LOW, marker=marker,
                ms=3.4, ls='none', mfc='white', mew=1.1)
        ax.plot(index + width / 2 + offset, shift[rows[high[rows]]].mean(0), color=HIGH, marker=marker,
                ms=3.4, ls='none', mfc='white', mew=1.1)
    zero_line(ax)
    ax.set_xticks(index); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('score shift the state adds\nto each contact')
    ax.set_title('state effect on the model', pad=4)
    ax = axes[3]
    rows = np.concatenate([r for _, _, r in picked])
    window = np.floor(transfer['anchor_time'][rows] / 7200.).astype(np.int64)
    difference = scores['background_subset'][rows] - scores['state_subset'][rows]
    values, counts = [], []
    for w in np.unique(window):
        mask = window == w
        values.append(float(difference[mask].mean())); counts.append(int(mask.sum()))
    colours = ['#B33A3A' if v < 0 else '#2F5D8A' for v in values]
    ax.scatter(np.arange(len(values)) + 1, values, s=np.clip(np.array(counts) * 2.5, 8, 34), c=colours,
               zorder=3, linewidths=0)
    ax.plot([.4, len(values) + .6], [np.median(values)] * 2, color='#4D4D4D', lw=1.6, zorder=4)
    zero_line(ax)
    ax.set_xlabel('two-hour window')
    ax.set_ylabel('score gain from the state\n(positive = state helps)')
    ax.set_title('per-window gain', pad=4)
    ax.text(.98, .03, '%d windows' % len(values), transform=ax.transAxes, fontsize=6.6,
            color='#4D4D4D', va='bottom', ha='right')
    ax.set_xticks(np.arange(1, len(values) + 1) if len(values) <= 8 else
                  np.linspace(1, len(values), 5).round().astype(int))
    ax.margins(x=.12)
    for letter, a in zip('ABCD', axes):
        panel_letter(a, letter, dx=-0.34, dy=1.20)
    handles = [Line2D([], [], color=LOW, lw=4, label='below the fitted-period median state'),
               Line2D([], [], color=HIGH, lw=4, label='above it'),
               Line2D([], [], color='#4D4D4D', marker='o', ls='none', ms=3.4, mfc='white', label='example 1'),
               Line2D([], [], color='#4D4D4D', marker='^', ls='none', ms=3.4, mfc='white', label='example 2')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.24), handlelength=1.4)
    fig.suptitle('%s: the same prefixes under different states' % SUBJECT_LABEL.get(subject, subject),
                 y=1.10, fontsize=8.5)
    folder.mkdir(parents=True, exist_ok=True)
    name = f'fig5_prefix_examples_{subject}'
    png, pdf = folder / f'{name}.png', folder / f'{name}.pdf'
    fig.savefig(png, dpi=600); fig.savefig(pdf); plt.close(fig)
    meta_out = dict(figure=name, subject=subject, timestamp=time.time(),
                    producer=str(Path(__file__).relative_to(ROOT)),
                    producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                    prefixes=[dict(key=str(k), n_selection_repeats=int(n),
                                   n_below=int((~high[r]).sum()), n_above=int(high[r].sum()))
                              for n, k, r in picked],
                    state_summary='first principal component of the fitted-period states, split at the '
                                  'fitted-period median; no held-out value enters the choice',
                    n_eligible_prefixes=len(eligible), upstream_card=record['upstream_card'],
                    family=record['family'], seed=record['seed'],
                    limit='an example display, not a test; the per-window panel is the quantitative part')
    atomic_json(folder / f'{name}.metadata.json', meta_out)
    return meta_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    index_path = args.root / 'm1_transfer' / 'm1_transfer_index.json'
    if not index_path.exists():
        print(json.dumps(dict(status='NOT_RUN', reason='no frozen transfer index'))); return
    folder = args.root / 'final_reports' / 'figures'
    metas = []
    for record in json.loads(index_path.read_text())['rows']:
        if record.get('source_mode') != 'event_only' or not str(record.get('contact', '')).endswith('.json'):
            continue
        if any(m['subject'] == record['subject'] for m in metas):
            continue
        try:
            meta = figure_for_subject(record['subject'], record, args.root, folder)
        except Exception as error:
            meta = dict(subject=record['subject'], error=str(error)[:300])
        if meta:
            metas.append(meta)
            print(json.dumps({k: meta.get(k) for k in ('figure', 'subject', 'error')}), flush=True)
    atomic_json(folder / 'prefix_example_index.json',
                dict(status='COMPLETE', timestamp=time.time(), figures=metas))
    print(json.dumps(dict(status='COMPLETE', n_subjects=len(metas))), flush=True)


if __name__ == '__main__':
    main()

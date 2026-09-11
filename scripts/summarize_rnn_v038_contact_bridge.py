#!/usr/bin/env python3
"""Patient/family-first summary, preserving event/bin sensitivity and sources."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_rnn_v038_contact_bridge import sha, write_json


PRIMARY_SUPPORT = 'fit_branch_same_prefix_k_shift'
PRIMARY_ENDPOINT = 'fit_branch_next_set_nll'
CONTROLS = ['static', 'B_mark', 'constant', 'shift_k']


def passing(card, endpoint, weighting, controls=CONTROLS, support=PRIMARY_SUPPORT):
    values = [card['comparisons'][support].get(c, {}).get(endpoint, {}).get(weighting) for c in controls]
    return card['parent_parity_ok'] and all(v is not None and v > 1e-6 for v in values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root
    manifest = json.loads((root / 'execution_manifest.json').read_text())
    lineage_path = Path('/data/hfosp_group_event_state_v0_3_8_review_repair/state_lineage.json')
    lineage = json.loads(lineage_path.read_text())
    parents = {r['checkpoint_sha256']: r for r in lineage['h1_cards']}
    cards, groups = [], defaultdict(list)
    for job in manifest['jobs']:
        card_path = root / 'per_subject' / job['id'] / 'card.json'
        card = json.loads(card_path.read_text())
        source = json.loads(Path(card['source_card']).read_text())
        checkpoint = source['state_provenance']['checkpoint']
        parent = parents[card['source_hashes'][checkpoint]]
        card['selected_state_class'] = parent['selected_state_class']
        if sha(card_path.parent / 'predictions_and_scores.npz') != card['predictions_sha256']:
            raise ValueError('prediction hash mismatch')
        card['result_path'] = str(card_path); card['result_sha256'] = sha(card_path)
        rich = source.get('conditional_rich_mark', {}).get('selection_scores', {})
        gains = {c: rich[c] - rich['prefix_plus_state'] for c in
                 ['prefix_only', 'prefix_plus_B_mark', 'prefix_plus_constant_state'] if c in rich}
        if rich:
            gains['wrong_time'] = rich['shifted_state_on_same_support'] - rich['correct_state_on_shift_support']
        card['rich_mark_control_floor_audit'] = {'gains': gains,
            'all_four_directional': len(gains) == 4 and all(v > 1e-6 for v in gains.values()),
            'scores_from_saved_cards_not_new_morphology_refit': True}
        cards.append(card); groups[(card['subject'], card['family'])].append(card)
    rows, seed_rows, flat_rows, blocks = [], [], [], []
    endpoints = ['all_identity_set_nll', 'suffix_identity_set_nll', 'next_after_two_set_nll',
                 'next_two_teacher_forced_set_nll', PRIMARY_ENDPOINT, 'fit_branch_next_set_accuracy', 'stop_bce']
    for (subject, family), group in sorted(groups.items()):
        group.sort(key=lambda c: c['seed'])
        row = {'subject': subject, 'family': family, 'n_registered_seeds': len(group),
               'n_parity_passed': sum(c['parent_parity_ok'] for c in group), 'endpoint_results': {},
               'rich_mark_four_control_seeds': [c['seed'] for c in group if c['rich_mark_control_floor_audit']['all_four_directional']]}
        for endpoint in endpoints:
            # STOP cannot be assessed on identity rows conditioned on K>0.
            support = 'same_prefix_shift' if endpoint == 'stop_bce' else PRIMARY_SUPPORT
            controls = ['static', 'B_mark', 'constant', 'shift'] if endpoint == 'stop_bce' else CONTROLS
            result = {'support': support, 'controls': controls}
            for weight in ['event_weighted_gain', 'block_equal_gain']:
                result[weight] = {'passing_seeds': [c['seed'] for c in group if passing(c, endpoint, weight, controls, support)],
                    'median_contrasts': {control: float(np.median(v)) if (v := [c['comparisons'][support][control][endpoint][weight]
                        for c in group if c['comparisons'][support][control][endpoint].get(weight) is not None and c['parent_parity_ok']]) else None
                        for control in controls}}
            result['passing_both_weightings'] = sorted(set(result['event_weighted_gain']['passing_seeds']) & set(result['block_equal_gain']['passing_seeds']))
            row['endpoint_results'][endpoint] = result
        rows.append(row)
        for c in group:
            line = {'subject': subject, 'family': family, 'seed': c['seed'], 'source_class': c['selected_state_class'],
                    'parity_ok': c['parent_parity_ok'], 'event_weight_update_l2': c['event_weight_update_l2'],
                    'primary_event_pass': passing(c, PRIMARY_ENDPOINT, 'event_weighted_gain'),
                    'primary_block_pass': passing(c, PRIMARY_ENDPOINT, 'block_equal_gain'),
                    'rich_four_controls_pass': c['rich_mark_control_floor_audit']['all_four_directional']}
            for control in ['static', 'B_mark', 'constant', 'shift_k', 'event_constant', 'background_constant', 'event_initial', 'event_mark_scramble', 'event_shift_k', 'background_shift_k']:
                cc = c['comparisons'][PRIMARY_SUPPORT].get(control, {}).get(PRIMARY_ENDPOINT, {})
                line[control + '_event_gain'] = cc.get('event_weighted_gain')
                line[control + '_block_gain'] = cc.get('block_equal_gain')
                line['n_events'] = cc.get('n_events', line.get('n_events'))
                line['n_blocks'] = cc.get('n_blocks', line.get('n_blocks'))
            seed_rows.append(line)
    for c in cards:
        for support, arms in c['comparisons'].items():
            for arm, endpoints_values in arms.items():
                for endpoint, result in endpoints_values.items():
                    base = {'subject': c['subject'], 'family': c['family'], 'seed': c['seed'],
                            'support': support, 'control': arm, 'endpoint': endpoint, 'parity_ok': c['parent_parity_ok']}
                    flat_rows.append({**base, **{k: v for k, v in result.items() if k != 'blocks'}})
                    blocks.extend({**base, **block} for block in result.get('blocks', []))

    def write_csv(name, records):
        keys = list(dict.fromkeys(k for r in records for k in r))
        with (root / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(records)

    write_csv('contact_bridge_seed_summary.csv', seed_rows)
    write_csv('contact_bridge_all_contrasts.csv', flat_rows)
    write_csv('contact_bridge_physical_bins.csv', blocks)
    summary = {'status': 'COMPLETE', 'n_evaluated': len(cards), 'n_registered': len(manifest['jobs']),
               'n_parent_parity_passed': sum(c['parent_parity_ok'] for c in cards), 'rows': rows,
               'seed_rows': seed_rows, 'primary_support': PRIMARY_SUPPORT, 'primary_endpoint': PRIMARY_ENDPOINT,
               'primary_controls': CONTROLS, 'original_lineage_sha256': sha(lineage_path),
               'result_manifest': [{'path': c['result_path'], 'sha256': c['result_sha256']} for c in cards],
               'retrospective_only': True, 'frozen_head_ablation_not_retrained_comparator': True,
               'directions_not_significance': True, 'claim_of_learned_event_dynamic_bridge_established': False,
               'seeds_are_not_patients': True}
    write_json(root / 'contact_bridge_summary.json', summary)

    # Diagnostic figure, not a revision of any accepted paper figure.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    e253 = [c for c in cards if c['subject'] == 'epilepsiae_253' and c['family'] == 'dual']
    e253.sort(key=lambda c: c['seed'])
    controls = ['static', 'B_mark', 'constant', 'shift_k', 'event_constant', 'background_constant', 'event_initial']
    labels = ['Static', 'Marked\nhistory', 'All state\nconstant', 'Wrong\ntime', 'Event\nconstant', 'Background\nconstant', 'Initial event\nweights']
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0), layout='constrained')
    matrices = []
    for weighting in ['event_weighted_gain', 'block_equal_gain']:
        matrices.append(np.array([[c['comparisons'][PRIMARY_SUPPORT][a][PRIMARY_ENDPOINT].get(weighting, np.nan)
                                   for a in controls] for c in e253]))
    vmax = max(0.01, max(np.nanmax(np.abs(v)) for v in matrices))
    for ax, matrix, title in zip(axes, matrices, ['A  Equal weight per event', 'B  Equal weight per 2 h bin']):
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_yticks(np.arange(5), [str(c['seed'])[-4:] for c in e253])
        ax.set_ylabel('Optimization seed'); ax.set_title(title, loc='left')
        for i in range(5):
            for k in range(len(labels)):
                v = matrix[i, k]
                ax.text(k, i, f'{v:+.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > 0.65 * vmax else 'black')
    fig.colorbar(im, ax=axes, label='Control NLL minus correct-state NLL', shrink=0.8)
    fig.suptitle('E253: next contact after the same two observed groups', fontsize=12)
    figs = root / 'figures'; figs.mkdir(exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(figs / ('contact_bridge_e253_next_contact.' + extension), dpi=220)
    plt.close(fig)
    (figs / 'README.md').write_text('### contact_bridge_e253_next_contact.png / .pdf / .svg\n'
        '这张诊断图比较 E253 五个固定 seed，在相同前两组和下一组大小下预测下一触点的变化。'
        '前缀分叉资格只在 FIT 建立，全部对照使用相同错时可配对事件；正值表示真实状态评分更好。'
        '左侧事件等权，右侧两小时物理窗等权，二者差异显示事件密集时段对结果的影响。\n'
        '**关注点**：初始化事件权重与冻结分支均值是固定读出的消融诊断，不是重新拟合的替代模型；图未获作者目视接受。\n')
    print(json.dumps({k: summary[k] for k in ['status', 'n_evaluated', 'n_parent_parity_passed']}))


if __name__ == '__main__':
    main()

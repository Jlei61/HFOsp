#!/usr/bin/env python3
"""Morning aggregation: sufficiency matrix, paired contrasts and evidence chains.

Nothing here re-runs science. It reads the cards, refuses to pair cells that do
not share code/data/targets/split, and writes the delivery tables.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from src.topic5_group_event_state.v0310 import audit
from src.topic5_group_event_state.v0310 import queue_plan as Q
from src.topic5_group_event_state.v035.contracts import atomic_json


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return 0
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def state_response(card, lead='2'):
    """Is the arm actually using its state, or has the readout collapsed?

    Two independent symptoms: swapping in a state from the wrong time costs
    exactly nothing, and the arm scores the same as the refitted constant. A
    saturated readout produces both.
    """
    metric = card.get('metrics', {}).get(lead, {})
    if metric.get('status') != 'SCORED':
        return dict(status='NOT_SCORED')
    donor = metric.get('wrong_time_control') or {}
    penalty = (None if donor.get('window_equal_weight') is None or donor.get('state_on_same_anchors') is None
               else donor['window_equal_weight'] - donor['state_on_same_anchors'])
    constant_gap = metric.get('gain_over_refitted_constant')
    saturated = (card.get('state_probes') or {}).get('readout_tanh_saturated_fraction')
    collapsed = (penalty is not None and abs(penalty) < 1e-9
                 and constant_gap is not None and abs(constant_gap) < 1e-3)
    return dict(status='SCORED', wrong_time_penalty=penalty, gap_to_refitted_constant=constant_gap,
                readout_tanh_saturated_fraction=saturated,
                state_arm_collapsed_to_constant=bool(collapsed))


def training_audit_rows(cards):
    rows = []
    for card in cards:
        inventory = card['model_inventory']
        totals = {k: inventory[k].get('_total_parameters') for k in inventory}
        response = state_response(card)
        for stage, value in card.get('stages', {}).items():
            sampling = value['sampling'].get('1') or next(iter(value['sampling'].values()), {})
            zero = value['gradient']['zero_gradient_updates']
            rows.append(dict(
                subject=card['subject'], seed=card['seed'], recipe=card['recipe'], family=card['family'],
                view=card['view'], source_mode=card['source_mode'], history_hours=card['history_hours'],
                cell_status=card['status'], state_width=card['state_width'],
                transition_rank=card['transition_rank'], readout_hidden=card['config']['readout_hidden'],
                observer_parameters=totals.get('observer'), residual_parameters=totals.get('residual'),
                background_parameters=totals.get('baseline'), constant_parameters=totals.get('constant'),
                stage=stage, budget=value['budget'], optimizer_updates=value['optimizer_updates'],
                stop_reason=value['stop_reason'], n_lr_drops=value['n_lr_drops'],
                final_lr=value['lr_levels'][-1]['lr'] if value['lr_levels'] else None,
                selected_update=value['selected_update'], initial_inner=value['initial_inner']['objective'],
                selected_inner=value['selected_inner'], final_inner=value['final_inner'],
                best_differs_from_final=value['best_differs_from_final'],
                fit_diagnostic_first=value['fit_diagnostic_first'], fit_diagnostic_last=value['fit_diagnostic_last'],
                median_pre_clip_norm=value['gradient']['median_pre_clip_norm'],
                max_pre_clip_norm=value['gradient']['max_pre_clip_norm'],
                clipped_fraction=value['gradient']['clipped_fraction'],
                nonfinite_events=value['gradient']['nonfinite_events'],
                n_parameters_with_any_zero_grad_update=sum(1 for v in zero.values() if v),
                max_zero_grad_updates=max(zero.values()) if zero else 0,
                lead2_equivalent_epochs=sampling.get('equivalent_epochs'),
                lead2_distinct_rows=sampling.get('distinct_rows'), lead2_pool_rows=sampling.get('pool_rows'),
                batch_schedule_sha256=value['batch_schedule_sha256'],
                sufficiency_verdict=card['training_sufficiency_vector']['verdict'],
                readout_tanh_saturated_fraction=(card['state_probes'] or {}).get('readout_tanh_saturated_fraction'),
                state_rms=(card['state_probes'] or {}).get('state_rms'),
                nonlinear_verdict=(card['nonlinear_usage'] or {}).get('verdict'),
                nonlinear_median_ratio=(card['nonlinear_usage'] or {}).get('median_nonlinear_over_linear'),
                optimisation_reading=(card['optimisation_diagnosis'] or {}).get('reading'),
                wrong_time_penalty_2h=response.get('wrong_time_penalty'),
                gap_to_refitted_constant_2h=response.get('gap_to_refitted_constant'),
                state_arm_collapsed_to_constant=response.get('state_arm_collapsed_to_constant'),
                initial_checkpoint_selected=(card['optimisation_diagnosis'] or {}).get('initial_checkpoint_selected'),
                data_sha256=card['data_sha256'][:16], split_sha256=card['split_sha256'][:16],
                target_sha256=card['target_sha256'][:16],
                trainer_sha256=card['source_hashes'].get('src/topic5_group_event_state/v0310/trainer.py', '')[:16],
                elapsed_seconds=round(card['elapsed_seconds']), card=card['_path']))
    return rows


def layer_rows(cards):
    """Spec section 10 per-layer table: shape, trainable count, decay group and
    how far each tensor actually moved from its initialisation."""
    rows = []
    for card in cards:
        groups = {}
        for stage, value in card.get('stages', {}).items():
            for name, record in value['parameter_groups'].items():
                groups[name] = dict(record, stage=stage)
        ratios = card['training_sufficiency_vector']['c_layers']['update_ratio']
        hashes = card['training_sufficiency_vector']['c_layers']['initial_parameter_hashes']
        zero = {}
        for stage, value in card.get('stages', {}).items():
            zero.update(value['gradient']['zero_gradient_updates'])
        for module, entries in card['model_inventory'].items():
            for name, entry in entries.items():
                if name.startswith('_'):
                    continue
                key = f'{module}.{name}'
                rows.append(dict(subject=card['subject'], seed=card['seed'], recipe=card['recipe'],
                                 family=card['family'], view=card['view'],
                                 source_mode=card['source_mode'], history_hours=card['history_hours'],
                                 state_width=card['state_width'],
                                 readout_hidden=card['config']['readout_hidden'],
                                 module=module, parameter=name, shape='x'.join(str(v) for v in entry['shape']),
                                 parameters=entry['parameters'],
                                 weight_decay_group=groups.get(key, {}).get('group', 'not_trained_in_any_stage'),
                                 trained_in_stage=groups.get(key, {}).get('stage'),
                                 absolute_l2_change=(ratios.get(module, {}).get(name, {}) or {}).get('absolute_l2_change'),
                                 relative_l2_change=(ratios.get(module, {}).get(name, {}) or {}).get('relative_l2_change'),
                                 zero_initialised=(ratios.get(module, {}).get(name, {}) or {}).get('zero_initialised'),
                                 initial_l2=(ratios.get(module, {}).get(name, {}) or {}).get('initial_l2'),
                                 final_l2=(ratios.get(module, {}).get(name, {}) or {}).get('final_l2'),
                                 zero_gradient_updates=zero.get(key),
                                 initial_hash=(hashes.get(module, {}) or {}).get(name, '')[:16],
                                 card=card['_path']))
    return rows


def background_parent_identity(group):
    """Spec 5.1 U3: the background-only parent must be ONE foundation shared by
    F, L and N, bound by hash. The trainer refits it deterministically per cell,
    so this checks post hoc that all three arms really stand on the same ground
    instead of three different fits."""
    import torch
    digests = {}
    for card in group:
        path = Path(card['checkpoint'])
        if not path.exists():
            digests[card['family']] = None
            continue
        state = torch.load(path, map_location='cpu', weights_only=False)
        h = hashlib.sha256()
        for key in sorted(state['baseline']):
            h.update(key.encode())
            h.update(np.ascontiguousarray(state['baseline'][key].numpy()).tobytes())
        digests[card['family']] = h.hexdigest()
    present = [v for v in digests.values() if v]
    return dict(per_family={k: (v or '')[:16] for k, v in digests.items()},
                shared=bool(present) and len(set(present)) == 1,
                n_arms_checked=len(present))


def contrast_table(cards):
    rows = []
    for key, group in sorted(audit.group_cards(cards).items()):
        try:
            audit.assert_pairable(group)
            pairable, reason = True, None
        except ValueError as error:
            pairable, reason = False, str(error)
        base = dict(subject=key[0], seed=key[1], recipe=key[2], history_hours=key[3], view=key[4],
                    source_mode=key[5], n_arms=len(group), families=''.join(sorted(c['family'] for c in group)),
                    complete_triple=set(c['family'] for c in group) == {'F', 'L', 'N'},
                    pairable=pairable, pairing_error=reason)
        for better, worse in (('N', 'L'), ('N', 'F'), ('L', 'F')):
            contrast = audit.paired_contrast(group, better, worse) if pairable else dict(status='NOT_PAIRABLE')
            base[f'{better}_over_{worse}'] = contrast.get('margin')
            base[f'{better}_over_{worse}_status'] = contrast.get('status')
        for card in group:
            metric = card['metrics'].get('2', {})
            if metric.get('status') == 'SCORED':
                base[f'{card["family"]}_gain_over_floored_control'] = metric['gain_over_floored_control']
                base[f'{card["family"]}_gain_over_background'] = metric['gain_over_frozen_background']
                donor = metric.get('wrong_time_control') or {}
                base[f'{card["family"]}_correct_over_shifted'] = (
                    None if donor.get('window_equal_weight') is None or donor.get('state_on_same_anchors') is None
                    else donor['window_equal_weight'] - donor['state_on_same_anchors'])
                base[f'{card["family"]}_n_two_hour_windows'] = metric['denominators']['n_two_hour_physical_windows']
                base[f'{card["family"]}_n_anchors'] = metric['denominators']['n_anchors']
            base[f'{card["family"]}_stop'] = card['stages']['event']['stop_reason'] if card.get('stages') else None
            probes = card.get('state_probes') or {}
            base[f'{card["family"]}_readout_tanh_saturated'] = probes.get('readout_tanh_saturated_fraction')
            base[f'{card["family"]}_state_rms'] = probes.get('state_rms')
            base[f'{card["family"]}_state_collapsed'] = state_response(card).get('state_arm_collapsed_to_constant')
        parent = background_parent_identity(group)
        base['background_parent_shared'] = parent['shared']
        base['background_parent_sha256'] = next(iter(sorted(
            v for v in parent['per_family'].values() if v)), None)
        base['background_parent_arms_checked'] = parent['n_arms_checked']
        base['any_arm_limited'] = any(c['training_sufficiency_vector']['any_arm_budget_limited'] or
                                      c['training_sufficiency_vector']['any_arm_wall_time_limited'] for c in group)
        rows.append(base)
    return rows


def matched_donor_rows(record, payload):
    """The reused contact producer scores the wrong-time arm only where a donor
    exists, so its arm-level loss sits on a smaller event set than the state
    arm. Recompute both arms on that SAME subset from the saved per-event
    scores; comparing the published means across different denominators is not
    a valid contrast."""
    from src.topic5_group_event_state.v039.frozen_transfer import equal_anchor_mean
    path = Path(payload['scores'])
    if not path.exists():
        return []
    rows = []
    with np.load(path, allow_pickle=True) as z:
        phase = z['phase']; branch = z['branch_mask']; donor = z['shift_donor']
        anchor = z['anchor_time']
        for endpoint, key, extra in (('exact_next_subset', '_subset', branch),
                                     ('next_subset_all', '_subset', np.ones(len(phase), bool)),
                                     ('stop', '_stop', np.ones(len(phase), bool))):
            mask = (phase == 'SELECTION') & extra & (donor >= 0)
            if not mask.any():
                continue
            values = {}
            for arm in ('state', 'shifted_state'):
                name = arm + key
                if name not in z:
                    values = {}; break
                per_anchor = equal_anchor_mean(z[name][mask], anchor[mask])
                values[arm] = float(per_anchor.mean())
            if len(values) != 2:
                continue
            rows.append(dict(subject=record['subject'], seed=record['seed'],
                             source_mode=record['source_mode'], family=record['family'],
                             recipe=record['recipe'], probe='contact_matched_donor',
                             arm='state_minus_wrong_time', endpoint=endpoint,
                             loss=values['shifted_state'] - values['state'],
                             state_loss_on_donor_subset=values['state'],
                             wrong_time_loss=values['shifted_state'],
                             n_events=int(mask.sum()),
                             n_anchors=int(np.unique(anchor[mask]).size),
                             upstream_card=record['upstream_card'],
                             note='both arms on the donor-eligible events only'))
    return rows


def state_drift_rows(root):
    """Does the frozen state stay in the range the readout was fitted on?

    The readout is fitted on the fitted-period states. If the dominant
    direction of the state is a slow trend, held-out queries land on one side
    of the fitted-period distribution and the readout is extrapolating. This
    measures that directly instead of inferring it from the losses.
    """
    index_path = root / 'm1_transfer' / 'm1_transfer_index.json'
    if not index_path.exists():
        return []
    rows = []
    for record in json.loads(index_path.read_text())['rows']:
        export = record.get('export')
        if not export or not Path(export).exists():
            continue
        with np.load(export) as z:
            phase = z['phase'].copy()
            for arm in ('state', 'initialized', 'fixed_history'):
                if arm not in z:
                    continue
                values = z[arm].copy()
                fit = phase == 'FIT'
                if fit.sum() < 20:
                    continue
                centre = values[fit].mean(0)
                _, singular, basis = np.linalg.svd(values[fit] - centre, full_matrices=False)
                score = (values - centre) @ basis[0]
                threshold = float(np.median(score[fit]))
                spread = float(score[fit].std()) or 1.0
                entry = dict(subject=record['subject'], seed=record['seed'],
                             source_mode=record['source_mode'], family=record['family'], arm=arm,
                             first_component_variance_share=float(singular[0] ** 2 / (singular ** 2).sum()),
                             fitted_fraction_above=float((score[fit] > threshold).mean()))
                for name in ('INNER', 'SELECTION'):
                    mask = phase == name
                    entry[f'{name.lower()}_fraction_above'] = (
                        float((score[mask] > threshold).mean()) if mask.any() else None)
                    entry[f'{name.lower()}_mean_shift_in_fitted_sd'] = (
                        float((score[mask].mean() - score[fit].mean()) / spread) if mask.any() else None)
                entry['reading'] = ('a held-out fraction near 1 or 0 means every held-out query sits on one '
                                    'side of the fitted-period median, so the readout is extrapolating')
                rows.append(entry)
    return rows


def history_contrast_rows(cards):
    """Spec section 3: horizons are paired on their COMMON scored anchors, and
    each horizon's own full support is reported beside it."""
    groups = {}
    for card in cards:
        if card['metrics'].get('2', {}).get('status') != 'SCORED':
            continue
        key = (card['subject'], card['seed'], card['recipe'], card['family'],
               card['view'], card['source_mode'])
        groups.setdefault(key, {})[card['history_hours']] = card
    rows = []
    for key, by_hours in sorted(groups.items()):
        if len(by_hours) < 2:
            continue
        loaded = {}
        for hours, card in by_hours.items():
            path = Path(card['scores'])
            if not path.exists():
                continue
            with np.load(path) as z:
                if '2h_anchor_time' not in z:
                    continue
                loaded[hours] = dict(anchor=z['2h_anchor_time'], state=z['2h_state_objective'],
                                     window=z['2h_window_id'])
        horizons = sorted(loaded)
        for i, short in enumerate(horizons):
            for long in horizons[i + 1:]:
                a_time, b_time = loaded[short]['anchor'], loaded[long]['anchor']
                common = np.intersect1d(a_time, b_time)
                if not common.size:
                    continue
                ia = np.searchsorted(a_time, common); ib = np.searchsorted(b_time, common)
                per_window = {}
                for w, d in zip(loaded[long]['window'][ib],
                                loaded[short]['state'][ia] - loaded[long]['state'][ib]):
                    per_window.setdefault(int(w), []).append(float(d))
                window_means = [float(np.mean(v)) for v in per_window.values()]
                rows.append(dict(
                    subject=key[0], seed=key[1], recipe=key[2], family=key[3], view=key[4],
                    source_mode=key[5], shorter_hours=short, longer_hours=long,
                    n_common_anchors=int(common.size),
                    n_common_two_hour_windows=len(window_means),
                    longer_minus_shorter_window_equal_weight=float(np.mean(window_means)),
                    longer_minus_shorter_anchor_equal_weight=float(
                        (loaded[short]['state'][ia] - loaded[long]['state'][ib]).mean()),
                    own_support_shorter=int(a_time.size), own_support_longer=int(b_time.size),
                    note='positive means the longer horizon predicted better on the shared anchors'))
    return rows


def contact_rows(root):
    rows = []
    index_path = root / 'm1_transfer' / 'm1_transfer_index.json'
    if not index_path.exists():
        return rows
    index = json.loads(index_path.read_text())
    for record in index['rows']:
        for kind in ('contact', 'expression'):
            path = record.get(kind)
            if not path or not str(path).endswith('.json') or not Path(path).exists():
                continue
            payload = json.loads(Path(path).read_text())
            if kind == 'contact':
                for arm, endpoints in payload.get('metrics', {}).items():
                    for endpoint, value in endpoints.items():
                        rows.append(dict(subject=record['subject'], seed=record['seed'],
                                         source_mode=record['source_mode'], family=record['family'],
                                         recipe=record['recipe'], probe='contact', arm=arm, endpoint=endpoint,
                                         loss=value['loss'], n_events=value['n_events'],
                                         n_anchors=value['n_anchors'], n_raw_blocks=value['n_raw_blocks'],
                                         upstream_card=record['upstream_card'],
                                         upstream_checkpoint_sha256=payload.get('upstream_checkpoint_sha256', '')[:16],
                                         upstream_training_sufficiency=record.get('upstream_training_sufficiency')))
                rows += matched_donor_rows(record, payload)
            else:
                # The expression probe nests its numbers per arm; flattening only
                # the endpoint level would silently drop every score.
                for endpoint, value in (payload.get('results') or {}).items():
                    if not isinstance(value, dict):
                        continue
                    shared = {k: v for k, v in value.items()
                              if isinstance(v, (int, float, str, bool))}
                    for arm, scores in (value.get('arms') or {}).items():
                        if not isinstance(scores, dict):
                            continue
                        rows.append(dict(subject=record['subject'], seed=record['seed'],
                                         source_mode=record['source_mode'], family=record['family'],
                                         recipe=record['recipe'], probe='expression', arm=arm,
                                         endpoint=endpoint, upstream_card=record['upstream_card'],
                                         upstream_training_sufficiency=record.get('upstream_training_sufficiency'),
                                         **shared,
                                         **{k: v for k, v in scores.items()
                                            if isinstance(v, (int, float, str, bool))}))
    return rows


def evidence_rows(root, cards):
    """One row per subject x seed x checkpoint. Never stitched across patients."""
    by_key = {}
    for card in cards:
        key = (card['subject'], card['seed'])
        entry = by_key.setdefault(key, dict(subject=card['subject'], seed=card['seed']))
        metric = card['metrics'].get('2', {})
        tag = f"{card['source_mode']}_{card['recipe']}_{card['family']}_H{card['history_hours']}"
        if metric.get('status') == 'SCORED':
            entry[f'{tag}_gain_over_floored_control'] = metric['gain_over_floored_control']
        entry[f'{tag}_checkpoint'] = card['checkpoint_sha256'][:16]
    m1 = root / 'm1_transfer' / 'm1_transfer_index.json'
    if m1.exists():
        for record in json.loads(m1.read_text())['rows']:
            key = (record.get('subject'), record.get('seed'))
            if key not in by_key:
                continue
            entry = by_key[key]
            entry['m1_family'] = record.get('family')
            entry['m1_upstream_card'] = record.get('upstream_card')
            path = record.get('contact')
            if path and str(path).endswith('.json') and Path(path).exists():
                payload = json.loads(Path(path).read_text())
                state = payload['metrics'].get('state', {}).get('exact_next_subset', {})
                prefix = payload['metrics'].get('background', {}).get('exact_next_subset', {})
                entry['contact_state_loss'] = state.get('loss')
                entry['contact_prefix_background_loss'] = prefix.get('loss')
                entry['contact_state_gain'] = (None if state.get('loss') is None or prefix.get('loss') is None
                                               else prefix['loss'] - state['loss'])
                entry['contact_n_events'] = state.get('n_events')
                entry['contact_upstream_checkpoint'] = payload.get('upstream_checkpoint_sha256', '')[:16]
    scores = root / 'seizure' / 'seizure_scores.json'
    if scores.exists():
        payload = json.loads(scores.read_text())
        for row in payload['per_cluster']:
            key = (row['subject'], row['seed'])
            if key not in by_key or row['arm'] != 'state':
                continue
            entry = by_key[key]
            entry.setdefault('seizure_clusters', 0)
            entry['seizure_clusters'] += 1
            entry[f"seizure_cluster{row['cluster_id']}_case_minus_control"] = row['case_minus_control']
            entry[f"seizure_cluster{row['cluster_id']}_matching"] = row['matching_status']
            entry['seizure_learned_state_equals_initialisation'] = row.get('learned_state_equals_initialisation')
    return [by_key[k] for k in sorted(by_key)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    out = root / 'final_reports'; out.mkdir(parents=True, exist_ok=True)
    cards = audit.load_cards(root / 'human_v0310')
    counts = {}
    for card in cards:
        phase = Path(card['_path']).parent.parent.name
        card['_phase'] = phase
        counts.setdefault(phase, {}).setdefault(card['status'], 0)
        counts[phase][card['status']] += 1
    n_audit = write_csv(out / 'training_audit.csv', training_audit_rows(cards))
    n_contrast = write_csv(out / 'paired_family_contrasts.csv', contrast_table(cards))
    n_layers = write_csv(out / 'layer_inventory.csv', layer_rows(cards))
    n_history = write_csv(out / 'history_contrasts.csv', history_contrast_rows(cards))
    n_drift = write_csv(out / 'state_drift.csv', state_drift_rows(root))
    windows = []
    for card in cards:
        path = Path(card['_path']).with_name('physical_windows.csv')
        if path.exists():
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    windows.append(row | dict(phase=card['_phase'], status=card['status']))
    n_windows = write_csv(out / 'paired_physical_windows.csv', windows)
    n_contact = write_csv(out / 'contact_endpoint_scores.csv', contact_rows(root))
    n_evidence = write_csv(out / 'same_checkpoint_evidence.csv', evidence_rows(root, cards))
    # Spec 5.1 U1: the shared recipe comes from the mean INNER across F/L/N, and
    # each family's own best recipe is reported alongside as a sensitivity.
    recipe_selection = {}
    for subject in Q.SUBJECTS:
        rows = [c for c in cards if c['subject'] == subject and c['source_mode'] == 'event_only'
                and c['view'] == 'joint' and abs(c['history_hours'] - 8.0) < 1e-9
                and c['seed'] == Q.SEEDS[0]]
        picked = audit.common_recipe_by_inner(rows)
        family_best = {}
        for family in ('F', 'L', 'N'):
            scored = [(c['stages']['event']['selected_inner'], c['recipe']) for c in rows
                      if c['family'] == family and c.get('stages', {}).get('event')]
            family_best[family] = min(scored)[1] if scored else None
        recipe_selection[subject] = dict(picked | dict(per_family_own_best_recipe=family_best))
    planned = {phase: len(Q.phase_cells(phase, {s: 'R1' for s in Q.SUBJECTS})) for phase in Q.DISPATCH_ORDER}
    queue = json.loads((root / 'queue_state.json').read_text()) if (root / 'queue_state.json').exists() else {}
    summary = dict(status='COMPLETE', schema='v0310_aggregation_v1', timestamp=time.time(),
                   n_cards=len(cards), by_phase=counts, planned_cells=planned,
                   common_recipe=queue.get('common_recipe'), recipe_selection=recipe_selection,
                   tables={'training_audit.csv': n_audit, 'layer_inventory.csv': n_layers,
                           'history_contrasts.csv': n_history, 'state_drift.csv': n_drift,
                           'paired_family_contrasts.csv': n_contrast,
                           'paired_physical_windows.csv': n_windows,
                           'contact_endpoint_scores.csv': n_contact,
                           'same_checkpoint_evidence.csv': n_evidence},
                   readout_saturation_caveat='a readout whose Tanh is nearly fully saturated stops '
                                             'responding to its state, so any advantage over that arm may '
                                             'reflect the scale of its state rather than the value of the '
                                             'other arm. This is recorded, not corrected: the contract '
                                             'forbids adding normalisation mid-run.',
                   pairing_rule='cells are compared only when subject, seed, recipe, horizon, view, source '
                                'mode, trainer source, data, split, targets and multi-task weights all match',
                   sufficiency_rule='any budget- or wall-limited arm forbids a sufficiency claim for the group')
    atomic_json(out / 'aggregation_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('n_cards', 'by_phase', 'tables', 'common_recipe')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

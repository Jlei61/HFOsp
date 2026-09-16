#!/usr/bin/env python3
"""S-A/S-B/S-C on frozen states. Runs only after the upstream manifest is frozen."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v0310 import seizure as S
from src.topic5_group_event_state.v0310 import seizure_scores as SS
from src.topic5_group_event_state.v039.transition import EventTransition
from src.topic5_group_event_state.v035.contracts import atomic_json

BUNDLE = '/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/human_data_v2'


def build_arms(card, data):
    cfg = card['config']
    state = torch.load(card['checkpoint'], map_location='cpu', weights_only=False)
    learned = EventTransition(data['input_dim'], cfg['family'], width=cfg['state_width'],
                              rank=cfg['transition_rank'], seed=cfg['seed'])
    learned.load_state_dict(state['observer'])
    initialised = EventTransition(data['input_dim'], cfg['family'], width=cfg['state_width'],
                                  rank=cfg['transition_rank'], seed=cfg['seed'])
    fixed = EventTransition(data['input_dim'], 'F')
    for m in (learned, initialised, fixed):
        m.requires_grad_(False)
    return dict(state=learned, initialized=initialised, fixed_history=fixed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--history-hours', type=float, default=2.0)
    p.add_argument('--source-mode', default='event_only')
    p.add_argument('--upstream-dir', required=True,
                   help='directory of frozen cells whose history matches --history-hours')
    p.add_argument('--deadline-epoch', type=float, default=0.)
    args = p.parse_args()
    torch.set_num_threads(2)
    out = args.root / 'seizure'; out.mkdir(parents=True, exist_ok=True)
    from src.topic5_group_event_state.v0310 import audit
    cards = audit.load_cards(Path(args.upstream_dir))
    # Spec 7.2: the seizure query history must be the history the model was
    # actually trained and frozen on. An H8 model truncated to H2 is not a
    # matched-history model and is refused here.
    cards = [c for c in cards if c['source_mode'] == args.source_mode and c['view'] == 'joint'
             and abs(float(c['history_hours']) - args.history_hours) < 1e-9
             and c.get('status') in ('COMPLETE', 'WALL_TIME_LIMITED') and c.get('stages', {}).get('event')]
    # The capacity directory holds one cell per recipe; picking the best INNER
    # across recipes would freeze a state at a capacity that was never selected.
    queue = args.root / 'queue_state.json'
    common = dict((json.loads(queue.read_text()).get('common_recipe') or {})) if queue.exists() else {}
    if common:
        cards = [c for c in cards if common.get(c['subject'], c['recipe']) == c['recipe']]
    # Every family's state goes through the same procedure. When the registered
    # selection lands on the fixed-history family, its "learned" and "untrained"
    # arms are literally the same object, so scoring only the selected family
    # would leave the learned-versus-untrained question unanswerable.
    best_inner = {}
    for card in cards:
        key = (card['subject'], card['seed'])
        current = best_inner.get(key)
        if current is None or (card['stages']['event']['selected_inner'], card['family']) < current:
            best_inner[key] = (card['stages']['event']['selected_inner'], card['family'])
    frozen = [dict(subject=c['subject'], seed=c['seed'], source_mode=args.source_mode,
                   family=c['family'], recipe=c['recipe'], upstream_card=c['_path'],
                   inner_selected=c['stages']['event']['selected_inner'],
                   is_primary_family=c['family'] == best_inner[(c['subject'], c['seed'])][1],
                   primary_family=best_inner[(c['subject'], c['seed'])][1],
                   primary_selected_by_tie_break=sum(
                       1 for x in cards if x['subject'] == c['subject'] and x['seed'] == c['seed']
                       and x['stages']['event']['selected_inner'] == best_inner[(c['subject'], c['seed'])][0]) > 1,
                   upstream_stop_reason=c['stages']['event']['stop_reason'])
              for c in sorted(cards, key=lambda c: (c['subject'], c['seed'], c['family']))]
    if not frozen:
        raise SystemExit(f'No frozen upstream state with history_hours={args.history_hours} in {args.upstream_dir}')
    per_cluster, summaries = [], {}
    for subject in sorted({r['subject'] for r in frozen}):
        picks = sorted([r for r in frozen if r['subject'] == subject], key=lambda r: r['seed'])
        data, seizures, records = S.load_subject(subject, BUNDLE)
        grid = S.query_grid(data, seizures, args.history_hours, 0.8)
        rows, ledger_summary, _ = S.build_ledger(subject, BUNDLE, args.history_hours)
        eligible_times = grid['times'][grid['eligible']]
        clusters = [r for r in rows if r['is_cluster_first'] and r['s_a_eligible']]
        subject_summary = dict(subject=subject, n_eligible_clusters=len(clusters),
                               n_eligible_queries=int(len(eligible_times)),
                               seeds=sorted({r['seed'] for r in picks}),
                               families=sorted({r['family'] for r in picks}),
                               primary_family=next((r['primary_family'] for r in picks), None),
                               primary_selected_by_tie_break=next(
                                   (r['primary_selected_by_tie_break'] for r in picks), None),
                               s_c_forward={},
                               arms={}, s_b={}, s_c=ledger_summary['s_c_status'],
                               s_c_reason=ledger_summary['s_c_reason'])
        if not clusters:
            summaries[subject] = subject_summary | dict(status='NOT_ESTIMABLE'); continue
        for pick in picks:
            card = json.loads(Path(pick['upstream_card']).read_text())
            models = build_arms(card, data)
            # If the upstream event stage selected its initial checkpoint, the
            # learned arm IS the initialised arm. That is not a bug, but it
            # must never be read as a learned-state result (clause C13).
            selected_initial = card['stages']['event']['selected_update'] == 0
            pick['upstream_selected_initial_checkpoint'] = bool(selected_initial)
            events = None
            for arm, model in models.items():
                states, load = SS.query_states(model, data, eligible_times, args.history_hours)
                events = load if events is None else events
                fit_mask_eligible = grid['phase'][grid['eligible']] == 'FIT'
                if fit_mask_eligible.sum() < 20:
                    subject_summary['arms'][f'{pick["seed"]}_{pick["family"]}_{arm}'] = dict(
                    family=pick['family'], is_primary_family=pick['is_primary_family'],status='NOT_ESTIMABLE',
                                                                           reason='fewer than 20 FIT queries')
                    continue
                reference = SS.fit_reference(states[fit_mask_eligible])
                distances = SS.mahalanobis(states, reference)
                full = np.zeros(len(grid['times'])); full[grid['eligible']] = distances
                load_full = np.zeros(len(grid['times'])); load_full[grid['eligible']] = load
                fit_load_null = []
                for t in grid['times'][(grid['phase'] == 'FIT') & grid['eligible']]:
                    value, n = SS.window_mean(load_full, grid['times'], t, S.PRE_ICTAL_WINDOW, grid['eligible'])
                    if value is not None and n >= 3:
                        fit_load_null.append(value)
                fit_load_null = np.asarray(fit_load_null, float)
                centres, labels = [], []
                taken, rate_taken = [], []
                for row in clusters:
                    onset = row['cluster_first_onset']
                    controls = [float(v) for v in row['control_onsets'].split(';') if v]
                    case_load, _ = SS.window_mean(load_full, grid['times'], onset, S.PRE_ICTAL_WINDOW, grid['eligible'])
                    case_q = SS.empirical_quantile(fit_load_null, case_load)

                    def rate_ok(t, case_q=case_q):
                        value, n = SS.window_mean(load_full, grid['times'], t, S.PRE_ICTAL_WINDOW, grid['eligible'])
                        if value is None or n < 3 or case_q is None:
                            return False
                        return abs(SS.empirical_quantile(fit_load_null, value) - case_q) <= 0.15
                    rate_controls, rate_taken, rate_match = S.matched_controls(
                        grid, seizures, onset, rate_taken, extra_predicate=rate_ok)
                    centres.append((onset, 'case')); labels.append(row)
                    for c in controls:
                        centres.append((c, 'control'))
                    for c in rate_controls:
                        centres.append((c, 'rate_matched_control'))
                    row['_controls'] = controls; row['_rate_controls'] = rate_controls
                    row['_rate_match'] = rate_match
                scored, null = SS.cluster_scores(full, load_full, grid['times'], grid['eligible'],
                                                 centres, grid['phase'], (grid['phase'] == 'FIT') & grid['eligible'])
                lookup = {round(r['centre'], 3): r for r in scored}
                case_rows, control_rows = [], []
                for row in clusters:
                    onset = row['cluster_first_onset']
                    case = lookup[round(onset, 3)]
                    ctrl = [lookup[round(c, 3)] for c in row['_controls'] if round(c, 3) in lookup]
                    rate = [lookup[round(c, 3)] for c in row['_rate_controls'] if round(c, 3) in lookup]
                    ctrl_mean = float(np.mean([c['primary'] for c in ctrl if c['primary'] is not None])) if ctrl else None
                    rate_mean = float(np.mean([c['primary'] for c in rate if c['primary'] is not None])) if rate else None
                    entry = dict(subject=subject, seed=pick['seed'], arm=arm, family=pick['family'],
                                 is_primary_family=pick['is_primary_family'],
                                 primary_family=pick['primary_family'],
                                 primary_selected_by_tie_break=pick['primary_selected_by_tie_break'],
                                 recipe=pick['recipe'], source_mode=args.source_mode,
                                 history_hours=args.history_hours, cluster_id=row['cluster_id'],
                                 phase=row['phase'], n_members=row['cluster_n_members'],
                                 onset_epoch=onset, n_primary_queries=case['n_primary_queries'],
                                 case_distance=case['primary'], case_fit_quantile=case['fit_quantile'],
                                 case_event_load=case['event_load'],
                                 n_controls=len(ctrl), control_mean_distance=ctrl_mean,
                                 case_minus_control=None if ctrl_mean is None or case['primary'] is None
                                 else case['primary'] - ctrl_mean,
                                 n_rate_matched_controls=len(rate), rate_matched_mean_distance=rate_mean,
                                 case_minus_rate_matched=None if rate_mean is None or case['primary'] is None
                                 else case['primary'] - rate_mean,
                                 minus6h_to_minus2h=case['minus6h_to_minus2h'],
                                 minus0p5h_to_onset=case['minus0p5h_to_onset'],
                                 upstream_selected_initial_checkpoint=bool(selected_initial),
                                 learned_state_equals_initialisation=bool(selected_initial),
                                 matching_status='MATCHING_LIMITED' if len(ctrl) < 2 else 'MATCHED',
                                 rate_matching_status='MATCHING_LIMITED' if len(rate) < 2 else 'MATCHED',
                                 trajectory=SS.trajectory_series(full, grid['times'], grid['eligible'], onset))
                    per_cluster.append(entry)
                    case_rows.append(dict(cluster_id=row['cluster_id'], primary=case['primary']))
                    control_rows += [dict(cluster_id=row['cluster_id'], primary=c['primary']) for c in ctrl]
                subject_summary['arms'][f'{pick["seed"]}_{pick["family"]}_{arm}'] = dict(
                    family=pick['family'], is_primary_family=pick['is_primary_family'],
                    status='SCORED', n_components=reference['n_components'],
                    explained_variance_ratio=reference['explained_variance_ratio'],
                    n_fit_null_windows=int(null.size))
                tag = f'{pick["seed"]}_{pick["family"]}_{arm}'
                subject_summary['s_b'][tag] = SS.leave_one_cluster_out(case_rows, control_rows)
                cluster_phase = {row['cluster_id']: row['phase'] for row in clusters}
                subject_summary.setdefault('s_c_forward', {})[tag] = SS.forward_in_time(
                    case_rows, control_rows, cluster_phase)
            if args.deadline_epoch and time.time() >= args.deadline_epoch:
                subject_summary['stopped_early'] = True
                break
        summaries[subject] = subject_summary | dict(status='SCORED')
        print(json.dumps({k: subject_summary[k] for k in ('subject', 'n_eligible_clusters',
                                                          'n_eligible_queries', 's_c')}), flush=True)
    flat = [{k: v for k, v in r.items() if k != 'trajectory'} for r in per_cluster]
    if flat:
        with (out / 'seizure_cluster_scores.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    atomic_json(out / 'seizure_scores.json',
                dict(status='COMPLETE', schema='v0310_seizure_scores_v1', timestamp=time.time(),
                     history_hours=args.history_hours, source_mode=args.source_mode,
                     subjects=summaries, per_cluster=per_cluster,
                     common_recipe=common,
                     score_definition='FIT PCA (<=4 components by FIT variance) then shrunk Mahalanobis '
                                      '(0.9*FIT covariance + 0.1*mean variance*I, floor 1e-6); PCA and '
                                      'standardisation use FIT only and no seizure label picks a dimension',
                     arm_scope='every family at the shared recipe is scored; the INNER-selected family is '
                               'flagged is_primary_family, the others are the registered sensitivity',
                     primary_window='-2h..-0.5h, queries averaged inside the window then one score per cluster',
                     reporting_limit='per-cluster case differences only; repeated 5-minute queries are not '
                                     'independent samples and no cohort p-value or detection rate is given',
                     s_c_note='a truly prospective score needs all upstream training, INNER selection and '
                              'downstream fitting to precede the test cluster. s_c_forward fits its readout '
                              'only on clusters inside the training partitions and applies it to a cluster '
                              'in the held-out partition; with one or two forward test seizures it is a '
                              'list of individual predictions, never a detection or false-alarm rate.',
                     upstream_selected_with_seizure_outcomes=False))
    print(json.dumps(dict(status='COMPLETE', n_cluster_rows=len(per_cluster))), flush=True)


if __name__ == '__main__':
    main()

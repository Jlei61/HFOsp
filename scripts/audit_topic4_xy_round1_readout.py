#!/usr/bin/env python3
"""Offline audit of all round1 observations. No SNN/heldout loading or reranking freeze."""
from pathlib import Path
import argparse
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from scripts.run_topic4_xy_research import training_contract, OUT
from src.topic4_xy_fig5_followup import read, write, sha
from src.topic4_xy_readout_audit import window_onsets, support_summary, support_gate_probability
from src.sef_hfo_events import detect_events
from src.topic4_xy_direction import onset_directions, direction_summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--maximum-candidates', type=int)
    args = parser.parse_args()
    output = OUT / 'readout_support_audit'
    if args.maximum_candidates:
        output = output / 'smoke'
    output.mkdir(parents=True, exist_ok=True)
    training, objective = training_contract()
    groups, pairs = training['groups'], training['pairs']
    all_rows = read(OUT / 'global/aggregate.json')['candidates']
    if args.maximum_candidates: all_rows = all_rows[:args.maximum_candidates]
    modes = ['frozen_causal_family', 'full_same_window_event_bar', 'full_same_window_run_bar',
             'full_detector_all_event_bar', 'full_detector_compound_event_bar',
             'full_detector_pure_event_bar', 'shifted_same_window_event_bar']
    units, candidates, inputs = [], [], {}
    for ci, row in enumerate(all_rows):
        pools = {k: [] for k in modes}
        frozen_by_seed = []
        for seed in [2511, 2512]:
            path = OUT / 'global/workers' / f'{row["candidate_id"]}_seed_{seed}.json'
            meta = read(path); inputs[str(path)] = sha(path)
            npz = path.with_suffix('.npz')
            if sha(npz) != meta['arrays']['sha256']:
                raise RuntimeError('worker checksum mismatch')
            inputs[str(npz)] = meta['arrays']['sha256']
            with np.load(npz) as z:
                names = z['contact_names'].astype(str)
                if list(names) != training['contact_names']:
                    raise RuntimeError('contact order mismatch')
                onsets = z['onsets'].astype(float); returned = z['event_returned']
                windows = np.column_stack([z['event_t_on_ms'], z['event_t_off_ms']])[returned]
                env = z['contact_envelope'].astype(float); dt = float(z['contact_envelope_dt_ms'])
                # Undo float32 storage only to recover the original integer E counts.
                ne = len(z['positions_E'])
                active = np.rint(z['active_fraction'].astype(float) * ne) / ne
                active_dt = float(z['active_fraction_bin_ms'])
                compound = z['detector_fragment_compound'].astype(bool)
                xy = z['contact_xy_mm'].copy()
            fragments = detect_events(active, active_dt, event_on_frac=meta['event_unit']['event_on_threshold'])
            if len(fragments) != meta['event_unit']['raw_detector_fragment_count']:
                raise RuntimeError('stored activity cannot reproduce detector fragments')
            fragment_windows = np.array([[e['t_on'], e['t_off']] for e in fragments])
            fragment_returned = np.array([e['returned'] for e in fragments])
            for c in meta['event_unit']['compound_fragments']:
                f = fragments[c['detector_fragment_index']]
                if f['t_on'] != c['trigger_t_on'] or f['t_off'] != c['trigger_t_off']:
                    raise RuntimeError('detector timing parity failed')
            frozen = onsets[returned]
            full_event = window_onsets(env, windows, dt)
            full_run = window_onsets(env, windows, dt, bar_scope='run')
            fragment_full = window_onsets(env, fragment_windows, dt)
            # Prespecified large offsets; condition on the original family windows.
            shifted = []
            for offset in (1000., 2000., 3000.):
                copy = env.copy(); copy[groups['SCL']] = np.roll(env[groups['SCL']], int(offset/dt), axis=1)
                shifted.append(window_onsets(copy, windows, dt))
            views = {'frozen_causal_family': frozen, 'full_same_window_event_bar': full_event,
                     'full_same_window_run_bar': full_run,
                     'full_detector_all_event_bar': fragment_full[fragment_returned],
                     'full_detector_compound_event_bar': fragment_full[fragment_returned & compound],
                     'full_detector_pure_event_bar': fragment_full[fragment_returned & ~compound],
                     'shifted_same_window_event_bar': np.concatenate(shifted)}
            frozen_by_seed.append(frozen)
            for name, values in views.items():
                summary = support_summary(values, groups, pairs)
                units.append(dict(candidate_id=row['candidate_id'], seed=seed, view=name, **summary))
                pools[name].append(values)
            direction = direction_summary(onset_directions(frozen, xy))
            write(output / 'per_subject' / f'{row["candidate_id"]}_{seed}.json',
                  {'candidate_id': row['candidate_id'], 'seed': seed, 'compound_fraction': float(compound.mean()),
                   'detector_count': len(fragments), 'direction': direction,
                   'n_final_families': len(onsets), 'n_final_returned': len(frozen),
                   'full_same_window_cross_change': {'root': support_summary(frozen, groups, pairs),
                                                     'full': support_summary(full_event, groups, pairs)}})
        base = dict(candidate_id=row['candidate_id'], domain=row['domain'], centers_mm=row['node_field']['centers_mm'],
                    J_direction=row['J_direction'], J_round1=row['J_round1'], D_cloud=row['D_cloud'],
                    D_direction=row['D_direction'], position_penalty=row['core_alignment_penalty'],
                    old_selection_eligible=row['selection_eligible'])
        for name, pool in pools.items():
            values = np.concatenate(pool)
            item = dict(base, view=name, **support_summary(values, groups, pairs))
            if name == 'shifted_same_window_event_bar':
                # Three repeated offsets are not three times the independent evidence.
                item['conditional_gate'] = None
            item['count_matched_gate_projection'] = support_gate_probability(
                pool, groups, pairs, sum(len(x) for x in frozen_by_seed), 128, 20263905+ci)
            candidates.append(item)
        if ci % 16 == 0: print(f'Audited {ci+1}/{len(all_rows)} geometries', flush=True)
        # Sample-size effect evaluated within the two existing topology strata.
        sample_rows = []
        for count in (sum(len(x) for x in frozen_by_seed), 80, 160, 320):
            sample_rows.append({'n': count, 'conditional_gate_probability': support_gate_probability(
                frozen_by_seed, groups, pairs, count, 128, 20260905 + ci)})
        write(output / 'per_subject' / f'{row["candidate_id"]}_sample_projection.json',
              {'not_new_independent_evidence': True, 'rows': sample_rows})
    patient = training['onsets_ms']
    counts = sorted({r['n_events'] for r in candidates if r['view'] == 'frozen_causal_family'})
    patient_counts = [{'n': n, 'conditional_gate_probability': support_gate_probability(
        [patient], groups, pairs, n, 256, 20261905+n)} for n in counts]
    summary = {'status': 'READOUT_SUPPORT_AUDIT_COMPLETE', 'candidate_count': len(all_rows),
               'patient': support_summary(patient, groups, pairs),
               'patient_count_matched_resampling': patient_counts, 'views': {},
               'inputs': inputs, 'upstream_results_modified': False,
               'sampling_note': 'Bootstrap is a diagnostic projection from finite existing events; it is not a new long run or independent confirmation.',
               'observation_note': 'Unrestricted envelopes can contain other concurrent roots; increased common recruitment does not establish single-family propagation or a correct patient observation model.'}
    for name in modes:
        rows = [r for r in candidates if r['view'] == name]
        values = [r['both_shafts_fraction'] for r in rows if r['both_shafts_fraction'] is not None]
        summary['views'][name] = {'n_geometries': len(rows),
                                'conditional_gate_pass': None if name.startswith('shifted') else sum(r['conditional_gate'] for r in rows),
                                'count_matched_gate_probability_median': float(np.median([r['count_matched_gate_projection'] for r in rows if r['count_matched_gate_projection'] is not None])),
                                'both_shafts_fraction_q0_q50_q100': np.quantile(values, [0, .5, 1]).tolist() if values else None,
                                'mean_contacts_median': float(np.median([r['mean_contacts'] for r in rows if r['mean_contacts'] is not None]))}
    no_prior = sorted(all_rows, key=lambda r:(r['J_direction'], r['candidate_id']))
    prior = sorted(all_rows, key=lambda r:(r['J_round1'], r['candidate_id']))
    cloud = sorted(all_rows, key=lambda r:(r['D_cloud'], r['candidate_id']))
    ranks0 = {r['candidate_id']: i for i, r in enumerate(no_prior)}
    ranks1 = {r['candidate_id']: i for i, r in enumerate(prior)}
    summary['score_audit'] = {'cloud_term_range': [min(r['D_cloud']/.1324583993682284 for r in all_rows), max(r['D_cloud']/.1324583993682284 for r in all_rows)],
      'direction_term_range': [min(r['D_direction']/.25438200471017514 for r in all_rows),max(r['D_direction']/.25438200471017514 for r in all_rows)],
      'prior_term_range': [min(.1*r['core_alignment_penalty'] for r in all_rows),max(.1*r['core_alignment_penalty'] for r in all_rows)],
      'number_rank_positions_changed_by_prior': sum(ranks0[c] != ranks1[c] for c in ranks0),
      'maximum_rank_change_from_prior': max(abs(ranks0[c]-ranks1[c]) for c in ranks0),
      'best_with_prior': prior[0]['candidate_id'], 'best_without_prior': no_prior[0]['candidate_id'],
      'cloud_score_spearman_with_primary': float(spearmanr([r['D_cloud'] for r in all_rows],[r['J_round1'] for r in all_rows]).statistic)}
    write(output / 'summary.json', summary)
    write(output / 'candidate_views.json', {'rows': candidates})
    write(output / 'unit_views.json', {'rows': units})
    with open(output / 'candidate_views.csv', 'w') as f:
        keys=[k for k in candidates[0] if k not in ('centers_mm','cross_pair_joint_counts','contact_count_histogram','recruitment_per_contact')]
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(candidates)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('inputs','patient_count_matched_resampling','patient')},indent=2))


if __name__ == '__main__': main()

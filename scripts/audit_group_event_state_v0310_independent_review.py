#!/usr/bin/env python3
"""Read-only audit of delivered v0310 artifacts; writes to a separate review root."""
import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from src.topic5_group_event_state.v0310.history import past_coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cache = {}

    def sha(path):
        p = str(path)
        if p not in cache:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(b)
            cache[p] = h.hexdigest()
        return cache[p]

    checks, failures = collections.Counter(), []

    def check(kind, actual, expected, path):
        checks[kind] += 1
        if actual != expected:
            failures.append(dict(kind=kind, path=str(path), expected=expected, actual=actual))

    cards = [(p, json.loads(p.read_text())) for p in sorted((a.root / 'human_v0310').rglob('card.json'))]
    stages, nonlinear, hashes = [], [], []
    pairs = collections.defaultdict(list)
    for p, c in cards:
        for key in ('checkpoint', 'scores', 'data'):
            path = c['data_path' if key == 'data' else key]
            check(key + '_sha256', sha(path), c[key + '_sha256'], p)
        for path, h in c['source_hashes'].items():
            check('source_sha256', sha(W / path), h, p)
        for name, s in c['stages'].items():
            stages.append(dict(card=str(p), family=c['family'], stage=name,
                               stop_reason=s['stop_reason'], selected=s['selected_update']))
            key = tuple(c[k] for k in ('subject', 'recipe', 'seed', 'view', 'source_mode', 'history_hours')) + (name,)
            pairs[key].append(s['batch_schedule_sha256'])
        if c['nonlinear_usage']:
            nonlinear.append(dict(card=str(p), selected=c['stages']['event']['selected_update'], **c['nonlinear_usage']))
        saved = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)
        advertised = c['training_sufficiency_vector']['c_layers']['initial_parameter_hashes']
        for mod, params in advertised.items():
            for name, h in params.items():
                val = saved[mod][name].detach().cpu().numpy()
                final_h = hashlib.sha256(np.ascontiguousarray(val).tobytes()).hexdigest()
                hashes.append(h == final_h)

    pre = json.loads((a.root / 'training_preflight.json').read_text())
    for path, h in pre['source_hashes'].items():
        check('preflight_source_sha256', sha(W / path), h, path)

    coverage = []
    for path in sorted({c['data_path'] for _, c in cards}):
        d = torch.load(path, map_location='cpu', weights_only=False)
        for hours in (.5, 2., 8., 16., 24.):
            for phase in ('FIT', 'INNER', 'SELECTION'):
                values = [past_coverage(d['observed_support'], s['anchor'], hours)
                          for s in d['samples'] if s['phase'] == phase]
                if values:
                    coverage.append(dict(subject=Path(path).stem, hours=hours, phase=phase,
                                         n=len(values), minimum=min(values), median=float(np.median(values)),
                                         n_below_80=int(np.sum(np.array(values) < .8)),
                                         n_below_90=int(np.sum(np.array(values) < .9))))

    contacts, drift, exports = [], [], []
    index = json.loads((a.root / 'm1_transfer/m1_transfer_index.json').read_text())
    for r in index['rows']:
        if not r.get('export'):
            continue
        meta = json.loads(Path(r['export']).with_suffix('.json').read_text())
        check('selected_export_source', meta['source_card'], r['upstream_card'], r['export'])
        check('export_sha256', sha(r['export']), meta['export_sha256'], r['export'])
        check('export_source_card_sha256', sha(meta['source_card']), meta['source_card_sha256'], r['export'])
        c = json.loads(Path(r['contact']).read_text())
        for key in ('checkpoint', 'scores', 'frozen_feature_card', 'upstream_card', 'prefix_card'):
            check('contact_' + key + '_sha256', sha(c[key]), c[key + '_sha256'], r['contact'])
        metrics = c['metrics']
        vals = {arm: metrics[arm]['exact_next_subset']['loss'] for arm in ('state', 'background', 'constant')}
        contacts.append(dict(subject=r['subject'], seed=r['seed'], source_mode=r['source_mode'],
                             family=r['family'], gain_parent=vals['background'] - vals['state'],
                             gain_floor=min(vals['background'], vals['constant']) - vals['state'],
                             state_selected=c['stages']['state']['selected_step'],
                             budget_stages=sum(s['stop_reason'] == 'BUDGET_LIMIT' for s in c['stages'].values()),
                             n_stages=len(c['stages']),
                             objective=c['training_contract']['objective']))
        with np.load(r['export']) as z:
            phase = z['phase']
            exports.append(dict(family=r['family'], state_equals_initialized=bool(np.array_equal(z['state'], z['initialized']))))
            # Match the delivered drift reference population exactly: all FIT anchors.
            for arm in ('state', 'initialized', 'fixed_history'):
                x = z[arm].astype(np.float64)
                fit, sel = phase == 'FIT', phase == 'SELECTION'
                centre = x[fit].mean(0)
                _, sv, vh = np.linalg.svd(x[fit] - centre, full_matrices=False)
                score = (x - centre) @ vh[0]
                lo, hi = score[fit].min(), score[fit].max()
                qlo, qhi = np.quantile(score[fit], [.01, .99])
                drift.append(dict(subject=r['subject'], seed=r['seed'], source_mode=r['source_mode'],
                                  family=r['family'], arm=arm,
                                  n_fit=int(fit.sum()), n_selection=int(sel.sum()),
                                  selection_fraction_above_fit_median=float(np.mean(score[sel] > np.median(score[fit]))),
                                  selection_fraction_outside_fit_pc1_range=float(np.mean((score[sel] < lo) | (score[sel] > hi))),
                                  selection_fraction_outside_fit_pc1_01_99=float(np.mean((score[sel] < qlo) | (score[sel] > qhi))),
                                  mean_shift_sd=float(score[sel].mean() / score[fit].std()),
                                  pc1_variance_fraction=float(sv[0]**2 / (sv**2).sum())))

    # Executable counterexamples: logical claims, not power calibration or human fits.
    fit = np.linspace(-2, 2, 401)
    test = np.linspace(.1, .9, 100)
    x = np.array([[1., -2.], [3., 4.]])
    beta = np.array([.7, -.2])
    reflection = np.diag([-1., 1.])
    eps = 1e-5
    examples = dict(
        all_above_median_but_within_support=dict(above=float(np.mean(test > np.median(fit))), outside=float(np.mean((test < fit.min()) | (test > fit.max())))),
        coordinate_sign_not_mechanism=dict(max_prediction_change=float(np.max(np.abs(x @ beta - (x @ reflection) @ (reflection @ beta))))),
        nonzero_tanh_branch_can_be_linear=dict(branch_over_linear=float(np.tanh(eps) / eps), nonlinear_remainder=float(abs(np.tanh(eps)-eps)/eps)),
        case_control_brier_depends_on_sampling=dict(constant_optimum_brier_when_prevalence_01=.01*.99, constant_optimum_brier_in_balanced_sample=.25))
    assert examples['all_above_median_but_within_support'] == dict(above=1., outside=0.)
    assert examples['coordinate_sign_not_mechanism']['max_prediction_change'] == 0.
    assert examples['nonzero_tanh_branch_can_be_linear']['branch_over_linear'] > .999
    assert examples['nonzero_tanh_branch_can_be_linear']['nonlinear_remainder'] < 1e-8

    def counts(xs):
        return dict(positive=sum(v > 1e-9 for v in xs), zero=sum(abs(v) <= 1e-9 for v in xs), negative=sum(v < -1e-9 for v in xs), median=float(np.median(xs)))

    summary = dict(schema='v0310_independent_review_v1', root=str(a.root),
                   script_sha256=sha(__file__), n_cards=len(cards), artifact_checks=dict(checks), failures=failures,
                   stage_stop_counts=dict(collections.Counter(s['stop_reason'] for s in stages)),
                   event_stop_counts=dict(collections.Counter(s['stop_reason'] for s in stages if s['stage']=='event')),
                   event_selected_origin=sum(s['selected']==0 for s in stages if s['stage']=='event'),
                   paired_batch_groups=len(pairs), paired_batch_mismatches=sum(len(set(v))>1 for v in pairs.values()),
                   advertised_initial_hashes_equal_final=dict(n=len(hashes), equal=sum(hashes), note='Label bug; does not invalidate separately stored update norms.'),
                   n_nonlinear_cards=len(nonlinear),
                   nonlinear_active_at_initialization=sum(n['selected']==0 and n['verdict']=='NONLINEAR_COMPONENT_ACTIVE' for n in nonlinear),
                   nonlinear_active_remainder_below_001=sum(n['median_tanh_remainder']<.001 and n['verdict']=='NONLINEAR_COMPONENT_ACTIVE' for n in nonlinear),
                   nonlinear_rows=nonlinear, contact_n=len(contacts), contact_parent_gain=counts([c['gain_parent'] for c in contacts]),
                   contact_floor_gain=counts([c['gain_floor'] for c in contacts]),
                   contact_budget_stages=sum(c['budget_stages'] for c in contacts), contact_total_stages=sum(c['n_stages'] for c in contacts),
                   contact_family_counts=dict(collections.Counter(c['family'] for c in contacts)),
                   exports_equal_initialization=sum(e['state_equals_initialized'] for e in exports),
                   counterexamples=examples,
                   scope='Artifact integrity and targeted logical/data diagnostics; not a rerun, new model validation, or scientific power calibration.')
    for name, rows in [('history_coverage', coverage), ('drift_support', drift), ('contact_comparisons', contacts)]:
        with (a.output / (name + '.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (a.output / 'audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('nonlinear_rows',)}, indent=2))


if __name__ == '__main__':
    main()

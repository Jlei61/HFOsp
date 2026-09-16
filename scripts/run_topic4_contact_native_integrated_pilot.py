#!/usr/bin/env python3
"""Consolidate five frozen observation terms and the native support prior.

Reuse the in-flight 12-unit native wave, then one genuinely adaptive 12-unit
batch and at most ten fresh-noise units. No live producer is edited.
"""
from pathlib import Path
import argparse, copy, fcntl, json, pickle, shutil, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_topic4_observable_loss_physical_pilot as execution
from scripts import run_topic4_native_activity_regularized_wave1 as native
from scripts import run_topic4_contact_timing_shape_pilot as prior
from scripts import run_topic4_multievent_distribution_v2_1 as engine
from scripts import run_topic4_xy_research as base
from src.topic4_contact_event_objective_v2 import worker_events
from src.topic4_multievent_condition_identity_v2_1 import condition_key

OUT = ROOT / 'results/topic4_sef_hfo/contact_native_integrated_pilot'
REV = ROOT / 'results/topic4_sef_hfo/contact_event_objective_revision_v2'
NATIVE = native.OUT
TRAIN = [(6101, 7101), (6102, 7101)]
CONFIRM = [(6101, 842901), (6102, 842901)]
KEYS = execution.KEYS
write = execution.write


def state(status, **kw):
    write(OUT / 'status.json', dict(status=status, updated_unix=time.time(), **kw))


def aggregate(candidate, units, lam):
    # Actual event count, not the two-label classifier, defines estimability.
    expected = len(units) == 2
    obs_ok = expected and all(u['observation']['loss'] is not None for u in units.values())
    reg_ok = expected and all(u['regularizer'].get('mean_unsupported_fraction') is not None
                              for u in units.values())
    obs = float(np.mean([u['observation']['loss'] for u in units.values()])) if obs_ok else None
    reg = float(np.mean([u['regularizer']['mean_unsupported_fraction'] for u in units.values()])) if reg_ok else None
    return dict(candidate_id=candidate['candidate_id'], candidate=candidate, units=units,
                ranking_eligible=obs_ok and reg_ok, observation_loss=obs, R_unsupported=reg,
                loss=obs + lam * reg if obs_ok and reg_ok else None,
                model_labels_used_for_ranking=False)


def freeze_lambda(rows):
    rr = [r for r in rows if r['ranking_eligible']]
    a = float(np.ptp(np.quantile([r['observation_loss'] for r in rr], [.25, .75])))
    b = float(np.ptp(np.quantile([r['R_unsupported'] for r in rr], [.25, .75])))
    if a <= 0 or b <= .01:
        raise RuntimeError('historical units cannot define the frozen regularizer scale')
    lam = .25 * a / b
    return dict(primary=lam, grid=[0., .5 * lam, lam, 2 * lam], historical_conditions=len(rr),
                observation_iqr=a, regularizer_iqr=b,
                rule='one R IQR equals 0.25 historical five-term observation-loss IQR',
                mechanism_prior_not_patient_estimate=True,
                new_native_wave_outputs_used_to_choose_lambda=False)


def distribution(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=None, median=None, variance=None, q05=None, q95=None)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                variance=float(x.var()), q05=float(np.quantile(x, .05)), q95=float(np.quantile(x, .95)))


def score_unit(path, obj, support):
    path = Path(path)
    cache = OUT / 'unit_cache' / (base.sha(path) + '.json')
    if cache.exists():
        return base.read(cache)
    engine.repaired_observation(path)
    table, details, names, worker = worker_events(path)
    if names != obj.names:
        raise RuntimeError('contact identity/order differs from the frozen objective')
    score = obj.score(table, worker.get('physical_status'))
    if table is None:
        reg = dict(n_events=0, mean_unsupported_fraction=None)
        events = []
    else:
        reg = native._regularizer_record(path, support)
        op = base.read(path.parent.parent / 'repaired_observation' / path.name)
        events = []
        for i, detail in enumerate(details):
            mask = table['participation'][i] > 0
            c = table['centroid'][i, mask]
            recruitment = table['recruitment'][i, mask, 0]
            widths = table['local_shape'][i, :, 17] - table['local_shape'][i, :, 1]
            events.append(dict(**detail, start_ms=op['events'][detail['event_id']]['window_ms'][0],
                               n_contacts=int(mask.sum()), centroid_span_ms=float(np.ptp(c)),
                               recruitment_span_ms=float(np.ptp(recruitment)),
                               contact_width_ms=[float(v) if m else None for v, m in zip(widths, mask)]))
    out = dict(worker_path=str(path), worker_json_sha256=base.sha(path),
               observation=score, regularizer=reg, events=events,
               topology_seed=int(worker['topology_seed']), dynamics_seed=int(worker['dynamics_seed']),
               static_array_identity=worker['static_array_identity'],
               distributions={k:distribution([e[k] for e in events]) for k in
                              ['local_width_ms', 'n_contacts', 'centroid_span_ms', 'recruitment_span_ms']})
    write(cache, out)
    return out


def score_rows(rows, phase, obj, lam):
    supports = {t:native._support(t) for t, _ in TRAIN}
    records = []
    for row in rows:
        units = {}
        for uid, value in row['units'].items():
            path = value['worker_path']
            topo = int(uid.split('_')[1])
            units[uid] = score_unit(path, obj, supports[topo])
        records.append(aggregate(row['candidate'], units, lam))
    report = dict(phase=phase, lambda_value=lam, candidates=records,
                  model_classifier_used=False, pooled_event_weighting=False)
    write(OUT / f'{phase}_scores.json', report)
    return records


def source_files():
    paths = [OUT/'protocol.md', REV/'objective.pkl', REV/'manifest.json',
             NATIVE/'design.json', NATIVE/'execution/A/candidate_manifest.json',
             prior.OUT/'design.json', prior.OUT/'baseline_train_scores.json',
             prior.OUT/'A_scores.json', prior.OUT/'B_scores.json',
             Path(__file__), ROOT/'scripts/analyze_topic4_contact_native_integrated_pilot.py',
             ROOT/'src/topic4_contact_event_objective_v2.py',
             ROOT/'src/topic4_native_activity_regularizer.py',
             ROOT/'scripts/run_topic4_native_activity_regularized_wave1.py',
             ROOT/'scripts/run_topic4_observable_loss_physical_pilot.py',
             ROOT/'scripts/analyze_topic4_observable_loss_physical_pilot.py']
    paths += [native.CANARY/f'support_topology_{t}_nearest.npz' for t, _ in TRAIN]
    return {str(p):base.sha(p) for p in paths}


def frozen():
    d = base.read(OUT / 'design.json')
    for path, digest in d['frozen_files'].items():
        if base.sha(Path(path)) != digest:
            raise RuntimeError(f'integrated frozen dependency changed: {path}')
    if base.sha(OUT/'objective.pkl') != d['objective_sha256']:
        raise RuntimeError('integrated observation reference changed')
    return d, pickle.load((OUT/'objective.pkl').open('rb'))


def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT/'unit_cache').mkdir(exist_ok=True)
    if (OUT/'design.json').exists():
        return frozen()
    hashes = source_files()
    state('PREPARING_FROM_COMPLETED_HISTORY', new_wave_scores_read=False)
    shutil.copyfile(REV/'objective.pkl', OUT/'objective.pkl')
    obj = pickle.load((OUT/'objective.pkl').open('rb'))
    history = [r for name in ['baseline_train', 'A', 'B']
               for r in base.read(prior.OUT/f'{name}_scores.json')['candidates']]
    rr = score_rows(history, 'history_unscaled', obj, 0.)
    lam = freeze_lambda(rr)
    scored = [aggregate(r['candidate'], r['units'], lam['primary']) for r in rr]
    write(OUT/'history_scores.json', dict(phase='history', candidates=scored))
    source = base.read(NATIVE/'design.json')
    old = base.read(prior.OUT/'design.json')
    d = dict(version='contact_native_integrated_pilot_v1', frozen_unix=time.time(),
             objective_sha256=base.sha(OUT/'objective.pkl'), frozen_files=hashes,
             weights={k:.2 for k in KEYS}, lambda_contract=lam,
             observation='five frozen terms; 19770 FIT events and 46 TRAIN waveforms',
             regularizer=source['regularizer'],
             regularizer_interpretation='delayed structural EE support proxy, not measured current or patient whole-field truth',
             anchors=old['anchors'], native_proposals=source['proposals'],
             train_pairs=TRAIN, confirmation_pairs=CONFIRM,
             parameter_names=prior.PARAMETERS,
             global_lower=prior.LOW.tolist(), global_upper=prior.HIGH.tolist(),
             anchor_half_width=prior.STEP.tolist(),
             initial_proposal_provenance='native wave A was selected from completed timing pilot under its old observation score; no retroactive claim of new-objective selection',
             adaptation='one additional six-condition batch generated from updated five-term-plus-R parents after native A completes; paired random directions frozen before dispatch',
             ranking='equal networks; every unit N >= 16 and no runaway; no mode-count, route or outside-core gate',
             auxiliary_checks='event count, background and primary-window fraction remain visible diagnostics, not TA/TB gates',
             confirmation='same pool: top two under observation only and top two under observation plus R, plus original point 1; physical-condition dedup; no confirmation outcomes in nomination',
             budget=dict(inherited_running_units=12, additional_adaptive_units=12,
                         maximum_confirmation_units=10, maximum_total_physical_units=34,
                         additional_units_from_this_handoff_at_most=22,
                         reused_completed_history_units=30, duration_ms=24000,
                         maximum_global_SNN_workers=8),
             discontinued_queue='observable_loss_physical_pilot stopped before first physical dispatch; inputs retained',
             waveform_review_limit='18 previous review waveforms were already used by the completed v1 review; not described as pristine holdout and not used here for training/nomination',
             stop='review after integrated pilot; no extra rounds, model freeze or Fig5')
    write(OUT/'design.json', d)
    state('PREPARED_WAITING_FOR_NATIVE_A', historical_conditions=len(scored),
          inherited_physical_units=12, additional_physical_units_started=0)
    return frozen()


def parents(pool, anchors):
    return [min([r for r in pool if r['ranking_eligible'] and
                 execution.anchor_index(r['candidate'], anchors) == i],
                key=lambda r:(r['loss'], r['candidate_id']))['candidate'] for i in range(3)]


def proposals(d, selected):
    path = OUT/'adaptive_proposals.json'
    if path.exists():
        return base.read(path)['candidates']
    candidates, draws = [], []
    for i, (anchor, parent) in enumerate(zip(d['anchors'], selected), 1):
        rng = np.random.default_rng(843200+i)
        direction = rng.choice([-1., 1.], 5)*rng.uniform(.35, 1., 5)
        low = np.maximum(prior.LOW, prior.vector(anchor)-prior.STEP)
        high = np.minimum(prior.HIGH, prior.vector(anchor)+prior.STEP)
        width = high-low
        for sign, label in [(1., 'plus'), (-1., 'minus')]:
            raw = prior.vector(parent)+sign*.65*prior.STEP*direction
            value = low+width-np.abs((raw-low)%(2*width)-width)
            cid = f'integrated_anchor{i}_B_{label}'
            candidates.append(prior.with_vector(parent, value, cid, parent['candidate_id'], 'integrated_B'))
            draws.append(dict(candidate_id=cid, seed=843200+i, sign=sign, direction=direction.tolist(),
                              parent_id=parent['candidate_id'], values=value.tolist(), bounds=[low.tolist(), high.tolist()]))
    if len({condition_key(c) for c in candidates}) != 6:
        raise RuntimeError('duplicate integrated proposals')
    write(path, dict(candidates=candidates, draws=draws, delayed_update=True))
    return candidates


def nominate(pool, reference):
    selected, roles, seen = {}, {}, {}
    for key, role in [('observation_loss', 'five_observation_terms'), ('loss', 'five_terms_plus_native_prior')]:
        picked = set()
        for r in sorted([r for r in pool if r['ranking_eligible']], key=lambda r:(r[key], r['candidate_id'])):
            ident = condition_key(r['candidate'])
            if ident in picked:
                continue
            picked.add(ident)
            cid = seen.setdefault(ident, r['candidate_id'])
            selected[cid] = next(x for x in pool if x['candidate_id'] == cid)
            roles.setdefault(cid, []).append(role)
            if len(picked) == 2:
                break
    ref = next(r for r in pool if r['candidate_id'] == reference)
    cid = seen.setdefault(condition_key(ref['candidate']), reference)
    selected[cid] = next(r for r in pool if r['candidate_id'] == cid)
    roles.setdefault(cid, []).append('starting_reference')
    return dict(nominees=list(selected.values()), roles=roles, max_conditions=5,
                selection_scope='same TRAIN pool; labels and confirmation not used')


def run_phase(phase, candidates, pairs, obj, lam, workers):
    execution.OUT = OUT
    files = execution.execution_files(phase, candidates, pairs)
    jobs = [(c['candidate_id'], t, n) for c in candidates for t, n in pairs]
    execution.run_jobs(phase, jobs, files, workers)
    frozen()
    audit = execution.parameter_audit(require_complete=True, execution=files[0], seed_pairs=pairs,
                                      output_path=OUT/f'{phase}_parameter_audit.json')
    if audit['status'] != 'PARAMETER_APPLICATION_AUDIT_PASS':
        raise RuntimeError('actual physical parameter audit failed')
    rows = [dict(candidate=c, units={f'topo_{t}_dyn_{n}':dict(worker_path=str(files[0]/'workers'/f'{engine._stem(c["candidate_id"],t,n)}.json'))
                                     for t, n in pairs}) for c in candidates]
    return score_rows(rows, phase, obj, lam)


def analyze(final=False):
    cmd = [engine.PYTHON, str(ROOT/'scripts/analyze_topic4_contact_native_integrated_pilot.py')]
    if final:
        cmd.append('--final')
    subprocess.run(cmd, cwd=ROOT, env=engine.ENV, check=True)


def run(workers):
    d, obj = prepare()
    lam = d['lambda_contract']['primary']
    analyze()
    # Wait on physical completion, not the old classifier-dependent scorer.
    folder = NATIVE/'execution/A'
    snapshot = folder/'runtime_snapshot.json'
    cc = d['native_proposals']
    jobs = [(c['candidate_id'], t, n) for c in cc for t, n in TRAIN]
    while True:
        complete = [j for j in jobs if engine._complete(folder/'workers'/f'{engine._stem(*j)}.json', snapshot)]
        if len(complete) == 12:
            break
        physical = base.read(NATIVE/'status.json')
        if physical['status'] in {'ERROR_REVIEW_REQUIRED', 'FAILURE_DRAINING', 'BLOCKED'}:
            raise RuntimeError(f'inherited physical batch failed: {physical}')
        state('WAITING_FOR_INHERITED_NATIVE_A', complete=len(complete), total=12,
              inherited_status=physical['status'], automatic_integrated_continuation=True)
        time.sleep(20)
    audit = execution.parameter_audit(require_complete=True, execution=folder, seed_pairs=TRAIN,
                                      output_path=OUT/'inherited_A_parameter_audit.json')
    if audit['status'] != 'PARAMETER_APPLICATION_AUDIT_PASS':
        raise RuntimeError('inherited parameter audit failed')
    rr = [dict(candidate=c, units={f'topo_{t}_dyn_{n}':dict(worker_path=str(folder/'workers'/f'{engine._stem(c["candidate_id"],t,n)}.json'))
                                    for t, n in TRAIN}) for c in cc]
    pool = base.read(OUT/'history_scores.json')['candidates'] + score_rows(rr, 'inherited_A', obj, lam)
    analyze()
    cc = proposals(d, parents(pool, d['anchors']))
    pool += run_phase('adaptive_B', cc, TRAIN, obj, lam, workers)
    analyze()
    nomination = nominate(pool, d['anchors'][0]['candidate_id'])
    if (OUT/'nomination.json').exists() and base.read(OUT/'nomination.json') != nomination:
        raise RuntimeError('frozen nomination changed on restart')
    write(OUT/'nomination.json', nomination)
    confirm = run_phase('confirmation', [r['candidate'] for r in nomination['nominees']], CONFIRM, obj, lam, workers)
    identities = []
    for r in confirm:
        training = next(x for x in pool if x['candidate_id'] == r['candidate_id'])
        for uid, unit in r['units'].items():
            original = training['units'][f'topo_{unit["topology_seed"]}_dyn_7101']
            same = unit['static_array_identity'] == original['static_array_identity']
            identities.append(dict(candidate_id=r['candidate_id'], unit=uid, static_identity_equal=same))
            if not same:
                raise RuntimeError('same-network noise replay changed static identity')
    write(OUT/'confirmation_static_identity.json', identities)
    state('PHYSICAL_COMPLETE_ANALYZING')
    analyze(final=True)
    state('PILOT_COMPLETE_PENDING_SCIENTIFIC_REVIEW', complete=24+2*len(confirm), total=24+2*len(confirm),
          inherited_units=12, additional_units=12+2*len(confirm), automatic_next_search=False,
          model_frozen=False, propagation_mechanism_accepted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error('workers must be 1 to 8')
    OUT.mkdir(exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.prepare_only:
                prepare()
            else:
                run(args.workers)
        except Exception as exc:
            state('ERROR_REVIEW_REQUIRED', error=repr(exc))
            raise

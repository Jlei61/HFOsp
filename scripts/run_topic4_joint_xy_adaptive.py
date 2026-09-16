#!/usr/bin/env python3
"""Persistent train-distribution search; every completed round diagnoses and adapts."""
from pathlib import Path
import argparse
import copy
import fcntl
import json
import os
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_topic4_xy_research as base
from scripts import run_topic4_xy_direction_research as runtime
from src.topic4_joint_xy import (observable_groups, joint_features, projections,
    projected_quantiles, joint_distance, proposal_rng, round_action)
from src.topic4_xy_direction import onset_directions, direction_histogram, direction_distance, direction_summary
from src.topic4_xy_readout_audit import support_summary
from src.topic4_xy_search import field_descriptor, geometry_allowed, canonical_centers, audit_geometry

OLD = base.OUT
OUT = ROOT / 'results/topic4_sef_hfo/joint_rank_space_dual_core_search'
CONFIG = ROOT / 'config/topic4_joint_xy_adaptive_v1.json'
read, write, sha = base.read, base.write, base.sha
SOURCES = ['src/topic4_joint_xy.py', 'scripts/run_topic4_joint_xy_adaptive.py',
           'scripts/paper_figures/plot_topic4_joint_xy_progress.py', 'config/topic4_joint_xy_adaptive_v1.json',
           'tests/test_topic4_joint_xy.py']


def status(name, **kwargs):
    write(OUT/'status.json', dict(status=name, updated_unix=time.time(), **kwargs))


def threshold_table(calibration, n):
    # Use the next LARGER calibration sample size: never loosen the threshold
    # because an observation happens to sit just above a smaller-N boundary.
    size = next((s for s in sorted(map(int, calibration)) if s >= n), max(map(int, calibration)))
    return calibration[str(size)]['q95']


class Objective:
    def __init__(self):
        self.training, self.old_objective = base.training_contract()
        self.xy = np.asarray(read(OLD/'direction_objective_v2.json')['contact_xy_mm'])
        self.groups = self.training['groups']; self.pairs = self.training['pairs']
        self.patient = self.training['onsets_ms']
        self.features = joint_features(self.patient, self.xy, self.groups)
        self.axes = projections(self.features.shape[1])
        self.reference_q = projected_quantiles(self.features, self.axes)
        self.patient_direction = direction_histogram(onset_directions(self.patient, self.xy))

    def metrics(self, table, *, reference=None, reference_q=None, patient_direction=None):
        t = np.asarray(table)
        if len(t) == 0:
            return {'n_events': 0, 'joint_distance': None, 'exploration_score': 1.05,
                    'D_support': None, 'D_order': None, 'D_lag': None, 'direction_distance': None,
                    'component_status': {k: 'NOT_ESTIMABLE_LOW_EVENTS' for k in ('D_support', 'D_order', 'D_lag')},
                    'direction': direction_summary(onset_directions(t, self.xy)),
                    'support': support_summary(t, self.groups, self.pairs)}
        vector = self.old_objective.component_vector(
            t, reference if reference is not None else self.training['reference'],
            self.groups, self.pairs, self.training['embedding'], composite=False,
            components=('D_support', 'D_order', 'D_lag'))
        f = joint_features(t, self.xy, self.groups)
        direction_view = onset_directions(t, self.xy)
        dr = direction_summary(direction_view)
        dr['histogram'] = direction_histogram(direction_view)
        score = joint_distance(f, self.axes, self.reference_q if reference_q is None else reference_q)
        return {'n_events': len(t), 'joint_distance': score,
                'exploration_score': (score if score is not None else 1.) + .05*max(0, 16-len(t))/16,
                **{k: vector[k]['value'] for k in ('D_support', 'D_order', 'D_lag')},
                'component_status': {k: vector[k]['status'] for k in ('D_support', 'D_order', 'D_lag')},
                'direction_distance': direction_distance(np.asarray(dr['histogram']) if dr.get('histogram') is not None else None,
                    self.patient_direction if patient_direction is None else patient_direction),
                'direction': dr, 'support': support_summary(t, self.groups, self.pairs)}

    def calibrate(self, plan):
        path = OUT/'patient_calibration.json'
        if path.exists(): return read(path)
        status('CALIBRATING_TRAINING_DISTRIBUTION')
        rng = np.random.default_rng(plan['calibration']['seed'])
        blocks = self.training['block_ids']; unique = np.unique(blocks); samples = {}
        keys = ('joint_distance', 'D_support', 'D_order', 'D_lag', 'direction_distance')
        for n in plan['calibration']['sample_sizes']:
            draws = []
            for _ in range(plan['calibration']['draws']):
                left = rng.choice(unique, len(unique)//2, replace=False)
                mask = np.isin(blocks, left)
                source = np.flatnonzero(mask); ref = self.patient[~mask]
                index = rng.choice(source, n, replace=False)
                reference = self.old_objective.patient_reference(ref, self.groups, self.pairs, self.training['embedding'])
                metric = self.metrics(self.patient[index], reference=reference,
                    reference_q=projected_quantiles(self.features[~mask], self.axes),
                    patient_direction=direction_histogram(onset_directions(ref, self.xy)))
                draws.append({k: metric[k] for k in keys})
            samples[str(n)] = {'q95': {k: float(np.quantile([d[k] for d in draws if d[k] is not None], .95)) for k in keys},
                               'draws': draws}
            write(OUT/'calibration_progress.json', {'completed_sizes': list(samples)})
            print('Calibrated patient event count', n, flush=True)
        # Sanity: preserve masks while destroying ranks/space by contact permutation.
        rng = np.random.default_rng(2026090603)
        altered = self.patient.copy()
        for row in altered:
            idx = np.flatnonzero(np.isfinite(row)); row[idx] = rng.permutation(row[idx])
        null_score = joint_distance(joint_features(altered, self.xy, self.groups), self.axes, self.reference_q)
        result = {'version': plan['version'], 'samples': samples,
                  'patient_training_sha256': self.training['sha256'], 'null_rank_shuffle_joint_distance': null_score,
                  'self_distance': joint_distance(self.features, self.axes, self.reference_q),
                  'heldout_opened': False, 'threshold_role': 'development tolerance from training block splits, not biological equivalence proof'}
        write(path, result); return result


def read_worker(path, obj, plan):
    meta = read(path); npz = Path(meta['arrays']['path'])
    if sha(npz) != meta['arrays']['sha256']: raise RuntimeError('worker checksum changed')
    with np.load(npz) as z:
        if list(z['contact_names'].astype(str)) != obj.training['contact_names']:
            raise RuntimeError('contact identity mismatch')
        env = z['contact_envelope'].astype(float); dt = float(z['contact_envelope_dt_ms'])
        if not np.isfinite(z['active_fraction']).all(): raise RuntimeError('invalid trajectory')
    table, obs = observable_groups(env, dt, **plan['observation'])
    return table, {'seed': meta['seed'], 'observation': obs,
                   'worker_path': str(path), 'worker_sha256': sha(path),
                   'graph_sha256': meta['network_cache_source']['sha256'],
                   'runaway': meta['simulation']['runaway_early_stop_ms'] is not None,
                   'geometry': meta['xy_geometry_audit'], 'metrics': obj.metrics(table)}


def score_candidates(rows, directory, seeds, obj, plan, output):
    if output.exists():
        saved = read(output)
        if saved['candidate_ids'] != [r['candidate_id'] for r in rows]: raise RuntimeError('score manifest drift')
        for row in saved['candidates']:
            for unit in row['units']:
                path = Path(unit['worker_path'])
                if sha(path) != unit['worker_sha256']: raise RuntimeError('cached score worker changed')
                meta = read(path)
                if sha(meta['arrays']['path']) != meta['arrays']['sha256']: raise RuntimeError('cached score arrays changed')
        return saved['candidates']
    scores = []
    for row in rows:
        tables, units = [], []
        for seed in seeds:
            t, u = read_worker(directory/f'{row["candidate_id"]}_seed_{seed}.json', obj, plan)
            tables.append(t); units.append(u)
        metric = obj.metrics(np.concatenate(tables))
        scores.append({'candidate': row, 'candidate_id': row['candidate_id'],
                       'units': units, **metric,
                       'explorable': metric['joint_distance'] is not None and not any(u['runaway'] for u in units)})
    write(output, {'candidates': scores, 'candidate_ids': [r['candidate_id'] for r in rows],
                   'objective_version': plan['version'], 'angle_penalty': 0., 'direction_loss_weight': 0.})
    return scores


def new_proposals(pool, pos, seed, number, plan, diagnosis):
    cfg = plan['search']; rng = proposal_rng(seed, number)
    seen = {r['candidate']['node_field']['field_sha256'] for r in pool}
    anchors = []
    for r in sorted([p for p in pool if p['explorable']], key=lambda r:r['exploration_score']):
        c = np.asarray(r['candidate']['node_field']['centers_mm'])
        if all(min(np.linalg.norm(c-a), np.linalg.norm(c-a[::-1])) > 2. for a in anchors): anchors.append(c)
        if len(anchors) == 4: break
    fraction = cfg['restart_random_fraction'] if diagnosis['action'] == 'increase_random_restart_fraction' else cfg['random_fraction']
    rows = []; attempts = 0
    while len(rows) < cfg['proposals_per_round']:
        attempts += 1
        if attempts > 100000: raise RuntimeError('proposal geometry exhausted')
        random_start = not anchors or rng.random() < fraction
        if random_start:
            centers = rng.uniform(.75, 19.25, size=(2, 2)); parent = None
        else:
            parent = int(rng.integers(len(anchors)))
            centers = anchors[parent] + rng.normal(size=(2, 2))*rng.choice(cfg['local_scales_mm'])
        domain = 'interior' if rng.random() < cfg['interior_fraction'] else 'whole_sheet'
        centers = canonical_centers(centers)
        if not geometry_allowed(centers, pos, domain=domain): continue
        field = field_descriptor(centers)
        if field['field_sha256'] in seen: continue
        seen.add(field['field_sha256'])
        rows.append(base.decorate({'candidate_id': f'joint_r{number:03d}_{len(rows):03d}', 'domain': domain,
                    'proposal': 'fresh_uniform_restart' if random_start else 'random_multi_anchor_local',
                    'proposal_round': number, 'random_master_seed': seed, 'anchor_index': parent,
                    'node_field': field, 'geometry': audit_geometry(pos, centers, 1499)}))
    return rows


def run_phase(name, rows, seeds, duration, plan, lock):
    # Adapt the existing memory-aware dispatcher without editing its locked source.
    original = base.OUT
    def network(seed):
        p = OLD/'network_records'/f'{seed}.json'
        if not p.exists():
            subprocess.run([base.PYTHON, str(ROOT/'scripts/prebuild_topic4_xy_network.py'), '--seed', str(seed)],
                           cwd=ROOT, env=base.ENV, check=True)
        r = read(p)
        if r['status'] != 'CORRECTED_GRAPH_VALIDATED' or sha(r['path']) != r['sha256']:
            raise RuntimeError('invalid corrected graph')
        return r
    old_network = base.network_record
    try:
        base.OUT = OUT/'execution'; runtime.OUT = base.OUT; base.network_record = network
        base.SEEDS[name] = seeds; base.DURATIONS[name] = duration
        runtime.MIN_BUDGET[name] = (plan['runtime']['confirmation_minimum_worker_gib'] if duration >= 20000 else
                                   18. if duration >= 12000 else plan['runtime']['screen_minimum_worker_gib'])
        worker_lock = read(OLD/'source_lock.json')['source_hashes']
        runtime.run_phase(name, rows, worker_lock, lock, plan['runtime']['maximum_workers'])
    finally:
        base.OUT = original; runtime.OUT = OLD; base.network_record = old_network


def is_qualified(result, calibration, plan, *, confirmation=False):
    n = result['n_events']; threshold = threshold_table(calibration['samples'], n)
    checks = {k: result[k] is not None and result[k] <= threshold[k] for k in threshold}
    checks['conditional_support'] = result['support']['conditional_gate']
    checks['no_runaway'] = not any(u['runaway'] for u in result['units'])
    checks['two_complete_cores'] = all(u['geometry']['minimum_clearance_mm'] >= plan['confirmation']['require_disk_clearance_mm'] for u in result['units'])
    checks['sufficient_events'] = n >= (plan['confirmation']['minimum_pooled_events'] if confirmation else 16)
    if confirmation:
        checks['six_networks'] = len(result['units']) == plan['confirmation']['n_seeds']
        checks['events_per_seed'] = all(u['metrics']['n_events'] >= plan['confirmation']['minimum_events_per_seed'] for u in result['units'])
        good = sum(u['metrics']['joint_distance'] is not None and u['metrics']['joint_distance'] <=
                   threshold_table(calibration['samples'], u['metrics']['n_events'])['joint_distance'] for u in result['units'])
        checks['independent_joint_distribution'] = good >= plan['confirmation']['minimum_good_joint_loss_seeds']
    return {'pass': all(checks.values()), 'checks': checks, 'thresholds': threshold}


def fig5_handoff(result, phase, plan, lock):
    from scripts import chain_topic4_xy_round1_fig5 as chain
    folder = OUT/'fig5_followup'; folder.mkdir(exist_ok=True)
    config = read(ROOT/'config/topic4_xy_round1_fig5_followup.json')
    seeds = [u['seed'] for u in result['units']]
    config.update(version='joint_rank_space_xy_fig5_v1', confirmation_seeds=seeds,
                  analysis_topology_seeds=seeds[:3],
                  candidate_rule='Pre-nominated joint-distribution candidate passing six-network confirmation; no runner-up selection on confirmation.')
    cfg = read(OUT/'execution'/phase/'execution_config.json')
    transition = ROOT/cfg['inputs']['transition_config']['path']
    if not transition.exists(): transition = base.ART/cfg['inputs']['transition_config']['path']
    if sha(transition) != cfg['inputs']['transition_config']['sha256']: raise RuntimeError('transition config drift')
    chain.fixed_json(folder/'analysis_plan.json', config)
    chain.fixed_json(folder/'runtime_lock.json', {'hashes': {**lock, str(folder/'analysis_plan.json'): sha(folder/'analysis_plan.json')},
                       'source_search': str(OUT), 'objective_version': plan['version']})
    record_paths = [OUT/'patient_calibration.json', OUT/'objective_contract.json', OUT/'qualified_substrate.json', transition]
    handoff = {'status': 'DEVELOPMENT_SUBSTRATE_QUALIFIED_FOR_FIG5', 'candidate': result['candidate'],
               'selection_result': read(OUT/'qualified_substrate.json')['selection_result'], 'confirmation_result': result,
               'selection_rule': config['candidate_rule'], 'runtime_lock_sha256': sha(folder/'runtime_lock.json'),
               'input_hashes': {str(p): sha(p) for p in record_paths}, 'transition_config': str(transition),
               'artifact_root': str(base.ART), 'network_cache': cfg['network_cache'],
               'networks': {str(s): cfg['corrected_networks'][str(s)] for s in seeds[:3]},
               'source_search': str(OUT), 'final_substrate_frozen': False, 'development_parameters_locked': True,
               'author_acceptance': False, 'patient_heldout_opened': False, 'ictal_opened': False,
               'claim_boundary': config['claim_boundary']}
    chain.fixed_json(folder/'development_substrate.json', handoff)
    status('QUALIFIED_BASE_FIG5_RUNNING', candidate_id=result['candidate_id'])
    chain.execute(OUT, folder, config, folder/'runtime_lock.json')
    status('QUALIFIED_BASE_FIG5_STAGE_COMPLETE', candidate_id=result['candidate_id'], fig5_status=read(folder/'status.json'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--score-only', action='store_true'); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    guard = open(OUT/'controller.lock', 'a'); fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(CONFIG)
    base.ENV['LD_LIBRARY_PATH'] = '/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib'
    hashes = {str(ROOT/p): sha(ROOT/p) for p in SOURCES}
    hashes.update({str(ROOT/p): h for p, h in read(OLD/'source_lock.json')['source_hashes'].items()})
    # The original Fig5 chain is a locked downstream consumer, not a new fit target.
    hashes.update(read(OLD/'fig5_followup/runtime_lock.json')['hashes'])
    contract_path = OUT/'objective_contract.json'
    if contract_path.exists():
        contract = read(contract_path)
        if contract['config_sha256'] != sha(CONFIG) or contract['source_hashes'] != hashes:
            raise RuntimeError('adaptive search code/config changed; create a new version')
    else:
        contract = {'version': plan['version'], 'config_sha256': sha(CONFIG), 'source_hashes': hashes,
                    'master_seed': secrets.randbits(32), 'random_seed_reuse': 'within-round common networks; distinct recorded proposal streams across rounds',
                    'created_unix': time.time(), 'patient_heldout_opened': False,
                    'no_direction_or_core_alignment_penalty': True}
        write(contract_path, contract)
    runtime.verify_amendment(hashes)
    obj = Objective(); calibration = obj.calibrate(plan)
    frozen_path = OUT/'analysis_input_lock.json'
    if not frozen_path.exists():
        fixed_paths = [OUT/'patient_calibration.json', contract_path, obj.training['path'],
                       OLD/'direction_objective_v2.json', OLD/'search_design.json']
        write(frozen_path, {'hashes': {str(p): sha(p) for p in fixed_paths}})
    fixed_inputs = read(frozen_path)['hashes']; runtime.verify_amendment(fixed_inputs)
    hashes.update(fixed_inputs)
    if args.prepare_only:
        status('PREPARED_JOINT_DISTRIBUTION_SEARCH'); return
    if (OUT/'qualified_substrate.json').exists() and not args.score_only:
        accepted = read(OUT/'qualified_substrate.json')
        if not is_qualified(accepted['candidate'], calibration, plan, confirmation=True)['pass']:
            raise RuntimeError('stored qualification no longer verifies')
        fig5_handoff(accepted['candidate'], accepted['phase'], plan, hashes)
        return
    design = read(OLD/'search_design.json')['candidates']
    # Exclude the unmatched-budget diagnostic from the adaptive search population.
    design = [r for r in design if r['node_field']['target_count'] == 1499]
    pool = score_candidates(design, OLD/'global/workers', [2511, 2512], obj, plan, OUT/'baseline_scores.json')
    write(OUT/'baseline_assessment.json', {'candidates': [{'candidate_id': r['candidate_id'],
        'assessment': is_qualified(r, calibration, plan)} for r in pool], 'n_geometries': len(pool)})
    if args.score_only:
        status('BASELINE_RESCORED', best=min(pool, key=lambda r:r['exploration_score'])['candidate_id']); return
    pos = base.positions()
    history = []; attempts = read(OUT/'confirmation_attempts.json') if (OUT/'confirmation_attempts.json').exists() else []
    for number in range(1, plan['search']['automatic_rounds_before_capacity_review']+1):
        directory = OUT/'rounds'/f'{number:03d}'; directory.mkdir(parents=True, exist_ok=True)
        best = min(pool, key=lambda r:r['exploration_score'])
        threshold = threshold_table(calibration['samples'], best['n_events'])
        diagnosis = round_action(history, best['exploration_score'],
                                 (best['D_support'] or 1.)/threshold['D_support'], best['n_events'])
        path = directory/'design.json'
        if path.exists(): rows = read(path)['candidates']
        else:
            rows = new_proposals(pool, pos, contract['master_seed'], number, plan, diagnosis)
            write(path, {'candidates': rows, 'preceding_diagnosis': diagnosis,
                         'seed_sequence': [contract['master_seed'], number, 0]})
        phase = f'round_{number:03d}'
        duration = plan['search']['low_yield_duration_ms'] if diagnosis['action'] == 'longer_paired_screen' else plan['search']['fit_duration_ms']
        status('RUNNING_RANDOM_MULTI_START_SEARCH', round=number, phase=phase, diagnosis=diagnosis)
        run_phase(phase, rows, plan['search']['fit_seeds'], duration, plan, hashes)
        scored = score_candidates(rows, OUT/'execution'/phase/'workers', plan['search']['fit_seeds'], obj, plan, directory/'scores.json')
        pool.extend(scored)
        best = min(pool, key=lambda r:r['exploration_score']); history.append(best['exploration_score'])
        assessments = [{'candidate_id': r['candidate_id'], 'assessment': is_qualified(r, calibration, plan)} for r in scored]
        report = {'round': number, 'best_candidate_id': best['candidate_id'], 'best_score': best['exploration_score'],
                  'best_metrics': best, 'n_candidates_seen': len(pool), 'assessments': assessments,
                  'next_action': round_action(history[:-1], best['exploration_score'],
                      (best['D_support'] or 1.)/threshold_table(calibration['samples'], best['n_events'])['D_support'], best['n_events']),
                  'loss_modified': False, 'initialization_reused': False, 'global_optimum_claim': False,
                  'fresh_random_proposals': sum(r['proposal'] == 'fresh_uniform_restart' for r in rows),
                  'local_random_proposals': sum(r['proposal'] == 'random_multi_anchor_local' for r in rows)}
        write(directory/'analysis.json', report)
        failures = [k for k, ok in is_qualified(best, calibration, plan)['checks'].items() if not ok]
        (directory/'analysis.md').write_text(
            f'# 第 {number} 轮自动分析\n\n当前最佳训练候选为 `{best["candidate_id"]}`；'
            f'累计比较 {len(pool)} 个几何，本轮全新随机起点 {report["fresh_random_proposals"]} 个、'
            f'多中心局部随机提案 {report["local_random_proposals"]} 个。\n\n'
            f'联合分布损失 {best["joint_distance"]}；群体观测事件 {best["n_events"]} 个。'
            f'未满足的开发验收项：{", ".join(failures) or "无，等待独立网络确认"}。\n\n'
            f'下一步诊断：`{report["next_action"]["diagnosis"]}`；动作：`{report["next_action"]["action"]}`。'
            '样本不足时增加观察时长；多轮停滞时增加随机重启。损失、观测与患者目标保持锁定，'
            '不会因为未通过而自动降低门槛。现有证据若无法区分优化覆盖不足与模型容量不足，会保留未确定判断。\n')
        subprocess.run([base.PYTHON, str(ROOT/'scripts/paper_figures/plot_topic4_joint_xy_progress.py'), '--out', str(OUT)],
                       cwd=ROOT, env=base.ENV, check=True)
        eligible = [r for r in pool if is_qualified(r, calibration, plan)['pass'] and
                    r['candidate_id'] not in [a['candidate_id'] for a in attempts]]
        pending_path = OUT/'nominations'/f'confirmation_{len(attempts):03d}.json'
        pending_nomination = read(pending_path) if pending_path.exists() else None
        if pending_nomination:
            eligible = ([next(r for r in pool if r['candidate_id'] == pending_nomination['candidate_id'])]
                        if number >= pending_nomination['round'] else [])
        if eligible:
            nominee = min(eligible, key=lambda r:r['exploration_score'])
            seeds = list(range(plan['confirmation']['seed_base']+len(attempts)*10,
                               plan['confirmation']['seed_base']+len(attempts)*10+plan['confirmation']['n_seeds']))
            confirmation = f'confirmation_{len(attempts):03d}'
            nomination = {'candidate_id': nominee['candidate_id'], 'candidate': nominee['candidate'],
                          'seeds': seeds, 'phase': confirmation, 'selected_before_new_networks': True, 'round': number}
            if pending_nomination:
                if pending_nomination != nomination: raise RuntimeError('pending nomination changed on resume')
            else: write(pending_path, nomination)
            status('CONFIRMING_PRENOMINATED_GEOMETRY', **nomination)
            run_phase(confirmation, [nominee['candidate']], seeds, plan['confirmation']['duration_ms'], plan, hashes)
            result = score_candidates([nominee['candidate']], OUT/'execution'/confirmation/'workers', seeds, obj, plan,
                                      OUT/'nominations'/f'{confirmation}_scores.json')[0]
            assessment = is_qualified(result, calibration, plan, confirmation=True)
            attempts.append({**nomination, 'assessment': assessment}); write(OUT/'confirmation_attempts.json', attempts)
            if assessment['pass']:
                write(OUT/'qualified_substrate.json', {'candidate': result, 'assessment': assessment,
                        'selection_result': nominee, 'phase': confirmation, 'development_working_point': True, 'uniqueness_established': False})
                if plan['auto_fig5_after_qualified_confirmation']: fig5_handoff(result, confirmation, plan, hashes)
                else: status('QUALIFIED_DEVELOPMENT_SUBSTRATE', candidate_id=result['candidate_id'])
                return
        status('ROUND_ANALYZED_CONTINUING', round=number, best_candidate=best['candidate_id'], next_action=report['next_action'])
    status('NEEDS_MODEL_CAPACITY_REVIEW', rounds=len(history), best_candidate=best['candidate_id'],
           explanation='Eight randomized batches complete. Goal remains active; inspect observation, objective controls and search coverage before a new version. Acceptance not relaxed.')


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        status('FAILED_NEEDS_ENGINEERING_DIAGNOSIS', reason=str(exc)); raise

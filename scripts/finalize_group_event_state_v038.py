#!/usr/bin/env python3
"""Patient-first v0.3.8 scientific closure summary and reports.

This finalizer is intentionally stricter than a queue counter.  It keeps the
four questions separate: temporal dynamics, trained long-history credit,
multi-endpoint pathology readout, and frozen seizure transfer.  Seed repeats
measure optimisation stability; the patient or held-out seizure remains the
scientific unit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.contracts import atomic_json


BASE = Path("/data/hfosp_group_event_state_v0_3_8_core_expansion")
LONG = ("epilepsiae_1096", "epilepsiae_253", "epilepsiae_958",
        "epilepsiae_1077", "epilepsiae_1125", "epilepsiae_916")
LONG_H2A = tuple(subject for subject in LONG if subject != "epilepsiae_916")
MEDIUM = ("epilepsiae_1146", "epilepsiae_384", "epilepsiae_548",
          "epilepsiae_583", "epilepsiae_922")
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)
FAMILIES = ("event", "grid", "dual")
ENDPOINTS = ("count", "burden", "community", "coupling", "mixture", "embedding", "mark")
ENDPOINT_LABELS = {
    "count": "未来事件数", "burden": "事件负荷/范围", "community": "community occupancy",
    "coupling": "跨 community coupling", "mixture": "repertoire mixture",
    "embedding": "连续 repertoire embedding", "mark": "条件频带/延迟/波形 mark",
}
LONG_CREDIT_FRACTION_THRESHOLD = 0.01


def _short_subject(subject: str) -> str:
    return str(subject).replace("epilepsiae_", "E")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cards(path: Path, repair: Path | list[Path] | None = None) -> list[dict[str, Any]]:
    cards = []
    for item in sorted(path.glob('**/card.json')):
        original = _read(item)
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        repairs = repair if isinstance(repair, list) else ([] if repair is None else [repair])
        for root in repairs:
            overlay = root / item.relative_to(path)
            if not overlay.exists():
                continue
            fixed = _read(overlay)
            if fixed.get('repair_source_sha256') != digest:
                raise ValueError(f'repair overlay source mismatch: {overlay}')
            if Path(fixed['repair_source_card']).resolve() != item.resolve():
                raise ValueError(f'repair overlay source path mismatch: {overlay}')
            if fixed.get('status') == 'COMPLETE' or original.get('status') == 'NOT_ESTIMABLE':
                original = fixed
                original['_loaded_overlay_card'] = str(overlay)
                original['_loaded_overlay_sha256'] = hashlib.sha256(overlay.read_bytes()).hexdigest()
            else:
                original['repair_unavailable'] = str(overlay)
        original['_loaded_source_card'] = str(item)
        original['_loaded_source_sha256'] = digest
        cards.append(original)
    return cards


def _attach_lineage(cards, lineage, repair_root):
    upstream = {row['source_card']: row for row in lineage.get('h1_cards', [])}
    downstream = {row['card']: row for row in lineage.get('downstream_bindings', [])}
    for card in cards:
        source = card['_loaded_source_card']
        if source in upstream:
            card['_state_lineage'] = upstream[source]
            row = upstream[source]
            replay = repair_root / 'checkpoint_replay' / row['scope'] / row['family'] / row['subject'] / f"seed{row['seed']}" / 'card.json'
            if replay.exists():
                result = _read(replay)
                if result.get('input_hashes', {}).get(row['checkpoint']) != row['checkpoint_sha256']:
                    raise ValueError(f'checkpoint replay lineage mismatch: {replay}')
                card['_replay_evidence'] = result
        elif source in downstream:
            binding = downstream[source]
            parent = upstream.get(binding.get('upstream_card'))
            if parent:
                card['_state_lineage'] = parent
        elif card.get('freeze_card') in downstream:
            card['_h2b_lineage'] = downstream[card['freeze_card']]
        if '_state_lineage' in card:
            row = card['_state_lineage']
            verified = repair_root / 'checkpoint_replay_verified' / row['scope'] / row['family'] / row['subject'] / f"seed{row['seed']}" / 'card.json'
            if verified.exists():
                result = _read(verified)
                if result.get('input_hashes', {}).get(row['checkpoint']) != row['checkpoint_sha256'] or result.get('verified_preprocessing') is not True:
                    raise ValueError(f'verified preprocessing lineage mismatch: {verified}')
                card['_verified_preprocessing_replay'] = result
            branch_path = repair_root / 'dual_branch_controls' / row['scope'] / row['family'] / row['subject'] / f"seed{row['seed']}" / 'card.json'
            if row['family'] == 'dual' and branch_path.exists():
                result = _read(branch_path)
                if result.get('source_checkpoint_sha256') != row['checkpoint_sha256'] or result.get('source_card') != row['source_card']:
                    raise ValueError(f'branch control checkpoint mismatch: {branch_path}')
                card['_branch_control_evidence'] = {**result, 'audit_path': str(branch_path),
                                                    'audit_sha256': hashlib.sha256(branch_path.read_bytes()).hexdigest()}
    return cards



def _floored_control(*, control_gain: float | None, baseline_gain: float | None) -> float | None:
    """Control-arm gain with the null floored at the strong baseline.

    A constant or matched-random arm is nested above the strong baseline, so it
    can score *worse* than simply not having a state block at all.  When it
    does, ``control - state`` measures how much the control hurt, not how much
    the state helped: for the retained short-scale candidate the random
    component is 0.34 log-score worse than the baseline, which is two thirds of
    its headline contrast.  A modeller offered a harmful control would fall
    back to the baseline, so the reportable contrast is the smaller of the two
    gains.
    """
    if control_gain is None or baseline_gain is None or not np.all(np.isfinite([control_gain, baseline_gain])):
        return None
    return float(min(float(control_gain), float(baseline_gain)))


def _control_harm(*, control_gain: float | None, baseline_gain: float | None) -> bool | None:
    """True when the control scores worse than the baseline it sits above."""
    if control_gain is None or baseline_gain is None or not np.all(np.isfinite([control_gain, baseline_gain])):
        return None
    return bool(float(control_gain) > float(baseline_gain))


def _code_provenance_audit(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Refuse to let one summary silently pool two versions of the training code.

    v0.3.7 gained this audit after a summary was built from cards that a later
    edit had superseded; the v0.3.8 finaliser dropped it, and the core run does
    contain two ``h2a.py`` versions split across seeds inside one comparison
    group.  Reported per source file so a benign queue-level split can be told
    apart from a split inside a single median.
    """
    versions: dict[str, set[str]] = {}
    groups = defaultdict(lambda: defaultdict(list))
    missing_cards = []
    missing = 0
    for card in cards:
        provenance = card.get("code_provenance")
        if not isinstance(provenance, dict) or not provenance.get("source_sha256"):
            missing += 1
            missing_cards.append({key: card.get(key) for key in ('subject', 'seed', 'status', '_loaded_source_card')})
            continue
        name = str(provenance.get("source_file", "unknown"))
        versions.setdefault(name, set()).add(str(provenance["source_sha256"]))
        source = Path(card.get('_loaded_source_card', 'unknown'))
        scope = card.get('_audit_scope', source.parent.parent.parent.name)
        family = card.get('_state_lineage', {}).get('family') or card.get('state_provenance', {}).get('family') or card.get('_audit_family', source.parent.parent.name)
        groups[(scope, family, card.get('subject'), name)][str(provenance['source_sha256'])].append(card.get('seed'))
    mixed = {name: sorted(hashes) for name, hashes in versions.items() if len(hashes) > 1}
    return {
        "n_cards": len(cards),
        "cards_without_code_provenance": missing,
        "missing_cards": missing_cards,
        "within_comparison_versions": [
            dict(zip(('scope', 'family', 'subject', 'source_file'), key),
                 seeds_by_sha256={sha: sorted(seeds, key=str) for sha, seeds in sorted(values.items())},
                 mixed_versions=len(values) > 1)
            for key, values in sorted(groups.items(), key=lambda item: str(item[0]))],
        "source_versions": {name: sorted(hashes) for name, hashes in sorted(versions.items())},
        "mixed_source_versions": mixed,
        "single_version_per_source_file": bool(not mixed),
        "rule": (
            "Hashes identify source versions, not semantic equivalence. Within-comparison "
            "mixtures require source review before pooling is used for confirmation. Separate "
            "queues remain separate; missing provenance is not a biological negative."
        ),
    }


def _event_updated(card):
    return card.get('_state_lineage', {}).get('modules', {}).get('event', {}).get('selected_changed') is True


def _h1_credit_multitarget_chain(h1, credit):
    lookup = {(row['subject'], row['horizon_hours']): row for row in credit['rows'] if row['branch'] == 'event'}
    rows = []
    endpoint_groups = (('burden',), ('community',), ('coupling',), ('embedding', 'mixture'), ('mark',))
    for cohort in h1.values():
        for row in cohort['dual']['rows']:
            bound = lookup.get((row['subject'], row['horizon_seconds'] / 3600))
            if bound is None: continue
            credit_seeds = {value['seed']: value for value in bound['joint_evidence']['per_seed']}
            seed_rows = []
            for seed in row['composite_with_learned_event_input_joint_evidence']['per_seed']:
                candidate = credit_seeds.get(seed['seed'], {})
                n_endpoints = sum(any(seed['seed'] in row['endpoint_evidence'][endpoint]['joint_evidence']['passing_seeds'] for endpoint in group)
                                  for group in endpoint_groups)
                seed_rows.append({'seed': seed['seed'], 'checkpoint_sha256': candidate.get('checkpoint_sha256'),
                                  'checks': {'learned_H1': all(seed['checks'].values()),
                                             'checkpoint_hash_match': seed.get('checkpoint_sha256') is not None and seed.get('checkpoint_sha256') == candidate.get('checkpoint_sha256'),
                                             'same_dual_head_credit': candidate.get('checks') is not None and all(candidate['checks'].values()),
                                             'two_non_count_endpoint_groups': n_endpoints >= 2}})
            fixed_pairs = _fixed_endpoint_pair_evidence(
                [{**seed, 'checks': {k: v for k, v in seed['checks'].items() if k != 'two_non_count_endpoint_groups'}} for seed in seed_rows],
                {'/'.join(group): set().union(*(row['endpoint_evidence'][endpoint]['joint_evidence']['passing_seeds'] for endpoint in group))
                 for group in endpoint_groups})
            joint = fixed_pairs['joint_evidence']
            branch_seeds = set(row['learned_event_joint_evidence']['passing_seeds'])
            event_pairs = _fixed_endpoint_pair_evidence(
                [{**seed, 'checks': {**{k: v for k, v in seed['checks'].items() if k != 'two_non_count_endpoint_groups'},
                                     'event_branch_temporal_value': seed['seed'] in branch_seeds}}
                 for seed in seed_rows],
                {'/'.join(group): set().union(*(row['endpoint_evidence'][endpoint]['joint_evidence']['passing_seeds'] for endpoint in group))
                 for group in endpoint_groups})
            rows.append({'subject': row['subject'], 'family': 'dual', 'horizon_seconds': row['horizon_seconds'],
                         'joint_evidence': joint, 'same_model_functional_state_candidate': joint['stable_positive_3_of_5'],
                         'fixed_endpoint_pair_evidence': fixed_pairs,
                         'event_branch_joint_evidence': event_pairs['joint_evidence'],
                         'event_branch_fixed_endpoint_pair_evidence': event_pairs,
                         'event_branch_functional_state_candidate': event_pairs['joint_evidence']['stable_positive_3_of_5'],
                         'boundary': 'composite H1 with learned event input and same-checkpoint credit; event-specific temporal value needs separate branch controls; not contact-decoder H2a or seizure H2b'})
    return {'rows': rows}


def _adapter_qualified(card, name):
    stage = card.get('stages', {}).get(name, {})
    if 'training_budget_exhausted' in stage:
        return stage['training_budget_exhausted'] is False and int(stage.get('epochs_run', 0)) > 0
    # Original cards retain epoch counts and the exact patience rule.
    return (int(stage.get('epochs_run', 0)) > 0 and
            int(stage.get('epochs_run', 0)) - int(stage.get('selected_epoch', 0)) >= int(stage.get('patience_epochs', 10**9)))


def _jsons(path: Path) -> list[dict[str, Any]]:
    return [_read(item) for item in sorted(path.glob("**/*.json"))
            if item.name != "queue_status.json"]


def _finite(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        if value is not None:
            number = float(value)
            if np.isfinite(number):
                output.append(number)
    return output


def _median(values: Iterable[Any]) -> float | None:
    items = _finite(values)
    return float(np.median(items)) if items else None


def _direction(values: Iterable[Any]) -> dict[str, Any]:
    items = _finite(values)
    return {
        "median": float(np.median(items)) if items else None,
        "positive_seeds": int(sum(value > 0.0 for value in items)),
        "negative_seeds": int(sum(value < 0.0 for value in items)),
        "estimated_seeds": len(items),
        "stable_positive_3_of_5": len(items) >= 3 and sum(value > 0.0 for value in items) >= 3,
    }


def _positive(value: Any) -> bool:
    return value is not None and np.isfinite(float(value)) and float(value) > 0


def _seed(card: dict[str, Any]) -> int:
    return int(card['seed'] if 'seed' in card else card['state_seed'])


def _joint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count complete within-seed evidence; never AND marginal seed votes."""
    seeds = [int(row['seed']) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError('duplicate seed in joint scientific evidence')
    passing = [int(row['seed']) for row in rows
               if row['checks'] and all(bool(v) for v in row['checks'].values())]
    return {'passing_seeds': passing, 'n_joint_positive': len(passing),
            'n_registered_seeds': len(SEEDS), 'n_observed_seeds': len(rows),
            'stable_positive_3_of_5': len(passing) >= 3,
            'per_seed': rows}


def _joint_contrasts(cards: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, Any]:
    keys = tuple(keys)
    return _joint([{'seed': _seed(card), 'checks': {
        key: _positive(card.get('primary_contrasts', {}).get(key)) for key in keys
    }} for card in cards])


def _fixed_endpoint_pair_evidence(common_rows, endpoint_seeds):
    """The same endpoint identities must replicate in the same seeds."""
    pairs = {}
    for first, second in combinations(sorted(endpoint_seeds), 2):
        pairs[f'{first}+{second}'] = _joint([{**{k: v for k, v in row.items() if k != 'checks'}, 'checks': {
            **row['checks'], f'endpoint:{first}': row['seed'] in endpoint_seeds[first],
            f'endpoint:{second}': row['seed'] in endpoint_seeds[second],
        }} for row in common_rows])
    best_name = max(pairs, key=lambda name: pairs[name]['n_joint_positive'])
    return {'best_pair': best_name, 'joint_evidence': pairs[best_name], 'all_pairs': pairs,
            'passing_pairs': [name for name, joint in pairs.items() if joint['stable_positive_3_of_5']]}


def _stage_qualified(card: dict[str, Any], name: str) -> bool:
    if name == 'random' and card.get('repair_unavailable'):
        return False
    stage = card.get('stages', {}).get(name, {})
    return (_positive(stage.get('first_step_gradient_norm'))
            and _positive(stage.get('peak_parameter_delta_from_stage_start'))
            and stage.get('training_budget_exhausted') is False)


def _group(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        grouped[str(card["subject"])].append(card)
    return dict(grouped)


def _score(card: dict[str, Any], arm: str, horizon: str) -> dict[str, Any] | None:
    try:
        return card["selection_scores"][arm]["by_horizon"][horizon]
    except (KeyError, TypeError):
        return None


def _difference(card: dict[str, Any], baseline: str, model: str, horizon: str,
                endpoint: str | None = None) -> float | None:
    left, right = _score(card, baseline, horizon), _score(card, model, horizon)
    if left is None or right is None:
        return None
    if endpoint is None:
        return float(left["total"] - right["total"])
    return float(left["endpoints"][endpoint] - right["endpoints"][endpoint])


def _training_stage(cards: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    rows = [card.get("stages", {}).get(stage) for card in cards]
    rows = [row for row in rows if isinstance(row, dict)]
    explored = [
        float(row.get("first_step_gradient_norm", 0.0) or 0.0) > 0.0
        and float(row.get("peak_parameter_delta_from_stage_start", 0.0) or 0.0) > 0.0
        for row in rows
    ]
    return {
        "n": len(rows),
        "selected_after_init": int(sum(not bool(row.get("selected_at_init", False)) for row in rows)),
        "path_explored": int(sum(explored)),
        "budget_exhausted": int(sum(bool(row.get("training_budget_exhausted", False)) for row in rows)),
        "median_selected_step": _median(row.get("selected_step") for row in rows),
        "median_first_step_gradient_norm": _median(row.get("first_step_gradient_norm") for row in rows),
        "median_peak_parameter_delta": _median(row.get("peak_parameter_delta_from_stage_start") for row in rows),
        "training_interpretable": len(rows) >= 3 and sum(explored) >= 3
        and sum(bool(row.get("training_budget_exhausted", False)) for row in rows) <= 1,
    }


def _random_background_index(cards: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(card["subject"]), int(card["seed"])): card for card in cards}


def _h1_family(cards: list[dict[str, Any]], family: str,
               random_background: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for subject, subject_cards in sorted(_group(cards).items()):
        horizons = sorted({int(float(value)) for card in subject_cards for value in card["horizons_seconds"]})
        relevant_stage = "event" if family == "dual" else "state"
        training = {
            "state": _training_stage(subject_cards, relevant_stage),
            "B_mark": _training_stage(subject_cards, "bmark"),
        }
        if family == "dual":
            training["background_state"] = _training_stage(subject_cards, "background_state")
            training["random_event"] = _training_stage(subject_cards, "random")
        else:
            training["random"] = _training_stage(subject_cards, "random")
        for seconds in horizons:
            key = str(seconds)
            if family == "event":
                strong, model, constant, random = "B_mark", "S_event", "S_event_constant", "random_frozen"
            elif family == "grid":
                strong, model, constant, random = "B_mark", "S_grid", "S_grid_constant", "random_frozen"
            else:
                strong, model, constant, random = (
                    "B_mark_current_background", "S_dual", "S_dual_constant_all", "random_event"
                )
            gains = [_difference(card, strong, model, key) for card in subject_cards]
            dynamic = [_difference(card, constant, model, key) for card in subject_cards]
            correct_time = [
                (card.get("time_shift_by_horizon", {}).get(key) or {}).get("gain")
                for card in subject_cards
            ]
            random_gain = (
                [_difference(card, random, model, key) for card in subject_cards]
                if family != "dual" else
                [_difference(card, "random_event", "S_dual", key) for card in subject_cards]
            )
            endpoint_evidence = {}
            for endpoint in ENDPOINTS:
                endpoint_gain = [_difference(card, strong, model, key, endpoint) for card in subject_cards]
                endpoint_dynamic = [_difference(card, constant, model, key, endpoint) for card in subject_cards]
                endpoint_time = []
                for card in subject_cards:
                    shifted = card.get("time_shift_by_horizon", {}).get(key)
                    endpoint_time.append(
                        None if shifted is None else
                        float(shifted["shifted"]["endpoints"][endpoint]
                              - shifted["correct"]["endpoints"][endpoint])
                    )
                endpoint_evidence[endpoint] = {
                    "gain_over_strong_baseline": _direction(endpoint_gain),
                    "dynamic_over_constant": _direction(endpoint_dynamic),
                    "correct_time_over_shifted": _direction(endpoint_time),
                }
                endpoint_joint = _joint([{'seed': card['seed'], 'checks': {
                    'baseline': _positive(endpoint_gain[i]),
                    'constant': _positive(endpoint_dynamic[i]),
                    'time_shift': _positive(endpoint_time[i]),
                }} for i, card in enumerate(subject_cards)])
                endpoint_evidence[endpoint]['joint_evidence'] = endpoint_joint
                endpoint_evidence[endpoint]['directional_dynamic_endpoint'] = endpoint_joint['stable_positive_3_of_5']
            support = [
                card.get("independent_windows_by_horizon", {}).get(key, {}).get("independent_windows")
                for card in subject_cards
            ]
            random_background_values = []
            if family == "dual":
                for card in subject_cards:
                    control = random_background.get((subject, int(card["seed"])))
                    random_background_values.append(
                        None if control is None else
                        control.get("by_horizon", {}).get(key, {}).get(
                            "learned_gain_over_random_background"
                        )
                    )
            dynamic_floored = [
                _floored_control(control_gain=dynamic[i], baseline_gain=gains[i])
                for i in range(len(subject_cards))
            ]
            random_floored = [
                _floored_control(control_gain=random_gain[i], baseline_gain=gains[i])
                for i in range(len(subject_cards))
            ]
            evidence = {
                "gain_over_strong_baseline": _direction(gains),
                "dynamic_over_constant": _direction(dynamic),
                "correct_time_over_shifted": _direction(correct_time),
                "gain_over_random_event_or_state": _direction(random_gain),
                "learned_background_over_random": _direction(random_background_values),
                # Reportable versions: a control that scores worse than the
                # strong baseline is harmful rather than null, and its raw
                # contrast credits the state with the control's own damage.
                "dynamic_over_constant_floored": _direction(dynamic_floored),
                "gain_over_random_floored": _direction(random_floored),
                "control_worse_than_strong_baseline": {
                    "constant_seeds": sum(
                        1 for i in range(len(subject_cards))
                        if _control_harm(control_gain=dynamic[i], baseline_gain=gains[i])
                    ),
                    "random_seeds": sum(
                        1 for i in range(len(subject_cards))
                        if _control_harm(control_gain=random_gain[i], baseline_gain=gains[i])
                    ),
                    "n_seeds": len(subject_cards),
                },
            }
            joint_rows = []
            for i, card in enumerate(subject_cards):
                event_random = _positive(random_gain[i]) and _stage_qualified(card, 'random')
                background_random = False
                if family == 'dual':
                    control = random_background.get((subject, int(card['seed'])), {})
                    control_stage = control.get('training', {})
                    background_random = (_positive(random_background_values[i])
                                         and _positive(control_stage.get('first_step_gradient_norm'))
                                         and _positive(control_stage.get('peak_parameter_delta_from_stage_start'))
                                         and control_stage.get('training_budget_exhausted') is False)
                checks = {
                    'baseline': _positive(gains[i]), 'constant': _positive(dynamic[i]),
                    'time_shift': _positive(correct_time[i]),
                    'qualified_random_component': event_random or background_random,
                    'baseline_training': _stage_qualified(card, 'q') and _stage_qualified(card, 'bmark'),
                    'state_training': _stage_qualified(card, relevant_stage),
                    'physical_windows': support[i] is not None and int(support[i]) >= 3,
                }
                if family == 'dual':
                    checks['state_training'] = checks['state_training'] or _stage_qualified(card, 'background_state')
                    checks['current_background_training'] = _stage_qualified(card, 'background_current')
                    checks['background_state_training'] = _stage_qualified(card, 'background_state')
                joint_rows.append({'seed': card['seed'], 'checks': checks})
            joint = _joint(joint_rows)
            learned_joint = _joint([{'seed': card['seed'], 'checkpoint_sha256': card.get('_state_lineage', {}).get('checkpoint_sha256'), 'checks': {
                **joint_rows[i]['checks'], 'selected_event_observer_updated': _event_updated(card),
                'checkpoint_replay': card.get('_replay_evidence', {}).get('full_endpoint_replay_qualified') is True,
            }} for i, card in enumerate(subject_cards)])
            event_rows = []
            for i, card in enumerate(subject_cards):
                checks = dict(learned_joint['per_seed'][i]['checks'])
                if family == 'dual':
                    audit = card.get('_branch_control_evidence', {})
                    branch = audit.get('by_horizon', {}).get(key, {}).get('event', {})
                    checks.update(event_stage_training=_stage_qualified(card, 'event'),
                                  matched_event_random=_positive(random_gain[i]) and _stage_qualified(card, 'random'),
                                  branch_score_replay=audit.get('score_parity', {}).get('all_endpoint_total_auditable') is True,
                                  event_branch_constant=_positive(branch.get('dynamic_over_branch_constant')),
                                  event_branch_time_shift=_positive(branch.get('correct_time_over_branch_shift')),
                                  event_increment_over_background=_positive(branch.get('increment_over_persistent_background')))
                event_rows.append({'seed': card['seed'], 'checkpoint_sha256': card.get('_state_lineage', {}).get('checkpoint_sha256'), 'checks': checks})
            event_joint = _joint(event_rows)
            independent = int(round(_median(support) or 0.0))
            rows.append({
                "subject": subject, "family": family, "horizon_seconds": seconds,
                "n_seeds": len(subject_cards), "independent_selection_windows": independent,
                "evidence": evidence, "endpoint_evidence": endpoint_evidence,
                "training": training,
                'joint_evidence': joint,
                'composite_with_learned_event_input_joint_evidence': learned_joint,
                'learned_event_joint_evidence': event_joint,
                'learned_event_dynamic_state_candidate': event_joint['stable_positive_3_of_5'],
                'branch_control_evidence': {str(card['seed']): {
                    **card.get('_branch_control_evidence', {}).get('by_horizon', {}).get(key, {}),
                    'score_parity': card.get('_branch_control_evidence', {}).get('score_parity'),
                    'audit_path': card.get('_branch_control_evidence', {}).get('audit_path'),
                    'audit_sha256': card.get('_branch_control_evidence', {}).get('audit_sha256'),
                } for card in subject_cards} if family == 'dual' else {},
                'selected_state_classes': {str(card['seed']): card.get('_state_lineage', {}).get('selected_state_class', 'UNVERIFIED') for card in subject_cards},
                'verified_preprocessing_replay': {str(card['seed']): {
                    'available': '_verified_preprocessing_replay' in card,
                    'strict_feature_and_endpoint_replay': card.get('_verified_preprocessing_replay', {}).get('full_endpoint_replay_qualified'),
                    'endpoint_score_replay': card.get('_verified_preprocessing_replay', {}).get('score_parity', {}).get('rebuilt_state', {}).get('all_endpoint_total_auditable'),
                    'feature_max_abs_differences': card.get('_verified_preprocessing_replay', {}).get('preprocessing_feature_max_abs_difference', {}),
                } for card in subject_cards},
                "directional_dynamic_state_candidate": joint['stable_positive_3_of_5'],
                "candidate_rule": (
                    "same seed must pass baseline, constant, time shift, trained random component, "
                    "all load-bearing optimization paths and >=3 non-overlapping windows; "
                    "then require >=3 registered seeds. This does not prove nonlinear state learning."
                ),
            })
    return {"rows": rows}


def _credit(cards: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        grouped[str(card["subject"])].append(card)
    for subject, group in sorted(grouped.items()):
        endpoint_rows = {}
        for endpoint in ENDPOINTS:
            values_2h = [card["endpoint_credit"][endpoint]["fraction_beyond_2h"] if int(card.get('audited_anchors', 0)) > 0 else None for card in group]
            values_6h = [card["endpoint_credit"][endpoint]["fraction_beyond_6h"] if int(card.get('audited_anchors', 0)) > 0 else None for card in group]
            endpoint_rows[endpoint] = {
                "fraction_beyond_2h": _direction(values_2h),
                "fraction_beyond_6h": _direction(values_6h),
                "measurable_beyond_6h_seeds": int(sum(
                    float(value) >= LONG_CREDIT_FRACTION_THRESHOLD for value in _finite(values_6h)
                )),
                "fraction_threshold": LONG_CREDIT_FRACTION_THRESHOLD,
            }
        training = [card["selected_state_training"] for card in group]
        selected = sum(not bool(row.get("selected_at_init", False)) for row in training)
        learned_endpoints = [
            endpoint for endpoint, row in endpoint_rows.items()
            if row["measurable_beyond_6h_seeds"] >= 3
        ]
        first_gradient = sum(
            float(row.get("first_step_gradient_norm", 0.0) or 0.0) > 0.0 for row in training
        )
        moved = sum(
            float(row.get("peak_parameter_delta_from_stage_start", 0.0) or 0.0) > 0.0
            for row in training
        )
        audited = _median(card.get("audited_anchors") for card in group)
        joint = _joint([{'seed': _seed(card), 'checks': {
            'selected_after_init': card['selected_state_training'].get('selected_at_init') is False,
            'gradient': _positive(card['selected_state_training'].get('first_step_gradient_norm')),
            'parameter_movement': _positive(card['selected_state_training'].get('peak_parameter_delta_from_stage_start')),
            'within_budget': card['selected_state_training'].get('training_budget_exhausted') is False,
            'audited_anchors': int(card.get('audited_anchors', 0)) >= 3,
            'two_endpoints': sum(float(card['endpoint_credit'][name]['fraction_beyond_6h'])
                                 >= LONG_CREDIT_FRACTION_THRESHOLD for name in ENDPOINTS) >= 2,
        }} for card in group])
        rows.append({
            "subject": subject, "n_seeds": len(group),
            'family': 'event', 'prediction_horizons_hours': sorted({card['horizon_hours'] for card in group}),
            'joint_evidence': joint,
            "audited_anchors_median": audited,
            "selected_after_init_seeds": int(selected),
            "first_step_gradient_positive_seeds": int(first_gradient),
            "parameter_moved_seeds": int(moved),
            "budget_exhausted_seeds": int(sum(bool(row.get("training_budget_exhausted", False)) for row in training)),
            "endpoint_credit": endpoint_rows,
            "endpoints_with_measurable_beyond_6h_in_3_of_5": learned_endpoints,
            "trained_multi_hour_credit_candidate": joint['stable_positive_3_of_5'],
            "interpretation": (
                "observer credit from real human endpoint losses; >=1% of endpoint gradient "
                "beyond 6 h is the predeclared measurable threshold; not physiological IED feedback"
            ),
        })
    return {"rows": rows}


def _h2a(cards: list[dict[str, Any]], family: str) -> dict[str, Any]:
    rows = []
    for subject, group in sorted(_group(cards).items()):
        keys = sorted(set().union(*(card.get("primary_contrasts", {}) for card in group)))
        contrasts = {key: _direction(card.get("primary_contrasts", {}).get(key) for card in group)
                     for key in keys}
        state_stage = [card.get("stages", {}).get("state", {}) for card in group]
        path_live = sum(float(row.get("peak_modulation_magnitude_during_training", 0.0) or 0.0) > 0.0
                        for row in state_stage)
        oracle = contrasts.get("oracle_sensitivity_grammar_gain", _direction([]))
        endpoint_keys = {
            "contact_subset": ("state_gain_over_B_mark_contact", "constant_unexplained_contact", "correct_time_paired_contact"),
            "continue_stop": ("state_gain_over_B_mark_stop", "constant_unexplained_stop", "correct_time_paired_stop"),
            "grammar": ("state_gain_over_B_mark_grammar", "constant_unexplained_grammar", "correct_time_paired_grammar"),
            "same_prefix_continuation": ("same_prefix_gain_over_B_mark", "same_prefix_constant_unexplained_grammar", "correct_time_same_prefix_grammar"),
            "conditional_rich_mark": ("conditional_rich_mark_gain_over_B_mark", "constant_unexplained_rich_mark", "correct_time_rich_mark"),
        }
        endpoint_joint = {name: _joint_contrasts(group, names) for name, names in endpoint_keys.items()}
        # Legacy whole-event scores are ineligible for a suffix claim, even if
        # somebody later adds a similarly named constant contrast to the card.
        suffix_rows = endpoint_joint['same_prefix_continuation']['per_seed']
        for card, row in zip(group, suffix_rows):
            row['checks']['strict_suffix_target'] = card.get('same_prefix_scoring_contract') == 'after_two_observed_groups_v1'
        endpoint_joint['same_prefix_continuation'] = _joint(suffix_rows)
        endpoint_candidates = {name: value['stable_positive_3_of_5'] for name, value in endpoint_joint.items()}
        non_synonymous = (
            "contact_subset", "continue_stop", "same_prefix_continuation",
            "conditional_rich_mark",
        )
        assay_sensitive = path_live >= 3 and oracle["stable_positive_3_of_5"]
        multi_joint = _joint([{'seed': _seed(card), 'checks': {
            'two_non_synonymous_endpoints': sum(_seed(card) in endpoint_joint[name]['passing_seeds'] for name in non_synonymous) >= 2,
            'adapter_path_live': _positive(state_stage[i].get('peak_modulation_magnitude_during_training')),
            'oracle_live': _positive(card.get('primary_contrasts', {}).get('oracle_sensitivity_grammar_gain')),
            'state_budget_qualified': _adapter_qualified(card, 'state'),
            'B_mark_budget_qualified': _adapter_qualified(card, 'B_mark'),
            'static_budget_qualified': _adapter_qualified(card, 'static'),
        }} for i, card in enumerate(group)])
        any_pair_diagnostic = multi_joint
        fixed_pairs = _fixed_endpoint_pair_evidence(
            [{'seed': row['seed'], 'checks': {k: v for k, v in row['checks'].items() if k != 'two_non_synonymous_endpoints'}}
             for row in multi_joint['per_seed']],
            {name: set(endpoint_joint[name]['passing_seeds']) for name in non_synonymous})
        multi_joint = fixed_pairs['joint_evidence']
        learned_pairs = _fixed_endpoint_pair_evidence(
            [{'seed': _seed(card), 'checks': {**{k: v for k, v in any_pair_diagnostic['per_seed'][i]['checks'].items() if k != 'two_non_synonymous_endpoints'},
                                             'selected_event_observer_updated': _event_updated(card)}} for i, card in enumerate(group)],
            {name: set(endpoint_joint[name]['passing_seeds']) for name in non_synonymous})
        learned_joint = learned_pairs['joint_evidence']
        rows.append({
            "subject": subject, "family": family, "n_seeds": len(group),
            "n_events": group[0].get("n_events", {}),
            "n_same_prefix_median": _median(card.get("n_same_prefix") for card in group),
            "n_shift_paired_median": _median(card.get("n_shift_paired") for card in group),
            "contrasts": contrasts, "endpoint_candidates": endpoint_candidates,
            "endpoint_joint_evidence": endpoint_joint, "multi_endpoint_joint_evidence": multi_joint,
            'any_two_endpoints_per_seed_diagnostic': any_pair_diagnostic,
            'fixed_endpoint_pair_evidence': fixed_pairs,
            'learned_upstream_multi_endpoint_joint_evidence': learned_joint,
            'learned_upstream_fixed_endpoint_pairs': learned_pairs,
            'learned_upstream_multi_endpoint_candidate': learned_joint['stable_positive_3_of_5'],
            'selected_upstream_state_classes': {str(_seed(card)): card.get('_state_lineage', {}).get('selected_state_class', 'UNVERIFIED') for card in group},
            "state_adapter_path_live_seeds": int(path_live),
            "oracle_sensitive_seeds": oracle["positive_seeds"],
            "assay_sensitive": assay_sensitive,
            "multiple_h2a_endpoints_directionally_supported": bool(
                multi_joint['stable_positive_3_of_5']
            ),
            "multi_endpoint_rule": (
                "same fixed pair of non-synonymous endpoints in the same >=3 seeds, with B_mark, FIT-constant, "
                "wrong-time, adapter and oracle checks; suffix requires corrected scoring contract"
            ),
        })
    return {"rows": rows}


def _dual_credit(cards, lineage):
    by_hash = {row['checkpoint_sha256']: row for row in lineage.get('h1_cards', [])}
    groups = defaultdict(list)
    for card in cards:
        if card.get('family') == 'dual' and 'horizon_hours' in card:
            groups[(card['subject'], float(card['horizon_hours']))].append(card)
    rows = []
    old_bins = ('6-8h', '8-16h', '16-32h', '>=32h')
    for (subject, horizon), group in sorted(groups.items()):
        for branch in ('event', 'background'):
            seed_rows = []; endpoint_seeds = defaultdict(list); deletions = defaultdict(list)
            for card in group:
                source = by_hash.get(card['source_hashes']['checkpoint.pt'])
                if source is None or source['subject'] != subject or source['seed'] != card['seed'] or source['family'] != 'dual':
                    raise ValueError('dual credit checkpoint binding mismatch')
                fractions = {}; auditable = set(card.get('score_parity', {}).get('auditable_endpoints', []))
                for endpoint in ENDPOINTS:
                    mass = defaultdict(float)
                    for anchor in card.get('anchors', []):
                        for age, cell in anchor['credit'].get(endpoint, {}).get(branch, {}).items():
                            mass[age] += float(cell['gradient_mass'])
                        for age in old_bins:
                            cell = anchor.get('deletions', {}).get(age, {}).get(branch, {})
                            if cell.get('removed_observations', 0) > 0 and endpoint in auditable:
                                value = cell.get('loss_increase', {}).get(endpoint)
                                if value is not None:
                                    deletions[(endpoint, age, card['seed'])].append(float(value))
                    total = sum(mass.values())
                    fractions[endpoint] = sum(mass[age] for age in old_bins) / total if total > 0 and endpoint in auditable else None
                    endpoint_seeds[endpoint].append(fractions[endpoint])
                stage = card.get('source_stages', {}).get('event' if branch == 'event' else 'background_state', {})
                checks = {'estimated': card.get('status') == 'COMPLETE',
                          'selected_observer_updated': source['modules'][branch]['selected_changed'],
                          'within_training_budget': stage.get('training_budget_exhausted') is False,
                          'state_replay': max(card.get('state_parity_max_abs', {'missing': float('inf')}).values()) < 1e-4,
                          'three_audited_anchors': int(card.get('audited_anchors', 0)) >= 3,
                          'two_endpoints_with_long_gradient': sum(v is not None and v >= LONG_CREDIT_FRACTION_THRESHOLD for v in fractions.values()) >= 2}
                seed_rows.append({'seed': card['seed'], 'checks': checks, 'endpoint_fraction_beyond_6h': fractions,
                                  'checkpoint_sha256': source['checkpoint_sha256'],
                                  'audited_anchors': card.get('audited_anchors', 0)})
            joint = _joint(seed_rows)
            rows.append({'subject': subject, 'family': 'dual', 'branch': branch, 'horizon_hours': horizon,
                         'joint_evidence': joint, 'trained_long_credit_candidate': joint['stable_positive_3_of_5'],
                         'endpoint_fraction_beyond_6h': {k: _direction(v) for k, v in endpoint_seeds.items()},
                         'history_deletion_mean_loss_increase_per_seed': [
                             {'endpoint': key[0], 'age_bin': key[1], 'seed': key[2], 'mean': float(np.mean(values)), 'n_anchors': len(values)}
                             for key, values in sorted(deletions.items())],
                         'interpretation': 'gradient reachability and branch-only deletion sensitivity; other inputs frozen; not physiological feedback; anchors may overlap'})
    return {'rows': rows, 'registered_units': len(cards),
            'not_estimable_units': sum(card.get('status') == 'NOT_ESTIMABLE' for card in cards)}


def _status_from_support(card: dict[str, Any]) -> tuple[str, dict[str, int]]:
    support = card.get("distance_survival", {}).get("support", {})
    counts = support.get('observed_seizures_by_phase', support.get("seizures_by_phase", {}))
    if support.get("status") != "ESTIMATED":
        return "NOT_ESTIMABLE", counts
    if int(counts.get("SELECTION", 0)) >= 3:
        return "REPEATED_HELD_OUT_SEIZURES", counts
    return "DESCRIPTIVE_ONLY", counts


def _hazard_qualification(card, family):
    required = {
        'event': ('B_history', 'B_history_plus_S_event', 'B_history_plus_random_history'),
        'grid': ('B_history', 'B_history_plus_S_grid', 'B_history_plus_random_history'),
        'dual': ('B_context', 'B_context_plus_S_dual', 'B_context_plus_random'),
        'background': ('B_history_current_background', 'B_history_current_background_plus_S_background',
                       'B_history_current_background_plus_random_background'),
    }[family] + ('B_history_plus_future_oracle',)
    fits = card.get('distance_survival', {})
    controls = card.get('hazard_instrument_controls', {})
    prefix = 'event_source' if family == 'event' else 'grid_source' if family == 'grid' else 'source'
    state_class = card.get('_h2b_lineage', {}).get('bindings', {}).get(prefix, {}).get('selected_state_class')
    updated = state_class in ({'learned_event_and_background', 'learned_background_only'} if family == 'background'
                              else {'learned_event_and_background'} if family == 'dual' else {'learned_event'})
    cluster = card.get('episode_support_audit', {}).get('cluster_gap_hours_sensitivity', {}).get('6', {}).get('time_shift', {})
    checks = {'readout_stationarity': all(fits.get(name, {}).get('training', {}).get('passes_stationarity') is True for name in required),
              'fitted_readouts_saved': all('fitted_readout' in fits.get(name, {}) for name in required),
              'oracle_sensitive': _positive(controls.get('future_oracle_gain_over_history')),
              'selected_upstream_updated': updated,
              'six_hour_cluster_sensitivity': int(cluster.get('phase_contained_clusters', 0)) >= 3}
    if family in ('event', 'grid'):
        checks['matched_history_capacity'] = _positive(controls.get(f'{family}_gain_over_matched_history_capacity'))
    return checks


def _h2b(cards: list[dict[str, Any]], *, enforce_evidence: bool = False) -> dict[str, Any]:
    rows = []
    any_candidates = []
    for subject, group in sorted(_group(cards).items()):
        statuses = [_status_from_support(card) for card in group]
        status = statuses[0][0] if statuses else "NOT_ESTIMABLE"
        counts = statuses[0][1] if statuses else {}
        keys = sorted(set().union(*(card.get("primary_contrasts", {}) for card in group)))
        contrasts = {key: _direction(card.get("primary_contrasts", {}).get(key) for card in group)
                     for key in keys}
        risk_keys = {
            "event": (
                "event_only_state_gain_over_mark_history", "event_correct_time_gain_over_shift",
                "event_dynamic_gain_over_fit_period_mean",
            ),
            "grid": (
                "grid_state_gain_over_mark_history", "grid_correct_time_gain_over_shift",
                "grid_dynamic_gain_over_fit_period_mean",
            ),
            "dual": (
                "state_gain_over_background_censored_logscore", "state_gain_over_random_capacity_control",
                "dual_correct_time_gain_over_shift", "dual_dynamic_gain_over_fit_period_mean",
            ),
            "background": (
                "background_state_gain_over_current_background_censored_logscore",
                "background_state_gain_over_random_background",
                "background_correct_time_gain_over_shift",
                "background_dynamic_gain_over_fit_period_mean",
            ),
        }
        risk_joint = {name: _joint_contrasts(group, names) for name, names in risk_keys.items()}
        for name in risk_joint:
            per_seed = risk_joint[name]['per_seed']
            for card, row in zip(group, per_seed):
                row['checks']['repeated_held_out_seizures'] = _status_from_support(card)[0] == 'REPEATED_HELD_OUT_SEIZURES'
            risk_joint[name] = _joint(per_seed)
        qualified_joint = {name: _joint([{'seed': card['seed'], 'checks': {
            **risk_joint[name]['per_seed'][i]['checks'], **_hazard_qualification(card, name),
        }} for i, card in enumerate(group)]) for name in risk_joint}
        risk_candidates = {name: row['stable_positive_3_of_5'] for name, row in
                           (qualified_joint if enforce_evidence else risk_joint).items()}
        field: dict[str, Any] = {}
        for card in group:
            for lead, targets in card.get("early_ictal_field_and_path", {}).items():
                for target, result in targets.items():
                    cell = field.setdefault(lead, {}).setdefault(target, defaultdict(list))
                    cell["status"].append(result.get("status"))
                    support = result.get("support", {})
                    cell["n_selection_seizures"].append(support.get("n_selection_seizures"))
                    for key, value in result.items():
                        if key.endswith("gain_over_background") or key.endswith("gain_over_mark_history") \
                                or key.endswith("gain_over_shift") \
                                or key.endswith("gain_over_fit_period_mean") \
                                or key.endswith("gain_over_random_capacity_control"):
                            cell[key].append(value)
        field_summary = {}
        field_candidate = False
        for lead, targets in field.items():
            field_summary[lead] = {}
            for target, values in targets.items():
                statuses = values.pop("status", [])
                selection_counts = values.pop("n_selection_seizures", [])
                result = {
                    "estimated_seeds": int(sum(value == "ESTIMATED" for value in statuses)),
                    "selection_seizures_median": _median(selection_counts),
                    "contrasts": {key: _direction(items) for key, items in values.items()},
                }
                families = {
                    "event": (
                        "event_only_state_gain_over_mark_history",
                        "event_correct_time_gain_over_shift",
                        "event_dynamic_gain_over_fit_period_mean",
                    ),
                    "grid": (
                        "grid_state_gain_over_mark_history",
                        "grid_correct_time_gain_over_shift",
                        "grid_dynamic_gain_over_fit_period_mean",
                    ),
                    "dual": (
                        "state_gain_over_background",
                        "state_gain_over_random_capacity_control",
                        "dual_correct_time_gain_over_shift",
                        "dual_dynamic_gain_over_fit_period_mean",
                    ),
                    "background": (
                        "background_state_gain_over_current_background",
                        "background_state_gain_over_random_background",
                        "background_correct_time_gain_over_shift",
                        "background_dynamic_gain_over_fit_period_mean",
                    ),
                }
                result['family_joint_evidence'] = {}
                for family, family_keys in families.items():
                    seed_rows = []
                    for card in group:
                        cell = card.get('early_ictal_field_and_path', {}).get(lead, {}).get(target, {})
                        checks = {key: _positive(cell.get(key)) for key in family_keys}
                        checks['estimable'] = cell.get('status') == 'ESTIMATED'
                        checks['repeated_seizures'] = int(cell.get('support', {}).get('n_selection_seizures', 0) or 0) >= 3
                        if enforce_evidence:
                            checks['common_trained_contacts'] = bool(cell.get('support', {}).get('common_fitted_columns'))
                            checks['field_oracle_sensitive'] = _positive(cell.get('future_field_oracle', {}).get('gain_over_history'))
                            checks['selected_upstream_updated'] = _hazard_qualification(card, family)['selected_upstream_updated']
                        seed_rows.append({'seed': card['seed'], 'checks': checks})
                    result['family_joint_evidence'][family] = _joint(seed_rows)
                result['family_candidates'] = {family: cell['stable_positive_3_of_5']
                                               for family, cell in result['family_joint_evidence'].items()}
                result["any_complete_dynamic_field_candidate"] = any(
                    result["family_candidates"].values()
                )
                field_candidate = field_candidate or result["any_complete_dynamic_field_candidate"]
                field_summary[lead][target] = result
        repeatable = status == "REPEATED_HELD_OUT_SEIZURES"
        any_candidate = repeatable and (any(risk_candidates.values()) or field_candidate)
        any_candidates.append(any_candidate)
        rows.append({
            "subject": subject, "n_seeds": len(group), "estimability": status,
            "seizures_by_phase": counts, "risk_contrasts": contrasts,
            'clinical_seizures_by_phase': group[0].get('distance_survival', {}).get('support', {}).get('seizures_by_phase', {}),
            'scored_seizures_by_phase': group[0].get('distance_survival', {}).get('support', {}).get('observed_seizures_by_phase'),
            "risk_candidates": risk_candidates, "risk_joint_evidence": risk_joint, "early_field_and_path": field_summary,
            'qualified_risk_joint_evidence': qualified_joint,
            'episode_support_audit': group[0].get('episode_support_audit', {}),
            'hazard_instrument_controls': {str(card['seed']): card.get('hazard_instrument_controls', {}) for card in group},
            'qualification_boundary': 'six-hour episode grouping is a post-review robustness diagnostic, not independently preregistered confirmation',
            "any_repeatable_cross_task_candidate": any_candidate,
        })
    return {"rows": rows, "any_repeatable_cross_task_candidate": bool(any(any_candidates))}


def _queue(path: Path) -> dict[str, Any]:
    return _read(path) if path.exists() else {"status": "PENDING", "path": str(path)}


def _markdown(summary: dict[str, Any], technical: bool) -> str:
    from scripts.group_event_state_v038_review_report import render
    return render(summary, technical)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=BASE)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument('--final-reviewed', action='store_true',
                        help='Use only after code, tests, report and figure review; checks all registered repair queues.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/data/hfosp_group_event_state_v0_3_8_review_repair/final_reports'))
    parser.add_argument('--repair-root', type=Path,
                        default=Path('/data/hfosp_group_event_state_v0_3_8_review_repair'))
    args = parser.parse_args()
    base = args.data_root
    lineage_path = args.repair_root / 'state_lineage.json'
    lineage = _read(lineage_path) if lineage_path.exists() else {}
    inventory_path = args.repair_root / 'model_inventory.json'
    inventory = _read(inventory_path) if inventory_path.exists() else {}
    evidence_cards = []
    def load_cards(path, repair=None):
        cards = _attach_lineage(_cards(path, repair), lineage, args.repair_root)
        for card in cards:
            card['_audit_scope'] = path.parent.name
            card['_audit_family'] = path.name
        evidence_cards.extend(cards)
        return cards
    queue_paths = {
        "master": base / "supervisor/queue_status.json",
        "h1_long": base / "h1_long_8h/supervisor/queue_status.json",
        "h1_medium": base / "h1_medium_2h/supervisor/queue_status.json",
        "h2a_long": base / "h2a_long_8h/supervisor/queue_status.json",
        "h2a_medium": base / "h2a_medium_2h/supervisor/queue_status.json",
        "h2b_long": base / "h2b_long_8h/supervisor/queue_status.json",
        "h2b_medium": base / "h2b_medium_2h/supervisor/queue_status.json",
        "trained_credit": base / "trained_credit_long/queue_status.json",
        "random_background_long": base / "dual_random_background_long_v2/queue_status.json",
        "random_background_medium": base / "dual_random_background_medium_v2/queue_status.json",
    }
    queues = {name: _queue(path) for name, path in queue_paths.items()}
    complete = queues["master"].get("status") == "COMPLETE"
    if not complete and not args.allow_incomplete:
        raise RuntimeError("v0.3.8 master queue is not complete")
    random_long = _jsons(base / "dual_random_background_long_v2")
    random_medium = _jsons(base / "dual_random_background_medium_v2")
    h1 = {}
    for cohort, h1_root, random_cards in (
        ("long_0.5_2_6_8h", base / "h1_long_8h", random_long),
        ("medium_0.5_2h", base / "h1_medium_2h", random_medium),
    ):
        random_index = _random_background_index(random_cards)
        h1[cohort] = {
            family: _h1_family(load_cards(h1_root / family, [args.repair_root / kind / h1_root.name / family
                                                           for kind in ('h1_random_budget', 'h1_remaining_controls')]), family, random_index)
            for family in FAMILIES
        }
    h2a = {
        "long_0.5_2_6_8h": {
            family: _h2a(load_cards(base / "h2a_long_8h" / family, [args.repair_root / kind / 'h2a_long_8h' / family for kind in ('h2a_suffix', 'h2a_budget', 'h2a_budget_complete')]), family) for family in FAMILIES
        },
        "medium_0.5_2h": {
            family: _h2a(load_cards(base / "h2a_medium_2h" / family, [args.repair_root / kind / 'h2a_medium_2h' / family for kind in ('h2a_suffix', 'h2a_budget', 'h2a_budget_medium')]), family) for family in FAMILIES
        },
    }
    h2b_cards = [card for scope in ('h2b_long_8h', 'h2b_medium_2h') for card in
                 load_cards(base / scope / 'outcomes', [args.repair_root / kind / scope / 'outcomes'
                                                       for kind in ('h2b_instrument', 'h2b_instrument_polished')])]
    output = args.output_dir
    if output.resolve() == (base / 'final_reports').resolve():
        raise ValueError('repair finalizer must preserve the original final reports')
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / ("summary_main.json" if complete else "summary_incremental.json")
    summary = {
        "format": "group_event_state_v0_3_8_core_scientific_closure_summary_v1",
        "status": "REVIEW_REPAIR_PARTIAL",
        "original_registered_queues_complete": complete,
        "entire_repair_goal_complete": False,
        "queues": queues,
        'repair_queues': {name: _queue(args.repair_root / name / 'queue_status.json')
                          for name in ('dual_credit_batch', 'dual_credit_remaining_batch', 'h2a_suffix_batch', 'random_budget_batch',
                                       'h2a_budget_batch', 'h2b_instrument_batch', 'h2b_instrument_polished_batch', 'checkpoint_replay_batch',
                                       'verified_preprocessing_batch', 'h2a_budget_complete_batch', 'dual_branch_controls_batch',
                                       'h2a_budget_medium_batch', 'h1_remaining_controls_batch')},
        'state_lineage': {'path': str(lineage_path), 'sha256': hashlib.sha256(lineage_path.read_bytes()).hexdigest() if lineage else None,
                          'state_classes': lineage.get('state_classes', {}), 'n_h1_cards': len(lineage.get('h1_cards', [])),
                          'n_downstream_bindings': len(lineage.get('downstream_bindings', []))},
        'model_inventory': {'path': str(inventory_path), 'sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest() if inventory else None,
                            'h2a_adapter_count': inventory.get('h2a_adapter_count'), 'frozen_decoder_count': inventory.get('frozen_decoder_count')},
        'source_card_manifest': [{k: card.get(k) for k in ('_loaded_source_card', '_loaded_source_sha256', '_loaded_overlay_card', '_loaded_overlay_sha256')}
                                 for card in evidence_cards],
        "cohorts": {
            "long": list(LONG), "long_h2a": list(LONG_H2A), "medium": list(MEDIUM),
            "seeds_are_optimization_repeats_not_independent_patients": list(SEEDS),
        },
        "h1": h1,
        "trained_long_credit": _credit(_jsons(base / "trained_credit_long")),
        'dual_checkpoint_bound_credit': _dual_credit(_jsons(args.repair_root / 'dual_credit'), lineage),
        "h2a": h2a,
        "h2b": _h2b(h2b_cards, enforce_evidence=True),
        "scientific_questions": {
            "dynamic_state": "strong baseline + constant + time shift + matching random component",
            "learned_long_credit": "selected human checkpoint + real held-out endpoint gradient",
            "multiple_pathology_endpoints": list(ENDPOINT_LABELS.values()),
            "seizure_transfer": "frozen interictal feature then risk/distance/early field/path",
        },
        "claim_boundaries": {
            "observer_update_is_not_H3": True,
            "fixed_tau_is_not_human_timescale_discovery": True,
            "not_estimable_is_not_scientific_null": True,
            "patient_is_statistical_unit": True,
            "state_selected_without_seizure_outcomes": True,
            "sealed_partition_opened": False,
        },
        "output_paths": {
            "summary_json": str(summary_path),
            "plain_report": str(output / "group_event_state_v0_3_8_core_closeout_plain.md"),
            "technical_report": str(output / "group_event_state_v0_3_8_core_closeout_technical.md"),
        },
    }
    summary['same_model_h1_credit_multitarget'] = _h1_credit_multitarget_chain(h1, summary['dual_checkpoint_bound_credit'])
    hazard_fits = [fit for card in h2b_cards for fit in card.get('distance_survival', {}).values()
                   if isinstance(fit, dict) and 'training' in fit]
    replay_diagnostics = {}
    for name in ('checkpoint_replay', 'checkpoint_replay_verified'):
        replay_cards = _cards(args.repair_root / name)
        replay_diagnostics[name] = {
            'completed_models': len(replay_cards),
            'strict_feature_and_endpoint_pass': sum(card.get('full_endpoint_replay_qualified') is True for card in replay_cards),
            'rebuilt_endpoint_score_pass': sum(card.get('score_parity', {}).get('rebuilt_state', {}).get('all_endpoint_total_auditable') is True for card in replay_cards),
            'per_model': [{k: card.get(k) for k in ('subject', 'seed', 'scope', 'family', 'full_endpoint_replay_qualified',
                                                   'selected_state_max_abs_difference', 'preprocessing_feature_max_abs_difference',
                                                   'score_parity', 'replay_bundle_sha256', '_loaded_source_sha256')} for card in replay_cards],
        }
    h2a_stage_audit = []
    h1_stage_audit = []
    for card in evidence_cards:
        if card.get('_loaded_source_card') == card.get('_state_lineage', {}).get('source_card'):
            for name, stage in card.get('stages', {}).items():
                # The rate stage is the innermost nested control.  Selecting it
                # at the origin removes this stage's increment only. Later
                # history/background stages may still learn; inspect them separately.
                h1_stage_audit.append({'subject': card['subject'], 'seed': _seed(card),
                                       'family': card['_state_lineage']['family'], 'stage': name,
                                       'qualified': _stage_qualified(card, name),
                                       'foundation_stage_at_origin': bool(
                                           name == 'q' and stage.get('selected_step') == 0),
                                       'training_budget_exhausted': stage.get('training_budget_exhausted'),
                                       'steps_run': stage.get('steps_run'), 'selected_step': stage.get('selected_step'),
                                       'repair_unavailable': card.get('repair_unavailable')})
        if 'state_provenance' not in card or not card.get('stages'): continue
        for name in ('static', 'state', 'B_mark'):
            stage = card['stages'].get(name, {})
            h2a_stage_audit.append({'subject': card['subject'], 'seed': _seed(card), 'family': card['state_provenance']['family'],
                                    'stage': name, 'qualified': _adapter_qualified(card, name),
                                    'epochs_run': stage.get('epochs_run'), 'selected_epoch': stage.get('selected_epoch')})
    summary['repair_diagnostics'] = {'estimated_hazard_readouts': len(hazard_fits),
                                     'stationary_hazard_readouts': sum(fit['training']['passes_stationarity'] for fit in hazard_fits),
                                     'h1_stage_qualification': h1_stage_audit,
                                     'h2a_stage_qualification': h2a_stage_audit,
                                     'branch_controls': {'n_cards': len(_cards(args.repair_root / 'dual_branch_controls')),
                                                         'score_replay_qualified': sum(card.get('score_parity', {}).get('all_endpoint_total_auditable') is True
                                                                                       for card in _cards(args.repair_root / 'dual_branch_controls'))},
                                     'preprocessing_replay': replay_diagnostics,
                                     'code_provenance': _code_provenance_audit(evidence_cards),
                                     'foundation_stage_at_origin': sorted({
                                         (row['subject'], row['family'])
                                         for row in h1_stage_audit
                                         if row.get('foundation_stage_at_origin')
                                     })}
    from scripts.finalize_group_event_state_v037 import _wrong_time_control_quality
    h1_cards = [card for card in evidence_cards if card.get('_loaded_source_card') == card.get('_state_lineage', {}).get('source_card')]
    summary['repair_diagnostics']['wrong_time_control_quality'] = {
        scope: _wrong_time_control_quality([card for card in h1_cards if card['_audit_scope'] == scope])
        for scope in sorted({card['_audit_scope'] for card in h1_cards})}
    summary['repair_diagnostics']['held_out_units'] = {
        'rule': 'Seeds are optimisation repeats; physical non-overlapping windows are reported separately by horizon.',
        'by_card': [{key: card.get(key) for key in ('subject', 'seed', '_audit_scope', '_audit_family',
                     'independent_windows_by_horizon', 'selection_window_audit')} for card in h1_cards]}
    summary['repair_diagnostics']['foundation_fallback_context'] = [
        {'subject': card['subject'], 'seed': card['seed'], 'family': card['_audit_family'],
         'selected_steps': {key: value.get('selected_step') for key, value in card['stages'].items()},
         'rule': 'Step zero is a selected fallback, not proof of failed optimisation or collapse of later nested stages.'}
        for card in h1_cards if card.get('stages', {}).get('q', {}).get('selected_step') == 0]
    if args.final_reviewed:
        if any(queue.get('status') != 'COMPLETE' for queue in summary['repair_queues'].values()):
            raise RuntimeError('final review cannot close unfinished or failed repair queues')
        if len(lineage.get('h1_cards', [])) != 165 or replay_diagnostics['checkpoint_replay_verified']['completed_models'] != 165:
            raise RuntimeError('final review requires all registered checkpoint lineage and preprocessing audits')
        if summary['repair_diagnostics']['branch_controls']['n_cards'] != 55:
            raise RuntimeError('final review requires all 55 registered branch controls')
        if inventory.get('h2a_adapter_count') != 120:
            raise RuntimeError('final review requires the 120 estimable adapter parameter inventories')
        summary['status'] = 'REVIEW_REPAIR_COMPLETE'
        summary['entire_repair_goal_complete'] = True
    h1_candidate_subjects = sorted({
        row["subject"] for cohort in summary["h1"].values()
        for family in cohort.values() for row in family["rows"]
        if row["directional_dynamic_state_candidate"]
    })
    six_hour_dynamic_subjects = sorted({
        row["subject"] for cohort in summary["h1"].values()
        for family in cohort.values() for row in family["rows"]
        if row["directional_dynamic_state_candidate"] and row["horizon_seconds"] == 21600
    })
    credit_subjects = sorted(
        row["subject"] for row in summary["trained_long_credit"]["rows"]
        if row["trained_multi_hour_credit_candidate"]
    )
    h2a_subjects = sorted({
        row["subject"] for cohort in summary["h2a"].values()
        for family in cohort.values() for row in family["rows"]
        if row["multiple_h2a_endpoints_directionally_supported"]
    })
    seizure_subjects = sorted(
        row["subject"] for row in summary["h2b"]["rows"]
        if row["any_repeatable_cross_task_candidate"]
    )
    summary["scientific_closure"] = {
        "dynamic_state_candidate_subjects": h1_candidate_subjects,
        "six_hour_dynamic_state_subjects": six_hour_dynamic_subjects,
        "trained_long_credit_subjects": credit_subjects,
        "multi_pathology_h2a_subjects": h2a_subjects,
        "repeatable_seizure_transfer_subjects": seizure_subjects,
        "patient_name_overlap_only_not_checkpoint_chain": sorted(
            set(h1_candidate_subjects) & set(credit_subjects)
            & set(h2a_subjects) & set(seizure_subjects)
        ),
        "single_subject_full_chain": [],
        'same_model_h1_credit_multitarget_subjects': sorted({row['subject'] for row in summary['same_model_h1_credit_multitarget']['rows']
                                                          if row['same_model_functional_state_candidate']}),
        'event_branch_h1_credit_multitarget_subjects': sorted({row['subject'] for row in summary['same_model_h1_credit_multitarget']['rows']
                                                            if row['event_branch_functional_state_candidate']}),
        'learned_event_dynamic_state_subjects': sorted({row['subject'] for cohort in h1.values() for family in cohort.values()
                                                       for row in family['rows'] if row['learned_event_dynamic_state_candidate']}),
        "checkpoint_bound_full_chain_status": "NOT_ESTABLISHED",
        "interpretation": (
            "joint-seed exploratory candidates only; checkpoint-bound unified state is not established; "
            + ("seizure transfer requires further qualification" if seizure_subjects
               else "no complete repeatable seizure-transfer candidate under corrected contrasts")
        ),
    }
    atomic_json(summary_path, summary)
    if complete:
        (output / "group_event_state_v0_3_8_core_closeout_plain.md").write_text(
            _markdown(summary, False), encoding="utf-8"
        )
        (output / "group_event_state_v0_3_8_core_closeout_technical.md").write_text(
            _markdown(summary, True), encoding="utf-8"
        )
    print(json.dumps({
        "status": summary["status"], "summary": str(summary_path),
        "h1_cards": sum(len(_cards(root / family)) for root in (
            base / "h1_long_8h", base / "h1_medium_2h"
        ) for family in FAMILIES),
        "h2a_cards": sum(len(_cards(root / family)) for root in (
            base / "h2a_long_8h", base / "h2a_medium_2h"
        ) for family in FAMILIES),
        "h2b_cards": len(h2b_cards),
    }, indent=2))


if __name__ == "__main__":
    main()

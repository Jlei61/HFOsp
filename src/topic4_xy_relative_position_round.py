"""Promote the prespecified relative-position soft prior for round 1 only."""
import copy
import numpy as np


def relative_position_ranking(report, *, weight=0.1):
    """Keep the measured fit intact; add a separately named constrained score."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('prior weight must be finite and nonnegative')
    result = copy.deepcopy(report)
    feasible = []
    for row in result['candidates']:
        score = row['J_direction']; prior = row['core_alignment_penalty']
        if not np.isfinite(prior) or not 0 <= prior <= 1 + 1e-12:
            raise ValueError('invalid relative-position penalty')
        if score is not None and not np.isfinite(score):
            raise ValueError('nonfinite event/direction fit')
        row['J_round1'] = None if score is None else float(score + weight * prior)
        if row['selection_eligible'] and row['J_round1'] is not None:
            feasible.append(row)
    def ranking(key):
        return [r['candidate_id'] for r in sorted(feasible, key=lambda r:(r[key],r['candidate_id']))]
    result['ranking_without_position_prior'] = ranking('J_direction')
    result['ranking'] = ranking('J_round1')
    result.update(round=1, round_primary_score='J_round1',
                  primary_selection_uses_core_prior=True, primary_core_prior_weight=float(weight),
                  relative_position_constraint='soft axial alignment, no fixed midpoint or separation')
    return result


def primary_and_unconstrained_nominees(report):
    """Nominate before independent seeds; retain the no-prior counterfactual."""
    byid = {r['candidate_id']:r for r in report['candidates']}
    selected = []
    for domain in ('whole_sheet','interior'):
        for key in ('ranking','ranking_without_position_prior'):
            ids = [c for c in report[key] if byid[c]['domain'] == domain]
            selected.extend(ids[:1])
    return list(dict.fromkeys(selected))

#!/usr/bin/env python3
"""Read-only component coverage of the fully closed V3 pool."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_replicated_search as run
from src.topic4_xy_component_search import COMPONENTS, local_anchors, select_racers
from src.topic4_xy_replicated_anchors import anchor_eligible


def main():
    out = run.OUT; plan = run.read(run.CONFIG)
    closure = run.read(out/'cycle_closure_review.json')
    if closure['status'] != 'EXHAUSTED_UNQUALIFIED_CYCLE_REPLAY_VERIFIED':
        raise RuntimeError('unclosed source cycle')
    run.v1.runtime.verify_amendment(closure['source_hashes'])
    sources = [out/'baseline_scores.json']
    pool = run.read(sources[0])['candidates']
    for n in range(1, closure['rounds']+1):
        stage = out/'rounds'/f'{n:03d}'; path = stage/'scores.json'; sources.append(path)
        pool.extend(run.read(path)['candidates'])
        for cid in run.read(stage/'race_nomination.json')['candidate_ids']:
            path = stage/f'combined_{cid}.json'; sources.append(path)
            row = run.read(path)['candidates'][0]
            pool = [row if r['candidate_id'] == cid else r for r in pool]
    def compact(row):
        return {k: row[k] for k in ('candidate_id', 'n_events', 'joint_distance', 'exploration_score', 'kernel_distances')}
    full = [r for r in pool if anchor_eligible(r, plan)]
    short = [r for r in pool if r['explorable'] and len(r['units']) == 2
             and all(u['geometry']['minimum_clearance_mm'] >= 0 for u in r['units'])]
    best = {label: {key: [compact(r) for r in sorted(rows, key=lambda r: r['kernel_distances'][key])[:3]]
                    for key in COMPONENTS} for label, rows in [('replicated', full), ('short_screen', short)]}
    sources += [out/'cycle_closure_review.json', Path(__file__), ROOT/'src/topic4_xy_component_search.py']
    result = {'status': 'COMPONENT_COVERAGE_REVIEW_COMPLETE_NOT_MODEL_QUALIFICATION',
        'n_geometries': len(pool), 'replicated_anchor_eligible': len(full), 'short_eligible': len(short),
        'best_components': best, 'proposed_local_anchors': [compact(r) for r in local_anchors(pool, plan)],
        'proposed_racers_without_new_proposals': [compact(r) for r in select_racers(pool, 0, plan)],
        'finding': 'V3 reserves rank and timing nominees but no participation nominee; its joint-score local anchors exclude component specialists.',
        'scope': 'Optimizer coverage hypothesis only. Component specialists still fail joint fit. No angle objective or changed acceptance.',
        'source_hashes': {str(p): run.sha(p) for p in sources}}
    run.write(out/'component_coverage_review.json', result)
    print({k: v for k, v in result.items() if k not in ('best_components', 'source_hashes')})


if __name__ == '__main__':
    main()

"""Local proposals must follow replicated training evidence, not short-run luck."""
import numpy as np


def anchor_eligible(row,plan):
    required=set(plan['search']['fit_seeds']+plan['search']['race_seeds'])
    seeds=[u['seed'] for u in row['units']]
    return (row['explorable'] and row['n_events']>=plan['search']['minimum_pool_events']
        and len(seeds)==len(required) and set(seeds)==required
        and np.isfinite(row['exploration_score'])
        and all(not u['runaway'] and u['geometry']['minimum_clearance_mm']>=0
                and bool(u['geometry']['full_disks_disjoint']) for u in row['units']))


def proposal_pool(pool,plan):
    # Preserve all geometry hashes for duplicate rejection, while masking the
    # low-fidelity candidates only from local-anchor selection.
    return [{**r,'explorable':anchor_eligible(r,plan)} for r in pool]


def incumbent(pool,plan):
    return min((r for r in pool if anchor_eligible(r,plan)),key=lambda r:r['exploration_score'],default=None)

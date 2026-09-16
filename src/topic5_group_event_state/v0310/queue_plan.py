"""Queue construction for the frozen v0.3.10 dispatch order.

Phases follow spec section 5.1. U2/U3/T1 depend on the common recipe that U1
selects on INNER only, so the queue is generated phase by phase from completed
cards rather than up front.
"""
from __future__ import annotations

RECIPES = {'R0': dict(state_width=16, transition_rank=8, readout_hidden=32),
           'R1': dict(state_width=32, transition_rank=16, readout_hidden=64),
           'R2': dict(state_width=64, transition_rank=16, readout_hidden=128)}
FAMILIES = ('F', 'L', 'N')
SUBJECTS = ('epilepsiae_1096', 'epilepsiae_1125', 'epilepsiae_253')
SEEDS = (20260905, 20260906, 20260907)


def cell(phase, subject, family, recipe, seed, hours, mode, view='joint'):
    identifier = f'{phase}_{subject}_{family}_{recipe}_s{seed}_H{hours}_{mode}' + (f'_{view}' if view != 'joint' else '')
    return dict(id=identifier, phase=phase, subject=subject, family=family, recipe=recipe,
                seed=seed, history_hours=hours, source_mode=mode, view=view,
                group=f'{phase}|{subject}|{recipe}|{seed}|{hours}|{mode}|{view}', **RECIPES[recipe])


NOT_IMPLEMENTED_PHASES = {
    'P2C': dict(registered='synthetic data seeds 39002/39003 x F/L/N x three optimisation seeds, '
                           'at most 18 cells, under the NEW training rules',
                reason='the new training rules (parameter groups, plateau schedule with two learning-rate '
                       'drops, gradient accumulation, exact resume, fixed multi-task weights) are '
                       'implemented against the human measurement bundle. Running this on the synthetic '
                       'generator with the older fixed-learning-rate instrument would not be "the new '
                       'rules" and is refused rather than relabelled.',
                partial_substitute='the B0 bootstrap already varies the learning rate on the same '
                                   'generator and data seed 39001, which is a learning-rate sensitivity '
                                   'but not an independent synthetic realisation')}


def phase_cells(phase, common_recipe=None, common_family=None):
    """common_recipe: subject -> recipe id chosen by U1 on INNER.
    common_family: subject -> family chosen on INNER, used by the single-view arm."""
    out = []
    if phase == 'U1':
        for subject in SUBJECTS:
            for recipe in ('R0', 'R1', 'R2'):
                for family in FAMILIES:
                    out.append(cell('U1', subject, family, recipe, SEEDS[0], 8.0, 'event_only'))
    elif phase == 'U2':
        for subject in SUBJECTS:
            recipe = (common_recipe or {}).get(subject)
            if not recipe:
                continue
            for seed in SEEDS[1:]:
                for family in FAMILIES:
                    out.append(cell('U2', subject, family, recipe, seed, 8.0, 'event_only'))
    elif phase == 'T1H2':
        for subject in SUBJECTS:
            recipe = (common_recipe or {}).get(subject)
            if not recipe:
                continue
            for family in FAMILIES:
                out.append(cell('T1H2', subject, family, recipe, SEEDS[0], 2.0, 'event_only'))
    elif phase == 'U3':
        for subject in SUBJECTS:
            recipe = (common_recipe or {}).get(subject)
            if not recipe:
                continue
            for seed in SEEDS:
                for family in FAMILIES:
                    out.append(cell('U3', subject, family, recipe, seed, 8.0, 'background_conditional'))
    elif phase == 'T1H05':
        for subject in SUBJECTS:
            recipe = (common_recipe or {}).get(subject)
            if not recipe:
                continue
            for family in FAMILIES:
                out.append(cell('T1H05', subject, family, recipe, SEEDS[0], 0.5, 'event_only'))
    elif phase == 'P2A':
        for subject in SUBJECTS:
            recipe = (common_recipe or {}).get(subject)
            if not recipe:
                continue
            for hours in (16.0, 24.0):
                for family in ('L', 'N'):
                    out.append(cell('P2A', subject, family, recipe, SEEDS[0], hours, 'event_only'))
    elif phase == 'P2B':
        # Registered as three patients x two views x three seeds = 18 cells at the
        # INNER-selected family, not at all three families.
        for subject in SUBJECTS:
            recipe = (common_recipe or {}).get(subject)
            family = (common_family or {}).get(subject)
            if not recipe or not family:
                continue
            for view in ('count', 'recruitment'):
                for seed in SEEDS:
                    out.append(cell('P2B', subject, family, recipe, seed, 8.0, 'event_only', view))
    else:
        raise ValueError(phase)
    return out


DISPATCH_ORDER = ('U1', 'U2', 'T1H2', 'U3', 'T1H05', 'P2A', 'P2B')

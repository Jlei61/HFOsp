"""Strict core-count phases; later arms never backfill an unfinished phase."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT

def validated_tiers(core_counts, policy):
    tiers = policy.get('core_count_tiers', [list(core_counts)])
    if not isinstance(tiers, list) or not tiers or any((not isinstance(t, list) or not t for t in tiers)):
        raise ValueError('core_count_tiers must contain nonempty lists')
    flat = [k for tier in tiers for k in tier]
    if any((type(k) is not int for k in flat)) or len(set(flat)) != len(flat) or set(flat) != set(core_counts):
        raise ValueError('Each planned core count must occur in exactly one tier')
    return tiers

def completed_core_counts(streams, core_counts, restarts):
    """Require all training and nominated-point confirmations for every restart."""
    done = set()
    for k in core_counts:
        rows = [s for s in streams if s['core_count'] == k]
        if len(rows) == restarts and {s['restart'] for s in rows} == set(range(restarts)) and all((s['optimization_and_confirmation_complete'] for s in rows)):
            done.add(k)
    return done

def dispatch_phase(tiers, completed):
    for index, tier in enumerate(tiers):
        unfinished = set(tier) - set(completed)
        if unfinished:
            return (index, unfinished)
    return (None, set())

def partition_tasks(tasks, allowed):
    return ([t for t in tasks if t[1] in allowed], [t for t in tasks if t[1] not in allowed])

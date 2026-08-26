"""Train-only ambiguous-prefix families for the H2a targeted analysis.

A prefix family is ambiguous when the same opening recruitment pattern is seen
often enough in train and is genuinely followed by more than one continuation.
Support is measured on train events only; validation and test never contribute
to the definition of a family.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from .event_marks import load_patient, recruitment_groups

MIN_SUPPORT = 50
MIN_BRANCH_SUPPORT = 10
MIN_ENTROPY_BITS = 0.5
DEPTHS = (1, 2, 3)


@lru_cache(maxsize=64)
def eligible_families(subject: str) -> dict[int, frozenset]:
    """``{prefix depth: frozenset of ambiguous prefixes}`` for one patient."""
    events = load_patient(subject)
    train = np.flatnonzero(events.split_mask("train"))
    groups_cache = [recruitment_groups(events, int(e)) for e in train]
    out: dict[int, frozenset] = {}
    for depth in DEPTHS:
        branches: dict[tuple, Counter] = defaultdict(Counter)
        for groups in groups_cache:
            if len(groups) <= depth:
                continue
            prefix = tuple(sorted(int(c) for g in groups[:depth] for c in g))
            nxt = tuple(sorted(int(c) for c in groups[depth]))
            branches[prefix][nxt] += 1
        keep = set()
        for prefix, counter in branches.items():
            total = sum(counter.values())
            if total < MIN_SUPPORT:
                continue
            probabilities = np.array([c / total for c in counter.values()])
            entropy = float(-(probabilities * np.log2(probabilities)).sum())
            strong = sum(1 for c in counter.values() if c >= MIN_BRANCH_SUPPORT)
            if entropy >= MIN_ENTROPY_BITS and strong >= 2:
                keep.add(prefix)
        if keep:
            out[depth] = frozenset(keep)
    return out


def cohort_families(subjects) -> dict[str, dict[int, set]]:
    return {s: {d: set(v) for d, v in eligible_families(s).items()} for s in subjects}

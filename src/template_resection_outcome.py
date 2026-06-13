"""Track E1 (Yuquan-only clinical capstone) — template-resection coverage metrics.

Pure metric functions for "how completely was the propagation-template network
treated (RF-thermocoagulation / resection)". Outcome association is gated on
hospital follow-up labels (absent) and lives in the runner, not here.

Spec: docs/superpowers/specs/2026-06-13-yuquan-template-resection-outcome-design.md

Contract clauses honored here (see spec §3/§4/§5):
  - Name-not-index: every set operation is on channel NAME sets (caller normalizes
    with norm().upper()); no positional alignment.
  - NA vs 0: an absent/empty target returns None (NA), distinct from a real 0.0
    ("denominator present, nothing covered"). 0 != NA is a science contract.
  - shared_endpoint_core gate: caller gates on stable_k>=2; this guards <2 templates -> None.
  - hfo_rate_core: reuses src.sef_hfo_soz_localization.build_rate_lookup for the
    bipolar->monopolar first-contact bridge (do not re-invent); frozen top-k rule.
  - discordant_candidate: rule-driven and NA-safe (None metric -> clause unevaluable).
"""
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from src.sef_hfo_soz_localization import build_rate_lookup, classify_montage


def coverage(target: Iterable[str], treated: Iterable[str]) -> Optional[float]:
    """|target ∩ treated| / |target|, on channel-name sets.

    Returns None (NA) when the target set is empty/absent — this is NOT 0.0.
    0.0 means "target present, nothing in it was treated"; None means "no target
    to evaluate" (e.g. missing source). The two are scientifically different.
    """
    target = set(target)
    if not target:
        return None
    return len(target & set(treated)) / len(target)


def shared_endpoint_core(endpoint_sets_per_template: Sequence[Iterable[str]]) -> Optional[Set[str]]:
    """Channels that are endpoints (source∪sink) in EVERY stable template = intersection.

    Caller gates on stable_k>=2; here we guard: <2 templates -> None (NA). An empty
    intersection with >=2 templates returns set() (a real "no shared core"), not None.
    """
    sets = [set(e) for e in endpoint_sets_per_template]
    if len(sets) < 2:
        return None
    core = sets[0]
    for s in sets[1:]:
        core = core & s
    return core


def hfo_rate_core(
    channel_metrics: Sequence[dict],
    soz_size: int,
    min_ch_events: int = 30,
) -> Tuple[Set[str], int]:
    """Size-matched HFO-rate core (monopolar) = top-k bridged contacts by event_rate.

    Frozen rule (spec §4.2):
      - bipolar->monopolar first-contact bridge + min_ch_events gate via
        src.sef_hfo_soz_localization.build_rate_lookup (reuse, not re-invent).
      - k = min(soz_size, n_available).
      - tie-break: stable sort by (-event_rate, channel_name).

    Returns (core_set, n_available). n_available = bridged contacts passing the
    event gate. Empty pool or soz_size==0 -> (set(), n_available).
    """
    montage = classify_montage(channel_metrics)
    rate_lookup = build_rate_lookup(channel_metrics, montage, min_ch_events)
    n_available = len(rate_lookup)
    if n_available == 0 or soz_size <= 0:
        return set(), n_available
    k = min(soz_size, n_available)
    ranked = sorted(rate_lookup.items(), key=lambda kv: (-kv[1], kv[0]))
    return {name for name, _ in ranked[:k]}, n_available


def discordant_candidate(
    early_end_cov: Optional[float],
    template_endpoint_cov: Optional[float],
    clinical_network_cov: Optional[float],
    template_source: Iterable[str],
    clinical_origin: Iterable[str],
    multi_session: bool,
) -> bool:
    """Rule-driven discordant-candidate flag (spec §5), NA-safe.

    True if ANY clause holds; a None (NA) metric makes its clause unevaluable
    (treated as not-firing, never a crash). Clauses:
      early_end_cov == 0
      OR template_endpoint_cov < 0.5
      OR clinical_network_cov < 0.5
      OR (template_source ∩ clinical_origin) == ∅  (both non-empty)
      OR multi_session
    """
    if multi_session:
        return True
    if early_end_cov is not None and early_end_cov == 0:
        return True
    if template_endpoint_cov is not None and template_endpoint_cov < 0.5:
        return True
    if clinical_network_cov is not None and clinical_network_cov < 0.5:
        return True
    ts, co = set(template_source), set(clinical_origin)
    if ts and co and not (ts & co):
        return True
    return False

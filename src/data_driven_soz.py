"""Data-driven ictal-onset SOZ audit (PR-T3-1).

Step 0 helpers:

- ``annotate_clinical_soz``: 3-state (SOZ/nonSOZ/unknown) annotation of an
  analysis channel set against a clinical SOZ list, using the canonical
  bipolar-to-any matcher from ``src.event_periodicity``.

Step 1 (M1 — HFO-onset rate enrichment), Step 2 (M2 ER-log-ratio +
Nyquist / filter padding guards), and the per-seizure aggregation /
ranking helpers will be added in their respective commits.

See ``docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md``.
"""

from __future__ import annotations

from typing import Dict, Iterable

from src.event_periodicity import _normalize_channel_name, match_bipolar_soz


SOZ_LABEL = "soz"
NON_SOZ_LABEL = "non_soz"
UNKNOWN_LABEL = "unknown"


def annotate_clinical_soz(
    analysis_channels: Iterable[str],
    clinical_soz: Iterable[str],
) -> Dict[str, str]:
    """Annotate each analysis channel as SOZ / non_soz / unknown.

    Plan §3.2 contract:

    - Bipolar ``X-Y``: if X or Y is in ``clinical_soz`` → ``"soz"``;
      else → ``"non_soz"``.
    - If ``X`` or ``Y`` is empty / whitespace-only (malformed name)
      → ``"unknown"``.
    - CAR / monopolar ``X``: same logic with single contact.

    The matcher reuses ``src.event_periodicity.match_bipolar_soz`` for the
    SOZ vs nonSOZ branch but adds the ``"unknown"`` branch the plan
    requires.

    Returns ``{channel_name: label}`` preserving the input ordering of
    ``analysis_channels``.
    """
    soz_set = {_normalize_channel_name(s) for s in clinical_soz}
    out: Dict[str, str] = {}
    for ch in analysis_channels:
        normalized = _normalize_channel_name(ch)
        parts = [p.strip() for p in normalized.split("-")]
        if any(not p for p in parts):
            out[ch] = UNKNOWN_LABEL
            continue
        out[ch] = match_bipolar_soz(ch, soz_set)
    return out


__all__ = [
    "annotate_clinical_soz",
    "SOZ_LABEL",
    "NON_SOZ_LABEL",
    "UNKNOWN_LABEL",
]

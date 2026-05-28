"""Centralized thresholds for HR burst detection + regime classification.

Centralizing here avoids magic numbers in hr_dynamics.py and lets the
sweep / CLI override per experiment.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BurstConfig:
    """Event (spike-level) detection thresholds, in HR time units.

    **Recalibrated 2026-05-28 (Stage 1 pre-Task-5 diagnostic)** to HR's
    *measured* fast-spike timescale, replacing the framework-time guessed
    values (x_threshold=1.0, min_burst_duration=5.0, bridge_gap=2.0) that
    assumed "1 HR unit ≈ 1 ms, bursts ≥5 ms". Diagnostic finding: an HR
    fast spike is only ~0.85 HR units wide above x=1.0, and intra-burst
    ISI ≈ 15 units >> the old bridge_gap=2.0. With the old thresholds
    `detect_bursts` returned 0 events across the entire (I, r, σ) grid,
    making the Stage 1 "excitable regime exists" exit-contract impossible
    to satisfy.

    Design decision (documented, reversible): the detected unit is a
    **spike-level excursion**, not a multi-spike burst envelope. At the
    excitable boundary where Stage 1 operates (I ∈ [-1, 1] with noise),
    noise-triggered activity is a single brief threshold crossing that
    returns to rest — so spike-level detection is both robust across
    parameters and the scientifically appropriate event unit there.
    The burst-vs-spike distinction only becomes load-bearing in Stage 2-3
    (propagation order / participation), where it will be revisited.

    Verified post-recalibration: a 2000-unit sweep shows σ=0 → 0 events
    (silent rest) for I ∈ [-1, 1], graded noise-triggered events for σ>0,
    no runaway — i.e. a clean excitable band exists.
    """
    x_threshold: float = 0.0          # spike-onset crossing (HR rest≈-1.3, peak≈+1.8)
    min_burst_duration: float = 0.3   # below this = sub-spike noise flicker; reject
    bridge_gap: float = 1.0           # merge gaps shorter than this within one event


@dataclass(frozen=True)
class RegimeConfig:
    """Regime classification thresholds (HR time units)."""
    max_burst_duration: float = 100.0  # exceeding this = "unstable"
    excitable_max_burst: float = 50.0  # all bursts shorter than this = excitable-compatible
    excitable_min_ibi: float = 30.0    # IBI shorter than this = repetitive-burst

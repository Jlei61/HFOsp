"""Centralized thresholds for HR burst detection + regime classification.

Centralizing here avoids magic numbers in hr_dynamics.py and lets the
sweep / CLI override per experiment.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BurstConfig:
    """Burst detection thresholds (HR time units; spec: 1 HR unit ≈ 1 ms)."""
    x_threshold: float = 1.0       # x crossing to call "above"
    min_burst_duration: float = 5.0  # below this is rejected as noise spike
    bridge_gap: float = 2.0          # gaps shorter than this don't split a burst


@dataclass(frozen=True)
class RegimeConfig:
    """Regime classification thresholds (HR time units)."""
    max_burst_duration: float = 100.0  # exceeding this = "unstable"
    excitable_max_burst: float = 50.0  # all bursts shorter than this = excitable-compatible
    excitable_min_ibi: float = 30.0    # IBI shorter than this = repetitive-burst

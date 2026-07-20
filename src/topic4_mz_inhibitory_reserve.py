"""Inhibitory-reserve slow geometry for the independent Topic-4 MZ route.

The frozen R0 screen uses ``q`` directly as an existing fast-system control
coordinate.  These helpers register how a later depletable fraction ``D_I``
maps to that coordinate without changing the recurrent E-to-E scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class InhibitoryReserveParameters:
    """One regional reserve law; all time constants are in ms."""

    q_rest: float = 0.90
    q_reserve: float = 0.80
    tau_recovery_ms: float = 20000.0
    tau_depletion_ms: float = 1000.0

    def validate(self) -> "InhibitoryReserveParameters":
        values = (
            self.q_rest,
            self.q_reserve,
            self.tau_recovery_ms,
            self.tau_depletion_ms,
        )
        if not all(np.isfinite(values)):
            raise ValueError("reserve parameters must be finite")
        if not 0.0 < self.q_reserve < self.q_rest <= 1.0:
            raise ValueError("reserve multipliers must satisfy 0<q_reserve<q_rest<=1")
        if self.tau_recovery_ms <= 0.0 or self.tau_depletion_ms <= 0.0:
            raise ValueError("reserve time constants must be positive")
        return self

    def effective_q(self, fraction: float | np.ndarray) -> np.ndarray:
        """Map a depletable fraction in [0,1] to the effective inhibition q."""

        checked = self.validate()
        value = np.asarray(fraction, dtype=float)
        if not np.all(np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError("depletable fraction must be finite and lie in [0,1]")
        return checked.q_reserve + (checked.q_rest - checked.q_reserve) * value

    def fraction_rhs(
        self,
        fraction: float | np.ndarray,
        inhibitory_use: float | np.ndarray,
    ) -> np.ndarray:
        """Evaluate dD/dt for a bounded non-negative use sensor."""

        checked = self.validate()
        value = np.asarray(fraction, dtype=float)
        use = np.asarray(inhibitory_use, dtype=float)
        if not np.all(np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError("depletable fraction must be finite and lie in [0,1]")
        if not np.all(np.isfinite(use)) or np.any(use < 0.0):
            raise ValueError("inhibitory use must be finite and non-negative")
        return (
            (1.0 - value) / checked.tau_recovery_ms
            - value * use / checked.tau_depletion_ms
        )

    def q_rhs(
        self,
        effective_q: float | np.ndarray,
        inhibitory_use: float | np.ndarray,
    ) -> np.ndarray:
        """Equivalent dq/dt, useful for slow-nullcline diagnostics."""

        checked = self.validate()
        q = np.asarray(effective_q, dtype=float)
        use = np.asarray(inhibitory_use, dtype=float)
        if (
            not np.all(np.isfinite(q))
            or np.any(q < checked.q_reserve)
            or np.any(q > checked.q_rest)
        ):
            raise ValueError("effective q must lie between q_reserve and q_rest")
        if not np.all(np.isfinite(use)) or np.any(use < 0.0):
            raise ValueError("inhibitory use must be finite and non-negative")
        return (
            (checked.q_rest - q) / checked.tau_recovery_ms
            - (q - checked.q_reserve) * use / checked.tau_depletion_ms
        )

    def q_nullcline(self, mean_use: float | np.ndarray) -> np.ndarray:
        """Return the frozen-cycle averaged q-nullcline for a fixed mean use."""

        checked = self.validate()
        use = np.asarray(mean_use, dtype=float)
        if not np.all(np.isfinite(use)) or np.any(use < 0.0):
            raise ValueError("mean use must be finite and non-negative")
        numerator = (
            checked.tau_depletion_ms * checked.q_rest
            + checked.tau_recovery_ms * use * checked.q_reserve
        )
        denominator = checked.tau_depletion_ms + checked.tau_recovery_ms * use
        return numerator / denominator


def reserve_floor_for_hold(
    q_hold: float | np.ndarray,
    mean_use: float | np.ndarray,
    *,
    q_rest: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
) -> np.ndarray:
    """Invert the averaged q-nullcline; this is not a free parameter sweep."""

    q = np.asarray(q_hold, dtype=float)
    use = np.asarray(mean_use, dtype=float)
    values = (q_rest, tau_recovery_ms, tau_depletion_ms)
    if not all(np.isfinite(values)) or not 0.0 < q_rest <= 1.0:
        raise ValueError("q_rest must be finite and lie in (0,1]")
    if tau_recovery_ms <= 0.0 or tau_depletion_ms <= 0.0:
        raise ValueError("time constants must be positive")
    if not np.all(np.isfinite(q)) or np.any((q <= 0.0) | (q >= q_rest)):
        raise ValueError("q_hold must be finite and lie below q_rest")
    if not np.all(np.isfinite(use)) or np.any(use <= 0.0):
        raise ValueError("mean_use must be finite and strictly positive")
    floor = q - (tau_depletion_ms / tau_recovery_ms) * (q_rest - q) / use
    return floor


def safe_q_intervals(
    q_nodes: Sequence[float],
    safe: Sequence[bool],
    *,
    maximum_spacing: float,
) -> list[list[float]]:
    """Group sorted safe q nodes without bridging an untested or failed gap."""

    q = np.asarray(q_nodes, dtype=float)
    mask = np.asarray(safe, dtype=bool)
    if q.ndim != 1 or mask.shape != q.shape or q.size == 0:
        raise ValueError("q_nodes and safe flags must be non-empty aligned vectors")
    if not np.all(np.isfinite(q)) or np.unique(q).size != q.size:
        raise ValueError("q_nodes must be finite and unique")
    if not np.isfinite(maximum_spacing) or maximum_spacing <= 0.0:
        raise ValueError("maximum_spacing must be positive")
    order = np.argsort(q)
    q = q[order]
    mask = mask[order]
    intervals: list[list[float]] = []
    current: list[float] = []
    for value, accepted in zip(q, mask):
        if not accepted:
            if current:
                intervals.append(current)
                current = []
            continue
        if current and value - current[-1] > maximum_spacing + 1.0e-12:
            intervals.append(current)
            current = []
        current.append(float(value))
    if current:
        intervals.append(current)
    return intervals


def interval_passes_gate(
    nodes: Iterable[float],
    *,
    minimum_width: float,
    minimum_nodes: int,
) -> bool:
    """Apply the preregistered width/node gate to one safe q interval."""

    values = np.asarray(list(nodes), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("interval nodes must be a finite non-empty vector")
    if minimum_width <= 0.0 or minimum_nodes < 2:
        raise ValueError("interval gate must require positive width and at least two nodes")
    return bool(
        values.size >= int(minimum_nodes)
        and float(np.max(values) - np.min(values)) >= float(minimum_width) - 1.0e-12
    )

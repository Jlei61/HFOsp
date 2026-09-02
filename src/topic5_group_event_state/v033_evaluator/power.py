"""D0-D5 replicates, power curves and effect tiers (plan Task 6).

Effect axis (P1): the Level-0 oracle held-out deviance gain in nats per anchor,
with the per-block signal-to-noise ratio alongside.  Power (P2): the fraction of
replicates whose block-bootstrap 95 % CI lower bound of the shared-dispersion
gain exceeds ``oracle.DETECTION_FLOOR_NATS``; the same rule on a state-free truth
gives the false-positive rate.  Effect tiers are defined on the oracle-gain
scale, never on beta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import resource
import time
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm

from . import dgp as D
from . import oracle as O
from .scaffold import Scaffold

EFFECT_TIER_TARGETS = {"small": 0.02, "medium": 0.05, "large": 0.15}   # nats / anchor, Level-0 oracle gain
EFFECT_AXIS = "level0_oracle_held_out_deviance_gain_nats_per_anchor"
DETECTION_RULE = ("block-bootstrap 95% CI lower bound of the shared-H-dispersion gain "
                  f"> {O.DETECTION_FLOOR_NATS:g} nats; blocks = non-overlapping horizon bins inside a target segment")
DEFAULT_POWER = 0.8
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class ReplicateSpec:
    kind: str
    beta_count: float
    beta_grammar: float
    replicate: int
    generator_seed: int | None = None
    noise_seed: int | None = None
    estimator_seed: int | None = None

    def resolved(self) -> dict[str, Any]:
        kind_index = D.DGP_KINDS.index(self.kind)
        base = 100_000 + 7919 * int(self.replicate) + 104_729 * kind_index \
            + int(round(1000 * self.beta_count)) * 13 + int(round(1000 * self.beta_grammar)) * 17
        gen = base if self.generator_seed is None else int(self.generator_seed)
        noise = base + 50_021 if self.noise_seed is None else int(self.noise_seed)
        est = int(self.replicate) if self.estimator_seed is None else int(self.estimator_seed)
        return {**asdict(self), "generator_seed": gen, "noise_seed": noise, "estimator_seed": est,
                "seeds_recorded": True}


def block_snr(block_means: Sequence[float]) -> float | None:
    b = np.asarray(block_means, dtype=np.float64)
    if b.size < 2:
        return None
    sd = float(b.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return float(b.mean() / sd)


def required_blocks_for_power(snr: float | None, *, power: float = DEFAULT_POWER, alpha: float = DEFAULT_ALPHA) -> int | None:
    """Independent blocks needed for a two-sided z-test at ``alpha`` to reach ``power`` at per-block SNR ``snr``."""

    if snr is None or not np.isfinite(snr) or snr <= 0:
        return None
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return int(math.ceil((z / float(snr)) ** 2))


def _summarise_level(level: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in level.items() if k not in ("table", "block_gain_means")}
    out["block_snr"] = block_snr(level.get("block_gain_means", []))
    out["required_blocks_for_80pct_power"] = required_blocks_for_power(out["block_snr"])
    return out


def run_replicate(scaffold: Scaffold, spec: ReplicateSpec, *, horizon: float, views: Sequence[str] = O.PRIMARY_VIEWS,
                  levels: Sequence[int] = O.LEVELS, n_steps: int = 200) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = spec.resolved()
    data = D.generate(scaffold, spec.kind, beta_count=spec.beta_count, beta_grammar=spec.beta_grammar,
                      generator_seed=resolved["generator_seed"], noise_seed=resolved["noise_seed"])
    cascades: dict[str, Any] = {}
    for view in views:
        cascade = O.run_cascade(scaffold, data, view=view, horizon=horizon, seed=resolved["estimator_seed"],
                                n_steps=n_steps, levels=tuple(levels))
        cascade["levels"] = [_summarise_level(lvl) for lvl in cascade["levels"]]
        cascades[view] = cascade
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "spec": resolved, "subject": scaffold.subject, "horizon_seconds": float(horizon), "carry": scaffold.carry,
        "dgp": data.as_meta(), "cascades": cascades,
        "resources": {"wall_seconds": time.perf_counter() - started, "peak_rss_mib": peak_kib / 1024.0},
    }


def _median_iqr(values: Sequence[float]) -> dict[str, float | None]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {"median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)),
            "n": int(v.size)}


def power_curve(replicates: Sequence[dict[str, Any]], *, view: str) -> dict[str, Any]:
    cells: dict[tuple, list[dict[str, Any]]] = {}
    for rep in replicates:
        cascade = rep.get("cascades", {}).get(view)
        if cascade is None:
            continue
        key = (rep["spec"]["kind"], float(rep["spec"]["beta_count"]), float(rep["spec"]["beta_grammar"]))
        cells.setdefault(key, []).append(cascade)
    out_cells = []
    for (kind, bc, bg), items in sorted(cells.items()):
        truth = bool(items[0]["truth_has_state"])
        levels = sorted({lvl["level"] for c in items for lvl in c["levels"]})
        rate = {str(l): float(np.mean([any(x["level"] == l and x["detected"] for x in c["levels"]) for c in items]))
                for l in levels}
        cell: dict[str, Any] = {
            "kind": kind, "beta_count": bc, "beta_grammar": bg, "truth_has_state": truth, "n_replicates": len(items),
            "gain_by_level": {str(l): _median_iqr([x["gain"] for c in items for x in c["levels"] if x["level"] == l]) for l in levels},
            "block_snr_by_level": {str(l): _median_iqr([x.get("block_snr") for c in items for x in c["levels"] if x["level"] == l]) for l in levels},
            "oracle_gain_level0": _median_iqr([x["gain"] for c in items for x in c["levels"] if x["level"] == 0]),
            "failure_location_counts": {},
        }
        for c in items:
            loc = c.get("failure_location")
            cell["failure_location_counts"][loc] = cell["failure_location_counts"].get(loc, 0) + 1
        if truth:
            cell["power_by_level"] = rate
        else:
            cell["false_positive_rate_by_level"] = rate
        out_cells.append(cell)
    return {"view": view, "effect_axis": EFFECT_AXIS, "detection_rule": DETECTION_RULE, "cells": out_cells}


def assign_effect_tiers(cells: Sequence[dict[str, Any]], *, beta_key: str) -> dict[str, Any]:
    """Pick, per tier, the cell whose median Level-0 oracle gain is closest to the tier target."""

    candidates = [c for c in cells if c.get("truth_has_state") and c["oracle_gain_level0"].get("median") is not None]
    out: dict[str, Any] = {"definition": {f"{k}_target_gain_nats": v for k, v in EFFECT_TIER_TARGETS.items()},
                           "effect_axis": EFFECT_AXIS}
    for tier, target in EFFECT_TIER_TARGETS.items():
        if not candidates:
            out[tier] = None
            continue
        best = min(candidates, key=lambda c: abs(c["oracle_gain_level0"]["median"] - target))
        snr0 = best["block_snr_by_level"].get("0", {}).get("median")
        out[tier] = {
            beta_key: best[beta_key], "kind": best.get("kind"),
            "oracle_gain_median": best["oracle_gain_level0"]["median"], "target_gain_nats": target,
            "block_snr_level0_median": snr0,
            "required_blocks_level0": required_blocks_for_power(snr0),
            "required_blocks_by_level": {l: required_blocks_for_power(v.get("median"))
                                         for l, v in best["block_snr_by_level"].items()},
        }
    return out

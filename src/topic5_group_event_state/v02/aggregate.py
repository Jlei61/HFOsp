"""Patient-first cohort aggregation of the future-block comparison.

Three disciplines this repository has already had to learn, all enforced here:

* **the patient is the denominator, not the anchor and not the seed.**  A 2 h
  window stepped every 5 min gives 24 overlapping anchors per independent
  window; seeds are repeated fits of one patient.  Both are reported, neither is
  a sample size.
* **an effect smaller than the seed-to-seed spread is not a per-patient
  finding.**  Every arm reports its across-seed spread next to its effect, and
  the two are compared explicitly.
* **cells flagged ``not_estimable`` are dropped and counted**, never silently
  averaged in as a weak negative.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PRIMARY_ENDPOINTS = (
    "count",
    "participation",
    "continuous",
    "continuous:size",
    "continuous:span",
    "continuous:band_energy",
    "continuous:band_peak",
    "continuous:embedding",
)


def sign_test_p(n_positive: int, n_total: int) -> float:
    """Two-sided exact binomial test against p=0.5."""

    if n_total == 0:
        return float("nan")
    k, n = int(n_positive), int(n_total)
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    observed = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= observed + 1e-15)))


def load_results(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(root).glob("per_subject/*.json")):
        out.append(json.loads(path.read_text()))
    return out


def _arm_scores(entry: Mapping[str, Any], arm: str) -> dict[str, float] | None:
    payload = entry["arms"].get(arm)
    if payload is None:
        return None
    return {k: v["nll_per_unit"] for k, v in payload["scores"].items()}


def _estimable(entry: Mapping[str, Any], arm: str, endpoint: str) -> bool:
    payload = entry["arms"].get(arm)
    if payload is None:
        return False
    flags = payload.get("estimability")
    if flags is None:
        return True
    return flags.get(endpoint, "ok") == "ok"


@dataclass(frozen=True)
class CohortCell:
    horizon_seconds: float
    arm: str
    endpoint: str
    per_subject: dict[str, float]
    n_subjects: int
    n_positive: int
    median: float
    p_sign: float
    n_dropped_not_estimable: int
    n_dropped_missing: int
    seed_spread_median: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "arm": self.arm,
            "endpoint": self.endpoint,
            "n_subjects": self.n_subjects,
            "n_positive": self.n_positive,
            "median_gain": self.median,
            "p_sign": self.p_sign,
            "n_dropped_not_estimable": self.n_dropped_not_estimable,
            "n_dropped_missing": self.n_dropped_missing,
            "seed_spread_median": self.seed_spread_median,
            "per_subject": self.per_subject,
        }


def gain_cells(
    results: Sequence[Mapping[str, Any]],
    *,
    reference_arm: str = "B_multiscale",
    arms: Sequence[str] | None = None,
    endpoints: Sequence[str] = PRIMARY_ENDPOINTS,
    seed_gains: Mapping[str, Mapping[str, Mapping[str, dict[str, float]]]] | None = None,
) -> list[CohortCell]:
    """Per (horizon, arm, endpoint) cohort summary of the gain over the baseline."""

    horizons: list[str] = sorted(
        {h for r in results for h in r["horizons"]}, key=lambda s: float(s[:-1])
    )
    all_arms = sorted({a for r in results for h in r["horizons"].values()
                       for a in h.get("arms", {})})
    use_arms = list(arms) if arms is not None else [
        a for a in all_arms if a not in (reference_arm, "intercept")
    ]
    cells: list[CohortCell] = []
    for hk in horizons:
        for arm in use_arms:
            for endpoint in endpoints:
                values: dict[str, float] = {}
                dropped_ne = dropped_missing = 0
                for r in results:
                    entry = r["horizons"].get(hk)
                    if entry is None or entry.get("status") != "ok":
                        dropped_missing += 1
                        continue
                    ref = _arm_scores(entry, reference_arm)
                    cur = _arm_scores(entry, arm)
                    if ref is None or cur is None or endpoint not in ref or endpoint not in cur:
                        dropped_missing += 1
                        continue
                    if not (_estimable(entry, arm, endpoint)
                            and _estimable(entry, reference_arm, endpoint)):
                        dropped_ne += 1
                        continue
                    values[r["subject"]] = float(ref[endpoint] - cur[endpoint])
                v = np.array(list(values.values()), dtype=float)
                spread = None
                if seed_gains:
                    per = [
                        seed_gains[s][hk][arm].get(endpoint)
                        for s in values
                        if s in seed_gains and hk in seed_gains[s]
                        and arm in seed_gains[s][hk]
                    ]
                    per = [x for x in per if x is not None and np.isfinite(x)]
                    spread = float(np.median(per)) if per else None
                cells.append(CohortCell(
                    horizon_seconds=float(hk[:-1]),
                    arm=arm,
                    endpoint=endpoint,
                    per_subject=values,
                    n_subjects=int(v.size),
                    n_positive=int((v > 0).sum()),
                    median=float(np.median(v)) if v.size else float("nan"),
                    p_sign=sign_test_p(int((v > 0).sum()), int(v.size)),
                    n_dropped_not_estimable=dropped_ne,
                    n_dropped_missing=dropped_missing,
                    seed_spread_median=spread,
                ))
    return cells


def denominator_table(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The honest denominators: anchors, independent windows, events per split."""

    rows: list[dict[str, Any]] = []
    for r in results:
        for hk, entry in r["horizons"].items():
            row = {"subject": r["subject"], "horizon_seconds": float(hk[:-1]),
                   "status": entry.get("status")}
            for split, d in entry.get("denominators", {}).items():
                row[f"{split}_anchors"] = d["n_anchors"]
                row[f"{split}_independent_windows"] = d["n_independent_windows"]
            rows.append(row)
    return rows


def summarise(cells: Iterable[CohortCell]) -> dict[str, Any]:
    return {"cells": [c.as_dict() for c in cells]}

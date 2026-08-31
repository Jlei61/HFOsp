"""Patient-first aggregation and hypothesis read-out for Group-Event State v0.1.

Two rules this module enforces mechanically, because this project has been
burnt by both before:

* **patient-first.** Events and sliding windows are never counted as independent
  samples.  Every comparison collapses seeds within a patient first, then treats
  patients as the unit.
* **the seed-noise floor is reported in the same units as the effect.** An arm
  difference smaller than the spread produced by merely changing the seed is not
  a finding, and the table has to make that visible without extra arithmetic.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ENDPOINTS = (
    "timing",
    "participation",
    "group_size",
    "delay",
    "band_energy",
    "band_peak",
    "cross_band_lag",
)


def load_runs(runs_dir: Path, tag: str = "main") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(Path(runs_dir, tag).glob("*/result.json")):
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def seed_payload_identity(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Guard against reporting N identical payloads as N seeds."""

    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for run in runs:
        key = (run["subject"], run["arm"])
        signature = json.dumps(
            {k: run["test"].get(k) for k in ENDPOINTS}, sort_keys=True
        )
        grouped[key].append((int(run["seed"]), signature))
    duplicates = []
    for key, items in grouped.items():
        signatures = {sig for _seed, sig in items}
        if len(items) > 1 and len(signatures) < len(items):
            duplicates.append({"subject": key[0], "arm": key[1], "n_seeds": len(items),
                               "n_distinct_payloads": len(signatures)})
    return {
        "n_groups": len(grouped),
        "n_groups_with_duplicate_payloads": len(duplicates),
        "duplicates": duplicates,
    }


def patient_arm_table(
    runs: Sequence[Mapping[str, Any]], field: str = "test"
) -> dict[tuple[str, str], dict[str, Any]]:
    """(patient, arm) -> per-endpoint median over seeds plus the seed spread."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["subject"], run["arm"])].append(run)
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        entry: dict[str, Any] = {
            "n_seeds": len(items),
            "seeds": sorted(int(i["seed"]) for i in items),
            "dataset": items[0]["dataset"],
            "n_events_test": items[0]["n_events_test"],
            "n_parameters": items[0]["n_parameters"],
            "selected_epoch": [int(i["selected_epoch"]) for i in items],
        }
        for endpoint in ENDPOINTS:
            values = np.array(
                [float(i.get(field, {}).get(endpoint, np.nan)) for i in items], dtype=float
            )
            values = values[np.isfinite(values)]
            entry[endpoint] = float(np.median(values)) if values.size else float("nan")
            entry[f"{endpoint}__seed_spread"] = (
                float(np.max(values) - np.min(values)) if values.size > 1 else 0.0
            )
        table[key] = entry
    return table


def _wilcoxon(deltas: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        nonzero = deltas[deltas != 0]
        if nonzero.size < 5:
            return float("nan")
        return float(wilcoxon(nonzero).pvalue)
    except Exception:
        return float("nan")


def _sign_test(deltas: np.ndarray) -> float:
    try:
        from scipy.stats import binomtest

        pos = int((deltas < 0).sum())  # negative delta = arm A has lower loss = better
        n = int((deltas != 0).sum())
        if n == 0:
            return float("nan")
        return float(binomtest(pos, n, 0.5).pvalue)
    except Exception:
        return float("nan")


def paired_comparison(
    table: Mapping[tuple[str, str], Mapping[str, Any]],
    arm_a: str,
    arm_b: str,
    endpoint: str,
) -> dict[str, Any]:
    """Patient-first paired comparison of two arms on one endpoint.

    Lower is better for every endpoint here (NLL, or MAE for group size), so a
    negative delta means ``arm_a`` wins.
    """

    subjects = sorted(
        {s for (s, a) in table if a == arm_a} & {s for (s, a) in table if a == arm_b}
    )
    deltas, spreads, rows = [], [], []
    for subject in subjects:
        a = table[(subject, arm_a)]
        b = table[(subject, arm_b)]
        if not (math.isfinite(a[endpoint]) and math.isfinite(b[endpoint])):
            continue
        delta = a[endpoint] - b[endpoint]
        noise = max(a[f"{endpoint}__seed_spread"], b[f"{endpoint}__seed_spread"])
        deltas.append(delta)
        spreads.append(noise)
        rows.append(
            {
                "subject": subject,
                "dataset": a["dataset"],
                f"{arm_a}": a[endpoint],
                f"{arm_b}": b[endpoint],
                "delta": delta,
                "seed_spread": noise,
                "beats_seed_noise": bool(abs(delta) > noise),
            }
        )
    deltas = np.asarray(deltas, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    if deltas.size == 0:
        return {"arm_a": arm_a, "arm_b": arm_b, "endpoint": endpoint, "n_patients": 0}
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "endpoint": endpoint,
        "n_patients": int(deltas.size),
        "n_patients_arm_a_better": int((deltas < 0).sum()),
        "median_delta": float(np.median(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "iqr_delta": [float(np.percentile(deltas, 25)), float(np.percentile(deltas, 75))],
        "median_seed_spread": float(np.median(spreads)),
        "effect_over_seed_noise": (
            float(abs(np.median(deltas)) / np.median(spreads))
            if np.median(spreads) > 0
            else float("inf")
        ),
        "n_patients_beating_seed_noise": int((np.abs(deltas) > spreads).sum()),
        "wilcoxon_p": _wilcoxon(deltas),
        "sign_test_p": _sign_test(deltas),
        "per_patient": rows,
    }


# Endpoints where a HIGHER value is better, so the delta sign flips.
HIGHER_IS_BETTER = (
    "participation_auc",
    "recruitment_order_spearman",
    "tied_group_agreement",
    "prefix_continuation_spearman",
    "prefix_next_contact_hit",
)


def _extract_scalar(run: Mapping[str, Any], name: str) -> float:
    if name == "participation_auc":
        return float(run.get("participation_auc_sampled", float("nan")))
    entry = run.get(name)
    if isinstance(entry, Mapping):
        return float(entry.get("median", float("nan")))
    return float("nan")


def derived_comparison(
    runs: Sequence[Mapping[str, Any]], arm_a: str, arm_b: str, name: str
) -> dict[str, Any]:
    """Patient-first comparison on a derived endpoint where higher is better."""

    per_patient: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        value = _extract_scalar(run, name)
        if math.isfinite(value):
            per_patient[run["subject"]][run["arm"]].append(value)
    deltas, spreads, rows = [], [], []
    for subject, arms in sorted(per_patient.items()):
        if arm_a not in arms or arm_b not in arms:
            continue
        a = float(np.median(arms[arm_a]))
        b = float(np.median(arms[arm_b]))
        noise = max(
            float(np.max(arms[arm_a]) - np.min(arms[arm_a])) if len(arms[arm_a]) > 1 else 0.0,
            float(np.max(arms[arm_b]) - np.min(arms[arm_b])) if len(arms[arm_b]) > 1 else 0.0,
        )
        deltas.append(a - b)
        spreads.append(noise)
        rows.append({"subject": subject, arm_a: a, arm_b: b, "delta": a - b,
                     "seed_spread": noise, "beats_seed_noise": bool(abs(a - b) > noise)})
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        return {"arm_a": arm_a, "arm_b": arm_b, "endpoint": name, "n_patients": 0}
    sp = np.asarray(spreads, dtype=float)
    return {
        "arm_a": arm_a, "arm_b": arm_b, "endpoint": name,
        "higher_is_better": True,
        "n_patients": int(arr.size),
        "n_patients_arm_a_better": int((arr > 0).sum()),
        "median_delta": float(np.median(arr)),
        "median_seed_spread": float(np.median(sp)),
        "effect_over_seed_noise": (
            float(abs(np.median(arr)) / np.median(sp)) if np.median(sp) > 0 else float("inf")
        ),
        "n_patients_beating_seed_noise": int((np.abs(arr) > sp).sum()),
        "wilcoxon_p": _wilcoxon(arr),
        "sign_test_p": _sign_test(-arr),  # _sign_test counts negatives as wins
        "per_patient": rows,
    }


def truncation_curve(runs: Sequence[Mapping[str, Any]], arm: str, endpoint: str) -> dict[str, Any]:
    """H1 probe: does letting the state live longer than K events help?"""

    per_patient: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run["arm"] != arm:
            continue
        for key, means in run.get("history_truncation", {}).items():
            value = means.get(endpoint)
            if value is not None and math.isfinite(value):
                per_patient[run["subject"]][key].append(float(value))
    keys = sorted({k for v in per_patient.values() for k in v})
    summary: dict[str, Any] = {"arm": arm, "endpoint": endpoint, "levels": {}}
    for key in keys:
        values = [float(np.median(v[key])) for v in per_patient.values() if v.get(key)]
        summary["levels"][key] = {
            "n_patients": len(values),
            "median": float(np.median(values)) if values else float("nan"),
        }
    full = "full_session"
    if full in keys:
        for key in keys:
            if key == full:
                continue
            deltas = [
                float(np.median(v[full])) - float(np.median(v[key]))
                for v in per_patient.values()
                if v.get(full) and v.get(key)
            ]
            arr = np.asarray(deltas, dtype=float)
            if arr.size:
                summary["levels"][key]["delta_full_minus_this"] = {
                    "n_patients": int(arr.size),
                    "median": float(np.median(arr)),
                    "n_full_better": int((arr < 0).sum()),
                    "sign_test_p": _sign_test(arr),
                    "wilcoxon_p": _wilcoxon(arr),
                }
    return summary


def wrong_time_comparison(runs: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """H1 probe: is the state's *alignment in time* doing the work?"""

    out: dict[str, Any] = {"arm": arm, "endpoints": {}}
    for endpoint in ENDPOINTS:
        per_patient: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            if run["arm"] != arm or "wrong_time_state" not in run:
                continue
            correct = run["test"].get(endpoint)
            wrong = run["wrong_time_state"].get(endpoint)
            if correct is None or wrong is None:
                continue
            if math.isfinite(correct) and math.isfinite(wrong):
                per_patient[run["subject"]].append(float(correct) - float(wrong))
        deltas = np.array([float(np.median(v)) for v in per_patient.values()])
        if deltas.size:
            out["endpoints"][endpoint] = {
                "n_patients": int(deltas.size),
                "median_delta_correct_minus_wrong": float(np.median(deltas)),
                "n_correct_better": int((deltas < 0).sum()),
                "sign_test_p": _sign_test(deltas),
                "wilcoxon_p": _wilcoxon(deltas),
            }
    return out

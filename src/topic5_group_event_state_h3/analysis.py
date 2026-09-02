"""Aggregation: blocks -> patients -> cohort, in that order and no other.

Three rules are enforced mechanically here rather than remembered:

*Seeds are repeated fits, not sample size.*  Seeds collapse to one number per
patient before anything crosses a patient boundary.

*Blocks are the within-patient denominator.*  Only the pre-registered disjoint
blocks are averaged, and the count is carried alongside every number so a thin
patient cannot silently weigh as much as a long one.

*Pairing is by key, never by position.*  ``np.array(list(d.values()))`` aligns two
dictionaries by insertion order, which quietly compares patient A against patient
B.  Every contrast here starts from ``sorted(set(a) & set(b))`` and reports what
it dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ENDPOINTS = ("count", "mark")


@dataclass
class RunRecord:
    subject: str
    arm: str
    seed: int
    payload: dict
    blocks: Mapping[str, np.ndarray]


def load_runs(machine_dir: Path, block_dir: Path) -> list[RunRecord]:
    out: list[RunRecord] = []
    for path in sorted(Path(machine_dir).glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            continue
        block_path = Path(payload["block_file"])
        if not block_path.exists():
            block_path = Path(block_dir) / block_path.name
        if not block_path.exists():
            continue
        with np.load(block_path) as z:
            blocks = {k: z[k] for k in z.files}
        out.append(
            RunRecord(payload["subject"], payload["arm"], int(payload["seed"]), payload, blocks)
        )
    return out


def per_run_block_scores(
    record: RunRecord, horizon: int, split: str = "development_test"
) -> dict[str, np.ndarray]:
    """The per-block held-out log scores of one run at one horizon."""

    prefix = f"{split}__{horizon}"
    if f"{prefix}__count_logscore" not in record.blocks:
        return {}
    return {
        "anchor_time": record.blocks[f"{prefix}__anchor_time"],
        "segment": record.blocks[f"{prefix}__segment"],
        "anchor_id": record.blocks[f"{prefix}__anchor_id"],
        "count": record.blocks[f"{prefix}__count_logscore"],
        "mark": record.blocks[f"{prefix}__mark_logscore"],
        "mark_groups": record.blocks[f"{prefix}__mark_group_logscore"],
        "has_events": record.blocks[f"{prefix}__has_events"],
        "count_true": record.blocks[f"{prefix}__count_true"],
    }


def collapse_seeds(
    records: Sequence[RunRecord],
    horizon: int,
    split: str = "development_test",
    *,
    seeds: Sequence[int] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """(subject, arm) -> per-block score, median over seeds, blocks aligned by key.

    Seeds are matched block-for-block on ``(segment, anchor_id)``; a seed that
    produced a different block set for the same patient would mean the support
    changed under us, which raises rather than silently intersecting.
    """

    allowed = None if seeds is None else set(int(s) for s in seeds)
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for rec in records:
        if allowed is not None and rec.seed not in allowed:
            continue
        grouped.setdefault((rec.subject, rec.arm), []).append(rec)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, runs in sorted(grouped.items()):
        per_seed = [(r.seed, per_run_block_scores(r, horizon, split)) for r in runs]
        per_seed = [(s, d) for s, d in per_seed if d]
        if not per_seed:
            continue
        reference = per_seed[0][1]
        ref_key = list(zip(reference["segment"].tolist(), reference["anchor_id"].tolist()))
        for seed, scores in per_seed[1:]:
            other = list(zip(scores["segment"].tolist(), scores["anchor_id"].tolist()))
            if other != ref_key:
                raise ValueError(
                    f"{key}: seed {seed} scored a different block set; the support "
                    "must not depend on the seed"
                )
        stacked = {
            endpoint: np.median(
                np.stack([d[endpoint] for _s, d in per_seed], axis=0), axis=0
            )
            for endpoint in ENDPOINTS
        }
        stacked["mark_groups"] = np.median(
            np.stack([d["mark_groups"] for _s, d in per_seed], axis=0), axis=0
        )
        stacked["has_events"] = reference["has_events"]
        stacked["count_true"] = reference["count_true"]
        stacked["anchor_time"] = reference["anchor_time"]
        stacked["block_key"] = np.asarray(ref_key, dtype=np.int64)
        stacked["n_seeds"] = len(per_seed)
        stacked["n_blocks"] = int(reference["count"].size)
        out[key] = stacked
    return out


def patient_means(
    collapsed: Mapping[tuple[str, str], Mapping[str, Any]], endpoint: str
) -> dict[str, dict[str, float]]:
    """(arm) -> {subject: mean held-out log score over that patient's blocks}."""

    out: dict[str, dict[str, float]] = {}
    for (subject, arm), scores in collapsed.items():
        values = np.asarray(scores[endpoint], dtype=np.float64)
        if endpoint == "mark":
            has = np.asarray(scores["has_events"], dtype=bool)
            values = values[has]
        if values.size == 0:
            continue
        out.setdefault(arm, {})[subject] = float(np.mean(values))
    return out


def paired_by_subject(
    a: Mapping[str, float], b: Mapping[str, float]
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Align two per-subject maps by key, reporting what each side lacked."""

    shared = sorted(set(a) & set(b))
    missing = sorted((set(a) ^ set(b)))
    return (
        shared,
        np.asarray([a[s] for s in shared], dtype=np.float64),
        np.asarray([b[s] for s in shared], dtype=np.float64),
        missing,
    )


def sign_test(delta: np.ndarray) -> dict[str, float]:
    from scipy import stats

    finite = delta[np.isfinite(delta)]
    n_pos = int((finite > 0).sum())
    n_neg = int((finite < 0).sum())
    n = n_pos + n_neg
    p = float(stats.binomtest(n_pos, n, 0.5).pvalue) if n else float("nan")
    return {"n_positive": n_pos, "n_negative": n_neg, "n_nonzero": n, "p_sign": p}


def wilcoxon(delta: np.ndarray) -> float:
    from scipy import stats

    finite = delta[np.isfinite(delta) & (delta != 0)]
    if finite.size < 5:
        return float("nan")
    return float(stats.wilcoxon(finite).pvalue)


def bootstrap_median_ci(
    delta: np.ndarray, *, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float]:
    finite = delta[np.isfinite(delta)]
    if finite.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(n_boot, finite.size), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def contrast(
    a: Mapping[str, float], b: Mapping[str, float], label: str, *, seed: int = 0
) -> dict[str, Any]:
    """Paired ``a - b`` across patients, with the denominator stated."""

    subjects, va, vb, missing = paired_by_subject(a, b)
    delta = va - vb
    lo, hi = bootstrap_median_ci(delta, seed=seed)
    return {
        "contrast": label,
        "n_patients": len(subjects),
        "subjects": subjects,
        "subjects_dropped_for_missing_arm": missing,
        "median_delta": float(np.median(delta)) if delta.size else float("nan"),
        "mean_delta": float(np.mean(delta)) if delta.size else float("nan"),
        "median_ci95": [lo, hi],
        "p_wilcoxon": wilcoxon(delta),
        **sign_test(delta),
        "per_subject_delta": {s: float(d) for s, d in zip(subjects, delta)},
    }


def seeds_are_distinct_fits(records: Sequence[RunRecord]) -> dict[str, Any]:
    """Two seeds that produced identical numbers are one fit, not two."""

    duplicates: list[str] = []
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.subject, rec.arm), []).append(rec)
    for key, runs in grouped.items():
        signatures = set()
        for run in runs:
            history = run.payload.get("fit", {}).get("history", [])
            signature = json.dumps(
                [round(float(h.get("train_loss", float("nan"))), 8) for h in history[:3]]
            )
            if signature in signatures:
                duplicates.append(f"{key[0]}|{key[1]}|seed{run.seed}")
            signatures.add(signature)
    return {"n_duplicate_seed_fits": len(duplicates), "duplicates": duplicates}


def per_seed_patient_means(
    records: Sequence[RunRecord], horizon: int, endpoint: str, split: str = "development_test"
) -> dict[tuple[str, str], dict[int, float]]:
    """(subject, arm) -> {seed: mean held-out score}, before seeds are collapsed."""

    out: dict[tuple[str, str], dict[int, float]] = {}
    for rec in records:
        scores = per_run_block_scores(rec, horizon, split)
        if not scores:
            continue
        values = np.asarray(scores[endpoint], dtype=np.float64)
        if endpoint == "mark":
            values = values[np.asarray(scores["has_events"], dtype=bool)]
        if values.size == 0:
            continue
        out.setdefault((rec.subject, rec.arm), {})[rec.seed] = float(np.mean(values))
    return out


def seed_swap_null(
    records: Sequence[RunRecord],
    horizon: int,
    endpoint: str,
    *,
    split: str = "development_test",
    primary_seeds: Sequence[int] = (0, 1, 2),
    null_seeds: Sequence[int] = (3, 4, 5),
    seed: int = 0,
) -> dict[str, Any]:
    """A null contrast with exactly the shape *and width* of the real one.

    Same patients, same blocks, same three-seed median on each side -- but the two
    sides are two disjoint seed groups of the *same arm* rather than two arms.
    Whatever a refit is worth once aggregated the way the real contrast aggregates,
    this measures it.

    Matching the aggregation width matters: a single-seed difference is noisier
    than a difference of two three-seed medians, so using one as the floor for the
    other overstates the noise and buries real effects.  This project has made
    exactly that cross-width comparison before.

    A cohort effect that does not clear this floor is a refit-sized effect,
    whatever its p-value says -- the p-value tests against zero, and zero is not
    the relevant comparison.
    """

    left_all = collapse_seeds(records, horizon, split, seeds=primary_seeds)
    right_all = collapse_seeds(records, horizon, split, seeds=null_seeds)
    have_left = {k: v for k, v in left_all.items() if v["n_seeds"] == len(primary_seeds)}
    have_right = {k: v for k, v in right_all.items() if v["n_seeds"] == len(null_seeds)}
    shared = sorted(set(have_left) & set(have_right))
    if not shared:
        return {
            "status": "insufficient_seeds",
            "note": (
                f"needs {len(primary_seeds)} + {len(null_seeds)} disjoint seeds per "
                "(patient, arm) to build a floor at the contrast's own aggregation width"
            ),
        }

    def _mean(scores: Mapping[str, Any]) -> float:
        values = np.asarray(scores[endpoint], dtype=np.float64)
        if endpoint == "mark":
            values = values[np.asarray(scores["has_events"], dtype=bool)]
        return float(np.mean(values)) if values.size else float("nan")

    left = {f"{sub}|{arm}": _mean(have_left[(sub, arm)]) for sub, arm in shared}
    right = {f"{sub}|{arm}": _mean(have_right[(sub, arm)]) for sub, arm in shared}
    stats = contrast(left, right, "seed_swap_null_same_arm_same_aggregation", seed=seed)
    delta = np.asarray(list(stats["per_subject_delta"].values()), dtype=np.float64)
    stats["status"] = "ok"
    stats["primary_seeds"] = list(primary_seeds)
    stats["null_seeds"] = list(null_seeds)
    stats["median_absolute_refit_delta"] = float(np.median(np.abs(delta)))
    stats["p90_absolute_refit_delta"] = float(np.percentile(np.abs(delta), 90))
    # The 81 cells are not 81 independent draws: every cell reuses the *same* six
    # seeds, so whatever those particular initialisations are worth repeats across
    # all of them and shifts the null's centre.  The spread -- which is what the
    # floor is -- is unaffected, and the primary contrast is immune because both
    # of its arms use the same three seeds.  Recorded so the off-centre null is
    # not later mistaken for a procedural fault.
    fits: dict[int, list[float]] = {}
    per_cell: dict[tuple[str, str], dict[int, float]] = {}
    for rec in records:
        objective = rec.payload.get("fit", {}).get("inner_validation_objective")
        if objective is not None and np.isfinite(objective):
            per_cell.setdefault((rec.subject, rec.arm), {})[rec.seed] = float(objective)
    all_seeds = sorted(set(primary_seeds) | set(null_seeds))
    for by_seed in per_cell.values():
        if set(by_seed) != set(all_seeds):
            continue
        order = sorted(by_seed, key=lambda sd: by_seed[sd])
        for rank, sd in enumerate(order, start=1):
            fits.setdefault(sd, []).append(float(rank))
    stats["per_seed_mean_fit_rank"] = {
        str(sd): float(np.mean(values)) for sd, values in sorted(fits.items())
    }
    stats["shared_seed_caveat"] = (
        "the null's cells all reuse the same seeds, so its centre carries a common "
        "seed-draw offset; the floor is its spread, and the primary contrast uses "
        "the same three seeds on both arms and is unaffected"
    )
    stats["note"] = (
        "two disjoint three-seed medians of the SAME arm, aggregated exactly like a "
        "real contrast; an effect below this scale is a refit-sized effect"
    )
    return stats

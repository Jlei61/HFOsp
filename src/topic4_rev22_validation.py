"""rev22-DCI selection-blind validation helpers: fixed-budget recall and C2ST.

Pure functions. These are validation-only readouts (spec section 9); they must never
be imported by the training/selection runners.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RECALL_OK = "OK"
RECALL_LOW_YIELD = "NOT_ESTIMABLE_LOW_YIELD"
C2ST_NOT_ESTIMABLE = "C2ST_NOT_ESTIMABLE_GROUPS"
REFERENCE_SUPPORT_BUDGET_OK = "OK"
REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE = "REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE"


def freeze_reference_support_budget(returned_counts, *, expected_units: int,
                                    min_events: int) -> dict:
    """Freeze the recall budget without discarding a low-yield reference unit."""
    counts = [int(value) for value in returned_counts]
    if len(counts) != int(expected_units):
        return {
            "status": REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE,
            "n_cov": None,
            "counts": counts,
            "reason": "MISSING_REFERENCE_UNIT",
        }
    n_cov = int(min(counts))
    if n_cov < int(min_events):
        return {
            "status": REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE,
            "n_cov": n_cov,
            "counts": counts,
            "reason": "REFERENCE_UNIT_BELOW_MINIMUM_YIELD",
        }
    return {
        "status": REFERENCE_SUPPORT_BUDGET_OK,
        "n_cov": n_cov,
        "counts": counts,
        "reason": None,
    }


def paired_unit_bootstrap(left, right, *, draws: int, seed: int,
                          higher_is_better: bool = True) -> dict:
    """Paired topology-unit contrast with a percentile interval.

    Positive output always means that ``left`` is better than ``right``.
    Missing or non-finite pairs are not silently removed: the contrast is not
    estimable unless every supplied pair is finite.
    """
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right):
        raise ValueError("paired inputs must be one-dimensional and equal length")
    if len(left) < 2:
        return {"status": "NOT_ESTIMABLE_TOO_FEW_UNITS", "delta": None,
                "lo": None, "hi": None, "n": int(len(left))}
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return {"status": "NOT_ESTIMABLE_MISSING_UNIT", "delta": None,
                "lo": None, "hi": None, "n": int(len(left))}
    delta = left - right
    if not higher_is_better:
        delta = -delta
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, len(delta), size=(int(draws), len(delta)))
    boot = delta[index].mean(axis=1)
    return {
        "status": "OK",
        "delta": float(delta.mean()),
        "lo": float(np.quantile(boot, 0.05)),
        "hi": float(np.quantile(boot, 0.95)),
        "n": int(len(delta)),
        "draws": int(draws),
        "seed": int(seed),
        "positive_is_better": True,
    }


# --------------------------------------------------------------------------- #
# fixed-budget recall
# --------------------------------------------------------------------------- #
def calibrate_coverage_radius(training_z: np.ndarray, n_cov: int, *, draws: int, seed: int,
                              quantile: float = 0.95) -> dict:
    """r_cov = median_b q95_{q in Q_b} min_{s in S_b} d(q, s) over frozen training draws."""
    z = np.asarray(training_z, float)
    if n_cov < 1 or n_cov >= len(z):
        raise ValueError("n_cov must be at least 1 and smaller than the training set")
    rng = np.random.default_rng(int(seed))
    radii = []
    for _ in range(int(draws)):
        support = rng.choice(len(z), size=int(n_cov), replace=False)
        mask = np.ones(len(z), bool)
        mask[support] = False
        nearest, _ = cKDTree(z[support]).query(z[mask], k=1)
        radii.append(float(np.quantile(nearest, quantile)))
    radii = np.asarray(radii)
    return {"r_cov": float(np.median(radii)), "n_cov": int(n_cov), "draws": int(draws),
            "quantile": float(quantile), "radius_q05": float(np.quantile(radii, 0.05)),
            "radius_q95": float(np.quantile(radii, 0.95)), "seed": int(seed)}


def fixed_budget_recall(model_z: np.ndarray, query_z: np.ndarray, *, n_cov: int, r_cov: float,
                        subsamples: int = 200, seed: int = 0) -> dict:
    """Fraction of query events within r_cov of a model support set of exactly n_cov events."""
    model_z = np.asarray(model_z, float)
    query_z = np.asarray(query_z, float)
    n_model = int(len(model_z))
    if n_model < 1:
        return {"status": RECALL_LOW_YIELD, "recall": None, "recall_all_events": None,
                "n_model": 0, "n_cov": int(n_cov)}
    nearest_all, _ = cKDTree(model_z).query(query_z, k=1)
    recall_all = float(np.mean(nearest_all <= r_cov))
    if n_model < n_cov:
        return {"status": RECALL_LOW_YIELD, "recall": None, "recall_all_events": recall_all,
                "n_model": n_model, "n_cov": int(n_cov)}
    if n_model == n_cov:
        return {"status": RECALL_OK, "recall": recall_all, "recall_q05": recall_all, "recall_q95": recall_all,
                "recall_all_events": recall_all, "n_model": n_model, "n_cov": int(n_cov), "subsamples": 1}
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(subsamples)):
        support = rng.choice(n_model, size=int(n_cov), replace=False)
        nearest, _ = cKDTree(model_z[support]).query(query_z, k=1)
        values.append(float(np.mean(nearest <= r_cov)))
    values = np.asarray(values)
    return {"status": RECALL_OK, "recall": float(values.mean()),
            "recall_q05": float(np.quantile(values, 0.05)), "recall_q95": float(np.quantile(values, 0.95)),
            "recall_all_events": recall_all, "n_model": n_model, "n_cov": int(n_cov),
            "subsamples": int(subsamples)}


def _sample_preserving_groups(indices: np.ndarray, groups: np.ndarray, n: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Sample without replacement while retaining at least one event per group."""
    indices = np.asarray(indices, int)
    unique = np.unique(groups[indices])
    if n < len(unique):
        raise ValueError("sample budget is smaller than the group count")
    mandatory = np.asarray([rng.choice(indices[groups[indices] == group]) for group in unique], int)
    if n == len(mandatory):
        return mandatory
    remaining = np.setdiff1d(indices, mandatory, assume_unique=False)
    extra = rng.choice(remaining, size=n - len(mandatory), replace=False)
    return np.concatenate([mandatory, extra])


def _paired_group_folds(patient_groups: np.ndarray, model_groups: np.ndarray, *,
                        n_splits: int, rng: np.random.Generator) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Create folds containing paired patient/model groups and return pair ids.

    Equal numbers of groups from each class are paired before assignment to folds. Every
    train and test partition therefore contains both classes. The pair ids also define a
    group-level permutation null: swapping the two labels inside a pair preserves the
    grouped dependence and the number of groups per class.
    """
    p_unique = np.unique(patient_groups)
    m_unique = np.unique(model_groups)
    if len(p_unique) != len(m_unique):
        raise ValueError("paired-group folds require equal class-specific group counts")
    n_pairs = min(len(p_unique), len(m_unique))
    if n_pairs < 2:
        return [], np.zeros(len(patient_groups) + len(model_groups), int)
    p_selected = p_unique.copy()
    m_selected = m_unique.copy()
    rng.shuffle(p_selected)
    rng.shuffle(m_selected)
    p_pair = {group: pair for pair, group in enumerate(p_selected)}
    m_pair = {group: pair for pair, group in enumerate(m_selected)}
    pair_ids = np.asarray(
        [p_pair[group] for group in patient_groups]
        + [m_pair[group] for group in model_groups], int,
    )
    fold_ids = np.arange(n_pairs) % min(int(n_splits), n_pairs)
    rng.shuffle(fold_ids)
    splits = []
    for fold in range(int(fold_ids.max()) + 1):
        test = np.flatnonzero(np.isin(pair_ids, np.flatnonzero(fold_ids == fold)))
        train = np.setdiff1d(np.arange(len(pair_ids)), test, assume_unique=True)
        splits.append((train, test))
    return splits, pair_ids


# --------------------------------------------------------------------------- #
# classifier two-sample test
# --------------------------------------------------------------------------- #
def classifier_two_sample_auc(patient_z: np.ndarray, patient_groups: np.ndarray,
                              model_z: np.ndarray, model_groups: np.ndarray, *, seed: int,
                              resamples: int = 20, n_splits: int = 5, permutations: int = 20,
                              regularization_c: float = 1.0) -> dict:
    """Class-balanced, group-separated linear C2ST in the frozen embedding.

    Patient recording blocks and model topology seeds are the groups; a group never crosses
    folds. The larger class is subsampled to the smaller class size in every resample.
    """
    patient_z = np.asarray(patient_z, float)
    model_z = np.asarray(model_z, float)
    patient_groups = np.asarray(patient_groups)
    model_groups = np.asarray(model_groups)
    if len(np.unique(patient_groups)) < 2 or len(np.unique(model_groups)) < 2 or len(model_z) < 4:
        return {"status": C2ST_NOT_ESTIMABLE, "auc": None}
    rng = np.random.default_rng(int(seed))
    n_group = min(len(np.unique(patient_groups)), len(np.unique(model_groups)))
    if n_group < 2:
        return {"status": C2ST_NOT_ESTIMABLE, "auc": None}

    sample_sizes = []

    def one(labels_permuted: bool) -> float | None:
        p_selected = rng.choice(np.unique(patient_groups), size=n_group, replace=False)
        m_selected = rng.choice(np.unique(model_groups), size=n_group, replace=False)
        p_pool = np.flatnonzero(np.isin(patient_groups, p_selected))
        m_pool = np.flatnonzero(np.isin(model_groups, m_selected))
        n = min(len(p_pool), len(m_pool))
        if n < max(4, n_group):
            return None
        p_idx = _sample_preserving_groups(p_pool, patient_groups, n, rng)
        m_idx = _sample_preserving_groups(m_pool, model_groups, n, rng)
        x = np.vstack([patient_z[p_idx], model_z[m_idx]])
        y = np.concatenate([np.zeros(len(p_idx), int), np.ones(len(m_idx), int)])
        p_group = patient_groups[p_idx]
        m_group = model_groups[m_idx]
        splits, pair_ids = _paired_group_folds(p_group, m_group, n_splits=n_splits, rng=rng)
        if len(splits) < 2:
            return None
        if labels_permuted:
            swap = rng.integers(0, 2, size=int(pair_ids.max()) + 1).astype(bool)
            y = np.where(swap[pair_ids], 1 - y, y)
        scores = np.full(len(y), np.nan)
        for train, test in splits:
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                return None
            clf = LogisticRegression(C=float(regularization_c), max_iter=2000)
            clf.fit(x[train], y[train])
            scores[test] = clf.decision_function(x[test])
        ok = np.isfinite(scores)
        if len(np.unique(y[ok])) < 2:
            return None
        if not labels_permuted:
            sample_sizes.append(int(n))
        return float(roc_auc_score(y[ok], scores[ok]))

    aucs = [v for v in (one(False) for _ in range(int(resamples))) if v is not None]
    perms = [v for v in (one(True) for _ in range(int(permutations))) if v is not None]
    if not aucs:
        return {"status": C2ST_NOT_ESTIMABLE, "auc": None}
    aucs = np.asarray(aucs)
    return {
        "status": "OK",
        "auc": float(aucs.mean()),
        "auc_oriented": float(0.5 + abs(aucs.mean() - 0.5)),
        "auc_q05": float(np.quantile(aucs, 0.05)), "auc_q95": float(np.quantile(aucs, 0.95)),
        "separability": float(2.0 * abs(aucs.mean() - 0.5)),
        "permutation_auc_median": float(np.median(perms)) if perms else None,
        "permutation_auc_q95": float(np.quantile(perms, 0.95)) if perms else None,
        "n_per_class": int(np.median(sample_sizes)), "resamples": int(len(aucs)),
    }

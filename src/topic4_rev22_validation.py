"""rev22-DCI selection-blind validation helpers: fixed-budget recall and C2ST.

Pure functions. These are validation-only readouts (spec section 9); they must never
be imported by the training/selection runners.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

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
    n = min(len(patient_z), len(model_z))
    p_codes = {g: i for i, g in enumerate(np.unique(patient_groups))}
    m_codes = {g: 10_000_000 + i for i, g in enumerate(np.unique(model_groups))}

    def one(labels_permuted: bool) -> float | None:
        p_idx = rng.choice(len(patient_z), size=n, replace=False) if len(patient_z) > n else np.arange(len(patient_z))
        m_idx = rng.choice(len(model_z), size=n, replace=False) if len(model_z) > n else np.arange(len(model_z))
        x = np.vstack([patient_z[p_idx], model_z[m_idx]])
        y = np.concatenate([np.zeros(len(p_idx), int), np.ones(len(m_idx), int)])
        groups = np.concatenate([[p_codes[g] for g in patient_groups[p_idx]],
                                 [m_codes[g] for g in model_groups[m_idx]]])
        if labels_permuted:
            y = rng.permutation(y)
        splits = min(int(n_splits), len(np.unique(groups)))
        if splits < 2:
            return None
        scores = np.full(len(y), np.nan)
        for train, test in GroupKFold(n_splits=splits).split(x, y, groups):
            if len(np.unique(y[train])) < 2:
                continue
            clf = LogisticRegression(C=float(regularization_c), max_iter=2000)
            clf.fit(x[train], y[train])
            scores[test] = clf.decision_function(x[test])
        ok = np.isfinite(scores)
        if len(np.unique(y[ok])) < 2:
            return None
        return float(roc_auc_score(y[ok], scores[ok]))

    aucs = [v for v in (one(False) for _ in range(int(resamples))) if v is not None]
    perms = [v for v in (one(True) for _ in range(int(permutations))) if v is not None]
    if not aucs:
        return {"status": C2ST_NOT_ESTIMABLE, "auc": None}
    aucs = np.asarray(aucs)
    return {
        "status": "OK",
        "auc": float(aucs.mean()),
        "auc_q05": float(np.quantile(aucs, 0.05)), "auc_q95": float(np.quantile(aucs, 0.95)),
        "separability": float(2.0 * abs(aucs.mean() - 0.5)),
        "permutation_auc_median": float(np.median(perms)) if perms else None,
        "permutation_auc_q95": float(np.quantile(perms, 0.95)) if perms else None,
        "n_per_class": int(n), "resamples": int(len(aucs)),
    }

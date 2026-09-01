"""Patient-trained direction labels in the shaft-aware event representation."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.topic4_shaft_aware import (
    build_event_features,
    transform_patient_embedding,
)


def _embedded_events(values, groups, embedding):
    features = build_event_features(np.asarray(values, dtype=float), groups)[
        "features"
    ]
    return transform_patient_embedding(features, embedding)


def _mahalanobis_squared(values, center, precision):
    delta = np.asarray(values, dtype=float) - np.asarray(center, dtype=float)
    return np.einsum("ni,ij,nj->n", delta, precision, delta)


def fit_direction_classifier(
    values,
    labels,
    block_ids,
    *,
    groups,
    embedding,
    n_splits=6,
    regularization_c=1.0,
    ood_quantile=0.99,
):
    """Fit old A/B direction labels without using model candidates.

    Cross-validation keeps recording blocks intact.  Class-conditional
    shrinkage distances provide an OOD diagnostic, but OOD events are never
    deleted before the all-event shaft objective is evaluated.
    """
    labels = np.asarray(labels, dtype=int)
    blocks = np.asarray(block_ids)
    z = _embedded_events(values, groups, embedding)
    if labels.shape != (len(z),) or blocks.shape != (len(z),):
        raise ValueError("direction labels and recording blocks must align")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("direction classifier requires labels 0 and 1")
    folds = []
    held_label = np.empty(len(labels), dtype=int)
    held_probability = np.empty(len(labels), dtype=float)
    splitter = GroupKFold(n_splits=min(int(n_splits), len(np.unique(blocks))))
    for fold, (train, test) in enumerate(splitter.split(z, labels, blocks)):
        model = LogisticRegression(
            C=float(regularization_c), class_weight="balanced",
            max_iter=2000, solver="lbfgs",
        ).fit(z[train], labels[train])
        predicted = model.predict(z[test])
        probability = model.predict_proba(z[test])[:, 1]
        held_label[test] = predicted
        held_probability[test] = probability
        folds.append({
            "fold": int(fold),
            "n_events": int(len(test)),
            "n_blocks": int(len(np.unique(blocks[test]))),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels[test], predicted)
            ),
            "roc_auc": float(roc_auc_score(labels[test], probability)),
        })

    model = LogisticRegression(
        C=float(regularization_c), class_weight="balanced",
        max_iter=2000, solver="lbfgs",
    ).fit(z, labels)
    centers, precisions, thresholds = [], [], []
    for mode in (0, 1):
        selected = z[labels == mode]
        covariance = LedoitWolf().fit(selected)
        distance = _mahalanobis_squared(
            selected, covariance.location_, covariance.precision_,
        )
        centers.append(covariance.location_)
        precisions.append(covariance.precision_)
        thresholds.append(float(np.quantile(distance, float(ood_quantile))))
    return {
        "coef": np.asarray(model.coef_[0], dtype=float),
        "intercept": float(model.intercept_[0]),
        "class_centers": np.asarray(centers, dtype=float),
        "class_precisions": np.asarray(precisions, dtype=float),
        "ood_distance_thresholds": np.asarray(thresholds, dtype=float),
        "ood_quantile": float(ood_quantile),
        "regularization_c": float(regularization_c),
        "n_train": int(len(labels)),
        "folds": folds,
        "heldout_balanced_accuracy": float(
            balanced_accuracy_score(labels, held_label)
        ),
        "heldout_roc_auc": float(roc_auc_score(labels, held_probability)),
        "training_balanced_accuracy": float(
            balanced_accuracy_score(labels, model.predict(z))
        ),
        "training_roc_auc": float(
            roc_auc_score(labels, model.predict_proba(z)[:, 1])
        ),
    }


def assign_direction_modes(values, *, groups, embedding, classifier: Mapping):
    """Assign every event; return OOD as a diagnostic rather than a filter."""
    z = _embedded_events(values, groups, embedding)
    coef = np.asarray(classifier["coef"], dtype=float)
    if z.ndim != 2 or z.shape[1] != len(coef):
        raise ValueError("classifier and shaft-aware embedding disagree")
    logit = z @ coef + float(classifier["intercept"])
    probability_b = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
    labels = (probability_b >= 0.5).astype(int)
    centers = np.asarray(classifier["class_centers"], dtype=float)
    precisions = np.asarray(classifier["class_precisions"], dtype=float)
    thresholds = np.asarray(
        classifier["ood_distance_thresholds"], dtype=float,
    )
    distance = np.asarray([
        _mahalanobis_squared(
            z[index:index + 1], centers[mode], precisions[mode],
        )[0]
        for index, mode in enumerate(labels)
    ])
    return {
        "labels": labels,
        "probability_B": probability_b,
        "embedding": z,
        "ood_distance": distance,
        "ood": distance > thresholds[labels],
    }


def all_event_shaft_participation(values, groups):
    """Count ICL-only, joint, SCL-only, and unreadable events."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("event values must be two-dimensional")
    icl = np.isfinite(values[:, groups["ICL"]]).any(axis=1)
    scl = np.isfinite(values[:, groups["SCL"]]).any(axis=1)
    count = max(1, len(values))
    return {
        "n_events": int(len(values)),
        "n_icl_only": int(np.sum(icl & ~scl)),
        "n_joint": int(np.sum(icl & scl)),
        "n_scl_only": int(np.sum(~icl & scl)),
        "n_unreadable": int(np.sum(~icl & ~scl)),
        "joint_fraction": float(np.sum(icl & scl) / count),
        "icl_participation_fraction": float(np.sum(icl) / count),
        "scl_participation_fraction": float(np.sum(scl) / count),
    }


def mode_conditioned_joint_support(values, labels, ood, groups):
    """Keep direction, joint-shaft participation, and patient support coupled."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    ood = np.asarray(ood, dtype=bool)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("event values and direction labels must align")
    if ood.shape != (len(values),):
        raise ValueError("OOD mask and event values must align")
    icl = np.isfinite(values[:, groups["ICL"]]).any(axis=1)
    scl = np.isfinite(values[:, groups["SCL"]]).any(axis=1)
    joint = icl & scl
    output = {}
    for mode, name in ((0, "A"), (1, "B")):
        selected = labels == mode
        n_mode = int(np.sum(selected))
        denominator = max(1, n_mode)
        n_joint = int(np.sum(selected & joint))
        n_in_distribution = int(np.sum(selected & ~ood))
        n_joint_in_distribution = int(np.sum(selected & joint & ~ood))
        output[name] = {
            "n_events": n_mode,
            "n_joint": n_joint,
            "n_in_distribution": n_in_distribution,
            "n_joint_in_distribution": n_joint_in_distribution,
            "joint_fraction": float(n_joint / denominator),
            "in_distribution_fraction": float(n_in_distribution / denominator),
            "joint_in_distribution_fraction": float(
                n_joint_in_distribution / denominator
            ),
        }
    return output

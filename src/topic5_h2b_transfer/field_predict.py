"""Predict a held-out seizure's early ictal field from a frozen interictal state.

Sample size drives the design: a patient contributes a median of ~5 TRAIN
seizures, so a state -> field regression cannot be fitted. Instead the patient's
own TRAIN fields are re-weighted by the similarity between the frozen state at
``onset - lead`` and the state before each TRAIN seizure.

Both arms of the comparison therefore draw on *identical* TRAIN fields:

    baseline    uniform weights  -> the patient-average field
    state arm   state-similarity weights

so the increment isolates exactly one thing -- whether the interictal state
says *which* of this patient's seizure patterns is coming.
"""

from __future__ import annotations

import numpy as np


def state_similarity_weights(
    query_state: np.ndarray,
    reference_states: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Softmax over cosine similarity between the query and each TRAIN state.

    Cosine, not Euclidean, so overall state magnitude (which tracks event rate
    and recording conditions) cannot drive the weighting on its own.
    ``temperature -> 0`` approaches nearest neighbour; ``-> inf`` approaches the
    uniform baseline, which makes the baseline a strict special case.
    """

    q = np.asarray(query_state, float).ravel()
    R = np.atleast_2d(np.asarray(reference_states, float))
    if R.shape[0] == 0:
        return np.empty(0)
    qn = np.linalg.norm(q)
    Rn = np.linalg.norm(R, axis=1)
    ok = np.isfinite(Rn) & (Rn > 0)
    sims = np.zeros(R.shape[0])
    if qn > 0 and ok.any():
        sims[ok] = (R[ok] @ q) / (Rn[ok] * qn)
    if not np.isfinite(temperature):
        return np.full(R.shape[0], 1.0 / R.shape[0])  # uniform: the baseline itself
    z = sims / max(float(temperature), 1e-12)
    z -= z.max()
    w = np.exp(z)
    total = w.sum()
    return w / total if total > 0 else np.full(R.shape[0], 1.0 / R.shape[0])


def predict_field(train_fields: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Weighted combination of TRAIN fields; uniform weights = patient average.

    Contacts missing from some TRAIN fields are averaged over the fields that
    do have them, rather than dragging the whole contact to NaN.
    """

    F = np.asarray(train_fields, float)
    if F.ndim != 2:
        raise ValueError("train_fields must be (n_train, n_contacts)")
    n, d = F.shape
    if n == 0:
        return np.full(d, np.nan)
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, float).ravel()
        if w.size != n:
            raise ValueError(f"weights must have one entry per TRAIN field, got {w.size} for {n}")
    valid = np.isfinite(F)
    W = np.where(valid, w[:, None], 0.0)
    denom = W.sum(axis=0)
    num = np.where(valid, F, 0.0) * W
    out = np.divide(num.sum(axis=0), denom, out=np.full(d, np.nan), where=denom > 0)
    return out

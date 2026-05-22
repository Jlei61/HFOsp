"""Smoke tests: --masked-features flag re-routes path globals to
`results/topic4_attractor_masked/...` in the 5 Topic 4 attractor scripts
and `mask_phantom=` is plumbed through to `build_rank_feature_matrix`.

These are minimal plumbing tests. Numerical contract for the masked rank
feature space (event-median impute) lives in tests/test_lagpat_rank_audit.py.
What we verify here is:
  1. `build_rank_feature_matrix(..., mask_phantom=True)` calls
     `build_masked_kmeans_features` and yields a feature matrix that differs
     from the legacy `mask_phantom=False` path on a synthetic fixture.
  2. Each of the 5 scripts' `_apply_masked_paths()` swaps OUT_DIR (and the
     load-bearing PR2_PER_SUBJECT_DIR / cohort summary globals) to
     `topic4_attractor_masked`.
  3. The `mask_phantom` kwarg is plumbed from `_run_one` / `_augment_one`
     callers through to `build_rank_feature_matrix` (via monkeypatch capture).

Per Topic 0 phantom rerun roadmap §5h.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _fresh_import(modname: str):
    """Reload module so prior in-process global mutations don't bleed in."""
    if modname in sys.modules:
        return importlib.reload(sys.modules[modname])
    return importlib.import_module(modname)


def _is_masked(p) -> bool:
    return "topic4_attractor_masked" in str(p)


def _is_legacy(p) -> bool:
    s = str(p)
    return "topic4_attractor" in s and not _is_masked(p)


def _is_pr2_masked(p) -> bool:
    return "interictal_propagation_masked" in str(p)


def _is_pr2_legacy(p) -> bool:
    s = str(p)
    return ("interictal_propagation" in s
            and "interictal_propagation_masked" not in s)


# ---------------------------------------------------------------------------
# 1. build_rank_feature_matrix: mask_phantom flag changes feature matrix
# ---------------------------------------------------------------------------


def test_build_rank_feature_matrix_mask_phantom_differs_from_legacy():
    """With phantom-polluted ranks on non-participating channels, mask_phantom
    must produce a different feature matrix from the legacy NaN->0 path."""
    from src.topic4_attractor_diagnostics import build_rank_feature_matrix

    rng = np.random.default_rng(0)
    n_ch = 10
    n_ev = 50
    # All events participate >= 6 channels so they survive the eligibility gate
    bools = np.zeros((n_ch, n_ev), dtype=bool)
    for e in range(n_ev):
        participating = rng.choice(n_ch, size=7, replace=False)
        bools[participating, e] = True
    # Ranks: argsort-of-argsort over noise; phantom channels still get finite
    # int ranks (this is the bug we're fixing).
    noise = rng.standard_normal((n_ch, n_ev))
    ranks = np.argsort(np.argsort(noise, axis=0), axis=0).astype(float)

    X_legacy, idx_legacy = build_rank_feature_matrix(
        ranks, bools, min_participating=6, mask_phantom=False
    )
    X_masked, idx_masked = build_rank_feature_matrix(
        ranks, bools, min_participating=6, mask_phantom=True
    )

    # Same event subset (eligibility unaffected by impute strategy)
    np.testing.assert_array_equal(idx_legacy, idx_masked)
    assert X_legacy.shape == X_masked.shape

    # But feature values differ — legacy has phantom ranks; masked has 0.5
    # at non-participating cells.
    assert not np.allclose(X_legacy, X_masked), (
        "mask_phantom=True must change the feature matrix on phantom-polluted "
        "fixtures"
    )

    # Non-participating cells in masked path = 0.5 (event_median impute)
    bools_subset = bools[:, idx_masked].T  # (n_ev, n_ch)
    np.testing.assert_array_almost_equal(
        X_masked[~bools_subset], 0.5,
        err_msg="masked non-participating cells must be 0.5 (event_median)"
    )


def test_build_rank_feature_matrix_default_is_masked():
    """`mask_phantom` defaults to True since Topic 0 §3.1 Phase 0 closure (2026-05-22).

    Pre-2026-05-22 the default was False (legacy phantom-contaminated path)
    for back-compat during broad re-derivation; closure flipped it to True.
    Explicit False still works (legacy path preserved for byte-level
    reproducibility) but emits DeprecationWarning — see
    test_build_rank_feature_matrix_explicit_false_emits_deprecation_warning.
    """
    import warnings
    from src.topic4_attractor_diagnostics import build_rank_feature_matrix

    rng = np.random.default_rng(1)
    n_ch, n_ev = 8, 20
    bools = np.ones((n_ch, n_ev), dtype=bool)  # all participate
    ranks = rng.standard_normal((n_ch, n_ev))

    X_default, _ = build_rank_feature_matrix(ranks, bools, min_participating=6)
    X_explicit_true, _ = build_rank_feature_matrix(
        ranks, bools, min_participating=6, mask_phantom=True
    )
    np.testing.assert_array_equal(X_default, X_explicit_true)


def test_build_rank_feature_matrix_explicit_false_emits_deprecation_warning():
    """Topic 0 §3.1 Phase 0 closure (2026-05-22): explicit `mask_phantom=False`
    is deprecated and must emit DeprecationWarning. The legacy path itself
    still works (for byte-level reproducibility of paper-locked numbers)."""
    import warnings
    from src.topic4_attractor_diagnostics import build_rank_feature_matrix

    rng = np.random.default_rng(2)
    n_ch, n_ev = 8, 20
    bools = np.ones((n_ch, n_ev), dtype=bool)
    ranks = rng.standard_normal((n_ch, n_ev))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_rank_feature_matrix(ranks, bools, min_participating=6, mask_phantom=False)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) >= 1, (
            "Expected DeprecationWarning when mask_phantom=False; "
            f"got categories {[w.category.__name__ for w in caught]}"
        )
        msg = str(deprecations[0].message)
        assert "phantom" in msg.lower() or "deprecated" in msg.lower(), (
            f"DeprecationWarning should mention phantom/deprecated: {msg!r}"
        )


# ---------------------------------------------------------------------------
# 2. Path-swap tests for the 5 scripts
# ---------------------------------------------------------------------------


def test_audit_topic4_step0_apply_masked_paths_swaps_globals():
    m = _fresh_import("audit_topic4_step0")

    assert _is_pr2_legacy(m.PR2_PER_SUBJECT_DIR)
    assert _is_legacy(m.OUT_DIR)
    assert _is_legacy(m.OUT_CSV)
    assert _is_legacy(m.OUT_SUMMARY)

    m._apply_masked_paths()

    assert _is_pr2_masked(m.PR2_PER_SUBJECT_DIR), m.PR2_PER_SUBJECT_DIR
    assert _is_masked(m.OUT_DIR), m.OUT_DIR
    assert _is_masked(m.OUT_CSV), m.OUT_CSV
    assert _is_masked(m.OUT_SUMMARY), m.OUT_SUMMARY


def test_run_attractor_step1_apply_masked_paths_swaps_globals():
    m = _fresh_import("run_attractor_step1")

    assert _is_pr2_legacy(m.PR2_PER_SUBJECT_DIR)
    assert _is_legacy(m.OUT_DIR)
    assert _is_legacy(m.PER_SUBJECT_DIR)
    assert _is_legacy(m.COHORT_CSV)
    assert _is_legacy(m.AUDIT_CSV)

    m._apply_masked_paths()

    assert _is_pr2_masked(m.PR2_PER_SUBJECT_DIR), m.PR2_PER_SUBJECT_DIR
    assert _is_masked(m.OUT_DIR), m.OUT_DIR
    assert _is_masked(m.PER_SUBJECT_DIR), m.PER_SUBJECT_DIR
    assert _is_masked(m.COHORT_CSV), m.COHORT_CSV
    assert _is_masked(m.AUDIT_CSV), m.AUDIT_CSV


def test_run_attractor_step1_sensitivity_apply_masked_paths_swaps_globals():
    m = _fresh_import("run_attractor_step1_sensitivity")

    assert _is_pr2_legacy(m.PR2_PER_SUBJECT_DIR)
    assert _is_legacy(m.OUT_DIR)
    assert _is_legacy(m.PER_SUBJECT_DIR)
    assert _is_legacy(m.SENS_CSV)
    assert _is_legacy(m.SENS_MD)

    m._apply_masked_paths()

    assert _is_pr2_masked(m.PR2_PER_SUBJECT_DIR), m.PR2_PER_SUBJECT_DIR
    assert _is_masked(m.OUT_DIR), m.OUT_DIR
    assert _is_masked(m.PER_SUBJECT_DIR), m.PER_SUBJECT_DIR
    assert _is_masked(m.SENS_CSV), m.SENS_CSV
    assert _is_masked(m.SENS_MD), m.SENS_MD


def test_augment_attractor_step1_kmeans_s_apply_masked_paths_swaps_globals():
    m = _fresh_import("augment_attractor_step1_kmeans_s")

    assert _is_pr2_legacy(m.PR2_PER_SUBJECT_DIR)
    assert _is_legacy(m.OUT_DIR)
    assert _is_legacy(m.PER_SUBJECT_DIR)
    assert _is_legacy(m.COHORT_CSV)

    m._apply_masked_paths()

    assert _is_pr2_masked(m.PR2_PER_SUBJECT_DIR), m.PR2_PER_SUBJECT_DIR
    assert _is_masked(m.OUT_DIR), m.OUT_DIR
    assert _is_masked(m.PER_SUBJECT_DIR), m.PER_SUBJECT_DIR
    assert _is_masked(m.COHORT_CSV), m.COHORT_CSV


def test_summarize_attractor_step1_apply_masked_paths_swaps_globals():
    m = _fresh_import("summarize_attractor_step1")

    assert _is_legacy(m.OUT_DIR)
    assert _is_legacy(m.PER_SUBJECT_DIR)
    assert _is_legacy(m.SUMMARY_MD)

    m._apply_masked_paths()

    assert _is_masked(m.OUT_DIR), m.OUT_DIR
    assert _is_masked(m.PER_SUBJECT_DIR), m.PER_SUBJECT_DIR
    assert _is_masked(m.SUMMARY_MD), m.SUMMARY_MD


# ---------------------------------------------------------------------------
# 3. mask_phantom plumbing: runner -> build_rank_feature_matrix
# ---------------------------------------------------------------------------


def test_run_attractor_step1_run_one_plumbs_mask_phantom(monkeypatch, tmp_path):
    """Verify `_run_one(..., mask_phantom=True)` forwards through to
    `build_rank_feature_matrix`. We monkeypatch the symbol that
    `run_step1_subject` (re-exported from `src.topic4_attractor_diagnostics`)
    uses to read features, then trigger one call.
    """
    import json as _json
    m = _fresh_import("run_attractor_step1")

    # Capture mask_phantom that arrives at build_rank_feature_matrix.
    captured = {}

    def fake_build(ranks, bools, *, min_participating, mask_phantom=False):
        captured["mask_phantom"] = bool(mask_phantom)
        # Return a tiny but valid feature matrix and event index.
        n_ev = 200
        return np.zeros((n_ev, ranks.shape[0]), dtype=float), np.arange(n_ev, dtype=int)

    # `run_step1_subject` calls build_rank_feature_matrix through
    # `src.topic4_attractor_diagnostics` module-level symbol.
    import src.topic4_attractor_diagnostics as td
    monkeypatch.setattr(td, "build_rank_feature_matrix", fake_build)

    # Stub the loader so we don't need a real subject dir.
    fake_subj_dir = tmp_path / "subj"
    fake_subj_dir.mkdir()
    monkeypatch.setattr(m, "_subject_dir", lambda dataset, subject: fake_subj_dir)
    n_ch = 10
    n_ev_raw = 200
    fake_loaded = {
        "ranks": np.tile(np.arange(n_ch).reshape(-1, 1).astype(float), (1, n_ev_raw)),
        "bools": np.ones((n_ch, n_ev_raw), dtype=bool),
        "channel_names": [f"E{i}" for i in range(n_ch)],
    }
    monkeypatch.setattr(m, "load_subject_propagation_events", lambda subj_dir: fake_loaded)

    # Stub the PR-2 JSON read (no real per_subject file in tmp_path).
    monkeypatch.setattr(
        m, "_read_pr2_labels",
        lambda json_path: (
            np.array([0, 1] * 100, dtype=int),  # 200 labels = matches valid_events at min_part=3
            [list(range(n_ch)), list(range(n_ch - 1, -1, -1))],
            [f"E{i}" for i in range(n_ch)],
            2, 200,
        ),
    )

    result = m._run_one("yuquan_x", "yuquan", "x", mask_phantom=True)
    assert captured.get("mask_phantom") is True, captured
    assert result.get("mask_phantom") is True


def test_augment_attractor_step1_augment_one_plumbs_mask_phantom(
    monkeypatch, tmp_path
):
    """`_augment_one(..., mask_phantom=True)` must forward to
    `build_rank_feature_matrix` via the script module's imported symbol."""
    import json as _json
    m = _fresh_import("augment_attractor_step1_kmeans_s")

    captured = {}

    def fake_build(ranks, bools, *, min_participating, mask_phantom=False):
        captured["mask_phantom"] = bool(mask_phantom)
        n_ev = 10
        return np.zeros((n_ev, ranks.shape[0]), dtype=float), np.arange(n_ev, dtype=int)

    monkeypatch.setattr(m, "build_rank_feature_matrix", fake_build)

    # Need a per-subject JSON that passes the early skip / excluded guards
    per = tmp_path / "yuquan_x.json"
    per.write_text(_json.dumps({
        "sid": "yuquan_x",
        "dataset": "yuquan",
        "subject": "x",
        "n_in_cluster": [5, 5],
    }))

    fake_subj_dir = tmp_path / "subj"
    fake_subj_dir.mkdir()
    monkeypatch.setattr(m, "_subject_dir", lambda dataset, subject: fake_subj_dir)
    # Stub PR-2 JSON loader to return a labels array length-matched to
    # `_valid_event_indices(bools, min_participating=3)` output below.
    monkeypatch.setattr(m, "_load_pr2_labels",
                        lambda jp: (np.array([0, 1] * 50, dtype=int), 2))

    n_ch = 10
    n_ev = 100
    fake_loaded = {
        "ranks": np.tile(np.arange(n_ch).reshape(-1, 1).astype(float), (1, n_ev)),
        "bools": np.ones((n_ch, n_ev), dtype=bool),
        "block_ids": np.zeros(n_ev, dtype=int),
        "channel_names": [f"E{i}" for i in range(n_ch)],
    }
    monkeypatch.setattr(m, "load_subject_propagation_events",
                        lambda subj_dir: fake_loaded)
    # `_valid_event_indices` at default min_participating=3 over all-true bools
    # returns indices 0..n_ev-1, matching the 100 labels we stubbed.

    out = m._augment_one(per, mask_phantom=True)
    assert captured.get("mask_phantom") is True, captured


def test_run_attractor_step1_sensitivity_run_one_plumbs_mask_phantom(
    monkeypatch, tmp_path
):
    """Same plumbing check for the sensitivity runner."""
    import json as _json
    m = _fresh_import("run_attractor_step1_sensitivity")

    captured = {}

    def fake_build(ranks, bools, *, min_participating, mask_phantom=False):
        captured["mask_phantom"] = bool(mask_phantom)
        n_ev = 100
        return np.zeros((n_ev, ranks.shape[0]), dtype=float), np.arange(n_ev, dtype=int)

    monkeypatch.setattr(m, "build_rank_feature_matrix", fake_build)

    fake_subj_dir = tmp_path / "subj"
    fake_subj_dir.mkdir()
    monkeypatch.setattr(m, "_subject_dir", lambda dataset, subject: fake_subj_dir)
    n_ch = 10
    n_ev = 100
    fake_loaded = {
        "ranks": np.tile(np.arange(n_ch).reshape(-1, 1).astype(float), (1, n_ev)),
        "bools": np.ones((n_ch, n_ev), dtype=bool),
        "channel_names": [f"E{i}" for i in range(n_ch)],
    }
    monkeypatch.setattr(m, "load_subject_propagation_events",
                        lambda subj_dir: fake_loaded)

    # Stage a fake PR-2 JSON at expected path
    pr2_root = tmp_path / "pr2"
    pr2_root.mkdir()
    monkeypatch.setattr(m, "PR2_PER_SUBJECT_DIR", pr2_root)
    (pr2_root / "yuquan_x.json").write_text(_json.dumps({
        "adaptive_cluster": {"labels": [0, 1] * 50}
    }))

    # Short-circuit fit_pca / fit_principal_curve so we don't hit numerical
    # paths — we only care that fake_build was called with mask_phantom=True.
    monkeypatch.setattr(
        m, "fit_pca",
        lambda X, n_components=3: {
            "scores": np.zeros((X.shape[0], n_components), dtype=float),
            "components": np.eye(n_components, X.shape[1], dtype=float),
            "explained_variance": np.ones(n_components, dtype=float),
        },
    )
    monkeypatch.setattr(
        m, "fit_principal_curve",
        lambda scores, max_iter=15: {
            "s": np.zeros(scores.shape[0], dtype=float),
            "residuals": np.ones(scores.shape[0], dtype=float),
            "residual_mean_sq": 1.0,
            "n_iter": 1,
            "converged": True,
            "splines": None,
            "s_grid": None,
            "spline_n_used": 0,
        },
    )
    # kmeans_centroids_from_labels: stub a non-degenerate axis
    monkeypatch.setattr(
        m, "kmeans_centroids_from_labels",
        lambda X, labels, n_clusters: np.vstack([
            np.zeros(X.shape[1], dtype=float),
            np.ones(X.shape[1], dtype=float),
        ]),
    )

    _ = m._run_one("yuquan_x", "yuquan", "x", mask_phantom=True)
    assert captured.get("mask_phantom") is True, captured

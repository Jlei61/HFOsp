"""Tests for src/topic5_field_reversal.py (TA/TB field-reversal gate).

Task 0 (this file, first test): substrate-general event loader — un-stub the
narrow path of `load_event_labels_ranks` in src/topic5_event_resolved_alignment.py.
narrow must return the SAME bundle schema as broad=True and pass the same C1
positional-alignment proof, on the narrow labels/lagpat pools.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.topic5_event_resolved_alignment import load_event_labels_ranks
from src.topic5_field_reversal import signed_reversal_corr

_NARROW_LABELS = Path("results/interictal_propagation_masked/per_subject")


def _a_narrow_subject():
    # first stable_k==2 narrow-labelled subject (deterministic pick)
    for p in sorted(_NARROW_LABELS.glob("*.json")):
        ac = json.load(open(p)).get("adaptive_cluster", {})
        if ac.get("stable_k") == 2 and ac.get("chosen_k") == 2:
            return p.stem
    return None


@pytest.mark.skipif(not _NARROW_LABELS.exists(), reason="narrow labels not mounted")
def test_narrow_loader_returns_broad_schema_and_passes_c1():
    ds_sid = _a_narrow_subject()
    if ds_sid is None:
        pytest.skip("no stable_k==2 narrow subject")
    dataset, subject = ds_sid.split("_", 1)
    bundle = load_event_labels_ranks(dataset, subject, broad=False)   # must NOT raise
    for k in ("masked", "bools", "labels", "channel_names", "cluster_template_ranks", "n_blocks"):
        assert k in bundle
    assert bundle["masked"].shape[0] == len(bundle["channel_names"])
    assert set(np.unique(bundle["labels"])) <= {0, 1}


# Task 1 tests: signed_reversal_corr

def _grid(n=81):
    yy, xx = np.mgrid[0:n, 0:n]
    return xx.astype(float), yy.astype(float)


def test_detects_perfect_reversal():
    xx, yy = _grid()
    S = np.ones_like(xx)
    f0 = {"T": xx + yy, "S": S}
    f1 = {"T": -(xx + yy), "S": S}          # exact reversal
    out = signed_reversal_corr(f0, f1)
    assert out["signed_corr"] < -0.99
    assert not out["insufficient_overlap"]


def test_no_y_mirror_is_applied():
    # F1 = x - y : corr(F0,F1)=0, but flip_y(F1) -> x+y -> corr +1.
    # A mirror-invariant impl would wrongly return +1; the no-mirror stat must return ~0.
    xx, yy = _grid()
    S = np.ones_like(xx)
    f0 = {"T": xx + yy, "S": S}
    f1 = {"T": xx - yy, "S": S}
    out = signed_reversal_corr(f0, f1)
    assert abs(out["signed_corr"]) < 0.05


def test_insufficient_overlap_flagged():
    xx, yy = _grid()
    S0 = np.zeros_like(xx); S0[:2, :2] = 1.0     # tiny support
    S1 = np.zeros_like(xx); S1[:2, :2] = 1.0
    out = signed_reversal_corr({"T": xx + yy, "S": S0}, {"T": -(xx + yy), "S": S1})
    assert out["insufficient_overlap"] is True
    assert out["n_overlap"] < 25


# Task 2 tests: build_reversal_fields

from src.topic5_field_reversal import build_reversal_fields
from src.propagation_contact_plane_readout import make_plane_grid


def _toy_plane(names, xy):
    return {"channels": [{"name": n, "x_norm": xy[n][0], "y_norm": xy[n][1],
                          "typical_rank": 0.0, "support": 1.0} for n in names]}


def test_both_fields_on_same_frame_and_sigma():
    names = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 7)]
    xy = {n: (0.1 * i, 0.0) for i, n in enumerate(names)}
    plane_ref = _toy_plane(names, xy)
    cav0 = {n: {"value": float(i), "support": 1.0} for i, n in enumerate(names)}
    cav1 = {n: {"value": float(len(names) - i), "support": 1.0} for i, n in enumerate(names)}  # reversed
    X, Y = make_plane_grid()
    out = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y)
    assert out["field0"] is not None and out["field1"] is not None
    # same sigma used for both (single float returned)
    assert out["field0"]["sigma_xy"] == out["field1"]["sigma_xy"] == out["sigma"]
    assert set(out["names_used"]) == set(names)


def test_membership_mismatch_names_used_is_cav1_on_plane():
    names = [f"A{i}" for i in range(1, 7)]
    xy = {n: (0.1 * i, 0.0) for i, n in enumerate(names)}
    plane_ref = _toy_plane(names, xy)
    cav0 = {n: {"value": 1.0, "support": 1.0} for n in names}
    cav1 = {n: {"value": 1.0, "support": 1.0} for n in names[:5]}     # A6 absent in cav1
    cav1["A6"] = {"value": np.nan, "support": 0.0}
    X, Y = make_plane_grid()
    out = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y)
    assert "A6" not in out["names_used"]


# Task 3 tests: within_shaft_reversal_gate (primary null)

from src.topic5_field_reversal import within_shaft_reversal_gate


def _two_shaft_plane():
    # two shafts A (x=0 column) and B (x=1 column), 6 contacts each, along-shaft y gradient
    names, xy = [], {}
    for sh, x in (("A", 0.0), ("B", 1.0)):
        for i in range(6):
            n = f"{sh}{i+1}"; names.append(n); xy[n] = (x, 0.15 * i)
    return {"channels": [{"name": n, "x_norm": xy[n][0], "y_norm": xy[n][1],
                          "typical_rank": 0.0, "support": 1.0} for n in names]}, names


def test_along_shaft_reversal_beats_within_shaft_null():
    plane, names = _two_shaft_plane()
    # cav0 rises along y; cav1 is the along-shaft reverse -> anti-correlated fields
    cav0 = {n: {"value": float(n[1:]), "support": 1.0} for n in names}
    cav1 = {n: {"value": float(7 - int(n[1:])), "support": 1.0} for n in names}
    rng = np.random.default_rng(0)
    out = within_shaft_reversal_gate(plane, cav0, cav1, X=None, Y=None, sigma=None,
                                     n_perm=200, rng=rng, overlap_min=10)
    assert out["signed_corr"] < 0
    assert out["percentile"] < 5.0          # observed below within-shaft null
    assert out["passed"] is True
    assert out["degenerate_null"] is False


def test_singleton_shafts_flagged_degenerate():
    # every contact on its own shaft -> nothing permutable within-shaft.
    # Toy-scaffolding note: parse_shaft() (src/propagation_skeleton_geometry.py _NAME_RE)
    # groups by ALPHABETIC PREFIX only, so "S0".."S7" would all share shaft "S" (one
    # 8-member shaft, not 8 singletons) -> effective_n=8, NOT degenerate at min_eff=6.
    # Use 8 distinct single-letter prefixes so each contact is genuinely alone on its
    # shaft (mirrors real SEEG naming: shaft-prefix + contact-ordinal). Only the naming
    # scheme changed; contact count (8), coordinates, values, and all gate params are
    # unchanged from the brief.
    names = [f"{chr(ord('A') + i)}1" for i in range(8)]
    plane = {"channels": [{"name": n, "x_norm": 0.1 * i, "y_norm": 0.0,
                           "typical_rank": 0.0, "support": 1.0} for i, n in enumerate(names)]}
    cav0 = {n: {"value": float(i), "support": 1.0} for i, n in enumerate(names)}
    cav1 = {n: {"value": float(8 - i), "support": 1.0} for i, n in enumerate(names)}
    rng = np.random.default_rng(0)
    out = within_shaft_reversal_gate(plane, cav0, cav1, X=None, Y=None, sigma=None,
                                     n_perm=50, rng=rng, min_eff=6, overlap_min=10)
    assert out["degenerate_null"] is True
    assert out["passed"] is False


# Task 4 tests: random_split_contrast (descriptive, non-inferential contrast)

from src.topic5_field_reversal import random_split_contrast


def _bundle_two_clusters(plane_names):
    # cluster 0/1 share a common spatial pattern `shared` (same for every event, regardless
    # of class) PLUS a class-specific `reversal` term that flips sign between the two clusters
    # (cluster 0 = +reversal, cluster 1 = -reversal). A label-blind random 50/50 split mixes
    # both clusters, so its per-contact mean is dominated by the SHARED term -> positive corr
    # between the two halves; the TRUE label split isolates the (larger-amplitude) reversal
    # term -> negative corr. masked (n_ch, n_ev).
    #
    # Toy-tuning note (see task-4-report.md): the brief's literal toy used ONLY the reversal
    # term (`rise` / `1-rise`, no shared component). Verified numerically that this collapses
    # observed AND every random split to the exact same -1: for a noiseless 2-cluster
    # population, ANY 50/50 split is itself an exact mirror-image pair (the two halves' cluster
    # counts are complementary: n1 vs 40-n1), so there is no way for a label-blind split to
    # differ in sign from the true label split -- 0/83 valid splits in a probe run were even
    # positive. Adding a `shared` pattern (here a half-sine, spatially different from `rise`)
    # with reversal amplitude 3x the shared amplitude fixes this while preserving the brief's
    # "cluster 0 rises / cluster 1 reversed" shape; verified over 8 seeds all give
    # split_median in [0.86, 0.99] vs observed_ab_corr == -0.77 (frac_positive == 1.0 in 7/8).
    n_ch = len(plane_names)
    rise = np.linspace(0, 1, n_ch)
    shared = np.sin(np.linspace(0, np.pi, n_ch))
    reversal = 3.0 * rise
    ev0 = np.tile((shared + reversal)[:, None], (1, 40))
    ev1 = np.tile((shared - reversal)[:, None], (1, 40))
    masked = np.hstack([ev0, ev1])
    labels = np.array([0] * 40 + [1] * 40)
    return {"masked": masked, "labels": labels, "channel_names": list(plane_names),
            "bools": np.isfinite(masked)}


def test_random_split_centers_positive_observed_negative():
    plane, names = _two_shaft_plane()
    bundle = _bundle_two_clusters(names)
    X, Y = make_plane_grid()
    rng = np.random.default_rng(1)
    out = random_split_contrast(bundle, plane, X=X, Y=Y, sigma=None, n_split=100,
                                rng=rng, overlap_min=10)
    assert out["observed_ab_corr"] < 0                 # true A/B reversed
    assert out["split_median"] > out["observed_ab_corr"]  # random halves not reversed
    assert out["note"] == "non_inferential"


# Task 4 review-fix tests: channel_floor None-corr guard + signed_reversal_corr None-field guard

from src.topic5_field_reversal import channel_floor


def test_channel_floor_well_behaved():
    plane, names = _two_shaft_plane()
    cav0 = {n: {"value": float(n[1:]), "support": 1.0} for n in names}
    cav1 = {n: {"value": float(7 - int(n[1:])), "support": 1.0} for n in names}
    X, Y = make_plane_grid()
    rng = np.random.default_rng(0)
    out = channel_floor(plane, cav0, cav1, X=X, Y=Y, sigma=None, n_perm=200,
                        rng=rng, overlap_min=10)
    assert np.isfinite(out["percentile"])
    assert np.isfinite(out["null_p05"])
    assert np.isfinite(out["null_p50"])
    assert np.isfinite(out["null_p95"])
    assert len(out["null_corrs"]) > 0


def test_channel_floor_tied_value_returns_empty_null_without_crash():
    # Repro (review finding): TB's cav1 has P,Q exactly tied at 5.0, plus a 3rd valid contact R
    # placed far outside the plane grid (X in [-0.5,1.5], Y in [-1,1]) so R contributes exactly
    # zero weight (float underflow) everywhere on the grid, yet still counts toward
    # MIN_PLANE_CONTACTS=3. The OBSERVED TB field is therefore exactly constant (5.0) over the
    # whole masked region -> zero variance -> signed_reversal_corr returns signed_corr=None
    # (insufficient_overlap=True). channel_shuffle's random permutations often move the "9.0"
    # value onto P or Q (in-grid), breaking the constant field, so those draws DO produce a
    # finite null entry -> null ends up non-empty. Before Fix 1, channel_floor then called
    # placement_in_distribution(None, non_empty_null) -> np.isfinite(None) -> TypeError.
    names = ["P", "Q", "R"]
    xy = {"P": (0.0, 0.0), "Q": (0.2, 0.0), "R": (100.0, 100.0)}
    plane = {"channels": [{"name": n, "x_norm": xy[n][0], "y_norm": xy[n][1],
                          "typical_rank": 0.0, "support": 1.0} for n in names]}
    cav0 = {"P": {"value": 1.0, "support": 1.0}, "Q": {"value": 2.0, "support": 1.0},
            "R": {"value": 3.0, "support": 1.0}}
    cav1 = {"P": {"value": 5.0, "support": 1.0}, "Q": {"value": 5.0, "support": 1.0},
            "R": {"value": 9.0, "support": 1.0}}
    X, Y = make_plane_grid()
    rng = np.random.default_rng(0)
    out = channel_floor(plane, cav0, cav1, X=X, Y=Y, sigma=0.15, n_perm=200,
                        rng=rng, overlap_min=10)
    assert out["null_corrs"] == []
    assert np.isnan(out["percentile"])


def test_signed_reversal_corr_none_field_no_crash():
    out = signed_reversal_corr(None, {"T": np.zeros((3, 3)), "S": np.ones((3, 3))})
    assert out["insufficient_overlap"] is True
    assert out["signed_corr"] is None


# Task 5 tests: contact_reversal_gate (head-to-head, no geometry)

from src.topic5_field_reversal import contact_reversal_gate


def test_contact_gate_detects_reversal():
    names = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 7)]
    cav0 = {n: {"value": float(i), "support": 1.0} for i, n in enumerate(names)}
    cav1 = {n: {"value": float(len(names) - i), "support": 1.0} for i, n in enumerate(names)}
    rng = np.random.default_rng(2)
    out = contact_reversal_gate(cav0, cav1, n_perm=200, rng=rng)
    assert out["signed_spearman"] < -0.9
    assert out["percentile"] < 5.0
    assert out["passed"] is True

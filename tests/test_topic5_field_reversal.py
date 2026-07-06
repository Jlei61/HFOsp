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

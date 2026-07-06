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

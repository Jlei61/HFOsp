"""Tests for src/topic4_m3_acceptance.py (Topic 4 M3 Layer-2 equivalence).

Pre-registered tolerance-band (TOST-style) acceptance test: the model's
per-subject median spatial-extent statistics (AF = axial_fraction,
LR = lateral_ratio) must fall INSIDE a band derived from real per-subject
medians, AND AF must not be a "short axial footprint". PASS is a
band-membership decision, NOT a non-significant p-value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic4_m3_acceptance import (  # noqa: E402
    layer2_equivalence,
    subject_tolerance_band,
)


# ---------- subject_tolerance_band ----------

def test_tolerance_band_matches_percentiles():
    # Known ref arrays: 11 evenly spaced values 0.85..0.95 for AF, 0.10..0.30 for LR.
    ref_af = np.linspace(0.85, 0.95, 11)
    ref_lr = np.linspace(0.10, 0.30, 11)
    band = subject_tolerance_band(ref_af, ref_lr, q=(10, 90))

    assert band["af_lo"] == pytest.approx(np.percentile(ref_af, 10))
    assert band["af_hi"] == pytest.approx(np.percentile(ref_af, 90))
    assert band["lr_lo"] == pytest.approx(np.percentile(ref_lr, 10))
    assert band["lr_hi"] == pytest.approx(np.percentile(ref_lr, 90))
    # Sanity: lo < hi.
    assert band["af_lo"] < band["af_hi"]
    assert band["lr_lo"] < band["lr_hi"]


def test_tolerance_band_respects_custom_q():
    ref_af = np.linspace(0.80, 1.00, 21)
    ref_lr = np.linspace(0.00, 0.40, 21)
    band = subject_tolerance_band(ref_af, ref_lr, q=(25, 75))
    assert band["af_lo"] == pytest.approx(np.percentile(ref_af, 25))
    assert band["af_hi"] == pytest.approx(np.percentile(ref_af, 75))


# ---------- layer2_equivalence: shared fixture ----------

# A band representing real subjects: AF tight around 0.9, LR around 0.2.
BAND = {"af_lo": 0.85, "af_hi": 0.95, "lr_lo": 0.10, "lr_hi": 0.30}


def test_model_inside_band_and_above_min_af_passes():
    # Model per-subject medians clustered comfortably inside the band.
    model_af = np.array([0.88, 0.90, 0.91, 0.89, 0.92])
    model_lr = np.array([0.18, 0.20, 0.22, 0.19, 0.21])
    res = layer2_equivalence(model_af, model_lr, BAND, min_af=0.75)

    assert res["pass_"] is True
    assert res["af_in_band"] is True
    assert res["lr_in_band"] is True
    # Margins positive when inside the band.
    assert res["af_margin"] > 0.0
    assert res["lr_margin"] > 0.0


def test_short_axial_footprint_fails_via_min_af_and_band():
    # AF median 0.3 is far below the band [0.85, 0.95]; LR happens to land in band.
    model_af = np.array([0.28, 0.30, 0.31, 0.30, 0.29])
    model_lr = np.array([0.18, 0.20, 0.22, 0.19, 0.21])
    res = layer2_equivalence(model_af, model_lr, BAND, min_af=0.75)

    # "Different" model must NOT pass; both the min_af gate and the band fail.
    assert res["pass_"] is False
    assert res["af_in_band"] is False
    assert res["af_median"] < BAND["af_lo"]
    assert res["af_median"] < res["min_af"]
    # AF margin is negative (outside band).
    assert res["af_margin"] < 0.0


def test_af_in_band_but_lr_far_outside_fails():
    # AF sits in band; LR median 0.9 is way above the band [0.10, 0.30].
    model_af = np.array([0.88, 0.90, 0.91, 0.89, 0.92])
    model_lr = np.array([0.85, 0.90, 0.95, 0.88, 0.92])
    res = layer2_equivalence(model_af, model_lr, BAND, min_af=0.75)

    assert res["pass_"] is False
    assert res["af_in_band"] is True
    assert res["lr_in_band"] is False
    assert res["lr_margin"] < 0.0


def test_min_af_gate_can_reject_inside_band_model():
    # AF median ~0.88 IS inside band, but a strict min_af=0.90 rejects it.
    model_af = np.array([0.87, 0.88, 0.89])
    model_lr = np.array([0.18, 0.20, 0.22])
    res = layer2_equivalence(model_af, model_lr, BAND, min_af=0.90)

    assert res["af_in_band"] is True
    assert res["lr_in_band"] is True
    assert res["af_median"] == pytest.approx(0.88)
    assert res["pass_"] is False  # blocked by min_af gate alone


def test_result_schema_and_no_pvalue_key():
    model_af = np.array([0.88, 0.90, 0.92])
    model_lr = np.array([0.18, 0.20, 0.22])
    res = layer2_equivalence(model_af, model_lr, BAND, min_af=0.75)

    # Required descriptive fields present.
    for key in (
        "pass_",
        "af_in_band",
        "lr_in_band",
        "af_median",
        "lr_median",
        "af_margin",
        "lr_margin",
        "min_af",
        "note",
    ):
        assert key in res, f"missing key {key!r}"

    # Types.
    assert isinstance(res["pass_"], bool)
    assert isinstance(res["af_in_band"], bool)
    assert isinstance(res["lr_in_band"], bool)
    assert isinstance(res["af_median"], float)
    assert isinstance(res["lr_median"], float)
    assert isinstance(res["af_margin"], float)
    assert isinstance(res["lr_margin"], float)
    assert isinstance(res["min_af"], float)
    assert isinstance(res["note"], str)

    # Note is non-empty and frames PASS as band-membership, not a p-value.
    assert len(res["note"]) > 0
    assert "p-value" not in res["note"].lower() or "not" in res["note"].lower()

    # No key implies a p-value-based pass.
    for key in res:
        kl = key.lower()
        assert "pval" not in kl
        assert "p_value" not in kl
        assert "pvalue" not in kl
        assert kl != "p"


def test_medians_are_computed_from_per_subject_arrays():
    # median([0.80, 0.90, 1.00]) == 0.90 ; median([0.10, 0.20, 0.30]) == 0.20
    model_af = np.array([0.80, 0.90, 1.00])
    model_lr = np.array([0.10, 0.20, 0.30])
    res = layer2_equivalence(model_af, model_lr, BAND, min_af=0.75)
    assert res["af_median"] == pytest.approx(0.90)
    assert res["lr_median"] == pytest.approx(0.20)

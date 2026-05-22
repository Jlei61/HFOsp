"""Unit tests for SEF-ITP Phase 1: H6 / H1 / H2 spatial geometry checks.

Plan: docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md
Module: src/sef_itp_phase1.py

All tests use synthetic numpy arrays with fixed seeds. NO real cohort data —
real-data integration is gated on Topic 0 Phase 0a + Phase 0b.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.sef_itp_phase1 import (
    _jaccard_set,
    _morans_i,
    _parse_channel,
    _shaft_stratified_shuffle,
    compute_h1_compactness,
    compute_h1_descriptive,
    compute_h1_full,
    compute_h1c_envelope,
    compute_h2_set_reversal,
    compute_h2_spatial_reversal,
    compute_h6_segregation,
    compute_participation_rate,
    pairwise_3d_euclidean,
    pairwise_shaft_ordinal,
)


SEED = 20260521


# =============================================================================
# Channel parsing
# =============================================================================


def test_parse_channel_monopolar():
    assert _parse_channel("A1") == ("A", 1)
    assert _parse_channel("LOF12") == ("LOF", 12)


def test_parse_channel_bipolar():
    assert _parse_channel("A1-A2") == ("A", 1)
    assert _parse_channel("LOF12-LOF13") == ("LOF", 12)


def test_parse_channel_primed():
    # primed name: A'1 -> ("A'", 1)
    prefix, num = _parse_channel("A'1")
    assert prefix == "A'"
    assert num == 1


# =============================================================================
# Distance metrics
# =============================================================================


def test_pairwise_3d_euclidean_basic():
    coords = np.array([[0, 0, 0], [3, 4, 0], [0, 0, 5]], dtype=float)
    D = pairwise_3d_euclidean(coords)
    assert D[0, 0] == 0
    assert D[0, 1] == pytest.approx(5.0)
    assert D[0, 2] == pytest.approx(5.0)
    assert D[1, 2] == pytest.approx(np.sqrt(9 + 16 + 25))
    # symmetric
    assert np.allclose(D, D.T)


def test_pairwise_shaft_ordinal_within_shaft_only():
    names = ["A1-A2", "A2-A3", "A3-A4", "B1-B2", "B2-B3"]
    D = pairwise_shaft_ordinal(names)
    # A1-A2 vs A2-A3: prefix A, ordinals 1, 2 → 1
    assert D[0, 1] == 1.0
    # A1-A2 vs A3-A4: ordinals 1, 3 → 2
    assert D[0, 2] == 2.0
    # cross-shaft: inf
    assert np.isinf(D[0, 3])
    assert np.isinf(D[2, 4])
    # within B
    assert D[3, 4] == 1.0


# =============================================================================
# H6 tests
# =============================================================================


def test_h6_segregation_synthetic_cross_shaft_depth_pattern():
    """TT-H6-1: spatial segregation BEYOND shaft assignment.

    Setup: 3 shafts (A/B/C) each with 5 contacts at z=0..4. High participation
    at z=0,1 on EVERY shaft (deep "hot" zone); low at z=3,4 (superficial). This
    spatial structure crosses shafts; shaft-stratified shuffle breaks the
    within-shaft depth ordering, so null should be lower than actual.
    """
    rng = np.random.default_rng(SEED)
    names = []
    coords = []
    participation = []
    for shaft_idx, prefix in enumerate(["A", "B", "C"]):
        x = shaft_idx * 15.0
        for depth in range(5):
            names.append(f"{prefix}{depth+1}-{prefix}{depth+2}")
            coords.append([x, 0, depth * 2.0])
            # high participation at depth 0,1; low at depth 3,4; mid at 2
            if depth <= 1:
                participation.append(0.85 + rng.uniform(-0.05, 0.05))
            elif depth == 2:
                participation.append(0.5 + rng.uniform(-0.05, 0.05))
            else:
                participation.append(0.15 + rng.uniform(-0.05, 0.05))
    coords = np.asarray(coords, dtype=float)
    participation = np.asarray(participation)

    out = compute_h6_segregation(
        participation,
        names,
        coords=coords,
        distance_metric="euclidean",
        n_permutations=500,
        rng=rng,
    )
    assert out["verdict"] in ("PASS", "PARTIAL"), f"got {out}"
    assert out["n_shafts"] == 3
    # Moran's I should be elevated (cross-shaft depth clustering)
    assert out["morans_i_actual"] > out["morans_i_null_median"], (
        f"actual {out['morans_i_actual']} should exceed null median {out['morans_i_null_median']}"
    )


def test_h6_segregation_shaft_only_pattern_returns_null():
    """When segregation = shaft (high on one shaft, low on another), shaft-stratified
    null correctly reports NULL — there's no spatial structure BEYOND shaft.

    User audit 2026-05-21: this is the INTENDED behavior. Shaft-stratified null
    asks "is there spatial structure beyond what implant geometry gives you?"
    When the only "structure" IS the implant geometry (shaft A high / shaft B low),
    null correctly says NULL. A v1 doc draft suggested PASS for this case — that
    was wrong and has been retracted in this test docstring.
    """
    rng = np.random.default_rng(SEED)
    names = [f"A{i+1}-A{i+2}" for i in range(5)] + [f"B{i+1}-B{i+2}" for i in range(5)]
    coords = np.array(
        [[0, 0, i] for i in range(5)] + [[10, 10, i] for i in range(5)], dtype=float
    )
    participation = np.array([0.9, 0.9, 0.8, 0.85, 0.9, 0.1, 0.1, 0.05, 0.1, 0.15])

    out = compute_h6_segregation(
        participation,
        names,
        coords=coords,
        distance_metric="euclidean",
        n_permutations=500,
        rng=rng,
    )
    # Shaft-stratified shuffle preserves per-shaft means → null is degenerate at actual
    assert out["verdict"] in ("NULL",), f"expected NULL (no structure beyond shaft), got {out}"


def test_h6_morans_i_known_autocorrelation():
    """TT-H6-2: Moran's I should be strongly positive for monotonically-arranged values."""
    n = 10
    values = np.linspace(0.1, 1.0, n)
    # 1D distance: |i - j|
    D = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    I = _morans_i(values, D)
    assert I > 0.2, f"expected strong positive autocorrelation, got I={I}"


def test_h6_shaft_stratified_shuffle_preserves_per_shaft_mean():
    """TT-H6-3: shaft-stratified shuffle should preserve per-shaft sum (not mean)."""
    rng = np.random.default_rng(SEED)
    shafts = ["A", "A", "A", "B", "B", "B"]
    values = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    original_a_sum = values[:3].sum()
    original_b_sum = values[3:].sum()

    for _ in range(50):
        out = _shaft_stratified_shuffle(values, shafts, rng)
        assert out[:3].sum() == pytest.approx(original_a_sum)
        assert out[3:].sum() == pytest.approx(original_b_sum)


def test_h6_excludes_single_shaft_subject():
    """TT-H6-4: single-shaft subject should be EXCLUDED (advisor 2026-05-21 fix)."""
    names = [f"A{i+1}-A{i+2}" for i in range(8)]
    coords = np.array([[0, 0, i] for i in range(8)], dtype=float)
    participation = np.random.default_rng(SEED).uniform(0, 1, size=8)

    out = compute_h6_segregation(
        participation,
        names,
        coords=coords,
        distance_metric="euclidean",
        n_permutations=100,
        rng=np.random.default_rng(SEED),
    )
    assert out["verdict"] == "EXCLUDED_SINGLE_SHAFT"
    assert out["n_shafts"] == 1


def test_h6_diffuse_random_participation_gives_null():
    """H6 NULL when participation rate is spatially random."""
    rng = np.random.default_rng(SEED)
    names = [f"A{i+1}-A{i+2}" for i in range(8)] + [f"B{i+1}-B{i+2}" for i in range(8)]
    coords = np.vstack(
        [np.column_stack([np.zeros(8), np.zeros(8), np.arange(8)]),
         np.column_stack([np.ones(8) * 10, np.ones(8) * 10, np.arange(8)])]
    ).astype(float)
    # random participation, no spatial structure
    participation = rng.uniform(0.3, 0.7, size=16)

    out = compute_h6_segregation(
        participation,
        names,
        coords=coords,
        distance_metric="euclidean",
        n_permutations=500,
        rng=rng,
    )
    # most likely NULL or PARTIAL; should NOT be FAIL or PASS
    assert out["verdict"] in ("NULL", "PARTIAL", "INSUFFICIENT_SPLIT"), f"got {out}"


# =============================================================================
# H1 tests
# =============================================================================


def _make_h1_synthetic(rng, target_pattern="compact"):
    """Synthetic SEEG-like layout for H1 tests.

    v1.0.7: non-uniform z packing — first 3 contacts of each shaft are very
    close (z=0/1/2), remaining 7 spread (z=15..45). This makes "first 3 of
    shaft" measurably more compact than "3 random contacts of shaft" under
    shaft-only matched null. v1.0.6's uniform layout would have made the
    null and actual statistically indistinguishable under shaft-only matching.

    Returns: (channel_names, coords, participation, hfo_rate, target_indices, valid_indices)
    """
    names = []
    coords = []
    n_per_shaft = 10
    # First 3 contacts tightly packed (z=0,1,2), rest spread (z=15..45)
    z_dense = np.array([0.0, 1.0, 2.0])
    z_spread = np.linspace(15.0, 45.0, n_per_shaft - 3)
    z_range = np.concatenate([z_dense, z_spread])
    for shaft_idx, prefix in enumerate(["A", "B", "C"]):
        shaft_x = shaft_idx * 50.0
        for i in range(n_per_shaft):
            names.append(f"{prefix}{i+1}-{prefix}{i+2}")
            coords.append([shaft_x, 0, z_range[i]])
    coords = np.asarray(coords, dtype=float)
    n_total = len(names)
    participation = rng.uniform(0.4, 0.9, size=n_total)
    hfo_rate = rng.uniform(1, 10, size=n_total)

    valid = list(range(n_total))

    if target_pattern == "compact":
        # target = the 3 tightly-packed contacts of shaft A (z=0,1,2)
        target = [0, 1, 2]
    elif target_pattern == "diffuse":
        # target = 1 random contact from each shaft (cross-shaft spread + z-unaligned).
        # User audit 2026-05-21: picking first contact of each shaft (z=0 aligned)
        # gives a subtly compact actual that beats the matched-null median; for a
        # genuinely diffuse target we pick random z within each shaft.
        target = [
            int(rng.integers(0, n_per_shaft)),
            int(rng.integers(0, n_per_shaft)) + n_per_shaft,
            int(rng.integers(0, n_per_shaft)) + 2 * n_per_shaft,
        ]
    else:
        raise ValueError(target_pattern)

    return names, coords, participation, hfo_rate, target, valid


def test_h1a_within_source_compact_synthetic():
    """TT-H1-1: compact source → significantly < null."""
    rng = np.random.default_rng(SEED)
    names, coords, participation, hfo_rate, target, valid = _make_h1_synthetic(
        rng, target_pattern="compact"
    )

    out = compute_h1_compactness(
        members=target,
        candidate_pool=valid,
        coords=coords,
        channel_names=names,
        participation=participation,
        hfo_rate=hfo_rate,
        distance_metric="euclidean",
        n_null=500,
        rng=rng,
    )
    assert out["verdict"] in ("PASS",), f"expected PASS, got {out}"
    assert out["C_actual"] < out["C_null_median"]


def test_h1_strict_diffuse_null_strict():
    """TT-H1-2b: diffuse target → NULL or FAIL_DIFFUSE, never PASS.

    Target = one channel per shaft (cross-shaft spread). Under v1.0.7 shaft-only
    matched-null on non-uniform shaft layout (dense low-z + spread high-z), the
    actual mean pairwise distance is at least as large as null median; depending
    on which contacts the diffuse target lands on, verdict is NULL or FAIL_DIFFUSE.

    Strict intent (user audit 2026-05-21): catch regressions where diffuse
    target wrongly PASSes. Both NULL and FAIL_DIFFUSE confirm "not compact".
    """
    rng = np.random.default_rng(SEED)
    names, coords, participation, hfo_rate, target, valid = _make_h1_synthetic(
        rng, target_pattern="diffuse"
    )

    out = compute_h1_compactness(
        members=target,
        candidate_pool=valid,
        coords=coords,
        channel_names=names,
        participation=participation,
        hfo_rate=hfo_rate,
        distance_metric="euclidean",
        n_null=500,
        rng=rng,
    )
    assert out["verdict"] in ("NULL", "FAIL_DIFFUSE"), (
        f"diffuse target must NOT be PASS, got {out}"
    )


def test_h1c_envelope_within_field():
    """TT-H1-3: endpoint inside high-participation field → ratio ≈ 1, PASS."""
    rng = np.random.default_rng(SEED)
    # 30 valid channels in a Gaussian cloud centered at origin
    coords = rng.normal(0, 5, size=(30, 3))
    # endpoint = 6 channels also inside the cloud (random subset)
    endpoint = list(rng.choice(30, size=6, replace=False))
    non_endpoint_pool = [i for i in range(30) if i not in set(endpoint)]

    out = compute_h1c_envelope(
        endpoint_indices=endpoint,
        non_endpoint_pool=non_endpoint_pool,
        coords=coords,
        n_null=500,
        rng=rng,
    )
    assert out["verdict"] == "PASS", f"expected PASS, got {out}"
    assert out["ratio_endpoint_to_non_endpoint"] < 2.0


def test_h1c_envelope_outside_field_fail():
    """TT-H1-4: endpoint outside field → ratio >> 1, FAIL."""
    rng = np.random.default_rng(SEED)
    # 30 valid channels: 24 near origin, 6 endpoint far away
    near = rng.normal(0, 1, size=(24, 3))
    far = rng.normal([20, 20, 20], 1, size=(6, 3))
    coords = np.vstack([near, far])
    endpoint = list(range(24, 30))
    non_endpoint_pool = list(range(24))  # explicit exclusion of endpoint

    out = compute_h1c_envelope(
        endpoint_indices=endpoint,
        non_endpoint_pool=non_endpoint_pool,
        coords=coords,
        n_null=500,
        rng=rng,
    )
    assert out["verdict"] == "FAIL", f"expected FAIL, got {out}"
    assert out["ratio_endpoint_to_non_endpoint"] > 3.0


def test_h1c_envelope_circularity_guard():
    """Advisor 2026-05-21 fix: H1c centroid uses NON-endpoint channels only.

    v1.0.7: explicit non_endpoint_pool parameter enforces this by contract —
    if centroid were from ALL valid channels (including endpoint), endpoint
    would be artifactually close to centroid → false PASS even at an extreme.
    """
    rng = np.random.default_rng(SEED)
    # 30 valid: 24 near origin, 6 at extreme (10, 10, 10)
    near = rng.normal(0, 1, size=(24, 3))
    far = rng.normal([10, 10, 10], 0.5, size=(6, 3))
    coords = np.vstack([near, far])
    endpoint = list(range(24, 30))
    non_endpoint_pool = list(range(24))

    out = compute_h1c_envelope(
        endpoint_indices=endpoint,
        non_endpoint_pool=non_endpoint_pool,
        coords=coords,
        n_null=500,
        rng=rng,
    )
    # ratio computed against non-endpoint centroid (origin-ish), endpoint at extreme
    # → ratio should be large (≈ 17/1 ≈ very large), FAIL
    assert out["ratio_endpoint_to_non_endpoint"] > 5.0, (
        f"circularity guard failed: ratio={out.get('ratio_endpoint_to_non_endpoint')}"
    )


def test_h1c_envelope_rejects_endpoint_in_pool():
    """v1.0.7 contract: non_endpoint_pool must not contain endpoint_indices."""
    rng = np.random.default_rng(SEED)
    coords = rng.normal(0, 5, size=(10, 3))
    endpoint = [0, 1, 2]
    # invalid: pool contains endpoint
    bad_pool = list(range(10))

    with pytest.raises(ValueError, match="non_endpoint_pool must NOT contain"):
        compute_h1c_envelope(
            endpoint_indices=endpoint,
            non_endpoint_pool=bad_pool,
            coords=coords,
            n_null=100,
            rng=rng,
        )


def test_h1_descriptive_compact_synthetic():
    """TT-H1-1a: descriptive layer returns small mean pairwise distance for compact target."""
    rng = np.random.default_rng(SEED)
    names, coords, participation, hfo_rate, target, valid = _make_h1_synthetic(
        rng, target_pattern="compact"
    )
    out = compute_h1_descriptive(
        members=target, coords=coords, channel_names=names, distance_metric="euclidean"
    )
    assert out["verdict"] == "DESCRIPTIVE"
    # compact target: first 3 of shaft A at z ≈ 0, 3.3, 6.7 → mean pairwise ≈ 4.4
    assert out["mean_pairwise_distance"] < 7.0
    assert out["diameter"] < 8.0
    assert out["radius_of_gyration"] is not None
    assert out["n_finite_pairs"] == 3  # C(3,2) = 3


def test_h1_descriptive_diffuse_synthetic():
    """TT-H1-1b: descriptive layer returns large mean pairwise distance for diffuse target."""
    rng = np.random.default_rng(SEED)
    names, coords, participation, hfo_rate, target, valid = _make_h1_synthetic(
        rng, target_pattern="diffuse"
    )
    out = compute_h1_descriptive(
        members=target, coords=coords, channel_names=names, distance_metric="euclidean"
    )
    assert out["verdict"] == "DESCRIPTIVE"
    # diffuse target: 1 from each shaft (x=0/50/100) → mean pairwise > 50
    assert out["mean_pairwise_distance"] > 50.0


def test_h1_descriptive_gated_on_coords():
    """Descriptive layer with euclidean but no coords returns DESCRIPTIVE_GATED_ON_COORDS."""
    out = compute_h1_descriptive(
        members=[0, 1, 2],
        coords=None,
        channel_names=["A1-A2", "A2-A3", "A3-A4"],
        distance_metric="euclidean",
    )
    assert out["verdict"] == "DESCRIPTIVE_GATED_ON_COORDS"


def test_h1_descriptive_shaft_ordinal_no_coords_needed():
    """Descriptive layer with shaft_ordinal works without coords."""
    out = compute_h1_descriptive(
        members=[0, 1, 2],
        coords=None,
        channel_names=["A1-A2", "A2-A3", "A3-A4"],
        distance_metric="shaft_ordinal",
    )
    assert out["verdict"] == "DESCRIPTIVE"
    assert out["mean_pairwise_distance"] == 4 / 3  # |1-2|+|1-3|+|2-3| = 4, /3 pairs
    assert out["radius_of_gyration"] is None  # not meaningful in 1D ordinal


def test_h1_full_three_layer_verdict():
    """H1 integrated runner returns three INDEPENDENT layers + overall verdict (v1.0.3)."""
    rng = np.random.default_rng(SEED)
    names, coords, participation, hfo_rate, _, valid = _make_h1_synthetic(rng)
    # compact source + compact sink
    source = [0, 1, 2]
    sink = [10, 11, 12]

    out = compute_h1_full(
        source_indices=source,
        sink_indices=sink,
        valid_indices=valid,
        coords=coords,
        channel_names=names,
        participation=participation,
        hfo_rate=hfo_rate,
        distance_metric="euclidean",
        n_null=500,
        rng=rng,
    )
    # three layer keys
    assert "descriptive" in out
    assert "strict" in out
    assert "envelope" in out
    # sub-keys
    assert "source" in out["descriptive"]
    assert "sink" in out["descriptive"]
    assert "source" in out["strict"]
    assert "sink" in out["strict"]
    # overall verdict still present
    assert "h1_overall_verdict" in out
    # descriptive numbers populated
    assert out["descriptive"]["source"]["verdict"] == "DESCRIPTIVE"
    assert out["descriptive"]["sink"]["verdict"] == "DESCRIPTIVE"
    # v1.0.4 (2026-05-21 user audit): h1a/h1b/h1c backward-compat keys REMOVED
    # — they would let downstream code collapse three layers back into one H1 narrative
    assert "h1a" not in out, "h1a backward-compat key must NOT be present (v1.0.4)"
    assert "h1b" not in out, "h1b backward-compat key must NOT be present"
    assert "h1c" not in out, "h1c backward-compat key must NOT be present"


def test_h1_full_envelope_skipped_blocks_overall_verdict():
    """v1.0.4 critical: when coords=None, envelope is SKIPPED_NO_COORDS, so
    h1_overall_verdict MUST be INCOMPLETE_GATED_ON_COORDS — NOT PASS/NULL/FAIL.

    User audit 2026-05-21: "envelope SKIPPED + overall = NULL/PASS" makes
    'necessary condition not tested' look like a complete H1 result. This test
    locks that the gated case cannot silently pose as a verdict.
    """
    rng = np.random.default_rng(SEED)
    names, _coords, participation, hfo_rate, _, valid = _make_h1_synthetic(rng)
    source = [0, 1, 2]
    sink = [10, 11, 12]

    out = compute_h1_full(
        source_indices=source,
        sink_indices=sink,
        valid_indices=valid,
        coords=None,  # no coords → envelope must skip
        channel_names=names,
        participation=participation,
        hfo_rate=hfo_rate,
        distance_metric="shaft_ordinal",  # works without coords
        n_null=200,
        rng=rng,
    )
    assert out["envelope"]["verdict"] == "SKIPPED_NO_COORDS"
    assert out["h1_overall_verdict"] == "INCOMPLETE_GATED_ON_COORDS", (
        f"envelope SKIPPED must block overall verdict, got "
        f"{out['h1_overall_verdict']!r} — this is the v1.0.4 critical fix"
    )
    # strict + descriptive may still have layer-specific verdicts; that's fine
    # — but overall must refuse to summarize


# =============================================================================
# H1 overall verdict aggregation table (v1.0.6, 2026-05-21 advisor catch)
#
# UNTESTABLE must never get smuggled into a binary verdict — readers must be
# able to tell "tested-and-not-passed" from "could-not-be-tested".
# FAIL_DIFFUSE must not collapse to NULL — anti-compact strict signal needs
# to surface as its own verdict.
# =============================================================================


def _h1_overall_with_strict_sink_states(
    monkeypatch, src_verdict: str, snk_verdict: str, env_verdict: str
) -> str:
    """Drive compute_h1_full by monkey-patching the three layer functions to
    return canned verdicts. Returns the resulting h1_overall_verdict."""
    from src import sef_itp_phase1 as mod

    def fake_strict_factory(verdict):
        def fn(*args, **kwargs):
            return {"verdict": verdict, "C_actual": 1.0, "C_null_median": 1.0,
                    "null_lower_p": 0.5, "null_upper_p": 0.5}
        return fn

    def fake_envelope(*args, **kwargs):
        return {"verdict": env_verdict, "ratio_endpoint_to_non_endpoint": 1.0}

    def fake_descriptive(*args, **kwargs):
        return {"verdict": "DESCRIPTIVE", "mean_pairwise_distance": 1.0}

    # Patch only the inner per-layer calls (compute_h1_full orchestrates them)
    call_count = {"strict": 0}
    def fake_strict_dispatcher(*args, **kwargs):
        verdicts = [src_verdict, snk_verdict]
        v = verdicts[call_count["strict"]]
        call_count["strict"] += 1
        return fake_strict_factory(v)()

    monkeypatch.setattr(mod, "compute_h1_descriptive", fake_descriptive)
    monkeypatch.setattr(mod, "compute_h1_compactness", fake_strict_dispatcher)
    monkeypatch.setattr(mod, "compute_h1c_envelope", fake_envelope)

    out = mod.compute_h1_full(
        source_indices=[0, 1, 2],
        sink_indices=[3, 4, 5],
        valid_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        coords=np.zeros((8, 3)),
        channel_names=[f"X{i}" for i in range(8)],
        participation=np.zeros(8),
        n_null=10,
        rng=np.random.default_rng(0),
        coord_units="mm",
    )
    return out["h1_overall_verdict"]


VERDICT_TABLE = [
    # (src, snk, env, expected_overall)
    ("PASS", "PASS", "PASS", "PASS"),
    ("PASS", "NULL", "PASS", "partial_PASS"),
    ("NULL", "PASS", "PASS", "partial_PASS"),
    ("NULL", "NULL", "PASS", "NULL"),

    # UNTESTABLE distinguished from NULL on both sides
    ("PASS", "INSUFFICIENT_NULL", "PASS", "PASS_one_side_untestable"),
    ("INSUFFICIENT_NULL", "PASS", "PASS", "PASS_one_side_untestable"),
    ("PASS", "INSUFFICIENT_CHANNELS", "PASS", "PASS_one_side_untestable"),
    ("NULL", "INSUFFICIENT_NULL", "PASS", "NULL_one_side_untestable"),
    ("INSUFFICIENT_NULL", "NULL", "PASS", "NULL_one_side_untestable"),
    ("INSUFFICIENT_NULL", "INSUFFICIENT_NULL", "PASS", "UNTESTABLE_BOTH_SIDES"),

    # FAIL_DIFFUSE surfaces (does not collapse to NULL)
    ("FAIL_DIFFUSE", "PASS", "PASS", "FAIL_DIFFUSE"),
    ("PASS", "FAIL_DIFFUSE", "PASS", "FAIL_DIFFUSE"),
    ("FAIL_DIFFUSE", "FAIL_DIFFUSE", "PASS", "FAIL_DIFFUSE"),
    ("FAIL_DIFFUSE", "NULL", "PASS", "FAIL_DIFFUSE"),

    # Envelope-level overrides
    ("PASS", "PASS", "FAIL", "FAIL"),                 # necessary condition override
    ("PASS", "PASS", "SKIPPED_NO_COORDS", "INCOMPLETE_GATED_ON_COORDS"),
    ("PASS", "PASS", "INSUFFICIENT_NON_ENDPOINT", "INCONCLUSIVE_ENVELOPE_INDETERMINATE"),
    ("PASS", "PASS", "DEGENERATE_CENTROID", "INCONCLUSIVE_ENVELOPE_INDETERMINATE"),
    ("PASS", "PASS", "INSUFFICIENT_NULL", "INCONCLUSIVE_ENVELOPE_INDETERMINATE"),
]


@pytest.mark.parametrize("src,snk,env,expected", VERDICT_TABLE)
def test_h1_overall_verdict_table(monkeypatch, src, snk, env, expected):
    """v1.0.6: full verdict aggregation table — UNTESTABLE ≠ NULL, FAIL_DIFFUSE ≠ NULL."""
    got = _h1_overall_with_strict_sink_states(monkeypatch, src, snk, env)
    assert got == expected, (
        f"src={src}, snk={snk}, env={env} → expected {expected!r}, got {got!r}"
    )


# =============================================================================
# H2 tests
# =============================================================================


def test_h2_set_reversal_perfect_swap():
    """TT-H2-1: perfect role swap → R_set strongly positive."""
    rng = np.random.default_rng(SEED)
    S_A = [0, 1, 2]
    K_A = [7, 8, 9]
    S_B = [7, 8, 9]  # = K_A
    K_B = [0, 1, 2]  # = S_A

    out = compute_h2_set_reversal(S_A, K_A, S_B, K_B, n_null=1000, rng=rng)
    assert out["R_set"] > 0.5, f"R_set should be high for perfect swap, got {out}"
    assert out["upper_p"] < 0.05
    assert out["verdict"] == "PASS"


def test_h2_set_reversal_no_swap():
    """TT-H2-2: disjoint roles, no swap → R_set ≈ 0, NULL."""
    rng = np.random.default_rng(SEED)
    S_A = [0, 1, 2]
    K_A = [3, 4, 5]
    S_B = [6, 7, 8]
    K_B = [9, 10, 11]

    out = compute_h2_set_reversal(S_A, K_A, S_B, K_B, n_null=1000, rng=rng)
    # all four sets disjoint → all Jaccards = 0 → R_set = 0
    assert abs(out["R_set"]) < 0.1
    assert out["verdict"] in ("NULL",)


def test_h2_spatial_reversal_two_blob_synthetic():
    """TT-H2-3: endpoints in two spatial blobs with perfect reversal → R_spatial > 0.5."""
    rng = np.random.default_rng(SEED)
    # blob A at (0,0,0), blob B at (10,10,10)
    coords_blob_a = rng.normal([0, 0, 0], 0.5, size=(6, 3))
    coords_blob_b = rng.normal([10, 10, 10], 0.5, size=(6, 3))
    coords = np.vstack([coords_blob_a, coords_blob_b])
    # S_A in blob A (idx 0-2), K_A in blob B (idx 6-8)
    # S_B in blob B (idx 9-11), K_B in blob A (idx 3-5) — reversal
    S_A = [0, 1, 2]
    K_A = [6, 7, 8]
    S_B = [9, 10, 11]
    K_B = [3, 4, 5]

    out = compute_h2_spatial_reversal(S_A, K_A, S_B, K_B, coords, n_null=1000, rng=rng)
    assert out["R_spatial"] > 0.7, f"expected R_spatial > 0.7 for two-blob reversal, got {out}"
    assert out["verdict"] == "PASS"


def test_h2_spatial_reversal_anti_swap_fail():
    """TT-H2-5: anti-reversal geometry.

    Setup: each template has source+sink within its own blob (degenerate template
    geometry where forward's source and sink are spatially close). Then:
      - d_same = d(S_A in A, K_A in A) + d(S_B in B, K_B in B) = small + small (small)
      - d_swap = d(S_A in A, K_B in B) + d(K_A in A, S_B in B) = large + large (large)
      - R_spatial = small / (large + small) → much < 0.5 → FAIL anti-reversal

    Interpretation: roles are organized BY TEMPLATE (within-blob), not BY POSITION
    (cross-blob extremes). This is the opposite of what SEF-ITP H2 predicts.
    """
    rng = np.random.default_rng(SEED)
    # Each template's source + sink in its own blob (degenerate within-blob clustering)
    coords_blob_a = rng.normal([0, 0, 0], 0.5, size=(6, 3))
    coords_blob_b = rng.normal([10, 10, 10], 0.5, size=(6, 3))
    coords = np.vstack([coords_blob_a, coords_blob_b])
    # template A: S_A and K_A both in blob A (within-blob "extremes")
    S_A = [0, 1, 2]
    K_A = [3, 4, 5]
    # template B: S_B and K_B both in blob B
    S_B = [6, 7, 8]
    K_B = [9, 10, 11]

    out = compute_h2_spatial_reversal(S_A, K_A, S_B, K_B, coords, n_null=1000, rng=rng)
    assert out["R_spatial"] < 0.4, f"expected R_spatial < 0.4 for anti-swap, got {out}"
    assert out["verdict"] == "FAIL"


def test_h2_jaccard_basic():
    assert _jaccard_set({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert _jaccard_set({1, 2}, {3, 4}) == 0.0
    assert _jaccard_set({1, 2, 3}, {1, 2, 3}) == 1.0
    assert _jaccard_set(set(), set()) == 0.0


# =============================================================================
# Utility / contract tests
# =============================================================================


def test_compute_participation_rate_basic():
    events = np.array([[1, 1, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1]], dtype=bool)
    rates = compute_participation_rate(events)
    assert rates[0] == pytest.approx(0.75)
    assert rates[1] == pytest.approx(0.25)
    assert rates[2] == 1.0


def test_compute_h6_segregation_rejects_mismatched_lengths():
    """Channel ordering invariant: assertion fires on mismatch."""
    participation = np.array([0.5, 0.5, 0.5])
    names = ["A1-A2", "A2-A3"]  # mismatch
    with pytest.raises(ValueError, match="channel_names"):
        compute_h6_segregation(participation, names, n_permutations=10)


def test_pairwise_3d_euclidean_rejects_2d():
    """coords must be (n_ch, 3); 2D coords rejected."""
    with pytest.raises(ValueError, match=r"\(n_ch, 3\)"):
        pairwise_3d_euclidean(np.array([[0, 0], [1, 1]]))


def test_compute_h1_compactness_requires_coords_for_euclidean():
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="euclidean requires coords"):
        compute_h1_compactness(
            members=[0, 1],
            candidate_pool=[0, 1, 2, 3, 4],
            coords=None,
            channel_names=["A1-A2", "A2-A3", "A3-A4", "B1-B2", "B2-B3"],
            participation=np.array([0.5] * 5),
            distance_metric="euclidean",
            n_null=100,
            rng=rng,
        )


# =============================================================================
# v1.0.5 coord_units contract (advisor 2026-05-21 #1 fix)
# =============================================================================


def test_compute_h1_descriptive_rejects_voxel_coords():
    """v1.0.5: H1 descriptive layer with coord_units='voxel' must raise."""
    rng = np.random.default_rng(SEED)
    names, coords, _, _, _, _ = _make_h1_synthetic(rng, target_pattern="compact")
    with pytest.raises(ValueError, match="coord_units must be 'mm'"):
        compute_h1_descriptive(
            members=[0, 1, 2],
            coords=coords,
            channel_names=names,
            distance_metric="euclidean",
            coord_units="voxel",  # WRONG
        )


def test_compute_h1_compactness_rejects_voxel_coords():
    """v1.0.5: H1 strict layer with coord_units='voxel' must raise."""
    rng = np.random.default_rng(SEED)
    names, coords, participation, hfo_rate, _, valid = _make_h1_synthetic(rng)
    with pytest.raises(ValueError, match="coord_units must be 'mm'"):
        compute_h1_compactness(
            members=[0, 1, 2],
            candidate_pool=valid,
            coords=coords,
            channel_names=names,
            participation=participation,
            hfo_rate=hfo_rate,
            distance_metric="euclidean",
            n_null=100,
            rng=rng,
            coord_units="voxel",  # WRONG
        )


def test_compute_h2_spatial_reversal_rejects_voxel_coords():
    """v1.0.5: H2 spatial reversal with coord_units='voxel' must raise."""
    rng = np.random.default_rng(SEED)
    coords = rng.normal(0, 1, size=(12, 3))
    with pytest.raises(ValueError, match="coord_units must be 'mm'"):
        compute_h2_spatial_reversal(
            S_A=[0, 1, 2], K_A=[3, 4, 5],
            S_B=[6, 7, 8], K_B=[9, 10, 11],
            coords=coords,
            n_null=100,
            rng=rng,
            coord_units="voxel",  # WRONG
        )


def test_compute_h1_descriptive_accepts_mm_coord_units():
    """v1.0.5: passing coord_units='mm' explicitly does NOT raise."""
    rng = np.random.default_rng(SEED)
    names, coords, _, _, _, _ = _make_h1_synthetic(rng, target_pattern="compact")
    out = compute_h1_descriptive(
        members=[0, 1, 2],
        coords=coords,
        channel_names=names,
        distance_metric="euclidean",
        coord_units="mm",
    )
    assert out["verdict"] == "DESCRIPTIVE"


def test_compute_h1_descriptive_coord_units_none_permitted():
    """v1.0.5: coord_units=None (default) is permitted for backward-compat with
    synthetic tests passing raw numpy arrays without metadata."""
    rng = np.random.default_rng(SEED)
    names, coords, _, _, _, _ = _make_h1_synthetic(rng, target_pattern="compact")
    out = compute_h1_descriptive(
        members=[0, 1, 2],
        coords=coords,
        channel_names=names,
        distance_metric="euclidean",
        coord_units=None,  # not provided — permitted
    )
    assert out["verdict"] == "DESCRIPTIVE"

import numpy as np

from src.topic5_template_axis_field import (
    INTERICTAL_FIELD_CONTRACT,
    align_activation_to_interictal_field,
    assess_axis_direction_validity,
    axis_passes_qc,
    build_interictal_template_field_record,
    interictal_field_fingerprint,
    classify_axis_pair,
    compute_template_axis_pair,
    make_field_scorer,
    make_normalized_plane,
    score_field,
    score_field_batch,
    score_scorer_bundle,
    score_scorer_bundle_batch,
    scorers_from_interictal_record,
    shared_bisector,
)


def _geometry(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(30, 3))
    shafts = np.array([f"S{i // 6}" for i in range(30)], object)
    return x, shafts


def test_axis_pair_detects_same_and_reversed_lines():
    x, shafts = _geometry()
    v = np.array([0.8, -0.4, 0.25])
    ea = x @ v
    rank_a = -ea
    same = compute_template_axis_pair(x, rank_a, rank_a + 0.01 * np.arange(len(x)), shafts,
                                      n_axis_boot=30, n_pair_boot=40, seed=2)
    rev = compute_template_axis_pair(x, rank_a, -rank_a, shafts,
                                     n_axis_boot=30, n_pair_boot=40, seed=2)
    assert same["status"] == "ok"
    assert same["axis_pair_estimable"] is True
    assert same["geometry_2d_supported"] is True
    assert same["relation"]["relation"] == "same"
    assert same["relation"]["direction_convention"] == "positive_early_to_late"
    assert same["shared_axis"]["direction_convention"] == "positive_early_to_late"
    assert same["relation"]["abs_cosine"] > 0.99
    assert rev["relation"]["relation"] == "reversed"
    assert rev["relation"]["abs_cosine"] > 0.99


def test_template_axis_positive_direction_is_early_to_late():
    x, shafts = _geometry(21)
    true_propagation = np.array([0.8, -0.4, 0.25])
    true_propagation /= np.linalg.norm(true_propagation)
    # Rank grows from early to late along the true propagation direction.
    rank = x @ true_propagation
    out = compute_template_axis_pair(
        x, rank, rank + 0.01 * np.arange(len(x)), shafts,
        n_axis_boot=30, n_pair_boot=40, seed=7,
    )
    axis = out["axis_a"]
    assert axis["axis_definition"] == "template_propagation_axis_v2"
    assert axis["direction_convention"] == "positive_early_to_late"
    assert float(np.asarray(axis["u"]) @ true_propagation) > 0.99
    assert np.corrcoef(axis["along"], rank)[0, 1] > 0.99
    assert np.corrcoef(axis["along"], -rank)[0, 1] < -0.99
    np.testing.assert_allclose(axis["u"], -np.asarray(axis["earliness_gradient_u"]))
    np.testing.assert_allclose(axis["propagation_vector"],
                               -np.asarray(axis["earliness_gradient_beta"]))
    assert "beta" not in axis
    assert "mu_A" not in axis and "mu_B" not in axis


def test_collinearity_is_line_not_direction():
    pos = classify_axis_pair(0.6)
    neg = classify_axis_pair(-0.6)
    off = classify_axis_pair(0.49)
    assert pos["collinear"] and pos["relation"] == "same"
    assert neg["collinear"] and neg["relation"] == "reversed"
    assert not off["collinear"] and off["relation"] == "different"
    assert np.isclose(pos["line_angle_deg"], neg["line_angle_deg"])


def test_shared_bisector_aligns_reversed_b_before_averaging():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-0.8, -0.6, 0.0])
    out = shared_bisector(a, b)
    assert out["status"] == "ok"
    assert out["b_alignment_sign"] == -1
    assert out["u"][0] > 0.9
    assert out["u"][1] > 0
    assert np.isclose(np.linalg.norm(out["u"]), 1.0)


def test_axis_qc_fails_closed_on_single_shaft_or_missing_loso():
    base = {"status": "ok", "n": 12, "n_shafts": 2, "effective_rank": 2,
            "bootstrap_cosine": 0.9, "loso_cosine": 0.7}
    assert axis_passes_qc(base)
    assert not axis_passes_qc(dict(base, n_shafts=1))
    assert not axis_passes_qc(dict(base, loso_cosine=np.nan))


def test_direction_validity_separates_estimability_geometry_and_stability():
    base = {"status": "ok", "n": 12, "n_shafts": 2, "effective_rank": 2,
            "bootstrap_cosine": 0.9, "loso_cosine": 0.7}
    valid = assess_axis_direction_validity(base)
    assert valid["estimable"] is True
    assert valid["geometry_2d_supported"] is True
    assert valid["strict_stability_pass"] is True

    single = assess_axis_direction_validity(dict(base, n_shafts=1, loso_cosine=np.nan))
    assert single["estimable"] is True
    assert single["geometry_2d_supported"] is False
    assert single["strict_stability_pass"] is False
    assert "single_shaft_geometry" in single["reason_codes"]


def test_normalized_plane_and_field_score_recover_pattern():
    x, _ = _geometry(4)
    plane = make_normalized_plane(x, [1, 0, 0])
    assert plane["status"] == "ok"
    pts = plane["points"]
    template = 2 * pts[:, 0] - 0.5 * pts[:, 1]
    support = np.linspace(0.2, 1.0, len(x))
    scorer = make_field_scorer(template, pts, support, plane["sigma"])
    out = score_field(scorer, template)
    assert out["abs_r"] > 0.99


def test_interictal_record_freezes_early_to_late_own_fields_without_ictal_input():
    x, shafts = _geometry(31)
    true_a = np.array([0.8, -0.4, 0.25])
    true_a /= np.linalg.norm(true_a)
    true_b = np.array([-0.75, 0.5, -0.1])
    true_b /= np.linalg.norm(true_b)
    rank_a, rank_b = x @ true_a, x @ true_b
    support_a = np.linspace(0.2, 1.0, len(x))
    support_b = np.linspace(1.0, 0.2, len(x))
    names = [f"S{i // 6}{i % 6 + 1}" for i in range(len(x))]
    record = build_interictal_template_field_record(
        subject_id="test_subject", dataset="test", subject="subject", stable_k=2,
        names=names, coords=x, rank_ta=rank_a, rank_tb=rank_b, shafts=shafts,
        support_ta=support_a, support_tb=support_b, support_source="unit_test",
        n_axis_boot=30, n_pair_boot=40, seed=11,
    )
    assert record["contract"] == INTERICTAL_FIELD_CONTRACT
    assert record["axis_direction_convention"] == "positive_early_to_late"
    assert float(np.asarray(record["axis_pair"]["axis_a"]["u"]) @ true_a) > 0.99
    assert float(np.asarray(record["axis_pair"]["axis_b"]["u"]) @ true_b) > 0.99
    assert record["direction_validity"]["ta"]["estimable"] is True
    assert record["direction_validity"]["tb"]["estimable"] is True
    assert record["interictal_field"]["status"] == "ok"
    assert record["interictal_field"]["fingerprint_sha256"] == interictal_field_fingerprint(record)
    assert set(("own_a", "own_b")).issubset(record["interictal_field"]["field_models"])

    scorers = scorers_from_interictal_record(record)
    assert score_field(scorers["own_a"], -rank_a)["abs_r"] > 0.99

    tampered = dict(record)
    tampered["interictal_field"] = dict(record["interictal_field"])
    tampered["interictal_field"]["support_a"] = np.asarray(
        record["interictal_field"]["support_a"], float
    ).copy()
    tampered["interictal_field"]["support_a"][0] += 0.01
    try:
        scorers_from_interictal_record(tampered)
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("tampered frozen artifact must fail closed")


def test_future_activation_is_name_joined_to_frozen_field_contact_order():
    x, shafts = _geometry(32)
    names = [f"S{i // 6}{i % 6 + 1}" for i in range(len(x))]
    rank = x[:, 0]
    support = np.ones(len(x))
    record = build_interictal_template_field_record(
        subject_id="test_subject", dataset="test", subject="subject", stable_k=2,
        names=names, coords=x, rank_ta=rank, rank_tb=-rank, shafts=shafts,
        support_ta=support, support_tb=support, support_source="unit_test",
        n_axis_boot=20, n_pair_boot=20, seed=12,
    )
    shuffled_names = list(reversed(names))
    shuffled_values = np.arange(len(names), dtype=float)[::-1]
    aligned = align_activation_to_interictal_field(record, shuffled_names, shuffled_values)
    np.testing.assert_allclose(aligned["values"], np.arange(len(names), dtype=float))
    assert aligned["n_matched"] == len(names)
    assert aligned["missing_names"] == []


def test_flipping_axis_sign_only_relabels_along_coordinate_not_field_score():
    x, _ = _geometry(14)
    u = np.array([0.7, -0.2, 0.4])
    pos = make_normalized_plane(x, u)
    neg = make_normalized_plane(x, -u)
    assert pos["status"] == neg["status"] == "ok"
    np.testing.assert_allclose(pos["points"][:, 0], -neg["points"][:, 0])
    np.testing.assert_allclose(pos["points"][:, 1], neg["points"][:, 1])
    template = np.linspace(-1.0, 1.0, len(x))
    activation = np.sin(np.linspace(0.0, 2.0, len(x)))
    support = np.linspace(0.2, 1.0, len(x))
    score_pos = score_field(make_field_scorer(template, pos["points"], support, pos["sigma"]),
                            activation)
    score_neg = score_field(make_field_scorer(template, neg["points"], support, neg["sigma"]),
                            activation)
    for key in ("r_identity", "r_mirror", "signed_r", "abs_r"):
        assert np.isclose(score_pos[key], score_neg[key])


def test_mirror_candidate_is_selected_by_absolute_not_signed_maximum():
    # A deliberately asymmetric plane; mirrored activation is the negative template.
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(24, 2))
    template = pts[:, 0] + 0.8 * pts[:, 1]
    support = np.ones(len(pts))
    scorer = make_field_scorer(template, pts, support, 0.4)
    activation = -(pts[:, 0] - 0.8 * pts[:, 1])
    out = score_field(scorer, activation)
    # Contract check: returned candidate is literally the larger absolute candidate.
    expected = max(abs(out["r_identity"]), abs(out["r_mirror"]))
    assert np.isclose(out["abs_r"], expected)


def test_bundle_recomputes_maxab_from_template_scores():
    rng = np.random.default_rng(8)
    pts = rng.normal(size=(20, 2))
    support = np.ones(20)
    sa = make_field_scorer(pts[:, 0], pts, support, 0.5)
    sb = make_field_scorer(pts[:, 1], pts, support, 0.5)
    out = score_scorer_bundle({"own_a": sa, "own_b": sb}, pts[:, 1])
    assert np.isclose(out["own_maxab"], max(out["own_a_abs"], out["own_b_abs"]))


def test_batch_scores_equal_rowwise_scores_and_reselect_maxab():
    rng = np.random.default_rng(19)
    pts = rng.normal(size=(22, 2))
    support = rng.uniform(0.1, 1.0, len(pts))
    sa = make_field_scorer(pts[:, 0], pts, support, 0.45)
    sb = make_field_scorer(pts[:, 1], pts, support, 0.45)
    values = rng.normal(size=(7, len(pts)))
    values[2, 3] = np.nan
    bundle = {"own_a": sa, "own_b": sb}
    batch = score_scorer_bundle_batch(bundle, values)
    for i, row in enumerate(values):
        single = score_scorer_bundle(bundle, row)
        for key in ("own_a_abs", "own_b_abs", "own_maxab"):
            assert np.isclose(batch[key][i], single[key], equal_nan=True)
    field_batch = score_field_batch(sa, values)
    assert np.allclose(field_batch["abs_r"], batch["own_a_abs"], equal_nan=True)

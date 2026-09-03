import numpy as np
import pytest

from src.topic4_rev22_interictal_objective import (
    COMPONENTS, PatientBlockViews, STATUS_LOW_JOINT, STATUS_OK, block_split_floors,
    censor_shaft, clipping_fractions, component_vector, coverage_distance,
    embedding_features, fit_training_embedding, identifiability_ratio, lag_view,
    minimax_proposal_scalar, normalized_excess, order_view, patient_reference,
    permute_within_shaft, pooled_candidate, recruitment_profile, standardized_excess,
    stretch_onsets, support_view, thin_recruitment,
)

N_CONTACTS = 8
GROUPS = {"ICL": np.arange(5), "SCL": np.arange(5, 8)}


def _pairs():
    out = {"ICL-ICL": [], "SCL-SCL": [], "ICL-SCL": []}
    for i in range(N_CONTACTS):
        for j in range(i + 1, N_CONTACTS):
            a, b = ("ICL" if i < 5 else "SCL"), ("ICL" if j < 5 else "SCL")
            key = f"{a}-{b}" if a == b else "ICL-SCL"
            out[key].append((i, j))
    return {k: np.asarray(v, dtype=int).reshape((-1, 2)) for k, v in out.items()}


PAIRS = _pairs()


def _patient(n=600, seed=0, minority_fraction=0.3):
    """Two directions: mode 0 propagates ICL->SCL, mode 1 SCL->ICL, jitter in ms."""
    rng = np.random.default_rng(seed)
    onsets = np.full((n, N_CONTACTS), np.nan)
    labels = (rng.random(n) < minority_fraction).astype(int)
    blocks = rng.integers(0, 12, n)
    for e in range(n):
        base = np.arange(N_CONTACTS, dtype=float) * 6.0
        if labels[e] == 0:
            base = base[::-1]
        recruited = rng.random(N_CONTACTS) < 0.85
        recruited[rng.integers(0, N_CONTACTS)] = True
        onsets[e, recruited] = 100.0 + base[recruited] + rng.normal(0, 1.5, recruited.sum())
    return onsets, labels, blocks


def _reference(onsets):
    embedding = fit_training_embedding(embedding_features(onsets, GROUPS), seed=1)
    return patient_reference(onsets, GROUPS, PAIRS, embedding), embedding


def test_embedding_distinguishes_first_recruited_from_not_recruited():
    onsets = np.array([[100.0, 110.0, np.nan, 120.0, 130.0, np.nan, 140.0, 150.0]])
    x = embedding_features(onsets, GROUPS)
    mask, order_block, physical_block = x[0, :8], x[0, 8:16], x[0, 16:24]
    assert mask[0] == 1.0 and mask[2] == 0.0
    assert order_block[0] == 1.0 and order_block[2] == 0.0
    assert physical_block[0] == 1.0 and physical_block[2] == 0.0
    assert x.shape[1] == 3 * N_CONTACTS + 4


def test_views_have_no_label_input_and_missing_contacts_stay_explicit():
    onsets, _, _ = _patient(50)
    view = support_view(onsets, GROUPS, PAIRS)
    assert view["n_events"] == 50
    assert np.isclose(view["count_histogram"].sum(), 1.0)
    order = order_view(onsets, PAIRS)
    for c in PAIRS:
        assert order["counts"][c].shape == (len(PAIRS[c]), 3)
        assert np.all(order["counts"][c].sum(axis=1) == order["joint_count"][c])


def test_order_and_lag_ignore_not_jointly_recruited_pairs():
    onsets = np.array([
        [100.0, 105.0, np.nan, np.nan, np.nan, 130.0, np.nan, np.nan],
        [100.0, np.nan, np.nan, np.nan, np.nan, 130.0, np.nan, np.nan],
    ])
    order = order_view(onsets, PAIRS)
    # pair (0,1) jointly recruited once -> exactly one state count in total
    idx = [k for k, (i, j) in enumerate(PAIRS["ICL-ICL"]) if (i, j) == (0, 1)][0]
    assert order["counts"]["ICL-ICL"][idx].sum() == 1
    assert order["joint_count"]["ICL-ICL"][idx] == 1
    lags = lag_view(onsets, PAIRS)
    assert len(lags["ICL-ICL"][idx]) == 1 and lags["ICL-ICL"][idx][0] == 5.0


def test_low_joint_support_is_flagged_not_zeroed():
    onsets, _, _ = _patient(400)
    reference, embedding = _reference(onsets)
    tiny = onsets[:3]
    vec = component_vector(tiny, reference, GROUPS, PAIRS, embedding)
    assert vec["D_order"]["status"] == STATUS_LOW_JOINT
    assert vec["D_lag"]["status"] == STATUS_LOW_JOINT
    assert vec["D_support"]["status"] == STATUS_OK
    assert vec["D_cover"]["status"] == STATUS_OK


def test_self_comparison_is_near_zero_and_controls_move_only_their_views():
    onsets, labels, blocks = _patient(600)
    reference, embedding = _reference(onsets)
    rng = np.random.default_rng(3)
    sample = onsets[rng.choice(len(onsets), 60, replace=False)]
    base = component_vector(sample, reference, GROUPS, PAIRS, embedding)
    assert base["D_order"]["status"] == STATUS_OK and base["D_lag"]["status"] == STATUS_OK

    # time stretch: lag worsens, support and order exactly unchanged
    stretched = component_vector(stretch_onsets(sample, 2.0), reference, GROUPS, PAIRS, embedding)
    assert stretched["D_lag"]["value"] > base["D_lag"]["value"]
    assert stretched["D_support"]["value"] == pytest.approx(base["D_support"]["value"], abs=1e-12)
    assert stretched["D_order"]["value"] == pytest.approx(base["D_order"]["value"], abs=1e-12)

    # within-shaft permutation: order worsens, support exactly unchanged
    permuted = component_vector(permute_within_shaft(sample, GROUPS, rng), reference, GROUPS, PAIRS, embedding)
    assert permuted["D_order"]["value"] > base["D_order"]["value"]
    assert permuted["D_support"]["value"] == pytest.approx(base["D_support"]["value"], abs=1e-12)

    # SCL censoring: support worsens, ICL-ICL order class exactly unchanged
    censored = component_vector(censor_shaft(sample, GROUPS, "SCL"), reference, GROUPS, PAIRS, embedding)
    assert censored["D_support"]["value"] > base["D_support"]["value"]
    assert censored["D_order"]["per_class"]["ICL-ICL"] == pytest.approx(
        base["D_order"]["per_class"]["ICL-ICL"], abs=1e-12)
    assert censored["D_order"]["status"] == STATUS_LOW_JOINT

    # minority removal: cover and order worsen without any label reaching the objective
    majority = onsets[labels == 1]
    sample_major = majority[rng.choice(len(majority), 60, replace=False)]
    collapsed = component_vector(sample_major, reference, GROUPS, PAIRS, embedding)
    assert collapsed["D_cover"]["value"] > base["D_cover"]["value"]
    assert collapsed["D_order"]["value"] > base["D_order"]["value"]


def test_coverage_uses_all_queries_and_penalizes_low_yield():
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(500, 3))
    dense = coverage_distance(queries[:200], queries)["value"]
    sparse = coverage_distance(queries[:5], queries)["value"]
    assert sparse > dense
    assert coverage_distance(np.zeros((0, 3)), queries)["value"] is None


def test_block_views_merge_equals_direct_reference():
    onsets, _, blocks = _patient(300)
    _, embedding = _reference(onsets)
    views = PatientBlockViews(onsets, blocks, GROUPS, PAIRS, embedding)
    subset = views.blocks[:5]
    merged = views.reference(subset)
    mask = np.isin(blocks, subset)
    direct = patient_reference(onsets[mask], GROUPS, PAIRS, embedding)
    for shaft in ("ICL", "SCL"):
        assert np.allclose(merged["support"]["recruitment"][shaft], direct["support"]["recruitment"][shaft])
    assert np.allclose(merged["support"]["count_histogram"], direct["support"]["count_histogram"])
    for c in PAIRS:
        assert np.array_equal(merged["order"]["counts"][c], direct["order"]["counts"][c])
        for a, b in zip(merged["lag"][c], direct["lag"][c]):
            assert np.array_equal(np.sort(a), np.sort(b))
    assert merged["z"].shape == direct["z"].shape


def test_thinning_matches_target_profile_and_keeps_onsets():
    onsets, _, _ = _patient(2000)
    rng = np.random.default_rng(5)
    source = recruitment_profile(onsets)
    target = source * np.array([1.0, 0.5, 0.25, 1.0, 0.5, 0.25, 1.0, 0.5])
    thinned = thin_recruitment(onsets, target, source, rng)
    got = recruitment_profile(thinned)
    assert np.allclose(got, target, atol=0.04)
    kept = np.isfinite(thinned)
    assert np.array_equal(thinned[kept], onsets[kept])
    assert np.all(kept <= np.isfinite(onsets))


def test_block_split_floors_are_deterministic_and_support_thinned_requests():
    onsets, _, blocks = _patient(300)
    _, embedding = _reference(onsets)
    views = PatientBlockViews(onsets, blocks, GROUPS, PAIRS, embedding)
    profile = recruitment_profile(onsets) * 0.6
    requests = [
        {"key": "n20", "n": 20, "components": ("D_support", "D_cover"), "thin_profile": None},
        {"key": "cand", "n": 40, "components": ("D_order", "D_lag"), "thin_profile": profile},
    ]
    a = block_split_floors(views, requests, draws=4, seed=11)
    b = block_split_floors(views, requests, draws=4, seed=11)
    assert a == b
    assert set(a["n20"]) == {"D_support", "D_cover"} and set(a["cand"]) == {"D_order", "D_lag"}
    for key in a:
        for k, f in a[key].items():
            assert f["draws"] >= 1 and f["q05"] <= f["q50"] <= f["q95"]


def test_pooled_candidate_jackknife_and_units():
    onsets, _, _ = _patient(400)
    reference, embedding = _reference(onsets)
    rng = np.random.default_rng(9)
    units = [onsets[rng.choice(len(onsets), 30, replace=False)] for _ in range(4)]
    out = pooled_candidate(units, reference, GROUPS, PAIRS, embedding)
    assert out["n_units"] == 4 and out["n_pooled_events"] == 120 and len(out["per_unit"]) == 4
    for k in COMPONENTS:
        assert out["pooled"][k]["status"] == STATUS_OK
        assert out["jackknife_sd"][k] is not None and out["jackknife_sd"][k] >= 0.0
    assert out["recruitment_profile"].shape == (N_CONTACTS,)


def test_excess_identifiability_and_minimax():
    assert normalized_excess(0.5, {"q50": 0.4, "q95": 0.6}) == pytest.approx(0.5, rel=1e-6)
    assert normalized_excess(0.3, {"q50": 0.4, "q95": 0.6}) == 0.0
    assert normalized_excess(None, {"q50": 0.4, "q95": 0.6}) is None
    assert standardized_excess(0.3, {"q50": 0.4, "q95": 0.6}) == pytest.approx(-0.5, rel=1e-6)
    strong = identifiability_ratio({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 0.01, "b": 0.02, "c": 0.01})
    weak = identifiability_ratio({"a": 1.0, "b": 1.1, "c": 0.9}, {"a": 1.0, "b": 1.0, "c": 1.0})
    assert strong["ratio"] > 1.0 > weak["ratio"]
    assert identifiability_ratio({"a": 1.0}, {"a": 0.1})["ratio"] is None
    assert minimax_proposal_scalar({"D_support": 0.2, "D_order": 1.5, "D_lag": None}, ["D_support", "D_order"]) == 1.5
    assert minimax_proposal_scalar({"D_support": 0.2, "D_lag": None}, ["D_support", "D_lag"]) is None
    assert minimax_proposal_scalar({"D_support": 0.2}, []) is None


def test_clipping_fractions():
    onsets = np.array([[0.0, 100.0, 400.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [0.0, 10.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    out = clipping_fractions(onsets)
    assert out["event_contact_fraction"] == pytest.approx(1 / 5)
    assert out["any_event_fraction"] == pytest.approx(0.5)

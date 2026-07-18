import numpy as np
import pytest

from scripts.run_topic5_old_cache_endpoint_vs_gradient_ownab import (
    align_activation_to_names,
    build_controlled_ownab_scorers,
    score_controlled_event,
    select_event_ids,
)
from src.topic5_tspectral_field_concordance import make_contact_permutations


def _synthetic_inputs():
    names = ["A1", "A2", "A3", "B1", "B2", "B3", "C1"]
    gradient_a = np.array([
        [0.00, 0.00], [0.16, 0.05], [0.34, -0.04], [0.52, 0.12],
        [0.68, -0.10], [0.83, 0.04], [1.00, 0.00],
    ])
    gradient_b = np.array([
        [1.00, 0.03], [0.84, -0.04], [0.66, 0.10], [0.49, -0.09],
        [0.31, 0.05], [0.15, -0.02], [0.00, 0.01],
    ])
    field_record = {
        "interictal_field": {
            "status": "ok",
            "contact_order": names,
            "earliness_a": np.linspace(1.0, -1.0, len(names)),
            "earliness_b": np.linspace(-1.0, 1.0, len(names)),
            "support_a": np.linspace(0.5, 1.0, len(names)),
            "support_b": np.linspace(1.0, 0.5, len(names)),
            "planes": {
                "own_a": {"points": gradient_a},
                "own_b": {"points": gradient_b},
            },
        }
    }
    endpoint_a_points = {
        name: [i / 6, (i % 3 - 1) * 0.2] for i, name in enumerate(names)
    }
    endpoint_b_points = {
        name: [(6 - i) / 6, ((i + 1) % 3 - 1) * 0.15] for i, name in enumerate(names)
    }
    endpoint_a = {"channels": [
        {"name": name, "x_norm": endpoint_a_points[name][0],
         "y_norm": endpoint_a_points[name][1]}
        for name in reversed(names)
    ]}
    endpoint_b = {"channels": [
        {"name": name, "x_norm": endpoint_b_points[name][0],
         "y_norm": endpoint_b_points[name][1]}
        for name in names[2:] + names[:2]
    ]}
    return field_record, endpoint_a, endpoint_b, endpoint_a_points, endpoint_b_points


def test_controlled_ownab_uses_frozen_order_and_only_changes_plane_geometry():
    field, endpoint_a, endpoint_b, endpoint_a_points, endpoint_b_points = _synthetic_inputs()
    result = build_controlled_ownab_scorers(
        field, endpoint_a, endpoint_b, validate_fingerprint=False
    )
    assert result["contact_order"] == field["interictal_field"]["contact_order"]
    expected_a = np.asarray([endpoint_a_points[name] for name in result["contact_order"]])
    expected_b = np.asarray([endpoint_b_points[name] for name in result["contact_order"]])
    np.testing.assert_allclose(result["points"]["endpoint"]["own_a"], expected_a)
    np.testing.assert_allclose(result["points"]["endpoint"]["own_b"], expected_b)
    np.testing.assert_allclose(
        result["scorers"]["endpoint"]["own_a"]["support"],
        result["scorers"]["gradient"]["own_a"]["support"],
    )
    np.testing.assert_allclose(
        result["scorers"]["endpoint"]["own_b"]["support"],
        result["scorers"]["gradient"]["own_b"]["support"],
    )
    assert all(
        result["sigmas"][representation][template] > 0
        for representation in ("endpoint", "gradient")
        for template in ("own_a", "own_b")
    )


def test_endpoint_a_and_b_keep_their_own_planes():
    field, endpoint_a, endpoint_b, _, _ = _synthetic_inputs()
    result = build_controlled_ownab_scorers(
        field, endpoint_a, endpoint_b, validate_fingerprint=False
    )
    assert not np.allclose(
        result["points"]["endpoint"]["own_a"],
        result["points"]["endpoint"]["own_b"],
    )


def test_same_permutation_matrix_is_used_for_both_representations():
    field, endpoint_a, endpoint_b, _, _ = _synthetic_inputs()
    result = build_controlled_ownab_scorers(
        field, endpoint_a, endpoint_b, validate_fingerprint=False
    )
    identical = {
        "endpoint": result["scorers"]["gradient"],
        "gradient": result["scorers"]["gradient"],
    }
    activation = np.linspace(-0.4, 1.3, len(result["contact_order"]))
    permutations = make_contact_permutations(
        result["contact_order"], np.ones(len(activation), bool), 40, 91,
        mode="all_contact",
    )
    scored = score_controlled_event(identical, activation, permutations)
    assert scored["endpoint"]["observed"] == pytest.approx(scored["gradient"]["observed"])
    np.testing.assert_allclose(scored["endpoint"]["null"], scored["gradient"]["null"])


def test_controlled_activation_join_is_by_exact_name_not_array_position():
    aligned = align_activation_to_names(
        ["B2", "A1", "A2"], [20.0, 10.0, 11.0], ["A1", "A2", "B2", "C1"]
    )
    np.testing.assert_allclose(aligned[:3], [10.0, 11.0, 20.0])
    assert np.isnan(aligned[3])


def test_controlled_ownab_fails_closed_below_six_common_contacts():
    field, endpoint_a, endpoint_b, _, _ = _synthetic_inputs()
    endpoint_b["channels"] = endpoint_b["channels"][:5]
    with pytest.raises(ValueError, match="fewer_than_6_common_contacts"):
        build_controlled_ownab_scorers(
            field, endpoint_a, endpoint_b, validate_fingerprint=False
        )


def test_strict_broadband_selector_is_intersected_with_old_cache_eligibility():
    assert select_event_ids(
        [1, 2, 4, 8], "accepted_strict_broadband", {0, 2, 8, 11}
    ) == [2, 8]
    assert select_event_ids([4, 1], "all_old_eligible", {1}) == [1, 4]


def test_gradient_prefers_predeclared_shared_plane_and_recomputes_shared_maxab():
    field, endpoint_a, endpoint_b, _, _ = _synthetic_inputs()
    own_a = np.asarray(field["interictal_field"]["planes"]["own_a"]["points"])
    shared = own_a.copy()
    shared[:, 0] = np.linspace(0.1, 0.9, len(shared))
    shared[:, 1] = np.linspace(-0.25, 0.25, len(shared))
    field["interictal_field"]["planes"]["shared"] = {"points": shared}
    field["interictal_field"]["field_models"] = {
        "shared_a": {}, "shared_b": {}
    }
    result = build_controlled_ownab_scorers(
        field,
        endpoint_a,
        endpoint_b,
        validate_fingerprint=False,
        gradient_field_policy="shared_else_own",
    )
    assert result["gradient_field_plane"] == "shared"
    assert result["score_prefixes"]["gradient"] == "shared"
    assert set(result["scorers"]["gradient"]) == {"shared_a", "shared_b"}
    np.testing.assert_allclose(result["points"]["gradient"]["shared_a"], shared)
    np.testing.assert_allclose(result["points"]["gradient"]["shared_b"], shared)

    activation = np.linspace(-1.0, 1.0, len(result["contact_order"]))
    permutations = make_contact_permutations(
        result["contact_order"], np.ones(len(activation), bool), 40, 19,
        mode="all_contact",
    )
    scored = score_controlled_event(
        result["scorers"], activation, permutations, result["score_prefixes"]
    )
    assert scored["gradient"]["field_prefix"] == "shared"
    assert scored["gradient"]["observed"] == pytest.approx(
        max(scored["gradient"]["a_abs"], scored["gradient"]["b_abs"])
    )
    assert scored["gradient"]["null"].shape == (40,)

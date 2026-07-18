import numpy as np

from scripts.paper_figures.plot_interictal_ab_direction_axis import (
    _fit_order_on_axis,
    _lateral_brain_coordinates,
    _lateral_propagation_direction,
    _route_endpoints,
)


def test_propagation_direction_is_early_to_late_not_earliness_gradient():
    along = np.linspace(-10.0, 10.0, 12)
    # Template A is early at +axis and late at -axis, so its propagation is -axis.
    order_a = np.linspace(1.0, 0.0, 12)
    # Template B is the reversed read: early at -axis and late at +axis.
    order_b = np.linspace(0.0, 1.0, 12)
    fit_a = _fit_order_on_axis(along, order_a)
    fit_b = _fit_order_on_axis(along, order_b)
    assert fit_a["propagation_sign_on_shared_axis"] == -1
    assert fit_b["propagation_sign_on_shared_axis"] == 1
    assert fit_a["pearson_r"] < -0.99
    assert fit_b["pearson_r"] > 0.99


def test_region_route_excludes_unresolved_labels_and_requires_repeated_contacts():
    regions = np.asarray(
        [
            "Superior frontal",
            "Superior frontal",
            "Deep / white matter",
            "Unassigned",
            "Pars opercularis",
            "Pars opercularis",
        ],
        dtype=object,
    )
    order = np.asarray([0.05, 0.15, 0.0, 1.0, 0.85, 0.95])
    route = _route_endpoints(regions, order)
    assert route["status"] == "ok"
    assert route["early_region"] == "Superior frontal"
    assert route["late_region"] == "Pars opercularis"
    assert all(row[0] not in {"Deep / white matter", "Unassigned"} for row in route["eligible_regions"])


def test_lateral_projection_draws_early_to_late_not_earliness_direction():
    direction, retained = _lateral_propagation_direction([0.0, 0.8, 0.6])
    # The input is already the early-to-late propagation vector; do not flip it again.
    np.testing.assert_allclose(direction, [-0.8, 0.6])
    assert retained == 1.0


def test_lateral_coordinate_map_puts_anterior_left_and_superior_up():
    case = {
        "mesh_vertices": np.asarray([[0.0, -10.0, -20.0], [0.0, 30.0, 40.0]]),
        "coords": np.asarray([[0.0, 25.0, -10.0], [0.0, -5.0, 30.0]]),
    }
    xy, _ = _lateral_brain_coordinates(case)
    assert xy[0, 0] < xy[1, 0]
    assert xy[0, 1] < xy[1, 1]

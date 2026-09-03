import numpy as np
from scipy import sparse

from scripts.audit_topic4_rev22_geometry_domain import (
    THRESHOLDS, assess_achieved_geometry, audit_grid_point, ee_edges,
    effective_source_count, grid_point_passes, largest_admissible_rectangle,
    validate_fast_path, weighted_connection_geometry,
)


def _toy_net(seed=0, n_e=40, n_i=8, in_degree=12):
    """Small anisotropic sparse E-to-E graph with two delay bins plus one GABA bin."""
    rng = np.random.default_rng(seed)
    n = n_e + n_i
    pos = rng.uniform(0.0, 4.0, size=(n, 2))
    bins = [sparse.lil_matrix((n, n_e)), sparse.lil_matrix((n, n_e))]
    for target in range(n):
        sources = rng.choice([s for s in range(n_e) if s != target], size=in_degree, replace=False)
        for source in sources:
            bins[int(rng.integers(0, 2))][target, source] = rng.uniform(0.2, 1.0)
    gaba = sparse.csc_matrix(rng.uniform(0.0, 1.0, size=(n, n_i)))
    return {
        "NE": n_e, "NI": n_i, "pos": pos,
        "ampa_by_delay": [b.tocsc() for b in bins],
        "gaba_by_delay": [gaba],
    }


def test_effective_source_count_matches_definition():
    rows = np.array([0, 0, 0, 1, 1, 2])
    data = np.array([1.0, 1.0, 1.0, 2.0, 0.5, 3.0])
    eff = effective_source_count(rows, data, n_e=4)
    assert eff[0] == 3.0                                      # three equal inputs
    assert np.isclose(eff[1], (2.5 ** 2) / (4.0 + 0.25))
    assert eff[2] == 1.0                                      # one input
    assert np.isnan(eff[3])                                   # no input


def test_weighted_connection_geometry_recovers_axial_orientation_and_aspect():
    angle = np.deg2rad(30.0)
    axis = np.array([np.cos(angle), np.sin(angle)])
    cross = np.array([-np.sin(angle), np.cos(angle)])
    displacement = np.vstack([axis * 3.0, -axis * 3.0, cross, -cross])
    geometry = weighted_connection_geometry(
        displacement[:, 0], displacement[:, 1], np.ones(4),
    )
    np.testing.assert_allclose(geometry["achieved_angle_deg"], 30.0, atol=1e-10)
    np.testing.assert_allclose(geometry["achieved_aspect_ratio"], 3.0, atol=1e-10)


def test_achieved_geometry_requires_response_above_topology_variability():
    theta = np.array([35.0, 45.0, 55.0])
    aspect = np.array([1.5, 2.0, 2.5])
    per_topology = {}
    for seed, jitter in zip((1, 2, 3, 4), (-0.2, -0.1, 0.1, 0.2)):
        records = []
        for i, requested_theta in enumerate(theta):
            for j, requested_aspect in enumerate(aspect):
                records.append({
                    "i": i, "j": j,
                    "achieved_angle_deg": requested_theta + jitter,
                    "achieved_aspect_ratio": requested_aspect * np.exp(jitter / 20.0),
                })
        per_topology[seed] = {"records": records}
    rectangle = {"i0": 0, "i1": 2, "j0": 0, "j1": 2}
    result = assess_achieved_geometry(per_topology, rectangle, theta, aspect, 1, 1)
    assert result["pass"] is True
    for block in per_topology.values():
        for record in block["records"]:
            record["achieved_angle_deg"] = 45.0
    failed = assess_achieved_geometry(per_topology, rectangle, theta, aspect, 1, 1)
    assert failed["pass"] is False


def _graph(net, base):
    pos = net["pos"]
    return {"n_e": base["n_e"], "row": base["row"], "data": base["data"],
            "dx": pos[base["col"], 0] - pos[base["row"], 0],
            "dy": pos[base["col"], 1] - pos[base["row"], 1]}


def test_reference_is_noop_and_off_reference_conserves_budget():
    net = _toy_net()
    base = ee_edges(net)
    graph = _graph(net, base)
    base_eff = effective_source_count(base["row"], base["data"], base["n_e"])
    ref = audit_grid_point(graph, base, base_eff, length_scale=0.38,
                           angle_deg=45.0, aspect_ratio=2.0)
    assert ref["status"] == "OK" and ref["exact_noop"]
    assert ref["edge_ratio_p01"] == ref["edge_ratio_p99"] == 1.0
    assert ref["effective_source_median_ratio"] == 1.0 and ref["budget_error_max"] == 0.0
    assert grid_point_passes(ref)
    off = audit_grid_point(graph, base, base_eff, length_scale=0.38,
                           angle_deg=30.0, aspect_ratio=1.5)
    assert off["status"] == "OK" and not off["exact_noop"]
    assert off["budget_error_max"] <= THRESHOLDS["budget_error_max"]
    assert off["edge_ratio_p01"] < 1.0 < off["edge_ratio_p99"]
    assert 0.0 < off["effective_source_p05_ratio"] <= off["effective_source_median_ratio"]
    # the frozen base must not be mutated by the audit
    after = ee_edges(net)
    assert np.array_equal(after["data"], base["data"])


def test_fast_path_reproduces_the_producer():
    net = _toy_net(seed=3)
    base = ee_edges(net)
    graph = _graph(net, base)
    checks = validate_fast_path(
        net, net["pos"], graph, base,
        [(45.0, 2.0), (22.5, 1.0), (67.5, 3.0), (45.0, 3.0)], length_scale=0.38,
    )
    assert len(checks) == 4
    assert checks[0]["producer_exact_noop"] and checks[0]["max_abs_difference"] == 0.0
    for check in checks:
        assert check["max_relative_difference"] <= 1e-12
        assert check["producer_topology_unchanged"] and check["producer_gaba_unchanged"]


def test_largest_rectangle_contains_reference_and_prefers_symmetry():
    mask = np.zeros((9, 5), bool)          # axis 0 theta, axis 1 AR
    mask[2:8, 1:4] = True                  # 6 x 3 block containing the reference (4, 2)
    mask[7, 3] = False                     # notch in one corner
    rect = largest_admissible_rectangle(mask, 4, 2, theta_values=np.arange(9) * 2.5 + 35.0,
                                        reference_angle=45.0)
    # 5 x 3 = 15 beats the 6 x 2 = 12 alternative the notch leaves behind
    assert (rect["i0"], rect["i1"], rect["j0"], rect["j1"]) == (2, 6, 1, 3)
    assert rect["cells"] == 15
    # genuine tie: two 6-cell maximal rectangles; the theta-symmetric one wins
    tie = np.zeros((5, 4), bool)
    tie[1:4, 1:3] = True                   # theta 1..3 x AR 1..2, centered on reference theta
    tie[2:4, 0:3] = True                   # theta 2..3 x AR 0..2, shifted off the reference
    rect = largest_admissible_rectangle(tie, 2, 1, theta_values=np.arange(5) * 2.5 + 40.0,
                                        reference_angle=45.0)
    assert (rect["i0"], rect["i1"], rect["j0"], rect["j1"]) == (1, 3, 1, 2)
    assert rect["cells"] == 6 and rect["theta_asymmetry"] == 0.0
    # reference cell failing -> no rectangle
    bad = mask.copy()
    bad[4, 2] = False
    assert largest_admissible_rectangle(bad, 4, 2) is None
    # only the reference passes -> single cell
    single = np.zeros((5, 5), bool)
    single[2, 2] = True
    assert largest_admissible_rectangle(single, 2, 2)["cells"] == 1


def test_graph_axis_reference_makes_realized_geometry_track_the_request():
    """Amendment v5.1 premise: with the reference bound to the graph's own kernel axis,
    the realized weighted geometry moves monotonically with the requested offset/aspect."""
    import numpy as np
    from scipy import sparse
    from scipy.stats import spearmanr
    from src.topic4_rev20_dual_core_mechanism import fixed_topology_ee_ellipse_redistribution
    from scripts.audit_topic4_rev22_geometry_domain import weighted_connection_geometry

    rng = np.random.default_rng(0)
    theta_ref, ar_ref, l, L, n, n_e, k = -22.80538396505847, 2.0, 0.38, 20.0, 1500, 1200, 30
    pos = rng.uniform(0.0, L, size=(n, 2))
    c, s = np.cos(np.radians(theta_ref)), np.sin(np.radians(theta_ref))
    l_par, l_perp = l * np.sqrt(ar_ref), l / np.sqrt(ar_ref)
    rows, cols = [], []
    for t in range(n_e):
        dz = pos[:n_e] - pos[t]
        u, v = c * dz[:, 0] + s * dz[:, 1], -s * dz[:, 0] + c * dz[:, 1]
        w = np.exp(-np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2))
        w[t] = 0.0
        keys = rng.standard_exponential(n_e) / np.where(w > 0, w, np.inf)
        src = np.argpartition(keys, k)[:k]
        rows += [t] * k
        cols += list(src)
    net = {"NE": n_e, "NI": n - n_e,
           "ampa_by_delay": [sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n_e))],
           "gaba_by_delay": [sparse.csr_matrix((n, n - n_e))]}

    def achieved(bins):
        m = bins[0].tocoo()
        keep = m.row < n_e
        d = pos[m.col[keep]] - pos[m.row[keep]]
        g = weighted_connection_geometry(d[:, 0], d[:, 1], m.data[keep])
        return g["achieved_angle_deg"], g["achieved_aspect_ratio"]

    base_angle, base_aspect = achieved(net["ampa_by_delay"])
    assert abs(((base_angle - theta_ref) + 90.0) % 180.0 - 90.0) < 6.0
    same, audit = fixed_topology_ee_ellipse_redistribution(
        net, pos, length_scale=l, angle_deg=theta_ref, aspect_ratio=ar_ref,
        reference_angle_deg=theta_ref, reference_aspect_ratio=ar_ref)
    assert audit["exact_noop"]
    offsets = [-15.0, 0.0, 15.0]
    angles = []
    for off in offsets:
        new, _ = fixed_topology_ee_ellipse_redistribution(
            net, pos, length_scale=l, angle_deg=theta_ref + off, aspect_ratio=ar_ref,
            reference_angle_deg=theta_ref, reference_aspect_ratio=ar_ref)
        a, _ = achieved(new["ampa_by_delay"])
        angles.append(((a - base_angle) + 90.0) % 180.0 - 90.0)
    assert spearmanr(offsets, angles).statistic == 1.0
    aspects = []
    for ar in (1.5, 2.0, 3.0):
        new, _ = fixed_topology_ee_ellipse_redistribution(
            net, pos, length_scale=l, angle_deg=theta_ref, aspect_ratio=ar,
            reference_angle_deg=theta_ref, reference_aspect_ratio=ar_ref)
        aspects.append(achieved(new["ampa_by_delay"])[1])
    assert spearmanr([1.5, 2.0, 3.0], aspects).statistic == 1.0

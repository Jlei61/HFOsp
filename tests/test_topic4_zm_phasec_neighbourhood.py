"""Phase-C full-field neighbourhood construction and fail-closed verdict."""
import os
import sys
from io import BytesIO

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_phasec_neighbourhood as C  # noqa: E402


def _observed():
    out = {}
    for p, phase in enumerate(C.DEFAULT_PHASES):
        out[phase] = {}
        for i, stage in enumerate(C.PRIMARY_STAGES):
            out[phase][stage] = {
                "z": np.array([0.82, 0.72, 0.62]) - 0.04 * i - 0.01 * p,
                "m": np.array([1.0, 2.0, 3.0]) + 0.5 * i + 0.1 * p,
                "S_G": 0.05 + 0.02 * i + 0.005 * p,
            }
    return out


def test_primary_path_is_full_field_convex_and_never_clipped():
    obs = _observed()
    rows = C.build_primary_convex_path(obs)
    assert len(rows) == 10
    assert all(r["status"] == "valid" for r in rows)
    mid = rows[1]["state"]
    np.testing.assert_allclose(
        mid["z"], 0.5 * (
            obs["rising"]["bounded_early"]["z"]
            + obs["rising"]["bounded_mid"]["z"]
        )
    )
    np.testing.assert_allclose(
        mid["m"], 0.5 * (
            obs["rising"]["bounded_early"]["m"]
            + obs["rising"]["bounded_mid"]["m"]
        )
    )
    assert rows[4]["state"]["S_G"] == obs["rising"]["bounded_late"]["S_G"]
    assert rows[9]["state"]["S_G"] == obs["peak"]["bounded_late"]["S_G"]
    assert {r["trajectory_id"] for r in rows} == {"rising", "peak"}

    bad = _observed()
    bad["peak"]["bounded_late"]["z"][0] = -0.1
    bad_rows = C.build_primary_convex_path(bad)
    assert bad_rows[-1]["status"] == "invalid_physical"
    assert bad_rows[-1]["state"]["z"][0] == -0.1  # invalid, never clipped to zero


def test_secondary_shell_is_locked_to_quarter_sd_and_fail_closed():
    obs = _observed()
    # Give the six observed full fields two genuine non-tangent residual modes;
    # the simpler helper above is deliberately almost rank-one.
    obs["rising"]["bounded_mid"]["z"] += np.array([0.01, -0.02, 0.03])
    obs["rising"]["bounded_late"]["m"] += np.array([0.04, -0.01, 0.02])
    obs["peak"]["bounded_early"]["z"] += np.array([-0.02, 0.03, -0.01])
    obs["peak"]["bounded_mid"]["m"] += np.array([0.02, 0.01, -0.03])
    n = len(obs["rising"]["bounded_mid"]["z"])
    parallel = np.concatenate([np.array([-1.0, 0.0, 1.0]), np.zeros(n + 1)])
    perpendicular = np.concatenate([np.array([1.0, -2.0, 1.0]), np.zeros(n + 1)])
    directions = {
        "pathology_parallel": parallel,
        "pathology_perpendicular": perpendicular,
    }
    core = np.array([True, False, False])
    axis = np.array([-1.0, 0.0, 1.0])
    envelopes = C.fit_physical_envelopes(
        [obs[p][s] for p in C.DEFAULT_PHASES for s in C.PRIMARY_STAGES],
        core, axis,
    )
    rows = C.build_secondary_shell(
        obs, pathology_directions=directions, core_mask=core,
        axis_coord=axis,
        envelopes=envelopes,
    )
    assert len(rows) == 8
    assert {r["step_robust_sd"] for r in rows} == {0.25}
    assert {r["basis_direction"] for r in rows} == {
        "fullfield_mode2", "fullfield_mode3",
        "pathology_parallel", "pathology_perpendicular",
    }
    assert all(r["status"] in {"valid", "invalid_physical"} for r in rows)
    try:
        C.build_secondary_shell(
            obs, pathology_directions=directions, core_mask=core,
            axis_coord=axis,
            envelopes=envelopes, step_sd=0.5,
        )
    except ValueError as exc:
        assert "locked" in str(exc)
    else:
        raise AssertionError("an oversized shell must be rejected")


def test_primary_names_and_two_phases_match_the_locked_ten_cells():
    rows = C.build_primary_convex_path(_observed())
    assert tuple(r["cell_id"] for r in rows) == C.PRIMARY_CELL_NAMES
    assert [r["cell_id"] for r in rows[:5]] == [
        "primary__rising__bounded_early",
        "primary__rising__early_mid_midpoint",
        "primary__rising__bounded_mid",
        "primary__rising__mid_late_midpoint",
        "primary__rising__bounded_late",
    ]
    assert [r["cell_id"] for r in rows[5:]] == [
        x.replace("rising", "peak") for x in C.PRIMARY_CELL_NAMES[:5]
    ]


def test_physical_gate_checks_hard_full_field_and_summary_envelopes_without_clipping():
    obs = _observed()
    observed = [obs[p][s] for p in C.DEFAULT_PHASES for s in C.PRIMARY_STAGES]
    core = np.array([True, False, False])
    axis = np.array([-1.0, 0.0, 1.0])
    env = C.fit_physical_envelopes(observed, core, axis)
    state = {
        "z": observed[0]["z"].copy(),
        "m": observed[0]["m"].copy(),
        "S_G": observed[0]["S_G"],
    }
    assert C.physical_status(
        state, full_field_envelope=env["full_field"],
        summary_envelope=env["summary7"], core_mask=core, axis_coord=axis,
    )["status"] == "valid"

    outlier = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state.items()}
    outlier["z"][0] = 1.2
    result = C.physical_status(
        outlier, full_field_envelope=env["full_field"],
        summary_envelope=env["summary7"], core_mask=core, axis_coord=axis,
    )
    assert result["status"] == "invalid_physical"
    assert "z_physical_boundary" in result["reasons"]
    assert outlier["z"][0] == 1.2
    assert result["clipped"] is False

    # A coordinated within-hard-bound core shift violates the seven-summary
    # envelope and remains unmodified.
    summary_bad = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state.items()}
    summary_bad["z"][0] += 0.12
    wide = (
        np.full_like(C._pack(summary_bad), -10.0),
        np.full_like(C._pack(summary_bad), 10.0),
    )
    result = C.physical_status(
        summary_bad, full_field_envelope=wide,
        summary_envelope=env["summary7"], core_mask=core, axis_coord=axis,
    )
    assert "summary7_envelope" in result["reasons"]


def test_geometry_directions_are_seed_local_and_parallel_perpendicular():
    obs = _observed()
    # Create nonzero empirical component scales in all three synthetic cells.
    obs["rising"]["bounded_mid"]["z"] += np.array([0.01, -0.01, 0.02])
    obs["peak"]["bounded_mid"]["m"] += np.array([-0.02, 0.01, 0.03])
    along = np.array([-1.0, 0.0, 1.0])
    perp = np.array([1.0, -2.0, 1.0])
    dirs = C.pathology_directions_from_geometry(
        obs, axis_coord=along, perpendicular_coord=perp
    )
    assert dirs["pathology_parallel"].shape == (7,)
    assert dirs["pathology_perpendicular"].shape == (7,)
    assert abs(np.dot(dirs["axis_coord"], dirs["perpendicular_coord"])) < 1e-10

    altered = _observed()
    altered["rising"]["bounded_mid"]["z"] += np.array([0.03, 0.0, -0.02])
    altered_dirs = C.pathology_directions_from_geometry(
        altered, axis_coord=along, perpendicular_coord=perp
    )
    assert not np.array_equal(
        dirs["pathology_parallel"], altered_dirs["pathology_parallel"]
    )


def test_summary7_uses_axial_field_projection_not_core_surround_difference():
    state = {
        "z": np.array([0.2, 0.8, 0.5, 0.6]),
        "m": np.array([4.0, 1.0, 2.0, 5.0]),
        "S_G": 0.3,
    }
    core = np.array([True, True, False, False])
    axis = np.array([-2.0, -1.0, 1.0, 2.0])
    summary = C.summary7(state, core, axis)
    assert tuple(C.SUMMARY7_NAMES) == (
        "z_core", "z_surround", "delta_z_parallel",
        "m_core", "m_surround", "delta_m_parallel", "S_G",
    )
    assert summary[2] == C._axial_projection(state["z"], axis)
    assert summary[5] == C._axial_projection(state["m"], axis)
    assert not np.isclose(summary[2], summary[0] - summary[1])


def test_coordinate_payload_is_float64_and_roundtrip_state_hash_is_lossless():
    obs = _observed()
    obs["rising"]["bounded_mid"]["z"] += np.array([0.01, -0.02, 0.03])
    obs["rising"]["bounded_late"]["m"] += np.array([0.04, -0.01, 0.02])
    obs["peak"]["bounded_early"]["z"] += np.array([-0.02, 0.03, -0.01])
    obs["peak"]["bounded_mid"]["m"] += np.array([0.02, 0.01, -0.03])
    coords = C.build_coordinate_set(
        obs,
        core_mask=np.array([True, False, False]),
        axis_coord=np.array([-1.0, 0.0, 1.0]),
        perpendicular_coord=np.array([1.0, -2.0, 1.0]),
    )
    arrays = C.coordinate_array_payload(coords)
    for key in (
        "z", "m", "S_G", "basis_directions_standardized",
        "trajectory_tangent_standardized", "component_scale",
        "full_field_envelope_lo", "full_field_envelope_hi",
        "axis_coord", "perpendicular_coord",
    ):
        assert arrays[key].dtype == np.float64
    encoded = C.deterministic_npz_bytes(arrays)
    with np.load(BytesIO(encoded), allow_pickle=False) as roundtrip:
        cells = list(coords["primary"]) + list(coords["secondary_shell"])
        for i, cell in enumerate(cells):
            restored = {
                "z": roundtrip["z"][i],
                "m": roundtrip["m"][i],
                "S_G": roundtrip["S_G"][i],
            }
            assert C.slow_state_sha256(restored) == C.slow_state_sha256(
                cell["state"]
            )
        assert C.semantic_array_sha256({
            key: roundtrip[key] for key in roundtrip.files
        }) == C.semantic_array_sha256(arrays)
    assert all(
        "standardized_distance_from_anchor_manifold" in cell
        and "reconstruction_error_standardized_rms" in cell
        for cell in cells
    )
    sign = coords["basis"]["fullfield_mode_sign_alignment"]
    assert set(sign) == {"fullfield_mode2", "fullfield_mode3"}
    assert all(
        row["rule"] in {
            "forward_trajectory_derivative",
            "deterministic_max_loading_fallback",
        }
        for row in sign.values()
    )


def test_coordinate_npz_is_deterministic_allow_pickle_false_and_write_once(tmp_path):
    arrays = {
        "z": np.arange(12, dtype=np.float32).reshape(3, 4),
        "cell_ids": np.asarray(["a", "b", "c"], dtype="U4"),
    }
    a = C.deterministic_npz_bytes(arrays)
    b = C.deterministic_npz_bytes(arrays)
    assert a == b
    path = tmp_path / "coords.npz"
    assert C.write_bytes_once(path, a) == "created"
    assert C.write_bytes_once(path, b) == "reused"
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["z"], arrays["z"])
        np.testing.assert_array_equal(data["cell_ids"], arrays["cell_ids"])
    changed = C.deterministic_npz_bytes({**arrays, "z": arrays["z"] + 1})
    try:
        C.write_bytes_once(path, changed)
    except RuntimeError as exc:
        assert "differs" in str(exc)
    else:
        raise AssertionError("coordinate overwrite must fail")


def _cell(rep, seed, idx, status, direction="forward"):
    return {
        "representation": rep,
        "seed": seed,
        "cell_id": f"c{idx}",
        "path_index": idx,
        "path_direction": direction,
        "status": status,
        "maturation_direction": direction if status == "pass" else None,
    }


def _expected(reps=("full_field",)):
    return {
        rep: {seed: [f"c{i}" for i in range(4)] for seed in (1, 3, 4)}
        for rep in reps
    }


def test_cell_contract_is_exact_2_by_3_jeffreys_and_five_of_six():
    rows = []
    for phase in C.DEFAULT_PHASES:
        for noise in C.DEFAULT_NOISES:
            rows.append({
                "phase": phase,
                "noise": noise,
                "status": "complete",
                "mature_pass": not (phase == "peak" and noise == "noise_resample_2"),
                "maturation_direction": "forward",
            })
    out = C.aggregate_cell(rows)
    assert out["status"] == "pass"
    assert (out["k"], out["n"]) == (5, 6)
    assert out["posterior_ci"][0] < out["posterior_median"] < out["posterior_ci"][1]
    assert C.aggregate_cell(rows[:-1])["status"] == "indeterminate"


def test_local_window_requires_adjacent_cells_in_two_seeds_same_direction():
    cells = []
    for seed in (1, 3):
        cells += [_cell("full_field", seed, i, "pass" if i in (1, 2) else "fail")
                  for i in range(4)]
    cells += [_cell("full_field", 4, i, "fail") for i in range(4)]
    out = C.adjudicate_phasec_neighbourhood(cells, _expected())
    assert out["verdict"] == "local_maturation_window"


def test_two_concordant_seed_windows_allow_indeterminate_third_seed():
    cells = []
    for seed in (1, 3):
        cells += [_cell("full_field", seed, i, "pass" if i in (1, 2) else "fail")
                  for i in range(4)]
    cells += [_cell("full_field", 4, i, "fail") for i in range(3)]
    out = C.adjudicate_phasec_neighbourhood(cells, _expected())
    assert out["verdict"] == "local_maturation_window"


def test_strict_no_window_requires_complete_three_seed_negative():
    cells = [_cell("full_field", seed, i, "fail")
             for seed in (1, 3, 4) for i in range(4)]
    out = C.adjudicate_phasec_neighbourhood(cells, _expected())
    assert out["verdict"] == "no_local_maturation_window"


def test_representation_sensitive_fixture_is_not_promoted():
    cells = []
    for rep in ("full_field", "axis_projection"):
        for seed in (1, 3, 4):
            passed = rep == "full_field" and seed in (1, 3)
            cells += [_cell(rep, seed, i, "pass" if passed and i in (1, 2) else "fail")
                      for i in range(4)]
    out = C.adjudicate_phasec_neighbourhood(
        cells, _expected(("full_field", "axis_projection"))
    )
    assert out["verdict"] == "representation_sensitive"


def test_missing_or_indeterminate_is_insufficient_coverage_not_negative():
    cells = [_cell("full_field", seed, i, "fail")
             for seed in (1, 3, 4) for i in range(4)]
    cells = [c for c in cells if not (c["seed"] == 4 and c["cell_id"] == "c3")]
    out = C.adjudicate_phasec_neighbourhood(cells, _expected())
    assert out["verdict"] == "insufficient_coverage"

import numpy as np

from src.topic4_core_field_stage3 import params_to_q
from src.topic4_spectral_field import (
    array_sha256,
    project_surface_to_spectral,
    sample_stationary_residual_pairs,
    spectral_field_h,
    spectral_surface,
    uniform_sheet_grid,
)


THETA = np.asarray([
    3.6202140551125526, 7.054341437477716, 0.33299422312135596,
    0.2384953189747353, 2.55249063283656, 11.847833907651358,
    3.577066400859459, 0.07885221110723895, 0.6168672885998744,
    0.9864584048804511, 12.287567587097564, 17.19786634663256,
    -0.3887262339692643, -0.4194174564300462, 2.116704332158924,
    1.042698924519583, 0.9426693059863966,
])


def test_spectral_field_is_bounded_and_budgeted():
    positions = uniform_sheet_grid(32)
    from src.topic4_spectral_field import fourier_wavevectors
    coefficients = np.zeros((len(fourier_wavevectors(5)), 2))
    coefficients[2, 1] = 0.7
    h, _ = spectral_field_h(
        coefficients, positions, max_harmonic=5, target_count=300.0,
    )
    assert np.all((h >= 0.0) & (h <= 1.0))
    assert np.isclose(h.sum(), 300.0, atol=1e-9)


def test_stationary_candidates_are_reproducible_and_antithetic():
    kwargs = dict(
        n_pairs=2, max_harmonic=7, seed=123,
        rms_amplitudes=[0.3, 0.7], active_max_harmonic=3,
    )
    left = sample_stationary_residual_pairs(**kwargs)
    right = sample_stationary_residual_pairs(**kwargs)
    assert array_sha256(left[0]["positive"]) == array_sha256(right[0]["positive"])
    assert np.array_equal(left[1]["positive"], -left[1]["negative"])
    from src.topic4_spectral_field import fourier_wavevectors
    vectors = fourier_wavevectors(7) * 20.0 / np.pi
    assert np.all(left[0]["positive"][np.linalg.norm(vectors, axis=1) > 3] == 0.0)


def test_old_stage3_field_projects_on_uniform_sheet_without_contacts():
    grid = uniform_sheet_grid(72)
    log_q = np.log(params_to_q(THETA, grid, K=3, L=20.0))
    coefficients = project_surface_to_spectral(
        log_q, grid, max_harmonic=9, L=20.0,
    )
    reconstructed = spectral_surface(
        coefficients, grid, max_harmonic=9, L=20.0,
    )
    reconstructed -= reconstructed.mean()
    target = log_q - log_q.mean()
    assert np.sqrt(np.mean((reconstructed - target) ** 2)) < 0.35


def test_builder_api_has_no_observation_geometry_argument():
    # Regression lock: field generation cannot silently become contact-driven.
    import inspect
    from src import topic4_spectral_field

    forbidden = {"contact_xy", "contacts", "shaft_ids", "onsets", "labels"}
    for name in (
        "project_surface_to_spectral", "sample_stationary_residual_pairs",
        "spectral_field_h",
    ):
        assert forbidden.isdisjoint(inspect.signature(
            getattr(topic4_spectral_field, name)
        ).parameters)


def test_v3_allocation_grid_is_uniform_and_observation_free():
    import inspect
    from scripts.freeze_topic4_rev10_sa_spectral_field_v3_candidates import (
        build_candidates,
        uniform_allocation_centers,
    )

    centers = uniform_allocation_centers(4, margin_mm=2.5, L=20.0)
    assert centers.shape == (16, 2)
    assert np.array_equal(np.unique(centers[:, 0]), np.linspace(2.5, 17.5, 4))
    assert np.array_equal(np.unique(centers[:, 1]), np.linspace(2.5, 17.5, 4))
    forbidden = {"contact_xy", "contacts", "shaft_ids", "onsets", "labels"}
    assert forbidden.isdisjoint(inspect.signature(build_candidates).parameters)


def test_uniform_allocation_direction_moves_continuous_field_mass():
    from src.topic4_spectral_field import (
        fourier_basis_2d,
        fourier_wavevectors,
        spectral_field_h,
        uniform_sheet_grid,
    )

    grid = uniform_sheet_grid(64, L=20.0)
    center = np.array([12.5, 12.5])
    width = 2.5
    surface = 4.0 * np.exp(
        -0.5 * np.sum((grid - center) ** 2, axis=1) / width ** 2
    )
    surface -= surface.mean()
    fitted, *_ = np.linalg.lstsq(
        fourier_basis_2d(grid, 9, L=20.0), surface, rcond=None,
    )
    coefficients = fitted.reshape(len(fourier_wavevectors(9, L=20.0)), 2)
    h, _ = spectral_field_h(
        coefficients, grid, max_harmonic=9, target_count=145.0, L=20.0,
    )
    radius = np.linalg.norm(grid - center, axis=1)

    assert np.linalg.norm(grid[np.argmax(h)] - center) < 0.5
    assert h[radius <= width].sum() / h.sum() > 0.5

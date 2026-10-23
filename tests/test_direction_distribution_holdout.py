import numpy as np
from scipy.integrate import quad
from scipy.stats import vonmises_fisher

from src.interictal_direction_distribution_holdout import (
    LOG_SPHERE_AREA, component_logpdf, fit_concentration, fit_template_density,
    mixture_nll, shuffle_training_directions, evaluate_models, METHODS,
)


def test_density_normalization_uniform_and_scipy_agreement():
    axis = np.array([[1., 0, 0]])
    for kappa in (0., 1e-8, .1, 3., 50., 1000.):
        def density(z):
            point = np.array([[z, np.sqrt(max(0, 1-z*z)), 0.]])
            return float(np.exp(component_logpdf(point, axis, np.array([kappa]))[0, 0]))
        integral = 2 * np.pi * quad(density, -1, 1, epsabs=1e-10)[0]
        np.testing.assert_allclose(integral, 1, atol=1e-9)
        points = np.array([[1., 0, 0], [0., 1, 0], [-1., 0, 0]])
        actual = component_logpdf(points, axis, np.array([kappa]))[:, 0]
        if kappa > 1e-6:
            np.testing.assert_allclose(actual, vonmises_fisher.logpdf(points, axis[0], kappa), atol=1e-9)
        elif kappa == 0:
            np.testing.assert_allclose(actual, -LOG_SPHERE_AREA)


def test_fixed_axis_concentration_optimum_and_negative_alignment():
    assert fit_concentration(-.5) == 0
    for c in (.01, .3, .8, .99):
        k = fit_concentration(c)
        np.testing.assert_allclose(1 / np.tanh(k) - 1/k, c, atol=1e-9)


def test_mixture_rotation_label_symmetry_and_probability_sum():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 3)); x /= np.linalg.norm(x, axis=1)[:, None]
    axes = np.array([[1., 0, 0], [0, 1., 0]])
    density = dict(axes=axes, kappas=np.array([2., 4.]), weights=np.array([.3, .7]))
    actual = mixture_nll(density, x)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = {**density, "axes": axes @ rotation}
    np.testing.assert_allclose(actual, mixture_nll(rotated, x @ rotation), atol=1e-12)
    flipped = {key: value[::-1] for key, value in density.items()}
    np.testing.assert_allclose(actual, mixture_nll(flipped, x), atol=1e-12)
    parts = np.exp(component_logpdf(x, axes, density["kappas"]))
    np.testing.assert_allclose(actual, -np.log(parts @ density["weights"]), atol=1e-12)


def test_density_training_only_and_shuffle_missingness_blocks():
    x = np.array([[1., 0, 0], [1., 0, 0], [.6, .8, 0],
                  [-1., 0, 0], [-1., 0, 0], [-.6, .8, 0], [np.nan]*3])
    model = dict(labels=np.array([0, 0, 0, 1, 1, 1, 0]), axes=np.array([[1., 0, 0], [-1., 0, 0]]))
    fitted = fit_template_density(model, x)
    np.testing.assert_allclose(fitted['weights'], [.5, .5])
    blocks = np.array([0, 0, 0, 1, 1, 1, 1])
    shuffled = shuffle_training_directions(x, blocks, np.random.default_rng(9))
    np.testing.assert_array_equal(np.isfinite(x), np.isfinite(shuffled))
    for block in (0, 1):
        np.testing.assert_allclose(np.sort(x[blocks == block], axis=0), np.sort(shuffled[blocks == block], axis=0))
    # Opposite held-out directions have worse loss under a nonuniform fixed model.
    asymmetric = dict(axes=np.array([[1., 0, 0]]), kappas=np.array([4.]), weights=np.array([1.]))
    assert mixture_nll(asymmetric, np.array([[1., 0, 0]]))[0] < mixture_nll(asymmetric, np.array([[-1., 0, 0]]))[0]


def test_test_directions_cannot_select_templates_or_change_denominator():
    templates = np.array([[0., 1., 2.], [2., 1., 0.]])
    ranks = np.repeat(templates.T, 4, axis=1)
    bools = np.ones_like(ranks, dtype=bool)
    axes = np.array([[1., 0, 0], [-1., 0, 0]])
    x = np.repeat(axes, 4, axis=0)
    models = {m: dict(templates=templates, axes=axes) for m in METHODS}
    densities = {m: dict(axes=axes, kappas=np.array([3., 3.]), weights=np.array([.5, .5])) for m in METHODS}
    correct = evaluate_models(models, densities, ranks, bools, x)
    reversed_direction = evaluate_models(models, densities, ranks, bools, -x)
    for m in METHODS:
        np.testing.assert_array_equal(correct['assignments'][m], reversed_direction['assignments'][m])
        assert correct['metrics'][m]['angular_mae_deg'] == 0
        assert reversed_direction['metrics'][m]['angular_mae_deg'] == 180
        # Marginal distribution is unchanged by swapping equally weighted modes.
        np.testing.assert_allclose(correct['metrics'][m]['nll'], reversed_direction['metrics'][m]['nll'])
    assert correct['finite_mask'].sum() == 8

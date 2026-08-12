import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick
from params import Params
from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive


def _positions(n=400):
    side = int(np.sqrt(n))
    xy = np.linspace(0.05, 3.95, side)
    x, y = np.meshgrid(xy, xy, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel()])


def test_local_and_permuted_share_exact_values_but_not_spatial_assignment():
    positions = _positions()
    base = dict(
        sigma_rate_per_ms=0.5, tau_ms=20.0, ell_mm=0.4,
        update_interval_ms=1.0, grid_spacing_mm=0.2, seed=17,
    )
    local = SpatialOUDrive(positions, 4.0, 0.1, SpatialOUConfig(mode="local", **base))
    permuted = SpatialOUDrive(
        positions, 4.0, 0.1, SpatialOUConfig(mode="permuted", **base),
    )
    for time_ms in (0.0, 1.0, 2.0, 7.0):
        left, right = local.step(time_ms), permuted.step(time_ms)
        np.testing.assert_allclose(np.sort(left), np.sort(right), atol=1e-12)
        assert not np.array_equal(left, right)
        assert abs(left.mean()) < 1e-12


def test_spatial_ou_is_reproducible_and_reports_continuous_trace():
    positions = _positions()
    config = SpatialOUConfig(
        mode="local", sigma_rate_per_ms=0.5, tau_ms=20.0, ell_mm=0.4,
        update_interval_ms=1.0, grid_spacing_mm=0.2, seed=23,
    )
    a = SpatialOUDrive(positions, 4.0, 0.1, config)
    b = SpatialOUDrive(positions, 4.0, 0.1, config)
    for time_ms in np.arange(0.0, 10.1, 0.1):
        np.testing.assert_array_equal(a.step(time_ms), b.step(time_ms))
    trace = a.trace_arrays()
    assert len(trace["time_ms"]) == 11
    assert np.all(trace["spatial_sd_rate_per_ms"] > 0.0)
    assert np.max(np.abs(trace["spatial_mean_rate_per_ms"])) < 1e-10


def test_local_drive_has_more_near_neighbor_coherence_than_permuted_control():
    positions = _positions()
    base = dict(
        sigma_rate_per_ms=0.5, tau_ms=10.0, ell_mm=0.4,
        update_interval_ms=1.0, grid_spacing_mm=0.2, seed=31,
    )
    local = SpatialOUDrive(positions, 4.0, 0.1, SpatialOUConfig(mode="local", **base))
    permuted = SpatialOUDrive(
        positions, 4.0, 0.1, SpatialOUConfig(mode="permuted", **base),
    )
    local_rows, permuted_rows = [], []
    for time_ms in np.arange(0.0, 201.0, 1.0):
        local_rows.append(local.step(time_ms))
        permuted_rows.append(permuted.step(time_ms))
    local_rows = np.asarray(local_rows)
    permuted_rows = np.asarray(permuted_rows)
    side = int(np.sqrt(len(positions)))
    left = np.arange(len(positions)).reshape(side, side)[:, :-1].ravel()
    right = np.arange(len(positions)).reshape(side, side)[:, 1:].ravel()
    local_variogram = np.mean((local_rows[:, left] - local_rows[:, right]) ** 2)
    permuted_variogram = np.mean(
        (permuted_rows[:, left] - permuted_rows[:, right]) ** 2
    )
    assert local_variogram < 0.5 * permuted_variogram


def test_external_drive_none_preserves_engine_baseline_hash():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1,
               nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, n_e, n_i = place_neurons(p, rng)
    net = build_connectivity_rot(
        p, pos, labels, n_e, n_i, rng, theta_EE=np.radians(45), AR=2.0,
    )
    net["rng"] = np.random.default_rng(1)
    result = simulate_kick(
        p, net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=np.full(n_e + n_i, 18.0),
        external_e_rate_drive=None,
    )
    assert hashlib.sha1(result["E_spk_bool"].tobytes()).hexdigest()[:16] == (
        "da5fc18c27d5340a"
    )
    assert result["external_e_rate_drive"] is None

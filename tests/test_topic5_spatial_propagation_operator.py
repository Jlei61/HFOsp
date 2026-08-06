"""Correctness tests for the spatial propagation operator.

These are not surface-behaviour tests.  Each one names a property that, if it
failed silently, would let the run produce numbers that look fine and mean
nothing -- a leak from the future, an operator whose sign is backwards, a
variant that does not actually freeze what it claims to freeze.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_propagation_operator import (  # noqa: E402
    CONFIGS, OperatorConfig, SPOModel, build_grid,
)
from src.topic5_virtual_seeg_operator import build_observation_operator  # noqa: E402


GRID = (14, 10)


def _contacts() -> np.ndarray:
    xs = np.linspace(0.0, 30.0, 6)
    return np.stack([xs, np.zeros_like(xs)], axis=-1)


def _setup(variant: str, microsteps: int = 3, seed: int = 0) -> SPOModel:
    contacts = _contacts()
    centres, shape, mask = build_grid(contacts, sigma_mm=3.0, max_cells_per_side=14)
    H = build_observation_operator(contacts, centres, sigma_mm=3.0)
    return SPOModel(OperatorConfig(
        variant=variant, n_contacts=len(contacts), grid_shape=shape,
        microsteps=microsteps, seed=seed,
        observation_operator=H, grid_mask=mask,
    ))


def _impulse(model: SPOModel, at: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive one contact and return the resulting activation field."""
    x = torch.zeros(1, model.config.n_contacts)
    x[0, at] = 1.0
    state = model.initial_state(1, x.device)
    t_norm = torch.zeros(1, 1)
    state, _, _ = model.step(state, x, torch.zeros_like(x), t_norm)
    return state


def _centre_of_mass_along_axis(a: torch.Tensor) -> float:
    ny = a.shape[1]
    coordinate = torch.arange(ny, dtype=torch.float32).view(1, ny, 1)
    total = a.sum().clamp_min(1e-9)
    return float((a * coordinate).sum() / total)


def _spread(a: torch.Tensor, dim: int) -> float:
    """Weighted standard deviation of a 2-D field along one axis."""
    assert a.dim() == 2, "pass a single field, not a batch"
    n = a.shape[dim]
    shape = [1, 1]
    shape[dim] = n
    coordinate = torch.arange(n, dtype=torch.float32).view(shape)
    total = a.sum().clamp_min(1e-9)
    mean = (a * coordinate).sum() / total
    return float(((a * (coordinate - mean) ** 2).sum() / total).sqrt())


# 1 -----------------------------------------------------------------------
def test_zero_transport_reduces_to_field_null():
    """With D and v pinned to zero the full operator must equal FIELD_NULL."""
    full = _setup("ANISOTROPIC_RECOVERY")
    null = _setup("FIELD_NULL")
    null.load_state_dict(full.state_dict())
    with torch.no_grad():
        full.operator.raw_D_parallel.fill_(-30.0)   # softplus -> ~0
        full.operator.raw_D_perp.fill_(-30.0)
        full.operator.v.fill_(0.0)
    a_full, _ = _impulse(full)
    a_null, _ = _impulse(null)
    assert torch.allclose(a_full, a_null, atol=1e-5)


# 2 -----------------------------------------------------------------------
def test_drift_sign_reverses_the_direction_of_travel():
    model = _setup("ANISOTROPIC_RECOVERY", microsteps=4)
    with torch.no_grad():
        model.operator.raw_D_parallel.fill_(-30.0)
        model.operator.raw_D_perp.fill_(-30.0)
        model.operator.v.fill_(0.6)
    forward, _ = _impulse(model, at=2)
    with torch.no_grad():
        model.operator.v.fill_(-0.6)
    backward, _ = _impulse(model, at=2)
    assert _centre_of_mass_along_axis(forward) > _centre_of_mass_along_axis(backward)


# 3 -----------------------------------------------------------------------
def test_axial_diffusion_increases_axial_spread():
    model = _setup("ANISOTROPIC_RECOVERY", microsteps=4)
    with torch.no_grad():
        model.operator.raw_D_perp.fill_(-30.0)
        model.operator.v.fill_(0.0)
        model.operator.raw_D_parallel.fill_(-30.0)
    tight, _ = _impulse(model, at=2)
    with torch.no_grad():
        model.operator.raw_D_parallel.fill_(4.0)
    wide, _ = _impulse(model, at=2)
    assert _spread(wide[0], dim=0) > _spread(tight[0], dim=0)


# 4 -----------------------------------------------------------------------
def test_transverse_diffusion_increases_transverse_spread():
    model = _setup("ANISOTROPIC_RECOVERY", microsteps=4)
    with torch.no_grad():
        model.operator.raw_D_parallel.fill_(-30.0)
        model.operator.v.fill_(0.0)
        model.operator.raw_D_perp.fill_(-30.0)
    tight, _ = _impulse(model, at=2)
    with torch.no_grad():
        model.operator.raw_D_perp.fill_(4.0)
    wide, _ = _impulse(model, at=2)
    assert _spread(wide[0], dim=1) > _spread(tight[0], dim=1)


# 5 -----------------------------------------------------------------------
def test_larger_decay_shortens_persistence():
    """Drive once, then let the field run; faster decay must leave less behind."""
    def remaining(gamma_raw: float) -> float:
        model = _setup("ANISOTROPIC_RECOVERY", microsteps=1)
        with torch.no_grad():
            model.operator.raw_gamma_a.fill_(gamma_raw)
            model.operator.raw_beta.fill_(-30.0)
            model.operator.raw_xi.fill_(-30.0)
        state = _impulse(model)
        a, r = state
        for _ in range(4):
            a, r = model.operator.step(a, r, None, "softplus")
        return float(a.sum())
    assert remaining(2.0) < remaining(-2.0)


# 6 -----------------------------------------------------------------------
def test_recovery_field_suppresses_later_activity():
    def total_after(beta_raw: float) -> float:
        model = _setup("ANISOTROPIC_RECOVERY", microsteps=1)
        with torch.no_grad():
            model.operator.raw_beta.fill_(beta_raw)
            model.operator.raw_xi.fill_(1.0)
            model.operator.raw_gamma_r.fill_(-2.0)
        a, r = _impulse(model)
        for _ in range(4):
            a, r = model.operator.step(a, r, None, "softplus")
        return float(a.sum())
    assert total_after(1.5) < total_after(-30.0)


# 7 -----------------------------------------------------------------------
def test_injection_and_readout_use_the_same_operator():
    """A contact must drive exactly the cells it reads, with the same weights."""
    model = _setup("ANISOTROPIC_RECOVERY")
    H = model.H.numpy()
    x = torch.zeros(1, model.config.n_contacts)
    x[0, 3] = 1.0
    injection = torch.einsum("bc,cm->bm", x, model.H).numpy()[0]
    assert np.allclose(injection, H[3])
    assert np.allclose(H.sum(axis=1), 1.0, atol=1e-5)
    # locality: a contact reads a strict subset of the grid
    assert (H[3] > 0).sum() < H.shape[1]


# 8 -----------------------------------------------------------------------
def test_prediction_at_step_t_cannot_see_step_t_plus_one():
    model = _setup("ANISOTROPIC_RECOVERY")
    c = model.config.n_contacts
    x = torch.zeros(1, 5, c)
    x[0, 0, 0] = 1.0
    x[0, 1, 1] = 1.0
    recruited = torch.cumsum(x, dim=1).clamp(0, 1)
    valid = torch.ones(1, 5, dtype=torch.bool)
    logits_a, _ = model(x, recruited, valid)
    # change only the LAST step; earlier logits must not move
    x2 = x.clone()
    x2[0, 4, 4] = 1.0
    recruited2 = torch.cumsum(x2, dim=1).clamp(0, 1)
    logits_b, _ = model(x2, recruited2, valid)
    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6)


# 9 -----------------------------------------------------------------------
def test_rollout_does_not_consume_ground_truth():
    """A free rollout takes a seed set and nothing else."""
    import inspect
    # Check the signature and the executable body, not the prose: the docstring
    # legitimately contains the words this test is looking for.
    signature = inspect.signature(SPOModel.rollout)
    assert set(signature.parameters) == {"self", "seed_set", "max_steps", "threshold"}
    body = inspect.getsource(SPOModel.rollout)
    body = body.split('"""')[2] if body.count('"""') >= 2 else body
    assert "target" not in body and "teacher" not in body
    model = _setup("ANISOTROPIC_RECOVERY")
    seed = torch.zeros(1, model.config.n_contacts)
    seed[0, 0] = 1.0
    produced = model.rollout(seed, max_steps=6)
    assert produced[0].equal(seed)
    assert all(p.shape == seed.shape for p in produced)


# 10 ----------------------------------------------------------------------
def test_a_contact_never_recruits_twice_in_a_rollout():
    model = _setup("ANISOTROPIC_RECOVERY")
    seed = torch.zeros(1, model.config.n_contacts)
    seed[0, 0] = 1.0
    produced = model.rollout(seed, max_steps=10, threshold=0.0)
    stacked = torch.cat(produced, dim=0)
    assert stacked.sum(dim=0).max() <= 1.0


# 11 ----------------------------------------------------------------------
@pytest.mark.parametrize("variant,frozen", [
    ("FIELD_NULL", ("D_parallel", "D_perp", "v")),
    ("ISOTROPIC_DIFFUSION", ("v", "beta", "xi")),
    ("ANISOTROPIC_DRIFT", ("beta", "xi")),
])
def test_each_variant_freezes_what_it_claims(variant, frozen):
    model = _setup(variant)
    estimates = model.parameter_estimates()
    for name in frozen:
        assert abs(estimates[name]) < 1e-8, f"{variant} left {name} free"


# 12 ----------------------------------------------------------------------
def test_isotropic_variant_ties_the_two_diffusions():
    model = _setup("ISOTROPIC_DIFFUSION")
    with torch.no_grad():
        model.operator.raw_D_parallel.fill_(0.7)
        model.operator.raw_D_perp.fill_(-3.0)   # must be ignored
    estimates = model.parameter_estimates()
    assert abs(estimates["D_parallel"] - estimates["D_perp"]) < 1e-8


# 13 ----------------------------------------------------------------------
def test_grid_mask_keeps_the_field_inside_the_observed_domain():
    contacts = _contacts()
    centres, shape, mask = build_grid(contacts, sigma_mm=3.0, max_cells_per_side=14)
    assert mask.shape == shape
    assert 0.0 < mask.mean() < 1.0, "mask is degenerate"
    model = _setup("ANISOTROPIC_RECOVERY", microsteps=5)
    a, r = _impulse(model)
    outside = torch.from_numpy(1.0 - mask)
    assert float((a[0] * outside).abs().max()) == 0.0
    assert float((r[0] * outside).abs().max()) == 0.0


# 14 ----------------------------------------------------------------------
def test_static_variant_has_no_field_and_no_spatial_parameters():
    model = _setup("STATIC")
    assert model.operator is None
    names = {n for n, _ in model.named_parameters()}
    assert not any("D_parallel" in n or "raw_v" in n for n in names)
    assert model.parameter_estimates() == {"variant": "STATIC"}


# 15 ----------------------------------------------------------------------
def test_parameter_count_is_tiny_next_to_a_free_graph():
    """The whole point: fewer numbers than an adjacency matrix has entries."""
    model = _setup("ANISOTROPIC_RECOVERY")
    n_cells = model.config.grid_shape[0] * model.config.grid_shape[1]
    spatial = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith("operator.") or n.startswith("raw_w"))
    assert spatial <= 12, f"{spatial} spatial parameters is not a low-dimensional operator"
    assert spatial < n_cells ** 2 / 100

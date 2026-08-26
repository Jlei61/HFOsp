"""Contract tests for the Raw-SEEG R0.1 model (Worker C).

Every test below maps to a clause of
``docs/archive/topic5/raw_seeg_state_scientific_spec_2026-08-21.md`` sections
5-7, or to a known failure mode this revision must not repeat (the Epi-PRSSM
v0.1 ``softplus(log tau)`` collapse).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.topic5_raw_seeg_state import contract  # noqa: E402
from src.topic5_raw_seeg_state.dynamics import DampedRotationDynamics  # noqa: E402
from src.topic5_raw_seeg_state.losses import (  # noqa: E402
    ACTIVE_DIM_STD_THRESHOLD,
    consistency_loss,
    consistency_ratio,
    latent_diagnostics,
    masked_forecast_loss,
    total_loss,
)
from src.topic5_raw_seeg_state.model import AttentionPool, RawSeegStateModel  # noqa: E402

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tiny_model(n_contacts: int = 5, n_shafts: int = 2, context_minutes: int | None = None,
                **kwargs) -> RawSeegStateModel:
    ctx = contract.CONTEXT_MINUTES if context_minutes is None else context_minutes
    model = RawSeegStateModel(
        n_contacts=n_contacts,
        n_shafts=n_shafts,
        context_minutes=ctx,
        dropout=0.0,
        **kwargs,
    )
    model.eval()
    return model


def _batch(model: RawSeegStateModel, batch: int = 2, seed: int = 0):
    """The seven contract inputs, in ``contract.ALLOWED_INPUT_KEYS`` order."""
    g = torch.Generator().manual_seed(seed)
    c = model.n_contacts
    t = model.context_minutes * contract.MINUTE_SAMPLES
    raw = torch.randn(batch, c, t, generator=g)
    coords = torch.randn(batch, c, 3, generator=g) * 10.0
    coord_valid = torch.ones(batch, c, dtype=torch.bool)
    shaft = torch.arange(c).remainder(model.n_shafts).expand(batch, c).contiguous()
    shaft_index = torch.arange(c).div(model.n_shafts, rounding_mode="floor").expand(
        batch, c
    ).contiguous()
    contact_valid = torch.ones(batch, c, dtype=torch.bool)
    minute_valid = torch.ones(batch, c, model.context_minutes, dtype=torch.bool)
    return raw, coords, coord_valid, shaft, shaft_index, contact_valid, minute_valid


def _graph_size(t: torch.Tensor) -> int:
    """Number of distinct autograd nodes reachable from ``t``."""
    seen: set = set()
    stack = [t.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        for nxt, _ in fn.next_functions:
            stack.append(nxt)
    return len(seen)


# ---------------------------------------------------------------------------
# 1-6  dynamics
# ---------------------------------------------------------------------------


def test_1_tau_is_hard_clamped_to_contract_bounds():
    dyn = DampedRotationDynamics()
    for extreme in (-50.0, 50.0):
        with torch.no_grad():
            dyn.log_tau.fill_(extreme)
        tau = dyn.tau
        assert torch.isfinite(tau).all()
        assert float(tau.min()) >= contract.TAU_MIN_MINUTES - 1e-6
        assert float(tau.max()) <= contract.TAU_MAX_MINUTES + 1e-6
    # mixed extremes in one vector
    with torch.no_grad():
        dyn.log_tau[::2] = -50.0
        dyn.log_tau[1::2] = 50.0
    tau = dyn.tau
    assert float(tau.min()) == pytest.approx(contract.TAU_MIN_MINUTES, rel=1e-6)
    assert float(tau.max()) == pytest.approx(contract.TAU_MAX_MINUTES, rel=1e-6)


def test_2_initial_taus_cover_the_whole_contract_range():
    dyn = DampedRotationDynamics()
    tau = dyn.tau.detach()
    assert tau.numel() == contract.N_ROTATION_MODES
    span_orders = math.log10(float(tau.max()) / float(tau.min()))
    assert span_orders >= 3.0, f"initial taus span only {span_orders:.2f} decades"
    assert abs(float(tau.min()) - contract.TAU_MIN_MINUTES) <= 0.10 * contract.TAU_MIN_MINUTES
    assert abs(float(tau.max()) - contract.TAU_MAX_MINUTES) <= 0.10 * contract.TAU_MAX_MINUTES


def test_3_regression_softplus_bug_median_tau_is_slow():
    """Epi-PRSSM v0.1 used softplus(log tau) and pinned every mode near 5.7 s.

    Under the R0.1 exp(clamp(.)) parametrisation the median initial time
    constant must be far above the minute scale.
    """
    dyn = DampedRotationDynamics()
    median_tau = float(dyn.tau.detach().median())
    assert median_tau > 30.0, f"median initial tau = {median_tau:.3f} min"


def test_4_bh_is_a_strict_contraction_for_every_positive_horizon():
    torch.manual_seed(11)
    horizons = [1e-3, 0.1, 1.0, 5.0, 100.0, 3000.0]
    for _ in range(200):
        dyn = DampedRotationDynamics().double()
        with torch.no_grad():
            dyn.log_tau.normal_(mean=2.0, std=4.0)
            dyn.omega_raw.normal_(mean=0.0, std=2.0)
            dyn.mu.normal_(mean=0.0, std=1.0)
        assert dyn.is_stable()
        z = torch.randn(3, contract.LATENT_DIM, dtype=torch.float64)
        r0 = torch.linalg.norm(z - dyn.mu, dim=-1)
        for h in horizons:
            zh = dyn(z, h)
            rh = torch.linalg.norm(zh - dyn.mu, dim=-1)
            assert bool((rh < r0).all()), f"non-contractive at h={h}"


def test_5_closed_form_equals_recursion_and_graph_depth_is_horizon_free():
    torch.manual_seed(3)
    dyn = DampedRotationDynamics().double()
    with torch.no_grad():
        dyn.log_tau.normal_(mean=2.0, std=1.5)
        dyn.omega_raw.normal_(mean=0.0, std=1.0)
        dyn.mu.normal_(mean=0.0, std=0.5)
    z = torch.randn(4, contract.LATENT_DIM, dtype=torch.float64)

    direct = dyn(z, 100.0)
    stepped = z
    for _ in range(100):
        stepped = dyn(stepped, 1.0)
    assert torch.allclose(direct, stepped, atol=1e-4, rtol=0.0)

    # graph depth: the closed form must not grow with the horizon.
    zg = z.clone().requires_grad_(True)
    n1 = _graph_size(dyn(zg, 1.0))
    n100 = _graph_size(dyn(zg, 100.0))
    assert n1 == n100, f"graph grew with horizon: {n1} -> {n100}"
    rec = zg
    for _ in range(10):
        rec = dyn(rec, 1.0)
    assert _graph_size(rec) > 5 * n100
    # gradients flow through the closed form without create_graph
    out = dyn(zg, 100.0).sum()
    (grad,) = torch.autograd.grad(out, zg, create_graph=False)
    assert torch.isfinite(grad).all()


def test_5b_negative_horizon_is_rejected():
    dyn = DampedRotationDynamics()
    z = torch.zeros(2, contract.LATENT_DIM)
    with pytest.raises(ValueError):
        dyn(z, -1.0)
    with pytest.raises(ValueError):
        dyn(z, torch.tensor([1.0, -0.5]))


def test_6_identity_mode_is_capacity_matched():
    dyn = DampedRotationDynamics()
    ident = DampedRotationDynamics(identity_mode=True)
    assert dyn.param_count() == ident.param_count()
    assert sorted(n for n, _ in dyn.named_parameters()) == sorted(
        n for n, _ in ident.named_parameters()
    )
    z = torch.randn(3, contract.LATENT_DIM)
    for h in (0.0, 1.0, 5.0, 10.0, 100.0):
        assert torch.equal(ident(z, h), z)
    # ...and the full model built on top keeps the same parameter budget.
    a = _tiny_model(context_minutes=2)
    b = _tiny_model(context_minutes=2, identity_dynamics=True)
    assert a.param_count()["total"] == b.param_count()["total"]


def test_6b_describe_modes_schema():
    dyn = DampedRotationDynamics()
    d = dyn.describe_modes()
    for key in ("tau_minutes", "omega_rad_per_min", "period_minutes"):
        assert key in d and len(d[key]) == contract.N_ROTATION_MODES
    zero = [i for i, w in enumerate(d["omega_rad_per_min"]) if w == 0.0]
    assert zero, "pure-decay modes must be reachable and present at init"
    for i in zero:
        assert math.isinf(d["period_minutes"][i])
    for i, w in enumerate(d["omega_rad_per_min"]):
        assert abs(w) <= contract.OMEGA_MAX_RAD_PER_MIN + 1e-9
        if w != 0.0:
            # a rotating mode must not start faster than its own decay scale
            assert d["period_minutes"][i] >= 0.5 * d["tau_minutes"][i]


def test_6c_no_mode_starts_inside_the_aliased_band():
    """OMEGA_MAX is the Nyquist rate of a minute-sampled state.

    Every horizon and the consistency step are whole minutes, so a period below
    ``2*pi/OMEGA_MAX`` = 2 min is an exact alias of a slower mode.  No mode may
    start there, and a mode whose tau is shorter than the floor must be recorded
    as capped rather than silently saturating ``atanh``.
    """
    dyn = DampedRotationDynamics()
    d = dyn.describe_modes()
    floor = 2.0 * math.pi / contract.OMEGA_MAX_RAD_PER_MIN
    assert d["min_period_minutes"] == pytest.approx(floor)
    for i, p_min in enumerate(d["period_minutes"]):
        assert p_min >= floor, f"mode {i} starts at period {p_min:.3f} < {floor:.3f} min"
    prov = d["init_provenance"]
    assert math.isfinite(prov["tanh_ratio_cap"]) and prov["tanh_ratio_cap"] < 1.0
    expected_capped = [
        i for i in range(1, contract.N_ROTATION_MODES, 2)
        if d["tau_minutes"][i] < floor
    ]
    assert prov["capped_modes"] == expected_capped
    for i in range(contract.N_ROTATION_MODES):
        req, real = prov["requested_period_minutes"][i], prov["realised_period_minutes"][i]
        if i in prov["capped_modes"]:
            assert real > req                      # cap lengthens the period
        else:
            assert real == pytest.approx(req, rel=1e-5)
        if i % 2 == 0:
            assert math.isinf(req) and math.isinf(real)
        else:
            assert req == pytest.approx(max(d["tau_minutes"][i], floor), rel=1e-5)
    assert all(torch.isfinite(p).all() for p in dyn.parameters())


# ---------------------------------------------------------------------------
# 7-10  encoder / model
# ---------------------------------------------------------------------------


def test_7_context_encoder_is_causal_over_minutes():
    model = _tiny_model(n_contacts=4)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=7)
    with torch.no_grad():
        base = model.encode_sequence(raw, coords, cvd, shaft, sidx, cv, mv)
        raw2 = raw.clone()
        raw2[:, :, -contract.MINUTE_SAMPLES:] += 5.0
        pert = model.encode_sequence(raw2, coords, cvd, shaft, sidx, cv, mv)
    assert torch.equal(base[:, 0], pert[:, 0]), "past state moved when the future changed"
    assert not torch.allclose(base[:, -1], pert[:, -1])
    # encode() is the last context position
    with torch.no_grad():
        assert torch.equal(model.encode(raw, coords, cvd, shaft, sidx, cv, mv), base[:, -1])


def test_8_invalid_contacts_cannot_influence_the_state():
    model = _tiny_model(n_contacts=5)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=8)
    cv[:, 3] = False              # contact 3 dead for both samples
    mv[0, 1, -2:] = False         # contact 1 artefacted in the last two minutes
    with torch.no_grad():
        z_ref = model.encode(raw, coords, cvd, shaft, sidx, cv, mv)

    trash = raw.clone()
    trash[:, 3] = torch.tensor(float("inf"))
    trash[0, 1, -2 * contract.MINUTE_SAMPLES:] = torch.tensor(float("nan"))
    trash[1, 3, :100] = -1e30
    coords_trash = coords.clone()
    coords_trash[:, 3] = torch.tensor(float("nan"))
    with torch.no_grad():
        z_trash = model.encode(trash, coords_trash, cvd, shaft, sidx, cv, mv)

    assert torch.isfinite(z_trash).all()
    assert torch.allclose(z_ref, z_trash, atol=1e-5, rtol=0.0), (
        "masked-out contacts / contact-minutes leaked into z"
    )


def test_8b_attention_pool_mask_equals_dropping_the_masked_entries():
    """The pooling mask must change the *denominator*, not just zero the values.

    Zeroing an invalid contact's token is not enough: if the mask is missing,
    the softmax still spends weight on it and the pooled minute token shrinks
    with the number of dead contacts.  The model-level invariance test cannot
    see that (both runs share the same dead contacts), so it is pinned here.
    """
    torch.manual_seed(4)
    pool = AttentionPool(8).eval()
    x = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [[True, True, False, True, False], [True, False, False, False, True]]
    )
    with torch.no_grad():
        full = pool(x, mask)
        kept0 = pool(x[0:1, [0, 1, 3]])
        kept1 = pool(x[1:2, [0, 4]])
        empty = pool(x, torch.zeros_like(mask))
    assert torch.allclose(full[0], kept0[0], atol=1e-6)
    assert torch.allclose(full[1], kept1[0], atol=1e-6)
    assert torch.equal(empty, torch.zeros_like(empty))
    # one empty row beside populated ones, on the (B, M, C) layout the contact
    # pool actually uses, and with a gradient (a NaN can hide in backward only)
    x4 = torch.randn(2, 3, 5, 8, requires_grad=True)
    mask4 = torch.ones(2, 3, 5, dtype=torch.bool)
    mask4[1, 2] = False
    out4 = pool(x4, mask4)
    assert torch.isfinite(out4).all()
    assert torch.equal(out4[1, 2], torch.zeros_like(out4[1, 2]))
    out4.sum().backward()
    assert torch.isfinite(x4.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in pool.parameters())


def test_8c_a_fully_masked_minute_stays_finite():
    """Every contact masked in one minute: no NaN forward OR backward.

    Softmax over an all -inf row is NaN, and a NaN that only appears in the
    backward pass is invisible to a forward-only check.
    """
    model = _tiny_model(n_contacts=4, context_minutes=3)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=81)
    mv[1, :, 0] = False          # every contact artefacted in minute 0 of sample 1
    out = model(raw, coords, cvd, shaft, sidx, cv, mv, return_diagnostics=True)
    assert torch.isfinite(out["z"]).all()
    for h, v in out["pred"].items():
        assert torch.isfinite(v).all(), f"non-finite prediction at h={h}"
    tokens = out["per_contact_minute_tokens"]
    assert torch.isfinite(tokens).all()
    # the dead minute produces exact zeros, not "small garbage"
    assert torch.equal(tokens[1, :, 0], torch.zeros_like(tokens[1, :, 0]))
    assert torch.equal(
        out["minute_tokens"][1, 0], torch.zeros_like(out["minute_tokens"][1, 0])
    )
    out["z"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "NaN in the backward pass"


def test_8c2_every_contact_dead_for_the_whole_window():
    """Degenerate extreme: no usable contact anywhere in the context."""
    model = _tiny_model(n_contacts=4, context_minutes=2)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=83)
    cv[:] = False
    mv[:] = False
    out = model(raw, coords, cvd, shaft, sidx, cv, mv, return_diagnostics=True)
    assert torch.isfinite(out["z"]).all()
    assert torch.equal(
        out["minute_tokens"], torch.zeros_like(out["minute_tokens"])
    )
    out["z"].sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_8d_contact_pooling_receives_the_validity_mask():
    """Pins the *call site*: zeroing the tokens is not a substitute for the mask.

    ``test_8b`` pins what ``AttentionPool`` does with a mask; this pins that the
    encoder actually hands it ``contact_valid & minute_valid``, oriented (B, M, C).
    """
    model = _tiny_model(n_contacts=4, context_minutes=2)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=82)
    cv[:, 2] = False
    mv[0, 1, 0] = False
    seen = {}
    original = model.contact_pool.forward

    def spy(x, mask=None):
        seen["mask"] = None if mask is None else mask.clone()
        return original(x, mask)

    model.contact_pool.forward = spy
    with torch.no_grad():
        model.encode(raw, coords, cvd, shaft, sidx, cv, mv)
    assert seen["mask"] is not None, "contact pooling must be masked"
    assert torch.equal(seen["mask"], (cv[:, :, None] & mv).permute(0, 2, 1))


def test_8e_unlocalised_contacts_cannot_leak_a_phantom_position():
    """coord_valid=False must gate the mm term to EXACTLY zero.

    Feeding zeros to ``coord_proj`` is not enough -- the projection has a bias,
    so an unlocalised contact would still be handed a constant phantom location.
    Whatever garbage sits in its ``coords_mm`` row, ``z`` must not move.
    """
    model = _tiny_model(n_contacts=5, context_minutes=2)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=84)
    cvd[:, 2] = False                       # contact 2 is recorded but not localised
    variants = []
    for filler in (0.0, float("nan"), 1e6, -1e6):
        c2 = coords.clone()
        c2[:, 2] = torch.tensor(filler)
        with torch.no_grad():
            variants.append(model.encode(raw, c2, cvd, shaft, sidx, cv, mv))
    for v in variants[1:]:
        assert torch.isfinite(v).all()
        assert torch.equal(variants[0], v), "unlocalised coordinate leaked into z"
    # sanity: a LOCALISED contact's coordinate does matter
    cvd_all = torch.ones_like(cvd)
    c3 = coords.clone()
    c3[:, 2] += 25.0
    with torch.no_grad():
        a = model.encode(raw, coords, cvd_all, shaft, sidx, cv, mv)
        b = model.encode(raw, c3, cvd_all, shaft, sidx, cv, mv)
    assert not torch.allclose(a, b)


def test_8f_a_subject_with_no_coordinates_at_all_still_works():
    """The five Yuquan subjects with recordings but no localisation artifact.

    With every coordinate missing the position floor is shaft_id + shaft_index
    alone, and that must still tell contacts apart: moving a signal from one
    contact to another has to change the state, otherwise the encoder is blind
    to which contact anything happened on.
    """
    model = _tiny_model(n_contacts=6, n_shafts=2, context_minutes=2)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=1, seed=85)
    cvd[:] = False
    coords = torch.full_like(coords, float("nan"))
    with torch.no_grad():
        z0 = model.encode(raw, coords, cvd, shaft, sidx, cv, mv)
    assert torch.isfinite(z0).all()
    perm = raw[:, torch.tensor([3, 1, 2, 0, 5, 4])].contiguous()
    with torch.no_grad():
        z1 = model.encode(perm, coords, cvd, shaft, sidx, cv, mv)
    assert not torch.allclose(z0, z1), "coordinate-less subject cannot tell contacts apart"
    # and it still trains
    out = model(raw, coords, cvd, shaft, sidx, cv, mv)
    out["z"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_8g_shaft_index_enters_the_position_encoding():
    model = _tiny_model(n_contacts=6, n_shafts=2, context_minutes=2)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=1, seed=86)
    permuted = sidx[:, torch.tensor([2, 1, 0, 3, 4, 5])].contiguous()
    assert not torch.equal(sidx, permuted)
    with torch.no_grad():
        z0 = model.encode(raw, coords, cvd, shaft, sidx, cv, mv)
        z1 = model.encode(raw, coords, cvd, shaft, permuted, cv, mv)
    assert not torch.allclose(z0, z1), "shaft_index is not reaching the encoder"
    # out-of-range indices are clamped, not an IndexError
    with torch.no_grad():
        wild = torch.full_like(sidx, model.max_shaft_index + 500)
        assert torch.isfinite(model.encode(raw, coords, cvd, shaft, wild, cv, mv)).all()


def test_9_forbidden_inputs_are_rejected_by_the_model():
    model = _tiny_model(n_contacts=3, context_minutes=2)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=1, seed=9)
    soz = torch.ones(1, 3)
    with pytest.raises(ValueError, match="forbidden"):
        model(raw, coords, cvd, shaft, sidx, cv, mv, soz=soz)
    with pytest.raises(ValueError, match="forbidden"):
        model.encode(raw, coords, cvd, shaft, sidx, cv, mv, lagpat_rank=soz)
    # disabling the gate must not open a side door for extra kwargs
    with pytest.raises(ValueError):
        model(raw, coords, cvd, shaft, sidx, cv, mv, check_inputs=False, soz=soz)
    with pytest.raises(ValueError, match="unrecognis"):
        model(raw, coords, cvd, shaft, sidx, cv, mv, something_else=soz)
    contract.assert_no_forbidden_inputs({"raw": raw})  # sanity: allowed key passes
    assert set(contract.ALLOWED_INPUT_KEYS) == {
        "raw", "coords_mm", "coord_valid", "shaft_id", "shaft_index",
        "contact_valid", "minute_valid",
    }


def test_10_forward_shape_contract():
    model = _tiny_model(n_contacts=7, n_shafts=3)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=10)
    out = model(raw, coords, cvd, shaft, sidx, cv, mv, return_diagnostics=True)
    assert out["z"].shape == (2, contract.LATENT_DIM)
    assert set(out["pred"]) == set(contract.HORIZONS_MIN)
    for h in contract.HORIZONS_MIN:
        assert out["pred"][h].shape == (2, 7, contract.N_FREQ_BINS)
        assert torch.isfinite(out["pred"][h]).all()
    assert out["per_contact_minute_tokens"].shape == (
        2, 7, contract.CONTEXT_MINUTES, contract.D_MODEL,
    )
    assert out["minute_tokens"].shape == (2, contract.CONTEXT_MINUTES, contract.D_MODEL)
    counts = model.param_count()
    assert counts["total"] == sum(p.numel() for p in model.parameters())


def test_10b_consistency_pair_matches_two_shifted_encoder_calls():
    model = _tiny_model(n_contacts=3, context_minutes=3)
    m = contract.MINUTE_SAMPLES
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=12)
    extra_raw = torch.randn(2, 3, m)
    raw_ext = torch.cat([raw, extra_raw], dim=-1)
    mv_ext = torch.ones(2, 3, model.context_minutes + 1, dtype=torch.bool)
    with torch.no_grad():
        z_now, z_next = model.encode_consistency_pair(raw_ext, coords, cvd, shaft, sidx, cv, mv_ext)
        z_now_ref = model.encode(raw, coords, cvd, shaft, sidx, cv, mv)
        z_next_ref = model.encode(raw_ext[:, :, m:], coords, cvd, shaft, sidx, cv, mv_ext[:, :, 1:])
    assert torch.allclose(z_now, z_now_ref, atol=1e-6)
    assert torch.allclose(z_next, z_next_ref, atol=1e-6)


def test_10c_gradient_checkpointing_matches_the_plain_path():
    model = _tiny_model(n_contacts=3, context_minutes=2)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(model, batch=2, seed=13)
    with torch.no_grad():
        plain = model.encode(raw, coords, cvd, shaft, sidx, cv, mv)
    model.use_checkpoint = True
    model.checkpoint_chunk = 16          # force several chunks (144 sequences)
    ckpt = model.encode(raw, coords, cvd, shaft, sidx, cv, mv)
    assert torch.allclose(plain, ckpt, atol=1e-6)
    ckpt.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_10d_the_horizon_enters_only_through_bh():
    """One shared decoder; the horizons differ only by the dynamics map.

    With identity dynamics all four horizon fields must be bit-identical (that
    is what makes baseline #4 an honest capacity-matched control); with the real
    map they must differ.
    """
    raw_model = _tiny_model(n_contacts=3, context_minutes=2)
    ident = _tiny_model(n_contacts=3, context_minutes=2, identity_dynamics=True)
    raw, coords, cvd, shaft, sidx, cv, mv = _batch(raw_model, batch=2, seed=14)
    with torch.no_grad():
        full = raw_model(raw, coords, cvd, shaft, sidx, cv, mv)["pred"]
        flat = ident(raw, coords, cvd, shaft, sidx, cv, mv)["pred"]
    hs = list(contract.HORIZONS_MIN)
    for h in hs[1:]:
        assert torch.equal(flat[hs[0]], flat[h])
        assert not torch.allclose(full[hs[0]], full[h])


# ---------------------------------------------------------------------------
# 11-12  losses
# ---------------------------------------------------------------------------


def _one_horizon(err: float, n_valid: int, n_total: int):
    pred = torch.zeros(1, n_total, 1)
    target = torch.full((1, n_total, 1), err)
    mask = torch.zeros(1, n_total, 1, dtype=torch.bool)
    mask[0, :n_valid, 0] = True
    return pred, target, mask


def test_11_horizons_are_equally_weighted_regardless_of_valid_count():
    p1, t1, m1 = _one_horizon(2.0, n_valid=1, n_total=100)     # sq err 4, 1 entry
    p5, t5, m5 = _one_horizon(1.0, n_valid=100, n_total=100)   # sq err 1, 100 entries
    total, per_h = masked_forecast_loss({1: p1, 5: p5}, {1: t1, 5: t5}, {1: m1, 5: m5})
    assert float(per_h[1]) == pytest.approx(4.0)
    assert float(per_h[5]) == pytest.approx(1.0)
    assert float(total) == pytest.approx(2.5)          # (4+1)/2, not pooled (104/101)
    assert float(total) != pytest.approx(104.0 / 101.0)


def test_11b_empty_horizon_drops_out_of_the_denominator():
    p1, t1, m1 = _one_horizon(2.0, n_valid=1, n_total=10)
    p5, t5, m5 = _one_horizon(1.0, n_valid=100, n_total=100)
    p9, t9, m9 = _one_horizon(9.0, n_valid=0, n_total=10)
    pred = {1: p1, 5: p5, 100: p9}
    tgt = {1: t1, 5: t5, 100: t9}
    msk = {1: m1, 5: m5, 100: m9}
    total, per_h = masked_forecast_loss(pred, tgt, msk)
    assert float(total) == pytest.approx(2.5)
    assert float(per_h[100]) == 0.0                    # documented: 0.0, not NaN
    # every horizon empty -> finite zero that still carries a graph
    pred_g = {h: v.clone().requires_grad_(True) for h, v in pred.items()}
    msk0 = {h: torch.zeros_like(v) for h, v in msk.items()}
    tot0, _ = masked_forecast_loss(pred_g, tgt, msk0)
    assert float(tot0) == 0.0
    tot0.backward()
    assert all(torch.isfinite(v.grad).all() for v in pred_g.values())


def test_11c_nan_targets_in_masked_cells_do_not_poison_the_loss():
    p, t, m = _one_horizon(1.0, n_valid=4, n_total=10)
    t = t.clone()
    t[0, 5:, 0] = float("nan")
    p = p.clone().requires_grad_(True)
    total, per_h = masked_forecast_loss({1: p}, {1: t}, {1: m})
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(p.grad).all()
    assert float(p.grad[0, 5:, 0].abs().max()) == 0.0


def test_11d_total_loss_combines_forecast_and_consistency():
    p1, t1, m1 = _one_horizon(2.0, n_valid=1, n_total=10)
    z_enc = torch.zeros(3, contract.LATENT_DIM)
    z_pred = torch.full((3, contract.LATENT_DIM), 0.5)
    cons = consistency_loss(z_enc, z_pred)
    assert float(cons) == pytest.approx(0.5 * 0.25)   # Huber, |r| < delta
    tot, parts = total_loss({1: p1}, {1: t1}, {1: m1}, z_enc, z_pred, lambda_cons=0.1)
    assert float(parts["forecast"]) == pytest.approx(4.0)
    assert float(parts["consistency"]) == pytest.approx(0.125)
    assert float(tot) == pytest.approx(4.0 + 0.1 * 0.125)
    tot0, parts0 = total_loss({1: p1}, {1: t1}, {1: m1}, z_enc, z_pred, lambda_cons=0.0)
    assert float(parts0["consistency"]) == 0.0
    assert float(tot0) == pytest.approx(4.0)


def test_12_consistency_ratio_endpoints():
    torch.manual_seed(5)
    z_now = torch.randn(6, contract.LATENT_DIM)
    z_next = z_now + torch.randn(6, contract.LATENT_DIM)
    perfect = consistency_ratio(z_next, z_next.clone(), z_now)
    assert perfect.ratio.shape == (6,)
    assert float(perfect.ratio.abs().max()) == pytest.approx(0.0, abs=1e-12)
    assert float(perfect.numerator.abs().max()) == pytest.approx(0.0, abs=1e-12)
    identity = consistency_ratio(z_next, z_now.clone(), z_now)
    assert torch.allclose(identity.ratio, torch.ones(6), atol=1e-6)
    assert torch.allclose(identity.numerator, identity.denominator, atol=1e-6)


def test_12b_ratio_carries_its_own_scale():
    """A small E_cons from a collapsed state must be distinguishable."""
    torch.manual_seed(6)
    moving_now = torch.randn(4, contract.LATENT_DIM)
    moving_next = moving_now + torch.randn(4, contract.LATENT_DIM)
    good = consistency_ratio(moving_next, moving_next + 0.02 * torch.randn(4, 32), moving_now)
    collapsed = torch.full((4, contract.LATENT_DIM), 0.3)
    dead = consistency_ratio(collapsed, collapsed + 1e-7, collapsed + 1e-6)
    assert float(good.ratio.median()) < 0.2 and float(dead.ratio.median()) < 0.2
    # same ratio band, opposite findings -- only the denominator separates them
    assert float(dead.denominator.max()) < 1e-4 < float(good.denominator.min())
    assert consistency_ratio(moving_next, moving_next, moving_now)._fields == (
        "ratio", "numerator", "denominator",
    )


def test_12c_latent_diagnostics_make_collapse_visible():
    torch.manual_seed(7)
    z_now = torch.randn(8, contract.LATENT_DIM)
    z_next = z_now + 0.5 * torch.randn(8, contract.LATENT_DIM)
    healthy = latent_diagnostics(z_now, z_next)
    assert set(healthy) == {"z_std_per_dim", "z_step_norm", "n_active_dims", "n_samples"}
    assert healthy["n_active_dims"] == contract.LATENT_DIM
    assert healthy["z_std_per_dim"] > 0.5 and healthy["z_step_norm"] > 0.5

    dead = torch.full((8, contract.LATENT_DIM), 0.7)
    collapsed = latent_diagnostics(dead, dead + 1e-7)
    assert collapsed["z_std_per_dim"] == 0.0
    assert collapsed["n_active_dims"] == 0
    assert collapsed["z_step_norm"] < 1e-5

    partial = z_now.clone()
    partial[:, 4:] = 0.0
    part = latent_diagnostics(partial, partial)
    assert part["n_active_dims"] == 4
    assert part["z_step_norm"] == 0.0

    # population std -> 0, never NaN, at B = 1; n_samples says why
    single = latent_diagnostics(z_now[:1], z_next[:1])
    assert single["n_samples"] == 1
    assert single["z_std_per_dim"] == 0.0 and single["z_step_norm"] > 0.0
    assert ACTIVE_DIM_STD_THRESHOLD > 0.0


# ---------------------------------------------------------------------------
# Regression: the patch stage must never launch one attention kernel wider than
# the CUDA grid limit
# ---------------------------------------------------------------------------


def test_patch_stage_splits_above_the_cuda_grid_limit():
    """Found while benchmarking real contact counts on the 3090.

    ``scaled_dot_product_attention`` maps its batch axis onto a CUDA grid
    dimension capped at 65535 blocks. The patch stage flattens
    (batch x contacts x minutes x windows) into that axis, so batch 4 at 139
    contacts is 4*139*10*12 = 66720 sequences and the kernel dies with
    "invalid configuration argument" -- a hard launch failure, not an OOM, so no
    batch-size ladder would have recovered from it. Twelve cohort subjects have
    >=137 contacts, so this was the common case.
    """
    from src.topic5_raw_seeg_state import model as M

    net = M.RawSeegStateModel(n_contacts=4, n_shafts=2)
    net.eval()
    limit = net.MAX_ATTENTION_ROWS
    assert limit <= 65535, "the split must stay under the CUDA grid limit"

    calls = {"n": 0, "max_rows": 0}
    real = net._patch_stage

    def spy(seq):
        calls["n"] += 1
        calls["max_rows"] = max(calls["max_rows"], int(seq.shape[0]))
        return real(seq)

    net._patch_stage = spy
    seq = torch.zeros(limit + 7, contract.PATCHES_PER_WINDOW, contract.PATCH_SAMPLES)
    with torch.no_grad():
        out = net._patch_stage_chunked(seq)
    assert out.shape[0] == limit + 7, "the split must be exact, not truncating"
    assert calls["n"] == 2, f"expected 2 chunks, got {calls['n']}"
    assert calls["max_rows"] <= limit

    calls.update(n=0, max_rows=0)
    small = torch.zeros(16, contract.PATCHES_PER_WINDOW, contract.PATCH_SAMPLES)
    with torch.no_grad():
        net._patch_stage_chunked(small)
    assert calls["n"] == 1, "a small stage must stay a single call"


def test_auto_batch_size_keeps_batch_times_contacts_bounded():
    import importlib.util as _u

    spec = _u.spec_from_file_location(
        "_run_patient",
        contract.REPO_ROOT / "scripts" / "topic5_raw_seeg_state" / "run_patient.py")
    rp = _u.module_from_spec(spec)
    spec.loader.exec_module(rp)
    for enc, dm in (("transformer", 128), ("transformer", 168), ("conformer", 128)):
        cap = rp.batch_contact_cap(enc, dm)
        for n_c in (24, 31, 87, 139, 183):
            bs = rp.auto_batch_size(n_c, encoder_kind=enc, d_model=dm)
            assert 1 <= bs <= 8
            assert bs * n_c <= cap or bs == 1, (
                f"{enc}/d={dm} C={n_c} -> batch {bs} exceeds the cap {cap}")
        assert (rp.auto_batch_size(183, encoder_kind=enc, d_model=dm)
                < rp.auto_batch_size(31, encoder_kind=enc, d_model=dm)), (
            "a bigger implantation must get a smaller batch")
    # the Conformer measured 17.9 GB against the Transformer's 9.8 GB at the same
    # batch x contacts, and OOMed at 82 contacts under the Transformer cap
    assert rp.batch_contact_cap("conformer") < rp.batch_contact_cap("transformer")
    assert rp.auto_batch_size(82, encoder_kind="conformer")         < rp.auto_batch_size(82, encoder_kind="transformer")
    # R0.2 audit: cap 220 selected batch 2 at 87 contacts and failed with an
    # asynchronous CUDA OOM during backward. This path must remain batch 1.
    assert rp.auto_batch_size(87, encoder_kind="conformer") == 1

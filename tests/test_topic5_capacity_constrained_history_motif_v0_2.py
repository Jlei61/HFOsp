"""Contract tests for Topic 5.2D v0.2.

Each test names the clause of the frozen design it protects.  Surface behaviour
("does it return a number?") is deliberately not what is checked — the tests
target the invariants that would otherwise fail silently and contaminate the
science: bypass leakage, future leakage, unmatched nulls, a suffix head smuggled
into the autonomous family, STOP steering a spatial checkpoint, and the
model-unseen split being touched before everything is frozen.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_data_v0_2 import (  # noqa: E402
    DATA_FRACTIONS,
    HORIZONS,
    PatientInput,
    build_sample_set,
    derive_recording_blocks,
    load_sample_set,
    rank_sets_for_event,
)
from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    CHECKPOINT_HORIZONS,
    MotifConfig,
    OrderedMotif,
    UnorderedBaseline,
    autonomous_suffix_field,
    checkpoint_objective,
    combine_logits,
    covariant_rotation,
    evaluate,
    expected_inclusion,
    horizon_losses,
    log_subset_probability,
    perturb_prefix_order,
    primary_field_kind,
    tensors_from_samples,
    unordered_features,
)
from src.topic5_structural_identifiability_v0_2 import (  # noqa: E402
    ANGLE_GRID_RAD,
    RANKS,
    aligned_dictionary,
    estimate_axis_2d,
    identity_permutation,
    isotropic_kernel,
    local_graph,
    local_kernel_sigma,
    orthonormal_truncation,
    rotate_axis,
    shaft_basis,
)

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"


# ---------------------------------------------------------------------------
# synthetic fixtures — no real patient needed for the structural invariants
# ---------------------------------------------------------------------------
def synthetic_patient(n_events: int = 240, n_contacts: int = 9, seed: int = 0) -> PatientInput:
    rng = np.random.default_rng(seed)
    # geometry is drawn from its own stream so that changing ``seed`` changes the
    # events and nothing else — that is what lets the event-blindness tests bite
    geometry_rng = np.random.default_rng(20260817)
    coords = np.column_stack([
        np.repeat(np.arange(3.0), 3) * 11.0,
        np.tile(np.arange(3.0), 3) * 4.0,
        geometry_rng.normal(0.0, 0.3, n_contacts),
    ])
    ranks = np.full((n_events, n_contacts), -1, dtype=np.int16)
    for event in range(n_events):
        size = int(rng.integers(5, n_contacts + 1))
        members = rng.permutation(n_contacts)[:size]
        ranks[event, members] = np.arange(size)
    times = np.cumsum(rng.exponential(4.0, n_events)) + 1_000_000.0
    split = np.zeros(n_events, dtype=np.int8)
    split[int(0.6 * n_events):int(0.75 * n_events)] = 1
    split[int(0.75 * n_events):int(0.9 * n_events)] = 2
    split[int(0.9 * n_events):] = -1
    lag = np.sort(rng.random((n_events, n_contacts)), axis=1)
    return PatientInput(
        dataset="SEEG", patient="synthetic", contact_names=[f"C{i}" for i in range(n_contacts)],
        shafts=["A"] * 4 + ["B"] * (n_contacts - 4), coords_3d_mm=coords,
        contacts_xy_mm=coords[:, :2], ranks=ranks, split=split, event_abs_time=times,
        event_lag_raw=lag.astype(np.float32), recording_block=derive_recording_blocks(times),
        provenance={},
    )


@pytest.fixture(scope="module")
def sample_bundle():
    patient = synthetic_patient()
    samples = build_sample_set(patient, prefix_len=3)
    rows = np.flatnonzero(samples.split >= 0)
    return patient, samples, tensors_from_samples(samples, rows)


def make_basis(patient: PatientInput, rank: int) -> np.ndarray:
    sigma = local_kernel_sigma(patient.coords_3d_mm)
    kernel = isotropic_kernel(patient.coords_3d_mm, sigma, local_graph(patient.coords_3d_mm))
    dictionary = aligned_dictionary(
        kernel, patient.coords_3d_mm, patient.contacts_xy_mm, np.array([1.0, 0.0]), patient.shafts
    )
    return orthonormal_truncation(dictionary, rank)[0]


def make_baseline_logits(batch) -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    return {
        "contact": torch.randn(batch.n_samples, batch.n_horizons, batch.n_contacts) * 0.3,
        "cardinality": torch.randn(batch.n_samples, batch.n_horizons, batch.max_cardinality) * 0.3,
        "suffix": torch.randn(batch.n_samples, batch.n_contacts) * 0.3,
    }


# ---------------------------------------------------------------------------
# D2.1 / D2.11 — the two unordered baselines cannot read rank order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level", ["U_MINIMAL", "U_FULL_SET"])
def test_unordered_baseline_is_bitwise_order_invariant(sample_bundle, level):
    _, _, batch = sample_bundle
    module = UnorderedBaseline(level, batch.n_contacts, unordered_features(batch, level).shape[1],
                               batch.n_horizons, batch.max_cardinality, rank=4)
    with torch.no_grad():
        reference = module(unordered_features(batch, level))
    for mode in ("swap_middle", "reverse_middle"):
        permuted = perturb_prefix_order(batch, mode)
        with torch.no_grad():
            candidate = module(unordered_features(permuted, level))
        for key in reference:
            assert torch.equal(candidate[key], reference[key]), f"{level}/{mode} leaked rank order"


def test_prefix_order_perturbation_preserves_everything_the_baseline_reads(sample_bundle):
    _, _, batch = sample_bundle
    permuted = perturb_prefix_order(batch, "swap_middle")
    assert torch.equal(permuted.cumulative_set, batch.cumulative_set)
    assert torch.equal(permuted.start_set, batch.start_set)
    assert torch.equal(permuted.prefix_sets[:, 0], batch.prefix_sets[:, 0])
    assert permuted.prefix_len == batch.prefix_len
    assert torch.equal(permuted.target_cardinality, batch.target_cardinality)
    assert not torch.equal(permuted.prefix_sets, batch.prefix_sets)


def test_bug_injected_baseline_fails_the_invariance_audit(sample_bundle):
    """The audit must be able to fail, or it proves nothing."""
    _, _, batch = sample_bundle
    level = "U_FULL_SET"
    width = unordered_features(batch, level).shape[1] + batch.n_contacts
    module = UnorderedBaseline(level, batch.n_contacts, width, batch.n_horizons,
                               batch.max_cardinality, rank=4)
    leaky = lambda piece: torch.cat([unordered_features(piece, level), piece.prefix_sets[:, -1]], dim=1)
    with torch.no_grad():
        reference = module(leaky(batch))
        candidate = module(leaky(perturb_prefix_order(batch, "swap_middle")))
    assert not torch.equal(candidate["contact"], reference["contact"])


# ---------------------------------------------------------------------------
# D2.2 — ordered state reads order, orderless bag does not
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR"])
def test_ordered_state_is_order_sensitive(sample_bundle, family):
    patient, _, batch = sample_bundle
    model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", family, 4, batch.n_contacts,
                                     batch.n_horizons, batch.max_cardinality),
                         make_basis(patient, 4))
    with torch.no_grad():
        reference = model.prefix_state(batch)
        permuted = model.prefix_state(perturb_prefix_order(batch, "swap_middle"))
    assert not torch.allclose(reference, permuted)


def test_orderless_bag_is_order_blind(sample_bundle):
    patient, _, batch = sample_bundle
    model = OrderedMotif(MotifConfig("H1_ALIGNED_ORDERLESS_BAG", "ORDERLESS_BAG", 4,
                                     batch.n_contacts, batch.n_horizons, batch.max_cardinality),
                         make_basis(patient, 4))
    with torch.no_grad():
        reference = model(batch)["contact"]
        permuted = model(perturb_prefix_order(batch, "swap_middle"))["contact"]
    assert torch.equal(reference, permuted)


def test_free_low_rank_state_is_order_sensitive(sample_bundle):
    _, _, batch = sample_bundle
    model = OrderedMotif(MotifConfig("H1_FREE_LOW_RANK", "AUTONOMOUS_SHARED_OPERATOR", 4,
                                     batch.n_contacts, batch.n_horizons, batch.max_cardinality,
                                     free_basis=True), None)
    with torch.no_grad():
        assert not torch.allclose(model.prefix_state(batch),
                                  model.prefix_state(perturb_prefix_order(batch, "swap_middle")))


# ---------------------------------------------------------------------------
# D2.3 — no future rank enters the prefix state
# ---------------------------------------------------------------------------
def test_future_targets_do_not_reach_the_prefix_state(sample_bundle):
    patient, _, batch = sample_bundle
    model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", "AUTONOMOUS_SHARED_OPERATOR", 4,
                                     batch.n_contacts, batch.n_horizons, batch.max_cardinality),
                         make_basis(patient, 4))
    scrambled = batch.index(torch.arange(batch.n_samples))
    scrambled.target_sets = torch.zeros_like(batch.target_sets)
    scrambled.target_valid = torch.zeros_like(batch.target_valid)
    scrambled.target_cardinality = torch.ones_like(batch.target_cardinality)
    with torch.no_grad():
        assert torch.equal(model.prefix_state(batch), model.prefix_state(scrambled))


# ---------------------------------------------------------------------------
# D2.4 / D2.6 — masks, denominators and no-repeat
# ---------------------------------------------------------------------------
def test_missing_horizon_masks_only_that_horizon(sample_bundle):
    _, samples, batch = sample_bundle
    for slot, horizon in enumerate(HORIZONS):
        expected = samples.n_rank_sets >= samples.prefix_len + horizon
        assert np.array_equal(samples.target_valid[:, slot], expected)
    assert (samples.target_valid[:, 0].sum() >= samples.target_valid[:, -1].sum())
    losses = horizon_losses(torch.zeros(batch.n_samples, batch.n_horizons, batch.n_contacts),
                            torch.zeros(batch.n_samples, batch.n_horizons, batch.max_cardinality),
                            batch)
    for slot in range(batch.n_horizons):
        assert float(losses["count"][slot]) == float(batch.target_valid[:, slot].sum())


def test_recruited_contacts_are_never_candidates(sample_bundle):
    _, samples, _ = sample_bundle
    for slot in range(len(HORIZONS)):
        assert not np.any(samples.target_available[:, slot] & samples.cumulative_set)
        valid = samples.target_valid[:, slot]
        assert np.all(samples.target_sets[valid, slot] <= samples.target_available[valid, slot])
    assert not np.any(samples.suffix5_field & samples.cumulative_set)
    assert not np.any(samples.full_suffix_field & samples.cumulative_set)


# ---------------------------------------------------------------------------
# D2.5 — exact subset law against brute-force enumeration
# ---------------------------------------------------------------------------
def test_exact_subset_law_matches_enumeration():
    torch.manual_seed(11)
    n_contacts, kmax = 7, 3
    logits = torch.randn(1, n_contacts, dtype=torch.float64)
    available = torch.ones(1, n_contacts, dtype=torch.bool)
    available[0, -1] = False
    weights = logits[0].exp().numpy()
    indices = [i for i in range(n_contacts) if available[0, i]]
    for order in range(1, kmax + 1):
        subsets = list(itertools.combinations(indices, order))
        norm = sum(np.prod([weights[i] for i in subset]) for subset in subsets)
        total = 0.0
        for subset in subsets:
            target = torch.zeros(1, n_contacts, dtype=torch.bool)
            target[0, list(subset)] = True
            value = float(log_subset_probability(
                logits, available, target, torch.tensor([order]), kmax))
            assert value == pytest.approx(
                math.log(float(np.prod([weights[i] for i in subset])) / norm), abs=1e-9)
            total += math.exp(value)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_inclusion_marginals_match_enumeration():
    torch.manual_seed(12)
    n_contacts, kmax = 6, 3
    logits = torch.randn(1, n_contacts, dtype=torch.float64)
    available = torch.ones(1, n_contacts, dtype=torch.bool)
    mixture = torch.softmax(torch.randn(1, kmax, dtype=torch.float64), dim=1)
    weights = logits[0].exp().numpy()
    expected = np.zeros(n_contacts)
    for order in range(1, kmax + 1):
        subsets = list(itertools.combinations(range(n_contacts), order))
        norm = sum(np.prod([weights[i] for i in subset]) for subset in subsets)
        for subset in subsets:
            probability = float(np.prod([weights[i] for i in subset])) / norm
            for index in subset:
                expected[index] += float(mixture[0, order - 1]) * probability
    got = expected_inclusion(logits, available, mixture, kmax)[0].numpy()
    assert np.abs(got - expected).max() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# D2.7 / D2.8 — the autonomous family shares one operator and one readout
# ---------------------------------------------------------------------------
def test_autonomous_family_has_no_horizon_specific_readout(sample_bundle):
    patient, _, batch = sample_bundle
    model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", "AUTONOMOUS_SHARED_OPERATOR", 4,
                                     batch.n_contacts, batch.n_horizons, batch.max_cardinality),
                         make_basis(patient, 4))
    assert model.c_out.shape == (4, 4)
    assert model.card_w.shape == (4,)
    assert not hasattr(model, "c_suffix")
    with torch.no_grad():
        assert model(batch)["suffix"] is None
    merged = combine_logits(make_baseline_logits(batch), model(batch))
    assert merged["suffix"] is None, "autonomous arm must not inherit the baseline suffix head"
    assert primary_field_kind("AUTONOMOUS_SHARED_OPERATOR") == "suffix5"


def test_direct_family_keeps_horizon_specific_readouts(sample_bundle):
    patient, _, batch = sample_bundle
    model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", "DIRECT_HORIZON_UPPER_BOUND", 4,
                                     batch.n_contacts, batch.n_horizons, batch.max_cardinality),
                         make_basis(patient, 4))
    assert model.c_out.shape == (batch.n_horizons, 4, 4)
    assert hasattr(model, "c_suffix")
    assert primary_field_kind("DIRECT_HORIZON_UPPER_BOUND") == "full_suffix"


def test_autonomous_suffix_is_the_no_repeat_accumulation(sample_bundle):
    _, _, batch = sample_bundle
    torch.manual_seed(5)
    contact = torch.randn(batch.n_samples, batch.n_horizons, batch.n_contacts)
    cardinality = torch.randn(batch.n_samples, batch.n_horizons, batch.max_cardinality)
    field = autonomous_suffix_field(contact, cardinality, batch)
    survive = torch.ones(batch.n_samples, batch.n_contacts)
    for horizon in range(batch.n_horizons):
        marginal = expected_inclusion(contact[:, horizon], batch.suffix_eval_mask,
                                      torch.softmax(cardinality[:, horizon], dim=1),
                                      batch.max_cardinality)
        survive = survive * (1.0 - marginal.clamp(0.0, 1.0 - 1e-6))
    assert torch.allclose(field, 1.0 - survive, atol=1e-6)
    assert float((field * (1 - batch.suffix_eval_mask.float())).abs().max()) == 0.0


def test_autonomous_suffix_ignores_the_true_remaining_length(sample_bundle):
    _, _, batch = sample_bundle
    torch.manual_seed(6)
    contact = torch.randn(batch.n_samples, batch.n_horizons, batch.n_contacts)
    cardinality = torch.randn(batch.n_samples, batch.n_horizons, batch.max_cardinality)
    scrambled = batch.index(torch.arange(batch.n_samples))
    scrambled.target_valid = torch.zeros_like(batch.target_valid)
    scrambled.n_horizons = batch.n_horizons
    assert torch.equal(autonomous_suffix_field(contact, cardinality, batch),
                       autonomous_suffix_field(contact, cardinality, scrambled))


# ---------------------------------------------------------------------------
# D2.9 / D2.10 — parameter-count contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR"])
def test_structured_arms_share_a_parameter_count(sample_bundle, family):
    patient, _, batch = sample_bundle
    sigma = local_kernel_sigma(patient.coords_3d_mm)
    kernel = isotropic_kernel(patient.coords_3d_mm, sigma, local_graph(patient.coords_3d_mm))
    axis = np.array([1.0, 0.0])
    counts = {}
    bases = {
        "aligned": orthonormal_truncation(
            aligned_dictionary(kernel, patient.coords_3d_mm, patient.contacts_xy_mm, axis,
                               patient.shafts), 4)[0],
        "angle": orthonormal_truncation(
            aligned_dictionary(kernel, patient.coords_3d_mm, patient.contacts_xy_mm,
                               rotate_axis(axis, ANGLE_GRID_RAD[3]), patient.shafts), 4)[0],
        "shaft": shaft_basis(patient.shafts, patient.coords_3d_mm, 4)[0],
    }
    bases["permuted"] = bases["aligned"][
        identity_permutation(patient.coords_3d_mm, patient.shafts,
                             local_graph(patient.coords_3d_mm), 1)]
    for name, basis in bases.items():
        model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", family, 4, batch.n_contacts,
                                         batch.n_horizons, batch.max_cardinality), basis)
        counts[name] = sum(p.numel() for p in model.parameters())
        assert np.abs(basis.T @ basis - np.eye(4)).max() < 1e-9
    assert len(set(counts.values())) == 1, counts
    free = OrderedMotif(MotifConfig("H1_FREE_LOW_RANK", family, 4, batch.n_contacts,
                                    batch.n_horizons, batch.max_cardinality, free_basis=True), None)
    assert sum(p.numel() for p in free.parameters()) > max(counts.values())


def test_structured_model_refuses_a_free_contact_readout(sample_bundle):
    _, _, batch = sample_bundle
    with pytest.raises(ValueError):
        OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", "AUTONOMOUS_SHARED_OPERATOR", 4,
                                 batch.n_contacts, batch.n_horizons, batch.max_cardinality), None)
    with pytest.raises(ValueError):
        OrderedMotif(MotifConfig("H1_FREE_LOW_RANK", "AUTONOMOUS_SHARED_OPERATOR", 4,
                                 batch.n_contacts, batch.n_horizons, batch.max_cardinality,
                                 free_basis=True), np.eye(batch.n_contacts, 4))


# ---------------------------------------------------------------------------
# D2.12 — the ablation removes the ordered path and nothing else
# ---------------------------------------------------------------------------
def test_ordered_path_ablation_returns_exactly_the_baseline(sample_bundle):
    patient, _, batch = sample_bundle
    baseline = make_baseline_logits(batch)
    for family in ("DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR"):
        model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", family, 4, batch.n_contacts,
                                         batch.n_horizons, batch.max_cardinality),
                             make_basis(patient, 4))
        with torch.no_grad():
            residual = model(batch, ordered_path=False)
        assert float(residual["contact"].abs().max()) == 0.0
        assert float(residual["cardinality"].abs().max()) == 0.0
        merged = combine_logits(baseline, residual)
        assert torch.equal(merged["contact"], baseline["contact"])
        assert torch.equal(merged["cardinality"], baseline["cardinality"])


# ---------------------------------------------------------------------------
# D2.15 — the full F is rotation covariant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR"])
def test_full_f_covariant_rotation_reproduces_logits(sample_bundle, family):
    patient, _, batch = sample_bundle
    model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", family, 4, batch.n_contacts,
                                     batch.n_horizons, batch.max_cardinality),
                         make_basis(patient, 4))
    torch.manual_seed(21)
    rotation = torch.linalg.qr(torch.randn(4, 4))[0]
    rotated = covariant_rotation(model, rotation)
    with torch.no_grad():
        reference, candidate = model(batch), rotated(batch)
    for key in ("contact", "cardinality"):
        assert torch.allclose(reference[key], candidate[key], atol=1e-5), key
    if reference["suffix"] is not None:
        assert torch.allclose(reference["suffix"], candidate["suffix"], atol=1e-5)


def test_restricted_transition_forms_are_not_rotation_covariant(sample_bundle):
    """DIAGONAL_ONLY is a sensitivity precisely because it is basis-order dependent."""
    patient, _, batch = sample_bundle
    model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", "AUTONOMOUS_SHARED_OPERATOR", 4,
                                     batch.n_contacts, batch.n_horizons, batch.max_cardinality,
                                     f_form="DIAGONAL_ONLY"), make_basis(patient, 4))
    torch.manual_seed(22)
    rotation = torch.linalg.qr(torch.randn(4, 4))[0]
    with torch.no_grad():
        assert not torch.allclose(model(batch)["contact"],
                                  covariant_rotation(model, rotation)(batch)["contact"], atol=1e-5)


# ---------------------------------------------------------------------------
# D2.16 / B4 — STOP cannot steer the spatial checkpoint
# ---------------------------------------------------------------------------
def test_stop_loss_cannot_enter_the_spatial_checkpoint(sample_bundle):
    patient, _, batch = sample_bundle
    baseline = make_baseline_logits(batch)
    model = OrderedMotif(MotifConfig("H1_PATIENT_ALIGNED", "AUTONOMOUS_SHARED_OPERATOR", 4,
                                     batch.n_contacts, batch.n_horizons, batch.max_cardinality),
                         make_basis(patient, 4))
    contact_xy = torch.as_tensor(patient.contacts_xy_mm, dtype=torch.float32)
    result = evaluate(model, baseline, batch, contact_xy)
    score = checkpoint_objective(result, "AUTONOMOUS_SHARED_OPERATOR")
    manual = sum(
        result.per_horizon["total_nll"][h - 1] / 3.0 for h in CHECKPOINT_HORIZONS
        if not math.isnan(result.per_horizon["total_nll"][h - 1])
    ) + result.scalars["suffix5_balanced_bce"]
    assert score == pytest.approx(manual)
    assert not any("stop" in key.lower() for key in result.scalars)
    for horizon in (4, 5):
        assert horizon not in CHECKPOINT_HORIZONS


# ---------------------------------------------------------------------------
# C3 — basis contracts
# ---------------------------------------------------------------------------
def test_rank_truncations_are_strictly_nested():
    patient = synthetic_patient()
    sigma = local_kernel_sigma(patient.coords_3d_mm)
    kernel = isotropic_kernel(patient.coords_3d_mm, sigma, local_graph(patient.coords_3d_mm))
    full = orthonormal_truncation(
        aligned_dictionary(kernel, patient.coords_3d_mm, patient.contacts_xy_mm,
                           np.array([1.0, 0.0]), patient.shafts), max(RANKS))[0]
    for rank in RANKS:
        if rank > full.shape[1]:
            continue
        assert np.array_equal(full[:, :rank], orthonormal_truncation(
            aligned_dictionary(kernel, patient.coords_3d_mm, patient.contacts_xy_mm,
                               np.array([1.0, 0.0]), patient.shafts), rank)[0])


def test_angle_null_preserves_kernel_rank_and_anisotropy_strength():
    patient = synthetic_patient()
    sigma = local_kernel_sigma(patient.coords_3d_mm)
    support = local_graph(patient.coords_3d_mm)
    kernel = isotropic_kernel(patient.coords_3d_mm, sigma, support)
    axis = np.array([1.0, 0.0])
    aligned = aligned_dictionary(kernel, patient.coords_3d_mm, patient.contacts_xy_mm, axis,
                                 patient.shafts)
    for angle in ANGLE_GRID_RAD:
        rotated = aligned_dictionary(kernel, patient.coords_3d_mm, patient.contacts_xy_mm,
                                     rotate_axis(axis, angle), patient.shafts)
        assert rotated.shape == aligned.shape
        # same isotropic block, so the kernel itself is untouched
        assert np.allclose(rotated[:, :patient.n_contacts], aligned[:, :patient.n_contacts])
        forward = rotated[:, patient.n_contacts:2 * patient.n_contacts]
        backward = rotated[:, 2 * patient.n_contacts:3 * patient.n_contacts]
        aligned_forward = aligned[:, patient.n_contacts:2 * patient.n_contacts]
        aligned_backward = aligned[:, 2 * patient.n_contacts:3 * patient.n_contacts]
        assert (forward + backward).sum() == pytest.approx(
            (aligned_forward + aligned_backward).sum(), rel=1e-9)


def test_shaft_basis_reads_no_event_outcome():
    first = synthetic_patient(seed=0)
    second = synthetic_patient(seed=99)
    assert not np.array_equal(first.ranks, second.ranks)
    assert np.allclose(shaft_basis(first.shafts, first.coords_3d_mm, 4)[0],
                       shaft_basis(second.shafts, second.coords_3d_mm, 4)[0])


def test_identity_permutation_keeps_the_basis_orthonormal():
    patient = synthetic_patient()
    basis = make_basis(patient, 4)
    permutation = identity_permutation(patient.coords_3d_mm, patient.shafts,
                                       local_graph(patient.coords_3d_mm), 5)
    permuted = basis[permutation]
    assert np.abs(permuted.T @ permuted - np.eye(4)).max() < 1e-9
    assert sorted(permutation.tolist()) == list(range(patient.n_contacts))
    shafts = np.asarray(patient.shafts)
    assert np.array_equal(shafts[permutation], shafts), "permutation must stay inside shafts"


def test_aligned_axis_is_undirected():
    rng = np.random.default_rng(0)
    displacement = rng.normal(0, 1, (200, 2)) @ np.array([[3.0, 0.0], [0.0, 0.4]])
    axis, _ = estimate_axis_2d(displacement)
    flipped, _ = estimate_axis_2d(-displacement)
    assert np.allclose(axis, flipped)


# ---------------------------------------------------------------------------
# D2.17 / D2.18 — data-fraction contracts
# ---------------------------------------------------------------------------
def test_training_fractions_are_strictly_nested_and_block_stratified():
    patient = synthetic_patient(n_events=600)
    samples = build_sample_set(patient, prefix_len=3)
    masks = {fraction: samples.fraction_mask(fraction) for fraction in DATA_FRACTIONS}
    assert np.all(masks[25] <= masks[50])
    assert np.all(masks[50] <= masks[100])
    assert masks[25].sum() < masks[50].sum() < masks[100].sum()
    blocks = samples.recording_block[masks[100]]
    for fraction in (25, 50):
        assert set(samples.recording_block[masks[fraction]]).issubset(set(blocks))


@pytest.mark.skipif(not (RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv").exists(),
                    reason="manifest not built yet")
def test_learning_curve_basis_contracts():
    import pandas as pd
    table = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    learning = table[table["block"] == "LEARNING"]
    assert set(learning["data_fraction"]) == {25, 50}
    aligned = learning[learning["structure"].isin(
        ["H1_PATIENT_ALIGNED", "H1_ANGLE_ROTATED_AXIS"])]
    end_to_end = aligned[aligned["basis_fraction"] == aligned["data_fraction"]]
    fixed = aligned[aligned["basis_fraction"] == 100]
    assert len(end_to_end) > 0 and len(fixed) > 0
    assert set(aligned["basis_fraction"]) == {25, 50, 100}
    others = learning[~learning["structure"].isin(["H1_PATIENT_ALIGNED", "H1_ANGLE_ROTATED_AXIS"])]
    assert set(others["basis_fraction"]) == {100}


@pytest.mark.skipif(not (RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv").exists(),
                    reason="manifest not built yet")
def test_near_one_dimensional_patients_leave_the_two_dimensional_denominator():
    import pandas as pd
    eligibility = pd.read_csv(RESULT_ROOT / "basis" / "BASIS_ELIGIBILITY.csv")
    table = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    near_1d = set(eligibility.loc[~eligibility["angle_null_eligible"], "patient"])
    assert near_1d, "the frozen cohort contains near-one-dimensional patients"
    angle = table[table["structure"] == "H1_ANGLE_ROTATED_AXIS"]
    assert not angle[angle["patient"].isin(near_1d)]["eligible"].any()
    assert set(angle.loc[angle["patient"].isin(near_1d), "ineligible_reason"]) == {
        "ANGLE_NULL_INELIGIBLE"}
    # they still take part in every comparison that does not need a rotated axis
    other = table[(table["structure"] == "H1_PATIENT_ALIGNED") & (table["rank"] <= 2)]
    assert other[other["patient"].isin(near_1d)]["eligible"].all()


@pytest.mark.skipif(not (RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv").exists(),
                    reason="manifest not built yet")
def test_manifest_never_reaches_the_model_unseen_split():
    import pandas as pd
    table = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    assert set(table["split_scored"].astype(str)) == {"2"}


# ---------------------------------------------------------------------------
# D2.19 — the model-unseen split is fenced off
# ---------------------------------------------------------------------------
def test_trainer_refuses_model_unseen_events(sample_bundle):
    patient, samples, _ = sample_bundle
    observed = np.flatnonzero(samples.split >= 0)
    assert not (samples.split[observed] == -1).any()
    assert (samples.split == -1).any(), "the fixture must contain a model-unseen tier"
    contaminated = np.arange(samples.n_samples)
    assert (samples.split[contaminated] == -1).any()


# ---------------------------------------------------------------------------
# data-construction contracts
# ---------------------------------------------------------------------------
def test_rank_sets_are_ordered_and_partition_the_participants():
    row = np.array([3, 0, -1, 1, 3, 2], dtype=np.int16)
    sets = rank_sets_for_event(row)
    assert [sorted(part.tolist()) for part in sets] == [[1], [3], [5], [0, 4]]
    assert sum(len(part) for part in sets) == int((row >= 0).sum())


def test_late_field_centroid_uses_the_final_fifth_of_the_event():
    patient = synthetic_patient(n_events=40, seed=3)
    samples = build_sample_set(patient, prefix_len=3)
    for position in range(min(10, samples.n_samples)):
        event = samples.event_index[position]
        sets = rank_sets_for_event(patient.ranks[event])
        tail_start = min(max(3, int(np.ceil(0.8 * len(sets)))), len(sets) - 1)
        contacts = np.concatenate(sets[tail_start:])
        assert np.allclose(samples.late_field_centroid[position],
                           patient.contacts_xy_mm[contacts].mean(axis=0), atol=1e-4)


def test_latency_proxy_is_relative_to_the_prefix_end():
    patient = synthetic_patient(n_events=40, seed=4)
    samples = build_sample_set(patient, prefix_len=3)
    for position in range(min(10, samples.n_samples)):
        event = samples.event_index[position]
        sets = rank_sets_for_event(patient.ranks[event])
        reference = patient.event_lag_raw[event][sets[2]].mean()
        for slot, horizon in enumerate(HORIZONS):
            index = 3 + horizon - 1
            if index < len(sets):
                expected = patient.event_lag_raw[event][sets[index]].mean() - reference
                assert samples.latency_proxy[position, slot] == pytest.approx(expected, abs=1e-4)


@pytest.mark.skipif(not (RESULT_ROOT / "sample_cache" / "prefix3").exists(),
                    reason="sample cache not built yet")
def test_real_cache_round_trips_and_keeps_split_hashes():
    audit = json.loads((RESULT_ROOT / "SPLIT_HASH_AUDIT.json").read_text())
    assert audit["seeg_split_parity_all_pass"]
    assert audit["seeg_model_unseen_equals_parent_heldout"]
    assert audit["nested_subsets_all_pass"]
    path = sorted((RESULT_ROOT / "sample_cache" / "prefix3").glob("*.npz"))[0]
    samples = load_sample_set(path)
    digest = hashlib.sha256(np.ascontiguousarray(samples.event_index).tobytes()).hexdigest()
    stored = audit["per_patient"][samples.patient]["prefix"]["3"]["event_ids_sha256"]
    assert digest == stored


# ---------------------------------------------------------------------------
# singleton-cardinality fast path must equal the general dynamic program
# ---------------------------------------------------------------------------
def test_singleton_fast_path_matches_the_general_subset_law():
    """26 of 28 SEEG patients never tie, so the closed form runs in production."""
    import src.topic5_strict_history_motif_v0_2 as motif
    torch.manual_seed(31)
    batch, n_contacts = 64, 11
    logits = torch.randn(batch, n_contacts, dtype=torch.float64)
    available = torch.rand(batch, n_contacts) > 0.3
    available[:, 0] = True
    target = torch.zeros(batch, n_contacts, dtype=torch.bool)
    picks = torch.stack([torch.multinomial(available[row].double(), 1)[0] for row in range(batch)])
    target[torch.arange(batch), picks] = True
    ones = torch.ones(batch, dtype=torch.long)

    fast = motif.log_subset_probability(logits, available, target, ones, 1)
    general = (torch.where(target, logits, torch.zeros_like(logits)).sum(dim=1)
               - motif.log_esp_suffix(logits, available, 1)[:, 0, 1])
    assert torch.allclose(fast, general, atol=1e-12)

    mixture = torch.ones(batch, 1, dtype=torch.float64)
    fast_marginal = motif.expected_inclusion(logits, available, mixture, 1)
    saved = motif.expected_inclusion.__wrapped__ if hasattr(motif.expected_inclusion, "__wrapped__") else None
    del saved
    # general path, forced by asking for kmax=2 with all mass on n=1
    padded = torch.zeros(batch, 2, dtype=torch.float64)
    padded[:, 0] = 1.0
    general_marginal = motif.expected_inclusion(logits, available, padded, 2)
    assert torch.allclose(fast_marginal, general_marginal, atol=1e-9)
    assert torch.allclose(fast_marginal.sum(dim=1), torch.ones(batch, dtype=torch.float64), atol=1e-9)


def test_singleton_fast_path_sampler_respects_availability():
    import src.topic5_strict_history_motif_v0_2 as motif
    torch.manual_seed(32)
    batch, n_contacts = 512, 9
    logits = torch.randn(batch, n_contacts)
    available = torch.rand(batch, n_contacts) > 0.4
    available[:, 0] = True
    generator = torch.Generator().manual_seed(5)
    drawn = motif.sample_subset(logits, available, torch.ones(batch, dtype=torch.long), 1, generator)
    assert torch.equal(drawn.sum(dim=1), torch.ones(batch, dtype=torch.long))
    assert bool((drawn & ~available).sum() == 0)
    empirical = drawn.double().mean(dim=0)
    exact = motif.expected_inclusion(logits, available,
                                     torch.ones(batch, 1), 1).mean(dim=0).double()
    assert float((empirical - exact).abs().max()) < 0.05


# --------------------------------------------------------------------------
# Review fixes (2026-08-19): the aggregation-side statistics added after the
# v0.2 matrix was already complete.  These are pure functions on tables, so
# they are tested directly rather than through a run.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aggregator():
    import importlib.util

    path = ROOT / "scripts" / "aggregate_topic5_capacity_v0_2.py"
    spec = importlib.util.spec_from_file_location("_agg_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_aware_interval_widens_when_retraining_moves_the_effect(aggregator):
    """The whole point of the seed-aware interval is to stop hiding run variance.

    Two cohorts with the same per-patient median must not get the same interval
    when one of them has runs that disagree with each other.
    """
    tight = {f"p{i}": [0.01, 0.01, 0.01] for i in range(20)}
    loose = {f"p{i}": [-0.19, 0.01, 0.21] for i in range(20)}
    tight_out = aggregator.seed_aware_interval(tight, "tight")
    loose_out = aggregator.seed_aware_interval(loose, "loose")

    assert tight_out["median"] == pytest.approx(loose_out["median"])
    tight_low, tight_high = tight_out["median_ci95_seed_aware"]
    loose_low, loose_high = loose_out["median_ci95_seed_aware"]
    assert (loose_high - loose_low) > 10 * (tight_high - tight_low)
    assert tight_out["crosses_zero"] is False
    assert loose_out["crosses_zero"] is True


def test_seed_aware_interval_refuses_a_cohort_too_small_to_bootstrap(aggregator):
    out = aggregator.seed_aware_interval({"p0": [0.1], "p1": [0.2]}, "two patients")
    assert "median_ci95_seed_aware" not in out
    assert out["n"] == 2


def test_seed_aware_interval_ignores_patients_with_no_finite_run(aggregator):
    runs = {"p0": [0.01] * 3, "p1": [], "p2": [float("nan")], "p3": [0.01] * 3,
            "p4": [0.01] * 3}
    out = aggregator.seed_aware_interval(runs, "with gaps")
    assert out["n"] == 3


def test_runs_contrast_keeps_every_run_pair_not_just_the_medians(aggregator):
    import pandas as pd

    frame = pd.DataFrame([
        {"patient": "p0", "family": "F", "structure": "REF", "primary_objective": 1.0},
        {"patient": "p0", "family": "F", "structure": "REF", "primary_objective": 3.0},
        {"patient": "p0", "family": "F", "structure": "CMP", "primary_objective": 0.5},
        {"patient": "p1", "family": "F", "structure": "REF", "primary_objective": 2.0},
        {"patient": "p1", "family": "F", "structure": "CMP", "primary_objective": 1.0},
    ])
    out = aggregator.runs_contrast(frame, {}, ("F", "REF"), ("F", "CMP"))
    assert sorted(out["p0"]) == [0.5, 2.5]
    assert out["p1"] == [1.0]


def test_common_target_objective_puts_both_families_on_one_scale(aggregator):
    """The two families close on different fields by design; a difference BETWEEN
    them therefore needs a single target, and suffix5 exists for every unit."""
    per_horizon = {"total_nll": [1.0, 2.0, 3.0, 4.0, 5.0]}
    scalars = {"suffix5_balanced_bce": 0.25, "full_suffix_balanced_bce": 0.75}

    direct = aggregator.objective_from(scalars, per_horizon, "DIRECT_HORIZON_UPPER_BOUND")
    autonomous = aggregator.objective_from(scalars, per_horizon, "AUTONOMOUS_SHARED_OPERATOR")
    assert direct != autonomous  # each closes on its own field: the frozen design

    common_direct = aggregator.objective_from(
        scalars, per_horizon, "AUTONOMOUS_SHARED_OPERATOR")
    assert common_direct == autonomous  # and the recompute uses one field for both

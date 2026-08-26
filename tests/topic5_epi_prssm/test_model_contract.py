"""Model contract: state isolation, graph support, resource bounds, observer limits."""
import inspect

import numpy as np
import pytest
import torch

from src.topic5_epi_prssm.contracts import FROZEN
from src.topic5_epi_prssm.cohort import load_tensors
from src.topic5_epi_prssm.graph_cells import GeneratorCell
from src.topic5_epi_prssm.model import EpiPRSSM, SlowState, build_cohort_batch
from src.topic5_epi_prssm.observer import PersistentObserver
from src.topic5_epi_prssm.resource_dynamics import ResourceState
from src.topic5_epi_prssm.rollout import cohort_scan, scan_loss
from src.topic5_epi_prssm.trainer import TrainConfig, make_split_batches

SUBJECTS = ("epilepsiae_1084", "yuquan_gaolan", "yuquan_zhangbichen")


@pytest.fixture(scope="module")
def patients():
    return load_tensors(SUBJECTS)


@pytest.fixture(scope="module")
def batch(patients):
    return build_cohort_batch(patients, [0, 0, 0], [128, 128, 128])


def test_three_state_objects_are_distinct_fields():
    fields = set(SlowState.__dataclass_fields__)
    assert {"state", "resource", "observer_state", "exposure"} <= fields
    source = inspect.getsource(SlowState)
    assert "state" in source and "observer_state" in source


def test_g0_never_touches_the_message_function():
    cell = GeneratorCell("G0", 8)
    assert cell.uses_messages is False
    assert cell.message is None
    A = torch.rand(1, 3, 7, 7)
    H = torch.randn(1, 7, 8)
    out = cell.propagate(H, torch.tensor([1.0]), A, torch.ones(1), torch.ones(1, 7))
    # G0's output must be reproducible with the graph deleted
    zeroed = cell.propagate(H, torch.tensor([1.0]), torch.zeros_like(A), torch.ones(1),
                            torch.ones(1, 7))
    assert torch.allclose(out, zeroed)


@pytest.mark.parametrize("level", ["G1", "G2", "G3"])
def test_messages_travel_only_along_graph_support(level):
    """Zeroing every edge into node 0 must remove node 0's dependence on other nodes."""
    cell = GeneratorCell(level, 8, use_resource=(level == "G3"))
    torch.manual_seed(0)
    A = torch.zeros(1, 3, 4, 4)
    A[0, 0, 1, 2] = 1.0                      # a single edge 1 -> 2, node 0 isolated
    H = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4)
    base = cell.propagate(H, torch.tensor([1.0]), A, torch.ones(1), mask)
    perturbed = H.clone()
    perturbed[0, 3] += 5.0                   # perturb an unconnected node
    moved = cell.propagate(perturbed, torch.tensor([1.0]), A, torch.ones(1), mask)
    if level == "G1":
        assert torch.allclose(base[0, 0], moved[0, 0], atol=1e-6)
    else:
        # G2/G3 additionally pool the state, which is a declared global term;
        # the graph message into node 0 must still be exactly zero
        msg = cell.message(H, A)
        assert torch.allclose(msg[0, 0], torch.zeros(8), atol=1e-7)


def test_state_gauge_is_bounded():
    cell = GeneratorCell("G2", 8)
    H = torch.randn(1, 6, 8) * 100.0
    A = torch.rand(1, 3, 6, 6)
    out = cell.propagate(H, torch.tensor([1.0]), A, torch.ones(1), torch.ones(1, 6))
    assert float(out.abs().max()) <= 8.0 + 1e-6


def test_padded_lanes_stay_at_zero(batch):
    model = EpiPRSSM(generator_level="G2", resource_arm="R1", adapter="node_film")
    result = cohort_scan(model, batch, 0, 16, model.initial_state(batch))
    for p, patient in enumerate(batch.patients):
        pad = result.state_minus[:, p, patient.n_contacts:, :]
        assert torch.allclose(pad, torch.zeros_like(pad))


@pytest.mark.parametrize("arm", ["R1", "R2", "R3"])
def test_resource_stays_inside_the_unit_interval(arm):
    resource = ResourceState(arm, 8, tau_r_seconds=1800.0)
    r = torch.ones(2)
    H = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5)
    for dt in (0.1, 1.0, 3600.0, 521725.848):
        r = resource.propagate(r, H, torch.full((2,), dt), mask, exposure=torch.full((2,), 3.0))
        assert torch.isfinite(r).all()
        assert float(r.min()) > 0.0 and float(r.max()) <= 1.0
    r = resource.absorb_event(r, torch.full((2,), 1.0))
    assert float(r.min()) > 0.0 and float(r.max()) <= 1.0


def test_primary_observer_has_no_resource_write_path():
    names = {name for name, _ in inspect.getmembers(PersistentObserver, inspect.isfunction)}
    assert not any("resource" in name for name in names)
    source = inspect.getsource(PersistentObserver)
    assert "correct_resource_every_event" not in source


def test_primary_observer_leaves_the_resource_untouched(batch):
    model = EpiPRSSM(generator_level="G2", resource_arm="R1", adapter="node_film")
    z = model.initial_state(batch)
    step = batch.gather(0, 4)
    moved = model.propagate(z, batch, step, 0)
    observed, _, _ = model.observe(moved, batch, step, 0)
    assert torch.equal(observed.resource, moved.resource)


def test_flexible_control_is_the_only_arm_that_writes_the_resource(batch):
    model = EpiPRSSM(generator_level="G2", resource_arm="R1", adapter="node_film",
                     flexible_resource_correction=True)
    with torch.no_grad():
        model.flexible.write.weight.fill_(1.0)
        model.flexible.log_gain.fill_(3.0)
    z = model.initial_state(batch)
    step = batch.gather(0, 4)
    moved = model.propagate(z, batch, step, 0)
    observed, _, penalty = model.observe(moved, batch, step, 0)
    assert not torch.equal(observed.resource, moved.resource)
    assert float(penalty) > 0.0


def test_correction_and_physical_transition_are_reported_separately(batch):
    model = EpiPRSSM(generator_level="G2", resource_arm="R0", adapter="node_film")
    on = cohort_scan(model, batch, 0, 16, model.initial_state(batch), correction_on=True)
    off = cohort_scan(model, batch, 0, 16, model.initial_state(batch), correction_on=False)
    assert float(on.correction_energy) >= 0.0
    assert float(off.correction_energy) == 0.0


def test_observer_off_rollout_never_reads_future_marks(batch):
    """Corrupting every mark must not change an observer-off trajectory."""
    model = EpiPRSSM(generator_level="G2", resource_arm="R2", adapter="node_film")
    z = model.initial_state(batch)
    clean = cohort_scan(model, batch, 0, 32, z, correction_on=False,
                        expected_load=torch.full((batch.n_patients,), 0.5))
    import copy
    corrupted = copy.copy(batch)
    poisoned = []
    for patient in batch.patients:
        clone = copy.copy(patient)
        clone.marks = torch.randn_like(patient.marks) * 10.0
        clone.load = torch.rand_like(patient.load)
        poisoned.append(clone)
    corrupted.patients = tuple(poisoned)
    dirty = cohort_scan(model, corrupted, 0, 32, z, correction_on=False,
                        expected_load=torch.full((batch.n_patients,), 0.5))
    assert torch.allclose(clean.state_minus, dirty.state_minus, atol=1e-6)
    assert torch.allclose(clean.resource_minus, dirty.resource_minus, atol=1e-6)


def test_future_load_does_not_leak_into_the_exposure_arm(batch):
    model = EpiPRSSM(generator_level="G2", resource_arm="R3", adapter="node_film",
                     tau_x_seconds=1800.0)
    z = model.initial_state(batch)
    a = cohort_scan(model, batch, 0, 24, z, correction_on=False,
                    expected_load=torch.full((batch.n_patients,), 0.4))
    import copy
    other = copy.copy(batch)
    other.patients = tuple(
        [_with_load(p, torch.ones_like(p.load)) for p in batch.patients])
    b = cohort_scan(model, other, 0, 24, z, correction_on=False,
                    expected_load=torch.full((batch.n_patients,), 0.4))
    assert torch.allclose(a.exposure_minus, b.exposure_minus, atol=1e-6)


def _with_load(patient, load):
    import copy
    clone = copy.copy(patient)
    clone.load = load
    return clone


def test_tbptt_truncates_gradient_but_not_forward_state(batch):
    model = EpiPRSSM(generator_level="G2", resource_arm="R0", adapter="node_film")
    z0 = model.initial_state(batch)
    whole = cohort_scan(model, batch, 0, 64, z0)
    first = cohort_scan(model, batch, 0, 32, z0)
    second = cohort_scan(model, batch, 32, 64, first.final.detach())
    assert torch.allclose(whole.state_minus[32:], second.state_minus, atol=1e-5)
    assert not second.state_minus.requires_grad or second.state_minus.grad_fn is not None


def test_integration_is_stable_at_the_largest_real_gap(patients):
    """The cohort's largest real inter-event interval is 5.2e5 s."""
    import copy
    stressed = []
    for patient in patients:
        clone = copy.copy(patient)
        clone.delta_t = torch.full_like(patient.delta_t, 521725.848)
        clone.log_delta_t = torch.log1p(clone.delta_t)
        stressed.append(clone)
    stress_batch = build_cohort_batch(stressed, [0, 0, 0], [32, 32, 32])
    for level in ("G0", "G1", "G2", "G3"):
        model = EpiPRSSM(generator_level=level, resource_arm="R1" if level == "G3" else "R0",
                         adapter="node_film")
        result = cohort_scan(model, stress_batch, 0, 32, model.initial_state(stress_batch))
        assert torch.isfinite(result.state_minus).all(), level
        assert torch.isfinite(result.resource_minus).all(), level


def test_patients_with_different_contact_counts_train_together(patients):
    config = TrainConfig(max_epochs=1, tbptt_length=16, max_train_events_per_patient=64)
    train_batch, _ = make_split_batches(patients, config)
    model = EpiPRSSM(generator_level="G2", resource_arm="R1", adapter="node_film")
    result = cohort_scan(model, train_batch, 0, 16, model.initial_state(train_batch))
    loss = scan_loss(model, train_batch, result, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_state_time_constants_are_initialised_where_they_are_claimed_to_be():
    """A softplus parametrisation silently turns log(300) into 5.7 s."""
    from src.topic5_epi_prssm.graph_cells import TAU_MAX_INIT, TAU_MIN_INIT
    cell = GeneratorCell("G2", 8)
    tau = cell.time_constants().detach().numpy()
    assert tau.min() == pytest.approx(TAU_MIN_INIT, rel=1e-3)
    assert tau.max() == pytest.approx(TAU_MAX_INIT, rel=1e-3)
    assert tau.min() < 60.0 < tau.max(), "the bank must straddle a minute"
    assert tau.max() > 3600.0, "the bank must reach the hour scale the hypothesis is about"


def test_a_single_optimiser_step_can_move_a_time_constant_multiplicatively():
    """The state must be able to reach hours inside the optimisation budget."""
    cell = GeneratorCell("G2", 8)
    before = cell.time_constants().detach().clone()
    with torch.no_grad():
        cell.log_tau += 0.05                      # one plausible optimiser step in log space
    after = cell.time_constants().detach()
    ratio = (after / before).numpy()
    assert np.allclose(ratio, np.exp(0.05), rtol=1e-5)
    with torch.no_grad():
        cell.log_tau += 3.0                       # a few hundred steps' worth
    assert float(cell.time_constants().max()) > 10 * float(before.max())


def test_a_no_state_reference_is_actually_state_free(patients):
    """The STOP and participation heads must not see H when the adapter is no_state."""
    patient = patients[0]
    model = EpiPRSSM(generator_level="G2", resource_arm="R1", adapter="no_state",
                     feature_dim=patient.node_features.shape[-1])
    index = torch.arange(0, 24)
    zeros = torch.zeros(24, patient.n_contacts, model.state_dim)
    loud = torch.randn(24, patient.n_contacts, model.state_dim) * 3.0
    resource_a = torch.ones(24)
    resource_b = torch.full((24,), 0.2)
    a = model.score_events(patient, index, zeros, resource_a)
    b = model.score_events(patient, index, loud, resource_b)
    for key in ("order_nll", "selection_nll", "stop_nll", "participation_nll"):
        assert torch.allclose(a[key], b[key], atol=1e-6), key


def test_a_state_adapter_does_see_the_state(patients):
    patient = patients[0]
    model = EpiPRSSM(generator_level="G2", resource_arm="R0", adapter="node_film",
                     feature_dim=patient.node_features.shape[-1])
    with torch.no_grad():
        model.adapter.node_head.weight.normal_(0, 0.5)
        model.decoder.participation_head.weight.normal_(0, 0.5)
    index = torch.arange(0, 24)
    zeros = torch.zeros(24, patient.n_contacts, model.state_dim)
    loud = torch.randn(24, patient.n_contacts, model.state_dim) * 3.0
    a = model.score_events(patient, index, zeros, torch.ones(24))
    b = model.score_events(patient, index, loud, torch.ones(24))
    assert not torch.allclose(a["order_nll"], b["order_nll"], atol=1e-6)


def test_the_timing_baseline_never_sees_which_contacts_took_part(patients):
    """Its state must be a function of the observable timing features and nothing else."""
    import copy
    if any(p.nuisance is None for p in patients):
        pytest.skip("nuisance features not built yet")
    batch = build_cohort_batch(patients, [0] * len(patients), [64] * len(patients))
    model = EpiPRSSM(generator_level="G0", resource_arm="R0", adapter="node_film",
                     state_from_nuisance=True,
                     nuisance_dim=patients[0].nuisance.shape[-1],
                     feature_dim=patients[0].node_features.shape[-1])
    clean = cohort_scan(model, batch, 0, 32, model.initial_state(batch))
    poisoned = copy.copy(batch)
    poisoned.patients = tuple(_poison(p) for p in patients)
    dirty = cohort_scan(model, poisoned, 0, 32, model.initial_state(poisoned))
    assert torch.allclose(clean.state_minus, dirty.state_minus, atol=1e-6)
    assert float(clean.correction_energy) == 0.0


def test_the_timing_baseline_does_track_the_timing_features(patients):
    import copy
    if any(p.nuisance is None for p in patients):
        pytest.skip("nuisance features not built yet")
    batch = build_cohort_batch(patients, [0] * len(patients), [64] * len(patients))
    model = EpiPRSSM(generator_level="G0", resource_arm="R0", adapter="node_film",
                     state_from_nuisance=True,
                     nuisance_dim=patients[0].nuisance.shape[-1],
                     feature_dim=patients[0].node_features.shape[-1])
    with torch.no_grad():
        model.nuisance_head.weight.normal_(0, 0.8)
    clean = cohort_scan(model, batch, 0, 32, model.initial_state(batch))
    shifted = copy.copy(batch)
    shifted.patients = tuple(_shift_nuisance(p) for p in patients)
    moved = cohort_scan(model, shifted, 0, 32, model.initial_state(shifted))
    assert not torch.allclose(clean.state_minus, moved.state_minus, atol=1e-6)


def _poison(patient):
    import copy
    clone = copy.copy(patient)
    clone.marks = torch.randn_like(patient.marks) * 5.0
    clone.load = torch.rand_like(patient.load)
    return clone


def _shift_nuisance(patient):
    import copy
    clone = copy.copy(patient)
    clone.nuisance = patient.nuisance + 1.0
    return clone

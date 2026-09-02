"""Phase 0 is a measurement gate, so its own instruments are tested first.

If the teachers do not carry the order content the plan says they carry, every gate
downstream reads the wrong thing.  These tests check the generative rules directly,
not the numbers that come out of training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_phase0_measurement_validity_v0_3 import (  # noqa: E402
    SET_ONLY_KINDS,
    TEACHER_KINDS,
    TeacherField,
    TeacherSpec,
    generate_ranks,
    order_information,
    start_weights,
)
from src.topic5_strict_history_motif_v0_2 import MotifConfig, OrderedMotif  # noqa: E402


def montage(n_shafts: int = 4, per_shaft: int = 8, seed: int = 0) -> np.ndarray:
    """A plausible SEEG layout: straight shafts fanning through the plane."""
    rng = np.random.default_rng(seed)
    points = []
    for shaft in range(n_shafts):
        angle = 0.4 * shaft
        origin = rng.normal(0.0, 6.0, 2)
        step = np.array([np.cos(angle), np.sin(angle)]) * 3.5
        points.extend(origin + step * index for index in range(per_shaft))
    return np.asarray(points)


def field_for(kind: str, seed: int = 0) -> TeacherField:
    return TeacherField(TeacherSpec(kind=kind, patient="TEST", seed=seed), montage(seed=seed))


# -- the generative rules ---------------------------------------------------


@pytest.mark.parametrize("kind", SET_ONLY_KINDS)
def test_set_only_teachers_ignore_the_order_of_the_prefix(kind):
    """T1 and T2 must return identical logits for every permutation of the same set.

    T2 is the one that matters: it travels along an axis and therefore *looks*
    directional, but it reads only the front of the recruited set.  If this test ever
    fails, T2 has become a second power teacher and the gate loses its ability to
    distinguish "no order information" from "blind instrument".
    """
    field = field_for(kind)
    rng = np.random.default_rng(1)
    prefix = rng.choice(field.n_contacts, size=4, replace=False)
    reference = field.logits(prefix)
    for _ in range(12):
        permuted = rng.permutation(prefix)
        assert np.array_equal(np.isfinite(reference), np.isfinite(field.logits(permuted)))
        finite = np.isfinite(reference)
        assert field.logits(permuted)[finite] == pytest.approx(reference[finite], abs=1e-12)


@pytest.mark.parametrize("kind", ("T3_TWO_MODE", "T4_HIDDEN_RELAY"))
def test_two_mode_teachers_change_their_mind_when_the_order_is_reversed(kind):
    field = field_for(kind)
    rng = np.random.default_rng(2)
    for _ in range(8):
        prefix = rng.choice(field.n_contacts, size=3, replace=False)
        forward = field.probabilities(prefix)
        backward = field.probabilities(prefix[::-1])
        if np.abs(field.projection[prefix[0]] - field.projection[prefix[1]]) < 1e-6:
            continue
        assert np.abs(forward - backward).max() > 1e-3


def test_ground_truth_order_information_is_zero_for_the_set_only_teachers():
    for kind in SET_ONLY_KINDS:
        field = field_for(kind)
        ranks = generate_ranks(field, np.full(60, 6), seed=5)
        truth = order_information(field, ranks, n_orders=24, max_events=40, seed=3)
        assert truth["order_information_nats"] == pytest.approx(0.0, abs=1e-9), kind


def test_ground_truth_order_information_is_large_for_the_two_mode_teacher():
    blind = field_for("T1_ORDER_BLIND")
    directed = field_for("T2_SINGLE_DIRECTED")
    two_mode = field_for("T3_TWO_MODE")
    values = {}
    for name, field in (("T1", blind), ("T2", directed), ("T3", two_mode)):
        ranks = generate_ranks(field, np.full(60, 6), seed=5)
        values[name] = order_information(
            field, ranks, n_orders=24, max_events=40, seed=3)["order_information_nats"]
    assert values["T3"] > 0.05
    assert values["T3"] > 20 * max(values["T1"], values["T2"], 1e-6)


def test_equal_weighting_would_inflate_the_two_mode_ground_truth():
    """The importance weighting is load-bearing, so its absence must be visible.

    Both estimates are taken from the SAME sampled permutations and differ only in
    how those permutations are weighted, so the comparison is paired and the noise
    of the permutation draw cancels.  Equal weighting credits orders the teacher
    would rarely produce, which inflates the ground truth; T1 and T2 are pure set
    functions, so the weighting cannot matter there and only T3 can show this.
    """
    field = field_for("T3_TWO_MODE")
    ranks = generate_ranks(field, np.full(60, 6), seed=5)
    weight = start_weights(field)
    rng = np.random.default_rng(3)

    weighted, naive = [], []
    for row in np.flatnonzero((ranks >= 0).sum(axis=1) > 3)[:40]:
        present = np.flatnonzero(ranks[row] >= 0)
        order = present[np.argsort(ranks[row][present])][:3]
        candidates = [order] + [rng.permutation(order) for _ in range(32)]
        stack = np.stack([field.probabilities(c) for c in candidates])

        log_weight = np.array(
            [field.log_path_probability(np.asarray(c), weight) for c in candidates])
        importance = np.exp(log_weight - log_weight.max())
        importance /= importance.sum()

        for label, marginal in (("weighted", importance @ stack), ("naive", stack.mean(axis=0))):
            support = (stack[0] > 0) & (marginal > 0)
            value = float((stack[0][support]
                           * np.log(stack[0][support] / marginal[support])).sum())
            (weighted if label == "weighted" else naive).append(value)

    # equal weighting inflates the estimate on average, not on every single event,
    # so the direction is asserted with a sign test rather than a made-up fraction
    from scipy import stats

    paired = np.asarray(naive) - np.asarray(weighted)
    assert np.mean(weighted) > 0.02
    assert np.mean(paired) > 0.0
    assert stats.binomtest(int((paired > 0).sum()), paired.size, 0.5,
                           alternative="greater").pvalue < 0.05


def test_events_from_the_directed_teacher_actually_travel_along_the_axis():
    """T2 has to be directional data, otherwise it is just a second copy of T1."""
    field = field_for("T2_SINGLE_DIRECTED")
    ranks = generate_ranks(field, np.full(120, 6), seed=5)
    steps = []
    for row in range(ranks.shape[0]):
        present = np.flatnonzero(ranks[row] >= 0)
        if present.size < 3:
            continue
        order = present[np.argsort(ranks[row][present])]
        steps.append(np.diff(field.projection[order]).mean())
    assert np.mean(steps) > 0.0
    assert np.mean(np.asarray(steps) > 0) > 0.7


@pytest.mark.parametrize("kind", TEACHER_KINDS)
def test_start_weights_are_a_distribution(kind):
    weight = start_weights(field_for(kind))
    assert weight.min() >= 0.0
    assert weight.sum() == pytest.approx(1.0)


def test_two_mode_events_start_near_the_middle_so_the_set_hides_the_direction():
    field = field_for("T3_TWO_MODE")
    weight = start_weights(field)
    centre = float(weight @ field.projection)
    assert abs(centre - np.median(field.projection)) < 0.1


# -- the student arm the gate needs ----------------------------------------


@pytest.fixture(scope="module")
def teacher_batch():
    """A real sample set built from one of the Phase 0 teachers."""
    from src.topic5_strict_history_data_v0_2 import PatientInput, build_sample_set
    from src.topic5_strict_history_motif_v0_2 import tensors_from_samples

    coords = montage(seed=4)
    field = TeacherField(TeacherSpec(kind="T3_TWO_MODE", patient="TEST", seed=4), coords)
    ranks = generate_ranks(field, np.full(240, 7), seed=11)
    n_events, n_contacts = ranks.shape
    split = np.zeros(n_events, dtype=np.int8)
    split[int(0.6 * n_events):int(0.8 * n_events)] = 1
    split[int(0.8 * n_events):] = 2
    times = np.cumsum(np.full(n_events, 3.0)) + 1_000_000.0
    patient = PatientInput(
        dataset="SEEG", patient="TEST__T3", contact_names=[f"c{i}" for i in range(n_contacts)],
        shafts=[f"s{i // 8}" for i in range(n_contacts)],
        coords_3d_mm=np.column_stack([coords, np.zeros(n_contacts)]),
        contacts_xy_mm=coords, ranks=ranks, split=split, event_abs_time=times,
        event_lag_raw=np.zeros((n_events, n_contacts)),
        recording_block=np.zeros(n_events, dtype=int), provenance={})
    samples = build_sample_set(patient, prefix_len=3)
    rows = np.flatnonzero(samples.split >= 0)
    return tensors_from_samples(samples, rows)


def bag_and_ordered(n_contacts: int = 24, rank: int = 4):
    shared = dict(structure="H1_FREE_LOW_RANK", rank=rank, n_contacts=n_contacts,
                  n_horizons=5, max_cardinality=2, f_form="FULL", free_basis=True)
    bag = OrderedMotif(MotifConfig(family="ORDERLESS_BAG", **shared), None)
    ordered = OrderedMotif(MotifConfig(family="DIRECT_HORIZON_UPPER_BOUND", **shared), None)
    return bag, ordered


def test_free_orderless_bag_and_free_ordered_differ_only_by_the_transition():
    """The capacity-matched control v0.2 never had: same architecture, no operator."""
    bag, ordered = bag_and_ordered()
    assert not hasattr(bag, "f_raw")
    assert hasattr(ordered, "f_raw")
    bag_parameters = sum(p.numel() for p in bag.parameters())
    ordered_parameters = sum(p.numel() for p in ordered.parameters())
    assert ordered_parameters - bag_parameters == ordered.f_raw.numel()


def test_free_orderless_bag_is_blind_to_the_prefix_order(teacher_batch):
    """A leak here would let the control read the very thing it must not read."""
    from src.topic5_strict_history_motif_v0_2 import perturb_prefix_order

    bag, _ = bag_and_ordered(n_contacts=teacher_batch.n_contacts)
    reference = bag.prefix_state(teacher_batch).detach().clone()
    shuffled = perturb_prefix_order(teacher_batch)
    assert torch.equal(bag.prefix_state(shuffled), reference)


def test_free_ordered_arm_is_not_blind_to_the_prefix_order(teacher_batch):
    from src.topic5_strict_history_motif_v0_2 import perturb_prefix_order

    _, ordered = bag_and_ordered(n_contacts=teacher_batch.n_contacts)
    reference = ordered.prefix_state(teacher_batch)
    shuffled = ordered.prefix_state(perturb_prefix_order(teacher_batch))
    assert not torch.allclose(reference, shuffled)

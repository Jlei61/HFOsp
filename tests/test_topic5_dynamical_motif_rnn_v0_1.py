"""Contract tests for the Topic 5.2 dynamical motif RNN v0.1-r2.

These cover the five engineering stop conditions in the spec: provenance,
leakage, nested zero-equivalence, decoder/replay behaviour and numerical
health.  A failure here blocks formal training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    ALL_MODELS,
    CapacityMatchedStaticReadout,
    GATE_RULES,
    MAIN_MODELS,
    MotifConfig,
    MotifRNN,
    NEW_PARAMETERS,
    StaticReadout,
    axis_shuffled_permutation,
    build_motif_event_tensors,
    causal_displacement,
    capacity_matched_static_rank,
    freeze_direction_scale,
    rollout_displacement_update,
    trainable_parameter_count,
)
from scripts.train_topic5_dynamical_motif_unit_v0_1 import selection_score  # noqa: E402
from src.topic5_dynamical_motif_rollout_v0_1 import (  # noqa: E402
    DecoderContract,
    SizeHead,
    energy_score,
    stochastic_rollout,
    summarise_sequences,
)
from src.topic5_shared_propagation_field import (  # noqa: E402
    conditional_k_subset_log_prob,
    sample_conditional_k_subset,
)

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"
PARENT_CACHE = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5/cache"


# ---------------------------------------------------------------- fixtures
def toy_config(model_id: str, **overrides) -> MotifConfig:
    rng = np.random.default_rng(7)
    n_nodes, n_contacts = 24, 5
    grid = np.stack(np.meshgrid(np.arange(6), np.arange(4), indexing="ij"), -1).reshape(-1, 2)
    node_xy = (grid.astype(float) * 3.0) + rng.normal(0, 0.15, size=(n_nodes, 2))
    distance = np.linalg.norm(node_xy[:, None, :] - node_xy[None, :, :], axis=-1)
    mask = ((distance > 0) & (distance <= 4.5)).astype(np.uint8)
    contact_index = np.linspace(0, n_nodes - 1, n_contacts).astype(int)
    H = np.zeros((n_contacts, n_nodes), dtype=np.float32)
    for row, node in enumerate(contact_index):
        weight = np.exp(-(distance[node] ** 2) / (2 * 2.0 ** 2))
        weight[weight < 0.05] = 0.0
        H[row] = weight / max(weight.sum(), 1e-9)
    payload = dict(
        model_id=model_id, n_contacts=n_contacts, n_nodes=n_nodes,
        observation_operator=H, node_xy_mm=node_xy, local_mask=mask,
        r_forward_mm=4.5, sigma_s_mm=2.0, seed=0, theta_init=0.4,
    )
    payload.update(overrides)
    return MotifConfig(**payload)


def toy_ranks(n_events: int = 64, n_contacts: int = 5, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ranks = np.full((n_events, n_contacts), -1, dtype=np.int16)
    for event in range(n_events):
        length = int(rng.integers(2, n_contacts + 1))
        order = rng.permutation(n_contacts)[:length]
        for rank, contact in enumerate(order):
            ranks[event, contact] = rank
    return ranks


def toy_contacts_xy(n_contacts: int = 5) -> np.ndarray:
    return np.stack([np.arange(n_contacts) * 3.0, np.zeros(n_contacts)], axis=1).astype(np.float32)


def test_capacity_matched_static_readout_respects_dm0_budget():
    config = toy_config("DM0_ISOTROPIC")
    dm0 = MotifRNN(config)
    covariates = np.column_stack([
        toy_contacts_xy(config.n_contacts), np.linspace(0.1, 0.9, config.n_contacts)])
    rank = capacity_matched_static_rank(
        config.n_contacts, covariates, trainable_parameter_count(dm0))
    static = CapacityMatchedStaticReadout(config.n_contacts, covariates, rank)
    assert trainable_parameter_count(static) <= trainable_parameter_count(dm0)
    if rank < config.n_contacts:
        larger = CapacityMatchedStaticReadout(config.n_contacts, covariates, rank + 1)
        assert trainable_parameter_count(larger) > trainable_parameter_count(dm0)
    x, recruited, displacement = _dummy_batch(config)
    logits, stops, gate = static(x, recruited, displacement)
    assert logits.shape == x.shape
    assert stops.shape == x.shape[:2]
    assert torch.all(gate == 0)


def test_contact_only_checkpoint_score_does_not_read_stop_loss():
    left = {"contact_nll": 1.0, "next_bce": 0.2, "stop_bce": 50.0}
    right = {"contact_nll": 1.1, "next_bce": 0.1, "stop_bce": 0.0}
    assert selection_score(left, "contact_nll", 1.0) < selection_score(
        right, "contact_nll", 1.0)
    assert selection_score(left, "joint", 1.0) > selection_score(right, "joint", 1.0)


# ------------------------------------------------------- frame provenance
@pytest.mark.skipif(not GEOMETRY_ROOT.exists(), reason="geometry frame cache not built")
def test_geometry_frame_reuses_parent_events_and_split():
    for directory in sorted(GEOMETRY_ROOT.iterdir()):
        provenance = json.loads((directory / "provenance.json").read_text())
        assert provenance["target_values_read"] is False
        events = np.load(directory / "events.npz", allow_pickle=True)
        parent = np.load(PARENT_CACHE / provenance["parent_views"][0] / "events_raw.npz",
                         allow_pickle=True)
        assert np.array_equal(events["ranks"], parent["ranks"])
        assert [str(v) for v in events["contact_names"]] == [str(v) for v in parent["contact_names"]]
        audit = provenance["split_audit"]
        assert audit["model_unseen_equals_parent_heldout"] is True
        assert audit["model_unseen_disjoint_from_train"] is True
        plane = np.load(directory / "plane.npz", allow_pickle=False)
        coords = np.asarray(plane["coords_3d_mm"], dtype=float)
        u, w = np.asarray(plane["basis_u"]), np.asarray(plane["basis_w"])
        origin = np.asarray(plane["basis_origin"])
        expected = (coords - origin) @ np.stack([u, w], axis=1)
        assert np.allclose(expected, np.asarray(plane["contacts_xy_mm"]), atol=1e-3)
        assert abs(float(u @ w)) < 1e-9
        assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-9


@pytest.mark.skipif(not GEOMETRY_ROOT.exists(), reason="geometry frame cache not built")
def test_geometry_frame_sign_rule_is_deterministic_and_target_free():
    for directory in sorted(GEOMETRY_ROOT.iterdir()):
        plane = np.load(directory / "plane.npz", allow_pickle=False)
        for key in ("basis_u", "basis_w"):
            vector = np.asarray(plane[key], dtype=float)
            assert vector[int(np.argmax(np.abs(vector)))] > 0


# ------------------------------------------------- nested zero-equivalence
def _dummy_batch(config: MotifConfig, steps: int = 4, batch: int = 6, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = np.zeros((batch, steps, config.n_contacts), np.float32)
    for b in range(batch):
        for t in range(steps):
            x[b, t, rng.integers(config.n_contacts)] = 1.0
    recruited = np.cumsum(x, axis=1).clip(0, 1)
    displacement = rng.normal(0, 3.0, size=(batch, steps, 2)).astype(np.float32)
    displacement[:, 0] = 0.0
    return (torch.from_numpy(x), torch.from_numpy(recruited.astype(np.float32)),
            torch.from_numpy(displacement))


@pytest.mark.parametrize("lower,upper", [
    ("DM0_ISOTROPIC", "DM1_FREE_AXIS"),
    ("DM1_FREE_AXIS", "DM2_LOCAL_DIRECTIONAL"),
    ("DM2_LOCAL_DIRECTIONAL", "DM3_AXIS_FEEDFORWARD_TRANSIENT"),
    ("DM2_LOCAL_DIRECTIONAL", "DM3_GAIN_MEMORY"),
    ("DM2_LOCAL_DIRECTIONAL", "DM3_SYMMETRIC_MATCHED"),
    ("DM2_LOCAL_DIRECTIONAL", "DM3_AXIS_SHUFFLED_TRIANGULAR"),
])
def test_zero_component_is_bit_exact(lower, upper):
    low = MotifRNN(toy_config(lower))
    high = MotifRNN(toy_config(upper))
    copied = high.load_warm_start(dict(low.state_dict()))
    assert copied, "warm start copied nothing"
    for name in NEW_PARAMETERS[upper]:
        value = getattr(high, name)
        if name in ("theta",):
            continue
        assert float(value) == 0.0, f"{upper}.{name} must start at exactly zero"
    x, recruited, displacement = _dummy_batch(low.config)
    with torch.no_grad():
        a = low(x, recruited, displacement)
        b = high(x, recruited, displacement)
    for left, right in zip(a, b):
        assert torch.equal(left, right), f"{upper} at zero component is not bit-identical to {lower}"


def test_m1_theta_is_irrelevant_when_eta_is_zero():
    base = MotifRNN(toy_config("DM1_FREE_AXIS", theta_init=0.0))
    other = MotifRNN(toy_config("DM1_FREE_AXIS", theta_init=1.1))
    other.load_warm_start({k: v for k, v in base.state_dict().items() if k != "theta"})
    x, recruited, displacement = _dummy_batch(base.config)
    with torch.no_grad():
        assert torch.allclose(base(x, recruited, displacement)[0],
                              other(x, recruited, displacement)[0], atol=1e-6)


# ------------------------------------------------------- operator contract
@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_operator_stays_on_frozen_local_support_and_non_negative(model_id):
    model = MotifRNN(toy_config(model_id))
    with torch.no_grad():
        model.log_g.fill_(0.5)
        if "beta" in dict(model.named_parameters()):
            model.beta.fill_(0.8)
        if "eta_raw" in dict(model.named_parameters()):
            model.eta_raw.fill_(0.4)
        if "gamma_raw" in dict(model.named_parameters()):
            model.gamma_raw.fill_(0.3)
        support = model.mask > 0
        for s in (-1.0, -0.3, 0.0, 0.3, 1.0):
            matrix = model.recurrent_matrix(s)
            assert torch.all(matrix >= 0), f"{model_id} produced a negative weight at s={s}"
            assert torch.all(matrix[~support] == 0), f"{model_id} left the frozen support"
            assert torch.all(torch.diagonal(matrix) == 0)


def test_column_normalisation_makes_outgoing_weight_sum_to_the_gain():
    model = MotifRNN(toy_config("DM2_LOCAL_DIRECTIONAL"))
    with torch.no_grad():
        model.beta.fill_(1.3)
        model.log_g.fill_(math.log(2.0))
        for s in (-1.0, 0.0, 0.7):
            matrix = model.recurrent_matrix(s)
            assert torch.allclose(matrix.sum(dim=0), torch.full((model.n_nodes,), 2.0), atol=1e-4)


def test_recurrent_drive_matches_the_dense_matrix():
    model = MotifRNN(toy_config("DM3_AXIS_FEEDFORWARD_TRANSIENT"))
    with torch.no_grad():
        model.beta.fill_(0.9)
        model.eta_raw.fill_(0.5)
        model.gamma_raw.fill_(0.4)
        model.log_g.fill_(0.2)
        terms = model.recurrent_terms()
        h = torch.randn(3, model.n_nodes)
        s = torch.tensor([-0.6, 0.0, 0.8])
        drive = model.recurrent_drive(h, s, terms)
        for row in range(3):
            dense = model.recurrent_matrix(float(s[row]))
            assert torch.allclose(drive[row], dense @ h[row], atol=1e-4)


def test_feedforward_is_strictly_triangular_and_transposed():
    model = MotifRNN(toy_config("DM3_AXIS_FEEDFORWARD_TRANSIENT"))
    with torch.no_grad():
        f_plus, f_minus = model.feedforward_pair()
        q = model.axial_position()
        order = torch.argsort(q)
        # Row = receiving node, column = source node, so activity may only move
        # from a lower axial position to a higher one: strictly lower triangular
        # once nodes are sorted along the axis.
        permuted = f_plus[order][:, order]
        assert torch.all(torch.triu(permuted) == 0), "F+ is not a strict forward cascade"
        assert torch.all(f_plus.sum() > 0)
        assert torch.equal(f_minus, f_plus.transpose(0, 1))
        assert torch.all(f_plus[model.mask == 0] == 0)


def test_symmetric_control_is_symmetric_on_the_same_pairs():
    model = MotifRNN(toy_config("DM3_SYMMETRIC_MATCHED"))
    reference = MotifRNN(toy_config("DM3_AXIS_FEEDFORWARD_TRANSIENT"))
    with torch.no_grad():
        f_plus, f_minus = model.feedforward_pair()
        symmetric = 0.5 * (f_plus + f_minus)
        assert torch.allclose(symmetric, symmetric.transpose(0, 1))
        left, right = reference.feedforward_pair()
        assert torch.equal((f_plus + f_minus) > 0, (left + right) > 0), "pair support differs"
        assert abs(float(symmetric.sum()) - float(0.5 * (left + right).sum())) < 1e-5


def test_axis_shuffled_control_keeps_triangularity_and_matches_non_zero_count():
    model = MotifRNN(toy_config("DM3_AXIS_SHUFFLED_TRIANGULAR", shuffle_seed=17))
    reference = MotifRNN(toy_config("DM3_AXIS_FEEDFORWARD_TRANSIENT"))
    report = model.calibrate_shuffle_radius()
    assert report["calibrated"] is True
    with torch.no_grad():
        f_plus, f_minus = model.feedforward_pair()
        left, _ = reference.feedforward_pair()
        assert abs(int((f_plus > 0).sum()) - int((left > 0).sum())) <= 1
        q = model.axial_position()[model.shuffle_permutation]
        order = torch.argsort(q)
        assert torch.all(torch.triu(f_plus[order][:, order]) == 0)
        assert torch.equal(f_minus, f_plus.transpose(0, 1))
        assert torch.all(f_plus[model.mask == 0] == 0)
    permutation = axis_shuffled_permutation(
        model.config.node_xy_mm, model.config.local_mask, 17)
    assert sorted(permutation.tolist()) == list(range(model.n_nodes))
    assert not np.array_equal(permutation, np.arange(model.n_nodes))


def test_gain_memory_control_moves_gain_up_and_leak_down():
    model = MotifRNN(toy_config("DM3_GAIN_MEMORY"))
    with torch.no_grad():
        base = model.recurrent_terms()
        model.delta_g.fill_(0.7)
        model.delta_kappa.fill_(0.9)
        moved = model.recurrent_terms()
    assert float(moved["gain"]) > float(base["gain"])
    assert float(moved["kappa"]) < float(base["kappa"])


def test_non_negative_projection_clamps_motif_parameters():
    model = MotifRNN(toy_config("DM3_AXIS_FEEDFORWARD_TRANSIENT"))
    with torch.no_grad():
        model.gamma_raw.fill_(-2.0)
        model.eta_raw.fill_(-1.0)
    model.project_constraints()
    assert float(model.gamma_raw) == 0.0
    assert float(model.eta_raw) == 0.0


# --------------------------------------------------------- direction gate
def test_direction_gate_is_zero_at_the_first_rank_and_reads_only_the_prefix():
    ranks = toy_ranks()
    xy = toy_contacts_xy()
    for rule in GATE_RULES:
        tensors = build_motif_event_tensors(ranks, xy, gate_rule=rule)
        displacement = tensors["displacement"].numpy()
        assert np.all(displacement[:, 0] == 0.0), f"{rule} leaks direction into the first rank"
        # Rewriting the tail must not change any earlier displacement.
        edited = ranks.copy()
        lengths = np.array([int(r[r >= 0].max()) + 1 for r in ranks])
        changed = 0
        for event in range(len(edited)):
            if lengths[event] < 4:
                continue
            last = np.flatnonzero(edited[event] == lengths[event] - 1)
            other = np.flatnonzero(edited[event] == -1)
            if last.size and other.size:
                edited[event, other[0]] = lengths[event] - 1
                edited[event, last[0]] = -1
                changed += 1
        assert changed > 0
        edited_tensors = build_motif_event_tensors(edited, xy, gate_rule=rule)
        limit = int(lengths.min())
        assert np.allclose(displacement[:, :max(limit - 1, 1)],
                           edited_tensors["displacement"].numpy()[:, :max(limit - 1, 1)])


def test_gate_rules_freeze_where_the_contract_says():
    # Deliberately non-uniform spacing: with an evenly advancing sequence every
    # rule gives the same constant mean displacement and the test is vacuous.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [9.0, 0.0], [10.0, 0.0], [30.0, 0.0], [31.0, 0.0]],
                  dtype=np.float32)
    ranks = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int16)
    frozen = {"M2-2RANK": 1, "M2-3RANK": 2}
    for rule in GATE_RULES:
        tensors = build_motif_event_tensors(ranks, xy, gate_rule=rule)
        d = tensors["displacement"].numpy()[0]
        if rule in frozen:
            k = frozen[rule]
            for t in range(k, d.shape[0]):
                assert np.allclose(d[t], d[k]), f"{rule} did not freeze at rank {k}"
            assert not np.allclose(d[k], d[max(k - 1, 0)]) or k == 0
        else:
            assert not np.allclose(d[1], d[3]), "online gate never updated"
            assert not np.allclose(d[3], d[5]), "online gate stopped updating"


def test_rollout_gate_matches_teacher_forced_gate_on_a_real_sequence():
    xy = torch.as_tensor(toy_contacts_xy(6))
    ranks = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int16)
    for rule in GATE_RULES:
        tensors = build_motif_event_tensors(ranks, xy.numpy(), gate_rule=rule)
        reference = tensors["displacement"].numpy()[0]
        centroid_start = xy[0][None, :]
        displacement = torch.zeros(1, 2)
        produced = [displacement.clone()]
        for step in range(1, reference.shape[0]):
            displacement = rollout_displacement_update(
                displacement, centroid_start, xy[step][None, :], step, rule)
            produced.append(displacement.clone())
        produced = torch.cat(produced).numpy()
        assert np.allclose(produced, reference, atol=1e-5), f"{rule} rollout gate drifted"


def test_freeze_direction_scale_uses_only_the_first_two_ranks():
    ranks = toy_ranks()
    xy = toy_contacts_xy()
    indices = np.arange(len(ranks))
    scale = freeze_direction_scale(ranks, xy, indices)
    edited = ranks.copy()
    edited[edited >= 2] = -1
    kept = np.flatnonzero((edited == 1).any(axis=1))
    assert abs(freeze_direction_scale(edited, xy, kept) - scale) < 1e-9
    # The default scale must not depend on any axis choice.
    assert scale > 0
    projected = freeze_direction_scale(ranks, xy, indices, axis_u=np.array([1.0, 0.0]))
    assert projected <= scale + 1e-9


# ----------------------------------------------------------------- decoder
def test_exact_subset_sampler_respects_cardinality_and_never_repeats():
    torch.manual_seed(0)
    logits = torch.randn(64, 7)
    available = torch.ones(64, 7, dtype=torch.bool)
    available[:, 0] = False
    k = torch.randint(0, 4, (64,))
    generator = torch.Generator().manual_seed(11)
    picked = sample_conditional_k_subset(logits, available, k, generator=generator)
    assert torch.equal(picked.sum(1), k)
    assert not bool((picked & ~available).any())


def test_exact_subset_sampler_matches_its_own_log_probability():
    torch.manual_seed(1)
    logits = torch.randn(1, 5) * 1.5
    available = torch.ones(1, 5, dtype=torch.bool)
    k = torch.tensor([2])
    generator = torch.Generator().manual_seed(5)
    draws = sample_conditional_k_subset(
        logits.expand(20000, -1).contiguous(), available.expand(20000, -1).contiguous(),
        k.expand(20000), generator=generator)
    keys: dict[tuple, int] = {}
    for row in draws.numpy():
        members = tuple(np.flatnonzero(row).tolist())
        keys[members] = keys.get(members, 0) + 1
    assert len(keys) == 10, "not every 2-subset of five contacts was drawn"
    for members, count in keys.items():
        target = torch.zeros(1, 5, dtype=torch.bool)
        target[0, list(members)] = True
        expected = float(torch.exp(conditional_k_subset_log_prob(logits, target, available)))
        assert abs(count / 20000 - expected) < 0.02, f"subset {members} off: {count/20000} vs {expected}"


def test_rollout_absorbs_stop_and_never_repeats_a_contact():
    config = toy_config("DM2_LOCAL_DIRECTIONAL")
    model = MotifRNN(config)
    with torch.no_grad():
        model.stop_head[-1].bias.fill_(-4.0)
    head = SizeHead(config.n_contacts)
    contract = DecoderContract(1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
    starts = torch.zeros(8, config.n_contacts)
    starts[:, 0] = 1.0
    out = stochastic_rollout(model, head, contract, starts, toy_contacts_xy(config.n_contacts),
                             torch.device("cpu"), mode="FULL_STOP", rng_label="unit")
    sequence = out["sequence"]
    for row, emitted in zip(sequence, out["n_emitted"]):
        counts = row[: emitted + 1].sum(axis=0)
        assert counts.max() <= 1, "a contact was recruited twice"
        assert row[emitted + 1:].sum() == 0, "STOP was not absorbing"


def _single_contact_size_head(n_contacts: int) -> SizeHead:
    head = SizeHead(n_contacts)
    with torch.no_grad():
        head.network[-1].weight.zero_()
        head.network[-1].bias.fill_(-20.0)
        head.network[-1].bias[0] = 20.0
    return head


def test_fixed_horizon_rollout_ignores_stop():
    config = toy_config("DM1_FREE_AXIS")
    model = MotifRNN(config)
    with torch.no_grad():
        model.stop_head[-1].bias.fill_(8.0)      # STOP would fire immediately
    head = _single_contact_size_head(config.n_contacts)
    contract = DecoderContract(1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
    starts = torch.zeros(4, config.n_contacts)
    starts[:, 0] = 1.0
    out = stochastic_rollout(model, head, contract, starts, toy_contacts_xy(config.n_contacts),
                             torch.device("cpu"), mode="FIXED_H", horizon=3, rng_label="unit")
    assert np.all(out["n_emitted"] == 3)
    stopped = stochastic_rollout(model, head, contract, starts,
                                 toy_contacts_xy(config.n_contacts), torch.device("cpu"),
                                 mode="FULL_STOP", rng_label="unit")
    assert np.all(stopped["n_emitted"] == 0), "STOP head was ignored in FULL_STOP mode"


def test_common_random_numbers_are_shared_across_models():
    starts = torch.zeros(6, 5)
    starts[:, 0] = 1.0
    head = SizeHead(5)
    contract = DecoderContract(1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
    outputs = []
    for model_id in ("DM1_FREE_AXIS", "DM2_LOCAL_DIRECTIONAL"):
        model = MotifRNN(toy_config(model_id))
        with torch.no_grad():
            model.contact_bias.zero_()
            model.readout_gain.fill_(0.0)        # identical logits by construction
            model.stop_head[-1].bias.fill_(-2.0)
        outputs.append(stochastic_rollout(model, head, contract, starts, toy_contacts_xy(5),
                                          torch.device("cpu"), rng_label="crn"))
    assert np.array_equal(outputs[0]["sequence"], outputs[1]["sequence"])


def test_rollout_is_reproducible_from_the_same_label():
    config = toy_config("DM3_AXIS_FEEDFORWARD_TRANSIENT")
    model = MotifRNN(config)
    head = SizeHead(config.n_contacts)
    contract = DecoderContract(1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
    starts = torch.zeros(5, config.n_contacts)
    starts[:, 1] = 1.0
    kwargs = dict(contacts_xy_mm=toy_contacts_xy(config.n_contacts), device=torch.device("cpu"),
                  rng_label="replay")
    first = stochastic_rollout(model, head, contract, starts, **kwargs)
    second = stochastic_rollout(model, head, contract, starts, **kwargs)
    assert np.array_equal(first["sequence"], second["sequence"])


# ---------------------------------------------------------------- scoring
def test_summary_and_energy_score_behave():
    sequence = np.zeros((2, 4, 5), dtype=np.uint8)
    sequence[0, 0, 0] = 1
    sequence[0, 1, 2] = 1
    sequence[0, 2, 4] = 1
    sequence[1, 0, 4] = 1
    sequence[1, 1, 3] = 1
    summary = summarise_sequences(sequence, np.array([2, 1]), toy_contacts_xy(5),
                                  np.array([1.0, 0.0]))
    assert summary["n_rank"].tolist() == [3, 2]
    assert summary["n_contact"].tolist() == [3, 2]
    assert summary["l_axis"][0] == pytest.approx(12.0)
    assert summary["r_last"][0][0] == pytest.approx(12.0)
    perfect = energy_score(np.array([[1.0, 1.0]] * 8), np.array([1.0, 1.0]))
    assert perfect == pytest.approx(0.0, abs=1e-9)
    assert energy_score(np.random.default_rng(0).normal(size=(64, 2)),
                        np.array([5.0, 5.0])) > perfect


def test_static_readout_has_no_recurrence_and_no_direction_input():
    baseline = StaticReadout(5, np.zeros((5, 3), dtype=np.float32))
    x, recruited, displacement = _dummy_batch(toy_config("DM0_ISOTROPIC"))
    with torch.no_grad():
        first = baseline(x, recruited, displacement)[0]
        second = baseline(x, recruited, displacement * 0 + 9.0)[0]
    assert torch.equal(first, second)


def _executable_identifiers(path: Path) -> set[str]:
    """Every name the module actually executes, ignoring prose and docstrings."""
    import ast
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.name)
            names.add(node.asname or "")
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


@pytest.mark.parametrize("module", [
    "src/topic5_dynamical_motif_rnn_v0_1.py",
    "src/topic5_dynamical_motif_rollout_v0_1.py",
])
def test_model_code_carries_no_template_or_seizure_dependency(module):
    import inspect
    signature = inspect.signature(MotifRNN.forward)
    assert list(signature.parameters) == ["self", "x", "recruited", "displacement"]
    names = _executable_identifiers(ROOT / module)
    forbidden = ("earliness", "seizure", "ictal", "bb150", "template_field",
                 "prefix_posterior", "full_train_mode", "prefix_mode", "suffix")
    for token in forbidden:
        hits = [name for name in names if token in str(name).lower()]
        assert not hits, f"{module} executes {token}-derived names: {hits}"


@pytest.mark.parametrize("model_id", MAIN_MODELS)
def test_forward_is_finite_and_gradients_flow(model_id):
    model = MotifRNN(toy_config(model_id))
    x, recruited, displacement = _dummy_batch(model.config)
    logits, stops, gate = model(x, recruited, displacement)
    assert torch.isfinite(logits).all() and torch.isfinite(stops).all() and torch.isfinite(gate).all()
    (logits.square().mean() + stops.square().mean()).backward()
    for name in NEW_PARAMETERS[model_id]:
        parameter = getattr(model, name)
        assert parameter.grad is not None, f"{model_id}.{name} received no gradient"
        assert torch.isfinite(parameter.grad).all()


def test_causal_displacement_handles_short_events():
    ranks = np.array([[0, -1, -1], [0, 1, -1]], dtype=np.int16)
    xy = toy_contacts_xy(3)
    tensors = build_motif_event_tensors(ranks, xy, gate_rule="M2-ONLINE")
    displacement = tensors["displacement"].numpy()
    assert np.all(displacement[0] == 0.0)
    assert np.allclose(displacement[1, 1], np.array([3.0, 0.0]))

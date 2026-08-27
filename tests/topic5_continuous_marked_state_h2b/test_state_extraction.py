from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.topic5_continuous_marked_state_h2b import contract
from src.topic5_continuous_marked_state_h2b.state_extraction import (
    InferenceRawAnchorReader,
    assert_anchor_outputs_bitwise_equal,
    atomic_state_cache,
    build_inference_anchor_inputs,
    build_wrong_time_candidates,
    exact_deterministic_history,
    extract_causal_state_features,
    freeze_and_assert,
    load_frozen_design,
    load_frozen_explicit_scaler,
    load_frozen_r16_checkpoint,
    materialize_inference_observation_embeddings,
)
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.state import ControlledPersistentState


E384_R16_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/"
    "optimizer_identifiability_r1_6/confirmation/prefix_high_lr_e12_c128/"
    "nested_extended_budget/epilepsiae_384"
)
E384_STABLE_HASHES = {
    1: "4113acf91e71736a1f5e9ea64c78389f3c0869877aa9c9f3bec71794620cecb6",
    3: "45eb3b8fe9ac81fcd31d9aa648923da26c6be58e00b78b3f779aa71e2c5b0069",
    4: "9eb27ed9a3d563a7911e053b90440a18f61786dc0844d371618a79a5aa05fdc8",
}
SOURCE_REPO = Path("/home/honglab/leijiaxin/HFOsp")
E384_COVERAGE = SOURCE_REPO / (
    "results/epi_prssm/continuous_marked_state/r1/r1_2/coverage/"
    "epilepsiae_384.npz"
)
E384_PREICTAL_QUERY = 1107877528.382813


class ToyFrozenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = ControlledPersistentState(observation_dim=2, state_dim=2)
        with torch.no_grad():
            self.state.generator.omega_raw.zero_()
            self.state.generator.q_raw.fill_(-8.0)
            self.state.generator.mu.zero_()
            self.state.correction.candidate.weight.zero_()
            self.state.correction.candidate.weight[:, :2].copy_(torch.eye(2))
            self.state.correction.candidate.bias.zero_()
            self.state.correction.gate.weight.zero_()
            self.state.correction.gate.bias.fill_(2.0)


@pytest.mark.parametrize("seed", [1, 3, 4])
def test_real_e384_stable_checkpoint_reconstructs_frozen_on_cpu(seed):
    checkpoint = E384_R16_ROOT / f"seed_{seed}/model.pt"
    if not checkpoint.exists():
        pytest.skip("canonical ignored R1.6 artifact is not mounted")
    model, provenance = load_frozen_r16_checkpoint(
        checkpoint, expected_sha256=E384_STABLE_HASHES[seed],
        expected_subject="epilepsiae_384", expected_seed=seed,
        device="cpu", require_stable_result=True,
    )
    assert model.state.dim == 8
    assert model.state_contact.out_features == 9
    assert provenance["state_frozen"] is True
    assert provenance["seizure_gradient_path"] is False
    assert all(not value.requires_grad for value in model.parameters())


@pytest.fixture(scope="module")
def real_e384_inference_support():
    if not E384_R16_ROOT.exists() or not E384_COVERAGE.exists():
        pytest.skip("canonical E384 R1.6/raw/coverage artifacts are not mounted")
    design, _, _ = load_frozen_design(SOURCE_REPO, "epilepsiae_384")
    coverage = CoverageTable.load(E384_COVERAGE)
    reader = InferenceRawAnchorReader(
        "epilepsiae_384", design.event_time, source_repo_root=SOURCE_REPO
    )
    return design, coverage, reader


def test_real_e384_inference_inventory_bypasses_only_training_guard(
        real_e384_inference_support):
    _, coverage, reader = real_e384_inference_support
    anchor, segment, continuity, source_minute, guard = (
        reader.inference_anchor_inventory(coverage)
    )
    assert int(reader.inference_usable.sum()) == 2509
    assert int((reader.inference_usable & ~reader.training_guard_free).sum()) == 606
    assert len(anchor) == 3871
    assert anchor.dtype == np.float64
    assert segment.dtype == np.int64
    assert reader.inference_min_valid_contact_fraction == pytest.approx(0.70)
    assert reader.training_min_valid_contact_fraction == pytest.approx(0.70)
    np.testing.assert_array_equal(continuity, coverage.session[segment])
    assert np.all(source_minute >= 0)
    assert int((~guard).sum()) == 560


def test_real_e384_same_session_coverage_rows_reset_independently(
        real_e384_inference_support):
    _, coverage, reader = real_e384_inference_support
    anchor, segment, continuity, _, _ = reader.inference_anchor_inventory(coverage)
    # E384 rows 0 and 1 share continuity-session 0 but are separated by a
    # genuine excluded gap.  State grouping must retain their unique row IDs.
    rows = np.asarray([0, 1], dtype=np.int64)
    selected = np.asarray([
        np.flatnonzero(segment == row)[0] for row in rows
    ])
    assert continuity[selected].tolist() == [0, 0]
    assert segment[selected].tolist() == [0, 1]
    model = freeze_and_assert(ToyFrozenModel())
    result = extract_causal_state_features(
        model,
        observation_time_epoch=anchor[selected],
        observation_coverage_segment_index=segment[selected],
        observation_embedding=np.asarray([[0.8, -0.2], [0.8, -0.2]], np.float32),
        explicit_observation=np.ones((2, 1, 4), dtype=np.float32),
        contact_mask=np.ones((2, 1), dtype=bool),
        anchor_time_epoch=anchor[selected],
        anchor_coverage_segment_index=segment[selected],
        deterministic_history=np.zeros((2, 3), dtype=np.float32),
        segment_start={int(row): float(coverage.start[row]) for row in rows},
        max_current_observation_age_seconds=30.0,
    )
    np.testing.assert_array_equal(result.persistent_state[0], result.persistent_state[1])
    assert result.causal_observation_count.tolist() == [1, 1]
    assert result.gap_reset.tolist() == [True, True]


def test_real_e384_inference_only_preictal_state_is_fresh_and_causal(
        real_e384_inference_support):
    checkpoint = E384_R16_ROOT / "seed_1/model.pt"
    if not checkpoint.exists():
        pytest.skip("canonical ignored R1.6 artifact is not mounted")
    model, _ = load_frozen_r16_checkpoint(
        checkpoint, expected_sha256=E384_STABLE_HASHES[1],
        expected_subject="epilepsiae_384", expected_seed=1, device="cpu",
    )
    design, coverage, reader = real_e384_inference_support
    query_time = np.asarray([E384_PREICTAL_QUERY], dtype=np.float64)
    query_segment = np.flatnonzero(
        (query_time[0] >= coverage.start) & (query_time[0] < coverage.stop)
    )
    assert query_segment.tolist() == [6]
    explicit_mean, explicit_scale, scaler_provenance = load_frozen_explicit_scaler(
        SOURCE_REPO, "epilepsiae_384"
    )
    inputs = build_inference_anchor_inputs(
        reader, coverage, explicit_mean=explicit_mean,
        explicit_scale=explicit_scale, allowed_segments=query_segment,
    )
    embedding = materialize_inference_observation_embeddings(
        model, inputs, device="cpu", batch_size=64
    )
    baseline_path = Path(
        "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/"
        "r1/r1_2/baselines/epilepsiae_384/seed_0/models.pt"
    )
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    history = exact_deterministic_history(
        design=design, subject="epilepsiae_384",
        history_scaler=baseline["history_scaler"],
        query_time_epoch=query_time,
        query_continuity_session=coverage.session[query_segment],
    )
    value = extract_causal_state_features(
        model,
        observation_time_epoch=inputs.anchor_time_epoch,
        observation_coverage_segment_index=inputs.coverage_segment_index,
        observation_embedding=embedding,
        explicit_observation=inputs.explicit,
        contact_mask=inputs.contact_mask,
        anchor_time_epoch=query_time,
        anchor_coverage_segment_index=query_segment,
        deterministic_history=history,
        segment_start={6: float(coverage.start[6])},
        max_current_observation_age_seconds=30.0,
    )
    assert embedding.shape == (len(inputs.anchor_time_epoch), 64)
    assert value.persistent_state.shape == (1, 8)
    assert value.anchor_time_epoch.dtype == np.float64
    assert np.isfinite(value.persistent_state).all()
    assert np.isfinite(value.memoryless_observation_code).all()
    assert value.observation_available.tolist() == [True]
    assert value.observation_age_seconds[0] == pytest.approx(10.382812976837158)
    current = np.flatnonzero(
        inputs.anchor_time_epoch == value.last_observation_time_epoch[0]
    )
    assert len(current) == 1
    assert inputs.training_guard_free[current[0]] is np.False_
    assert inputs.provenance["training_guard_free_used_for_inference"] is False
    assert inputs.provenance["training_minute_usable_used_for_inference"] is False
    assert inputs.provenance["state_reset_at_every_coverage_segment_row"] is True
    assert scaler_provenance["explicit_scaler_source"] == (
        "frozen_bridge_train_anchors_only"
    )

    # A real future observation exists in the same row.  Arbitrarily changing
    # every post-query embedding/window cannot change the query output.
    future = inputs.anchor_time_epoch > query_time[0]
    assert bool(future.any())
    changed_embedding = embedding.copy()
    changed_embedding[future] = 12345.0
    changed_explicit = inputs.explicit.copy()
    changed_explicit[future] = -12345.0
    perturbed = extract_causal_state_features(
        model,
        observation_time_epoch=inputs.anchor_time_epoch,
        observation_coverage_segment_index=inputs.coverage_segment_index,
        observation_embedding=changed_embedding,
        explicit_observation=changed_explicit,
        contact_mask=inputs.contact_mask,
        anchor_time_epoch=query_time,
        anchor_coverage_segment_index=query_segment,
        deterministic_history=history,
        segment_start={6: float(coverage.start[6])},
        max_current_observation_age_seconds=30.0,
    )
    assert_anchor_outputs_bitwise_equal(value, perturbed)


def _extract(model, embedding, explicit, *, query_time=(20.0,), query_segment=(0,)):
    return extract_causal_state_features(
        model,
        observation_time_epoch=np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        observation_coverage_segment_index=np.asarray([0, 0, 0], dtype=np.int64),
        observation_embedding=np.asarray(embedding, dtype=np.float32),
        explicit_observation=np.asarray(explicit, dtype=np.float32),
        contact_mask=np.ones((3, 1), dtype=bool),
        anchor_time_epoch=np.asarray(query_time, dtype=np.float64),
        anchor_coverage_segment_index=np.asarray(query_segment, dtype=np.int64),
        deterministic_history=np.zeros((len(query_time), 3), dtype=np.float32),
        segment_start={0: 0.0},
    )


def test_model_is_frozen_and_future_perturbation_is_bitwise_invariant():
    model = freeze_and_assert(ToyFrozenModel())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    embedding = np.asarray([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.7]])
    explicit = np.arange(12, dtype=np.float32).reshape(3, 1, 4)
    reference = _extract(model, embedding, explicit)
    changed_embedding = embedding.copy()
    changed_embedding[2] = [100.0, -100.0]
    changed_explicit = explicit.copy()
    changed_explicit[2] = -999.0
    perturbed = _extract(model, changed_embedding, changed_explicit)
    assert_anchor_outputs_bitwise_equal(reference, perturbed)
    assert reference.causal_observation_count.tolist() == [2]
    assert reference.last_observation_time_epoch.dtype == np.float64


def test_memoryless_uses_only_current_observation_while_persistent_carries():
    model = freeze_and_assert(ToyFrozenModel())
    explicit = np.ones((3, 1, 4), dtype=np.float32)
    first = _extract(
        model,
        [[0.9, 0.0], [0.1, 0.0], [0.0, 0.0]],
        explicit,
    )
    second = _extract(
        model,
        [[-0.9, 0.0], [0.1, 0.0], [0.0, 0.0]],
        explicit,
    )
    assert not np.array_equal(first.persistent_state, second.persistent_state)
    np.testing.assert_array_equal(
        first.memoryless_observation_code,
        second.memoryless_observation_code,
    )


def test_stale_window_cannot_masquerade_as_current_observation():
    model = freeze_and_assert(ToyFrozenModel())
    result = extract_causal_state_features(
        model,
        observation_time_epoch=np.asarray([10.0], dtype=np.float64),
        observation_coverage_segment_index=np.asarray([0], dtype=np.int64),
        observation_embedding=np.asarray([[0.8, -0.2]], dtype=np.float32),
        explicit_observation=np.ones((1, 1, 4), dtype=np.float32),
        contact_mask=np.ones((1, 1), dtype=bool),
        anchor_time_epoch=np.asarray([50.0], dtype=np.float64),
        anchor_coverage_segment_index=np.asarray([0], dtype=np.int64),
        deterministic_history=np.zeros((1, 3), dtype=np.float32),
        segment_start={0: 0.0},
        max_current_observation_age_seconds=30.0,
    )
    assert np.isfinite(result.persistent_state).all()
    assert np.isnan(result.memoryless_observation_code).all()
    assert np.isnan(result.current_explicit_observation).all()
    assert result.observation_age_seconds.tolist() == [40.0]
    assert result.observation_available.tolist() == [False]


def test_segment_change_resets_state_and_never_carries_across_gap():
    model = freeze_and_assert(ToyFrozenModel())
    result = extract_causal_state_features(
        model,
        observation_time_epoch=np.asarray([10.0, 1010.0], dtype=np.float64),
        observation_coverage_segment_index=np.asarray([0, 1], dtype=np.int64),
        observation_embedding=np.asarray([[0.8, -0.2], [0.8, -0.2]], np.float32),
        explicit_observation=np.ones((2, 1, 4), dtype=np.float32),
        contact_mask=np.ones((2, 1), dtype=bool),
        anchor_time_epoch=np.asarray([10.0, 1010.0], dtype=np.float64),
        anchor_coverage_segment_index=np.asarray([0, 1], dtype=np.int64),
        deterministic_history=np.zeros((2, 3), dtype=np.float32),
        segment_start={0: 0.0, 1: 1000.0},
    )
    np.testing.assert_array_equal(result.persistent_state[0], result.persistent_state[1])
    assert result.gap_reset.tolist() == [True, True]
    assert result.causal_observation_count.tolist() == [1, 1]


def test_wrong_time_candidates_are_same_segment_soft_ranked_and_excluded():
    value = build_wrong_time_candidates(
        target_time_epoch=np.asarray([100.0]),
        target_segment=np.asarray([2]),
        target_confounders=np.asarray([[0.0, 0.0]], np.float32),
        donor_time_epoch=np.asarray([0.0, 50.0, 200.0, 300.0, 110.0]),
        donor_segment=np.asarray([2, 2, 2, 3, 2]),
        donor_state=np.arange(10, dtype=np.float32).reshape(5, 2),
        donor_confounders=np.asarray([
            [0.1, 0.1], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]
        ], np.float32),
        n_donors=3, min_separation_seconds=20.0,
        global_exclusion_intervals=[(40.0, 60.0)],
        target_exclusion_start=np.asarray([190.0]),
        target_exclusion_stop=np.asarray([210.0]),
    )
    assert value.valid.sum() == 1
    assert value.donor_index[0, 0] == 0
    assert value.donor_time_epoch.dtype == np.float64


def test_atomic_cache_records_causal_and_dtype_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "RESULT_ROOT", tmp_path)
    model = freeze_and_assert(ToyFrozenModel())
    features = _extract(
        model,
        [[0.2, 0.0], [0.1, 0.0], [0.0, 0.0]],
        np.ones((3, 1, 4), dtype=np.float32),
    )
    target = tmp_path / "state_cache" / "fixture.npz"
    manifest = atomic_state_cache(
        target, features=features, query_id=np.asarray(["q0"]),
        provenance={
            "checkpoint_sha256": "a" * 64,
            "checkpoint_result_sha256": "b" * 64,
            "source_hashes": {"fixture": "c" * 64},
            "current_observation_max_age_seconds": 30.0,
        },
    )
    assert manifest["max_source_time_le_anchor"] is True
    assert manifest["all_parameters_frozen"] is True
    assert manifest["gap_reset"] is True
    assert manifest["time_dtype"] == "float64"
    assert json.loads(target.with_suffix(".manifest.json").read_text())["cache_sha256"]
    with np.load(target, allow_pickle=False) as cached:
        assert cached["anchor_time_epoch"].dtype == np.float64
        assert cached["max_source_time_epoch"][0] <= cached["anchor_time_epoch"][0]


def test_atomic_cache_rejects_source_after_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "RESULT_ROOT", tmp_path)
    model = freeze_and_assert(ToyFrozenModel())
    features = _extract(
        model,
        [[0.2, 0.0], [0.1, 0.0], [0.0, 0.0]],
        np.ones((3, 1, 4), dtype=np.float32),
    )
    object.__setattr__(
        features, "last_observation_time_epoch",
        np.asarray([features.anchor_time_epoch[0] + 1.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="after its anchor"):
        atomic_state_cache(
            tmp_path / "bad.npz", features=features,
            query_id=np.asarray(["q0"]),
            provenance={
                "checkpoint_sha256": "a" * 64,
                "checkpoint_result_sha256": "b" * 64,
                "source_hashes": {"fixture": "c" * 64},
                "current_observation_max_age_seconds": 30.0,
            },
        )

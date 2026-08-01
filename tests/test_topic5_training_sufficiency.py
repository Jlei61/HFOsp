"""Contract tests for the Topic 5 RNN training/objective sufficiency audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.train_topic5_interictal_rank_distribution import load_records
from src.topic5_rank_distribution import LinearStateSequenceRNN
from src.topic5_training_sufficiency import (
    Objective,
    aggregate_patient_metric,
    calibrate_offset_instrumented,
    development_records,
    evaluate_decomposed,
    objective_from_name,
    patient_first_summary,
    plan_cells,
    plateau_verdict,
    scheduled_forward,
    train_coverage_instrumented,
)

N_SUBJECTS = 34
N_CONTACTS = 5
N_FEATURES = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_subject(rng: np.random.Generator, n_events: int):
    """Singleton rank sets so every step feeds exactly one contact."""
    groups = np.full((n_events, N_CONTACTS), -1, dtype=np.int16)
    counts = np.zeros(n_events, dtype=np.int16)
    for event in range(n_events):
        length = int(rng.integers(2, N_CONTACTS + 1))
        order = rng.permutation(N_CONTACTS)[:length]
        groups[event, order] = np.arange(length, dtype=np.int16)
        counts[event] = length
    return groups, counts


def _write_dataset(root: Path, *, target_values_read: bool = False) -> Path:
    """A miniature but structurally faithful frozen rank dataset."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "per_subject").mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    audit_rows = []
    for index in range(N_SUBJECTS):
        dataset = "epilepsiae" if index < 18 else "yuquan"
        subject = f"{dataset}_{index:03d}"
        n_events = int(rng.integers(40, 90))
        groups, counts = _synthetic_subject(rng, n_events)
        split = np.zeros(n_events, dtype=np.uint8)
        split[int(round(n_events * 0.8)) :] = 1
        path = root / "per_subject" / f"{subject}.npz"
        np.savez_compressed(
            path,
            contact_features=rng.normal(size=(N_CONTACTS, N_FEATURES)).astype(
                np.float32
            ),
            contact_names=np.asarray(
                [f"A{position + 1}" for position in range(N_CONTACTS)]
            ),
            event_group_ids=groups,
            event_group_count=counts,
            event_split=split,
            event_source_index=np.arange(n_events, dtype=np.int64),
            contact_coords=rng.normal(size=(N_CONTACTS, 3)).astype(np.float32),
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "subject": subject,
                    "dataset_npz_sha256": _sha256(path),
                    "forbidden_inputs_present": {
                        "inter_event_interval": False,
                        "ictal_target": False,
                    },
                }
            )
        )
        audit_rows.append({"subject": subject, "dataset": dataset, "status": "ok"})
    pd.DataFrame(audit_rows).to_csv(root / "subject_audit.csv", index=False)
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "n_subjects_ok": N_SUBJECTS,
                "target_values_read": bool(target_values_read),
                "ab_or_kmeans_labels_read": False,
                "split_contract": "chronological first 80% calibration, last 20% held out",
            }
        )
    )
    return root


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory) -> Path:
    return _write_dataset(tmp_path_factory.mktemp("rank_dataset"))


@pytest.fixture(scope="module")
def records(dataset_root: Path):
    return load_records(dataset_root)


def _model(seed: int = 3):
    torch.manual_seed(int(seed))
    return LinearStateSequenceRNN(
        N_FEATURES,
        hidden_size=6,
        contact_embedding_dim=5,
        contact_encoder_hidden=5,
        local_offset_dim=2,
    )


def _train(records_list, *, batch_size=16, seed=17, cycles=1, updates=2, objective="teacher_forced_one_step", optimizer="adamw", weight_decay=1e-4):
    model = _model()
    snapshots, rows, coverage = train_coverage_instrumented(
        model,
        records_list,
        coverage_cycles=cycles,
        updates_per_patient=updates,
        batch_size=batch_size,
        learning_rate=1e-3,
        local_learning_rate=2e-3,
        weight_decay=weight_decay,
        gradient_clip=1.0,
        local_offset_dim=2,
        device=torch.device("cpu"),
        seed=seed,
        objective=objective_from_name(objective),
        optimizer_name=optimizer,
    )
    return snapshots, rows, coverage


# ---------------------------------------------------------------- test 1


def test_chunk_size_is_a_memory_chunk_not_an_update(records):
    """512 vs 1024 style chunking must not move the optimizer trajectory."""
    subset = [records[key] for key in sorted(records)[:4]]
    small, rows_small, _ = _train(subset, batch_size=8)
    large, rows_large, _ = _train(subset, batch_size=4096)

    assert max(row["n_backward_chunks"] for row in rows_small) > 1
    assert all(row["n_backward_chunks"] == 1 for row in rows_large)
    assert len(rows_small) == len(rows_large)

    left = small[max(small)]["model_state"]
    right = large[max(large)]["model_state"]
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=1e-5, atol=1e-6)
    for subject in small[max(small)]["offsets"]:
        torch.testing.assert_close(
            small[max(small)]["offsets"][subject],
            large[max(large)]["offsets"][subject],
            rtol=1e-5,
            atol=1e-6,
        )


def test_chunked_gradients_match_the_single_chunk_gradient(records):
    record = records[sorted(records)[0]]
    from scripts.train_topic5_interictal_rank_distribution import _batch
    from src.topic5_minimal_sequence_kernel import decomposed_next_set_stop_loss

    device = torch.device("cpu")
    segment = record.train_indices[:20]
    objective = objective_from_name("teacher_forced_one_step")

    def _gradient(chunk_size: int):
        model = _model()
        offset = torch.zeros((N_CONTACTS, 2), dtype=torch.float32)
        model.zero_grad(set_to_none=True)
        for start in range(0, len(segment), chunk_size):
            chunk = segment[start : start + chunk_size]
            batch = _batch(
                record, chunk, device, rank_shuffle=False, rng=np.random.default_rng(0)
            )
            outputs = scheduled_forward(
                model,
                batch["contact_features"],
                batch["contact_mask"],
                batch["group_ids"],
                batch["group_count"],
                offset,
                objective=objective,
                self_feed_probability=0.0,
            )
            loss = decomposed_next_set_stop_loss(
                outputs, batch["group_ids"], batch["group_count"]
            )
            (loss["total"] * (len(chunk) / len(segment))).backward()
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }

    whole = _gradient(len(segment))
    split = _gradient(5)
    for name in whole:
        torch.testing.assert_close(whole[name], split[name], rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- test 2


def test_coverage_cycle_and_update_counts_are_exact(records):
    subset = [records[key] for key in sorted(records)[:3]]
    _, rows, coverage = _train(subset, cycles=3, updates=4)
    frame = pd.DataFrame(rows)
    assert len(frame) == 3 * len(subset) * 4
    for (cycle, subject), group in frame.groupby(["coverage_cycle", "subject"]):
        assert len(group) == 4
        assert int(group.n_events.sum()) == int(
            coverage[subject]["events_available"]
        )
    assert set(frame.coverage_cycle) == {1, 2, 3}
    for subject, entry in coverage.items():
        assert entry["completed_cycles"] == 3
        assert entry["drawn"] == entry["events_available"] * 3


def test_nested_cycle_snapshot_equals_a_standalone_short_run(records):
    """Reading cycles {1,2,4} from one run must be exact, not approximate."""
    subset = [records[key] for key in sorted(records)[:3]]
    long_run, _, _ = _train(subset, cycles=3, updates=2, seed=41)
    short_run, _, _ = _train(subset, cycles=1, updates=2, seed=41)
    for key, value in short_run[1]["model_state"].items():
        torch.testing.assert_close(long_run[1]["model_state"][key], value)


# ---------------------------------------------------------------- test 3


def test_development_split_seals_the_outer_heldout(records):
    inner, audit = development_records(records, 0.10)
    assert len(audit) == N_SUBJECTS
    for subject, record in inner.items():
        original = records[subject]
        sealed = set(original.eval_indices.tolist())
        assert sealed, "fixture must have an outer heldout"
        assert not sealed & set(record.train_indices.tolist())
        assert not sealed & set(record.eval_indices.tolist())
        assert set(record.train_indices.tolist()) | set(
            record.eval_indices.tolist()
        ) == set(original.train_indices.tolist())
        assert np.all(record.event_split[original.eval_indices] == 2)
        # inner validation must be chronologically after inner training
        assert record.train_indices.max() < record.eval_indices.min()


def test_development_training_never_touches_sealed_events(records):
    inner, _ = development_records(records, 0.10)
    subset = [inner[key] for key in sorted(inner)[:3]]
    sealed = {
        record.subject: set(records[record.subject].eval_indices.tolist())
        for record in subset
    }
    _, rows, coverage = _train(subset, cycles=1, updates=2)
    for record in subset:
        assert coverage[record.subject]["events_available"] == int(
            record.train_indices.size
        )
        assert not sealed[record.subject] & set(record.train_indices.tolist())


# ---------------------------------------------------------------- test 4


def test_target_seal_is_enforced_by_the_loader(tmp_path):
    unsealed = _write_dataset(tmp_path / "unsealed", target_values_read=True)
    with pytest.raises(RuntimeError, match="sealed ictal targets"):
        load_records(unsealed)


def test_frozen_dataset_manifest_declares_the_seal():
    root = Path(__file__).resolve().parents[1] / (
        "results/topic5_interictal_rank_distribution/dataset_v0_4"
    )
    if not (root / "dataset_manifest.json").is_file():
        pytest.skip("frozen dataset artifact is not present")
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    assert manifest["target_values_read"] is False
    assert manifest["ab_or_kmeans_labels_read"] is False
    assert int(manifest["n_subjects_ok"]) == 34


# ---------------------------------------------------------------- test 5


@pytest.mark.parametrize("name", ["self_fed_2step", "self_fed_3step", "scheduled_sampling"])
def test_self_fed_history_never_repeats_a_contact(records, name):
    from scripts.train_topic5_interictal_rank_distribution import _batch

    record = records[sorted(records)[0]]
    device = torch.device("cpu")
    batch = _batch(
        record,
        record.train_indices[:24],
        device,
        rank_shuffle=False,
        rng=np.random.default_rng(0),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(5)
    outputs = scheduled_forward(
        _model(),
        batch["contact_features"],
        batch["contact_mask"],
        batch["group_ids"],
        batch["group_count"],
        torch.zeros((N_CONTACTS, 2)),
        objective=objective_from_name(name),
        self_feed_probability=1.0,
        generator=generator,
    )
    # every model-generated contact is distinct within its event: the number of
    # distinct emitted contacts equals the number of self-fed steps
    torch.testing.assert_close(
        outputs["model_emitted"].sum(1).to(torch.int64),
        outputs["model_fed_steps_per_event"],
    )
    # a self-fed contact is never one the model's own history already holds
    assert torch.all(
        outputs["model_emitted"] <= outputs["fed_recruited"]
    )
    # the true prefix bookkeeping is untouched by self-feeding
    torch.testing.assert_close(
        outputs["true_recruited"].sum(1).to(torch.int64), batch["group_count"]
    )
    model_fed, tie_fallback, eligible = outputs["self_feed_counters"].tolist()
    assert eligible >= model_fed + tie_fallback
    if name != "scheduled_sampling":
        assert model_fed > 0
        assert int(outputs["model_fed_steps_per_event"].sum()) > 0


def test_self_fed_preserves_the_teacher_forced_denominator(records):
    from scripts.train_topic5_interictal_rank_distribution import _batch
    from src.topic5_minimal_sequence_kernel import decomposed_next_set_stop_loss

    record = records[sorted(records)[0]]
    device = torch.device("cpu")
    batch = _batch(
        record,
        record.train_indices[:24],
        device,
        rank_shuffle=False,
        rng=np.random.default_rng(0),
    )
    model = _model()
    offset = torch.zeros((N_CONTACTS, 2))
    reference = scheduled_forward(
        model,
        batch["contact_features"],
        batch["contact_mask"],
        batch["group_ids"],
        batch["group_count"],
        offset,
        objective=objective_from_name("teacher_forced_one_step"),
        self_feed_probability=0.0,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(2)
    self_fed = scheduled_forward(
        model,
        batch["contact_features"],
        batch["contact_mask"],
        batch["group_ids"],
        batch["group_count"],
        offset,
        objective=objective_from_name("self_fed_3step"),
        self_feed_probability=1.0,
        generator=generator,
    )
    torch.testing.assert_close(
        reference["candidate_mask"], self_fed["candidate_mask"]
    )
    torch.testing.assert_close(
        reference["true_recruited"], self_fed["true_recruited"]
    )
    left = decomposed_next_set_stop_loss(
        reference, batch["group_ids"], batch["group_count"]
    )
    right = decomposed_next_set_stop_loss(
        self_fed, batch["group_ids"], batch["group_count"]
    )
    torch.testing.assert_close(left["decision_mask"], right["decision_mask"])
    torch.testing.assert_close(left["nonterminal_mask"], right["nonterminal_mask"])
    # the fed history differs, so the likelihood must actually change
    assert not torch.allclose(left["total"], right["total"])


def test_teacher_forced_objective_matches_the_frozen_forward(records):
    from scripts.train_topic5_interictal_rank_distribution import _batch

    record = records[sorted(records)[0]]
    device = torch.device("cpu")
    batch = _batch(
        record,
        record.train_indices[:16],
        device,
        rank_shuffle=False,
        rng=np.random.default_rng(0),
    )
    model = _model()
    offset = torch.zeros((N_CONTACTS, 2))
    frozen = model(**batch, local_offset=offset)
    audited = scheduled_forward(
        model,
        batch["contact_features"],
        batch["contact_mask"],
        batch["group_ids"],
        batch["group_count"],
        offset,
        objective=objective_from_name("teacher_forced_one_step"),
        self_feed_probability=0.0,
    )
    for key in ("contact_logits", "stop_logits", "candidate_mask"):
        torch.testing.assert_close(frozen[key], audited[key])


# ---------------------------------------------------------------- test 6


def test_state_and_history_freeze_after_stop(records):
    from scripts.train_topic5_interictal_rank_distribution import _batch

    record = records[sorted(records)[0]]
    device = torch.device("cpu")
    indices = record.train_indices[:32]
    batch = _batch(
        record, indices, device, rank_shuffle=False, rng=np.random.default_rng(0)
    )
    counts = batch["group_count"]
    assert int(counts.max()) > int(counts.min()), "need mixed event lengths"
    generator = torch.Generator(device=device)
    generator.manual_seed(9)
    outputs = scheduled_forward(
        _model(),
        batch["contact_features"],
        batch["contact_mask"],
        batch["group_ids"],
        batch["group_count"],
        torch.zeros((N_CONTACTS, 2)),
        objective=objective_from_name("self_fed_3step"),
        self_feed_probability=1.0,
        generator=generator,
    )
    logits = outputs["contact_logits"]
    for row in range(logits.shape[0]):
        stop_step = int(counts[row])
        for step in range(stop_step + 1, logits.shape[1]):
            torch.testing.assert_close(logits[row, step], logits[row, stop_step])


# ---------------------------------------------------------------- test 7


def test_identical_seed_reproduces_training(records):
    subset = [records[key] for key in sorted(records)[:3]]
    first, rows_first, _ = _train(subset, batch_size=16, seed=101, cycles=2, objective="scheduled_sampling")
    second, rows_second, _ = _train(subset, batch_size=16, seed=101, cycles=2, objective="scheduled_sampling")
    for key, value in first[2]["model_state"].items():
        torch.testing.assert_close(second[2]["model_state"][key], value, rtol=0, atol=0)
    assert [row["loss"] for row in rows_first] == [
        row["loss"] for row in rows_second
    ]

    other, _, _ = _train(subset, batch_size=16, seed=202, cycles=2, objective="scheduled_sampling")
    assert not all(
        torch.allclose(other[2]["model_state"][key], value)
        for key, value in first[2]["model_state"].items()
    )


# ---------------------------------------------------------------- test 8


def test_native_rollout_is_source_conditioned_and_paired(records):
    from src.topic5_training_sufficiency import paired_native_rollout

    record = records[sorted(records)[0]]
    device = torch.device("cpu")
    indices = record.eval_indices
    observed = np.asarray(record.group_ids[indices], dtype=np.int16)
    source = observed == 0
    uniforms = np.random.default_rng(4).random((observed.shape[0], N_CONTACTS))
    features = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask = torch.ones((1, N_CONTACTS), dtype=torch.bool, device=device)
    offset = torch.zeros((N_CONTACTS, 2))

    groups, counts = paired_native_rollout(
        _model(), features, mask, offset, source, uniforms
    )
    # the revealed first rank set is retained verbatim
    assert np.all(groups[source] == 0)
    # ranks are contiguous from zero and never repeated inside an event
    for event, length in zip(groups, counts):
        present = np.sort(event[event >= 0])
        assert present.tolist() == list(range(int(length)))
    assert np.all(counts >= 1)

    # identical uniforms and identical parameters reproduce the run exactly
    again, again_counts = paired_native_rollout(
        _model(), features, mask, offset, source, uniforms
    )
    np.testing.assert_array_equal(groups, again)
    np.testing.assert_array_equal(counts, again_counts)

    # a different model with the same uniforms must still be a valid rollout
    other, other_counts = paired_native_rollout(
        _model(seed=99), features, mask, offset, source, uniforms
    )
    assert np.all(other[source] == 0)
    assert np.all(other_counts >= 1)


def test_resume_plan_skips_complete_and_blocks_partial(tmp_path):
    cells = ["a", "b", "c"]
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "DONE.json").write_text("{}")
    (tmp_path / "b").mkdir()
    plan = plan_cells(cells, tmp_path)
    assert plan["complete"] == ["a"]
    assert plan["blocked"] == ["b"]
    assert plan["pending"] == ["c"]
    assert len(plan["complete"]) + len(plan["pending"]) + len(plan["blocked"]) == 3

    (tmp_path / "b" / "DONE.json").write_text("{}")
    second = plan_cells(cells, tmp_path)
    assert second["pending"] == ["c"]
    assert not second["blocked"]


# ---------------------------------------------------------------- test 9


def test_patient_first_aggregation_ignores_event_counts():
    rows = [
        {"subject": "big", "seed": 1, "value": 1.0, "n_events": 100000},
        {"subject": "big", "seed": 2, "value": 1.0, "n_events": 100000},
        {"subject": "small_a", "seed": 1, "value": 5.0, "n_events": 10},
        {"subject": "small_b", "seed": 1, "value": 5.0, "n_events": 10},
    ]
    summary = aggregate_patient_metric(rows, value_key="value")
    assert summary["n_patients"] == 3
    assert summary["median"] == 5.0
    assert summary["per_patient"]["big"] == 1.0
    assert summary["n_seeds_per_patient"]["big"] == 2

    pooled = float(
        np.average(
            [row["value"] for row in rows],
            weights=[row["n_events"] for row in rows],
        )
    )
    assert abs(pooled - summary["median"]) > 3.0


def test_patient_first_summary_handles_missing_values():
    summary = patient_first_summary({"a": 1.0, "b": float("nan"), "c": 3.0})
    assert summary["n_patients"] == 2
    assert summary["median"] == 2.0


# ---------------------------------------------------------------- extras


def test_plateau_rule_needs_two_quiet_cycles():
    still_moving = plateau_verdict([2.0, 1.5, 1.0, 0.5])
    assert still_moving["plateau_reached"] is False

    settled = plateau_verdict([2.0, 1.5, 1.4995, 1.4991])
    assert settled["plateau_reached"] is True
    assert settled["plateau_at_cycle"] == 4

    assert plateau_verdict([1.0])["plateau_reached"] is False


def test_objective_feed_schedules_are_frozen():
    two = objective_from_name("self_fed_2step")
    three = objective_from_name("self_fed_3step")
    assert [two.feeds_model_at_step(step) for step in range(6)] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert [three.feeds_model_at_step(step) for step in range(6)] == [
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert two.max_consecutive_model_steps == 1
    assert three.max_consecutive_model_steps == 2
    assert objective_from_name("teacher_forced_one_step").feeds_model_at_step(3) is False

    ramp = objective_from_name("scheduled_sampling")
    assert ramp.feeds_model_at_step(0) is False
    assert ramp.self_feed_probability(0.0) == 0.0
    assert ramp.self_feed_probability(1.0) == pytest.approx(0.5)
    assert ramp.self_feed_probability(0.5) == pytest.approx(0.25)


def test_offset_calibration_snapshots_two_budgets(records):
    record = records[sorted(records)[0]]
    model = _model()
    snapshots, rows, coverage = calibrate_offset_instrumented(
        model,
        record,
        coverage_cycles=4,
        updates_per_cycle=2,
        batch_size=16,
        local_learning_rate=2e-3,
        weight_decay=1e-4,
        gradient_clip=1.0,
        local_offset_dim=2,
        device=torch.device("cpu"),
        seed=7,
        objective=objective_from_name("teacher_forced_one_step"),
        snapshot_cycles=(2, 4),
    )
    assert sorted(snapshots) == [2, 4]
    assert not torch.allclose(snapshots[2], snapshots[4])
    assert len(rows) == 4 * 2
    assert coverage["completed_cycles"] == 4
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_evaluate_decomposition_is_additive(records):
    record = records[sorted(records)[0]]
    metrics = evaluate_decomposed(
        _model(),
        record,
        torch.zeros((N_CONTACTS, 2)),
        device=torch.device("cpu"),
        batch_size=16,
    )
    assert metrics["n_events"] == int(record.eval_indices.size)
    assert metrics["n_decisions"] > metrics["n_nonterminal_decisions"] > 0
    assert np.isfinite(metrics["contact_choice_nll"])
    assert np.isfinite(metrics["stop_contribution_nll"])
    assert np.isfinite(metrics["event_total_nll"])


def test_unknown_objective_and_optimizer_are_rejected(records):
    with pytest.raises(ValueError):
        objective_from_name("reinforce")
    with pytest.raises(ValueError):
        Objective(name="mystery").feeds_model_at_step(0)
    subset = [records[key] for key in sorted(records)[:2]]
    with pytest.raises(ValueError, match="weight decay 0"):
        _train(subset, batch_size=16, optimizer="adam", weight_decay=1e-4)


@pytest.mark.parametrize("mode", ["ordered", "frozen", "shuffled"])
def test_native_rollout_history_modes_are_matched(records, mode):
    """The ablation must change only what the state is fed, nothing else."""
    from src.topic5_training_sufficiency import paired_native_rollout

    record = records[sorted(records)[0]]
    device = torch.device("cpu")
    observed = np.asarray(record.group_ids[record.eval_indices], dtype=np.int16)
    source = observed == 0
    uniforms = np.random.default_rng(12).random((observed.shape[0], N_CONTACTS))
    features = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask = torch.ones((1, N_CONTACTS), dtype=torch.bool, device=device)
    offset = torch.zeros((N_CONTACTS, 2))

    groups, counts = paired_native_rollout(
        _model(), features, mask, offset, source, uniforms,
        history_mode=mode, history_seed=5,
    )
    # the revealed source and the contiguous-rank contract hold in every arm
    assert np.all(groups[source] == 0)
    for event, length in zip(groups, counts):
        present = np.sort(event[event >= 0])
        assert present.tolist() == list(range(int(length)))

    again, again_counts = paired_native_rollout(
        _model(), features, mask, offset, source, uniforms,
        history_mode=mode, history_seed=5,
    )
    np.testing.assert_array_equal(groups, again)
    np.testing.assert_array_equal(counts, again_counts)


def test_native_rollout_history_modes_actually_differ(records):
    from src.topic5_training_sufficiency import paired_native_rollout

    record = records[sorted(records)[0]]
    device = torch.device("cpu")
    observed = np.asarray(record.group_ids[record.eval_indices], dtype=np.int16)
    source = observed == 0
    # bias the inverse-CDF draws away from the STOP action so events run long
    # enough for a shuffled history to have anything to shuffle
    uniforms = 0.5 + 0.5 * np.random.default_rng(13).random(
        (observed.shape[0], N_CONTACTS)
    )
    features = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask = torch.ones((1, N_CONTACTS), dtype=torch.bool, device=device)
    offset = torch.zeros((N_CONTACTS, 2))
    def _state_coupled_model():
        # an untrained toy model is dominated by its per-contact bias, so the
        # hidden state barely reaches the output; amplify the state-to-output
        # path so the ablation has something observable to change
        model = _model()
        with torch.no_grad():
            model.action_query.weight.mul_(50.0)
            model.stop_head.bias.fill_(-20.0)  # keep events running
        return model

    runs = {
        mode: paired_native_rollout(
            _state_coupled_model(), features, mask, offset, source, uniforms,
            history_mode=mode, history_seed=5,
        )[0]
        for mode in ("ordered", "frozen", "shuffled")
    }
    assert not np.array_equal(runs["ordered"], runs["frozen"])
    assert not np.array_equal(runs["ordered"], runs["shuffled"])

    with pytest.raises(ValueError):
        paired_native_rollout(
            _model(), features, mask, offset, source, uniforms,
            history_mode="mystery",
        )

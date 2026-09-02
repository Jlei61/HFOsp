import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_topic5_multiscale_effective_scaffold_v0_5",
    ROOT / "scripts/audit_topic5_multiscale_effective_scaffold_v0_5.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

BLOCK_SPEC = importlib.util.spec_from_file_location(
    "analyse_topic5_multiscale_block_heldout_v0_5",
    ROOT / "scripts/analyse_topic5_multiscale_block_heldout_v0_5.py",
)
BLOCK_MODULE = importlib.util.module_from_spec(BLOCK_SPEC)
assert BLOCK_SPEC.loader is not None
BLOCK_SPEC.loader.exec_module(BLOCK_MODULE)

MIXTURE_SPEC = importlib.util.spec_from_file_location(
    "repair_topic5_multiscale_train_mixture_v0_5",
    ROOT / "scripts/repair_topic5_multiscale_train_mixture_v0_5.py",
)
MIXTURE_MODULE = importlib.util.module_from_spec(MIXTURE_SPEC)
assert MIXTURE_SPEC.loader is not None
MIXTURE_SPEC.loader.exec_module(MIXTURE_MODULE)

SCORER_SPEC = importlib.util.spec_from_file_location(
    "score_topic5_multiscale_early_ictal_v0_5",
    ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py",
)
SCORER_MODULE = importlib.util.module_from_spec(SCORER_SPEC)
assert SCORER_SPEC.loader is not None
SCORER_SPEC.loader.exec_module(SCORER_MODULE)


def test_densify_groups_removes_dropped_rank_gaps():
    groups = np.asarray([[0, 2, -1, 4], [1, -1, 3, 3]], dtype=np.int16)
    observed = MODULE.densify_groups(groups)
    expected = np.asarray([[0, 1, -1, 2], [0, -1, 1, 1]], dtype=np.int16)
    assert np.array_equal(observed, expected)


def test_relative_latency_span_requires_two_rank_sets():
    groups = np.asarray([[0, 1, -1], [0, 0, -1]], dtype=np.int16)
    lag = np.asarray([[0.010, 0.035, np.nan], [0.100, 0.120, np.nan]])
    span = MODULE.relative_latency_span_ms(lag, groups)
    assert np.isclose(span[0], 25.0)
    assert np.isnan(span[1])


def test_strict_block_heldout_removes_only_test_events_from_exposed_blocks():
    split = np.asarray([0, 0, 1, 2, 2, 2, -1], dtype=np.int8)
    source_index = np.asarray([3, 4, 8, 9, 10, 12, 13], dtype=np.int64)
    # Block 1 contains validation plus the first test event; blocks 2 and 3
    # are truly unseen.  The split=-1 event must not count as exposure.
    raw_blocks = np.asarray(
        [9, 9, 9, 0, 0, 8, 8, 8, 1, 1, 2, 7, 3, 3], dtype=np.int64
    )
    strict, audit = BLOCK_MODULE.strict_unseen_test_events(
        split, source_index, raw_blocks
    )
    # Returned indices refer to the compact split>=0 ordering.
    assert strict.tolist() == [4, 5]
    assert audit["n_test_events"] == 3
    assert audit["n_strict_unseen_block_test_events"] == 2
    assert audit["n_test_events_removed_boundary_block"] == 1
    assert audit["n_strict_test_blocks"] == 2


def test_recovery_subjects_are_exactly_the_five_missing_spatial_patients():
    assert set(MODULE.RECOVERY_SUBJECTS) == {
        "epilepsiae_1077", "epilepsiae_1096", "epilepsiae_1125",
        "epilepsiae_139", "epilepsiae_635",
    }


def test_stage_f_plot_uses_named_sem_column_and_matching_completion_path():
    plot_source = (ROOT / "scripts/plot_topic5_multiscale_stage_f_v0_5.py").read_text()
    driver_source = (ROOT / "scripts/run_topic5_multiscale_stage_f_v0_5.py").read_text()
    assert 'summary["median"] - summary["sem"]' in plot_source
    assert 'summary["median"] + summary["sem"]' in plot_source
    assert "summary.sem" not in plot_source
    assert "figures/stage_f_v0_5_target_free_mechanism.png" in driver_source


def test_full_event_mode_assignment_does_not_collapse_with_prefix_classifier(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # Full fields form two clean train-defined modes, while a prefix-only hard
    # classifier has collapsed every event to mode 0.  Mixture components must
    # use the former, not silently lose mode 1.
    ranks = np.asarray([
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [3, 2, 1, 0],
        [2, 3, 1, 0],
    ], dtype=np.int16)
    event_features = MIXTURE_MODULE.features(ranks)
    centers = np.stack([
        event_features[:2].mean(axis=0),
        event_features[2:].mean(axis=0),
    ])
    np.savez_compressed(
        cache / "events.npz",
        ranks=ranks,
        mode=np.zeros(4, dtype=np.int8),
        event_source_index=np.arange(10, 14, dtype=np.int64),
    )
    np.savez_compressed(cache / "train_only_modes.npz", centers=centers)
    observed = MIXTURE_MODULE.full_event_train_modes(cache)
    assert observed.tolist() == [0, 0, 1, 1]

    records = [{"event_source_index": value, "mode": 0} for value in range(10, 14)]
    remapped = MIXTURE_MODULE.remap_record_full_train_modes(records, cache)
    assert [row["mode"] for row in remapped] == [0, 0, 1, 1]


def test_early_inventory_uses_only_prefrozen_attenuation_fields(tmp_path):
    subject = "patient_x"
    root = tmp_path / "attenuation/fields/per_patient" / subject
    for alpha in (0.25, 0.50, 0.75, 1.00):
        path = root / "L3_ADDED" / f"alpha{alpha:.2f}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, value=np.asarray([1.0]))
    available = SCORER_MODULE.available_attenuation_conditions(tmp_path, subject)
    assert available == {
        f"ATTEN|L3_ADDED|{alpha:.2f}" for alpha in (0.25, 0.50, 0.75, 1.00)
    }
    canonical = SCORER_MODULE.expected_condition_inventory("canonical_full", available)
    seed_removed = SCORER_MODULE.expected_condition_inventory("seed_removed", available)
    assert not any("L3_MATCHED_LOCAL" in value for value in canonical)
    assert len(canonical) == 19
    assert len(seed_removed) == 17

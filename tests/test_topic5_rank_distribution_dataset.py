from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.build_topic5_interictal_rank_distribution_dataset import (
    _eligible_subjects,
)
from scripts.train_topic5_interictal_rank_distribution import (
    BalancedSubjectSampler,
    SubjectRecord,
    train_shared_coverage,
)
from src.topic5_rank_distribution import FullHistorySequenceGRU


def test_v04_cohort_audit_resolves_exact_34_subjects():
    cfg = {
        "cohort": {
            "audit": "results/topic4_attractor_masked/step0_audit.csv",
            "eligible_column": "eligible_for_main",
            "expected_subjects": 34,
            "expected_epilepsiae": 18,
            "expected_yuquan": 16,
        }
    }
    subjects, rows = _eligible_subjects(cfg)
    assert len(subjects) == 34
    assert rows["dataset"].value_counts().to_dict() == {
        "epilepsiae": 18,
        "yuquan": 16,
    }
    assert "epilepsiae_916" not in subjects


def test_v04_cohort_source_has_no_ictal_value_columns():
    path = Path("results/topic4_attractor_masked/step0_audit.csv")
    columns = set(pd.read_csv(path, nrows=1).columns)
    assert not any("energy" in column.lower() for column in columns)
    assert not any("seizure" in column.lower() for column in columns)


def test_shared_sampler_balances_datasets_and_cycles_patients_without_replacement():
    records = [
        SimpleNamespace(dataset="epilepsiae", subject=f"e{index}")
        for index in range(3)
    ] + [
        SimpleNamespace(dataset="yuquan", subject=f"y{index}")
        for index in range(2)
    ]
    sampler = BalancedSubjectSampler(records, np.random.default_rng(4))
    draws = [sampler.draw(step) for step in range(6)]
    assert [draw.dataset for draw in draws] == [
        "epilepsiae",
        "yuquan",
        "epilepsiae",
        "yuquan",
        "epilepsiae",
        "yuquan",
    ]
    assert len({draw.subject for draw in draws[0::2]}) == 3
    assert len({draw.subject for draw in draws[1:4:2]}) == 2


def test_formal_shared_training_records_one_complete_event_cycle():
    torch = __import__("pytest").importorskip("torch")
    records = []
    for index, dataset in enumerate(("epilepsiae", "yuquan")):
        groups = np.array(
            [[0, 1, -1], [1, 0, -1], [0, -1, 1], [-1, 0, 1]],
            dtype=np.int16,
        )
        records.append(
            SubjectRecord(
                subject=f"s{index}",
                dataset=dataset,
                path=Path(f"s{index}.npz"),
                contact_features=np.zeros((3, 8), np.float32),
                contact_names=np.array(["A1", "A2", "A3"]),
                group_ids=groups,
                group_count=np.full(4, 2, np.int16),
                event_split=np.zeros(4, np.uint8),
                event_source_index=np.arange(4),
                input_sha256="test",
            )
        )
    model = FullHistorySequenceGRU(
        8,
        hidden_size=8,
        contact_embedding_dim=8,
        contact_encoder_hidden=8,
        local_offset_dim=4,
    )
    _, offsets, _, coverage = train_shared_coverage(
        model,
        records,
        coverage_cycles=1,
        updates_per_patient=2,
        batch_size=2,
        learning_rate=1e-3,
        local_learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=1.0,
        local_offset_dim=4,
        device=torch.device("cpu"),
        seed=4,
        rank_shuffle=False,
    )
    assert set(offsets) == {"s0", "s1"}
    assert all(value["completed_cycles"] == 1 for value in coverage.values())
    assert all(value["drawn"] == 4 for value in coverage.values())

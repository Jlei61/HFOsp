from types import SimpleNamespace

import numpy as np

from src.topic5_continuous_marked_state_r1 import optimizer_audit


def test_nested_time_split_keeps_alignment_selection_unseen(monkeypatch) -> None:
    design = SimpleNamespace(
        subject="example",
        anchor_time=np.arange(100, dtype=np.float64),
        anchor_ids=lambda split: np.arange(100, dtype=np.int64),
    )
    monkeypatch.setattr(
        optimizer_audit.contract, "load_split", lambda subject: (100.0, 120.0)
    )
    split = optimizer_audit.nested_time_split(design)
    assert split.base_train_ids.tolist() == list(range(60))
    assert split.prefix_refit_ids.tolist() == list(range(80))
    assert split.base_select_lower == 60.0
    assert split.base_select_upper == 80.0
    assert split.alignment_select_lower == 80.0
    assert split.alignment_select_upper == 100.0
    assert max(split.prefix_refit_ids) < split.alignment_select_lower


def test_nested_time_split_rejects_invalid_fractions(monkeypatch) -> None:
    design = SimpleNamespace(
        subject="example",
        anchor_time=np.arange(100, dtype=np.float64),
        anchor_ids=lambda split: np.arange(100, dtype=np.int64),
    )
    monkeypatch.setattr(
        optimizer_audit.contract, "load_split", lambda subject: (100.0, 120.0)
    )
    for base, alignment in ((0.8, 0.6), (0.0, 0.8), (0.6, 1.0)):
        try:
            optimizer_audit.nested_time_split(
                design, base_fraction=base, alignment_fraction=alignment
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid nested fractions were accepted")


def test_fixed_overfit_segment_stays_inside_one_session(monkeypatch) -> None:
    design = SimpleNamespace(
        subject="example",
        anchor_time=np.arange(100, dtype=np.float64),
        anchor_session=np.asarray([0] * 20 + [1] * 80),
        session_label=np.asarray([0, 1]),
        anchor_ids=lambda split: np.arange(100, dtype=np.int64),
    )
    monkeypatch.setattr(
        optimizer_audit.contract, "load_split", lambda subject: (100.0, 120.0)
    )
    split = optimizer_audit.nested_time_split(design)
    anchors, lower, upper = optimizer_audit.fixed_overfit_segment(
        design, split, maximum_anchors=16
    )
    assert anchors.tolist() == list(range(20, 36))
    assert lower == 20.0
    assert upper == 36.0
    assert len(set(design.anchor_session[anchors])) == 1

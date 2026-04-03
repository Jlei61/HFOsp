from scripts.pr25_loso_validation import (
    _aggregate_subject_summaries,
    _score_candidate,
    _summarize_subject_rows,
)


def test_aggregate_subject_summaries_computes_global_metrics() -> None:
    rows = [
        {"n_manual": 5, "n_detected": 6, "TP": 4, "FP": 2, "FN": 1},
        {"n_manual": 3, "n_detected": 4, "TP": 2, "FP": 2, "FN": 1},
    ]
    out = _aggregate_subject_summaries(rows)
    assert out["n_subjects"] == 2
    assert out["n_manual"] == 8
    assert out["n_detected"] == 10
    assert out["TP"] == 6
    assert out["FP"] == 4
    assert out["FN"] == 2
    assert out["recall"] == 0.75
    assert out["precision"] == 0.6
    assert out["fp_per_manual"] == 0.5


def test_score_candidate_prioritizes_recall_gate_then_lower_fp() -> None:
    strong = _score_candidate(
        {"macro_recall": 0.85, "macro_precision": 0.30, "macro_fp_per_manual": 1.0}, 0.80
    )
    weak_recall = _score_candidate(
        {"macro_recall": 0.79, "macro_precision": 0.90, "macro_fp_per_manual": 0.1}, 0.80
    )
    assert strong > weak_recall

    lower_fp = _score_candidate(
        {"macro_recall": 0.85, "macro_precision": 0.30, "macro_fp_per_manual": 0.5}, 0.80
    )
    higher_fp = _score_candidate(
        {"macro_recall": 0.85, "macro_precision": 0.80, "macro_fp_per_manual": 1.5}, 0.80
    )
    assert lower_fp > higher_fp


def test_summarize_subject_rows_adds_macro_metrics() -> None:
    rows = [
        {"n_manual": 10, "n_detected": 10, "TP": 9, "FP": 1, "FN": 1, "recall": 0.9, "precision": 0.9, "f1": 0.9, "fp_per_manual": 0.1},
        {"n_manual": 2, "n_detected": 5, "TP": 1, "FP": 4, "FN": 1, "recall": 0.5, "precision": 0.2, "f1": 0.286, "fp_per_manual": 2.0},
    ]
    out = _summarize_subject_rows(rows)
    assert out["recall"] == 10 / 12
    assert out["macro_recall"] == 0.7
    assert out["macro_precision"] == 0.55
    assert out["macro_fp_per_manual"] == 1.05

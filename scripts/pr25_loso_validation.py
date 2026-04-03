"""Cross-patient PR2.5 validation with held-out subject folds.

This script treats the current spatial-extent detector as a candidate model and
evaluates whether one global parameter set generalizes to held-out labeled
patients across Yuquan and Epilepsiae.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pr2_seizure_validation import DEFAULT_ROOT, run_validation
from src.epilepsiae_dataset import survey_epilepsiae_dataset


DEFAULT_YUQUAN_LABELED_SUBJECTS = [
    "gaolan",
    "litengsheng",
    "sunyuanxin",
    "xuxinyi",
    "chenziyang",
    "hanyuxuan",
    "huanghanwen",
    "zhangjinhan",
]


def _parse_csv_arg(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _candidate_grid(args: argparse.Namespace) -> List[Dict[str, float]]:
    grid = []
    for per_channel_k, min_active_frac, min_duration_sec, merge_gap_sec, min_channel_consec_sec in itertools.product(
        _parse_csv_arg(args.per_channel_k_grid),
        _parse_csv_arg(args.min_active_frac_grid),
        _parse_csv_arg(args.min_duration_grid),
        _parse_csv_arg(args.merge_gap_grid),
        _parse_csv_arg(args.min_channel_consec_grid),
    ):
        grid.append(
            {
                "per_channel_k": float(per_channel_k),
                "min_active_frac": float(min_active_frac),
                "min_duration_sec": float(min_duration_sec),
                "merge_gap_sec": float(merge_gap_sec),
                "min_channel_consec_sec": float(min_channel_consec_sec),
            }
        )
    return grid


def _discover_epilepsiae_subjects(limit: int | None = None) -> List[str]:
    inventory = survey_epilepsiae_dataset()
    block_ok = {}
    for row in inventory.block_rows:
        subject = str(row["subject"])
        ready = (
            bool(row["head_exists"])
            and bool(row["data_exists"])
            and int(row["intracranial_channels"] or 0) >= 2
        )
        block_ok[subject] = bool(block_ok.get(subject, False) or ready)
    subjects = [
        str(row["subject"])
        for row in inventory.subject_rows
        if int(row["n_complete_eeg_intervals"] or 0) > 0 and bool(block_ok.get(str(row["subject"]), False))
    ]
    subjects = sorted(set(subjects))
    return subjects if limit is None else subjects[:limit]


def _subject_pool(args: argparse.Namespace) -> List[Dict[str, str]]:
    yuquan_subjects = (
        DEFAULT_YUQUAN_LABELED_SUBJECTS
        if str(args.yuquan_subjects).strip().lower() == "default"
        else _parse_csv_arg(args.yuquan_subjects)
    )
    epilepsiae_subjects = (
        _discover_epilepsiae_subjects(limit=args.max_epilepsiae_subjects)
        if str(args.epilepsiae_subjects).strip().lower() == "auto"
        else _parse_csv_arg(args.epilepsiae_subjects)
    )
    pool = [{"dataset": "yuquan", "subject": s} for s in yuquan_subjects]
    pool.extend({"dataset": "epilepsiae", "subject": s} for s in epilepsiae_subjects)
    return pool


def _aggregate_subject_summaries(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    n_manual = sum(int(r["n_manual"]) for r in rows)
    n_detected = sum(int(r["n_detected"]) for r in rows)
    tp = sum(int(r["TP"]) for r in rows)
    fp = sum(int(r["FP"]) for r in rows)
    fn = sum(int(r["FN"]) for r in rows)
    recall = (tp / n_manual) if n_manual else 0.0
    precision = (tp / n_detected) if n_detected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_per_manual = (fp / n_manual) if n_manual else math.inf
    return {
        "n_subjects": len(rows),
        "n_manual": n_manual,
        "n_detected": n_detected,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fp_per_manual": fp_per_manual,
    }


def _macro_average_subject_summaries(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "macro_recall": 0.0,
            "macro_precision": 0.0,
            "macro_f1": 0.0,
            "macro_fp_per_manual": math.inf,
        }
    return {
        "macro_recall": sum(float(r["recall"]) for r in rows) / len(rows),
        "macro_precision": sum(float(r["precision"]) for r in rows) / len(rows),
        "macro_f1": sum(float(r["f1"]) for r in rows) / len(rows),
        "macro_fp_per_manual": sum(float(r["fp_per_manual"]) for r in rows) / len(rows),
    }


def _summarize_subject_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    out = _aggregate_subject_summaries(rows)
    out.update(_macro_average_subject_summaries(rows))
    return out


def _score_candidate(summary: Dict[str, float], recall_target: float) -> Tuple[int, float, float, float]:
    meets_recall = int(float(summary["macro_recall"]) >= float(recall_target))
    return (
        meets_recall,
        -float(summary["macro_fp_per_manual"]),
        float(summary["macro_recall"]),
        float(summary["macro_precision"]),
    )


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_subject(
    item: Dict[str, str],
    params: Dict[str, float],
    *,
    data_root: Path,
    output_root: Path,
    cache_root: Path,
    n_jobs: int,
    make_plots: bool,
    write_audit: bool,
) -> Dict[str, object]:
    dataset = str(item["dataset"])
    subject = str(item["subject"])
    return run_validation(
        dataset=dataset,
        subject=subject,
        data_root=data_root,
        output_dir=output_root / dataset / subject,
        cache_dir=cache_root / dataset / subject,
        n_jobs=n_jobs,
        make_plots=make_plots,
        write_audit=write_audit,
        **params,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="LOSO validation for PR2.5 seizure detector")
    ap.add_argument("--yuquan-subjects", default="default")
    ap.add_argument("--epilepsiae-subjects", default="auto")
    ap.add_argument("--max-epilepsiae-subjects", type=int, default=None)
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--output-dir", default="results/pr25_loso")
    ap.add_argument("--cache-dir", default="results/pr25_loso_cache")
    ap.add_argument("--per-channel-k-grid", default="4.0,5.0,6.0")
    ap.add_argument("--min-active-frac-grid", default="0.30,0.40,0.50")
    ap.add_argument("--min-duration-grid", default="30,45")
    ap.add_argument("--merge-gap-grid", default="3,5,10")
    ap.add_argument("--min-channel-consec-grid", default="0,5,10")
    ap.add_argument("--recall-target", type=float, default=0.80)
    ap.add_argument("--fp-same-order-threshold", type=float, default=2.0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument(
        "--plots-for-heldout",
        action="store_true",
        help="Render plots/audits for held-out evaluations only",
    )
    args = ap.parse_args()

    subject_pool = _subject_pool(args)
    if len(subject_pool) < 2:
        raise ValueError("Need at least 2 labeled subjects for held-out validation.")

    output_root = Path(args.output_dir)
    cache_root = Path(args.cache_dir)
    data_root = Path(args.data_root)
    grid = _candidate_grid(args)

    baseline_params = {
        "per_channel_k": 5.0,
        "min_active_frac": 0.40,
        "min_duration_sec": 30.0,
        "merge_gap_sec": 5.0,
        "min_channel_consec_sec": 5.0,
    }
    baseline_rows: List[Dict[str, object]] = []
    baseline_root = output_root / "baseline_subjects"
    for item in subject_pool:
        row = _run_subject(
            item,
            baseline_params,
            data_root=data_root,
            output_root=baseline_root,
            cache_root=cache_root / "baseline",
            n_jobs=args.n_jobs,
            make_plots=False,
            write_audit=False,
        )
        if row is None:
            raise RuntimeError(f"Failed baseline validation for {item['dataset']}/{item['subject']}")
        baseline_rows.append(row)
    baseline_summary = _summarize_subject_rows(baseline_rows)

    candidate_rows: List[Dict[str, object]] = []
    heldout_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []

    for heldout in subject_pool:
        train_items = [x for x in subject_pool if x != heldout]
        best_candidate = None
        best_score = None
        best_train_summary = None
        for candidate_idx, params in enumerate(grid):
            train_subject_rows = []
            for item in train_items:
                row = _run_subject(
                    item,
                    params,
                    data_root=data_root,
                    output_root=output_root / "train_cache_only",
                    cache_root=cache_root / "grid",
                    n_jobs=args.n_jobs,
                    make_plots=False,
                    write_audit=False,
                )
                if row is None:
                    raise RuntimeError(f"Failed train validation for {item['dataset']}/{item['subject']}")
                train_subject_rows.append(row)
            train_summary = _summarize_subject_rows(train_subject_rows)
            score = _score_candidate(train_summary, args.recall_target)
            candidate_row = {
                "heldout_dataset": heldout["dataset"],
                "heldout_subject": heldout["subject"],
                "candidate_idx": candidate_idx,
                **params,
                **train_summary,
                "meets_recall_target": int(train_summary["macro_recall"] >= float(args.recall_target)),
            }
            candidate_rows.append(candidate_row)
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = dict(params)
                best_train_summary = dict(train_summary)

        heldout_summary = _run_subject(
            heldout,
            best_candidate,
            data_root=data_root,
            output_root=output_root / "heldout_subjects",
            cache_root=cache_root / "heldout",
            n_jobs=args.n_jobs,
            make_plots=args.plots_for_heldout,
            write_audit=args.plots_for_heldout,
        )
        if heldout_summary is None:
            raise RuntimeError(f"Failed held-out validation for {heldout['dataset']}/{heldout['subject']}")
        heldout_rows.append(
            {
                "heldout_dataset": heldout["dataset"],
                "heldout_subject": heldout["subject"],
                **best_candidate,
                "train_recall": best_train_summary["recall"],
                "train_precision": best_train_summary["precision"],
                "train_fp_per_manual": best_train_summary["fp_per_manual"],
                "train_macro_recall": best_train_summary["macro_recall"],
                "train_macro_precision": best_train_summary["macro_precision"],
                "train_macro_fp_per_manual": best_train_summary["macro_fp_per_manual"],
                "heldout_recall": heldout_summary["recall"],
                "heldout_precision": heldout_summary["precision"],
                "heldout_f1": heldout_summary["f1"],
                "heldout_fp_per_manual": heldout_summary["fp_per_manual"],
                "heldout_FP": heldout_summary["FP"],
                "heldout_FN": heldout_summary["FN"],
                "heldout_manual": heldout_summary["n_manual"],
                "heldout_detected": heldout_summary["n_detected"],
                "heldout_median_onset_err_s": heldout_summary["median_onset_err_s"],
            }
        )
        fold_rows.append(
            {
                "heldout_dataset": heldout["dataset"],
                "heldout_subject": heldout["subject"],
                **best_candidate,
                "train_recall": best_train_summary["recall"],
                "train_precision": best_train_summary["precision"],
                "train_fp_per_manual": best_train_summary["fp_per_manual"],
                "train_macro_recall": best_train_summary["macro_recall"],
                "train_macro_precision": best_train_summary["macro_precision"],
                "train_macro_fp_per_manual": best_train_summary["macro_fp_per_manual"],
                "heldout_recall": heldout_summary["recall"],
                "heldout_precision": heldout_summary["precision"],
                "heldout_f1": heldout_summary["f1"],
                "heldout_fp_per_manual": heldout_summary["fp_per_manual"],
                "heldout_pass_recall": int(float(heldout_summary["recall"]) >= float(args.recall_target)),
                "heldout_pass_fp_same_order": int(
                    float(heldout_summary["fp_per_manual"]) <= float(args.fp_same_order_threshold)
                ),
            }
        )

    baseline_by_dataset = {}
    heldout_by_dataset = {}
    for dataset_name in sorted({str(r["dataset"]) for r in baseline_rows}):
        baseline_by_dataset[dataset_name] = _summarize_subject_rows(
            [r for r in baseline_rows if str(r["dataset"]) == dataset_name]
        )
    for dataset_name in sorted({str(r["heldout_dataset"]) for r in heldout_rows}):
        dataset_rows = [r for r in heldout_rows if str(r["heldout_dataset"]) == dataset_name]
        heldout_by_dataset[dataset_name] = {
            "n_folds": len(dataset_rows),
            "mean_heldout_recall": sum(float(r["heldout_recall"]) for r in dataset_rows) / len(dataset_rows),
            "mean_heldout_precision": sum(float(r["heldout_precision"]) for r in dataset_rows) / len(dataset_rows),
            "mean_heldout_fp_per_manual": sum(float(r["heldout_fp_per_manual"]) for r in dataset_rows) / len(dataset_rows),
        }

    heldout_aggregate = {
        "n_folds": len(fold_rows),
        "mean_heldout_recall": sum(float(r["heldout_recall"]) for r in fold_rows) / len(fold_rows),
        "mean_heldout_precision": sum(float(r["heldout_precision"]) for r in fold_rows) / len(fold_rows),
        "mean_heldout_fp_per_manual": sum(float(r["heldout_fp_per_manual"]) for r in fold_rows) / len(fold_rows),
        "n_folds_pass_recall": sum(int(r["heldout_pass_recall"]) for r in fold_rows),
        "n_folds_pass_fp_same_order": sum(int(r["heldout_pass_fp_same_order"]) for r in fold_rows),
        "recall_target": float(args.recall_target),
        "fp_same_order_threshold": float(args.fp_same_order_threshold),
    }

    _write_csv(output_root / "baseline_subject_summary.csv", baseline_rows)
    _write_csv(output_root / "candidate_scores.csv", candidate_rows)
    _write_csv(output_root / "heldout_subject_summary.csv", heldout_rows)
    _write_csv(output_root / "fold_summary.csv", fold_rows)
    with open(output_root / "loso_summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "subject_pool": subject_pool,
                "baseline_params": baseline_params,
                "baseline_summary": baseline_summary,
                "baseline_by_dataset": baseline_by_dataset,
                "heldout_summary": heldout_aggregate,
                "heldout_by_dataset": heldout_by_dataset,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print("=== PR2.5 LOSO summary ===")
    print(f"Subjects: {len(subject_pool)}")
    print(
        f"Baseline recall={baseline_summary['recall']:.3f}, "
        f"precision={baseline_summary['precision']:.3f}, "
        f"fp/manual={baseline_summary['fp_per_manual']:.3f} | "
        f"macro_recall={baseline_summary['macro_recall']:.3f} "
        f"macro_fp/manual={baseline_summary['macro_fp_per_manual']:.3f}"
    )
    print(
        f"Held-out mean recall={heldout_aggregate['mean_heldout_recall']:.3f}, "
        f"mean precision={heldout_aggregate['mean_heldout_precision']:.3f}, "
        f"mean fp/manual={heldout_aggregate['mean_heldout_fp_per_manual']:.3f}"
    )
    print(
        f"Pass recall folds={heldout_aggregate['n_folds_pass_recall']}/{heldout_aggregate['n_folds']}, "
        f"pass fp-order folds={heldout_aggregate['n_folds_pass_fp_same_order']}/{heldout_aggregate['n_folds']}"
    )
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()

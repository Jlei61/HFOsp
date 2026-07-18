#!/usr/bin/env python3
"""Compare legacy and current fields on the identical old activation cache.

Current-field routing is availability based and fixed before scoring:
``shared_a/shared_b`` when both frozen shared models exist, otherwise
``own_a/own_b``.  The old cache activation, event denominator and all-contact
shuffle contract are otherwise held fixed.  A legacy all-20 group is shown,
plus legacy/current results on the exact current-field overlap denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import (  # noqa: E402
    plot_paired_data_null_groups,
)
from scripts.run_topic5_tspectral_field_concordance import FIELD_ROOT, _seed  # noqa: E402
from scripts.run_topic5_unstratified_channel_scaffold_diagnostic import (  # noqa: E402
    OLD_CACHE,
    OLD_RESULT,
)
from src.topic5_template_axis_field import scorers_from_interictal_record  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    exact_name_align_matrix,
    fold_seizure_null_draws,
    jsonable,
    make_contact_permutations,
    paired_sign_flip_p,
    score_observed_bundle,
    score_permutation_matrix,
)


OUT = ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance"
PAPER = ROOT / "results/paper-ready-figure/fig3-sup-tspectral-field-concordance"
PAPER_FIGURES = PAPER / "figures"
CONTRACT = "topic5_old_cache_legacy_vs_current_shared_else_own_v1"
BASE_SEED = 20260717
MIN_CONTACTS = 6


def select_shared_else_own_scorers(
    scorers: Mapping[str, Mapping[str, object]],
) -> tuple[dict, str, str]:
    """Select one pre-declared plane without outcome-based switching."""
    if all(key in scorers for key in ("shared_a", "shared_b")):
        return ({key: scorers[key] for key in ("shared_a", "shared_b")},
                "shared", "shared_maxab")
    if all(key in scorers for key in ("own_a", "own_b")):
        return ({key: scorers[key] for key in ("own_a", "own_b")},
                "own_fallback", "own_maxab")
    raise ValueError("neither_complete_shared_nor_own_field_pair")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_summary(rows: Sequence[Mapping[str, object]], *, seed: int) -> dict:
    data = np.asarray([float(row["data"]) for row in rows], float)
    null = np.asarray([float(row["null"]) for row in rows], float)
    margin = data - null
    return {
        "n_subjects": len(rows),
        "n_seizures": int(sum(int(row["n_seizures"]) for row in rows)),
        "data_median": float(np.median(data)),
        "data_iqr_low": float(np.percentile(data, 25)),
        "data_iqr_high": float(np.percentile(data, 75)),
        "null_median": float(np.median(null)),
        "null_iqr_low": float(np.percentile(null, 25)),
        "null_iqr_high": float(np.percentile(null, 75)),
        "margin_median": float(np.median(margin)),
        "margin_iqr_low": float(np.percentile(margin, 25)),
        "margin_iqr_high": float(np.percentile(margin, 75)),
        "n_data_gt_null": int(np.sum(margin > 0)),
        "wilcoxon_one_sided_data_gt_null_p": float(
            wilcoxon(data, null, alternative="greater").pvalue
        ),
        "two_sided_subject_sign_flip_p": float(
            paired_sign_flip_p(margin, n_perm=100000, seed=seed)
        ),
    }


def _legacy_rows() -> list[dict]:
    artifact = json.loads(OLD_RESULT.read_text())
    return [{
        "subject": str(row["subject_id"]),
        "dataset": str(row["dataset"]),
        "data": float(row["real_median_abs_corr"]),
        "null": float(row["channel_null_median"]),
        "n_seizures": int(row["n_seizures"]),
        "field_plane": "legacy_selected_maxab",
    } for row in artifact["per_subject"]
        if row.get("status") == "ok"
        and row.get("real_median_abs_corr") is not None
        and row.get("channel_null_median") is not None]


def _score_current(n_perm: int, seed: int) -> tuple[list[dict], list[dict]]:
    rows, drops = [], []
    for meta_path in sorted(OLD_CACHE.glob("*.json")):
        subject = meta_path.stem
        cache_path = OLD_CACHE / f"{subject}.npz"
        field_path = FIELD_ROOT / f"{subject}.json"
        if not cache_path.exists() or not field_path.exists():
            drops.append({"subject": subject, "drop_reason": "missing_cache_or_field"})
            continue
        record = json.loads(field_path.read_text())
        try:
            all_scorers = scorers_from_interictal_record(record)
            scorers, plane, score_key = select_shared_else_own_scorers(all_scorers)
        except Exception as exc:
            drops.append({"subject": subject,
                          "drop_reason": f"field_unavailable:{type(exc).__name__}:{exc}"})
            continue
        meta = json.loads(meta_path.read_text())
        target_names = [str(v) for v in record["interictal_field"]["contact_order"]]
        event_data, event_null, used = [], [], []
        with np.load(cache_path, allow_pickle=True) as cache:
            source_names = [str(v) for v in meta.get("channels", cache["channels"].tolist())]
            for seizure_idx in [int(v) for v in meta.get("eligible_idxs", [])]:
                key = f"bb150_auc__{seizure_idx}"
                if key not in cache.files:
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "drop_reason": "missing_bb150_activation"})
                    continue
                activation = np.asarray(cache[key], float)
                aligned = exact_name_align_matrix(
                    record, source_names, activation[:, None]
                )["values"][:, 0]
                matched = np.isfinite(aligned)
                if int(matched.sum()) < MIN_CONTACTS:
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "drop_reason": f"fewer_than_6_contacts:{matched.sum()}"})
                    continue
                observed = score_observed_bundle(scorers, aligned).get(score_key)
                if observed is None or not np.isfinite(observed):
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "drop_reason": "nonfinite_observed"})
                    continue
                permutations = make_contact_permutations(
                    target_names, matched, n_perm,
                    _seed(f"old-cache-hybrid:{subject}:{seizure_idx}", seed),
                    mode="all_contact",
                )
                null = score_permutation_matrix(
                    scorers, aligned[None, :], permutations, chunk_draws=100
                )[score_key][:, 0]
                event_data.append(float(observed))
                event_null.append(null[:, None])
                used.append(seizure_idx)
        if not event_data:
            drops.append({"subject": subject, "drop_reason": "no_resolvable_events"})
            continue
        folded = fold_seizure_null_draws(event_null)[:, 0]
        rows.append({
            "subject": subject, "dataset": subject.split("_", 1)[0],
            "data": float(np.median(event_data)),
            "null": float(np.median(folded)),
            "null_p95": float(np.percentile(folded, 95)),
            "margin": float(np.median(event_data) - np.median(folded)),
            "n_seizures": len(event_data),
            "seizure_idxs": ";".join(map(str, used)),
            "field_plane": plane, "score_key": score_key,
            "n_channel_shuffle_draws": n_perm,
        })
    return rows, drops


def _plot(groups: list[dict]) -> tuple[Path, Path]:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    png = PAPER_FIGURES / "old_cache_legacy_vs_current_shared_else_own.png"
    pdf = png.with_suffix(".pdf")
    plot_paired_data_null_groups(
        groups, png, pdf, ylabel="Old-cache field concordance |r|", seed=BASE_SEED,
    )
    return png, pdf


def _write_readme(stats: pd.DataFrame) -> None:
    readme = PAPER_FIGURES / "README.md"
    existing = readme.read_text() if readme.exists() else "# Fig3 supplement figures\n"
    marker = "### old_cache_legacy_vs_current_shared_else_own.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    lines = "; ".join(
        f"{r.comparison_label}: n={r.n_subjects}, p={r.wilcoxon_one_sided_data_gt_null_p:.4g}"
        for r in stats.itertuples()
    )
    addition = f"""### old_cache_legacy_vs_current_shared_else_own.png / old_cache_legacy_vs_current_shared_else_own.pdf

三组全部使用旧 `bb150_auc` clinical-onset cache与all-contact channel shuffle。第一组是历史field的全20名；第二组把历史field限制到当前field可加载的19名；第三组在完全相同19名上使用当前冻结field，有完整shared A/B时取shared maxAB，否则取own maxAB。每个null draw均在contact层重新计算所选field的mirror与A/B max，随后先折叠seizure再进入cohort。

**关注点**：{lines}。E916当前field为`axis_not_available`，因此当前结果不能诚实写成n=20；第二、三组才是field版本的同分母比较。
"""
    readme.write_text(existing + addition)


def run(args: argparse.Namespace) -> dict:
    if args.n_perm < 1000:
        raise ValueError("n_perm must be >=1000")
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    cache_paths = sorted(OLD_CACHE.glob("*.npz"))
    hashes_before = {str(p.relative_to(ROOT)): _sha256(p) for p in cache_paths}
    legacy_all = _legacy_rows()
    current, drops = _score_current(args.n_perm, args.seed)
    current_ids = {row["subject"] for row in current}
    legacy_overlap = [row for row in legacy_all if row["subject"] in current_ids]

    comparisons = [
        ("legacy_all20", "Legacy field · all20", legacy_all),
        ("legacy_overlap", "Legacy field · overlap", legacy_overlap),
        ("current_hybrid_overlap", "Current shared/own · overlap", current),
    ]
    stats_rows, groups = [], []
    for comparison_id, label, rows in comparisons:
        summary = _paired_summary(rows, seed=_seed(comparison_id, args.seed))
        stats_rows.append({"comparison_id": comparison_id, "comparison_label": label,
                           **summary})
        groups.append({
            "label": label,
            "rows": [{"subject_id": r["subject"], "data": r["data"],
                      "null": r["null"], "n_seizures": r["n_seizures"]} for r in rows],
            "summary": {"n": len(rows)},
            "display_p": summary["wilcoxon_one_sided_data_gt_null_p"],
            "p_label": "one-sided p",
        })
    stats = pd.DataFrame(stats_rows)

    old_lookup = {row["subject"]: row for row in legacy_overlap}
    paired = []
    for row in sorted(current, key=lambda r: r["subject"]):
        old = old_lookup[row["subject"]]
        paired.append({
            "subject": row["subject"], "field_plane_current": row["field_plane"],
            "legacy_data": old["data"], "legacy_null": old["null"],
            "legacy_margin": old["data"] - old["null"],
            "current_data": row["data"], "current_null": row["null"],
            "current_margin": row["data"] - row["null"],
            "current_minus_legacy_margin": (
                row["data"] - row["null"] - (old["data"] - old["null"])
            ),
            "n_seizures": row["n_seizures"],
        })
    paired_frame = pd.DataFrame(paired)
    difference = paired_frame.current_minus_legacy_margin.to_numpy(float)
    paired_test = {
        "n_subjects": len(difference),
        "median_current_minus_legacy_margin": float(np.median(difference)),
        "wilcoxon_two_sided_p": float(wilcoxon(difference, alternative="two-sided").pvalue),
        "sign_flip_two_sided_p": float(paired_sign_flip_p(
            difference, n_perm=100000, seed=_seed("field-version-paired", args.seed)
        )),
    }

    subject_path = OUT / "old_cache_current_shared_else_own_subject.csv"
    stats_path = OUT / "old_cache_legacy_vs_current_field_cohort.csv"
    paired_path = OUT / "old_cache_legacy_vs_current_field_paired_subject.csv"
    drop_path = OUT / "old_cache_current_shared_else_own_drop_inventory.csv"
    pd.DataFrame(current).to_csv(subject_path, index=False)
    stats.to_csv(stats_path, index=False)
    paired_frame.to_csv(paired_path, index=False)
    pd.DataFrame(drops).to_csv(drop_path, index=False)
    hashes_after = {str(p.relative_to(ROOT)): _sha256(p) for p in cache_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("old cache NPZ changed")
    png, pdf = _plot(groups)
    _write_readme(stats)

    summary = {
        "contract": CONTRACT,
        "activation": "old bb150_auc cache; clinical onset 0-10 s",
        "current_field_routing": "shared_a/shared_b if complete else own_a/own_b",
        "routing_is_outcome_independent": True,
        "null": "1000 all-contact permutations; recompute mirror and selected-plane maxAB",
        "counts": {
            "legacy_all_subjects": len(legacy_all),
            "legacy_overlap_subjects": len(legacy_overlap),
            "current_subjects": len(current),
            "current_shared_subjects": sum(r["field_plane"] == "shared" for r in current),
            "current_own_fallback_subjects": sum(r["field_plane"] == "own_fallback" for r in current),
        },
        "cohort_statistics": stats.to_dict("records"),
        "paired_field_version_test": paired_test,
        "drops": drops,
        "cache_npz_unchanged": True,
        "outputs": {
            "subject": str(subject_path.relative_to(ROOT)),
            "cohort": str(stats_path.relative_to(ROOT)),
            "paired_subject": str(paired_path.relative_to(ROOT)),
            "drops": str(drop_path.relative_to(ROOT)),
            "figure_png": str(png.relative_to(ROOT)),
            "figure_pdf": str(pdf.relative_to(ROOT)),
        },
    }
    summary_path = OUT / "old_cache_legacy_vs_current_field_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    for path in (subject_path, stats_path, paired_path, summary_path):
        (PAPER / path.name).write_text(path.read_text())
    print(stats.to_string(index=False), flush=True)
    print(json.dumps(paired_test, indent=2), flush=True)
    print(f"[done] {png}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

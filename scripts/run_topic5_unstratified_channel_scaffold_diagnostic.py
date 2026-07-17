#!/usr/bin/env python3
"""Diagnose the old unstratified channel-shuffle scaffold conclusion.

This is a controlled bridge analysis, not a replacement for the registered
phenotype-matched analysis.  It produces two key controls:

1. Re-score the historical all-eligible, onset-aligned 1--150 Hz activation
   cache with the current frozen ``template_propagation_axis_v2`` fields.
2. Re-extract exact 1--150 Hz energy for every accepted T_spectral seizure,
   ignore phenotype labels, and compare current own maxAB with an all-contact
   channel-shuffle null at the six pre-specified fixed windows.

Every seizure is folded to one subject value before cohort inference.  Each
shuffle draw recomputes mirror choice and TA/TB maxAB from the contact-level
activation, rather than shuffling an already-computed similarity statistic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
from scripts.run_topic5_tspectral_field_concordance import (  # noqa: E402
    FIELD_ROOT,
    FIXED_WINDOW_ORDER,
    PAPER_FIGURES,
    _cache_npz_paths,
    _field_quality,
    _fixed_field_subject_rows,
    _hash_manifest,
    _load_subject_caches,
    _process_event,
    _seed,
)
from src.topic5_template_axis_field import scorers_from_interictal_record  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    exact_name_align_matrix,
    fixed_window_sign_flip_maxt,
    fold_seizure_null_draws,
    jsonable,
    make_contact_permutations,
    paired_sign_flip_p,
    score_observed_bundle,
    score_permutation_matrix,
)


OUT = ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance"
OLD_CACHE = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
OLD_RESULT = (ROOT / "results/topic5_ictal_recruitment/axis_alignment/"
              "axis_alignment_broadband150_max_ab_B1000.json")
CHECKPOINT_DIR = OUT / "per_subject/unstratified_channel_scaffold"
WINDOW_LABEL = {
    "distal": "Distal baseline",
    "pre20": "Pre −20–0 s",
    "pre10": "Pre −10–0 s",
    "post10": "Post 0–10 s",
    "post20": "Post 0–20 s",
    "late20_30": "Late 20–30 s",
}
CONTRACT = "topic5_unstratified_channel_scaffold_diagnostic_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_summary(rows: Sequence[Mapping[str, object]], *, seed: int) -> dict:
    data = np.asarray([float(row["data"]) for row in rows], float)
    null = np.asarray([float(row["null"]) for row in rows], float)
    delta = data - null
    if len(delta) == 0:
        raise ValueError("empty paired comparison")
    return {
        "n_subjects": int(len(delta)),
        "n_seizures": int(sum(int(row.get("n_seizures", 0)) for row in rows)),
        "data_median": float(np.median(data)),
        "data_iqr_low": float(np.percentile(data, 25)),
        "data_iqr_high": float(np.percentile(data, 75)),
        "null_median": float(np.median(null)),
        "null_iqr_low": float(np.percentile(null, 25)),
        "null_iqr_high": float(np.percentile(null, 75)),
        "margin_mean": float(np.mean(delta)),
        "margin_median": float(np.median(delta)),
        "n_data_gt_null": int(np.sum(delta > 0)),
        "wilcoxon_one_sided_data_gt_null_p": float(
            wilcoxon(data, null, alternative="greater").pvalue
        ),
        "two_sided_subject_sign_flip_p": float(
            paired_sign_flip_p(delta, n_perm=100000, seed=seed)
        ),
    }


def _old_historical_rows() -> list[dict]:
    artifact = json.loads(OLD_RESULT.read_text())
    return [
        {
            "subject_id": str(row["subject_id"]),
            "dataset": str(row["dataset"]),
            "data": float(row["real_median_abs_corr"]),
            "null": float(row["channel_null_median"]),
            "n_seizures": int(row["n_seizures"]),
        }
        for row in artifact["per_subject"]
        if row.get("status") == "ok"
        and row.get("real_median_abs_corr") is not None
        and row.get("channel_null_median") is not None
    ]


def _rescore_old_cache_current_axis(n_perm: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold the old activation/event/window fixed and change only field/scorer."""
    subject_rows, drops = [], []
    for meta_path in sorted(OLD_CACHE.glob("*.json")):
        subject = meta_path.stem
        field_path = FIELD_ROOT / f"{subject}.json"
        npz_path = OLD_CACHE / f"{subject}.npz"
        if not field_path.exists() or not npz_path.exists():
            drops.append({"subject": subject, "reason": "missing_field_or_cache"})
            continue
        field_record = json.loads(field_path.read_text())
        try:
            scorers = scorers_from_interictal_record(field_record)
        except Exception as exc:
            drops.append({"subject": subject,
                          "reason": f"field_unavailable:{type(exc).__name__}:{exc}"})
            continue
        meta = json.loads(meta_path.read_text())
        with np.load(npz_path, allow_pickle=True) as data:
            names = [str(value) for value in meta.get("channels", data["channels"].tolist())]
            target_names = [str(value) for value in
                            field_record["interictal_field"]["contact_order"]]
            event_obs, event_null, used = [], [], []
            for seizure_idx in [int(value) for value in meta.get("eligible_idxs", [])]:
                key = f"bb150_auc__{seizure_idx}"
                if key not in data.files:
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "reason": "missing_bb150_activation"})
                    continue
                activation = np.asarray(data[key], float)
                if activation.shape != (len(names),):
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "reason": "activation_channel_shape_mismatch"})
                    continue
                aligned = exact_name_align_matrix(
                    field_record, names, activation[:, None]
                )["values"][:, 0]
                matched = np.isfinite(aligned)
                if int(matched.sum()) < 6:
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "reason": f"fewer_than_6_finite_contacts:{matched.sum()}"})
                    continue
                observed = score_observed_bundle(scorers, aligned).get("own_maxab")
                if observed is None or not np.isfinite(observed):
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "reason": "nonfinite_observed_maxab"})
                    continue
                permutations = make_contact_permutations(
                    target_names, matched, n_perm,
                    _seed(f"old-cache-current-axis:{subject}:{seizure_idx}", seed),
                    mode="all_contact",
                )
                null = score_permutation_matrix(
                    scorers, aligned[None, :], permutations, chunk_draws=100
                )["own_maxab"][:, 0]
                event_obs.append(float(observed))
                event_null.append(null[:, None])
                used.append(seizure_idx)
        if not event_obs:
            drops.append({"subject": subject, "reason": "no_resolvable_seizure"})
            continue
        folded = fold_seizure_null_draws(event_null)[:, 0]
        subject_rows.append({
            "subject_id": subject,
            "dataset": subject.split("_", 1)[0],
            "data": float(np.median(event_obs)),
            "null": float(np.median(folded)),
            "null_p95": float(np.percentile(folded, 95)),
            "n_seizures": int(len(event_obs)),
            "seizure_idxs": ";".join(map(str, used)),
            "n_channel_shuffle_draws": int(n_perm),
        })
    return subject_rows, drops


def _checkpoint_valid(path: Path, *, n_perm: int, seed: int,
                      cache_hash: str, field_hash: str) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    return bool(
        payload.get("contract") == CONTRACT
        and payload.get("n_perm") == n_perm
        and payload.get("seed") == seed
        and payload.get("cache_sha256") == cache_hash
        and payload.get("field_sha256") == field_hash
    )


def _run_current_unstratified(n_perm: int, seed: int,
                              *, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All accepted T_spectral seizures, one exact 1--150 Hz readout each."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    subject_rows, inventory = [], []
    cache_records = _load_subject_caches()
    for number, (cache_root, sidecar_path, meta, selectors) in enumerate(cache_records, 1):
        subject = str(meta["subject"])
        dataset = subject.split("_", 1)[0]
        accepted = sorted(int(value) for value in selectors["accepted"])
        if not accepted:
            continue
        field_path = FIELD_ROOT / f"{subject}.json"
        cache_path = cache_root / f"{subject}.npz"
        cache_hash = _sha256(cache_path)
        field_hash = _sha256(field_path) if field_path.exists() else "missing"
        checkpoint = CHECKPOINT_DIR / f"{subject}.json"
        if resume and _checkpoint_valid(
                checkpoint, n_perm=n_perm, seed=seed,
                cache_hash=cache_hash, field_hash=field_hash):
            payload = json.loads(checkpoint.read_text())
            subject_rows.extend(payload.get("subject_rows", []))
            inventory.extend(payload.get("event_inventory", []))
            print(f"[{number:02d}/{len(cache_records):02d}] {subject}: resume", flush=True)
            continue
        print(f"[{number:02d}/{len(cache_records):02d}] {subject}: "
              f"{len(accepted)} accepted seizures", flush=True)
        local_inventory = [
            {"dataset": dataset, "subject": subject, "seizure_idx": idx,
             "status": "pending", "readout": "broadband_1_150",
             "phenotype_ignored_for_inclusion": True}
            for idx in accepted
        ]
        lookup = {int(row["seizure_idx"]): row for row in local_inventory}
        local_rows = []
        events = []
        if not field_path.exists():
            for row in local_inventory:
                row.update(status="drop", drop_reason="missing_axis_or_field")
        else:
            field_record = json.loads(field_path.read_text())
            quality = _field_quality(field_record)
            try:
                scorers = scorers_from_interictal_record(field_record)
            except Exception as exc:
                reason = f"field_unavailable:{type(exc).__name__}:{exc}"
                for row in local_inventory:
                    row.update(status="drop", drop_reason=reason)
            else:
                for event_number, seizure_idx in enumerate(accepted, 1):
                    print(f"  seizure {seizure_idx} [{event_number}/{len(accepted)}]",
                          flush=True)
                    try:
                        bands = _process_event(
                            subject, dataset, seizure_idx,
                            "unstratified_all_accepted", meta, field_record,
                            scorers, quality, n_perm, seed,
                            band_keys_override=["broadband_1_150"],
                        )
                        events.append({
                            "subject": subject, "dataset": dataset,
                            "seizure_idx": seizure_idx,
                            "phenotype": "unstratified_all_accepted",
                            "bands": bands,
                        })
                        lookup[seizure_idx]["status"] = "included"
                    except Exception as exc:
                        lookup[seizure_idx].update(
                            status="drop",
                            drop_reason=f"{type(exc).__name__}:{exc}",
                        )
                        print(f"    DROP {type(exc).__name__}: {exc}", flush=True)
                if events:
                    local_rows, _ = _fixed_field_subject_rows(
                        subject, dataset, quality, events
                    )
                    for row in local_rows:
                        row["readout_family"] = "unstratified_all_accepted_broadband_1_150"
                        row["readout_contract"] = (
                            "all accepted T_spectral seizures -> exact 1-150 Hz"
                        )
                        row["n_broadband_seizures"] = int(row["n_seizures"])
                        row["n_gamma_seizures"] = 0
        payload = {
            "contract": CONTRACT, "subject": subject, "n_perm": n_perm,
            "seed": seed, "cache_sha256": cache_hash, "field_sha256": field_hash,
            "subject_rows": local_rows, "event_inventory": local_inventory,
        }
        checkpoint.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n")
        subject_rows.extend(local_rows)
        inventory.extend(local_inventory)
    return pd.DataFrame(subject_rows), pd.DataFrame(inventory)


def _unstratified_window_statistics(frame: pd.DataFrame, *, n_perm: int,
                                    seed: int) -> pd.DataFrame:
    source = "own_all_contact_delta_null_median"
    pivot = frame.pivot(index="subject", columns="fixed_window", values=source)
    pivot = pivot.reindex(columns=FIXED_WINDOW_ORDER).dropna(how="any")
    inference = fixed_window_sign_flip_maxt(
        pivot.to_numpy(float), n_perm=n_perm, seed=seed
    )
    rows = []
    for j, window in enumerate(FIXED_WINDOW_ORDER):
        window_frame = frame[frame.fixed_window == window].set_index("subject").loc[pivot.index]
        paired = [
            {"subject_id": subject,
             "data": float(window_frame.loc[subject, "own_maxab"]),
             "null": float(window_frame.loc[subject,
                                             "own_all_contact_null_median_folded"]),
             "n_seizures": int(window_frame.loc[subject, "n_seizures"])}
            for subject in pivot.index
        ]
        summary = _paired_summary(paired, seed=_seed(f"unstrat:{window}", seed))
        rows.append({
            "fixed_window": window,
            "window_start_sec": float(np.nanmedian(window_frame.window_start_sec)),
            "window_end_sec": float(np.nanmedian(window_frame.window_end_sec)),
            **summary,
            "two_sided_sign_flip_raw_p": float(inference["raw_p"][j]),
            "two_sided_sign_flip_maxt_p": float(inference["maxt_p"][j]),
            "n_sign_permutations_maxt": int(inference["n_permutations"]),
        })
    return pd.DataFrame(rows)


def _high_quality_collinear_mask(frame: pd.DataFrame) -> pd.Series:
    """Pre-existing strict 2-D quality plus same/reversed A/B collinearity."""
    return (
        frame["axis_quality_tier"].astype(str).eq("strict_2d")
        & frame["axis_relation"].astype(str).isin(("same", "reversed"))
    )


def _strict_reversed_mask(frame: pd.DataFrame) -> pd.Series:
    """The previously discussed high-quality reverse-collinear stratum."""
    return (
        frame["axis_quality_tier"].astype(str).eq("strict_2d")
        & frame["axis_relation"].astype(str).eq("reversed")
    )


def _strict_reversed_subject_ids() -> set[str]:
    subjects = set()
    for path in sorted(FIELD_ROOT.glob("*.json")):
        quality = _field_quality(json.loads(path.read_text()))
        if (quality.get("axis_quality_tier") == "strict_2d"
                and quality.get("axis_relation") == "reversed"):
            subjects.add(path.stem)
    return subjects


def _plot_window_board(subject: pd.DataFrame, stats: pd.DataFrame, *,
                       stem: str = "unstratified_channel_scaffold_by_window",
                       ylabel: str = "Unstratified BB 1–150 Hz field concordance |r|") -> None:
    groups = []
    stat_lookup = stats.set_index("fixed_window")
    for window in FIXED_WINDOW_ORDER:
        window_frame = subject[subject.fixed_window == window].sort_values("subject")
        rows = [
            {"subject_id": str(row.subject), "data": float(row.own_maxab),
             "null": float(row.own_all_contact_null_median_folded),
             "n_seizures": int(row.n_seizures)}
            for row in window_frame.itertuples()
        ]
        groups.append({
            "label": WINDOW_LABEL[window], "rows": rows,
            "summary": {"n": len(rows)},
            "display_p": float(stat_lookup.loc[window,
                                                "two_sided_sign_flip_maxt_p"]),
            "p_label": "maxT p",
        })
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    output = PAPER_FIGURES / f"{stem}.png"
    plot_paired_data_null_groups(
        groups, output, output.with_suffix(".pdf"),
        ylabel=ylabel,
        seed=20260716,
    )


def _plot_contract_decomposition(
        groups: list[dict], *,
        stem: str = "channel_scaffold_contract_decomposition") -> None:
    output = PAPER_FIGURES / f"{stem}.png"
    plot_paired_data_null_groups(
        groups, output, output.with_suffix(".pdf"),
        ylabel="Field concordance |r|", seed=20260717,
    )


def _write_readme(comparison: pd.DataFrame) -> None:
    readme = PAPER_FIGURES / "README.md"
    existing = readme.read_text() if readme.exists() else "# Fig3 supplement figures\n"
    marker = "### unstratified_channel_scaffold_by_window.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    post = comparison[comparison.comparison_id.str.contains("post10")]
    lines = [
        f"{row.comparison_label}: n={row.n_subjects}, "
        f"one-sided Wilcoxon p={row.wilcoxon_one_sided_data_gt_null_p:.4g}"
        for row in post.itertuples()
    ]
    addition = f"""### unstratified_channel_scaffold_by_window.png / unstratified_channel_scaffold_by_window.pdf

严格复用既有 Fig3 `Data`–`Null` violin、box、subject 配对线和显著性括号。每名患者先对其全部 accepted `T_spectral` seizure 取中位数；所有 seizure 均使用精确 1–150 Hz，不按频谱表型分组，Null 为 all-contact channel shuffle。图中六个固定窗口的 p 值是 subject-level two-sided sign-flip maxT，同时校正六个窗口。

**关注点**：这是对旧“粗 scaffold”结论的当前轴诊断，不替代正式 phenotype-matched / within-shaft 主分析。

### unstratified_channel_scaffold_high_quality_collinear_by_window.png / unstratified_channel_scaffold_high_quality_collinear_by_window.pdf

与上一张图使用完全相同的全部 accepted `T_spectral` seizure、精确 1–150 Hz、own maxAB、subject-first folding 和 all-contact channel-shuffle 合同，只保留预先定义为 `strict_2d` 且 A/B relation 属于 `same` 或 `reversed` 的患者。绘图仍严格复用既有 Fig3 成对 Data–Null painter，六窗口 p 值在这个固定质量子集中重新执行 subject-level two-sided sign-flip maxT。

**关注点**：这是预先存在轴质量与共线性联合分层，不按当前 concordance 或 p 值筛选；`same` 与 `reversed` 均属于共线，未从两者中选择结果更强的一组。

### unstratified_channel_scaffold_strict_reversed_by_window.png / unstratified_channel_scaffold_strict_reversed_by_window.pdf

只保留 `axis_quality_tier == strict_2d` 且 `axis_relation == reversed` 的严格反向共线患者，其他 readout、subject-first folding、channel-shuffle 和六窗口合同与不分型全 cohort 图一致。当前精确 1–150 Hz `T_spectral` 分母中共有4名患者；sign-flip 使用全部16种 subject 符号组合精确枚举。

**关注点**：这才是“高质量反向共线”子集；不要与上一张同时包含 `same` 的宽共线分层混称。

### strict_reversed_channel_scaffold_contract_decomposition.png / strict_reversed_channel_scaffold_contract_decomposition.pdf

在严格反向共线层内并列旧 field+旧 onset、当前 field+相同旧 activation、当前 field+全部 accepted `T_spectral` 和当前 field+phenotype-matched `T_spectral`。全部使用相同的 Data–channel-null median subject 配对图及旧图的单侧 Wilcoxon显示语法。

**关注点**：旧 activation 两组均含5名患者，包括当前精确1–150 Hz因 Nyquist 不足而整名退出的 E583；当前 `T_spectral` 两组仅余4名，因此它们不是同一分母。

### channel_scaffold_contract_decomposition.png / channel_scaffold_contract_decomposition.pdf

同一绘图函数下并列历史旧轴结果、相同旧 activation 换当前冻结 TA/TB field、当前 `T_spectral` 全 accepted seizure 不分型结果，以及当前 phenotype-matched 结果。前两组固定旧的临床/EEG onset 0–10 s activation，后两组固定当前 `T_spectral` 0–10 s；全部使用 channel-shuffle null。

**关注点**：{'; '.join(lines)}。不同组的分母和时间零点不同，因此用于定位结果变化来自哪项合同，不作为四组之间的直接效应量比较。
"""
    readme.write_text(existing + addition)


def run(args: argparse.Namespace) -> dict:
    if args.n_perm < 1000:
        raise ValueError("channel scaffold diagnostic requires at least 1000 shuffles")
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    tspectral_paths = _cache_npz_paths()
    hashes_before = _hash_manifest(tspectral_paths)
    old_historical = _old_historical_rows()
    old_current, old_drops = _rescore_old_cache_current_axis(
        args.n_perm, args.seed
    )
    unstratified, inventory = _run_current_unstratified(
        args.n_perm, args.seed, resume=args.resume
    )
    if unstratified.empty:
        raise RuntimeError("unstratified current-axis analysis produced no subjects")
    window_stats = _unstratified_window_statistics(
        unstratified, n_perm=args.cohort_permutations, seed=args.seed
    )
    high_quality_collinear = unstratified[
        _high_quality_collinear_mask(unstratified)
    ].copy()
    if high_quality_collinear.empty:
        raise RuntimeError("no strict-2D collinear subjects in unstratified analysis")
    high_quality_collinear_stats = _unstratified_window_statistics(
        high_quality_collinear, n_perm=args.cohort_permutations,
        seed=_seed("high-quality-collinear", args.seed),
    )
    strict_reversed = unstratified[_strict_reversed_mask(unstratified)].copy()
    if strict_reversed.empty:
        raise RuntimeError("no strict-2D reversed subjects in unstratified analysis")
    strict_reversed_stats = _unstratified_window_statistics(
        strict_reversed, n_perm=args.cohort_permutations,
        seed=_seed("strict-reversed", args.seed),
    )

    matched_path = OUT / "phenotype_matched_fixed_window_subject.csv"
    matched = pd.read_csv(matched_path)
    matched_post = matched[matched.fixed_window == "post10"].sort_values("subject")
    matched_rows = [
        {"subject_id": str(row.subject), "data": float(row.own_maxab),
         "null": float(row.own_all_contact_null_median_folded),
         "n_seizures": int(row.n_seizures)}
        for row in matched_post.itertuples()
    ]
    unstrat_post = unstratified[
        unstratified.fixed_window == "post10"
    ].sort_values("subject")
    unstrat_rows = [
        {"subject_id": str(row.subject), "data": float(row.own_maxab),
         "null": float(row.own_all_contact_null_median_folded),
         "n_seizures": int(row.n_seizures)}
        for row in unstrat_post.itertuples()
    ]

    current_subjects = {row["subject_id"] for row in old_current}
    historical_intersection = [row for row in old_historical
                               if row["subject_id"] in current_subjects]
    comparison_specs = [
        ("historical_old_axis_all20_post10", "Legacy field · old onset",
         old_historical),
        ("historical_old_axis_current19_post10", "Old field · overlap set",
         historical_intersection),
        ("old_activation_current_axis_post10", "Current own · old onset",
         old_current),
        ("current_unstratified_post10", "Current own · all Tₛ",
         unstrat_rows),
        ("current_matched_post10", "Current own · matched Tₛ",
         matched_rows),
    ]
    comparison_rows, plot_groups = [], []
    for comparison_id, label, rows in comparison_specs:
        summary = _paired_summary(rows, seed=_seed(comparison_id, args.seed))
        comparison_rows.append({
            "comparison_id": comparison_id,
            "comparison_label": label.replace("\n", " / "),
            **summary,
        })
        if comparison_id != "historical_old_axis_current19_post10":
            plot_groups.append({
                "label": label, "rows": rows,
                "summary": {
                    "n": summary["n_subjects"],
                    "wilcoxon_p_data_gt_null_median": summary[
                        "wilcoxon_one_sided_data_gt_null_p"
                    ],
                },
                "display_p": summary["wilcoxon_one_sided_data_gt_null_p"],
                "p_label": "one-sided p",
            })
    comparison = pd.DataFrame(comparison_rows)

    strict_reversed_ids = _strict_reversed_subject_ids()
    strict_specs = [
        ("strict_reversed_historical_old_axis_post10", "Legacy field · old onset",
         [row for row in old_historical if row["subject_id"] in strict_reversed_ids]),
        ("strict_reversed_old_activation_current_axis_post10", "Current own · old onset",
         [row for row in old_current if row["subject_id"] in strict_reversed_ids]),
        ("strict_reversed_current_unstratified_post10", "Current own · all Tₛ",
         [row for row in unstrat_rows if row["subject_id"] in strict_reversed_ids]),
        ("strict_reversed_current_matched_post10", "Current own · matched Tₛ",
         [row for row in matched_rows if row["subject_id"] in strict_reversed_ids]),
    ]
    strict_comparison_rows, strict_plot_groups = [], []
    for comparison_id, label, rows in strict_specs:
        summary = _paired_summary(rows, seed=_seed(comparison_id, args.seed))
        strict_comparison_rows.append({
            "comparison_id": comparison_id, "comparison_label": label,
            **summary,
        })
        strict_plot_groups.append({
            "label": label, "rows": rows,
            "summary": {
                "n": summary["n_subjects"],
                "wilcoxon_p_data_gt_null_median": summary[
                    "wilcoxon_one_sided_data_gt_null_p"
                ],
            },
            "display_p": summary["wilcoxon_one_sided_data_gt_null_p"],
            "p_label": "one-sided p",
        })
    strict_comparison = pd.DataFrame(strict_comparison_rows)

    hashes_after = _hash_manifest(tspectral_paths)
    invariant = hashes_before == hashes_after
    if not invariant:
        raise RuntimeError("T_spectral cache NPZ hash changed during diagnostic")

    tables = {
        "unstratified_channel_scaffold_subject_by_window.csv": unstratified,
        "unstratified_channel_scaffold_cohort_by_window.csv": window_stats,
        "unstratified_channel_scaffold_high_quality_collinear_subject_by_window.csv":
            high_quality_collinear,
        "unstratified_channel_scaffold_high_quality_collinear_cohort_by_window.csv":
            high_quality_collinear_stats,
        "unstratified_channel_scaffold_strict_reversed_subject_by_window.csv":
            strict_reversed,
        "unstratified_channel_scaffold_strict_reversed_cohort_by_window.csv":
            strict_reversed_stats,
        "unstratified_channel_scaffold_event_inventory.csv": inventory,
        "old_activation_current_axis_channel_subject.csv": pd.DataFrame(old_current),
        "old_activation_current_axis_channel_drops.csv": pd.DataFrame(old_drops),
        "channel_scaffold_contract_decomposition.csv": comparison,
        "strict_reversed_channel_scaffold_contract_decomposition.csv":
            strict_comparison,
    }
    for filename, table in tables.items():
        table.to_csv(OUT / filename, index=False)
    PAPER_FIGURES.parent.mkdir(parents=True, exist_ok=True)
    for filename in tables:
        shutil.copy2(OUT / filename, PAPER_FIGURES.parent / filename)
    _plot_window_board(unstratified, window_stats)
    _plot_window_board(
        high_quality_collinear, high_quality_collinear_stats,
        stem="unstratified_channel_scaffold_high_quality_collinear_by_window",
        ylabel="Collinear-field concordance |r|",
    )
    _plot_window_board(
        strict_reversed, strict_reversed_stats,
        stem="unstratified_channel_scaffold_strict_reversed_by_window",
        ylabel="Strict reversed-field concordance |r|",
    )
    _plot_contract_decomposition(plot_groups)
    _plot_contract_decomposition(
        strict_plot_groups,
        stem="strict_reversed_channel_scaffold_contract_decomposition",
    )
    _write_readme(comparison)

    result = {
        "contract": CONTRACT,
        "n_channel_shuffles": args.n_perm,
        "n_cohort_sign_permutations": args.cohort_permutations,
        "current_axis": "template_propagation_axis_v2 positive_early_to_late",
        "unstratified_current_contract": (
            "all accepted T_spectral seizures; exact 1-150 Hz; delta energy; "
            "own maxAB; subject-first; all-contact channel shuffle"
        ),
        "historical_bridge_contract": (
            "all old eligible seizures; old onset-aligned bb150_auc 0-10 s; "
            "current frozen own maxAB; subject-first; all-contact channel shuffle"
        ),
        "tspectral_cache_npz_unchanged": invariant,
        "counts": {
            "unstratified_subjects": int(unstratified.subject.nunique()),
            "unstratified_included_events": int(
                (inventory.status == "included").sum()),
            "unstratified_dropped_events": int((inventory.status == "drop").sum()),
            "old_activation_current_axis_subjects": len(old_current),
            "old_activation_current_axis_events": int(
                sum(row["n_seizures"] for row in old_current)),
            "high_quality_collinear_subjects": int(
                high_quality_collinear.subject.nunique()),
            "high_quality_collinear_events": int(
                high_quality_collinear.loc[
                    high_quality_collinear.fixed_window == "post10", "n_seizures"
                ].sum()),
            "strict_reversed_current_subjects": int(
                strict_reversed.subject.nunique()),
            "strict_reversed_current_events": int(
                strict_reversed.loc[
                    strict_reversed.fixed_window == "post10", "n_seizures"
                ].sum()),
        },
        "comparison": comparison.to_dict("records"),
        "unstratified_window_statistics": window_stats.to_dict("records"),
        "high_quality_collinear_definition": (
            "axis_quality_tier == strict_2d and axis_relation in {same,reversed}"
        ),
        "high_quality_collinear_window_statistics":
            high_quality_collinear_stats.to_dict("records"),
        "strict_reversed_definition": (
            "axis_quality_tier == strict_2d and axis_relation == reversed"
        ),
        "strict_reversed_window_statistics": strict_reversed_stats.to_dict("records"),
        "strict_reversed_contract_comparison": strict_comparison.to_dict("records"),
    }
    (OUT / "channel_scaffold_diagnostic_summary.json").write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--cohort-permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

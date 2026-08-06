#!/usr/bin/env python3
"""Summarize full-cohort interictal and target-eligible ictal consistency.

Interictal consistency is the correlation between pairwise contact-precedence
probabilities in free model rollouts and untouched test20 events.  Early-ictal
consistency is the frozen model-field max absolute Spearman correlation with
the same patient's clinical-onset [0, 10] s, 1--150 Hz energy field.  The two
metrics are kept separate because only the latter requires an ictal target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_topic5_history_conditioned_field_v0_4 import (  # noqa: E402
    _exact_signed_rank,
)


MODEL = "full_history_gru"
CONTROL = "rank_shuffle_gru"
DEVELOPMENT_SUBJECTS = {
    "epilepsiae_1073",
    "epilepsiae_1146",
    "yuquan_chenziyang",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_median_ci(values: np.ndarray, *, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True)
    return list(map(float, np.quantile(np.median(draws, axis=1), [0.025, 0.975])))


def _paired_summary(values: np.ndarray, *, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    positive = int(np.sum(values > 1e-9))
    negative = int(np.sum(values < -1e-9))
    tied = int(len(values) - positive - negative)
    nonzero = values[np.abs(values) > 1e-9]
    if len(nonzero) <= 20:
        test = _exact_signed_rank(values, tolerance=1e-9)
        p_value = float(test["p_two_sided_exact"])
        method = "exact sign-flip distribution of signed ranks"
    elif len(nonzero):
        p_value = float(
            wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto").pvalue
        )
        method = "two-sided Wilcoxon signed-rank, scipy auto method"
    else:
        p_value = 1.0
        method = "all differences tied"
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "bootstrap_ci95": _bootstrap_median_ci(values, seed=seed),
        "n_positive": positive,
        "n_negative": negative,
        "n_tied": tied,
        "p_two_sided": p_value,
        "test_method": method,
    }


def _permutation_spearman(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    draws: int = 100_000,
) -> dict:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    observed = float(spearmanr(left, right).statistic)
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator <= 0:
        raise ValueError("constant vector in cross-metric association")
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < int(draws):
        current = min(5000, int(draws) - completed)
        permutations = np.stack([rng.permutation(len(right_rank)) for _ in range(current)])
        values = (right_rank[permutations] @ left_rank) / denominator
        extreme += int(np.count_nonzero(np.abs(values) >= abs(observed) - 1e-12))
        completed += current
    return {
        "n": int(len(left)),
        "spearman_rho": observed,
        "permutation_p_two_sided": float((extreme + 1) / (draws + 1)),
        "permutations": int(draws),
    }


def _load_unit_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("units/*/*/seed_*/DONE.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE" or payload.get("model") not in {MODEL, CONTROL}:
            continue
        rows.append(
            {
                "subject": str(payload["subject"]),
                "dataset": str(payload["subject"]).split("_", 1)[0],
                "model": str(payload["model"]),
                "seed": int(payload["seed"]),
                "n_contacts": int(payload["n_contacts"]),
                "n_test_events": int(payload["n_events"]["test20"]),
                "heldout_event_nll": float(payload["test"]["heldout_event_nll"]),
                "precedence_correlation": float(payload["rollout_errors"]["precedence_correlation"]),
                "precedence_mae": float(payload["rollout_errors"]["precedence_mae"]),
                "rank_wasserstein": float(payload["rollout_errors"]["rank_wasserstein"]),
                "participation_mae": float(payload["rollout_errors"]["participation_mae"]),
                "source_root": str(root.relative_to(ROOT)),
                "done_path": str(path.relative_to(ROOT)),
                "done_sha256": _sha256(path),
                "ictal_target_read": bool(payload["ictal_target_read"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-config",
        type=Path,
        default=ROOT / "config/topic5_patient_specific_interictal_consistency_v0_1.yaml",
    )
    parser.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "config/topic5_patient_specific_target_free_rnn_bridge_v0_1.yaml",
    )
    args = parser.parse_args()
    extension_config = yaml.safe_load(args.extension_config.read_text(encoding="utf-8"))
    bridge_config = yaml.safe_load(args.bridge_config.read_text(encoding="utf-8"))
    for key in ("models", "training", "readout", "split"):
        if extension_config[key] != bridge_config[key]:
            raise RuntimeError(f"extension/bridge {key} contract mismatch")

    output = ROOT / extension_config["output_root"]
    bridge = ROOT / bridge_config["output_root"]
    rows = _load_unit_rows(bridge) + _load_unit_rows(output)
    seed_frame = pd.DataFrame(rows)
    if seed_frame.empty:
        raise RuntimeError("no complete interictal units")
    duplicated = seed_frame.duplicated(["subject", "model", "seed"], keep=False)
    if bool(duplicated.any()):
        raise RuntimeError("duplicate subject/model/seed across source roots")
    expected = 34 * 2 * 3
    if len(seed_frame) != expected:
        raise RuntimeError(f"expected {expected} complete units, found {len(seed_frame)}")
    if bool(seed_frame.ictal_target_read.any()):
        raise RuntimeError("an interictal unit read an ictal target")

    patient = (
        seed_frame.groupby(["subject", "dataset", "model"], as_index=False)
        .median(numeric_only=True)
    )
    if patient.subject.nunique() != 34:
        raise RuntimeError("full interictal cohort is not 34 patients")
    wide = patient.pivot(index="subject", columns="model", values="precedence_correlation")
    wide_nll = patient.pivot(index="subject", columns="model", values="heldout_event_nll")
    subject_meta = patient.loc[patient.model == MODEL, ["subject", "dataset", "n_contacts", "n_test_events"]].set_index("subject")
    interictal = subject_meta.join(
        wide.rename(columns={
            MODEL: "interictal_rnn_precedence_r",
            CONTROL: "interictal_rank_shuffle_precedence_r",
        })
    ).join(
        wide_nll.rename(columns={
            MODEL: "interictal_rnn_test_nll",
            CONTROL: "interictal_rank_shuffle_test_nll",
        })
    )
    interictal["interictal_consistency_gain"] = (
        interictal.interictal_rnn_precedence_r
        - interictal.interictal_rank_shuffle_precedence_r
    )
    interictal["interictal_nll_gain"] = (
        interictal.interictal_rank_shuffle_test_nll
        - interictal.interictal_rnn_test_nll
    )
    interictal["development_subject"] = interictal.index.isin(DEVELOPMENT_SUBJECTS)

    ictal_path = bridge / "early_ictal_patient_metrics.csv"
    ictal_raw = pd.read_csv(ictal_path)
    ictal = (
        ictal_raw.loc[(ictal_raw.model == MODEL) & (ictal_raw.band == "1_150")]
        .set_index("subject")
        [[
            "n_seizures",
            "observed_max_abs_rho",
            "all_contact_null_median",
            "all_contact_margin",
            "all_contact_p",
            "within_shaft_null_median",
            "within_shaft_margin",
            "within_shaft_p",
            "development_supportive",
        ]]
        .rename(
            columns={
                "observed_max_abs_rho": "early_ictal_rnn_max_abs_rho",
                "all_contact_null_median": "early_ictal_all_contact_null_median",
                "all_contact_margin": "early_ictal_consistency_margin",
                "all_contact_p": "early_ictal_all_contact_p",
                "within_shaft_null_median": "early_ictal_within_shaft_null_median",
                "within_shaft_margin": "early_ictal_within_shaft_margin",
                "within_shaft_p": "early_ictal_within_shaft_p",
            }
        )
    )
    # Reconstruct the margins from their two scored components instead of
    # reusing the serialized margin column.  Several patients have exact tied
    # rational rank correlations; serializing a pre-subtracted float can break
    # those ties at ~1e-16 and alter the exact signed-rank distribution.
    ictal["early_ictal_consistency_margin"] = (
        ictal.early_ictal_rnn_max_abs_rho
        - ictal.early_ictal_all_contact_null_median
    )
    ictal["early_ictal_within_shaft_margin"] = (
        ictal.early_ictal_rnn_max_abs_rho
        - ictal.early_ictal_within_shaft_null_median
    )
    metrics = interictal.join(ictal, how="left").reset_index()
    output.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(output / "interictal_consistency_seed_metrics.csv", index=False)
    patient.to_csv(output / "interictal_consistency_patient_model_metrics.csv", index=False)
    metrics.to_csv(output / "patient_specific_consistency_metrics.csv", index=False)

    inter_all = metrics.interictal_consistency_gain.to_numpy(float)
    inter_primary = metrics.loc[~metrics.development_subject, "interictal_consistency_gain"].to_numpy(float)
    strict = metrics.loc[metrics.early_ictal_rnn_max_abs_rho.notna()].copy()
    strict_primary = strict.loc[~strict.development_supportive.astype(bool)].copy()
    statistics = {
        "status": "COMPLETE",
        "contract": extension_config["contract"],
        "metric_definitions": {
            "interictal_prediction_consistency": (
                "Pearson r across all directed contact-pair precedence probabilities: "
                "free RNN rollouts versus untouched chronological test20 events."
            ),
            "early_ictal_prediction_consistency": (
                "Patient-level median max absolute Spearman rho between frozen RNN-derived "
                "contact fields and clinical-onset [0,10] s, 1-150 Hz energy fields; "
                "candidate maximization is repeated inside each channel-shuffle null draw."
            ),
        },
        "denominators": {
            "interictal_full": 34,
            "interictal_development_excluded": int((~metrics.development_subject).sum()),
            "early_ictal_target_eligible_full": int(len(strict)),
            "early_ictal_primary": int(len(strict_primary)),
            "development_subjects": sorted(DEVELOPMENT_SUBJECTS),
        },
        "interictal": {
            "full_34_rnn_precedence_r_median": float(metrics.interictal_rnn_precedence_r.median()),
            "full_34_rank_shuffle_precedence_r_median": float(metrics.interictal_rank_shuffle_precedence_r.median()),
            "rnn_minus_rank_shuffle_full_34": _paired_summary(inter_all, seed=202608031),
            "rnn_minus_rank_shuffle_development_excluded_31": _paired_summary(inter_primary, seed=202608032),
            "by_dataset": {
                dataset: _paired_summary(
                    group.interictal_consistency_gain.to_numpy(float),
                    seed=202608040 + index,
                )
                for index, (dataset, group) in enumerate(metrics.groupby("dataset"))
            },
        },
        "early_ictal": {
            "full_16_rnn_abs_rho_median": float(strict.early_ictal_rnn_max_abs_rho.median()),
            "full_16_null_median": float(strict.early_ictal_all_contact_null_median.median()),
            "rnn_minus_all_contact_null_full_16": _paired_summary(
                strict.early_ictal_consistency_margin.to_numpy(float), seed=202608051
            ),
            "rnn_minus_all_contact_null_primary_15": _paired_summary(
                strict_primary.early_ictal_consistency_margin.to_numpy(float), seed=202608052
            ),
            "rnn_minus_within_shaft_null_primary_15": _paired_summary(
                strict_primary.early_ictal_within_shaft_margin.to_numpy(float), seed=202608053
            ),
        },
        "cross_metric_association": {
            "full_16": _permutation_spearman(
                strict.interictal_rnn_precedence_r.to_numpy(float),
                strict.early_ictal_rnn_max_abs_rho.to_numpy(float),
                seed=202608061,
            ),
            "primary_15": _permutation_spearman(
                strict_primary.interictal_rnn_precedence_r.to_numpy(float),
                strict_primary.early_ictal_rnn_max_abs_rho.to_numpy(float),
                seed=202608062,
            ),
        },
        "seals": {
            "interictal_ictal_target_read": False,
            "interictal_other_patient_events_used": False,
            "empirical_ab_used": False,
        },
        "sources": {
            "extension_config": {
                "path": str(args.extension_config.resolve().relative_to(ROOT)),
                "sha256": _sha256(args.extension_config),
            },
            "bridge_config": {
                "path": str(args.bridge_config.resolve().relative_to(ROOT)),
                "sha256": _sha256(args.bridge_config),
            },
            "early_ictal_patient_metrics": {
                "path": str(ictal_path.relative_to(ROOT)),
                "sha256": _sha256(ictal_path),
            },
            "n_done_files": int(len(seed_frame)),
        },
    }
    statistics_path = output / "PATIENT_SPECIFIC_CONSISTENCY_STATISTICS.json"
    statistics_path.write_text(
        json.dumps(statistics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    inter_formal = statistics["interictal"]["rnn_minus_rank_shuffle_development_excluded_31"]
    ictal_formal = statistics["early_ictal"]["rnn_minus_all_contact_null_primary_15"]
    association = statistics["cross_metric_association"]["primary_15"]
    report = f"""# 患者特异 RNN 两类一致性指标 v0.1

## 指标

1. **间期预测一致性**：模型自由生成与 untouched test20 中所有有向 contact pair 的先后概率相关。它直接衡量模型是否恢复患者自己的传播排序，而不是只看 NLL。
2. **发作期预测一致性**：冻结 RNN-derived contact field 与同患者 clinical-onset 后 0--10 s、1--150 Hz energy field 的 max absolute Spearman。发作 target 不进入训练；它是跨状态空间预测一致性，不是逐次发作路径预测。

## 全 cohort 统计

- 间期共 34 人；图中显示全部 34 人。排除三名 development patients 后，RNN 相对 rank-shuffle 的 precedence correlation 增量中位数为 `{inter_formal['median']:.3f}`，{inter_formal['n_positive']}/{inter_formal['n']} 为正，`P={inter_formal['p_two_sided']:.4g}`。
- 发作期 exact clinical-onset target 可用者共 16 人；图中显示全部 16 人。排除 E1146 后 primary 15 人中，RNN field 相对 all-contact channel-shuffle null 的一致性 margin 中位数为 `{ictal_formal['median']:.3f}`，{ictal_formal['n_positive']}/{ictal_formal['n']} 为正，`P={ictal_formal['p_two_sided']:.4g}`。
- 两项患者级一致性强弱在 15 人中不相关：Spearman `rho={association['spearman_rho']:.3f}`，permutation `P={association['permutation_p_two_sided']:.3f}`。因此队列层面两项均成立，不代表“间期模型拟合最好的人一定具有最强发作对应”。

## 边界

- early-ictal 分母不是 34，因为 18 人没有当前合同所需的 exact clinical-onset target；不能把缺失 target 当成阴性患者。
- 发作期一致性在 all-contact null 下成立；within-shaft sensitivity 和相对完整静态 scaffold 的增量仍未建立。
"""
    (output / "PATIENT_SPECIFIC_CONSISTENCY_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(statistics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

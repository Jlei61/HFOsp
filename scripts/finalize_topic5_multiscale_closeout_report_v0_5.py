#!/usr/bin/env python3
"""Replace the closeout report's frozen final-results block after scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_REPORT = (
    ROOT / "docs/archive/topic5/"
    "multiscale_effective_scaffold_v0_5_closeout_2026-08-14.md"
)
BEGIN = "<!-- FINAL_RESULTS_BEGIN -->"
END = "<!-- FINAL_RESULTS_END -->"


def f(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "NA"


def paired_line(label: str, result: dict) -> str:
    return (
        f"- {label}：中位 `{f(result.get('median'))}`，"
        f"{int(result.get('n_positive', 0))}/{int(result.get('n', 0))} 正向，"
        f"单侧患者级 `P={f(result.get('wilcoxon_p_greater'))}`；"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    out, report_path = args.out_root.resolve(), args.report.resolve()
    for relative in (
        "PIPELINE_COMPLETE.json", "FINAL_CLAIM_ADJUDICATION.json",
        "ATTENUATED_FIELDS_FROZEN.json", "ATTENUATION_PER_PATIENT_AUC.csv",
        "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json",
    ):
        if not (out / relative).exists():
            raise FileNotFoundError(out / relative)
    claims = json.loads((out / "FINAL_CLAIM_ADJUDICATION.json").read_text())
    early = json.loads((out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json").read_text())
    marker = json.loads((out / "ATTENUATED_FIELDS_FROZEN.json").read_text())
    auc = pd.read_csv(out / "ATTENUATION_PER_PATIENT_AUC.csv")
    auc_n = {
        target: int(group.loc[
            group.inferential_eligible.astype(str).str.lower().eq("true"), "subject"
        ].nunique())
        for target, group in auc.groupby("target")
    }
    primary = early["primary_interaction"]
    d1 = early["D1_L3_full_margin_gt_zero"]
    d2 = early["D2_L3_minus_L2m_seed_removed_signed_oracle"]
    d2_auc = early["D2_L3_added_attenuation_auc_seed_removed_gt_zero"]
    mixture = early["nonoracle_L3_mixture_margin_gt_zero"]
    l3_suffix = early["L3_minus_suffix_full_signed_oracle"]
    template = early["template_oracle_margin_gt_zero"]
    claim = claims["claims"]
    status_word = lambda value: "支持" if value else "未支持"
    block = f"""### 8.1 Stage F target-free 扰动与冻结

- 504/504 arm-target attenuation units 完成；matched-local control 的严格患者级可推断人数为
  `{int(marker.get('eligible_local_control_patients', 0))}`；
- 四类 attenuation AUC 的 inferential denominator：
  `L1={auc_n.get('L1_ADDED', 0)}`、`L2m={auc_n.get('L2M_ADDED', 0)}`、
  `L3={auc_n.get('L3_ADDED', 0)}`、`L3-matched-local={auc_n.get('L3_MATCHED_LOCAL', 0)}`；
- intact、template、mixture、attenuated、gain-adjusted fields 和 synchronized null maps 均在
  target authorization 前冻结；本节所有 perturbation fields 都没有读取 early-ictal values。

### 8.2 Locked internal early-ictal benchmark

目标固定为 17 位患者、167 次发作、clinical onset 后 0–10 s、1–150 Hz broadband energy；
这是项目内已经看过 target 后锁定的 mechanistic follow-up，不是 independent confirmation。

{paired_line('D1：L3 canonical-full signed best-mode field 相对 synchronized all-contact null', d1)}
{paired_line('D2-direct：L3−L2m seed-removed signed field correspondence', d2)}
{paired_line('D2-attenuation：削弱 L3 selected shortcuts 后 seed-removed concordance damage AUC', d2_auc)}
{paired_line('非 oracle：train-prevalence mixture 相对 synchronized all-contact null', mixture)}
{paired_line('L3 相对 split-matched suffix reassignment 的 cross-state 增量', l3_suffix)}
{paired_line('train-only oracle template 相对 synchronized all-contact null', template)}

Primary early-ictal interaction 为 `rho={f(primary.get('spearman_rho'))}`，单侧 patient-label
permutation `P={f(primary.get('permutation_p_greater'))}`，coherent synchronized spatial-null
interaction `P={f(primary.get('spatial_null', {}).get('spatial_null_p_greater'))}`；联合主判据取较大值
`P={f(primary.get('joint_primary_p_greater'))}`，bootstrap 95% CI
`[{f(primary.get('bootstrap_95_ci', [None, None])[0])}, {f(primary.get('bootstrap_95_ci', [None, None])[1])}]`。

以上两个 `J` interaction 使用的是正式训练和 target 解封前冻结的 event-mean sparse exceedance
burden。原 event-median estimand 在 28/28 患者中精确为零，已按
`J_ESTIMAND_PREFREEZE_REPAIR.json` 标记为退化 sensitivity；因此不能将本结果包装成未经修订的
原始 preregistered `J`。

### 8.3 预冻结 claim adjudication

- target-free `(L3−L2m) × J`：**{status_word(claim['PRIMARY_TARGET_FREE_NONLOCALITY_INTERACTION']['supported'])}**；
- 总体 prefix–suffix information：**{status_word(claim['KEY_SUFFIX_INFORMATION']['supported_all_transitions'])}**；
- distal-specific suffix information：**{status_word(claim['KEY_SUFFIX_INFORMATION']['supported_distal_specificity'])}**；
- D1 cross-state field correspondence：**{status_word(claim['D1_CROSS_STATE_FIELD_CORRESPONDENCE']['supported'])}**；
- early-ictal `(L3−L2m) × J`：**{status_word(claim['PRIMARY_EARLY_NONLOCALITY_INTERACTION']['supported'])}**；
- D2 shortcut-specific cross-state contribution（两项 Holm family）：
  **{status_word(claim['D2_SHORTCUT_SPECIFIC_CROSS_STATE_CONTRIBUTION']['supported'])}**。

允许表述必须以 `FINAL_CLAIM_ADJUDICATION.json` 为准。无论数值方向如何，都不能把 effective
scaffold 写成 anatomical/white-matter connectivity，也不能把 broadband energy field 写成 arrival
time 或 recruitment order。按预冻结决策树，下一条单一机制扩展为：
`{claims['next_extension_rule']}`。
"""
    original = report_path.read_text()
    old_status = (
        "状态：**执行中；Stage A–E 完成，Stage F target-free attenuation/gain/field freeze "
        "运行中，target 尚未解封。本文不是最终验收版本。**"
    )
    new_status = (
        "状态：**A–H 全流程完成；17/167 locked internal benchmark、Figure 6、source-data "
        "export 与 machine closeout 均已生成，待用户终审后 commit/push。**"
    )
    if old_status not in original:
        raise RuntimeError("closeout report running-status line changed unexpectedly")
    original = original.replace(old_status, new_status, 1)
    stage_rows = {
        "| F | mechanism、arm-specific attenuation、gain-adjusted fields 在 target 前冻结 | `STAGE_F_TARGET_FREE_COMPLETE.json` 与五套 field/metric manifests | 运行中 |":
        "| F | mechanism、arm-specific attenuation、gain-adjusted fields 在 target 前冻结 | `STAGE_F_TARGET_FREE_COMPLETE.json` 与五套 field/metric manifests | 已完成 |",
        "| G | authorization 后读取 17/167 broadband target，patient-first locked benchmark | `TARGET_UNSEAL_AUTHORIZATION.json`、`TARGET_UNLOCK_RECORD.json`、`EARLY_ICTAL_*` | 待 F 完成后自动运行 |":
        "| G | authorization 后读取 17/167 broadband target，patient-first locked benchmark | `TARGET_UNSEAL_AUTHORIZATION.json`、`TARGET_UNLOCK_RECORD.json`、`EARLY_ICTAL_*` | 已完成 |",
        "| H | Figure 6、source data、逐项 closeout audit、中文结论 | `FIGURE6_*`、`CLOSEOUT_AUDIT.json`、本文 §8 | 待 G 完成后自动运行 |":
        "| H | Figure 6、source data、逐项 closeout audit、中文结论 | `FIGURE6_*`、`CLOSEOUT_AUDIT.json`、本文 §8 | 已完成 |",
    }
    for pending, completed in stage_rows.items():
        if original.count(pending) != 1:
            raise RuntimeError(f"closeout report stage row changed unexpectedly: {pending}")
        original = original.replace(pending, completed, 1)
    if original.count(BEGIN) != 1 or original.count(END) != 1:
        raise RuntimeError("closeout report final-results markers are missing or duplicated")
    before, remainder = original.split(BEGIN, 1)
    _old, after = remainder.split(END, 1)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(before + BEGIN + "\n" + block.strip() + "\n" + END + after)
    temporary.replace(report_path)
    print(report_path)


if __name__ == "__main__":
    main()

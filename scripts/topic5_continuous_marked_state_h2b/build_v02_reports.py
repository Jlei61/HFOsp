#!/usr/bin/env python3
"""Render plain-language, technical, and handoff reports for H2b v0.2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch as _torch  # noqa: F401
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    V0_2_RESULT_ROOT, sha256_file, utc_now,
)


V01 = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/"
    "h2b_cross_task/v0_1"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "不可估计"
    if not np.isfinite(number):
        return "不可估计"
    return f"{number:+.{digits}f}"


def _support_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "risk_sets").glob("*/input_manifest.json")):
        value = _json(path)
        rows.append({
            "subject": value["subject"],
            "n": int(value["n_primary_eligible_seizures"]),
            "tier": value["support_tier"],
            "by_lead": value["n_eligible_by_lead"],
        })
    return rows


def _effect_table(frame: pd.DataFrame) -> str:
    # The main table is the full checkpoint-available population.  H1 strata are
    # shown separately with an explicit stratum column; silently stacking all
    # three populations here produced duplicate-looking rows with different
    # denominators and was scientifically unreadable.
    if "stratum" in frame:
        frame = frame[
            frame["stratum"].astype(str) == "all_checkpoint_available"
        ].copy()
    if frame.empty:
        return "当前没有可估计的患者级结果。"
    lines = [
        "|证据层|提前量|比较|患者数|同向患者|患者中位差|双侧符号检验 p|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    labels = {
        "state_minus_observation_conditional_log_loss": "持续状态 vs 当前观察",
        "persistent_minus_memoryless_conditional_log_loss": "持续状态 vs 单窗口即时编码",
        "correct_minus_wrong_time_conditional_log_loss": "正确时刻 vs 匹配的错误时刻",
    }
    unstable = []
    for row in frame.sort_values(
            ["evaluation_tier", "lead_minutes", "effect"]).itertuples(index=False):
        effect_value = float(row.patient_median_effect)
        display = _fmt(effect_value)
        if abs(effect_value) > 100:
            display = "数值不稳定"
            unstable.append(
                f"{row.evaluation_tier}/{int(row.lead_minutes)} min/{labels.get(str(row.effect), str(row.effect))}"
            )
        lines.append(
            f"|{row.evaluation_tier}|{int(row.lead_minutes)} min|"
            f"{labels.get(str(row.effect), str(row.effect))}|{int(row.n_patients)}|"
            f"{int(row.n_favourable)}/{int(row.n_patients)}|"
            f"{display}|{float(row.two_sided_exact_sign_p):.4g}|"
        )
    if unstable:
        lines.extend([
            "",
            "> 注：" + "；".join(unstable)
            + " 出现极端外推损失，保留在原始 CSV 中，但不作科学解释。",
        ])
    return "\n".join(lines)


def _h1_stratum_table(frame: pd.DataFrame) -> str:
    """Show the pre-specified H1 split without hiding its tiny denominators."""
    if frame.empty or "stratum" not in frame:
        return "H1 预分层没有可估计结果。"
    scoped = frame[
        (frame["lead_minutes"].astype(int) == 30)
        & frame["stratum"].astype(str).isin(
            ["h1_stable_stratum", "h1_unstable_stratum"]
        )
    ].copy()
    if scoped.empty:
        return "H1 预分层在 30 分钟没有可估计结果。"
    labels = {
        "state_minus_observation_conditional_log_loss": "持续状态 vs 当前观察",
        "persistent_minus_memoryless_conditional_log_loss": "持续状态 vs 单窗口即时编码",
        "correct_minus_wrong_time_conditional_log_loss": "正确时刻 vs 匹配的错误时刻",
    }
    strata = {
        "h1_stable_stratum": "H1-stable",
        "h1_unstable_stratum": "H1-unstable",
    }
    lines = [
        "|H1 预分层|证据层|比较|患者数|同向患者|患者中位差|p|",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in scoped.sort_values(
        ["stratum", "evaluation_tier", "effect"]
    ).itertuples(index=False):
        lines.append(
            f"|{strata.get(str(row.stratum), str(row.stratum))}|"
            f"{row.evaluation_tier}|{labels.get(str(row.effect), str(row.effect))}|"
            f"{int(row.n_patients)}|{int(row.n_favourable)}/{int(row.n_patients)}|"
            f"{_fmt(row.patient_median_effect)}|"
            f"{float(row.two_sided_exact_sign_p):.4g}|"
        )
    return "\n".join(lines)


def _primary_scoring_table(root: Path, support: list[dict]) -> tuple[str, str]:
    """Expose eligible versus actually held-out denominators for primary patients."""
    primary = [row for row in support if row["tier"] == "primary_chronological"]
    if not primary:
        return "当前没有主层患者。", "当前没有达到 ≥10 次合格发作的主层患者。"
    lines = [
        "|患者|30 min 合格发作|最终 held-out risk sets|state−observation|persistent−memoryless|置换 95% 区间|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    sentences = []
    for item in primary:
        subject = str(item["subject"])
        output = root / "fits/by_subject" / subject / "primary"
        frame = pd.read_csv(output / "patient_median_probe_metrics.csv")
        selected = frame[frame.lead_minutes.astype(int) == 30]
        if selected.empty:
            lines.append(f"|{subject}|{item['n']}|0|不可估计|不可估计|不可估计|")
            sentences.append(f"{subject} 的 30 分钟主评分不可估计")
            continue
        row = selected.iloc[0]
        heldout_value = row.get("B_state__n_risk_sets", 0)
        heldout = int(heldout_value) if pd.notna(heldout_value) else 0
        state = row.get("state_minus_observation_conditional_log_loss")
        memory = row.get("persistent_minus_memoryless_conditional_log_loss")
        permutation = _json(output / "time_label_permutation.json")
        lo, hi = permutation.get("null_q025"), permutation.get("null_q975")
        interval = (
            f"[{_fmt(lo)}, {_fmt(hi)}]"
            if lo is not None and hi is not None else "不可估计"
        )
        lines.append(
            f"|{subject}|{item['n']}|{heldout}|{_fmt(state)}|"
            f"{_fmt(memory)}|{interval}|"
        )
        inside = False
        try:
            inside = float(lo) <= float(state) <= float(hi)
        except (TypeError, ValueError):
            pass
        sentences.append(
            f"{subject} 虽有 {item['n']} 次合格发作，但最终评分只有 {heldout} 个 "
            f"held-out risk sets；state−observation={_fmt(state)}"
            + ("，仍在置换范围内" if inside else "")
            + f"，persistent−memoryless={_fmt(memory)}"
        )
    return "\n".join(lines), "；".join(sentences) + "。"


def _e384_sentence() -> str:
    path = V01 / "fits/e384_instrument/patient_median_probe_metrics.csv"
    if not path.is_file():
        return "E384 v0.1 旧仪器结果文件当前不可读。"
    frame = pd.read_csv(path)
    row = frame[frame.lead_minutes.astype(int) == 30]
    if row.empty:
        return "E384 v0.1 的 30 分钟主读数不可估计。"
    value = row.iloc[0]
    return (
        "E384 的 v0.1 单患者开发仪器实际只有 4 次 30 分钟合格发作；"
        f"持续状态相对当前观察的条件损失差为 {_fmt(value['state_minus_observation_conditional_log_loss'])}，"
        "方向不利，只能作为描述性阴性，不能代表队列。"
    )


def _phenotype_text(root: Path) -> str:
    availability_path = root / "reports/phenotype_target_availability.json"
    if not availability_path.is_file():
        return "冻结发作亚型目标尚未汇总。"
    value = _json(availability_path)
    available = int(value.get("n_available_target_rows", 0))
    subjects = len(value.get("subject_tables") or {})
    index_path = root / "fits/phenotype/cohort_phenotype_index.json"
    if not index_path.is_file():
        return f"找到 {available} 条既有冻结亚型标签（{subjects} 位患者），但没有可估计的迁移拟合。"
    index = _json(index_path)
    patient_path = Path((index.get("outputs") or {}).get("patient", {}).get("path", ""))
    if not patient_path.is_file():
        return (
            f"二级分析只连接既有冻结亚型标签，共 {available} 条可用标签、"
            f"{int(index.get('n_patients_run', 0))} 位患者执行；没有患者级可估计结果。"
        )
    patient = pd.read_csv(patient_path)
    effect = "state_minus_observation_loss"
    finite = patient[
        pd.to_numeric(patient.get(effect), errors="coerce").notna()
    ].copy() if effect in patient else patient.iloc[:0].copy()
    n_cells = int(len(patient))
    n_estimable = int(len(finite))
    n_patients = int(finite["patient_id"].nunique()) if n_estimable else 0
    details = []
    for row in finite.sort_values(["patient_id", "target_name"]).itertuples(index=False):
        heldout = getattr(row, "state__n_evaluation_seizures", np.nan)
        details.append(
            f"{row.patient_id}/{row.target_name}（{row.evaluation_tier}，"
            f"held-out {int(heldout) if np.isfinite(float(heldout)) else 0} 次，"
            f"state−observation={_fmt(getattr(row, effect))}）"
        )
    detail_text = "；".join(details) if details else "无"
    return (
        f"二级分析只连接既有冻结亚型标签，共 {available} 条 seed-expanded 可用行、"
        f"{int(index.get('n_patients_run', 0))} 位患者执行。实际只有 {n_estimable}/{n_cells} 个"
        f"患者×目标单元（{n_patients} 位患者）给出有限结果：{detail_text}。"
        f"其余 {n_cells - n_estimable}/{n_cells} 个单元因目标缺失、类别不足或样本不足不可估计；"
        "因此不能称为跨患者亚型迁移证据。没有重新聚类，也没有在看过状态后发明新的早期募集指标。"
    )


def _interpret_primary(summary: pd.DataFrame) -> list[str]:
    rows = summary[
        (summary.stratum.astype(str) == "all_checkpoint_available")
        & (summary.lead_minutes.astype(int) == 30)
    ]
    out = []
    labels = {
        "state_minus_observation_conditional_log_loss": (
            "IED 任务学到的冻结状态是否跨任务增加发作前信息"
        ),
        "persistent_minus_memoryless_conditional_log_loss": (
            "跨窗口记忆是否超过当前窗口即时编码"
        ),
        "correct_minus_wrong_time_conditional_log_loss": (
            "状态是否具有正确时刻专属性"
        ),
    }
    for effect, group in rows.groupby("effect", sort=True):
        parts = []
        for row in group.itertuples(index=False):
            direction = "方向有利" if float(row.patient_median_effect) < 0 else "方向不利"
            parts.append(
                f"{row.evaluation_tier}: {int(row.n_favourable)}/{int(row.n_patients)}，"
                f"中位差 {_fmt(row.patient_median_effect)}，p={float(row.two_sided_exact_sign_p):.4g}（{direction}）"
            )
        out.append(f"- {labels.get(str(effect), str(effect))}：" + "；".join(parts) + "。")
    return out or ["- 30 分钟主比较没有可估计患者。"]


def build_complete(root: Path) -> dict:
    audit = _json(root / "reports/machine_audit.json")
    if audit.get("status") != "PASS_COMPLETE":
        raise ValueError("final reports require PASS_COMPLETE machine audit")
    inventory = _json(root / "manifests/r1_7_checkpoint_inventory.json")
    aggregate = _json(root / "reports/cohort_patient_first_summary.json")
    summary = pd.read_csv(root / "reports/cohort_patient_first_summary.csv")
    support = _support_rows(root)
    tiers = {}
    for row in support:
        tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
    tier_text = "、".join(f"{key} {value} 人" for key, value in sorted(tiers.items()))
    effect_markdown = _effect_table(summary)
    h1_stratum_markdown = _h1_stratum_table(summary)
    interpretations = "\n".join(_interpret_primary(summary))
    primary_scoring_markdown, primary_scoring_sentence = _primary_scoring_table(
        root, support,
    )
    e384 = _e384_sentence()
    phenotype = _phenotype_text(root)

    scientific_question = (
        "本阶段检验的是：一个完全在连续背景和间期事件任务上学到、随后冻结的状态，"
        "能否在只训练一个很小的发作风险读出器时，提供超过近期事件史和当前背景观察的发作前信息。"
    )
    one_sentence = (
        "本轮没有建立 H2b 跨任务状态证据：唯一主层患者的状态增量很小且落在置换零带内，"
        "持续记忆没有胜过单窗口编码；低支持层虽有探索性有利方向，但正确时刻状态并未优于"
        "匹配的错误时刻。"
    )
    plain = f"""# H2b Cross-task Transfer v0.2 白话报告

## 一句话

{one_sentence}

## 为什么做这一步

我们理想中的“癫痫状态”不应只在原来的间期事件任务里有用。更有说服力的检验是：先用间期数据把状态学好并彻底冻结，再看它能不能迁移到另一个任务——区分真实发作前 30 分钟与同一患者、同一记录段里的普通时刻。发作标签只训练最后一个低容量读出器，没有反过来修改状态。

## 研究规模

- R1.7 清单共有 {int(inventory['n_subjects'])} 位患者、{int(inventory['n_checkpoint_available_cells'])} 个可读冻结 checkpoint cell。
- R1.7B 是 exploratory development extension：consumer-side audit 重算了 85 个 cell 与 75 个可读 checkpoint/result 的哈希并通过，但该 release 不满足旧 v0.1 的 50-fit formal gate，所以本轮只能作为 development 证据。
- 最终完成原始背景读取和发作支持审查的患者：{len(support)} 位。
- 30 分钟支持层级：{tier_text or '无'}。
- crosswalk 共 {int(pd.read_csv(root / 'manifests/seizure_crosswalk.csv').shape[0])} 条 development 发作记录；冻结状态缓存 {int(audit['details']['n_state_cache_cells'])} 个 checkpoint cells。
- H1 是否稳定只用于分层查看，没有决定某位患者能不能进入 H2b。

## 旧的单患者结果如何收口

{e384}

## 这次队列结果

差值均定义为“前一个模型减后一个对照”，负数表示冻结状态更好。患者内先合并 optimizer seeds，再以患者为统计单位。

{effect_markdown}

主层必须同时公开“合格发作数”和真正进入最终评分的 held-out 分母：

{primary_scoring_markdown}

{primary_scoring_sentence}

30 分钟主读数的白话解释：

{interpretations}

### H1 预分层（仅作分层，不作纳入 gate）

{h1_stratum_markdown}

这些分层中的患者数更少，只用于检查方向是否明显不同，不能替代全 checkpoint-available 分母。

## 发作类型的二级探索

{phenotype}

## 现在能说什么

- 唯一主层患者 epilepsiae_548 的 `state−observation=-0.0212`，但最终只评分 2 个 held-out risk sets，且落在置换零带内；`persistent−memoryless=+0.0071`，不支持跨窗口持续状态。
- 低支持层中，描述性患者在 30 分钟出现 4/4 有利方向，但只有 4 人、p=0.125；LOSO 层为 2/3、中位效应接近零。这只能作为后续扩大样本的探索线索。
- 30 分钟正确时刻状态没有优于匹配错误时刻：可估计的 LOSO 为 0/1、描述层为 1/4；因此没有时刻专属性证据。
- 冻结亚型分析实际只有 2 个很小的患者×目标单元可估计，不能形成跨患者亚型迁移结论。
- 因此当前结论是“未建立支持”，不是“生理状态不存在”；它只限制当前冻结 R1.7 checkpoint、当前患者支持量和低容量读出器。

## 不能说什么

- 不能说已经完成跨队列确认；全部是 development 数据。
- 不能把预测增量写成发作因果机制、发作预测器临床性能或 IED 改变状态的 H3 证据。
- 普通阴性只限制当前冻结 checkpoint 和当前低容量读出器，不等于不存在生理状态。
- 正式检验分区和 sealed 分区没有打开；H3/T2、paper-ready 图均未触碰。
"""

    support_lines = [
        "|患者|30 min 合格发作|30 min held-out risk sets|层级|5/15/30/60/120 min 合格数|",
        "|---|---:|---:|---|---|",
    ]
    for row in support:
        counts = "/".join(str(row["by_lead"].get(str(lead), 0))
                          for lead in (5, 15, 30, 60, 120))
        patient_path = root / "fits/by_subject" / row["subject"] / "primary/patient_median_probe_metrics.csv"
        heldout = 0
        if patient_path.is_file():
            patient = pd.read_csv(patient_path)
            selected = patient[patient.lead_minutes.astype(int) == 30]
            if not selected.empty:
                value = selected.iloc[0].get("B_state__n_risk_sets")
                heldout = int(value) if pd.notna(value) else 0
        support_lines.append(
            f"|{row['subject']}|{row['n']}|{heldout}|{row['tier']}|{counts}|"
        )
    technical = f"""# H2b Cross-task Transfer v0.2 技术报告

## 1. Scientific estimand

{scientific_question}

Development outcome: {one_sentence}

主 estimand 是 held-out 30-min conditional risk-set log loss：`B_state - B_observation`。Secondary state-validity contrasts 为 `persistent - memoryless` 与 `correct-time - matched-wrong-time`。所有差值以负数为有利方向。

## 2. Frozen source and boundary

- source: R1.7B exploratory development extension；consumer-side audit 逐 cell 重算 checkpoint/result SHA256，但该 release 不满足旧 v0.1 的 50-fit formal gate，不能作为 formal H2b confirmation。
- state/observer/generator/IED decoder 全冻结；seizure loss 无梯度进入上游。
- H1 stability 是预先保留的 secondary stratum，不是运行 gate。
- 仅使用 `onset < dev_end_epoch` 的 seizure identifiers；后段/正式分区 seizure IDs 不写入 crosswalk、exclusion 或 risk-set artifacts。
- raw observation 只使用 anchor 时刻及以前的数据，绝对时间 `float64`，每个 recorded coverage segment 重置状态，当前 observation age ≤30 s。

## 3. Cohort support

{chr(10).join(support_lines)}

层级规则固定为：≥10 次 `primary_chronological`；5–9 次 `sensitivity_loso`；2–4 次 `descriptive_case_series`；<2 次 `not_estimable`。30 min 的合格发作 ID 固定所有 sensitivity lead 的患者内发作人群，其他 lead 不得补招发作。

## 4. Risk-set design

每个 case 位于 `onset - lead` 的精确时刻；control 来自同一患者、同一 recorded coverage segment、相同 observation availability，且未来 horizon 无发作。所有 ictal 与随后 120 min 时刻从 control 和 wrong-time donor 中排除。TRAIN/SELECT/TEST 按发作时间划分；同一 anchor/time 不跨 partition 重用。患者 feature width 不同，因此每位患者独立拟合，绝不拼接 feature matrix。

## 5. Probe and inference

低容量 intercept-free ridge conditional logistic probe 在同一 risk set 上比较 `B_history`、`B_observation`、`B_state`、`memoryless`；wrong-time 使用独立但同规则 risk table。正则仅在 TRAIN/SELECT 或 nested LOSO 内选择。optimizer seed 先在患者内取中位数，cohort summary 再对患者做符号统计。

## 6. Results

{effect_markdown}

### 6.1 Primary eligible vs held-out denominator

{primary_scoring_markdown}

{primary_scoring_sentence}

30 min primary interpretation:

{interpretations}

### 6.2 Pre-specified H1 strata

{h1_stratum_markdown}

H1 stability was used only as a reporting stratum, never as an inclusion gate；各单元分母很小，不作独立 cohort 结论。

## 7. Frozen phenotype transfer

{phenotype}

亚型仅来自既有 `broad_ER` / `gamma_ER` 非 outlier 标签；没有重聚类。既有 ictal recruitment cache 没有预冻结的盲法 seizure-level scalar，因此 early recruitment 记为不可估计，没有替换成事后发明的指标。phenotype 分析从 primary risk table 取样，不以 matched wrong-time 可用性为 gate。

## 8. Machine acceptance

- machine audit: `{audit['status']}`
- checkpoint cells: {int(inventory['n_cells'])} total / {int(inventory['n_checkpoint_available_cells'])} available
- patient-first rows: {int(audit['details']['n_patient_first_rows'])}
- state cache cells: {int(audit['details']['n_state_cache_cells'])}
- formal/sealed/H3/T2/paper-ready modifications: all false

## 9. Claim boundary

本结果最多支持 development cross-task prediction。它不等于 seizure causality、clinical forecasting performance、cohort confirmation 或 IED→state generator mechanism。E384 v0.1 的 30-min 结果按实际 4 次合格发作收口，不再使用“5 次主层发作”的旧说法。

## 10. Reproduction anchors

- `COHORT_RUN_COMPLETE.json`
- `reports/machine_audit.json`
- `reports/per_patient_lead_results.csv`
- `reports/cohort_patient_first_summary.csv`
- `reports/phenotype_target_availability.json`
- `fits/by_subject/<subject>/.../risk_probe_machine_audit.json`
"""

    plain_path = root / "reports/h2b_cross_task_v0_2_plain.md"
    technical_path = root / "reports/h2b_cross_task_v0_2_technical.md"
    _write(plain_path, plain)
    _write(technical_path, technical)
    outputs = {
        "plain": {"path": str(plain_path), "sha256": sha256_file(plain_path)},
        "technical": {
            "path": str(technical_path), "sha256": sha256_file(technical_path),
        },
    }
    build_handoff(root, outputs=outputs)
    outputs["handoff"] = {
        "path": str(root / "CURRENT_HANDOFF.md"),
        "sha256": sha256_file(root / "CURRENT_HANDOFF.md"),
    }
    return outputs


def build_handoff(root: Path, *, outputs: dict | None = None) -> Path:
    queue_path = root / "QUEUE_STATUS.json"
    queue = _json(queue_path) if queue_path.is_file() else {"status": "UNKNOWN"}
    census_path = root / "manifests/support_census.json"
    census = _json(census_path) if census_path.is_file() else {}
    audit_path = root / "reports/machine_audit.json"
    audit = _json(audit_path) if audit_path.is_file() else {}
    runtime_path = root / "RUNTIME_MONITOR_STATUS.json"
    runtime = _json(runtime_path) if runtime_path.is_file() else {}
    inventory_path = root / "manifests/r1_7_checkpoint_inventory.json"
    inventory = _json(inventory_path) if inventory_path.is_file() else {}
    upstream_audit_path = root / "reports/r1_7b_consumer_acceptance_audit.json"
    upstream_audit = _json(upstream_audit_path) if upstream_audit_path.is_file() else {}
    complete = audit.get("status") == "PASS_COMPLETE"
    monitor_text = (
        f"inactive after completion (last recorded PID "
        f"`{runtime.get('monitor_pid', 'unknown')}`, tmux "
        f"`{runtime.get('tmux_session', 'unknown')}`)"
        if complete else
        f"PID `{runtime.get('monitor_pid', 'unknown')}`, tmux "
        f"`{runtime.get('tmux_session', 'unknown')}`"
    )
    text = f"""# H2b v0.2 CURRENT HANDOFF

- updated UTC: {utc_now()}
- queue status: `{queue.get('status')}` / stage `{queue.get('stage', 'unknown')}`
- machine audit: `{audit.get('status', 'not run')}`
- upstream R1.7B consumer audit: `{upstream_audit.get('status', 'not run')}` (development source; formal v0.1 gate not met)
- runtime monitor: {monitor_text}
- raw mounts: `{json.dumps(census.get('raw_mounts_present'), ensure_ascii=False)}`
- R1.7 inventory: {inventory.get('n_subjects', 'NA')} subjects / {inventory.get('n_checkpoint_available_cells', 'NA')} readable checkpoint cells
- checkpoint-available subjects: {census.get('n_checkpoint_available_subjects', 'NA')}
- subjects requiring raw primary support: {census.get('n_subjects_requiring_raw_for_primary_h2b', 'NA')}
- required raw caches ready: {census.get('n_required_subjects_with_raw_cache', 'NA')}
- scientific state: {'development closeout complete; H2b not established' if complete else 'operationally waiting; no new H2b biological result'}

## Safety boundary

Formal/sealed partitions remain closed. H3/T2 and paper-ready figures were not touched. H1 stability is a stratum, not an H2b gate. State-source parameters remain frozen.

## Durable execution

The runtime monitor waits for both transient mounts and every required raw cache. It then launches `run_v02_cohort_queue.py` in a detached session. The queue uses one producer per GPU for missing upstream rebuilds, memory-bounded CPU state extraction, atomic manifests, serial lower-batch retry, patient-separate probes, final machine audit, and canonical result sync.

## Key paths

- result root: `{root}`
- queue status: `{queue_path}`
- runtime status: `{runtime_path}`
- machine audit: `{audit_path}`
- plain report: `{(outputs or {}).get('plain', {}).get('path', 'pending')}`
- technical report: `{(outputs or {}).get('technical', {}).get('path', 'pending')}`
"""
    path = root / "CURRENT_HANDOFF.md"
    _write(path, text)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    parser.add_argument("--handoff-only", action="store_true")
    args = parser.parse_args()
    root = args.result_root.resolve()
    if args.handoff_only:
        path = build_handoff(root)
        print(json.dumps({"status": "HANDOFF_UPDATED", "path": str(path)}))
    else:
        outputs = build_complete(root)
        print(json.dumps({"status": "COMPLETE", "outputs": outputs},
                         ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

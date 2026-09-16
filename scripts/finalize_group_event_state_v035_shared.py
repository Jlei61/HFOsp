#!/usr/bin/env python3
"""Aggregate the shared S_N/S_G pilot into machine, plain and technical reports."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/data/hfosp_group_event_state_v0_3_5_shared")
SUBJECTS = ("epilepsiae_253", "epilepsiae_1096", "epilepsiae_1125")
SEEDS = (20260903, 20260904, 20260905)
DOC = ROOT / "docs/archive/topic5"


def med(values):
    clean = [float(v) for v in values if v is not None]
    return None if not clean else statistics.median(clean)


def fmt(value):
    return "NA" if value is None else f"{float(value):.4g}"


def main() -> None:
    queue = json.loads((OUT / "supervisor/queue_status.json").read_text())
    rows, h2a, h2b, producers = [], [], [], []
    for subject in SUBJECTS:
        for seed in SEEDS:
            for family in ("S_N", "S_G"):
                unit = next(iter((OUT / "shared_producer" / subject / family).glob(f"*_state_seed{seed}")), None)
                if unit is None or not (unit / "card.json").exists():
                    continue
                pc = json.loads((unit / "card.json").read_text())
                with np.load(pc["state_trajectory"], allow_pickle=False) as trajectory:
                    state_pre = np.asarray(trajectory["state_pre"], dtype=np.float64)
                    state_post = np.asarray(trajectory["state_post"], dtype=np.float64)
                producers.append({"subject": subject, "seed": seed, "family": family,
                                  "selected_epoch": pc["selected_epoch"],
                                  "epochs_run": len(pc.get("history", [])),
                                  "selected_at_budget_edge": (
                                      pc.get("selected_epoch") == len(pc.get("history", [])) - 1
                                      and len(pc.get("history", [])) >= 20
                                  ),
                                  "mean_state_sd": float(np.std(state_pre, axis=0).mean()),
                                  "event_update_rms": float(np.sqrt(np.mean((state_post - state_pre) ** 2))),
                                  "selection_targets_read": pc["selection_targets_read"]})
                ev = OUT / "frozen_evaluator" / subject / family / f"seed{seed}" / "card.json"
                if ev.exists():
                    card = json.loads(ev.read_text())
                    for horizon, hr in card["horizons"].items():
                        for endpoint, er in hr.get("endpoints", {}).items():
                            if er.get("status") != "ESTIMATED": continue
                            rows.append({"subject": subject, "seed": seed,
                                         "source_family": family, "target_family": family,
                                         "horizon_seconds": int(horizon), "endpoint": endpoint,
                                         "n_independent_selection_windows": hr.get("n_independent_selection_windows"),
                                         **er["contrasts"]})
                if family == "S_G":
                    prefix = OUT / "same_prefix" / subject / f"seed{seed}" / "card.json"
                    if prefix.exists():
                        card = json.loads(prefix.read_text())["conditional_continuation"]
                        for offset, values in card["arms"].items():
                            for endpoint in ("grammar", "contact_nll", "stop_bce", "next_bce"):
                                correct = values["rate_plus_mark"][endpoint]["mean_on_shift_support"]
                                shifted = values["block_shift_mark"][endpoint]["mean_on_shift_support"]
                                rate = values["rate_only"][endpoint]["mean_on_shift_support"]
                                period = values["period_mean_mark"][endpoint]["mean_on_shift_support"]
                                h2a.append({"subject": subject, "seed": seed, "offset": offset,
                                            "endpoint": endpoint,
                                            "state_gain_over_rate": None if rate is None or correct is None else rate - correct,
                                            "state_gain_over_period_mean": None if period is None or correct is None else period - correct,
                                            "correct_time_gain_over_shift": None if correct is None or shifted is None else shifted - correct})
                binding = OUT / "shared_h2b" / subject / family / f"seed{seed}" / "shared_state_binding.json"
                if binding.exists():
                    bc = json.loads(binding.read_text())
                    hc = json.loads(Path(bc["h2b_card"]).read_text())
                    hazard = hc["distance_survival"]
                    h2b.append({"subject": subject, "seed": seed, "family": family,
                                "task": "seizure_distance", "lead": None, "endpoint": "hazard",
                                "seizures_by_phase": hazard["seizures_by_phase"],
                                **hazard.get("arms", {}).get("registered_contrasts", {})})
                    for lead, endpoints in hc.get("early_ictal_field_and_path", {}).items():
                        for endpoint, result in endpoints.items():
                            if not isinstance(result, dict) or result.get("status") == "NOT_ESTIMABLE":
                                continue
                            h2b.append({"subject": subject, "seed": seed, "family": family,
                                        "task": "early_ictal_field", "lead": lead,
                                        "endpoint": endpoint,
                                        "state_gain_over_q": result.get("state_gain_over_q"),
                                        "correct_time_gain_over_shift": result.get("correct_time_gain_over_shift"),
                                        "state_gain_over_period_mean": result.get("mark_gain_over_period_mean"),
                                        "support": result.get("support")})
            for source, target, dirname in (
                ("S_N", "S_G", "sn_to_sg"), ("S_G", "S_N", "sg_to_sn"),
                ("S_N+S_G", "S_N", "combined_to_sn"), ("S_N+S_G", "S_G", "combined_to_sg"),
            ):
                path = OUT / "cross_evaluator" / subject / f"seed{seed}" / dirname / "card.json"
                if not path.exists(): continue
                card = json.loads(path.read_text())
                for horizon, hr in card["horizons"].items():
                    for endpoint, er in hr.get("endpoints", {}).items():
                        if er.get("status") != "ESTIMATED": continue
                        rows.append({"subject": subject, "seed": seed,
                                     "source_family": source, "target_family": target,
                                     "horizon_seconds": int(horizon), "endpoint": endpoint,
                                     "n_independent_selection_windows": hr.get("n_independent_selection_windows"),
                                     **er["contrasts"]})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_family"], row["target_family"], row["horizon_seconds"], row["endpoint"])].append(row)
    summary_rows = []
    for key, values in sorted(grouped.items()):
        subject_medians = defaultdict(list)
        for row in values:
            subject_medians[row["subject"]].append(row)
        by_subject = []
        for subject, subject_rows in subject_medians.items():
            by_subject.append({"subject": subject,
                               "state_gain_over_q": med([r["state_gain_over_q"] for r in subject_rows]),
                               "state_gain_over_constant": med([r["state_gain_over_constant"] for r in subject_rows]),
                               "correct_time_gain_over_block_shift": med([r["correct_time_gain_over_block_shift"] for r in subject_rows]),
                               "n_independent_selection_windows": med([r.get("n_independent_selection_windows") for r in subject_rows]),
                               "n_independent_block_shift_windows": med([r.get("n_independent_block_shift_windows") for r in subject_rows])})
        summary_rows.append({"source_family": key[0], "target_family": key[1],
                             "horizon_seconds": key[2], "endpoint": key[3],
                             "n_subjects": len(by_subject), "by_subject": by_subject,
                             "n_state_gain_positive": sum(r["state_gain_over_q"] is not None and r["state_gain_over_q"] > 0 for r in by_subject),
                             "n_correct_time_positive": sum(r["correct_time_gain_over_block_shift"] is not None and r["correct_time_gain_over_block_shift"] > 0 for r in by_subject),
                             "n_joint_positive": sum(r["state_gain_over_q"] is not None and r["state_gain_over_q"] > 0 and r["correct_time_gain_over_block_shift"] is not None and r["correct_time_gain_over_block_shift"] > 0 for r in by_subject),
                             "cohort_median_state_gain_over_q": med([r["state_gain_over_q"] for r in by_subject]),
                             "cohort_median_correct_time_gain": med([r["correct_time_gain_over_block_shift"] for r in by_subject]),
                             "independent_selection_windows_by_subject": [r["n_independent_selection_windows"] for r in by_subject],
                             "independent_shift_windows_by_subject": [r["n_independent_block_shift_windows"] for r in by_subject]})
    h2a_grouped = defaultdict(list)
    for row in h2a:
        h2a_grouped[(row["offset"], row["endpoint"])].append(row)
    h2a_summary = []
    for key, values in sorted(h2a_grouped.items()):
        by_subject = defaultdict(list)
        for row in values:
            by_subject[row["subject"]].append(row["correct_time_gain_over_shift"])
        subject_rows = []
        for subject in sorted(by_subject):
            current = [r for r in values if r["subject"] == subject]
            subject_rows.append({
                "subject": subject,
                "state_gain_over_rate": med([r["state_gain_over_rate"] for r in current]),
                "state_gain_over_period_mean": med([r["state_gain_over_period_mean"] for r in current]),
                "correct_time_gain_over_shift": med([r["correct_time_gain_over_shift"] for r in current]),
            })
        h2a_summary.append({
            "offset": key[0], "endpoint": key[1], "n_subjects": len(subject_rows),
            "n_state_positive": sum(r["state_gain_over_rate"] is not None and r["state_gain_over_rate"] > 0 for r in subject_rows),
            "n_correct_time_positive": sum(r["correct_time_gain_over_shift"] is not None and r["correct_time_gain_over_shift"] > 0 for r in subject_rows),
            "n_joint_positive": sum(r["state_gain_over_rate"] is not None and r["state_gain_over_rate"] > 0 and r["correct_time_gain_over_shift"] is not None and r["correct_time_gain_over_shift"] > 0 for r in subject_rows),
            "cohort_median_state_gain": med([r["state_gain_over_rate"] for r in subject_rows]),
            "cohort_median_correct_time_gain": med([r["correct_time_gain_over_shift"] for r in subject_rows]),
            "by_subject": subject_rows,
        })
    payload = {
        "format": "group_event_state_v0_3_5_shared_final_summary_v1",
        "queue": queue, "producers": producers, "evaluator_rows": rows,
        "nested_summary": summary_rows, "h2a_rows": h2a, "h2a_summary": h2a_summary,
        "h2b_rows": h2b,
        "scientific_scope": "three-patient development pilot; no cohort confirmation",
        "development_targets_read": False, "sealed_partition_opened": False,
    }
    report_dir = OUT / "final_reports"; report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "shared_state_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    own = [r for r in summary_rows if r["source_family"] == r["target_family"]]
    table = ["| 状态 | 目标 | 未来窗 | 端点 | 患者数 | 独立评价窗 / 错时窗 | 超过动态基线 | 正确时刻优于错时 | 两条件同过 |",
             "|---|---|---:|---|---:|---|---:|---:|---:|"]
    for row in own:
        support = ",".join(fmt(v) for v in row["independent_selection_windows_by_subject"])
        shift = ",".join(fmt(v) for v in row["independent_shift_windows_by_subject"])
        table.append(f"| {row['source_family']} | {row['target_family']} | {row['horizon_seconds']/3600:g} h | {row['endpoint']} | {row['n_subjects']} | {support} / {shift} | {fmt(row['cohort_median_state_gain_over_q'])} ({row['n_state_gain_positive']}/{row['n_subjects']}) | {fmt(row['cohort_median_correct_time_gain'])} ({row['n_correct_time_positive']}/{row['n_subjects']}) | {row['n_joint_positive']}/{row['n_subjects']} |")
    h2a_table = ["| 事件偏移 | 端点 | 患者数 | 状态超过 rate | 正确时刻优于错时 | 两条件同过 |",
                 "|---|---|---:|---:|---:|---:|"]
    for row in h2a_summary:
        h2a_table.append(
            f"| {row['offset']} | {row['endpoint']} | {row['n_subjects']} | "
            f"{fmt(row['cohort_median_state_gain'])} ({row['n_state_positive']}/{row['n_subjects']}) | "
            f"{fmt(row['cohort_median_correct_time_gain'])} ({row['n_correct_time_positive']}/{row['n_subjects']}) | "
            f"{row['n_joint_positive']}/{row['n_subjects']} |"
        )
    training_edges = sum(bool(row["selected_at_budget_edge"]) for row in producers)
    plain = """# Group-Event State v0.3.5 共享状态首轮报告\n\n## 一句话\n\n这一轮第一次让同一个 `S_N` 或 `S_G` 状态同时面对 2、6、8 小时的未来，而不是每个时间窗各训一套模型。它是三位患者的 development pilot；是否有状态增量、是否对得上时刻，必须由下表共同判断，不能只看事件率。\n\n## 已完成的科学结构\n\n- `S_N` 预测未来事件负荷；`S_G` 预测局部传播续接、community 占用、跨 community 耦合和 repertoire mixture。\n- contact decoder 的患者内骨架冻结，但其时段适配层按共享长窗分区重新训练，避免旧短窗 checkpoint 偷看新留出。\n- 每个 horizon 的小读出独立，但 producer 和状态轨迹共享。\n- 主层排除跨发作未来块；发作预测只读取冻结状态。\n\n## 结果表\n\n""" + "\n".join(table) + f"""\n\n正值表示状态臂更好，括号内是患者同向数；“两条件同过”要求同一患者既超过动态基线，又优于错时状态。独立窗一列按患者列出互不重叠墙钟窗，不能把重叠网格锚点当样本量。三位患者只能用于探索方向，不能写成队列结论。若前一列为正而时刻列为零，仍更像一个慢水平或阶段标签，而不是时刻特异的持续状态。\n\n## H2a：相同开头后的传播续接\n\n""" + "\n".join(h2a_table) + f"""\n\n只有“状态超过 rate”与“正确时刻优于错时”同时成立，才能说跨事件状态调制了相同开头之后的 contact 传播；只过其中一项不能验收。\n\n## H2b：冻结迁移到发作\n\n冻结探针共生成 {len(h2b)} 条任务记录。缺少足够 FIT 或 SELECTION 发作的任务保持 `NOT_ESTIMABLE`；不把空分母写成阴性，也不把发作结局回传到状态 producer。\n\n## 训练充分性\n\n{training_edges}/{len(producers)} 个 producer 的最佳点贴在 20 epoch 预算末端；这些单元必须延长到 60 epoch 后再裁决。状态轨迹非零只能证明模型产生了动态数值，不能证明它包含所需科学信息。\n\n## 结论边界\n\n本轮最多建立 marked-history predictive state 的 development 证据。它还不是独立生理潜状态；只有冻结后能稳定预测发作距离或发作早期空间场，才接近癫痫易感状态。H2b 缺少留出发作的患者记为不可估，不记为阴性。\n"""
    technical = "# Group-Event State v0.3.5 共享状态技术报告\n\n" + \
        f"- producer cells: {len(producers)}\n- producer budget-edge cells: {training_edges}\n- evaluator rows: {len(rows)}\n- H2a rows: {len(h2a)}\n- H2b task rows: {len(h2b)}\n\n" + \
        "机器证据：`/data/hfosp_group_event_state_v0_3_5_shared/final_reports/shared_state_summary.json`。所有 producer 均在 FIT 训练、INNER 选 checkpoint；SELECTION 只由冻结 evaluator 读取。冻结 ridge 在平均损失尺度定义，count 使用四臂共享 dispersion 的负二项 likelihood。每个 horizon 同时报重叠锚点与互不重叠墙钟窗，正式 development target 与 sealed/test 均未打开。\n\n" + "\n".join(table) + "\n\n" + "\n".join(h2a_table) + "\n"
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "group_event_state_v0_3_5_shared_state_plain_2026-09-04.md").write_text(plain)
    (DOC / "group_event_state_v0_3_5_shared_state_technical_2026-09-04.md").write_text(technical)


if __name__ == "__main__":
    main()

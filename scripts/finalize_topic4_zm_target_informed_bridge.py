#!/usr/bin/env python3
"""Finalize figures, provenance and the collaborator report after rev5 stops."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
TERMINAL = {
    "NO_STAGE1_ZM_ONLY_ELIGIBLE",
    "NO_SELECTION_CANDIDATE_CONFIRMED_2_OF_3",
    "FROZEN_CONFIRMATION_PASS",
    "FROZEN_CONFIRMATION_FAIL",
}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _fmt(value, digits=3):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _report(out, terminal_status):
    target = _load(out / "clinical_target.json")
    stage1 = (_load(out / "existing_candidate_rescore.json")
              if (out / "existing_candidate_rescore.json").exists() else None)
    selection = (_load(out / "selection_results.json")
                 if (out / "selection_results.json").exists() else None)
    confirmation = (_load(out / "confirmation_results.json")
                    if (out / "confirmation_results.json").exists() else None)
    nulls = (_load(out / "selection_aware_null.json")
             if (out / "selection_aware_null.json").exists() else None)
    lines = [
        "# Topic 4 rev5：data-driven Z/M 发作早期能量梯度拟合",
        "",
        f"终态：`{terminal_status}`",
        "",
        "## 科学问题",
        "",
        "在完整保留 Fig.4 学得的 node、E→E 和 E→I 底物后，只调整 Z/M 慢变量，判断模型能否先进入广泛、持续且频率升高的高活动态，再比较其最早合格状态窗与 E1146 发作早期的触点能量梯度。",
        "",
        "这是 target-informed development，不是盲预测，也不识别患者的 Z/M 生物机制。",
        "",
        "## 冻结临床目标",
        "",
        f"- 完整发作：{len(target['complete_seizure_indices'])}；开发目标：{len(target['development_seizure_indices'])}；展示 seizure：{target['display_seizure_idx']}。",
        "- 数值损失使用 matched 10–150 Hz；Fig.3 的 1–150 Hz 保留为临床主口径敏感性。",
        f"- seizure 2 parity：shared-A={target['display_parity']['shared_a_signed']:.6f}，direct rank={target['display_parity']['direct_early_rank_correlation']:.6f}。",
        "",
        "## Stage 1",
        "",
    ]
    if stage1 is None:
        lines.append("Stage 1 尚未形成可重评分 artifact。")
    else:
        primary = [row for row in stage1["records"] if row.get("primary_zm_only")]
        evaluable = [row for row in primary if row.get("status") == "BRIDGE_EVALUABLE"]
        lines.extend([
            f"- full-dose Z/M 候选：{len(primary)}；bridge-evaluable：{len(evaluable)}。",
            f"- 全部候选总数（含 2%/5% 等历史 edge-dose 对照）：{stage1['n_candidates']}。",
        ])
        if evaluable:
            best = min(evaluable, key=lambda row: row["J_early_bridge"])
            lines.append(
                f"- fit seed 最佳 full-dose 候选：`{best['candidate_id']}`，"
                f"J_early={best['J_early_bridge']:.3f}，"
                f"contact rho={best['field']['early_spearman']:.3f}。")
        else:
            qualified = [row for row in primary
                         if row.get("model_ictal_qualification")]
            frequency_pass = [
                row for row in qualified
                if (row["model_ictal_qualification"].get("clauses") or {}).get(
                    "contact_frequency_increased") is True
            ]
            duty_pass = [
                row for row in qualified
                if (row["model_ictal_qualification"].get("clauses") or {}).get(
                    "joint_broad_recruitment_duty") is True
            ]
            if frequency_pass:
                closest = max(
                    frequency_pass,
                    key=lambda row: row["model_ictal_qualification"]["joint_duty"])
                q = closest["model_ictal_qualification"]
                lines.append(
                    f"- 频率通过候选中最高一秒 recruitment duty："
                    f"`{closest['candidate_id']}`，duty={q['joint_duty']:.3f}，"
                    f"contact centroid shift={q['contact_centroid_shift_hz']:.2f} Hz。")
            if duty_pass:
                closest = max(
                    duty_pass,
                    key=lambda row: row["model_ictal_qualification"][
                        "contact_centroid_shift_hz"])
                q = closest["model_ictal_qualification"]
                lines.append(
                    f"- duty 通过候选中最大频率变化：`{closest['candidate_id']}`，"
                    f"duty={q['joint_duty']:.3f}，"
                    f"contact centroid shift={q['contact_centroid_shift_hz']:.2f} Hz。")
            comparators = [row for row in stage1["records"]
                           if row.get("edge_dose_comparator")]
            comparator_evaluable = [row for row in comparators
                                    if row.get("status") == "BRIDGE_EVALUABLE"]
            lines.append(
                f"- edge-expression 历史对照 bridge-evaluable："
                f"{len(comparator_evaluable)}/{len(comparators)}；这些臂不能成为 Z/M-only winner。")
    lines.extend(["", "## Selection 与 confirmation", ""])
    if selection is not None:
        for row in selection["candidate_summary"]:
            lines.append(
                f"- `{row['candidate_id']}`：eligible {row['n_eligible']}/{row['n_runs']}，"
                f"median J_early={_fmt(row['median_J_early_bridge'])}，"
                f"worst J_early={_fmt(row['worst_J_early_bridge'])}。")
    else:
        lines.append("未进入 selection 或 selection 未完成。")
    if confirmation is not None:
        lines.append(
            f"- 冻结候选 `{confirmation['candidate_id']}`：confirmation eligible "
            f"{confirmation['n_eligible']}/{confirmation['n_seeds']}。")
        for row in confirmation["records"]:
            lines.append(
                f"- seed {row['seed']}：{row['status']}，J_early={_fmt(row.get('J_early_bridge'))}，"
                f"rho={_fmt((row.get('field') or {}).get('early_spearman'))}。")
    lines.extend(["", "## Selection-aware null", ""])
    if nulls is None:
        lines.append("没有可评价的 primary winner，因此未运行 selection-aware null。")
    else:
        for row in nulls["nulls"]:
            lines.append(
                f"- {row['mode']}：observed min J_early={row['observed_minimum_J_early']:.3f}，"
                f"null median={row['null_median']:.3f}，P={row['lower_tail_p']:.4f}。")
    lines.extend([
        "", "## 结论边界", "",
        "- `FROZEN_CONFIRMATION_PASS` 仅支持一个开发阶段、患者目标导向的 Z/M 工作点。",
        "- 若 full-dose Z/M 未通过，2%/5% E→I 轨迹仍只能说明 fast-substrate expression 会移动动力学边界，不能等效解释为 Z/M 参数。",
        "- full-dose 候选若未先通过模型内部资格，就不计算患者损失；因此该负结果不是患者目标拟合失败，而是拟合前的动力学容量失败。",
        "- 模型 current proxy、模型频率和患者 SEEG 不共享物理单位；本轮比较的是 baseline-normalized 接触点能量组织。",
        "- 没有新的未见患者或未见发作单元，因此不作泛化声明。",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_target_informed_bridge_v1.json")
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = _load(config_path)
    out = ROOT / config["output_root"]
    controller = out / "selection_controller.json"
    while True:
        status = _load(controller).get("status") if controller.exists() else None
        if status in TERMINAL:
            break
        time.sleep(args.interval_seconds)
    if status == "FROZEN_CONFIRMATION_PASS":
        subprocess.run([
            PYTHON, "scripts/paper_figures/plot_fig5_target_informed_zm_bridge.py",
            "--config", str(config_path.relative_to(ROOT)),
        ], check=True, cwd=ROOT)
    report = _report(out, status)
    _atomic(out / "final_report.md", report)
    tracked = [
        config_path,
        out / "clinical_target.json",
        out / "existing_candidate_rescore.json",
        out / "selection_controller.json",
    ]
    tracked.extend(path for path in (
        out / "selection_results.json", out / "confirmation_results.json",
        out / "selection_aware_null.json", out / "WORKPOINT_TARGET_INFORMED_FROZEN.json",
    ) if path.exists())
    provenance = {
        "status": "TARGET_INFORMED_BRIDGE_FINALIZED",
        "terminal_status": status,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "sha256": {str(path.relative_to(ROOT)): _sha(path) for path in tracked},
        "finalized_epoch": time.time(),
    }
    _atomic(out / "final_provenance.json", json.dumps(provenance, indent=2) + "\n")
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "Topic 4 rev5", f"Finalized: {status}"],
                       check=False)
    print(json.dumps({"status": status, "report": str(out / 'final_report.md')}))


if __name__ == "__main__":
    main()

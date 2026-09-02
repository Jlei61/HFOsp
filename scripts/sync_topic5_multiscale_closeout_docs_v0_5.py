#!/usr/bin/env python3
"""Synchronize v0.5 status documents from frozen final artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
SPEC = ROOT / "docs/superpowers/specs/2026-08-13-topic5-patient-specific-multiscale-effective-scaffold-v0-5-design.md"
PLAN = ROOT / "docs/superpowers/plans/2026-08-13-topic5-patient-specific-multiscale-effective-scaffold-v0-5.md"
TOPIC = ROOT / "docs/topic5_seizure_subtyping.md"
INDEX = ROOT / "docs/archive/topic5/INDEX.md"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"expected one frozen status block in {path}, found {text.count(old)}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.replace(old, new, 1))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    claims = json.loads((out / "FINAL_CLAIM_ADJUDICATION.json").read_text())
    early = json.loads((out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json").read_text())
    if not (out / "CLOSEOUT_AUDIT.json").exists():
        raise RuntimeError("doc sync requires a completed machine closeout audit")
    audit = json.loads((out / "CLOSEOUT_AUDIT.json").read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("doc sync refuses a failed closeout audit")
    d1 = claims["claims"]["D1_CROSS_STATE_FIELD_CORRESPONDENCE"]
    d2 = claims["claims"]["D2_SHORTCUT_SPECIFIC_CROSS_STATE_CONTRIBUTION"]
    ei = claims["claims"]["PRIMARY_EARLY_NONLOCALITY_INTERACTION"]
    d1_summary = early["D1_L3_full_margin_gt_zero"]
    spec_old = (
        "> 状态：**合同已锁定并执行中：Stage A–E 完成，531/531 正式训练单元完成，Stage F target-free\n"
        "> attenuation/gain/field freeze 运行中；early-ictal target 仍物理封存。**"
    )
    spec_new = (
        "> 状态：**A–H 全流程完成：531/531 正式训练单元、Stage F target-free field/attenuation freeze、\n"
        "> 17 人/167 seizures locked internal benchmark、Figure 6 与 machine closeout audit 均完成；\n"
        "> 待用户终审后 commit/push。**"
    )
    replace_once(SPEC, spec_old, spec_new)
    plan_old = (
        "> 状态：**执行中：Stage A–E 已完成，531/531 正式单元完成；Stage F target-free\n"
        "> attenuation/gain/field freeze 运行中，target 尚未解封。**"
    )
    plan_new = (
        "> 状态：**A–H 已执行完成：531/531 正式训练、Stage F target-free freeze、17 人/167 seizures\n"
        "> locked internal benchmark、Figure 6/source data 与 standalone machine closeout audit 均完成；\n"
        "> 待用户终审后 commit/push。**"
    )
    replace_once(PLAN, plan_old, plan_new)
    topic_old = (
        "**2026-08-14 v0.5 执行状态（Stage A–E 完成，Stage F 运行中）**：full-parent builder 已自动扫描\n"
        "全部 34 位 K=2 parent，正式 spatial 分母为 **28 人/42 fits**，early-ictal metadata intersection 为\n"
        "**17 人/167 seizures**。531/531 正式训练单元完成，0 unresolved OOM、0 nonfinite；target 仍物理\n"
        "封存。唯一 target-free primary——cross-fitted nonlocality `J_lat` 与 heldout distal `L3-L2m`\n"
        "增益的 interaction——当前未确认（全 census Spearman rho=0.168，单侧 permutation P=0.195；\n"
        "去 6–7-contact、去最高 J 与去近一维几何敏感性均已在解封前冻结）。真实 prefix–suffix association\n"
        "对总体序列预测有稳定增量（L3 相对 suffix-reassignment 中位 +0.02368 nats，24/28，P=3.16e-5），\n"
        "但不特异集中在 distal transitions。`L2m` 是精确匹配 added degree、reciprocity 与 distance bins 后\n"
        "从头训练的 random-nonlocal control，不是 frozen rewiring。Stage F 正在冻结 arm-specific\n"
        "attenuation、gain-adjusted 与全部 model fields；只有其完成并写入授权 manifest 后，才读取 0–10 s、\n"
        "1–150 Hz broadband energy。后者仍是 `locked internal mechanistic follow-up`，不是独立确认。"
    )
    topic_new = (
        "**2026-08-14 v0.5 收口状态（A–H 完成，待用户终审）**：full-parent builder 扫描全部 34 位\n"
        "K=2 parent，正式 spatial 分母为 **28 人/42 fits**，locked early-ictal benchmark 为\n"
        "**17 人/167 seizures**。531/531 正式训练单元、Stage F target-free attenuation/gain/field freeze、\n"
        "synchronized null maps、Figure 6/source data 与 machine closeout audit 均完成。唯一 target-free\n"
        "primary——cross-fitted nonlocality `J_lat` 与 heldout distal `L3-L2m` gain 的 interaction——未确认\n"
        "（rho=0.168，单侧 permutation P=0.195）；真实 prefix–suffix association 改善总体预测\n"
        "（+0.02368 nats，24/28，P=3.16e-5），但不是 distal-specific。locked internal early-ictal D1\n"
        f"为{'支持' if d1['supported'] else '未支持'}（median null-relative margin={float(d1_summary['median']):+.4f}, "
        f"P={float(d1_summary['wilcoxon_p_greater']):.4g}）；early `(L3−L2m)×J` 为"
        f"{'支持' if ei['supported'] else '未支持'}，D2 shortcut-specific cross-state contribution 为"
        f"{'支持' if d2['supported'] else '未支持'}。该 benchmark 是 locked internal mechanistic follow-up，"
        "不是独立确认；broadband energy field 也不是 arrival/recruitment order。"
    )
    replace_once(TOPIC, topic_old, topic_new)
    index_old = (
        "### `multiscale_effective_scaffold_v0_5_closeout_2026-08-14.md` — **v0.5 正式 28 人/42 fits；Stage F/G/H 执行中**\n"
        "- full-parent builder 将早期手工预期 26 人/40 fits 修正为正式 **28 人/42 fits**；531/531 target-free training units 完成，17 人/167 seizures target 仍物理封存。\n"
        "- 真实 prefix–suffix association 对总体 heldout contact NLL 有稳定增量（24/28，`P=3.16e-5`），但 task-selected nonlocal 相对 macro-matched L2m 的 distal 优势及其与 cross-fitted nonlocality J 的 interaction 未确认。\n"
        "- Stage F 正在冻结 arm-specific attenuation、matched-local、gain-adjusted 与全部 model fields；closeout 文档目前是执行中骨架，必须等 explicit unseal、locked internal scoring、Figure 6 QA 与用户终审后定稿。"
    )
    index_new = (
        "### `multiscale_effective_scaffold_v0_5_closeout_2026-08-14.md` — **v0.5 A–H 完成；28 人/42 fits + 17 人/167 seizures**\n"
        "- 531/531 target-free training units、504 attenuation targets、126 matched-local searches、gain/field/null freeze、locked internal scoring、Figure 6/source data 和 machine closeout audit 已完成；待用户终审后 commit/push。\n"
        "- 真实 prefix–suffix association 改善总体 heldout contact NLL（24/28，`P=3.16e-5`），但 task-selected nonlocal 相对 L2m 的 distal 优势及 `J×gain` interaction 未确认。\n"
        f"- 17/167 early-ictal D1 为{'支持' if d1['supported'] else '未支持'}；primary early interaction 为"
        f"{'支持' if ei['supported'] else '未支持'}；D2 shortcut-specific contribution 为"
        f"{'支持' if d2['supported'] else '未支持'}。只称 locked internal mechanistic follow-up，不称 independent confirmation。"
    )
    replace_once(INDEX, index_old, index_new)
    print(json.dumps({"status": "PASS", "updated": [str(SPEC), str(PLAN), str(TOPIC), str(INDEX)]}, indent=2))


if __name__ == "__main__":
    main()

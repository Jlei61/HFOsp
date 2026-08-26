#!/usr/bin/env python3
"""Aggregate R1.2b seed-first and write the three-goal route audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2b import (
    R1_2B_REVISION, R1_2B_SUBJECTS,
)


ARMS = ("joint_explicit", "joint_explicit_raw")
SEEDS = (0, 1, 2)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE":
        raise ValueError(f"incomplete artifact: {path}")
    if value.get("sealed_opened") is not False:
        raise ValueError(f"sealed flag is not false: {path}")
    if value.get("r1_2b_revision") != R1_2B_REVISION:
        raise ValueError(f"revision mismatch: {path}")
    if contract.sha256_file(value["checkpoint"]) != value["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    return value


def _median(values) -> float:
    finite = np.asarray(values, dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("patient-first endpoint contains non-finite values")
    return float(np.median(finite))


def _summary(rows: list[dict], key: str, *, favourable: str = "negative") -> dict:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "n_patients": int(len(values)), "median": float(np.median(values)),
        "n_favourable": int(np.sum(values < 0) if favourable == "negative" else np.sum(values > 0)),
        "favourable_direction": favourable,
        "values": values.tolist(),
    }


def _fmt(value) -> str:
    return f"{float(value):+.6f}"


def main() -> None:
    root = contract.RESULT_ROOT / "r1_2b"
    r1_2_root = contract.RESULT_ROOT / "r1_2"
    rows: list[dict] = []
    raw_seed_pairs: dict[str, list[float]] = {}
    for subject in R1_2B_SUBJECTS:
        arm_runs = {
            arm: [
                _load(root / "joint" / subject / f"{arm}_seed_{seed}" / "result.json")
                for seed in SEEDS
            ]
            for arm in ARMS
        }
        raw_seed_pairs[subject] = [
            arm_runs["joint_explicit_raw"][index]["final_validation"]["filtered"]["joint_nll_per_event"]
            - arm_runs["joint_explicit"][index]["final_validation"]["filtered"]["joint_nll_per_event"]
            for index in range(len(SEEDS))
        ]
        row = {"subject": subject}
        for arm, prefix in (
            ("joint_explicit", "explicit"),
            ("joint_explicit_raw", "raw"),
        ):
            runs = arm_runs[arm]
            endpoint = {
                "filtered_minus_no_state": [
                    run["contrasts"]["filtered_minus_no_state_joint_nll"] for run in runs
                ],
                "filtered_minus_validation_off": [
                    run["contrasts"]["filtered_minus_validation_correction_off_joint_nll"] for run in runs
                ],
                "timing_minus_validation_off": [
                    run["contrasts"]["filtered_minus_validation_correction_off_timing_nll"] for run in runs
                ],
                "mark_minus_validation_off": [
                    run["contrasts"]["filtered_minus_validation_correction_off_mark_nll"] for run in runs
                ],
                "matched_filtered_minus_wrong_time": [
                    run["contrasts"]["matched_filtered_minus_wrong_time_joint_nll"] for run in runs
                ],
                "joint_minus_frozen": [
                    run["frozen_r1_2_reference"]["joint_minus_frozen_filtered_nll"] for run in runs
                ],
                "selected_epochs": [run["selected_epochs"] for run in runs],
                "final_raw_gain": [run["final_raw_gain"] for run in runs],
            }
            for horizon in (5, 10, 20):
                endpoint[f"h{horizon}_off_minus_filtered"] = [
                    run["horizon_correction_off"]["horizons"][str(horizon)]
                    ["correction_off_minus_filtered"]["joint_nll_per_event"]
                    for run in runs
                ]
                endpoint[f"h{horizon}_off_minus_filtered_mark"] = [
                    run["horizon_correction_off"]["horizons"][str(horizon)]
                    ["correction_off_minus_filtered"]["mark_nll_per_event"]
                    for run in runs
                ]
            for key, values in endpoint.items():
                row[f"{prefix}_{key}"] = _median(values)
        row["raw_minus_explicit_filtered"] = _median(raw_seed_pairs[subject])
        cache = json.loads((root / "cache" / subject / "manifest.json").read_text())
        row["train_anchors"] = cache["n_train_anchors"]
        row["validation_anchors"] = cache["n_validation_anchors"]
        rows.append(row)

    report_root = root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "r1_2b_patient_first.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    endpoints = [
        key for key in rows[0]
        if key not in {"subject", "train_anchors", "validation_anchors"}
    ]
    summary = {
        "status": "COMPLETE", "contract": contract.REVISION,
        "r1_2b_revision": R1_2B_REVISION,
        "n_subjects": len(rows), "n_seeds_per_arm": len(SEEDS),
        "n_joint_fits": len(rows) * len(ARMS) * len(SEEDS),
        "patient_first": {
            key: _summary(
                rows, key,
                favourable=("positive" if "off_minus_filtered" in key else "negative"),
            )
            for key in endpoints
        },
        "rows": rows,
        "seed_paired_raw_minus_explicit": raw_seed_pairs,
        "ordinary_negative_results_are_not_gates": True,
        "sealed_opened": False,
    }
    contract.atomic_json(report_root / "r1_2b_summary.json", summary)

    r0_card_path = contract.R0_RESULT_ROOT / "manifests/HYPOTHESIS_EVIDENCE_CARD.json"
    r0_audit_path = contract.R0_RESULT_ROOT / "manifests/FINAL_PACKAGE_AUDIT.json"
    r0_card = json.loads(r0_card_path.read_text())
    r0_audit = json.loads(r0_audit_path.read_text())
    r0_h2a = r0_card["hypotheses"]["H2a"]
    r0_h2b = r0_card["hypotheses"]["H2b"]
    r0_h3a = r0_card["hypotheses"]["H3a"]
    r0_h3b = r0_card["hypotheses"]["H3b"]
    r1_2_summary_path = r1_2_root / "reports/r1_2_summary.json"
    r1_2_summary = json.loads(r1_2_summary_path.read_text())
    r1_2_audit_path = r1_2_root / "manifests/FINAL_PACKAGE_AUDIT.json"
    r1_2_audit = json.loads(r1_2_audit_path.read_text())

    patient = summary["patient_first"]
    table = [
        "| 患者 | joint-exp: filtered−no-state | joint-exp: filtered−off | joint-exp: filtered−wrong | joint-raw−joint-exp | H20 off−filtered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['subject']} | {_fmt(row['explicit_filtered_minus_no_state'])} | "
            f"{_fmt(row['explicit_filtered_minus_validation_off'])} | "
            f"{_fmt(row['explicit_matched_filtered_minus_wrong_time'])} | "
            f"{_fmt(row['raw_minus_explicit_filtered'])} | "
            f"{_fmt(row['explicit_h20_off_minus_filtered'])} |"
        )
    route_verdict = (
        "三步没有改变 H1–H3 的问题定义，属于同一路线上的逐级校正：R0.1 关闭错误工具，"
        "R1.2 把发生时刻、mark 与完整 recorded support 接入同一个 likelihood，R1.2b 再补上"
        "R1.2 漏掉的有限联合对齐。科学问题没有偏离；但 R1.2b 只微调空间汇聚尾层，并不是"
        "原计划设想的 raw temporal last-block end-to-end 检验。raw 上游仍是 epoch-0 特征，"
        "这是一项明确的实现覆盖缺口，不能被工程 PASS 掩盖。"
    )
    plain = f"""# Continuous marked-state 三阶段路线审计：白话版

## 一句话

{route_verdict}

R1.2b 的三位患者结果如下。表中前四列负值有利于左侧 arm；最后一列正值表示关闭未来
background correction 后更差，即持续读取背景 observation 有帮助。

{chr(10).join(table)}

## 三次 goal 分别解决了什么

### Goal 1：R0.1 验收与止错

R0.1 最重要的产出不是一个新的“慢状态阳性”，而是把历史信息、raw/spectral bridge、
H3-S0 exposure screen 和不能支持的机制结论分层。冻结结论是：H2a 的开发级证据最强；
H1 只支持历史预测记忆，尚不是 raw-informed autonomous physical-time state；H2b 有提示但
未解决；H3a 支持 recent multi-IED termination/extent memory，但事件次数已足够解释，
generator edge 与 H3b 未解决。工程审计为 `{r0_audit['all_pass']}`。它没有把频谱预测或
有限窗口结果继续冒充 H1–H3 的最终模型。

### Goal 2：R1.2 full-anchor exact event model

R1.2 在6位患者、全部可读 development anchors、完整 recorded intervals 上联合评分下一
IED 的发生时刻和 mark。history timing 稳定胜 static（6/6，中位
{_fmt(r1_2_summary['patient_first']['history_timing_minus_static']['median'])}），但 persistent
state 的 filtered−no-state、validation-off 和 wrong-time 三把尺子均只有2/6有利、中位0；
raw−explicit 也近零。它把“只学 contact shape、没有 time likelihood”的旧偏差纠正了，
但冻结 observer 且本轮三位 Bridge 均选 epoch 0，使这个阴性只约束该配置。

### Goal 3：R1.2b 有限联合微调

R1.2b 没有换问题、换分母或加辅助 loss，只让最后空间汇聚层以 state LR 的0.1倍在同一个
IED likelihood 上对齐；上游 explicit/raw 编码仍冻结。三 seed 先在患者内取中位数。

- joint-exp filtered−no-state：{patient['explicit_filtered_minus_no_state']['n_favourable']}/3有利，患者中位 {_fmt(patient['explicit_filtered_minus_no_state']['median'])}；
- joint-exp filtered−validation-off：{patient['explicit_filtered_minus_validation_off']['n_favourable']}/3有利，患者中位 {_fmt(patient['explicit_filtered_minus_validation_off']['median'])}；
- joint-exp matched filtered−wrong-time：{patient['explicit_matched_filtered_minus_wrong_time']['n_favourable']}/3有利，患者中位 {_fmt(patient['explicit_matched_filtered_minus_wrong_time']['median'])}；
- joint-raw−joint-exp：{patient['raw_minus_explicit_filtered']['n_favourable']}/3有利，患者中位 {_fmt(patient['raw_minus_explicit_filtered']['median'])}；
- H20 correction-off−filtered：{patient['explicit_h20_off_minus_filtered']['n_favourable']}/3为正，患者中位 {_fmt(patient['explicit_h20_off_minus_filtered']['median'])}。

620 的三 seed 全部选择 epoch 0，所以整行严格为0；958和黄瀚文三 seed 均选择 epoch 4。
因此本轮最稳妥的读法是：有限联合对齐在2/3患者把 frozen 阴性变成了 predictive-filter
增量，而且未来 correction 对 mark 有用；但0/3患者通过 matched wrong-time，raw residual
也基本为0。**这不支持“正确时刻专属的 raw-informed state”，更不支持 autonomous state。**

## 对 H1–H3 的当前边界

- **H1：** R0.1 支持 event history 有预测信息。本轮2/3有 filtered/off/H20 增量，但0/3
  通过 matched wrong-time，所以当前只支持部分患者的 predictive filter，不支持
  time-specific persistent state。
- **H2a：** 只看 mark，且必须把 group-size/STOP 与 subset identity 分开。timing 阳性不能
  代答 recruitment field。旧图零假设在34人中 real decoder 胜 degree-preserving rewire
  {r0_h2a['decoder_graph_degree_rewire']['n_favourable']}/34（中位
  {_fmt(r0_h2a['decoder_graph_degree_rewire']['median_delta'])}），仍是开发级 predictive
  spatial-identity 证据，不是本轮 persistent-state 复现。R1.2b mark correction 在2/3有利，
  但 wrong-time 在0/3有利，因此仍不能称 state-dependent recruitment。
- **H2b：** 上一版 decoder readout 在361次发作、27人中为 +0.446 SD、20/27同向，
  p=0.019，但高可观测性203次更弱且严格六维 matching 为0/361可行；状态含义与混杂仍
  未解决。本次三个 goal 都没有重训 seizure probe，所以这条不升级也不作废。
- **H3a：** 既有 H3-S0 结果支持分布式 recent-event exposure 在 mark 上胜单个 current IED，
  主要由 STOP、部分 participation-rank 承担；真实 elapsed-time decay 没有稳定胜 matched
  event-count。它仍是预测筛查，未实现 exposure→persistent-state generator edge，也无法排除
  潜在状态同时造成连续相似 IED 的替代解释。R1.2b 没有触碰这条边，因此不升级也不推翻。
- **H3b：** 旧 case-crossover 为
  {r0_h3b['accepted_previous_case_crossover']['n_favourable_patients']}/{r0_h3b['accepted_previous_case_crossover']['n_patients']}
  人同向、p={r0_h3b['accepted_previous_case_crossover']['two_sided_sign_p']:.1f}；它不是
  T2-specific frozen-state probe，当前仍无支持。

## 路线偏差判断

**没有科学问题层面的偏离，但 R1.2b 只完成了原计划的一部分。** 已完成的每一步都在把模型
从“预测事件形状”拉回“连续背景→持续状态→下一事件 timing+mark”的原目标。当前最大缺口
不是多做 null，而是上游 raw observer 从未真正学会为 IED likelihood 提取背景信息。下一步
应做可学习 raw temporal last block 的小规模确认，同时把 H3 作为独立探索分支进入最小
T2 exposure→state matrix；两条线不应以普通阴性互相 gate。
"""
    (report_root / "combined_route_audit_plain_2026-08-25.md").write_text(plain)

    technical = f"""# Continuous marked-state 三阶段路线审计：技术版

## 1. 审计结论

{route_verdict}

## 2. 产物与完整性

- R0.1 final audit：`{r0_audit_path}`，all_pass={r0_audit['all_pass']}，sealed=false；
- R1.2 final audit：`{r1_2_audit_path}`，status={r1_2_audit.get('status')}，sealed=false；
- R1.2b：3 subjects × 2 arms × 3 seeds = 18 fits；本报告按 seed→patient 两级聚合；
- R1.2b 使用 R1.2 相同 `FullAnchorDesign`、baseline checkpoint、coverage 与 split hash；
- observer trainable set 限于 pool token、单层 spatial Transformer、output LayerNorm，raw arm
  另含 scalar raw_gain；observer/state LR=3e-5/3e-4；
- H5/H10/H20 为 event-observed correction-off，真实未来 IED 更新 deterministic history，
  recorded gaps 从四点 Gauss survival integral 排除；重叠窗口仅作 supportive diagnostic。

## 3. 患者优先结果

{chr(10).join(table)}

完整患者×endpoint 数值：`{csv_path}`；机器摘要：`{report_root / 'r1_2b_summary.json'}`。

## 4. 解释约束

1. R0.1 的常规 observation prototype 与 R1.2 exact event model 不是同一仪器，数值不可直接
   合并；这里只审计论证阶梯。
2. R1.2 的六人 state 阴性来自 frozen observer；固定三位 Bridge 都选 epoch0。R1.2b 仅使
   最后空间块适配 IED likelihood，上游 raw temporal features 仍是未训练初始特征。
3. R1.2b 若胜 frozen R1.2，定位为 target-alignment bottleneck；若不胜，只否定这一级有限
   微调，不否定 trained raw backbone、长背景 context 或其他 observation family。
4. filtered、wrong-time 与 correction-off 必须分层。只有 filtered 阳性时称 predictive
   filter；matched wrong-time 同向后称 time-specific estimate；H5/H10/H20 仍支持后才接近
   controlled predictive state。fully-generative rollout 本轮未做。
5. H2b/T2/H3 未运行；development partition 没有被扩成 formal cohort。
6. R1.2b 中620三 seed 均选择epoch0；958与黄瀚文选择epoch4。患者优先结果为2/3
   filtered/off/H20有利、0/3 matched wrong-time有利、raw−explicit中位近0。因此 joint
   alignment 瓶颈只得到部分支持，time-specific/raw-informed state 不支持。

## 5. 与原 H1–H3 的对齐矩阵

| 原问题 | R0.1 | R1.2 | R1.2b | 当前缺口 |
|---|---|---|---|---|
| H1 背景中是否有持续预测状态 | history/bridge 开发筛查 | exact timing+mark、full support、frozen observer | limited joint alignment + H5/H10/H20 | trained raw temporal observer；必要时 fully-generative supportive rollout |
| H2a 状态是否改变下一 IED recruitment | 开发级 mark 信号 | persistent mark 未稳定复现 | joint mark 与 wrong-time/off 拆分 | participation/subset 与 STOP 分离后的患者级复现 |
| H2b preictal state | suggestive/未决 | 未运行 | 未运行 | 冻结 state 后按 seizure mode 的 causal lead probe |
| H3 IED 是否塑造状态/网络 | H3-S0 25–200-event accumulation，count 足以 | T1 foundation only | T1 alignment only | T2 real/delayed/state-matched exposure→state edge，count vs physical |

## 6. 科学路线判定

- **对齐：** 时间发生过程、mark、recorded support、causal state carry、raw correction-off 都是
  原问题所需的承重部件；没有退回 contact-shape-only 或 frequency forecasting。
- **未越界：** 没把 history prediction 写成 latent mechanism；没把 H3-S0 写成 generator
  因果；没用三位 development patient 代替34人。
- **不足：** observer 的 raw upstream 学习覆盖不够，R1.2b 仍不是“raw end-to-end”完整检验；
  R1.2 mark decoder 当前报告 group-size/subset，但尚未恢复旧 sequential contact-RNN 的 order
  endpoint，因此 repertoire 结论仍受限。

## 7. 下一步建议

本轮结束后不再围绕 R1.2 frozen arm 加防御性 null。优先做一个固定三患者、三seed的
`joint_raw_last_temporal_block`：在相同 exact likelihood 下解冻 raw temporal Transformer 最后
一层和空间尾层，仍用0.1× LR、同分母、同H5/H10/H20；这是判断“raw信息不存在”与“raw从未
学会”之间最短实验。与此同时，H3作为独立探索分支按已冻结25/50/100/200/400-event证据进入
最小 T2：先做 load 与 participation 两种 source 的 real/delayed/state-matched、event-count
与 matched physical，普通阴性不 gate 另一 source/clock。正式34人与H2b继续保持封存。
"""
    (report_root / "combined_route_audit_technical_2026-08-25.md").write_text(technical)

    provenance = {
        "status": "COMPLETE", "r1_2b_revision": R1_2B_REVISION,
        "r0_evidence_card": str(r0_card_path),
        "r0_evidence_card_sha256": contract.sha256_file(r0_card_path),
        "r0_final_audit": str(r0_audit_path),
        "r0_final_audit_sha256": contract.sha256_file(r0_audit_path),
        "r1_2_summary": str(r1_2_summary_path),
        "r1_2_summary_sha256": contract.sha256_file(r1_2_summary_path),
        "r1_2_final_audit": str(r1_2_audit_path),
        "r1_2_final_audit_sha256": contract.sha256_file(r1_2_audit_path),
        "r1_2b_summary": str(report_root / "r1_2b_summary.json"),
        "r1_2b_summary_sha256": contract.sha256_file(report_root / "r1_2b_summary.json"),
        "plain_report": str(report_root / "combined_route_audit_plain_2026-08-25.md"),
        "plain_report_sha256": contract.sha256_file(report_root / "combined_route_audit_plain_2026-08-25.md"),
        "technical_report": str(report_root / "combined_route_audit_technical_2026-08-25.md"),
        "technical_report_sha256": contract.sha256_file(report_root / "combined_route_audit_technical_2026-08-25.md"),
        "scientific_route_deviation": False,
        "implementation_coverage_gap": "upstream raw observer remains epoch-zero frozen features",
        "sealed_opened": False,
    }
    contract.atomic_json(report_root / "combined_route_audit_manifest.json", provenance)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

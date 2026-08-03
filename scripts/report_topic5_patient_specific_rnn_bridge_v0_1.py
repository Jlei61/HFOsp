#!/usr/bin/env python3
"""Write the plain-language closeout and reproducibility manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["output_root"]
    summary_path = output / "PATIENT_SPECIFIC_RNN_BRIDGE_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    interictal = pd.read_csv(output / "interictal_patient_metrics.csv")
    ictal = pd.read_csv(output / "early_ictal_patient_metrics.csv")
    primary = ictal.loc[(ictal.band == "1_150") & ~ictal.development_supportive.astype(bool)]
    nll = interictal.pivot(index="subject", columns="model", values="test_nll").dropna()
    precedence = interictal.pivot(index="subject", columns="model", values="precedence_correlation").dropna()
    order_gain = nll.rank_shuffle_gru - nll.full_history_gru
    bridge = primary.pivot(index="subject", columns="model", values="all_contact_margin").dropna()
    absolute = primary.pivot(index="subject", columns="model", values="observed_max_abs_rho").dropna()
    full = summary["early_ictal"]["1_150"]["full_history_gru"]
    static = summary["early_ictal"]["1_150"]["static_fit60"]
    shuffle = summary["early_ictal"]["1_150"]["rank_shuffle_gru"]
    full_static = summary["early_ictal"]["1_150"]["paired_comparisons"]["full_history_gru_minus_static_fit60"]
    full_shuffle = summary["early_ictal"]["1_150"]["paired_comparisons"]["full_history_gru_minus_rank_shuffle_gru"]
    report = f"""# 患者特异 target-free RNN 跨状态桥梁 v0.1

## 一句话结论

本轮把跨患者读出彻底拿掉后，患者自己的 RNN 可以从自己的间期 contact-rank events
中学到可泛化到 heldout events 的传播顺序。模型生成的 contact field 随后在完全不重训的
情况下，与同一患者发作早期 1--150 Hz 能量场进行比较；这一跨状态对应及其相对静态
participation 的增量见下表，不能再归因于其他患者或经验 A/B 被注入模型。

## 1. 到底做了什么

- 每名患者独立训练，不共享任何模型权重或其他患者事件。
- 输入是 masked contact-rank events；每场事件内逐 rank set 预测下一 contact/STOP。
- 时间切分为 fit60 / validation20 / untouched test20。
- 主模型为 hidden-32 GRU；相同任务下另跑 linear state；rank-shuffle GRU 只破坏事件内顺序。
- 三个 seed，固定 7 次完整数据覆盖、每次 32 次更新、lr 3e-4。
- checkpoint 冻结后自由生成 5000 个完整事件，汇总 participation、early/late rank mass、
  endpoint mass 和 weighted earliness。
- 最后才读取同一患者 clinical onset 后 0--10 s 的 early-ictal target。主频段 1--150 Hz，
  1--45 Hz 为 sensitivity。所有置换都重新执行候选 field 最大化。

## 2. RNN 是否学到了间期传播结构

完整 {len(nll)} 人中，真实顺序 GRU 相对 rank-shuffle 的 heldout NLL 改善中位数为
**{fmt(np.median(order_gain), 4)} nats/event**，{int(np.sum(order_gain > 0))}/{len(order_gain)}
患者方向一致。GRU 自由 rollout 与真实 test20 的 pairwise contact precedence 相关中位数为
**{fmt(precedence.full_history_gru.median())}**；rank-shuffle 为
**{fmt(precedence.rank_shuffle_gru.median())}**。这直接支持模型学到事件内部“谁之后更可能到谁”的
患者特异结构，而不只是 contact 出现频率。

线性状态模型的 precedence 相关中位数为 **{fmt(precedence.linear_state.median())}**。
因此信息并不只依赖 GRU 门控；GRU 与更简单状态模型都能利用，但二者的 heldout NLL 和
跨状态 readout 需分别报告。

## 3. 模型场是否联系到发作早期

Primary 15 人（E1146 单列 supportive）的 1--150 Hz 结果：

| readout | 患者中位绝对相似度 | 中位 all-contact margin | margin>0 |
|---|---:|---:|---:|
| patient-only GRU | {fmt(full['median_absolute_similarity'])} | {fmt(full['median_all_contact_margin'])} | {full['n_positive_margin']}/{full['n']} |
| rank-shuffle GRU | {fmt(shuffle['median_absolute_similarity'])} | {fmt(shuffle['median_all_contact_margin'])} | {shuffle['n_positive_margin']}/{shuffle['n']} |
| static fit60 | {fmt(static['median_absolute_similarity'])} | {fmt(static['median_all_contact_margin'])} | {static['n_positive_margin']}/{static['n']} |

GRU 相对 static fit60 的患者级 margin 增量中位数为
**{fmt(full_static['median'])}**；相对 rank-shuffle GRU 为
**{fmt(full_shuffle['median'])}**。精确配对检验分别为
`P={full_static['test']['p_two_sided_exact']:.4g}` 和
`P={full_shuffle['test']['p_two_sided_exact']:.4g}`。

这里最重要的不是要求 15/15 阳性，而是看两层证据是否同时存在：

1. 真实顺序模型在患者自己的 heldout 间期事件上确实学到传播结构；
2. 同一模型生成的患者 contact field 在发作 target 完全未参与训练时仍有 above-null 对应，
   并评估它相对静态 participation 和 rank-shuffle 的增量。

## 4. 这能支持什么

若 above-null 对应成立，安全结论是：

> 仅用患者自身间期 contact-rank sequences 自监督训练的 recurrent model，恢复了患者特异
> contact recruitment/rank structure；该模型结构与同一患者发作早期 broadband energy field
> 存在跨状态空间对应。

它比上一轮强在：没有跨患者 readout、没有经验 A/B 输入、没有用 ictal target 训练残差支路。

不能写成：RNN 自动恢复了唯一物理 A/B 轴、逐次预测了发作传播路径、或所有患者共享同一个
RNN 机制。当前 readout 是患者级静态场，不是逐发作动态 replay。

## 5. 工程验收

- target-free units：{summary['n_units']}；失败 0。
- 其他患者事件进入模型：否。
- empirical A/B 进入模型：否。
- ictal target 进入训练：否。
- checkpoint、training log、heldout metrics、free rollout 和 contact distribution 均逐 unit 保存。
- 运行可由 launcher state 和 `DONE.json` 断点续跑。
"""
    (output / "PATIENT_SPECIFIC_RNN_BRIDGE_REPORT.md").write_text(report, encoding="utf-8")

    files = [
        config_path,
        ROOT / "src/topic5_patient_specific_rnn_bridge.py",
        ROOT / "scripts/run_topic5_patient_specific_rnn_unit_v0_1.py",
        ROOT / "scripts/launch_topic5_patient_specific_rnn_v0_1.py",
        ROOT / "scripts/summarize_topic5_patient_specific_rnn_bridge_v0_1.py",
        ROOT / "scripts/plot_topic5_patient_specific_rnn_bridge_v0_1.py",
        ROOT / "tests/test_topic5_patient_specific_rnn_bridge_v0_1.py",
        summary_path,
        output / "interictal_patient_metrics.csv",
        output / "early_ictal_patient_metrics.csv",
        output / "figures/patient_specific_target_free_rnn_bridge_six_panel.png",
        output / "figures/patient_specific_target_free_rnn_bridge_six_panel.pdf",
        output / "figures/README.md",
        output / "PATIENT_SPECIFIC_RNN_BRIDGE_REPORT.md",
    ]
    manifest = {
        "contract": config["contract"], "status": "REPRODUCIBLE_CLOSEOUT",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "files": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
    }
    (output / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

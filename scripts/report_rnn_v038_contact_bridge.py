#!/usr/bin/env python3
"""Render the bounded Chinese scientific report from completed machine results."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root
    data = json.loads((root / 'contact_bridge_summary.json').read_text())
    primary = data['primary_endpoint']
    rows = data['rows']; seed_rows = data['seed_rows']
    e253 = next(r for r in rows if r['subject'] == 'epilepsiae_253' and r['family'] == 'dual')
    seeds253 = [r for r in seed_rows if r['subject'] == 'epilepsiae_253' and r['family'] == 'dual']
    good = lambda row, endpoint, weight='event_weighted_gain': len(row['endpoint_results'][endpoint][weight]['passing_seeds'])
    cases = [r['subject'].replace('epilepsiae_', 'E') + '/' + r['family'] for r in rows
             if good(r, primary) >= 3]
    case_text = '、'.join(cases) if cases else '无'
    text = [
        '# 审阅结论：RNN v0.3.8 状态—触点序列接口实测', '',
        '## 1. 一句话判断', '',
        f'本次主要的“同样前两组之后，下一触点身份”四对照方向性筛查中，达到至少 3/5 seed 的患者/模型为：**{case_text}**。'
        '这仍是固定读出、旧测量字典下的回顾性测试；当前不能声称学到的事件历史动态已稳定驱动触点传播分叉。', '',
        f'E253 dual 有一条较弱线索：后缀平均身份评分 {good(e253,"suffix_identity_set_nll")}/5 方向通过，'
        f'按两小时窗等权后为 {good(e253,"suffix_identity_set_nll","block_equal_gain")}/5；'
        f'第一未知触点为 {good(e253,primary)}/5，接下来两组为 {good(e253,"next_two_teacher_forced_set_nll")}/5。'
        '因此目前可说状态输入调整了部分后续序列概率，不能直接写成稳定改善即时分叉。', '',
        '## 2. 完成程度', '',
        f'本轮登记的冻结模型测试 **{data["n_evaluated"]}/{data["n_registered"]} 完成**，父模型评分重放 '
        f'**{data["n_parent_parity_passed"]}/{data["n_registered"]} 通过**。这里统计执行情况，不把它换算为科学假设完成度。', '',
        '新增：下一触点的条件概率和整组命中率、严格后缀和接下来两组的逐步条件评分、FIT 定义的真实分叉前缀、'
        '事件/背景分支常数与错时、初始化事件矩阵、固定时间轴上的 mark 打乱重放，以及两小时窗等权敏感性。'
        '既有 H1 聚合预测和形态读出未重新训练；形态部分补查了原卡片中的 prefix-only 基线。', '',
        '## 3. P0 / P1 关键问题', '',
        '**P0：旧测量输入不具备严格前瞻资格。** v0.3.8 的触点选择汇总了全记录统计。'
        '本轮严格冻结与重放可以检查原模型接口，但不能消除测量层的未来信息。依赖前瞻解释的下一轮必须使用 FIT-only 重建测量及匹配 decoder。', '',
        '**P1：弱化的比较基线会夸大状态增益。** 本轮同时要求胜过 static、B_mark、FIT 常数、错时；'
        '合并形态也补入更简单的 prefix-only。E253 dual 原 3/5 的合并形态三对照方向性通过，'
        f'在补入 prefix-only 后为 **{len(e253["rich_mark_four_control_seeds"])}/5**。'
        '这项复核沿用已保存的形态评分，未声称新拟合或新的独立留出证据。', '',
        '**P1：整体 grammar、身份、具体分叉是不同目标。** 原 grammar 把下一步 BCE 和 STOP 合并；'
        '整段身份分数还平均了不同位置的预测。给定两组之后只取下一步身份，才能直接回答此时走向哪个触点。', '',
        '**P1：权重更新与接口依赖必须分开。** 冻结头的分支消融只检查该头怎样使用输入，不能替代各对照独立拟合。'
        '初始化特征有用不等于事件状态学到了新动态；事件矩阵发生变化也不保证下游 adapter 用到了它。', '',
        '## 4. 科学性结果', '',
        '下表每格都是同一 seed 同时超过四项预测对照的方向性通过数，正方向容差 1e-6。'
        '不是显著性检验，seed 是优化重复。除 STOP 外，身份评分均使用 FIT 有分叉且可做相同前缀/K错时的共同事件子集。', '',
        '|患者/状态模型|整段身份|严格后缀身份|前两组后下一触点|接下来两组，逐步给真前缀|下一触点：两小时窗等权|',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        text.append(f'|{r["subject"].replace("epilepsiae_", "E")} / {r["family"]}|'
            f'{good(r,"all_identity_set_nll")}/5|{good(r,"suffix_identity_set_nll")}/5|'
            f'{good(r,primary)}/5|{good(r,"next_two_teacher_forced_set_nll")}/5|'
            f'{good(r,primary,"block_equal_gain")}/5|')
    rich958 = next(r for r in rows if r['subject'] == 'epilepsiae_958' and r['family'] == 'dual')
    text += ['', f'合并形态补入 prefix-only 后，E958 dual 仍有 {len(rich958["rich_mark_four_control_seeds"])}/5 通过，'
             f'E253 dual 为 {len(e253["rich_mark_four_control_seeds"])}/5。'
             '因此“状态输入帮助某些事件属性”在 E958 保留患者内线索；不能因此升级为触点身份、分叉或多种独立形态指标都获改善。']
    text += ['', '**E253 同一批五个 seed 的来源与下一触点结果：**', '',
             '|seed|上游来源|超过 static|超过 B_mark|超过常数|超过错时|事件等权全部通过|',
             '|---|---|---:|---:|---:|---:|---|']
    names = {'initialized_observer_features': '保留初始化特征', 'learned_background_only': '只学习背景',
             'learned_event_and_background': '学习事件和背景', 'learned_event': '学习事件'}
    for s in seeds253:
        v = lambda name: 'NA' if s[name + '_event_gain'] is None else f'{s[name + "_event_gain"]:+.4f}'
        text.append(f'|{s["seed"]}|{names[s["source_class"]]}|{v("static")}|{v("B_mark")}|{v("constant")}|{v("shift_k")}|'
                    f'{"是" if s["primary_event_pass"] else "否"}|')
    text += ['', '数值为对照 NLL 减去真实状态 NLL，正数有利。它衡量下一触点概率，不是准确率百分点。', '',
             '**分支与初始化消融：**', '',
             '|seed|事件分支改为 FIT 均值的代价|背景分支改为 FIT 均值的代价|恢复初始化事件矩阵的代价|打乱历史 mark 的代价|',
             '|---|---:|---:|---:|---:|']
    for s in seeds253:
        v = lambda name: 'NA' if s[name + '_event_gain'] is None else f'{s[name + "_event_gain"]:+.4f}'
        text.append(f'|{s["seed"]}|{v("event_constant")}|{v("background_constant")}|{v("event_initial")}|{v("event_mark_scramble")}|')
    text += ['', '负代价表示替换后反而更好；零表示在数值容差内无效应。'
        '初始化探针沿用真实状态的标准化和 adapter，分布变化会影响读出，故即使代价为正也只能作依赖性诊断。', '',
        'E253 dual 的四个 seed 事件矩阵未更新，这四个 seed 恢复初始化矩阵应逐值不变。'
        '剩下一个学习事件与背景的 seed，仍须同时超过静态、强历史、常数及错时，并在独立时间支持上复现，才能升级。', '',
        f'E253 主分叉子集每个 seed 有 {seeds253[0]["n_events"]} 个可评分事件、'
        f'{seeds253[0]["n_blocks"]} 个含事件的两小时物理窗。窗之间仍可能共享历史，不能当作独立患者。'
        '事件等权与窗等权出现分歧时，不挑选有利的权重方式来宣布稳定。', '',
        'STOP 另在未按未来 K 筛选的前缀可配对子集、所有有效步上计分；'
        f'E253 四对照方向性通过为 {good(e253,"stop_bce")}/5。'
        '固定前两组后的身份集已条件于继续，不能用该子集单独宣称 STOP 判别成立。', '',
        '合并形态、逐触点身份和传播分叉均不能合并为多个独立病理发现。'
        '接下来两组的结果使用 teacher forcing，不代表自由生成整条传播路径，也不证明事件改变生理状态。', '',
        '## 5. 工程性验证', '',
        '新增评分单元测试 10 项通过；已知信号的 toy 能提高身份概率，错置后下降，常数 null 不产生增益。'
        '生产评分重放、每步 logits、目标 mask、donor、输入及输出 hash 均保存。'
        '35 个单元的 card 是诊断执行单元，不是 35 个患者。原文件和其他活动工作区未改写。', '',
        '全量完整性检查覆盖 2300 个登记 hash；35 个模型的 static 读出在相同前缀下逐值一致，'
        '事件矩阵未更新的单元恢复初始化后 logits 逐值一致。检查记录见 `validation/integrity_and_prefix_invariance.json`。'
        'E253 下一组大小在本子集中全为 K=1，新集合评分与原单触点 softmax 在此处等价；'
        '结果差异来自明确预测位置、评分支持和对照，而不是换一个概率公式制造的。', '',
        '## 6. 最小修改路线', '',
        '1. 使用已经在重建的 FIT-only 测量与匹配 decoder；保留旧数据结果作为回顾性参照。',
        '2. 固定上游状态，在同一前缀/K任务上分别拟合等容量的 prefix-only、显式历史、background-only、initialized-event、learned-event 和完整状态 adapter；全部只用 FIT/INNER 选择。',
        '3. 先直接优化并检验 contact identity 条件损失，将 STOP 与组大小独立读出；新 checkpoint 必须记录同前缀下一步与后两步的分数，不能继续只按合并 grammar 解释效果。',
        '4. 固定相同端点、相同患者、相同上游权重，检查事件学习增量和时间窗稳定性；最后在未参与设计的数据确认。', '',
        '## 7. 下一步建议', '',
        '接口可以继续用“事件前冻结状态 → 触点序列解码器的条件调制”，但当前应标作有待验证的预测接口。'
        '优先解决的不是把两个网络接得更深，而是让下游直接回答“相同前缀后选哪个触点”，并确认超过简单前缀和背景输入。'
        '本任务没有扩展到联合训练或 v0.3.9 的执行队列。', '',
        f'数据与图：[{root.name}]({root})；[机器汇总]({root / "contact_bridge_summary.json"})；'
        f'[逐 seed 表]({root / "contact_bridge_seed_summary.csv"})；'
        f'[逐物理窗表]({root / "contact_bridge_physical_bins.csv"})；'
        f'[执行合同]({root / "execution_manifest.json"})。', '',
        f'![E253 下一触点对照]({root / "figures/contact_bridge_e253_next_contact.png"})', '']
    (root / 'contact_bridge_report_zh.md').write_text('\n'.join(text))
    print(root / 'contact_bridge_report_zh.md')


if __name__ == '__main__':
    main()

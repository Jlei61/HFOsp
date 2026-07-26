# Topic 5 / Figure 6：set-valued transition-skeleton graph RNN（v0.8）

**日期**：2026-07-26
**状态**：**INVALIDATED**；真实 heldout contact rank 无并列，v0.8 假设不成立
**继承**：34人、masked contact rank、chronological 80/20、LOSO、
target sealed、clinical onset `[0,10] s`、BB `1–150 Hz`

## 1. 阻断性修正

本节是假设性描述，后被真实数据审计否定。原先假设一个 rank 可以同时包含多个触点。v0.7 的 teacher-forced loss
把正确下一 rank 当成“若干正确类别之一”，而 free rollout 每步只抽一个触点。
因此同 rank 触点被错误拆成虚假的先后顺序，free-rollout 的 rank distribution、
pairwise precedence 和 whole-path 指标均不能作为科学结果。

v0.7 已完成的结果仅保留为工程诊断，不进入统计或论文。

## 2. v0.8 自监督任务

每一步动作严格定义为“下一 rank 的完整触点集合”：

1. STOP 是独立二元判断；
2. 未停止时，对所有尚未参与的触点同时输出 Bernoulli 概率；
3. teacher forcing 使用 multi-label BCE 学习完整下一集合；
4. free rollout 同时采样多个触点，所有被采样触点写入同一 rank；
5. `group_count` 始终表示 rank-set 数量，不再混用为参与触点数；
6. 若采样集合为空，则从同一 Bernoulli 分布重新采样；有限次失败后才用最高概率触点
   作为数值 fallback。

模型输入仍只有 contact-level interictal rank event，不使用 A/B label、IEI 或发作数据。

## 3. 结构先验与 rank

沿用 v0.7 的 train80-only patient transition skeleton：

- rank 0：无 recurrent state；
- rank 1：患者特异双向多路径 skeleton；
- rank 2：显式 forward/reverse 两个共享参数的方向状态；
- rank 3：rank 2 + global recruitment；
- rank 4：rank 3 + local surround suppression。

无约束 full-rank/low-rank 结果只保留为 sensitivity。主结果必须比较真实患者路径、
无历史 rank 0 和保留轴与密度的 edge-weight shuffle。

## 4. 重新启动门

先做 3 位固定患者 × 3 seeds：

1. 检查生成事件确实含 multi-contact tied ranks；
2. 真实路径相对 rank 0 和 weight shuffle 的 heldout set NLL；
3. participation、conditional rank distribution、pairwise precedence；
4. label-free whole-path distance；
5. rank 1/2/4 的最小结构比较。

只有 precedence 与 whole-path 两项在多数 patient-seed 上同时改善，才重启 34 人。
若只改善 teacher-forced set NLL，则结论仍是“结构被使用，但未复现传播动力学”，
不得进入发作期。

## 5. 发作期边界

间期门通过前，禁止读取任何发作 target。通过后仍需先冻结模型、rank 和完整训练覆盖
合同，再评估 clinical onset `[0,10] s`、`1–150 Hz` baseline-robust-z 静态能量场。

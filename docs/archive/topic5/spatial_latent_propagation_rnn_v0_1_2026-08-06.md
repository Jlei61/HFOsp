# 患者特异空间潜变量传播 RNN v0.1 — 阶段报告

> 完整数字与逐患者表：`results/topic5_spatial_latent_propagation_rnn_v0_1/`
> （`CLOSEOUT_REPORT.md` / `cohort_statistics.json` / `RECOVERY_GATE.json`）
> spec：`docs/superpowers/specs/2026-08-06-topic5-spatial-latent-propagation-rnn-v0_1.md`

## 一句话

把状态从「每个触点一个节点」挪到「患者自己那张传播平面上的一片组织」，触点只当观测口。这样问了一个以前问不了的问题：模型能不能预测一个它从没训练过的触点。

## 这一版能说什么、不能说什么

先用**答案已知**的合成数据检验：造一张稀疏空间连接图、用它生成事件、再让模型把图找回来。三层分开评分，结论是——

- 哪些连接存在：**认不出**
- 活动整体往哪走：**认不出**
- 各组织块往前推的相对排序：**认得出**

所以「这位患者的连接图长这样、和别人不一样」以及「删掉模型认为重要的连接会怎样」这两类说法在这一版里**没有依据**——图是一大堆同样拟合得一样好的解里随便一个。预测层面的结论不受影响，因为它们不需要知道哪条连接是对的。

## 队列

- 21 位患者同时有冻结的事件记录和真实触点坐标（按精确名字对齐）。
- 每位患者的平面都是用**整段记录**估出来的，所以这一版是回溯性的，
  不能说明这套几何在记录之前就能知道。

## 完成度

- 队列单元：计划 315，完成 48
- 冻结配置：`{"microsteps": 6, "wiring_strength": 0.3, "edge_budget": 6.0}`

（内部归档代号：SLP-RNN v0.1, RECOVERY_GATE.json, FROZEN_CONFIG.json, leave_contact_out_summary.json, static_baseline_verification.json）
# Continuous marked-state R1.2b 有限联合微调冻结合同

**冻结日期：** 2026-08-25
**结果根：** `results/epi_prssm/continuous_marked_state/r1/r1_2b/`
**上游：** R1.2 六人 full-anchor/full-recorded-support 冻结结果

## 1. 科学问题

R1.2 将 Bridge observer 冻结后再训练 persistent state；本轮固定三位患者的
Bridge explicit 与 explicit+raw 均选择 epoch 0。R1.2b 只回答一个有限问题：

> R1.2 的持续状态阴性，是否主要由 observer 没有在同一个 IED timing+mark
> likelihood 上进行目标对齐造成？

本轮不回答 H2b，不加入 exposure→state edge，不运行 T2/H3，也不打开 sealed
partition 或扩到 34 人。

## 2. 不变合同

- 患者固定为 `epilepsiae_620`、`epilepsiae_958`、`yuquan_huanghanwen`；
- 沿用 R1.2 全部 admissible development anchors 与完整 TRAIN/validation
  recorded support；
- 事件按严格 `z(t-)` 评分；
- deterministic history timing/mark baseline 冻结且各 arm 共用；
- 唯一目标仍是 exact marked point-process timing+mark likelihood，无额外重建、
  consistency、seizure 或 contrastive loss；
- `state_dim=8`，state LR=`3e-4`，observer LR=`3e-5`；
- epoch 0 与 1–4 通过 TRAIN 内后20%选择，再按所选 epoch 在完整 TRAIN refit；
- seeds 固定为 0/1/2，科学聚合先在患者内取 seed 中位数，再跨患者描述；
- ordinary negative result 不阻断其他 arm，但任何 split/sealed、coverage、future
  leakage、hash 或 likelihood 合同错误使对应产物失效。

## 3. 唯一新增自由度

两臂为 `joint_explicit` 与 `joint_explicit_raw`。二者共享相同的 state、decoder、
history baseline、最后空间块初始化和训练预算。

只训练 observer 的最后空间汇聚块（单层 spatial Transformer、pool token、output
LayerNorm）；raw 臂额外训练一个从 0 初始化的标量 `raw_gain`。以下上游组件全部
冻结并预缓存为 per-contact node：explicit projection、coordinate projection、shaft
embedding、raw tokenizer、raw temporal Transformer。observer 参数组学习率严格为
state 参数组的 0.1 倍。

因此本轮若为阴性，只能否定“冻结的 epoch-0 上游特征 + 最后空间层有限联合对齐”这
一配置；不能否定从头或充分预训练的 raw backbone，也不能否定更长背景上下文。

## 4. 对照与读数

每个 arm/seed 报告：

1. filtered − initial no-state；
2. filtered − validation correction-off；
3. filtered − all correction-off；
4. filtered − same-session matched wrong-time；
5. timing、full mark、group-size/STOP、subset identity 分项；
6. joint arm − 对应冻结 R1.2 seed-0 arm；
7. raw − explicit 的同 seed 配对差；
8. H5/H10/H20 event-observed correction-off。

H5/H10/H20 使用对最大 H=20 均合格的同一组 validation anchors；若超过64个，按
时间顺序等距取64个，不按结果挑选。anchor 后关闭所有未来 background observation
correction，保留真实未来 IED timing/mark 对 deterministic history 的更新。每个窗口
重新将 validation recorded coverage 与 `[anchor, 第H个未来事件]` 相交，并在真实事件
处切开四点 Gauss quadrature；未记录缺口不进入积分。窗口允许重叠，因此该层是
supportive rollout diagnostic，不把重叠窗口当独立生物样本。

## 5. 允许的结论

- 联合臂稳定胜冻结臂：支持“R1.2 存在 observer-target alignment 瓶颈”；
- filtered 胜 no-state，但 correction-off/wrong-time 不支持：只称 predictive filter；
- filtered、matched wrong-time、validation-off 与 H5/H10/H20 同向：才升级为
  time-specific/controlled predictive state 的开发级证据；
- mark 改善必须拆分 group-size/STOP 与 subset identity，不能用 timing 或 STOP 代答
  repertoire shaping；
- 三位患者、三个 seed 仍是 development pilot，不作 34 人队列结论。

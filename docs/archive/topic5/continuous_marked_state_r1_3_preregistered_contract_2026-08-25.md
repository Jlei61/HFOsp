# Continuous marked-state R1.3 正式冻结合同

**冻结日期：** 2026-08-25
**结果根：** `results/epi_prssm/continuous_marked_state/r1/r1_3/`
**工程 smoke：** `r1_3_smoke/`，黄瀚文 seed 0、1+1 epoch，只证明运行与梯度覆盖，
不进入正式聚合。
**实现 revision：** `r1_3_full_raw_temporal_exact_target_isolated_increment_v2`；
v1 smoke 在正式结果前发现 raw 臂共同读出会额外训练，已仅作为失效诊断保留。

## 1. 唯一科学问题

> 在 exact IED timing + sequential tied-group mark likelihood 下，让完整 raw
> patch tokenizer 与两层 temporal Transformer 学习后，raw waveform 是否在
> trained explicit observer、deterministic event history 与跨窗口 persistent
> memory 之外增加预测信息？

本轮不回答 seizure、H2b、H3、34 人队列或 autonomous physiology。

## 2. 固定数据与分母

- 患者：`epilepsiae_620`、`epilepsiae_958`、`yuquan_huanghanwen`；
- seeds：0/1/2；
- arms：`explicit`、`explicit_raw`；
- 与 R1.2/R1.2b 完全相同的 full anchors、event rows、four-point quadrature、
  recorded coverage、history baseline、state dimension 8 和 development split；
- event 只用严格 pre-event `z(t-)`；
- TRAIN 尾部 20% chronological inner-validation 选择 0–4 epoch；
- development validation 不参与 epoch 选择；sealed partition 保持关闭；
- 聚合顺序为 seed -> patient，三人结果不称队列比例。

## 3. 配对初始化

每个患者/seed：

1. `explicit` 从同 seed 的 R1.2b joint-explicit persistent core 与 spatial tail
   初始化，并训练完整 explicit MLP、coordinate/shaft embedding 与 spatial fusion；
2. `explicit_raw` 从已完成的同 seed R1.3 explicit checkpoint 初始化；
3. raw gate 固定以 `0.02` 开始，避免精确零 gate 阻断 tokenizer 梯度；
4. raw 臂只开放 raw tokenizer、raw positional/valid projection、两层 temporal
   Transformer、raw norm/gate；explicit/coordinate/shaft、spatial fusion、state
   correction、generator 和所有 timing/mark readout 全部冻结；
5. raw 与 explicit 的差异因此严格定位为 raw residual 的增量，不能来自共同读出层
   多训练四轮。raw 全栈仍通过冻结的下游映射接受 exact event likelihood 梯度。
6. raw 的 epoch-0 选择基线临时令 gate=0，与 paired explicit 逐元素相同；进入训练前
   恢复 gate=0.02。若 inner-validation 最终选择 epoch 0，正式 checkpoint 也令 gate=0，
   因而 no-update 的 raw-explicit 差严格为 0，不包含随机 raw 残差。

## 4. 训练

- 唯一 loss：exact recorded-support timing likelihood + exact tied-group sequential
  mark likelihood；
- 不加入频谱、waveform、seizure、contrastive、KL 或 latent consistency loss；
- explicit observer-alignment 2 epoch：训练 explicit/coordinate/shaft、spatial
  fusion 与 state-to-output readout，冻结 generator 和 observation correction；
- explicit joint-alignment 2 epoch：额外开放 observation correction；
- paired raw 的两个阶段均只训练 raw 全栈与 raw gate，共同参数始终冻结；阶段名称仅
  保持调度与审计格式一致，不给 raw 臂额外的共同参数容量；
- continuous generator `K` 与 `mu` 全程冻结，避免 observer 坐标变化被新 generator
  吸收；
- LR：state/readout `3e-4`，explicit/spatial `3e-5`，raw temporal `1e-5`；
- raw observer 使用 AMP，state、matrix exponential 和 exact likelihood FP32；
- chronological truncated BPTT，初始 chunk 8；CUDA OOM 时整 fit 从同初始点以
  chunk 4/2/1 重试；
- explicit features 与 contact mask 每个 anchor 只缓存一次；raw waveform 仍从冻结
  minute-chunked Zarr 流式读取，不建立患者级内存副本。

## 5. 主要比较

每个 arm/seed 固定报告：

1. persistent - memoryless；
2. correct-time - 五个 same-session、至少相隔 30 min 的 matched wrong-time donor
   中位数；
3. raw persistent - paired explicit persistent；
4. timing、selecting size、terminal STOP、first-group subset、later-group
   continuation、same-prefix continuation；
5. raw tokenizer、temporal layer 0/1 的 selection-stage 最大梯度与最终参数更新量；
6. selected epoch 0 必须报告为 no-update，不解释为患者生物学阴性。

## 6. 允许结论

- persistent 胜 memoryless：跨窗口 predictive memory；
- correct 胜 strict matched wrong：time-specific persistent estimate；
- raw 胜 paired explicit：raw waveform 在显式统计之外有增量；
- 仅 STOP/size 改善：termination/extent memory；
- first/later subset 改善：recruitment/repertoire predictive memory；
- raw 梯度非零但 selected epoch 0：raw 路径被真实训练但未获 inner-validation 采用；
- 普通阴性不 gate T2-S1，H3 的结论强度由所用 pre-event T1 质量单独约束。

## 7. 无效条件

以下任一项只使对应 fit 无效并重跑，不把普通数值阴性当 blocker：

- formal split 被打开；
- design/coverage/cache/checkpoint hash 不一致；
- raw arm 任一 tokenizer/temporal block selection gradient 非有限或精确为 0；
- exact likelihood 出现 NaN/Inf；
- raw arm不是从同 seed completed explicit checkpoint 初始化；
- recorded gaps 被计入 survival integral。

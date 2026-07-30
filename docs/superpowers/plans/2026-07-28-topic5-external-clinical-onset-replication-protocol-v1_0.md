# Topic 5 external clinical-onset replication protocol v1.0 plan

## Activation 0：独立患者 inventory

1. 审计未使用缓存患者的 exact clinical-onset 与 target provenance；
2. 审计新 Epilepsiae/医院患者；
3. 区分 new patients 与 current-patient new seizures；
4. 无新患者时保持 `READY_BUT_BLOCKED_NO_INDEPENDENT_PATIENT_COHORT`。

## Activation 1：target 打开前冻结

1. 生成新患者 manifest；
2. 冻结 exact onset、`[0,10] s`、`1–150 Hz` target；
3. 冻结 raw field、regularized candidates、GRU/rank-shuffle/first-order；
4. 冻结 orientation-free primary、signed secondary；
5. 冻结 within-shaft primary null 与 geometry sensitivity；
6. 冻结 \(\delta_{\mathrm{static}}=0.05\)；
7. 生成 `POWER_AND_PRECISION_FREEZE.json`。

## Activation 2：target-blind interictal fit

1. 每名新患者建立 chronological train80/heldout20；
2. train60/validation20 选择 regularized estimator；
3. 按冻结流程拟合 full/rank-shuffle GRU；
4. 验证 contact field、fingerprint、denominator 和 target seal。

## Activation 3：一次性 target evaluation

1. 先评估 raw participation 的 orientation-free within-shaft margin；
2. 再评估 signed direction/heterogeneity；
3. 再比较 best regularized、full GRU、rank-shuffle 和 first-order；
4. 报告 patient bootstrap CI、patient-first tests 与 equivalence UCB；
5. 不因结果重新选择 polarity、field、亚组或 null。

## Activation 4：决策

- Endpoint 1 复制：支持 independent static contact-topography replication；
- Endpoint 1 未复制：停止，不用 signed 或 GRU 次级结果挽救；
- UCB < 0.05：允许写无科学上有意义的 GRU static increment；
- UCB ≥ 0.05：写 `INCONCLUSIVE_FOR_EQUIVALENCE`；
- 动态模型只按 spec §9 的三项联合门重开。

# RNN connectivity motif / cross-state v0.4 正式结果

## 结论

本轮工程验收通过。1426/1426 个正式训练单元完成，0 failed、0 OOM、0 nonfinite；完整冻结测试 140/140 通过，PNG/PDF/SVG 与 Figure 6 A–F 均完成科学和视觉核对。

科学结论分为四层：

1. **患者内传播充分性成立。** Dense、uniform sparse、fixed local、spatial growth 与带 wiring cost 的多类 leaky RNN 均能在留出间期事件上提供 next-contact 增益，并在删除已给定起点后的自由推演中保持非塌缩的传播排序。
2. **经济连接约束成立。** Spatial + cost 的患者中位 normalized wiring cost 为 0.301，低于 dense 的 0.911，同时保留可用的间期传播计算。该结果支持“同一任务可由更经济的 recurrent wiring 实现”，不等于恢复了解剖连接组。
3. **冻结场的 early-ictal 队列级对应未建立。** Primary cohort 为 target 解封前确定的实际交集 n=10、26 seizures；endpoint 为 clinical onset 后 0–10 s、1–150 Hz broadband energy，使用 canonical-full maxAB 相对 5000 次同步 all-contact permutation。Spatial + cost 的 null-relative margin 中位数为 0.110，但仅 6/10 为正，P=0.160；相对 no-recurrence 的中位增量为 0.071，Holm q=0.422；相对 dense 为 -0.009。控制 interictal fidelity 后，Spatial + cost 相对 no-recurrence 的估计为 0.050，95% CI [-0.034, 0.152]，permutation P=0.057。因此只能报告数值趋势，不能报告结构特异的跨状态优势。
4. **可干预计算 motif 未建立。** Effective operator 的跨 seed 与 split-half 稳定性成立，但 local-backbone 与 long-range-connector 的双重富集、与任务表现的患者级关系、相对 matched random lesion 的特异损害、以及超越 order-shuffle proposal 的证据未同时成立。

## 图的科学含义

主图：`results/topic5_rnn_motif_cross_state_benchmark_v0_4/figures/topic5_figure6_rnn_connectivity_motifs.png/.pdf/.svg`。

- **A**：同一患者几何上的 dense、sparse、local 与 spatial + cost 连接约束。
- **B**：观察到的 TA/TB contact-rank events 与给定第一 rank 后的自由推演；给定起点不算模型自发发现 A/B。
- **C**：全患者 recurrence gain 与 free-rollout rank correlation。
- **D**：间期场 fidelity 与 normalized wiring cost 的充分性空间；不用于 early-ictal model selection。
- **E**：冻结 RNN TA/TB 场与真实 early-ictal broadband energy 场，以及患者级 null-relative maxAB；支持数值趋势，不支持结构特异优势。
- **F**：target-free effective influence 与 matched perturbation；没有显著 lesion effect 时不命名为 established motif。

E1146 在 target 解封前固定，只作辅助空间示例，不进入主 P 值。RNN TA/TB 使用 early-to-late 顺序色标，真实 early-ictal 数据使用 broadband-power 色标，避免把模型输出与数据值混为同一物理量。

## 统计与冻结合同

- early-ictal target：clinical onset 后 0–10 s、1–150 Hz、baseline-normalized robust-z log broadband power。
- primary null：5000 次同步 all-contact label permutation；每次重做 support、mirror 与 maxAB。
- within-shaft permutation：仅 sensitivity，不替代 primary null。
- canonical full field：与 Human/SNN 场合同同构的跨状态 endpoint。
- seed-removed field：检验传播是否超越给定第一 rank 的机制性 secondary endpoint。
- target 解封后没有训练、模型选择或主评分修改。

## 工程验收

- 正式训练：1426/1426；总训练与 smoke 单元：1435。
- 每单元 `config.json` 与 `input_hashes.json`：1435/1435。
- 冻结测试：140 passed、0 failed、0 skipped。
- 20 个计算型 postprocess producer 与冻结 snapshot 逐字节一致。
- 制图 producer 在 target 解封后按最终主图要求修订，单独记录在 `POST_UNSEAL_FIGURE_AMENDMENT.json`；该修订只改变显示，并加入 full-versus-start-removed 的 secondary 配对展示，不改变模型、冻结场、primary scoring 或主要模型比较。
- 最终验收：`FINAL_ACCEPTANCE.json::engineering_accepted=true`；`PIPELINE_COMPLETE.json::status=COMPLETE`。

## 可以写 / 不可以写

可以写：

> 多类 recurrent connectivity constraints 足以在患者内自监督任务中生成留出间期传播；空间生长和 wiring cost 可在保留传播计算的同时降低总布线成本。

可以补充：

> 多种冻结 RNN 场对 early-ictal broadband field 呈正向数值趋势，但当前 n=10 队列未支持显著的跨状态对应或某一连接约束的特异优势。

不可以写：

- RNN 恢复了患者真实解剖连接组或独立发现病理轴；
- structured RNN 显著优于普通 recurrent model 地复现 early-ictal field；
- local-backbone + connector 已被确认为癫痫样传播 motif；
- early-ictal field correspondence 等同于发作预测或因果机制。

## 关键产物

- 正式报告：`TOPIC5_RNN_MOTIF_FINAL_REPORT_ZH.md`
- 最终验收：`FINAL_ACCEPTANCE.json`
- 逐项验收：`COMPLETION_AUDIT.json`
- 完成标记：`PIPELINE_COMPLETE.json`
- 图源清单：`figures/figure6_source_manifest.json`
- 视觉验收：`VISUAL_QA.json`
- 跨状态统计：`early_ictal_model_contrasts.json`
- 理论统计：`EFFECTIVE_MOTIF_SUMMARY.json`、`MATCHED_LESION_SUMMARY.json`

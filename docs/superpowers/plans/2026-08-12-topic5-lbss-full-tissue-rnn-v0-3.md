# Topic 5.1 Full-tissue LBSS-RNN v0.3 执行计划

> 对应 spec：`docs/superpowers/specs/2026-08-12-topic5-lbss-full-tissue-rnn-v0-3-design.md`
>
> 状态：**A–G 全部完成；closeout audit PASS（2026-08-13）**。正式结果根为 `results/topic5_lbss_full_tissue_rnn_v0_3/`，Figure 6 候选位于 `results/paper-ready-figure/fig6_lbss_full_tissue_rnn/figures/`。本计划不再追加模型或事后调参。

## A｜旧 latent-domain 审计与状态修正

1. 对 31 fits 重算 H coverage、contact-private nodes、未表示面积与 local-edge gap crossing；
2. 将 v0.2/LBSS spatial closeout 标记为 `CONTACT_DILATED_DOMAIN_SENSITIVITY`；
3. 保留 34 人 contact-space interictal learning 与 17 人 full-cohort early-ictal field-transfer 结果，它们不依赖本次 spatial topology claim；
4. 生成 `stage_a_latent_domain_audit`，同时显示 E1146 与 cohort。

## B｜Full-tissue geometry implementation

1. 在 `src/topic5_virtual_seeg_operator.py` 新增显式版本化 API，不改变历史 `sample_latent_nodes/resolve_node_count`；
2. 实现 offset-convex-envelope、PCA fallback、support/background node placement 与 zero-H audit；
3. 新建 v0.3 cache builder，读取冻结的 v0.2 events/contact plane，只替换 nodes、H、D 和 geometry provenance；
4. 新结果根目录：`results/topic5_lbss_full_tissue_rnn_v0_3/`，禁止覆盖 v0.2；
5. 单测：domain 与 contact-support union 不同、zero-H nodes 存在、H rows sum 1、每 contact 至少 3 nodes、determinism、node cap、strong connectivity、无 contact bypass。

阶段图：

- `stage_b_e1146_full_tissue_geometry.png/.pdf`
- E1146 contact-dilated v0.2 与 full-tissue v0.3 并列；contacts、zero-H tissue nodes、H-supported nodes 和 local backbone 用视觉编码直接区分。

## C｜工程 smoke 与资源冻结

1. E1146、E1084、Yuquan chengshuai：5 arms × 1 seed；
2. 检查 OOM、吞吐、梯度、resume、checkpoint、free rollout 与 field builder；
3. 根据实测显存冻结并发，不改 batch、node density 或科学合同；
4. 用 nohup/tmux 运行，单元级日志、DONE/FAILED、watcher；
5. 0 unresolved OOM、0 NaN 后进入正式 cohort。

阶段图：training loss、GPU memory、node coverage、candidate-pool separation。

## D｜Target-free 正式训练

运行 31 fits × 5 arms × 3 seeds = 465 units。优先完成 L0/L1/L2/L3，再完成 order-shuffle。所有正式模型只读取间期数据。

聚合并作图：

- overall 与 distal next-contact NLL；
- free rollout；
- empirical interictal field fidelity；
- L3 相对 L0/L1/L2；
- true-order 相对 shuffle；
- v0.2 contact-dilated 与 v0.3 full-tissue sensitivity。

阶段反思：若 v0.3 改变结论，解释为 latent-domain contract 的影响；不得选择性保留更好看的版本。

## D2｜空间连接与超参数的 target-free 决策

正式 D 阶段结束后，先读取患者级间期结果，不读取 early-ictal 数值。

1. 若 L3 在 overall、distal、rollout 和 seed stability 上显示选择性优势，记录
   `CURRENT_SPATIAL_CONTRACT_RETAINED`，不调参；
2. 若 L3 未同时优于 L0/L1/L2，运行冻结的
   `development_spatial_search_v0_4`：13 个单因素配置 × 3 fits × 3 seeds；
   所有配置比较先在 fit 内合并三个 seed，再在三个 development fits 间汇总；
3. 根据预写规则合成一个 joint candidate，选择最多两个非基准配置；
4. 对候选运行 5 matched arms × 3 fits × 3 seeds；
5. 只有 matched development confirmation 成立时，才冻结一个新配置并补
   31 fits × 5 arms × 3 seeds 的 full-cohort confirmation；患者级确认统计排除
   三位 development 患者，全部 spatial cohort 同时作为支持性结果；
   若无候选成立，写明搜索边界并保留 v0.3 原合同；
6. 全过程 target marker 必须不存在，所有产物 `target_values_read=false`。
7. 若新配置通过 development-excluded full-cohort confirmation，自动建立独立
   `topic5_lbss_full_tissue_rnn_v0_4_selected` artifact root；不得覆盖 v0.3。该 root
   以只读 symlink 消费 selected 465 units 与冻结 cache，完整重跑 E 阶段并写
   `PRIMARY_ARTIFACT_POINTER.json` 后才进入 F。Figure/closeout 跟随该 pointer。

该阶段不是为了刷 overall AUC/NLL；首要问题是 task-selected nonlocal shortcuts
是否对 distal propagation 提供超出 local-only、extra-local 和 random-LR 的增量。

## E｜Pathway 与 attenuation 冻结

在任何 early-ictal target access 前完成：

1. canonical-full 与 seed-removed fields；
2. source/target density、contact-space effective influence、distal reach；
3. arm-specific added-edge targets；
4. L3 matched-local controls；
5. 四档 attenuation rollouts 与 fields；
6. `MODEL_FIELD_MANIFEST.json`、`PATHWAY_MANIFEST.json`、`ATTENUATED_FIELD_MANIFEST.json`；
7. 文件 hash 与 `target_values_read=false`。

阶段图：真实 order 与 shuffle 的 nonlocal effective pathway formation；不显示发作 target。

zero-H dynamic engagement 是主 L3 的诊断审计，不扩成五臂第二套模型矩阵。它在最终
空间模型决定并完成主 pipeline 后，对 primary artifact root 的 31 fits × 3 seeds
运行；报告 zero-H state magnitude 与 clamp 后 heldout NLL 变化，不参与模型选择。

E 阶段结束后写 `INTERICTAL_POSTPROCESS_PRETARGET_COMPLETE.json` 并暂停。不得由
watcher 自动继续进入 F；只有 D2 的空间模型决策完成后，才显式授权 target unseal。

## F｜Early-ictal 外部 benchmark

1. 先生成 Figure 3D 17 人到 full-tissue spatial cohort 的逐患者 join/attrition table；冻结结果为 12 人/141 seizures，5 位无 full-tissue spatial model 的患者逐名列出；
2. 只读取 E 阶段冻结 fields；
3. 主 external benchmark 复用 Figure 3 的 0–10 s phenotype-matched readout、all-contact synchronized shuffle 与 maxAB；并列报告 11 人/92 seizures 的 strict-broadband 1–150 Hz sensitivity；E1146 场图固定使用 15 次 strict-broadband seizures；
4. 统计 D1 full-field、D2 seed-removed increment 与 attenuation dose-response；
5. 患者级 paired statistics；seizures 不作独立样本。

## G｜Figure 6 候选与收口

主图遵循现有 paper-ready Figure 风格，不靠堆文字表达：

- A：真实 SEEG layout 叠加 full-tissue RNN，contacts 仅为 readout；
- B：E1146 data 与 free-generated TA/TB rank sequences；
- C：34 人 contact-space interictal learnability；
- D：E1146 frozen RNN TA/TB fields 与 early-ictal field；
- E：17 人/实际可评价分母的跨状态统计；
- F：full-tissue local backbone + selective nonlocal shortcut 的 geometry/functional result；
- G：distal propagation matched contrasts；
- H：shortcut attenuation 与 early-ictal contribution。

每个阶段结束检查：模型是否仍在回答“局部 backbone + 少量选择性非局部 pathway 的癫痫样传播充分性”，而不是回到 generic wiring economy 或逐边 connectome recovery。

最终交付：代码、测试、运行 manifest、逐患者表、PNG/PDF/SVG、中文 README、收口报告。用户终审前不 commit。

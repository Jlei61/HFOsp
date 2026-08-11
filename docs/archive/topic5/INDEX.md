# Topic 5 Archive Index

> **主入口**：`docs/topic5_seizure_subtyping.md`（§5 历史文档索引含完整 backlink）
> **范围**：以 ictal seizure 本身为研究对象（subtype / pre-ictal / propagation / outcome）。
> **不属于**：interictal 事件内部传播（topic1）、IEI/PSD（topic2）、SOZ 空间归因（topic3）、模型层（topic4）。

## 主线（network-axis pivot）

### `rnn_motif_cross_state_v0_4_review_closeout_2026-08-10.md` — **间期生成充分性闭合；跨状态对应趋势未确认；局部有效组织阳性**
- 冻结模型无需重训；补齐 16→15→10 队列排除链、early-ictal target 患者内可靠性、canonical/full 场分解、布线资源定义和自由推演 Kendall 诊断。缺失 5 人均因冻结物理坐标模型少于 8 个 exact geometry/event contacts，不是评分 join 静默丢失。
- Q1 为强阳性但只限"排序"：真实顺序 recurrent models 能自由生成 heldout 间期传播（Kendall τb 相对无循环 +0.260、相对顺序打乱 +0.260），**参与集合 Jaccard 相对无循环为 −0.013（p=0.48），即不更会挑哪些触点参与**。Q2 仍是正向但未确认：source 有稳定贡献，recurrence 正向，true-order 与 wiring cost 无额外跨状态增量。Q3 仅支持 local effective influence enrichment；matched perturbation 因 n=5/7（long-range 0 人可估计）记为不确定。
- 第二轮整体审阅修复：这套工作散在**三个分支**（审阅收口 / 正式流水线 `527e3555`+`15a67b50` / 用户目视验收那批图所在的 `71c22692`），审阅写在最旧的一支上，已依次合并——主图 Panel F 标题恢复为已锁的 `Effective influence test`（冻结判决本就是 `M6_motif_claim_pass = false`），Panel E 三张场图统一到同一块 physical-mm 平面；布线与 rollout 患者级聚合改为与冻结间期表一致的两段式并加 parity 断言（原口径 294 行中 118 行不一致）；排除 6 个 smoke 训练单元。10% 边数是预设稀疏预算而非结果，结果是总几何长度 4.93%。全部图已用合并树重画，**需重新目视验收**（`VISUAL_QA.json` 已过期）。
- 下一阶段 `Local-Backbone Selective-Shortcut RNN` v0.2 已按 2026-08-11 审阅完成七项 P0 修订并锁定执行；v0.4 收口与新合同分开提交，结果目录不互相覆盖。

### `stable_interaction_identifiability_v2_1_multiround_2026-07-31.md` — **RNNv2.1 五轮结构审计：single fixed graph 在 4 位可辨识患者中 bounded negative**
- 修正 test endpoint-specific oracle Gate，不增加模型容量，完成 D1 baseline/envelope/diversity、D2 M2 operator、D4 unseen-start、D3 real-minus-null split stability 和 D0 patient-matched sensitivity+specificity。
- D0 为 4/6 PASS；这 4 位 human real-minus-null stability 全为负。另 2 位分别因 sensitivity 与 fixed-vs-mixture specificity 不足保持未裁决。
- Unseen-start NLL 5/6 改善，但 precedence 仅 2/6；不授权 shared-backbone modulation、event drive、process noise 或 full 34-patient 扩展。SNN 不参与 Gate。

### `stable_interaction_graph_rnn_v2_development_2026-07-31.md` — **SIG-RNN v2 development：feedback 有增量；稳定结构尚未裁决**
- 通用 12-contact synthetic feedback graph 在独立 graph/event seeds、9,600 个训练事件和未改阈值下通过工程校准；首轮 2,400-event G0-A 失败仍原样保留，未事后改门。
- 六患者中 SIG1 相对匹配 phase-only noGraph 的 NLL 与自由生成 precedence 均为 6/6 改善；但相对每位患者最强的 phase-matched mixture 或 latent time template，两端点同时改善仅 1/6。
- 旧 `G1` 实际是在 development test 上分别为 NLL 和 precedence 选择最小值的 endpoint-specific oracle stress test；它只能说明 current single fixed graph 未取得已见分布预测优势，不能裁决 stable structure。
- v2.1 已重开 patient-matched identifiability、chronological observable stability 和 unseen-start/compositional generalization；在这些结构特异实验完成前不扩 34 人，也不加 event drive/process noise。SNN 与 RNN 独立，SNN 不在任何 RNN Gate 中。

### `shared_propagation_field_rnn_multiround_review_2026-07-31.md` — **v0.1 七轮复核：输出不反馈的 autonomous trajectory 未被选择**
- 六患者 development 中，M4 相对 M3 在 10%–100% nested learning curve 六档均为 0/6，`d={2,4,6}` 也均为 0/6；该结果只约束由第一 rank 初始化、生成 contact 不反馈 latent state 的 deterministic autonomous latent-trajectory null。
- 既有 SNN Round 5 只作 exploratory compatibility check：legacy artifacts 未满足同条件 nested event-count / `N_min` 合同，且 first-rank lookup 已形成捷径，因此不能据此判 G0 正或负。
- v0.1 按窄合同完成 bounded negative；不解释 latent weight，不否定稳定 contact interaction，也不参与 human-to-SNN mechanism mapping。

### `shared_propagation_field_rnn_ladder_pilot_2026-07-30.md` — **RNNv2 自主 shared-field 六患者公平比较 bounded negative**
- 六名 target-blind development patients × 3 seeds × 8 models 全部训练充分；M4 虽超过 static 6/6、stationary M1 5/6，但相对 phase-matched mixture 与低维时间模板均为 0/6，M4-phase 也只在 2/6 超过模板。
- 旧 train80 内新增独立 development test、checkpoint/provenance、低学习率自动复核、重复 IWAE/prior-predictive 与多次自由 rollout；旧 heldout20 未读取。按 stop rule 不扩到 34 人。
- SNN 仅审计既有 source/sink/paired artifacts，未重跑；方向由低阈值 kernel/core 位置产生，isotropic 不是方向消失 null。该审计后续已从 RNN Gate 中删除，不能评分 G0。

### `rnn_training_and_objective_sufficiency_v0_1_report_2026-07-30.md` — **训练充分性关闭；整场生成阴性拆成"训练不足"+"读出方式"两个成分**
- 1,068 个单元零失败。上一轮冻结训练预算比延长训练预算差 **0.134 nats/decision（34/34 人，P=1.16e-10，LOSO 结构确认）**，约为既有顺序增量（0.0257）的 5 倍；容量、优化器家族、权重衰减、显存分块均已排除为限制因素，学习率仍是限制（3e-4 优于上一轮的 1e-3，位于预注册网格边缘）。跑到预注册上限 8 遍仍未满足连续两遍改善 <0.002 的收敛判据，故只能写"接近收敛"，**不得写"已收敛"**。
- **测试的三种 rollout-aware 目标不支持曝光偏差作为主要解释**（不等于普遍排除）：它们（每 2 步 / 每 3 步自喂、渐增 schedule）在 development 与外层留出上都同时损害局部预测与整场生成，呈单调剂量反应，一步预测护栏全部失守。
- **方法学更正**：上一轮用来评价生成的复合发生器（静态骨架＋顺序残差＋经验终止）**随模型变好而系统性变差**——外层留出上收敛模型经它读出后成对先后相关塌到 0.014（不用历史的静态对照 0.184）。改用模型自身联合分布读出，六个整场端点全部显著改善（30–33/34，两队列同向）；无重训的匹配消融显示这一改善**部分**来自事件内顺序（相对身份打乱对照，五端点显著），但成对先后的绝对水平主要由静态解码结构承担（相对冻结状态仅 +0.034，p=0.121）。**两台发生器是两个不同被估量**：模型自身＝模型生成能力主读出，复合＝顺序残差分解的敏感性。
- **预注册整场门槛仍未达到**：用上一轮自己的标准重评分，上一轮 9/34（本轮精确复现）、静态对照 10/34、延长训练+模型自身读出 13/34、rollout-aware+模型自身读出 14/34，门槛 17/34。**不得写"RNN 自由生成了真实的完整双向传播事件"。**

### `rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md` — **当前 RNN 阶段性总入口：科学验收通过，训练充分性仍开放**
- 已接受对象为稳定 static contact scaffold + 短程 within-event ordered information；linear-state 可改善 heldout next-contact，并在自由生成中恢复局部 transition fingerprint。
- 完整 suffix rank/precedence 和双向 axis read-back 未恢复，但该阴性目前只限 frozen teacher-forced training contract。正式模型仅一轮 exact coverage，最终 linear-state 未独立调 learning rate / training budget，也未用 self-fed rollout loss。
- 下一轮只允许做 convergence 与 objective-sufficiency 审计；执行 prompt 见 `docs/superpowers/plans/2026-07-30-topic5-rnn-training-sufficiency-agent-prompt.md`。

### `minimal_sequence_kernel_closeout_v0_2_report_2026-07-30.md` — **where / how / when 分层后的最小序列结构最终验收**
- 34 人 × 3 seeds 的同分母重评分将 heldout likelihood 精确拆为 contact choice 与 STOP：contact identity 的增量集中在当前和前一 rank，第三 rank 主要改善 STOP；更早历史无额外 contact 信息。
- 可识别对象改为 linear-state 的输入—输出 lag kernel \(K_k=CA^kB\)。显式 FIR-H3 未优于无序基线，固定方向跨数据集确认失败；patient-mean early-ictal association 仍主要来自 static scaffold。
- `when` 已隔离为新分支：exact 1–150 Hz seizure residual 可靠性当前不可辨识，IEI-aware Gate 1 仅有未跨两数据集复现的 cohort feasibility signal。本轮定位固定为 Extended Data / Supplementary bounded result。

### `ordered_history_architecture_audit_v0_1_report_2026-07-29.md` — **最新 RNN 条件信息、架构与跨状态综合验收**
- 34 人 × 3 seeds 的 target-blind 架构审计显示：linear-state 相对 unordered-prefix 的患者中位 NLL 增益为 0.0257（26/34，7-family maxT P=0.00032），相对同架构 rank-shuffle 为 0.0419（31/34）；容量匹配后 linear 结果保留。
- 但 7 个预注册递归家族仅 linear-state 通过 family-wise inference，故顺序证据具有架构依赖性；clinical-onset `[0,10] s`、`1–150 Hz` reused target 上，ordered residual 超越 static + unordered 与 matched shuffle 的条件增量均未建立。当前只进入 supplementary sequence-identification / boundary result，不支持脑流形、真实时间慢变量或逐发作预测。

### `rnn_overall_integrated_acceptance_2026-07-28.md` — **上一版 RNN 总验收基线（已由 2026-07-29 架构审计细化）**
- 统一收口 full-rank、low-rank、persistent path、symmetric-axis、competitive/source、internal-state、fixed early-ictal readout 与 H1/H2/H3 history-necessity 全部分支。
- 最终接受对象为 target-blind 的稳定 interictal participation scaffold + 最近 2–3 个 rank set 的有序短历史；early-ictal 只保留 reused-target 的 sign-free static morphology。full history、正 low-rank mode、path/axis/competition/source 和 GRU-specific static transfer 均未建立。
- 机器可读状态：`results/topic5_rnn_overall_acceptance/FINAL_ACCEPTANCE.json`；论文层级固定为 `SUPPLEMENTARY_BOUNDED_COMPUTATIONAL_RESULT`。

### `interictal_scaffold_reliability_history_necessity_v0_1_report_2026-07-28.md` — **34人 target-blind 静态可靠性与 H1/H2/H3 历史必要性**
- train80–heldout20 participation field Spearman 中位 0.893，34/34 为正；约 200 个事件时 30/34 已接近 full train80 estimate。
- H2>H1、H3>H2、ordered H3>matched H3 shuffle 均通过；full history 不超过 H3。accepted sequence reference 因此锁为 H3，不再扩大 GRU 历史。

### `static_scaffold_fixed_readout_validation_v0_1_report_2026-07-28.md` — **静态 contact topography 分项验收：跨状态形态保留，GRU-specific 增量未建立**
- strict clinical-onset 16 人/106 seizures 的 participation readout、signed primary、强空间 null、target-free regularized baseline、teacher/free 分解和 baseline-power confound audit 已全部完成。
- orientation-free contact morphology 在 within-shaft 与 geometry-smooth null 下保留；真实顺序相对 rank-shuffle 有 heldout 增益，但 positive signed direction、unbounded-history necessity 和 GRU-specific static increment 均未建立。等待真正独立 clinical-onset patient cohort 复制。

### `static_contact_topography_claim_consistency_audit_2026-07-28.md` — **当前论文口径全文一致性审计**
- 扫描 21 个 manuscript-facing 文件，20 处敏感表述全部归入明确边界/否定、其他经验合同或历史模型阶段，`UNSAFE_CURRENT_CLAIM=0`。
- 当前唯一 Figure 6 source 为 `docs/paper-draft/figure6_static_contact_topography_bounded_result.md`；早期 RNN 版本只保留为 provenance。

### `rnn_postreview_closeout_and_static_scaffold_goal_2026-07-28.md` — **RNN 审阅后收口与 fixed signed scaffold 新 goal**
- 上一 goal 经复核拆成 static contact topography、interictal order sensitivity 和 target-reused state read-back 三层；Figure 降为补充探索性候选，structured-axis RNN 冻结。
- 新 fixed participation Phase 1 显示 absolute morphology margin 稳定，但 positive signed margin 在 all-contact、within-shaft 与 geometry-smooth null 下均有明显患者异质性；已启动正则化非递归 baseline、teacher-forced/free-rollout 和 confound 分解。

### `rnn_internal_state_reduction_v0_1_report_2026-07-28.md` — **静态 scaffold、ordered-history 诊断与探索性 state read-back 分层**
- 冻结 34 人 × 3 seeds 既有 GRU，102/102 hidden extraction、扰动和随机子空间单元全部完成；真实 prefix 顺序打乱对 ordered GRU 的 NLL 影响显著大于 rank-shuffle，对应 32/34 患者为正。
- strict clinical-onset 16 人/106 seizures 中，固定 participation 支持静态 contact scaffold，但 full GRU 未稳定超过 static/unordered/rank-shuffle；去 participation 后的 PC1/PC2 迁移仅作 target-reused exploratory candidate。下一步先做 fixed signed readout、强空间 null 和正则化非递归基线。

### `symmetric_axis_competitive_propagation_rnn_v2_3_result_2026-07-27.md` — **categorical RNN 可预测，但 physical-axis 机制门失败**
- 22 人 × 3 seeds × 5 conditions 的 330 个正式模型全部完成；full 相对 node bias 22/22 为正，history 相对 instantaneous 18/22 为正。
- delayed competition、matched physical axis 与 source-conditioned direction 均未过冻结门；模型约恢复 ordered-history Markov cohort-median benefit 的 58%，不开放 latent-state 解释或 early-ictal transfer。

### `interictal_transition_signal_decomposition_v0_1_result_2026-07-27.md` — **Markov transition signal 分解与 v2.3 开发许可**
- 31 人显示 symmetric、跨局部几何且依赖 ordered multi-step history 的 heldout transition signal；22 人 physical-axis residual通过，但 source-conditioned增益很小且 14/22 axis coefficient 为负。
- 冻结决策为允许起草最小 v2.3 recurrent observation model，不是 shared anatomical axis 或 early-ictal transfer 的机制结论。

### `symmetric_axis_propagation_state_v2_2_1_closeout_2026-07-27.md` — **symmetric-axis propagation-state RNN 按预注册停止点收口**
- 66/66 formal runs 审计完整；Claim 1/2 为已执行失败，Claim 3/4 为 `LOCKED_NOT_RUN`，early-ictal transfer 同时受间期 gate 与 0 exact source metadata 阻断。
- 在同一 22 人 heldout 合同下，Markov 21/22 优于 node-bias，而 full/isotropic 均仅 1/22 为正；校准显示 next-set size 1.00 被预测为约 1.65，local/axis kernel 中位 Frobenius cosine 0.979。结论只限“当前非负线性单状态 observation mapping 不足”，不否定共享病理轴。

### `persistent_path_mode_rnn_closeout_and_v2_pivot_2026-07-26.md` — **Figure 6 旧 RNN 收口与 v2.2 propagation-state 转向**
- 冻结 v0.7/v0.9/v1.0：局部历史可学，但离散、event-persistent path mode 不是合适科学对象；不再调 K、hidden size 或开放发作期 target。
- 下一版改为共同的近似对称 effective scaffold + observed source + 单一 propagation state + scalar STOP；三位 geometry-complete development 后以 22 人做 physical-axis formal，跨状态主任务回到 clinical-onset early-ictal energy field。

### `persistent_path_mode_rnn_formal_result_2026-07-26.md` — **Figure 6 正式 34 人 structured graph RNN bounded-negative**
- 34 人 × 3 seeds × 5 conditions 的 510 个 LOSO runs 全部完成；局部 next-set NLL 可学，但 participation 与完整 rank distribution 的自由生成主门未通过。
- graph / mode-collapse 结构必要性未成立，path-direction posterior 保持高熵；按预注册合同不读取 clinical-onset 发作期 target。给出可保留口径、禁止口径和若重开时必须修改的训练目标。

### `fig6_interictal_operator_phase0_stagea_pilot_2026-07-24.md` — **Figure 6 计算桥：selected h64 的 Stage-A engineering screen 停止于 suffix-static gate**
- 以 masked contact-rank 的单事件 prefix→suffix/STOP 任务训练 contact-query GRU；40/40 患者、532,793 个间期事件通过数据与泄漏审计，Stage A 不读取 ictal values。
- 13 折 target-free one-SE 选择 h64；13 人 one-seed screen 中 next-set 对 strongest static 的患者级 CI 为正，但 suffix 中位为负且 CI 跨 0；两项均 13/13 超过 rank-shuffle。按 stop rule 不启动正式三 seed gate、Mode recovery 或 ictal readout。

### `fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md` — **当前执行合同：Figure 3 发作相关 gradient R3 全量重算**
- 以正式 n=17 / 167 seizures 为唯一母清单，统一使用 outcome-independent adaptive 81×81 gradient grid、subject-fixed sigma、corrected mirror abs-max、shared-else-own maxAB 与 coherent all-contact null；R2 只作同输入 paired sensitivity。
- Stage 1 重算 cohort Data-vs-Null 与七频带 inheritance/specificity；Stage 2 更新 Fig3-B R3 score provenance，并在 Fig3-C 仍保留时同步重算 7 名 shared-only 轨迹和 spatial null。配套回填表 `fig3_ictal_gradient_r3_full_recompute_run_form_2026-07-18.md`。

### `field_concordance_multiband_unified_handoff_2026-07-18.md` — **SUPERSEDED：旧 R2 七频带合同**
- 该版本虽锁定 n=17 / 167 parent cohort、共同 permutation 与 subject-first fold，但 primary metric 仍是 contact-evaluated R2；只保留 provenance，不再执行。旧 form 同样作废。

### `fig3a_raw_spectral_context_acceptance_2026-07-18.md` — **Fig3-A 正式画图合同与验收**
- 锁定 E1146 seizure 7 的 raw SEEG + SCL9 TFR + 四频带 2×2 布局、严格时间轴对齐、row-shared y 轴、clinical-onset shading 和可/不可报告边界。
- canonical producer：`scripts/paper_figures/plot_fig3_raw_spectral_context.py`；输出：`results/paper-ready-figure/fig3a_raw_spectral_context/figures/`。

### `axis_alignment_AB_result_2026-06-14.md` — **现阶段主线结果**：间期传播轴 ↔ 发作早期激活的轴对齐（A 线 primary + B 线 secondary）
- 18 Epilepsiae 队列：粗"共享网络主轴"稳（broadband 稳赢全通道 null，FDR + LOSO 扛住）；细对齐仅快活动（hfa）稳（过最严 joint）；符号自由共线，非逐点重放。
- 含完整方法 / 定稿数值表 / 工件清单 / handoff。计划全貌：`network_axis_pivot_plan_2026-06-13.md`（A/B 段已标 ✅ 执行）。
- 定稿表 `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.md`。

### `hfa_joint_confirm_2026-06-15.md` — hfa×joint 冻结复验（split-half + 负对照）
- 唯一过最严 joint null 的 hfa 细对齐：full 干净复现（Wilcox=0.022）但**奇数半不显著（0.078）→ 非 split-half 稳健**；负对照四层全部非显著=非假阳性。
- 结论 = real-but-not-robust，**维持灵敏度档、不升 primary**；升格须独立第二队列。主线粗骨架不受影响。

### `v3p_preictal_nonaxis_trajectory_2026-07-05.md` — V3p preictal-only 非轴向轨迹完整硬门阴性
- 只看 EEG onset 前 −120~−10 s；narrow、`broad_expanded`、`broad_core` 分层报告，三层均 tier 0，完整个体支持为 0。
- broad 的少数 single-null nominal hits 被 rate / lag1 / phase / block / 双 span 预设硬门筛掉，不算潜在阳性。
- 结论边界：未支持稳定一致的 preictal non-axis ramp；不等于发作前没有任何 state change，也不裁决 onset 后变化。

### `contact_similarity_ladder_2026-07-01.md` — 触点相似性几何阶梯（R1 无几何 / R2 同平面触点核 / R3 场）
- n=18（两激活量），场统计量数值抬高主要来自平面几何平滑；但平滑同时抬高信号与零假设，超零假设被试数反而随 R1→R3 下降。网格步无可分辨增益；R3 与 A 线主统计逐位一致。
- R2b native-3D sensitivity 与 2D plane 等价通过。定位是灵敏度/稳健性复核，不是新的队列级主张；主线粗骨架结论不受影响。

## PR 系列

### `rnn_connectivity_motif_cross_state_v0_4_result_2026-08-10.md` — **多类 recurrent wiring 足以生成间期传播，但跨状态与可干预 motif 未建立**
- 1426/1426 正式单元完成、140 项冻结测试通过；dense、sparse、local 与 spatial/economic leaky RNN 均达到患者内传播充分性，Spatial + cost 以更低 wiring cost 保留任务能力。
- frozen RNN field 对 clinical-onset 0–10 s、1–150 Hz early-ictal field 只有数值趋势（primary n=10），结构特异优势、多重校正后跨状态对应和 matched-lesion motif 均未成立；Figure 6 与可/不可写边界已冻结。

### `pr1_seizure_clustering/` — Per-subject seizure subtyping (z-ER tensor + 1−Spearman + UPGMA)
- `pr1_zer_cohort_2026-05-10.md` — **主结果文档**：cohort z-ER subtyping，含 sentinel 视觉裁定、audit fix 历史、over_split 规则演化
- 见 `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/README.md` 的 cohort 视觉骨架
- 计划档：`docs/superpowers/specs/topic5_pr1_seizure_clustering.md`（plan v2）

### PR-0：v2.3 Layer A ictal ER timing atlas
（追授 topic5 PR-0；详细 spec 见 `docs/superpowers/specs/`）

### Bridge → Ictal-template echo 谱系（Topic 1 × Topic 5：间期传播模板是否在发作期复演）
- `bridge_q1/bridge_q1_results_2026-05-10.md` — Q1 cohort（verdict NULL-locked, n=9, power floor）
- `bridge_q1prime/bridge_q1prime_results_2026-05-10.md` + `q1prime_overnight_exploration_2026-05-10.md` — Q1′ case-series（INDETERMINATE）
- `echo_gate/stage1_proxy_triage_2026-06-08.md` — **Stage 1** ER 代理 echo gate：= 共享粗锚，非 specific-path-replay
- `dynamic_echo/stage2b_sentinel_2026-06-12.md` — **Stage 2b** early-ictal 动态模板 echo sentinel：**gate NOT PASSED**（B=500 n=3）；有模板相关结构但非稳定早期路径复演 → 粗解剖/杆级锚为主；未进 cohort。（Stage 2 first-onset recruitment 量错失败，未单独归档，见此文档"谱系"段）
### `constructive_event_generation_sufficiency_v0_1_report_2026-07-30.md` — **局部 transition 可生成，但完整双向事件充分性 Gate 失败**
- 34 人 × 3 seeds 的 102 个 source-conditioned free-running 单元全部完成；history 改善 first-order transition fingerprint，但不改善 suffix rank/precedence，只有 9/34 人至少两项达到人体 split-half 经验范围。
- 独立 rank-progress STOP 在 34/34 人必要；22 位 train-only 双模态+物理轴合格患者中，history 未改善 template 或 signed-axis fidelity。Gate C 与 SNN bridge 按合同锁定。

### `spatial_latent_propagation_rnn_v0_1_2026-08-06.md` — **状态从触点搬到组织平面；连接身份不可辨识，预测腿仍可读**
- 触点改为观测口而非节点，第一次可以问「能不能预测没训练过的触点」。
- 合成可辨识性检验：逐边身份与整体行进方向都认不出，只有各组织块往前推的相对排序认得出。
- 因此患者间图差异与定向删连接两类说法在本版无依据；预测层面的比较不受影响。

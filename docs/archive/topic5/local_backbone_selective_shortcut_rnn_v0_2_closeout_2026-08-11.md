# Topic 5.1：Local-backbone selective-shortcut RNN v0.2 收口报告

> **分母更正（2026-08-11）**：本报告的 10 人/24 seizures 是旧 `outer_*` exact-join 的
> LBSS 物理子集敏感性，不是 Figure 3D 的正式发作 cohort。正式 RNN external benchmark 已
> 按 34 人间期 + 17 人/167 seizures 发作母清单重做，见
> `rnn_full_cohort_field_transfer_v0_1_2026-08-11.md`。本报告 Claim A–C 的 21 人 LBSS 机制结果
> 保留；涉及 10/24 的 Q2 表述均不得作为 primary cross-state 结论引用。

日期：2026-08-11（当日复审后修订）
状态：**科学合同执行完成，六项 spec 要求的审计已补做，Figure 6 已重建并纳入 Topic 5 RNN closeout**

> 修订摘要：复审把执行结果逐条对回 spec，改动集中在四处——(1) 正对照按 geometry 逐条裁决后，收回「这不是检测能力不足」这一说法（§4）；(2) Claim A 的 comparator 补做等价审计并加硬前置（§5）；(3) Claim C 补做跨 seed 稳定性与「换 seed」对照，效应量由 0.452 下修到 +0.27~+0.35，判决由 SUPPORTED 改为 PARTIALLY_SUPPORTED（§7.1）；(4) Claim D2 的 min-of-controls 统计量补做零假设标定，撤回「显著方向相反」的读法（§8.2）。图与判决名的修订见 §10.3–§10.5 与 §12。**没有任何模型、field、评分被重跑，所有已冻结数值不变。**

## 1. 一句话结论

患者内有序间期 contact-rank 序列能够被局部 recurrent RNN 学习并自由生成，真实 rank order 对这一计算很重要；但在固定局部 backbone 上额外加入 task-selected nonlocal shortcuts，并未比额外局部边、随机 nonlocal shortcuts 或纯局部模型更好地解释远端间期传播，也没有形成 early-ictal 场的结构特异增量。

因此本轮支持：

> **局部 recurrence + 真实事件顺序足以产生患者特异的间期传播计算。**

本轮不支持：

> **少量 task-selected nonlocal shortcuts 是这一传播或跨状态场对应所必需的特异 motif。**

## 2. 这轮到底做了什么

所有模型使用相同的患者 tissue plane、contact read-in/readout、leaky RNN、next-rank/STOP 任务、训练预算、free-rollout decoder 和患者内 chronological split。只改变局部 backbone 以外新增边来自哪里：

- `L0_LOCAL_ONLY`：只有固定、强连通的 degree-balanced 局部 backbone；
- `L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL`：同一 backbone，加等数量、仍偏局部的可学习边；
- `L2_LOCAL_PLUS_RANDOM_LR`：同一 backbone，加等数量的固定随机 nonlocal 边；
- `L3_LOCAL_PLUS_LEARNED_LR`：同一 backbone，加等数量、由间期任务选择的 nonlocal 边；
- `C_L3_ORDER_SHUFFLED`：保持 rank 1，使用 derangement 打乱后续 rank-set 顺序后训练 L3。

这里的 `nonlocal` 只表示相对局部 backbone 更远、可在一个 ordinal rank update 内通信；模型没有传导速度和物理轴突延迟，不能把这些边称为真实白质通路。

## 3. 工程执行与数据边界

- 间期 cohort：21 位患者，31 个 fit，5 个 arms，3 个 seeds；
- 正式训练：465/465 单元完成，0 失败、0 unresolved OOM、0 非有限值；
- 正式训练墙钟约 7.45 小时；
- target-free intact fields：630 个 fit-seed-template fields；
- pathway analysis：372 个 fit-seed units；
- attenuation：6,296 个 draw-level 指标行、1,968 个 fit-seed-template field rows；
- early-ictal primary cohort：10 位患者、24 次发作；
- target：clinical onset 后 0–10 秒、1–150 Hz broadband energy；
- primary null：所有评分 contacts 同步做 all-contact label shuffle，5,000 次；
- sensitivity null：within-shaft shuffle；
- patient-first：先 seizure 内评分，再患者内取中位数，最终以患者为统计单位；
- 相关测试：103/103 通过。

模型、intact fields、attenuated fields 和各自 manifest 全部冻结后，才生成 target-unseal authorization。Target 解封后没有训练、模型选择或 field 重建。

## 4. 功能可检测性正对照（2026-08-11 修订：按 geometry 逐条裁决）

初版只报告三套几何的中位数，并据此写“这不是检测能力不足”。逐 geometry 展开后这个结论只对一部分对比成立：

| 指标 | syn-1084 | syn-1146 | syn-chengshuai | 中位数 | 真实队列同一对比 | 灵敏度是否建立 |
|---|---|---|---|---|---|---|
| L3−L0 distal | +0.0042 | +0.1083 | **−0.0195** | +0.0042 | +0.0023 | **否** |
| L3−L1 distal | +0.0300 | +0.0461 | +0.0136 | +0.0300 | +0.0056 | 是 |
| L3−L2 distal | +0.0302 | +0.0421 | +0.0209 | +0.0302 | −0.0060 | 是 |
| true−shuffle distal | +0.0719 | +0.2839 | **−0.1081** | +0.0719 | — | **否** |
| 关闭 L3 边后 distal NLL 上升 | +0.1258 | +0.2870 | +0.1052 | +0.1258 | — | 是（仅绝对损害） |

两处问题：

1. 原 `functional_class_detected` 是五个「三套几何的中位数 > 0」的 AND，没有幅度阈值，也不要求三套几何同向。L3−L0 与 true−shuffle 各有一套几何为负，且植入效应的 L3−L0 中位数（+0.0042）与真实数据被判为 null 的同一量（+0.0023）同量级，因此**该对照没有为 L3-vs-local-only 建立可检测下限**。
2. spec §8.3 要求对照证明「关闭 selected 边**选择性**损害 distal」，但记录的是 distal NLL 的绝对上升，从未计算 distal−local selectivity。真实数据失败的正是 selectivity 双重解离，所以该终点的灵敏度同样未建立。

**可以写**：当比较对象是额外局部容量或随机 nonlocal 容量时，本流水线能检出植入的非局部通信需求。
**不可以写**：L3 相对 local-only 的 distal null「不是检测能力不足」。

逐条裁决归档：`results/topic5_lbss_rnn_v0_2/synthetic_detectability/FUNCTIONAL_DETECTABILITY_ADJUDICATION.json`。

## 5. Claim A：局部 recurrence 是否足够

**支持。**

`L0_LOCAL_ONLY` 相对 matched no-recurrence：

- held-out contact NLL 改善中位数：+0.13880；
- 20/21 患者同向；
- 95% bootstrap CI：[0.09715, 0.19914]；
- Wilcoxon P=1.91×10⁻⁶。

只给第一 rank 后，L0 自由生成的 propagation-rank correlation 中位数为 0.771；生成长度/真实长度中位数为 1.10。局部 backbone 不只是 teacher-forced one-step predictor，而能生成整场留出事件。

**comparator 边界（2026-08-11 补做）**：这里的 no-recurrence 臂不是本轮训练的，而是从 v0.4 run 直接读入的 `M0_NO_REC__rnn`。spec §4 要求先做 checkpoint/config/hash 等价审计才允许它进入 matched contrast，初版没有做。补做结果（93 个 unit、93 个 input-cache 文件）：输入 cache 逐字节相同，train/validation/test 事件数、contact 数、node 数、batch size 全部一致，共享 config key 无一处取值不同，v0.4 侧全部收敛。仅有的差异是 LBSS 侧多出 `added_fraction` / `resume_every_epochs` / `gradient_clip=5.0`——后两者是训练稳定性设置，**方向上有利于 LBSS 一侧**，因此 Claim A 的正向结论应理解为「至少这么大」而非精确幅度。归档：`results/topic5_lbss_rnn_v0_2/NO_REC_EQUIVALENCE_AUDIT.json`；`analyse_topic5_lbss_interictal_v0_2.py` 现在没有这份审计就拒绝导入该 comparator。

## 6. Claim B：task-selected nonlocal shortcuts 是否有选择性增量

**不支持。**

在 distal transitions 上，L3 相对：

- L0：+0.00228，11/21 同向，P=0.919，Holm q=1；
- L1：+0.00557，12/21 同向，P=0.708，Holm q=1；
- L2：−0.00602，9/21 同向，P=0.374，Holm q=1。

在全部 transitions 上，L3 与 L0、L1 基本相同；相对 L2 反而有很小但一致的劣势（中位 −0.00210，3/21 同向，P=0.00375）。这一差值数值很小，不应包装成随机 nonlocal 生物学更优；它只说明 task-selected nonlocal 并没有产生预期优势。

各 recurrent arms 的自由生成相关都处于同一平台：L0 0.771、L1 0.771、L2 0.758、L3 0.771。当前序列数据能强烈区分 recurrence 和真实顺序，但不能辨识哪一种新增连接 pool 更正确。

## 7. Claim C：真实顺序是否选择了功能性 shortcut organization

结论分成两层。

### 7.1 真实顺序确实重要，并会改变粗空间组织（效应量已按 seed 对照下修）

- L3 相对 order-shuffle 的 all-step NLL gain：+0.12405；
- 21/21 患者同向，P=9.54×10⁻⁷；
- shuffle 自由生成相关中位数降至 0.450。

粗空间组织这一层初版报告为「扣除 candidate-proposal exposure 后中位 0.4524，21/21，Holm q=2.86×10⁻⁶」。该统计量只扣掉了 proposal exposure 的差异，**没有扣掉换随机种子本身带来的漂移**，spec §9 要求的跨 seed 相似性从未计算。补做后：

- 跨 seed 可重复性（同一 arm，三个 seed 两两相关）：true-order 的 endpoint pattern 中位 r=+0.051、effective-influence 中位 r=−0.015；shuffle 侧分别为 +0.087 与 +0.070。**两个 arm 都没有能扛过换 seed 的粗 pattern**，因此 spec §9 定义的 "consensus pathway" 在本轮**不成立**，不得使用该措辞。
- 同一指标下，「换 seed」与「毁掉顺序」的对照（患者层）：endpoint 0.144 → 0.402，差 +0.352，19/21，P=6.7×10⁻⁵；effective-influence 0.167 → 0.432，差 +0.265，19/21，P=2.0×10⁻⁴。

所以结论保留但收窄：**毁掉真实顺序对粗空间组织的扰动明显大于换一个随机种子**，这一点稳健；但公布的 0.4524 中约有 0.14–0.17 来自 seed 漂移，承重量应取 +0.27~+0.35。同时，由于单 seed pattern 本身不可复现，这只能说「顺序影响了模型形成的粗计算」，不能说「顺序选出了一条可辨识的通路」。

归档：`results/topic5_lbss_rnn_v0_2/pathway_analysis/ACROSS_SEED_PATTERN_STABILITY.json` + `across_seed_pattern_stability.csv` + `order_vs_seed_pattern_control{,_patient}.csv`。

### 7.2 但没有证明这种改变就是 selective nonlocal shortcut 功能

- true-order 相对 shuffle 的 distal gain：+0.05982，14/21，P=0.0547，Holm q=0.109；
- L3 selected-nonlocal attenuation 相对 matched-local 的 distal-selectivity AUC：−0.01983，7/17，P=0.611；
- matched-local inference 可用 17/21 患者，其余 4 人因无法满足冻结 calipers 仅作描述性分析。

L3 attenuation 相对 extra-local 更有 distal selectivity（中位 +0.05460，15/21，P=0.00554），但相对 random nonlocal 为 −0.01680，且相对 matched-local 为负。因此不能把它解释成“任务选择出了特异的远端 pathway”；任意 nonlocal 或部分局部高影响边也能产生相似或更强的远端损害。

## 8. Claim D：冻结间期场是否对应 early-ictal 场

### 8.1 D1：存在正向趋势，但未确认

L3 canonical-full field 相对 synchronized all-contact null：

- patient-level margin 中位数：+0.10887；
- 6/10 患者为正；
- 95% bootstrap CI：[−0.06039, 0.22233]；
- Wilcoxon P=0.08398，Claim-D Holm q=0.16797。

数值方向与“间期场和发作早期场有共同空间成分”一致，但当前 10 人 cohort 不足以确认。

而且这一趋势不是 L3 特有：

- L0 canonical margin：+0.11894；
- L1：+0.11985；
- L2：+0.11433；
- L3：+0.10887。

L3 相对 L0/L1/L2 均无优势；控制间期 field fidelity 后的 L3 model effect 也均跨零。

### 8.2 D2：不支持 shortcut-specific 跨状态贡献

去除真实第一 rank/source 后，L3 相对三个 controls 的逐患者最小增量：

- 中位数 −0.01089；
- 1/10 患者为正；
- 95% bootstrap CI：[−0.06092, −0.00300]；
- Holm q=0.01172。

**统计口径更正（2026-08-11）**：初版在这一行写「方向与假设相反」，容易被读成「L3 显著更差」。该统计量是「L3 减去三个 control 中最好那个」，只要 L3 不是四臂中唯一最好的，它就为负——**在四臂可交换的零假设下它的中位数本来就是负的**，双侧 Wilcoxon 的显著性不能读作 L3 更差。按「把四个臂之一随机指派为 selected 臂」重标定（20,000 次）：参考中位数 −0.0073（2.5–97.5% 为 [−0.0195, +0.0006]），实测 −0.0109 落在参考分布内，p=0.167；可交换下预期「selected 臂最好」的患者数为 2.51/10，实测 1/10，p=0.243。

因此正确表述是：**L3 不优于三个 control，也没有证据表明它更差**。归档：`results/topic5_lbss_rnn_v0_2/CONJUNCTION_STATISTIC_CALIBRATION.json`。

对各类新增边做 attenuation 后，L3 的 seed-removed early-ictal concordance damage AUC 也没有超过 extra-local、random-nonlocal 和 matched-local：逐患者最小优势中位数 −0.01293，3/10，Holm q=0.16797（同一 min-of-controls 口径，同样只能读作「不更好」）。

### 8.3 起点/source 的贡献

L3 的 full-field margin 相对 seed-removed field 增加中位数 +0.02018，9/10 患者同向，P=0.00977。L0、L1、L2 也有同方向变化。

所以当前跨状态趋势的一部分明确来自真实第一 rank/source 与宽尺度静态 scaffold，不能全部归因于后续 recurrent propagation。

### 8.4 Null 边界

Primary all-contact shuffle 是本论文数据主线使用的口径。Within-shaft sensitivity 下，L3 canonical margin 中位数 +0.05788，7/10，P=0.232，未确认。因此当前 RNN 结果不能独立声称跨状态对应超过 shaft geometry；它应与论文已有 data-level 结果一起解释。

## 9. Figure 6 逐 panel 批注

- **A**：真实患者组织平面上的固定局部 backbone 和少量 task-selected nonlocal edges。红边是模型有效边，不是真实白质束。
- **B**：预先指定患者的 held-out TA/TB 事件与同起点自由生成。它说明模型能生成两类条件传播分布，不表示逐事件重放。
- **C**：21 人结果。最清楚的信号是 true-order 明显优于 shuffle；selected nonlocal 对 local/extra/random 没有增量。
- **D**：true-order 和 shuffle 形成不同的 coarse endpoint/effective-influence pattern；右侧散点是「毁掉顺序」的效应，虚线是「同一 arm 换随机种子」的同指标参考水平，**承重量是散点与虚线之间的距离**，不是散点本身，也不是 exact edge identity。
- **E**：预先指定代表患者的 data TA、frozen RNN TA 和 early-ictal 0–10 s broadband energy，只作直观示例。该病例的评分是 mirror/maxAB 下的 |r|=0.64，其带符号相关为 **−0.64**，图内已标注；不看这行标注会把一对反相关的场读成一致。
- **F**：10 人外部 benchmark。四个 recurrent arms 均有类似正向趋势；selected nonlocal 的 attenuation 没有特异损害跨状态一致性。

最终图路径：

`results/topic5_lbss_rnn_v0_2/figures/topic5_figure6_lbss_rnn.{png,pdf,svg}`

paper-ready 副本：

`results/paper-ready-figure/fig6_lbss_rnn/figures/`

## 10. 工程审计与修复记录

### 10.1 Matched-local eligibility P0

初版 attenuation 会在零合法匹配时中断，并在 seed 聚合时用中位数传播 eligibility，可能把一个失败 seed 静默变成可推断。修订后：

- 零合法匹配写显式 `NO_VALID_MATCH`，不伪造 field；
- eligibility 和 `n_valid_matched_draws` 按 seed→fit→patient 取最小值；
- 17/21 患者可做 matched-local inference，4/21 为 `DESCRIPTIVE_ONLY`；
- 所有 inferentially eligible patient metrics 均为有限值。

### 10.2 CPU/GPU execution switch

先在同一患者、同一 seed、四个 alpha 上验证 CPU/GPU rollout JSON 和 fields 逐位相同，再从拥塞的 GPU6 切换到 CPU64。切换不改变 checkpoint、mask、随机数、scorer 或输出合同。

### 10.3 收口后审计补做（2026-08-11 复审）

复审把执行结果逐条对回 spec，发现六项 spec 要求的产物或审计没有落地。全部只从已冻结产物重算，未训练、未重建 field、未重新评分：

| 缺口 | spec 依据 | 补做产物 |
|---|---|---|
| v0.4 no-recurrence comparator 未做等价审计 | §4 | `NO_REC_EQUIVALENCE_AUDIT.json`（并在分析脚本中改为硬前置） |
| 正对照被压缩成单一 `median>0` 标志 | §8.3 | `synthetic_detectability/FUNCTIONAL_DETECTABILITY_ADJUDICATION.json` |
| 跨 seed pattern 相似性从未计算 | §9 | `pathway_analysis/ACROSS_SEED_PATTERN_STABILITY.json` + 两张 CSV |
| D2 conjunction 统计量无参考分布 | §12 判读纪律 | `CONJUNCTION_STATISTIC_CALIBRATION.json` |
| 六个 plan D2 聚合产物缺失 | plan D2 | `interictal_per_event.csv`、`order_shuffle_effective_strength.json`、`candidate_exposure_audit.{json,csv}`、`distal_transition_summary.{json,csv}`、`rollout_diagnostics.{json,csv}`、`training_trajectory_summary.{json,csv}` |
| 预注册 claim ledger 未按原名输出 | plan Milestone I | `FINAL_ACCEPTANCE.json` |

汇总记录：`results/topic5_lbss_rnn_v0_2/CLOSEOUT_AUDIT_COMPLETION.json`。

补做同时暴露一个 order-shuffle 的实现边界：`derange_rank_sets` 用的是**随机循环移位**，它满足 spec §7 的「无固定点」，但保留了 rank 集合的相对循环次序，只破坏每个事件 m−1 个相邻转移中的 1 个。实测跨 93 个 unit 的 mean Kendall distance 中位数为 0.431（min 0.352 / max 0.606），均匀随机置换的参考值是 0.5。对已通过的顺序主效应，这个控制偏保守（真实效应只会更大）；但对 `true_order_vs_shuffle_distal`（P=0.0547，边缘）这类比较，它可能造成检验力不足。**本轮不改算法**——改了就要重跑 93 个 shuffle unit 及其全部下游；已加测试锁住「这是循环移位而非均匀 derangement」，避免后续误读。

### 10.4 Target 解封后的 Figure-only recovery

初次制图因 Matplotlib `Colorbar.set_ticks` API 兼容性失败。统计和 target scoring 已完成，失败仅发生在 H_figure。修复只涉及 API 与布局，没有重新训练、重建 field、重算统计或更换代表患者。原失败记录和新旧 producer hash 均保存在：

`results/topic5_lbss_rnn_v0_2/POST_UNSEAL_FIGURE_RECOVERY.json`

### 10.5 Figure 6 重建（2026-08-11 复审）

复审目视检查发现初版 Figure 6 有四处标签压在数据上（C 右侧 y 轴标题压住左图 Shuffle 列、D 右侧 y 轴标题压住 shuffle 地图的触点、F 右侧 y 轴标题压住左图轨迹、F 左侧 y 轴标题越界进 E），E 的两条 colorbar 刻度文字 "late"/"low" 相连成一串，且 F 的左右两块用**同一个灰蓝色分别表示 Local 和 Extra**。因此 `FIGURE_VISUAL_QA: PASS_AFTER_LAYOUT_RECOVERY` 不成立。

重建内容（只改绘图，不改任何数值）：全图统一一套「连接条件 → 颜色」映射；加宽面板内间距并让等比例地图靠左对齐以让出标签空间；两条 colorbar 收缩留白；A 加图元图例、B 加 recruitment-rank 色标与坐标含义、D 加 source/target 图例；E 标注该代表病例的 |r|=0.64 与其**带符号相关 −0.64**（评分口径是 mirror/maxAB 下的 |r|，不标注会让读者把反相关读成一致）；D 右侧改为绘制「毁掉顺序」的效应并以虚线画出「同一 arm 换 seed」的参考水平。绘图脚本现在带 `assert_no_label_overlap`，任何标签压到相邻面板会直接报错而不是靠人眼。

## 11. 可以写与不可以写

### 可以写

> Patient-specific ordered interictal rank sequences were sufficient to train recurrent networks that freely generated held-out propagation from the observed first rank. A strongly connected local recurrent backbone was sufficient, and shuffling the true rank order abolished much of the generative advantage.

> Destroying the true event order perturbed the coarse contact-space organization of the learned computation substantially more than changing the random seed did, but multiple local and nonlocal connectivity solutions achieved comparable propagation fidelity.

> Frozen model-generated fields showed a positive but non-confirmatory correspondence with early-ictal broadband energy.

### 不可以写

- “模型恢复了患者真实白质连接或真实 nonlocal pathways”；
- “task-selected shortcuts 显著优于随机或局部连接”；
- “selected shortcuts 对 distal propagation 具有已证实的功能双重解离”；
- “RNN 学到的 shortcut network 在发作早期被特异复用”；
- “模型预测了某次发作”或“发现了跨事件发作状态”；
- “geometry 是完全 train-only 独立发现的患者病理轴”；
- “L3 相对 local-only 的 distal null 不是检测能力问题”（正对照没有为该对比建立灵敏度）；
- “真实顺序选出了一条可辨识的 shortcut 通路 / consensus pathway”（粗 pattern 不跨 seed 复现）；
- “selected nonlocal 在跨状态上显著更差”（min-of-controls 统计量在零假设下即为负）。

## 12. 最终科学判决

判决名沿用 plan Milestone I 预先固定的七项；初版把 B1/B2 合并、并把 attenuation 双重解离并入 Claim C，属于结果出来后改动 claim ledger，已还原。机器可读版本：`results/topic5_lbss_rnn_v0_2/FINAL_ACCEPTANCE.json`。

```text
CLAIM_A_LOCAL_BACKBONE_SUFFICIENT:                            SUPPORTED
CLAIM_B1_NONLOCAL_INCREMENT:                                  NOT_SUPPORTED
CLAIM_B2_SELECTIVE_NONLOCAL_BENEFIT:                          NOT_SUPPORTED
CLAIM_C_TRUE_ORDER_SELECTS_FUNCTIONAL_SHORTCUT_ORGANIZATION:  PARTIALLY_SUPPORTED
CLAIM_D1_EARLY_ICTAL_FIELD_CORRESPONDENCE:                    INCONCLUSIVE
CLAIM_D2_SHORTCUT_SPECIFIC_CROSS_STATE_CONTRIBUTION:          NOT_SUPPORTED
ATTENUATION_DOUBLE_DISSOCIATION:                              NOT_SUPPORTED
ENGINEERING_EXECUTION:                                        COMPLETE_AFTER_AUDIT_BACKFILL
FIGURE_VISUAL_QA:                                             PASS_AFTER_LAYOUT_REBUILD
```

各判决的附带边界：

- **A**：comparator 借自 v0.4 且 LBSS 侧多出 gradient clipping，方向上有利于 L0，读作「至少这么大」。
- **B1**：正对照未为该对比建立可检测下限 → 这是 absence of evidence，不是 evidence of absence。
- **B2**：正对照对这两个对比**确实**建立了灵敏度，因此 NOT_SUPPORTED 在这里更有分量。
- **C**：顺序对粗组织的扰动显著大于 seed 扰动（+0.27~+0.35），但两个 arm 的 pattern 都不跨 seed 复现 → 只到「顺序影响粗计算」，到不了「选出通路」，故为 PARTIALLY。
- **D1**：n=10，within-shaft null 下未确认。
- **D2 / 双重解离**：min-of-controls 统计量在零假设下即为负，重标定后 L3 只是「不更好」。

本轮最有价值的结论不是“找到了癫痫 long-range motif”，而是把模型空间收窄到：

1. recurrence 确实必要；
2. local backbone 已经足够；
3. 真实顺序携带稳定计算信息；
4. 当前数据没有要求一种唯一的 nonlocal shortcut organization；
5. early-ictal benchmark 仍主要反映宽尺度 scaffold/source，而非 selected-shortcut 特异复用。

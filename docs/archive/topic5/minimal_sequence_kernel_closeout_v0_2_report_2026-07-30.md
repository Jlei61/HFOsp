# Topic 5 最小序列结构收口 v0.2：where / how / when 分层验收

## 1. 审阅结论

### 一句话判断

本轮已经按冻结合同完成。结果支持“稳定的患者特异 contact 招募先验 + 很短的事件内顺序修正”，但不支持把该修正解释为唯一的 recurrent mechanism、真实时间的发作状态，或 patient-mean early-ictal field 的额外预测来源。

### 完成度

**科学合同执行完成度：100/100。**

- 34 人、3 seeds，共 102 个既有模型单元完成同分母重评分；
- 34 人、3 seeds，共 102 个 FIR-H3 单元完成训练与 held-out 评价；
- contact choice、STOP、continue 与 total NLL 已精确分解；
- linear-state 已展开为输入—输出 lag kernel；
- FIR-H3、跨数据集冻结迁移、rank-set 容差、matched context、early-ictal Gate 0 和跨事件 Gate 1 均已完成；
- 两张图均已生成并目视验收。

**科学结论强度：78/100。** 扣分不是实验欠项，而是结果本身的边界：跨数据集确认非对称、FIR-H3 未优于无序基线、ordered residual 未增加 early-ictal association，真正的 seizure-specific 1–150 Hz 动态 target 仍不可辨识。

---

## 2. 本轮实际回答了什么

本轮把三个科学对象彻底拆开：

| 层级 | 问题 | 本轮结论 |
|---|---|---|
| **Where** | 哪些 contact 容易参与？ | 稳定、患者特异；也是当前跨状态对应的主要来源 |
| **How** | 一次事件内部下一步走向哪里？ | 当前 rank 和前一 rank 含小但稳定的额外信息 |
| **When** | 患者如何随真实时间进入发作？ | 当前 event-reset 模型没有检验；独立 feasibility gate 尚未通过 |

冻结的预测分解为：

\[
\ell_{e,t+1,c}
=
\alpha_{p,c}
+f_{\mathrm{set}}(U_{e,t},c)
+f_{\mathrm{ord}}(S_{e,t-2:t},c).
\]

其中 `static prior`、`unordered prefix` 和 `ordered history` 是相加的预测成分，不是因果加工链。模型每场事件开始时 reset；一个 step 是 rank step，不是秒、分钟或小时。

---

## 3. 数据与评估合同

### 3.1 队列

- 总计 34 人：Epilepsiae 18 人，Yuquan 16 人；
- 每位患者 3 seeds；
- chronological train80 / heldout20；
- 所有主比较以患者为统计单位。

### 3.2 同分母保证

H1、H2、H3、full、rank shuffle、linear-state、FIR-H3 及 lag deletion 使用相同：

- held-out decision identities；
- eligible-contact mask；
- candidate set；
- STOP 定义；
- event denominator。

102/102 重评分单元通过合同审计；原始 joint likelihood 的重构最大误差为 \(9.54\times10^{-7}\)。FIR-H3 的冻结无序基线和决策计数也在 102/102 单元一致。

### 3.3 likelihood 分解

每个 held-out decision 被拆成：

1. `contact-choice NLL`：已知事件继续时，下一个 contact/rank set 的身份；
2. `STOP contribution NLL`：模型对继续或结束的判断；
3. `total NLL`：两者联合。

零容差 heldout 中没有有效 multi-contact ties；全数据 5,902,546 个 rank sets 中仅 73 个 ties。因此本数据下无法把独立的 cardinality head 作为主要可辨识对象。

---

## 4. 主要结果

### 4.1 Where：静态 contact scaffold 是最稳的结果

- train80–heldout20 participation Spearman 中位数：**0.893**；
- structured shaft-preserving null 合格的 33 人中：**33/33** 真实 heldout 高于 null；
- patient-mean early-ictal field 的 orientation-free absolute association 中位 margin：**0.106**，**15/16** 为正；
- within-shaft null 下仍为 **14/16** 为正。

**批注：** 这说明跨状态对应首先是稳定空间招募倾向，而不是 RNN hidden state 的特异贡献。它支持“同一病理 scaffold 在间期和发作早期被重复使用”的保守版本，但不是逐触点动态 replay。

### 4.2 How：第二个 rank 提供 contact identity 信息，第三个 rank 主要帮助 STOP

| 比较 | contact-choice 增益，中位 nats/decision | 阳性患者 | Wilcoxon P | 解释 |
|---|---:|---:|---:|---|
| H2 − H1 | **0.0110** | 28/34 | 0.000164 | 前一 rank 对下一 contact 有额外信息 |
| H3 − H2 | 0.00151 | 22/34 | 0.121 | 第三 rank 对 contact identity 无稳定增益 |
| H3 − rank shuffle | **0.0295** | 30/34 | 0.000193 | 真实局部顺序优于同分母顺序打乱 |
| Full − H3 | **−0.00418** | 9/34 | 0.00300 | 更早历史不但无益，且损害 contact 泛化 |

H3 − H2 的 STOP contribution 为 **0.00978 nats/decision，34/34 为正，P=1.16×10⁻¹⁰**。

**批注：** “最近 2–3 rank 有用”必须进一步收窄：对传播到哪里而言，证据集中在当前 rank 和前一 rank；第三 rank 的主要价值是判断事件是否结束。不能把总 NLL 改善全部写成传播路径信息。

### 4.3 线性状态的 lag kernel：可解释对象是 \(K_k=CA^kB\)

删除单个 lag 后的 contact-choice 损失：

| lag | 中位损失 nats/decision | 阳性患者 | Wilcoxon P |
|---|---:|---:|---:|
| \(K_0\) | **0.0389** | 30/34 | 8.87×10⁻⁸ |
| \(K_1\) | **0.00684** | 24/34 | 0.0374 |
| \(K_2\) | 0.00158 | — | 0.301 |
| \(K_{3+}\) | 0.00018 | — | 0.973 |

跨 seed 的 contact-kernel cosine 中位约为 0.73、0.72、0.71、0.70（\(K_0\) 至 \(K_3\)）。

有限时域 Hankel 谱的 90% 与 95% energy rank 中位均为 2，effective order 中位约 1.29。但本轮没有做 balanced truncation 后的 held-out NLL 等价检验。

**批注：** \(K_k\) 不随 hidden-state 坐标旋转，是比 hidden PCA 更可靠的输入—输出对象。Hankel 结果只能写成“预测映射具有集中的低阶谱”，不能写成“脑内存在二维癫痫流形”。

### 4.4 FIR-H3 没有取代 linear-state

显式模型为：

\[
\Delta\ell_{\mathrm{ord}}
=K_0x_t+K_1x_{t-1}+K_2x_{t-2},
\]

无序 baseline 先冻结，FIR branch 只学习 ordered residual。

- FIR-H3 − retrained unordered，contact-choice 中位：**+0.00499 nats**，P=0.427；
- FIR-H3 − selected linear-state，contact-choice 中位：**−0.00843 nats**，P=0.078；
- FIR-H3 − selected linear-state，total NLL 中位：**−0.0235 nats**，P=0.00218。

**批注：** 显式 H3 并未达到与 linear-state 等价的门。因此最终不能简单删除 state 并宣称这是普通三阶 Markov；当前更安全的表述是，简单线性状态以参数共享的方式利用了很短的历史，但其独立确认仍不足。

### 4.5 架构结论

既有 target-blind 架构审计中，linear-state 是唯一通过 family-wise 校正的递归家族；GRU、rate RNN 和 low-rank families 没有稳定的额外收益。本轮进一步显示：

- linear-state − unordered 的 contact-choice 中位增益为 **0.00951 nats**，P=0.0112，但 bootstrap CI 跨 0；
- linear-state − matched rank shuffle 为 **0.0250 nats**，29/34 为正，P=9.72×10⁻⁵；
- 更复杂的非线性或 low-rank 结构没有必要性证据。

**批注：** 这不是“大脑动力学是线性的”。它只表示在当前任务和样本量下，简单线性滤波器泛化最好。

---

## 5. 稳健性与独立确认

### 5.1 rank-set 时间容差

contact-choice 增益在 0、1、2 ms 下方向保留；到 5 和 10 ms 时消失。随着容差增加，被合并的 near-simultaneous ranks 大量增加。

**批注：** 结果对小幅 peak jitter 稳健，但不是对任意 rank 分组都稳健。论文应明确写“0–2 ms sensitivity range”，不能笼统写 timing-definition independent。

### 5.2 跨数据集冻结确认

预先冻结 Epilepsiae → Yuquan：

- linear-state：−0.00648 nats，7/16 为正，P=0.130；
- FIR-H3：+0.00404 nats，9/16 为正，P=0.433。

反向 Yuquan → Epilepsiae sensitivity：

- linear-state：+0.0371 nats，16/18 为正，P=0.00233；
- FIR-H3：+0.0128 nats，13/18 为正，P=0.167。

**批注：** 固定方向确认失败，反向只有 linear-state 阳性。不能写成 two-cohort replicated；更像两个数据集在事件数量、contact 数和记录结构上的异质性。

### 5.3 matched observed contexts

精确相同 unordered prefix、相同长度和 candidate mask 的重复上下文数量足够进行 H2/H3 audit，但朴素 empirical ordered lookup 在 heldout 上反而更差。

**批注：** exact context 极度稀疏，未能提供数据层面的反事实确认。reverse/reset 仍只能解释为“模型输出依赖顺序”，不能升级为神经机制因果证据。

---

## 6. 跨状态与 when 分支

### 6.1 patient-mean early-ictal field

在控制 static 与 unordered 后：

- ordered − unordered 的 absolute field association 中位：**−0.0986**；
- true order − shuffle 的增量中位：**+0.00850**，P=0.604。

**结论：** 当前 patient-mean early-ictal correspondence 主要来自 static scaffold；ordered residual 的额外跨状态价值未建立。

### 6.2 Gate 0：seizure-specific target 可靠性

冻结的 strict target 为 clinical onset、1–150 Hz、0–10 s，共 16 人、106 seizures：

- patient-mean split-half ρ 中位：**0.821**；
- 仅 5 人有至少 4 次 seizures；
- 每次 seizure 只有一个聚合场，无法从现有 artifact 估计 exact BB150 seizure-specific residual 的交叉验证可靠性。

1–45 Hz 的 0–5/5–10 s proxy 在 5 人中显示 residual matched-minus-mismatched 中位 0.717，bootstrap CI [0.414, 0.919]，但不能替代冻结的 1–150 Hz target。

**Gate 0：`BLOCKED_EXACT_BB150_SEIZURE_RESIDUAL_RELIABILITY_UNIDENTIFIABLE`。**

### 6.3 Gate 1：跨事件状态 feasibility

本轮只用固定的线性、带真实 IEI 的 1 h half-life state 做自监督 screen，并在记录间隔超过 10 min 时 reset；没有训练新 RNN。

- recent unordered average − static：+0.0222 nats/contact，33/34 为正；
- time-state − best non-state control：+0.00438 nats/contact，26/34 为正，P=0.0118；
- Epilepsiae：+0.00946，17/18，P=7.63×10⁻⁵；
- Yuquan：+0.000883，9/16，P=0.744；
- 30/34 同时超过 circular-shift 与 block-shuffle pairing null。

**Gate 1：`PROVISIONAL_COHORT_SIGNAL_NOT_REPLICATED_ACROSS_DATASETS`。**

**批注：** 这提示跨事件历史可能存在，但尚未在两个数据集分别复现，更没有连接到 seizure-specific early-ictal residual。它属于下一篇/下一阶段的 `when` 可行性结果，不进入本轮 within-event RNN 的主结论。

---

## 7. P0 / P1 问题

### P0

无运行或数据泄漏层面的 P0。所有主比较的 decision/mask/denominator 已核对，within-event rescore 未读取 ictal target。

### P1

1. **独立数据集确认失败。** 当前顺序增益不能写成跨队列稳健复制。
2. **FIR-H3 未达到 linear-state 等价。** 不能把结果直接降维成已确认的 lag-3 Markov kernel。
3. **第三 rank 的主要收益来自 STOP。** 不能用总 NLL 把事件长度结构写成传播 identity。
4. **exact seizure-specific BB150 target 不可辨识。** 不能启动动态 seizure prediction。
5. **matched-context 数据验证阴性。** 模型 intervention 不是神经因果扰动证据。

---

## 8. 六联图的科学含义

| Panel | 科学问题 | 结论 |
|---|---|---|
| A | 预测信息如何分层？ | static + unordered + recent ordered；仅为 within-event memory |
| B | contact prior 是否稳定？ | chronological heldout 高于 shaft-preserving null |
| C | contact 与 STOP 分别需要多长历史？ | contact 在 H2 后饱和，H3 主要改善 STOP |
| D | 是否需要复杂 RNN？ | linear-state 最稳定；非线性与 low-rank 无必要性 |
| E | 可识别的 lag 结构是什么？ | \(K_0\) 主导，\(K_1\) 较小，\(K_2/K_{3+}\) 无稳定 contact 增益 |
| F | ordered history 是否解释 early-ictal field？ | 跨状态 association 主要为 static；ordered residual 未建立 |

辅助三联图分别展示 rank-set 容差、early-ictal target reliability gate 和跨事件 state feasibility，不与主图的 within-event claim 混合。

---

## 9. 允许和禁止的论文口径

### 允许

> 在稳定的患者特异 contact 招募骨架之上，间期群体事件包含小幅但可检测的短程事件内顺序信息。该信息主要由当前和前一个 rank set 提供，并可由简单线性递归状态压缩；更早历史、复杂非线性和显式 FIR-H3 未显示稳定必要性。患者平均 early-ictal field 的对应主要来自静态 scaffold，而非 ordered residual。

### 禁止

- RNN 恢复了真实癫痫 latent state 或脑流形；
- rank-step memory 是生物学时间常数；
- ordered history 可预测发作时间；
- ordered residual 独立解释 early-ictal recruitment；
- 结果已在两个独立数据集双向确认；
- Hankel 谱证明脑内只有一到两个状态维度。

---

## 10. 最小下一步

本论文的 within-event architecture zoo 到此冻结，不再增加 GRU、low-rank rank、seed 或门控。

若未来继续 `when` 分支，唯一合理顺序是：

1. 从原始 seizure 数据构建每次 seizure 的 exact 1–150 Hz、clinical-onset `[0,10] s` 可重复 target；
2. 先证明 seizure-specific residual 高于测量噪声；
3. 再冻结 IEI-aware inter-event model，并在两个数据集分别复现 next-event prediction；
4. 最后才做 leave-one-seizure-out residual prediction。

---

## 11. 交付物

- 机器验收：`results/topic5_minimal_sequence_kernel_closeout/FINAL_ACCEPTANCE.json`
- 主汇总：`results/topic5_minimal_sequence_kernel_closeout/MINIMAL_SEQUENCE_KERNEL_SUMMARY.json`
- 辅助汇总：`results/topic5_minimal_sequence_kernel_closeout/MINIMAL_SEQUENCE_AUXILIARY_SUMMARY.json`
- 六联图：`results/paper-ready-figure/fig_topic5_minimal_sequence_kernel_closeout/figures/topic5_minimal_sequence_kernel_closeout.{png,pdf}`
- 辅助三联图：`results/paper-ready-figure/fig_topic5_minimal_sequence_kernel_closeout/figures/topic5_sequence_definition_and_when_audit.{png,pdf}`
- 图说明：`results/paper-ready-figure/fig_topic5_minimal_sequence_kernel_closeout/figures/README.md`
- 冻结 spec：`docs/superpowers/specs/2026-07-30-topic5-minimal-sequence-kernel-closeout-v0_2.md`
- 执行 plan：`docs/superpowers/plans/2026-07-30-topic5-minimal-sequence-kernel-closeout-v0_2.md`

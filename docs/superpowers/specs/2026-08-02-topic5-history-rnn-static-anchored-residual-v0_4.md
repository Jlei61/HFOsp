# Topic 5 发作前历史对静态 A/B 病理场的动态残差修正 v0.4

英文工作名：**History-conditioned early-ictal field refinement**

## 1. 本轮只回答什么

论文已有结果表明，患者特异的静态 A/B 间期病理场与发作早期 contact-energy field 存在无符号、双候选对应。此前 direct-transfer 阴性的一部分来自把这个集合值问题强行改成单一有符号场预测。

本轮不再重新发现 A/B，不再扩展模型 zoo，也不要求 RNN 单独生成完整发作场。唯一问题是：

\[
\boxed{
\text{冻结且已经有效的静态 A/B 候选场}
+
\text{发作前间期历史的受限残差修正}
\rightarrow
\text{更高的 early-ictal maxAB concordance}
}
\]

这里的输出不是唯一方向的场，而是一个无符号、集合值的候选：

\[
\widehat{\mathcal F}_{p,s}
=
\{\widehat F^A_{p,s},\widehat F^B_{p,s}\}.
\]

因此本轮检验的是“发作前历史是否能修正静态病理场的表达”，不是“RNN 是否预测了唯一发作传播方向”，也不是“历史如何因果塑造网络”。

---

## 2. 允许和不允许的结论

### 2.1 阳性结果最多支持

> 在稳定的患者特异静态 A/B 病理场之外，发作前间期历史包含可被受限 recurrent state 利用的、与发作早期空间能量场有关的增量信息。

若联合微调模型超过冻结状态模型，才可进一步写：

> early-ictal target supervision 改变了 target-blind recurrent dynamics 的有效读出。

若正确发作历史还优于同患者其他发作的历史，才可称为 seizure-matched history information。

### 2.2 不允许升级为

- 唯一有符号发作场或发作方向的前瞻预测；
- 发作时刻预测或临床预警器；
- 间期历史因果塑造发作；
- RNN hidden unit 对应细胞级 E/I 状态；
- decay 参数是生物时间常数；
- 静态 A/B 基底是完全前瞻得到的。

---

## 3. 数据与 endpoint 合同

### 3.1 Primary target

主 endpoint 恢复为论文现有核心口径：

- `clinical_onset` 对齐；
- onset 后 `[0,10] s`；
- **1–45 Hz broadband contact-energy field**；
- 每次发作单独构建；
- onset 前 `600 s` guard；
- contact name exact join，禁止 fuzzy join。

1–150 Hz 只作固定敏感性分析：主模型、超参数和 seeds 全部冻结后，用 **1–45 Hz 训练所得预测**直接评分 1–150 Hz target；不得用 1–150 Hz 选择模型或重新调参。此前 v0.3 的 1–150 Hz maxAB 结果只作 provenance/sensitivity，不再称为本轮 primary anchor。

### 3.2 发作前间期历史

对患者 \(p\)、发作 \(s\)：

\[
\mathcal H_{p,s}
=
\{X_{p,e}:t_{p,e}<t^{onset}_{p,s}-600\ \mathrm{s}\}.
\]

仅使用同一连续记录段内的历史，并排除前一次发作后的 postictal 区间。事件按真实时间排序，输入包括：

- target-blind event embedding；
- event participation/rank 信息；
- 真实 event time 和 IEI；
- segment/reset metadata。

最后一个事件后的状态必须纯衰减到统一 cutoff，不能停在最后事件时刻。

### 3.3 冻结静态 A/B 基底

只读使用：

```text
results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json
```

读取并按 contact name 对齐：

- `field_models.own_a.template_field`；
- `field_models.own_b.template_field`；
- `contact_order`。

禁止从 early-ictal target 重新估计 A/B、axis、plane、contact set 或候选方向。

必须在文档和图注中明确：静态 A/B 是由患者**全记录的间期数据**回顾性估计的，可能包含该次发作后的间期事件。它没有读取 early-ictal target，因此不是 target leakage；但整体模型不是完全前瞻预测器，只有 residual-history 支路严格 causal。本轮不重建 prefix-only static A/B。

### 3.4 Cohort

- `epilepsiae_1146`：只作工程 smoke，不作模型选择或科学证据；
- 其余 15 位 strict clinical-onset 患者：primary evaluation cohort；
- 每个 outer fold 留出 1 位患者，target supervision 只使用其余 14 位 primary 患者；
- 不再运行 `1146` supportive formal fold。

评分触点为 static field、rank dataset 与 target 的 exact-name 交集。每患者、每次发作的实际 contact denominator 必须写入结果。

---

## 4. P0：outer-fold 内必须使用同一编码坐标

对每个 heldout patient \(p\)，使用一个 target-blind outer-fold encoder：

\[
E^{(-p)}.
\]

该 encoder 使用审计过的 c30 LOSO checkpoint，且没有看过 heldout patient \(p\)。在这个 outer fold 内：

- 14 位 target-training patients；
- heldout patient \(p\)；

的所有事件都必须由**同一个** \(E^{(-p)}\) 编码。禁止把不同 subject-specific LOSO checkpoints 产生的 hidden coordinates 混进同一个共享监督 readout。

cache 必须按 outer fold 组织，并保存 encoder checkpoint hash：

```text
cache/outer_<heldout_patient>/<encoded_subject>/...
```

不同 outer folds 可以有不同的 \(E^{(-p)}\)，同一个 outer fold 内不可以混用。

---

## 5. 四个冻结模型

### 5.1 共同的静态锚定形式

定义：

\[
\operatorname{center}(v)=v-\bar v,
\]

\[
\operatorname{unit}_{\epsilon}(v)
=
\frac{\operatorname{center}(v)}
{\sqrt{\|\operatorname{center}(v)\|_2^2+\epsilon}}.
\]

对历史分支产生的 raw residual \(\delta^k\)：

\[
r^k=
\begin{cases}
0,&\|\operatorname{center}(\delta^k)\|_2<\epsilon_r,\\
\dfrac{\operatorname{center}(\delta^k)}
{\|\operatorname{center}(\delta^k)\|_2+\epsilon},&\text{otherwise}.
\end{cases}
\]

最终候选场：

\[
\widehat F^k
=
\operatorname{unit}_{\epsilon}
\left[
\operatorname{unit}_{\epsilon}(F^k_{static})+g_k r^k
\right],
\quad k\in\{A,B\}.
\]

其中 \(g_k=\sigma(a_k)\in(0,1)\)，初始化为 \(10^{-3}\)。A/B gains 为 outer-fold training cohort 共享标量，不设 patient-specific gain。初始化时必须记录最终候选场与 static A/B 的 L2 差和夹角。工程测试中显式令 \(g=0\) 时，输出必须逐位等于规范化 static A/B。微小 residual 不能因单位化被放大。

### 5.2 M0：`STATIC_AB`

只输出冻结的 A/B 候选场，不训练任何参数。

### 5.3 M1：`STATIC_AB_PLUS_FROZEN_HISTORY_HEAD`

- 使用同一 outer-fold encoder \(E^{(-p)}\) 的冻结 `TimeDecayHistoryGRU`；
- 状态跨事件持续，并衰减到 cutoff；
- **完全训练** A/B contact-query heads 和 residual gains；
- recurrent weights 与 decay 全程冻结。

M1 回答：已有 target-blind recurrent state 是否已经包含可读出的 early-ictal 信息。

### 5.4 M2：`STATIC_AB_PLUS_TIME_AWARE_NONRECURRENT`

不使用 recurrent state。对同一 causal history 计算一个固定、无超参数扫描的 summary：

1. 以 cutoff 为参照、固定指数衰减时间常数 \(\tau_0=2\) h 的 event-embedding EWMA；
2. 全历史 event-embedding mean；
3. 全历史 event-embedding element-wise max；
4. last-event embedding；
5. `log1p(event_count)`；
6. `log1p(history_span_seconds)`。

拼接后只允许经过一个共享线性投影压到 16 维，再通过与 M1 同类型的 A/B contact-query residual heads 和 gains；不加非线性隐藏层。M2 回答：简单的活动负荷与最近事件汇总是否已经足够，不需要 recurrence。

### 5.5 M3：`STATIC_AB_PLUS_JOINT_RNN`

结构与 M1 相同。训练前 30 epochs 与 M1 共用同一个 frozen-recurrent head-training checkpoint；随后分叉：M1 再训练 30 epochs、但 recurrent/decay 继续冻结，M3 则在相同 mini-batch 顺序和 head optimizer state 下联合微调：

- HistoryGRU recurrent weights；
- decay；
- A/B residual heads；
- gains。

EventRNN/within-event encoder、contact embedding、event normalization 和 static A/B 始终冻结。不增加第二层 GRU、attention、dense contact-to-contact mixing 或 patient-ID embedding。

M3−M1 是关键解释对比：它检验 early-ictal supervision 是否需要改变 recurrent dynamics，而不只是训练一个更好的输出头。

---

## 6. 训练损失

### 6.1 与 maxAB 一致的 soft-rank loss

对 prediction 使用 soft-rank：

\[
\widetilde r_i(v)
=1+\sum_{j\neq i}
\sigma\left(\frac{v_j-v_i}{\tau_r}\right),
\quad \tau_r=0.1.
\]

Target 使用精确 mid-rank。对两个候选分别计算绝对 rank correlation \(r_A,r_B\)，再用 \(\tau_m=0.05\) 的 bounded soft maximum：

\[
m=
\sum_{k\in\{A,B\}}
\operatorname{softmax}(r_k/\tau_m)r_k.
\]

每次发作：

\[
\mathcal L_s
=1-m_s
+\lambda_g(g_A^2+g_B^2)
+\lambda_{anchor}\|\theta_{rec}-\theta_{rec,0}\|_2^2.
\]

冻结：

- \(\lambda_g=10^{-3}\)；
- \(\lambda_{anchor}=10^{-4}\)；
- anchor penalty **只作用于 HistoryGRU recurrent weights 与 decay**；
- 新 residual heads 和 gains 不进入 anchor penalty。

先对同一患者的 seizures 取平均，再对患者取平均。A/B 与正负号的选择必须留在 maxAB loss 内，禁止恢复单一 signed-output loss。

### 6.2 单一固定训练器

不再用 `1146` 做 learning-rate × epoch 搜索。所有 outer folds 预先冻结：

- optimizer：`AdamW`，`weight_decay=0`；
- common frozen-recurrent head stage：30 epochs，LR `3e-4`；
- M1：从 common stage 继续 30 epochs，recurrent/decay 保持冻结；
- M2：总计 60 epochs，LR `3e-4`；
- M3：从同 seed common-stage checkpoint 继续联合训练 30 epochs；
- M3 recurrent/decay LR `1e-4`；
- M3 heads/gains LR `3e-4`；
- gradient clipping `1.0`；
- hidden size `16`；
- event chunk `256`，chunk 间状态与梯度不断开；
- fixed seeds：`11, 29, 47`。

不读取 heldout target early stop，不选择最好 seed。正式预测对三个 seed 的 A/B candidate fields 分别做 contact-wise 平均，再计算 exact maxAB。

`epilepsiae_1146` 只允许做 1 epoch 的 shape、gradient、OOM、determinism 和 output-schema smoke；不得依据其科学指标修改上述配置。

---

## 7. 正式评分和针对性对照

### 7.1 Primary metric

每次 seizure：

\[
T_{p,s}(M)
=
\max
\left(
|\rho_S(\widehat F^A_M,Y)|,
|\rho_S(\widehat F^B_M,Y)|
\right).
\]

聚合固定为：

```text
per-seizure maxAB -> per-patient median -> patient-level cohort statistic
```

### 7.2 Primary 与解释性 contrasts

Primary：

\[
\Delta_{M3-M0}=T(M3)-T(M0).
\]

必须同时报告：

- \(M3-M1\)：target supervision 是否改变 recurrent dynamics；
- \(M3-M2\)：recurrence 是否超过固定时间汇总；
- \(M1-M0\)：冻结 target-blind state 是否可读；
- \(M2-M0\)：简单时间汇总是否可读。

每项报告 patient median、bootstrap 95% CI、正/负/并列数及 tie-tolerant exact Wilcoxon。没有复合 hard gate。

### 7.3 Event-order shuffle

不重训模型。对每次发作的整段 causal history，将事件身份随机分配到原有时间槽，保留：

- event multiset；
- event count；
- time slots / IEI distribution；
- cutoff；
- static A/B 和 contact set。

每次发作使用 32 个固定随机置换，比较 M3 true-order 与其 shuffle 平均。不能只打乱最近 64 个事件。

### 7.4 Within-patient history swap

不重训模型。对具有至少两次合格发作的患者，把发作 \(s\) 的 static A/B、contact set 和 target 保持不变，改用同一患者其他发作的历史状态。遍历所有合格 donor histories，并以 donor-score 中位数作为 swapped 对照。

它回答的是 residual 是否匹配到具体发作历史，而不是患者指纹。只有一场合格发作的患者记为 NA，不进入该对比。

### 7.5 Zero-state 的位置

`zero_state` 不再是科学对照。因为本模型没有 residual bias，zero state 理论上就应还原 static A/B。它只保留为单元测试，不进入正式统计、图或结论。

### 7.6 Matched channel null

- primary 1–45 Hz：每患者 5000 次 all-contact target-label shuffle；
- 同一 permutations 用于所有模型；
- 每次 seizure 独立打乱；
- 每个 draw 重新完成 A/B candidate 和 absolute-sign selection；
- 每个 draw 先 seizure maxAB，再 patient median。

---

## 8. 预先冻结的解释层级

| 结果 | 允许解释 |
|---|---|
| `M3 <= M0` | 当前联合 RNN residual 未改善静态 A/B 场 |
| `M3 > M0` 且 `M3 ≈ M1` | 已有 target-blind state 可读；无证据表明 target supervision 改变 dynamics |
| `M3 > M0` 且 `M3 ≈ M2` | 简单时间汇总足够；不需要 recurrence |
| `M3 > M1,M2` 但 correct-history ≈ swapped-history | 联合 RNN 有额外容量，但未建立发作匹配的历史信息 |
| `M3 > M1,M2` 且 correct-history > swapped-history | 支持 seizure-matched history-conditioned field refinement |
| true-order > order-shuffle | 支持真实时间组织提供额外信息 |
| true-order ≈ order-shuffle | 增量来自历史内容/负荷，不支持顺序特异性 |

任何 secondary contrast 阴性都不停止剩余 formal folds，也不触发临时改模型。

---

## 9. 工程验收

必须通过：

1. primary endpoint manifest 明确为 1–45 Hz，1–150 Hz 标记为 no-retrain sensitivity；
2. 同一 outer fold 的所有 patients 使用同一 encoder checkpoint hash；
3. heldout patient target 对训练器不可见；
4. `g=0` 时逐位还原 static A/B；
5. residual norm 小于阈值时修正严格为零；
6. 初始化候选场与 static A/B 的差异接近零并被记录；
7. anchor penalty 不作用于新 heads/gains；
8. cutoff decay 使用 `cutoff-last_event_time`；
9. chunked 与 unchunked 短序列输出和梯度一致；
10. A/B 交换和候选整体翻号不改变 loss；
11. M1 与 M3 只在 recurrent/decay 是否可训练上不同；
12. order shuffle 覆盖完整 causal history；
13. history swap 只在患者内进行并保持 contact alignment；
14. channel null 每个 draw 重做 A/B/sign selection；
15. 每 fold 有 seed、epoch、loss、gain、decay、gradient、显存日志及 `DONE.json`/`FAILED.json`；
16. 0 OOM / NaN；资源不足时只改并发、checkpointing 或 batch accumulation，不改科学合同。

这些是正确性条件，不是科学阳性 gate。

---

## 10. 输出

```text
results/topic5_history_conditioned_field_refinement_v0_4/
├── cache/outer_<heldout_patient>/
├── development_smoke/
├── per_subject/
├── logs/
├── figures/
│   └── README.md
├── history_conditioned_field_patient_metrics.csv
├── history_conditioned_field_channel_null.csv
├── HISTORY_CONDITIONED_FIELD_SUMMARY.json
└── REPRODUCIBILITY_MANIFEST.json
```

本轮不覆盖 v0.2/v0.3 产物；它们保留为 signed-readout、1–150 Hz sensitivity 和旧 frozen-state transfer 的方法学 provenance。

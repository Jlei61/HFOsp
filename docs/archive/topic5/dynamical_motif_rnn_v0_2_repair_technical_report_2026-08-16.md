# Topic 5.2 动力学 motif RNN v0.2：技术修复报告

日期：2026-08-16  
结果根：`results/topic5_dynamical_motif_rnn_v0_1/`  
修复根：`results/topic5_dynamical_motif_rnn_v0_1/repair_v0_2/`

## 1. 修复结论

\[
\boxed{
\text{ORDERED HISTORY + GENERIC RECURRENCE SUPPORTED; SPECIFIC SPATIAL MOTIFS NOT IDENTIFIED}
}
\]

原 v0.1 的“motif 无增益”方向保留，但原解释不够完整。修复后可以区分三层：

1. 真实 prefix–suffix 关系有预测信息；
2. 已拟合 RNN 内部使用普通局部递归路径；
3. 该路径对 next-contact 任务不是唯一必要实现，M1–M3 额外空间限制也没有增量泛化。

## 2. 修复项

### 2.1 Checkpoint 选择

原正式 checkpoint 用 contact + STOP 联合验证指标。新增 `selection_metric=contact_nll`，重新训练：

\[
28\ \text{patients}\times4\ \text{models}\times1\ \text{seed}=112/112,
\]

0 failure、0 nonfinite。所有 hard-transition 主比较使用这一批 checkpoint；联合指标 checkpoint 仅作敏感性。

### 2.2 容量匹配的无递归历史基线

旧 `STATIC_READOUT` 有两张自由 \(C\times C\) 矩阵，17/28 患者的参数量大于 DM0。新增共享输出基的低秩模型：

\[
\ell_t=b+U V_{start}^{\top}x_1+U V_{hist}^{\top}r_t+c_{geom},
\]

其中 \(r_t\) 为累计 recruited mask。rank 在读取结果前取满足参数总数不超过对应 DM0 的最大值。模型使用相同 split、相同 next-rank/STOP loss；28 位患者各 3 seed，共 84/84，所有患者满足参数上限。

### 2.3 RNN 路径 2×2 消融

原消融混合了“有记忆无空间混合”和“无记忆有空间混合”，无法归因。修复为：

| 历史状态 | 空间递归混合 | 条件 |
|---|---|---|
| 有 | 有 | 完整 DM0 |
| 有 | 无 | recurrent gain 置近零，保留 leak accumulation |
| 无 | 有 | 每个 rank 前重置状态 |
| 无 | 无 | 每个 rank 前重置且 recurrent gain 置近零 |

由于 \(Wh\) 只作用于上一状态，无历史时开关空间混合逐位相同；数据也得到 28/28 精确并列。这一结构性零值用于确认消融逻辑，不作科学阳性。

### 2.4 困难转移

阈值全部由 train split 冻结：

- 前两次预测；
- 第三次及以后；
- contact-centroid 距离位于 train 75% 分位以上；
- 事件长度位于 train 75% 分位以上。

### 2.5 Motif 算子可见度

旧 synthetic generator 未包含 \(\gamma F\)，所以不能审计 M3。新增审计直接使用实际 `MotifRNN`、真实 model-unseen prefix 和冻结读出；人为设置已知 \(\eta,\beta,\gamma\)，用 Bernoulli KL 扫描同族候选。它检验前向算子是否能影响输出，不检验有限事件下 optimizer 能否恢复参数。

### 2.6 发作特征 parity

主评分只纳入与冻结 0–10 s、1–150 Hz 特征逐事件 parity 通过的 224 seizures / 17 patients；pseudo-onset 2464 个。非 parity 事件不再进入结果。

## 3. 结果

### 3.1 有序历史

真实后段配对相对 shuffled continuation：

| n | 中位误差增加 | 95% bootstrap CI | 正方向 | 单侧 P |
|---:|---:|---:|---:|---:|
| 28 | +0.023683 | [+0.018289, +0.034179] | 24/28 | 3.159×10⁻⁵ |

### 3.2 Contact-only checkpoint 下的 motif 增量

| 比较 | all 中位增益 | 正/负/并列 | P | late P | distal P | long-event P |
|---|---:|---:|---:|---:|---:|---:|
| elongated vs even | 0 | 7/3/18 | 0.385 | 0.138 | 0.188 | 0.615 |
| direction-biased vs elongated | 0 | 3/1/24 | 0.313 | 0.188 | 0.938 | 0.938 |
| forward relay vs direction-biased | 0 | 1/0/27 | 0.500 | 0.500 | 1.000 | 1.000 |

Contact-only checkpoints 中非零参数数：\(\eta\) 10/28、\(\beta\) 4/28、\(\gamma\) 1/28。非零参数没有形成队列级 model-unseen gain。

### 3.3 剂量曲线

- 轻微 elongation 在 calibration 上可有很小改善，但跨患者不稳定；较强 elongation 明显增加误差；
- directional bias 在零附近达到最低，正负增强均增加误差；
- forward relay 从最小非零剂量开始单调增加误差。

独立 validation 选点再读 held-out 时，\(\eta,\beta,\gamma\) 的队列中位收益均为 0。

### 3.4 算子可见度

所有预设强度均在 28/28 患者精确恢复到正确 grid value。代表性 truth-vs-zero KL：

| 参数 | 强度 | 中位 KL / available contact |
|---|---:|---:|
| elongation \(\eta\) | 0.4 | 0.004690 |
| directional bias \(\beta\) | 0.6 | 0.006231 |
| relay \(\gamma\)，zero parent | 0.1 | 0.013300 |
| relay \(\gamma\)，directional parent | 0.1 | 0.009162 |

因此三项均能改变输出；M3 尤其不是死算子。该结果不替代 finite-sample identifiability map。

### 3.5 RNN 内部实现与任务必要性

#### 冻结 RNN 内部消融

| 比较 | endpoint | 中位误差增加 | 95% CI | 正方向 | P |
|---|---|---:|---:|---:|---:|
| 去掉历史状态上的局部递归混合 | next contact | +0.102240 | [+0.058801,+0.166116] | 24/28 | 9.425×10⁻⁷ |
| 同上 | STOP | +0.771916 | [+0.363961,+1.091558] | 27/28 | 7.451×10⁻⁹ |
| 仅保留 leak history、不含空间混合 | next contact | −0.002628 | [−0.014893,+0.000490] | 11/28 | 0.943 |
| 同上 | STOP | +0.044105 | [−0.001885,+0.102096] | 18/28 | 0.093 |

完整 RNN 中起作用的是“历史状态 × 局部空间递归”的组合，不是单独的 ordinal counter。

#### 跨模型容量匹配比较

| endpoint | RNN−history 中位增益 | 95% CI | 正方向 | P |
|---|---:|---:|---:|---:|
| next contact | −0.002978 | [−0.024552,+0.026852] | 14/28 | 0.540 |
| STOP | +0.031398 | [+0.018086,+0.049655] | 28/28 | 3.725×10⁻⁹ |

因此 post-training lesion 证明 RNN **采用**该路径；容量匹配模型比较说明 next-contact 任务并不**要求所有可行模型都采用**该路径。STOP 是当前唯一稳定的跨架构递归增益。

### 3.6 未解释方向结构

真实 rank sequence 相对 event-wise order shuffle 的方向连续性 excess 中位数 +0.06234，28/28 为正。真实减生成的连续性差：

| 模型 | 中位差 | 95% CI | 正方向 |
|---|---:|---:|---:|
| even local | +0.01395 | [+0.00159,+0.03462] | 21/28 |
| elongated | +0.01542 | [−0.00050,+0.03386] | 18/28 |
| direction-biased | +0.01688 | [+0.00133,+0.03450] | 20/28 |
| forward relay | +0.01691 | [+0.00222,+0.03223] | 20/28 |

新增 motif 未缩小数据—模型差距。

### 3.7 发作期

Parity-only：17 patients、224 seizures、2464 pseudo-onsets。静态场之外加入 IED motif 后：

- 真实 onset 绝对增量：median \(\Delta error=-1.7706\)，3/17 改善；
- real-minus-pseudo：median +0.3816，15/17，P=0.0133。

第二项只表示该负贡献在真实 onset 相对没那么差；由于真实 onset 的绝对增量仍为负，不能支持 incremental reuse。主图删除 seizure panel。

### 3.8 探索性时间线索

控制 rank step 后，事件内时间代理与距离的偏相关中位数 +0.13206，27/28 为正。变量来自 spectral mass-centre timing，不等于 onset latency 或传导速度；只用于决定下一版是否加入时间监督。

## 4. 新 Figure 6

输出：`results/paper-ready-figure/fig6_dynamical_motif_rnn_v0_2/figures/`

- A：有序 SEEG rank-set 输入、完整 tissue RNN、下一 rank 与 STOP 输出，以及四种局部规则；
- B：E1146 真实和 order-aware RNN 生成的 TA/TB 序列；TA/TB 仅为训练后展示标签；
- C：打乱 continuation 后的患者级留出损失；
- D：三种 motif 的剂量—下一触点误差曲线；
- E：contact-only checkpoint 在全部、较晚和较长空间转移上的留出增益；
- F：同图区分“RNN 内部使用局部递归”与“任务跨架构需要递归”，STOP 小效应另给缩放 inset；
- G：四种模型均未解释完的真实方向连续性。

新图采用最终双栏宽 7.15 inch、四行布局。PNG 原图和 PDF 首页分别目检；3217×3982 px，PDF 1 页，最小 PDF word box 高度 6.54 pt，SVG 保留 112 个文字节点。旧 seizure panel、无 \(\gamma\) 的 synthetic panel 和塌缩到零的 M3 替代对照均不进入主图。

## 5. 工程验收

- contact-only sensitivity：112/112；
- capacity-matched baseline：84/84；
- operator visibility：28/28 patients，所有 grid truth 精确恢复；
- contract tests：44 passed；
- failure / nonfinite / OOM：0 / 0 / 0；
- PNG/PDF/SVG：同一脚本状态重新生成，`FIGURE_VISUAL_QA.json` = PASS。

## 6. 最终 claim boundary

允许：

> The fitted RNN uses local recurrent mixing, but next-contact prediction is not uniquely diagnostic of that implementation. Additional anisotropic, directional and axis-aligned feedforward constraints do not improve held-out prediction, whereas recurrent state retains a reproducible advantage for event termination.

禁止：

- 把 motif 阴性外推成脑组织没有方向传播；
- 把 post-training lesion 写成该机制在所有模型中必要；
- 把 operator visibility 写成 finite-sample recovery；
- 把 parity-only seizure real-minus-pseudo 差写成绝对 reuse 阳性。


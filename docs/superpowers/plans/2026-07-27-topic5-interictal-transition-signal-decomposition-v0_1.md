# Topic 5 interictal transition signal decomposition v0.1 execution plan

> 对应 spec：
> `docs/superpowers/specs/2026-07-27-topic5-interictal-transition-signal-decomposition-v0_1.md`

## Milestone A：v2.2.1 收口

1. 重建 66/66 epoch audit。
2. 将 Claim 2 与 Claim 3/4 的状态分开：
   - Claim 2 = `FAIL`
   - Claim 3/4 = `LOCKED_NOT_RUN`
3. 从现有 raw NLL 形成 node / Markov / isotropic / full 五项比较。
4. 从 frozen checkpoints 复算：
   - next-set cardinality；
   - positive / negative / STOP NLL；
   - step 1/2/3/4+；
   - hazard 均值与离散度。
5. 做 operator identifiability：
   - local-axis Frobenius cosine；
   - full-isotropic operator distance；
   - learned axis–PCA1 cosine；
   - gamma→0 的 heldout logit 改变量。
6. 生成 v2.2.1 closeout 图、README、状态 JSON 和正式归档。

停止条件：任一复算 NLL 与 frozen artifact 不一致，先修评分合同，不进入 Milestone B。

## Milestone B：共享评分内核

1. 把 conditional-nonempty set likelihood、STOP 和 event-first aggregation 写成纯函数。
2. 用 toy events 验证：
   - tie set；
   - terminal STOP；
   - eligible mask；
   - no future-length leakage；
   - Markov 与 node 共用 denominator。
3. 保存 dataset、代码和 spec checksum。

## Milestone C：31 人 coordinate-free decomposition

1. 跑 `node_bias`、accepted probability Markov 和 directed-logit Markov。
2. 分解 \(L^S\) 与 \(L^A\)。
3. 评估 source-only / last-rank / last-2 / last-3 / full-prefix history。
4. 输出 patient-level metrics；不做 axis claim。

## Milestone D：22 人 physical-axis decomposition

1. same-shaft / distance / combined geometry。
2. 32 个 train-only physical-axis candidates。
3. axis residual 相对 local geometry 的 heldout benefit。
4. source-conditioned skew component。
5. 跨 shaft 正式比较使用含正、负 eligible contacts 的 conditional-nonempty
   prefix likelihood；预先要求至少 20 个 heldout events 和 50 个 heldout
   prefixes 含跨-shaft next contact。
6. positive-contact NLL 仅作 descriptive calibration，不参与 go/no-go。
7. 跨 shaft P 值与其余正式比较进入同一个 BH-FDR family。

若 local geometry 已解释 Markov 且 axis residual 非阳性，停止，不优化更多方向或
新增 nonlinear basis。

## Milestone E：clinical-onset metadata 独立工作流

1. 从 71 次 seizure metadata inventory 生成 blinded annotation registry。
2. 字段固定为：
   - patient_id / seizure_id
   - clinical_onset_time / contacts
   - montage / reference / source document
   - reviewer_1 / reviewer_2 / consensus
   - confidence / exact join / exclusion reason
3. 不自动填 SOZ、energy-top、A/B source 或 patient-level focus。
4. 输出 readiness 状态；没有人工 exact source 时保持 0 eligible。

## Milestone F：统计、图和决策

1. patient-first Wilcoxon + BH-FDR。
2. 检查正效应患者数、dataset 分层和 leave-one-patient-out robustness。
3. 生成一张四块诊断图：
   - local geometry；
   - symmetric vs skew；
   - physical-axis/source-conditioned residual；
   - history depth。
4. 按 spec §6 写 `GO_V2_3_RNN`、`GO_MINIMAL_OPERATOR` 或
   `STOP_SYSTEM_IDENTIFICATION`。
5. 更新 archive、paper-draft/README、archive index、figure index 和 checksum
   manifest。

## 执行顺序

```text
v2.2.1 closeout
  ↓ scoring contract PASS
coordinate-free decomposition
  ↓ real residual beyond node bias
geometry / symmetry / direction decomposition
  ↓ axis and source-conditioned evidence
history-depth test
  ↓ only then decide whether a recurrent state is justified
```
